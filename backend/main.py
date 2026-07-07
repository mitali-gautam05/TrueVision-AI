from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
import numpy as np
from PIL import Image
import io
import gc
import traceback
import logging
import threading

# =====================================================
# LOGGING
# =====================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

# =====================================================
# LIMIT THREADS — critical for Render free tier RAM
# =====================================================
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

# =====================================================
# FASTAPI APP
# =====================================================
app = FastAPI(title="TrueVision AI Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================================
# PATHS
# =====================================================
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

# =====================================================
# GPU MEMORY GROWTH
# =====================================================
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        log.info("GPU memory growth enabled")
    except RuntimeError as e:
        log.warning(f"GPU config error: {e}")
else:
    log.info("No GPU — running on CPU (expected on Render)")

# =====================================================
# MODEL STATE (loaded in background, NOT at import time)
# =====================================================
mobilenet_model = None
resnet_model    = None
models_loading  = False
models_error    = None

def load_models():
    """
    Runs in a background thread. Uvicorn binds to the port and starts
    serving immediately — Render's health check passes right away instead
    of waiting 60-90s for both models to load, which was causing Render
    to kill the instance mid-boot before it ever served a request.
    """
    global mobilenet_model, resnet_model, models_loading, models_error
    models_loading = True

    mobilenet_path = os.path.join(MODEL_DIR, "mobilenet_model.h5")
    resnet_path    = os.path.join(MODEL_DIR, "resnet_model.h5")

    if not os.path.exists(mobilenet_path):
        log.error(f"MobileNet NOT found: {mobilenet_path}")
        models_error = "MobileNet file missing"
        models_loading = False
        return
    if not os.path.exists(resnet_path):
        log.error(f"ResNet NOT found: {resnet_path}")
        models_error = "ResNet file missing"
        models_loading = False
        return

    try:
        log.info("Loading MobileNet...")
        mobilenet_model = tf.keras.models.load_model(
            mobilenet_path,
            compile=False
        )
        log.info("MobileNet loaded ✅")
    except Exception as e:
        log.error(f"MobileNet FAILED: {e}")
        traceback.print_exc()
        models_error = f"MobileNet load failed: {e}"

    try:
        log.info("Loading ResNet...")
        resnet_model = tf.keras.models.load_model(
            resnet_path,
            compile=False
        )
        log.info("ResNet loaded ✅")
    except Exception as e:
        log.error(f"ResNet FAILED: {e}")
        traceback.print_exc()
        models_error = f"ResNet load failed: {e}"

    gc.collect()
    models_loading = False
    log.info("Models loaded — memory cleaned up")

# =====================================================
# STARTUP — kick off background loading, don't block boot
# =====================================================
@app.on_event("startup")
def startup_event():
    thread = threading.Thread(target=load_models, daemon=True)
    thread.start()
    log.info("Model loading started in background thread")

# =====================================================
# PREPROCESS
# =====================================================
IMG_SIZE = 224

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE))
    arr   = np.array(image, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

# =====================================================
# HOME
# =====================================================
@app.get("/")
def home():
    return {
        "message":         "TrueVision AI Backend Running",
        "mobilenet_ready": mobilenet_model is not None,
        "resnet_ready":    resnet_model    is not None,
        "models_loading":  models_loading,
        "models_error":    models_error,
        "keras_version":   tf.keras.__version__,
        "tf_version":      tf.__version__,
    }

# =====================================================
# HEALTH  (ping with UptimeRobot / cron-job.org to prevent cold starts)
# =====================================================
@app.get("/health")
def health():
    status = "ok" if (mobilenet_model and resnet_model) else "degraded"
    return {"status": status, "models_loading": models_loading}

# =====================================================
# DEBUG
# =====================================================
@app.get("/debug")
def debug():
    models_exist    = os.path.exists(MODEL_DIR)
    files_in_models = os.listdir(MODEL_DIR) if models_exist else []
    return {
        "BASE_DIR":          BASE_DIR,
        "MODEL_DIR":         MODEL_DIR,
        "models_dir_exists": models_exist,
        "files_in_models":   files_in_models,
        "mobilenet_loaded":  mobilenet_model is not None,
        "resnet_loaded":     resnet_model    is not None,
        "models_loading":    models_loading,
        "models_error":      models_error,
        "keras_version":     tf.keras.__version__,
        "tf_version":        tf.__version__,
    }

# =====================================================
# PREDICT
# =====================================================
@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    if mobilenet_model is None or resnet_model is None:
        if models_loading:
            log.info("Predict called while models still loading")
            return JSONResponse(
                status_code=503,
                content={
                    "status": "warming_up",
                    "error": "Models are still loading, please retry in a few seconds."
                }
            )
        log.error("Predict called but models not loaded")
        return JSONResponse(
            status_code=500,
            content={"error": f"Models not loaded. {models_error or 'Check Render logs.'}"}
        )

    processed_image = None

    try:
        # --- read & validate ---
        image_bytes = await file.read()

        if len(image_bytes) == 0:
            return JSONResponse(
                status_code=400,
                content={"error": "Uploaded file is empty."}
            )

        # --- preprocess ---
        try:
            processed_image = preprocess_image(image_bytes)
        except Exception as e:
            log.error(f"Preprocessing failed: {e}")
            return JSONResponse(
                status_code=400,
                content={"error": f"Invalid image: {e}"}
            )

        log.info(f"Running inference: {file.filename}")

        # --- inference ---
        mobilenet_pred = float(
            mobilenet_model.predict(processed_image, verbose=0)[0][0]
        )
        gc.collect()
        resnet_pred = float(
            resnet_model.predict(processed_image, verbose=0)[0][0]
        )
        gc.collect()

        log.info(f"MobileNet: {mobilenet_pred:.3f} | ResNet: {resnet_pred:.3f}")

        # --- ensemble ---
        confidence = (mobilenet_pred + resnet_pred) / 2
        label = (
            "FAKE Handwriting Detected"
            if confidence > 0.35
            else "REAL Handwriting Detected"
        )

        log.info(f"Result: {label} | Confidence: {confidence:.3f}")

        return JSONResponse({
            "label":      label,
            "confidence": round(confidence, 3),
            "mobilenet":  round(mobilenet_pred, 3),
            "resnet":     round(resnet_pred, 3),
        })

    except Exception as e:
        log.error(f"Prediction error: {e}")
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

    finally:
        # --- always free memory after every request ---
        if processed_image is not None:
            del processed_image
        gc.collect()
        log.info("Memory cleaned after request")