from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os
import traceback
import logging

# =====================================================
# LOGGING SETUP
# =====================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

# =====================================================
# FASTAPI APP
# =====================================================
app = FastAPI(title="TrueVision AI Backend")

# =====================================================
# CORS
# =====================================================
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
# LIMIT TF MEMORY (CRITICAL FOR RENDER FREE TIER)
# Prevents OOM kill by capping GPU/CPU memory growth
# =====================================================
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"   # suppress TF noise

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        log.info("GPU memory growth enabled")
    except RuntimeError as e:
        log.warning(f"GPU config error: {e}")
else:
    log.info("No GPU found — running on CPU (expected on Render)")

# =====================================================
# LOAD MODELS
# =====================================================
mobilenet_model = None
resnet_model    = None

def load_models():
    global mobilenet_model, resnet_model

    mobilenet_path = os.path.join(MODEL_DIR, "mobilenet_model.h5")
    resnet_path    = os.path.join(MODEL_DIR, "resnet_model.h5")

    # --- check files exist before loading ---
    if not os.path.exists(mobilenet_path):
        log.error(f"MobileNet model NOT found at: {mobilenet_path}")
        return
    if not os.path.exists(resnet_path):
        log.error(f"ResNet model NOT found at: {resnet_path}")
        return

    try:
        log.info("Loading MobileNet model...")
        mobilenet_model = tf.keras.models.load_model(mobilenet_path)
        log.info("MobileNet loaded ✅")
    except Exception as e:
        log.error(f"MobileNet load FAILED: {e}")
        traceback.print_exc()

    try:
        log.info("Loading ResNet model...")
        resnet_model = tf.keras.models.load_model(resnet_path)
        log.info("ResNet loaded ✅")
    except Exception as e:
        log.error(f"ResNet load FAILED: {e}")
        traceback.print_exc()

# Load at startup
load_models()

# =====================================================
# IMAGE PREPROCESS
# =====================================================
IMG_SIZE = 224

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE))
    arr   = np.array(image, dtype=np.float32) / 255.0
    arr   = np.expand_dims(arr, axis=0)
    return arr

# =====================================================
# HOME / HEALTH CHECK
# =====================================================
@app.get("/")
def home():
    return {
        "message":        "TrueVision AI Backend Running",
        "mobilenet_ready": mobilenet_model is not None,
        "resnet_ready":    resnet_model    is not None,
    }

@app.get("/health")
def health():
    status = "ok" if (mobilenet_model and resnet_model) else "degraded"
    return {"status": status}

# =====================================================
# PREDICT
# =====================================================
@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    # --- guard: models must be loaded ---
    if mobilenet_model is None or resnet_model is None:
        log.error("Predict called but models are not loaded")
        return JSONResponse(
            status_code=500,
            content={
                "error": (
                    "Models are not loaded. "
                    "Check Render logs for MODEL LOADING ERROR."
                )
            }
        )

    try:
        # --- read & validate file ---
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
            log.error(f"Image preprocessing failed: {e}")
            return JSONResponse(
                status_code=400,
                content={"error": f"Invalid image file: {e}"}
            )

        # --- inference ---
        log.info(f"Running inference on: {file.filename}")

        mobilenet_pred = float(
            mobilenet_model.predict(processed_image, verbose=0)[0][0]
        )
        resnet_pred = float(
            resnet_model.predict(processed_image, verbose=0)[0][0]
        )

        log.info(f"MobileNet: {mobilenet_pred:.3f} | ResNet: {resnet_pred:.3f}")

        # --- ensemble ---
        confidence = (mobilenet_pred + resnet_pred) / 2
        threshold  = 0.35

        label = (
            "FAKE Handwriting Detected"
            if confidence > threshold
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
        traceback.print_exc()   # full traceback visible in Render logs
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )