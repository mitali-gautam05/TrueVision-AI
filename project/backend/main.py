from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os

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
# SAFE BASE PATH (IMPORTANT FOR RENDER)
# =====================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

# =====================================================
# LOAD MODELS SAFELY
# =====================================================
try:
    mobilenet_model = tf.keras.models.load_model(
        os.path.join(MODEL_DIR, "mobilenet_model.h5")
    )

    resnet_model = tf.keras.models.load_model(
        os.path.join(MODEL_DIR, "resnet_model.h5")
    )

except Exception as e:
    mobilenet_model = None
    resnet_model = None
    print("MODEL LOADING ERROR:", e)

# =====================================================
# IMAGE PREPROCESS
# =====================================================
IMG_SIZE = 224

def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# =====================================================
# HOME ROUTE
# =====================================================
@app.get("/")
def home():
    return {
        "message": "TrueVision AI Backend Running Successfully"
    }

# =====================================================
# PREDICT ROUTE
# =====================================================
@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    try:
        if mobilenet_model is None or resnet_model is None:
            return JSONResponse(
                status_code=500,
                content={"error": "Models not loaded properly"}
            )

        image_bytes = await file.read()
        processed_image = preprocess_image(image_bytes)

        # =================================================
        # PREDICTIONS
        # =================================================
        mobilenet_pred = float(
            mobilenet_model.predict(processed_image)[0][0]
        )

        resnet_pred = float(
            resnet_model.predict(processed_image)[0][0]
        )

        # =================================================
        # ENSEMBLE
        # =================================================
        confidence = (mobilenet_pred + resnet_pred) / 2

        threshold = 0.35

        label = (
            "FAKE Handwriting Detected"
            if confidence > threshold
            else "REAL Handwriting Detected"
        )

        # =================================================
        # RESPONSE
        # =================================================
        return JSONResponse({
            "label": label,
            "confidence": round(confidence, 3),
            "mobilenet": round(mobilenet_pred, 3),
            "resnet": round(resnet_pred, 3)
        })

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )