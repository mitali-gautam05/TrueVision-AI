from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import tensorflow as tf
import numpy as np
from PIL import Image
import io

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
# LOAD MODELS
# =====================================================

# Change paths according to your files
mobilenet_model = tf.keras.models.load_model(
    "models/mobilenet_model.h5"
)

resnet_model = tf.keras.models.load_model(
    "models/resnet_model.h5"
)

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

        image_bytes = await file.read()

        processed_image = preprocess_image(image_bytes)

        # =================================================
        # MODEL PREDICTIONS
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

        # =================================================
        # LABEL
        # =================================================
        threshold = 0.35

        if confidence > threshold:
            label = "FAKE Handwriting Detected"
        else:
            label = "REAL Handwriting Detected"

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
            content={
                "error": str(e)
            }
        )