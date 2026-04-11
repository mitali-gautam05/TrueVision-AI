from fastapi import FastAPI, File, UploadFile
import numpy as np
from PIL import Image
import io

from model_utils import load_models, predict_image

app = FastAPI(title="Handwriting Detection API")

# Load models once
mobilenet, resnet = load_models()


@app.get("/")
def home():
    return {"message": "API is running"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        img_array = np.array(image)

        label, confidence, pred_m, pred_r = predict_image(
            mobilenet, resnet, img_array
        )

        return {
            "label": label,
            "confidence": confidence,
            "mobilenet": pred_m,
            "resnet": pred_r
        }

    except Exception as e:
        return {"error": str(e)}