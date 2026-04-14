from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from PIL import Image
import io

from model_utils import load_models, predict_image

app = FastAPI(title="Handwriting Detection API")

# ---------------- CORS (IMPORTANT for frontend) ----------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # later you can restrict
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- Lazy Model Loading ----------------
mobilenet = None
resnet = None

def get_models():
    global mobilenet, resnet
    if mobilenet is None or resnet is None:
        mobilenet, resnet = load_models()
    return mobilenet, resnet


@app.get("/")
def home():
    return {"message": "API is running"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        img_array = np.array(image)

        # Load models only when needed
        mobilenet, resnet = get_models()

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