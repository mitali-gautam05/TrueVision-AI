import tensorflow as tf
import numpy as np
import os

IMG_SIZE = (224, 224)

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Model paths
MOBILENET_PATH = os.path.join(BASE_DIR, "models", "mobilenet_model.keras")
RESNET_PATH = os.path.join(BASE_DIR, "models", "resnet_model.keras")

# Global variables (avoid reloading every request)
mobilenet = None
resnet = None


# Load models ONLY ONCE
def load_models():
    global mobilenet, resnet

    if mobilenet is None or resnet is None:
        mobilenet = tf.keras.models.load_model(
            MOBILENET_PATH,
            compile=False,
            safe_mode=False   # 🔥 critical fix
        )
        resnet = tf.keras.models.load_model(
            RESNET_PATH,
            compile=False,
            safe_mode=False   # 🔥 critical fix
        )

    return mobilenet, resnet


# Preprocess for MobileNet
def preprocess_mobilenet(img):
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32)
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    return img


# Preprocess for ResNet
def preprocess_resnet(img):
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32)
    img = tf.keras.applications.resnet50.preprocess_input(img)
    return img


# Prediction
def predict_image(img):
    mobilenet, resnet = load_models()   # load once

    img = img.astype(np.float32)

    # MobileNet
    img_m = preprocess_mobilenet(img)
    pred_m = mobilenet.predict(tf.expand_dims(img_m, 0), verbose=0)[0][0]

    # ResNet
    img_r = preprocess_resnet(img)
    pred_r = resnet.predict(tf.expand_dims(img_r, 0), verbose=0)[0][0]

    # Ensemble
    final_pred = (pred_m + pred_r) / 2

    threshold = 0.35

    label = "FAKE ❌" if final_pred > threshold else "REAL ✅"
    confidence = final_pred if final_pred > threshold else 1 - final_pred

    return label, float(confidence), float(pred_m), float(pred_r)