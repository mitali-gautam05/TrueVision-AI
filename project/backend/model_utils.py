import tensorflow as tf
import numpy as np
import os

IMG_SIZE = (224, 224)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MOBILENET_PATH = os.path.join(BASE_DIR, "models", "mobilenet_model.keras")
RESNET_PATH = os.path.join(BASE_DIR, "models", "resnet_model.keras")

mobilenet = None
resnet = None


def load_models():
    global mobilenet, resnet

    if mobilenet is None or resnet is None:
        mobilenet = tf.keras.models.load_model(MOBILENET_PATH, compile=False)
        resnet = tf.keras.models.load_model(RESNET_PATH, compile=False)

    return mobilenet, resnet


def preprocess_mobilenet(img):
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32)
    return tf.keras.applications.mobilenet_v2.preprocess_input(img)


def preprocess_resnet(img):
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32)
    return tf.keras.applications.resnet50.preprocess_input(img)


def predict_image(img):
    mobilenet, resnet = load_models()

    img = img.astype(np.float32)

    pred_m = mobilenet.predict(
        tf.expand_dims(preprocess_mobilenet(img), 0), verbose=0
    )[0][0]

    pred_r = resnet.predict(
        tf.expand_dims(preprocess_resnet(img), 0), verbose=0
    )[0][0]

    final_pred = (pred_m + pred_r) / 2

    threshold = 0.35

    label = "FAKE ❌" if final_pred > threshold else "REAL ✅"
    confidence = final_pred if final_pred > threshold else 1 - final_pred

    return {
        "label": label,
        "confidence": float(confidence),
        "mobilenet": float(pred_m),
        "resnet": float(pred_r)
    }