import tensorflow as tf
import numpy as np

IMG_SIZE = (224, 224)


# Load both models (cached once in Streamlit)
@tf.keras.utils.register_keras_serializable()
def load_models():
    mobilenet = tf.keras.models.load_model("models/mobilenet_model.keras")
    resnet = tf.keras.models.load_model("models/resnet_model.keras")
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


# Final prediction (ENSEMBLE)
def predict_image(mobilenet, resnet, img):
    """
    img: numpy array of shape (H, W, 3), RGB
    """
    img = img.astype(np.float32)

    # MobileNet prediction
    img_m = preprocess_mobilenet(img)
    pred_m = mobilenet.predict(tf.expand_dims(img_m, 0), verbose=0)[0][0]

    # ResNet prediction
    img_r = preprocess_resnet(img)
    pred_r = resnet.predict(tf.expand_dims(img_r, 0), verbose=0)[0][0]

    # Ensemble (average)
    final_pred = (pred_m + pred_r) / 2

    # Threshold (same as your logic)
    threshold = 0.35

    label = "FAKE ❌" if final_pred > threshold else "REAL ✅"
    confidence = final_pred if final_pred > threshold else 1 - final_pred

    return label, float(confidence), float(pred_m), float(pred_r)