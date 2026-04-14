import streamlit as st
import requests
from PIL import Image

API_URL = "https://truevision-ai-1-2qy1.onrender.com/predict"

st.set_page_config(page_title="Handwriting Detection")

st.title("✍️ AI Handwriting Detection")
st.write("Detect REAL vs FAKE handwriting")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image")

    if st.button("Predict"):
        with st.spinner("Analyzing..."):

            files = {
    "file": (uploaded_file.name, uploaded_file.getvalue(), "image/jpeg")
}
            response = requests.post(API_URL, files=files)

            result = response.json()

            if "error" in result:
                st.error(result["error"])
            else:
                st.success(result["label"])
                st.write(f"Confidence: {result['confidence']:.3f}")

                st.markdown("### Model Insights")
                st.write(f"MobileNet: {result['mobilenet']:.3f}")
                st.write(f"ResNet: {result['resnet']:.3f}")