import streamlit as st
import requests
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import base64
import os

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="TrueVision AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# SESSION STATE
# =====================================================
if "history" not in st.session_state:
    st.session_state.history = []

# =====================================================
# LOAD BACKGROUND
# =====================================================
def get_base64(file_path):
    try:
        with open(file_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except Exception:
        return ""

# =====================================================
# PREMIUM CSS
# =====================================================
st.markdown(f"""
<style>

/* Hide Branding */
header {{
    background: transparent !important;
}}

#MainMenu {{
    visibility:hidden;
}}

footer {{
    visibility:hidden;
}}

[data-testid="collapsedControl"] {{
    display:flex !important;
    position:fixed;
    top:12px;
    left:12px;
    z-index:99999;
    background:rgba(15,23,42,0.85);
    border-radius:12px;
    padding:6px;
}}

[data-testid="collapsedControl"] svg {{
    fill:white !important;
}}

footer {{
    visibility: hidden;
}}

#MainMenu {{
    visibility: hidden;
}}

.block-container {{
    padding-top: 1rem;
    padding-bottom: 2rem;
    max-width: 1400px;
}}

/* Background */
.stApp {{
    background:
    linear-gradient(rgba(4,8,20,0.78), rgba(4,8,20,0.84)),
    url("data:image/jpg;base64,{bg}");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}}

/* Sidebar */
section[data-testid="stSidebar"] {{
    background: rgba(8,15,30,0.94);
    border-right: 1px solid rgba(255,255,255,0.08);
}}

section[data-testid="stSidebar"] * {{
    color: white !important;
}}

/* Titles */
.hero-title {{
    font-size: 72px;
    font-weight: 900;
    line-height: 1;
    margin-bottom: 10px;
    color: white;
}}

.hero-blue {{
    color: #38bdf8;
}}

.hero-sub {{
    font-size: 22px;
    color: #dbeafe;
    margin-bottom: 14px;
}}

.hero-tag {{
    font-size: 18px;
    color: #94a3b8;
    margin-bottom: 30px;
}}

/* Cards */
.card {{
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 22px;
    padding: 28px;
    backdrop-filter: blur(12px);
    box-shadow: 0 10px 35px rgba(0,0,0,0.22);
    transition: 0.3s ease;
    min-height: 230px;
}}

.card:hover {{
    transform: translateY(-6px);
    box-shadow: 0 14px 40px rgba(56,189,248,0.15);
}}

/* Buttons */
.stButton > button {{
    width: 100%;
    background: linear-gradient(90deg,#06b6d4,#2563eb);
    color: white;
    border: none;
    border-radius: 14px;
    padding: 14px;
    font-size: 18px;
    font-weight: 700;
}}

.stButton > button:hover {{
    transform: scale(1.02);
}}

/* Download Button */
[data-testid="stDownloadButton"] button {{
    width: 100%;
    background: linear-gradient(90deg,#f59e0b,#ea580c) !important;
    color: white !important;
    border: none !important;
    border-radius: 14px !important;
    padding: 14px !important;
    font-weight: 700 !important;
}}

/* =====================================================
FILE UPLOADER DARK FIX
===================================================== */

/* Outer uploader */
[data-testid="stFileUploader"] {{
    background: rgba(255,255,255,0.04) !important;
    border-radius: 20px !important;
    padding: 14px !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
}}

/* Inner drag box */
[data-testid="stFileUploaderDropzone"] {{
    background: rgba(15,23,42,0.92) !important;
    border: 2px dashed #38bdf8 !important;
    border-radius: 18px !important;
    padding: 35px !important;
}}

/* Hover effect */
[data-testid="stFileUploaderDropzone"]:hover {{
    background: rgba(30,41,59,0.95) !important;
    border-color: #22d3ee !important;
}}

/* Text inside uploader */
[data-testid="stFileUploaderDropzone"] * {{
    color: white !important;
}}

/* Browse files button */
[data-testid="stFileUploaderDropzone"] button {{
    background: linear-gradient(90deg,#06b6d4,#2563eb) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 700 !important;
}}

[data-testid="stFileUploaderDropzone"] button p {{
    color: white !important;
}}

/* Uploaded file selected row */
[data-testid="stFileUploaderFile"] {{
    background: rgba(255,255,255,0.06) !important;
    border-radius: 12px !important;
    padding: 10px !important;
}}

/* Metrics */
[data-testid="metric-container"] {{
    background: rgba(255,255,255,0.06);
    border-radius: 16px;
    padding: 10px;
    border: 1px solid rgba(255,255,255,0.08);
}}

/* Text */
h1,h2,h3,h4,p,label,div,span {{
    color: white !important;
}}

hr {{
    border: 1px solid rgba(255,255,255,0.06);
}}

</style>
""", unsafe_allow_html=True)

# =====================================================
# API
# =====================================================
API_URL = "https://truevision-ai-6.onrender.com/predict"

# =====================================================
# SIDEBAR
# =====================================================
st.sidebar.image("assets/logo.png", width=150)
st.sidebar.markdown("## TrueVision AI")
st.sidebar.caption("Forgery Detection Platform")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "🔍 Detection",
        "📊 Analytics",
        "ℹ️ About"
    ]
)

st.sidebar.markdown("---")
st.sidebar.success("AI Powered Verification")

# =====================================================
# HOME
# =====================================================
if page == "🏠 Home":

    col1, col2 = st.columns([1,5])

    with col1:
        st.image("assets/logo.png", width=120)

    with col2:
        st.markdown("""
        <div class="hero-title">
        TrueVision <span class="hero-blue">AI</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(
        '<div class="hero-sub">Next Generation Handwriting Authentication Platform</div>',
        unsafe_allow_html=True)

        st.markdown(
        '<div class="hero-tag">Verify authenticity. Detect forgery. Build trust instantly.</div>',
        unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("""
        <div class="card">
        <h2>⚡ Instant Detection</h2>
        <p>Upload handwriting image and receive AI prediction within seconds.</p>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="card">
        <h2>🧠 Dual Model AI</h2>
        <p>MobileNet + ResNet ensemble for smarter predictions.</p>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class="card">
        <h2>🔐 Enterprise Ready</h2>
        <p>Perfect for banking, legal docs, signatures and fraud checks.</p>
        </div>
        """, unsafe_allow_html=True)

# =====================================================
# DETECTION
# =====================================================
elif page == "🔍 Detection":

    st.markdown("""
    <div class="hero-title" style="font-size:52px;">
    Forgery <span class="hero-blue">Detection</span>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload handwriting image",
        type=["jpg","jpeg","png"]
    )

    if uploaded_file:

        image = Image.open(uploaded_file)
        st.image(image, use_container_width=True)

        if st.button("Analyze Now"):

            with st.spinner("Running AI Analysis..."):

                files = {
                    "file": (
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        "image/jpeg"
                    )
                }

                response = requests.post(API_URL, files=files)
                result = response.json()

                if "error" in result:
                    st.error(result["error"])

                else:
                    label = result["label"]
                    confidence = result["confidence"]
                    mobilenet = result["mobilenet"]
                    resnet = result["resnet"]

                    a,b,c = st.columns(3)

                    with a:
                        st.metric("Prediction", label)

                    with b:
                        st.metric("Confidence", f"{confidence:.3f}")

                    with c:
                        st.metric("Status", "Completed")

                    st.progress(float(confidence))

                    col1, col2 = st.columns(2)

                    with col1:
                        st.metric("MobileNet", f"{mobilenet:.3f}")

                    with col2:
                        st.metric("ResNet", f"{resnet:.3f}")

                    fig, ax = plt.subplots(figsize=(8,4))
                    bars = ax.bar(
                        ["MobileNet","ResNet"],
                        [mobilenet,resnet]
                    )

                    ax.set_ylim(0,1)
                    ax.set_title("Model Confidence Comparison")

                    for bar in bars:
                        h = bar.get_height()
                        ax.text(
                            bar.get_x()+bar.get_width()/2,
                            h+0.02,
                            f"{h:.2f}",
                            ha="center"
                        )

                    st.pyplot(fig)

                    # ==========================================
                    # AFTER GRAPH FEATURES
                    # ==========================================

                    st.markdown("## Final Verdict")

                    if "FAKE" in label:
                        st.error("⚠️ Possible Forgery Detected")
                    else:
                        st.success("✅ Likely Genuine Handwriting")

                    st.markdown("## Confidence Progress")
                    st.progress(float(confidence))

                    # st.markdown("## Model Agreement")

                    mob_label = "FAKE" if mobilenet > 0.35 else "REAL"
                    res_label = "FAKE" if resnet > 0.35 else "REAL"

                    if mob_label == res_label:
                        st.success(f"Both Models Agree : {mob_label}")
                    else:
                        st.warning("Models Disagree")

                    st.session_state.history.append({
                        "Time": datetime.now().strftime("%H:%M:%S"),
                        "File": uploaded_file.name,
                        "Result": label,
                        "Confidence": round(confidence,3)
                    })

                    report = f"""
TrueVision AI Report
-------------------
File: {uploaded_file.name}
Prediction: {label}
Confidence: {confidence:.3f}
MobileNet: {mobilenet:.3f}
ResNet: {resnet:.3f}
Generated: {datetime.now()}
"""

                    st.markdown("## Download Report")

                    st.download_button(
                        "Download TXT Report",
                        report,
                        file_name="truevision_report.txt",
                        mime="text/plain"
                    )

# =====================================================
# ANALYTICS
# =====================================================
elif page == "📊 Analytics":

    st.markdown("""
    <div class="hero-title" style="font-size:52px;">
    Analytics <span class="hero-blue">Dashboard</span>
    </div>
    """, unsafe_allow_html=True)

    if len(st.session_state.history) == 0:
        st.warning("No predictions yet.")
    else:
        df = pd.DataFrame(st.session_state.history[::-1])
        st.dataframe(df, use_container_width=True)

# =====================================================
# ABOUT
# =====================================================
elif page == "ℹ️ About":

    st.markdown("""
    <div class="hero-title" style="font-size:52px;">
    About <span class="hero-blue">TrueVision AI</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
### Premium handwriting authentication platform

Built using:
- Streamlit
- FastAPI
- TensorFlow
- MobileNet
- ResNet

Use Cases:
- Signature Verification
- Legal Documents
- Banking Security
- Fraud Detection
""")