import streamlit as st
import requests
import time
from PIL import Image
import io
import os

# ─── CONFIG ───────────────────────────────────────────────────────────────────
BACKEND_URL = "https://truevision-ai-6.onrender.com"
MAX_RETRIES = 5          # how many times to retry on timeout/cold-start
RETRY_DELAY = 10         # seconds to wait between retries
REQUEST_TIMEOUT = 60     # seconds before a single request gives up

# ─── PAGE SETUP ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TrueVision AI",
    page_icon="🔍",
    layout="centered",
)

# ─── SAFE IMAGE HELPER ────────────────────────────────────────────────────────
def safe_image(path: str, **kwargs):
    """Display an image only if the file actually exists."""
    if os.path.exists(path):
        st.image(path, **kwargs)

# ─── BACKEND HELPERS ──────────────────────────────────────────────────────────

def wake_backend() -> bool:
    """
    Ping /health to wake the backend.
    Returns True when the backend responds OK, False if it never wakes.

    WHY THIS EXISTS:
    Render free tier puts the server to sleep after 15 min of inactivity.
    The first request triggers a cold start (~30-50 s).
    We ping /health (lightweight) first so the heavy /predict call
    doesn't hit a sleeping server.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.get(f"{BACKEND_URL}/health", timeout=REQUEST_TIMEOUT)
            if r.status_code == 200:
                return True
        except requests.exceptions.Timeout:
            pass
        except requests.exceptions.ConnectionError:
            pass

        if attempt < MAX_RETRIES:
            st.toast(f"⏳ Backend waking up… attempt {attempt}/{MAX_RETRIES}", icon="🔄")
            time.sleep(RETRY_DELAY)

    return False


def predict_with_retry(image_bytes: bytes, filename: str) -> dict | None:
    """
    POST the image to /predict with automatic retry on cold-start failures.

    Returns the JSON dict from the backend, or None on total failure.

    HOW RETRY LOGIC WORKS:
    ┌─────────────┐    timeout/error    ┌──────────────────┐
    │  attempt N  │ ─────────────────► │  wait RETRY_DELAY │
    └─────────────┘                    └──────────────────┘
          ▲                                      │
          └──────────────────────────────────────┘
                    (up to MAX_RETRIES times)
    """
    files = {"file": (filename, image_bytes, "image/jpeg")}

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.post(
                f"{BACKEND_URL}/predict",
                files=files,
                timeout=REQUEST_TIMEOUT,
            )
            response.raise_for_status()   # raises for 4xx / 5xx
            return response.json()

        except requests.exceptions.Timeout:
            msg = f"⏳ Request timed out (attempt {attempt}/{MAX_RETRIES})"
        except requests.exceptions.ConnectionError:
            msg = f"🔌 Connection error (attempt {attempt}/{MAX_RETRIES})"
        except requests.exceptions.HTTPError as e:
            # 5xx errors are worth retrying; 4xx are not
            if response.status_code < 500:
                st.error(f"❌ Server rejected the request: {e}")
                return None
            msg = f"🚨 Server error {response.status_code} (attempt {attempt}/{MAX_RETRIES})"
        except Exception as e:
            st.error(f"❌ Unexpected error: {e}")
            return None

        # show progress and wait before next attempt
        if attempt < MAX_RETRIES:
            st.warning(f"{msg} — retrying in {RETRY_DELAY}s…")
            time.sleep(RETRY_DELAY)
        else:
            st.error(
                f"{msg}\n\n"
                "The backend is not responding after several attempts.\n"
                "Please wait 60 seconds and refresh the page."
            )

    return None


# ─── UI ───────────────────────────────────────────────────────────────────────

safe_image("assets/logo.png", width=180)
st.title("TrueVision AI — Handwriting Forgery Detector")
st.markdown(
    "Upload a handwriting sample and our ensemble model "
    "(MobileNet + ResNet) will tell you if it's **genuine** or **forged**."
)

st.divider()

uploaded_file = st.file_uploader(
    "Upload handwriting image",
    type=["png", "jpg", "jpeg"],
    help="Supported formats: PNG, JPG, JPEG",
)

if uploaded_file:
    # show the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded sample", use_container_width=True)

    if st.button("🔍 Analyse", type="primary", use_container_width=True):

        # ── Step 1: wake the backend ──────────────────────────────────────────
        with st.spinner("Contacting backend… (may take up to 60 s on first request)"):
            backend_alive = wake_backend()

        if not backend_alive:
            st.error(
                "❌ Could not reach the backend after multiple attempts.\n\n"
                "**What to do:**\n"
                "1. Wait 60 seconds and try again.\n"
                "2. Visit the health check directly: "
                f"[{BACKEND_URL}/health]({BACKEND_URL}/health)\n"
                "3. If it still fails, the Render service may be down."
            )
            st.stop()

        # ── Step 2: send image for prediction ─────────────────────────────────
        uploaded_file.seek(0)          # rewind buffer after PIL read
        image_bytes = uploaded_file.read()

        with st.spinner("Analysing handwriting…"):
            result = predict_with_retry(image_bytes, uploaded_file.name)

        # ── Step 3: display result ─────────────────────────────────────────────
        if result:
            prediction = result.get("prediction", "Unknown")
            confidence = result.get("confidence", 0.0)

            st.divider()

            if prediction.lower() == "genuine":
                st.success(f"✅ **GENUINE** handwriting detected")
            else:
                st.error(f"⚠️ **FORGED** handwriting detected")

            st.metric("Confidence", f"{confidence * 100:.1f}%")

            with st.expander("Full response from backend"):
                st.json(result)

# ─── SIDEBAR — BACKEND STATUS ─────────────────────────────────────────────────
with st.sidebar:
    safe_image("assets/logo.png", width=120)
    st.header("Backend Status")

    if st.button("🩺 Check backend health"):
        with st.spinner("Pinging backend…"):
            try:
                r = requests.get(f"{BACKEND_URL}/health", timeout=15)
                if r.status_code == 200:
                    data = r.json()
                    st.success("✅ Backend is awake")
                    st.json(data)
                else:
                    st.warning(f"⚠️ Backend returned status {r.status_code}")
            except requests.exceptions.Timeout:
                st.warning("⏳ Backend is cold-starting. Wait 30 s and try again.")
            except Exception as e:
                st.error(f"❌ {e}")

    st.caption(f"Backend URL: `{BACKEND_URL}`")
    st.caption("Free tier may sleep after 15 min of inactivity.")