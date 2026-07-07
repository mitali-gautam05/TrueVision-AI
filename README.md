# TrueVision AI

> A deep learning system that classifies handwriting samples as **Real or Forged** using an ensemble of MobileNet and ResNet50 models — served via a FastAPI backend and Streamlit frontend.

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.19-FF6F00?logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-Frontend-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Overview

Handwriting forgery is a serious concern in document verification, forensic analysis, and signature authentication. TrueVision AI addresses this problem using computer vision and transfer learning.

Two pretrained CNN architectures — MobileNet and ResNet50 — are independently trained on handwriting samples and combined via ensemble averaging for more reliable predictions than either model alone. The system is built with a clean separation between the ML backend (FastAPI) and the user interface (Streamlit).

---

## Live Demo

- **Frontend (Streamlit):** deployed and publicly accessible
- **Backend (FastAPI on Render):** `https://truevision-ai-6.onrender.com`
  - Swagger docs: `https://truevision-ai-6.onrender.com/docs`
  - Health check: `https://truevision-ai-6.onrender.com/health`

> Note: the backend runs on Render's free tier, which sleeps after periods of inactivity. The first request after idle time may take 20–60 seconds while it spins back up — this is expected, not a bug.

---

## Features

- Binary classification: Real vs Forged handwriting
- Ensemble model combining MobileNet and ResNet50 predictions
- Tunable confidence threshold (default: 0.35)
- REST API with auto-generated Swagger docs via FastAPI
- Interactive image upload UI via Streamlit with confidence visualizations
- Background model loading — server binds to its port immediately so the platform's health checks pass even while models are still loading
- Automatic retry + warm-up handling on the frontend for cold starts
- Prediction history and a downloadable analysis report

---

## Architecture

```
User uploads image (Streamlit UI)
          │
          ▼
  FastAPI Backend receives image
          │
          ▼
  Image Preprocessing Pipeline
          │
          ├─────────────────────┐
          ▼                     ▼
     MobileNet            ResNet50
     Inference            Inference
          │                     │
          └──────────┬──────────┘
                     ▼
          Ensemble Averaging
          (mean of both outputs)
                     │
                     ▼
        Threshold → REAL / FORGED
                     │
                     ▼
        Confidence Score returned
          to Streamlit frontend
```

---

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.11 |
| Deep Learning | TensorFlow 2.19, Keras 3.13 |
| Models | MobileNet, ResNet50 (pretrained, fine-tuned) |
| Backend | FastAPI, Uvicorn |
| Frontend | Streamlit |
| Image Processing | Pillow, NumPy |
| Visualization | Matplotlib, Pandas |
| Model Storage | Git LFS |
| Deployment | Render (backend), Streamlit Community Cloud (frontend) |

---

## Project Structure

```
project/
│
├── backend/
│   ├── main.py               # FastAPI app — routes, background model loading, inference
│   └── models/
│       ├── mobilenet_model.h5
│       └── resnet_model.h5
│
├── frontend/
│   ├── app.py                # Streamlit UI — upload, analysis, history, reports
│   ├── requirements.txt
│   └── assets/
│       ├── bg.jpg
│       └── logo.png
│
├── .gitattributes            # Git LFS tracking rules for model files
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Model Details

**MobileNet**
Lightweight depthwise separable CNN optimized for speed and low memory usage. Suitable for fast inference without sacrificing meaningful feature extraction.

**ResNet50**
50-layer residual network with skip connections that enable deeper feature learning. Captures fine-grained texture and stroke patterns better than shallow architectures.

**Ensemble Strategy**
Both models output a probability score. The final prediction is the arithmetic mean of both outputs, passed through a threshold of **0.35** (tunable in `backend/main.py`) to determine Real vs Forged.

This ensemble approach reduces individual model bias and improves generalization across varied handwriting styles.

---

## Getting Started

### Prerequisites

- Python 3.11+
- pip
- Git and [Git LFS](https://git-lfs.com/) (model files are tracked via LFS)

### 1. Clone the repository

```bash
git clone https://github.com/mitali-gautam05/TrueVision-AI.git
cd TrueVision-AI
git lfs pull
```

### 2. Create and activate a virtual environment

**Windows**
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux / macOS**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the backend

```bash
cd backend
uvicorn main:app --reload
```

API will be live at `http://127.0.0.1:8000`
Interactive docs at `http://127.0.0.1:8000/docs`

### 5. Run the frontend

Open a new terminal:

```bash
cd frontend
streamlit run app.py
```

UI will open at `http://localhost:8501`

> If running the frontend against the deployed backend instead of localhost, update `API_URL` in `frontend/app.py`.

---

## Use Cases

- **Signature verification** — detect forged signatures on legal or financial documents
- **Document fraud detection** — flag suspicious handwriting in identity or medical records
- **Forensic handwriting analysis** — assist investigators with preliminary authenticity checks

---

## Deployment Notes

- Model files are large (~215MB combined) and are tracked with **Git LFS** rather than committed directly, to stay within GitHub's per-file size limits.
- The backend loads both models in a background thread on startup so the web server can bind to its port immediately — this prevents the hosting platform from killing the instance mid-boot during slow model loads.
- The `/predict` endpoint returns a `503` with a `"warming_up"` status if a request arrives before models finish loading, and the frontend automatically retries in that case.

---

## Roadmap

- [ ] Docker containerization for unified backend + frontend deployment
- [ ] Add logging and monitoring (e.g. MLflow or Weights & Biases)
- [ ] Expand dataset with more handwriting styles and languages
- [ ] Improve model accuracy with data augmentation and fine-tuning
- [ ] Add user authentication for API access control
- [ ] Real-time video-based handwriting verification

---

## License

This project is licensed under the MIT License.