# AI Handwriting Detection System

> A deep learning system that classifies handwriting samples as **Real or Forged** using an ensemble of MobileNet and ResNet50 models — served via a FastAPI backend and Streamlit frontend.

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-Frontend-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Overview

Handwriting forgery is a serious concern in document verification, forensic analysis, and signature authentication. This project addresses that problem using computer vision and transfer learning.

Two pretrained CNN architectures — MobileNet and ResNet50 — are independently trained on handwriting samples and combined via ensemble averaging for more reliable predictions than either model alone. The system is built with a clean separation between the ML backend (FastAPI) and user interface (Streamlit), following a production-style modular architecture.

---

## Demo

> Backend API docs: `http://127.0.0.1:8000/docs` after running locally

> Frontend: `http://localhost:8501` after running Streamlit

---

## Features

- Binary classification: Real vs Forged handwriting
- Ensemble model combining MobileNet and ResNet50 predictions
- Tunable confidence threshold (default: 0.35)
- REST API with auto-generated Swagger docs via FastAPI
- Interactive image upload UI via Streamlit
- Confidence score returned alongside prediction
- Modular codebase — models, preprocessing, and API are fully decoupled

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
        Threshold → REAL / FAKE
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
| Deep Learning | TensorFlow, Keras |
| Models | MobileNet, ResNet50 (pretrained, fine-tuned) |
| Backend | FastAPI, Uvicorn |
| Frontend | Streamlit |
| Image Processing | Pillow, NumPy |
| Deployment | Render (backend), Streamlit Cloud (frontend) |

---

## Project Structure

```
project/
│
├── backend/
│   ├── main.py              # FastAPI app — routes and request handling
│   ├── model_utils.py       # Model loading, preprocessing, ensemble logic
│   └── models/
│       ├── mobilenet_model.keras
│       └── resnet_model.keras
│
├── frontend/
│   └── app.py               # Streamlit UI — image upload and result display
│
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
Both models output a probability score. The final prediction is the arithmetic mean of both outputs, passed through a threshold of **0.35** (tunable in `model_utils.py`) to determine Real vs Fake.

This ensemble approach reduces individual model bias and improves generalization across varied handwriting styles.

---

## Getting Started

### Prerequisites

- Python 3.11+
- pip
- Git

### 1. Clone the repository

```bash
git clone https://github.com/mitali-gautam05/TrueVision-AI.git
cd handwriting-detection
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
python -m uvicorn main:app --reload
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

---

## Use Cases

- **Signature verification** — detect forged signatures on legal or financial documents
- **Document fraud detection** — flag suspicious handwriting in identity or medical records
- **Forensic handwriting analysis** — assist investigators with preliminary authenticity checks

---

## Roadmap

- [ ] Docker containerization for unified backend + frontend deployment
- [ ] Add logging and monitoring (e.g. MLflow or Weights & Biases)
- [ ] Expand dataset with more handwriting styles and languages
- [ ] Improve model accuracy with data augmentation and fine-tuning
- [ ] Add user authentication for API access control
- [ ] Real-time video-based handwriting verification

---



