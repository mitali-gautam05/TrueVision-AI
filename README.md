# TrueVision AI

> A deep learning system that classifies handwriting samples as **Real or Forged** using an ensemble of MobileNet and ResNet50 models, powered by FastAPI and an interactive frontend.

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.19-FF6F00?logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)


---

# Overview

TrueVision AI is a deep learning based handwriting forgery detection system that classifies handwriting samples as **Real** or **Forged**.

The application uses an ensemble of **MobileNet** and **ResNet50** models trained on handwriting images. Both model predictions are averaged to improve robustness and reduce prediction bias.

The backend is developed using **FastAPI**, while the frontend communicates with it through REST APIs.

---

# Features

- Detect forged handwriting images
- Ensemble prediction using MobileNet + ResNet50
- REST API using FastAPI
- Automatic Swagger documentation
- Confidence score generation
- Background model loading
- Prediction history support
- Clean modular project structure

---

# Architecture

```text
User Uploads Image
        │
        ▼
 FastAPI Backend
        │
        ▼
Image Preprocessing
        │
 ┌──────────────┐
 ▼              ▼
MobileNet    ResNet50
 ▼              ▼
 └──────┬───────┘
        ▼
 Ensemble Average
        ▼
Real / Forged
        ▼
 Return Prediction
```

---

# Tech Stack

| Category | Technology |
|----------|------------|
| Language | Python 3.11 |
| Deep Learning | TensorFlow, Keras |
| Backend | FastAPI |
| Image Processing | Pillow, NumPy |
| Server | Uvicorn |
| Version Control | Git, Git LFS |

---

# Project Structure

```text
TrueVision-AI/
│
├── backend/
│   ├── main.py
│   ├── requirements.txt
│   └── models/
│       ├── mobilenet_model.h5
│       └── resnet_model.h5
│
├── frontend/
│   ├── app.py
│   ├── requirements.txt
│   └── assets/
│
├── README.md
└── .gitignore
```

---

# Model Details

### MobileNet

A lightweight CNN architecture optimized for fast inference while maintaining high accuracy.

### ResNet50

A deep residual network capable of extracting complex handwriting features.

### Ensemble

The final confidence score is calculated as:

```
(MobileNet Prediction + ResNet Prediction) / 2
```

Default threshold:

```
0.35
```

Above threshold → Forged

Below threshold → Real

---

# Installation

## Clone Repository

```bash
git clone https://github.com/mitali-gautam05/TrueVision-AI.git
cd TrueVision-AI
git lfs pull
```

---

## Create Virtual Environment

Windows

```bash
python -m venv venv
venv\Scripts\activate
```

Linux/Mac

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## Install Dependencies

Backend

```bash
cd backend
pip install -r requirements.txt
```

Frontend

```bash
cd ../frontend
pip install -r requirements.txt
```

---

# Running Locally

## Start Backend

```bash
cd backend
uvicorn main:app --reload
```

Backend runs at

```
http://127.0.0.1:8000
```

Swagger Docs

```
http://127.0.0.1:8000/docs
```

---

## Start Frontend

Open another terminal

```bash
cd frontend
python app.py
```

Frontend opens at

```
http://localhost:7860
```

---

# API Endpoints

| Endpoint | Description |
|----------|-------------|
| / | Home |
| /predict | Predict handwriting |
| /health | Health status |
| /debug | Debug information |
| /docs | Swagger UI |

---

# Future Improvements

- Docker support
- Cloud deployment
- Better ensemble strategies
- More handwriting datasets
- User authentication
- Performance optimization

