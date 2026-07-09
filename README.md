# TrueVision AI

A deep learning-based handwriting forgery detection system that classifies handwriting samples as **Real** or **Forged** using an ensemble of **MobileNet** and **ResNet50** models. The project follows a client-server architecture with a **Streamlit frontend** and a **FastAPI backend**.

---

## Overview

Handwriting verification plays an important role in document authentication, forensic investigations, banking, and legal processes. TrueVision AI leverages transfer learning and ensemble learning to improve handwriting forgery detection accuracy.

The application allows users to upload handwriting images through a Streamlit interface. Images are processed by a FastAPI backend, where two independently trained deep learning models generate predictions. The final decision is obtained by averaging the outputs of both models.

---

## Features

- Binary handwriting classification (Real vs Forged)
- Ensemble prediction using MobileNet and ResNet50
- REST API powered by FastAPI
- Interactive web interface built with Streamlit
- Automatic Swagger API documentation
- Background model loading for faster startup
- Confidence score for every prediction
- Modular project structure for easy maintenance

---

## System Architecture

```
                  User
                    │
                    ▼
         Streamlit Frontend
                    │
             HTTP Request
                    │
                    ▼
            FastAPI Backend
                    │
        Image Preprocessing
                    │
        ┌───────────┴───────────┐
        ▼                       ▼
   MobileNet Model        ResNet50 Model
        │                       │
        └───────────┬───────────┘
                    ▼
          Ensemble Averaging
                    │
                    ▼
         Real / Forged Prediction
                    │
                    ▼
           Response to Frontend
```

---

## Technology Stack

| Category | Technology |
|----------|------------|
| Programming Language | Python 3.11 |
| Deep Learning | TensorFlow, Keras |
| Backend | FastAPI |
| Frontend | Streamlit |
| Image Processing | Pillow, NumPy |
| Server | Uvicorn |
| Version Control | Git, Git LFS |

---

## Project Structure

```text
TrueVision-AI/
│
├── backend/
│   ├── main.py                     # FastAPI backend
│   ├── models/
│   │   ├── mobilenet_model.h5
│   │   └── resnet_model.h5
│   └── __pycache__/
│
├── frontend/
│   ├── app.py                      # Streamlit frontend
│   └── assets/
│
├── .devcontainer/                  # VS Code Dev Container configuration
├── .gitattributes                  # Git LFS configuration
├── .gitignore                      # Git ignore rules
├── README.md                       # Project documentation
├── requirements.txt                # Project dependencies
├── runtime.txt                     # Python runtime version
├── Procfile                        # Process configuration (deployment)
│
└── venv_tf/                        # Local virtual environment (not committed)
```
---

## Model Details

### MobileNet

A lightweight convolutional neural network optimized for efficient feature extraction and fast inference.

### ResNet50

A deep residual neural network capable of learning complex handwriting patterns through residual connections.

### Ensemble Strategy

Both models independently generate prediction scores.

The final confidence score is calculated as:

```
Final Score = (MobileNet Prediction + ResNet50 Prediction) / 2
```

A threshold of **0.35** is used for classification:

- Score > 0.35 → Forged Handwriting
- Score ≤ 0.35 → Real Handwriting

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/mitali-gautam05/TrueVision-AI.git
cd TrueVision-AI
```

If using Git LFS:

```bash
git lfs install
git lfs pull
```

---

### 2. Create a Virtual Environment

Windows

```bash
python -m venv venv
venv\Scripts\activate
```

Linux/macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

---

### 3. Install Dependencies

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

## Running the Project

### Step 1: Start the Backend

```bash
cd backend
uvicorn main:app --reload
```

Backend URL

```
http://127.0.0.1:8000
```

Swagger Documentation

```
http://127.0.0.1:8000/docs
```

---

### Step 2: Start the Frontend

Open another terminal.

```bash
cd frontend
streamlit run app.py
```

The application will be available at:

```
http://localhost:8501
```

---

## API Endpoints

| Method | Endpoint | Description |
|---------|----------|-------------|
| GET | `/` | API status |
| GET | `/health` | Health check |
| GET | `/debug` | Debug information |
| POST | `/predict` | Predict handwriting authenticity |
| GET | `/docs` | Swagger API documentation |

---

## Workflow

1. Upload a handwriting image using the Streamlit interface.
2. The image is sent to the FastAPI backend.
3. The backend preprocesses the image.
4. MobileNet and ResNet50 independently generate prediction scores.
5. The scores are averaged to produce the final confidence value.
6. The prediction and confidence score are displayed to the user.

---

## Future Enhancements

- Docker containerization
- Cloud deployment
- Support for additional handwriting datasets
- Improved ensemble techniques
- User authentication
- Batch prediction support
- Model monitoring and logging

---


