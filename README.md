# ✍️ AI Handwriting Detection System

A deep learning-based system to classify handwriting as **REAL or FAKE** using an ensemble of MobileNet and ResNet50 models. The project follows a production-style architecture with a FastAPI backend and a Streamlit frontend.

---

## 🚀 Project Overview

This project detects whether a given handwriting sample is authentic or forged using computer vision and deep learning techniques. It leverages two pretrained CNN architectures and combines their predictions for improved accuracy.

---

## 🧠 Features

* ✅ Binary classification: **REAL vs FAKE handwriting**
* ✅ Ensemble model (MobileNet + ResNet50)
* ✅ Image preprocessing pipeline
* ✅ REST API using FastAPI
* ✅ Interactive UI using Streamlit
* ✅ Confidence score + model insights
* ✅ Modular and scalable architecture

---

## 🏗️ Project Structure

```
project/
│
├── backend/
│   ├── main.py              # FastAPI backend
│   ├── model_utils.py      # Model loading & prediction logic
│   └── models/
│       ├── mobilenet_model.keras
│       └── resnet_model.keras
│
├── frontend/
│   └── app.py              # Streamlit UI
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Tech Stack

* Python
* TensorFlow / Keras
* FastAPI
* Streamlit
* NumPy, Pillow

---

## 🧪 How It Works

1. User uploads an image via Streamlit UI
2. Image is sent to FastAPI backend
3. Backend preprocesses the image
4. Predictions are made using:

   * MobileNet
   * ResNet50
5. Final prediction is computed using **ensemble averaging**
6. Result is returned with confidence score

---

## 📊 Model Details

* **MobileNet**: Lightweight CNN for fast inference
* **ResNet50**: Deep architecture for higher feature extraction
* **Ensemble Strategy**: Average of both model outputs
* **Threshold**: 0.35 (tunable)

---

## 🖥️ Run Locally

### 1️⃣ Clone the repository

```bash
git clone https://github.com/mitali-gautam05/TrueVision-AI.git
cd handwriting-detection
```

---

### 2️⃣ Create virtual environment

```bash
python -m venv venv
venv\Scripts\activate   # Windows
```

---

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

### 4️⃣ Run backend (FastAPI)

```bash
cd backend
python -m uvicorn main:app --reload
```

👉 Open: http://127.0.0.1:8000/docs

---

### 5️⃣ Run frontend (Streamlit)

```bash
cd frontend
streamlit run app.py
```

---

## 🌐 Deployment

* Backend: Render (FastAPI)
* Frontend: Streamlit Cloud

---

## 📈 Future Improvements

* 🔹 Model optimization 
* 🔹 Add logging & monitoring
* 🔹 Improve dataset & accuracy
* 🔹 Add user authentication
* 🔹 Deploy using Docker

---

## 🎯 Use Cases

* Signature verification
* Document fraud detection
* Forensic handwriting analysis

---

## Open to suggestions for improving model accuracy, performance, and deployment.

