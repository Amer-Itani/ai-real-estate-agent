# 🏠 AI Real Estate Agent

An end-to-end AI system that predicts house prices from natural language descriptions using:

* LLM (Groq) for feature extraction & interpretation
* Machine Learning model (Random Forest) for price prediction
* FastAPI for backend
* Streamlit for UI
* Docker for deployment

---

## 🚀 Features

* Natural language → structured features
* Missing feature detection (no silent defaults)
* ML prediction pipeline
* LLM-based interpretation of results
* Streamlit interactive UI

---

## 🧠 Architecture

User Query → LLM (Feature Extraction) → Validation → ML Model → LLM (Interpretation)

---

## ⚙️ Run Locally

### 1. Start FastAPI

uvicorn app.main:app --reload

### 2. Start Streamlit

streamlit run streamlit_app.py

---

## 🐳 Docker

Build:
docker build -t ai-real-estate .

Run:
docker run -p 8000:8000 -e GROQ_API_KEY=your_key_here ai-real-estate

---

## ☁️ Deployment

Deployed using Railway (FastAPI backend)

---

## 📌 Notes

* Model trained in Google Colab
* Dataset: Ames Housing
* Some dataset statistics are approximated for demo purposes
