# 🧠 Cancer Microenvironment Classification — AI-Powered scRNA-seq Analysis

A full-stack deep-learning application for classifying **single-cell RNA sequencing (scRNA-seq)** data from the **cancer tumor microenvironment (TME)**.
Built using **PyTorch**, **ONNX**, and **FastAPI**, with a responsive **HTML/JS front-end**.

---

## 🚀 Overview

## App Link : https://hw-classification.onrender.com 

This project simulates a real-world bioinformatics pipeline — from model training to deployment.

* **Backend**: FastAPI + ONNX runtime
* **Model**: Deep MLP (PyTorch → exported to ONNX)
* **Frontend**: HTML + JavaScript (AJAX calls to FastAPI)


It classifies individual cells into **Cancer**, **T_Cell**, or **Fibroblast**, based on gene-expression features.

---

## 🧩 Features

✅ Deep neural-network classification (8-layer MLP)
✅ End-to-end preprocessing (scaling, encoding, feature alignment)
✅ Dual inference backends (ONNX or PyTorch fallback)
✅ Interactive web UI for real-time prediction
✅ Modular structure — easy to extend for new datasets
✅ Ready for cloud deployment (Railway, Docker, or Azure)

---

## 🧠 Tech Stack

| Layer      | Technology                  |
| :--------- | :-------------------------- |
| Frontend   | HTML, CSS, JavaScript       |
| Backend    | FastAPI, Uvicorn            |
| ML         | PyTorch, ONNX, ONNX Runtime |
| Data       | scikit-learn, NumPy, Pandas |
| Deployment | Docker, Railway (optional)  |

---

## 🧪 Local Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/<your-username>/hw-classification.git
cd hw-classification
```

### 2️⃣ Create a Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate        # On Windows
# or
source venv/bin/activate     # On macOS/Linux
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Run the API

```bash
uvicorn src.main:app --reload
```

Your API will start at 👉 **[http://127.0.0.1:8000](http://127.0.0.1:8000)**

---

## 🧭 Folder Structure

```
hw-classification/
│
├── models/                   # trained model artifacts
│   ├── cancer_classifier.pth
│   └── cancer_classifier.onnx
│
├── scalers/                  # preprocessing artifacts
│   ├── scaler_X.pkl
│   └── label_encoder.pkl
│
├── src/                      # backend code
│   ├── main.py               # FastAPI entrypoint
│   ├── inference.py          # ONNX + Torch inference logic
│   └── __init__.py
│
├── static/                   # frontend assets
│   ├── style.css
│   └── script.js
│
├── templates/                # HTML frontend
│   └── index.html
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Usage (Web UI)

1. Start the server → `uvicorn src.main:app --reload`
2. Open your browser → [http://127.0.0.1:8000](http://127.0.0.1:8000)
3. Fill in the following fields:

   * Gene_E_Housekeeping
   * Gene_A_Oncogene
   * Gene_B_Immune
   * Gene_C_Stromal
   * Gene_D_Therapy
   * Pathway_Score_Inflam
   * UMAP_1
   * Disease_Status (Tumor / Normal)
4. Click **Predict** to see:

   ```
   Prediction: Cancer
   Probabilities: [0.997, 0.002, 0.001]
   ```

---

## 🧰 API Endpoints

| Method | Endpoint   | Description                        |
| :----- | :--------- | :--------------------------------- |
| `GET`  | `/`        | Renders web UI                     |
| `POST` | `/predict` | Returns classification result JSON |

Example `POST` body:

```json
{
  "Gene_E_Housekeeping": 5.56,
  "Gene_A_Oncogene": 14.88,
  "Gene_B_Immune": 10.53,
  "Gene_C_Stromal": 3.20,
  "Gene_D_Therapy": 9.92,
  "Pathway_Score_Inflam": 9.58,
  "UMAP_1": 7.81,
  "Disease_Status": "Tumor"
}
```

Response:

```json
{
  "backend": "onnx",
  "prediction": "Cancer",
  "classes": ["Cancer", "Fibroblast", "T_Cell"],
  "probabilities": [0.997, 0.002, 0.001]
}
```

---

## 🧱 Model Summary

* Architecture: 8-layer MLP
* Hidden sizes: 512 → 256 → 128 → 64 → 32 → Output(3)
* Activation: LeakyReLU
* Regularization: BatchNorm + Dropout(0.3)
* Optimizer: AdamW
* Loss: CrossEntropyLoss
* Export: ONNX (for production inference)

---

## 🧮 Dataset Reference

Synthetic scRNA-seq dataset simulating tumor microenvironment interactions:

* Features: 8 (Gene + Pathway + Embedding)
* Classes: Cancer, T_Cell, Fibroblast
* Source: Synthetic (log-normal generated for benchmark use)

---

**Procfile**

```
web: uvicorn src.main:app --host 0.0.0.0 --port 8000
```

