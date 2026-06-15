# ⚛️ QuantumML Arena – Classical vs Quantum Machine Learning

A full-stack web application that compares **classical machine learning** models with **hybrid quantum-classical** models on real-world datasets using **live inference**.

![QuantumML Arena](https://img.shields.io/badge/QuantumML-Arena-blueviolet?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.9+-blue?style=flat-square)
![Node](https://img.shields.io/badge/Node.js-18+-green?style=flat-square)
![React](https://img.shields.io/badge/React-18-61dafb?style=flat-square)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?style=flat-square)

---

## 📋 Table of Contents

- [Features](#-features)
- [Datasets & Models](#-datasets--models)
- [Prerequisites](#-prerequisites)
- [Step-by-Step Setup Guide](#-step-by-step-setup-guide)
- [Running the Application](#-running-the-application)
- [Troubleshooting](#-troubleshooting)
- [Project Structure](#-project-structure)

---

## ✨ Features

- **Side-by-side comparison** of classical vs quantum models
- **Live inference** with real-time predictions
- **Battle Arena** — gamified 10-round model competition
- **Interactive Charts** — bar charts, pie charts, accuracy comparisons
- **Circuit Visualization** — quantum circuit diagrams
- **Dark/Light theme** toggle
- **3 Datasets**: Breast Cancer, Crop Recommendation, IMDB Sentiment

---

## 🧠 Datasets & Models

| Dataset | Classical Model | Quantum Model |
|---------|----------------|---------------|
| 🩺 Breast Cancer | Classical DNN (Keras) | Hybrid QNN (6-qubit PennyLane) |
| 🌾 Crop Recommendation | Classical KNN | Quantum KNN (Swap Test) |
| 🎬 IMDB Sentiment | Classical Transformer (DistilBERT) | Quantum Transformer (DistilBERT + PennyLane) |

---

## 📦 Prerequisites

Make sure the following are installed on the target computer:

| Tool | Minimum Version | Check Command | Download |
|------|----------------|---------------|----------|
| **Python** | 3.9+ | `python3 --version` | [python.org](https://www.python.org/downloads/) |
| **pip** | 21+ | `pip3 --version` | Comes with Python |
| **Node.js** | 18+ | `node --version` | [nodejs.org](https://nodejs.org/) |
| **npm** | 9+ | `npm --version` | Comes with Node.js |
| **Git** | Any | `git --version` | [git-scm.com](https://git-scm.com/) |

> **Note for Windows users**: Use **PowerShell** or **Command Prompt** (not Git Bash) for best compatibility. Replace `python3` with `python` and `pip3` with `pip` if needed.

---

## 🚀 Step-by-Step Setup Guide

### Step 1: Copy the Project Folder

Copy the entire `untitled folder` to the target PC (USB drive, Google Drive, zip file, etc.).  
Place it somewhere accessible, e.g. `Desktop`.

```
Desktop/
└── untitled folder/
    ├── quantum-ml-app/        ← The web app
    └── projectfinalhqnn/      ← Pre-trained model files
```

> ⚠️ **Important**: The `projectfinalhqnn` folder contains the pre-trained DistilBERT model (~270MB). Both folders must be at the same level.

---

### Step 2: Open Terminal / Command Prompt

**macOS/Linux:**
```bash
cd ~/Desktop/untitled\ folder/quantum-ml-app
```

**Windows (PowerShell):**
```powershell
cd "$HOME\Desktop\untitled folder\quantum-ml-app"
```

---

### Step 3: Set Up the Backend (Python)

#### 3a. Create a Python virtual environment

**macOS/Linux:**
```bash
cd backend
python3 -m venv venv
source venv/bin/activate
```

**Windows:**
```powershell
cd backend
python -m venv venv
.\venv\Scripts\Activate.ps1
```

> You should see `(venv)` at the beginning of your terminal prompt.

#### 3b. Install Python dependencies

```bash
pip install -r requirements.txt
```

> ⏳ This will take **5–15 minutes** depending on internet speed.  
> The largest packages are `torch` (~2GB), `transformers`, `qiskit`, and `qiskit-aer`.

#### 3c. Verify installation

```bash
python -c "import torch; import pennylane; import qiskit; import transformers; print('All imports OK ✅')"
```

---

### Step 4: Set Up the Frontend (Node.js)

Open a **new terminal window/tab** (keep the backend terminal open):

**macOS/Linux:**
```bash
cd ~/Desktop/untitled\ folder/quantum-ml-app/frontend
npm install
```

**Windows:**
```powershell
cd "$HOME\Desktop\untitled folder\quantum-ml-app\frontend"
npm install
```

> ⏳ This takes about **1–2 minutes**.

---

### Step 5: Start the Application

You need **two terminal windows** running simultaneously.

#### Terminal 1 — Backend Server

```bash
cd backend
source venv/bin/activate        # macOS/Linux
# OR
.\venv\Scripts\Activate.ps1     # Windows

uvicorn main:app --reload --port 8000
```

✅ You should see:
```
INFO:     Uvicorn running on http://127.0.0.1:8000
🚀 Bootstrapping datasets...
INFO:     Application startup complete.
```

#### Terminal 2 — Frontend Server

```bash
cd frontend
npm run dev
```

✅ You should see:
```
VITE ready in 200 ms
➜  Local:   http://localhost:5173/
```

---

### Step 6: Open the App

Open your browser and go to:

```
http://localhost:5173
```

🎉 **The QuantumML Arena should now be running!**

---

## 🛑 Stopping the Application

Press `Ctrl + C` in both terminal windows to stop the servers.

---

## 🔧 Troubleshooting

### ❌ "Module not found" errors
```bash
# Make sure venv is activated
source venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
```

### ❌ "Port already in use"
```bash
# macOS/Linux — kill existing processes
lsof -ti:8000 | xargs kill -9
lsof -ti:5173 | xargs kill -9

# Windows
netstat -aon | findstr :8000
taskkill /PID <PID_NUMBER> /F
```

### ❌ scikit-learn version warning
The pre-trained KNN models were trained with `scikit-learn==1.6.1`. If you see version mismatch warnings, make sure the correct version is installed:
```bash
pip install scikit-learn==1.6.1
```

### ❌ Breast Cancer Battle shows "Error loading"
This was fixed. If it happens, restart the backend server:
```bash
# Kill and restart
lsof -ti:8000 | xargs kill -9
uvicorn main:app --reload --port 8000
```

### ❌ "torch" installation fails on Windows
Try installing PyTorch separately first:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### ❌ IMDB Transformer loads slowly
The first prediction with the IMDB models loads the DistilBERT model (~270MB) into memory. This may take **30–60 seconds** on the first run. Subsequent predictions are much faster.

### ❌ "Pre-trained sentiment model directory not found"
Make sure the `projectfinalhqnn` folder is at the same level as `quantum-ml-app`:
```
untitled folder/
├── quantum-ml-app/
└── projectfinalhqnn/
    └── classicalsentimentanlaysis/
        └── sentiment_model/     ← DistilBERT model files
```

---

## 📁 Project Structure

```
quantum-ml-app/
├── backend/
│   ├── main.py                    # FastAPI server (API endpoints)
│   ├── requirements.txt           # Python dependencies
│   ├── data/                      # Dataset CSV files
│   │   ├── breast_cancer.csv
│   │   ├── Crop_recommendation.csv
│   │   └── imdb.csv
│   ├── models/                    # Pre-trained model artifacts
│   │   ├── cknn_artifacts/        # Classical KNN (scaler, LDA, model)
│   │   ├── qknn_checkpoints/     # Quantum KNN (training data, labels)
│   │   ├── hqnn_model/           # Hybrid QNN (PennyLane weights)
│   │   └── classical_dnn/        # Classical DNN (Keras model)
│   └── wrappers/                  # Model wrapper classes
│       ├── base.py                # BaseModel interface
│       ├── classical_dnn.py
│       ├── hqnn.py
│       ├── classical_knn.py
│       ├── quantum_knn.py
│       ├── pretrained_transformer.py
│       └── quantum_transformer.py
├── frontend/
│   ├── src/
│   │   ├── App.tsx                # Main application
│   │   ├── api.ts                 # API client
│   │   ├── index.css              # Full design system
│   │   └── components/
│   │       ├── InputForm.tsx       # Dataset input sliders
│   │       ├── ResultCard.tsx      # Prediction result cards
│   │       ├── BattleArena.tsx     # Battle mode overlay
│   │       ├── Charts.tsx          # Chart components
│   │       └── UI.tsx              # Shared UI components
│   ├── package.json
│   └── index.html
└── README.md                      # This file
```

---

## 📝 Quick Reference Commands

| Action | Command |
|--------|---------|
| Start backend | `cd backend && source venv/bin/activate && uvicorn main:app --reload --port 8000` |
| Start frontend | `cd frontend && npm run dev` |
| Open app | http://localhost:5173 |
| Stop servers | `Ctrl + C` in each terminal |
| Reinstall Python deps | `pip install -r requirements.txt` |
| Reinstall Node deps | `npm install` |

---

## 👨‍💻 Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | React 18, TypeScript, Vite, Framer Motion, Chart.js |
| Backend | Python, FastAPI, Uvicorn |
| Classical ML | Scikit-learn, Keras/TensorFlow, HuggingFace Transformers |
| Quantum ML | PennyLane, Qiskit, Qiskit-Aer |
| Styling | Vanilla CSS with glassmorphism design system |

---

**Made with ❤️ for the Major Project — Hybrid Quantum-Classical Machine Learning Framework**
