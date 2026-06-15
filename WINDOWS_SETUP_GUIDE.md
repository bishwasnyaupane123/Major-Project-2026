# ⚛️ QuantumML Arena — Windows Setup Guide

Hello! This guide will help you easily set up and run the **QuantumML Arena** on your Windows laptop.

---

## 🛑 Step 0: Ensure Prerequisites

Before you start, make sure you have the following installed on your laptop:

1. **Python 3.9 or higher**
   - Download: [python.org/downloads](https://www.python.org/downloads/)
   - ⚠️ **CRITICAL:** When installing Python, make sure to check the box that says **"Add Python to PATH"** at the bottom of the installer window!
   
2. **Node.js (LTS Version, e.g., 18+ or 20+)**
   - Download: [nodejs.org](https://nodejs.org/)

To verify you have them installed, open the **Command Prompt** (press `Win + R`, type `cmd`, and press Enter) and run:
- `python --version`
- `node --version`
- `npm --version`

*(All of these should output a version number without errors. If not, reinstall them.)*

---

## 📁 Step 1: Extract the Files Properly

Make sure you extracted the `.zip` file into a folder on your Desktop (or any other accessible location). You should have a folder structure like this:
```
untitled folder/
├── quantum-ml-app/          ← The web app files (You should be inside this folder!)
└── projectfinalhqnn/        ← Pre-trained models (Must be next to the quantum-ml-app folder)
```

---

## 🚀 Step 2: The Easiest Way to Run (One-Click)

Inside the `quantum-ml-app` folder, you will find a file named **`start_windows.bat`**. 

1. **Double-click `start_windows.bat`**
2. The script will automatically:
   - Create a Python virtual environment.
   - Install all the heavy Python machine learning libraries (this might take 5-15 minutes the first time depending on your internet).
   - Install the Node.js frontend packages.
   - Start the backend and frontend servers in two separate windows.
3. Once both windows are loaded, the app will be accessible at:
   **http://localhost:5173** in your web browser.

> Note: Leave both black terminal windows open while you are using the app! Close them when you are done.

---

## 🛠️ Step 2 (Alternative): The Manual Way

If the batch script doesn't work for some reason, you can do it manually using **Command Prompt**.

### Termial 1: Setup Backend
Open a Command Prompt window. Type the following commands, pressing Enter after each one:

```cmd
cd "path\to\untitled folder\quantum-ml-app\backend"
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```
*Wait until it says "Application startup complete."*

### Terminal 2: Setup Frontend
Open a **second** Command Prompt window:

```cmd
cd "path\to\untitled folder\quantum-ml-app\frontend"
npm install
npm run dev
```

Finally, open your browser to **http://localhost:5173**

---

## ❓ Troubleshooting Common Errors

### ❌ "Command not found: python" or "npm is not recognized"
You did not check "Add Python to PATH" or didn't install Node.js. Reinstall Python/Node.js and ensure you add them to the system PATH.

### ❌ Error installing `torch` or `scikit-learn`
If you get a massive red error while installing Python dependencies, run this inside the `backend` folder after activating the virtual environment:
```cmd
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### ❌ "Port already in use"
Another server is running. Open Task Manager and end any rogue `node.exe` or `python.exe` background tasks, then start again.

### ❌ Pre-trained Sentiment Model Not Found
Make sure the `projectfinalhqnn` folder wasn't accidentally deleted or placed inside `quantum-ml-app`. They must sit side-by-side in the parent directory.
