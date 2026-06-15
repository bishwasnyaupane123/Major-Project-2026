#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────
#  QUANTUM ML ARENA – One-shot setup script for macOS
#  Run: chmod +x setup.sh && ./setup.sh
# ──────────────────────────────────────────────────────────────────────
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo ""
echo "🚀 QuantumML Arena – Setup Script"
echo "=================================="

# ── 1. Check Python ──
if ! command -v python3 &>/dev/null; then
  echo "❌ Python 3 not found. Install via: brew install python@3.11"
  exit 1
fi
PYTHON_VER=$(python3 -c 'import sys; print(sys.version_info.minor)')
echo "✅ Python 3.${PYTHON_VER} found"

# ── 2. Backend venv ──
echo ""
echo "📦 Setting up backend virtual environment..."
cd "$SCRIPT_DIR/backend"

if [ ! -d "venv" ]; then
  python3 -m venv venv
  echo "✅ Virtual environment created"
fi

source venv/bin/activate

echo "📦 Installing backend dependencies..."
pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet
echo "✅ Backend dependencies installed"

# ── 3. Create data folders ──
mkdir -p data models
echo "✅ data/ and models/ directories ready"

# ── 4. Check for datasets ──
echo ""
echo "🗄️  Checking datasets..."
MISSING=()
[ ! -f "data/breast_cancer.csv" ] && MISSING+=("breast_cancer.csv")
[ ! -f "data/Crop_recommendation.csv" ] && MISSING+=("Crop_recommendation.csv")
[ ! -f "data/imdb.csv" ] && MISSING+=("imdb.csv (optional)")

if [ ${#MISSING[@]} -gt 0 ]; then
  echo "⚠️  Missing datasets: ${MISSING[*]}"
  echo "   Place them in:  quantum-ml-app/backend/data/"
  echo "   Sources:"
  echo "   • Breast Cancer: kaggle.com/datasets/uciml/breast-cancer-wisconsin-data"
  echo "   • Crop Rec.:    kaggle.com/datasets/atharvaingle/crop-recommendation-dataset"
  echo "   • IMDB:         kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews"
else
  echo "✅ All datasets present"
fi

deactivate

# ── 5. Frontend ──
echo ""
echo "⚛️  Setting up frontend..."
cd "$SCRIPT_DIR/frontend"

if [ ! -d "node_modules" ]; then
  npm install --silent
  echo "✅ Frontend dependencies installed"
else
  echo "✅ Frontend already set up"
fi

echo ""
echo "════════════════════════════════════"
echo "✅ Setup complete!"
echo ""
echo "To start the app, run:"
echo "  ./start.sh"
echo "════════════════════════════════════"
