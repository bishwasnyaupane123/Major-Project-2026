#!/bin/bash
# QuantumML Arena — Quick Start Script
# Run this script to start both servers at once

echo "⚛️  QuantumML Arena — Starting..."
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# --- Backend ---
echo "🔧 Starting Backend Server..."
cd "$SCRIPT_DIR/backend"

if [ ! -d "venv" ]; then
    echo "📦 Creating Python virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    echo "📦 Installing Python dependencies (this may take a while)..."
    pip install -r requirements.txt
else
    source venv/bin/activate
fi

uvicorn main:app --reload --port 8000 &
BACKEND_PID=$!
echo "✅ Backend started (PID: $BACKEND_PID)"

# --- Frontend ---
echo "🔧 Starting Frontend Server..."
cd "$SCRIPT_DIR/frontend"

if [ ! -d "node_modules" ]; then
    echo "📦 Installing Node.js dependencies..."
    npm install
fi

npm run dev &
FRONTEND_PID=$!
echo "✅ Frontend started (PID: $FRONTEND_PID)"

echo ""
echo "═══════════════════════════════════════════"
echo "  ⚛️  QuantumML Arena is running!"
echo "  🌐 Open: http://localhost:5173"
echo "  Press Ctrl+C to stop both servers"
echo "═══════════════════════════════════════════"
echo ""

# Wait and cleanup on Ctrl+C
trap "echo ''; echo 'Stopping servers...'; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; echo 'Done!'; exit 0" INT
wait
