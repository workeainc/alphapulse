#!/bin/bash
# AlphaPulse Backend Startup Script
# This starts the intelligent production backend with HEAD A fully implemented

echo ""
echo "============================================================"
echo " AlphaPulse Intelligent Production Backend"
echo "============================================================"
echo ""
echo "Features:"
echo "  ✓ HEAD A: 69 technical indicators with weighted scoring"
echo "  ✓ 9-Head SDE Consensus System"
echo "  ✓ Adaptive timeframe selection (regime-based)"
echo "  ✓ Multi-stage quality filtering (98-99% rejection)"
echo "  ✓ Live Binance WebSocket streaming (1m candles)"
echo "  ✓ Real-time signal generation"
echo ""
echo "Backend will start on: http://localhost:8000"
echo "API Documentation: http://localhost:8000/docs"
echo ""
echo "Starting server..."
echo "============================================================"
echo ""

# Change to script directory
cd "$(dirname "$0")"

# Set UTF-8 encoding
export PYTHONIOENCODING=utf-8

# Start the backend
python main.py

