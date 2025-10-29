#!/usr/bin/env pwsh
# AlphaPulse Backend Startup Script
# This starts the intelligent production backend with HEAD A fully implemented

Write-Host "`n============================================================" -ForegroundColor Cyan
Write-Host " AlphaPulse Intelligent Production Backend" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Features:" -ForegroundColor Yellow
Write-Host "  ✓ HEAD A: 69 technical indicators with weighted scoring" -ForegroundColor White
Write-Host "  ✓ 9-Head SDE Consensus System" -ForegroundColor White
Write-Host "  ✓ Adaptive timeframe selection (regime-based)" -ForegroundColor White
Write-Host "  ✓ Multi-stage quality filtering (98-99% rejection)" -ForegroundColor White
Write-Host "  ✓ Live Binance WebSocket streaming (1m candles)" -ForegroundColor White
Write-Host "  ✓ Real-time signal generation" -ForegroundColor White
Write-Host ""
Write-Host "Backend will start on: http://localhost:8000" -ForegroundColor Cyan
Write-Host "API Documentation: http://localhost:8000/docs" -ForegroundColor Gray
Write-Host ""
Write-Host "Starting server..." -ForegroundColor Yellow
Write-Host "============================================================`n" -ForegroundColor Cyan

# Set UTF-8 encoding for proper emoji/unicode support
$env:PYTHONIOENCODING = "utf-8"

# Change to backend directory
Set-Location $PSScriptRoot

# Start the backend
python main.py

