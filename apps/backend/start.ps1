# AlphaPulse MTF System Startup Script
# Author: AI Assistant
# Date: October 27, 2025

Write-Host ""
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "  ALPHAPULSE MTF ENTRY SYSTEM - STARTUP" -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host ""

# Check Docker services
Write-Host "[1/5] Checking Docker services..." -ForegroundColor Yellow

$postgres = docker ps --filter "name=alphapulse_postgres" --format "{{.Status}}" 2>$null
$redis = docker ps --filter "name=bowery_redis" --format "{{.Status}}" 2>$null

if ($postgres -match "Up") {
    Write-Host "  [OK] PostgreSQL is running" -ForegroundColor Green
} else {
    Write-Host "  [ERROR] PostgreSQL is not running" -ForegroundColor Red
    Write-Host "  Starting PostgreSQL..." -ForegroundColor Yellow
    docker start alphapulse_postgres | Out-Null
    Start-Sleep -Seconds 3
}

if ($redis -match "Up") {
    Write-Host "  [OK] Redis is running" -ForegroundColor Green
} else {
    Write-Host "  [ERROR] Redis is not running" -ForegroundColor Red
    Write-Host "  Starting Redis..." -ForegroundColor Yellow
    docker start bowery_redis | Out-Null
    Start-Sleep -Seconds 2
}

# Check database connection
Write-Host ""
Write-Host "[2/5] Verifying database connection..." -ForegroundColor Yellow

$dbCheck = docker exec -i alphapulse_postgres psql -U alpha_emon -d alphapulse -c "SELECT 1;" 2>$null
if ($LASTEXITCODE -eq 0) {
    Write-Host "  [OK] Database accessible" -ForegroundColor Green
} else {
    Write-Host "  [ERROR] Cannot connect to database" -ForegroundColor Red
    Write-Host "  Please check database logs: docker logs alphapulse_postgres" -ForegroundColor Yellow
    exit 1
}

# Check configuration files
Write-Host ""
Write-Host "[3/5] Checking configuration files..." -ForegroundColor Yellow

if (Test-Path "config\mtf_config.yaml") {
    Write-Host "  [OK] mtf_config.yaml found" -ForegroundColor Green
} else {
    Write-Host "  [ERROR] mtf_config.yaml not found" -ForegroundColor Red
    exit 1
}

if (Test-Path "config\symbol_config.yaml") {
    Write-Host "  [OK] symbol_config.yaml found" -ForegroundColor Green
} else {
    Write-Host "  [ERROR] symbol_config.yaml not found" -ForegroundColor Red
    exit 1
}

# Check MTF tables
Write-Host ""
Write-Host "[4/5] Verifying MTF database tables..." -ForegroundColor Yellow

$tableCheck = docker exec -i alphapulse_postgres psql -U alpha_emon -d alphapulse -c "\dt ai_signals_mtf" 2>$null
if ($LASTEXITCODE -eq 0) {
    Write-Host "  [OK] ai_signals_mtf table exists" -ForegroundColor Green
} else {
    Write-Host "  [WARNING] ai_signals_mtf table not found" -ForegroundColor Yellow
    Write-Host "  Run migrations: Get-Content 'src\database\migrations\101_mtf_entry_fields.sql' | docker exec -i alphapulse_postgres psql -U alpha_emon -d alphapulse" -ForegroundColor Yellow
}

# Start system
Write-Host ""
Write-Host "[5/5] Starting AlphaPulse MTF System..." -ForegroundColor Yellow
Write-Host ""
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "  SYSTEM STARTING - Monitor logs below" -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Press Ctrl+C to stop the system" -ForegroundColor Yellow
Write-Host ""

# Run the system
try {
    python main_scaled.py
} catch {
    Write-Host ""
    Write-Host "================================================================================" -ForegroundColor Red
    Write-Host "  ERROR: System failed to start" -ForegroundColor Red
    Write-Host "================================================================================" -ForegroundColor Red
    Write-Host ""
    Write-Host "Error details: $_" -ForegroundColor Red
    Write-Host ""
    Write-Host "Troubleshooting steps:" -ForegroundColor Yellow
    Write-Host "1. Check Python dependencies: pip install -r requirements.txt" -ForegroundColor Yellow
    Write-Host "2. Check Docker services: docker ps" -ForegroundColor Yellow
    Write-Host "3. Check database: docker logs alphapulse_postgres" -ForegroundColor Yellow
    Write-Host "4. Check Redis: docker logs bowery_redis" -ForegroundColor Yellow
    Write-Host ""
    exit 1
}

Write-Host ""
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "  SYSTEM STOPPED" -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host ""

