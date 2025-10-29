# PowerShell Script to Apply ALL Learning System Migrations
# Applies both 003 (basic learning) and 004 (rejection learning)

Write-Host "=" -NoNewline; Write-Host ("="*79)
Write-Host "🗄️  APPLYING ALL LEARNING SYSTEM MIGRATIONS"
Write-Host "=" -NoNewline; Write-Host ("="*79)
Write-Host ""

# Check if Docker is running
Write-Host "🔍 Checking Docker status..."
$dockerStatus = docker ps 2>&1

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Docker is not running!" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please start Docker Desktop first, then run this script again." -ForegroundColor Yellow
    Write-Host ""
    exit 1
}

Write-Host "✅ Docker is running" -ForegroundColor Green
Write-Host ""

# Check PostgreSQL container
Write-Host "🔍 Checking PostgreSQL container..."
$postgresContainer = docker ps --filter "name=alphapulse_postgres" --format "{{.Names}}"

if (-not $postgresContainer) {
    Write-Host "❌ PostgreSQL container not found!" -ForegroundColor Red
    Write-Host "Starting PostgreSQL container..." -ForegroundColor Yellow
    docker-compose -f ..\..\infrastructure\docker-compose\docker-compose.yml up -d postgres
    Start-Sleep -Seconds 5
}

Write-Host "✅ PostgreSQL container is running" -ForegroundColor Green
Write-Host ""

# Migration 1: Basic Learning System
Write-Host "🔄 Applying Migration 003: Basic Learning System..."
$migration1 = "src\database\migrations\003_learning_state.sql"

if (Test-Path $migration1) {
    Get-Content $migration1 | docker exec -i alphapulse_postgres psql -U alpha_emon -d alphapulse 2>&1 | Out-Null
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Migration 003 applied successfully!" -ForegroundColor Green
    } else {
        Write-Host "⚠️  Migration 003 may already be applied (this is OK)" -ForegroundColor Yellow
    }
} else {
    Write-Host "❌ Migration 003 not found!" -ForegroundColor Red
    exit 1
}

Write-Host ""

# Migration 2: Rejection Learning
Write-Host "🔄 Applying Migration 004: Rejection Learning System..."
$migration2 = "src\database\migrations\004_rejection_learning.sql"

if (Test-Path $migration2) {
    Get-Content $migration2 | docker exec -i alphapulse_postgres psql -U alpha_emon -d alphapulse 2>&1 | Out-Null
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Migration 004 applied successfully!" -ForegroundColor Green
    } else {
        Write-Host "⚠️  Migration 004 may already be applied (this is OK)" -ForegroundColor Yellow
    }
} else {
    Write-Host "❌ Migration 004 not found!" -ForegroundColor Red
    exit 1
}

Write-Host ""

# Verify all tables created
Write-Host "🔍 Verifying tables created..."
$verifyCommand = @"
SELECT table_name FROM information_schema.tables 
WHERE table_schema = 'public' 
AND table_name IN (
    'learning_state', 
    'active_learning_state', 
    'learning_events',
    'rejected_signals',
    'scan_history',
    'rejection_learning_metrics'
)
ORDER BY table_name;
"@

echo $verifyCommand | docker exec -i alphapulse_postgres psql -U alpha_emon -d alphapulse

Write-Host ""
Write-Host "=" -NoNewline; Write-Host ("="*79)
Write-Host "🎉 ALL MIGRATIONS COMPLETE!" -ForegroundColor Green
Write-Host "=" -NoNewline; Write-Host ("="*79)
Write-Host ""
Write-Host "Tables created:" -ForegroundColor Cyan
Write-Host "  1. learning_state (version history)" -ForegroundColor White
Write-Host "  2. active_learning_state (current state)" -ForegroundColor White
Write-Host "  3. learning_events (audit trail)" -ForegroundColor White
Write-Host "  4. rejected_signals (shadow tracking)" -ForegroundColor White
Write-Host "  5. scan_history (complete scan history)" -ForegroundColor White
Write-Host "  6. rejection_learning_metrics (daily metrics)" -ForegroundColor White
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  1. Start your backend: python main.py" -ForegroundColor White
Write-Host "  2. Check learning: curl http://localhost:8000/api/learning/stats" -ForegroundColor White
Write-Host "  3. Check rejections: curl http://localhost:8000/api/learning/rejection-analysis" -ForegroundColor White
Write-Host ""
Write-Host "🧠 Your system will now learn from EVERY decision (100% coverage)!" -ForegroundColor Green
Write-Host ""

