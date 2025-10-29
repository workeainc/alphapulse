# PowerShell Script to Apply Learning System Database Migration
# Run this after Docker is started

Write-Host "=" -NoNewline; Write-Host ("="*79)
Write-Host "🗄️  APPLYING LEARNING SYSTEM DATABASE MIGRATION"
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

# Check if PostgreSQL container is running
Write-Host "🔍 Checking PostgreSQL container..."
$postgresContainer = docker ps --filter "name=alphapulse_postgres" --format "{{.Names}}"

if (-not $postgresContainer) {
    Write-Host "❌ PostgreSQL container not found!" -ForegroundColor Red
    Write-Host ""
    Write-Host "Starting PostgreSQL container..." -ForegroundColor Yellow
    docker-compose -f ..\..\infrastructure\docker-compose\docker-compose.yml up -d postgres
    Start-Sleep -Seconds 5
}

Write-Host "✅ PostgreSQL container is running" -ForegroundColor Green
Write-Host ""

# Apply migration
Write-Host "🔄 Applying learning system migration..."
Write-Host ""

$migrationPath = "src\database\migrations\003_learning_state.sql"

if (Test-Path $migrationPath) {
    Get-Content $migrationPath | docker exec -i alphapulse_postgres psql -U alpha_emon -d alphapulse
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host ""
        Write-Host "✅ Migration applied successfully!" -ForegroundColor Green
        Write-Host ""
        
        # Verify tables created
        Write-Host "🔍 Verifying tables created..."
        $verifyCommand = @"
SELECT table_name FROM information_schema.tables 
WHERE table_schema = 'public' 
AND table_name IN ('learning_state', 'active_learning_state', 'learning_events')
ORDER BY table_name;
"@
        
        echo $verifyCommand | docker exec -i alphapulse_postgres psql -U alpha_emon -d alphapulse
        
        Write-Host ""
        Write-Host "=" -NoNewline; Write-Host ("="*79)
        Write-Host "🎉 MIGRATION COMPLETE!" -ForegroundColor Green
        Write-Host "=" -NoNewline; Write-Host ("="*79)
        Write-Host ""
        Write-Host "Next steps:"
        Write-Host "  1. Start your backend: python main.py" -ForegroundColor Cyan
        Write-Host "  2. Check learning system: curl http://localhost:8000/api/learning/stats" -ForegroundColor Cyan
        Write-Host ""
        
    } else {
        Write-Host ""
        Write-Host "❌ Migration failed!" -ForegroundColor Red
        Write-Host "Check the error messages above for details." -ForegroundColor Yellow
        Write-Host ""
        exit 1
    }
} else {
    Write-Host "❌ Migration file not found: $migrationPath" -ForegroundColor Red
    Write-Host "Make sure you're in the apps/backend directory" -ForegroundColor Yellow
    Write-Host ""
    exit 1
}

