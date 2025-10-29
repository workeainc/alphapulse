# AlphaPulse Database Backup Script
# This script backs up your PostgreSQL database and Redis data

$ErrorActionPreference = "Stop"

# Configuration
$BACKUP_DIR = "backups"
$DATE = Get-Date -Format "yyyy-MM-dd_HHmmss"
$POSTGRES_BACKUP = "$BACKUP_DIR/alphapulse_db_$DATE.dump"
$POSTGRES_SQL_BACKUP = "$BACKUP_DIR/alphapulse_db_$DATE.sql"
$REDIS_BACKUP = "$BACKUP_DIR/redis_$DATE.rdb"
$VOLUME_BACKUP = "$BACKUP_DIR/postgres_volume_$DATE.tar.gz"

Write-Host "🔒 AlphaPulse Database Backup Script" -ForegroundColor Cyan
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host ""

# Create backup directory if it doesn't exist
if (!(Test-Path $BACKUP_DIR)) {
    Write-Host "📁 Creating backup directory..." -ForegroundColor Yellow
    New-Item -ItemType Directory -Path $BACKUP_DIR | Out-Null
}

# Check if Docker is running
Write-Host "🐳 Checking Docker status..." -ForegroundColor Yellow
try {
    docker ps | Out-Null
} catch {
    Write-Host "❌ ERROR: Docker is not running!" -ForegroundColor Red
    Write-Host "   Please start Docker Desktop and try again." -ForegroundColor Red
    exit 1
}

# Check if containers are running
$postgresRunning = docker ps --filter "name=alphapulse_postgres" --format "{{.Names}}"
$redisRunning = docker ps --filter "name=alphapulse_redis" --format "{{.Names}}"

if (!$postgresRunning) {
    Write-Host "⚠️  WARNING: PostgreSQL container is not running!" -ForegroundColor Red
    Write-Host "   Start containers with: docker-compose up -d" -ForegroundColor Yellow
    exit 1
}

Write-Host "✅ Docker is running" -ForegroundColor Green
Write-Host ""

# Backup PostgreSQL Database (Custom Format)
Write-Host "📦 Backing up PostgreSQL database (custom format)..." -ForegroundColor Yellow
try {
    docker exec alphapulse_postgres pg_dump -U alpha_emon -d alphapulse -F c -f /tmp/backup.dump
    docker cp alphapulse_postgres:/tmp/backup.dump $POSTGRES_BACKUP
    Write-Host "   ✅ PostgreSQL backup saved: $POSTGRES_BACKUP" -ForegroundColor Green
    
    # Get file size
    $size = (Get-Item $POSTGRES_BACKUP).Length / 1MB
    Write-Host "   📊 Backup size: $([math]::Round($size, 2)) MB" -ForegroundColor Cyan
} catch {
    Write-Host "   ❌ Failed to backup PostgreSQL: $_" -ForegroundColor Red
}

# Backup PostgreSQL Database (SQL Format - Human Readable)
Write-Host "📝 Backing up PostgreSQL database (SQL format)..." -ForegroundColor Yellow
try {
    docker exec alphapulse_postgres pg_dump -U alpha_emon -d alphapulse > $POSTGRES_SQL_BACKUP
    Write-Host "   ✅ SQL backup saved: $POSTGRES_SQL_BACKUP" -ForegroundColor Green
    
    # Get file size
    $size = (Get-Item $POSTGRES_SQL_BACKUP).Length / 1MB
    Write-Host "   📊 Backup size: $([math]::Round($size, 2)) MB" -ForegroundColor Cyan
} catch {
    Write-Host "   ❌ Failed to backup SQL: $_" -ForegroundColor Red
}

# Backup Redis (if running)
if ($redisRunning) {
    Write-Host "🔴 Backing up Redis data..." -ForegroundColor Yellow
    try {
        docker exec alphapulse_redis redis-cli SAVE | Out-Null
        docker cp alphapulse_redis:/data/dump.rdb $REDIS_BACKUP
        Write-Host "   ✅ Redis backup saved: $REDIS_BACKUP" -ForegroundColor Green
    } catch {
        Write-Host "   ⚠️  Redis backup skipped (not critical)" -ForegroundColor Yellow
    }
} else {
    Write-Host "⚠️  Redis container not running, skipping..." -ForegroundColor Yellow
}

# Get database statistics
Write-Host ""
Write-Host "📊 Database Statistics:" -ForegroundColor Cyan
Write-Host "======================" -ForegroundColor Cyan

try {
    Write-Host ""
    Write-Host "Database Size:" -ForegroundColor Yellow
    docker exec alphapulse_postgres psql -U alpha_emon -d alphapulse -c "SELECT pg_size_pretty(pg_database_size('alphapulse')) as database_size;"
    
    Write-Host ""
    Write-Host "Table Count:" -ForegroundColor Yellow
    docker exec alphapulse_postgres psql -U alpha_emon -d alphapulse -t -c "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public';"
    
    Write-Host ""
    Write-Host "Tables:" -ForegroundColor Yellow
    docker exec alphapulse_postgres psql -U alpha_emon -d alphapulse -c "\dt"
} catch {
    Write-Host "   ⚠️  Could not retrieve database statistics" -ForegroundColor Yellow
}

# Summary
Write-Host ""
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host "✅ BACKUP COMPLETED!" -ForegroundColor Green
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Backup files created:" -ForegroundColor Yellow
Write-Host "  📦 Custom dump: $POSTGRES_BACKUP" -ForegroundColor White
Write-Host "  📝 SQL format:  $POSTGRES_SQL_BACKUP" -ForegroundColor White
if ($redisRunning) {
    Write-Host "  🔴 Redis data:  $REDIS_BACKUP" -ForegroundColor White
}
Write-Host ""
Write-Host "💡 TIP: Keep these backups in a safe location!" -ForegroundColor Cyan
Write-Host "    Consider copying to an external drive or cloud storage." -ForegroundColor Cyan
Write-Host ""
Write-Host "📅 Backup Date: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Gray
Write-Host ""

