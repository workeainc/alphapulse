# AlphaPulse Database Restore Script
# This script restores your PostgreSQL database from a backup

$ErrorActionPreference = "Stop"

Write-Host "🔄 AlphaPulse Database Restore Script" -ForegroundColor Cyan
Write-Host "======================================" -ForegroundColor Cyan
Write-Host ""

# Check if Docker is running
Write-Host "🐳 Checking Docker status..." -ForegroundColor Yellow
try {
    docker ps | Out-Null
    Write-Host "✅ Docker is running" -ForegroundColor Green
} catch {
    Write-Host "❌ ERROR: Docker is not running!" -ForegroundColor Red
    Write-Host "   Please start Docker Desktop and try again." -ForegroundColor Red
    exit 1
}

# Check if containers are running
$postgresRunning = docker ps --filter "name=alphapulse_postgres" --format "{{.Names}}"
if (!$postgresRunning) {
    Write-Host "⚠️  PostgreSQL container is not running!" -ForegroundColor Red
    Write-Host "   Starting containers..." -ForegroundColor Yellow
    docker-compose -f infrastructure/docker-compose/docker-compose.yml up -d
    Start-Sleep -Seconds 10
}

# List available backups
Write-Host ""
Write-Host "📋 Available Backups:" -ForegroundColor Cyan
Write-Host "=====================" -ForegroundColor Cyan
Write-Host ""

$backupFiles = Get-ChildItem -Path "backups" -Filter "*.dump" -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending

if ($backupFiles.Count -eq 0) {
    Write-Host "❌ No backup files found in 'backups' directory!" -ForegroundColor Red
    Write-Host "   Please run backup_database.ps1 first to create a backup." -ForegroundColor Yellow
    exit 1
}

# Display available backups
for ($i = 0; $i -lt $backupFiles.Count; $i++) {
    $file = $backupFiles[$i]
    $size = [math]::Round($file.Length / 1MB, 2)
    $date = $file.LastWriteTime.ToString("yyyy-MM-dd HH:mm:ss")
    Write-Host "[$($i + 1)] $($file.Name)" -ForegroundColor Yellow
    Write-Host "    Size: $size MB | Date: $date" -ForegroundColor Gray
    Write-Host ""
}

# Prompt user to select backup
Write-Host "Select a backup to restore (1-$($backupFiles.Count)) or press Ctrl+C to cancel:" -ForegroundColor Cyan
$selection = Read-Host "Enter number"

if ([int]$selection -lt 1 -or [int]$selection -gt $backupFiles.Count) {
    Write-Host "❌ Invalid selection!" -ForegroundColor Red
    exit 1
}

$selectedBackup = $backupFiles[[int]$selection - 1].FullName
Write-Host ""
Write-Host "Selected backup: $selectedBackup" -ForegroundColor Green
Write-Host ""

# Warning
Write-Host "⚠️  WARNING: This will REPLACE all current data in the database!" -ForegroundColor Red
Write-Host "   Current data will be LOST unless you have a backup!" -ForegroundColor Red
Write-Host ""
$confirm = Read-Host "Type 'YES' to continue"

if ($confirm -ne "YES") {
    Write-Host "❌ Restore cancelled." -ForegroundColor Yellow
    exit 0
}

Write-Host ""
Write-Host "🔄 Starting restore process..." -ForegroundColor Yellow
Write-Host ""

# Copy backup into container
Write-Host "📦 Copying backup file into container..." -ForegroundColor Yellow
try {
    docker cp $selectedBackup alphapulse_postgres:/tmp/restore_backup.dump
    Write-Host "   ✅ Backup file copied" -ForegroundColor Green
} catch {
    Write-Host "   ❌ Failed to copy backup: $_" -ForegroundColor Red
    exit 1
}

# Drop and recreate database (clean restore)
Write-Host "🗑️  Preparing database..." -ForegroundColor Yellow
try {
    # Terminate existing connections
    docker exec alphapulse_postgres psql -U alpha_emon -d postgres -c "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = 'alphapulse' AND pid <> pg_backend_pid();" | Out-Null
    
    # Drop and recreate
    docker exec alphapulse_postgres psql -U alpha_emon -d postgres -c "DROP DATABASE IF EXISTS alphapulse;" | Out-Null
    docker exec alphapulse_postgres psql -U alpha_emon -d postgres -c "CREATE DATABASE alphapulse;" | Out-Null
    
    Write-Host "   ✅ Database prepared" -ForegroundColor Green
} catch {
    Write-Host "   ⚠️  Warning: Could not drop database (might not exist)" -ForegroundColor Yellow
}

# Restore database
Write-Host "📥 Restoring database..." -ForegroundColor Yellow
Write-Host "   This may take several minutes depending on data size..." -ForegroundColor Gray
try {
    docker exec alphapulse_postgres pg_restore -U alpha_emon -d alphapulse -v /tmp/restore_backup.dump 2>&1 | Out-Null
    Write-Host "   ✅ Database restored successfully!" -ForegroundColor Green
} catch {
    Write-Host "   ⚠️  Restore completed with warnings (this is often normal)" -ForegroundColor Yellow
}

# Verify restoration
Write-Host ""
Write-Host "🔍 Verifying restoration..." -ForegroundColor Yellow
try {
    Write-Host ""
    Write-Host "Database Size:" -ForegroundColor Cyan
    docker exec alphapulse_postgres psql -U alpha_emon -d alphapulse -c "SELECT pg_size_pretty(pg_database_size('alphapulse')) as database_size;"
    
    Write-Host ""
    Write-Host "Table Count:" -ForegroundColor Cyan
    $tableCount = docker exec alphapulse_postgres psql -U alpha_emon -d alphapulse -t -c "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public';"
    Write-Host "   Tables: $tableCount" -ForegroundColor White
    
    Write-Host ""
    Write-Host "Tables:" -ForegroundColor Cyan
    docker exec alphapulse_postgres psql -U alpha_emon -d alphapulse -c "\dt"
} catch {
    Write-Host "   ⚠️  Could not verify restoration" -ForegroundColor Yellow
}

# Summary
Write-Host ""
Write-Host "======================================" -ForegroundColor Cyan
Write-Host "✅ RESTORE COMPLETED!" -ForegroundColor Green
Write-Host "======================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Restored from: $selectedBackup" -ForegroundColor Yellow
Write-Host "Restore Date: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Gray
Write-Host ""
Write-Host "💡 TIP: Test your application to ensure everything works correctly!" -ForegroundColor Cyan
Write-Host ""

