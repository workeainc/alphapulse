# Quick Database Backup Script
# Fast backup without statistics or prompts

$ErrorActionPreference = "Stop"
$BACKUP_DIR = "backups"
$DATE = Get-Date -Format "yyyy-MM-dd_HHmmss"

# Create backup directory
if (!(Test-Path $BACKUP_DIR)) { New-Item -ItemType Directory -Path $BACKUP_DIR | Out-Null }

Write-Host "⚡ Quick Backup Starting..." -ForegroundColor Yellow

try {
    docker ps | Out-Null
    docker exec alphapulse_postgres pg_dump -U alpha_emon -d alphapulse -F c -f /tmp/backup.dump
    docker cp alphapulse_postgres:/tmp/backup.dump "$BACKUP_DIR/quick_backup_$DATE.dump"
    
    $size = [math]::Round((Get-Item "$BACKUP_DIR/quick_backup_$DATE.dump").Length / 1MB, 2)
    Write-Host "✅ Backup completed! Size: $size MB" -ForegroundColor Green
    Write-Host "   Saved: $BACKUP_DIR/quick_backup_$DATE.dump" -ForegroundColor Cyan
} catch {
    Write-Host "❌ Backup failed: $_" -ForegroundColor Red
    exit 1
}

