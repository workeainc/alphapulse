# Backup Docker Volumes (Works WITHOUT Docker Running!)
# This copies your entire Docker data directory

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "  DOCKER VOLUMES BACKUP" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$sourcePath = "C:\Users\EA Soft Lab\AppData\Local\Docker\wsl\data"
$backupPath = "D:\AlphaPulse_Database_Backup\docker_data_backup_$(Get-Date -Format 'yyyy-MM-dd_HHmmss')"

# Check source exists
if (!(Test-Path $sourcePath)) {
    Write-Host "❌ Source directory not found!" -ForegroundColor Red
    Write-Host "   Expected: $sourcePath" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Checking alternative locations..." -ForegroundColor Cyan
    
    $altPath = "C:\Users\EA Soft Lab\AppData\Local\Docker\wsl"
    if (Test-Path $altPath) {
        Write-Host "✅ Found Docker WSL directory" -ForegroundColor Green
        Get-ChildItem $altPath -Recurse -Directory | Select-Object FullName
    }
    
    Read-Host "`nPress Enter to exit"
    exit 1
}

# Get size
Write-Host "Calculating size..." -ForegroundColor Yellow
$size = (Get-ChildItem $sourcePath -Recurse -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum).Sum
$sizeGB = [math]::Round($size / 1GB, 2)

Write-Host "Source: $sourcePath" -ForegroundColor White
Write-Host "Size: $sizeGB GB" -ForegroundColor Cyan
Write-Host ""
Write-Host "Backup will be saved to:" -ForegroundColor Yellow
Write-Host "$backupPath" -ForegroundColor White
Write-Host ""

$confirm = Read-Host "Start backup? (Y/N)"

if ($confirm -ne "Y" -and $confirm -ne "y") {
    Write-Host "Backup cancelled." -ForegroundColor Yellow
    exit 0
}

Write-Host ""
Write-Host "Starting backup..." -ForegroundColor Green
Write-Host "This may take several minutes depending on data size..." -ForegroundColor Gray
Write-Host ""

try {
    # Create backup directory
    New-Item -ItemType Directory -Path $backupPath -Force | Out-Null
    
    # Copy with progress
    $files = Get-ChildItem -Path $sourcePath -Recurse -File
    $totalFiles = $files.Count
    $current = 0
    
    foreach ($file in $files) {
        $current++
        $percent = [math]::Round(($current / $totalFiles) * 100, 1)
        Write-Progress -Activity "Backing up Docker volumes" -Status "$percent% Complete" -PercentComplete $percent
        
        $relativePath = $file.FullName.Substring($sourcePath.Length)
        $destPath = Join-Path $backupPath $relativePath
        $destDir = Split-Path $destPath -Parent
        
        if (!(Test-Path $destDir)) {
            New-Item -ItemType Directory -Path $destDir -Force | Out-Null
        }
        
        Copy-Item $file.FullName -Destination $destPath -Force
    }
    
    Write-Progress -Activity "Backing up Docker volumes" -Completed
    
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Green
    Write-Host "  BACKUP COMPLETE!" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "Backup location: $backupPath" -ForegroundColor Cyan
    Write-Host "Size: $sizeGB GB" -ForegroundColor White
    Write-Host "Files backed up: $totalFiles" -ForegroundColor White
    Write-Host ""
    Write-Host "Your data is now safely backed up!" -ForegroundColor Green
    Write-Host "You can restore this even on a different computer." -ForegroundColor Cyan
    
} catch {
    Write-Host ""
    Write-Host "❌ Error during backup: $_" -ForegroundColor Red
}

Write-Host ""
Read-Host "Press Enter to close"


