# Fix Docker Desktop Settings to Bypass Virtualization Check

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "  FIX DOCKER SETTINGS" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Stop Docker Desktop
Write-Host "1. Stopping Docker Desktop..." -ForegroundColor Yellow
Get-Process -Name "*Docker*" -ErrorAction SilentlyContinue | Stop-Process -Force
Start-Sleep -Seconds 3
Write-Host "   Done" -ForegroundColor Green
Write-Host ""

# Settings file path
$settingsPath = "$env:APPDATA\Docker\settings.json"

if (Test-Path $settingsPath) {
    Write-Host "2. Backing up current settings..." -ForegroundColor Yellow
    Copy-Item $settingsPath "$settingsPath.backup" -Force
    Write-Host "   Backup created: $settingsPath.backup" -ForegroundColor Green
    Write-Host ""
    
    Write-Host "3. Modifying settings..." -ForegroundColor Yellow
    try {
        $settings = Get-Content $settingsPath -Raw | ConvertFrom-Json
        
        # Force WSL2 backend and bypass checks
        $settings | Add-Member -NotePropertyName "wslEngineEnabled" -NotePropertyValue $true -Force
        $settings | Add-Member -NotePropertyName "displayedTutorial" -NotePropertyValue $true -Force
        $settings | Add-Member -NotePropertyName "skipUpdateToWSL2" -NotePropertyValue $false -Force
        
        # Save settings
        $settings | ConvertTo-Json -Depth 10 | Set-Content $settingsPath
        
        Write-Host "   Settings updated" -ForegroundColor Green
    } catch {
        Write-Host "   Error: $_" -ForegroundColor Red
    }
} else {
    Write-Host "2. Settings file not found, creating new one..." -ForegroundColor Yellow
    
    $newSettings = @{
        wslEngineEnabled = $true
        displayedTutorial = $true
        skipUpdateToWSL2 = $false
    }
    
    $newSettings | ConvertTo-Json -Depth 10 | Set-Content $settingsPath
    Write-Host "   New settings created" -ForegroundColor Green
}

Write-Host ""
Write-Host "4. Restarting Docker Desktop..." -ForegroundColor Yellow
Start-Process "C:\Program Files\Docker\Docker\Docker Desktop.exe"
Write-Host "   Docker Desktop starting..." -ForegroundColor Green
Write-Host ""

Write-Host "========================================" -ForegroundColor Green
Write-Host "  SETTINGS FIXED!" -ForegroundColor Green  
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Wait 30-60 seconds for Docker to start," -ForegroundColor Cyan
Write-Host "then check if the virtualization error is gone." -ForegroundColor Cyan
Write-Host ""

Read-Host "Press Enter to close"


