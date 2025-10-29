# Download and Repair Docker Desktop
# This script downloads the latest Docker Desktop and runs it to repair the installation

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "  DOCKER DESKTOP REPAIR SCRIPT" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if running as admin
$currentPrincipal = New-Object Security.Principal.WindowsPrincipal([Security.Principal.WindowsIdentity]::GetCurrent())
$isAdmin = $currentPrincipal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

if (-not $isAdmin) {
    Write-Host "WARNING: Not running as Administrator" -ForegroundColor Yellow
    Write-Host "The installer will request elevation when it runs.`n" -ForegroundColor Gray
}

# Download URL for Docker Desktop
$dockerUrl = "https://desktop.docker.com/win/main/amd64/Docker%20Desktop%20Installer.exe"
$downloadPath = "$env:TEMP\DockerDesktopInstaller.exe"

Write-Host "Step 1: Downloading Docker Desktop Installer..." -ForegroundColor Yellow
Write-Host "This may take a few minutes (file is ~500MB)`n" -ForegroundColor Gray

try {
    # Use .NET WebClient for better progress
    $webClient = New-Object System.Net.WebClient
    
    # Register progress event
    Register-ObjectEvent -InputObject $webClient -EventName DownloadProgressChanged -SourceIdentifier WebClient.DownloadProgressChanged -Action {
        Write-Progress -Activity "Downloading Docker Desktop" -Status "$($EventArgs.ProgressPercentage)% Complete" -PercentComplete $EventArgs.ProgressPercentage
    } | Out-Null
    
    # Start download
    $webClient.DownloadFile($dockerUrl, $downloadPath)
    
    # Cleanup
    Unregister-Event -SourceIdentifier WebClient.DownloadProgressChanged
    $webClient.Dispose()
    
    Write-Host "   ✅ Download complete!" -ForegroundColor Green
    Write-Host ""
} catch {
    Write-Host "   ❌ Download failed: $_" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please download manually from:" -ForegroundColor Yellow
    Write-Host "https://www.docker.com/products/docker-desktop/" -ForegroundColor Cyan
    Write-Host ""
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host "Step 2: Running Docker Desktop Installer..." -ForegroundColor Yellow
Write-Host "The installer will:" -ForegroundColor Gray
Write-Host "  - Detect existing installation" -ForegroundColor Gray
Write-Host "  - Repair corrupted files" -ForegroundColor Gray
Write-Host "  - Fix registry keys" -ForegroundColor Gray
Write-Host "  - May ask to restart computer" -ForegroundColor Gray
Write-Host ""

$confirm = Read-Host "Ready to run installer? (Y/N)"

if ($confirm -ne "Y" -and $confirm -ne "y") {
    Write-Host "Installation cancelled." -ForegroundColor Yellow
    Write-Host "Installer saved at: $downloadPath" -ForegroundColor Cyan
    exit 0
}

Write-Host ""
Write-Host "Launching installer..." -ForegroundColor Cyan
Write-Host "(UAC prompt will appear - click YES)" -ForegroundColor Yellow
Write-Host ""

try {
    # Run installer
    Start-Process -FilePath $downloadPath -Wait -Verb RunAs
    
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Green
    Write-Host "  INSTALLATION COMPLETE!" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "Next steps:" -ForegroundColor Yellow
    Write-Host "1. RESTART YOUR COMPUTER (if installer didn't already)" -ForegroundColor White
    Write-Host "2. After restart, Docker Desktop should start automatically" -ForegroundColor White
    Write-Host "3. Look for Docker whale icon in system tray" -ForegroundColor White
    Write-Host "4. Test with: docker ps" -ForegroundColor White
    Write-Host ""
    
    $restart = Read-Host "Restart computer now? (Y/N)"
    
    if ($restart -eq "Y" -or $restart -eq "y") {
        Write-Host ""
        Write-Host "Restarting in 10 seconds..." -ForegroundColor Yellow
        Write-Host "SAVE YOUR WORK!" -ForegroundColor Red
        Start-Sleep -Seconds 10
        Restart-Computer -Force
    } else {
        Write-Host ""
        Write-Host "Please restart manually when ready." -ForegroundColor Yellow
    }
    
} catch {
    Write-Host "❌ Error running installer: $_" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please run the installer manually:" -ForegroundColor Yellow
    Write-Host $downloadPath -ForegroundColor Cyan
}

Write-Host ""
Read-Host "Press Enter to close"



