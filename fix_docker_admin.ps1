# Fix Docker - Run as Administrator
# Right-click this file and select "Run with PowerShell as Administrator"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Docker Fix Script (Administrator)" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if running as Administrator
$currentPrincipal = New-Object Security.Principal.WindowsPrincipal([Security.Principal.WindowsIdentity]::GetCurrent())
$isAdmin = $currentPrincipal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

if (-not $isAdmin) {
    Write-Host "ERROR: This script must be run as Administrator!" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please:" -ForegroundColor Yellow
    Write-Host "1. Right-click on this file" -ForegroundColor White
    Write-Host "2. Select 'Run with PowerShell as Administrator'" -ForegroundColor White
    Write-Host ""
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host "Running with Administrator privileges..." -ForegroundColor Green
Write-Host ""

# Enable Virtual Machine Platform
Write-Host "Step 1: Enabling Virtual Machine Platform..." -ForegroundColor Yellow
try {
    dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart
    Write-Host "  Success!" -ForegroundColor Green
} catch {
    Write-Host "  Warning: $_" -ForegroundColor Yellow
}

Write-Host ""

# Enable WSL
Write-Host "Step 2: Enabling Windows Subsystem for Linux..." -ForegroundColor Yellow
try {
    dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
    Write-Host "  Success!" -ForegroundColor Green
} catch {
    Write-Host "  Warning: $_" -ForegroundColor Yellow
}

Write-Host ""

# Set WSL 2 as default
Write-Host "Step 3: Setting WSL 2 as default..." -ForegroundColor Yellow
try {
    wsl --set-default-version 2
    Write-Host "  Success!" -ForegroundColor Green
} catch {
    Write-Host "  Warning: $_" -ForegroundColor Yellow
}

Write-Host ""

# Check current status
Write-Host "Step 4: Checking current status..." -ForegroundColor Yellow
Write-Host ""
Write-Host "Virtualization Status:" -ForegroundColor Cyan
try {
    $virt = Get-ComputerInfo | Select-Object HyperVRequirementVirtualizationFirmwareEnabled
    if ($virt.HyperVRequirementVirtualizationFirmwareEnabled) {
        Write-Host "  Virtualization in BIOS: ENABLED" -ForegroundColor Green
    } else {
        Write-Host "  Virtualization in BIOS: DISABLED" -ForegroundColor Red
        Write-Host "  You need to enable it in BIOS settings!" -ForegroundColor Yellow
    }
} catch {
    Write-Host "  Could not check virtualization status" -ForegroundColor Yellow
}

Write-Host ""

# Check WSL status
Write-Host "WSL Status:" -ForegroundColor Cyan
wsl --status

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  IMPORTANT NEXT STEPS" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$needsRestart = $false

# Check if features were just enabled
$featureCheck = dism.exe /online /get-featureinfo /featurename:VirtualMachinePlatform
if ($featureCheck -match "Restart Required") {
    $needsRestart = $true
}

if ($needsRestart) {
    Write-Host "A RESTART IS REQUIRED for changes to take effect!" -ForegroundColor Red
    Write-Host ""
    Write-Host "After restart:" -ForegroundColor Yellow
    Write-Host "1. Docker Desktop will start automatically" -ForegroundColor White
    Write-Host "2. Wait for Docker whale icon in system tray" -ForegroundColor White
    Write-Host "3. Test with: docker ps" -ForegroundColor White
    Write-Host ""
    $restart = Read-Host "Do you want to restart now? (Y/N)"
    if ($restart -eq "Y" -or $restart -eq "y") {
        Write-Host ""
        Write-Host "Restarting in 10 seconds..." -ForegroundColor Yellow
        Write-Host "Save your work now!" -ForegroundColor Red
        Start-Sleep -Seconds 10
        Restart-Computer -Force
    } else {
        Write-Host ""
        Write-Host "Please restart manually when ready." -ForegroundColor Yellow
    }
} else {
    Write-Host "Features are already enabled!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Trying to start Docker Desktop..." -ForegroundColor Yellow
    Start-Process "C:\Program Files\Docker\Docker\Docker Desktop.exe"
    Write-Host ""
    Write-Host "Wait 30-60 seconds for Docker to start," -ForegroundColor Cyan
    Write-Host "then check the system tray for Docker whale icon." -ForegroundColor Cyan
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Script completed!" -ForegroundColor Green
Write-Host ""
Read-Host "Press Enter to close"

