# COMPLETE Docker Fix Script - Run as Administrator
# This script will fix ALL Docker/WSL2 issues

# Check Administrator
$currentPrincipal = New-Object Security.Principal.WindowsPrincipal([Security.Principal.WindowsIdentity]::GetCurrent())
$isAdmin = $currentPrincipal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

if (-not $isAdmin) {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Red
    Write-Host "  MUST RUN AS ADMINISTRATOR!" -ForegroundColor Red
    Write-Host "========================================" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please:" -ForegroundColor Yellow
    Write-Host "1. Right-click on PowerShell" -ForegroundColor White
    Write-Host "2. Select 'Run as Administrator'" -ForegroundColor White
    Write-Host "3. Navigate to: D:\Emon Work\AlphaPuls\" -ForegroundColor White
    Write-Host "4. Run: .\COMPLETE_DOCKER_FIX.ps1" -ForegroundColor White
    Write-Host ""
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  COMPLETE DOCKER FIX SCRIPT" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Step 1: Enable Hyper-V
Write-Host "Step 1: Enabling Hyper-V Platform..." -ForegroundColor Yellow
try {
    Enable-WindowsOptionalFeature -Online -FeatureName Microsoft-Hyper-V -All -NoRestart
    Write-Host "  Hyper-V Enabled!" -ForegroundColor Green
} catch {
    Write-Host "  Note: $($_.Exception.Message)" -ForegroundColor Gray
}

# Step 2: Enable Virtual Machine Platform
Write-Host ""
Write-Host "Step 2: Enabling Virtual Machine Platform..." -ForegroundColor Yellow
try {
    Enable-WindowsOptionalFeature -Online -FeatureName VirtualMachinePlatform -All -NoRestart
    Write-Host "  Virtual Machine Platform Enabled!" -ForegroundColor Green
} catch {
    Write-Host "  Note: $($_.Exception.Message)" -ForegroundColor Gray
}

# Step 3: Enable WSL
Write-Host ""
Write-Host "Step 3: Enabling Windows Subsystem for Linux..." -ForegroundColor Yellow
try {
    Enable-WindowsOptionalFeature -Online -FeatureName Microsoft-Windows-Subsystem-Linux -All -NoRestart
    Write-Host "  WSL Enabled!" -ForegroundColor Green
} catch {
    Write-Host "  Note: $($_.Exception.Message)" -ForegroundColor Gray
}

# Step 4: Enable Hypervisor at boot
Write-Host ""
Write-Host "Step 4: Enabling Hypervisor at boot..." -ForegroundColor Yellow
try {
    bcdedit /set hypervisorlaunchtype auto
    Write-Host "  Hypervisor boot enabled!" -ForegroundColor Green
} catch {
    Write-Host "  Note: $($_.Exception.Message)" -ForegroundColor Gray
}

# Step 5: Set WSL 2 as default
Write-Host ""
Write-Host "Step 5: Setting WSL 2 as default version..." -ForegroundColor Yellow
try {
    wsl --set-default-version 2
    Write-Host "  WSL 2 set as default!" -ForegroundColor Green
} catch {
    Write-Host "  Note: Will work after restart" -ForegroundColor Gray
}

# Check status
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  CURRENT STATUS" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check virtualization
Write-Host "Virtualization Check:" -ForegroundColor Yellow
$virt = Get-ComputerInfo | Select-Object HyperVRequirementVirtualizationFirmwareEnabled, HyperVRequirementVMMonitorModeExtensions
if ($virt.HyperVRequirementVirtualizationFirmwareEnabled) {
    Write-Host "  BIOS Virtualization: ENABLED" -ForegroundColor Green
} else {
    Write-Host "  BIOS Virtualization: DISABLED" -ForegroundColor Red
    Write-Host "  ** YOU MUST ENABLE THIS IN BIOS! **" -ForegroundColor Red
}

if ($virt.HyperVRequirementVMMonitorModeExtensions) {
    Write-Host "  VM Monitor Extensions: ENABLED" -ForegroundColor Green
} else {
    Write-Host "  VM Monitor Extensions: DISABLED" -ForegroundColor Red
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  NEXT STEPS - IMPORTANT!" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "A SYSTEM RESTART IS REQUIRED!" -ForegroundColor Red
Write-Host ""
Write-Host "After restart:" -ForegroundColor Yellow
Write-Host "  1. Docker Desktop will start automatically" -ForegroundColor White
Write-Host "  2. Look for Docker whale icon in system tray" -ForegroundColor White
Write-Host "  3. Test with: docker ps" -ForegroundColor White
Write-Host "  4. Start containers: docker-compose up -d" -ForegroundColor White
Write-Host ""

$restart = Read-Host "Restart computer now? (Y/N)"

if ($restart -eq "Y" -or $restart -eq "y") {
    Write-Host ""
    Write-Host "Restarting in 10 seconds..." -ForegroundColor Yellow
    Write-Host "SAVE YOUR WORK NOW!" -ForegroundColor Red
    Write-Host ""
    Start-Sleep -Seconds 10
    Restart-Computer -Force
} else {
    Write-Host ""
    Write-Host "Please restart manually as soon as possible." -ForegroundColor Yellow
    Write-Host "Docker will NOT work until you restart!" -ForegroundColor Red
    Write-Host ""
}

Write-Host "========================================" -ForegroundColor Green
Write-Host "Script completed!" -ForegroundColor Green
Write-Host ""
Read-Host "Press Enter to close"


