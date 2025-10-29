# Test Docker Status - Run this after Docker Desktop starts

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "  DOCKER STATUS CHECK" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check Docker process
Write-Host "1. Checking Docker Desktop process..." -ForegroundColor Yellow
$dockerProcess = Get-Process -Name "Docker Desktop" -ErrorAction SilentlyContinue
if ($dockerProcess) {
    Write-Host "   ✅ Docker Desktop IS RUNNING" -ForegroundColor Green
} else {
    Write-Host "   ❌ Docker Desktop NOT RUNNING" -ForegroundColor Red
    Write-Host "   Please launch Docker Desktop first!" -ForegroundColor Yellow
    Write-Host ""
    Read-Host "Press Enter to exit"
    exit
}

Write-Host ""

# Check Docker service
Write-Host "2. Checking Docker service..." -ForegroundColor Yellow
$service = Get-Service -Name "com.docker.service" -ErrorAction SilentlyContinue
if ($service.Status -eq "Running") {
    Write-Host "   ✅ Docker Service IS RUNNING" -ForegroundColor Green
} else {
    Write-Host "   ⏳ Docker Service: $($service.Status)" -ForegroundColor Yellow
    Write-Host "   Wait a bit longer for Docker to start..." -ForegroundColor Gray
}

Write-Host ""

# Test Docker command
Write-Host "3. Testing Docker command..." -ForegroundColor Yellow
try {
    $result = docker version 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "   ✅ DOCKER IS WORKING!" -ForegroundColor Green
    } else {
        Write-Host "   ⏳ Docker starting... wait 30 more seconds" -ForegroundColor Yellow
    }
} catch {
    Write-Host "   ⏳ Docker not ready yet" -ForegroundColor Yellow
}

Write-Host ""

# Test docker ps
Write-Host "4. Testing docker ps..." -ForegroundColor Yellow
try {
    docker ps 2>&1 | Out-Null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "   ✅ DOCKER DAEMON IS RESPONDING!" -ForegroundColor Green
        Write-Host ""
        Write-Host "========================================" -ForegroundColor Green
        Write-Host "  ✅ DOCKER IS FULLY WORKING!" -ForegroundColor Green
        Write-Host "========================================" -ForegroundColor Green
        Write-Host ""
        Write-Host "You can now:" -ForegroundColor Cyan
        Write-Host "  1. Start AlphaPulse containers" -ForegroundColor White
        Write-Host "  2. Create backups" -ForegroundColor White
        Write-Host "  3. Access your data" -ForegroundColor White
    } else {
        Write-Host "   ⏳ Wait 30 more seconds..." -ForegroundColor Yellow
    }
} catch {
    Write-Host "   ⏳ Docker daemon not ready yet" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Press any key to close..." -ForegroundColor Gray
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")



