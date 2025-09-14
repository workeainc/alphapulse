@echo off
REM AlphaPlus Docker Deployment Script for Windows
REM This script deploys the entire AlphaPlus project (frontend + backend + database)

echo 🚀 Starting AlphaPlus Docker Deployment...

REM Check if Docker is running
docker info >nul 2>&1
if errorlevel 1 (
    echo ❌ Docker is not running. Please start Docker Desktop and try again.
    pause
    exit /b 1
)

REM Check if Docker Compose is available
docker-compose --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Docker Compose is not available. Please ensure Docker Desktop is properly installed.
    pause
    exit /b 1
)

REM Create .env file if it doesn't exist
if not exist .env (
    echo 📝 Creating .env file from template...
    copy env.example .env
    echo ⚠️  Please review and update .env file with your configuration before continuing.
    echo Press any key to continue...
    pause >nul
)

REM Stop any existing containers
echo 🛑 Stopping existing containers...
docker-compose down --remove-orphans

REM Remove old images (optional)
echo 🧹 Cleaning up old images...
docker system prune -f

REM Build and start services
echo 🔨 Building and starting services...
docker-compose up --build -d

REM Wait for services to be ready
echo ⏳ Waiting for services to be ready...
timeout /t 30 /nobreak >nul

REM Check service health
echo 🏥 Checking service health...
docker-compose ps

REM Test backend health (using PowerShell if available)
echo 🔍 Testing backend health...
powershell -Command "try { Invoke-WebRequest -Uri 'http://localhost:8000/health' -UseBasicParsing | Out-Null; Write-Host '✅ Backend is healthy!' } catch { Write-Host '❌ Backend health check failed. Check logs with: docker-compose logs backend' }" 2>nul
if errorlevel 1 (
    echo ❌ Backend health check failed. Check logs with: docker-compose logs backend
)

REM Test frontend
echo 🔍 Testing frontend...
powershell -Command "try { Invoke-WebRequest -Uri 'http://localhost:3000' -UseBasicParsing | Out-Null; Write-Host '✅ Frontend is accessible!' } catch { Write-Host '❌ Frontend check failed. Check logs with: docker-compose logs frontend' }" 2>nul
if errorlevel 1 (
    echo ❌ Frontend check failed. Check logs with: docker-compose logs frontend
)

echo.
echo 🎉 AlphaPlus deployment completed!
echo.
echo 📊 Services are running on:
echo    Frontend: http://localhost:3000
echo    Backend API: http://localhost:8000
echo    Database: localhost:5432
echo    Redis: localhost:6379
echo.
echo 📋 Useful commands:
echo    View logs: docker-compose logs -f
echo    Stop services: docker-compose down
echo    Restart services: docker-compose restart
echo    Update services: docker-compose up --build -d
echo.
pause
