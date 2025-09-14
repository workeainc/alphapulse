@echo off
REM AlphaPulse Simple Development Deployment Script for Windows
REM This script deploys the core AlphaPulse system using Docker Compose

echo 🚀 Starting AlphaPulse Simple Development Deployment...

REM Check if Docker is running
docker info >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Docker is not running. Please start Docker and try again.
    pause
    exit /b 1
)

REM Check if Docker Compose is available
docker-compose --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Docker Compose is not installed. Please install Docker Compose and try again.
    pause
    exit /b 1
)

REM Stop any existing containers
echo [INFO] Stopping any existing containers...
docker-compose -f docker-compose.simple.yml down --remove-orphans

REM Remove old images to ensure fresh build
echo [INFO] Removing old images...
docker-compose -f docker-compose.simple.yml down --rmi all --volumes --remove-orphans

REM Build and start services
echo [INFO] Building and starting services...
docker-compose -f docker-compose.simple.yml up --build -d

REM Wait for services to be ready
echo [INFO] Waiting for services to be ready...
timeout /t 30 /nobreak >nul

REM Check service health
echo [INFO] Checking service health...

REM Check PostgreSQL
docker-compose -f docker-compose.simple.yml exec -T postgres pg_isready -U alpha_emon -d alphapulse >nul 2>&1
if errorlevel 1 (
    echo [WARNING] PostgreSQL is still starting up...
) else (
    echo [SUCCESS] PostgreSQL is ready
)

REM Check Redis
docker-compose -f docker-compose.simple.yml exec -T redis redis-cli ping >nul 2>&1
if errorlevel 1 (
    echo [WARNING] Redis is still starting up...
) else (
    echo [SUCCESS] Redis is ready
)

REM Check Backend
curl -f http://localhost:8000/health >nul 2>&1
if errorlevel 1 (
    echo [WARNING] Backend API is still starting up...
) else (
    echo [SUCCESS] Backend API is ready
)

REM Check Frontend
curl -f http://localhost:3000 >nul 2>&1
if errorlevel 1 (
    echo [WARNING] Frontend is still starting up...
) else (
    echo [SUCCESS] Frontend is ready
)

REM Display service URLs
echo.
echo [SUCCESS] 🎉 AlphaPulse Development Environment is starting up!
echo.
echo 📊 Service URLs:
echo    Frontend Dashboard:     http://localhost:3000
echo    Backend API:           http://localhost:8000
echo    API Documentation:     http://localhost:8000/docs
echo.
echo 🗄️  Database:
echo    PostgreSQL:            localhost:5432
echo    Database:              alphapulse
echo    Username:              alpha_emon
echo    Password:              Emon_@17711
echo.
echo 🔧 Useful Commands:
echo    View logs:             docker-compose -f docker-compose.simple.yml logs -f
echo    Stop services:         docker-compose -f docker-compose.simple.yml down
echo    Restart services:      docker-compose -f docker-compose.simple.yml restart
echo    View containers:       docker-compose -f docker-compose.simple.yml ps
echo.

REM Wait a bit more and check final status
timeout /t 10 /nobreak >nul

echo [INFO] Final health check...
curl -f http://localhost:8000/health >nul 2>&1
if errorlevel 1 (
    echo [WARNING] ⚠️  Some services may still be starting up. Please wait a few more minutes.
    echo [WARNING] You can check the logs with: docker-compose -f docker-compose.simple.yml logs -f
) else (
    curl -f http://localhost:3000 >nul 2>&1
    if errorlevel 1 (
        echo [WARNING] ⚠️  Some services may still be starting up. Please wait a few more minutes.
        echo [WARNING] You can check the logs with: docker-compose -f docker-compose.simple.yml logs -f
    ) else (
        echo [SUCCESS] ✅ All services are running successfully!
        echo.
        echo [SUCCESS] 🌐 You can now access the AlphaPulse dashboard at: http://localhost:3000
    )
)

echo.
echo [INFO] Deployment completed! 🚀
pause
