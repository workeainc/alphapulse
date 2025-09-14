@echo off
REM AlphaPlus Production Deployment Script for Windows
REM Deploys the complete AlphaPlus system with Docker

echo 🚀 Starting AlphaPlus Production Deployment...

REM Check if Docker is running
docker info >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Docker is not running. Please start Docker and try again.
    exit /b 1
)

echo [SUCCESS] Docker is running

REM Check if Docker Compose is available
docker-compose --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Docker Compose is not installed. Please install Docker Compose and try again.
    exit /b 1
)

echo [SUCCESS] Docker Compose is available

REM Create necessary directories
echo [INFO] Creating necessary directories...
if not exist logs mkdir logs
if not exist data mkdir data
if not exist ssl mkdir ssl
if not exist nginx\ssl mkdir nginx\ssl

REM Copy environment file if it doesn't exist
if not exist .env (
    echo [INFO] Creating .env file from production template...
    copy production.env .env
    echo [WARNING] Please review and update the .env file with your specific configuration
)

REM Stop any existing containers
echo [INFO] Stopping existing containers...
docker-compose -f docker-compose.production.yml down --remove-orphans

REM Remove old images to force rebuild
echo [INFO] Removing old images...
docker-compose -f docker-compose.production.yml down --rmi all

REM Build and start services
echo [INFO] Building and starting services...
docker-compose -f docker-compose.yml up --build -d

REM Wait for services to be healthy
echo [INFO] Waiting for services to be healthy...

REM Wait for PostgreSQL
echo [INFO] Waiting for PostgreSQL to be ready...
set timeout=60
:wait_postgres
docker exec alphapulse_postgres pg_isready -U alpha_emon -d alphapulse >nul 2>&1
if %errorlevel% equ 0 (
    echo [SUCCESS] PostgreSQL is ready
    goto wait_redis
)
timeout /t 2 /nobreak >nul
set /a timeout-=2
if %timeout% leq 0 (
    echo [ERROR] PostgreSQL failed to start within 60 seconds
    exit /b 1
)
goto wait_postgres

:wait_redis
REM Wait for Redis
echo [INFO] Waiting for Redis to be ready...
set timeout=30
:wait_redis_loop
docker exec alphapulse_redis redis-cli ping >nul 2>&1
if %errorlevel% equ 0 (
    echo [SUCCESS] Redis is ready
    goto wait_backend
)
timeout /t 2 /nobreak >nul
set /a timeout-=2
if %timeout% leq 0 (
    echo [ERROR] Redis failed to start within 30 seconds
    exit /b 1
)
goto wait_redis_loop

:wait_backend
REM Wait for Backend
echo [INFO] Waiting for Backend to be ready...
set timeout=120
:wait_backend_loop
curl -f http://localhost:8000/api/v1/production/health >nul 2>&1
if %errorlevel% equ 0 (
    echo [SUCCESS] Backend is ready
    goto wait_frontend
)
timeout /t 5 /nobreak >nul
set /a timeout-=5
if %timeout% leq 0 (
    echo [ERROR] Backend failed to start within 120 seconds
    echo [INFO] Checking backend logs...
    docker logs alphapulse_backend
    exit /b 1
)
goto wait_backend_loop

:wait_frontend
REM Wait for Frontend
echo [INFO] Waiting for Frontend to be ready...
set timeout=60
:wait_frontend_loop
curl -f http://localhost:3000 >nul 2>&1
if %errorlevel% equ 0 (
    echo [SUCCESS] Frontend is ready
    goto show_status
)
timeout /t 5 /nobreak >nul
set /a timeout-=5
if %timeout% leq 0 (
    echo [ERROR] Frontend failed to start within 60 seconds
    echo [INFO] Checking frontend logs...
    docker logs alphapulse_frontend
    exit /b 1
)
goto wait_frontend_loop

:show_status
REM Show deployment status
echo [INFO] Deployment completed! Here's the status:
echo.
echo 📊 Service Status:
docker-compose -f docker-compose.production.yml ps

echo.
echo 🌐 Access URLs:
echo   Frontend Dashboard: http://localhost:3000
echo   Backend API: http://localhost:8000
echo   API Documentation: http://localhost:8000/docs
echo   Health Check: http://localhost:8000/api/v1/production/health
echo   Prometheus: http://localhost:9090
echo   Grafana: http://localhost:3001 (admin/admin123)
echo   Nginx: http://localhost:80

echo.
echo 📋 Database Information:
echo   PostgreSQL Host: localhost:5432
echo   Database: alphapulse
echo   Username: alpha_emon
echo   Password: Emon_@17711

echo.
echo 🔧 Management Commands:
echo   View logs: docker-compose -f docker-compose.production.yml logs -f [service]
echo   Stop services: docker-compose -f docker-compose.production.yml down
echo   Restart services: docker-compose -f docker-compose.production.yml restart [service]
echo   Scale services: docker-compose -f docker-compose.production.yml up -d --scale [service]=[count]

echo.
echo [SUCCESS] AlphaPlus Production Deployment Completed Successfully!
echo [INFO] All services are running and healthy

REM Run a quick test
echo [INFO] Running quick system test...
curl -f http://localhost:8000/api/v1/production/status >nul 2>&1
if %errorlevel% equ 0 (
    echo [SUCCESS] System test passed - AlphaPlus is fully operational!
) else (
    echo [WARNING] System test failed - check the logs for issues
)

echo.
echo 🎉 AlphaPlus is now running in production mode!
echo    Visit http://localhost:3000 to access the dashboard
echo    Visit http://localhost:8000/docs to explore the API

pause
