@echo off
REM AlphaPulse Production Deployment Script for Windows
REM This script automates the deployment of the AlphaPulse Performance Dashboard

setlocal enabledelayedexpansion

REM Configuration
set DOCKER_IMAGE=alphapulse/dashboard
set DOCKER_TAG=latest
set NAMESPACE=alphapulse

REM Functions
:log_info
echo [INFO] %~1
goto :eof

:log_success
echo [SUCCESS] %~1
goto :eof

:log_warning
echo [WARNING] %~1
goto :eof

:log_error
echo [ERROR] %~1
goto :eof

:check_prerequisites
call :log_info "Checking prerequisites..."

REM Check Docker
docker --version >nul 2>&1
if errorlevel 1 (
    call :log_error "Docker is not installed. Please install Docker Desktop first."
    exit /b 1
)

REM Check Docker Compose
docker-compose --version >nul 2>&1
if errorlevel 1 (
    call :log_error "Docker Compose is not installed. Please install Docker Compose first."
    exit /b 1
)

call :log_success "Prerequisites check passed"
goto :eof

:build_docker_image
call :log_info "Building Docker image..."

docker build -t %DOCKER_IMAGE%:%DOCKER_TAG% .

if errorlevel 1 (
    call :log_error "Failed to build Docker image"
    exit /b 1
)

call :log_success "Docker image built successfully"
goto :eof

:deploy_docker_compose
call :log_info "Deploying with Docker Compose..."

REM Stop existing containers
docker-compose down

REM Start services
docker-compose up -d

REM Wait for services to be healthy
call :log_info "Waiting for services to be healthy..."
timeout /t 30 /nobreak >nul

REM Check service health
curl -f http://localhost:8000/api/health >nul 2>&1
if errorlevel 1 (
    call :log_error "Dashboard health check failed"
    exit /b 1
)

call :log_success "Dashboard is healthy and running"
goto :eof

:run_tests
call :log_info "Running deployment tests..."

REM Test API endpoints
curl -f http://localhost:8000/api/health >nul 2>&1
if errorlevel 1 (
    call :log_error "Health endpoint test failed"
    exit /b 1
)

curl -f http://localhost:8000/api/metrics >nul 2>&1
if errorlevel 1 (
    call :log_error "Metrics endpoint test failed"
    exit /b 1
)

call :log_success "All deployment tests passed"
goto :eof

:show_status
call :log_info "Deployment Status:"
echo ====================

docker-compose ps
echo.
docker-compose logs --tail=20 dashboard
goto :eof

:cleanup
call :log_info "Cleaning up..."

docker-compose down -v

call :log_success "Cleanup completed"
goto :eof

REM Main script
:main
set DEPLOYMENT_TYPE=%1
if "%DEPLOYMENT_TYPE%"=="" set DEPLOYMENT_TYPE=docker

call :log_info "Starting AlphaPulse Production Deployment"
call :log_info "Deployment type: %DEPLOYMENT_TYPE%"

if "%DEPLOYMENT_TYPE%"=="docker" (
    call check_prerequisites
    call build_docker_image
    call deploy_docker_compose
    call run_tests
    call show_status
) else if "%DEPLOYMENT_TYPE%"=="cleanup" (
    call cleanup
) else (
    echo Usage: %0 {docker^|cleanup}
    echo   docker  - Deploy using Docker Compose
    echo   cleanup - Clean up deployment
    exit /b 1
)

call :log_success "Deployment completed successfully!"
goto :eof

REM Handle script arguments
if "%1"=="cleanup" (
    call cleanup
    goto :eof
)

call main %*
