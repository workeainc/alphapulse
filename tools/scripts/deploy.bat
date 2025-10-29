@echo off
REM AlphaPulse Production Deployment Script
REM Week 10: Production Deployment

setlocal enabledelayedexpansion

REM Colors for output (Windows doesn't support ANSI colors by default)
set "RED=[ERROR]"
set "GREEN=[SUCCESS]"
set "YELLOW=[WARNING]"
set "BLUE=[INFO]"

REM Configuration
set "NAMESPACE=alphapulse"
set "DEPLOYMENT_TYPE=%1"
if "%DEPLOYMENT_TYPE%"=="" set "DEPLOYMENT_TYPE=docker"

echo %BLUE% 🚀 AlphaPulse Production Deployment
echo %BLUE% Week 10: Production Deployment
echo ==================================

REM Function to check prerequisites
:check_prerequisites
echo %YELLOW% Checking prerequisites...
docker --version >nul 2>&1
if errorlevel 1 (
    echo %RED% ❌ Docker is not installed
    exit /b 1
)

docker-compose --version >nul 2>&1
if errorlevel 1 (
    echo %RED% ❌ Docker Compose is not installed
    exit /b 1
)

if "%DEPLOYMENT_TYPE%"=="kubernetes" (
    kubectl version --client >nul 2>&1
    if errorlevel 1 (
        echo %RED% ❌ kubectl is not installed
        exit /b 1
    )
    
    kubectl cluster-info >nul 2>&1
    if errorlevel 1 (
        echo %RED% ❌ Kubernetes cluster is not accessible
        exit /b 1
    )
)

echo %GREEN% ✅ Prerequisites check passed
goto :eof

REM Function to create necessary directories
:create_directories
echo %YELLOW% Creating necessary directories...
if not exist "logs\backend" mkdir "logs\backend"
if not exist "logs\frontend" mkdir "logs\frontend"
if not exist "logs\nginx" mkdir "logs\nginx"
if not exist "data\models" mkdir "data\models"
if not exist "data\cache" mkdir "data\cache"
if not exist "data\backups" mkdir "data\backups"
if not exist "ssl" mkdir "ssl"
if not exist "monitoring\grafana\provisioning\dashboards" mkdir "monitoring\grafana\provisioning\dashboards"
if not exist "monitoring\grafana\provisioning\datasources" mkdir "monitoring\grafana\provisioning\datasources"
if not exist "monitoring\prometheus" mkdir "monitoring\prometheus"

echo %GREEN% ✅ Directories created
goto :eof

REM Function to generate SSL certificates
:generate_ssl_certificates
echo %YELLOW% Generating SSL certificates...
if not exist "ssl\alphapulse.key" (
    openssl req -x509 -nodes -days 365 -newkey rsa:2048 -keyout ssl\alphapulse.key -out ssl\alphapulse.crt -subj "/C=US/ST=State/L=City/O=AlphaPulse/CN=alphapulse.example.com"
    echo %GREEN% ✅ SSL certificates generated
) else (
    echo %GREEN% ✅ SSL certificates already exist
)
goto :eof

REM Function to set environment variables
:set_environment
echo %YELLOW% Setting environment variables...
if not exist ".env" (
    (
        echo # AlphaPulse Production Environment
        echo POSTGRES_PASSWORD=alphapulse_secure_password_2025
        echo REDIS_PASSWORD=redis_secure_password_2025
        echo GRAFANA_PASSWORD=grafana_admin_2025
        echo.
        echo # API Keys (if needed)
        echo CCXT_API_KEY=
        echo CCXT_SECRET=
        echo.
        echo # JWT Secret
        echo JWT_SECRET=alphapulse_jwt_secret_2025_production
        echo.
        echo # Monitoring
        echo PROMETHEUS_RETENTION=200h
        echo GRAFANA_PLUGINS=grafana-piechart-panel,grafana-worldmap-panel
    ) > .env
    echo %GREEN% ✅ Environment file created
) else (
    echo %GREEN% ✅ Environment file already exists
)
goto :eof

REM Function to deploy with Docker Compose
:deploy_docker
echo %YELLOW% Deploying with Docker Compose...

echo %BLUE% Building Docker images...
docker-compose -f docker-compose.prod.yml build

echo %BLUE% Starting services...
docker-compose -f docker-compose.prod.yml up -d

echo %BLUE% Waiting for services to be ready...
timeout /t 30 /nobreak >nul

echo %BLUE% Checking service health...
docker-compose -f docker-compose.prod.yml ps

echo %GREEN% ✅ Docker deployment completed
goto :eof

REM Function to deploy with Kubernetes
:deploy_kubernetes
echo %YELLOW% Deploying with Kubernetes...

echo %BLUE% Creating namespace...
kubectl apply -f k8s\namespace.yaml

echo %BLUE% Applying configurations...
kubectl apply -f k8s\configmap.yaml
kubectl apply -f k8s\secrets.yaml

echo %BLUE% Deploying database and cache...
kubectl apply -f k8s\postgres.yaml
kubectl apply -f k8s\redis.yaml

echo %BLUE% Waiting for database to be ready...
kubectl wait --for=condition=ready pod -l app=alphapulse-postgres -n %NAMESPACE% --timeout=300s
kubectl wait --for=condition=ready pod -l app=alphapulse-redis -n %NAMESPACE% --timeout=300s

echo %BLUE% Deploying backend...
kubectl apply -f k8s\backend.yaml

echo %BLUE% Deploying frontend...
kubectl apply -f k8s\frontend.yaml

echo %BLUE% Deploying monitoring...
kubectl apply -f k8s\monitoring.yaml

echo %BLUE% Deploying ingress...
kubectl apply -f k8s\ingress.yaml

echo %BLUE% Waiting for all pods to be ready...
kubectl wait --for=condition=ready pod -l app=alphapulse-backend -n %NAMESPACE% --timeout=300s
kubectl wait --for=condition=ready pod -l app=alphapulse-frontend -n %NAMESPACE% --timeout=300s

echo %GREEN% ✅ Kubernetes deployment completed
goto :eof

REM Function to show deployment status
:show_status
echo %BLUE% Deployment Status:
echo ==================

if "%DEPLOYMENT_TYPE%"=="docker" (
    docker-compose -f docker-compose.prod.yml ps
    echo.
    echo %BLUE% Service URLs:
    echo Frontend: http://localhost:3000
    echo Backend API: http://localhost:8000
    echo Dashboard: http://localhost:8050
    echo Prometheus: http://localhost:9090
    echo Grafana: http://localhost:3001 (admin/admin)
) else (
    kubectl get pods -n %NAMESPACE%
    echo.
    kubectl get services -n %NAMESPACE%
    echo.
    kubectl get ingress -n %NAMESPACE%
)
goto :eof

REM Function to show logs
:show_logs
echo %BLUE% Recent logs:
echo =============

if "%DEPLOYMENT_TYPE%"=="docker" (
    docker-compose -f docker-compose.prod.yml logs --tail=20
) else (
    kubectl logs -n %NAMESPACE% -l app=alphapulse-backend --tail=20
)
goto :eof

REM Main deployment flow
:main
call :check_prerequisites
call :create_directories
call :generate_ssl_certificates
call :set_environment

if "%DEPLOYMENT_TYPE%"=="docker" (
    call :deploy_docker
) else (
    call :deploy_kubernetes
)

call :show_status
echo.
echo %GREEN% 🎉 AlphaPulse production deployment completed!
echo.
echo %BLUE% Next steps:
echo 1. Update your hosts file to point domains to localhost
echo 2. Access the dashboard at http://localhost:3000
echo 3. Monitor services with Grafana at http://localhost:3001
echo 4. Check logs with: scripts\deploy.bat logs
goto :eof

REM Handle command line arguments
if "%1"=="deploy" goto :main
if "%1"=="status" goto :show_status
if "%1"=="logs" goto :show_logs
if "%1"=="cleanup" (
    if "%DEPLOYMENT_TYPE%"=="docker" (
        docker-compose -f docker-compose.prod.yml down -v
    ) else (
        kubectl delete namespace %NAMESPACE%
    )
    echo %GREEN% ✅ Cleanup completed
    goto :eof
)
if "%1"=="" goto :main

echo Usage: %0 {deploy^|status^|logs^|cleanup} [docker^|kubernetes]
echo   deploy: Deploy AlphaPulse (default)
echo   status: Show deployment status
echo   logs: Show recent logs
echo   cleanup: Remove all resources
echo   Second argument: docker (default) or kubernetes
exit /b 1
