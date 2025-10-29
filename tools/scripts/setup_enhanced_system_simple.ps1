# Enhanced AlphaPlus System Setup Script (Simplified)
# Handles database migrations, dependencies, and system initialization

param(
    [switch]$SkipDocker,
    [switch]$SkipDependencies,
    [switch]$SkipMigrations,
    [switch]$Help
)

# Configuration
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$BackendDir = Join-Path $ProjectRoot "backend"
$DockerComposeFile = Join-Path $ProjectRoot "docker\docker-compose.enhanced.yml"
$MigrationFile = Join-Path $BackendDir "migrations\001_enhanced_cache_integration.sql"

# Logging function
function Write-Log {
    param([string]$Message)
    Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $Message" -ForegroundColor Green
}

function Write-Warn {
    param([string]$Message)
    Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] WARNING: $Message" -ForegroundColor Yellow
}

function Write-Error {
    param([string]$Message)
    Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] ERROR: $Message" -ForegroundColor Red
}

function Write-Info {
    param([string]$Message)
    Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] INFO: $Message" -ForegroundColor Blue
}

# Check if Docker is installed
function Test-Docker {
    try {
        $null = docker --version
        Write-Log "Docker is installed"
        return $true
    }
    catch {
        Write-Error "Docker is not installed or not in PATH"
        return $false
    }
}

# Check if Docker Compose is available
function Test-DockerCompose {
    try {
        $null = docker-compose --version
        Write-Log "Docker Compose is available"
        return $true
    }
    catch {
        Write-Error "Docker Compose is not available"
        return $false
    }
}

# Check if required files exist
function Test-RequiredFiles {
    $requiredFiles = @(
        $DockerComposeFile,
        $MigrationFile,
        (Join-Path $BackendDir "requirements.enhanced.txt"),
        (Join-Path $BackendDir "Dockerfile.enhanced")
    )
    
    foreach ($file in $requiredFiles) {
        if (-not (Test-Path $file)) {
            Write-Error "Required file not found: $file"
            return $false
        }
    }
    
    Write-Log "All required files found"
    return $true
}

# Install Python dependencies
function Install-PythonDependencies {
    Write-Log "Installing Python dependencies..."
    
    try {
        Set-Location $BackendDir
        pip install -r requirements.enhanced.txt
        Write-Log "Python dependencies installed successfully"
    }
    catch {
        Write-Error "Failed to install Python dependencies: $($_.Exception.Message)"
        return $false
    }
    finally {
        Set-Location $ProjectRoot
    }
    
    return $true
}

# Start Docker services
function Start-DockerServices {
    Write-Log "Starting Docker services..."
    
    try {
        # Stop existing services if running
        docker-compose -f $DockerComposeFile down
        
        # Start PostgreSQL and Redis first
        Write-Log "Starting PostgreSQL and Redis..."
        docker-compose -f $DockerComposeFile up -d postgres redis
        
        # Wait for services to be ready
        Start-Sleep -Seconds 10
        
        Write-Log "Docker services started successfully"
        return $true
    }
    catch {
        Write-Error "Failed to start Docker services: $($_.Exception.Message)"
        return $false
    }
}

# Wait for PostgreSQL to be ready
function Wait-PostgreSQLReady {
    Write-Log "Waiting for PostgreSQL to be ready..."
    
    $maxAttempts = 30
    $attempt = 1
    
    while ($attempt -le $maxAttempts) {
        try {
            $result = docker exec alphapulse_postgres pg_isready -U alpha_emon -d alphapulse 2>$null
            if ($LASTEXITCODE -eq 0) {
                Write-Log "PostgreSQL is ready"
                return $true
            }
        }
        catch {
            # Ignore errors during startup
        }
        
        Write-Info "Waiting for PostgreSQL to be ready (attempt $attempt/$maxAttempts)..."
        Start-Sleep -Seconds 5
        $attempt++
    }
    
    Write-Error "PostgreSQL failed to become ready"
    return $false
}

# Run database migration
function Invoke-DatabaseMigration {
    Write-Log "Running database migration..."
    
    try {
        # Copy migration file to container
        docker cp $MigrationFile alphapulse_postgres:/tmp/001_enhanced_cache_integration.sql
        
        # Run migration
        docker exec alphapulse_postgres psql -U alpha_emon -d alphapulse -f /tmp/001_enhanced_cache_integration.sql
        
        Write-Log "Database migration completed successfully"
        return $true
    }
    catch {
        Write-Error "Failed to run database migration: $($_.Exception.Message)"
        return $false
    }
}

# Build and start enhanced services
function Start-EnhancedServices {
    Write-Log "Building and starting enhanced services..."
    
    try {
        docker-compose -f $DockerComposeFile up -d --build
        Write-Log "Enhanced services started successfully"
        return $true
    }
    catch {
        Write-Error "Failed to start enhanced services: $($_.Exception.Message)"
        return $false
    }
}

# Test system
function Test-System {
    Write-Log "Testing system..."
    
    try {
        # Wait for services to be ready
        Start-Sleep -Seconds 15
        
        # Test API health endpoint
        $response = Invoke-WebRequest -Uri "http://localhost:8000/api/health" -UseBasicParsing -TimeoutSec 10
        if ($response.StatusCode -eq 200) {
            Write-Log "API health check passed"
        } else {
            Write-Warn "API health check returned status: $($response.StatusCode)"
        }
        
        # Test cache statistics endpoint
        try {
            $cacheResponse = Invoke-WebRequest -Uri "http://localhost:8000/api/cache/stats" -UseBasicParsing -TimeoutSec 10
            if ($cacheResponse.StatusCode -eq 200) {
                Write-Log "Cache statistics endpoint working"
            }
        }
        catch {
            Write-Warn "Cache statistics endpoint not available yet"
        }
        
        Write-Log "System tests completed"
        return $true
    }
    catch {
        Write-Warn "System tests failed: $($_.Exception.Message)"
        return $false
    }
}

# Display system information
function Show-SystemInfo {
    Write-Host ""
    Write-Host "Enhanced AlphaPlus System Setup Complete!" -ForegroundColor Green
    Write-Host ""
    Write-Host "System Access:" -ForegroundColor Blue
    Write-Host "   Backend API: http://localhost:8000" -ForegroundColor White
    Write-Host "   API Documentation: http://localhost:8000/docs" -ForegroundColor White
    Write-Host "   Grafana Dashboard: http://localhost:3000 (admin/admin123)" -ForegroundColor White
    Write-Host "   Prometheus: http://localhost:9090" -ForegroundColor White
    Write-Host "   Redis Commander: http://localhost:8081" -ForegroundColor White
    Write-Host ""
    Write-Host "Management Commands:" -ForegroundColor Blue
    Write-Host "   View logs: docker-compose -f $DockerComposeFile logs -f" -ForegroundColor White
    Write-Host "   Stop services: docker-compose -f $DockerComposeFile down" -ForegroundColor White
    Write-Host "   Restart services: docker-compose -f $DockerComposeFile restart" -ForegroundColor White
    Write-Host ""
    Write-Host "Documentation: docs/QUICK_START_ENHANCED.md" -ForegroundColor Blue
    Write-Host ""
}

# Main execution
function Main {
    Write-Host "Enhanced AlphaPlus System Setup" -ForegroundColor Green
    Write-Host "=====================================" -ForegroundColor Green
    Write-Host ""
    
    # Check prerequisites
    if (-not $SkipDocker) {
        if (-not (Test-Docker)) { return }
        if (-not (Test-DockerCompose)) { return }
    }
    
    if (-not (Test-RequiredFiles)) { return }
    
    # Install dependencies
    if (-not $SkipDependencies) {
        if (-not (Install-PythonDependencies)) { return }
    }
    
    # Start Docker services
    if (-not $SkipDocker) {
        if (-not (Start-DockerServices)) { return }
        if (-not (Wait-PostgreSQLReady)) { return }
    }
    
    # Run database migration
    if (-not $SkipMigrations) {
        if (-not (Invoke-DatabaseMigration)) { return }
    }
    
    # Start enhanced services
    if (-not $SkipDocker) {
        if (-not (Start-EnhancedServices)) { return }
    }
    
    # Test system
    Test-System
    
    # Show system information
    Show-SystemInfo
}

# Handle help parameter
if ($Help) {
    Write-Host "Enhanced AlphaPlus System Setup Script" -ForegroundColor Green
    Write-Host ""
    Write-Host "Usage: .\setup_enhanced_system_simple.ps1 [options]" -ForegroundColor White
    Write-Host ""
    Write-Host "Options:" -ForegroundColor White
    Write-Host "  -SkipDocker      Skip Docker-related operations" -ForegroundColor White
    Write-Host "  -SkipDependencies Skip Python dependency installation" -ForegroundColor White
    Write-Host "  -SkipMigrations  Skip database migrations" -ForegroundColor White
    Write-Host "  -Help           Show this help message" -ForegroundColor White
    Write-Host ""
    exit 0
}

# Run main function
Main
