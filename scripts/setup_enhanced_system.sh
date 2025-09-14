#!/bin/bash

# Enhanced AlphaPlus System Setup Script
# Handles database migrations, dependencies, and system initialization

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKEND_DIR="$PROJECT_ROOT/backend"
DOCKER_COMPOSE_FILE="$PROJECT_ROOT/docker/docker-compose.enhanced.yml"
MIGRATION_FILE="$BACKEND_DIR/migrations/001_enhanced_cache_integration.sql"

# Logging function
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
}

info() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] INFO: $1${NC}"
}

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    # Check if Docker is running
    if ! docker info &> /dev/null; then
        error "Docker is not running. Please start Docker first."
        exit 1
    fi
    
    log "✅ Prerequisites check passed"
}

# Install Python dependencies
install_python_dependencies() {
    log "Installing Python dependencies..."
    
    if [ -f "$BACKEND_DIR/requirements.enhanced.txt" ]; then
        cd "$BACKEND_DIR"
        pip install -r requirements.enhanced.txt
        log "✅ Python dependencies installed"
    else
        warn "Enhanced requirements file not found, skipping Python dependency installation"
    fi
}

# Setup database
setup_database() {
    log "Setting up database..."
    
    # Start PostgreSQL container if not running
    if ! docker ps | grep -q "alphapulse_postgres"; then
        info "Starting PostgreSQL container..."
        docker-compose -f "$DOCKER_COMPOSE_FILE" up -d postgres
        
        # Wait for PostgreSQL to be ready
        log "Waiting for PostgreSQL to be ready..."
        local max_attempts=30
        local attempt=1
        
        while [ $attempt -le $max_attempts ]; do
            if docker exec alphapulse_postgres pg_isready -U alpha_emon -d alphapulse &> /dev/null; then
                log "✅ PostgreSQL is ready"
                break
            else
                info "Waiting for PostgreSQL to be ready (attempt $attempt/$max_attempts)..."
                sleep 5
                attempt=$((attempt + 1))
            fi
        done
        
        if [ $attempt -gt $max_attempts ]; then
            error "PostgreSQL failed to become ready"
            exit 1
        fi
    else
        log "✅ PostgreSQL container is already running"
    fi
    
    # Run database migration
    if [ -f "$MIGRATION_FILE" ]; then
        log "Running database migration..."
        docker exec alphapulse_postgres psql -U alpha_emon -d alphapulse -f /docker-entrypoint-initdb.d/001_enhanced_cache_integration.sql
        
        if [ $? -eq 0 ]; then
            log "✅ Database migration completed successfully"
        else
            error "Database migration failed"
            exit 1
        fi
    else
        warn "Migration file not found: $MIGRATION_FILE"
    fi
}

# Setup Redis
setup_redis() {
    log "Setting up Redis..."
    
    # Start Redis container if not running
    if ! docker ps | grep -q "alphapulse_redis"; then
        info "Starting Redis container..."
        docker-compose -f "$DOCKER_COMPOSE_FILE" up -d redis
        
        # Wait for Redis to be ready
        log "Waiting for Redis to be ready..."
        local max_attempts=30
        local attempt=1
        
        while [ $attempt -le $max_attempts ]; do
            if docker exec alphapulse_redis redis-cli ping &> /dev/null; then
                log "✅ Redis is ready"
                break
            else
                info "Waiting for Redis to be ready (attempt $attempt/$max_attempts)..."
                sleep 5
                attempt=$((attempt + 1))
            fi
        done
        
        if [ $attempt -gt $max_attempts ]; then
            error "Redis failed to become ready"
            exit 1
        fi
    else
        log "✅ Redis container is already running"
    fi
}

# Build and start services
build_and_start_services() {
    log "Building and starting enhanced services..."
    
    # Build the enhanced backend
    docker-compose -f "$DOCKER_COMPOSE_FILE" build backend_enhanced
    
    # Start all services
    docker-compose -f "$DOCKER_COMPOSE_FILE" up -d
    
    log "✅ Enhanced services started"
}

# Wait for services to be healthy
wait_for_services() {
    log "Waiting for services to be healthy..."
    
    local max_attempts=30
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        info "Checking service health (attempt $attempt/$max_attempts)..."
        
        # Check if all services are healthy
        if docker-compose -f "$DOCKER_COMPOSE_FILE" ps | grep -q "unhealthy"; then
            warn "Some services are not healthy yet, waiting..."
            sleep 10
            attempt=$((attempt + 1))
        else
            log "✅ All services are healthy!"
            break
        fi
    done
    
    if [ $attempt -gt $max_attempts ]; then
        error "Services failed to become healthy within the expected time"
        docker-compose -f "$DOCKER_COMPOSE_FILE" ps
        exit 1
    fi
}

# Run system tests
run_system_tests() {
    log "Running system tests..."
    
    # Wait a bit for services to fully initialize
    sleep 10
    
    # Test API health endpoint
    if curl -f http://localhost:8000/api/health &> /dev/null; then
        log "✅ API health check passed"
    else
        error "API health check failed"
        return 1
    fi
    
    # Test cache endpoint
    if curl -f http://localhost:8000/api/cache/stats &> /dev/null; then
        log "✅ Cache endpoint test passed"
    else
        warn "Cache endpoint test failed (this might be normal if no data yet)"
    fi
    
    # Test system overview endpoint
    if curl -f http://localhost:8000/api/system/overview &> /dev/null; then
        log "✅ System overview endpoint test passed"
    else
        warn "System overview endpoint test failed"
    fi
    
    log "✅ System tests completed"
}

# Display system information
display_system_info() {
    log "Enhanced AlphaPlus System Setup Complete!"
    echo
    echo "🌐 System Access URLs:"
    echo "   • Backend API: http://localhost:8000"
    echo "   • API Documentation: http://localhost:8000/docs"
    echo "   • Grafana Dashboard: http://localhost:3000 (admin/admin123)"
    echo "   • Prometheus: http://localhost:9090"
    echo "   • Redis Commander: http://localhost:8081"
    echo
    echo "📊 Test Endpoints:"
    echo "   • Health Check: curl http://localhost:8000/api/health"
    echo "   • Cache Stats: curl http://localhost:8000/api/cache/stats"
    echo "   • System Overview: curl http://localhost:8000/api/system/overview"
    echo
    echo "🔧 Management Commands:"
    echo "   • View logs: docker-compose -f $DOCKER_COMPOSE_FILE logs -f"
    echo "   • Stop services: docker-compose -f $DOCKER_COMPOSE_FILE down"
    echo "   • Restart services: docker-compose -f $DOCKER_COMPOSE_FILE restart"
    echo
    echo "📁 Important Files:"
    echo "   • Docker Compose: $DOCKER_COMPOSE_FILE"
    echo "   • Migration: $MIGRATION_FILE"
    echo "   • Requirements: $BACKEND_DIR/requirements.enhanced.txt"
    echo
}

# Main setup function
main() {
    log "🚀 Starting Enhanced AlphaPlus System Setup..."
    
    # Check if we're in the right directory
    if [ ! -f "$DOCKER_COMPOSE_FILE" ]; then
        error "Docker Compose file not found: $DOCKER_COMPOSE_FILE"
        error "Please run this script from the project root directory"
        exit 1
    fi
    
    # Run setup steps
    check_prerequisites
    install_python_dependencies
    setup_database
    setup_redis
    build_and_start_services
    wait_for_services
    run_system_tests
    display_system_info
    
    log "🎉 Enhanced AlphaPlus System Setup Complete!"
}

# Handle command line arguments
case "${1:-}" in
    "help"|"-h"|"--help")
        echo "Enhanced AlphaPlus System Setup Script"
        echo
        echo "Usage: $0 [command]"
        echo
        echo "Commands:"
        echo "  (no args)  - Run full setup"
        echo "  help       - Show this help message"
        echo "  deps       - Install Python dependencies only"
        echo "  db         - Setup database only"
        echo "  redis      - Setup Redis only"
        echo "  build      - Build and start services only"
        echo "  test       - Run system tests only"
        echo "  info       - Display system information only"
        ;;
    "deps")
        check_prerequisites
        install_python_dependencies
        ;;
    "db")
        check_prerequisites
        setup_database
        ;;
    "redis")
        check_prerequisites
        setup_redis
        ;;
    "build")
        check_prerequisites
        build_and_start_services
        ;;
    "test")
        run_system_tests
        ;;
    "info")
        display_system_info
        ;;
    "")
        main
        ;;
    *)
        error "Unknown command: $1"
        echo "Run '$0 help' for usage information"
        exit 1
        ;;
esac
