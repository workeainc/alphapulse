#!/bin/bash

# Enhanced AlphaPlus Deployment Script
# Deploys the enhanced system with Redis cache integration

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="alphapulse_enhanced"
DOCKER_COMPOSE_FILE="docker/docker-compose.enhanced.yml"
BACKEND_DIR="backend"
LOG_DIR="logs"

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

# Check if Docker is installed
check_docker() {
    if ! command -v docker &> /dev/null; then
        error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    log "Docker and Docker Compose are installed"
}

# Check if required files exist
check_files() {
    local missing_files=()
    
    if [ ! -f "$DOCKER_COMPOSE_FILE" ]; then
        missing_files+=("$DOCKER_COMPOSE_FILE")
    fi
    
    if [ ! -d "$BACKEND_DIR" ]; then
        missing_files+=("$BACKEND_DIR")
    fi
    
    if [ ${#missing_files[@]} -ne 0 ]; then
        error "Missing required files/directories:"
        for file in "${missing_files[@]}"; do
            echo "  - $file"
        done
        exit 1
    fi
    
    log "All required files are present"
}

# Create necessary directories
create_directories() {
    log "Creating necessary directories..."
    
    mkdir -p "$LOG_DIR"
    mkdir -p "docker/nginx/ssl"
    mkdir -p "docker/monitoring"
    mkdir -p "docker/grafana"
    
    log "Directories created successfully"
}

# Build and deploy services
deploy_services() {
    log "Starting deployment of AlphaPlus Enhanced with Cache..."
    
    # Stop existing services if running
    if docker-compose -f "$DOCKER_COMPOSE_FILE" ps | grep -q "Up"; then
        warn "Stopping existing services..."
        docker-compose -f "$DOCKER_COMPOSE_FILE" down
    fi
    
    # Build and start services
    log "Building and starting services..."
    docker-compose -f "$DOCKER_COMPOSE_FILE" up -d --build
    
    log "Services started successfully"
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
            log "All services are healthy!"
            break
        fi
    done
    
    if [ $attempt -gt $max_attempts ]; then
        error "Services failed to become healthy within the expected time"
        docker-compose -f "$DOCKER_COMPOSE_FILE" ps
        exit 1
    fi
}

# Initialize database
initialize_database() {
    log "Initializing database..."
    
    # Wait for PostgreSQL to be ready
    local max_attempts=30
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if docker exec alphapulse_postgres pg_isready -U alpha_emon -d alphapulse &> /dev/null; then
            log "PostgreSQL is ready"
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
    
    # Run database initialization
    log "Running database initialization..."
    docker exec alphapulse_postgres psql -U alpha_emon -d alphapulse -f /docker-entrypoint-initdb.d/init_enhanced_data_tables.sql
    
    log "Database initialized successfully"
}

# Test the system
test_system() {
    log "Testing the enhanced system..."
    
    # Wait a bit for services to fully start
    sleep 10
    
    # Test health endpoint
    local max_attempts=10
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if curl -f http://localhost:8000/api/health &> /dev/null; then
            log "Health check passed"
            break
        else
            info "Health check failed (attempt $attempt/$max_attempts), retrying..."
            sleep 5
            attempt=$((attempt + 1))
        fi
    done
    
    if [ $attempt -gt $max_attempts ]; then
        error "Health check failed after multiple attempts"
        exit 1
    fi
    
    # Test cache endpoint
    if curl -f http://localhost:8000/api/cache/stats &> /dev/null; then
        log "Cache endpoint test passed"
    else
        warn "Cache endpoint test failed"
    fi
    
    # Test WebSocket endpoint
    if curl -f http://localhost:8000/api/websocket/stats &> /dev/null; then
        log "WebSocket endpoint test passed"
    else
        warn "WebSocket endpoint test failed"
    fi
    
    log "System testing completed"
}

# Display service information
display_info() {
    log "AlphaPlus Enhanced with Cache deployment completed successfully!"
    echo
    echo "Service Information:"
    echo "==================="
    echo "Backend API:     http://localhost:8000"
    echo "API Documentation: http://localhost:8000/docs"
    echo "Health Check:    http://localhost:8000/api/health"
    echo "System Overview: http://localhost:8000/api/system/overview"
    echo
    echo "Monitoring:"
    echo "==========="
    echo "Grafana Dashboard: http://localhost:3000 (admin/admin123)"
    echo "Prometheus:        http://localhost:9090"
    echo "Redis Commander:   http://localhost:8081"
    echo
    echo "Database:"
    echo "========="
    echo "PostgreSQL:       localhost:5432"
    echo "Redis:           localhost:6379"
    echo
    echo "Useful Commands:"
    echo "==============="
    echo "View logs:        docker-compose -f $DOCKER_COMPOSE_FILE logs -f"
    echo "Stop services:    docker-compose -f $DOCKER_COMPOSE_FILE down"
    echo "Restart services: docker-compose -f $DOCKER_COMPOSE_FILE restart"
    echo "View status:      docker-compose -f $DOCKER_COMPOSE_FILE ps"
    echo
}

# Main deployment function
main() {
    log "Starting AlphaPlus Enhanced with Cache deployment..."
    
    # Check prerequisites
    check_docker
    check_files
    
    # Create directories
    create_directories
    
    # Deploy services
    deploy_services
    
    # Wait for services to be healthy
    wait_for_services
    
    # Initialize database
    initialize_database
    
    # Test the system
    test_system
    
    # Display information
    display_info
    
    log "Deployment completed successfully!"
}

# Handle script arguments
case "${1:-}" in
    "stop")
        log "Stopping AlphaPlus Enhanced services..."
        docker-compose -f "$DOCKER_COMPOSE_FILE" down
        log "Services stopped"
        ;;
    "restart")
        log "Restarting AlphaPlus Enhanced services..."
        docker-compose -f "$DOCKER_COMPOSE_FILE" restart
        log "Services restarted"
        ;;
    "logs")
        log "Showing service logs..."
        docker-compose -f "$DOCKER_COMPOSE_FILE" logs -f
        ;;
    "status")
        log "Service status:"
        docker-compose -f "$DOCKER_COMPOSE_FILE" ps
        ;;
    "clean")
        log "Cleaning up AlphaPlus Enhanced deployment..."
        docker-compose -f "$DOCKER_COMPOSE_FILE" down -v
        docker system prune -f
        log "Cleanup completed"
        ;;
    "help"|"-h"|"--help")
        echo "AlphaPlus Enhanced Deployment Script"
        echo
        echo "Usage: $0 [COMMAND]"
        echo
        echo "Commands:"
        echo "  (no args)  Deploy the enhanced system"
        echo "  stop       Stop all services"
        echo "  restart    Restart all services"
        echo "  logs       Show service logs"
        echo "  status     Show service status"
        echo "  clean      Stop and remove all containers and volumes"
        echo "  help       Show this help message"
        ;;
    *)
        main
        ;;
esac
