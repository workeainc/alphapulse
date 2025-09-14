#!/bin/bash
# AlphaPlus Production Deployment Script
# Automated deployment with health checks and rollback capabilities

set -e

# Configuration
PROJECT_NAME="alphapulse"
COMPOSE_FILE="docker-compose.production.yml"
ENV_FILE="production.env"
BACKUP_DIR="./backups"
LOG_FILE="./logs/deployment.log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a $LOG_FILE
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a $LOG_FILE
    exit 1
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1" | tee -a $LOG_FILE
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1" | tee -a $LOG_FILE
}

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."
    
    # Check if Docker is installed
    if ! command -v docker &> /dev/null; then
        error "Docker is not installed. Please install Docker first."
    fi
    
    # Check if Docker Compose is installed
    if ! command -v docker-compose &> /dev/null; then
        error "Docker Compose is not installed. Please install Docker Compose first."
    fi
    
    # Check if environment file exists
    if [ ! -f "$ENV_FILE" ]; then
        error "Environment file $ENV_FILE not found. Please create it first."
    fi
    
    # Check if compose file exists
    if [ ! -f "$COMPOSE_FILE" ]; then
        error "Docker Compose file $COMPOSE_FILE not found."
    fi
    
    success "Prerequisites check passed"
}

# Create necessary directories
create_directories() {
    log "Creating necessary directories..."
    
    mkdir -p $BACKUP_DIR
    mkdir -p ./logs
    mkdir -p ./ssl/certbot/conf
    mkdir -p ./ssl/certbot/www
    mkdir -p ./nginx/ssl
    mkdir -p ./monitoring/grafana/dashboards
    mkdir -p ./monitoring/grafana/datasources
    
    success "Directories created"
}

# Backup current deployment
backup_current() {
    log "Creating backup of current deployment..."
    
    BACKUP_NAME="backup_$(date +%Y%m%d_%H%M%S)"
    BACKUP_PATH="$BACKUP_DIR/$BACKUP_NAME"
    
    mkdir -p $BACKUP_PATH
    
    # Backup database
    if docker-compose -f $COMPOSE_FILE ps postgres | grep -q "Up"; then
        log "Backing up database..."
        docker-compose -f $COMPOSE_FILE exec -T postgres pg_dump -U alphapulse_user alphapulse > $BACKUP_PATH/database.sql
    fi
    
    # Backup configuration files
    cp $ENV_FILE $BACKUP_PATH/
    cp $COMPOSE_FILE $BACKUP_PATH/
    cp -r ./nginx $BACKUP_PATH/
    cp -r ./monitoring $BACKUP_PATH/
    
    success "Backup created: $BACKUP_PATH"
}

# Generate SSL certificates
generate_ssl() {
    log "Generating SSL certificates..."
    
    if [ ! -f "./nginx/ssl/cert.pem" ] || [ ! -f "./nginx/ssl/key.pem" ]; then
        log "Generating self-signed SSL certificates..."
        chmod +x ./scripts/generate-ssl.sh
        ./scripts/generate-ssl.sh
    else
        warning "SSL certificates already exist. Skipping generation."
    fi
}

# Pull latest images
pull_images() {
    log "Pulling latest Docker images..."
    
    docker-compose -f $COMPOSE_FILE pull
    
    success "Images pulled successfully"
}

# Deploy services
deploy_services() {
    log "Deploying services..."
    
    # Stop existing services
    docker-compose -f $COMPOSE_FILE down --remove-orphans
    
    # Start services
    docker-compose -f $COMPOSE_FILE up -d
    
    success "Services deployed"
}

# Wait for services to be healthy
wait_for_health() {
    log "Waiting for services to be healthy..."
    
    local max_attempts=30
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        log "Health check attempt $attempt/$max_attempts"
        
        # Check if all services are running
        if docker-compose -f $COMPOSE_FILE ps | grep -q "Exit"; then
            warning "Some services exited. Checking logs..."
            docker-compose -f $COMPOSE_FILE logs --tail=50
        fi
        
        # Check backend health
        if curl -f http://localhost:8000/health > /dev/null 2>&1; then
            success "Backend is healthy"
            break
        fi
        
        if [ $attempt -eq $max_attempts ]; then
            error "Services failed to become healthy after $max_attempts attempts"
        fi
        
        sleep 10
        ((attempt++))
    done
}

# Run post-deployment tests
run_tests() {
    log "Running post-deployment tests..."
    
    # Test API endpoints
    if curl -f http://localhost:8000/api/health > /dev/null 2>&1; then
        success "API health check passed"
    else
        error "API health check failed"
    fi
    
    # Test frontend
    if curl -f http://localhost:3000 > /dev/null 2>&1; then
        success "Frontend health check passed"
    else
        error "Frontend health check failed"
    fi
    
    # Test database connection
    if docker-compose -f $COMPOSE_FILE exec -T postgres pg_isready -U alphapulse_user -d alphapulse > /dev/null 2>&1; then
        success "Database health check passed"
    else
        error "Database health check failed"
    fi
    
    # Test Redis connection
    if docker-compose -f $COMPOSE_FILE exec -T redis-master redis-cli ping > /dev/null 2>&1; then
        success "Redis health check passed"
    else
        error "Redis health check failed"
    fi
}

# Display deployment status
show_status() {
    log "Deployment Status:"
    echo ""
    docker-compose -f $COMPOSE_FILE ps
    echo ""
    
    log "Service URLs:"
    echo "Frontend: http://localhost:3000"
    echo "Backend API: http://localhost:8000"
    echo "Grafana: http://localhost:3001"
    echo "Prometheus: http://localhost:9090"
    echo ""
    
    log "Useful Commands:"
    echo "View logs: docker-compose -f $COMPOSE_FILE logs -f"
    echo "Stop services: docker-compose -f $COMPOSE_FILE down"
    echo "Restart services: docker-compose -f $COMPOSE_FILE restart"
    echo "Scale backend: docker-compose -f $COMPOSE_FILE up -d --scale backend=3"
}

# Rollback function
rollback() {
    log "Rolling back to previous deployment..."
    
    # Find latest backup
    LATEST_BACKUP=$(ls -t $BACKUP_DIR | head -n1)
    
    if [ -z "$LATEST_BACKUP" ]; then
        error "No backup found for rollback"
    fi
    
    log "Rolling back to: $LATEST_BACKUP"
    
    # Stop current services
    docker-compose -f $COMPOSE_FILE down
    
    # Restore from backup
    cp "$BACKUP_DIR/$LATEST_BACKUP/$ENV_FILE" ./
    cp "$BACKUP_DIR/$LATEST_BACKUP/$COMPOSE_FILE" ./
    
    # Restart services
    docker-compose -f $COMPOSE_FILE up -d
    
    success "Rollback completed"
}

# Main deployment function
main() {
    log "Starting AlphaPlus production deployment..."
    
    case "${1:-deploy}" in
        "deploy")
            check_prerequisites
            create_directories
            backup_current
            generate_ssl
            pull_images
            deploy_services
            wait_for_health
            run_tests
            show_status
            success "Deployment completed successfully!"
            ;;
        "rollback")
            rollback
            ;;
        "status")
            show_status
            ;;
        "logs")
            docker-compose -f $COMPOSE_FILE logs -f
            ;;
        *)
            echo "Usage: $0 {deploy|rollback|status|logs}"
            echo "  deploy   - Deploy the application"
            echo "  rollback - Rollback to previous version"
            echo "  status   - Show deployment status"
            echo "  logs     - Show service logs"
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"
