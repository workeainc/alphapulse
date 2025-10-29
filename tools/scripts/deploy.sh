#!/bin/bash

# AlphaPulse Production Deployment Script
# Week 10: Production Deployment

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
NAMESPACE="alphapulse"
DEPLOYMENT_TYPE="${1:-docker}" # docker or kubernetes

echo -e "${BLUE}🚀 AlphaPulse Production Deployment${NC}"
echo -e "${BLUE}Week 10: Production Deployment${NC}"
echo "=================================="

# Function to check prerequisites
check_prerequisites() {
    echo -e "${YELLOW}Checking prerequisites...${NC}"
    
    if ! command -v docker &> /dev/null; then
        echo -e "${RED}❌ Docker is not installed${NC}"
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        echo -e "${RED}❌ Docker Compose is not installed${NC}"
        exit 1
    fi
    
    if [ "$DEPLOYMENT_TYPE" = "kubernetes" ]; then
        if ! command -v kubectl &> /dev/null; then
            echo -e "${RED}❌ kubectl is not installed${NC}"
            exit 1
        fi
        
        if ! kubectl cluster-info &> /dev/null; then
            echo -e "${RED}❌ Kubernetes cluster is not accessible${NC}"
            exit 1
        fi
    fi
    
    echo -e "${GREEN}✅ Prerequisites check passed${NC}"
}

# Function to create necessary directories
create_directories() {
    echo -e "${YELLOW}Creating necessary directories...${NC}"
    
    mkdir -p logs/{backend,frontend,nginx}
    mkdir -p data/{models,cache,backups}
    mkdir -p ssl
    mkdir -p monitoring/{grafana/provisioning/{dashboards,datasources},prometheus}
    
    echo -e "${GREEN}✅ Directories created${NC}"
}

# Function to generate SSL certificates (self-signed for testing)
generate_ssl_certificates() {
    echo -e "${YELLOW}Generating SSL certificates...${NC}"
    
    if [ ! -f "ssl/alphapulse.key" ] || [ ! -f "ssl/alphapulse.crt" ]; then
        openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
            -keyout ssl/alphapulse.key \
            -out ssl/alphapulse.crt \
            -subj "/C=US/ST=State/L=City/O=AlphaPulse/CN=alphapulse.example.com"
        echo -e "${GREEN}✅ SSL certificates generated${NC}"
    else
        echo -e "${GREEN}✅ SSL certificates already exist${NC}"
    fi
}

# Function to set environment variables
set_environment() {
    echo -e "${YELLOW}Setting environment variables...${NC}"
    
    if [ ! -f ".env" ]; then
        cat > .env << EOF
# AlphaPulse Production Environment
POSTGRES_PASSWORD=alphapulse_secure_password_2025
REDIS_PASSWORD=redis_secure_password_2025
GRAFANA_PASSWORD=grafana_admin_2025

# API Keys (if needed)
CCXT_API_KEY=
CCXT_SECRET=

# JWT Secret
JWT_SECRET=alphapulse_jwt_secret_2025_production

# Monitoring
PROMETHEUS_RETENTION=200h
GRAFANA_PLUGINS=grafana-piechart-panel,grafana-worldmap-panel
EOF
        echo -e "${GREEN}✅ Environment file created${NC}"
    else
        echo -e "${GREEN}✅ Environment file already exists${NC}"
    fi
}

# Function to deploy with Docker Compose
deploy_docker() {
    echo -e "${YELLOW}Deploying with Docker Compose...${NC}"
    
    # Build images
    echo -e "${BLUE}Building Docker images...${NC}"
    docker-compose -f docker-compose.prod.yml build
    
    # Start services
    echo -e "${BLUE}Starting services...${NC}"
    docker-compose -f docker-compose.prod.yml up -d
    
    # Wait for services to be ready
    echo -e "${BLUE}Waiting for services to be ready...${NC}"
    sleep 30
    
    # Check service health
    echo -e "${BLUE}Checking service health...${NC}"
    docker-compose -f docker-compose.prod.yml ps
    
    echo -e "${GREEN}✅ Docker deployment completed${NC}"
}

# Function to deploy with Kubernetes
deploy_kubernetes() {
    echo -e "${YELLOW}Deploying with Kubernetes...${NC}"
    
    # Create namespace
    echo -e "${BLUE}Creating namespace...${NC}"
    kubectl apply -f k8s/namespace.yaml
    
    # Apply configurations
    echo -e "${BLUE}Applying configurations...${NC}"
    kubectl apply -f k8s/configmap.yaml
    kubectl apply -f k8s/secrets.yaml
    
    # Deploy database and cache
    echo -e "${BLUE}Deploying database and cache...${NC}"
    kubectl apply -f k8s/postgres.yaml
    kubectl apply -f k8s/redis.yaml
    
    # Wait for database to be ready
    echo -e "${BLUE}Waiting for database to be ready...${NC}"
    kubectl wait --for=condition=ready pod -l app=alphapulse-postgres -n $NAMESPACE --timeout=300s
    kubectl wait --for=condition=ready pod -l app=alphapulse-redis -n $NAMESPACE --timeout=300s
    
    # Deploy backend
    echo -e "${BLUE}Deploying backend...${NC}"
    kubectl apply -f k8s/backend.yaml
    
    # Deploy frontend
    echo -e "${BLUE}Deploying frontend...${NC}"
    kubectl apply -f k8s/frontend.yaml
    
    # Deploy monitoring
    echo -e "${BLUE}Deploying monitoring...${NC}"
    kubectl apply -f k8s/monitoring.yaml
    
    # Deploy ingress
    echo -e "${BLUE}Deploying ingress...${NC}"
    kubectl apply -f k8s/ingress.yaml
    
    # Wait for all pods to be ready
    echo -e "${BLUE}Waiting for all pods to be ready...${NC}"
    kubectl wait --for=condition=ready pod -l app=alphapulse-backend -n $NAMESPACE --timeout=300s
    kubectl wait --for=condition=ready pod -l app=alphapulse-frontend -n $NAMESPACE --timeout=300s
    
    echo -e "${GREEN}✅ Kubernetes deployment completed${NC}"
}

# Function to show deployment status
show_status() {
    echo -e "${BLUE}Deployment Status:${NC}"
    echo "=================="
    
    if [ "$DEPLOYMENT_TYPE" = "docker" ]; then
        docker-compose -f docker-compose.prod.yml ps
        echo ""
        echo -e "${BLUE}Service URLs:${NC}"
        echo "Frontend: http://localhost:3000"
        echo "Backend API: http://localhost:8000"
        echo "Dashboard: http://localhost:8050"
        echo "Prometheus: http://localhost:9090"
        echo "Grafana: http://localhost:3001 (admin/admin)"
    else
        kubectl get pods -n $NAMESPACE
        echo ""
        kubectl get services -n $NAMESPACE
        echo ""
        kubectl get ingress -n $NAMESPACE
    fi
}

# Function to show logs
show_logs() {
    echo -e "${BLUE}Recent logs:${NC}"
    echo "============="
    
    if [ "$DEPLOYMENT_TYPE" = "docker" ]; then
        docker-compose -f docker-compose.prod.yml logs --tail=20
    else
        kubectl logs -n $NAMESPACE -l app=alphapulse-backend --tail=20
    fi
}

# Main deployment flow
main() {
    check_prerequisites
    create_directories
    generate_ssl_certificates
    set_environment
    
    if [ "$DEPLOYMENT_TYPE" = "docker" ]; then
        deploy_docker
    else
        deploy_kubernetes
    fi
    
    show_status
    echo ""
    echo -e "${GREEN}🎉 AlphaPulse production deployment completed!${NC}"
    echo ""
    echo -e "${BLUE}Next steps:${NC}"
    echo "1. Update your hosts file to point domains to localhost"
    echo "2. Access the dashboard at http://localhost:3000"
    echo "3. Monitor services with Grafana at http://localhost:3001"
    echo "4. Check logs with: ./scripts/deploy.sh logs"
}

# Handle command line arguments
case "${1:-deploy}" in
    "deploy")
        main
        ;;
    "status")
        show_status
        ;;
    "logs")
        show_logs
        ;;
    "cleanup")
        if [ "$DEPLOYMENT_TYPE" = "docker" ]; then
            docker-compose -f docker-compose.prod.yml down -v
        else
            kubectl delete namespace $NAMESPACE
        fi
        echo -e "${GREEN}✅ Cleanup completed${NC}"
        ;;
    *)
        echo "Usage: $0 {deploy|status|logs|cleanup} [docker|kubernetes]"
        echo "  deploy: Deploy AlphaPulse (default)"
        echo "  status: Show deployment status"
        echo "  logs: Show recent logs"
        echo "  cleanup: Remove all resources"
        echo "  Second argument: docker (default) or kubernetes"
        exit 1
        ;;
esac
