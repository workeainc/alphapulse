#!/bin/bash

# AlphaPulse Production Deployment Script
# This script automates the deployment of the AlphaPulse Performance Dashboard

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
DOCKER_IMAGE="alphapulse/dashboard"
DOCKER_TAG="latest"
NAMESPACE="alphapulse"
REGISTRY_URL=""

# Functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    # Check kubectl if using Kubernetes
    if [ "$1" = "k8s" ]; then
        if ! command -v kubectl &> /dev/null; then
            log_error "kubectl is not installed. Please install kubectl first."
            exit 1
        fi
    fi
    
    log_success "Prerequisites check passed"
}

build_docker_image() {
    log_info "Building Docker image..."
    
    docker build -t ${DOCKER_IMAGE}:${DOCKER_TAG} .
    
    if [ $? -eq 0 ]; then
        log_success "Docker image built successfully"
    else
        log_error "Failed to build Docker image"
        exit 1
    fi
}

push_docker_image() {
    if [ -n "$REGISTRY_URL" ]; then
        log_info "Pushing Docker image to registry..."
        
        docker tag ${DOCKER_IMAGE}:${DOCKER_TAG} ${REGISTRY_URL}/${DOCKER_IMAGE}:${DOCKER_TAG}
        docker push ${REGISTRY_URL}/${DOCKER_IMAGE}:${DOCKER_TAG}
        
        if [ $? -eq 0 ]; then
            log_success "Docker image pushed successfully"
        else
            log_error "Failed to push Docker image"
            exit 1
        fi
    else
        log_warning "No registry URL specified, skipping image push"
    fi
}

deploy_docker_compose() {
    log_info "Deploying with Docker Compose..."
    
    # Stop existing containers
    docker-compose down
    
    # Start services
    docker-compose up -d
    
    # Wait for services to be healthy
    log_info "Waiting for services to be healthy..."
    sleep 30
    
    # Check service health
    if curl -f http://localhost:8000/api/health > /dev/null 2>&1; then
        log_success "Dashboard is healthy and running"
    else
        log_error "Dashboard health check failed"
        exit 1
    fi
}

deploy_kubernetes() {
    log_info "Deploying to Kubernetes..."
    
    # Create namespace if it doesn't exist
    kubectl create namespace ${NAMESPACE} --dry-run=client -o yaml | kubectl apply -f -
    
    # Apply Kubernetes manifests
    kubectl apply -f k8s/ -n ${NAMESPACE}
    
    # Wait for deployment to be ready
    log_info "Waiting for deployment to be ready..."
    kubectl wait --for=condition=available --timeout=300s deployment/alphapulse-dashboard -n ${NAMESPACE}
    
    if [ $? -eq 0 ]; then
        log_success "Kubernetes deployment successful"
        
        # Get service URL
        SERVICE_URL=$(kubectl get service alphapulse-dashboard-service -n ${NAMESPACE} -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
        if [ -n "$SERVICE_URL" ]; then
            log_info "Dashboard available at: http://${SERVICE_URL}"
        fi
    else
        log_error "Kubernetes deployment failed"
        exit 1
    fi
}

run_tests() {
    log_info "Running deployment tests..."
    
    # Test API endpoints
    if curl -f http://localhost:8000/api/health > /dev/null 2>&1; then
        log_success "Health endpoint test passed"
    else
        log_error "Health endpoint test failed"
        exit 1
    fi
    
    # Test metrics endpoint
    if curl -f http://localhost:8000/api/metrics > /dev/null 2>&1; then
        log_success "Metrics endpoint test passed"
    else
        log_error "Metrics endpoint test failed"
        exit 1
    fi
    
    log_success "All deployment tests passed"
}

show_status() {
    log_info "Deployment Status:"
    echo "===================="
    
    if [ "$1" = "k8s" ]; then
        kubectl get pods -n ${NAMESPACE}
        echo ""
        kubectl get services -n ${NAMESPACE}
        echo ""
        kubectl get ingress -n ${NAMESPACE}
    else
        docker-compose ps
        echo ""
        docker-compose logs --tail=20 dashboard
    fi
}

cleanup() {
    log_info "Cleaning up..."
    
    if [ "$1" = "k8s" ]; then
        kubectl delete -f k8s/ -n ${NAMESPACE} --ignore-not-found=true
        kubectl delete namespace ${NAMESPACE} --ignore-not-found=true
    else
        docker-compose down -v
    fi
    
    log_success "Cleanup completed"
}

# Main script
main() {
    DEPLOYMENT_TYPE=${1:-"docker"}
    
    log_info "Starting AlphaPulse Production Deployment"
    log_info "Deployment type: ${DEPLOYMENT_TYPE}"
    
    case $DEPLOYMENT_TYPE in
        "docker"|"compose")
            check_prerequisites
            build_docker_image
            push_docker_image
            deploy_docker_compose
            run_tests
            show_status
            ;;
        "k8s"|"kubernetes")
            check_prerequisites "k8s"
            build_docker_image
            push_docker_image
            deploy_kubernetes
            run_tests
            show_status "k8s"
            ;;
        "cleanup")
            cleanup ${2:-"docker"}
            ;;
        *)
            echo "Usage: $0 {docker|k8s|cleanup}"
            echo "  docker  - Deploy using Docker Compose"
            echo "  k8s     - Deploy to Kubernetes"
            echo "  cleanup - Clean up deployment"
            exit 1
            ;;
    esac
    
    log_success "Deployment completed successfully!"
}

# Handle script arguments
case "${1:-}" in
    "cleanup")
        cleanup ${2:-"docker"}
        ;;
    *)
        main "$@"
        ;;
esac
