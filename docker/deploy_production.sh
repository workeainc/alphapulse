#!/bin/bash

# AlphaPlus Production Deployment Script
# Deploys the complete AlphaPlus system with Docker

set -e

echo "🚀 Starting AlphaPlus Production Deployment..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    print_error "Docker is not running. Please start Docker and try again."
    exit 1
fi

print_success "Docker is running"

# Check if Docker Compose is available
if ! command -v docker-compose &> /dev/null; then
    print_error "Docker Compose is not installed. Please install Docker Compose and try again."
    exit 1
fi

print_success "Docker Compose is available"

# Create necessary directories
print_status "Creating necessary directories..."
mkdir -p logs
mkdir -p data
mkdir -p ssl
mkdir -p nginx/ssl

# Copy environment file if it doesn't exist
if [ ! -f .env ]; then
    print_status "Creating .env file from production template..."
    cp production.env .env
    print_warning "Please review and update the .env file with your specific configuration"
fi

# Stop any existing containers
print_status "Stopping existing containers..."
docker-compose -f docker-compose.production.yml down --remove-orphans || true

# Remove old images to force rebuild
print_status "Removing old images..."
docker-compose -f docker-compose.production.yml down --rmi all || true

# Build and start services
print_status "Building and starting services..."
docker-compose -f docker-compose.yml up --build -d

# Wait for services to be healthy
print_status "Waiting for services to be healthy..."

# Wait for PostgreSQL
print_status "Waiting for PostgreSQL to be ready..."
timeout=60
while ! docker exec alphapulse_postgres pg_isready -U alpha_emon -d alphapulse > /dev/null 2>&1; do
    sleep 2
    timeout=$((timeout - 2))
    if [ $timeout -le 0 ]; then
        print_error "PostgreSQL failed to start within 60 seconds"
        exit 1
    fi
done
print_success "PostgreSQL is ready"

# Wait for Redis
print_status "Waiting for Redis to be ready..."
timeout=30
while ! docker exec alphapulse_redis redis-cli ping > /dev/null 2>&1; do
    sleep 2
    timeout=$((timeout - 2))
    if [ $timeout -le 0 ]; then
        print_error "Redis failed to start within 30 seconds"
        exit 1
    fi
done
print_success "Redis is ready"

# Wait for Backend
print_status "Waiting for Backend to be ready..."
timeout=120
while ! curl -f http://localhost:8000/api/v1/production/health > /dev/null 2>&1; do
    sleep 5
    timeout=$((timeout - 5))
    if [ $timeout -le 0 ]; then
        print_error "Backend failed to start within 120 seconds"
        print_status "Checking backend logs..."
        docker logs alphapulse_backend
        exit 1
    fi
done
print_success "Backend is ready"

# Wait for Frontend
print_status "Waiting for Frontend to be ready..."
timeout=60
while ! curl -f http://localhost:3000 > /dev/null 2>&1; do
    sleep 5
    timeout=$((timeout - 5))
    if [ $timeout -le 0 ]; then
        print_error "Frontend failed to start within 60 seconds"
        print_status "Checking frontend logs..."
        docker logs alphapulse_frontend
        exit 1
    fi
done
print_success "Frontend is ready"

# Wait for Prometheus
print_status "Waiting for Prometheus to be ready..."
timeout=30
while ! curl -f http://localhost:9090 > /dev/null 2>&1; do
    sleep 2
    timeout=$((timeout - 2))
    if [ $timeout -le 0 ]; then
        print_warning "Prometheus failed to start within 30 seconds (optional service)"
    fi
done
print_success "Prometheus is ready"

# Wait for Grafana
print_status "Waiting for Grafana to be ready..."
timeout=30
while ! curl -f http://localhost:3001 > /dev/null 2>&1; do
    sleep 2
    timeout=$((timeout - 2))
    if [ $timeout -le 0 ]; then
        print_warning "Grafana failed to start within 30 seconds (optional service)"
    fi
done
print_success "Grafana is ready"

# Show deployment status
print_status "Deployment completed! Here's the status:"
echo ""
echo "📊 Service Status:"
docker-compose -f docker-compose.production.yml ps

echo ""
echo "🌐 Access URLs:"
echo "  Frontend Dashboard: http://localhost:3000"
echo "  Backend API: http://localhost:8000"
echo "  API Documentation: http://localhost:8000/docs"
echo "  Health Check: http://localhost:8000/api/v1/production/health"
echo "  Prometheus: http://localhost:9090"
echo "  Grafana: http://localhost:3001 (admin/admin123)"
echo "  Nginx: http://localhost:80"

echo ""
echo "📋 Database Information:"
echo "  PostgreSQL Host: localhost:5432"
echo "  Database: alphapulse"
echo "  Username: alpha_emon"
echo "  Password: Emon_@17711"

echo ""
echo "🔧 Management Commands:"
echo "  View logs: docker-compose -f docker-compose.production.yml logs -f [service]"
echo "  Stop services: docker-compose -f docker-compose.production.yml down"
echo "  Restart services: docker-compose -f docker-compose.production.yml restart [service]"
echo "  Scale services: docker-compose -f docker-compose.production.yml up -d --scale [service]=[count]"

echo ""
print_success "AlphaPlus Production Deployment Completed Successfully!"
print_status "All services are running and healthy"

# Run a quick test
print_status "Running quick system test..."
if curl -f http://localhost:8000/api/v1/production/status > /dev/null 2>&1; then
    print_success "System test passed - AlphaPlus is fully operational!"
else
    print_warning "System test failed - check the logs for issues"
fi

echo ""
echo "🎉 AlphaPlus is now running in production mode!"
echo "   Visit http://localhost:3000 to access the dashboard"
echo "   Visit http://localhost:8000/docs to explore the API"
