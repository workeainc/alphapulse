#!/bin/bash

# AlphaPulse Development Deployment Script
# This script deploys the entire AlphaPulse system using Docker Compose

set -e  # Exit on any error

echo "🚀 Starting AlphaPulse Development Deployment..."

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

# Check if Docker Compose is available
if ! command -v docker-compose &> /dev/null; then
    print_error "Docker Compose is not installed. Please install Docker Compose and try again."
    exit 1
fi

# Stop any existing containers
print_status "Stopping any existing containers..."
docker-compose -f docker-compose.development.yml down --remove-orphans || true

# Remove old images to ensure fresh build
print_status "Removing old images..."
docker-compose -f docker-compose.development.yml down --rmi all --volumes --remove-orphans || true

# Build and start services
print_status "Building and starting services..."
docker-compose -f docker-compose.development.yml up --build -d

# Wait for services to be ready
print_status "Waiting for services to be ready..."
sleep 30

# Check service health
print_status "Checking service health..."

# Check PostgreSQL
if docker-compose -f docker-compose.development.yml exec -T postgres pg_isready -U alpha_emon -d alphapulse > /dev/null 2>&1; then
    print_success "PostgreSQL is ready"
else
    print_warning "PostgreSQL is still starting up..."
fi

# Check Redis
if docker-compose -f docker-compose.development.yml exec -T redis redis-cli ping > /dev/null 2>&1; then
    print_success "Redis is ready"
else
    print_warning "Redis is still starting up..."
fi

# Check Backend
if curl -f http://localhost:8000/health > /dev/null 2>&1; then
    print_success "Backend API is ready"
else
    print_warning "Backend API is still starting up..."
fi

# Check Frontend
if curl -f http://localhost:3000 > /dev/null 2>&1; then
    print_success "Frontend is ready"
else
    print_warning "Frontend is still starting up..."
fi

# Check Monitoring
if curl -f http://localhost:8001/health > /dev/null 2>&1; then
    print_success "Monitoring Dashboard is ready"
else
    print_warning "Monitoring Dashboard is still starting up..."
fi

# Display service URLs
echo ""
print_success "🎉 AlphaPulse Development Environment is starting up!"
echo ""
echo "📊 Service URLs:"
echo "   Frontend Dashboard:     http://localhost:3000"
echo "   Backend API:           http://localhost:8000"
echo "   API Documentation:     http://localhost:8000/docs"
echo "   Monitoring Dashboard:  http://localhost:8001"
echo ""
echo "🗄️  Database:"
echo "   PostgreSQL:            localhost:5432"
echo "   Database:              alphapulse"
echo "   Username:              alpha_emon"
echo "   Password:              Emon_@17711"
echo ""
echo "🔧 Useful Commands:"
echo "   View logs:             docker-compose -f docker-compose.development.yml logs -f"
echo "   Stop services:         docker-compose -f docker-compose.development.yml down"
echo "   Restart services:      docker-compose -f docker-compose.development.yml restart"
echo "   View containers:       docker-compose -f docker-compose.development.yml ps"
echo ""

# Wait a bit more and check final status
sleep 10

print_status "Final health check..."
if curl -f http://localhost:8000/health > /dev/null 2>&1 && curl -f http://localhost:3000 > /dev/null 2>&1; then
    print_success "✅ All services are running successfully!"
    echo ""
    print_success "🌐 You can now access the AlphaPulse dashboard at: http://localhost:3000"
else
    print_warning "⚠️  Some services may still be starting up. Please wait a few more minutes."
    print_warning "You can check the logs with: docker-compose -f docker-compose.development.yml logs -f"
fi

echo ""
print_status "Deployment completed! 🚀"
