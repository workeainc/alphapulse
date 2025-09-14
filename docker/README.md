# AlphaPlus Docker Deployment Guide

This guide will help you deploy the entire AlphaPlus project (frontend + backend + database) using Docker.

## 🚀 Quick Start

### Prerequisites

- Docker Desktop installed and running
- Docker Compose available
- At least 8GB RAM available for Docker
- Ports 3000, 8000, 5432, 6379, 80, 443 available

### One-Command Deployment

#### Windows
```bash
cd docker
deploy.bat
```

#### Linux/Mac
```bash
cd docker
chmod +x deploy.sh
./deploy.sh
```

## 📋 What Gets Deployed

The deployment includes:

- **Frontend**: Next.js dashboard (port 3000)
- **Backend**: FastAPI trading system (port 8000)
- **Database**: PostgreSQL 15 (port 5432)
- **Cache**: Redis 7 (port 6379)
- **Task Queue**: Celery workers for background processing
- **Reverse Proxy**: Nginx for production routing (ports 80/443)

## 🔧 Manual Deployment

### 1. Environment Setup

Copy the environment template and configure it:
```bash
cp env.example .env
# Edit .env with your configuration
```

### 2. Development Deployment
```bash
docker-compose up --build -d
```

### 3. Production Deployment
```bash
docker-compose -f docker-compose.prod.yml up --build -d
```

## 🌐 Accessing Services

After deployment, access your services at:

- **Frontend Dashboard**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

## 📊 Monitoring & Management

### View Logs
```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f backend
docker-compose logs -f frontend
```

### Service Status
```bash
docker-compose ps
```

### Stop Services
```bash
docker-compose down
```

### Restart Services
```bash
docker-compose restart
```

## 🔒 Production Considerations

### Security
- Change default passwords in `.env`
- Use strong `SECRET_KEY`
- Enable SSL/TLS with proper certificates
- Restrict network access

### Performance
- Adjust Celery worker concurrency
- Configure Redis persistence
- Set appropriate PostgreSQL settings
- Monitor resource usage

### Backup
- Database volumes are persistent
- Model cache volumes preserve ML models
- Feature cache volumes preserve computed features

## 🐛 Troubleshooting

### Common Issues

#### Port Already in Use
```bash
# Check what's using the port
netstat -ano | findstr :8000  # Windows
lsof -i :8000                 # Linux/Mac

# Stop conflicting services or change ports in docker-compose.yml
```

#### Database Connection Issues
```bash
# Check database logs
docker-compose logs postgres

# Verify database is running
docker-compose exec postgres psql -U alphapulse_user -d alphapulse
```

#### Frontend Build Issues
```bash
# Clear node_modules and rebuild
docker-compose exec frontend rm -rf node_modules
docker-compose up --build frontend
```

### Health Checks

All services include health checks. Monitor them with:
```bash
docker-compose ps
```

## 📁 File Structure

```
docker/
├── docker-compose.yml          # Development configuration
├── docker-compose.prod.yml     # Production configuration
├── Dockerfile.backend          # Backend container
├── Dockerfile.frontend         # Frontend container
├── nginx.conf                  # Nginx configuration
├── nginx.prod.conf             # Production Nginx config
├── deploy.sh                   # Linux/Mac deployment script
├── deploy.bat                  # Windows deployment script
├── env.example                 # Environment template
└── README.md                   # This file
```

## 🔄 Updates & Maintenance

### Update Services
```bash
docker-compose pull
docker-compose up --build -d
```

### Clean Up
```bash
# Remove unused containers, networks, images
docker system prune -a

# Remove all data (WARNING: Destructive!)
docker-compose down -v
```

## 📞 Support

If you encounter issues:

1. Check the logs: `docker-compose logs -f`
2. Verify environment variables in `.env`
3. Ensure all ports are available
4. Check Docker Desktop resources

## 🎯 Next Steps

After successful deployment:

1. Access the frontend dashboard
2. Configure trading parameters
3. Set up data sources
4. Train initial ML models
5. Monitor system performance

---

**Happy Trading! 🚀📈**
