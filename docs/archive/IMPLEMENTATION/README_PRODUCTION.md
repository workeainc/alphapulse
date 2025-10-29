# 🚀 AlphaPulse Performance Dashboard - Production Deployment Guide

## **Overview**

The AlphaPulse Performance Dashboard is a production-ready, scalable monitoring system for TimescaleDB performance optimization. This guide covers deployment using Docker, Docker Compose, and Kubernetes.

## **🏗️ Architecture**

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Load Balancer │    │   Nginx Proxy   │    │   Dashboard     │
│   (Optional)    │───▶│   (SSL/TLS)     │───▶│   (FastAPI)     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │                       │
                                ▼                       ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │   Redis Cache   │    │  TimescaleDB    │
                       │   (Optional)    │    │   Database      │
                       └─────────────────┘    └─────────────────┘
```

## **📋 Prerequisites**

### **System Requirements**
- **CPU**: 2+ cores
- **Memory**: 4GB+ RAM
- **Storage**: 20GB+ available space
- **OS**: Linux, macOS, or Windows with Docker support

### **Software Requirements**
- **Docker**: 20.10+
- **Docker Compose**: 2.0+
- **kubectl**: 1.24+ (for Kubernetes deployment)
- **curl**: For health checks

## **🚀 Quick Start (Docker Compose)**

### **1. Clone and Setup**
```bash
git clone <repository-url>
cd backend
chmod +x deploy.sh
```

### **2. Deploy**
```bash
# Deploy with Docker Compose
./deploy.sh docker

# Or manually
docker-compose up -d
```

### **3. Access Dashboard**
- **URL**: http://localhost:8000
- **Health Check**: http://localhost:8000/api/health
- **API Docs**: http://localhost:8000/docs

## **🐳 Docker Deployment**

### **Build Image**
```bash
docker build -t alphapulse/dashboard:latest .
```

### **Run Container**
```bash
docker run -d \
  --name alphapulse-dashboard \
  -p 8000:8000 \
  -e DATABASE_URL=postgresql://user:pass@host:5432/db \
  alphapulse/dashboard:latest
```

### **Health Check**
```bash
curl -f http://localhost:8000/api/health
```

## **☸️ Kubernetes Deployment**

### **1. Prerequisites**
- Kubernetes cluster (1.24+)
- kubectl configured
- Docker image in accessible registry

### **2. Deploy**
```bash
# Create namespace
kubectl create namespace alphapulse

# Apply manifests
kubectl apply -f k8s/ -n alphapulse

# Check status
kubectl get pods -n alphapulse
```

### **3. Access Dashboard**
```bash
# Port forward for local access
kubectl port-forward svc/alphapulse-dashboard-service 8000:80 -n alphapulse

# Or get external IP
kubectl get svc -n alphapulse
```

## **🔧 Configuration**

### **Environment Variables**

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | - | PostgreSQL connection string |
| `DASHBOARD_HOST` | `0.0.0.0` | Dashboard bind address |
| `DASHBOARD_PORT` | `8000` | Dashboard port |
| `LOG_LEVEL` | `INFO` | Logging level |
| `REDIS_URL` | - | Redis connection string |

### **Database Configuration**
```bash
# TimescaleDB connection
DATABASE_URL=postgresql://username:password@host:5432/database_name

# Example
DATABASE_URL=postgresql://alphapulse_user:alphapulse_password@timescaledb:5432/alphapulse
```

## **📊 Monitoring & Observability**

### **Prometheus Metrics**
- **Endpoint**: `/metrics`
- **Scrape Interval**: 15s
- **Metrics**: System health, performance, optimization status

### **Health Checks**
- **Liveness**: `/api/health`
- **Readiness**: `/api/health`
- **Startup**: 30s delay

### **Logging**
- **Format**: JSON structured logging
- **Levels**: DEBUG, INFO, WARNING, ERROR
- **Output**: stdout/stderr

## **🔒 Security**

### **SSL/TLS**
- **Nginx**: Automatic SSL redirect
- **Certificates**: Let's Encrypt integration
- **Protocols**: TLS 1.2, 1.3

### **Rate Limiting**
- **API**: 10 requests/second
- **Dashboard**: 30 requests/second
- **Burst**: Configurable burst limits

### **Security Headers**
- **X-Frame-Options**: DENY
- **X-Content-Type-Options**: nosniff
- **X-XSS-Protection**: 1; mode=block
- **HSTS**: 1 year

## **📈 Scaling**

### **Horizontal Pod Autoscaler**
- **Min Replicas**: 3
- **Max Replicas**: 10
- **CPU Target**: 70%
- **Memory Target**: 80%

### **Load Balancing**
- **Strategy**: Least connections
- **Health Checks**: Active monitoring
- **Failover**: Automatic failover

## **🔄 Updates & Maintenance**

### **Rolling Updates**
```bash
# Update deployment
kubectl set image deployment/alphapulse-dashboard dashboard=new-image:tag -n alphapulse

# Monitor rollout
kubectl rollout status deployment/alphapulse-dashboard -n alphapulse
```

### **Rollback**
```bash
# Rollback to previous version
kubectl rollout undo deployment/alphapulse-dashboard -n alphapulse
```

## **🚨 Troubleshooting**

### **Common Issues**

#### **Dashboard Not Starting**
```bash
# Check logs
docker-compose logs dashboard
kubectl logs -f deployment/alphapulse-dashboard -n alphapulse

# Check health
curl -v http://localhost:8000/api/health
```

#### **Database Connection Issues**
```bash
# Test database connectivity
docker exec -it alphapulse_timescaledb psql -U alphapulse_user -d alphapulse

# Check database logs
docker-compose logs timescaledb
```

#### **Performance Issues**
```bash
# Check resource usage
docker stats
kubectl top pods -n alphapulse

# Monitor metrics
curl http://localhost:8000/api/metrics
```

### **Debug Mode**
```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
docker-compose up dashboard
```

## **📚 API Reference**

### **Endpoints**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Dashboard HTML |
| `/api/health` | GET | System health status |
| `/api/metrics` | GET | Performance metrics |
| `/api/optimization-status` | GET | Optimization status |
| `/api/performance-report` | GET | Detailed performance report |
| `/api/alerts` | GET | Active alerts |
| `/ws` | WebSocket | Real-time updates |

### **Response Format**
```json
{
  "overall_score": 85.5,
  "database_health": "good",
  "performance_score": 8.5,
  "optimization_status": "monitoring",
  "active_alerts": 1,
  "timestamp": "2024-01-01T00:00:00Z"
}
```

## **🔮 Advanced Features**

### **Redis Caching**
- **Session Storage**: User sessions
- **Query Cache**: Performance metrics
- **Rate Limiting**: API throttling

### **Nginx Features**
- **Gzip Compression**: Automatic compression
- **Static File Serving**: Optimized delivery
- **Load Balancing**: Multiple backend support

### **Monitoring Stack**
- **Prometheus**: Metrics collection
- **Grafana**: Visualization
- **Alertmanager**: Alerting

## **📞 Support**

### **Getting Help**
- **Documentation**: This README
- **Issues**: GitHub Issues
- **Discussions**: GitHub Discussions

### **Contributing**
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## **📄 License**

This project is licensed under the MIT License - see the LICENSE file for details.

---

**🎉 Congratulations!** You now have a production-ready, scalable monitoring dashboard for your TimescaleDB performance optimization system.
