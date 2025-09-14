# 🚀 Enhanced AlphaPlus Quick Start Guide

## Overview

This guide will help you quickly set up and run the **Enhanced AlphaPlus System** with Redis cache integration for ultra-low latency trading.

## Prerequisites

- **Docker** (version 20.10 or higher)
- **Docker Compose** (version 2.0 or higher)
- **Git** (for cloning the repository)
- **8GB RAM** minimum (16GB recommended)
- **10GB free disk space**

## Quick Setup (Automated)

### 1. Clone and Navigate
```bash
git clone <your-repo-url>
cd AlphaPlus
```

### 2. Run Automated Setup
```bash
# Make setup script executable
chmod +x scripts/setup_enhanced_system.sh

# Run full automated setup
./scripts/setup_enhanced_system.sh
```

This script will automatically:
- ✅ Check prerequisites
- ✅ Install Python dependencies
- ✅ Set up PostgreSQL with TimescaleDB
- ✅ Set up Redis cache
- ✅ Run database migrations
- ✅ Build and start all services
- ✅ Run system tests
- ✅ Display access information

## Manual Setup (Step by Step)

If you prefer manual setup or need to troubleshoot:

### 1. Install Python Dependencies
```bash
cd backend
pip install -r requirements.enhanced.txt
```

### 2. Start Database and Cache
```bash
# Start PostgreSQL and Redis
docker-compose -f docker/docker-compose.enhanced.yml up -d postgres redis

# Wait for services to be ready
docker-compose -f docker/docker-compose.enhanced.yml ps
```

### 3. Run Database Migration
```bash
# Run the enhanced cache migration
docker exec alphapulse_postgres psql -U alpha_emon -d alphapulse -f /docker-entrypoint-initdb.d/001_enhanced_cache_integration.sql
```

### 4. Start Enhanced Services
```bash
# Build and start all services
docker-compose -f docker/docker-compose.enhanced.yml up -d --build
```

### 5. Verify Setup
```bash
# Check service status
docker-compose -f docker/docker-compose.enhanced.yml ps

# Test API endpoints
curl http://localhost:8000/api/health
curl http://localhost:8000/api/cache/stats
```

## System Access

Once setup is complete, you can access:

| Service | URL | Credentials |
|---------|-----|-------------|
| **Backend API** | http://localhost:8000 | - |
| **API Documentation** | http://localhost:8000/docs | - |
| **Grafana Dashboard** | http://localhost:3000 | admin/admin123 |
| **Prometheus** | http://localhost:9090 | - |
| **Redis Commander** | http://localhost:8081 | - |

## Key Features

### 🚀 **Ultra-Low Latency**
- **Memory Cache**: <1ms response time
- **Redis Cache**: <5ms response time
- **WebSocket**: <20ms real-time updates

### 📊 **Performance Monitoring**
- Real-time cache hit rates
- WebSocket connection metrics
- Database performance tracking
- System health monitoring

### 🔧 **API Endpoints**
```bash
# Health check
curl http://localhost:8000/api/health

# Cache statistics
curl http://localhost:8000/api/cache/stats

# System overview
curl http://localhost:8000/api/system/overview

# Market data (from cache)
curl "http://localhost:8000/api/market/data?symbol=BTC/USDT&timeframe=1m&limit=100"

# Real-time data
curl http://localhost:8000/api/real-time/BTC/USDT/1m
```

### 🔌 **WebSocket Connection**
```javascript
// Connect to WebSocket
const ws = new WebSocket('ws://localhost:8000/ws');

// Subscribe to data
ws.send(JSON.stringify({
    type: 'subscribe',
    symbols: ['BTC/USDT', 'ETH/USDT'],
    timeframes: ['1m', '5m']
}));

// Listen for updates
ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    console.log('Received:', data);
};
```

## Management Commands

### Service Management
```bash
# View logs
docker-compose -f docker/docker-compose.enhanced.yml logs -f

# Stop services
docker-compose -f docker/docker-compose.enhanced.yml down

# Restart services
docker-compose -f docker/docker-compose.enhanced.yml restart

# Update and rebuild
docker-compose -f docker/docker-compose.enhanced.yml up -d --build
```

### Database Management
```bash
# Access PostgreSQL
docker exec -it alphapulse_postgres psql -U alpha_emon -d alphapulse

# Access Redis
docker exec -it alphapulse_redis redis-cli

# Backup database
docker exec alphapulse_postgres pg_dump -U alpha_emon alphapulse > backup.sql
```

### Cache Management
```bash
# Clear cache
curl -X POST http://localhost:8000/api/cache/clear

# Get cache statistics
curl http://localhost:8000/api/cache/stats
```

## Troubleshooting

### Common Issues

#### 1. **Services Not Starting**
```bash
# Check Docker status
docker info

# Check service logs
docker-compose -f docker/docker-compose.enhanced.yml logs

# Restart Docker
sudo systemctl restart docker
```

#### 2. **Database Connection Issues**
```bash
# Check PostgreSQL status
docker exec alphapulse_postgres pg_isready -U alpha_emon -d alphapulse

# Restart PostgreSQL
docker-compose -f docker/docker-compose.enhanced.yml restart postgres
```

#### 3. **Redis Connection Issues**
```bash
# Check Redis status
docker exec alphapulse_redis redis-cli ping

# Restart Redis
docker-compose -f docker/docker-compose.enhanced.yml restart redis
```

#### 4. **Port Conflicts**
```bash
# Check port usage
netstat -tulpn | grep :8000
netstat -tulpn | grep :6379
netstat -tulpn | grep :5432

# Stop conflicting services
sudo systemctl stop <conflicting-service>
```

### Performance Optimization

#### 1. **Increase Cache Size**
Edit `docker/docker-compose.enhanced.yml`:
```yaml
redis:
  command: redis-server --appendonly yes --maxmemory 1gb --maxmemory-policy allkeys-lru
```

#### 2. **Optimize Database**
```sql
-- Increase shared buffers
ALTER SYSTEM SET shared_buffers = '256MB';
ALTER SYSTEM SET effective_cache_size = '1GB';

-- Reload configuration
SELECT pg_reload_conf();
```

#### 3. **Monitor Performance**
```bash
# View real-time metrics
curl http://localhost:8000/api/system/overview

# Check Grafana dashboard
open http://localhost:3000
```

## Next Steps

1. **Explore the API**: Visit http://localhost:8000/docs
2. **Monitor Performance**: Check Grafana dashboard
3. **Test WebSocket**: Connect to real-time data stream
4. **Customize Configuration**: Modify settings in `docker/docker-compose.enhanced.yml`
5. **Scale Up**: Add more Redis instances or database replicas

## Support

- **Documentation**: Check `docs/enhanced_cache_implementation_guide.md`
- **Issues**: Create an issue in the repository
- **Logs**: Check service logs for detailed error information

---

**🎉 Congratulations!** Your Enhanced AlphaPlus System is now running with ultra-low latency cache integration.
