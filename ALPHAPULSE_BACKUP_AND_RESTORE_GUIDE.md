# 🔒 AlphaPulse Complete Backup & Restore Guide

## 📋 Overview
This document contains ALL configurations for your AlphaPulse database, Redis, and Docker setup.
**IMPORTANT: You have 1 year of valuable trading data stored in your database!**

---

## 🗄️ Database Configuration

### Current Database Setup
- **Type**: TimescaleDB (PostgreSQL with time-series extensions)
- **Database Name**: `alphapulse`
- **Username**: `alpha_emon`
- **Password**: `Emon_@17711`
- **Host**: `localhost` (or `postgres` when using Docker)
- **Port**: `5432` (internal) / `55433` (external when using Docker)
- **Connection URL**: `postgresql://alpha_emon:Emon_@17711@localhost:5432/alphapulse`

### Database Location
- **Docker Volume**: `postgres_data` (for development)
- **Production Volume**: `postgres_data_prod` (for production)
- **Physical Location**: `C:\Users\EA Soft Lab\AppData\Local\Docker\wsl\data\`

---

## 🔴 Redis Configuration

### Current Redis Setup
- **Host**: `localhost` (or `redis` when using Docker)
- **Port**: `6379`
- **Database**: `0`
- **Password**: None (default setup)
- **Connection URL**: `redis://localhost:6379`

### Redis Location
- **Docker Volume**: `redis_data` (for development)
- **Production Volume**: `redis_master_data` & `redis_replica_data` (for production)

---

## 🐳 Docker Volumes (YOUR DATA IS HERE!)

### Active Volumes
```yaml
volumes:
  postgres_data:          # Your 1 year of trading data
  redis_data:            # Cache and temporary data
  postgres_data_prod:    # Production database (if used)
  redis_master_data:     # Production Redis master
  redis_replica_data:    # Production Redis replica
  prometheus_data:       # Monitoring data
  grafana_data:         # Dashboard data
```

---

## 💾 BACKUP YOUR DATA (CRITICAL!)

### Method 1: Export Database using Docker (Recommended)

After Docker starts, run these commands:

```powershell
# 1. Make sure Docker is running
docker ps

# 2. Create backup directory
mkdir backups
cd backups

# 3. Backup PostgreSQL Database (ALL YOUR 1 YEAR DATA!)
docker exec alphapulse_postgres pg_dump -U alpha_emon -d alphapulse -F c -f /tmp/alphapulse_backup.dump

# 4. Copy backup from container to your computer
docker cp alphapulse_postgres:/tmp/alphapulse_backup.dump ./alphapulse_backup_$(Get-Date -Format "yyyy-MM-dd_HHmmss").dump

# 5. Backup Redis data (optional, mostly cache)
docker exec alphapulse_redis redis-cli SAVE
docker cp alphapulse_redis:/data/dump.rdb ./redis_backup_$(Get-Date -Format "yyyy-MM-dd_HHmmss").rdb

# 6. Export all Docker volumes (SAFEST METHOD)
docker run --rm -v postgres_data:/data -v D:\Backups:/backup ubuntu tar czf /backup/postgres_data_backup.tar.gz /data

cd ..
```

### Method 2: Backup Entire Docker Volume

```powershell
# This backs up the ENTIRE volume including all data
docker run --rm -v postgres_data:/data -v ${PWD}/backups:/backup ubuntu tar czf /backup/postgres_full_backup_$(date +%Y%m%d).tar.gz -C /data .
```

### Method 3: SQL Dump (Human-Readable)

```powershell
# Create SQL text backup
docker exec alphapulse_postgres pg_dump -U alpha_emon -d alphapulse > ./backups/alphapulse_backup_$(Get-Date -Format "yyyy-MM-dd").sql
```

---

## 🔄 RESTORE YOUR DATA

### Restore from Dump File

```powershell
# 1. Make sure containers are running
docker-compose up -d

# 2. Copy backup into container
docker cp ./backups/alphapulse_backup_YYYY-MM-DD.dump alphapulse_postgres:/tmp/

# 3. Restore the database
docker exec alphapulse_postgres pg_restore -U alpha_emon -d alphapulse -c /tmp/alphapulse_backup.dump
```

### Restore from SQL File

```powershell
# Restore from SQL file
docker exec -i alphapulse_postgres psql -U alpha_emon -d alphapulse < ./backups/alphapulse_backup_YYYY-MM-DD.sql
```

### Restore Entire Volume

```powershell
# Restore complete volume from backup
docker run --rm -v postgres_data:/data -v ${PWD}/backups:/backup ubuntu tar xzf /backup/postgres_full_backup_YYYYMMDD.tar.gz -C /data
```

---

## 🚀 Quick Start Commands

### Start AlphaPulse (Development)

```powershell
cd "D:\Emon Work\AlphaPuls\infrastructure\docker-compose"
docker-compose -f docker-compose.yml up -d
```

### Start AlphaPulse (Production)

```powershell
cd "D:\Emon Work\AlphaPuls\infrastructure\docker-compose"
docker-compose -f docker-compose.production.yml up -d
```

### Check Status

```powershell
docker ps
docker-compose logs -f
```

### Stop Everything

```powershell
docker-compose down
```

### Stop and REMOVE Volumes (⚠️ DANGER - DELETES DATA!)

```powershell
docker-compose down -v  # DON'T RUN THIS unless you want to delete everything!
```

---

## 📊 Database Connection Details for Applications

### Python Connection

```python
import asyncpg

# Connection parameters
conn = await asyncpg.connect(
    host='localhost',
    port=5432,
    user='alpha_emon',
    password='Emon_@17711',
    database='alphapulse'
)
```

### SQLAlchemy Connection

```python
from sqlalchemy import create_engine

DATABASE_URL = "postgresql://alpha_emon:Emon_@17711@localhost:5432/alphapulse"
engine = create_engine(DATABASE_URL)
```

### Docker Internal Connection (from backend container)

```python
DATABASE_URL = "postgresql://alpha_emon:Emon_@17711@postgres:5432/alphapulse"
```

---

## 🔧 Important Docker Commands

### View Volumes

```powershell
docker volume ls
docker volume inspect postgres_data
```

### View Volume Size (Check your data size)

```powershell
docker system df -v
```

### Access Database Directly

```powershell
# Connect to database via psql
docker exec -it alphapulse_postgres psql -U alpha_emon -d alphapulse

# Common commands inside psql:
# \dt              - List all tables
# \d+ table_name   - Describe table
# \l               - List databases
# \q               - Quit
```

### Access Redis Directly

```powershell
docker exec -it alphapulse_redis redis-cli
```

---

## 📁 All Configuration Files Locations

### Docker Compose Files
- Main: `infrastructure/docker-compose/docker-compose.yml`
- Production: `infrastructure/docker-compose/docker-compose.production.yml`
- Development: `infrastructure/docker-compose/docker-compose.development.yml`
- Enhanced: `infrastructure/docker-compose/docker-compose.enhanced.yml`
- External Services: `infrastructure/docker-compose/docker-compose.external-services.yml`

### Dockerfile Locations
- Backend: `infrastructure/docker/Dockerfile.backend`
- Backend Prod: `infrastructure/docker/Dockerfile.backend.production`
- Frontend: `infrastructure/docker/Dockerfile.web`
- Frontend Prod: `infrastructure/docker/Dockerfile.frontend.production`

### Configuration Files
- Backend Config: `apps/backend/src/app/core/config.py`
- Unified Config: `apps/backend/src/app/core/unified_config.py`
- Database Connection: `apps/backend/src/database/connection.py`
- Env Template: `env.template`
- Backend Env Template: `apps/backend/config/config/config.env.template`

---

## ⚙️ Environment Variables Reference

Create a `.env` file in the root directory with these variables:

```bash
# Database Configuration
TIMESCALEDB_HOST=localhost
TIMESCALEDB_PORT=5432
TIMESCALEDB_DATABASE=alphapulse
TIMESCALEDB_USERNAME=alpha_emon
TIMESCALEDB_PASSWORD=Emon_@17711
TIMESCALEDB_URL=postgresql://alpha_emon:Emon_@17711@localhost:5432/alphapulse
DATABASE_URL=postgresql://alpha_emon:Emon_@17711@localhost:5432/alphapulse

# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
REDIS_URL=redis://localhost:6379
REDIS_ENABLED=true

# For Docker Internal Communication
DB_HOST=postgres
DB_PORT=5432
DB_NAME=alphapulse
DB_USER=alpha_emon
DB_PASSWORD=Emon_@17711

# Application Settings
APP_NAME=AlphaPulse Trading Bot
VERSION=1.0.0
DEBUG=false
LOG_LEVEL=INFO
ENVIRONMENT=production

# Server Settings
HOST=0.0.0.0
PORT=8000
WORKERS=4

# Trading Configuration
TRADING_ENABLED=false
PAPER_TRADING=true
RISK_LIMIT_PERCENT=2.0
MAX_OPEN_POSITIONS=10
DEFAULT_POSITION_SIZE=0.01

# API Keys (Add your actual keys)
BINANCE_API_KEY=your_key_here
BINANCE_SECRET_KEY=your_secret_here
COINGECKO_API_KEY=your_key_here
NEWS_API_KEY=your_key_here
TWITTER_API_KEY=your_key_here
HUGGINGFACE_API_KEY=your_key_here

# Security
SECRET_KEY=your_secret_key_here
JWT_SECRET_KEY=your_jwt_secret_key_here
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Performance
MAX_WORKERS=4
CHUNK_SIZE=1000
CACHE_TTL=3600
DB_POOL_SIZE=20
DB_MAX_OVERFLOW=30
```

---

## 🛡️ CRITICAL SAFETY NOTES

### ⚠️ BEFORE YOU DO ANYTHING:

1. **BACKUP YOUR DATA FIRST!** Run the backup commands above
2. **Never run `docker-compose down -v`** - This will delete your 1 year of data!
3. **Keep multiple backups** - Store them in different locations
4. **Test your backups** - Make sure they actually work
5. **After Windows restart** - Docker volumes persist, your data is safe

### Docker Volume Persistence

- ✅ Volumes survive container restarts
- ✅ Volumes survive `docker-compose down`
- ✅ Volumes survive `docker-compose up`
- ✅ Volumes survive system reboot (after Docker starts)
- ❌ Volumes are DELETED with `docker-compose down -v`
- ❌ Volumes are DELETED with `docker volume rm`

---

## 📞 Quick Troubleshooting

### Database Connection Issues

```powershell
# Check if container is running
docker ps | grep postgres

# Check container logs
docker logs alphapulse_postgres

# Verify database exists
docker exec alphapulse_postgres psql -U alpha_emon -d alphapulse -c "\dt"
```

### Redis Connection Issues

```powershell
# Check Redis
docker ps | grep redis
docker logs alphapulse_redis

# Test connection
docker exec alphapulse_redis redis-cli ping
```

### View Database Size

```powershell
docker exec alphapulse_postgres psql -U alpha_emon -d alphapulse -c "SELECT pg_size_pretty(pg_database_size('alphapulse'));"
```

---

## 🎯 RECOMMENDED WORKFLOW AFTER REBOOT

1. **Start Docker Desktop** (wait for it to fully start)
2. **Verify Docker is running**: `docker ps`
3. **Start your containers**: `docker-compose up -d`
4. **Check everything is healthy**: `docker ps` (all should be "healthy")
5. **Verify database connection**: Use the quick test commands above
6. **Create a backup** (if you haven't recently)

---

## 📦 Backup Schedule Recommendation

- **Daily**: Automated SQL dumps
- **Weekly**: Full volume backups
- **Monthly**: Off-site backup copies
- **Before updates**: Always backup before changing anything!

---

## 🎓 Additional Resources

### Docker Volume Location (Windows)
```
C:\Users\EA Soft Lab\AppData\Local\Docker\wsl\data\
```

This is where Docker stores ALL your volumes in WSL2.
This directory is managed by Docker Desktop.

### Manual Volume Inspection
```powershell
# List all Docker volumes with details
docker volume ls

# Inspect specific volume
docker volume inspect postgres_data

# See mount point
docker volume inspect postgres_data --format '{{ .Mountpoint }}'
```

---

## ✅ Verification Checklist

After restore, verify everything works:

- [ ] Docker containers are running: `docker ps`
- [ ] Database is accessible: `docker exec alphapulse_postgres psql -U alpha_emon -d alphapulse -c "SELECT 1;"`
- [ ] Redis is working: `docker exec alphapulse_redis redis-cli ping`
- [ ] Tables exist: `docker exec alphapulse_postgres psql -U alpha_emon -d alphapulse -c "\dt"`
- [ ] Data is present: `docker exec alphapulse_postgres psql -U alpha_emon -d alphapulse -c "SELECT COUNT(*) FROM market_data;"`
- [ ] Backend connects successfully
- [ ] Frontend can reach backend

---

## 🆘 Emergency Data Recovery

If Docker Desktop gets corrupted or uninstalled:

1. **DO NOT PANIC!** - Your data is still in the WSL2 filesystem
2. **Reinstall Docker Desktop**
3. **Volumes should automatically reappear**
4. **If not, restore from your backup files**

---

**Created**: $(Get-Date)  
**For**: AlphaPulse Trading System  
**Data Importance**: CRITICAL (1 year of trading data)  
**Status**: Active System with Production Data

---

**🔥 REMEMBER: Always create a backup BEFORE making changes!**

