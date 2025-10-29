# AlphaPulse Database Setup - Complete ✅

## Overview

The AlphaPulse PostgreSQL/TimescaleDB database is now running and ready for use.

**Status:** ✅ **COMPLETE**  
**Date:** October 26, 2025  
**Port:** 55433 (to avoid conflict with other PostgreSQL instances)  

---

## Database Configuration

### Connection Details

```
Host: localhost
Port: 55433
Database: alphapulse
User: alpha_emon
Password: Emon_@17711
```

### Connection String

```
postgresql://alpha_emon:Emon_%4017711@localhost:55433/alphapulse
```

**Note:** The `@` symbol in the password is URL-encoded as `%40`

---

## Docker Container

### Container Information

- **Container Name:** `alphapulse_postgres`
- **Image:** `timescale/timescaledb:latest-pg15`
- **PostgreSQL Version:** 15.13
- **TimescaleDB Version:** 2.22.1
- **Status:** Running ✅
- **Port Mapping:** `55433:5432`

### Docker Commands

```bash
# Start the database
cd docker
docker-compose up -d postgres

# Stop the database
docker-compose stop postgres

# View logs
docker logs alphapulse_postgres

# Access database shell
docker exec -it alphapulse_postgres psql -U alpha_emon -d alphapulse

# Check status
docker ps --filter "name=alphapulse_postgres"
```

---

## Migration Status

### Database Migration: NOT NEEDED ✅

The database is completely fresh with **zero tables**. This is actually perfect because:

1. **Code Already Updated** ✅
   - All model code uses `SignalRecommendation` instead of `Trade`
   - Table name is `signal_recommendations` (not `trades`)
   - All imports updated throughout codebase

2. **No Migration Required** ✅
   - When you create tables, they'll automatically use the new schema
   - No need to rename `trades` → `signal_recommendations`
   - Migration script exists if you ever need it for existing data

3. **Migration Script Available** ✅
   - Location: `backend/database/migrations/rename_trades_to_recommendations.py`
   - Ready to use if you ever import old data
   - Supports both forward and rollback migrations

---

## Table Schema

### Primary Tables (Will be created on first use)

When you run the application, these tables will be created with the **new** schema:

1. **`signal_recommendations`** (formerly `trades`)
   - Stores trading signal recommendations
   - Fields: `suggested_entry_price`, `suggested_stop_loss`, `suggested_take_profit`, etc.
   - Status values: `'pending'`, `'user_executed'`, `'expired'`, `'cancelled'`

2. **`signals`**
   - Stores raw trading signals from analysis
   - Linked to signal_recommendations via foreign key

3. **`logs`**
   - System and signal generation logs

4. **`feedback`**
   - User feedback on signal outcomes

5. **`performance_metrics`**
   - System performance tracking

6. **`market_regimes`**
   - Market regime classification data

---

## TimescaleDB Features

### Extension Installed ✅

TimescaleDB extension is installed and ready:
- Version: 2.22.1
- Status: Active
- Features: Hypertables, continuous aggregates, compression

### Hypertables

When tables are created, time-series tables will automatically be converted to hypertables for:
- Optimized time-series queries
- Automatic data compression
- Data retention policies
- Improved query performance

---

## Environment Variables

### For Backend Services

Add to your `.env` file:

```bash
# Database Configuration
DATABASE_URL=postgresql://alpha_emon:Emon_%4017711@localhost:55433/alphapulse
TIMESCALEDB_URL=postgresql://alpha_emon:Emon_%4017711@localhost:55433/alphapulse

# Database Pool Settings
DB_POOL_SIZE=20
DB_MAX_OVERFLOW=10
DB_POOL_TIMEOUT=30
```

### For Docker Compose Services

Already configured in `docker/docker-compose.yml`:

```yaml
environment:
  - DATABASE_URL=postgresql://alpha_emon:Emon_@17711@postgres:5432/alphapulse
```

**Note:** Inside Docker network, use `postgres` as hostname and port `5432`

---

## Next Steps

### 1. Initialize Database Schema

Run one of these scripts to create tables:

```bash
# Option A: Using Alembic migrations
cd backend
alembic upgrade head

# Option B: Using init script
python database/migrations/init_db.py

# Option C: Let the application create tables on first run
python backend/main.py
```

### 2. Verify Tables Created

```bash
# List all tables
docker exec alphapulse_postgres psql -U alpha_emon -d alphapulse -c "\dt"

# Check signal_recommendations table
docker exec alphapulse_postgres psql -U alpha_emon -d alphapulse -c "\d signal_recommendations"
```

### 3. Verify TimescaleDB Hypertables

```bash
docker exec alphapulse_postgres psql -U alpha_emon -d alphapulse -c "SELECT * FROM timescaledb_information.hypertables;"
```

---

## Troubleshooting

### Container Won't Start

```bash
# Check logs
docker logs alphapulse_postgres

# Remove and recreate
cd docker
docker-compose down postgres
docker-compose up -d postgres
```

### Port Already in Use

If port 55433 is also taken, edit `docker/docker-compose.yml`:

```yaml
ports:
  - "55434:5432"  # Change to different port
```

Then update connection strings accordingly.

### Connection Refused

```bash
# Ensure container is running
docker ps --filter "name=alphapulse_postgres"

# Check if PostgreSQL is accepting connections
docker exec alphapulse_postgres pg_isready -U alpha_emon
```

### Password Authentication Failed

Ensure you're using URL-encoded password in connection strings:
- Raw password: `Emon_@17711`
- URL-encoded: `Emon_%4017711` (@ becomes %40)

---

## Database Backup & Restore

### Backup

```bash
# Backup entire database
docker exec alphapulse_postgres pg_dump -U alpha_emon alphapulse > alphapulse_backup.sql

# Backup specific table
docker exec alphapulse_postgres pg_dump -U alpha_emon -t signal_recommendations alphapulse > recommendations_backup.sql
```

### Restore

```bash
# Restore database
docker exec -i alphapulse_postgres psql -U alpha_emon alphapulse < alphapulse_backup.sql
```

---

## Performance Tuning

TimescaleDB has been auto-tuned for your system:

```
shared_buffers = 5977MB
effective_cache_size = 17933MB
maintenance_work_mem = 2047MB
work_mem = 3060kB
max_worker_processes = 39
timescaledb.max_background_workers = 16
```

These settings are optimized for:
- Available memory: 23.35 GB
- CPU cores: 20
- PostgreSQL version: 15

---

## Security Notes

### Production Deployment

For production, change these defaults:

1. **Change Password:**
   ```sql
   ALTER USER alpha_emon WITH PASSWORD 'new_secure_password';
   ```

2. **Use Environment Variables:**
   - Don't hardcode credentials
   - Use Docker secrets or env files

3. **Enable SSL:**
   ```yaml
   environment:
     POSTGRES_SSL_MODE: require
   ```

4. **Restrict Access:**
   - Don't expose port publicly
   - Use VPN or internal network only

---

## Summary

✅ **Database Running:** Port 55433  
✅ **TimescaleDB Active:** Version 2.22.1  
✅ **No Migration Needed:** Fresh database  
✅ **Schema Updated:** All code uses new `SignalRecommendation` model  
✅ **Ready for Use:** Can create tables and start application  

**Next:** Initialize tables by running the application or migration scripts.

---

**Document Created:** October 26, 2025  
**Last Updated:** October 26, 2025  
**Status:** COMPLETE ✅

