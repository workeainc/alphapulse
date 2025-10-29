# Simple Solution: Run AlphaPulse WITHOUT Docker

## The Problem

Docker Desktop has multiple issues on your system and isn't worth the hassle right now.

## The Simple Solution

**Install PostgreSQL and Redis directly on Windows** - No Docker needed!

---

## Step-by-Step Guide

### STEP 1: Install PostgreSQL on Windows

1. **Download PostgreSQL**
   - Go to: https://www.postgresql.org/download/windows/
   - Click "Download the installer"
   - Choose version 15.x (latest stable)
   - Download the Windows x86-64 installer

2. **Run the Installer**
   - Run the downloaded `.exe` file
   - Click "Next" through the wizard
   - **IMPORTANT:** Set a password for postgres user (remember it!)
   - Use port `5432` (default)
   - Install all components
   - Complete the installation

3. **Verify Installation**
   ```powershell
   # Should show PostgreSQL service running
   Get-Service -Name "*postgres*"
   ```

---

### STEP 2: Install Redis on Windows (Optional)

Redis is only needed for caching, not critical.

1. **Download Redis**
   - Go to: https://github.com/microsoftarchive/redis/releases
   - Download latest `.msi` file
   - Or use: `choco install redis` if you have Chocolatey

2. **Install and Start**
   - Run the installer
   - Redis will start automatically

---

### STEP 3: Create Your Database

```powershell
# Open PowerShell and run:
psql -U postgres

# In psql console:
CREATE DATABASE alphapulse;
CREATE USER alpha_emon WITH PASSWORD 'Emon_@17711';
GRANT ALL PRIVILEGES ON DATABASE alphapulse TO alpha_emon;
\q
```

---

### STEP 4: Update AlphaPulse Configuration

Update your `.env` file or create one:

```bash
# Database Configuration
TIMESCALEDB_HOST=localhost
TIMESCALEDB_PORT=5432
TIMESCALEDB_DATABASE=alphapulse
TIMESCALEDB_USERNAME=alpha_emon
TIMESCALEDB_PASSWORD=Emon_@17711
DATABASE_URL=postgresql://alpha_emon:Emon_@17711@localhost:5432/alphapulse

# Redis (if installed)
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_URL=redis://localhost:6379

# Other settings
DEBUG=true
LOG_LEVEL=INFO
```

---

### STEP 5: Install TimescaleDB Extension

TimescaleDB is needed for time-series data:

```powershell
# Download TimescaleDB
# Go to: https://www.timescale.com/download
# Select PostgreSQL 15, Windows
# Download and run installer
```

Then enable it:
```sql
psql -U postgres -d alphapulse
CREATE EXTENSION IF NOT EXISTS timescaledb;
\q
```

---

### STEP 6: Run Your AlphaPulse Backend

```powershell
# Navigate to backend directory
cd apps/backend

# Install Python dependencies (if not already done)
pip install -r requirements.txt

# Run the backend
python -m src.main
# or
uvicorn src.main:app --reload
```

---

## Advantages of This Approach

✅ **No Docker issues** - Everything runs natively on Windows
✅ **Faster** - No virtualization overhead  
✅ **Easier debugging** - Direct access to everything
✅ **More stable** - No WSL2 or Hyper-V dependencies
✅ **Better performance** - Native Windows services

---

## Your Data Migration Options

### Option A: Start Fresh (Easiest)

Just start with a new database and begin collecting data again.

### Option B: Export Data from Docker Volume (If Docker worked at some point)

If Docker ever worked and you have data in volumes:

1. Get Docker working temporarily
2. Run this:
   ```powershell
   docker exec alphapulse_postgres pg_dump -U alpha_emon -d alphapulse > backup.sql
   ```
3. Then restore to Windows PostgreSQL:
   ```powershell
   psql -U postgres -d alphapulse < backup.sql
   ```

### Option C: Access Docker Volume Directly

The data is in: `C:\Users\EA Soft Lab\AppData\Local\Docker\wsl\`

You can:
1. Find the `.vhdx` file containing your data
2. Mount it
3. Extract the PostgreSQL data files
4. Import them

(This is technical but possible if you really need the old data)

---

## Quick Setup Script

```powershell
# Run this after installing PostgreSQL

# Create database
$env:PGPASSWORD = "your_postgres_password"
psql -U postgres -c "CREATE DATABASE alphapulse;"
psql -U postgres -c "CREATE USER alpha_emon WITH PASSWORD 'Emon_@17711';"
psql -U postgres -c "GRANT ALL PRIVILEGES ON DATABASE alphapulse TO alpha_emon;"

# Test connection
psql -U alpha_emon -d alphapulse -c "SELECT version();"
```

---

## Running AlphaPulse

Once PostgreSQL is installed and configured:

```powershell
# 1. Start PostgreSQL (usually automatic)
# Check: Get-Service postgresql*

# 2. Start Redis (if installed)
# Check: Get-Service redis

# 3. Run AlphaPulse backend
cd D:\Emon Work\AlphaPuls\apps\backend
python -m src.main

# 4. Run frontend (in another terminal)
cd D:\Emon Work\AlphaPuls\apps\web
npm run dev
```

---

## Troubleshooting

### PostgreSQL won't start
```powershell
# Check service
Get-Service postgresql*

# Start manually
Start-Service postgresql-x64-15
```

### Can't connect to database
```powershell
# Test connection
psql -U postgres -d alphapulse

# If password issues, edit: C:\Program Files\PostgreSQL\15\data\pg_hba.conf
# Change "md5" to "trust" temporarily for testing
```

### Port already in use
```powershell
# Check what's using port 5432
netstat -ano | findstr :5432

# Kill the process or change PostgreSQL port in postgresql.conf
```

---

## Summary

**Forget Docker for now!**

1. Install PostgreSQL on Windows (10 minutes)
2. Create alphapulse database (2 minutes)
3. Update your `.env` file (1 minute)
4. Run AlphaPulse backend directly (1 minute)

**Total time: 15 minutes** vs. hours fighting with Docker!

---

## Need Your Old Data?

If you absolutely need to recover your 1 year of data from Docker volumes, we can:

1. Try to get Docker working one more time just long enough to export
2. Or manually extract data from the Docker volume files
3. Or help you rebuild from any other backups you have

But honestly, **starting fresh might be faster and cleaner** at this point!

---

**This is the SIMPLEST solution - no virtualization, no Docker, just native Windows PostgreSQL!**


