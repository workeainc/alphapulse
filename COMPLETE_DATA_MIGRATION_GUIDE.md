# Complete Data Migration Guide - Docker to PostgreSQL on Windows

## Your Situation

✅ **GOOD NEWS:** Your data exists!
- **File:** `docker_data.vhdx`
- **Size:** 28.58 GB (contains your 1 year of trading data!)
- **Location:** `C:\Users\EA Soft Lab\AppData\Local\Docker\wsl\disk\docker_data.vhdx`

❌ **Problem:** Docker Desktop won't start
✅ **Solution:** Extract data directly from VHDX, then use PostgreSQL on Windows

---

## Complete Migration Plan (30-45 minutes)

###  PHASE 1: Extract Data from Docker (10-15 minutes)

**Run as Administrator:**

```powershell
.\EXTRACT_DATABASE_FROM_DOCKER.ps1
```

This will:
1. Mount your Docker VHDX file
2. Find PostgreSQL data inside
3. Copy to `D:\AlphaPulse_Database_Backup\exported_data`
4. Unmount VHDX

---

### PHASE 2: Install PostgreSQL on Windows (10 minutes)

**Run:**

```powershell
.\install_postgres_windows.ps1
```

Or manually:
1. Go to https://www.postgresql.org/download/windows/
2. Download PostgreSQL 15.x installer
3. Install with defaults
4. Set password for 'postgres' user (remember it!)
5. Use port 5432

---

### PHASE 3: Create AlphaPulse Database (2 minutes)

```powershell
# Open PowerShell
psql -U postgres

# In psql console:
CREATE DATABASE alphapulse;
CREATE USER alpha_emon WITH PASSWORD 'Emon_@17711';
GRANT ALL PRIVILEGES ON DATABASE alphapulse TO alpha_emon;
\q
```

---

### PHASE 4: Import Your Data (10-20 minutes)

**Method A: If extraction created SQL file**

```powershell
psql -U postgres -d alphapulse < D:\AlphaPulse_Database_Backup\exported_data\backup.sql
```

**Method B: If extraction created data directory**

1. Stop PostgreSQL service:
```powershell
Stop-Service postgresql-x64-15
```

2. Copy exported data to PostgreSQL data directory:
```powershell
$pgDataDir = "C:\Program Files\PostgreSQL\15\data"
Copy-Item "D:\AlphaPulse_Database_Backup\exported_data\postgres_data\*" -Destination "$pgDataDir\" -Recurse -Force
```

3. Fix ownership (run as postgres user):
```powershell
icacls "$pgDataDir\*" /grant postgres:F /T
```

4. Start PostgreSQL:
```powershell
Start-Service postgresql-x64-15
```

---

### PHASE 5: Update AlphaPulse Configuration (2 minutes)

Create or update `.env` file in project root:

```bash
# Database Configuration
TIMESCALEDB_HOST=localhost
TIMESCALEDB_PORT=5432
TIMESCALEDB_DATABASE=alphapulse
TIMESCALEDB_USERNAME=alpha_emon
TIMESCALEDB_PASSWORD=Emon_@17711
DATABASE_URL=postgresql://alpha_emon:Emon_@17711@localhost:5432/alphapulse

# Redis (optional, can skip for now)
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_URL=redis://localhost:6379

# Application
DEBUG=false
LOG_LEVEL=INFO
```

---

### PHASE 6: Test Everything (5 minutes)

```powershell
# 1. Test database connection
psql -U alpha_emon -d alphapulse -c "SELECT COUNT(*) FROM information_schema.tables;"

# 2. Check if your data is there
psql -U alpha_emon -d alphapulse -c "\dt"

# 3. Run AlphaPulse backend
cd D:\Emon Work\AlphaPuls\apps\backend
python -m src.main
```

---

## Alternative: Simpler Approach (If Extraction is Too Complex)

If extracting data proves difficult:

### Option B: Start Fresh + Keep Old VHDX as Backup

1. Install PostgreSQL on Windows (10 min)
2. Start fresh with new database
3. Keep the `docker_data.vhdx` file as backup
4. Try to extract data later when you have more time

**Advantages:**
- Get working immediately
- Can extract old data later
- Less risky

---

## Troubleshooting

### VHDX Won't Mount

**Error:** "Access Denied" or "In Use"

**Solution:**
```powershell
# Make sure Docker is completely stopped
Get-Process *docker* | Stop-Process -Force

# Try mounting again
Mount-VHD -Path "C:\Users\EA Soft Lab\AppData\Local\Docker\wsl\disk\docker_data.vhdx" -ReadOnly
```

### Can't Find PostgreSQL Data in VHDX

**Manual Search:**
```powershell
# After mounting (assume drive Z:)
Get-ChildItem Z:\ -Recurse -Directory | Where-Object {$_.Name -like "*postgres*"}
```

Look for directories containing files like:
- `PG_VERSION`
- `base/`
- `global/`
- `pg_wal/`

### PostgreSQL Won't Start After Import

**Check logs:**
```powershell
Get-Content "C:\Program Files\PostgreSQL\15\data\log\*.log" -Tail 50
```

**Common fixes:**
- Check permissions on data directory
- Verify postgresql.conf settings
- Check port 5432 is not in use

---

## Summary: Your Best Options

### Option 1: Full Migration (Recommended if you need the data)
1. Extract data from VHDX ✓
2. Install PostgreSQL ✓
3. Import data ✓
4. Run AlphaPulse ✓

**Time:** 45 minutes
**Complexity:** Medium
**Result:** All your data in working system

### Option 2: Fresh Start (Recommended if time is valuable)
1. Install PostgreSQL ✓
2. Create new database ✓
3. Run AlphaPulse ✓
4. Keep VHDX as backup ✓

**Time:** 15 minutes
**Complexity:** Easy
**Result:** Working system immediately, data extractable later

---

## My Recommendation

**Start with Option 2 (Fresh Start):**
1. Install PostgreSQL on Windows NOW (15 minutes)
2. Get AlphaPulse working with new database
3. Keep the 28.58 GB VHDX file safe
4. Extract old data when you have time (weekend project)
5. Then merge/import old data if needed

**Why?**
- You'll be productive immediately
- Less risk of data corruption
- Can extract old data anytime
- VHDX file isn't going anywhere

**The VHDX file is your backup** - it's safe where it is!

---

## What To Do Right Now

**Immediate Action:**

```powershell
# Run this to install PostgreSQL and get working:
.\install_postgres_windows.ps1
```

**Then:**
1. Create alphapulse database (2 minutes)
2. Update .env file (1 minute)
3. Run AlphaPulse backend (1 minute)
4. **You're working!**

**Later (when you have time):**
1. Run extraction script on the VHDX
2. Import old data
3. Merge with new data if needed

---

**Your data is SAFE in that 28.58 GB file. Let's get you working first, then we can extract the old data!**


