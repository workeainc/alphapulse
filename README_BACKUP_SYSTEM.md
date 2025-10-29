# 🔒 AlphaPulse Backup System - Quick Reference

## 📁 Files Created in Root Directory

All your database, Redis, and Docker configurations are now saved in the root directory for easy access!

### Main Documentation
- **`ALPHAPULSE_BACKUP_AND_RESTORE_GUIDE.md`** - Complete guide with all configurations

### Backup Scripts (PowerShell)
- **`backup_database.ps1`** - Full backup with statistics
- **`restore_database.ps1`** - Interactive restore from backup
- **`quick_backup.ps1`** - Fast backup without prompts

---

## 🚀 Quick Commands

### Create a Backup

```powershell
# Full backup with details
.\backup_database.ps1

# Quick backup (fast)
.\quick_backup.ps1
```

### Restore a Backup

```powershell
.\restore_database.ps1
```

---

## 📊 Your Database Info

```yaml
Database Type: TimescaleDB (PostgreSQL + Time-series)
Database Name: alphapulse
Username: alpha_emon
Password: Emon_@17711
Port: 5432 (internal) / 55433 (external)
Data Location: Docker volume 'postgres_data'
Physical Path: C:\Users\EA Soft Lab\AppData\Local\Docker\wsl\data\
```

---

## 🔴 Your Redis Info

```yaml
Host: localhost (or 'redis' in Docker)
Port: 6379
Database: 0
Password: None
Data Location: Docker volume 'redis_data'
```

---

## 🐳 Docker Commands

### Start AlphaPulse

```powershell
cd infrastructure/docker-compose
docker-compose up -d
```

### Check Status

```powershell
docker ps
docker-compose logs -f
```

### Stop AlphaPulse

```powershell
docker-compose down
# ⚠️ NEVER use 'down -v' - it deletes your data!
```

---

## ⚠️ CRITICAL REMINDERS

### ✅ SAFE Commands (Keep your 1 year of data)
- `docker-compose up -d` - Start containers
- `docker-compose down` - Stop containers
- `docker-compose restart` - Restart containers
- System reboot - Data persists after restart

### ❌ DANGEROUS Commands (Will delete your data!)
- `docker-compose down -v` - Deletes all volumes and data
- `docker volume rm postgres_data` - Deletes database
- `docker system prune -a --volumes` - Deletes everything

---

## 📋 What Data is in Docker Volumes?

| Volume Name | Contains | Importance |
|------------|----------|------------|
| `postgres_data` | **1 YEAR OF TRADING DATA** | 🔥 CRITICAL |
| `redis_data` | Cache & temporary data | Medium |
| `postgres_data_prod` | Production database (if used) | 🔥 CRITICAL |

---

## 🎯 Workflow After Windows Restart

1. Open Docker Desktop (wait for it to start completely)
2. Check Docker: `docker ps`
3. If no containers running, start them:
   ```powershell
   cd infrastructure/docker-compose
   docker-compose up -d
   ```
4. Wait 30 seconds for services to be healthy
5. Verify: `docker ps` (should show healthy status)
6. Test backend: `http://localhost:8000`
7. Test frontend: `http://localhost:3000`

---

## 💾 Backup Strategy Recommendation

| Frequency | Method | Retention |
|-----------|--------|-----------|
| **Daily** | `.\quick_backup.ps1` | Keep 7 days |
| **Weekly** | `.\backup_database.ps1` | Keep 4 weeks |
| **Monthly** | Full backup + copy to external drive | Keep 12 months |
| **Before updates** | Always backup! | Keep until verified |

---

## 🆘 Emergency Scenarios

### Scenario 1: Docker Desktop Won't Start
1. Check WSL2 is enabled: `wsl --status`
2. Restart computer
3. Reinstall Docker Desktop (data is safe in WSL)
4. Your volumes will reappear automatically

### Scenario 2: Lost Data / Corrupted Database
1. Run: `.\restore_database.ps1`
2. Select your most recent backup
3. Follow the prompts

### Scenario 3: Need to Move to New Computer
1. Create backup: `.\backup_database.ps1`
2. Copy entire `backups/` folder to USB/cloud
3. On new computer:
   - Install Docker Desktop
   - Copy backup files
   - Run: `.\restore_database.ps1`

### Scenario 4: Accidentally Ran `docker-compose down -v`
1. DON'T PANIC! Your backup files still exist
2. Start containers: `docker-compose up -d`
3. Restore: `.\restore_database.ps1`
4. Select your latest backup

---

## 📞 Quick Tests

### Test Database Connection

```powershell
docker exec alphapulse_postgres psql -U alpha_emon -d alphapulse -c "SELECT 1;"
```

### Test Redis Connection

```powershell
docker exec alphapulse_redis redis-cli ping
```

### Check Database Size

```powershell
docker exec alphapulse_postgres psql -U alpha_emon -d alphapulse -c "SELECT pg_size_pretty(pg_database_size('alphapulse'));"
```

### List All Tables

```powershell
docker exec alphapulse_postgres psql -U alpha_emon -d alphapulse -c "\dt"
```

### Access Database CLI

```powershell
docker exec -it alphapulse_postgres psql -U alpha_emon -d alphapulse
```

---

## 🎓 Important File Locations

### Configuration Files
- Docker Compose: `infrastructure/docker-compose/docker-compose.yml`
- Backend Config: `apps/backend/src/app/core/config.py`
- Database Connection: `apps/backend/src/database/connection.py`
- Environment Template: `env.template`

### Data Directories
- Backups: `backups/` (create this by running backup script)
- Docker Volumes: `C:\Users\EA Soft Lab\AppData\Local\Docker\wsl\data\`

---

## ✅ Next Steps After Reboot

1. **Reboot your computer now** (to enable Virtual Machine Platform)
2. **After reboot:**
   - Docker Desktop will start automatically
   - Wait for Docker whale icon to appear in system tray
   - Run: `docker ps` to verify Docker is working
3. **Start AlphaPulse:**
   ```powershell
   cd infrastructure/docker-compose
   docker-compose up -d
   ```
4. **Create your first backup:**
   ```powershell
   .\backup_database.ps1
   ```

---

## 📚 Full Documentation

See **`ALPHAPULSE_BACKUP_AND_RESTORE_GUIDE.md`** for:
- Complete connection details
- All environment variables
- Detailed troubleshooting
- Advanced backup methods
- Production configurations
- Security notes

---

**🔥 IMPORTANT: Your 1 year of trading data is safe in Docker volumes!**

**💡 TIP: Create a backup RIGHT NOW after your reboot!**

```powershell
# After Docker starts, run this:
.\backup_database.ps1
```

---

**Created**: 2025-10-29  
**Purpose**: Preserve AlphaPulse database and configurations  
**Status**: Ready to use after reboot  
**Data Safety**: ✅ All files created for easy backup/restore

