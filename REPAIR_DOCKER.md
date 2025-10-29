# Docker Desktop Repair Guide

## Problem Identified

**Error:** `cannot find registry key "SOFTWARE\\Docker Inc.\\Docker Desktop"`

**Cause:** Docker Desktop's registry keys are missing or corrupted, preventing it from starting.

**Solution:** Repair or reinstall Docker Desktop

---

## Option 1: Quick Repair (Try This First)

### Step 1: Download Docker Desktop Installer
1. Go to: https://www.docker.com/products/docker-desktop/
2. Click "Download for Windows"
3. Save the installer

### Step 2: Run Installer to Repair
1. **Close all Docker processes** (if any are running)
2. Run the downloaded `Docker Desktop Installer.exe`
3. The installer will detect existing installation
4. Choose **"Repair"** or just install over existing
5. Click **"OK"** on prompts
6. **IMPORTANT:** When installation completes, restart computer

### Step 3: After Restart
1. Docker Desktop should start automatically
2. Or launch it from Start Menu
3. Wait for whale icon in system tray
4. Test with: `docker ps`

---

## Option 2: Clean Reinstall (If Repair Doesn't Work)

### Step 1: Uninstall Docker Desktop

**Method A - Windows Settings:**
1. Press `Win + I` (Settings)
2. Go to "Apps" > "Installed apps"
3. Find "Docker Desktop"
4. Click three dots > "Uninstall"
5. Follow prompts

**Method B - Control Panel:**
1. Open Control Panel
2. Go to "Programs and Features"
3. Find "Docker Desktop"
4. Right-click > "Uninstall"
5. Follow prompts

### Step 2: Clean Up (Optional but Recommended)
Delete these folders if they exist:
- `C:\Program Files\Docker`
- `C:\ProgramData\Docker`
- `C:\Users\<YourName>\AppData\Local\Docker`
- `C:\Users\<YourName>\AppData\Roaming\Docker`

**IMPORTANT:** Your data in Docker volumes is stored elsewhere and is SAFE!
- Volume location: `C:\Users\EA Soft Lab\AppData\Local\Docker\wsl\data`
- **DO NOT DELETE** the `wsl\data` folder!

### Step 3: Reinstall Docker Desktop
1. Download fresh installer from: https://www.docker.com/products/docker-desktop/
2. Run installer as Administrator
3. Accept defaults
4. Restart computer when prompted

### Step 4: After Restart
1. Docker Desktop starts automatically
2. Your volumes will be automatically discovered
3. Test: `docker volume ls` - should show `postgres_data` and `redis_data`
4. Start AlphaPulse: `docker-compose up -d`

---

## Your Data is SAFE!

**Don't worry!** Your 1 year of trading data is stored in Docker volumes:
- Location: `C:\Users\EA Soft Lab\AppData\Local\Docker\wsl\data\`
- This is **separate** from Docker Desktop installation
- **Will NOT be deleted** when uninstalling/reinstalling Docker
- Docker will automatically find the volumes after reinstall

---

## Alternative: Use Docker Without Docker Desktop

If Docker Desktop continues to have issues, you can use Docker Engine directly with WSL2:

### Install Docker in WSL2:
```bash
# In WSL2 terminal
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER
```

Then access your existing volumes through WSL2.

---

## Quick Commands After Fix

Once Docker is working:

```powershell
# Test Docker
docker ps

# List your volumes (your data!)
docker volume ls

# Start AlphaPulse
cd infrastructure/docker-compose
docker-compose up -d

# Create backup
cd ../..
.\backup_database.ps1
```

---

## Need Help?

If you encounter issues:
1. Check Docker Desktop logs at: `C:\Users\EA Soft Lab\AppData\Local\Docker\log\`
2. Check Windows Event Viewer > Application > Docker events
3. Ensure Hyper-V and WSL2 are still enabled (they should be)

---

**Next Step:** Download Docker Desktop installer and run it to repair the installation!

Download here: https://www.docker.com/products/docker-desktop/



