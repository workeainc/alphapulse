# Extract PostgreSQL Database from Docker VHDX
# Run this as Administrator

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "  EXTRACT DATA FROM DOCKER VHDX" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if running as admin
$currentPrincipal = New-Object Security.Principal.WindowsPrincipal([Security.Principal.WindowsIdentity]::GetCurrent())
$isAdmin = $currentPrincipal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

if (-not $isAdmin) {
    Write-Host "❌ This script must be run as Administrator!" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please:" -ForegroundColor Yellow
    Write-Host "1. Right-click PowerShell" -ForegroundColor White
    Write-Host "2. Select 'Run as Administrator'" -ForegroundColor White
    Write-Host "3. Navigate to: D:\Emon Work\AlphaPuls\" -ForegroundColor White
    Write-Host "4. Run: .\EXTRACT_DATABASE_FROM_DOCKER.ps1" -ForegroundColor White
    Write-Host ""
    Read-Host "Press Enter to exit"
    exit 1
}

$vhdxPath = "C:\Users\EA Soft Lab\AppData\Local\Docker\wsl\disk\docker_data.vhdx"
$exportPath = "D:\AlphaPulse_Database_Backup\exported_data"

# Check VHDX exists
if (!(Test-Path $vhdxPath)) {
    Write-Host "❌ VHDX file not found: $vhdxPath" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

$sizeGB = [math]::Round((Get-Item $vhdxPath).Length / 1GB, 2)
Write-Host "Found Docker volume:" -ForegroundColor Green
Write-Host "  File: $vhdxPath" -ForegroundColor White
Write-Host "  Size: $sizeGB GB`n" -ForegroundColor Yellow

# Create export directory
if (!(Test-Path $exportPath)) {
    New-Item -ItemType Directory -Path $exportPath -Force | Out-Null
}

Write-Host "Step 1: Mounting VHDX..." -ForegroundColor Yellow
try {
    Mount-VHD -Path $vhdxPath -ReadOnly
    Write-Host "  ✅ Mounted successfully!" -ForegroundColor Green
    
    # Find the mounted drive
    Start-Sleep -Seconds 2
    $mountedDisk = Get-Disk | Where-Object {$_.FriendlyName -like "*Virtual*" -and $_.Size -gt 1GB} | Select-Object -First 1
    
    if ($mountedDisk) {
        $partition = Get-Partition -DiskNumber $mountedDisk.Number | Where-Object {$_.Type -eq 'Basic'}
        if ($partition.DriveLetter) {
            $driveLetter = $partition.DriveLetter
            Write-Host "  Drive Letter: $driveLetter`:" -ForegroundColor Cyan
        } else {
            Write-Host "  Assigning drive letter..." -ForegroundColor Yellow
            $partition | Set-Partition -NewDriveLetter Z
            $driveLetter = "Z"
        }
        
        Write-Host ""
        Write-Host "Step 2: Searching for PostgreSQL data..." -ForegroundColor Yellow
        
        # Common PostgreSQL paths in Docker volumes
        $pgPaths = @(
            "$driveLetter`:\var\lib\docker\volumes\postgres_data\_data",
            "$driveLetter`:\docker\volumes\postgres_data\_data",
            "$driveLetter`:\volumes\postgres_data\_data"
        )
        
        $found = $false
        foreach ($path in $pgPaths) {
            if (Test-Path $path) {
                Write-Host "  ✅ Found PostgreSQL data at: $path" -ForegroundColor Green
                Write-Host ""
                Write-Host "Step 3: Copying data..." -ForegroundColor Yellow
                Write-Host "  This may take several minutes..." -ForegroundColor Gray
                
                Copy-Item -Path $path -Destination "$exportPath\postgres_data" -Recurse -Force
                
                Write-Host "  ✅ Data copied to: $exportPath\postgres_data" -ForegroundColor Green
                $found = $true
                break
            }
        }
        
        if (!$found) {
            Write-Host "  Listing all directories to find data..." -ForegroundColor Cyan
            Get-ChildItem "$driveLetter`:\" -Recurse -Directory -ErrorAction SilentlyContinue | 
                Where-Object {$_.Name -like "*postgres*"} | 
                Select-Object -First 10 FullName
        }
        
    } else {
        Write-Host "  ❌ Could not find mounted disk" -ForegroundColor Red
    }
    
} catch {
    Write-Host "  ❌ Error: $_" -ForegroundColor Red
} finally {
    Write-Host ""
    Write-Host "Step 4: Unmounting VHDX..." -ForegroundColor Yellow
    try {
        Dismount-VHD -Path $vhdxPath
        Write-Host "  ✅ Unmounted" -ForegroundColor Green
    } catch {
        Write-Host "  ⚠️  Could not unmount: $_" -ForegroundColor Yellow
    }
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "  EXTRACTION COMPLETE!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Data exported to: $exportPath" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "1. Install PostgreSQL on Windows" -ForegroundColor White
Write-Host "2. Run the import script to load this data" -ForegroundColor White
Write-Host ""

Read-Host "Press Enter to close"


