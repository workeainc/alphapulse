# Install PostgreSQL on Windows (Alternative to Docker)
# This allows you to run PostgreSQL directly on Windows

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "  INSTALL POSTGRESQL ON WINDOWS" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "This will:" -ForegroundColor Yellow
Write-Host "  1. Download PostgreSQL for Windows" -ForegroundColor White
Write-Host "  2. Install it (no Docker needed!)" -ForegroundColor White
Write-Host "  3. Allow you to restore your database backup" -ForegroundColor White
Write-Host ""

# PostgreSQL download URL (version 15 with TimescaleDB support)
$pgUrl = "https://get.enterprisedb.com/postgresql/postgresql-15.10-1-windows-x64.exe"
$downloadPath = "$env:TEMP\postgresql_installer.exe"

Write-Host "Downloading PostgreSQL..." -ForegroundColor Cyan
Write-Host "(This is about 250MB, may take a few minutes)" -ForegroundColor Gray
Write-Host ""

try {
    $webClient = New-Object System.Net.WebClient
    
    # Progress
    Register-ObjectEvent -InputObject $webClient -EventName DownloadProgressChanged -SourceIdentifier WebClient.DownloadProgressChanged -Action {
        Write-Progress -Activity "Downloading PostgreSQL" -Status "$($EventArgs.ProgressPercentage)% Complete" -PercentComplete $EventArgs.ProgressPercentage
    } | Out-Null
    
    $webClient.DownloadFile($pgUrl, $downloadPath)
    
    Unregister-Event -SourceIdentifier WebClient.DownloadProgressChanged
    $webClient.Dispose()
    
    Write-Host "✅ Download complete!" -ForegroundColor Green
    Write-Host ""
    
    Write-Host "Now running installer..." -ForegroundColor Yellow
    Write-Host ""
    Write-Host "IMPORTANT INSTALLATION NOTES:" -ForegroundColor Red
    Write-Host "  1. Remember the password you set for 'postgres' user" -ForegroundColor White
    Write-Host "  2. Use port 5432 (default)" -ForegroundColor White
    Write-Host "  3. Install all components" -ForegroundColor White
    Write-Host ""
    
    $confirm = Read-Host "Ready to run installer? (Y/N)"
    
    if ($confirm -eq "Y" -or $confirm -eq "y") {
        Start-Process -FilePath $downloadPath -Wait
        
        Write-Host ""
        Write-Host "========================================" -ForegroundColor Green
        Write-Host "  INSTALLATION COMPLETE!" -ForegroundColor Green
        Write-Host "========================================" -ForegroundColor Green
        Write-Host ""
        Write-Host "PostgreSQL is now installed on Windows!" -ForegroundColor Cyan
        Write-Host ""
        Write-Host "Next steps:" -ForegroundColor Yellow
        Write-Host "  1. Your database is running on port 5432" -ForegroundColor White
        Write-Host "  2. Username: postgres" -ForegroundColor White
        Write-Host "  3. Password: (what you set during install)" -ForegroundColor White
        Write-Host ""
        Write-Host "You can now restore your AlphaPulse database!" -ForegroundColor Green
    } else {
        Write-Host "Installation cancelled." -ForegroundColor Yellow
        Write-Host "Installer saved at: $downloadPath" -ForegroundColor Cyan
    }
    
} catch {
    Write-Host "❌ Error: $_" -ForegroundColor Red
    Write-Host ""
    Write-Host "Manual download:" -ForegroundColor Yellow
    Write-Host "Go to: https://www.postgresql.org/download/windows/" -ForegroundColor Cyan
}

Write-Host ""
Read-Host "Press Enter to close"


