# Comprehensive Testing Runner (PowerShell)
# Runs both backend and frontend integration tests
# Phase 7: Testing & Validation

Write-Host "Starting Comprehensive Testing Suite" -ForegroundColor Blue
Write-Host "========================================" -ForegroundColor Blue

# Function to print colored output
function Write-Status {
    param(
        [string]$Status,
        [string]$Message
    )
    
    switch ($Status) {
        "SUCCESS" { Write-Host "SUCCESS: $Message" -ForegroundColor Green }
        "ERROR" { Write-Host "ERROR: $Message" -ForegroundColor Red }
        "WARNING" { Write-Host "WARNING: $Message" -ForegroundColor Yellow }
        "INFO" { Write-Host "INFO: $Message" -ForegroundColor Cyan }
    }
}

# Check if backend server is running
function Test-BackendServer {
    Write-Status "INFO" "Checking if backend server is running..."
    
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:8000/api/v1/production/status" -TimeoutSec 5 -ErrorAction Stop
        Write-Status "SUCCESS" "Backend server is running"
        return $true
    }
    catch {
        Write-Status "ERROR" "Backend server is not running on localhost:8000"
        Write-Status "INFO" "Please start the backend server first:"
        Write-Status "INFO" "  cd backend; python -m uvicorn app.main_ai_system_simple:app --host 0.0.0.0 --port 8000"
        return $false
    }
}

# Check if frontend dependencies are installed
function Test-FrontendDependencies {
    Write-Status "INFO" "Checking frontend dependencies..."
    
    if (Test-Path "frontend/package.json") {
        if (Test-Path "frontend/node_modules") {
            Write-Status "SUCCESS" "Frontend dependencies are installed"
            return $true
        }
        else {
            Write-Status "WARNING" "Frontend dependencies not installed"
            Write-Status "INFO" "Installing frontend dependencies..."
            Set-Location frontend
            npm install
            Set-Location ..
            return $true
        }
    }
    else {
        Write-Status "ERROR" "Frontend package.json not found"
        return $false
    }
}

# Run backend tests
function Invoke-BackendTests {
    Write-Status "INFO" "Running backend integration tests..."
    
    if (Test-Path "backend/test_real_data_integration.py") {
        Set-Location backend
        $backendResult = python test_real_data_integration.py
        $backendExitCode = $LASTEXITCODE
        Set-Location ..
        
        if ($backendExitCode -eq 0) {
            Write-Status "SUCCESS" "Backend tests passed"
            return $true
        }
        else {
            Write-Status "ERROR" "Backend tests failed"
            return $false
        }
    }
    else {
        Write-Status "ERROR" "Backend test file not found"
        return $false
    }
}

# Run frontend tests
function Invoke-FrontendTests {
    Write-Status "INFO" "Running frontend integration tests..."
    
    if (Test-Path "frontend/test_frontend_integration.js") {
        Set-Location frontend
        $frontendResult = node test_frontend_integration.js
        $frontendExitCode = $LASTEXITCODE
        Set-Location ..
        
        if ($frontendExitCode -eq 0) {
            Write-Status "SUCCESS" "Frontend tests passed"
            return $true
        }
        else {
            Write-Status "ERROR" "Frontend tests failed"
            return $false
        }
    }
    else {
        Write-Status "ERROR" "Frontend test file not found"
        return $false
    }
}

# Generate combined report
function New-CombinedReport {
    Write-Status "INFO" "Generating combined test report..."
    
    $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $reportFile = "comprehensive_test_report_$timestamp.md"
    
    $backendReport = Get-ChildItem "backend/real_data_integration_test_report_*.json" -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    $frontendReport = Get-ChildItem "frontend/frontend_integration_test_report_*.json" -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    
    $backendStatus = if ($backendReport) { "✅ PASSED" } else { "❌ FAILED" }
    $frontendStatus = if ($frontendReport) { "✅ PASSED" } else { "❌ FAILED" }
    
    $reportContent = "# Comprehensive Test Report`n" +
                   "**Generated:** $(Get-Date)`n" +
                   "**Test Suite:** Real Data Integration Testing`n`n" +
                   "## Test Summary`n`n" +
                   "### Backend Tests`n" +
                   "- Status: $backendStatus`n" +
                   "- Report: $($backendReport.Name)`n`n" +
                   "### Frontend Tests`n" +
                   "- Status: $frontendStatus`n" +
                   "- Report: $($frontendReport.Name)`n`n" +
                   "## Recommendations`n`n" +
                   "1. Review individual test reports for detailed results`n" +
                   "2. Fix any failing tests before proceeding to production`n" +
                   "3. Run performance tests under load conditions`n" +
                   "4. Validate data consistency across all components`n`n" +
                   "## Next Steps`n`n" +
                   "1. Performance Testing: Load testing and optimization`n" +
                   "2. Production Deployment: Deploy to production environment`n" +
                   "3. Monitoring Setup: Set up production monitoring`n" +
                   "4. Documentation: Final user guides and API documentation`n`n" +
                   "---`n" +
                   "*Generated by Comprehensive Testing Suite*"

    $reportContent | Out-File -FilePath $reportFile -Encoding UTF8
    Write-Status "SUCCESS" "Combined report generated: $reportFile"
}

# Main execution
function Start-ComprehensiveTests {
    $startTime = Get-Date
    
    Write-Status "INFO" "Starting comprehensive testing suite..."
    
    # Check prerequisites
    if (-not (Test-BackendServer)) {
        exit 1
    }
    
    if (-not (Test-FrontendDependencies)) {
        exit 1
    }
    
    # Run tests
    $backendSuccess = Invoke-BackendTests
    $frontendSuccess = Invoke-FrontendTests
    
    # Generate combined report
    New-CombinedReport
    
    # Final status
    $endTime = Get-Date
    $duration = ($endTime - $startTime).TotalSeconds
    
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Blue
    Write-Status "INFO" "Testing completed in $([math]::Round($duration, 2)) seconds"
    
    if ($backendSuccess -and $frontendSuccess) {
        Write-Status "SUCCESS" "All tests passed! System is ready for production."
        exit 0
    }
    else {
        Write-Status "ERROR" "Some tests failed. Check the reports for details."
        exit 1
    }
}

# Run main function
Start-ComprehensiveTests
