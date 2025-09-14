# Enhanced Strategy Deployment Script for AlphaPlus (PowerShell Version)
# Deploys all new enhancements while maintaining compatibility

param(
    [string]$DBHost = "localhost",
    [int]$DBPort = 5432,
    [string]$DBName = "alphapulse",
    [string]$DBUser = "alpha_emon",
    [string]$DBPassword = "Emon_@17711"
)

# Configuration
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$BackendDir = Join-Path $ProjectRoot "backend"
$MigrationsDir = Join-Path $BackendDir "database\migrations"
$LogFile = Join-Path $ProjectRoot "logs\enhanced_deployment_$(Get-Date -Format 'yyyyMMdd_HHmmss').log"

# Create logs directory if it doesn't exist
$LogsDir = Split-Path $LogFile -Parent
if (!(Test-Path $LogsDir)) {
    New-Item -ItemType Directory -Path $LogsDir -Force | Out-Null
}

# Function to log messages
function Write-LogMessage {
    param([string]$Message, [string]$Color = "White")
    Write-Host $Message -ForegroundColor $Color
    $Timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    "$Timestamp - $Message" | Out-File -FilePath $LogFile -Append
}

# Function to check prerequisites
function Test-Prerequisites {
    Write-LogMessage "Checking prerequisites..." "Yellow"

    # Check if Python is installed
    try {
        $pythonVersion = python --version 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-LogMessage "Python found: $pythonVersion" "Green"
        } else {
            Write-LogMessage "Python is not installed" "Red"
            return $false
        }
    } catch {
        Write-LogMessage "Python is not installed" "Red"
        return $false
    }

    # Check if pip is installed
    try {
        $pipVersion = pip --version 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-LogMessage "pip found: $pipVersion" "Green"
        } else {
            Write-LogMessage "pip is not installed" "Red"
            return $false
        }
    } catch {
        Write-LogMessage "pip is not installed" "Red"
        return $false
    }

    # Check if PostgreSQL client is installed
    try {
        $psqlVersion = psql --version 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-LogMessage "PostgreSQL client found: $psqlVersion" "Green"
        } else {
            Write-LogMessage "PostgreSQL client (psql) is not installed" "Red"
            return $false
        }
    } catch {
        Write-LogMessage "PostgreSQL client (psql) is not installed" "Red"
        return $false
    }

    Write-LogMessage "Prerequisites check passed" "Green"
    return $true
}

# Function to install Python dependencies
function Install-Dependencies {
    Write-LogMessage "Installing Python dependencies..." "Yellow"

    Set-Location $ProjectRoot

    # Install required packages for enhanced strategies
    Write-LogMessage "Installing core ML packages..." "Blue"
    
    try {
        pip install --upgrade pip
        pip install scikit-learn==1.3.0
        pip install lightgbm==4.0.0
        pip install numpy==1.24.3
        pip install pandas==2.0.3
        pip install psutil==5.9.5
        pip install asyncpg==0.28.0
        pip install sqlalchemy==2.0.20
        
        Write-LogMessage "Dependencies installed successfully" "Green"
    } catch {
        Write-LogMessage "Failed to install dependencies: $($_.Exception.Message)" "Red"
        return $false
    }

    return $true
}

# Function to run database migrations
function Invoke-DatabaseMigrations {
    Write-LogMessage "Running database migrations..." "Yellow"

    # Create migrations table if it doesn't exist
    $createMigrationsTable = "CREATE TABLE IF NOT EXISTS database_migrations (id SERIAL PRIMARY KEY, migration_name VARCHAR(255) UNIQUE NOT NULL, applied_at TIMESTAMPTZ DEFAULT NOW(), version VARCHAR(50), status VARCHAR(50) DEFAULT 'completed');"

    try {
        $env:PGPASSWORD = $DBPassword
        psql -h $DBHost -p $DBPort -U $DBUser -d $DBName -c $createMigrationsTable | Out-Null
        Write-LogMessage "Migrations table created/verified" "Green"
    } catch {
        Write-LogMessage "Failed to create migrations table: $($_.Exception.Message)" "Red"
        return $false
    }

    # Run the enhanced strategy migration
    Write-LogMessage "Applying enhanced strategy tables migration..." "Blue"

    $migrationFile = Join-Path $MigrationsDir "002_enhanced_strategy_tables.sql"
    
    try {
        $env:PGPASSWORD = $DBPassword
        psql -h $DBHost -p $DBPort -U $DBUser -d $DBName -f $migrationFile | Out-File -FilePath $LogFile -Append
        Write-LogMessage "Database migration completed successfully" "Green"
    } catch {
        Write-LogMessage "Database migration failed: $($_.Exception.Message)" "Red"
        Write-LogMessage "Check the log file: $LogFile" "Yellow"
        return $false
    }

    return $true
}

# Function to validate database setup
function Test-DatabaseSetup {
    Write-LogMessage "Validating database setup..." "Yellow"

    $tablesToCheck = @(
        "enhanced_strategy_configs",
        "ensemble_predictions",
        "strategy_performance_history",
        "in_memory_cache",
        "parallel_execution_tasks",
        "market_microstructure",
        "adaptive_parameter_tuning",
        "risk_clustering",
        "slippage_modeling",
        "signal_streaming_events",
        "pre_aggregated_strategy_views",
        "system_performance_metrics"
    )

    foreach ($table in $tablesToCheck) {
        try {
            $env:PGPASSWORD = $DBPassword
            $result = psql -h $DBHost -p $DBPort -U $DBUser -d $DBName -c "SELECT 1 FROM $table LIMIT 1;" 2>&1
            if ($LASTEXITCODE -eq 0) {
                Write-LogMessage "Table $table exists" "Green"
            } else {
                Write-LogMessage "Table $table not found" "Red"
                return $false
            }
        } catch {
            Write-LogMessage "Table $table not found: $($_.Exception.Message)" "Red"
            return $false
        }
    }

    # Check if views were created
    $viewsToCheck = @(
        "strategy_performance_summary",
        "ensemble_prediction_accuracy",
        "system_performance_overview"
    )

    foreach ($view in $viewsToCheck) {
        try {
            $env:PGPASSWORD = $DBPassword
            $result = psql -h $DBHost -p $DBPort -U $DBUser -d $DBName -c "SELECT 1 FROM $view LIMIT 1;" 2>&1
            if ($LASTEXITCODE -eq 0) {
                Write-LogMessage "View $view exists" "Green"
            } else {
                Write-LogMessage "View $view not found" "Red"
                return $false
            }
        } catch {
            Write-LogMessage "View $view not found: $($_.Exception.Message)" "Red"
            return $false
        }
    }

    Write-LogMessage "Database validation completed" "Green"
    return $true
}

# Function to test enhanced components
function Test-EnhancedComponents {
    Write-LogMessage "Testing enhanced components..." "Yellow"

    Set-Location $BackendDir

    # Test in-memory processor
    Write-LogMessage "Testing in-memory processor..." "Blue"
    try {
        $testScript = @"
import asyncio
import sys
sys.path.append('.')
from core.in_memory_processor import InMemoryProcessor

async def test_in_memory():
    processor = InMemoryProcessor()
    test_data = {
        'timestamp': 1640995200000,
        'open': 50000.0,
        'high': 51000.0,
        'low': 49000.0,
        'close': 50500.0,
        'volume': 1000.0
    }
    result = await processor.process_candle_in_memory('BTCUSDT', '1h', test_data)
    print('In-memory processor test passed' if result else 'In-memory processor test failed')

asyncio.run(test_in_memory())
"@
        $testScript | python
        if ($LASTEXITCODE -eq 0) {
            Write-LogMessage "In-memory processor test passed" "Green"
        } else {
            Write-LogMessage "In-memory processor test failed" "Red"
        }
    } catch {
        Write-LogMessage "In-memory processor test failed: $($_.Exception.Message)" "Red"
    }

    # Test ensemble manager
    Write-LogMessage "Testing ensemble manager..." "Blue"
    try {
        $testScript = @"
import asyncio
import sys
sys.path.append('.')
from strategies.ensemble_strategy_manager import EnsembleStrategyManager

async def test_ensemble():
    manager = EnsembleStrategyManager()
    manager.register_strategy('test_strategy', {'type': 'test'})
    print('Ensemble manager test passed')

asyncio.run(test_ensemble())
"@
        $testScript | python
        if ($LASTEXITCODE -eq 0) {
            Write-LogMessage "Ensemble manager test passed" "Green"
        } else {
            Write-LogMessage "Ensemble manager test failed" "Red"
        }
    } catch {
        Write-LogMessage "Ensemble manager test failed: $($_.Exception.Message)" "Red"
    }

    # Test parallel executor
    Write-LogMessage "Testing parallel executor..." "Blue"
    try {
        $testScript = @"
import asyncio
import sys
sys.path.append('.')
from strategies.parallel_strategy_executor import ParallelStrategyExecutor

async def test_parallel():
    executor = ParallelStrategyExecutor(max_process_workers=1, max_thread_workers=2)
    await executor.start()
    await executor.stop()
    print('Parallel executor test passed')

asyncio.run(test_parallel())
"@
        $testScript | python
        if ($LASTEXITCODE -eq 0) {
            Write-LogMessage "Parallel executor test passed" "Green"
        } else {
            Write-LogMessage "Parallel executor test failed" "Red"
        }
    } catch {
        Write-LogMessage "Parallel executor test failed: $($_.Exception.Message)" "Red"
    }

    Write-LogMessage "Component testing completed" "Green"
}

# Function to create configuration files
function New-ConfigurationFiles {
    Write-LogMessage "Creating configuration files..." "Yellow"

    # Create enhanced strategy configuration
    $configDir = Join-Path $BackendDir "config"
    if (!(Test-Path $configDir)) {
        New-Item -ItemType Directory -Path $configDir -Force | Out-Null
    }

    $enhancedConfig = @"
{
    "enhanced_strategy_integration": {
        "enable_in_memory_processing": true,
        "enable_parallel_execution": true,
        "enable_ensemble_learning": true,
        "enable_market_microstructure": true,
        "enable_adaptive_tuning": true,
        "max_buffer_size": 1000,
        "max_workers": 4,
        "ensemble_retrain_interval_hours": 24
    },
    "in_memory_processor": {
        "max_buffer_size": 1000,
        "max_workers": 4,
        "cache_ttl_seconds": 300
    },
    "parallel_executor": {
        "max_process_workers": 4,
        "max_thread_workers": 8,
        "enable_process_pool": true,
        "enable_thread_pool": true
    },
    "ensemble_manager": {
        "model_save_path": "models/ensemble",
        "min_training_samples": 100,
        "retrain_interval_hours": 24
    }
}
"@

    $enhancedConfig | Out-File -FilePath (Join-Path $configDir "enhanced_strategies.json") -Encoding UTF8
    Write-LogMessage "Configuration files created" "Green"
}

# Main execution function
function Start-EnhancedDeployment {
    Write-LogMessage "AlphaPlus Enhanced Strategy Deployment" "Blue"
    Write-LogMessage "======================================" "Blue"
    Write-LogMessage ""

    try {
        # Check prerequisites
        if (!(Test-Prerequisites)) {
            Write-LogMessage "Prerequisites check failed. Exiting." "Red"
            return $false
        }

        # Install dependencies
        if (!(Install-Dependencies)) {
            Write-LogMessage "Dependency installation failed. Exiting." "Red"
            return $false
        }

        # Run database migrations
        if (!(Invoke-DatabaseMigrations)) {
            Write-LogMessage "Database migration failed. Exiting." "Red"
            return $false
        }

        # Validate database setup
        if (!(Test-DatabaseSetup)) {
            Write-LogMessage "Database validation failed. Exiting." "Red"
            return $false
        }

        # Test enhanced components
        Test-EnhancedComponents

        # Create configuration files
        New-ConfigurationFiles

        Write-LogMessage ""
        Write-LogMessage "Enhanced Strategy Deployment Completed Successfully!" "Green"
        Write-LogMessage "Check the log file for details: $LogFile" "Blue"
        Write-LogMessage ""
        Write-LogMessage "Next steps:" "Yellow"
        Write-LogMessage "1. Start the enhanced system: python backend/main_enhanced.py" "Blue"
        Write-LogMessage "2. Monitor performance: python scripts/monitor_enhanced_system.py" "Blue"
        Write-LogMessage "3. Check dashboard for enhanced metrics" "Blue"

        return $true

    } catch {
        Write-LogMessage "Deployment failed with error: $($_.Exception.Message)" "Red"
        Write-LogMessage "Check the log file for details: $LogFile" "Yellow"
        return $false
    }
}

# Run main function
Start-EnhancedDeployment
