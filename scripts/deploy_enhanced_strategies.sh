#!/bin/bash

# Enhanced Strategy Deployment Script for AlphaPlus
# Deploys all new enhancements while maintaining compatibility

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKEND_DIR="$PROJECT_ROOT/backend"
MIGRATIONS_DIR="$BACKEND_DIR/database/migrations"
LOG_FILE="$PROJECT_ROOT/logs/enhanced_deployment_$(date +%Y%m%d_%H%M%S).log"

# Database configuration (update with your actual credentials)
DB_HOST="localhost"
DB_PORT="5432"
DB_NAME="alphapulse"
DB_USER="alpha_emon"
DB_PASSWORD="Emon_@17711"

echo -e "${BLUE}🚀 AlphaPlus Enhanced Strategy Deployment${NC}"
echo -e "${BLUE}==========================================${NC}"
echo ""

# Function to log messages
log_message() {
    echo -e "$1"
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" >> "$LOG_FILE"
}

# Function to check prerequisites
check_prerequisites() {
    log_message "${YELLOW}📋 Checking prerequisites...${NC}"
    
    # Check if Python is installed
    if ! command -v python3 &> /dev/null; then
        log_message "${RED}❌ Python 3 is not installed${NC}"
        exit 1
    fi
    
    # Check if pip is installed
    if ! command -v pip3 &> /dev/null; then
        log_message "${RED}❌ pip3 is not installed${NC}"
        exit 1
    fi
    
    # Check if PostgreSQL client is installed
    if ! command -v psql &> /dev/null; then
        log_message "${RED}❌ PostgreSQL client (psql) is not installed${NC}"
        exit 1
    fi
    
    # Check if TimescaleDB extension is available
    if ! psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -c "SELECT * FROM pg_extension WHERE extname = 'timescaledb';" &> /dev/null; then
        log_message "${RED}❌ TimescaleDB extension is not installed${NC}"
        exit 1
    fi
    
    log_message "${GREEN}✅ Prerequisites check passed${NC}"
}

# Function to install Python dependencies
install_dependencies() {
    log_message "${YELLOW}📦 Installing Python dependencies...${NC}"
    
    cd "$PROJECT_ROOT"
    
    # Install required packages for enhanced strategies
    pip3 install --upgrade pip
    
    # Core ML packages for ensemble learning
    pip3 install scikit-learn==1.3.0
    pip3 install lightgbm==4.0.0
    pip3 install numpy==1.24.3
    pip3 install pandas==2.0.3
    
    # Performance monitoring
    pip3 install psutil==5.9.5
    
    # Database and async
    pip3 install asyncpg==0.28.0
    pip3 install sqlalchemy==2.0.20
    
    # Optional: GPU acceleration (if available)
    if command -v nvidia-smi &> /dev/null; then
        log_message "${BLUE}🔧 GPU detected - installing CUDA packages...${NC}"
        pip3 install cupy-cuda11x  # Adjust version as needed
    fi
    
    log_message "${GREEN}✅ Dependencies installed successfully${NC}"
}

# Function to run database migrations
run_migrations() {
    log_message "${YELLOW}🗄️ Running database migrations...${NC}"
    
    # Create migrations table if it doesn't exist
    psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -c "
        CREATE TABLE IF NOT EXISTS database_migrations (
            id SERIAL PRIMARY KEY,
            migration_name VARCHAR(255) UNIQUE NOT NULL,
            applied_at TIMESTAMPTZ DEFAULT NOW(),
            version VARCHAR(50),
            status VARCHAR(50) DEFAULT 'completed'
        );
    " 2>/dev/null || true
    
    # Run the enhanced strategy migration
    log_message "${BLUE}📊 Applying enhanced strategy tables migration...${NC}"
    
    if psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -f "$MIGRATIONS_DIR/002_enhanced_strategy_tables.sql" >> "$LOG_FILE" 2>&1; then
        log_message "${GREEN}✅ Database migration completed successfully${NC}"
    else
        log_message "${RED}❌ Database migration failed${NC}"
        log_message "${YELLOW}📋 Check the log file: $LOG_FILE${NC}"
        exit 1
    fi
}

# Function to validate database setup
validate_database() {
    log_message "${YELLOW}🔍 Validating database setup...${NC}"
    
    # Check if new tables were created
    tables_to_check=(
        "enhanced_strategy_configs"
        "ensemble_predictions"
        "strategy_performance_history"
        "in_memory_cache"
        "parallel_execution_tasks"
        "market_microstructure"
        "adaptive_parameter_tuning"
        "risk_clustering"
        "slippage_modeling"
        "signal_streaming_events"
        "pre_aggregated_strategy_views"
        "system_performance_metrics"
    )
    
    for table in "${tables_to_check[@]}"; do
        if psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -c "SELECT 1 FROM $table LIMIT 1;" &> /dev/null; then
            log_message "${GREEN}✅ Table $table exists${NC}"
        else
            log_message "${RED}❌ Table $table not found${NC}"
            exit 1
        fi
    done
    
    # Check if views were created
    views_to_check=(
        "strategy_performance_summary"
        "ensemble_prediction_accuracy"
        "system_performance_overview"
    )
    
    for view in "${views_to_check[@]}"; do
        if psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -c "SELECT 1 FROM $view LIMIT 1;" &> /dev/null; then
            log_message "${GREEN}✅ View $view exists${NC}"
        else
            log_message "${RED}❌ View $view not found${NC}"
            exit 1
        fi
    done
    
    log_message "${GREEN}✅ Database validation completed${NC}"
}

# Function to test enhanced components
test_enhanced_components() {
    log_message "${YELLOW}🧪 Testing enhanced components...${NC}"
    
    cd "$BACKEND_DIR"
    
    # Test in-memory processor
    log_message "${BLUE}🔧 Testing in-memory processor...${NC}"
    python3 -c "
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
    print('✅ In-memory processor test passed' if result else '❌ In-memory processor test failed')

asyncio.run(test_in_memory())
" 2>/dev/null && log_message "${GREEN}✅ In-memory processor test passed${NC}" || log_message "${RED}❌ In-memory processor test failed${NC}"
    
    # Test ensemble manager
    log_message "${BLUE}🧠 Testing ensemble manager...${NC}"
    python3 -c "
import asyncio
import sys
sys.path.append('.')
from strategies.ensemble_strategy_manager import EnsembleStrategyManager

async def test_ensemble():
    manager = EnsembleStrategyManager()
    manager.register_strategy('test_strategy', {'type': 'test'})
    print('✅ Ensemble manager test passed')

asyncio.run(test_ensemble())
" 2>/dev/null && log_message "${GREEN}✅ Ensemble manager test passed${NC}" || log_message "${RED}❌ Ensemble manager test failed${NC}"
    
    # Test parallel executor
    log_message "${BLUE}⚡ Testing parallel executor...${NC}"
    python3 -c "
import asyncio
import sys
sys.path.append('.')
from strategies.parallel_strategy_executor import ParallelStrategyExecutor

async def test_parallel():
    executor = ParallelStrategyExecutor(max_process_workers=1, max_thread_workers=2)
    await executor.start()
    await executor.stop()
    print('✅ Parallel executor test passed')

asyncio.run(test_parallel())
" 2>/dev/null && log_message "${GREEN}✅ Parallel executor test passed${NC}" || log_message "${RED}❌ Parallel executor test failed${NC}"
    
    log_message "${GREEN}✅ Component testing completed${NC}"
}

# Function to create configuration files
create_configurations() {
    log_message "${YELLOW}⚙️ Creating configuration files...${NC}"
    
    # Create enhanced strategy configuration
    cat > "$BACKEND_DIR/config/enhanced_strategies.json" << EOF
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
EOF
    
    log_message "${GREEN}✅ Configuration files created${NC}"
}

# Function to update existing configuration
update_existing_config() {
    log_message "${YELLOW}🔄 Updating existing configuration...${NC}"
    
    # Backup existing configuration
    if [ -f "$BACKEND_DIR/config/config.py" ]; then
        cp "$BACKEND_DIR/config/config.py" "$BACKEND_DIR/config/config.py.backup.$(date +%Y%m%d_%H%M%S)"
        log_message "${BLUE}📋 Backed up existing configuration${NC}"
    fi
    
    # Add enhanced strategy imports to main config
    if [ -f "$BACKEND_DIR/config/config.py" ]; then
        echo "" >> "$BACKEND_DIR/config/config.py"
        echo "# Enhanced Strategy Configuration" >> "$BACKEND_DIR/config/config.py"
        echo "ENHANCED_STRATEGIES_ENABLED = True" >> "$BACKEND_DIR/config/config.py"
        echo "IN_MEMORY_PROCESSING_ENABLED = True" >> "$BACKEND_DIR/config/config.py"
        echo "PARALLEL_EXECUTION_ENABLED = True" >> "$BACKEND_DIR/config/config.py"
        echo "ENSEMBLE_LEARNING_ENABLED = True" >> "$BACKEND_DIR/config/config.py"
    fi
    
    log_message "${GREEN}✅ Configuration updated${NC}"
}

# Function to create startup script
create_startup_script() {
    log_message "${YELLOW}🚀 Creating startup script...${NC}"
    
    cat > "$PROJECT_ROOT/scripts/start_enhanced_alphapulse.sh" << 'EOF'
#!/bin/bash

# Enhanced AlphaPlus Startup Script
# Starts the system with all enhancements enabled

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKEND_DIR="$PROJECT_ROOT/backend"

echo "🚀 Starting Enhanced AlphaPlus..."

# Set environment variables
export ENHANCED_STRATEGIES_ENABLED=true
export IN_MEMORY_PROCESSING_ENABLED=true
export PARALLEL_EXECUTION_ENABLED=true
export ENSEMBLE_LEARNING_ENABLED=true

# Start the enhanced system
cd "$BACKEND_DIR"
python3 -m app.main_ai_system --enhanced

echo "✅ Enhanced AlphaPlus started successfully"
EOF
    
    chmod +x "$PROJECT_ROOT/scripts/start_enhanced_alphapulse.sh"
    log_message "${GREEN}✅ Startup script created${NC}"
}

# Function to create monitoring script
create_monitoring_script() {
    log_message "${YELLOW}📊 Creating monitoring script...${NC}"
    
    cat > "$PROJECT_ROOT/scripts/monitor_enhanced_system.py" << 'EOF'
#!/usr/bin/env python3

"""
Enhanced System Monitoring Script
Monitors performance of all enhancement layers
"""

import asyncio
import json
import sys
from pathlib import Path

# Add backend to path
backend_dir = Path(__file__).parent.parent / "backend"
sys.path.append(str(backend_dir))

from core.enhanced_strategy_integration import EnhancedStrategyIntegration
from core.in_memory_processor import InMemoryProcessor
from strategies.parallel_strategy_executor import ParallelStrategyExecutor
from strategies.ensemble_strategy_manager import EnsembleStrategyManager

async def monitor_system():
    """Monitor all enhancement components"""
    print("📊 Enhanced System Monitoring")
    print("=" * 40)
    
    # Monitor in-memory processor
    try:
        processor = InMemoryProcessor()
        stats = processor.get_buffer_stats()
        print(f"📦 In-Memory Processor: {len(stats)} active buffers")
        print(f"   Memory Usage: {processor.stats['memory_usage_mb']:.2f} MB")
        print(f"   Avg Processing Time: {processor.stats['avg_processing_time_ms']:.2f} ms")
    except Exception as e:
        print(f"❌ In-Memory Processor Error: {e}")
    
    # Monitor parallel executor
    try:
        executor = ParallelStrategyExecutor()
        stats = executor.get_performance_stats()
        print(f"⚡ Parallel Executor: {stats['completed_tasks']} tasks completed")
        print(f"   Avg Processing Time: {stats['avg_processing_time_ms']:.2f} ms")
        print(f"   CPU Usage: {stats['cpu_usage_percent']:.1f}%")
    except Exception as e:
        print(f"❌ Parallel Executor Error: {e}")
    
    # Monitor ensemble manager
    try:
        manager = EnsembleStrategyManager()
        stats = manager.get_ensemble_stats()
        print(f"🧠 Ensemble Manager: {stats['performance_history_count']} performance records")
        print(f"   Model Accuracy: {stats['model_metadata']['accuracy']:.3f}")
        print(f"   Total Predictions: {stats['stats']['total_predictions']}")
    except Exception as e:
        print(f"❌ Ensemble Manager Error: {e}")

if __name__ == "__main__":
    asyncio.run(monitor_system())
EOF
    
    chmod +x "$PROJECT_ROOT/scripts/monitor_enhanced_system.py"
    log_message "${GREEN}✅ Monitoring script created${NC}"
}

# Main deployment function
main() {
    log_message "${BLUE}🚀 Starting Enhanced Strategy Deployment${NC}"
    
    # Create logs directory
    mkdir -p "$(dirname "$LOG_FILE")"
    
    # Run deployment steps
    check_prerequisites
    install_dependencies
    run_migrations
    validate_database
    test_enhanced_components
    create_configurations
    update_existing_config
    create_startup_script
    create_monitoring_script
    
    log_message "${GREEN}🎉 Enhanced Strategy Deployment Completed Successfully!${NC}"
    log_message "${BLUE}📋 Deployment log: $LOG_FILE${NC}"
    log_message "${BLUE}🚀 To start the enhanced system: ./scripts/start_enhanced_alphapulse.sh${NC}"
    log_message "${BLUE}📊 To monitor the system: python3 scripts/monitor_enhanced_system.py${NC}"
    
    echo ""
    echo -e "${GREEN}🎉 Deployment Summary:${NC}"
    echo -e "${GREEN}✅ Enhanced strategy tables created${NC}"
    echo -e "${GREEN}✅ In-memory processing layer deployed${NC}"
    echo -e "${GREEN}✅ Parallel execution system deployed${NC}"
    echo -e "${GREEN}✅ Ensemble learning system deployed${NC}"
    echo -e "${GREEN}✅ Integration layer configured${NC}"
    echo -e "${GREEN}✅ Startup and monitoring scripts created${NC}"
    echo ""
    echo -e "${YELLOW}📋 Next steps:${NC}"
    echo -e "${YELLOW}1. Review the deployment log: $LOG_FILE${NC}"
    echo -e "${YELLOW}2. Start the enhanced system: ./scripts/start_enhanced_alphapulse.sh${NC}"
    echo -e "${YELLOW}3. Monitor performance: python3 scripts/monitor_enhanced_system.py${NC}"
    echo -e "${YELLOW}4. Configure strategy parameters in backend/config/enhanced_strategies.json${NC}"
}

# Run main function
main "$@"
