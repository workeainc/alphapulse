"""
Phase 3: SDE Calibration System Implementation
Database migration for advanced calibration system
"""

import asyncio
import logging
import asyncpg
import json
from datetime import datetime
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

# Database configuration
db_config = {
    'host': 'localhost',
    'port': 5432,
    'user': 'alpha_emon',
    'password': 'Emon_@17711',
    'database': 'alphapulse'
}

async def create_calibration_tables(pool: asyncpg.Pool):
    """Create calibration system tables"""
    
    tables = [
        # Calibration History
        """
        CREATE TABLE IF NOT EXISTS sde_calibration_history (
            id SERIAL PRIMARY KEY,
            model_name VARCHAR(50) NOT NULL,
            symbol VARCHAR(20) NOT NULL,
            timeframe VARCHAR(10) NOT NULL,
            raw_probability DECIMAL(5,4) NOT NULL,
            calibrated_probability DECIMAL(5,4) NOT NULL,
            calibration_method VARCHAR(20) NOT NULL,
            reliability_score DECIMAL(5,4) NOT NULL,
            calibration_timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        )
        """,
        
        # Calibration Metrics
        """
        CREATE TABLE IF NOT EXISTS sde_calibration_metrics (
            id SERIAL PRIMARY KEY,
            model_name VARCHAR(50) NOT NULL,
            symbol VARCHAR(20) NOT NULL,
            timeframe VARCHAR(10) NOT NULL,
            calibration_method VARCHAR(20) NOT NULL,
            brier_score DECIMAL(8,6) NOT NULL,
            reliability_score DECIMAL(5,4) NOT NULL,
            resolution_score DECIMAL(8,6) NOT NULL,
            uncertainty_score DECIMAL(8,6) NOT NULL,
            calibration_error DECIMAL(8,6) NOT NULL,
            sample_size INTEGER NOT NULL,
            calculation_timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        )
        """,
        
        # Calibration Configuration
        """
        CREATE TABLE IF NOT EXISTS sde_calibration_config (
            id SERIAL PRIMARY KEY,
            config_name VARCHAR(100) NOT NULL UNIQUE,
            config_type VARCHAR(50) NOT NULL,
            config_data JSONB NOT NULL,
            description TEXT,
            is_active BOOLEAN DEFAULT TRUE,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        )
        """,
        
        # Model Performance Tracking
        """
        CREATE TABLE IF NOT EXISTS sde_model_performance (
            id SERIAL PRIMARY KEY,
            model_name VARCHAR(50) NOT NULL,
            symbol VARCHAR(20) NOT NULL,
            timeframe VARCHAR(10) NOT NULL,
            period_start TIMESTAMP WITH TIME ZONE NOT NULL,
            period_end TIMESTAMP WITH TIME ZONE NOT NULL,
            total_signals INTEGER NOT NULL,
            winning_signals INTEGER NOT NULL,
            losing_signals INTEGER NOT NULL,
            win_rate DECIMAL(5,4) NOT NULL,
            avg_profit DECIMAL(10,6) NOT NULL,
            avg_loss DECIMAL(10,6) NOT NULL,
            profit_factor DECIMAL(8,4) NOT NULL,
            sharpe_ratio DECIMAL(8,4),
            max_drawdown DECIMAL(8,4),
            total_return DECIMAL(10,6) NOT NULL,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        )
        """,
        
        # Calibration Drift Detection
        """
        CREATE TABLE IF NOT EXISTS sde_calibration_drift (
            id SERIAL PRIMARY KEY,
            model_name VARCHAR(50) NOT NULL,
            symbol VARCHAR(20) NOT NULL,
            timeframe VARCHAR(10) NOT NULL,
            drift_type VARCHAR(30) NOT NULL, -- 'feature_drift', 'concept_drift', 'calibration_drift'
            drift_score DECIMAL(8,6) NOT NULL,
            drift_threshold DECIMAL(8,6) NOT NULL,
            drift_detected BOOLEAN NOT NULL,
            drift_details JSONB,
            detection_timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        )
        """
    ]
    
    for i, table_sql in enumerate(tables):
        try:
            async with pool.acquire() as conn:
                await conn.execute(table_sql)
                logger.info(f"✅ Created calibration table {i+1}")
        except Exception as e:
            logger.error(f"❌ Failed to create calibration table {i+1}: {e}")
            raise

async def create_calibration_indexes(pool: asyncpg.Pool):
    """Create performance indexes for calibration tables"""
    
    indexes = [
        # Calibration History indexes
        "CREATE INDEX IF NOT EXISTS idx_calibration_history_model_symbol_tf ON sde_calibration_history(model_name, symbol, timeframe)",
        "CREATE INDEX IF NOT EXISTS idx_calibration_history_timestamp ON sde_calibration_history(calibration_timestamp)",
        "CREATE INDEX IF NOT EXISTS idx_calibration_history_method ON sde_calibration_history(calibration_method)",
        
        # Calibration Metrics indexes
        "CREATE INDEX IF NOT EXISTS idx_calibration_metrics_model_symbol_tf ON sde_calibration_metrics(model_name, symbol, timeframe)",
        "CREATE INDEX IF NOT EXISTS idx_calibration_metrics_timestamp ON sde_calibration_metrics(calculation_timestamp)",
        "CREATE INDEX IF NOT EXISTS idx_calibration_metrics_method ON sde_calibration_metrics(calibration_method)",
        
        # Model Performance indexes
        "CREATE INDEX IF NOT EXISTS idx_model_performance_model_symbol_tf ON sde_model_performance(model_name, symbol, timeframe)",
        "CREATE INDEX IF NOT EXISTS idx_model_performance_period ON sde_model_performance(period_start, period_end)",
        "CREATE INDEX IF NOT EXISTS idx_model_performance_win_rate ON sde_model_performance(win_rate)",
        
        # Calibration Drift indexes
        "CREATE INDEX IF NOT EXISTS idx_calibration_drift_model_symbol_tf ON sde_calibration_drift(model_name, symbol, timeframe)",
        "CREATE INDEX IF NOT EXISTS idx_calibration_drift_type ON sde_calibration_drift(drift_type)",
        "CREATE INDEX IF NOT EXISTS idx_calibration_drift_detected ON sde_calibration_drift(drift_detected)",
        "CREATE INDEX IF NOT EXISTS idx_calibration_drift_timestamp ON sde_calibration_drift(detection_timestamp)"
    ]
    
    for i, index_sql in enumerate(indexes):
        try:
            async with pool.acquire() as conn:
                await conn.execute(index_sql)
                logger.info(f"✅ Created calibration index {i+1}")
        except Exception as e:
            logger.error(f"❌ Failed to create calibration index {i+1}: {e}")
            raise

async def insert_calibration_configs(pool: asyncpg.Pool):
    """Insert default calibration configurations"""
    
    configs = [
        {
            'config_name': 'sde_calibration_default',
            'config_type': 'calibration',
            'config_data': {
                'default_method': 'isotonic',
                'min_sample_size': {
                    'isotonic': 100,
                    'platt': 50,
                    'temperature': 30
                },
                'reliability_threshold': 0.8,
                'calibration_window_days': 30,
                'recalibration_frequency': 'daily',
                'confidence_level': 0.95
            },
            'description': 'Default calibration configuration'
        },
        {
            'config_name': 'sde_drift_detection_default',
            'config_type': 'drift_detection',
            'config_data': {
                'feature_drift_threshold': 0.1,
                'concept_drift_threshold': 0.15,
                'calibration_drift_threshold': 0.2,
                'detection_window_days': 7,
                'min_samples_for_drift': 50,
                'alert_on_drift': True
            },
            'description': 'Drift detection configuration'
        },
        {
            'config_name': 'sde_performance_tracking_default',
            'config_type': 'performance_tracking',
            'config_data': {
                'tracking_periods': ['daily', 'weekly', 'monthly'],
                'min_signals_for_performance': 10,
                'performance_metrics': ['win_rate', 'profit_factor', 'sharpe_ratio', 'max_drawdown'],
                'benchmark_comparison': True,
                'alert_thresholds': {
                    'win_rate_min': 0.5,
                    'profit_factor_min': 1.2,
                    'max_drawdown_max': 0.2
                }
            },
            'description': 'Performance tracking configuration'
        }
    ]
    
    for config in configs:
        try:
            async with pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO sde_calibration_config (config_name, config_type, config_data, description)
                    VALUES ($1, $2, $3, $4)
                    ON CONFLICT (config_name) DO UPDATE SET
                        config_data = EXCLUDED.config_data,
                        updated_at = NOW()
                """, config['config_name'], config['config_type'], json.dumps(config['config_data']), config['description'])
                logger.info(f"✅ Inserted/updated calibration config: {config['config_name']}")
        except Exception as e:
            logger.error(f"❌ Failed to insert calibration config {config['config_name']}: {e}")
            raise

async def run_migration():
    """Run the SDE calibration system migration"""
    logger.info("🚀 Starting SDE Calibration System Migration")
    
    try:
        # Create database connection
        pool = await asyncpg.create_pool(**db_config)
        logger.info("✅ Database connection established")
        
        # Create calibration tables
        await create_calibration_tables(pool)
        logger.info("✅ Calibration tables created")
        
        # Wait for tables to be fully created
        await asyncio.sleep(3)
        
        # Create calibration indexes
        await create_calibration_indexes(pool)
        logger.info("✅ Calibration indexes created")
        
        # Insert calibration configurations
        await insert_calibration_configs(pool)
        logger.info("✅ Calibration configurations inserted")
        
        # Close connection
        await pool.close()
        
        logger.info("🎉 SDE Calibration System Migration completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Migration failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(run_migration())
