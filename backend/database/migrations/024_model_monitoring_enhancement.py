#!/usr/bin/env python3
"""
Database Migration: Enhanced Model Monitoring and Drift Detection
Migration: 024_model_monitoring_enhancement.py

Adds tables for:
- Live performance tracking
- Drift detection logs
- Interpretability data
- Performance alerts
- Model health monitoring
"""

import asyncio
import logging
from datetime import datetime
from sqlalchemy import create_engine, text
import os

logger = logging.getLogger(__name__)

# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://alpha_emon:Emon_%4017711@localhost:5432/alphapulse")

async def run_migration():
    """Run the enhanced model monitoring migration"""
    engine = create_engine(DATABASE_URL)
    
    try:
        logger.info("🚀 Starting Enhanced Model Monitoring Migration (Partial)...")
        
        # Only create missing tables and add missing columns
        
        # 1. Create model health monitoring table (missing)
        logger.info("🏥 Creating model_health_monitoring table...")
        with engine.connect() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS model_health_monitoring (
                    id SERIAL,
                    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    model_id VARCHAR(100) NOT NULL,
                    symbol VARCHAR(20) NOT NULL,
                    health_score DECIMAL(5,2) NOT NULL, -- 0-100
                    status VARCHAR(20) NOT NULL, -- 'healthy', 'warning', 'critical', 'degraded'
                    last_training_time TIMESTAMPTZ,
                    last_prediction_time TIMESTAMPTZ,
                    total_predictions BIGINT DEFAULT 0,
                    successful_predictions BIGINT DEFAULT 0,
                    failed_predictions BIGINT DEFAULT 0,
                    avg_inference_time_ms DECIMAL(8,3),
                    memory_usage_mb DECIMAL(8,2),
                    cpu_usage_percent DECIMAL(5,2),
                    model_size_mb DECIMAL(8,2),
                    version VARCHAR(50),
                    deployment_environment VARCHAR(50),
                    health_checks JSONB,
                    metadata JSONB,
                    PRIMARY KEY (timestamp, id)
                )
            """))
            conn.commit()
        
        # Create hypertable
        try:
            with engine.connect() as conn:
                conn.execute(text("""
                    SELECT create_hypertable('model_health_monitoring', 'timestamp', 
                        if_not_exists => TRUE, migrate_data => TRUE)
                """))
                conn.commit()
                logger.info("✅ Created hypertable for model_health_monitoring")
        except Exception as e:
            logger.info(f"ℹ️ Hypertable for model_health_monitoring may already exist: {e}")
        
        # Create indexes
        with engine.connect() as conn:
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_health_monitoring_model_timestamp 
                ON model_health_monitoring (model_id, timestamp DESC)
            """))
            conn.commit()
            
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_health_monitoring_status 
                ON model_health_monitoring (status, timestamp DESC)
            """))
            conn.commit()
            
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_health_monitoring_health_score 
                ON model_health_monitoring (health_score DESC, timestamp DESC)
            """))
            conn.commit()
            
        # 2. Create backtest baseline table (missing)
        logger.info("📊 Creating backtest_baseline table...")
        with engine.connect() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS backtest_baseline (
                    id SERIAL PRIMARY KEY,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    model_id VARCHAR(100) NOT NULL UNIQUE,
                    symbol VARCHAR(20) NOT NULL,
                    baseline_name VARCHAR(100) NOT NULL,
                    precision DECIMAL(5,4) NOT NULL,
                    recall DECIMAL(5,4) NOT NULL,
                    f1_score DECIMAL(5,4) NOT NULL,
                    sharpe_ratio DECIMAL(8,4) NOT NULL,
                    profit_factor DECIMAL(8,4) NOT NULL,
                    win_rate DECIMAL(5,4) NOT NULL,
                    max_drawdown DECIMAL(8,4),
                    total_trades INTEGER NOT NULL,
                    backtest_start_date DATE NOT NULL,
                    backtest_end_date DATE NOT NULL,
                    backtest_duration_days INTEGER NOT NULL,
                    confidence_interval_lower DECIMAL(5,4),
                    confidence_interval_upper DECIMAL(5,4),
                    baseline_metadata JSONB,
                    is_active BOOLEAN DEFAULT TRUE
                )
            """))
            conn.commit()
            
            # Create indexes
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_backtest_baseline_model 
                ON backtest_baseline (model_id)
            """))
            conn.commit()
            
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_backtest_baseline_active 
                ON backtest_baseline (is_active, model_id)
            """))
            conn.commit()
            
        # 3. Create monitoring configuration table (missing)
        logger.info("⚙️ Creating monitoring_configuration table...")
        with engine.connect() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS monitoring_configuration (
                    id SERIAL PRIMARY KEY,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    config_name VARCHAR(100) NOT NULL UNIQUE,
                    config_type VARCHAR(50) NOT NULL, -- 'drift_detection', 'performance_monitoring', 'alerting'
                    model_id VARCHAR(100),
                    psi_threshold DECIMAL(5,4) DEFAULT 0.1,
                    kl_threshold DECIMAL(5,4) DEFAULT 0.1,
                    performance_degradation_threshold DECIMAL(5,2) DEFAULT 10.0,
                    alert_threshold DECIMAL(5,2) DEFAULT 5.0,
                    monitoring_window_hours INTEGER DEFAULT 24,
                    retraining_threshold DECIMAL(5,2) DEFAULT 15.0,
                    auto_retraining_enabled BOOLEAN DEFAULT TRUE,
                    alert_channels TEXT[], -- ['email', 'slack', 'webhook']
                    notification_recipients TEXT[],
                    is_active BOOLEAN DEFAULT TRUE,
                    configuration_json JSONB,
                    metadata JSONB
                )
            """))
            conn.commit()
            
            # Create indexes
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_monitoring_config_type 
                ON monitoring_configuration (config_type, is_active)
            """))
            conn.commit()
            
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_monitoring_config_model 
                ON monitoring_configuration (model_id, is_active)
            """))
            conn.commit()
            
            # Insert default monitoring configurations
            logger.info("📝 Inserting default monitoring configurations...")
            conn.execute(text("""
                INSERT INTO monitoring_configuration 
                (config_name, config_type, psi_threshold, kl_threshold, performance_degradation_threshold, 
                 alert_threshold, monitoring_window_hours, retraining_threshold, auto_retraining_enabled, 
                 alert_channels, is_active, configuration_json)
                VALUES 
                ('default_drift_detection', 'drift_detection', 0.1, 0.1, 10.0, 5.0, 24, 15.0, TRUE, 
                 ARRAY['email', 'slack'], TRUE, '{"enabled": true, "methods": ["psi", "kl_divergence"]}'),
                ('default_performance_monitoring', 'performance_monitoring', 0.1, 0.1, 10.0, 5.0, 24, 15.0, TRUE, 
                 ARRAY['email', 'slack'], TRUE, '{"enabled": true, "metrics": ["precision", "recall", "sharpe"]}'),
                ('default_alerting', 'alerting', 0.1, 0.1, 10.0, 5.0, 24, 15.0, TRUE, 
                 ARRAY['email', 'slack'], TRUE, '{"enabled": true, "channels": ["email", "slack"]}')
                ON CONFLICT (config_name) DO NOTHING
            """))
            conn.commit()
            
        # 4. Add monitoring columns to existing ml_predictions table
        logger.info("🔧 Adding monitoring columns to ml_predictions table...")
        try:
            with engine.connect() as conn:
                conn.execute(text("""
                    ALTER TABLE ml_predictions 
                    ADD COLUMN IF NOT EXISTS monitoring_metadata JSONB
                """))
                conn.commit()
                
                conn.execute(text("""
                    ALTER TABLE ml_predictions 
                    ADD COLUMN IF NOT EXISTS drift_score DECIMAL(8,6)
                """))
                conn.commit()
                
                conn.execute(text("""
                    ALTER TABLE ml_predictions 
                    ADD COLUMN IF NOT EXISTS performance_score DECIMAL(5,4)
                """))
                conn.commit()
                
                conn.execute(text("""
                    ALTER TABLE ml_predictions 
                    ADD COLUMN IF NOT EXISTS health_status VARCHAR(20)
                """))
                conn.commit()
                
        except Exception as e:
            logger.warning(f"⚠️ Some columns may already exist in ml_predictions: {e}")
            
        logger.info("✅ Enhanced Model Monitoring Migration completed successfully!")
        
        # Log table creation summary
        tables_created = [
            'model_health_monitoring',
            'backtest_baseline',
            'monitoring_configuration'
        ]
        
        logger.info(f"📋 Created {len(tables_created)} missing tables:")
        for table in tables_created:
            logger.info(f"   - {table}")
        
        logger.info("🔧 Enhanced existing ml_predictions table with monitoring columns")
        logger.info("⚙️ Inserted default monitoring configurations")
        
    except Exception as e:
        logger.error(f"❌ Migration failed: {e}")
        raise
    finally:
        engine.dispose()

if __name__ == "__main__":
    asyncio.run(run_migration())
