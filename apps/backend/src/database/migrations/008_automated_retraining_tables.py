#!/usr/bin/env python3
"""
Migration 008: Automated Retraining Tables
Add tables for automated retraining logs, performance tracking, and drift detection
"""

import asyncio
import asyncpg
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

async def run_migration():
    """Run the automated retraining tables migration"""
    
    # Database configuration
    DB_CONFIG = {
        'host': 'localhost',
        'port': 5432,
        'database': 'alphapulse',
        'user': 'alpha_emon',
        'password': 'Emon_@17711'
    }
    
    try:
        logger.info("🔄 Starting Migration 008: Automated Retraining Tables")
        
        # Connect to database
        conn = await asyncpg.connect(**DB_CONFIG)
        
        # Create retraining logs table
        logger.info("📊 Creating retraining_logs table...")
        await conn.execute("""
        CREATE TABLE IF NOT EXISTS retraining_logs (
            timestamp TIMESTAMPTZ NOT NULL,
            id SERIAL,
            event_type VARCHAR(50) NOT NULL, -- 'scheduled', 'performance_triggered', 'drift_triggered', 'manual'
            model_type VARCHAR(50) NOT NULL,
            trigger_reason TEXT,
            training_samples INTEGER,
            validation_samples INTEGER,
            old_performance NUMERIC(6,4),
            new_performance NUMERIC(6,4),
            improvement NUMERIC(6,4),
            training_duration_seconds NUMERIC(10,2),
            status VARCHAR(20) NOT NULL, -- 'success', 'failed', 'in_progress'
            error_message TEXT,
            model_version VARCHAR(50),
            feature_drift_score NUMERIC(6,4),
            data_quality_score NUMERIC(6,4),
            kubernetes_pod_name VARCHAR(100),
            resource_usage JSONB,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            PRIMARY KEY (timestamp, id)
        );
        """)
        
        # Create TimescaleDB hypertable for retraining_logs
        logger.info("📊 Creating hypertable for retraining_logs...")
        await conn.execute("""
        SELECT create_hypertable('retraining_logs', 'timestamp', if_not_exists => TRUE);
        """)
        
        # Create performance tracking table
        logger.info("📊 Creating model_performance_history table...")
        await conn.execute("""
        CREATE TABLE IF NOT EXISTS model_performance_history (
            timestamp TIMESTAMPTZ NOT NULL,
            id SERIAL,
            model_type VARCHAR(50) NOT NULL,
            model_version VARCHAR(50) NOT NULL,
            prediction_target VARCHAR(50) NOT NULL, -- 'regime_change', 'sector_rotation', 'price_direction'
            accuracy_score NUMERIC(6,4),
            precision_score NUMERIC(6,4),
            recall_score NUMERIC(6,4),
            f1_score NUMERIC(6,4),
            auc_roc_score NUMERIC(6,4),
            mse_score NUMERIC(10,6),
            mae_score NUMERIC(10,6),
            training_samples INTEGER,
            validation_samples INTEGER,
            test_samples INTEGER,
            feature_count INTEGER,
            training_duration_seconds NUMERIC(10,2),
            inference_latency_ms NUMERIC(10,2),
            memory_usage_mb NUMERIC(10,2),
            cpu_usage_percent NUMERIC(5,2),
            drift_score NUMERIC(6,4),
            confidence_score NUMERIC(6,4),
            is_active BOOLEAN DEFAULT FALSE,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            PRIMARY KEY (timestamp, id)
        );
        """)
        
        # Create TimescaleDB hypertable for model_performance_history
        logger.info("📊 Creating hypertable for model_performance_history...")
        await conn.execute("""
        SELECT create_hypertable('model_performance_history', 'timestamp', if_not_exists => TRUE);
        """)
        
        # Create drift detection table
        logger.info("📊 Creating feature_drift_metrics table...")
        await conn.execute("""
        CREATE TABLE IF NOT EXISTS feature_drift_metrics (
            timestamp TIMESTAMPTZ NOT NULL,
            id SERIAL,
            feature_name VARCHAR(100) NOT NULL,
            feature_type VARCHAR(50) NOT NULL, -- 'ohlcv', 'sentiment', 'flow', 'correlation'
            ks_statistic NUMERIC(10,6),
            ks_p_value NUMERIC(10,6),
            drift_severity VARCHAR(20), -- 'low', 'medium', 'high'
            drift_detected BOOLEAN DEFAULT FALSE,
            reference_mean NUMERIC(20,8),
            reference_std NUMERIC(20,8),
            current_mean NUMERIC(20,8),
            current_std NUMERIC(20,8),
            mean_drift_percent NUMERIC(8,4),
            std_drift_percent NUMERIC(8,4),
            sample_size_reference INTEGER,
            sample_size_current INTEGER,
            detection_method VARCHAR(50) DEFAULT 'ks_test',
            model_affected VARCHAR(100),
            created_at TIMESTAMPTZ DEFAULT NOW(),
            PRIMARY KEY (timestamp, id)
        );
        """)
        
        # Create TimescaleDB hypertable for feature_drift_metrics
        logger.info("📊 Creating hypertable for feature_drift_metrics...")
        await conn.execute("""
        SELECT create_hypertable('feature_drift_metrics', 'timestamp', if_not_exists => TRUE);
        """)
        
        # Create retraining configuration table
        logger.info("📊 Creating retraining_config table...")
        await conn.execute("""
        CREATE TABLE IF NOT EXISTS retraining_config (
            id SERIAL PRIMARY KEY,
            config_name VARCHAR(100) UNIQUE NOT NULL,
            config_value JSONB NOT NULL,
            is_active BOOLEAN DEFAULT TRUE,
            description TEXT,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            updated_at TIMESTAMPTZ DEFAULT NOW()
        );
        """)
        
        # Create retraining schedules table
        logger.info("📊 Creating retraining_schedules table...")
        await conn.execute("""
        CREATE TABLE IF NOT EXISTS retraining_schedules (
            id SERIAL PRIMARY KEY,
            schedule_name VARCHAR(100) UNIQUE NOT NULL,
            cron_expression VARCHAR(100) NOT NULL,
            trigger_type VARCHAR(50) NOT NULL, -- 'scheduled', 'performance', 'drift', 'manual'
            model_types TEXT[], -- Array of model types to retrain
            is_active BOOLEAN DEFAULT TRUE,
            last_run TIMESTAMPTZ,
            next_run TIMESTAMPTZ,
            run_count INTEGER DEFAULT 0,
            success_count INTEGER DEFAULT 0,
            failure_count INTEGER DEFAULT 0,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            updated_at TIMESTAMPTZ DEFAULT NOW()
        );
        """)
        
        # Create indexes for better query performance
        logger.info("📊 Creating indexes...")
        
        # Indexes for retraining_logs (after hypertable creation)
        await conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_retraining_logs_event_type ON retraining_logs (event_type);
        CREATE INDEX IF NOT EXISTS idx_retraining_logs_model_type ON retraining_logs (model_type);
        CREATE INDEX IF NOT EXISTS idx_retraining_logs_status ON retraining_logs (status);
        """)
        
        # Indexes for model_performance_history (after hypertable creation)
        await conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_model_performance_model_type ON model_performance_history (model_type);
        CREATE INDEX IF NOT EXISTS idx_model_performance_target ON model_performance_history (prediction_target);
        CREATE INDEX IF NOT EXISTS idx_model_performance_active ON model_performance_history (is_active);
        """)
        
        # Indexes for feature_drift_metrics (after hypertable creation)
        await conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_feature_drift_feature_name ON feature_drift_metrics (feature_name);
        CREATE INDEX IF NOT EXISTS idx_feature_drift_detected ON feature_drift_metrics (drift_detected);
        CREATE INDEX IF NOT EXISTS idx_feature_drift_severity ON feature_drift_metrics (drift_severity);
        """)
        
        # Note: Continuous aggregates will be added in a future migration
        logger.info("📊 Skipping continuous aggregates for now (will be added in future migration)")
        
        # Set up compression policies
        logger.info("📊 Setting up compression policies...")
        
        # Compress retraining_logs after 7 days
        await conn.execute("""
        SELECT add_compression_policy('retraining_logs', INTERVAL '7 days');
        """)
        
        # Compress model_performance_history after 30 days
        await conn.execute("""
        SELECT add_compression_policy('model_performance_history', INTERVAL '30 days');
        """)
        
        # Compress feature_drift_metrics after 7 days
        await conn.execute("""
        SELECT add_compression_policy('feature_drift_metrics', INTERVAL '7 days');
        """)
        
        # Set up retention policies
        logger.info("📊 Setting up retention policies...")
        
        # Keep retraining_logs for 90 days
        await conn.execute("""
        SELECT add_retention_policy('retraining_logs', INTERVAL '90 days');
        """)
        
        # Keep model_performance_history for 1 year
        await conn.execute("""
        SELECT add_retention_policy('model_performance_history', INTERVAL '1 year');
        """)
        
        # Keep feature_drift_metrics for 90 days
        await conn.execute("""
        SELECT add_retention_policy('feature_drift_metrics', INTERVAL '90 days');
        """)
        
        # Insert default retraining configuration
        logger.info("📊 Inserting default retraining configuration...")
        await conn.execute("""
        INSERT INTO retraining_config (config_name, config_value, description)
        VALUES (
            'default_retraining_config',
            '{
                "retraining_schedule": "0 2 * * *",
                "performance_check_interval": 60,
                "drift_check_interval": 30,
                "performance_threshold": 0.7,
                "drift_threshold": 0.1,
                "data_threshold": 1000,
                "max_model_versions": 3,
                "rollback_threshold": 0.05,
                "max_training_time": 3600,
                "memory_limit": "4Gi",
                "cpu_limit": "2"
            }',
            'Default automated retraining configuration'
        ) ON CONFLICT (config_name) DO NOTHING;
        """)
        
        # Insert default retraining schedules
        logger.info("📊 Inserting default retraining schedules...")
        await conn.execute("""
        INSERT INTO retraining_schedules (schedule_name, cron_expression, trigger_type, model_types, description)
        VALUES 
        (
            'daily_retraining',
            '0 2 * * *',
            'scheduled',
            ARRAY['ensemble', 'xgboost', 'catboost', 'random_forest'],
            'Daily retraining at 2 AM UTC'
        ),
        (
            'weekly_retraining',
            '0 3 * * 0',
            'scheduled',
            ARRAY['ensemble', 'xgboost', 'catboost', 'random_forest', 'neural_network'],
            'Weekly retraining on Sunday at 3 AM UTC'
        ),
        (
            'performance_monitoring',
            '0 * * * *',
            'performance',
            ARRAY['ensemble', 'xgboost', 'catboost'],
            'Hourly performance monitoring'
        ),
        (
            'drift_monitoring',
            '*/30 * * * *',
            'drift',
            ARRAY['ensemble'],
            'Drift monitoring every 30 minutes'
        )
        ON CONFLICT (schedule_name) DO NOTHING;
        """)
        
        # Close connection
        await conn.close()
        
        logger.info("✅ Migration 008: Automated Retraining Tables completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Migration 008 failed: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(run_migration())
    if success:
        print("✅ Migration completed successfully")
    else:
        print("❌ Migration failed")
        exit(1)
