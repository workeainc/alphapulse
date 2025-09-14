#!/usr/bin/env python3
"""
Migration 008: Automated Retraining Tables (Simplified)
Add core tables for automated retraining logs, performance tracking, and drift detection
"""

import asyncio
import asyncpg
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

async def run_migration():
    """Run the simplified automated retraining tables migration"""
    
    # Database configuration
    DB_CONFIG = {
        'host': 'localhost',
        'port': 5432,
        'database': 'alphapulse',
        'user': 'alpha_emon',
        'password': 'Emon_@17711'
    }
    
    try:
        logger.info("🔄 Starting Migration 008: Automated Retraining Tables (Simplified)")
        
        # Connect to database
        conn = await asyncpg.connect(**DB_CONFIG)
        
        # Create retraining logs table (simplified)
        logger.info("📊 Creating retraining_logs table...")
        await conn.execute("""
        CREATE TABLE IF NOT EXISTS retraining_logs (
            id SERIAL PRIMARY KEY,
            timestamp TIMESTAMPTZ NOT NULL,
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
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
        """)
        
        # Create performance tracking table (simplified)
        logger.info("📊 Creating model_performance_history table...")
        await conn.execute("""
        CREATE TABLE IF NOT EXISTS model_performance_history (
            id SERIAL PRIMARY KEY,
            timestamp TIMESTAMPTZ NOT NULL,
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
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
        """)
        
        # Create drift detection table (simplified)
        logger.info("📊 Creating feature_drift_metrics table...")
        await conn.execute("""
        CREATE TABLE IF NOT EXISTS feature_drift_metrics (
            id SERIAL PRIMARY KEY,
            timestamp TIMESTAMPTZ NOT NULL,
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
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
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
        
        # Create basic indexes
        logger.info("📊 Creating basic indexes...")
        
        # Indexes for retraining_logs
        await conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_retraining_logs_timestamp ON retraining_logs (timestamp DESC);
        CREATE INDEX IF NOT EXISTS idx_retraining_logs_event_type ON retraining_logs (event_type);
        CREATE INDEX IF NOT EXISTS idx_retraining_logs_model_type ON retraining_logs (model_type);
        CREATE INDEX IF NOT EXISTS idx_retraining_logs_status ON retraining_logs (status);
        """)
        
        # Indexes for model_performance_history
        await conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_model_performance_timestamp ON model_performance_history (timestamp DESC);
        CREATE INDEX IF NOT EXISTS idx_model_performance_model_type ON model_performance_history (model_type);
        CREATE INDEX IF NOT EXISTS idx_model_performance_target ON model_performance_history (prediction_target);
        CREATE INDEX IF NOT EXISTS idx_model_performance_active ON model_performance_history (is_active);
        """)
        
        # Indexes for feature_drift_metrics
        await conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_feature_drift_timestamp ON feature_drift_metrics (timestamp DESC);
        CREATE INDEX IF NOT EXISTS idx_feature_drift_feature_name ON feature_drift_metrics (feature_name);
        CREATE INDEX IF NOT EXISTS idx_feature_drift_detected ON feature_drift_metrics (drift_detected);
        CREATE INDEX IF NOT EXISTS idx_feature_drift_severity ON feature_drift_metrics (drift_severity);
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
        INSERT INTO retraining_schedules (schedule_name, cron_expression, trigger_type, model_types)
        VALUES 
        (
            'daily_retraining',
            '0 2 * * *',
            'scheduled',
            ARRAY['ensemble', 'xgboost', 'catboost', 'random_forest']
        ),
        (
            'weekly_retraining',
            '0 3 * * 0',
            'scheduled',
            ARRAY['ensemble', 'xgboost', 'catboost', 'random_forest', 'neural_network']
        ),
        (
            'performance_monitoring',
            '0 * * * *',
            'performance',
            ARRAY['ensemble', 'xgboost', 'catboost']
        ),
        (
            'drift_monitoring',
            '*/30 * * * *',
            'drift',
            ARRAY['ensemble']
        )
        ON CONFLICT (schedule_name) DO NOTHING;
        """)
        
        # Close connection
        await conn.close()
        
        logger.info("✅ Migration 008: Automated Retraining Tables (Simplified) completed successfully")
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
