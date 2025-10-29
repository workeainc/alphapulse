#!/usr/bin/env python3
"""
Phase 6: Advanced ML Model Integration Migration
Adds ML model performance tracking, model health monitoring, and advanced ML integration tables
"""

import asyncpg
import logging
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

async def upgrade(connection: asyncpg.Connection):
    """Upgrade database schema for Phase 6 Advanced ML Integration"""
    try:
        logger.info("🔄 Starting Phase 6: Advanced ML Model Integration migration...")
        
        # 1. ML Model Performance Tracking Table
        await connection.execute("""
            CREATE TABLE IF NOT EXISTS ml_model_performance (
                id SERIAL PRIMARY KEY,
                model_name VARCHAR(100) NOT NULL,
                model_type VARCHAR(50) NOT NULL,
                symbol VARCHAR(20),
                timeframe VARCHAR(10),
                timestamp TIMESTAMPTZ DEFAULT NOW(),
                accuracy FLOAT,
                precision FLOAT,
                recall FLOAT,
                f1_score FLOAT,
                auc_score FLOAT,
                latency_ms FLOAT,
                throughput_per_sec FLOAT,
                memory_usage_mb FLOAT,
                gpu_usage_percent FLOAT,
                prediction_count INTEGER DEFAULT 0,
                error_count INTEGER DEFAULT 0,
                drift_score FLOAT,
                health_score FLOAT,
                metadata JSONB,
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)
        logger.info("✅ Created ml_model_performance table")
        
        # 2. Model Health Monitoring Table
        await connection.execute("""
            CREATE TABLE IF NOT EXISTS model_health_monitoring (
                id SERIAL PRIMARY KEY,
                model_name VARCHAR(100) NOT NULL,
                timestamp TIMESTAMPTZ DEFAULT NOW(),
                health_status VARCHAR(20) DEFAULT 'healthy',
                overall_health_score FLOAT,
                feature_drift_score FLOAT,
                concept_drift_score FLOAT,
                performance_drift_score FLOAT,
                data_quality_score FLOAT,
                model_stability_score FLOAT,
                alert_level VARCHAR(20) DEFAULT 'none',
                alert_message TEXT,
                recommendations JSONB,
                metadata JSONB,
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)
        logger.info("✅ Created model_health_monitoring table")
        
        # 3. Advanced ML Integration Results Table
        await connection.execute("""
            CREATE TABLE IF NOT EXISTS advanced_ml_integration_results (
                id SERIAL PRIMARY KEY,
                signal_id VARCHAR(100) NOT NULL,
                symbol VARCHAR(20) NOT NULL,
                timeframe VARCHAR(10) NOT NULL,
                timestamp TIMESTAMPTZ DEFAULT NOW(),
                catboost_prediction FLOAT,
                catboost_confidence FLOAT,
                drift_detection_score FLOAT,
                chart_pattern_score FLOAT,
                candlestick_pattern_score FLOAT,
                volume_analysis_score FLOAT,
                ensemble_prediction FLOAT,
                ensemble_confidence FLOAT,
                ml_health_score FLOAT,
                processing_time_ms FLOAT,
                model_versions JSONB,
                feature_importance JSONB,
                prediction_explanations JSONB,
                metadata JSONB,
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)
        logger.info("✅ Created advanced_ml_integration_results table")
        
        # 4. ML Model Registry Table
        await connection.execute("""
            CREATE TABLE IF NOT EXISTS ml_model_registry (
                id SERIAL PRIMARY KEY,
                model_name VARCHAR(100) NOT NULL UNIQUE,
                model_type VARCHAR(50) NOT NULL,
                model_path VARCHAR(500) NOT NULL,
                onnx_path VARCHAR(500),
                version VARCHAR(20) NOT NULL,
                status VARCHAR(20) DEFAULT 'active',
                accuracy FLOAT,
                training_date TIMESTAMPTZ,
                last_updated TIMESTAMPTZ DEFAULT NOW(),
                model_size_mb FLOAT,
                input_features JSONB,
                output_classes JSONB,
                hyperparameters JSONB,
                performance_metrics JSONB,
                metadata JSONB,
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)
        logger.info("✅ Created ml_model_registry table")
        
        # 5. Model Training History Table
        await connection.execute("""
            CREATE TABLE IF NOT EXISTS model_training_history (
                id SERIAL PRIMARY KEY,
                model_name VARCHAR(100) NOT NULL,
                training_run_id VARCHAR(100) NOT NULL,
                timestamp TIMESTAMPTZ DEFAULT NOW(),
                training_status VARCHAR(20) DEFAULT 'completed',
                training_duration_seconds FLOAT,
                training_samples INTEGER,
                validation_samples INTEGER,
                test_samples INTEGER,
                initial_accuracy FLOAT,
                final_accuracy FLOAT,
                accuracy_improvement FLOAT,
                loss_history JSONB,
                metrics_history JSONB,
                hyperparameters JSONB,
                feature_importance JSONB,
                training_logs TEXT,
                metadata JSONB,
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)
        logger.info("✅ Created model_training_history table")
        
        # 6. Add ML-related columns to existing signals table
        await connection.execute("""
            ALTER TABLE signals 
            ADD COLUMN IF NOT EXISTS ml_model_confidence FLOAT DEFAULT 0.0,
            ADD COLUMN IF NOT EXISTS ml_health_score FLOAT DEFAULT 0.0,
            ADD COLUMN IF NOT EXISTS catboost_prediction FLOAT DEFAULT 0.0,
            ADD COLUMN IF NOT EXISTS drift_detection_score FLOAT DEFAULT 0.0,
            ADD COLUMN IF NOT EXISTS chart_pattern_ml_score FLOAT DEFAULT 0.0,
            ADD COLUMN IF NOT EXISTS candlestick_ml_score FLOAT DEFAULT 0.0,
            ADD COLUMN IF NOT EXISTS volume_ml_score FLOAT DEFAULT 0.0,
            ADD COLUMN IF NOT EXISTS ml_processing_time_ms FLOAT DEFAULT 0.0,
            ADD COLUMN IF NOT EXISTS ml_model_versions JSONB,
            ADD COLUMN IF NOT EXISTS ml_prediction_explanations JSONB
        """)
        logger.info("✅ Added ML columns to signals table")
        
        # 7. Create indexes for performance
        await connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_ml_model_performance_model_timestamp 
            ON ml_model_performance(model_name, timestamp DESC)
        """)
        
        await connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_ml_model_performance_symbol_timeframe 
            ON ml_model_performance(symbol, timeframe, timestamp DESC)
        """)
        
        await connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_model_health_monitoring_model_timestamp 
            ON model_health_monitoring(model_name, timestamp DESC)
        """)
        
        await connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_advanced_ml_integration_results_signal_id 
            ON advanced_ml_integration_results(signal_id)
        """)
        
        await connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_advanced_ml_integration_results_symbol_timeframe 
            ON advanced_ml_integration_results(symbol, timeframe, timestamp DESC)
        """)
        
        await connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_ml_model_registry_model_type_status 
            ON ml_model_registry(model_type, status)
        """)
        
        await connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_model_training_history_model_timestamp 
            ON model_training_history(model_name, timestamp DESC)
        """)
        
        logger.info("✅ Created performance indexes")
        
        # 8. Insert default ML model registry entries
        await connection.execute("""
            INSERT INTO ml_model_registry (model_name, model_type, model_path, version, status, metadata)
            VALUES 
            ('catboost_nightly_incremental', 'catboost', 'models/catboost_nightly_incremental_20250814_151525.model', '1.0.0', 'active', '{"description": "Nightly incremental CatBoost model", "training_frequency": "daily"}'),
            ('xgboost_weekly_quick', 'xgboost', 'models/xgboost_weekly_quick_20250814_151525.model', '1.0.0', 'active', '{"description": "Weekly quick XGBoost model", "training_frequency": "weekly"}'),
            ('lightgbm_monthly_full', 'lightgbm', 'models/lightgbm_monthly_full_20250814_151525.model', '1.0.0', 'active', '{"description": "Monthly full LightGBM model", "training_frequency": "monthly"}')
            ON CONFLICT (model_name) DO NOTHING
        """)
        logger.info("✅ Inserted default ML model registry entries")
        
        logger.info("✅ Phase 6: Advanced ML Model Integration migration completed successfully")
        
    except Exception as e:
        logger.error(f"❌ Error in Phase 6 migration: {e}")
        raise

async def downgrade(connection: asyncpg.Connection):
    """Downgrade database schema for Phase 6"""
    try:
        logger.info("🔄 Rolling back Phase 6: Advanced ML Model Integration migration...")
        
        # Drop tables in reverse order
        await connection.execute("DROP TABLE IF EXISTS model_training_history CASCADE")
        await connection.execute("DROP TABLE IF EXISTS ml_model_registry CASCADE")
        await connection.execute("DROP TABLE IF EXISTS advanced_ml_integration_results CASCADE")
        await connection.execute("DROP TABLE IF EXISTS model_health_monitoring CASCADE")
        await connection.execute("DROP TABLE IF EXISTS ml_model_performance CASCADE")
        
        # Remove ML columns from signals table
        await connection.execute("""
            ALTER TABLE signals 
            DROP COLUMN IF EXISTS ml_model_confidence,
            DROP COLUMN IF EXISTS ml_health_score,
            DROP COLUMN IF EXISTS catboost_prediction,
            DROP COLUMN IF EXISTS drift_detection_score,
            DROP COLUMN IF EXISTS chart_pattern_ml_score,
            DROP COLUMN IF EXISTS candlestick_ml_score,
            DROP COLUMN IF EXISTS volume_ml_score,
            DROP COLUMN IF EXISTS ml_processing_time_ms,
            DROP COLUMN IF EXISTS ml_model_versions,
            DROP COLUMN IF EXISTS ml_prediction_explanations
        """)
        
        logger.info("✅ Phase 6 rollback completed successfully")
        
    except Exception as e:
        logger.error(f"❌ Error in Phase 6 rollback: {e}")
        raise
