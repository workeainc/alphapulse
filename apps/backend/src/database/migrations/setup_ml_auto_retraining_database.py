#!/usr/bin/env python3
"""
ML Auto-Retraining Database Setup
Extends existing TimescaleDB infrastructure for ML model registry and evaluation
"""

import psycopg2
import logging
from datetime import datetime, timedelta
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Database configuration
DB_CONFIG = {
    'host': 'localhost',
    'database': 'alphapulse',
    'user': 'alpha_emon',
    'password': 'Emon_@17711',
    'port': 5432
}

def create_ml_auto_retraining_tables():
    """Create ML auto-retraining tables in TimescaleDB"""
    
    logger.info("🔌 Connecting to database...")
    
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        
        logger.info("✅ Database connection established")
        
        # 1. ML Model Registry Table
        logger.info("📝 Creating ml_models table...")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ml_models (
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                model_name TEXT NOT NULL,
                version INTEGER NOT NULL,
                status TEXT NOT NULL CHECK (status IN ('staging','production','archived','failed','canary')),
                regime TEXT NOT NULL,
                symbol TEXT NOT NULL,
                trained_on_daterange TSRANGE,
                featureset_hash TEXT,
                params JSONB,
                metrics JSONB,
                artifact_uri TEXT,
                training_duration_seconds INTEGER,
                training_samples INTEGER,
                validation_samples INTEGER,
                model_size_mb DECIMAL(10,2),
                created_by TEXT DEFAULT 'auto_retraining_system',
                PRIMARY KEY (created_at, model_name, version)
            )
        """)
        
        # 2. ML Evaluation History Table
        logger.info("📝 Creating ml_eval_history table...")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ml_eval_history (
                evaluated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                model_name TEXT NOT NULL,
                candidate_version INTEGER NOT NULL,
                baseline_version INTEGER,
                test_window TSRANGE,
                metrics JSONB,
                drift JSONB,
                decision TEXT NOT NULL CHECK (decision IN ('promote','reject','rollback')),
                notes TEXT,
                evaluation_duration_seconds INTEGER,
                test_samples INTEGER,
                drift_threshold_exceeded BOOLEAN DEFAULT FALSE,
                performance_improvement DECIMAL(5,4),
                risk_score DECIMAL(5,4),
                PRIMARY KEY (evaluated_at, model_name, candidate_version)
            )
        """)
        
        # 3. ML Training Jobs Table
        logger.info("📝 Creating ml_training_jobs table...")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ml_training_jobs (
                created_at TIMESTAMPTZ DEFAULT NOW(),
                job_id TEXT NOT NULL,
                model_name TEXT NOT NULL,
                regime TEXT NOT NULL,
                symbol TEXT NOT NULL,
                status TEXT NOT NULL CHECK (status IN ('pending','running','completed','failed','cancelled')),
                started_at TIMESTAMPTZ,
                completed_at TIMESTAMPTZ,
                training_data_start TIMESTAMPTZ,
                training_data_end TIMESTAMPTZ,
                training_samples INTEGER,
                validation_samples INTEGER,
                params JSONB,
                metrics JSONB,
                error_message TEXT,
                metadata JSONB,
                last_run TIMESTAMPTZ,
                updated_at TIMESTAMPTZ DEFAULT NOW(),
                PRIMARY KEY (created_at, job_id)
            )
        """)
        
        # 4. ML Performance Tracking Table
        logger.info("📝 Creating ml_performance_tracking table...")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ml_performance_tracking (
                timestamp TIMESTAMPTZ NOT NULL,
                model_name TEXT NOT NULL,
                model_version INTEGER NOT NULL,
                regime TEXT NOT NULL,
                symbol TEXT NOT NULL,
                pattern_id VARCHAR(100),
                prediction_confidence DECIMAL(5,4) NOT NULL,
                actual_outcome VARCHAR(20),
                profit_loss DECIMAL(18,8),
                prediction_correct BOOLEAN,
                market_conditions JSONB,
                feature_values JSONB,
                created_at TIMESTAMPTZ DEFAULT NOW(),
                PRIMARY KEY (timestamp, model_name, model_version, pattern_id)
            )
        """)
        
        # Convert to TimescaleDB hypertables
        logger.info("🔄 Converting tables to TimescaleDB hypertables...")
        
        try:
            cursor.execute("SELECT create_hypertable('ml_models', 'created_at', if_not_exists => TRUE)")
            logger.info("✅ Converted ml_models to hypertable")
        except Exception as e:
            logger.warning(f"⚠️ ml_models hypertable creation: {e}")
        
        try:
            cursor.execute("SELECT create_hypertable('ml_eval_history', 'evaluated_at', if_not_exists => TRUE)")
            logger.info("✅ Converted ml_eval_history to hypertable")
        except Exception as e:
            logger.warning(f"⚠️ ml_eval_history hypertable creation: {e}")
        
        try:
            cursor.execute("SELECT create_hypertable('ml_training_jobs', 'created_at', if_not_exists => TRUE)")
            logger.info("✅ Converted ml_training_jobs to hypertable")
        except Exception as e:
            logger.warning(f"⚠️ ml_training_jobs hypertable creation: {e}")
        
        try:
            cursor.execute("SELECT create_hypertable('ml_performance_tracking', 'timestamp', if_not_exists => TRUE)")
            logger.info("✅ Converted ml_performance_tracking to hypertable")
        except Exception as e:
            logger.warning(f"⚠️ ml_performance_tracking hypertable creation: {e}")
        
        # Create performance indexes
        logger.info("📊 Creating performance indexes...")
        
        # ML Models indexes
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_ml_models_name_version 
            ON ml_models(model_name, version)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_ml_models_status_regime 
            ON ml_models(status, regime, symbol)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_ml_models_created_at 
            ON ml_models(created_at DESC)
        """)
        
        # ML Evaluation History indexes
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_ml_eval_model_version 
            ON ml_eval_history(model_name, candidate_version)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_ml_eval_decision 
            ON ml_eval_history(decision, evaluated_at DESC)
        """)
        
        # ML Training Jobs indexes
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_ml_training_jobs_status 
            ON ml_training_jobs(status, created_at DESC)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_ml_training_jobs_model_regime 
            ON ml_training_jobs(model_name, regime, symbol)
        """)
        
        # ML Performance Tracking indexes
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_ml_performance_model_version 
            ON ml_performance_tracking(model_name, model_version, timestamp DESC)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_ml_performance_regime_symbol 
            ON ml_performance_tracking(regime, symbol, timestamp DESC)
        """)
        
        # Insert default ML model configurations
        logger.info("⚙️ Inserting default ML model configurations...")
        
        default_models = [
            {
                'model_name': 'alphaplus_pattern_classifier',
                'regime': 'trending',
                'symbol': 'BTCUSDT',
                'status': 'staging',
                'version': 1,
                'params': {
                    'algorithm': 'xgboost',
                    'n_estimators': 400,
                    'max_depth': 5,
                    'learning_rate': 0.05,
                    'subsample': 0.9,
                    'colsample_bytree': 0.9,
                    'reg_lambda': 1.0
                },
                'metrics': {
                    'f1': 0.0,
                    'precision': 0.0,
                    'recall': 0.0,
                    'roc_auc': 0.0,
                    'samples': 0
                }
            },
            {
                'model_name': 'alphaplus_pattern_classifier',
                'regime': 'sideways',
                'symbol': 'BTCUSDT',
                'status': 'staging',
                'version': 1,
                'params': {
                    'algorithm': 'xgboost',
                    'n_estimators': 400,
                    'max_depth': 5,
                    'learning_rate': 0.05,
                    'subsample': 0.9,
                    'colsample_bytree': 0.9,
                    'reg_lambda': 1.0
                },
                'metrics': {
                    'f1': 0.0,
                    'precision': 0.0,
                    'recall': 0.0,
                    'roc_auc': 0.0,
                    'samples': 0
                }
            },
            {
                'model_name': 'alphaplus_pattern_classifier',
                'regime': 'volatile',
                'symbol': 'BTCUSDT',
                'status': 'staging',
                'version': 1,
                'params': {
                    'algorithm': 'xgboost',
                    'n_estimators': 400,
                    'max_depth': 5,
                    'learning_rate': 0.05,
                    'subsample': 0.9,
                    'colsample_bytree': 0.9,
                    'reg_lambda': 1.0
                },
                'metrics': {
                    'f1': 0.0,
                    'precision': 0.0,
                    'recall': 0.0,
                    'roc_auc': 0.0,
                    'samples': 0
                }
            },
            {
                'model_name': 'alphaplus_pattern_classifier',
                'regime': 'consolidation',
                'symbol': 'BTCUSDT',
                'status': 'staging',
                'version': 1,
                'params': {
                    'algorithm': 'xgboost',
                    'n_estimators': 400,
                    'max_depth': 5,
                    'learning_rate': 0.05,
                    'subsample': 0.9,
                    'colsample_bytree': 0.9,
                    'reg_lambda': 1.0
                },
                'metrics': {
                    'f1': 0.0,
                    'precision': 0.0,
                    'recall': 0.0,
                    'roc_auc': 0.0,
                    'samples': 0
                }
            }
        ]
        
        for model_config in default_models:
            cursor.execute("""
                INSERT INTO ml_models (
                    model_name, version, status, regime, symbol, params, metrics
                ) VALUES (
                    %s, %s, %s, %s, %s, %s::jsonb, %s::jsonb
                ) ON CONFLICT (created_at, model_name, version) DO NOTHING
            """, (
                model_config['model_name'],
                model_config['version'],
                model_config['status'],
                model_config['regime'],
                model_config['symbol'],
                json.dumps(model_config['params']),
                json.dumps(model_config['metrics'])
            ))
        
        # Commit all changes
        conn.commit()
        
        logger.info("✅ All tables created and configured successfully")
        
        # Verify table creation
        logger.info("📖 Verifying table creation...")
        
        tables_to_check = [
            'ml_models',
            'ml_eval_history', 
            'ml_training_jobs',
            'ml_performance_tracking'
        ]
        
        for table in tables_to_check:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            count = cursor.fetchone()[0]
            logger.info(f"   {table}: {count} records")
        
        logger.info("🎯 ML Auto-Retraining Database Setup Complete!")
        logger.info("✅ All tables created and configured")
        logger.info("✅ TimescaleDB hypertables configured")
        logger.info("✅ Performance indexes created")
        logger.info("✅ Default ML model configurations inserted")
        
    except Exception as e:
        logger.error(f"❌ Database setup failed: {e}")
        if conn:
            conn.rollback()
        raise
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()
            logger.info("✅ Database connection closed")

if __name__ == "__main__":
    create_ml_auto_retraining_tables()
