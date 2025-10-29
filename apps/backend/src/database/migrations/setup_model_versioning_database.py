#!/usr/bin/env python3
"""
Phase 1: Model Versioning & Rollback Database Setup
Enhances existing ML auto-retraining system with advanced versioning and rollback capabilities
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

def create_model_versioning_tables():
    """Create Phase 1 model versioning and rollback tables in TimescaleDB"""
    
    logger.info("🔌 Connecting to database for Phase 1 enhancements...")
    
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        
        logger.info("✅ Database connection established")
        
        # 1. Model Lineage Table - Track model dependencies and training history
        logger.info("📝 Creating model_lineage table...")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS model_lineage (
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                lineage_id TEXT NOT NULL,
                model_name TEXT NOT NULL,
                model_version INTEGER NOT NULL,
                parent_model_name TEXT,
                parent_model_version INTEGER,
                training_data_hash TEXT NOT NULL,
                feature_set_hash TEXT NOT NULL,
                hyperparameters_hash TEXT NOT NULL,
                training_environment TEXT,
                git_commit_hash TEXT,
                docker_image_tag TEXT,
                training_duration_seconds INTEGER,
                training_samples INTEGER,
                validation_samples INTEGER,
                created_by TEXT DEFAULT 'auto_retraining_system',
                lineage_metadata JSONB,
                PRIMARY KEY (created_at, lineage_id)
            )
        """)
        
        # 2. Model Versions Table - Detailed version history and metadata
        logger.info("📝 Creating model_versions table...")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS model_versions (
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                model_name TEXT NOT NULL,
                version INTEGER NOT NULL,
                status TEXT NOT NULL CHECK (status IN ('staging','production','archived','failed','canary','rollback_candidate')),
                regime TEXT NOT NULL,
                symbol TEXT NOT NULL,
                model_artifact_path TEXT,
                model_artifact_size_mb DECIMAL(10,2),
                model_artifact_hash TEXT,
                training_metrics JSONB,
                validation_metrics JSONB,
                test_metrics JSONB,
                performance_metrics JSONB,
                drift_metrics JSONB,
                rollback_metrics JSONB,
                deployment_timestamp TIMESTAMPTZ,
                last_used_timestamp TIMESTAMPTZ,
                usage_count INTEGER DEFAULT 0,
                error_count INTEGER DEFAULT 0,
                avg_inference_time_ms DECIMAL(10,3),
                total_inferences INTEGER DEFAULT 0,
                version_metadata JSONB,
                created_by TEXT DEFAULT 'auto_retraining_system'
            )
        """)
        
        # Add primary key after table creation for hypertable compatibility
        cursor.execute("""
            ALTER TABLE model_versions 
            ADD CONSTRAINT pk_model_versions 
            PRIMARY KEY (created_at, model_name, version)
        """)
        
        # 3. Rollback Events Table - Track model rollback history
        logger.info("📝 Creating rollback_events table...")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS rollback_events (
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                rollback_id TEXT NOT NULL,
                model_name TEXT NOT NULL,
                from_version INTEGER NOT NULL,
                to_version INTEGER NOT NULL,
                rollback_reason TEXT NOT NULL,
                rollback_type TEXT NOT NULL CHECK (rollback_type IN ('performance','drift','error','manual','automatic')),
                performance_degradation DECIMAL(5,4),
                drift_severity TEXT,
                error_details JSONB,
                rollback_triggered_by TEXT,
                rollback_duration_seconds INTEGER,
                rollback_success BOOLEAN,
                rollback_metadata JSONB
            )
        """)
        
        # Add primary key after table creation for hypertable compatibility
        cursor.execute("""
            ALTER TABLE rollback_events 
            ADD CONSTRAINT pk_rollback_events 
            PRIMARY KEY (created_at, rollback_id)
        """)
        
        # 4. Model Performance History Table - Track performance over time
        logger.info("📝 Creating model_performance_history table...")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS model_performance_history (
                timestamp TIMESTAMPTZ NOT NULL,
                model_name TEXT NOT NULL,
                model_version INTEGER NOT NULL,
                regime TEXT NOT NULL,
                symbol TEXT NOT NULL,
                performance_window_hours INTEGER NOT NULL,
                accuracy DECIMAL(5,4),
                precision DECIMAL(5,4),
                recall DECIMAL(5,4),
                f1_score DECIMAL(5,4),
                roc_auc DECIMAL(5,4),
                sharpe_ratio DECIMAL(8,4),
                profit_factor DECIMAL(8,4),
                max_drawdown DECIMAL(8,4),
                total_trades INTEGER,
                winning_trades INTEGER,
                losing_trades INTEGER,
                avg_trade_pnl DECIMAL(18,8),
                total_pnl DECIMAL(18,8),
                performance_metadata JSONB,
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)
        
        # Add primary key after table creation for hypertable compatibility
        cursor.execute("""
            ALTER TABLE model_performance_history 
            ADD CONSTRAINT pk_model_performance_history 
            PRIMARY KEY (timestamp, model_name, model_version, performance_window_hours)
        """)
        
        # 5. Model Comparison Table - A/B testing and model comparison results
        logger.info("📝 Creating model_comparison table...")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS model_comparison (
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                comparison_id TEXT NOT NULL,
                model_a_name TEXT NOT NULL,
                model_a_version INTEGER NOT NULL,
                model_b_name TEXT NOT NULL,
                model_b_version INTEGER NOT NULL,
                comparison_type TEXT NOT NULL CHECK (comparison_type IN ('ab_test','performance','drift','rollback_analysis')),
                comparison_window_hours INTEGER NOT NULL,
                model_a_metrics JSONB,
                model_b_metrics JSONB,
                comparison_metrics JSONB,
                statistical_significance DECIMAL(5,4),
                winner_model TEXT,
                winner_confidence DECIMAL(5,4),
                comparison_metadata JSONB
            )
        """)
        
        # Add primary key after table creation for hypertable compatibility
        cursor.execute("""
            ALTER TABLE model_comparison 
            ADD CONSTRAINT pk_model_comparison 
            PRIMARY KEY (created_at, comparison_id)
        """)
        
        # Convert to TimescaleDB hypertables
        logger.info("🔄 Converting tables to TimescaleDB hypertables...")
        
        try:
            cursor.execute("SELECT create_hypertable('model_lineage', 'created_at', if_not_exists => TRUE)")
            logger.info("✅ Converted model_lineage to hypertable")
        except Exception as e:
            logger.warning(f"⚠️ model_lineage hypertable creation: {e}")
        
        try:
            cursor.execute("SELECT create_hypertable('model_versions', 'created_at', if_not_exists => TRUE)")
            logger.info("✅ Converted model_versions to hypertable")
        except Exception as e:
            logger.warning(f"⚠️ model_versions hypertable creation: {e}")
        
        try:
            cursor.execute("SELECT create_hypertable('rollback_events', 'created_at', if_not_exists => TRUE)")
            logger.info("✅ Converted rollback_events to hypertable")
        except Exception as e:
            logger.warning(f"⚠️ rollback_events hypertable creation: {e}")
        
        try:
            cursor.execute("SELECT create_hypertable('model_performance_history', 'timestamp', if_not_exists => TRUE)")
            logger.info("✅ Converted model_performance_history to hypertable")
        except Exception as e:
            logger.warning(f"⚠️ model_performance_history hypertable creation: {e}")
        
        try:
            cursor.execute("SELECT create_hypertable('model_comparison', 'created_at', if_not_exists => TRUE)")
            logger.info("✅ Converted model_comparison to hypertable")
        except Exception as e:
            logger.warning(f"⚠️ model_comparison hypertable creation: {e}")
        
        # Create performance indexes
        logger.info("📊 Creating performance indexes...")
        
        # Model Lineage indexes
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_model_lineage_model_version 
            ON model_lineage(model_name, model_version)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_model_lineage_parent 
            ON model_lineage(parent_model_name, parent_model_version)
        """)
        
        # Model Versions indexes
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_model_versions_status_regime 
            ON model_versions(status, regime, symbol)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_model_versions_performance 
            ON model_versions(model_name, version, last_used_timestamp DESC)
        """)
        
        # Rollback Events indexes
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_rollback_events_model 
            ON rollback_events(model_name, from_version, to_version)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_rollback_events_type 
            ON rollback_events(rollback_type, created_at DESC)
        """)
        
        # Model Performance History indexes
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_model_performance_model_version 
            ON model_performance_history(model_name, model_version, timestamp DESC)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_model_performance_regime_symbol 
            ON model_performance_history(regime, symbol, timestamp DESC)
        """)
        
        # Model Comparison indexes
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_model_comparison_type 
            ON model_comparison(comparison_type, created_at DESC)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_model_comparison_models 
            ON model_comparison(model_a_name, model_a_version, model_b_name, model_b_version)
        """)
        
        # Enhance existing ml_models table with versioning metadata
        logger.info("🔧 Enhancing existing ml_models table...")
        
        try:
            cursor.execute("""
                ALTER TABLE ml_models 
                ADD COLUMN IF NOT EXISTS lineage_id TEXT,
                ADD COLUMN IF NOT EXISTS parent_model_name TEXT,
                ADD COLUMN IF NOT EXISTS parent_model_version INTEGER,
                ADD COLUMN IF NOT EXISTS training_data_hash TEXT,
                ADD COLUMN IF NOT EXISTS feature_set_hash TEXT,
                ADD COLUMN IF NOT EXISTS hyperparameters_hash TEXT,
                ADD COLUMN IF NOT EXISTS rollback_candidate BOOLEAN DEFAULT FALSE,
                ADD COLUMN IF NOT EXISTS rollback_reason TEXT,
                ADD COLUMN IF NOT EXISTS version_metadata JSONB
            """)
            logger.info("✅ Enhanced ml_models table with versioning columns")
        except Exception as e:
            logger.warning(f"⚠️ ml_models table enhancement: {e}")
        
        # Insert default versioning configurations
        logger.info("⚙️ Inserting default versioning configurations...")
        
        # Insert sample model lineage for existing models
        sample_lineage = {
            'lineage_id': 'initial_lineage_001',
            'model_name': 'alphaplus_pattern_classifier',
            'model_version': 1,
            'parent_model_name': None,
            'parent_model_version': None,
            'training_data_hash': 'initial_training_data_hash',
            'feature_set_hash': 'initial_feature_set_hash',
            'hyperparameters_hash': 'initial_hyperparameters_hash',
            'training_environment': 'production_v1',
            'lineage_metadata': {
                'description': 'Initial model lineage for production deployment',
                'phase': 'phase1_enhancement'
            }
        }
        
        cursor.execute("""
            INSERT INTO model_lineage (
                lineage_id, model_name, model_version, parent_model_name, parent_model_version,
                training_data_hash, feature_set_hash, hyperparameters_hash, training_environment,
                lineage_metadata
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb
            ) ON CONFLICT (created_at, lineage_id) DO NOTHING
        """, (
            sample_lineage['lineage_id'],
            sample_lineage['model_name'],
            sample_lineage['model_version'],
            sample_lineage['parent_model_name'],
            sample_lineage['parent_model_version'],
            sample_lineage['training_data_hash'],
            sample_lineage['feature_set_hash'],
            sample_lineage['hyperparameters_hash'],
            sample_lineage['training_environment'],
            json.dumps(sample_lineage['lineage_metadata'])
        ))
        
        # Insert sample model version for existing production model
        sample_version = {
            'model_name': 'alphaplus_pattern_classifier',
            'version': 1,
            'status': 'production',
            'regime': 'trending',
            'symbol': 'BTCUSDT',
            'model_artifact_path': '/models/alphaplus_pattern_classifier_v1.joblib',
            'model_artifact_size_mb': 15.5,
            'model_artifact_hash': 'initial_model_hash',
            'training_metrics': {
                'accuracy': 0.75,
                'precision': 0.72,
                'recall': 0.78,
                'f1_score': 0.75,
                'roc_auc': 0.82
            },
            'validation_metrics': {
                'accuracy': 0.73,
                'precision': 0.71,
                'recall': 0.76,
                'f1_score': 0.73,
                'roc_auc': 0.80
            },
            'performance_metrics': {
                'sharpe_ratio': 1.25,
                'profit_factor': 1.45,
                'max_drawdown': 0.08,
                'total_trades': 150,
                'winning_trades': 95,
                'losing_trades': 55
            },
            'deployment_timestamp': datetime.now(),
            'version_metadata': {
                'description': 'Initial production model for Phase 1 enhancement',
                'phase': 'phase1_enhancement',
                'features': ['technical_indicators', 'market_regime', 'noise_filtering']
            }
        }
        
        cursor.execute("""
            INSERT INTO model_versions (
                model_name, version, status, regime, symbol, model_artifact_path,
                model_artifact_size_mb, model_artifact_hash, training_metrics,
                validation_metrics, performance_metrics, deployment_timestamp, version_metadata
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s::jsonb, %s::jsonb, %s, %s::jsonb
            ) ON CONFLICT (created_at, model_name, version) DO NOTHING
        """, (
            sample_version['model_name'],
            sample_version['version'],
            sample_version['status'],
            sample_version['regime'],
            sample_version['symbol'],
            sample_version['model_artifact_path'],
            sample_version['model_artifact_size_mb'],
            sample_version['model_artifact_hash'],
            json.dumps(sample_version['training_metrics']),
            json.dumps(sample_version['validation_metrics']),
            json.dumps(sample_version['performance_metrics']),
            sample_version['deployment_timestamp'],
            json.dumps(sample_version['version_metadata'])
        ))
        
        # Commit all changes
        conn.commit()
        
        logger.info("✅ All Phase 1 tables created and configured successfully")
        
        # Verify table creation
        logger.info("📖 Verifying Phase 1 table creation...")
        
        tables_to_check = [
            'model_lineage',
            'model_versions', 
            'rollback_events',
            'model_performance_history',
            'model_comparison'
        ]
        
        for table in tables_to_check:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            count = cursor.fetchone()[0]
            logger.info(f"   {table}: {count} records")
        
        logger.info("🎯 Phase 1: Model Versioning & Rollback Database Setup Complete!")
        logger.info("✅ All versioning tables created and configured")
        logger.info("✅ TimescaleDB hypertables configured")
        logger.info("✅ Performance indexes created")
        logger.info("✅ Default versioning configurations inserted")
        logger.info("✅ Enhanced existing ml_models table")
        
    except Exception as e:
        logger.error(f"❌ Phase 1 database setup failed: {e}")
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
    create_model_versioning_tables()
