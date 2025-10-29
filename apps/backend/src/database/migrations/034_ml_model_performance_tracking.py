"""
Migration: Add ML Model Performance Tracking Tables
Phase 1.3: Database schema updates for model performance tracking
"""

import asyncio
import asyncpg
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

async def upgrade(connection: asyncpg.Connection):
    """Upgrade database schema"""
    try:
        logger.info("🔄 Adding ML model performance tracking tables...")
        
        # Create ML model performance tracking table
        await connection.execute("""
            CREATE TABLE IF NOT EXISTS ml_model_performance (
                id SERIAL PRIMARY KEY,
                model_name VARCHAR(100) NOT NULL,
                model_type VARCHAR(50) NOT NULL,
                symbol VARCHAR(20),
                timeframe VARCHAR(10),
                accuracy FLOAT,
                precision FLOAT,
                recall FLOAT,
                f1_score FLOAT,
                roc_auc FLOAT,
                prediction_latency_ms FLOAT,
                training_time_seconds FLOAT,
                last_training_at TIMESTAMPTZ,
                model_version VARCHAR(50),
                is_active BOOLEAN DEFAULT TRUE,
                created_at TIMESTAMPTZ DEFAULT NOW(),
                updated_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)
        
        # Create drift detection logs table
        await connection.execute("""
            CREATE TABLE IF NOT EXISTS drift_detection_logs (
                id SERIAL PRIMARY KEY,
                feature_name VARCHAR(100) NOT NULL,
                symbol VARCHAR(20),
                timeframe VARCHAR(10),
                drift_score FLOAT NOT NULL,
                drift_type VARCHAR(50),
                severity VARCHAR(20),
                confidence FLOAT,
                detection_method VARCHAR(50),
                reference_window_start TIMESTAMPTZ,
                reference_window_end TIMESTAMPTZ,
                detection_window_start TIMESTAMPTZ,
                detection_window_end TIMESTAMPTZ,
                alert_sent BOOLEAN DEFAULT FALSE,
                retraining_triggered BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)
        
        # Create ONNX model registry table
        await connection.execute("""
            CREATE TABLE IF NOT EXISTS onnx_model_registry (
                id SERIAL PRIMARY KEY,
                model_name VARCHAR(100) NOT NULL UNIQUE,
                original_model_path VARCHAR(255),
                onnx_model_path VARCHAR(255) NOT NULL,
                model_type VARCHAR(50) NOT NULL,
                input_shape JSONB,
                output_shape JSONB,
                conversion_time_seconds FLOAT,
                model_size_mb FLOAT,
                inference_latency_ms FLOAT,
                accuracy_comparison FLOAT,
                is_active BOOLEAN DEFAULT TRUE,
                created_at TIMESTAMPTZ DEFAULT NOW(),
                updated_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)
        
        # Create indexes for efficient querying
        try:
            await connection.execute("""
                CREATE INDEX IF NOT EXISTS idx_ml_model_performance_model_name 
                ON ml_model_performance(model_name)
            """)
            
            await connection.execute("""
                CREATE INDEX IF NOT EXISTS idx_ml_model_performance_symbol_timeframe 
                ON ml_model_performance(symbol, timeframe)
            """)
            
            await connection.execute("""
                CREATE INDEX IF NOT EXISTS idx_ml_model_performance_created_at 
                ON ml_model_performance(created_at DESC)
            """)
            
            await connection.execute("""
                CREATE INDEX IF NOT EXISTS idx_drift_detection_logs_feature_symbol 
                ON drift_detection_logs(feature_name, symbol)
            """)
            
            await connection.execute("""
                CREATE INDEX IF NOT EXISTS idx_drift_detection_logs_created_at 
                ON drift_detection_logs(created_at DESC)
            """)
            
            await connection.execute("""
                CREATE INDEX IF NOT EXISTS idx_drift_detection_logs_severity 
                ON drift_detection_logs(severity)
            """)
            
            await connection.execute("""
                CREATE INDEX IF NOT EXISTS idx_onnx_model_registry_model_name 
                ON onnx_model_registry(model_name)
            """)
            
            await connection.execute("""
                CREATE INDEX IF NOT EXISTS idx_onnx_model_registry_model_type 
                ON onnx_model_registry(model_type)
            """)
            
            logger.info("✅ Indexes created successfully")
        except Exception as e:
            logger.warning(f"⚠️ Could not create some indexes: {e}")
        
        # Add columns to existing signals table for ML model tracking
        await connection.execute("""
            ALTER TABLE signals 
            ADD COLUMN IF NOT EXISTS ml_model_used VARCHAR(100),
            ADD COLUMN IF NOT EXISTS ml_prediction_confidence FLOAT,
            ADD COLUMN IF NOT EXISTS ml_model_version VARCHAR(50),
            ADD COLUMN IF NOT EXISTS drift_score_at_prediction FLOAT
        """)
        
        # Create TimescaleDB hypertables for time-series data
        try:
            await connection.execute("""
                SELECT create_hypertable('ml_model_performance', 'created_at', if_not_exists => TRUE)
            """)
            logger.info("✅ ml_model_performance converted to hypertable")
        except Exception as e:
            logger.warning(f"⚠️ Could not create hypertable for ml_model_performance: {e}")
        
        try:
            await connection.execute("""
                SELECT create_hypertable('drift_detection_logs', 'created_at', if_not_exists => TRUE)
            """)
            logger.info("✅ drift_detection_logs converted to hypertable")
        except Exception as e:
            logger.warning(f"⚠️ Could not create hypertable for drift_detection_logs: {e}")
        
        # Add columns to existing signals table for ML model tracking
        try:
            await connection.execute("""
                ALTER TABLE signals 
                ADD COLUMN IF NOT EXISTS ml_model_used VARCHAR(100),
                ADD COLUMN IF NOT EXISTS ml_prediction_confidence FLOAT,
                ADD COLUMN IF NOT EXISTS ml_model_version VARCHAR(50),
                ADD COLUMN IF NOT EXISTS drift_score_at_prediction FLOAT
            """)
            logger.info("✅ Added ML tracking columns to signals table")
        except Exception as e:
            logger.warning(f"⚠️ Could not add ML tracking columns to signals table: {e}")
        
        logger.info("✅ ML model performance tracking tables created successfully")
        
    except Exception as e:
        logger.error(f"❌ Error creating ML model performance tables: {e}")
        raise

async def downgrade(connection: asyncpg.Connection):
    """Downgrade database schema"""
    try:
        logger.info("🔄 Removing ML model performance tracking tables...")
        
        # Drop tables
        await connection.execute("DROP TABLE IF EXISTS ml_model_performance CASCADE")
        await connection.execute("DROP TABLE IF EXISTS drift_detection_logs CASCADE")
        await connection.execute("DROP TABLE IF EXISTS onnx_model_registry CASCADE")
        
        # Remove columns from signals table
        await connection.execute("""
            ALTER TABLE signals 
            DROP COLUMN IF EXISTS ml_model_used,
            DROP COLUMN IF EXISTS ml_prediction_confidence,
            DROP COLUMN IF EXISTS ml_model_version,
            DROP COLUMN IF EXISTS drift_score_at_prediction
        """)
        
        logger.info("✅ ML model performance tracking tables removed successfully")
        
    except Exception as e:
        logger.error(f"❌ Error removing ML model performance tables: {e}")
        raise

async def main():
    """Run migration"""
    try:
        # Connect to database
        connection = await asyncpg.connect(
            host='localhost',
            port=5432,
            user='alpha_emon',
            password='Emon_@17711',
            database='alphapulse'
        )
        
        # Run upgrade
        await upgrade(connection)
        
        # Close connection
        await connection.close()
        
        logger.info("🎉 Migration completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Migration failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
