"""
Migration 051: Advanced Model Fusion & Calibration (Phase 7)
Implements advanced model fusion, probability calibration, and performance tracking
"""

import asyncio
import asyncpg
import logging
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def run_migration():
    """Execute Phase 7 migration for Advanced Model Fusion & Calibration"""
    
    # Database connection parameters
    DB_CONFIG = {
        'host': 'localhost',
        'port': 5432,
        'user': 'alpha_emon',
        'password': 'Emon_@17711',
        'database': 'alphapulse'
    }
    
    try:
        # Connect to database
        conn = await asyncpg.connect(**DB_CONFIG)
        logger.info("✅ Connected to database successfully")
        
        # Create tables for Phase 7
        
        # 1. Model Fusion Configuration
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS sde_model_fusion_config (
                id SERIAL PRIMARY KEY,
                fusion_method VARCHAR(50) NOT NULL DEFAULT 'weighted_average',
                model_weights JSONB NOT NULL DEFAULT '{"catboost": 0.4, "logistic": 0.2, "decision_tree": 0.2, "rule_based": 0.2}',
                consensus_threshold DECIMAL(5,4) NOT NULL DEFAULT 0.7000,
                min_agreement_count INTEGER NOT NULL DEFAULT 3,
                calibration_method VARCHAR(50) NOT NULL DEFAULT 'isotonic',
                confidence_threshold DECIMAL(5,4) NOT NULL DEFAULT 0.8500,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                is_active BOOLEAN DEFAULT TRUE
            )
        """)
        logger.info("✅ Created sde_model_fusion_config table")
        
        # 2. Model Calibration Data
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS sde_model_calibration (
                id SERIAL PRIMARY KEY,
                model_name VARCHAR(100) NOT NULL,
                calibration_type VARCHAR(50) NOT NULL, -- 'isotonic', 'platt', 'temperature'
                calibration_params JSONB NOT NULL DEFAULT '{}',
                calibration_data JSONB NOT NULL DEFAULT '{}',
                accuracy_before DECIMAL(5,4),
                accuracy_after DECIMAL(5,4),
                brier_score_before DECIMAL(5,4),
                brier_score_after DECIMAL(5,4),
                calibration_date TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                is_active BOOLEAN DEFAULT TRUE,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
        """)
        logger.info("✅ Created sde_model_calibration table")
        
        # 3. Ensemble Predictions
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS sde_ensemble_predictions (
                id SERIAL PRIMARY KEY,
                symbol VARCHAR(20) NOT NULL,
                timeframe VARCHAR(10) NOT NULL,
                timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
                model_predictions JSONB NOT NULL, -- {"catboost": 0.75, "logistic": 0.68, ...}
                ensemble_prediction DECIMAL(5,4) NOT NULL,
                consensus_score DECIMAL(5,4) NOT NULL,
                agreement_count INTEGER NOT NULL,
                confidence_score DECIMAL(5,4) NOT NULL,
                calibrated_confidence DECIMAL(5,4),
                signal_direction VARCHAR(10), -- 'LONG', 'SHORT', 'FLAT'
                signal_strength DECIMAL(5,4),
                fusion_method VARCHAR(50) NOT NULL,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
        """)
        logger.info("✅ Created sde_ensemble_predictions table")
        
        # 4. Model Performance Tracking
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS sde_model_performance (
                id SERIAL PRIMARY KEY,
                model_name VARCHAR(100) NOT NULL,
                symbol VARCHAR(20) NOT NULL,
                timeframe VARCHAR(10) NOT NULL,
                period_start TIMESTAMP WITH TIME ZONE NOT NULL,
                period_end TIMESTAMP WITH TIME ZONE NOT NULL,
                total_predictions INTEGER NOT NULL DEFAULT 0,
                correct_predictions INTEGER NOT NULL DEFAULT 0,
                accuracy DECIMAL(5,4) NOT NULL DEFAULT 0.0000,
                precision DECIMAL(5,4),
                recall DECIMAL(5,4),
                f1_score DECIMAL(5,4),
                auc_score DECIMAL(5,4),
                brier_score DECIMAL(5,4),
                profit_factor DECIMAL(8,4),
                sharpe_ratio DECIMAL(8,4),
                max_drawdown DECIMAL(8,4),
                avg_win DECIMAL(8,4),
                avg_loss DECIMAL(8,4),
                win_rate DECIMAL(5,4),
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
        """)
        logger.info("✅ Created sde_model_performance table")
        
        # 5. Calibration History
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS sde_calibration_history (
                id SERIAL PRIMARY KEY,
                model_name VARCHAR(100) NOT NULL,
                calibration_type VARCHAR(50) NOT NULL,
                calibration_date TIMESTAMP WITH TIME ZONE NOT NULL,
                pre_calibration_metrics JSONB NOT NULL DEFAULT '{}',
                post_calibration_metrics JSONB NOT NULL DEFAULT '{}',
                calibration_samples INTEGER NOT NULL DEFAULT 0,
                validation_samples INTEGER NOT NULL DEFAULT 0,
                is_successful BOOLEAN DEFAULT TRUE,
                error_message TEXT,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
        """)
        logger.info("✅ Created sde_calibration_history table")
        
        # 6. Model Drift Detection
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS sde_model_drift (
                id SERIAL PRIMARY KEY,
                model_name VARCHAR(100) NOT NULL,
                symbol VARCHAR(20) NOT NULL,
                timeframe VARCHAR(10) NOT NULL,
                drift_type VARCHAR(50) NOT NULL, -- 'concept', 'data', 'performance'
                drift_score DECIMAL(5,4) NOT NULL,
                drift_threshold DECIMAL(5,4) NOT NULL DEFAULT 0.1000,
                is_drift_detected BOOLEAN NOT NULL DEFAULT FALSE,
                drift_metrics JSONB NOT NULL DEFAULT '{}',
                detection_date TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                last_training_date TIMESTAMP WITH TIME ZONE,
                recommended_action VARCHAR(100), -- 'retrain', 'recalibrate', 'monitor'
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
        """)
        logger.info("✅ Created sde_model_drift table")
        
        # Insert default fusion configuration
        await conn.execute("""
            INSERT INTO sde_model_fusion_config 
            (fusion_method, model_weights, consensus_threshold, min_agreement_count, calibration_method, confidence_threshold)
            VALUES 
            ('weighted_average', '{"catboost": 0.4, "logistic": 0.2, "decision_tree": 0.2, "rule_based": 0.2}', 0.7000, 3, 'isotonic', 0.8500)
            ON CONFLICT DO NOTHING
        """)
        logger.info("✅ Inserted default fusion configuration")
        
        # Create indexes for performance
        indexes = [
            ("idx_ensemble_predictions_symbol_time", "sde_ensemble_predictions", "(symbol, timeframe, timestamp)"),
            ("idx_ensemble_predictions_timestamp", "sde_ensemble_predictions", "(timestamp DESC)"),
            ("idx_model_performance_model_symbol", "sde_model_performance", "(model_name, symbol, timeframe)"),
            ("idx_model_performance_period", "sde_model_performance", "(period_start, period_end)"),
            ("idx_model_calibration_model", "sde_model_calibration", "(model_name, calibration_type)"),
            ("idx_model_drift_model_symbol", "sde_model_drift", "(model_name, symbol, timeframe)"),
            ("idx_model_drift_detection_date", "sde_model_drift", "(detection_date DESC)"),
            ("idx_calibration_history_model", "sde_calibration_history", "(model_name, calibration_date DESC)")
        ]
        
        for index_name, table_name, columns in indexes:
            try:
                await conn.execute(f"CREATE INDEX IF NOT EXISTS {index_name} ON {table_name} {columns}")
                logger.info(f"✅ Created index: {index_name}")
            except Exception as e:
                logger.warning(f"⚠️ Index {index_name} already exists or failed: {e}")
        
        # Verify table creation
        tables = [
            'sde_model_fusion_config',
            'sde_model_calibration', 
            'sde_ensemble_predictions',
            'sde_model_performance',
            'sde_calibration_history',
            'sde_model_drift'
        ]
        
        for table in tables:
            result = await conn.fetchval(f"SELECT COUNT(*) FROM {table}")
            logger.info(f"✅ Table {table}: {result} rows")
        
        await conn.close()
        logger.info("✅ Phase 7 migration completed successfully")
        
    except Exception as e:
        logger.error(f"❌ Migration failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(run_migration())
