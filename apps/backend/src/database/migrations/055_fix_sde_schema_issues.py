"""
Migration 055: Fix SDE Schema Issues
Fix missing columns and incorrect column names in SDE tables
"""

import asyncio
import logging
import asyncpg
from datetime import datetime

logger = logging.getLogger(__name__)

async def run_migration():
    """Run the SDE schema fixes migration"""
    try:
        # Database connection
        db_pool = await asyncpg.create_pool(
            host='localhost',
            port=5432,
            user='postgres',
            password='Emon_@17711',
            database='alphapulse'
        )
        
        async with db_pool.acquire() as conn:
            logger.info("🚀 Starting SDE Schema Fixes Migration")
            
            # Fix 1: Add missing columns to sde_model_consensus_tracking
            logger.info("📝 Fix 1: Adding missing columns to sde_model_consensus_tracking")
            try:
                await conn.execute("""
                    ALTER TABLE sde_model_consensus_tracking 
                    ADD COLUMN IF NOT EXISTS signal_id VARCHAR(50),
                    ADD COLUMN IF NOT EXISTS symbol VARCHAR(20),
                    ADD COLUMN IF NOT EXISTS timeframe VARCHAR(10),
                    ADD COLUMN IF NOT EXISTS consensus_direction VARCHAR(10),
                    ADD COLUMN IF NOT EXISTS agreeing_heads_count INTEGER DEFAULT 0,
                    ADD COLUMN IF NOT EXISTS min_agreeing_heads INTEGER DEFAULT 2,
                    ADD COLUMN IF NOT EXISTS min_head_probability DECIMAL(5,4) DEFAULT 0.6,
                    ADD COLUMN IF NOT EXISTS processing_time_ms INTEGER,
                    ADD COLUMN IF NOT EXISTS consensus_reason TEXT
                """)
                logger.info("✅ Added missing columns to sde_model_consensus_tracking")
            except Exception as e:
                logger.warning(f"⚠️ Some columns may already exist: {e}")
            
            # Fix 2: Add missing columns to sde_ensemble_predictions
            logger.info("📝 Fix 2: Adding missing columns to sde_ensemble_predictions")
            try:
                await conn.execute("""
                    ALTER TABLE sde_ensemble_predictions 
                    ADD COLUMN IF NOT EXISTS symbol VARCHAR(20),
                    ADD COLUMN IF NOT EXISTS timeframe VARCHAR(10),
                    ADD COLUMN IF NOT EXISTS timestamp TIMESTAMPTZ DEFAULT NOW(),
                    ADD COLUMN IF NOT EXISTS model_predictions JSONB,
                    ADD COLUMN IF NOT EXISTS consensus_score DECIMAL(5,4),
                    ADD COLUMN IF NOT EXISTS agreement_count INTEGER DEFAULT 0,
                    ADD COLUMN IF NOT EXISTS confidence_score DECIMAL(5,4),
                    ADD COLUMN IF NOT EXISTS calibrated_confidence DECIMAL(5,4),
                    ADD COLUMN IF NOT EXISTS signal_direction VARCHAR(10),
                    ADD COLUMN IF NOT EXISTS signal_strength DECIMAL(5,4),
                    ADD COLUMN IF NOT EXISTS fusion_method VARCHAR(50),
                    ADD COLUMN IF NOT EXISTS processing_time_ms INTEGER
                """)
                logger.info("✅ Added missing columns to sde_ensemble_predictions")
            except Exception as e:
                logger.warning(f"⚠️ Some columns may already exist: {e}")
            
            # Fix 3: Add missing columns to sde_signal_limits
            logger.info("📝 Fix 3: Adding missing columns to sde_signal_limits")
            try:
                await conn.execute("""
                    ALTER TABLE sde_signal_limits 
                    ADD COLUMN IF NOT EXISTS account_id VARCHAR(50),
                    ADD COLUMN IF NOT EXISTS symbol VARCHAR(20),
                    ADD COLUMN IF NOT EXISTS timeframe VARCHAR(10),
                    ADD COLUMN IF NOT EXISTS timestamp TIMESTAMPTZ DEFAULT NOW(),
                    ADD COLUMN IF NOT EXISTS account_limit_reached BOOLEAN DEFAULT FALSE,
                    ADD COLUMN IF NOT EXISTS symbol_limit_reached BOOLEAN DEFAULT FALSE,
                    ADD COLUMN IF NOT EXISTS max_account_signals INTEGER DEFAULT 10,
                    ADD COLUMN IF NOT EXISTS max_symbol_signals INTEGER DEFAULT 5,
                    ADD COLUMN IF NOT EXISTS current_account_signals INTEGER DEFAULT 0,
                    ADD COLUMN IF NOT EXISTS current_symbol_signals INTEGER DEFAULT 0,
                    ADD COLUMN IF NOT EXISTS limit_window_hours INTEGER DEFAULT 24
                """)
                logger.info("✅ Added missing columns to sde_signal_limits")
            except Exception as e:
                logger.warning(f"⚠️ Some columns may already exist: {e}")
            
            # Fix 4: Add missing columns to sde_signal_quality_metrics
            logger.info("📝 Fix 4: Adding missing columns to sde_signal_quality_metrics")
            try:
                await conn.execute("""
                    ALTER TABLE sde_signal_quality_metrics 
                    ADD COLUMN IF NOT EXISTS symbol VARCHAR(20),
                    ADD COLUMN IF NOT EXISTS timeframe VARCHAR(10),
                    ADD COLUMN IF NOT EXISTS timestamp TIMESTAMPTZ DEFAULT NOW(),
                    ADD COLUMN IF NOT EXISTS overall_quality_score DECIMAL(5,4),
                    ADD COLUMN IF NOT EXISTS quality_level VARCHAR(20),
                    ADD COLUMN IF NOT EXISTS validation_passed BOOLEAN DEFAULT FALSE,
                    ADD COLUMN IF NOT EXISTS rejection_reasons TEXT[],
                    ADD COLUMN IF NOT EXISTS market_regime VARCHAR(50),
                    ADD COLUMN IF NOT EXISTS confidence_score DECIMAL(5,4),
                    ADD COLUMN IF NOT EXISTS volatility_score DECIMAL(5,4),
                    ADD COLUMN IF NOT EXISTS volume_score DECIMAL(5,4),
                    ADD COLUMN IF NOT EXISTS trend_score DECIMAL(5,4),
                    ADD COLUMN IF NOT EXISTS momentum_score DECIMAL(5,4),
                    ADD COLUMN IF NOT EXISTS processing_time_ms INTEGER
                """)
                logger.info("✅ Added missing columns to sde_signal_quality_metrics")
            except Exception as e:
                logger.warning(f"⚠️ Some columns may already exist: {e}")
            
            # Fix 5: Create missing tables if they don't exist
            logger.info("📝 Fix 5: Creating missing tables")
            
            # Create sde_ensemble_predictions if it doesn't exist
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS sde_ensemble_predictions (
                    prediction_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    symbol VARCHAR(20) NOT NULL,
                    timeframe VARCHAR(10) NOT NULL,
                    timestamp TIMESTAMPTZ DEFAULT NOW(),
                    ensemble_prediction DECIMAL(5,4) NOT NULL,
                    model_predictions JSONB,
                    consensus_score DECIMAL(5,4),
                    agreement_count INTEGER DEFAULT 0,
                    confidence_score DECIMAL(5,4),
                    calibrated_confidence DECIMAL(5,4),
                    signal_direction VARCHAR(10),
                    signal_strength DECIMAL(5,4),
                    fusion_method VARCHAR(50),
                    processing_time_ms INTEGER,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)
            
            # Create sde_signal_limits if it doesn't exist
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS sde_signal_limits (
                    limit_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    account_id VARCHAR(50) NOT NULL,
                    symbol VARCHAR(20) NOT NULL,
                    timeframe VARCHAR(10) NOT NULL,
                    timestamp TIMESTAMPTZ DEFAULT NOW(),
                    account_limit_reached BOOLEAN DEFAULT FALSE,
                    symbol_limit_reached BOOLEAN DEFAULT FALSE,
                    max_account_signals INTEGER DEFAULT 10,
                    max_symbol_signals INTEGER DEFAULT 5,
                    current_account_signals INTEGER DEFAULT 0,
                    current_symbol_signals INTEGER DEFAULT 0,
                    limit_window_hours INTEGER DEFAULT 24,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)
            
            # Create sde_signal_quality_metrics if it doesn't exist
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS sde_signal_quality_metrics (
                    metric_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    symbol VARCHAR(20) NOT NULL,
                    timeframe VARCHAR(10) NOT NULL,
                    timestamp TIMESTAMPTZ DEFAULT NOW(),
                    overall_quality_score DECIMAL(5,4) DEFAULT 0.0,
                    quality_level VARCHAR(20) DEFAULT 'REJECT',
                    validation_passed BOOLEAN DEFAULT FALSE,
                    rejection_reasons TEXT[],
                    market_regime VARCHAR(50) DEFAULT 'unknown',
                    confidence_score DECIMAL(5,4),
                    volatility_score DECIMAL(5,4),
                    volume_score DECIMAL(5,4),
                    trend_score DECIMAL(5,4),
                    momentum_score DECIMAL(5,4),
                    processing_time_ms INTEGER,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)
            
            logger.info("✅ Created missing tables")
            
            # Fix 6: Create indexes for performance
            logger.info("📝 Fix 6: Creating performance indexes")
            
            indexes = [
                ("idx_sde_ensemble_symbol_time", "sde_ensemble_predictions", "symbol, timestamp"),
                ("idx_sde_ensemble_timestamp", "sde_ensemble_predictions", "timestamp"),
                ("idx_sde_limits_account_symbol", "sde_signal_limits", "account_id, symbol"),
                ("idx_sde_limits_timestamp", "sde_signal_limits", "timestamp"),
                ("idx_sde_quality_symbol_time", "sde_signal_quality_metrics", "symbol, timestamp"),
                ("idx_sde_quality_timestamp", "sde_signal_quality_metrics", "timestamp"),
                ("idx_sde_consensus_symbol_time", "sde_model_consensus_tracking", "symbol, timestamp"),
                ("idx_sde_consensus_timestamp", "sde_model_consensus_tracking", "timestamp")
            ]
            
            for index_name, table_name, columns in indexes:
                try:
                    await conn.execute(f"""
                        CREATE INDEX IF NOT EXISTS {index_name} ON {table_name} ({columns})
                    """)
                except Exception as e:
                    logger.warning(f"⚠️ Index {index_name} may already exist: {e}")
            
            logger.info("✅ Created performance indexes")
            
            logger.info("🎉 SDE Schema Fixes Migration completed successfully!")
            
    except Exception as e:
        logger.error(f"❌ Migration failed: {e}")
        raise
    finally:
        if 'db_pool' in locals():
            await db_pool.close()

if __name__ == "__main__":
    asyncio.run(run_migration())
