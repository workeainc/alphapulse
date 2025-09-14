"""
Migration 052: Advanced Calibration System (Phase 8)
Implements advanced calibration tables for 90%+ signal accuracy
"""

import asyncio
import asyncpg
import logging
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def run_migration():
    """Execute Phase 8 migration for Advanced Calibration System"""
    
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
        
        # Create tables for Phase 8
        
        # 1. Advanced Calibration History
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS sde_calibration_history (
                id SERIAL PRIMARY KEY,
                symbol VARCHAR(20) NOT NULL,
                timeframe VARCHAR(10) NOT NULL,
                raw_probability DECIMAL(5,4) NOT NULL,
                calibrated_probability DECIMAL(5,4) NOT NULL,
                calibration_methods JSONB NOT NULL DEFAULT '{}',
                reliability_score DECIMAL(5,4) NOT NULL,
                confidence_interval_lower DECIMAL(5,4),
                confidence_interval_upper DECIMAL(5,4),
                market_regime VARCHAR(20),
                calibration_date TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
        """)
        logger.info("✅ Created sde_calibration_history table")
        
        # 2. Dynamic Threshold Configuration
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS sde_dynamic_thresholds (
                id SERIAL PRIMARY KEY,
                market_regime VARCHAR(20) NOT NULL,
                volatility_level VARCHAR(20) NOT NULL,
                min_confidence_threshold DECIMAL(5,4) NOT NULL,
                min_consensus_heads INTEGER NOT NULL,
                min_probability_threshold DECIMAL(5,4) NOT NULL,
                calibration_weight_isotonic DECIMAL(5,4) NOT NULL DEFAULT 0.4000,
                calibration_weight_platt DECIMAL(5,4) NOT NULL DEFAULT 0.3000,
                calibration_weight_temperature DECIMAL(5,4) NOT NULL DEFAULT 0.2000,
                calibration_weight_ensemble DECIMAL(5,4) NOT NULL DEFAULT 0.1000,
                is_active BOOLEAN DEFAULT TRUE,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
        """)
        logger.info("✅ Created sde_dynamic_thresholds table")
        
        # 3. Feature Importance Tracking
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS sde_feature_importance (
                id SERIAL PRIMARY KEY,
                symbol VARCHAR(20) NOT NULL,
                timeframe VARCHAR(10) NOT NULL,
                feature_name VARCHAR(100) NOT NULL,
                importance_score DECIMAL(8,6) NOT NULL,
                feature_type VARCHAR(50) NOT NULL, -- 'technical', 'sentiment', 'volume', 'regime'
                model_head VARCHAR(50) NOT NULL, -- 'head_a', 'head_b', 'head_c', 'head_d'
                performance_impact DECIMAL(5,4),
                last_updated TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
        """)
        logger.info("✅ Created sde_feature_importance table")
        
        # 4. Accuracy Performance Tracking
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS sde_accuracy_performance (
                id SERIAL PRIMARY KEY,
                symbol VARCHAR(20) NOT NULL,
                timeframe VARCHAR(10) NOT NULL,
                period_start TIMESTAMP WITH TIME ZONE NOT NULL,
                period_end TIMESTAMP WITH TIME ZONE NOT NULL,
                total_signals INTEGER NOT NULL DEFAULT 0,
                correct_signals INTEGER NOT NULL DEFAULT 0,
                accuracy_rate DECIMAL(5,4) NOT NULL DEFAULT 0.0000,
                brier_score DECIMAL(8,6),
                calibration_error DECIMAL(8,6),
                reliability_score DECIMAL(5,4),
                market_regime VARCHAR(20),
                volatility_level VARCHAR(20),
                avg_confidence DECIMAL(5,4),
                avg_calibration_adjustment DECIMAL(8,6),
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
        """)
        logger.info("✅ Created sde_accuracy_performance table")
        
        # 5. Model Consensus Optimization
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS sde_consensus_optimization (
                id SERIAL PRIMARY KEY,
                market_regime VARCHAR(20) NOT NULL,
                volatility_level VARCHAR(20) NOT NULL,
                min_agreeing_heads INTEGER NOT NULL,
                min_probability_threshold DECIMAL(5,4) NOT NULL,
                consensus_weight_head_a DECIMAL(5,4) NOT NULL DEFAULT 0.4000,
                consensus_weight_head_b DECIMAL(5,4) NOT NULL DEFAULT 0.2000,
                consensus_weight_head_c DECIMAL(5,4) NOT NULL DEFAULT 0.2000,
                consensus_weight_head_d DECIMAL(5,4) NOT NULL DEFAULT 0.2000,
                direction_agreement_required BOOLEAN DEFAULT TRUE,
                confidence_threshold DECIMAL(5,4) NOT NULL DEFAULT 0.8500,
                optimization_date TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                performance_improvement DECIMAL(8,6),
                is_active BOOLEAN DEFAULT TRUE,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
        """)
        logger.info("✅ Created sde_consensus_optimization table")
        
        # Insert default dynamic thresholds
        default_thresholds = [
            ('bullish', 'low', 0.80, 2, 0.70, 0.50, 0.30, 0.15, 0.05),
            ('bullish', 'medium', 0.85, 3, 0.75, 0.45, 0.30, 0.20, 0.05),
            ('bullish', 'high', 0.90, 4, 0.80, 0.40, 0.30, 0.25, 0.05),
            ('bearish', 'low', 0.85, 3, 0.75, 0.30, 0.50, 0.15, 0.05),
            ('bearish', 'medium', 0.90, 4, 0.80, 0.25, 0.55, 0.15, 0.05),
            ('bearish', 'high', 0.95, 4, 0.85, 0.20, 0.60, 0.15, 0.05),
            ('sideways', 'low', 0.85, 3, 0.75, 0.40, 0.30, 0.25, 0.05),
            ('sideways', 'medium', 0.90, 3, 0.80, 0.35, 0.30, 0.30, 0.05),
            ('sideways', 'high', 0.95, 4, 0.85, 0.30, 0.30, 0.35, 0.05),
            ('volatile', 'low', 0.90, 4, 0.80, 0.20, 0.20, 0.55, 0.05),
            ('volatile', 'medium', 0.95, 4, 0.85, 0.15, 0.15, 0.65, 0.05),
            ('volatile', 'high', 0.98, 4, 0.90, 0.10, 0.10, 0.75, 0.05)
        ]
        
        for threshold in default_thresholds:
            await conn.execute("""
                INSERT INTO sde_dynamic_thresholds 
                (market_regime, volatility_level, min_confidence_threshold, min_consensus_heads,
                 min_probability_threshold, calibration_weight_isotonic, calibration_weight_platt,
                 calibration_weight_temperature, calibration_weight_ensemble)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                ON CONFLICT DO NOTHING
            """, *threshold)
        
        logger.info("✅ Inserted default dynamic thresholds")
        
        # Insert default consensus optimization
        default_consensus = [
            ('bullish', 'low', 2, 0.70, 0.50, 0.20, 0.20, 0.10, True, 0.80),
            ('bullish', 'medium', 3, 0.75, 0.45, 0.20, 0.20, 0.15, True, 0.85),
            ('bullish', 'high', 4, 0.80, 0.40, 0.20, 0.20, 0.20, True, 0.90),
            ('bearish', 'low', 3, 0.75, 0.30, 0.50, 0.20, 0.00, True, 0.85),
            ('bearish', 'medium', 4, 0.80, 0.25, 0.55, 0.20, 0.00, True, 0.90),
            ('bearish', 'high', 4, 0.85, 0.20, 0.60, 0.20, 0.00, True, 0.95),
            ('sideways', 'low', 3, 0.75, 0.40, 0.30, 0.20, 0.10, True, 0.85),
            ('sideways', 'medium', 3, 0.80, 0.35, 0.30, 0.20, 0.15, True, 0.90),
            ('sideways', 'high', 4, 0.85, 0.30, 0.30, 0.20, 0.20, True, 0.95),
            ('volatile', 'low', 4, 0.80, 0.20, 0.20, 0.20, 0.40, True, 0.90),
            ('volatile', 'medium', 4, 0.85, 0.15, 0.15, 0.20, 0.50, True, 0.95),
            ('volatile', 'high', 4, 0.90, 0.10, 0.10, 0.20, 0.60, True, 0.98)
        ]
        
        for consensus in default_consensus:
            await conn.execute("""
                INSERT INTO sde_consensus_optimization 
                (market_regime, volatility_level, min_agreeing_heads, min_probability_threshold,
                 consensus_weight_head_a, consensus_weight_head_b, consensus_weight_head_c,
                 consensus_weight_head_d, direction_agreement_required, confidence_threshold)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                ON CONFLICT DO NOTHING
            """, *consensus)
        
        logger.info("✅ Inserted default consensus optimization")
        
        # Create indexes for performance
        indexes = [
            ("idx_calibration_history_symbol_time", "sde_calibration_history", "(symbol, timeframe, calibration_date DESC)"),
            ("idx_calibration_history_date", "sde_calibration_history", "(calibration_date DESC)"),
            ("idx_dynamic_thresholds_regime", "sde_dynamic_thresholds", "(market_regime, volatility_level)"),
            ("idx_feature_importance_symbol", "sde_feature_importance", "(symbol, timeframe, feature_type)"),
            ("idx_feature_importance_model", "sde_feature_importance", "(model_head, feature_type)"),
            ("idx_accuracy_performance_symbol", "sde_accuracy_performance", "(symbol, timeframe, period_start)"),
            ("idx_accuracy_performance_regime", "sde_accuracy_performance", "(market_regime, volatility_level)"),
            ("idx_consensus_optimization_regime", "sde_consensus_optimization", "(market_regime, volatility_level)")
        ]
        
        for index_name, table_name, columns in indexes:
            try:
                await conn.execute(f"CREATE INDEX IF NOT EXISTS {index_name} ON {table_name} {columns}")
                logger.info(f"✅ Created index: {index_name}")
            except Exception as e:
                logger.warning(f"⚠️ Index {index_name} already exists or failed: {e}")
        
        # Verify table creation
        tables = [
            'sde_calibration_history',
            'sde_dynamic_thresholds', 
            'sde_feature_importance',
            'sde_accuracy_performance',
            'sde_consensus_optimization'
        ]
        
        for table in tables:
            result = await conn.fetchval(f"SELECT COUNT(*) FROM {table}")
            logger.info(f"✅ Table {table}: {result} rows")
        
        await conn.close()
        logger.info("✅ Phase 8 migration completed successfully")
        
    except Exception as e:
        logger.error(f"❌ Migration failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(run_migration())
