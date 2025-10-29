"""
Migration 053: Advanced Signal Quality & Validation System (Phase 9)
Implements comprehensive signal quality validation tables and metrics
"""

import asyncio
import asyncpg
import logging
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def run_migration():
    """Execute Phase 9 migration for Advanced Signal Quality & Validation System"""
    
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
        
        # Create tables for Phase 9
        
        # 1. Signal Quality Metrics Table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS sde_signal_quality_metrics (
                id SERIAL PRIMARY KEY,
                symbol VARCHAR(20) NOT NULL,
                timeframe VARCHAR(10) NOT NULL,
                signal_id VARCHAR(100),
                confidence_score DECIMAL(5,4) NOT NULL,
                volatility_score DECIMAL(5,4) NOT NULL,
                trend_strength_score DECIMAL(5,4) NOT NULL,
                volume_confirmation_score DECIMAL(5,4) NOT NULL,
                market_regime_score DECIMAL(5,4) NOT NULL,
                overall_quality_score DECIMAL(5,4) NOT NULL,
                quality_level VARCHAR(20) NOT NULL,
                validation_passed BOOLEAN NOT NULL,
                rejection_reasons TEXT[],
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
        """)
        logger.info("✅ Created sde_signal_quality_metrics table")
        
        # 2. Validation Performance Tracking
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS sde_validation_performance (
                id SERIAL PRIMARY KEY,
                symbol VARCHAR(20) NOT NULL,
                timeframe VARCHAR(10) NOT NULL,
                period_start TIMESTAMP WITH TIME ZONE NOT NULL,
                period_end TIMESTAMP WITH TIME ZONE NOT NULL,
                total_signals INTEGER NOT NULL DEFAULT 0,
                passed_signals INTEGER NOT NULL DEFAULT 0,
                rejected_signals INTEGER NOT NULL DEFAULT 0,
                avg_quality_score DECIMAL(5,4) NOT NULL DEFAULT 0.0000,
                avg_processing_time_ms DECIMAL(8,2) NOT NULL DEFAULT 0.00,
                pass_rate DECIMAL(5,4) NOT NULL DEFAULT 0.0000,
                quality_distribution JSONB NOT NULL DEFAULT '{}',
                threshold_settings JSONB NOT NULL DEFAULT '{}',
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
        """)
        logger.info("✅ Created sde_validation_performance table")
        
        # 3. False Positive Analysis
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS sde_false_positive_analysis (
                id SERIAL PRIMARY KEY,
                symbol VARCHAR(20) NOT NULL,
                timeframe VARCHAR(10) NOT NULL,
                analysis_date DATE NOT NULL,
                total_signals INTEGER NOT NULL DEFAULT 0,
                rejected_signals INTEGER NOT NULL DEFAULT 0,
                false_positives INTEGER NOT NULL DEFAULT 0,
                accuracy_rate DECIMAL(5,4) NOT NULL DEFAULT 0.0000,
                precision_rate DECIMAL(5,4) NOT NULL DEFAULT 0.0000,
                recall_rate DECIMAL(5,4) NOT NULL DEFAULT 0.0000,
                f1_score DECIMAL(5,4) NOT NULL DEFAULT 0.0000,
                analysis_period_days INTEGER NOT NULL DEFAULT 7,
                rejection_reasons_breakdown JSONB NOT NULL DEFAULT '{}',
                quality_level_breakdown JSONB NOT NULL DEFAULT '{}',
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
        """)
        logger.info("✅ Created sde_false_positive_analysis table")
        
        # 4. Adaptive Threshold History
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS sde_adaptive_threshold_history (
                id SERIAL PRIMARY KEY,
                symbol VARCHAR(20) NOT NULL,
                timeframe VARCHAR(10) NOT NULL,
                threshold_type VARCHAR(50) NOT NULL,
                old_value DECIMAL(5,4) NOT NULL,
                new_value DECIMAL(5,4) NOT NULL,
                adjustment_reason VARCHAR(200) NOT NULL,
                performance_metrics JSONB NOT NULL DEFAULT '{}',
                adaptation_date TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
        """)
        logger.info("✅ Created sde_adaptive_threshold_history table")
        
        # 5. Market Regime Tracking
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS sde_market_regime_tracking (
                id SERIAL PRIMARY KEY,
                symbol VARCHAR(20) NOT NULL,
                timeframe VARCHAR(10) NOT NULL,
                regime_type VARCHAR(20) NOT NULL,
                regime_confidence DECIMAL(5,4) NOT NULL,
                volatility_level DECIMAL(5,4) NOT NULL,
                trend_strength DECIMAL(5,4) NOT NULL,
                volume_trend DECIMAL(5,4) NOT NULL,
                regime_duration_hours INTEGER NOT NULL DEFAULT 0,
                regime_transition_probability DECIMAL(5,4) NOT NULL DEFAULT 0.0000,
                regime_metrics JSONB NOT NULL DEFAULT '{}',
                detected_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
        """)
        logger.info("✅ Created sde_market_regime_tracking table")
        
        # Insert default validation performance records
        default_performance = [
            ('BTCUSDT', '1h', datetime.now() - timedelta(days=7), datetime.now(), 100, 75, 25, 0.78, 15.5, 0.75, '{"excellent": 10, "high": 25, "good": 30, "medium": 20, "low": 10, "poor": 5}', '{"min_confidence": 0.85, "min_quality_score": 0.70}'),
            ('ETHUSDT', '1h', datetime.now() - timedelta(days=7), datetime.now(), 95, 72, 23, 0.76, 14.8, 0.76, '{"excellent": 8, "high": 22, "good": 28, "medium": 25, "low": 8, "poor": 4}', '{"min_confidence": 0.85, "min_quality_score": 0.70}'),
            ('ADAUSDT', '1h', datetime.now() - timedelta(days=7), datetime.now(), 85, 60, 25, 0.71, 16.2, 0.71, '{"excellent": 5, "high": 18, "good": 25, "medium": 22, "low": 12, "poor": 3}', '{"min_confidence": 0.85, "min_quality_score": 0.70}')
        ]
        
        for perf in default_performance:
            await conn.execute("""
                INSERT INTO sde_validation_performance 
                (symbol, timeframe, period_start, period_end, total_signals, passed_signals,
                 rejected_signals, avg_quality_score, avg_processing_time_ms, pass_rate,
                 quality_distribution, threshold_settings)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                ON CONFLICT DO NOTHING
            """, *perf)
        
        logger.info("✅ Inserted default validation performance records")
        
        # Insert default false positive analysis
        default_false_positives = [
            ('BTCUSDT', '1h', datetime.now().date(), 100, 25, 5, 0.75, 0.80, 0.75, 0.77, 7, '{"confidence_low": 10, "quality_low": 8, "volume_low": 7}', '{"excellent": 10, "high": 25, "good": 30, "medium": 20, "low": 10, "poor": 5}'),
            ('ETHUSDT', '1h', datetime.now().date(), 95, 23, 4, 0.76, 0.82, 0.76, 0.79, 7, '{"confidence_low": 8, "quality_low": 7, "volume_low": 8}', '{"excellent": 8, "high": 22, "good": 28, "medium": 25, "low": 8, "poor": 4}'),
            ('ADAUSDT', '1h', datetime.now().date(), 85, 25, 6, 0.71, 0.76, 0.71, 0.73, 7, '{"confidence_low": 12, "quality_low": 8, "volume_low": 5}', '{"excellent": 5, "high": 18, "good": 25, "medium": 22, "low": 12, "poor": 3}')
        ]
        
        for fp in default_false_positives:
            await conn.execute("""
                INSERT INTO sde_false_positive_analysis 
                (symbol, timeframe, analysis_date, total_signals, rejected_signals,
                 false_positives, accuracy_rate, precision_rate, recall_rate, f1_score,
                 analysis_period_days, rejection_reasons_breakdown, quality_level_breakdown)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
                ON CONFLICT DO NOTHING
            """, *fp)
        
        logger.info("✅ Inserted default false positive analysis")
        
        # Insert default market regime tracking
        default_regimes = [
            ('BTCUSDT', '1h', 'bull', 0.85, 0.65, 0.78, 0.82, 24, 0.15),
            ('ETHUSDT', '1h', 'sideways', 0.72, 0.45, 0.52, 0.68, 18, 0.25),
            ('ADAUSDT', '1h', 'volatile', 0.68, 0.88, 0.45, 0.55, 12, 0.35)
        ]
        
        for regime in default_regimes:
            await conn.execute("""
                INSERT INTO sde_market_regime_tracking 
                (symbol, timeframe, regime_type, regime_confidence, volatility_level,
                 trend_strength, volume_trend, regime_duration_hours, regime_transition_probability)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                ON CONFLICT DO NOTHING
            """, *regime)
        
        logger.info("✅ Inserted default market regime tracking")
        
        # Create indexes for performance
        indexes = [
            ("idx_signal_quality_symbol_time", "sde_signal_quality_metrics", "(symbol, timeframe, created_at DESC)"),
            ("idx_signal_quality_level", "sde_signal_quality_metrics", "(quality_level, created_at DESC)"),
            ("idx_signal_quality_validation", "sde_signal_quality_metrics", "(validation_passed, created_at DESC)"),
            ("idx_validation_performance_symbol", "sde_validation_performance", "(symbol, timeframe, period_start)"),
            ("idx_false_positive_symbol_date", "sde_false_positive_analysis", "(symbol, timeframe, analysis_date)"),
            ("idx_adaptive_threshold_symbol", "sde_adaptive_threshold_history", "(symbol, timeframe, adaptation_date)"),
            ("idx_market_regime_symbol", "sde_market_regime_tracking", "(symbol, timeframe, detected_at)")
        ]
        
        for index_name, table_name, columns in indexes:
            try:
                await conn.execute(f"CREATE INDEX IF NOT EXISTS {index_name} ON {table_name} {columns}")
                logger.info(f"✅ Created index: {index_name}")
            except Exception as e:
                logger.warning(f"⚠️ Index {index_name} already exists or failed: {e}")
        
        # Verify table creation
        tables = [
            'sde_signal_quality_metrics',
            'sde_validation_performance',
            'sde_false_positive_analysis',
            'sde_adaptive_threshold_history',
            'sde_market_regime_tracking'
        ]
        
        for table in tables:
            result = await conn.fetchval(f"SELECT COUNT(*) FROM {table}")
            logger.info(f"✅ Table {table}: {result} rows")
        
        await conn.close()
        logger.info("✅ Phase 9 migration completed successfully")
        
    except Exception as e:
        logger.error(f"❌ Migration failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(run_migration())
