"""
Migration 048: Divergence Analysis Integration
Add divergence analysis tables and integrate with SDE framework
"""

import asyncio
import asyncpg
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

async def run_migration():
    """Run the divergence analysis integration migration"""
    try:
        # Database connection
        db_pool = await asyncpg.create_pool(
            host='localhost',
            port=5432,
            user='alpha_emon',
            password='Emon_@17711',
            database='alphapulse'
        )
        
        async with db_pool.acquire() as conn:
            logger.info("🚀 Starting divergence analysis integration migration...")
            
            # Create sde_divergence_analysis table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS sde_divergence_analysis (
                    id SERIAL PRIMARY KEY,
                    symbol VARCHAR(20) NOT NULL,
                    timeframe VARCHAR(10) NOT NULL,
                    
                    -- RSI Divergence
                    rsi_divergence_type VARCHAR(30),
                    rsi_confidence DECIMAL(6,4),
                    rsi_score DECIMAL(6,4),
                    rsi_price_points JSONB,
                    rsi_indicator_points JSONB,
                    rsi_correlation_score DECIMAL(6,4),
                    rsi_confirmations JSONB,
                    
                    -- MACD Divergence
                    macd_divergence_type VARCHAR(30),
                    macd_confidence DECIMAL(6,4),
                    macd_score DECIMAL(6,4),
                    macd_price_points JSONB,
                    macd_indicator_points JSONB,
                    macd_correlation_score DECIMAL(6,4),
                    macd_confirmations JSONB,
                    
                    -- Volume Divergence
                    volume_divergence_type VARCHAR(30),
                    volume_confidence DECIMAL(6,4),
                    volume_score DECIMAL(6,4),
                    volume_price_points JSONB,
                    volume_indicator_points JSONB,
                    volume_correlation_score DECIMAL(6,4),
                    volume_confirmations JSONB,
                    
                    -- Combined Divergence
                    combined_divergence_type VARCHAR(30),
                    combined_confidence DECIMAL(6,4),
                    combined_score DECIMAL(6,4),
                    combined_price_points JSONB,
                    combined_indicator_points JSONB,
                    combined_correlation_score DECIMAL(6,4),
                    combined_confirmations JSONB,
                    
                    -- Overall Metrics
                    overall_confidence DECIMAL(6,4),
                    divergence_score DECIMAL(6,4),
                    confirmation_count INTEGER,
                    
                    -- Metadata
                    analysis_timestamp TIMESTAMP DEFAULT NOW(),
                    created_at TIMESTAMP DEFAULT NOW()
                )
            """)
            logger.info("✅ Created sde_divergence_analysis table")
            
            # Create sde_divergence_config table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS sde_divergence_config (
                    id SERIAL PRIMARY KEY,
                    config_name VARCHAR(50) UNIQUE NOT NULL,
                    config_type VARCHAR(30) NOT NULL,
                    
                    -- RSI Configuration
                    rsi_period INTEGER DEFAULT 14,
                    rsi_overbought INTEGER DEFAULT 70,
                    rsi_oversold INTEGER DEFAULT 30,
                    rsi_min_swing_distance INTEGER DEFAULT 5,
                    rsi_confirmation_threshold DECIMAL(6,4) DEFAULT 0.6,
                    
                    -- MACD Configuration
                    macd_fast_period INTEGER DEFAULT 12,
                    macd_slow_period INTEGER DEFAULT 26,
                    macd_signal_period INTEGER DEFAULT 9,
                    macd_min_swing_distance INTEGER DEFAULT 5,
                    macd_confirmation_threshold DECIMAL(6,4) DEFAULT 0.6,
                    
                    -- Volume Configuration
                    volume_period INTEGER DEFAULT 20,
                    volume_min_swing_distance INTEGER DEFAULT 3,
                    volume_confirmation_threshold DECIMAL(6,4) DEFAULT 0.5,
                    volume_threshold DECIMAL(6,4) DEFAULT 1.5,
                    
                    -- Analysis Windows
                    short_term_window INTEGER DEFAULT 10,
                    medium_term_window INTEGER DEFAULT 20,
                    long_term_window INTEGER DEFAULT 50,
                    
                    -- Strength Thresholds
                    weak_threshold DECIMAL(6,4) DEFAULT 0.3,
                    moderate_threshold DECIMAL(6,4) DEFAULT 0.5,
                    strong_threshold DECIMAL(6,4) DEFAULT 0.7,
                    extreme_threshold DECIMAL(6,4) DEFAULT 0.9,
                    
                    -- Metadata
                    is_active BOOLEAN DEFAULT true,
                    created_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW()
                )
            """)
            logger.info("✅ Created sde_divergence_config table")
            
            # Create sde_divergence_performance table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS sde_divergence_performance (
                    id SERIAL PRIMARY KEY,
                    symbol VARCHAR(20) NOT NULL,
                    timeframe VARCHAR(10) NOT NULL,
                    divergence_type VARCHAR(30) NOT NULL,
                    indicator_type VARCHAR(20) NOT NULL,
                    
                    -- Performance Metrics
                    accuracy DECIMAL(6,4),
                    precision DECIMAL(6,4),
                    recall DECIMAL(6,4),
                    f1_score DECIMAL(6,4),
                    profit_factor DECIMAL(6,4),
                    win_rate DECIMAL(6,4),
                    avg_win DECIMAL(10,4),
                    avg_loss DECIMAL(10,4),
                    max_drawdown DECIMAL(6,4),
                    
                    -- Sample Size
                    total_signals INTEGER,
                    winning_signals INTEGER,
                    losing_signals INTEGER,
                    
                    -- Time Period
                    start_date DATE,
                    end_date DATE,
                    
                    -- Metadata
                    analysis_timestamp TIMESTAMP DEFAULT NOW(),
                    created_at TIMESTAMP DEFAULT NOW()
                )
            """)
            logger.info("✅ Created sde_divergence_performance table")
            
            # Create sde_divergence_signals table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS sde_divergence_signals (
                    id SERIAL PRIMARY KEY,
                    signal_id VARCHAR(50),
                    symbol VARCHAR(20) NOT NULL,
                    timeframe VARCHAR(10) NOT NULL,
                    
                    -- Divergence Signal Details
                    divergence_type VARCHAR(30) NOT NULL,
                    indicator_type VARCHAR(20) NOT NULL,
                    strength VARCHAR(20) NOT NULL,
                    confidence DECIMAL(6,4),
                    divergence_score DECIMAL(6,4),
                    
                    -- Price and Indicator Points
                    price_points JSONB,
                    indicator_points JSONB,
                    correlation_score DECIMAL(6,4),
                    confirmation_signals JSONB,
                    
                    -- Signal Integration
                    sde_integrated BOOLEAN DEFAULT false,
                    sde_confidence_boost DECIMAL(6,4),
                    sde_reasoning JSONB,
                    
                    -- Signal Status
                    status VARCHAR(20) DEFAULT 'active',
                    entry_price DECIMAL(12,4),
                    exit_price DECIMAL(12,4),
                    pnl DECIMAL(12,4),
                    
                    -- Timestamps
                    signal_timestamp TIMESTAMP,
                    entry_timestamp TIMESTAMP,
                    exit_timestamp TIMESTAMP,
                    created_at TIMESTAMP DEFAULT NOW()
                )
            """)
            logger.info("✅ Created sde_divergence_signals table")
            
            # Insert default divergence configuration
            await conn.execute("""
                INSERT INTO sde_divergence_config (
                    config_name, config_type,
                    rsi_period, rsi_overbought, rsi_oversold, rsi_min_swing_distance, rsi_confirmation_threshold,
                    macd_fast_period, macd_slow_period, macd_signal_period, macd_min_swing_distance, macd_confirmation_threshold,
                    volume_period, volume_min_swing_distance, volume_confirmation_threshold, volume_threshold,
                    short_term_window, medium_term_window, long_term_window,
                    weak_threshold, moderate_threshold, strong_threshold, extreme_threshold
                ) VALUES (
                    'default_divergence_config', 'divergence',
                    14, 70, 30, 5, 0.6,
                    12, 26, 9, 5, 0.6,
                    20, 3, 0.5, 1.5,
                    10, 20, 50,
                    0.3, 0.5, 0.7, 0.9
                ) ON CONFLICT (config_name) DO NOTHING
            """)
            logger.info("✅ Inserted default divergence configuration")
            
            # Create indexes for performance
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_sde_divergence_analysis_symbol_timeframe 
                ON sde_divergence_analysis(symbol, timeframe)
            """)
            
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_sde_divergence_analysis_timestamp 
                ON sde_divergence_analysis(analysis_timestamp)
            """)
            
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_sde_divergence_signals_symbol_timeframe 
                ON sde_divergence_signals(symbol, timeframe)
            """)
            
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_sde_divergence_signals_status 
                ON sde_divergence_signals(status)
            """)
            
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_sde_divergence_performance_symbol_timeframe 
                ON sde_divergence_performance(symbol, timeframe)
            """)
            
            logger.info("✅ Created performance indexes")
            
            logger.info("🎉 Divergence analysis integration migration completed successfully!")
            
    except Exception as e:
        logger.error(f"❌ Migration failed: {e}")
        raise
    finally:
        if 'db_pool' in locals():
            await db_pool.close()

if __name__ == "__main__":
    asyncio.run(run_migration())
