#!/usr/bin/env python3
"""
Migration: Phase 6 Database Enhancements
Real-time streaming, ML training datasets, and performance optimizations
"""

import asyncio
import logging
import asyncpg
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database connection parameters
DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'database': 'alphapulse',
    'user': 'alpha_emon',
    'password': 'Emon_@17711'
}

async def create_phase6_enhancements():
    """Create Phase 6 database enhancements"""
    
    conn = None
    try:
        # Connect to database
        conn = await asyncpg.connect(**DB_CONFIG)
        logger.info("🔗 Connected to database for Phase 6 enhancements")
        
        # Create ML training dataset table
        ml_dataset_table = """
        CREATE TABLE IF NOT EXISTS volume_analysis_ml_dataset (
            id SERIAL,
            symbol VARCHAR(20) NOT NULL,
            timeframe VARCHAR(10) NOT NULL,
            timestamp TIMESTAMPTZ NOT NULL,
            features JSONB NOT NULL,
            targets JSONB NOT NULL,
            metadata JSONB,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            PRIMARY KEY (timestamp, id)
        );
        """
        
        # Create real-time streaming views
        real_time_streaming_view = """
        CREATE MATERIALIZED VIEW IF NOT EXISTS real_time_volume_stream AS
        SELECT 
            symbol,
            timeframe,
            timestamp,
            volume_ratio,
            volume_positioning_score,
            order_book_imbalance,
            vwap,
            cumulative_volume_delta,
            volume_breakout,
            volume_pattern_type,
            volume_pattern_confidence,
            liquidity_score,
            spoofing_detected,
            whale_activity
        FROM volume_analysis
        WHERE timestamp >= NOW() - INTERVAL '1 hour'
        ORDER BY timestamp DESC;
        """
        
        # Create symbol-specific statistics table
        symbol_stats_table = """
        CREATE TABLE IF NOT EXISTS symbol_volume_statistics (
            symbol VARCHAR(20) PRIMARY KEY,
            timeframe VARCHAR(10) NOT NULL,
            avg_volume_ratio DECIMAL(6,3),
            avg_volatility DECIMAL(8,6),
            avg_liquidity_score DECIMAL(3,2),
            volume_breakout_frequency DECIMAL(5,2),
            pattern_success_rate DECIMAL(5,2),
            last_updated TIMESTAMPTZ DEFAULT NOW(),
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
        """
        
        # Create performance monitoring table
        performance_monitoring_table = """
        CREATE TABLE IF NOT EXISTS volume_analysis_performance (
            id SERIAL,
            timestamp TIMESTAMPTZ NOT NULL,
            symbol VARCHAR(20) NOT NULL,
            timeframe VARCHAR(10) NOT NULL,
            analysis_duration_ms INTEGER,
            memory_usage_mb DECIMAL(8,2),
            cpu_usage_percent DECIMAL(5,2),
            database_latency_ms INTEGER,
            streaming_latency_ms INTEGER,
            compression_ratio DECIMAL(5,2),
            created_at TIMESTAMPTZ DEFAULT NOW(),
            PRIMARY KEY (timestamp, id)
        );
        """
        
        # Execute table creation
        tables = [
            ("ML Training Dataset", ml_dataset_table),
            ("Real-time Streaming View", real_time_streaming_view),
            ("Symbol Statistics", symbol_stats_table),
            ("Performance Monitoring", performance_monitoring_table)
        ]
        
        for name, command in tables:
            try:
                logger.info(f"Creating {name}...")
                await conn.execute(command)
                logger.info(f"✅ {name} created successfully")
            except Exception as e:
                logger.warning(f"⚠️ {name} creation warning: {e}")
                continue
        
        # Create TimescaleDB hypertables for new tables
        hypertables = [
            "SELECT create_hypertable('volume_analysis_ml_dataset', 'timestamp', if_not_exists => TRUE, chunk_time_interval => INTERVAL '1 day');",
            "SELECT create_hypertable('volume_analysis_performance', 'timestamp', if_not_exists => TRUE, chunk_time_interval => INTERVAL '1 hour');"
        ]
        
        for command in hypertables:
            try:
                await conn.execute(command)
                logger.info("✅ Hypertable created successfully")
            except Exception as e:
                logger.warning(f"⚠️ Hypertable creation warning: {e}")
        
        # Create indexes for performance
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_ml_dataset_symbol_timeframe ON volume_analysis_ml_dataset (symbol, timeframe, timestamp DESC);",
            "CREATE INDEX IF NOT EXISTS idx_ml_dataset_features ON volume_analysis_ml_dataset USING GIN (features);",
            "CREATE INDEX IF NOT EXISTS idx_ml_dataset_targets ON volume_analysis_ml_dataset USING GIN (targets);",
            "CREATE INDEX IF NOT EXISTS idx_performance_symbol ON volume_analysis_performance (symbol, timestamp DESC);",
            "CREATE INDEX IF NOT EXISTS idx_symbol_stats_symbol ON symbol_volume_statistics (symbol);"
        ]
        
        for command in indexes:
            try:
                await conn.execute(command)
                logger.info("✅ Index created successfully")
            except Exception as e:
                logger.warning(f"⚠️ Index creation warning: {e}")
        
        # Enable compression on new tables
        compression_commands = [
            """
            ALTER TABLE volume_analysis_ml_dataset SET (
                timescaledb.compress,
                timescaledb.compress_segmentby = 'symbol,timeframe',
                timescaledb.compress_orderby = 'timestamp DESC'
            );
            """,
            """
            ALTER TABLE volume_analysis_performance SET (
                timescaledb.compress,
                timescaledb.compress_segmentby = 'symbol,timeframe',
                timescaledb.compress_orderby = 'timestamp DESC'
            );
            """
        ]
        
        for command in compression_commands:
            try:
                await conn.execute(command)
                logger.info("✅ Compression enabled successfully")
            except Exception as e:
                logger.warning(f"⚠️ Compression warning: {e}")
        
        # Set compression policies
        compression_policies = [
            "SELECT add_compression_policy('volume_analysis_ml_dataset', INTERVAL '7 days');",
            "SELECT add_compression_policy('volume_analysis_performance', INTERVAL '24 hours');"
        ]
        
        for command in compression_policies:
            try:
                await conn.execute(command)
                logger.info("✅ Compression policy set successfully")
            except Exception as e:
                logger.warning(f"⚠️ Compression policy warning: {e}")
        
        logger.info("✅ Phase 6 database enhancements completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Phase 6 enhancements failed: {e}")
        return False
        
    finally:
        if conn:
            await conn.close()

async def main():
    """Main migration function"""
    logger.info("🚀 Starting Phase 6 Database Enhancements...")
    
    try:
        success = await create_phase6_enhancements()
        if success:
            logger.info("✅ Phase 6 Database Enhancements completed successfully!")
        else:
            logger.error("❌ Phase 6 enhancements failed")
            
    except Exception as e:
        logger.error(f"❌ Phase 6 enhancements failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
