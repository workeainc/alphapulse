"""
Migration: Create volume_analysis table
Revision: 051_create_volume_analysis_table
"""

import asyncio
import asyncpg
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

async def create_volume_analysis_table():
    """Create volume_analysis table with TimescaleDB hypertable"""
    
    # Database connection parameters
    db_config = {
        'host': 'postgres',  # Docker service name
        'port': 5432,
        'user': 'alpha_emon',
        'password': 'Emon_@17711',
        'database': 'alphapulse'
    }
    
    try:
        # Connect to database
        conn = await asyncpg.connect(**db_config)
        logger.info("✅ Connected to database")
        
        # Create volume_analysis table
        volume_analysis_table = """
        CREATE TABLE IF NOT EXISTS volume_analysis (
            timestamp TIMESTAMPTZ NOT NULL,
            symbol VARCHAR(20) NOT NULL,
            timeframe VARCHAR(10) DEFAULT '1h',
            volume_ratio DECIMAL(8,4),
            volume_trend VARCHAR(20),
            order_book_imbalance DECIMAL(8,4),
            volume_positioning_score DECIMAL(8,4),
            buy_volume_ratio DECIMAL(8,4),
            sell_volume_ratio DECIMAL(8,4),
            volume_breakout BOOLEAN DEFAULT FALSE,
            volume_analysis TEXT,
            analysis_confidence DECIMAL(5,4) DEFAULT 0.5,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
        """
        
        await conn.execute(volume_analysis_table)
        logger.info("✅ Created volume_analysis table")
        
        # Create TimescaleDB hypertable
        try:
            await conn.execute("""
                SELECT create_hypertable('volume_analysis', 'timestamp', 
                    if_not_exists => TRUE, 
                    chunk_time_interval => INTERVAL '1 hour'
                );
            """)
            logger.info("✅ Created TimescaleDB hypertable for volume_analysis")
        except Exception as e:
            logger.warning(f"⚠️ Hypertable creation warning: {e}")
        
        # Create indexes
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_volume_analysis_symbol_timestamp ON volume_analysis (symbol, timestamp DESC);",
            "CREATE INDEX IF NOT EXISTS idx_volume_analysis_timeframe ON volume_analysis (timeframe);",
            "CREATE INDEX IF NOT EXISTS idx_volume_analysis_volume_trend ON volume_analysis (volume_trend);",
            "CREATE INDEX IF NOT EXISTS idx_volume_analysis_confidence ON volume_analysis (analysis_confidence DESC);"
        ]
        
        for index in indexes:
            await conn.execute(index)
        
        logger.info("✅ Created indexes for volume_analysis table")
        
        # Insert sample data
        sample_data = [
            (datetime.now(), 'BTC/USDT', '1h', 1.5, 'increasing', 0.2, 0.7, 0.6, 0.4, True, 'High volume with bullish positioning', 0.8),
            (datetime.now(), 'ETH/USDT', '1h', 0.8, 'stable', -0.1, 0.5, 0.5, 0.5, False, 'Normal volume conditions', 0.6),
            (datetime.now(), 'SOL/USDT', '1h', 2.1, 'increasing', 0.3, 0.8, 0.7, 0.3, True, 'Volume breakout with strong buying', 0.9)
        ]
        
        await conn.executemany("""
            INSERT INTO volume_analysis (
                timestamp, symbol, timeframe, volume_ratio, volume_trend, 
                order_book_imbalance, volume_positioning_score, buy_volume_ratio, 
                sell_volume_ratio, volume_breakout, volume_analysis, analysis_confidence
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
        """, sample_data)
        
        logger.info("✅ Inserted sample data into volume_analysis table")
        
        await conn.close()
        logger.info("✅ Volume analysis table creation completed successfully")
        
    except Exception as e:
        logger.error(f"❌ Error creating volume_analysis table: {e}")
        raise

async def main():
    """Main function to run the migration"""
    logger.info("🚀 Starting volume_analysis table creation...")
    await create_volume_analysis_table()
    logger.info("✅ Volume analysis table migration completed!")

if __name__ == "__main__":
    asyncio.run(main())
