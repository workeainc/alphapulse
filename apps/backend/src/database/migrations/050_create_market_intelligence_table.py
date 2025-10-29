#!/usr/bin/env python3
"""
Migration: Create Market Intelligence Table
Basic table for storing market intelligence data
"""

import asyncio
import logging
import os
import asyncpg
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set environment variable for psql authentication
os.environ['PGPASSWORD'] = 'Emon_@17711'

# Database connection parameters
DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'database': 'alphapulse',
    'user': 'alpha_emon',
    'password': 'Emon_@17711'
}

async def create_market_intelligence_table():
    """Create market intelligence table"""
    
    market_intelligence_table = """
    CREATE TABLE IF NOT EXISTS market_intelligence (
        id SERIAL,
        timestamp TIMESTAMPTZ NOT NULL,
        btc_dominance NUMERIC(10,4),
        total2_value NUMERIC(20,8),
        total3_value NUMERIC(20,8),
        market_sentiment_score NUMERIC(4,3),
        news_sentiment_score NUMERIC(4,3),
        volume_positioning_score NUMERIC(4,3),
        fear_greed_index INTEGER,
        market_regime VARCHAR(50),
        volatility_index NUMERIC(6,4),
        trend_strength NUMERIC(4,3),
        created_at TIMESTAMPTZ DEFAULT NOW()
    );
    """
    
    try:
        # Connect to database
        conn = await asyncpg.connect(**DB_CONFIG)
        
        # Create table
        await conn.execute(market_intelligence_table)
        logger.info("✅ Market intelligence table created successfully")
        
        # Create TimescaleDB hypertable (skip if already exists)
        try:
            await conn.execute(
                "SELECT create_hypertable('market_intelligence', 'timestamp', if_not_exists => TRUE, chunk_time_interval => INTERVAL '1 hour');"
            )
            logger.info("✅ Market intelligence hypertable created successfully")
        except Exception as e:
            logger.warning(f"⚠️ Hypertable creation skipped (may already exist): {e}")
        
        # Create indexes (non-unique for TimescaleDB)
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_market_intelligence_timestamp ON market_intelligence (timestamp DESC);"
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_market_intelligence_regime ON market_intelligence (market_regime, timestamp DESC);"
        )
        logger.info("✅ Market intelligence indexes created successfully")
        
        await conn.close()
        return True
        
    except Exception as e:
        logger.error(f"❌ Error creating market intelligence table: {e}")
        return False

async def main():
    """Main function to run the migration"""
    logger.info("🚀 Starting market intelligence table creation...")
    
    success = await create_market_intelligence_table()
    
    if success:
        logger.info("✅ Market intelligence table creation completed successfully")
    else:
        logger.error("❌ Market intelligence table creation failed")
    
    return success

if __name__ == "__main__":
    asyncio.run(main())
