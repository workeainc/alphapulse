#!/usr/bin/env python3
"""
Migration: Create Candles Table
Table for storing OHLCV candlestick data
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
    'host': 'postgres',
    'port': 5432,
    'database': 'alphapulse',
    'user': 'alpha_emon',
    'password': 'Emon_@17711'
}

async def create_candles_table():
    """Create candles table"""

    candles_table = """
    CREATE TABLE IF NOT EXISTS candles (
        id SERIAL,
        ts TIMESTAMPTZ NOT NULL,
        symbol VARCHAR(50) NOT NULL,
        tf VARCHAR(10) NOT NULL,
        o NUMERIC(20,8) NOT NULL,
        h NUMERIC(20,8) NOT NULL,
        l NUMERIC(20,8) NOT NULL,
        c NUMERIC(20,8) NOT NULL,
        v NUMERIC(20,8) NOT NULL,
        created_at TIMESTAMPTZ DEFAULT NOW(),
        PRIMARY KEY (ts, symbol, tf)
    );
    """

    try:
        # Connect to database
        conn = await asyncpg.connect(**DB_CONFIG)

        # Create table
        await conn.execute(candles_table)
        logger.info("✅ Candles table created successfully")

        # Create TimescaleDB hypertable
        try:
            await conn.execute(
                "SELECT create_hypertable('candles', 'ts', if_not_exists => TRUE, chunk_time_interval => INTERVAL '1 hour');"
            )
            logger.info("✅ Candles hypertable created successfully")
        except Exception as e:
            logger.warning(f"⚠️ Hypertable creation skipped (may already exist): {e}")

        # Create indexes
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_candles_symbol_tf ON candles (symbol, tf, ts DESC);"
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_candles_ts ON candles (ts DESC);"
        )
        logger.info("✅ Candles indexes created successfully")

        # Insert some sample data for testing
        sample_data = [
            ('BTC/USDT', '1h', 45000.0, 45100.0, 44900.0, 45050.0, 1000000.0),
            ('BTC/USDT', '1h', 45050.0, 45200.0, 45000.0, 45150.0, 1200000.0),
            ('BTC/USDT', '1h', 45150.0, 45300.0, 45100.0, 45250.0, 1100000.0),
            ('ETH/USDT', '1h', 3000.0, 3010.0, 2990.0, 3005.0, 500000.0),
            ('ETH/USDT', '1h', 3005.0, 3020.0, 3000.0, 3015.0, 600000.0),
        ]
        
        for symbol, tf, o, h, l, c, v in sample_data:
            await conn.execute("""
                INSERT INTO candles (ts, symbol, tf, o, h, l, c, v)
                VALUES (NOW() - INTERVAL '1 hour' * $1, $2, $3, $4, $5, $6, $7, $8)
                ON CONFLICT (ts, symbol, tf) DO NOTHING
            """, len(sample_data) - sample_data.index((symbol, tf, o, h, l, c, v)), symbol, tf, o, h, l, c, v)
        
        logger.info("✅ Sample data inserted successfully")

        await conn.close()
        return True

    except Exception as e:
        logger.error(f"❌ Error creating candles table: {e}")
        return False

async def main():
    """Main function to run the migration"""
    logger.info("🚀 Starting candles table creation...")

    success = await create_candles_table()

    if success:
        logger.info("✅ Candles table creation completed successfully")
    else:
        logger.error("❌ Candles table creation failed")

    return success

if __name__ == "__main__":
    asyncio.run(main())
