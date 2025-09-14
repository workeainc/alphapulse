#!/usr/bin/env python3
"""
Migration: Create Price Action ML Predictions Table
Table for storing price action ML predictions
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

async def create_price_action_ml_predictions_table():
    """Create price action ML predictions table"""

    price_action_ml_predictions_table = """
    CREATE TABLE IF NOT EXISTS price_action_ml_predictions (
        id SERIAL,
        timestamp TIMESTAMPTZ NOT NULL,
        symbol VARCHAR(50) NOT NULL,
        timeframe VARCHAR(10) NOT NULL,
        support_resistance_score NUMERIC(10,4),
        market_structure_score NUMERIC(10,4),
        demand_supply_score NUMERIC(10,4),
        pattern_ml_score NUMERIC(10,4),
        combined_price_action_score NUMERIC(10,4),
        price_action_confidence NUMERIC(10,4),
        nearest_support NUMERIC(20,8),
        nearest_resistance NUMERIC(20,8),
        structure_type VARCHAR(50),
        trend_alignment BOOLEAN,
        zone_type VARCHAR(50),
        breakout_probability NUMERIC(10,4),
        hold_probability NUMERIC(10,4),
        support_resistance_context TEXT,
        market_structure_context TEXT,
        demand_supply_context TEXT,
        pattern_ml_context TEXT,
        created_at TIMESTAMPTZ DEFAULT NOW(),
        PRIMARY KEY (timestamp, symbol, timeframe)
    );
    """

    try:
        # Connect to database
        conn = await asyncpg.connect(**DB_CONFIG)

        # Create table
        await conn.execute(price_action_ml_predictions_table)
        logger.info("✅ Price action ML predictions table created successfully")

        # Create TimescaleDB hypertable
        try:
            await conn.execute(
                "SELECT create_hypertable('price_action_ml_predictions', 'timestamp', if_not_exists => TRUE, chunk_time_interval => INTERVAL '1 hour');"
            )
            logger.info("✅ Price action ML predictions hypertable created successfully")
        except Exception as e:
            logger.warning(f"⚠️ Hypertable creation skipped (may already exist): {e}")

        # Create indexes
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_price_action_ml_timestamp ON price_action_ml_predictions (timestamp DESC);"
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_price_action_ml_symbol_timeframe ON price_action_ml_predictions (symbol, timeframe, timestamp DESC);"
        )
        logger.info("✅ Price action ML predictions indexes created successfully")

        await conn.close()
        return True

    except Exception as e:
        logger.error(f"❌ Error creating price action ML predictions table: {e}")
        return False

async def main():
    """Main function to run the migration"""
    logger.info("🚀 Starting price action ML predictions table creation...")

    success = await create_price_action_ml_predictions_table()

    if success:
        logger.info("✅ Price action ML predictions table creation completed successfully")
    else:
        logger.error("❌ Price action ML predictions table creation failed")

    return success

if __name__ == "__main__":
    asyncio.run(main())

