#!/usr/bin/env python3
"""
Migration: Add Missing Phase 1 Columns
Add missing Phase 1 columns to enhanced_market_intelligence table
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

async def add_missing_phase1_columns():
    """Add missing Phase 1 columns to enhanced_market_intelligence table"""
    
    # Add missing Phase 1 columns
    add_missing_columns = """
    ALTER TABLE enhanced_market_intelligence 
    ADD COLUMN IF NOT EXISTS sector_rotation_strength NUMERIC(4,3),
    ADD COLUMN IF NOT EXISTS capital_flow_heatmap JSONB,
    ADD COLUMN IF NOT EXISTS sector_performance_ranking JSONB,
    ADD COLUMN IF NOT EXISTS rotation_confidence NUMERIC(4,3),
    ADD COLUMN IF NOT EXISTS weighted_coin_sentiment JSONB,
    ADD COLUMN IF NOT EXISTS whale_sentiment_proxy NUMERIC(4,3),
    ADD COLUMN IF NOT EXISTS sentiment_divergence_score NUMERIC(4,3),
    ADD COLUMN IF NOT EXISTS multi_timeframe_sentiment JSONB,
    ADD COLUMN IF NOT EXISTS ensemble_model_weights JSONB;
    """
    
    try:
        # Connect to database
        conn = await asyncpg.connect(**DB_CONFIG)
        logger.info("✅ Connected to database successfully")
        
        # Add missing columns
        logger.info("Adding missing Phase 1 columns to enhanced_market_intelligence table...")
        await conn.execute(add_missing_columns)
        logger.info("✅ Missing Phase 1 columns added to enhanced_market_intelligence table")
        
        await conn.close()
        logger.info("✅ Missing Phase 1 columns migration completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Error adding missing Phase 1 columns: {e}")
        raise

async def main():
    """Main migration function"""
    logger.info("🚀 Starting Missing Phase 1 Columns Migration...")
    await add_missing_phase1_columns()
    logger.info("✅ Missing Phase 1 Columns Migration completed!")

if __name__ == "__main__":
    asyncio.run(main())
