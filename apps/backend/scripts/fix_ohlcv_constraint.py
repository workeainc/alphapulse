#!/usr/bin/env python3
"""
Add missing unique constraint to ohlcv_data table
"""

import asyncio
import asyncpg
import logging

logger = logging.getLogger(__name__)

async def add_ohlcv_constraint():
    """Add unique constraint to ohlcv_data table"""
    db_url = "postgresql://alpha_emon:Emon_%4017711@localhost:5432/alphapulse"
    
    try:
        # Create database connection
        conn = await asyncpg.connect(db_url)
        logger.info("✅ Database connection established")
        
        # Check if constraint already exists
        check_query = """
            SELECT 1 FROM information_schema.table_constraints 
            WHERE constraint_name = 'uk_ohlcv_symbol_timeframe_timestamp' 
            AND table_name = 'ohlcv_data'
        """
        exists = await conn.fetchval(check_query)
        
        if not exists:
            # Add the constraint
            alter_query = """
                ALTER TABLE ohlcv_data 
                ADD CONSTRAINT uk_ohlcv_symbol_timeframe_timestamp 
                UNIQUE (symbol, timeframe, timestamp)
            """
            await conn.execute(alter_query)
            logger.info("✅ Added unique constraint to ohlcv_data table")
        else:
            logger.info("ℹ️ Unique constraint already exists on ohlcv_data table")
        
        logger.info("🎉 OHLCV constraint migration completed")
        
    except Exception as e:
        logger.error(f"❌ Migration failed: {e}")
        raise
    finally:
        if 'conn' in locals():
            await conn.close()
            logger.info("🔌 Database connection closed")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(add_ohlcv_constraint())
