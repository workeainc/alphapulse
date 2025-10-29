#!/usr/bin/env python3
"""
Fix unique constraint on ohlcv_data table
Creates unique index if it doesn't exist
"""

import asyncio
import logging
import asyncpg
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DB_CONFIG = {
    'host': 'localhost',
    'port': 55433,
    'database': 'alphapulse',
    'user': 'alpha_emon',
    'password': 'Emon_@17711'
}

async def fix_unique_constraint():
    """Create unique constraint/index if it doesn't exist"""
    try:
        conn = await asyncpg.connect(**DB_CONFIG)
        
        try:
            # Check if unique index exists
            index_exists = await conn.fetchval("""
                SELECT EXISTS (
                    SELECT 1 FROM pg_indexes
                    WHERE tablename = 'ohlcv_data'
                    AND indexname = 'idx_ohlcv_unique'
                )
            """)
            
            if index_exists:
                logger.info("✅ Unique index 'idx_ohlcv_unique' already exists")
                return True
            
            # Check if there's already a unique constraint
            constraint_exists = await conn.fetchval("""
                SELECT EXISTS (
                    SELECT 1 FROM pg_constraint
                    WHERE conrelid = 'ohlcv_data'::regclass
                    AND contype = 'u'
                    AND (pg_get_constraintdef(oid) LIKE '%symbol%' 
                         AND pg_get_constraintdef(oid) LIKE '%timeframe%'
                         AND pg_get_constraintdef(oid) LIKE '%timestamp%')
                )
            """)
            
            if constraint_exists:
                logger.info("✅ Unique constraint already exists")
                return True
            
            # Create unique index
            logger.info("Creating unique index on (symbol, timeframe, timestamp)...")
            
            await conn.execute("""
                CREATE UNIQUE INDEX IF NOT EXISTS idx_ohlcv_unique 
                ON ohlcv_data (symbol, timeframe, timestamp)
            """)
            
            logger.info("✅ Unique index created successfully")
            logger.info("   This prevents duplicate records and enables ON CONFLICT handling")
            
            return True
            
        finally:
            await conn.close()
            
    except Exception as e:
        logger.error(f"❌ Error creating unique constraint: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(fix_unique_constraint())
    sys.exit(0 if success else 1)

