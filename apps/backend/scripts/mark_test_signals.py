#!/usr/bin/env python3
"""
Mark existing signals as test data
Preserves all existing signals for reference while marking them as test data
"""

import asyncio
import logging
import asyncpg
from datetime import datetime
from typing import Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Database configuration (matching main.py)
DB_CONFIG = {
    'host': 'localhost',
    'port': 55433,
    'database': 'alphapulse',
    'user': 'alpha_emon',
    'password': 'Emon_@17711'
}

async def check_column_exists(conn, table_name: str, column_name: str) -> bool:
    """Check if a column exists in a table"""
    query = """
        SELECT EXISTS (
            SELECT 1
            FROM information_schema.columns
            WHERE table_name = $1 AND column_name = $2
        )
    """
    exists = await conn.fetchval(query, table_name, column_name)
    return exists

async def add_is_test_data_column(conn):
    """Add is_test_data column if it doesn't exist"""
    column_exists = await check_column_exists(conn, 'live_signals', 'is_test_data')
    
    if not column_exists:
        logger.info("Adding 'is_test_data' column to live_signals table...")
        await conn.execute("""
            ALTER TABLE live_signals
            ADD COLUMN IF NOT EXISTS is_test_data BOOLEAN DEFAULT FALSE
        """)
        logger.info("✅ Column 'is_test_data' added successfully")
    else:
        logger.info("✅ Column 'is_test_data' already exists")

async def mark_test_signals() -> Dict[str, Any]:
    """Mark all existing signals as test data"""
    
    results = {
        'total_signals': 0,
        'marked_as_test': 0,
        'already_marked': 0,
        'errors': []
    }
    
    try:
        # Create database connection
        conn = await asyncpg.connect(**DB_CONFIG)
        logger.info("✅ Connected to database")
        
        try:
            # Add column if needed
            await add_is_test_data_column(conn)
            
            # Get total count of signals
            total_count = await conn.fetchval("SELECT COUNT(*) FROM live_signals")
            results['total_signals'] = total_count or 0
            logger.info(f"Found {total_count} signals in database")
            
            if total_count == 0:
                logger.info("No signals to mark. Database is clean.")
                return results
            
            # Count already marked
            already_marked = await conn.fetchval("""
                SELECT COUNT(*) 
                FROM live_signals 
                WHERE is_test_data = TRUE
            """)
            results['already_marked'] = already_marked or 0
            
            # Mark all existing signals as test data
            # Also add a timestamp marker for when real data collection begins
            marked_count = await conn.fetchval("""
                UPDATE live_signals
                SET is_test_data = TRUE
                WHERE is_test_data IS NULL OR is_test_data = FALSE
                RETURNING COUNT(*)
            """)
            
            results['marked_as_test'] = marked_count or 0
            
            # Create a marker timestamp in a metadata table if it doesn't exist
            try:
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS system_metadata (
                        key VARCHAR(100) PRIMARY KEY,
                        value TEXT NOT NULL,
                        updated_at TIMESTAMPTZ DEFAULT NOW()
                    )
                """)
                
                await conn.execute("""
                    INSERT INTO system_metadata (key, value)
                    VALUES ('historical_data_collection_start', $1)
                    ON CONFLICT (key) DO UPDATE
                    SET value = $1, updated_at = NOW()
                """, datetime.now().isoformat())
                
                logger.info("✅ Marked historical data collection start time")
                
            except Exception as e:
                logger.warning(f"⚠️ Could not update metadata table: {e}")
            
            logger.info(f"✅ Marked {results['marked_as_test']} signals as test data")
            logger.info(f"   Total signals: {results['total_signals']}")
            logger.info(f"   Already marked: {results['already_marked']}")
            logger.info(f"   Newly marked: {results['marked_as_test']}")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Error marking test signals: {e}")
            results['errors'].append(str(e))
            raise
        finally:
            await conn.close()
            
    except Exception as e:
        logger.error(f"❌ Failed to connect to database: {e}")
        results['errors'].append(f"Connection error: {str(e)}")
        raise

async def main():
    """Main execution"""
    logger.info("=" * 80)
    logger.info("MARK TEST SIGNALS SCRIPT")
    logger.info("=" * 80)
    logger.info("This script marks all existing signals as test data")
    logger.info("Preserves all data for reference while distinguishing test from real")
    logger.info("=" * 80)
    
    try:
        results = await mark_test_signals()
        
        logger.info("\n" + "=" * 80)
        logger.info("SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Total signals: {results['total_signals']}")
        logger.info(f"Marked as test: {results['marked_as_test']}")
        logger.info(f"Already marked: {results['already_marked']}")
        
        if results['errors']:
            logger.info(f"\nErrors: {len(results['errors'])}")
            for error in results['errors']:
                logger.error(f"  - {error}")
        else:
            logger.info("\n✅ All signals successfully marked as test data!")
        
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"❌ Script failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())

