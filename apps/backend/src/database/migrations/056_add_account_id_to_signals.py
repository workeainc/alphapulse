"""
Migration 056: Add account_id column to signals table
Add missing account_id column for SDE signal limits functionality
"""

import asyncio
import logging
import asyncpg
from datetime import datetime

logger = logging.getLogger(__name__)

async def run_migration():
    """Run the migration to add account_id to signals table"""
    try:
        # Database connection
        db_pool = await asyncpg.create_pool(
            host='localhost',
            port=5432,
            user='postgres',
            password='Emon_@17711',
            database='alphapulse'
        )
        
        async with db_pool.acquire() as conn:
            logger.info("🚀 Starting Migration 056: Add account_id to signals table")
            
            # Check if account_id column already exists
            column_exists = await conn.fetchval("""
                SELECT EXISTS (
                    SELECT 1 FROM information_schema.columns 
                    WHERE table_name = 'signals' AND column_name = 'account_id'
                )
            """)
            
            if not column_exists:
                # Add account_id column to signals table
                logger.info("📝 Adding account_id column to signals table")
                await conn.execute("""
                    ALTER TABLE signals 
                    ADD COLUMN account_id VARCHAR(50) DEFAULT 'default'
                """)
                
                # Create index on account_id for better performance
                await conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_signals_account_id 
                    ON signals(account_id)
                """)
                
                logger.info("✅ Successfully added account_id column to signals table")
            else:
                logger.info("ℹ️ account_id column already exists in signals table")
            
            # Update existing records to have a default account_id if they don't have one
            updated_count = await conn.execute("""
                UPDATE signals 
                SET account_id = 'default' 
                WHERE account_id IS NULL
            """)
            
            logger.info(f"✅ Updated {updated_count} existing signals with default account_id")
            
            # Create additional indexes for better SDE performance
            logger.info("📝 Creating additional indexes for SDE performance")
            
            # Index for signal limits queries
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_signals_account_symbol_outcome 
                ON signals(account_id, symbol, outcome)
            """)
            
            # Index for signal history queries
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_signals_symbol_timestamp 
                ON signals(symbol, ts)
            """)
            
            logger.info("✅ Successfully created additional indexes")
            
            logger.info("🎉 Migration 056 completed successfully")
            
        await db_pool.close()
        
    except Exception as e:
        logger.error(f"❌ Migration 056 failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(run_migration())
