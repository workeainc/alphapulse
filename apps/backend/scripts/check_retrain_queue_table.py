#!/usr/bin/env python3
"""
Check if retrain_queue table exists and its structure
"""

import asyncio
import logging
from sqlalchemy import text
from ..src.database.connection_simple import SimpleTimescaleDBConnection

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def check_retrain_queue():
    """Check if retrain_queue table exists"""
    
    try:
        db_connection = SimpleTimescaleDBConnection()
        session_factory = await db_connection.get_async_session()
        
        async with session_factory as session:
            # Check if retrain_queue table exists
            result = await session.execute(text("""
                SELECT table_name, table_type 
                FROM information_schema.tables 
                WHERE table_name = 'retrain_queue'
            """))
            
            tables = result.fetchall()
            if tables:
                logger.info("✅ retrain_queue table exists")
                
                # Get table structure
                result = await session.execute(text("""
                    SELECT column_name, data_type, is_nullable, column_default
                    FROM information_schema.columns 
                    WHERE table_name = 'retrain_queue'
                    ORDER BY ordinal_position
                """))
                
                columns = result.fetchall()
                logger.info("📋 retrain_queue table structure:")
                for col in columns:
                    logger.info(f"  - {col.column_name}: {col.data_type} (nullable: {col.is_nullable})")
                
                return True
            else:
                logger.error("❌ retrain_queue table does not exist")
                return False
                
    except Exception as e:
        logger.error(f"❌ Error checking retrain_queue table: {e}")
        return False

async def main():
    """Main function"""
    logger.info("🚀 Checking retrain_queue table")
    logger.info("=" * 40)
    
    exists = await check_retrain_queue()
    
    if exists:
        logger.info("✅ retrain_queue table is available")
    else:
        logger.error("❌ retrain_queue table is missing")
    
    return exists

if __name__ == "__main__":
    asyncio.run(main())
