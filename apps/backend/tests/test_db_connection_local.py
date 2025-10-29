#!/usr/bin/env python3
"""
Test database connection with localhost
"""

import asyncio
import asyncpg
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_connection():
    """Test database connection"""
    try:
        # Connect to database
        conn = await asyncpg.connect(
            host='localhost',
            port=5432,
            database='alphapulse',
            user='alpha_emon',
            password='Emon_@17711'
        )
        
        logger.info("✅ Database connection successful!")
        
        # Test query
        result = await conn.fetchval('SELECT version()')
        logger.info(f"✅ Database version: {result}")
        
        # Check if enhanced_signals table exists
        table_exists = await conn.fetchval("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'enhanced_signals'
            )
        """)
        
        if table_exists:
            logger.info("✅ enhanced_signals table exists")
            
            # Check current columns
            columns = await conn.fetch("""
                SELECT column_name, data_type 
                FROM information_schema.columns 
                WHERE table_name = 'enhanced_signals' 
                ORDER BY column_name
            """)
            
            logger.info(f"✅ Table has {len(columns)} columns")
            
            # Check for sentiment columns
            sentiment_columns = [col for col in columns if col['column_name'].startswith('sentiment_')]
            logger.info(f"✅ Found {len(sentiment_columns)} sentiment columns")
            
        else:
            logger.warning("⚠️ enhanced_signals table does not exist")
        
        await conn.close()
        return True
        
    except Exception as e:
        logger.error(f"❌ Database connection failed: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_connection())
    if success:
        logger.info("🚀 Database connection test completed successfully!")
    else:
        logger.error("💥 Database connection test failed!")
