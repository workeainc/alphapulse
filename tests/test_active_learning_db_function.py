#!/usr/bin/env python3
"""
Debug script to test the process_labeled_item database function directly
"""

import asyncio
import logging
from sqlalchemy import text
from ..database.connection_simple import SimpleTimescaleDBConnection

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_db_function():
    """Test the process_labeled_item function directly"""
    
    try:
        db_connection = SimpleTimescaleDBConnection()
        session_factory = await db_connection.get_async_session()
        
        async with session_factory as session:
            # First, let's check if there are any items in the queue
            result = await session.execute(text("""
                SELECT id, signal_id, status FROM active_learning_queue 
                WHERE model_id LIKE 'test_model_%' 
                ORDER BY id DESC LIMIT 5
            """))
            
            items = result.fetchall()
            logger.info(f"Found {len(items)} test items in queue:")
            for item in items:
                logger.info(f"  ID: {item.id}, Signal ID: {item.signal_id}, Status: {item.status}")
            
            if not items:
                logger.error("❌ No test items found in queue")
                return False
            
            # Try to process the first item
            test_item_id = items[0].id
            logger.info(f"🔍 Testing process_labeled_item with queue_id={test_item_id}")
            
            result = await session.execute(text("""
                SELECT process_labeled_item(:queue_id, :manual_label, :labeled_by, :labeling_notes)
            """), {
                'queue_id': test_item_id,
                'manual_label': 'BUY',
                'labeled_by': 'test_user',
                'labeling_notes': 'Test labeling'
            })
            
            retrain_id = result.scalar()
            logger.info(f"📊 Function returned: {retrain_id}")
            
            if retrain_id:
                logger.info("✅ Database function worked correctly")
                return True
            else:
                logger.error("❌ Database function returned NULL")
                return False
                
    except Exception as e:
        logger.error(f"❌ Error testing database function: {e}")
        return False

async def main():
    """Main function"""
    logger.info("🚀 Testing Active Learning Database Function")
    logger.info("=" * 50)
    
    success = await test_db_function()
    
    if success:
        logger.info("✅ Database function test passed")
    else:
        logger.error("❌ Database function test failed")
    
    return success

if __name__ == "__main__":
    asyncio.run(main())
