#!/usr/bin/env python3
"""
Fix the process_labeled_item function to handle NULL signal_id gracefully
"""

import asyncio
import logging
from sqlalchemy import text
from ..src.database.connection_simple import SimpleTimescaleDBConnection

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def fix_process_labeled_item_function():
    """Fix the process_labeled_item function to handle NULL signal_id"""
    
    try:
        db_connection = SimpleTimescaleDBConnection()
        session_factory = await db_connection.get_async_session()
        
        async with session_factory as session:
            # Create an improved version of the function that handles NULL signal_id
            await session.execute(text("""
                CREATE OR REPLACE FUNCTION process_labeled_item(
                    p_queue_id INTEGER,
                    p_manual_label VARCHAR(10),
                    p_labeled_by VARCHAR(100),
                    p_labeling_notes TEXT DEFAULT NULL
                ) RETURNS INTEGER AS $$
                DECLARE
                    retrain_id INTEGER;
                    signal_id_val INTEGER;
                BEGIN
                    -- Update the queue item
                    UPDATE active_learning_queue 
                    SET 
                        manual_label = p_manual_label,
                        labeled_by = p_labeled_by,
                        labeled_at = NOW(),
                        labeling_notes = p_labeling_notes,
                        status = 'labeled',
                        updated_at = NOW()
                    WHERE id = p_queue_id;
                    
                    -- Get the signal_id for retrain queue
                    SELECT signal_id INTO signal_id_val 
                    FROM active_learning_queue 
                    WHERE id = p_queue_id;
                    
                    -- Add to retrain queue if we have a signal_id
                    IF signal_id_val IS NOT NULL THEN
                        INSERT INTO retrain_queue (
                            signal_id, reason, priority, status
                        ) VALUES (
                            signal_id_val, 
                            'active_learning_labeled', 
                            2,  -- Medium priority for active learning
                            'pending'
                        ) RETURNING id INTO retrain_id;
                        
                        -- Update the queue item with retrain queue reference
                        UPDATE active_learning_queue 
                        SET retrain_queue_id = retrain_id, status = 'processed'
                        WHERE id = p_queue_id;
                        
                        RETURN retrain_id;
                    ELSE
                        -- If no signal_id, just mark as processed without adding to retrain queue
                        UPDATE active_learning_queue 
                        SET status = 'processed'
                        WHERE id = p_queue_id;
                        
                        RETURN 0;  -- Return 0 to indicate success but no retrain queue entry
                    END IF;
                END;
                $$ LANGUAGE plpgsql;
            """))
            
            logger.info("✅ Fixed process_labeled_item function")
            return True
                
    except Exception as e:
        logger.error(f"❌ Error fixing function: {e}")
        return False

async def main():
    """Main function"""
    logger.info("🚀 Fixing Active Learning Database Function")
    logger.info("=" * 50)
    
    success = await fix_process_labeled_item_function()
    
    if success:
        logger.info("✅ Function fix completed")
    else:
        logger.error("❌ Function fix failed")
    
    return success

if __name__ == "__main__":
    asyncio.run(main())
