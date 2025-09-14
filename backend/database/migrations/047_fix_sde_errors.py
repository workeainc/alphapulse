"""
Migration 047: Fix SDE Framework Errors
- Add missing raw_probability column to sde_signal_history
- Fix numeric precision in sde_signal_validation
- Create sde_news_blackout table if missing
"""

import asyncio
import asyncpg
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

async def run_migration():
    """Run the migration to fix SDE errors"""
    
    # Database configuration
    db_config = {
        'host': 'localhost',
        'port': 5432,
        'user': 'alpha_emon',
        'password': 'Emon_@17711',
        'database': 'alphapulse'
    }
    
    try:
        # Connect to database
        pool = await asyncpg.create_pool(**db_config)
        
        async with pool.acquire() as conn:
            logger.info("🔧 Starting SDE Error Fixes Migration")
            
            # Fix 1: Add raw_probability column to sde_signal_history
            logger.info("📝 Fix 1: Adding raw_probability column to sde_signal_history")
            try:
                await conn.execute("""
                    ALTER TABLE sde_signal_history 
                    ADD COLUMN IF NOT EXISTS raw_probability DECIMAL(5,4)
                """)
                logger.info("✅ raw_probability column added successfully")
            except Exception as e:
                logger.warning(f"⚠️ raw_probability column already exists or error: {e}")
            
            # Fix 2: Update numeric precision in sde_signal_validation
            logger.info("📝 Fix 2: Updating numeric precision in sde_signal_validation")
            try:
                # Update consensus_score precision
                await conn.execute("""
                    ALTER TABLE sde_signal_validation 
                    ALTER COLUMN consensus_score TYPE DECIMAL(6,4)
                """)
                logger.info("✅ consensus_score precision updated")
                
                # Update confluence_score precision
                await conn.execute("""
                    ALTER TABLE sde_signal_validation 
                    ALTER COLUMN confluence_score TYPE DECIMAL(6,4)
                """)
                logger.info("✅ confluence_score precision updated")
                
                # Update execution_quality_score precision
                await conn.execute("""
                    ALTER TABLE sde_signal_validation 
                    ALTER COLUMN execution_quality_score TYPE DECIMAL(6,4)
                """)
                logger.info("✅ execution_quality_score precision updated")
                
                # Update final_confidence precision
                await conn.execute("""
                    ALTER TABLE sde_signal_validation 
                    ALTER COLUMN final_confidence TYPE DECIMAL(6,4)
                """)
                logger.info("✅ final_confidence precision updated")
                
                # Update confidence_threshold precision
                await conn.execute("""
                    ALTER TABLE sde_signal_validation 
                    ALTER COLUMN confidence_threshold TYPE DECIMAL(6,4)
                """)
                logger.info("✅ confidence_threshold precision updated")
                
            except Exception as e:
                logger.error(f"❌ Error updating numeric precision: {e}")
            
            # Fix 3: Ensure sde_news_blackout table exists with correct structure
            logger.info("📝 Fix 3: Ensuring sde_news_blackout table structure")
            try:
                # Check if table exists
                table_exists = await conn.fetchval("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_schema = 'public' 
                        AND table_name = 'sde_news_blackout'
                    )
                """)
                
                if not table_exists:
                    logger.info("📝 Creating sde_news_blackout table")
                    await conn.execute("""
                        CREATE TABLE sde_news_blackout (
                            id SERIAL PRIMARY KEY,
                            symbol VARCHAR(20) NOT NULL,
                            event_type VARCHAR(50) NOT NULL,
                            event_impact VARCHAR(20) NOT NULL,
                            event_title TEXT NOT NULL,
                            event_description TEXT,
                            start_time TIMESTAMP WITH TIME ZONE NOT NULL,
                            end_time TIMESTAMP WITH TIME ZONE NOT NULL,
                            blackout_active BOOLEAN DEFAULT true,
                            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                        )
                    """)
                    logger.info("✅ sde_news_blackout table created")
                else:
                    logger.info("✅ sde_news_blackout table already exists")
                    
            except Exception as e:
                logger.error(f"❌ Error with sde_news_blackout table: {e}")
            
            # Fix 4: Update sde_model_consensus_tracking numeric precision
            logger.info("📝 Fix 4: Updating sde_model_consensus_tracking numeric precision")
            try:
                # Update probability columns precision
                await conn.execute("""
                    ALTER TABLE sde_model_consensus_tracking 
                    ALTER COLUMN head_a_probability TYPE DECIMAL(6,4)
                """)
                await conn.execute("""
                    ALTER TABLE sde_model_consensus_tracking 
                    ALTER COLUMN head_b_probability TYPE DECIMAL(6,4)
                """)
                await conn.execute("""
                    ALTER TABLE sde_model_consensus_tracking 
                    ALTER COLUMN head_c_probability TYPE DECIMAL(6,4)
                """)
                await conn.execute("""
                    ALTER TABLE sde_model_consensus_tracking 
                    ALTER COLUMN head_d_probability TYPE DECIMAL(6,4)
                """)
                await conn.execute("""
                    ALTER TABLE sde_model_consensus_tracking 
                    ALTER COLUMN consensus_probability TYPE DECIMAL(6,4)
                """)
                await conn.execute("""
                    ALTER TABLE sde_model_consensus_tracking 
                    ALTER COLUMN min_head_probability TYPE DECIMAL(6,4)
                """)
                logger.info("✅ sde_model_consensus_tracking precision updated")
                
            except Exception as e:
                logger.error(f"❌ Error updating sde_model_consensus_tracking precision: {e}")
            
            # Fix 5: Update sde_calibration_history numeric precision
            logger.info("📝 Fix 5: Updating sde_calibration_history numeric precision")
            try:
                await conn.execute("""
                    ALTER TABLE sde_calibration_history 
                    ALTER COLUMN raw_probability TYPE DECIMAL(6,4)
                """)
                await conn.execute("""
                    ALTER TABLE sde_calibration_history 
                    ALTER COLUMN calibrated_probability TYPE DECIMAL(6,4)
                """)
                await conn.execute("""
                    ALTER TABLE sde_calibration_history 
                    ALTER COLUMN reliability_score TYPE DECIMAL(6,4)
                """)
                logger.info("✅ sde_calibration_history precision updated")
                
            except Exception as e:
                logger.error(f"❌ Error updating sde_calibration_history precision: {e}")
            
            logger.info("🎉 SDE Error Fixes Migration Completed Successfully!")
            
    except Exception as e:
        logger.error(f"❌ Migration failed: {e}")
        raise
    finally:
        if 'pool' in locals():
            await pool.close()

if __name__ == "__main__":
    asyncio.run(run_migration())
