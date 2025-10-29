#!/usr/bin/env python3
"""
Add unique constraints to existing tables for ON CONFLICT support
"""

import asyncio
import asyncpg
import logging

logger = logging.getLogger(__name__)

async def add_unique_constraints():
    """Add unique constraints to existing tables"""
    db_url = "postgresql://alpha_emon:Emon_%4017711@localhost:5432/alphapulse"
    
    try:
        # Create database connection
        conn = await asyncpg.connect(db_url)
        logger.info("✅ Database connection established")
        
        # Add unique constraints
        constraints = [
            {
                'table': 'volume_profile_analysis',
                'constraint': 'uk_volume_profile_symbol_timeframe_timestamp',
                'columns': '(symbol, timeframe, timestamp)'
            },
            {
                'table': 'psychological_levels_analysis', 
                'constraint': 'uk_psychological_analysis_symbol_timeframe_timestamp',
                'columns': '(symbol, timeframe, timestamp)'
            },
            {
                'table': 'psychological_levels',
                'constraint': 'uk_psychological_levels_symbol_type_price_timestamp',
                'columns': '(symbol, level_type, price_level, timestamp)'
            }
        ]
        
        for constraint_info in constraints:
            try:
                # Check if constraint already exists
                check_query = """
                    SELECT 1 FROM information_schema.table_constraints 
                    WHERE constraint_name = $1 AND table_name = $2
                """
                exists = await conn.fetchval(check_query, constraint_info['constraint'], constraint_info['table'])
                
                if not exists:
                    # Add the constraint
                    alter_query = f"""
                        ALTER TABLE {constraint_info['table']} 
                        ADD CONSTRAINT {constraint_info['constraint']} 
                        UNIQUE {constraint_info['columns']}
                    """
                    await conn.execute(alter_query)
                    logger.info(f"✅ Added constraint {constraint_info['constraint']} to {constraint_info['table']}")
                else:
                    logger.info(f"ℹ️ Constraint {constraint_info['constraint']} already exists on {constraint_info['table']}")
                    
            except Exception as e:
                logger.error(f"❌ Error adding constraint {constraint_info['constraint']}: {e}")
        
        logger.info("🎉 Unique constraints migration completed")
        
    except Exception as e:
        logger.error(f"❌ Migration failed: {e}")
        raise
    finally:
        if 'conn' in locals():
            await conn.close()
            logger.info("🔌 Database connection closed")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(add_unique_constraints())
