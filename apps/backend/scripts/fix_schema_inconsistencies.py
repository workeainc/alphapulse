#!/usr/bin/env python3
"""
Database Schema Fix Script for AlphaPlus
Fixes field name inconsistencies in existing tables
"""

import asyncio
import asyncpg
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class DatabaseSchemaFixer:
    """Fixes database schema inconsistencies"""
    
    def __init__(self, db_url: str = None):
        self.db_url = db_url or "postgresql://alpha_emon:Emon_%4017711@localhost:5432/alphapulse"
        self.conn = None
    
    async def initialize(self):
        """Initialize database connection"""
        try:
            self.conn = await asyncpg.connect(self.db_url)
            logger.info("✅ Database connection established")
        except Exception as e:
            logger.error(f"❌ Database connection failed: {e}")
            raise
    
    async def close(self):
        """Close database connection"""
        if self.conn:
            await self.conn.close()
            logger.info("🔌 Database connection closed")
    
    async def fix_schema_inconsistencies(self):
        """Fix schema inconsistencies"""
        try:
            await self.initialize()
            
            # Check and fix each table
            await self.fix_psychological_levels_analysis()
            await self.fix_volume_profile_analysis()
            await self.fix_order_book_levels()
            await self.fix_market_microstructure()
            await self.fix_psychological_levels()
            await self.fix_psychological_level_interactions()
            
            logger.info("🎉 Schema inconsistencies fixed successfully!")
            
        except Exception as e:
            logger.error(f"❌ Schema fix failed: {e}")
            raise
        finally:
            await self.close()
    
    async def fix_psychological_levels_analysis(self):
        """Fix psychological_levels_analysis table"""
        logger.info("🔧 Fixing psychological_levels_analysis table...")
        
        try:
            # Check if analysis_timestamp column exists
            columns = await self.conn.fetch("""
                SELECT column_name FROM information_schema.columns 
                WHERE table_name = 'psychological_levels_analysis'
            """)
            
            column_names = [row['column_name'] for row in columns]
            
            if 'analysis_timestamp' in column_names and 'timestamp' not in column_names:
                # Rename analysis_timestamp to timestamp
                await self.conn.execute("""
                    ALTER TABLE psychological_levels_analysis 
                    RENAME COLUMN analysis_timestamp TO timestamp
                """)
                logger.info("✅ Renamed analysis_timestamp to timestamp")
            elif 'timestamp' in column_names:
                logger.info("ℹ️ timestamp column already exists")
            else:
                logger.warning("⚠️ Neither analysis_timestamp nor timestamp column found")
                
        except Exception as e:
            logger.error(f"❌ Error fixing psychological_levels_analysis: {e}")
    
    async def fix_volume_profile_analysis(self):
        """Fix volume_profile_analysis table"""
        logger.info("🔧 Fixing volume_profile_analysis table...")
        
        try:
            # Check if analysis_timestamp column exists
            columns = await self.conn.fetch("""
                SELECT column_name FROM information_schema.columns 
                WHERE table_name = 'volume_profile_analysis'
            """)
            
            column_names = [row['column_name'] for row in columns]
            
            if 'analysis_timestamp' in column_names and 'timestamp' not in column_names:
                # Rename analysis_timestamp to timestamp
                await self.conn.execute("""
                    ALTER TABLE volume_profile_analysis 
                    RENAME COLUMN analysis_timestamp TO timestamp
                """)
                logger.info("✅ Renamed analysis_timestamp to timestamp")
            elif 'timestamp' in column_names:
                logger.info("ℹ️ timestamp column already exists")
            else:
                logger.warning("⚠️ Neither analysis_timestamp nor timestamp column found")
                
        except Exception as e:
            logger.error(f"❌ Error fixing volume_profile_analysis: {e}")
    
    async def fix_order_book_levels(self):
        """Fix order_book_levels table"""
        logger.info("🔧 Fixing order_book_levels table...")
        
        try:
            # Check if level_timestamp column exists
            columns = await self.conn.fetch("""
                SELECT column_name FROM information_schema.columns 
                WHERE table_name = 'order_book_levels'
            """)
            
            column_names = [row['column_name'] for row in columns]
            
            if 'level_timestamp' in column_names and 'timestamp' not in column_names:
                # Rename level_timestamp to timestamp
                await self.conn.execute("""
                    ALTER TABLE order_book_levels 
                    RENAME COLUMN level_timestamp TO timestamp
                """)
                logger.info("✅ Renamed level_timestamp to timestamp")
            elif 'timestamp' in column_names:
                logger.info("ℹ️ timestamp column already exists")
            else:
                logger.warning("⚠️ Neither level_timestamp nor timestamp column found")
                
        except Exception as e:
            logger.error(f"❌ Error fixing order_book_levels: {e}")
    
    async def fix_market_microstructure(self):
        """Fix market_microstructure table"""
        logger.info("🔧 Fixing market_microstructure table...")
        
        try:
            # Check if analysis_timestamp column exists
            columns = await self.conn.fetch("""
                SELECT column_name FROM information_schema.columns 
                WHERE table_name = 'market_microstructure'
            """)
            
            column_names = [row['column_name'] for row in columns]
            
            if 'analysis_timestamp' in column_names and 'timestamp' not in column_names:
                # Rename analysis_timestamp to timestamp
                await self.conn.execute("""
                    ALTER TABLE market_microstructure 
                    RENAME COLUMN analysis_timestamp TO timestamp
                """)
                logger.info("✅ Renamed analysis_timestamp to timestamp")
            elif 'timestamp' in column_names:
                logger.info("ℹ️ timestamp column already exists")
            else:
                logger.warning("⚠️ Neither analysis_timestamp nor timestamp column found")
                
        except Exception as e:
            logger.error(f"❌ Error fixing market_microstructure: {e}")
    
    async def fix_psychological_levels(self):
        """Fix psychological_levels table"""
        logger.info("🔧 Fixing psychological_levels table...")
        
        try:
            # Check if level_timestamp column exists
            columns = await self.conn.fetch("""
                SELECT column_name FROM information_schema.columns 
                WHERE table_name = 'psychological_levels'
            """)
            
            column_names = [row['column_name'] for row in columns]
            
            if 'level_timestamp' in column_names and 'timestamp' not in column_names:
                # Rename level_timestamp to timestamp
                await self.conn.execute("""
                    ALTER TABLE psychological_levels 
                    RENAME COLUMN level_timestamp TO timestamp
                """)
                logger.info("✅ Renamed level_timestamp to timestamp")
            elif 'timestamp' in column_names:
                logger.info("ℹ️ timestamp column already exists")
            else:
                logger.warning("⚠️ Neither level_timestamp nor timestamp column found")
                
        except Exception as e:
            logger.error(f"❌ Error fixing psychological_levels: {e}")
    
    async def fix_psychological_level_interactions(self):
        """Fix psychological_level_interactions table"""
        logger.info("🔧 Fixing psychological_level_interactions table...")
        
        try:
            # Check if interaction_timestamp column exists
            columns = await self.conn.fetch("""
                SELECT column_name FROM information_schema.columns 
                WHERE table_name = 'psychological_level_interactions'
            """)
            
            column_names = [row['column_name'] for row in columns]
            
            if 'interaction_timestamp' in column_names and 'timestamp' not in column_names:
                # Rename interaction_timestamp to timestamp
                await self.conn.execute("""
                    ALTER TABLE psychological_level_interactions 
                    RENAME COLUMN interaction_timestamp TO timestamp
                """)
                logger.info("✅ Renamed interaction_timestamp to timestamp")
            elif 'timestamp' in column_names:
                logger.info("ℹ️ timestamp column already exists")
            else:
                logger.warning("⚠️ Neither interaction_timestamp nor timestamp column found")
                
        except Exception as e:
            logger.error(f"❌ Error fixing psychological_level_interactions: {e}")

async def main():
    """Main schema fix function"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    fixer = DatabaseSchemaFixer()
    await fixer.fix_schema_inconsistencies()

if __name__ == "__main__":
    asyncio.run(main())
