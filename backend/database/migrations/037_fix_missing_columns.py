"""
Migration: Fix Missing Database Columns
Fix missing columns that are causing errors in the system
"""

import asyncio
import asyncpg
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

async def upgrade(connection: asyncpg.Connection):
    """Upgrade database schema"""
    try:
        logger.info("🔄 Fixing missing database columns...")
        
        # Fix comprehensive_analysis table - add missing columns
        try:
            await connection.execute("""
                ALTER TABLE comprehensive_analysis 
                ADD COLUMN IF NOT EXISTS technical_confidence FLOAT DEFAULT 0.0,
                ADD COLUMN IF NOT EXISTS pattern_confidence FLOAT DEFAULT 0.0,
                ADD COLUMN IF NOT EXISTS volume_confidence FLOAT DEFAULT 0.0,
                ADD COLUMN IF NOT EXISTS sentiment_confidence FLOAT DEFAULT 0.0,
                ADD COLUMN IF NOT EXISTS market_regime_confidence FLOAT DEFAULT 0.0
            """)
            logger.info("✅ Added missing columns to comprehensive_analysis table")
        except Exception as e:
            logger.warning(f"⚠️ Could not add columns to comprehensive_analysis: {e}")
        
        # Fix volume_analysis table - ensure timeframe column exists and is not null
        try:
            # First check if timeframe column exists
            result = await connection.fetchval("""
                SELECT COUNT(*) FROM information_schema.columns 
                WHERE table_name = 'volume_analysis' AND column_name = 'timeframe'
            """)
            
            if result == 0:
                # Add timeframe column if it doesn't exist
                await connection.execute("""
                    ALTER TABLE volume_analysis 
                    ADD COLUMN timeframe VARCHAR(10) DEFAULT '1h'
                """)
                logger.info("✅ Added timeframe column to volume_analysis table")
            
            # Update existing records with null timeframe to have a default value
            await connection.execute("""
                UPDATE volume_analysis 
                SET timeframe = '1h' 
                WHERE timeframe IS NULL
            """)
            logger.info("✅ Updated null timeframe values in volume_analysis table")
            
        except Exception as e:
            logger.warning(f"⚠️ Could not fix volume_analysis timeframe: {e}")
        
        # Fix signals table - add any missing columns
        try:
            await connection.execute("""
                ALTER TABLE signals 
                ADD COLUMN IF NOT EXISTS technical_confidence FLOAT DEFAULT 0.0,
                ADD COLUMN IF NOT EXISTS pattern_confidence FLOAT DEFAULT 0.0,
                ADD COLUMN IF NOT EXISTS volume_confidence FLOAT DEFAULT 0.0,
                ADD COLUMN IF NOT EXISTS sentiment_confidence FLOAT DEFAULT 0.0,
                ADD COLUMN IF NOT EXISTS market_regime_confidence FLOAT DEFAULT 0.0,
                ADD COLUMN IF NOT EXISTS ml_model_confidence FLOAT DEFAULT 0.0,
                ADD COLUMN IF NOT EXISTS ensemble_confidence FLOAT DEFAULT 0.0
            """)
            logger.info("✅ Added missing columns to signals table")
        except Exception as e:
            logger.warning(f"⚠️ Could not add columns to signals table: {e}")
        
        # Fix any other tables that might have missing columns
        try:
            # Check and fix market_regime_data table
            await connection.execute("""
                ALTER TABLE market_regime_data 
                ADD COLUMN IF NOT EXISTS confidence_score FLOAT DEFAULT 0.0,
                ADD COLUMN IF NOT EXISTS regime_strength FLOAT DEFAULT 0.0
            """)
            logger.info("✅ Added missing columns to market_regime_data table")
        except Exception as e:
            logger.warning(f"⚠️ Could not add columns to market_regime_data: {e}")
        
        # Create indexes for better performance
        try:
            await connection.execute("""
                CREATE INDEX IF NOT EXISTS idx_comprehensive_analysis_symbol_timeframe 
                ON comprehensive_analysis(symbol, timeframe)
            """)
            
            await connection.execute("""
                CREATE INDEX IF NOT EXISTS idx_volume_analysis_symbol_timeframe 
                ON volume_analysis(symbol, timeframe)
            """)
            
            await connection.execute("""
                CREATE INDEX IF NOT EXISTS idx_signals_technical_confidence 
                ON signals(technical_confidence DESC)
            """)
            
            logger.info("✅ Created performance indexes")
        except Exception as e:
            logger.warning(f"⚠️ Could not create some indexes: {e}")
        
        logger.info("✅ Database column fixes completed successfully")
        
    except Exception as e:
        logger.error(f"❌ Error fixing database columns: {e}")
        raise

async def downgrade(connection: asyncpg.Connection):
    """Downgrade database schema"""
    try:
        logger.info("🔄 Removing added columns...")
        
        # Remove columns from comprehensive_analysis table
        await connection.execute("""
            ALTER TABLE comprehensive_analysis 
            DROP COLUMN IF EXISTS technical_confidence,
            DROP COLUMN IF EXISTS pattern_confidence,
            DROP COLUMN IF EXISTS volume_confidence,
            DROP COLUMN IF EXISTS sentiment_confidence,
            DROP COLUMN IF EXISTS market_regime_confidence
        """)
        
        # Remove columns from signals table
        await connection.execute("""
            ALTER TABLE signals 
            DROP COLUMN IF EXISTS technical_confidence,
            DROP COLUMN IF EXISTS pattern_confidence,
            DROP COLUMN IF EXISTS volume_confidence,
            DROP COLUMN IF EXISTS sentiment_confidence,
            DROP COLUMN IF EXISTS market_regime_confidence,
            DROP COLUMN IF EXISTS ml_model_confidence,
            DROP COLUMN IF EXISTS ensemble_confidence
        """)
        
        # Remove columns from market_regime_data table
        await connection.execute("""
            ALTER TABLE market_regime_data 
            DROP COLUMN IF EXISTS confidence_score,
            DROP COLUMN IF EXISTS regime_strength
        """)
        
        logger.info("✅ Database column removal completed")
        
    except Exception as e:
        logger.error(f"❌ Error removing database columns: {e}")
        raise

async def main():
    """Run migration"""
    try:
        # Connect to database
        connection = await asyncpg.connect(
            host='localhost',
            port=5432,
            user='alpha_emon',
            password='Emon_@17711',
            database='alphapulse'
        )
        
        # Run upgrade
        await upgrade(connection)
        
        # Close connection
        await connection.close()
        
        logger.info("🎉 Database column fixes completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Database column fixes failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
