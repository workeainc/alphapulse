#!/usr/bin/env python3
"""
Fix Primary Keys for TimescaleDB
Script to fix primary key constraints for hypertables
"""

import asyncio
import logging
import sys
import os
from datetime import datetime
from pathlib import Path

import asyncpg

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PrimaryKeyFixer:
    """Fix primary keys for TimescaleDB"""
    
    def __init__(self):
        # Database configuration
        self.host = 'localhost'
        self.port = 5432
        self.database = 'alphapulse'
        self.username = 'alpha_emon'
        self.password = 'Emon_@17711'
        
        self.connection = None
        
    async def initialize(self):
        """Initialize database connection"""
        try:
            logger.info("Initializing database connection...")
            
            self.connection = await asyncpg.connect(
                host=self.host,
                port=self.port,
                database=self.database,
                user=self.username,
                password=self.password
            )
            
            # Test connection
            await self.connection.execute("SELECT 1")
            
            logger.info("Database connection initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize database connection: {e}")
            raise
    
    async def fix_market_data_table(self):
        """Fix market_data table primary key"""
        try:
            logger.info("Fixing market_data table primary key...")
            
            # Drop existing primary key if it exists
            await self.connection.execute("""
                ALTER TABLE market_data DROP CONSTRAINT IF EXISTS market_data_pkey;
            """)
            
            # Add new composite primary key with timestamp
            await self.connection.execute("""
                ALTER TABLE market_data ADD PRIMARY KEY (symbol, timestamp);
            """)
            
            # Convert to hypertable
            await self.connection.execute("""
                SELECT create_hypertable('market_data', 'timestamp', if_not_exists => TRUE);
            """)
            
            logger.info("market_data table fixed successfully")
            
        except Exception as e:
            logger.error(f"Failed to fix market_data table: {e}")
            raise
    
    async def fix_candlestick_data_table(self):
        """Fix candlestick_data table primary key"""
        try:
            logger.info("Fixing candlestick_data table primary key...")
            
            # Drop existing primary key if it exists
            await self.connection.execute("""
                ALTER TABLE candlestick_data DROP CONSTRAINT IF EXISTS candlestick_data_pkey;
            """)
            
            # Add new composite primary key with timestamp
            await self.connection.execute("""
                ALTER TABLE candlestick_data ADD PRIMARY KEY (symbol, timestamp);
            """)
            
            # Convert to hypertable
            await self.connection.execute("""
                SELECT create_hypertable('candlestick_data', 'timestamp', if_not_exists => TRUE);
            """)
            
            logger.info("candlestick_data table fixed successfully")
            
        except Exception as e:
            logger.error(f"Failed to fix candlestick_data table: {e}")
            raise
    
    async def cleanup(self):
        """Clean up resources"""
        if self.connection:
            await self.connection.close()

async def main():
    """Main function to fix primary keys"""
    fixer = PrimaryKeyFixer()
    
    try:
        await fixer.initialize()
        
        # Fix primary keys for tables
        await fixer.fix_market_data_table()
        await fixer.fix_candlestick_data_table()
        
        logger.info("Primary key fixes completed successfully!")
        
    except Exception as e:
        logger.error(f"Primary key fix failed: {e}")
        sys.exit(1)
    
    finally:
        await fixer.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
