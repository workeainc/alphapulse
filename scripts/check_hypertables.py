#!/usr/bin/env python3
"""
Check and Convert Hypertables
Script to check if tables are hypertables and convert them if needed
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

class HypertableChecker:
    """Check and convert hypertables"""
    
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
    
    async def check_hypertables(self):
        """Check which tables are hypertables"""
        try:
            logger.info("Checking hypertables...")
            
            # Get list of hypertables
            hypertables = await self.connection.fetch("""
                SELECT hypertable_name 
                FROM timescaledb_information.hypertables;
            """)
            
            logger.info(f"Found {len(hypertables)} hypertables:")
            for table in hypertables:
                logger.info(f"  - {table['hypertable_name']}")
            
            return [table['hypertable_name'] for table in hypertables]
            
        except Exception as e:
            logger.error(f"Failed to check hypertables: {e}")
            raise
    
    async def convert_to_hypertables(self):
        """Convert regular tables to hypertables"""
        try:
            logger.info("Converting tables to hypertables...")
            
            # Convert market_data to hypertable
            await self.connection.execute("""
                SELECT create_hypertable('market_data', 'timestamp', if_not_exists => TRUE);
            """)
            
            # Convert candlestick_data to hypertable
            await self.connection.execute("""
                SELECT create_hypertable('candlestick_data', 'timestamp', if_not_exists => TRUE);
            """)
            
            # Convert signals to hypertable (using 'ts' column)
            await self.connection.execute("""
                SELECT create_hypertable('signals', 'ts', if_not_exists => TRUE);
            """)
            
            logger.info("Tables converted to hypertables successfully")
            
        except Exception as e:
            logger.error(f"Failed to convert to hypertables: {e}")
            raise
    
    async def cleanup(self):
        """Clean up resources"""
        if self.connection:
            await self.connection.close()

async def main():
    """Main function to check and convert hypertables"""
    checker = HypertableChecker()
    
    try:
        await checker.initialize()
        
        # Check existing hypertables
        existing_hypertables = await checker.check_hypertables()
        
        # Convert tables to hypertables if needed
        tables_to_convert = ['market_data', 'candlestick_data', 'signals']
        for table in tables_to_convert:
            if table not in existing_hypertables:
                logger.info(f"Converting {table} to hypertable...")
                await checker.convert_to_hypertables()
                break
        else:
            logger.info("All tables are already hypertables")
        
        logger.info("Hypertable check and conversion completed successfully!")
        
    except Exception as e:
        logger.error(f"Hypertable check failed: {e}")
        sys.exit(1)
    
    finally:
        await checker.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
