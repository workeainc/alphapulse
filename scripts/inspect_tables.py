#!/usr/bin/env python3
"""
Inspect Table Structure
Script to inspect the structure of existing tables
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

class TableInspector:
    """Inspect table structure"""
    
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
    
    async def inspect_table_structure(self, table_name):
        """Inspect the structure of a specific table"""
        try:
            logger.info(f"Inspecting table structure for: {table_name}")
            
            # Get column information
            columns = await self.connection.fetch("""
                SELECT 
                    column_name,
                    data_type,
                    is_nullable,
                    column_default
                FROM information_schema.columns 
                WHERE table_name = $1 
                AND table_schema = 'public'
                ORDER BY ordinal_position;
            """, table_name)
            
            logger.info(f"Columns in {table_name}:")
            for column in columns:
                logger.info(f"  - {column['column_name']}: {column['data_type']} (nullable: {column['is_nullable']})")
            
            # Get sample data
            sample_data = await self.connection.fetch(f"""
                SELECT * FROM {table_name} LIMIT 3;
            """)
            
            if sample_data:
                logger.info(f"Sample data from {table_name}:")
                for row in sample_data:
                    logger.info(f"  - {dict(row)}")
            else:
                logger.info(f"No data found in {table_name}")
            
            return columns
            
        except Exception as e:
            logger.error(f"Failed to inspect table {table_name}: {e}")
            raise
    
    async def cleanup(self):
        """Clean up resources"""
        if self.connection:
            await self.connection.close()

async def main():
    """Main function to inspect tables"""
    inspector = TableInspector()
    
    try:
        await inspector.initialize()
        
        # Inspect key tables
        tables_to_inspect = ['market_data', 'candlestick_data', 'signals']
        
        for table_name in tables_to_inspect:
            await inspector.inspect_table_structure(table_name)
            logger.info("-" * 50)
        
        logger.info("Table inspection completed successfully!")
        
    except Exception as e:
        logger.error(f"Table inspection failed: {e}")
        sys.exit(1)
    
    finally:
        await inspector.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
