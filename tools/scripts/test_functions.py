#!/usr/bin/env python3
"""
Test Functions
Script to test and fix the query optimization functions
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

class FunctionTester:
    """Test and fix functions"""
    
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
    
    async def test_optimized_market_data_function(self):
        """Test the optimized market data function"""
        try:
            logger.info("Testing optimized market data function...")
            
            # Create the function
            await self.connection.execute("""
                CREATE OR REPLACE FUNCTION get_optimized_market_data(
                    p_symbol VARCHAR(20),
                    p_start_time TIMESTAMPTZ,
                    p_end_time TIMESTAMPTZ
                ) RETURNS TABLE (
                    symbol VARCHAR(20),
                    ts TIMESTAMPTZ,
                    open NUMERIC,
                    high NUMERIC,
                    low NUMERIC,
                    close NUMERIC,
                    volume NUMERIC
                ) AS $$
                BEGIN
                    RETURN QUERY
                    SELECT 
                        md.symbol,
                        md.timestamp as ts,
                        md.open,
                        md.high,
                        md.low,
                        md.close,
                        md.volume
                    FROM market_data md
                    WHERE md.symbol = p_symbol
                    AND md.timestamp BETWEEN p_start_time AND p_end_time
                    ORDER BY md.timestamp;
                END;
                $$ LANGUAGE plpgsql;
            """)
            
            logger.info("Optimized market data function created successfully")
            
        except Exception as e:
            logger.error(f"Failed to create optimized market data function: {e}")
            raise
    
    async def test_signals_function(self):
        """Test the signals function"""
        try:
            logger.info("Testing signals function...")
            
            # Create the function
            await self.connection.execute("""
                CREATE OR REPLACE FUNCTION get_latest_signals_with_context(
                    p_limit INTEGER DEFAULT 100
                ) RETURNS TABLE (
                    signal_id INTEGER,
                    symbol VARCHAR(20),
                    signal_type VARCHAR(50),
                    confidence FLOAT,
                    ts TIMESTAMPTZ,
                    price NUMERIC,
                    volume NUMERIC
                ) AS $$
                BEGIN
                    RETURN QUERY
                    SELECT 
                        s.id,
                        s.symbol,
                        s.direction as signal_type,
                        s.proba as confidence,
                        s.ts,
                        md.close as price,
                        md.volume
                    FROM signals s
                    LEFT JOIN market_data md ON s.symbol = md.symbol AND s.ts = md.timestamp
                    ORDER BY s.ts DESC
                    LIMIT p_limit;
                END;
                $$ LANGUAGE plpgsql;
            """)
            
            logger.info("Signals function created successfully")
            
        except Exception as e:
            logger.error(f"Failed to create signals function: {e}")
            raise
    
    async def cleanup(self):
        """Clean up resources"""
        if self.connection:
            await self.connection.close()

async def main():
    """Main function to test functions"""
    tester = FunctionTester()
    
    try:
        await tester.initialize()
        
        # Test functions
        await tester.test_optimized_market_data_function()
        await tester.test_signals_function()
        
        logger.info("Function tests completed successfully!")
        
    except Exception as e:
        logger.error(f"Function test failed: {e}")
        sys.exit(1)
    
    finally:
        await tester.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
