#!/usr/bin/env python3
"""
Check and Create Base Tables
Script to check existing tables and create base tables if needed
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

class TableChecker:
    """Check and create base tables"""
    
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
    
    async def check_existing_tables(self):
        """Check what tables exist in the database"""
        try:
            logger.info("Checking existing tables...")
            
            # Get list of tables
            tables = await self.connection.fetch("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public' 
                ORDER BY table_name;
            """)
            
            logger.info(f"Found {len(tables)} existing tables:")
            for table in tables:
                logger.info(f"  - {table['table_name']}")
            
            return [table['table_name'] for table in tables]
            
        except Exception as e:
            logger.error(f"Failed to check existing tables: {e}")
            raise
    
    async def create_base_tables(self):
        """Create base tables if they don't exist"""
        try:
            logger.info("Creating base tables...")
            
            # Create market_data table (without SERIAL primary key for TimescaleDB)
            await self.connection.execute("""
                CREATE TABLE IF NOT EXISTS market_data (
                    symbol VARCHAR(20) NOT NULL,
                    timestamp TIMESTAMPTZ NOT NULL,
                    open NUMERIC(20, 8) NOT NULL,
                    high NUMERIC(20, 8) NOT NULL,
                    low NUMERIC(20, 8) NOT NULL,
                    close NUMERIC(20, 8) NOT NULL,
                    volume NUMERIC(20, 8) NOT NULL,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    PRIMARY KEY (symbol, timestamp)
                );
            """)
            
            # Create candlestick_data table (without SERIAL primary key for TimescaleDB)
            await self.connection.execute("""
                CREATE TABLE IF NOT EXISTS candlestick_data (
                    symbol VARCHAR(20) NOT NULL,
                    timestamp TIMESTAMPTZ NOT NULL,
                    indicators JSONB,
                    patterns JSONB,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    PRIMARY KEY (symbol, timestamp)
                );
            """)
            
            # Create signals table (without SERIAL primary key for TimescaleDB)
            await self.connection.execute("""
                CREATE TABLE IF NOT EXISTS signals (
                    symbol VARCHAR(20) NOT NULL,
                    signal_type VARCHAR(50) NOT NULL,
                    confidence FLOAT NOT NULL,
                    timestamp TIMESTAMPTZ NOT NULL,
                    features JSONB,
                    outcome VARCHAR(20),
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    PRIMARY KEY (symbol, timestamp)
                );
            """)
            
            # Create TimescaleDB hypertables
            await self.connection.execute("""
                SELECT create_hypertable('market_data', 'timestamp', if_not_exists => TRUE);
            """)
            
            await self.connection.execute("""
                SELECT create_hypertable('candlestick_data', 'timestamp', if_not_exists => TRUE);
            """)
            
            await self.connection.execute("""
                SELECT create_hypertable('signals', 'timestamp', if_not_exists => TRUE);
            """)
            
            # Create basic indexes
            await self.connection.execute("""
                CREATE INDEX IF NOT EXISTS idx_market_data_symbol_time 
                ON market_data (symbol, timestamp DESC);
            """)
            
            await self.connection.execute("""
                CREATE INDEX IF NOT EXISTS idx_candlestick_data_symbol_time 
                ON candlestick_data (symbol, timestamp DESC);
            """)
            
            await self.connection.execute("""
                CREATE INDEX IF NOT EXISTS idx_signals_symbol_time 
                ON signals (symbol, timestamp DESC);
            """)
            
            logger.info("Base tables created successfully")
            
        except Exception as e:
            logger.error(f"Failed to create base tables: {e}")
            raise
    
    async def insert_sample_data(self):
        """Insert some sample data for testing"""
        try:
            logger.info("Inserting sample data...")
            
            # Insert sample market data
            await self.connection.execute("""
                INSERT INTO market_data (symbol, timestamp, open, high, low, close, volume)
                VALUES 
                ('BTCUSDT', NOW() - INTERVAL '1 hour', 45000.0, 45100.0, 44900.0, 45050.0, 1000.0),
                ('BTCUSDT', NOW() - INTERVAL '30 minutes', 45050.0, 45200.0, 45000.0, 45150.0, 1200.0),
                ('BTCUSDT', NOW(), 45150.0, 45300.0, 45100.0, 45250.0, 1500.0),
                ('ETHUSDT', NOW() - INTERVAL '1 hour', 3000.0, 3010.0, 2990.0, 3005.0, 500.0),
                ('ETHUSDT', NOW() - INTERVAL '30 minutes', 3005.0, 3020.0, 3000.0, 3015.0, 600.0),
                ('ETHUSDT', NOW(), 3015.0, 3030.0, 3010.0, 3025.0, 700.0)
                ON CONFLICT DO NOTHING;
            """)
            
            # Insert sample candlestick data
            await self.connection.execute("""
                INSERT INTO candlestick_data (symbol, timestamp, indicators, patterns)
                VALUES 
                ('BTCUSDT', NOW() - INTERVAL '1 hour', '{"rsi": 65.5, "macd": 0.5}', '{"doji": 0.8}'),
                ('BTCUSDT', NOW() - INTERVAL '30 minutes', '{"rsi": 68.2, "macd": 0.7}', '{"hammer": 0.9}'),
                ('BTCUSDT', NOW(), '{"rsi": 70.1, "macd": 0.9}', '{"bullish_engulfing": 0.85}'),
                ('ETHUSDT', NOW() - INTERVAL '1 hour', '{"rsi": 55.3, "macd": 0.2}', '{"spinning_top": 0.6}'),
                ('ETHUSDT', NOW() - INTERVAL '30 minutes', '{"rsi": 58.7, "macd": 0.4}', '{"morning_star": 0.75}'),
                ('ETHUSDT', NOW(), '{"rsi": 62.4, "macd": 0.6}', '{"three_white_soldiers": 0.8}')
                ON CONFLICT DO NOTHING;
            """)
            
            # Insert sample signals
            await self.connection.execute("""
                INSERT INTO signals (symbol, signal_type, confidence, timestamp, features, outcome)
                VALUES 
                ('BTCUSDT', 'BUY', 0.85, NOW() - INTERVAL '1 hour', '{"rsi": 65.5, "macd": 0.5}', 'win'),
                ('BTCUSDT', 'SELL', 0.72, NOW() - INTERVAL '30 minutes', '{"rsi": 68.2, "macd": 0.7}', 'loss'),
                ('BTCUSDT', 'BUY', 0.91, NOW(), '{"rsi": 70.1, "macd": 0.9}', NULL),
                ('ETHUSDT', 'BUY', 0.78, NOW() - INTERVAL '1 hour', '{"rsi": 55.3, "macd": 0.2}', 'win'),
                ('ETHUSDT', 'HOLD', 0.65, NOW() - INTERVAL '30 minutes', '{"rsi": 58.7, "macd": 0.4}', NULL),
                ('ETHUSDT', 'BUY', 0.83, NOW(), '{"rsi": 62.4, "macd": 0.6}', NULL)
                ON CONFLICT DO NOTHING;
            """)
            
            logger.info("Sample data inserted successfully")
            
        except Exception as e:
            logger.error(f"Failed to insert sample data: {e}")
            raise
    
    async def cleanup(self):
        """Clean up resources"""
        if self.connection:
            await self.connection.close()

async def main():
    """Main function to check and create tables"""
    checker = TableChecker()
    
    try:
        await checker.initialize()
        
        # Check existing tables
        existing_tables = await checker.check_existing_tables()
        
        # Create base tables if needed
        if not existing_tables or 'market_data' not in existing_tables:
            await checker.create_base_tables()
            await checker.insert_sample_data()
        else:
            logger.info("Base tables already exist, skipping creation")
        
        logger.info("Table check and creation completed successfully!")
        
    except Exception as e:
        logger.error(f"Table check failed: {e}")
        sys.exit(1)
    
    finally:
        await checker.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
