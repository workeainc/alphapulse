"""
Database Setup Script for AlphaPulse Trading Bot
Phase 1 Implementation - TimescaleDB setup for time-series data
"""

import psycopg2
import psycopg2.extras
import logging
from datetime import datetime
import os
from typing import Optional

logger = logging.getLogger(__name__)

class TimescaleDBSetup:
    """Setup and configure TimescaleDB for AlphaPulse trading bot"""
    
    def __init__(self, host: str = "localhost", port: int = 5432, 
                 database: str = "alphapulse", username: str = "postgres", 
                 password: str = "Emon_@17711"):
        self.host = host
        self.port = port
        self.database = database
        self.username = username
        self.password = password
        self.connection = None
    
    def connect(self) -> bool:
        """Establish connection to TimescaleDB"""
        try:
            self.connection = psycopg2.connect(
                host=self.host,
                port=self.port,
                database=self.database,
                user=self.username,
                password=self.password
            )
            self.connection.autocommit = True
            logger.info("✅ Connected to TimescaleDB successfully")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to connect to TimescaleDB: {e}")
            return False
    
    def disconnect(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()
            logger.info("🔌 Disconnected from TimescaleDB")
    
    def create_database(self) -> bool:
        """Create the database if it doesn't exist"""
        try:
            # Connect to default postgres database first
            conn = psycopg2.connect(
                host=self.host,
                port=self.port,
                database="postgres",
                user=self.username,
                password=self.password
            )
            conn.autocommit = True
            cursor = conn.cursor()
            
            # Check if database exists
            cursor.execute("SELECT 1 FROM pg_database WHERE datname = %s", (self.database,))
            exists = cursor.fetchone()
            
            if not exists:
                cursor.execute(f"CREATE DATABASE {self.database}")
                logger.info(f"✅ Created database: {self.database}")
            else:
                logger.info(f"ℹ️ Database {self.database} already exists")
            
            cursor.close()
            conn.close()
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to create database: {e}")
            return False
    
    def create_user(self) -> bool:
        """Create the alpha_emon user if it doesn't exist"""
        try:
            # Connect to the alphapulse database
            if not self.connection:
                if not self.connect():
                    return False
            
            cursor = self.connection.cursor()
            
            # Check if user exists
            cursor.execute("SELECT 1 FROM pg_user WHERE usename = 'alpha_emon'")
            exists = cursor.fetchone()
            
            if not exists:
                # Create the user with password
                cursor.execute("CREATE USER alpha_emon WITH PASSWORD 'Emon17711'")
                logger.info("✅ Created user: alpha_emon")
            else:
                logger.info("ℹ️ User alpha_emon already exists")
            
            # Grant necessary privileges
            cursor.execute("GRANT ALL PRIVILEGES ON DATABASE alphapulse TO alpha_emon")
            cursor.execute("GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO alpha_emon")
            cursor.execute("GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO alpha_emon")
            cursor.execute("ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO alpha_emon")
            cursor.execute("ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO alpha_emon")
            
            logger.info("✅ Granted privileges to alpha_emon user")
            cursor.close()
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to create user: {e}")
            return False
    
    def setup_timescale_extension(self) -> bool:
        """Enable TimescaleDB extension"""
        try:
            if not self.connection:
                if not self.connect():
                    return False
            
            cursor = self.connection.cursor()
            
            # Enable TimescaleDB extension
            cursor.execute("CREATE EXTENSION IF NOT EXISTS timescaledb")
            logger.info("✅ TimescaleDB extension enabled")
            
            cursor.close()
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to setup TimescaleDB extension: {e}")
            return False
    
    def create_tables(self) -> bool:
        """Create all necessary tables for the trading bot"""
        try:
            if not self.connection:
                if not self.connect():
                    return False
            
            cursor = self.connection.cursor()
            
            # Drop existing tables to recreate with correct schema
            cursor.execute("DROP TABLE IF EXISTS candlestick_patterns CASCADE")
            cursor.execute("DROP TABLE IF EXISTS trading_signals CASCADE")
            cursor.execute("DROP TABLE IF EXISTS portfolio CASCADE")
            cursor.execute("DROP TABLE IF EXISTS trades CASCADE")
            cursor.execute("DROP TABLE IF EXISTS candlesticks CASCADE")
            cursor.execute("DROP TABLE IF EXISTS technical_indicators CASCADE")
            logger.info("✅ Dropped existing tables")
            
            # 1. Instruments table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS instruments (
                    instrument_id SERIAL PRIMARY KEY,
                    symbol VARCHAR(20) NOT NULL UNIQUE,
                    exchange VARCHAR(50) NOT NULL,
                    base_asset VARCHAR(20) NOT NULL,
                    quote_asset VARCHAR(20) NOT NULL,
                    is_active BOOLEAN DEFAULT true,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    updated_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)
            
            # 2. Candlestick data table (will be converted to hypertable)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS candlesticks (
                    id SERIAL,
                    timestamp TIMESTAMPTZ NOT NULL,
                    instrument_id INTEGER NOT NULL,
                    open DECIMAL(20,8) NOT NULL,
                    high DECIMAL(20,8) NOT NULL,
                    low DECIMAL(20,8) NOT NULL,
                    close DECIMAL(20,8) NOT NULL,
                    volume DECIMAL(20,8) NOT NULL,
                    interval VARCHAR(10) NOT NULL,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    PRIMARY KEY (id, timestamp)
                )
            """)
            
            # 3. Technical indicators table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS technical_indicators (
                    id SERIAL,
                    timestamp TIMESTAMPTZ NOT NULL,
                    instrument_id INTEGER NOT NULL,
                    interval VARCHAR(10) NOT NULL,
                    indicator_name VARCHAR(50) NOT NULL,
                    indicator_value DECIMAL(20,8) NOT NULL,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    PRIMARY KEY (id, timestamp)
                )
            """)
            
            # 4. Candlestick patterns table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS candlestick_patterns (
                    pattern_id SERIAL,
                    timestamp TIMESTAMPTZ NOT NULL,
                    instrument_id INTEGER NOT NULL,
                    interval VARCHAR(10) NOT NULL,
                    pattern_name VARCHAR(100) NOT NULL,
                    confidence DECIMAL(5,4) NOT NULL,
                    pattern_data JSONB,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    PRIMARY KEY (pattern_id, timestamp)
                )
            """)
            
            # 5. Trading signals table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS trading_signals (
                    signal_id SERIAL,
                    timestamp TIMESTAMPTZ NOT NULL,
                    instrument_id INTEGER NOT NULL,
                    signal_type VARCHAR(10) NOT NULL,
                    strength VARCHAR(20) NOT NULL,
                    confidence DECIMAL(5,4) NOT NULL,
                    entry_price DECIMAL(20,8) NOT NULL,
                    stop_loss DECIMAL(20,8) NOT NULL,
                    take_profit DECIMAL(20,8) NOT NULL,
                    position_size DECIMAL(20,8) NOT NULL,
                    reasoning TEXT[],
                    patterns_detected TEXT[],
                    indicators_confirming TEXT[],
                    risk_reward_ratio DECIMAL(10,4) NOT NULL,
                    is_executed BOOLEAN DEFAULT false,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    PRIMARY KEY (signal_id, timestamp)
                )
            """)
            
            # 6. Portfolio table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS portfolio (
                    portfolio_id SERIAL,
                    timestamp TIMESTAMPTZ NOT NULL,
                    total_value DECIMAL(20,8) NOT NULL,
                    cash_balance DECIMAL(20,8) NOT NULL,
                    positions_value DECIMAL(20,8) NOT NULL,
                    pnl DECIMAL(20,8) NOT NULL,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    PRIMARY KEY (portfolio_id, timestamp)
                )
            """)
            
            # 7. Trades table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    trade_id SERIAL,
                    timestamp TIMESTAMPTZ NOT NULL,
                    instrument_id INTEGER NOT NULL,
                    signal_id INTEGER,
                    trade_type VARCHAR(10) NOT NULL,
                    quantity DECIMAL(20,8) NOT NULL,
                    price DECIMAL(20,8) NOT NULL,
                    total_value DECIMAL(20,8) NOT NULL,
                    commission DECIMAL(20,8) DEFAULT 0,
                    status VARCHAR(20) DEFAULT 'pending',
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    PRIMARY KEY (trade_id, timestamp)
                )
            """)
            
            logger.info("✅ All tables created successfully")
            cursor.close()
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to create tables: {e}")
            return False
    
    def create_hypertables(self) -> bool:
        """Convert regular tables to TimescaleDB hypertables for time-series optimization"""
        try:
            if not self.connection:
                if not self.connect():
                    return False
            
            cursor = self.connection.cursor()
            
            # Convert candlesticks table to hypertable
            cursor.execute("""
                SELECT create_hypertable('candlesticks', 'timestamp', 
                                       if_not_exists => TRUE,
                                       chunk_time_interval => INTERVAL '1 day')
            """)
            logger.info("✅ Candlesticks hypertable created")
            
            # Convert technical indicators table to hypertable
            cursor.execute("""
                SELECT create_hypertable('technical_indicators', 'timestamp', 
                                       if_not_exists => TRUE,
                                       chunk_time_interval => INTERVAL '1 day')
            """)
            logger.info("✅ Technical indicators hypertable created")
            
            # Convert candlestick patterns table to hypertable
            cursor.execute("""
                SELECT create_hypertable('candlestick_patterns', 'timestamp', 
                                       if_not_exists => TRUE,
                                       chunk_time_interval => INTERVAL '1 day')
            """)
            logger.info("✅ Candlestick patterns hypertable created")
            
            # Convert trading signals table to hypertable
            cursor.execute("""
                SELECT create_hypertable('trading_signals', 'timestamp', 
                                       if_not_exists => TRUE,
                                       chunk_time_interval => INTERVAL '1 day')
            """)
            logger.info("✅ Trading signals hypertable created")
            
            # Convert portfolio table to hypertable
            cursor.execute("""
                SELECT create_hypertable('portfolio', 'timestamp', 
                                       if_not_exists => TRUE,
                                       chunk_time_interval => INTERVAL '1 day')
            """)
            logger.info("✅ Portfolio hypertable created")
            
            # Convert trades table to hypertable
            cursor.execute("""
                SELECT create_hypertable('trades', 'timestamp', 
                                       if_not_exists => TRUE,
                                       chunk_time_interval => INTERVAL '1 day')
            """)
            logger.info("✅ Trades hypertable created")
            
            cursor.close()
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to create hypertables: {e}")
            return False
    
    def create_indexes(self) -> bool:
        """Create performance indexes for faster queries"""
        try:
            if not self.connection:
                if not self.connect():
                    return False
            
            cursor = self.connection.cursor()
            
            # Indexes for candlesticks table
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_candlesticks_instrument_timestamp 
                ON candlesticks (instrument_id, timestamp DESC)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_candlesticks_interval 
                ON candlesticks (interval, timestamp DESC)
            """)
            
            # Indexes for technical indicators
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_indicators_instrument_timestamp 
                ON technical_indicators (instrument_id, timestamp DESC)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_indicators_name 
                ON technical_indicators (indicator_name, timestamp DESC)
            """)
            
            # Indexes for patterns
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_patterns_instrument_timestamp 
                ON candlestick_patterns (instrument_id, timestamp DESC)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_patterns_name 
                ON candlestick_patterns (pattern_name, timestamp DESC)
            """)
            
            # Indexes for signals
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_signals_instrument_timestamp 
                ON trading_signals (instrument_id, timestamp DESC)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_signals_type 
                ON trading_signals (signal_type, timestamp DESC)
            """)
            
            # Indexes for trades
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_trades_instrument_timestamp 
                ON trades (instrument_id, timestamp DESC)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_trades_status 
                ON trades (status, timestamp DESC)
            """)
            
            logger.info("✅ All indexes created successfully")
            cursor.close()
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to create indexes: {e}")
            return False
    
    def insert_sample_data(self) -> bool:
        """Insert sample data for testing"""
        try:
            if not self.connection:
                if not self.connect():
                    return False
            
            cursor = self.connection.cursor()
            
            # Insert sample instruments
            sample_instruments = [
                ('BTCUSDT', 'binance', 'BTC', 'USDT'),
                ('ETHUSDT', 'binance', 'ETH', 'USDT'),
                ('ADAUSDT', 'binance', 'ADA', 'USDT')
            ]
            
            for symbol, exchange, base, quote in sample_instruments:
                cursor.execute("""
                    INSERT INTO instruments (symbol, exchange, base_asset, quote_asset)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (symbol) DO NOTHING
                """, (symbol, exchange, base, quote))
            
            logger.info("✅ Sample instruments inserted")
            cursor.close()
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to insert sample data: {e}")
            return False
    
    def setup_complete(self) -> bool:
        """Run complete database setup"""
        logger.info("🚀 Starting TimescaleDB setup for AlphaPulse...")
        
        try:
            # 1. Create database
            if not self.create_database():
                return False
            
            # 2. Connect to the new database
            if not self.connect():
                return False
            
            # 3. Create alpha_emon user
            if not self.create_user():
                return False
            
            # 4. Setup TimescaleDB extension
            if not self.setup_timescale_extension():
                return False
            
            # 5. Create tables
            if not self.create_tables():
                return False
            
            # 6. Create hypertables
            if not self.create_hypertables():
                return False
            
            # 7. Create indexes
            if not self.create_indexes():
                return False
            
            # 8. Insert sample data
            if not self.insert_sample_data():
                return False
            
            logger.info("🎉 TimescaleDB setup completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Database setup failed: {e}")
            return False
        finally:
            self.disconnect()

def main():
    """Main function to run database setup"""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Get database configuration from environment or use defaults
    db_config = {
        'host': os.getenv('DB_HOST', 'localhost'),
        'port': int(os.getenv('DB_PORT', '5432')),
        'database': os.getenv('DB_NAME', 'alphapulse'),
        'username': os.getenv('DB_USER', 'postgres'),
        'password': os.getenv('DB_PASSWORD', 'Emon_@17711')
    }
    
    # Run setup
    setup = TimescaleDBSetup(**db_config)
    success = setup.setup_complete()
    
    if success:
        print("\n🎉 Database setup completed successfully!")
        print("📊 Your AlphaPulse trading bot now has optimized time-series storage!")
        print("\n📋 Next steps:")
        print("1. Test the WebSocket client")
        print("2. Test the signal generator")
        print("3. Run a test pipeline with real-time data")
    else:
        print("\n❌ Database setup failed. Check the logs above for details.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
