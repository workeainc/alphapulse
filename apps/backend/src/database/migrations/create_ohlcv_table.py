#!/usr/bin/env python3
"""
Create OHLCV table for ML auto-retraining system
"""

import psycopg2
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Database configuration
DB_CONFIG = {
    'host': 'localhost',
    'database': 'alphapulse',
    'user': 'alpha_emon',
    'password': 'Emon_@17711',
    'port': 5432
}

def create_ohlcv_table():
    """Create OHLCV table for market data"""
    
    logger.info("🔌 Connecting to database...")
    
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        
        logger.info("✅ Database connection established")
        
        # Create OHLCV table
        logger.info("📝 Creating ohlcv table...")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ohlcv (
                time TIMESTAMPTZ NOT NULL,
                symbol VARCHAR(20) NOT NULL,
                open DECIMAL(18,8) NOT NULL,
                high DECIMAL(18,8) NOT NULL,
                low DECIMAL(18,8) NOT NULL,
                close DECIMAL(18,8) NOT NULL,
                volume DECIMAL(18,8) NOT NULL,
                created_at TIMESTAMPTZ DEFAULT NOW(),
                PRIMARY KEY (time, symbol)
            )
        """)
        
        # Convert to TimescaleDB hypertable
        logger.info("🔄 Converting ohlcv to TimescaleDB hypertable...")
        try:
            cursor.execute("SELECT create_hypertable('ohlcv', 'time', if_not_exists => TRUE)")
            logger.info("✅ Converted ohlcv to hypertable")
        except Exception as e:
            logger.warning(f"⚠️ ohlcv hypertable creation: {e}")
        
        # Create indexes
        logger.info("📊 Creating performance indexes...")
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_ohlcv_symbol_time 
            ON ohlcv(symbol, time DESC)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_ohlcv_time_desc 
            ON ohlcv(time DESC)
        """)
        
        # Commit changes
        conn.commit()
        
        logger.info("✅ OHLCV table created and configured successfully")
        
        # Verify table creation
        cursor.execute("SELECT COUNT(*) FROM ohlcv")
        count = cursor.fetchone()[0]
        logger.info(f"📖 OHLCV table: {count} records")
        
    except Exception as e:
        logger.error(f"❌ OHLCV table creation failed: {e}")
        if conn:
            conn.rollback()
        raise
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()
            logger.info("✅ Database connection closed")

if __name__ == "__main__":
    create_ohlcv_table()
