#!/usr/bin/env python3
"""
Manual Database Setup for AlphaPlus Ultra-Low Latency System
Bypasses Alembic compatibility issues with Python 3.13
"""

import asyncio
import logging
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from typing import Dict, Any
import json

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ManualDatabaseSetup:
    """Manual database setup for ultra-low latency system"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {
            "host": "localhost",
            "port": 5432,
            "database": "alphapulse",
            "username": "alpha_emon",
            "password": "Emon_@17711"
        }
        
    def get_connection(self):
        """Get database connection"""
        return psycopg2.connect(
            host=self.config["host"],
            port=self.config["port"],
            database=self.config["database"],
            user=self.config["username"],
            password=self.config["password"]
        )
    
    def setup_database(self):
        """Setup the complete database schema"""
        try:
            logger.info("🚀 Starting manual database setup...")
            
            conn = self.get_connection()
            conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
            cursor = conn.cursor()
            
            # Step 1: Enable TimescaleDB extension
            logger.info("📦 Enabling TimescaleDB extension...")
            cursor.execute("CREATE EXTENSION IF NOT EXISTS timescaledb;")
            
            # Step 2: Create ultra-low latency pattern detection table
            logger.info("📊 Creating ultra-low latency patterns table...")
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ultra_low_latency_patterns (
                    pattern_id UUID DEFAULT gen_random_uuid(),
                    symbol VARCHAR(20) NOT NULL,
                    timeframe VARCHAR(10) NOT NULL,
                    pattern_name VARCHAR(100) NOT NULL,
                    pattern_type VARCHAR(20) NOT NULL,
                    confidence DECIMAL(4,3) NOT NULL,
                    strength VARCHAR(20) NOT NULL,
                    timestamp TIMESTAMPTZ NOT NULL,
                    price_level DECIMAL(20,8) NOT NULL,
                    volume_confirmation BOOLEAN NOT NULL DEFAULT FALSE,
                    volume_confidence DECIMAL(4,3) NOT NULL DEFAULT 0.0,
                    trend_alignment VARCHAR(20) NOT NULL DEFAULT 'neutral',
                    detection_method VARCHAR(50) NOT NULL DEFAULT 'vectorized',
                    processing_latency_ms INTEGER,
                    metadata JSONB,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    PRIMARY KEY (timestamp, pattern_id)
                );
            """)
            
            # Convert to TimescaleDB hypertable
            cursor.execute("""
                SELECT create_hypertable('ultra_low_latency_patterns', 'timestamp', 
                    chunk_time_interval => INTERVAL '1 hour',
                    if_not_exists => TRUE
                );
            """)
            
            # Note: TimescaleDB dimensions must be integer, timestamp, or date types
            # We'll use partitioning by symbol through the primary key instead
            
            # Step 3: Create ultra-low latency signals table
            logger.info("📈 Creating ultra-low latency signals table...")
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ultra_low_latency_signals (
                    signal_id UUID DEFAULT gen_random_uuid(),
                    pattern_id UUID,
                    symbol VARCHAR(20) NOT NULL,
                    timeframe VARCHAR(10) NOT NULL,
                    signal_type VARCHAR(20) NOT NULL,
                    confidence DECIMAL(4,3) NOT NULL,
                    strength VARCHAR(20) NOT NULL,
                    entry_price DECIMAL(20,8) NOT NULL,
                    stop_loss DECIMAL(20,8),
                    take_profit DECIMAL(20,8),
                    risk_reward_ratio DECIMAL(6,2),
                    timestamp TIMESTAMPTZ NOT NULL,
                    processing_latency_ms INTEGER,
                    ensemble_score DECIMAL(4,3),
                    market_regime VARCHAR(50),
                    volatility_context DECIMAL(6,3),
                    metadata JSONB,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    PRIMARY KEY (timestamp, signal_id)
                );
            """)
            
            # Convert to TimescaleDB hypertable
            cursor.execute("""
                SELECT create_hypertable('ultra_low_latency_signals', 'timestamp', 
                    chunk_time_interval => INTERVAL '1 hour',
                    if_not_exists => TRUE
                );
            """)
            
            # Note: TimescaleDB dimensions must be integer, timestamp, or date types
            # We'll use partitioning by symbol through the primary key instead
            
            # Step 4: Create performance metrics table
            logger.info("⚡ Creating performance metrics table...")
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ultra_low_latency_performance (
                    metric_id UUID DEFAULT gen_random_uuid(),
                    component VARCHAR(50) NOT NULL,
                    metric_name VARCHAR(100) NOT NULL,
                    metric_value DECIMAL(10,3) NOT NULL,
                    metric_unit VARCHAR(20),
                    timestamp TIMESTAMPTZ NOT NULL,
                    metadata JSONB,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    PRIMARY KEY (timestamp, metric_id)
                );
            """)
            
            # Convert to TimescaleDB hypertable
            cursor.execute("""
                SELECT create_hypertable('ultra_low_latency_performance', 'timestamp', 
                    chunk_time_interval => INTERVAL '1 hour',
                    if_not_exists => TRUE
                );
            """)
            
            # Step 5: Create shared memory buffers table
            logger.info("🔄 Creating shared memory buffers table...")
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS shared_memory_buffers (
                    buffer_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    buffer_name VARCHAR(100) NOT NULL UNIQUE,
                    buffer_type VARCHAR(50) NOT NULL,
                    current_size INTEGER NOT NULL DEFAULT 0,
                    max_size INTEGER NOT NULL,
                    overflow_count INTEGER NOT NULL DEFAULT 0,
                    last_updated TIMESTAMPTZ NOT NULL,
                    status VARCHAR(20) NOT NULL DEFAULT 'active',
                    metadata JSONB,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
            """)
            
            # Step 6: Create optimized indexes
            logger.info("🔍 Creating optimized indexes...")
            
            # BRIN indexes for time-series data (no CONCURRENTLY for hypertables)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_ultra_patterns_timestamp_brin 
                ON ultra_low_latency_patterns USING BRIN (timestamp) 
                WITH (pages_per_range = 128);
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_ultra_signals_timestamp_brin 
                ON ultra_low_latency_signals USING BRIN (timestamp) 
                WITH (pages_per_range = 128);
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_ultra_performance_timestamp_brin 
                ON ultra_low_latency_performance USING BRIN (timestamp) 
                WITH (pages_per_range = 128);
            """)
            
            # Partial indexes for high-confidence patterns only
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_ultra_patterns_high_confidence 
                ON ultra_low_latency_patterns (symbol, pattern_name, timestamp DESC) 
                WHERE confidence >= 0.8;
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_ultra_signals_high_confidence 
                ON ultra_low_latency_signals (symbol, signal_type, timestamp DESC) 
                WHERE confidence >= 0.8;
            """)
            
            # Covering indexes for common queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_ultra_patterns_covering 
                ON ultra_low_latency_patterns (symbol, timestamp DESC) 
                INCLUDE (pattern_name, pattern_type, confidence, strength, price_level, detection_method);
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_ultra_signals_covering 
                ON ultra_low_latency_signals (symbol, timestamp DESC) 
                INCLUDE (signal_type, confidence, strength, entry_price, stop_loss, take_profit, ensemble_score);
            """)
            
            # GIN indexes for JSONB metadata fields
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_ultra_patterns_metadata_gin 
                ON ultra_low_latency_patterns USING GIN (metadata);
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_ultra_signals_metadata_gin 
                ON ultra_low_latency_signals USING GIN (metadata);
            """)
            
            # Composite indexes for multi-column queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_ultra_patterns_symbol_pattern_time 
                ON ultra_low_latency_patterns (symbol, pattern_name, timestamp DESC);
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_ultra_signals_symbol_type_time 
                ON ultra_low_latency_signals (symbol, signal_type, timestamp DESC);
            """)
            
            # Functional indexes for computed values
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_ultra_patterns_confidence_score 
                ON ultra_low_latency_patterns (confidence DESC, strength);
            """)
            
            # Partitioned indexes for recent data only
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_ultra_patterns_recent 
                ON ultra_low_latency_patterns (symbol, timestamp DESC) 
                WHERE timestamp >= NOW() - INTERVAL '24 hours';
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_ultra_signals_recent 
                ON ultra_low_latency_signals (symbol, timestamp DESC) 
                WHERE timestamp >= NOW() - INTERVAL '24 hours';
            """)
            
            # Step 7: Create continuous aggregates for pre-computed statistics
            logger.info("📊 Creating continuous aggregates...")
            cursor.execute("""
                CREATE MATERIALIZED VIEW IF NOT EXISTS ultra_patterns_hourly_stats
                WITH (timescaledb.continuous) AS
                SELECT 
                    time_bucket('1 hour', timestamp) AS bucket,
                    symbol,
                    pattern_name,
                    COUNT(*) as pattern_count,
                    AVG(confidence) as avg_confidence,
                    MAX(confidence) as max_confidence,
                    AVG(processing_latency_ms) as avg_latency_ms
                FROM ultra_low_latency_patterns
                GROUP BY bucket, symbol, pattern_name;
            """)
            
            cursor.execute("""
                CREATE MATERIALIZED VIEW IF NOT EXISTS ultra_signals_hourly_stats
                WITH (timescaledb.continuous) AS
                SELECT 
                    time_bucket('1 hour', timestamp) AS bucket,
                    symbol,
                    signal_type,
                    COUNT(*) as signal_count,
                    AVG(confidence) as avg_confidence,
                    AVG(ensemble_score) as avg_ensemble_score,
                    AVG(processing_latency_ms) as avg_latency_ms
                FROM ultra_low_latency_signals
                GROUP BY bucket, symbol, signal_type;
            """)
            
            # Step 8: Create compression policies for older data
            logger.info("🗜️ Setting up compression policies...")
            cursor.execute("""
                ALTER TABLE ultra_low_latency_patterns SET (
                    timescaledb.compress,
                    timescaledb.compress_segmentby = 'symbol',
                    timescaledb.compress_orderby = 'timestamp DESC'
                );
            """)
            
            cursor.execute("""
                ALTER TABLE ultra_low_latency_signals SET (
                    timescaledb.compress,
                    timescaledb.compress_segmentby = 'symbol',
                    timescaledb.compress_orderby = 'timestamp DESC'
                );
            """)
            
            # Set compression policies (compress data older than 1 day)
            cursor.execute("""
                SELECT add_compression_policy('ultra_low_latency_patterns', INTERVAL '1 day');
            """)
            
            cursor.execute("""
                SELECT add_compression_policy('ultra_low_latency_signals', INTERVAL '1 day');
            """)
            
            # Set retention policies (keep data for 30 days)
            cursor.execute("""
                SELECT add_retention_policy('ultra_low_latency_patterns', INTERVAL '30 days');
            """)
            
            cursor.execute("""
                SELECT add_retention_policy('ultra_low_latency_signals', INTERVAL '30 days');
            """)
            
            # Step 9: Insert initial shared memory buffer configurations
            logger.info("💾 Initializing shared memory buffers...")
            cursor.execute("""
                INSERT INTO shared_memory_buffers (buffer_name, buffer_type, max_size, last_updated, status) VALUES
                ('candlestick_data', 'redis_stream', 1000, NOW(), 'active'),
                ('pattern_detection', 'redis_stream', 500, NOW(), 'active'),
                ('signal_generation', 'redis_stream', 200, NOW(), 'active'),
                ('market_analysis', 'redis_stream', 1000, NOW(), 'active')
                ON CONFLICT (buffer_name) DO NOTHING;
            """)
            
            cursor.close()
            conn.close()
            
            logger.info("✅ Manual database setup completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Manual database setup failed: {e}")
            return False
    
    def verify_tables(self):
        """Verify that all required tables exist"""
        try:
            logger.info("🔍 Verifying database tables...")
            
            conn = self.get_connection()
            cursor = conn.cursor()
            
            # Check for required tables
            required_tables = [
                "ultra_low_latency_patterns",
                "ultra_low_latency_signals",
                "ultra_low_latency_performance",
                "shared_memory_buffers"
            ]
            
            for table in required_tables:
                cursor.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = %s
                    );
                """, (table,))
                
                exists = cursor.fetchone()[0]
                if not exists:
                    logger.error(f"❌ Required table {table} not found")
                    return False
                else:
                    logger.info(f"✅ Table {table} exists")
            
            # Check for hypertables
            cursor.execute("""
                SELECT hypertable_name 
                FROM timescaledb_information.hypertables 
                WHERE hypertable_name IN ('ultra_low_latency_patterns', 'ultra_low_latency_signals', 'ultra_low_latency_performance');
            """)
            
            hypertables = [row[0] for row in cursor.fetchall()]
            logger.info(f"✅ Found hypertables: {hypertables}")
            
            # Check for indexes
            cursor.execute("""
                SELECT indexname 
                FROM pg_indexes 
                WHERE tablename LIKE 'ultra_low_latency_%' 
                AND indexname LIKE 'idx_%';
            """)
            
            indexes = [row[0] for row in cursor.fetchall()]
            logger.info(f"✅ Found {len(indexes)} indexes for ultra-low latency tables")
            
            cursor.close()
            conn.close()
            
            logger.info("✅ Database verification completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Database verification failed: {e}")
            return False

def main():
    """Main function to run manual database setup"""
    try:
        # Load config from file if available
        config = None
        try:
            with open("config/deployment_config.json", "r") as f:
                config_data = json.load(f)
                config = config_data.get("database", {})
        except FileNotFoundError:
            logger.warning("Config file not found, using default settings")
        
        # Create setup instance
        setup = ManualDatabaseSetup(config)
        
        # Run setup
        if setup.setup_database():
            # Verify setup
            if setup.verify_tables():
                logger.info("🎉 Database setup and verification completed successfully!")
                return True
            else:
                logger.error("❌ Database verification failed")
                return False
        else:
            logger.error("❌ Database setup failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ Manual database setup failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        exit(1)
