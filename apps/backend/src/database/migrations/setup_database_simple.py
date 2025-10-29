#!/usr/bin/env python3
"""
Simple Database Setup for AlphaPlus Ultra-Low Latency System
Creates basic tables without complex indexes for testing
"""

import logging
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_database_simple():
    """Setup basic database schema without complex indexes"""
    try:
        logger.info("🚀 Starting simple database setup...")
        
        conn = psycopg2.connect(
            host="localhost",
            port=5432,
            database="alphapulse",
            user="alpha_emon",
            password="Emon_@17711"
        )
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
        
        # Step 6: Create basic indexes only
        logger.info("🔍 Creating basic indexes...")
        
        # Basic indexes for patterns
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_patterns_symbol_time 
            ON ultra_low_latency_patterns (symbol, timestamp DESC);
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_patterns_confidence 
            ON ultra_low_latency_patterns (confidence DESC);
        """)
        
        # Basic indexes for signals
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_signals_symbol_time 
            ON ultra_low_latency_signals (symbol, timestamp DESC);
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_signals_confidence 
            ON ultra_low_latency_signals (confidence DESC);
        """)
        
        # Step 7: Insert initial shared memory buffer configurations
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
        
        logger.info("✅ Simple database setup completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Simple database setup failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def verify_tables():
    """Verify that all required tables exist"""
    try:
        logger.info("🔍 Verifying database tables...")
        
        conn = psycopg2.connect(
            host="localhost",
            port=5432,
            database="alphapulse",
            user="alpha_emon",
            password="Emon_@17711"
        )
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
        
        cursor.close()
        conn.close()
        
        logger.info("✅ Database verification completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Database verification failed: {e}")
        return False

def main():
    """Main function to run simple database setup"""
    try:
        # Run setup
        if setup_database_simple():
            # Verify setup
            if verify_tables():
                logger.info("🎉 Simple database setup and verification completed successfully!")
                return True
            else:
                logger.error("❌ Database verification failed")
                return False
        else:
            logger.error("❌ Database setup failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ Simple database setup failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        exit(1)
