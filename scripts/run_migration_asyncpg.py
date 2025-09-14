#!/usr/bin/env python3
"""
Direct Migration Runner using asyncpg
Simplified version that runs the migration directly without SQLAlchemy
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

class AsyncpgMigrationRunner:
    """Direct migration runner using asyncpg"""
    
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
    
    async def run_migration(self):
        """Run the advanced optimizations migration"""
        try:
            logger.info("Starting advanced optimizations migration...")
            
            # 1. Create advanced indexes
            logger.info("Creating advanced indexes...")
            await self._create_advanced_indexes()
            
            # 2. Create materialized views
            logger.info("Creating materialized views...")
            await self._create_materialized_views()
            
            # 3. Create continuous aggregates
            logger.info("Creating continuous aggregates...")
            await self._create_continuous_aggregates()
            
            # 4. Create data tiering policies
            logger.info("Creating data tiering policies...")
            await self._create_data_tiering_policies()
            
            # 5. Create performance monitoring tables
            logger.info("Creating performance monitoring tables...")
            await self._create_performance_monitoring_tables()
            
            # 6. Create cache optimization tables
            logger.info("Creating cache optimization tables...")
            await self._create_cache_optimization_tables()
            
            # 7. Create query optimization functions
            logger.info("Creating query optimization functions...")
            await self._create_query_optimization_functions()
            
            # 8. Create data lifecycle management
            logger.info("Creating data lifecycle management...")
            await self._create_data_lifecycle_management()
            
            logger.info("Advanced optimizations migration completed successfully!")
            
        except Exception as e:
            logger.error(f"Migration failed: {e}")
            raise
    
    async def _create_advanced_indexes(self):
        """Create advanced indexes for performance optimization"""
        try:
            # BRIN indexes for time-series data
            await self.connection.execute("""
                CREATE INDEX IF NOT EXISTS idx_market_data_time_brin 
                ON market_data USING BRIN (timestamp) WITH (pages_per_range = 128);
            """)
            
            await self.connection.execute("""
                CREATE INDEX IF NOT EXISTS idx_candlestick_data_time_brin 
                ON candlestick_data USING BRIN (timestamp) WITH (pages_per_range = 128);
            """)
            
            # Partial indexes for high-confidence signals (using 'proba' instead of 'confidence')
            await self.connection.execute("""
                CREATE INDEX IF NOT EXISTS idx_signals_high_confidence 
                ON signals (ts DESC, symbol) 
                WHERE proba > 0.7;
            """)
            
            # Covering indexes with INCLUDE
            await self.connection.execute("""
                CREATE INDEX IF NOT EXISTS idx_market_data_covering 
                ON market_data (symbol, timestamp DESC) 
                INCLUDE (open, high, low, close, volume);
            """)
            
            # GIN indexes for JSONB fields
            await self.connection.execute("""
                CREATE INDEX IF NOT EXISTS idx_candlestick_indicators_gin 
                ON candlestick_data USING GIN (indicators);
            """)
            
            await self.connection.execute("""
                CREATE INDEX IF NOT EXISTS idx_signals_indicators_gin 
                ON signals USING GIN (indicators);
            """)
            
            logger.info("Advanced indexes created successfully")
            
        except Exception as e:
            logger.error(f"Failed to create advanced indexes: {e}")
            raise
    
    async def _create_materialized_views(self):
        """Create materialized views for pre-joined data"""
        try:
            # Market data with indicators view
            await self.connection.execute("""
                CREATE MATERIALIZED VIEW IF NOT EXISTS market_data_with_indicators AS
                SELECT 
                    md.symbol,
                    md.timestamp,
                    md.open,
                    md.high,
                    md.low,
                    md.close,
                    md.volume,
                    cd.indicators,
                    cd.patterns
                FROM market_data md
                LEFT JOIN candlestick_data cd ON md.symbol = cd.symbol AND md.timestamp = cd.timestamp
                WHERE md.timestamp >= NOW() - INTERVAL '7 days';
            """)
            
            # Signals with context view (using actual column names)
            await self.connection.execute("""
                CREATE MATERIALIZED VIEW IF NOT EXISTS signals_with_context AS
                SELECT 
                    s.*,
                    md.open,
                    md.high,
                    md.low,
                    md.close,
                    md.volume
                FROM signals s
                LEFT JOIN market_data md ON s.symbol = md.symbol AND s.ts = md.timestamp
                WHERE s.ts >= NOW() - INTERVAL '30 days';
            """)
            
            # Performance summary view (using actual column names)
            await self.connection.execute("""
                CREATE MATERIALIZED VIEW IF NOT EXISTS performance_summary AS
                SELECT 
                    symbol,
                    DATE_TRUNC('day', ts) as date,
                    COUNT(*) as total_signals,
                    AVG(proba) as avg_confidence,
                    SUM(CASE WHEN outcome = 'win' THEN 1 ELSE 0 END) as wins,
                    SUM(CASE WHEN outcome = 'loss' THEN 1 ELSE 0 END) as losses
                FROM signals
                WHERE ts >= NOW() - INTERVAL '90 days'
                GROUP BY symbol, DATE_TRUNC('day', ts);
            """)
            
            logger.info("Materialized views created successfully")
            
        except Exception as e:
            logger.error(f"Failed to create materialized views: {e}")
            raise
    
    async def _create_continuous_aggregates(self):
        """Create continuous aggregates for time-series data"""
        try:
            # 5-minute OHLCV aggregates
            await self.connection.execute("""
                CREATE MATERIALIZED VIEW IF NOT EXISTS market_data_5min_agg
                WITH (timescaledb.continuous) AS
                SELECT 
                    symbol,
                    time_bucket('5 minutes', timestamp) AS bucket,
                    FIRST(open, timestamp) AS open,
                    MAX(high) AS high,
                    MIN(low) AS low,
                    LAST(close, timestamp) AS close,
                    SUM(volume) AS volume
                FROM market_data
                GROUP BY symbol, bucket;
            """)
            
            # 1-hour OHLCV aggregates
            await self.connection.execute("""
                CREATE MATERIALIZED VIEW IF NOT EXISTS market_data_1hour_agg
                WITH (timescaledb.continuous) AS
                SELECT 
                    symbol,
                    time_bucket('1 hour', timestamp) AS bucket,
                    FIRST(open, timestamp) AS open,
                    MAX(high) AS high,
                    MIN(low) AS low,
                    LAST(close, timestamp) AS close,
                    SUM(volume) AS volume
                FROM market_data
                GROUP BY symbol, bucket;
            """)
            
            logger.info("Continuous aggregates created successfully")
            
        except Exception as e:
            logger.error(f"Failed to create continuous aggregates: {e}")
            raise
    
    async def _create_data_tiering_policies(self):
        """Create data tiering and retention policies"""
        try:
            # Enable columnstore for compression
            await self.connection.execute("""
                ALTER TABLE market_data SET (timescaledb.compress, timescaledb.compress_segmentby = 'symbol');
            """)
            
            await self.connection.execute("""
                ALTER TABLE candlestick_data SET (timescaledb.compress, timescaledb.compress_segmentby = 'symbol');
            """)
            
            # Set up compression policies
            await self.connection.execute("""
                SELECT add_compression_policy('market_data', INTERVAL '7 days', if_not_exists => TRUE);
            """)
            
            await self.connection.execute("""
                SELECT add_compression_policy('candlestick_data', INTERVAL '7 days', if_not_exists => TRUE);
            """)
            
            # Set up retention policies
            await self.connection.execute("""
                SELECT add_retention_policy('market_data', INTERVAL '90 days', if_not_exists => TRUE);
            """)
            
            await self.connection.execute("""
                SELECT add_retention_policy('candlestick_data', INTERVAL '90 days', if_not_exists => TRUE);
            """)
            
            logger.info("Data tiering policies created successfully")
            
        except Exception as e:
            logger.error(f"Failed to create data tiering policies: {e}")
            raise
    
    async def _create_performance_monitoring_tables(self):
        """Create performance monitoring tables"""
        try:
            # Query performance metrics table
            await self.connection.execute("""
                CREATE TABLE IF NOT EXISTS query_performance_metrics (
                    id SERIAL PRIMARY KEY,
                    query_type VARCHAR(100) NOT NULL,
                    execution_time_ms FLOAT NOT NULL,
                    rows_returned INTEGER,
                    cache_hit BOOLEAN DEFAULT FALSE,
                    timestamp TIMESTAMPTZ DEFAULT NOW(),
                    query_hash VARCHAR(64),
                    user_agent TEXT
                );
            """)
            
            # Cache performance metrics table
            await self.connection.execute("""
                CREATE TABLE IF NOT EXISTS cache_performance_metrics (
                    id SERIAL PRIMARY KEY,
                    cache_type VARCHAR(50) NOT NULL,
                    hit_rate FLOAT NOT NULL,
                    miss_rate FLOAT NOT NULL,
                    avg_response_time_ms FLOAT,
                    memory_usage_mb FLOAT,
                    timestamp TIMESTAMPTZ DEFAULT NOW()
                );
            """)
            
            logger.info("Performance monitoring tables created successfully")
            
        except Exception as e:
            logger.error(f"Failed to create performance monitoring tables: {e}")
            raise
    
    async def _create_cache_optimization_tables(self):
        """Create cache optimization tables"""
        try:
            # UI cache table for pre-computed data
            await self.connection.execute("""
                CREATE TABLE IF NOT EXISTS ui_cache (
                    id SERIAL PRIMARY KEY,
                    cache_key VARCHAR(255) UNIQUE NOT NULL,
                    cache_data JSONB NOT NULL,
                    ttl TIMESTAMPTZ NOT NULL,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    updated_at TIMESTAMPTZ DEFAULT NOW()
                );
            """)
            
            # Create index on cache key and TTL
            await self.connection.execute("""
                CREATE INDEX IF NOT EXISTS idx_ui_cache_key_ttl 
                ON ui_cache (cache_key, ttl);
            """)
            
            logger.info("Cache optimization tables created successfully")
            
        except Exception as e:
            logger.error(f"Failed to create cache optimization tables: {e}")
            raise
    
    async def _create_query_optimization_functions(self):
        """Create query optimization functions"""
        try:
            # Function to get optimized market data
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
            
            # Function to get latest signals with context
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
            
            logger.info("Query optimization functions created successfully")
            
        except Exception as e:
            logger.error(f"Failed to create query optimization functions: {e}")
            raise
    
    async def _create_data_lifecycle_management(self):
        """Create data lifecycle management"""
        try:
            # Create a function to clean up old cache entries
            await self.connection.execute("""
                CREATE OR REPLACE FUNCTION cleanup_expired_cache()
                RETURNS INTEGER AS $$
                DECLARE
                    deleted_count INTEGER;
                BEGIN
                    DELETE FROM ui_cache WHERE ttl < NOW();
                    GET DIAGNOSTICS deleted_count = ROW_COUNT;
                    RETURN deleted_count;
                END;
                $$ LANGUAGE plpgsql;
            """)
            
            # Create a function to analyze table performance
            await self.connection.execute("""
                CREATE OR REPLACE FUNCTION analyze_table_performance()
                RETURNS TABLE (
                    table_name TEXT,
                    total_rows BIGINT,
                    table_size TEXT,
                    index_size TEXT,
                    cache_hit_ratio FLOAT
                ) AS $$
                BEGIN
                    RETURN QUERY
                    SELECT 
                        schemaname||'.'||tablename as table_name,
                        n_tup_ins + n_tup_upd + n_tup_del as total_rows,
                        pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as table_size,
                        pg_size_pretty(pg_indexes_size(schemaname||'.'||tablename)) as index_size,
                        heap_blks_hit::float / (heap_blks_hit + heap_blks_read) as cache_hit_ratio
                    FROM pg_stat_user_tables
                    WHERE schemaname = 'public'
                    ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;
                END;
                $$ LANGUAGE plpgsql;
            """)
            
            logger.info("Data lifecycle management created successfully")
            
        except Exception as e:
            logger.error(f"Failed to create data lifecycle management: {e}")
            raise
    
    async def cleanup(self):
        """Clean up resources"""
        if self.connection:
            await self.connection.close()

async def main():
    """Main function to run the migration"""
    runner = AsyncpgMigrationRunner()
    
    try:
        await runner.initialize()
        await runner.run_migration()
        logger.info("Migration completed successfully!")
        
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        sys.exit(1)
    
    finally:
        await runner.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
