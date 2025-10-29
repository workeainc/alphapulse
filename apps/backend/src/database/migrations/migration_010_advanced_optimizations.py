"""
Migration 010: Advanced Database Optimizations
Implements hybrid storage, materialized views, and advanced indexing
"""

import asyncio
import logging
from datetime import datetime
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

class AdvancedOptimizationsMigration:
    """Migration for advanced database optimizations"""
    
    def __init__(self, db_session_factory):
        self.db_session_factory = db_session_factory
        self.migration_version = "010"
        self.migration_name = "advanced_optimizations"
        
        logger.info(f"🚀 Initializing migration {self.migration_version}: {self.migration_name}")
    
    async def upgrade(self):
        """Apply advanced optimizations migration"""
        try:
            logger.info("🔄 Starting advanced optimizations migration...")
            
            async with self.db_session_factory() as session:
                # Step 1: Create advanced indexes
                await self._create_advanced_indexes(session)
                
                # Step 2: Create materialized views
                await self._create_materialized_views(session)
                
                # Step 3: Create continuous aggregates
                await self._create_continuous_aggregates(session)
                
                # Step 4: Create data tiering policies
                await self._create_data_tiering_policies(session)
                
                # Step 5: Create performance monitoring tables
                await self._create_performance_monitoring_tables(session)
                
                # Step 6: Create cache optimization tables
                await self._create_cache_optimization_tables(session)
                
                # Step 7: Create query optimization functions
                await self._create_query_optimization_functions(session)
                
                # Step 8: Create data lifecycle management
                await self._create_data_lifecycle_management(session)
                
                await session.commit()
                logger.info("✅ Advanced optimizations migration completed successfully")
                
        except Exception as e:
            logger.error(f"❌ Error in advanced optimizations migration: {e}")
            raise
    
    async def downgrade(self):
        """Rollback advanced optimizations migration"""
        try:
            logger.info("🔄 Rolling back advanced optimizations migration...")
            
            async with self.db_session_factory() as session:
                # Drop materialized views
                await self._drop_materialized_views(session)
                
                # Drop advanced indexes
                await self._drop_advanced_indexes(session)
                
                # Drop performance monitoring tables
                await self._drop_performance_monitoring_tables(session)
                
                # Drop cache optimization tables
                await self._drop_cache_optimization_tables(session)
                
                # Drop query optimization functions
                await self._drop_query_optimization_functions(session)
                
                await session.commit()
                logger.info("✅ Advanced optimizations migration rollback completed")
                
        except Exception as e:
            logger.error(f"❌ Error rolling back advanced optimizations migration: {e}")
            raise
    
    async def _create_advanced_indexes(self, session: AsyncSession):
        """Create advanced indexes for optimal performance"""
        try:
            logger.info("📊 Creating advanced indexes...")
            
            # BRIN indexes for time-series data
            await session.execute(text("""
                CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_candlestick_data_timestamp_brin 
                ON candlestick_data USING BRIN (timestamp) 
                WITH (pages_per_range = 128);
            """))
            
            await session.execute(text("""
                CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_signals_timestamp_brin 
                ON signals USING BRIN (timestamp) 
                WITH (pages_per_range = 128);
            """))
            
            await session.execute(text("""
                CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_trades_timestamp_brin 
                ON trades USING BRIN (timestamp) 
                WITH (pages_per_range = 128);
            """))
            
            # Partial indexes for filtered queries
            await session.execute(text("""
                CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_signals_high_confidence 
                ON signals (symbol, timestamp DESC, confidence DESC) 
                WHERE confidence >= 0.8;
            """))
            
            await session.execute(text("""
                CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_signals_active 
                ON signals (symbol, timestamp DESC) 
                WHERE status = 'active';
            """))
            
            await session.execute(text("""
                CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_trades_winners 
                ON trades (symbol, timestamp DESC, pnl DESC) 
                WHERE pnl > 0;
            """))
            
            # Covering indexes for common queries
            await session.execute(text("""
                CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_candlestick_covering 
                ON candlestick_data (symbol, timestamp DESC) 
                INCLUDE (open, high, low, close, volume, indicators, patterns);
            """))
            
            await session.execute(text("""
                CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_signals_covering 
                ON signals (symbol, timestamp DESC) 
                INCLUDE (side, strategy, confidence, strength, price, stop_loss, take_profit, status, metadata);
            """))
            
            # GIN indexes for JSONB fields
            await session.execute(text("""
                CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_candlestick_indicators_gin 
                ON candlestick_data USING GIN (indicators);
            """))
            
            await session.execute(text("""
                CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_signals_metadata_gin 
                ON signals USING GIN (metadata);
            """))
            
            logger.info("✅ Advanced indexes created successfully")
            
        except Exception as e:
            logger.error(f"Error creating advanced indexes: {e}")
            raise
    
    async def _create_materialized_views(self, session: AsyncSession):
        """Create materialized views for pre-joined data"""
        try:
            logger.info("📊 Creating materialized views...")
            
            # Market data with indicators view
            await session.execute(text("""
                CREATE MATERIALIZED VIEW IF NOT EXISTS market_data_with_indicators
                WITH (timescaledb.continuous) AS
                SELECT 
                    time_bucket('1 minute', cd.timestamp) AS bucket,
                    cd.symbol,
                    cd.timeframe,
                    cd.open,
                    cd.high,
                    cd.low,
                    cd.close,
                    cd.volume,
                    cd.indicators->>'rsi_14' AS rsi_14,
                    cd.indicators->>'macd' AS macd,
                    cd.indicators->>'macd_signal' AS macd_signal,
                    cd.indicators->>'ema_9' AS ema_9,
                    cd.indicators->>'ema_21' AS ema_21,
                    cd.indicators->>'bb_upper' AS bb_upper,
                    cd.indicators->>'bb_lower' AS bb_lower,
                    cd.patterns
                FROM candlestick_data cd
                WHERE cd.timestamp >= NOW() - INTERVAL '7 days'
                GROUP BY bucket, cd.symbol, cd.timeframe, cd.open, cd.high, cd.low, cd.close, cd.volume, 
                         cd.indicators, cd.patterns
                ORDER BY bucket DESC, cd.symbol;
            """))
            
            # Signals with context view
            await session.execute(text("""
                CREATE MATERIALIZED VIEW IF NOT EXISTS signals_with_context
                WITH (timescaledb.continuous) AS
                SELECT 
                    time_bucket('1 minute', s.timestamp) AS bucket,
                    s.id,
                    s.symbol,
                    s.side,
                    s.strategy,
                    s.confidence,
                    s.strength,
                    s.price,
                    s.stop_loss,
                    s.take_profit,
                    s.status,
                    s.metadata,
                    cd.open AS market_open,
                    cd.high AS market_high,
                    cd.low AS market_low,
                    cd.close AS market_close,
                    cd.volume AS market_volume,
                    cd.indicators->>'rsi_14' AS market_rsi,
                    cd.indicators->>'macd' AS market_macd,
                    t.pnl,
                    t.exit_price,
                    t.exit_timestamp,
                    CASE 
                        WHEN t.pnl > 0 THEN 'win'
                        WHEN t.pnl < 0 THEN 'loss'
                        ELSE 'pending'
                    END AS trade_result
                FROM signals s
                LEFT JOIN candlestick_data cd ON s.symbol = cd.symbol 
                    AND s.timestamp = cd.timestamp
                LEFT JOIN trades t ON s.id = t.signal_id
                WHERE s.timestamp >= NOW() - INTERVAL '30 days'
                GROUP BY bucket, s.id, s.symbol, s.side, s.strategy, s.confidence, s.strength,
                         s.price, s.stop_loss, s.take_profit, s.status, s.metadata,
                         cd.open, cd.high, cd.low, cd.close, cd.volume, cd.indicators,
                         t.pnl, t.exit_price, t.exit_timestamp
                ORDER BY bucket DESC, s.symbol;
            """))
            
            # Performance summary view
            await session.execute(text("""
                CREATE MATERIALIZED VIEW IF NOT EXISTS performance_summary
                WITH (timescaledb.continuous) AS
                SELECT 
                    time_bucket('1 hour', t.timestamp) AS bucket,
                    t.symbol,
                    t.strategy,
                    COUNT(*) AS total_trades,
                    COUNT(CASE WHEN t.pnl > 0 THEN 1 END) AS winning_trades,
                    COUNT(CASE WHEN t.pnl < 0 THEN 1 END) AS losing_trades,
                    AVG(t.pnl) AS avg_pnl,
                    SUM(t.pnl) AS total_pnl,
                    MAX(t.pnl) AS max_win,
                    MIN(t.pnl) AS max_loss,
                    CASE 
                        WHEN COUNT(*) > 0 THEN 
                            COUNT(CASE WHEN t.pnl > 0 THEN 1 END)::FLOAT / COUNT(*)
                        ELSE 0 
                    END AS win_rate,
                    AVG(s.confidence) AS avg_signal_confidence
                FROM trades t
                LEFT JOIN signals s ON t.signal_id = s.id
                WHERE t.timestamp >= NOW() - INTERVAL '90 days'
                GROUP BY bucket, t.symbol, t.strategy
                ORDER BY bucket DESC, t.symbol, t.strategy;
            """))
            
            # UI cache view
            await session.execute(text("""
                CREATE MATERIALIZED VIEW IF NOT EXISTS ui_cache
                WITH (timescaledb.continuous) AS
                SELECT 
                    time_bucket('5 minutes', NOW()) AS bucket,
                    jsonb_build_object(
                        'symbols', (
                            SELECT jsonb_object_agg(symbol, jsonb_build_object(
                                'price', close,
                                'change_24h', (close - LAG(close, 1440) OVER (PARTITION BY symbol ORDER BY timestamp)) / LAG(close, 1440) OVER (PARTITION BY symbol ORDER BY timestamp) * 100,
                                'volume_24h', SUM(volume) OVER (PARTITION BY symbol ORDER BY timestamp ROWS BETWEEN 1440 PRECEDING AND CURRENT ROW),
                                'rsi', indicators->>'rsi_14',
                                'macd', indicators->>'macd',
                                'trend', CASE 
                                    WHEN indicators->>'ema_9' > indicators->>'ema_21' THEN 'bullish'
                                    ELSE 'bearish'
                                END
                            ))
                            FROM (
                                SELECT DISTINCT ON (symbol) 
                                    symbol, close, volume, indicators, timestamp
                                FROM candlestick_data 
                                WHERE timestamp >= NOW() - INTERVAL '1 day'
                                ORDER BY symbol, timestamp DESC
                            ) latest_data
                        ),
                        'signals', (
                            SELECT jsonb_agg(jsonb_build_object(
                                'id', id,
                                'symbol', symbol,
                                'side', side,
                                'confidence', confidence,
                                'price', price,
                                'timestamp', timestamp
                            ))
                            FROM (
                                SELECT id, symbol, side, confidence, price, timestamp
                                FROM signals 
                                WHERE timestamp >= NOW() - INTERVAL '1 hour'
                                ORDER BY timestamp DESC
                                LIMIT 50
                            ) recent_signals
                        ),
                        'performance', (
                            SELECT jsonb_build_object(
                                'total_trades', COUNT(*),
                                'win_rate', COUNT(CASE WHEN pnl > 0 THEN 1 END)::FLOAT / COUNT(*),
                                'total_pnl', SUM(pnl),
                                'avg_pnl', AVG(pnl)
                            )
                            FROM trades 
                            WHERE timestamp >= NOW() - INTERVAL '24 hours'
                        )
                    ) AS ui_data
                FROM candlestick_data
                WHERE timestamp >= NOW() - INTERVAL '1 minute'
                GROUP BY bucket;
            """))
            
            logger.info("✅ Materialized views created successfully")
            
        except Exception as e:
            logger.error(f"Error creating materialized views: {e}")
            raise
    
    async def _create_continuous_aggregates(self, session: AsyncSession):
        """Create continuous aggregates for efficient querying"""
        try:
            logger.info("📊 Creating continuous aggregates...")
            
            # Add refresh policies for materialized views
            await session.execute(text("""
                SELECT add_continuous_aggregate_policy('market_data_with_indicators',
                    start_offset => INTERVAL '1 hour',
                    end_offset => INTERVAL '1 minute',
                    schedule_interval => INTERVAL '1 minute'
                );
            """))
            
            await session.execute(text("""
                SELECT add_continuous_aggregate_policy('signals_with_context',
                    start_offset => INTERVAL '2 hours',
                    end_offset => INTERVAL '1 minute',
                    schedule_interval => INTERVAL '5 minutes'
                );
            """))
            
            await session.execute(text("""
                SELECT add_continuous_aggregate_policy('performance_summary',
                    start_offset => INTERVAL '3 hours',
                    end_offset => INTERVAL '1 hour',
                    schedule_interval => INTERVAL '1 hour'
                );
            """))
            
            await session.execute(text("""
                SELECT add_continuous_aggregate_policy('ui_cache',
                    start_offset => INTERVAL '10 minutes',
                    end_offset => INTERVAL '5 minutes',
                    schedule_interval => INTERVAL '5 minutes'
                );
            """))
            
            logger.info("✅ Continuous aggregates created successfully")
            
        except Exception as e:
            logger.error(f"Error creating continuous aggregates: {e}")
            raise
    
    async def _create_data_tiering_policies(self, session: AsyncSession):
        """Create data tiering policies for storage optimization"""
        try:
            logger.info("📊 Creating data tiering policies...")
            
            # Compression policies
            await session.execute(text("""
                SELECT add_compression_policy('candlestick_data', INTERVAL '7 days');
            """))
            
            await session.execute(text("""
                SELECT add_compression_policy('signals', INTERVAL '7 days');
            """))
            
            await session.execute(text("""
                SELECT add_compression_policy('trades', INTERVAL '7 days');
            """))
            
            await session.execute(text("""
                SELECT add_compression_policy('candlestick_patterns', INTERVAL '7 days');
            """))
            
            # Retention policies
            await session.execute(text("""
                SELECT add_retention_policy('candlestick_data', INTERVAL '90 days');
            """))
            
            await session.execute(text("""
                SELECT add_retention_policy('signals', INTERVAL '180 days');
            """))
            
            await session.execute(text("""
                SELECT add_retention_policy('trades', INTERVAL '365 days');
            """))
            
            await session.execute(text("""
                SELECT add_retention_policy('candlestick_patterns', INTERVAL '90 days');
            """))
            
            logger.info("✅ Data tiering policies created successfully")
            
        except Exception as e:
            logger.error(f"Error creating data tiering policies: {e}")
            raise
    
    async def _create_performance_monitoring_tables(self, session: AsyncSession):
        """Create performance monitoring tables"""
        try:
            logger.info("📊 Creating performance monitoring tables...")
            
            # Query performance tracking
            await session.execute(text("""
                CREATE TABLE IF NOT EXISTS query_performance_log (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMPTZ NOT NULL,
                    query_hash VARCHAR(64) NOT NULL,
                    query_text TEXT NOT NULL,
                    execution_time_ms FLOAT NOT NULL,
                    rows_returned INTEGER NOT NULL,
                    rows_scanned INTEGER NOT NULL,
                    cache_hit BOOLEAN NOT NULL,
                    index_used VARCHAR(100),
                    table_name VARCHAR(100),
                    user_agent VARCHAR(200),
                    ip_address INET,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );
            """))
            
            # Create hypertable for query performance
            await session.execute(text("""
                SELECT create_hypertable('query_performance_log', 'timestamp', 
                    if_not_exists => TRUE, 
                    chunk_time_interval => INTERVAL '1 day'
                );
            """))
            
            # Cache performance tracking
            await session.execute(text("""
                CREATE TABLE IF NOT EXISTS cache_performance_log (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMPTZ NOT NULL,
                    cache_type VARCHAR(50) NOT NULL, -- 'redis', 'memory', 'database'
                    operation VARCHAR(20) NOT NULL, -- 'get', 'set', 'delete'
                    key_pattern VARCHAR(200),
                    response_time_ms FLOAT NOT NULL,
                    cache_hit BOOLEAN NOT NULL,
                    data_size_bytes INTEGER,
                    ttl_seconds INTEGER,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );
            """))
            
            # Create hypertable for cache performance
            await session.execute(text("""
                SELECT create_hypertable('cache_performance_log', 'timestamp', 
                    if_not_exists => TRUE, 
                    chunk_time_interval => INTERVAL '1 hour'
                );
            """))
            
            # System performance metrics
            await session.execute(text("""
                CREATE TABLE IF NOT EXISTS system_performance_metrics (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMPTZ NOT NULL,
                    metric_name VARCHAR(100) NOT NULL,
                    metric_value FLOAT NOT NULL,
                    metric_unit VARCHAR(20),
                    component VARCHAR(50), -- 'database', 'cache', 'api', 'websocket'
                    metadata JSONB,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );
            """))
            
            # Create hypertable for system metrics
            await session.execute(text("""
                SELECT create_hypertable('system_performance_metrics', 'timestamp', 
                    if_not_exists => TRUE, 
                    chunk_time_interval => INTERVAL '1 hour'
                );
            """))
            
            logger.info("✅ Performance monitoring tables created successfully")
            
        except Exception as e:
            logger.error(f"Error creating performance monitoring tables: {e}")
            raise
    
    async def _create_cache_optimization_tables(self, session: AsyncSession):
        """Create cache optimization tables"""
        try:
            logger.info("📊 Creating cache optimization tables...")
            
            # Cache hit rate tracking
            await session.execute(text("""
                CREATE TABLE IF NOT EXISTS cache_hit_rate_metrics (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMPTZ NOT NULL,
                    cache_type VARCHAR(50) NOT NULL,
                    data_type VARCHAR(50) NOT NULL, -- 'candlestick', 'signals', 'patterns'
                    total_requests INTEGER NOT NULL,
                    cache_hits INTEGER NOT NULL,
                    cache_misses INTEGER NOT NULL,
                    hit_rate FLOAT NOT NULL,
                    avg_response_time_ms FLOAT NOT NULL,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );
            """))
            
            # Create hypertable for cache metrics
            await session.execute(text("""
                SELECT create_hypertable('cache_hit_rate_metrics', 'timestamp', 
                    if_not_exists => TRUE, 
                    chunk_time_interval => INTERVAL '1 hour'
                );
            """))
            
            # Cache warming strategies
            await session.execute(text("""
                CREATE TABLE IF NOT EXISTS cache_warming_strategies (
                    id SERIAL PRIMARY KEY,
                    strategy_name VARCHAR(100) UNIQUE NOT NULL,
                    data_type VARCHAR(50) NOT NULL,
                    priority INTEGER NOT NULL DEFAULT 1,
                    ttl_seconds INTEGER NOT NULL,
                    max_items INTEGER NOT NULL,
                    is_active BOOLEAN NOT NULL DEFAULT true,
                    last_executed TIMESTAMPTZ,
                    execution_count INTEGER DEFAULT 0,
                    success_count INTEGER DEFAULT 0,
                    avg_execution_time_ms FLOAT DEFAULT 0.0,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    updated_at TIMESTAMPTZ DEFAULT NOW()
                );
            """))
            
            # Insert default cache warming strategies
            await session.execute(text("""
                INSERT INTO cache_warming_strategies 
                (strategy_name, data_type, priority, ttl_seconds, max_items, is_active)
                VALUES 
                ('latest_signals', 'signals', 1, 300, 100, true),
                ('latest_patterns', 'patterns', 2, 600, 50, true),
                ('market_overview', 'candlestick', 3, 3600, 1000, true),
                ('performance_summary', 'performance', 4, 1800, 100, true)
                ON CONFLICT (strategy_name) DO NOTHING;
            """))
            
            logger.info("✅ Cache optimization tables created successfully")
            
        except Exception as e:
            logger.error(f"Error creating cache optimization tables: {e}")
            raise
    
    async def _create_query_optimization_functions(self, session: AsyncSession):
        """Create query optimization functions"""
        try:
            logger.info("📊 Creating query optimization functions...")
            
            # Function to get optimized market data
            await session.execute(text("""
                CREATE OR REPLACE FUNCTION get_optimized_market_data(
                    p_symbol VARCHAR(20),
                    p_timeframe VARCHAR(10),
                    p_hours INTEGER DEFAULT 24
                )
                RETURNS TABLE (
                    timestamp TIMESTAMPTZ,
                    open DECIMAL(20,8),
                    high DECIMAL(20,8),
                    low DECIMAL(20,8),
                    close DECIMAL(20,8),
                    volume DECIMAL(20,8),
                    rsi_14 DECIMAL(10,4),
                    macd DECIMAL(20,8),
                    ema_9 DECIMAL(20,8),
                    ema_21 DECIMAL(20,8),
                    bb_upper DECIMAL(20,8),
                    bb_lower DECIMAL(20,8)
                )
                LANGUAGE plpgsql
                AS $$
                BEGIN
                    RETURN QUERY
                    SELECT 
                        mdi.bucket as timestamp,
                        mdi.open,
                        mdi.high,
                        mdi.low,
                        mdi.close,
                        mdi.volume,
                        mdi.rsi_14::DECIMAL(10,4),
                        mdi.macd::DECIMAL(20,8),
                        mdi.ema_9::DECIMAL(20,8),
                        mdi.ema_21::DECIMAL(20,8),
                        mdi.bb_upper::DECIMAL(20,8),
                        mdi.bb_lower::DECIMAL(20,8)
                    FROM market_data_with_indicators mdi
                    WHERE mdi.symbol = p_symbol 
                        AND mdi.timeframe = p_timeframe
                        AND mdi.bucket >= NOW() - INTERVAL '1 hour' * p_hours
                    ORDER BY mdi.bucket DESC;
                END;
                $$;
            """))
            
            # Function to get optimized signals
            await session.execute(text("""
                CREATE OR REPLACE FUNCTION get_optimized_signals(
                    p_symbol VARCHAR(20) DEFAULT NULL,
                    p_hours INTEGER DEFAULT 24,
                    p_min_confidence DECIMAL(5,4) DEFAULT 0.0
                )
                RETURNS TABLE (
                    id VARCHAR(50),
                    symbol VARCHAR(20),
                    side VARCHAR(10),
                    strategy VARCHAR(50),
                    confidence DECIMAL(5,4),
                    strength VARCHAR(20),
                    price DECIMAL(20,8),
                    timestamp TIMESTAMPTZ,
                    market_rsi DECIMAL(10,4),
                    market_macd DECIMAL(20,8),
                    pnl DECIMAL(20,8),
                    trade_result VARCHAR(10)
                )
                LANGUAGE plpgsql
                AS $$
                BEGIN
                    RETURN QUERY
                    SELECT 
                        swc.id,
                        swc.symbol,
                        swc.side,
                        swc.strategy,
                        swc.confidence,
                        swc.strength,
                        swc.price,
                        swc.bucket as timestamp,
                        swc.market_rsi::DECIMAL(10,4),
                        swc.market_macd::DECIMAL(20,8),
                        swc.pnl,
                        swc.trade_result
                    FROM signals_with_context swc
                    WHERE (p_symbol IS NULL OR swc.symbol = p_symbol)
                        AND swc.bucket >= NOW() - INTERVAL '1 hour' * p_hours
                        AND swc.confidence >= p_min_confidence
                    ORDER BY swc.bucket DESC;
                END;
                $$;
            """))
            
            # Function to get performance summary
            await session.execute(text("""
                CREATE OR REPLACE FUNCTION get_performance_summary(
                    p_symbol VARCHAR(20) DEFAULT NULL,
                    p_hours INTEGER DEFAULT 24
                )
                RETURNS TABLE (
                    symbol VARCHAR(20),
                    strategy VARCHAR(50),
                    total_trades BIGINT,
                    winning_trades BIGINT,
                    losing_trades BIGINT,
                    win_rate DECIMAL(5,4),
                    avg_pnl DECIMAL(20,8),
                    total_pnl DECIMAL(20,8),
                    max_win DECIMAL(20,8),
                    max_loss DECIMAL(20,8),
                    avg_signal_confidence DECIMAL(5,4)
                )
                LANGUAGE plpgsql
                AS $$
                BEGIN
                    RETURN QUERY
                    SELECT 
                        ps.symbol,
                        ps.strategy,
                        ps.total_trades,
                        ps.winning_trades,
                        ps.losing_trades,
                        ps.win_rate,
                        ps.avg_pnl,
                        ps.total_pnl,
                        ps.max_win,
                        ps.max_loss,
                        ps.avg_signal_confidence
                    FROM performance_summary ps
                    WHERE (p_symbol IS NULL OR ps.symbol = p_symbol)
                        AND ps.bucket >= NOW() - INTERVAL '1 hour' * p_hours
                    ORDER BY ps.total_pnl DESC;
                END;
                $$;
            """))
            
            logger.info("✅ Query optimization functions created successfully")
            
        except Exception as e:
            logger.error(f"Error creating query optimization functions: {e}")
            raise
    
    async def _create_data_lifecycle_management(self, session: AsyncSession):
        """Create data lifecycle management"""
        try:
            logger.info("📊 Creating data lifecycle management...")
            
            # Data lifecycle policies table
            await session.execute(text("""
                CREATE TABLE IF NOT EXISTS data_lifecycle_policies (
                    id SERIAL PRIMARY KEY,
                    table_name VARCHAR(100) NOT NULL,
                    policy_type VARCHAR(50) NOT NULL, -- 'compression', 'retention', 'archival'
                    policy_config JSONB NOT NULL,
                    is_active BOOLEAN NOT NULL DEFAULT true,
                    last_executed TIMESTAMPTZ,
                    execution_count INTEGER DEFAULT 0,
                    success_count INTEGER DEFAULT 0,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    updated_at TIMESTAMPTZ DEFAULT NOW()
                );
            """))
            
            # Insert default lifecycle policies
            await session.execute(text("""
                INSERT INTO data_lifecycle_policies 
                (table_name, policy_type, policy_config, is_active)
                VALUES 
                ('candlestick_data', 'compression', '{"interval": "7 days"}', true),
                ('candlestick_data', 'retention', '{"interval": "90 days"}', true),
                ('signals', 'compression', '{"interval": "7 days"}', true),
                ('signals', 'retention', '{"interval": "180 days"}', true),
                ('trades', 'compression', '{"interval": "7 days"}', true),
                ('trades', 'retention', '{"interval": "365 days"}', true),
                ('candlestick_patterns', 'compression', '{"interval": "7 days"}', true),
                ('candlestick_patterns', 'retention', '{"interval": "90 days"}', true)
                ON CONFLICT DO NOTHING;
            """))
            
            # Data archival tracking
            await session.execute(text("""
                CREATE TABLE IF NOT EXISTS data_archival_log (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMPTZ NOT NULL,
                    table_name VARCHAR(100) NOT NULL,
                    operation VARCHAR(50) NOT NULL, -- 'compressed', 'archived', 'deleted'
                    records_affected INTEGER NOT NULL,
                    storage_saved_bytes BIGINT,
                    execution_time_ms FLOAT,
                    status VARCHAR(20) NOT NULL, -- 'success', 'failed', 'partial'
                    error_message TEXT,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );
            """))
            
            # Create hypertable for archival log
            await session.execute(text("""
                SELECT create_hypertable('data_archival_log', 'timestamp', 
                    if_not_exists => TRUE, 
                    chunk_time_interval => INTERVAL '1 day'
                );
            """))
            
            logger.info("✅ Data lifecycle management created successfully")
            
        except Exception as e:
            logger.error(f"Error creating data lifecycle management: {e}")
            raise
    
    async def _drop_materialized_views(self, session: AsyncSession):
        """Drop materialized views"""
        try:
            views = [
                'market_data_with_indicators',
                'signals_with_context',
                'performance_summary',
                'ui_cache'
            ]
            
            for view in views:
                await session.execute(text(f"DROP MATERIALIZED VIEW IF EXISTS {view} CASCADE;"))
            
            logger.info("✅ Materialized views dropped successfully")
            
        except Exception as e:
            logger.error(f"Error dropping materialized views: {e}")
            raise
    
    async def _drop_advanced_indexes(self, session: AsyncSession):
        """Drop advanced indexes"""
        try:
            indexes = [
                'idx_candlestick_data_timestamp_brin',
                'idx_signals_timestamp_brin',
                'idx_trades_timestamp_brin',
                'idx_signals_high_confidence',
                'idx_signals_active',
                'idx_trades_winners',
                'idx_candlestick_covering',
                'idx_signals_covering',
                'idx_candlestick_indicators_gin',
                'idx_signals_metadata_gin'
            ]
            
            for index in indexes:
                await session.execute(text(f"DROP INDEX IF EXISTS {index};"))
            
            logger.info("✅ Advanced indexes dropped successfully")
            
        except Exception as e:
            logger.error(f"Error dropping advanced indexes: {e}")
            raise
    
    async def _drop_performance_monitoring_tables(self, session: AsyncSession):
        """Drop performance monitoring tables"""
        try:
            tables = [
                'query_performance_log',
                'cache_performance_log',
                'system_performance_metrics'
            ]
            
            for table in tables:
                await session.execute(text(f"DROP TABLE IF EXISTS {table} CASCADE;"))
            
            logger.info("✅ Performance monitoring tables dropped successfully")
            
        except Exception as e:
            logger.error(f"Error dropping performance monitoring tables: {e}")
            raise
    
    async def _drop_cache_optimization_tables(self, session: AsyncSession):
        """Drop cache optimization tables"""
        try:
            tables = [
                'cache_hit_rate_metrics',
                'cache_warming_strategies'
            ]
            
            for table in tables:
                await session.execute(text(f"DROP TABLE IF EXISTS {table} CASCADE;"))
            
            logger.info("✅ Cache optimization tables dropped successfully")
            
        except Exception as e:
            logger.error(f"Error dropping cache optimization tables: {e}")
            raise
    
    async def _drop_query_optimization_functions(self, session: AsyncSession):
        """Drop query optimization functions"""
        try:
            functions = [
                'get_optimized_market_data',
                'get_optimized_signals',
                'get_performance_summary'
            ]
            
            for func in functions:
                await session.execute(text(f"DROP FUNCTION IF EXISTS {func} CASCADE;"))
            
            logger.info("✅ Query optimization functions dropped successfully")
            
        except Exception as e:
            logger.error(f"Error dropping query optimization functions: {e}")
            raise
