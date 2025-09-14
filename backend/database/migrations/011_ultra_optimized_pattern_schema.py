"""
Migration 011: Ultra-Optimized Pattern Detection Schema
Implements advanced indexing, compression, and performance optimizations
"""

import asyncio
import logging
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB, TIMESTAMP
from sqlalchemy.sql import text

# revision identifiers, used by Alembic
revision = '011_ultra_optimized_pattern_schema'
down_revision = '010_advanced_optimizations'
depends_on = None

logger = logging.getLogger(__name__)

def upgrade():
    """
    Create ultra-optimized pattern detection schema with advanced indexing
    """
    logger.info("🚀 Starting ultra-optimized pattern schema migration")
    
    # 1. Enhanced Candlestick Patterns Table with Advanced Features
    op.create_table(
        'ultra_optimized_patterns',
        sa.Column('id', sa.BigInteger(), nullable=False),
        sa.Column('pattern_id', sa.String(50), nullable=False),
        sa.Column('symbol', sa.String(20), nullable=False),
        sa.Column('timeframe', sa.String(10), nullable=False),
        sa.Column('pattern_name', sa.String(100), nullable=False),
        sa.Column('pattern_category', sa.String(20), nullable=False),  # single, double, triple, complex
        sa.Column('pattern_type', sa.String(20), nullable=False),  # reversal, continuation, indecision
        sa.Column('direction', sa.String(10), nullable=False),  # bullish, bearish, neutral
        sa.Column('confidence', sa.Numeric(precision=4, scale=3), nullable=False),
        sa.Column('strength', sa.String(20), nullable=False),  # weak, moderate, strong
        sa.Column('price_level', sa.Numeric(precision=20, scale=8), nullable=False),
        sa.Column('volume_confirmation', sa.Boolean(), nullable=False, default=False),
        sa.Column('volume_confidence', sa.Numeric(precision=4, scale=3), nullable=False, default=0.0),
        sa.Column('volume_pattern_type', sa.String(50), nullable=True),
        sa.Column('volume_strength', sa.String(20), nullable=True),
        sa.Column('volume_context', JSONB, nullable=True),
        sa.Column('trend_alignment', sa.String(20), nullable=False, default='neutral'),
        sa.Column('multi_timeframe_boost', sa.Numeric(precision=4, scale=3), nullable=False, default=0.0),
        sa.Column('processing_time_ms', sa.Numeric(precision=8, scale=2), nullable=True),
        sa.Column('vectorized_operations', sa.Boolean(), nullable=False, default=True),
        sa.Column('cache_hit', sa.Boolean(), nullable=False, default=False),
        sa.Column('technical_indicators', JSONB, nullable=True),  # RSI, MACD, Bollinger Bands, etc.
        sa.Column('market_conditions', JSONB, nullable=True),  # Volatility, trend strength, etc.
        sa.Column('pattern_metadata', JSONB, nullable=True),  # Body ratio, shadows, etc.
        sa.Column('performance_metrics', JSONB, nullable=True),  # Processing stats, cache hits, etc.
        sa.Column('data_points_used', sa.Integer(), nullable=False),
        sa.Column('data_quality_score', sa.Numeric(precision=4, scale=3), nullable=False),
        sa.Column('status', sa.String(20), nullable=False, default='active'),  # active, completed, failed
        sa.Column('timestamp', TIMESTAMP(timezone=True), nullable=False),
        sa.Column('created_at', TIMESTAMP(timezone=True), nullable=False, server_default=sa.text('NOW()')),
        sa.Column('updated_at', TIMESTAMP(timezone=True), nullable=False, server_default=sa.text('NOW()')),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('pattern_id')
    )
    
    # 2. Pattern Detection Performance Metrics Table
    op.create_table(
        'pattern_detection_metrics',
        sa.Column('id', sa.BigInteger(), nullable=False),
        sa.Column('symbol', sa.String(20), nullable=False),
        sa.Column('timeframe', sa.String(10), nullable=False),
        sa.Column('detection_session_id', sa.String(50), nullable=False),
        sa.Column('total_patterns_detected', sa.Integer(), nullable=False, default=0),
        sa.Column('cache_hits', sa.Integer(), nullable=False, default=0),
        sa.Column('cache_misses', sa.Integer(), nullable=False, default=0),
        sa.Column('avg_processing_time_ms', sa.Numeric(precision=8, scale=2), nullable=False, default=0.0),
        sa.Column('total_processing_time_ms', sa.Numeric(precision=12, scale=2), nullable=False, default=0.0),
        sa.Column('vectorized_operations', sa.Integer(), nullable=False, default=0),
        sa.Column('parallel_operations', sa.Integer(), nullable=False, default=0),
        sa.Column('memory_usage_mb', sa.Numeric(precision=8, scale=2), nullable=True),
        sa.Column('cpu_usage_percent', sa.Numeric(precision=5, scale=2), nullable=True),
        sa.Column('patterns_per_second', sa.Numeric(precision=8, scale=2), nullable=True),
        sa.Column('cache_hit_rate', sa.Numeric(precision=5, scale=4), nullable=True),
        sa.Column('performance_metadata', JSONB, nullable=True),
        sa.Column('timestamp', TIMESTAMP(timezone=True), nullable=False),
        sa.Column('created_at', TIMESTAMP(timezone=True), nullable=False, server_default=sa.text('NOW()')),
        sa.PrimaryKeyConstraint('id')
    )
    
    # 3. Sliding Window Buffer State Table
    op.create_table(
        'sliding_window_buffers',
        sa.Column('id', sa.BigInteger(), nullable=False),
        sa.Column('buffer_key', sa.String(100), nullable=False),  # symbol_timeframe
        sa.Column('symbol', sa.String(20), nullable=False),
        sa.Column('timeframe', sa.String(10), nullable=False),
        sa.Column('buffer_size', sa.Integer(), nullable=False),
        sa.Column('data_points_count', sa.Integer(), nullable=False, default=0),
        sa.Column('first_timestamp', TIMESTAMP(timezone=True), nullable=True),
        sa.Column('last_timestamp', TIMESTAMP(timezone=True), nullable=True),
        sa.Column('last_update_time', TIMESTAMP(timezone=True), nullable=False),
        sa.Column('memory_usage_mb', sa.Numeric(precision=8, scale=2), nullable=True),
        sa.Column('buffer_metadata', JSONB, nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=False, default=True),
        sa.Column('created_at', TIMESTAMP(timezone=True), nullable=False, server_default=sa.text('NOW()')),
        sa.Column('updated_at', TIMESTAMP(timezone=True), nullable=False, server_default=sa.text('NOW()')),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('buffer_key')
    )
    
    # 4. Pattern Detection Cache Table
    op.create_table(
        'pattern_detection_cache',
        sa.Column('id', sa.BigInteger(), nullable=False),
        sa.Column('cache_key', sa.String(64), nullable=False),  # MD5 hash of data
        sa.Column('symbol', sa.String(20), nullable=False),
        sa.Column('timeframe', sa.String(10), nullable=False),
        sa.Column('data_hash', sa.String(64), nullable=False),
        sa.Column('patterns_data', JSONB, nullable=False),  # Serialized pattern results
        sa.Column('cache_size_bytes', sa.BigInteger(), nullable=False),
        sa.Column('hit_count', sa.Integer(), nullable=False, default=0),
        sa.Column('last_accessed', TIMESTAMP(timezone=True), nullable=False),
        sa.Column('expires_at', TIMESTAMP(timezone=True), nullable=False),
        sa.Column('created_at', TIMESTAMP(timezone=True), nullable=False, server_default=sa.text('NOW()')),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('cache_key')
    )
    
    # 5. Multi-Timeframe Pattern Correlation Table
    op.create_table(
        'multi_timeframe_patterns',
        sa.Column('id', sa.BigInteger(), nullable=False),
        sa.Column('correlation_id', sa.String(50), nullable=False),
        sa.Column('symbol', sa.String(20), nullable=False),
        sa.Column('base_timeframe', sa.String(10), nullable=False),
        sa.Column('correlated_timeframes', JSONB, nullable=False),  # Array of timeframes
        sa.Column('pattern_name', sa.String(100), nullable=False),
        sa.Column('correlation_strength', sa.Numeric(precision=4, scale=3), nullable=False),
        sa.Column('timeframe_weights', JSONB, nullable=False),  # Weights for each timeframe
        sa.Column('combined_confidence', sa.Numeric(precision=4, scale=3), nullable=False),
        sa.Column('consensus_direction', sa.String(10), nullable=False),
        sa.Column('consensus_strength', sa.String(20), nullable=False),
        sa.Column('correlation_metadata', JSONB, nullable=True),
        sa.Column('timestamp', TIMESTAMP(timezone=True), nullable=False),
        sa.Column('created_at', TIMESTAMP(timezone=True), nullable=False, server_default=sa.text('NOW()')),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('correlation_id')
    )
    
    # 6. Pattern Performance Analytics Table
    op.create_table(
        'pattern_performance_analytics',
        sa.Column('id', sa.BigInteger(), nullable=False),
        sa.Column('symbol', sa.String(20), nullable=False),
        sa.Column('timeframe', sa.String(10), nullable=False),
        sa.Column('pattern_name', sa.String(100), nullable=False),
        sa.Column('analysis_period', sa.String(20), nullable=False),  # daily, weekly, monthly
        sa.Column('total_occurrences', sa.Integer(), nullable=False, default=0),
        sa.Column('successful_signals', sa.Integer(), nullable=False, default=0),
        sa.Column('failed_signals', sa.Integer(), nullable=False, default=0),
        sa.Column('success_rate', sa.Numeric(precision=5, scale=4), nullable=False, default=0.0),
        sa.Column('avg_confidence', sa.Numeric(precision=4, scale=3), nullable=False, default=0.0),
        sa.Column('avg_strength', sa.Numeric(precision=4, scale=3), nullable=False, default=0.0),
        sa.Column('avg_processing_time_ms', sa.Numeric(precision=8, scale=2), nullable=False, default=0.0),
        sa.Column('volume_confirmation_rate', sa.Numeric(precision=5, scale=4), nullable=False, default=0.0),
        sa.Column('trend_alignment_rate', sa.Numeric(precision=5, scale=4), nullable=False, default=0.0),
        sa.Column('performance_metrics', JSONB, nullable=True),
        sa.Column('period_start', TIMESTAMP(timezone=True), nullable=False),
        sa.Column('period_end', TIMESTAMP(timezone=True), nullable=False),
        sa.Column('created_at', TIMESTAMP(timezone=True), nullable=False, server_default=sa.text('NOW()')),
        sa.PrimaryKeyConstraint('id')
    )
    
    logger.info("✅ Created ultra-optimized pattern tables")
    
    # 7. Create TimescaleDB Hypertables
    connection = op.get_bind()
    
    # Convert to TimescaleDB hypertables
    connection.execute(text("""
        SELECT create_hypertable(
            'ultra_optimized_patterns',
            'timestamp',
            chunk_time_interval => INTERVAL '1 hour',
            if_not_exists => TRUE
        );
    """))
    
    connection.execute(text("""
        SELECT create_hypertable(
            'pattern_detection_metrics',
            'timestamp',
            chunk_time_interval => INTERVAL '1 hour',
            if_not_exists => TRUE
        );
    """))
    
    connection.execute(text("""
        SELECT create_hypertable(
            'multi_timeframe_patterns',
            'timestamp',
            chunk_time_interval => INTERVAL '1 hour',
            if_not_exists => TRUE
        );
    """))
    
    connection.execute(text("""
        SELECT create_hypertable(
            'pattern_performance_analytics',
            'period_start',
            chunk_time_interval => INTERVAL '1 day',
            if_not_exists => TRUE
        );
    """))
    
    logger.info("✅ Created TimescaleDB hypertables")
    
    # 8. Create Advanced Indexes for Ultra-Fast Queries
    
    # BRIN indexes for time-series data (ultra-fast, low space)
    connection.execute(text("""
        CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_ultra_patterns_timestamp_brin 
        ON ultra_optimized_patterns USING BRIN (timestamp) 
        WITH (pages_per_range = 128);
    """))
    
    connection.execute(text("""
        CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_pattern_metrics_timestamp_brin 
        ON pattern_detection_metrics USING BRIN (timestamp) 
        WITH (pages_per_range = 128);
    """))
    
    # Partial indexes for filtered queries
    connection.execute(text("""
        CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_ultra_patterns_high_confidence 
        ON ultra_optimized_patterns (symbol, pattern_name, timestamp DESC) 
        WHERE confidence >= 0.8;
    """))
    
    connection.execute(text("""
        CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_ultra_patterns_volume_confirmed 
        ON ultra_optimized_patterns (symbol, pattern_name, timestamp DESC) 
        WHERE volume_confirmation = true;
    """))
    
    connection.execute(text("""
        CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_ultra_patterns_vectorized 
        ON ultra_optimized_patterns (symbol, timestamp DESC) 
        WHERE vectorized_operations = true;
    """))
    
    # Covering indexes for common queries
    connection.execute(text("""
        CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_ultra_patterns_covering 
        ON ultra_optimized_patterns (symbol, timeframe, timestamp DESC) 
        INCLUDE (pattern_name, confidence, strength, volume_confirmation, trend_alignment);
    """))
    
    connection.execute(text("""
        CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_pattern_metrics_covering 
        ON pattern_detection_metrics (symbol, timeframe, timestamp DESC) 
        INCLUDE (total_patterns_detected, avg_processing_time_ms, cache_hit_rate);
    """))
    
    # GIN indexes for JSONB fields
    connection.execute(text("""
        CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_ultra_patterns_metadata_gin 
        ON ultra_optimized_patterns USING GIN (pattern_metadata);
    """))
    
    connection.execute(text("""
        CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_ultra_patterns_indicators_gin 
        ON ultra_optimized_patterns USING GIN (technical_indicators);
    """))
    
    connection.execute(text("""
        CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_ultra_patterns_performance_gin 
        ON ultra_optimized_patterns USING GIN (performance_metrics);
    """))
    
    # Composite indexes for multi-column queries
    connection.execute(text("""
        CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_ultra_patterns_symbol_pattern_conf 
        ON ultra_optimized_patterns (symbol, pattern_name, confidence DESC, timestamp DESC);
    """))
    
    connection.execute(text("""
        CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_ultra_patterns_type_direction 
        ON ultra_optimized_patterns (pattern_type, direction, confidence DESC, timestamp DESC);
    """))
    
    connection.execute(text("""
        CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_ultra_patterns_category_strength 
        ON ultra_optimized_patterns (pattern_category, strength, confidence DESC, timestamp DESC);
    """))
    
    # Functional indexes for computed values
    connection.execute(text("""
        CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_ultra_patterns_confidence_score 
        ON ultra_optimized_patterns ((confidence * CASE 
            WHEN strength = 'strong' THEN 1.0
            WHEN strength = 'moderate' THEN 0.7
            WHEN strength = 'weak' THEN 0.4
            ELSE 0.1
        END)) DESC;
    """))
    
    connection.execute(text("""
        CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_ultra_patterns_volume_score 
        ON ultra_optimized_patterns ((confidence + volume_confidence + multi_timeframe_boost)) DESC;
    """))
    
    # Cache-specific indexes
    connection.execute(text("""
        CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_pattern_cache_expires 
        ON pattern_detection_cache (expires_at) WHERE expires_at < NOW();
    """))
    
    connection.execute(text("""
        CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_pattern_cache_hits 
        ON pattern_detection_cache (hit_count DESC, last_accessed DESC);
    """))
    
    # Buffer-specific indexes
    connection.execute(text("""
        CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_sliding_buffers_active 
        ON sliding_window_buffers (symbol, timeframe) WHERE is_active = true;
    """))
    
    connection.execute(text("""
        CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_sliding_buffers_update_time 
        ON sliding_window_buffers (last_update_time DESC) WHERE is_active = true;
    """))
    
    logger.info("✅ Created advanced indexes for ultra-fast queries")
    
    # 9. Create Continuous Aggregates for Performance Analytics
    connection.execute(text("""
        CREATE MATERIALIZED VIEW IF NOT EXISTS ultra_patterns_hourly_stats
        WITH (timescaledb.continuous) AS
        SELECT 
            time_bucket('1 hour', timestamp) AS bucket,
            symbol,
            timeframe,
            pattern_name,
            COUNT(*) as pattern_count,
            AVG(confidence) as avg_confidence,
            AVG(processing_time_ms) as avg_processing_time,
            COUNT(*) FILTER (WHERE volume_confirmation = true) as volume_confirmed_count,
            COUNT(*) FILTER (WHERE cache_hit = true) as cache_hit_count
        FROM ultra_optimized_patterns
        GROUP BY bucket, symbol, timeframe, pattern_name;
    """))
    
    connection.execute(text("""
        CREATE MATERIALIZED VIEW IF NOT EXISTS ultra_patterns_daily_stats
        WITH (timescaledb.continuous) AS
        SELECT 
            time_bucket('1 day', timestamp) AS bucket,
            symbol,
            pattern_name,
            COUNT(*) as pattern_count,
            AVG(confidence) as avg_confidence,
            AVG(processing_time_ms) as avg_processing_time,
            COUNT(*) FILTER (WHERE volume_confirmation = true) as volume_confirmed_count,
            COUNT(*) FILTER (WHERE cache_hit = true) as cache_hit_count
        FROM ultra_optimized_patterns
        GROUP BY bucket, symbol, pattern_name;
    """))
    
    logger.info("✅ Created continuous aggregates for performance analytics")
    
    # 10. Set up compression policies
    connection.execute(text("""
        ALTER TABLE ultra_optimized_patterns SET (
            timescaledb.compress,
            timescaledb.compress_segmentby = 'symbol,timeframe',
            timescaledb.compress_orderby = 'timestamp DESC'
        );
    """))
    
    connection.execute(text("""
        ALTER TABLE pattern_detection_metrics SET (
            timescaledb.compress,
            timescaledb.compress_segmentby = 'symbol,timeframe',
            timescaledb.compress_orderby = 'timestamp DESC'
        );
    """))
    
    connection.execute(text("""
        ALTER TABLE multi_timeframe_patterns SET (
            timescaledb.compress,
            timescaledb.compress_segmentby = 'symbol',
            timescaledb.compress_orderby = 'timestamp DESC'
        );
    """))
    
    # Add compression policies
    connection.execute(text("""
        SELECT add_compression_policy('ultra_optimized_patterns', INTERVAL '7 days');
    """))
    
    connection.execute(text("""
        SELECT add_compression_policy('pattern_detection_metrics', INTERVAL '7 days');
    """))
    
    connection.execute(text("""
        SELECT add_compression_policy('multi_timeframe_patterns', INTERVAL '7 days');
    """))
    
    logger.info("✅ Set up compression policies")
    
    # 11. Create retention policies
    connection.execute(text("""
        SELECT add_retention_policy('ultra_optimized_patterns', INTERVAL '90 days');
    """))
    
    connection.execute(text("""
        SELECT add_retention_policy('pattern_detection_metrics', INTERVAL '30 days');
    """))
    
    connection.execute(text("""
        SELECT add_retention_policy('pattern_detection_cache', INTERVAL '7 days');
    """))
    
    logger.info("✅ Set up retention policies")
    
    # 12. Create functions for pattern analysis
    connection.execute(text("""
        CREATE OR REPLACE FUNCTION get_pattern_performance_stats(
            p_symbol VARCHAR(20),
            p_timeframe VARCHAR(10),
            p_start_time TIMESTAMPTZ,
            p_end_time TIMESTAMPTZ
        ) RETURNS TABLE (
            pattern_name VARCHAR(100),
            total_count BIGINT,
            avg_confidence NUMERIC,
            avg_processing_time NUMERIC,
            volume_confirmation_rate NUMERIC,
            cache_hit_rate NUMERIC
        ) AS $$
        BEGIN
            RETURN QUERY
            SELECT 
                upp.pattern_name,
                COUNT(*) as total_count,
                AVG(upp.confidence) as avg_confidence,
                AVG(upp.processing_time_ms) as avg_processing_time,
                COUNT(*) FILTER (WHERE upp.volume_confirmation = true)::NUMERIC / COUNT(*) as volume_confirmation_rate,
                COUNT(*) FILTER (WHERE upp.cache_hit = true)::NUMERIC / COUNT(*) as cache_hit_rate
            FROM ultra_optimized_patterns upp
            WHERE upp.symbol = p_symbol
                AND upp.timeframe = p_timeframe
                AND upp.timestamp BETWEEN p_start_time AND p_end_time
            GROUP BY upp.pattern_name
            ORDER BY total_count DESC;
        END;
        $$ LANGUAGE plpgsql;
    """))
    
    connection.execute(text("""
        CREATE OR REPLACE FUNCTION get_cache_performance_stats(
            p_start_time TIMESTAMPTZ DEFAULT NOW() - INTERVAL '24 hours'
        ) RETURNS TABLE (
            cache_hit_rate NUMERIC,
            avg_cache_size_bytes BIGINT,
            total_cache_entries BIGINT,
            expired_entries BIGINT
        ) AS $$
        BEGIN
            RETURN QUERY
            SELECT 
                COUNT(*) FILTER (WHERE pdc.hit_count > 0)::NUMERIC / COUNT(*) as cache_hit_rate,
                AVG(pdc.cache_size_bytes) as avg_cache_size_bytes,
                COUNT(*) as total_cache_entries,
                COUNT(*) FILTER (WHERE pdc.expires_at < NOW()) as expired_entries
            FROM pattern_detection_cache pdc
            WHERE pdc.created_at >= p_start_time;
        END;
        $$ LANGUAGE plpgsql;
    """))
    
    logger.info("✅ Created analysis functions")
    
    logger.info("🎉 Ultra-optimized pattern schema migration completed successfully!")

def downgrade():
    """
    Downgrade ultra-optimized pattern schema
    """
    logger.info("🔄 Starting ultra-optimized pattern schema downgrade")
    
    connection = op.get_bind()
    
    # Drop functions
    connection.execute(text("DROP FUNCTION IF EXISTS get_pattern_performance_stats(VARCHAR, VARCHAR, TIMESTAMPTZ, TIMESTAMPTZ);"))
    connection.execute(text("DROP FUNCTION IF EXISTS get_cache_performance_stats(TIMESTAMPTZ);"))
    
    # Drop continuous aggregates
    connection.execute(text("DROP MATERIALIZED VIEW IF EXISTS ultra_patterns_hourly_stats CASCADE;"))
    connection.execute(text("DROP MATERIALIZED VIEW IF EXISTS ultra_patterns_daily_stats CASCADE;"))
    
    # Drop tables
    op.drop_table('pattern_performance_analytics')
    op.drop_table('multi_timeframe_patterns')
    op.drop_table('pattern_detection_cache')
    op.drop_table('sliding_window_buffers')
    op.drop_table('pattern_detection_metrics')
    op.drop_table('ultra_optimized_patterns')
    
    logger.info("✅ Ultra-optimized pattern schema downgrade completed")
