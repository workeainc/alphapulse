"""
Migration: Ultra-Low Latency Pattern Storage Schema
Enhanced TimescaleDB schema for vectorized pattern detection with optimized indexes
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB, TIMESTAMP, UUID
import uuid

# revision identifiers, used by Alembic.
revision = '009_ultra_low_latency_pattern_schema'
down_revision = '008_create_enhanced_data_tables'
branch_labels = None
depends_on = None

def upgrade():
    """Create ultra-low latency pattern storage schema"""
    
    # Ultra-Low Latency Pattern Detection Table
    op.create_table(
        'ultra_low_latency_patterns',
        sa.Column('pattern_id', UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column('symbol', sa.String(20), nullable=False),
        sa.Column('timeframe', sa.String(10), nullable=False),
        sa.Column('pattern_name', sa.String(100), nullable=False),
        sa.Column('pattern_type', sa.String(20), nullable=False),  # 'bullish', 'bearish', 'neutral'
        sa.Column('confidence', sa.Numeric(precision=4, scale=3), nullable=False),
        sa.Column('strength', sa.String(20), nullable=False),  # 'weak', 'moderate', 'strong'
        sa.Column('timestamp', TIMESTAMP(timezone=True), nullable=False),
        sa.Column('price_level', sa.Numeric(precision=20, scale=8), nullable=False),
        sa.Column('volume_confirmation', sa.Boolean, nullable=False, default=False),
        sa.Column('volume_confidence', sa.Numeric(precision=4, scale=3), nullable=False, default=0.0),
        sa.Column('trend_alignment', sa.String(20), nullable=False, default='neutral'),
        sa.Column('detection_method', sa.String(50), nullable=False, default='vectorized'),  # 'talib', 'vectorized', 'ensemble'
        sa.Column('processing_latency_ms', sa.Integer, nullable=True),  # Track processing speed
        sa.Column('metadata', JSONB, nullable=True),
        sa.Column('created_at', TIMESTAMP(timezone=True), nullable=False, server_default=sa.text('NOW()')),
        sa.Column('updated_at', TIMESTAMP(timezone=True), nullable=False, server_default=sa.text('NOW()'))
    )
    
    # Convert to TimescaleDB hypertable for ultra-fast time-series queries
    op.execute("""
        SELECT create_hypertable(
            'ultra_low_latency_patterns', 
            'timestamp',
            chunk_time_interval => INTERVAL '1 hour',
            if_not_exists => TRUE
        );
    """)
    
    # Add space dimension for parallel processing
    op.execute("""
        SELECT add_dimension(
            'ultra_low_latency_patterns', 
            'symbol', 
            chunk_time_interval => INTERVAL '1 hour',
            if_not_exists => TRUE
        );
    """)
    
    # Ultra-Low Latency Signal Generation Table
    op.create_table(
        'ultra_low_latency_signals',
        sa.Column('signal_id', UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column('pattern_id', UUID(as_uuid=True), nullable=True),
        sa.Column('symbol', sa.String(20), nullable=False),
        sa.Column('timeframe', sa.String(10), nullable=False),
        sa.Column('signal_type', sa.String(20), nullable=False),  # 'buy', 'sell', 'hold'
        sa.Column('confidence', sa.Numeric(precision=4, scale=3), nullable=False),
        sa.Column('strength', sa.String(20), nullable=False),
        sa.Column('entry_price', sa.Numeric(precision=20, scale=8), nullable=False),
        sa.Column('stop_loss', sa.Numeric(precision=20, scale=8), nullable=True),
        sa.Column('take_profit', sa.Numeric(precision=20, scale=8), nullable=True),
        sa.Column('risk_reward_ratio', sa.Numeric(precision=6, scale=2), nullable=True),
        sa.Column('timestamp', TIMESTAMP(timezone=True), nullable=False),
        sa.Column('processing_latency_ms', sa.Integer, nullable=True),
        sa.Column('ensemble_score', sa.Numeric(precision=4, scale=3), nullable=True),  # Multi-model ensemble
        sa.Column('market_regime', sa.String(50), nullable=True),  # 'trending', 'ranging', 'volatile'
        sa.Column('volatility_context', sa.Numeric(precision=6, scale=3), nullable=True),
        sa.Column('metadata', JSONB, nullable=True),
        sa.Column('created_at', TIMESTAMP(timezone=True), nullable=False, server_default=sa.text('NOW()')),
        sa.Column('updated_at', TIMESTAMP(timezone=True), nullable=False, server_default=sa.text('NOW()'))
    )
    
    # Convert to TimescaleDB hypertable
    op.execute("""
        SELECT create_hypertable(
            'ultra_low_latency_signals', 
            'timestamp',
            chunk_time_interval => INTERVAL '1 hour',
            if_not_exists => TRUE
        );
    """)
    
    # Add space dimension
    op.execute("""
        SELECT add_dimension(
            'ultra_low_latency_signals', 
            'symbol', 
            chunk_time_interval => INTERVAL '1 hour',
            if_not_exists => TRUE
        );
    """)
    
    # Performance Metrics Table
    op.create_table(
        'ultra_low_latency_performance',
        sa.Column('metric_id', UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column('component', sa.String(50), nullable=False),  # 'websocket', 'pattern_detection', 'signal_generation'
        sa.Column('metric_name', sa.String(100), nullable=False),  # 'avg_latency_ms', 'throughput_per_sec', 'error_rate'
        sa.Column('metric_value', sa.Numeric(precision=10, scale=3), nullable=False),
        sa.Column('metric_unit', sa.String(20), nullable=True),  # 'ms', 'count', 'percentage'
        sa.Column('timestamp', TIMESTAMP(timezone=True), nullable=False),
        sa.Column('metadata', JSONB, nullable=True),
        sa.Column('created_at', TIMESTAMP(timezone=True), nullable=False, server_default=sa.text('NOW()'))
    )
    
    # Convert to TimescaleDB hypertable
    op.execute("""
        SELECT create_hypertable(
            'ultra_low_latency_performance', 
            'timestamp',
            chunk_time_interval => INTERVAL '1 hour',
            if_not_exists => TRUE
        );
    """)
    
    # Shared Memory Buffer Status Table
    op.create_table(
        'shared_memory_buffers',
        sa.Column('buffer_id', UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column('buffer_name', sa.String(100), nullable=False, unique=True),
        sa.Column('buffer_type', sa.String(50), nullable=False),  # 'redis_stream', 'memory_buffer'
        sa.Column('current_size', sa.Integer, nullable=False, default=0),
        sa.Column('max_size', sa.Integer, nullable=False),
        sa.Column('overflow_count', sa.Integer, nullable=False, default=0),
        sa.Column('last_updated', TIMESTAMP(timezone=True), nullable=False),
        sa.Column('status', sa.String(20), nullable=False, default='active'),  # 'active', 'overflow', 'error'
        sa.Column('metadata', JSONB, nullable=True),
        sa.Column('created_at', TIMESTAMP(timezone=True), nullable=False, server_default=sa.text('NOW()')),
        sa.Column('updated_at', TIMESTAMP(timezone=True), nullable=False, server_default=sa.text('NOW()'))
    )
    
    # Create optimized indexes for ultra-low latency queries
    
    # BRIN indexes for time-series data (ultra-fast, low space)
    op.execute("""
        CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_ultra_patterns_timestamp_brin 
        ON ultra_low_latency_patterns USING BRIN (timestamp) 
        WITH (pages_per_range = 128);
    """)
    
    op.execute("""
        CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_ultra_signals_timestamp_brin 
        ON ultra_low_latency_signals USING BRIN (timestamp) 
        WITH (pages_per_range = 128);
    """)
    
    op.execute("""
        CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_ultra_performance_timestamp_brin 
        ON ultra_low_latency_performance USING BRIN (timestamp) 
        WITH (pages_per_range = 128);
    """)
    
    # Partial indexes for high-confidence patterns only
    op.execute("""
        CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_ultra_patterns_high_confidence 
        ON ultra_low_latency_patterns (symbol, pattern_name, timestamp DESC) 
        WHERE confidence >= 0.8;
    """)
    
    op.execute("""
        CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_ultra_signals_high_confidence 
        ON ultra_low_latency_signals (symbol, signal_type, timestamp DESC) 
        WHERE confidence >= 0.8;
    """)
    
    # Covering indexes for common queries (INCLUDE all needed columns)
    op.execute("""
        CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_ultra_patterns_covering 
        ON ultra_low_latency_patterns (symbol, timestamp DESC) 
        INCLUDE (pattern_name, pattern_type, confidence, strength, price_level, detection_method);
    """)
    
    op.execute("""
        CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_ultra_signals_covering 
        ON ultra_low_latency_signals (symbol, timestamp DESC) 
        INCLUDE (signal_type, confidence, strength, entry_price, stop_loss, take_profit, ensemble_score);
    """)
    
    # GIN indexes for JSONB metadata fields
    op.execute("""
        CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_ultra_patterns_metadata_gin 
        ON ultra_low_latency_patterns USING GIN (metadata);
    """)
    
    op.execute("""
        CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_ultra_signals_metadata_gin 
        ON ultra_low_latency_signals USING GIN (metadata);
    """)
    
    # Composite indexes for multi-column queries
    op.execute("""
        CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_ultra_patterns_symbol_pattern_time 
        ON ultra_low_latency_patterns (symbol, pattern_name, timestamp DESC);
    """)
    
    op.execute("""
        CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_ultra_signals_symbol_type_time 
        ON ultra_low_latency_signals (symbol, signal_type, timestamp DESC);
    """)
    
    # Functional indexes for computed values
    op.execute("""
        CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_ultra_patterns_confidence_score 
        ON ultra_low_latency_patterns ((confidence * CASE 
            WHEN strength = 'strong' THEN 1.0
            WHEN strength = 'moderate' THEN 0.7
            WHEN strength = 'weak' THEN 0.4
            ELSE 0.1
        END)) DESC;
    """)
    
    # Partitioned indexes for recent data only
    op.execute("""
        CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_ultra_patterns_recent 
        ON ultra_low_latency_patterns (symbol, timestamp DESC) 
        WHERE timestamp >= NOW() - INTERVAL '24 hours';
    """)
    
    op.execute("""
        CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_ultra_signals_recent 
        ON ultra_low_latency_signals (symbol, timestamp DESC) 
        WHERE timestamp >= NOW() - INTERVAL '24 hours';
    """)
    
    # Create continuous aggregates for pre-computed statistics
    op.execute("""
        CREATE MATERIALIZED VIEW ultra_patterns_hourly_stats
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
    
    op.execute("""
        CREATE MATERIALIZED VIEW ultra_signals_hourly_stats
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
    
    # Create compression policies for older data
    op.execute("""
        ALTER TABLE ultra_low_latency_patterns SET (
            timescaledb.compress,
            timescaledb.compress_segmentby = 'symbol',
            timescaledb.compress_orderby = 'timestamp DESC'
        );
    """)
    
    op.execute("""
        ALTER TABLE ultra_low_latency_signals SET (
            timescaledb.compress,
            timescaledb.compress_segmentby = 'symbol',
            timescaledb.compress_orderby = 'timestamp DESC'
        );
    """)
    
    # Set compression policies (compress data older than 1 day)
    op.execute("""
        SELECT add_compression_policy('ultra_low_latency_patterns', INTERVAL '1 day');
    """)
    
    op.execute("""
        SELECT add_compression_policy('ultra_low_latency_signals', INTERVAL '1 day');
    """)
    
    # Set retention policies (keep data for 30 days)
    op.execute("""
        SELECT add_retention_policy('ultra_low_latency_patterns', INTERVAL '30 days');
    """)
    
    op.execute("""
        SELECT add_retention_policy('ultra_low_latency_signals', INTERVAL '30 days');
    """)
    
    # Insert initial shared memory buffer configurations
    op.execute("""
        INSERT INTO shared_memory_buffers (buffer_name, buffer_type, max_size, last_updated, status) VALUES
        ('candlestick_data', 'redis_stream', 1000, NOW(), 'active'),
        ('pattern_detection', 'redis_stream', 500, NOW(), 'active'),
        ('signal_generation', 'redis_stream', 200, NOW(), 'active'),
        ('market_analysis', 'redis_stream', 1000, NOW(), 'active');
    """)

def downgrade():
    """Rollback ultra-low latency pattern storage schema"""
    
    # Drop continuous aggregates
    op.execute("DROP MATERIALIZED VIEW IF EXISTS ultra_signals_hourly_stats CASCADE;")
    op.execute("DROP MATERIALIZED VIEW IF EXISTS ultra_patterns_hourly_stats CASCADE;")
    
    # Drop indexes
    op.execute("DROP INDEX IF EXISTS idx_ultra_patterns_timestamp_brin;")
    op.execute("DROP INDEX IF EXISTS idx_ultra_signals_timestamp_brin;")
    op.execute("DROP INDEX IF EXISTS idx_ultra_performance_timestamp_brin;")
    op.execute("DROP INDEX IF EXISTS idx_ultra_patterns_high_confidence;")
    op.execute("DROP INDEX IF EXISTS idx_ultra_signals_high_confidence;")
    op.execute("DROP INDEX IF EXISTS idx_ultra_patterns_covering;")
    op.execute("DROP INDEX IF EXISTS idx_ultra_signals_covering;")
    op.execute("DROP INDEX IF EXISTS idx_ultra_patterns_metadata_gin;")
    op.execute("DROP INDEX IF EXISTS idx_ultra_signals_metadata_gin;")
    op.execute("DROP INDEX IF EXISTS idx_ultra_patterns_symbol_pattern_time;")
    op.execute("DROP INDEX IF EXISTS idx_ultra_signals_symbol_type_time;")
    op.execute("DROP INDEX IF EXISTS idx_ultra_patterns_confidence_score;")
    op.execute("DROP INDEX IF EXISTS idx_ultra_patterns_recent;")
    op.execute("DROP INDEX IF EXISTS idx_ultra_signals_recent;")
    
    # Drop tables
    op.drop_table('shared_memory_buffers')
    op.drop_table('ultra_low_latency_performance')
    op.drop_table('ultra_low_latency_signals')
    op.drop_table('ultra_low_latency_patterns')
