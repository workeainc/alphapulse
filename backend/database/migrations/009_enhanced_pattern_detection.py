"""
Migration: Enhanced Pattern Detection System
Adds new tables and indexes for ultra-fast pattern detection
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB, TIMESTAMP

# revision identifiers
revision = '009_enhanced_pattern_detection'
down_revision = '008_create_enhanced_data_tables'
depends_on = None

def upgrade():
    """
    Upgrade: Create enhanced pattern detection tables and indexes
    """
    
    # Create enhanced candlestick patterns table with performance optimizations
    op.create_table(
        'enhanced_candlestick_patterns',
        sa.Column('pattern_id', sa.BigInteger(), nullable=False),
        sa.Column('symbol', sa.String(20), nullable=False),
        sa.Column('timeframe', sa.String(10), nullable=False),
        sa.Column('pattern_name', sa.String(100), nullable=False),
        sa.Column('timestamp', TIMESTAMP(timezone=True), nullable=False),
        sa.Column('confidence', sa.Numeric(precision=4, scale=3), nullable=False),
        sa.Column('strength', sa.String(20), nullable=False),  # weak, moderate, strong
        sa.Column('direction', sa.String(10), nullable=False),  # bullish, bearish, neutral
        sa.Column('price_level', sa.Numeric(precision=20, scale=8), nullable=False),
        sa.Column('volume_confirmation', sa.Boolean(), nullable=False, default=False),
        sa.Column('volume_confidence', sa.Numeric(precision=4, scale=3), nullable=False, default=0.0),
        sa.Column('volume_pattern_type', sa.String(50), nullable=True),
        sa.Column('volume_strength', sa.String(20), nullable=True),
        sa.Column('volume_context', JSONB, nullable=True),
        sa.Column('trend_alignment', sa.String(20), nullable=False),
        sa.Column('detection_method', sa.String(20), nullable=False),  # talib, ml, hybrid, vectorized
        sa.Column('ml_confidence', sa.Numeric(precision=4, scale=3), nullable=True),
        sa.Column('talib_confidence', sa.Numeric(precision=4, scale=3), nullable=True),
        sa.Column('noise_filter_passed', sa.Boolean(), nullable=False, default=True),
        sa.Column('atr_percent', sa.Numeric(precision=6, scale=4), nullable=True),
        sa.Column('body_ratio', sa.Numeric(precision=6, scale=4), nullable=True),
        sa.Column('detection_latency_ms', sa.Numeric(precision=8, scale=2), nullable=True),
        sa.Column('metadata', JSONB, nullable=True),
        sa.Column('created_at', TIMESTAMP(timezone=True), nullable=False, server_default=sa.text('NOW()')),
        sa.Column('updated_at', TIMESTAMP(timezone=True), nullable=False, server_default=sa.text('NOW()')),
        sa.PrimaryKeyConstraint('pattern_id')
    )
    
    # Create sliding window cache table for ultra-fast access
    op.create_table(
        'pattern_sliding_windows',
        sa.Column('window_id', sa.BigInteger(), nullable=False),
        sa.Column('symbol', sa.String(20), nullable=False),
        sa.Column('timeframe', sa.String(10), nullable=False),
        sa.Column('window_data', JSONB, nullable=False),  # Cached OHLCV data
        sa.Column('last_updated', TIMESTAMP(timezone=True), nullable=False, server_default=sa.text('NOW()')),
        sa.Column('cache_hits', sa.Integer(), nullable=False, default=0),
        sa.Column('cache_misses', sa.Integer(), nullable=False, default=0),
        sa.PrimaryKeyConstraint('window_id')
    )
    
    # Create ML model metadata table
    op.create_table(
        'pattern_ml_models',
        sa.Column('model_id', sa.BigInteger(), nullable=False),
        sa.Column('pattern_name', sa.String(100), nullable=False),
        sa.Column('model_type', sa.String(50), nullable=False),  # xgboost, random_forest, etc.
        sa.Column('model_version', sa.String(20), nullable=False),
        sa.Column('training_samples', sa.Integer(), nullable=True),
        sa.Column('accuracy_score', sa.Numeric(precision=5, scale=4), nullable=True),
        sa.Column('precision_score', sa.Numeric(precision=5, scale=4), nullable=True),
        sa.Column('recall_score', sa.Numeric(precision=5, scale=4), nullable=True),
        sa.Column('f1_score', sa.Numeric(precision=5, scale=4), nullable=True),
        sa.Column('feature_importance', JSONB, nullable=True),
        sa.Column('model_path', sa.String(500), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=False, default=True),
        sa.Column('created_at', TIMESTAMP(timezone=True), nullable=False, server_default=sa.text('NOW()')),
        sa.Column('updated_at', TIMESTAMP(timezone=True), nullable=False, server_default=sa.text('NOW()')),
        sa.PrimaryKeyConstraint('model_id')
    )
    
    # Create pattern validation table for post-detection validation
    op.create_table(
        'pattern_validations',
        sa.Column('validation_id', sa.BigInteger(), nullable=False),
        sa.Column('pattern_id', sa.BigInteger(), nullable=False),
        sa.Column('follow_through_confirmed', sa.Boolean(), nullable=True),
        sa.Column('volume_expansion', sa.Boolean(), nullable=True),
        sa.Column('price_movement_pct', sa.Numeric(precision=8, scale=4), nullable=True),
        sa.Column('validation_period_hours', sa.Integer(), nullable=True),
        sa.Column('validation_score', sa.Numeric(precision=4, scale=3), nullable=True),
        sa.Column('validation_status', sa.String(20), nullable=False, default='pending'),  # pending, confirmed, failed
        sa.Column('validated_at', TIMESTAMP(timezone=True), nullable=True),
        sa.Column('metadata', JSONB, nullable=True),
        sa.Column('created_at', TIMESTAMP(timezone=True), nullable=False, server_default=sa.text('NOW()')),
        sa.PrimaryKeyConstraint('validation_id'),
        sa.ForeignKeyConstraint(['pattern_id'], ['enhanced_candlestick_patterns.pattern_id'])
    )
    
    # Create multi-symbol correlation table
    op.create_table(
        'pattern_correlations',
        sa.Column('correlation_id', sa.BigInteger(), nullable=False),
        sa.Column('primary_pattern_id', sa.BigInteger(), nullable=False),
        sa.Column('correlated_symbol', sa.String(20), nullable=False),
        sa.Column('correlation_type', sa.String(20), nullable=False),  # btc_dominance, alt_correlation, etc.
        sa.Column('correlation_strength', sa.Numeric(precision=4, scale=3), nullable=False),
        sa.Column('correlation_direction', sa.String(10), nullable=False),  # positive, negative, neutral
        sa.Column('correlation_data', JSONB, nullable=True),
        sa.Column('created_at', TIMESTAMP(timezone=True), nullable=False, server_default=sa.text('NOW()')),
        sa.PrimaryKeyConstraint('correlation_id'),
        sa.ForeignKeyConstraint(['primary_pattern_id'], ['enhanced_candlestick_patterns.pattern_id'])
    )
    
    # Create performance metrics table
    op.create_table(
        'pattern_performance_metrics',
        sa.Column('metric_id', sa.BigInteger(), nullable=False),
        sa.Column('detector_type', sa.String(50), nullable=False),  # vectorized, talib, ml, hybrid
        sa.Column('total_detections', sa.BigInteger(), nullable=False, default=0),
        sa.Column('avg_latency_ms', sa.Numeric(precision=8, scale=2), nullable=True),
        sa.Column('cache_hit_rate', sa.Numeric(precision=5, scale=4), nullable=True),
        sa.Column('throughput_per_sec', sa.Numeric(precision=10, scale=2), nullable=True),
        sa.Column('memory_usage_mb', sa.Numeric(precision=8, scale=2), nullable=True),
        sa.Column('cpu_usage_pct', sa.Numeric(precision=5, scale=2), nullable=True),
        sa.Column('accuracy_score', sa.Numeric(precision=5, scale=4), nullable=True),
        sa.Column('false_positive_rate', sa.Numeric(precision=5, scale=4), nullable=True),
        sa.Column('false_negative_rate', sa.Numeric(precision=5, scale=4), nullable=True),
        sa.Column('metrics_timestamp', TIMESTAMP(timezone=True), nullable=False, server_default=sa.text('NOW()')),
        sa.Column('metadata', JSONB, nullable=True),
        sa.PrimaryKeyConstraint('metric_id')
    )
    
    # Create TimescaleDB hypertables
    op.execute("SELECT create_hypertable('enhanced_candlestick_patterns', 'timestamp', if_not_exists => TRUE)")
    op.execute("SELECT create_hypertable('pattern_performance_metrics', 'metrics_timestamp', if_not_exists => TRUE)")
    
    # Create optimized indexes for ultra-fast queries
    
    # Primary pattern table indexes
    op.create_index('idx_enhanced_patterns_symbol_timeframe_timestamp', 
                   'enhanced_candlestick_patterns', ['symbol', 'timeframe', 'timestamp'])
    op.create_index('idx_enhanced_patterns_pattern_name_confidence', 
                   'enhanced_candlestick_patterns', ['pattern_name', 'confidence'])
    op.create_index('idx_enhanced_patterns_direction_strength', 
                   'enhanced_candlestick_patterns', ['direction', 'strength'])
    op.create_index('idx_enhanced_patterns_detection_method', 
                   'enhanced_candlestick_patterns', ['detection_method'])
    op.create_index('idx_enhanced_patterns_volume_confirmation', 
                   'enhanced_candlestick_patterns', ['volume_confirmation'])
    op.create_index('idx_enhanced_patterns_trend_alignment', 
                   'enhanced_candlestick_patterns', ['trend_alignment'])
    
    # Partial indexes for high-confidence patterns
    op.execute("""
        CREATE INDEX idx_enhanced_patterns_high_confidence 
        ON enhanced_candlestick_patterns (symbol, pattern_name, timestamp DESC) 
        WHERE confidence >= 0.8
    """)
    
    op.execute("""
        CREATE INDEX idx_enhanced_patterns_volume_confirmed 
        ON enhanced_candlestick_patterns (symbol, pattern_name, timestamp DESC) 
        WHERE volume_confirmation = true
    """)
    
    op.execute("""
        CREATE INDEX idx_enhanced_patterns_noise_filtered 
        ON enhanced_candlestick_patterns (symbol, pattern_name, timestamp DESC) 
        WHERE noise_filter_passed = true
    """)
    
    # Covering indexes for common queries
    op.execute("""
        CREATE INDEX idx_enhanced_patterns_covering 
        ON enhanced_candlestick_patterns (symbol, timeframe, timestamp DESC) 
        INCLUDE (pattern_name, confidence, direction, strength, price_level, volume_confirmation)
    """)
    
    # Sliding window cache indexes
    op.create_index('idx_sliding_windows_symbol_timeframe', 
                   'pattern_sliding_windows', ['symbol', 'timeframe'])
    op.create_index('idx_sliding_windows_last_updated', 
                   'pattern_sliding_windows', ['last_updated'])
    
    # ML model indexes
    op.create_index('idx_ml_models_pattern_name', 
                   'pattern_ml_models', ['pattern_name'])
    op.create_index('idx_ml_models_active', 
                   'pattern_ml_models', ['is_active'])
    op.create_index('idx_ml_models_accuracy', 
                   'pattern_ml_models', ['accuracy_score'])
    
    # Validation indexes
    op.create_index('idx_validations_pattern_id', 
                   'pattern_validations', ['pattern_id'])
    op.create_index('idx_validations_status', 
                   'pattern_validations', ['validation_status'])
    op.create_index('idx_validations_score', 
                   'pattern_validations', ['validation_score'])
    
    # Correlation indexes
    op.create_index('idx_correlations_primary_pattern', 
                   'pattern_correlations', ['primary_pattern_id'])
    op.create_index('idx_correlations_symbol', 
                   'pattern_correlations', ['correlated_symbol'])
    op.create_index('idx_correlations_strength', 
                   'pattern_correlations', ['correlation_strength'])
    
    # Performance metrics indexes
    op.create_index('idx_performance_detector_type', 
                   'pattern_performance_metrics', ['detector_type'])
    op.create_index('idx_performance_timestamp', 
                   'pattern_performance_metrics', ['metrics_timestamp'])
    
    # GIN indexes for JSONB fields
    op.execute("CREATE INDEX idx_enhanced_patterns_metadata_gin ON enhanced_candlestick_patterns USING GIN (metadata)")
    op.execute("CREATE INDEX idx_enhanced_patterns_volume_context_gin ON enhanced_candlestick_patterns USING GIN (volume_context)")
    op.execute("CREATE INDEX idx_sliding_windows_data_gin ON pattern_sliding_windows USING GIN (window_data)")
    op.execute("CREATE INDEX idx_ml_models_feature_importance_gin ON pattern_ml_models USING GIN (feature_importance)")
    op.execute("CREATE INDEX idx_correlations_data_gin ON pattern_correlations USING GIN (correlation_data)")
    op.execute("CREATE INDEX idx_performance_metadata_gin ON pattern_performance_metrics USING GIN (metadata)")
    
    # BRIN indexes for time-series data
    op.execute("""
        CREATE INDEX idx_enhanced_patterns_timestamp_brin 
        ON enhanced_candlestick_patterns USING BRIN (timestamp) 
        WITH (pages_per_range = 128)
    """)
    
    op.execute("""
        CREATE INDEX idx_performance_timestamp_brin 
        ON pattern_performance_metrics USING BRIN (metrics_timestamp) 
        WITH (pages_per_range = 128)
    """)
    
    # Create compression policies for TimescaleDB
    op.execute("""
        SELECT add_compression_policy('enhanced_candlestick_patterns', INTERVAL '7 days')
    """)
    
    op.execute("""
        SELECT add_compression_policy('pattern_performance_metrics', INTERVAL '30 days')
    """)
    
    # Create retention policies
    op.execute("""
        SELECT add_retention_policy('enhanced_candlestick_patterns', INTERVAL '90 days')
    """)
    
    op.execute("""
        SELECT add_retention_policy('pattern_performance_metrics', INTERVAL '365 days')
    """)

def downgrade():
    """
    Downgrade: Remove enhanced pattern detection tables and indexes
    """
    
    # Drop compression and retention policies
    op.execute("SELECT remove_compression_policy('enhanced_candlestick_patterns')")
    op.execute("SELECT remove_compression_policy('pattern_performance_metrics')")
    op.execute("SELECT remove_retention_policy('enhanced_candlestick_patterns')")
    op.execute("SELECT remove_retention_policy('pattern_performance_metrics')")
    
    # Drop indexes
    op.drop_index('idx_enhanced_patterns_symbol_timeframe_timestamp')
    op.drop_index('idx_enhanced_patterns_pattern_name_confidence')
    op.drop_index('idx_enhanced_patterns_direction_strength')
    op.drop_index('idx_enhanced_patterns_detection_method')
    op.drop_index('idx_enhanced_patterns_volume_confirmation')
    op.drop_index('idx_enhanced_patterns_trend_alignment')
    op.drop_index('idx_sliding_windows_symbol_timeframe')
    op.drop_index('idx_sliding_windows_last_updated')
    op.drop_index('idx_ml_models_pattern_name')
    op.drop_index('idx_ml_models_active')
    op.drop_index('idx_ml_models_accuracy')
    op.drop_index('idx_validations_pattern_id')
    op.drop_index('idx_validations_status')
    op.drop_index('idx_validations_score')
    op.drop_index('idx_correlations_primary_pattern')
    op.drop_index('idx_correlations_symbol')
    op.drop_index('idx_correlations_strength')
    op.drop_index('idx_performance_detector_type')
    op.drop_index('idx_performance_timestamp')
    
    # Drop partial indexes
    op.execute("DROP INDEX IF EXISTS idx_enhanced_patterns_high_confidence")
    op.execute("DROP INDEX IF EXISTS idx_enhanced_patterns_volume_confirmed")
    op.execute("DROP INDEX IF EXISTS idx_enhanced_patterns_noise_filtered")
    op.execute("DROP INDEX IF EXISTS idx_enhanced_patterns_covering")
    
    # Drop GIN indexes
    op.execute("DROP INDEX IF EXISTS idx_enhanced_patterns_metadata_gin")
    op.execute("DROP INDEX IF EXISTS idx_enhanced_patterns_volume_context_gin")
    op.execute("DROP INDEX IF EXISTS idx_sliding_windows_data_gin")
    op.execute("DROP INDEX IF EXISTS idx_ml_models_feature_importance_gin")
    op.execute("DROP INDEX IF EXISTS idx_correlations_data_gin")
    op.execute("DROP INDEX IF EXISTS idx_performance_metadata_gin")
    
    # Drop BRIN indexes
    op.execute("DROP INDEX IF EXISTS idx_enhanced_patterns_timestamp_brin")
    op.execute("DROP INDEX IF EXISTS idx_performance_timestamp_brin")
    
    # Drop tables
    op.drop_table('pattern_performance_metrics')
    op.drop_table('pattern_correlations')
    op.drop_table('pattern_validations')
    op.drop_table('pattern_ml_models')
    op.drop_table('pattern_sliding_windows')
    op.drop_table('enhanced_candlestick_patterns')
