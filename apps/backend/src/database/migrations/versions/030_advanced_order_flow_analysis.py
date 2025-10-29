"""
Migration: Advanced Order Flow Analysis Tables
Add comprehensive tables for advanced order flow analysis including toxicity, market maker vs taker, large orders, and patterns
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB, TIMESTAMP, NUMERIC

# revision identifiers, used by Alembic.
revision = '030_advanced_order_flow_analysis'
down_revision = '029_dynamic_support_resistance'
branch_labels = None
depends_on = None

def upgrade():
    """Create advanced order flow analysis tables"""
    
    # ==================== ORDER FLOW TOXICITY ANALYSIS ====================
    
    # Order flow toxicity analysis table
    op.create_table(
        'order_flow_toxicity_analysis',
        sa.Column('id', sa.BigInteger(), nullable=False),
        sa.Column('symbol', sa.String(20), nullable=False),
        sa.Column('timeframe', sa.String(10), nullable=False),
        sa.Column('timestamp', TIMESTAMP(timezone=True), nullable=False),
        sa.Column('toxicity_score', NUMERIC(10, 6), nullable=False),  # -1 to +1
        sa.Column('bid_toxicity', NUMERIC(10, 6), nullable=True),  # -1 to +1
        sa.Column('ask_toxicity', NUMERIC(10, 6), nullable=True),  # -1 to +1
        sa.Column('large_order_ratio', NUMERIC(10, 6), nullable=True),  # 0 to 1
        sa.Column('order_size_distribution', JSONB, nullable=True),  # Size distribution stats
        sa.Column('toxicity_trend', sa.String(20), nullable=True),  # 'increasing', 'decreasing', 'stable'
        sa.Column('toxicity_confidence', NUMERIC(4, 3), nullable=True),  # 0 to 1
        sa.Column('market_impact_score', NUMERIC(10, 6), nullable=True),  # 0 to 1
        sa.Column('analysis_metadata', JSONB, nullable=True),
        sa.Column('created_at', TIMESTAMP(timezone=True), server_default=sa.text('NOW()'), nullable=False),
        sa.PrimaryKeyConstraint('id', 'timestamp')
    )
    
    # Create TimescaleDB hypertable for toxicity analysis
    op.execute("""
        SELECT create_hypertable('order_flow_toxicity_analysis', 'timestamp', 
            if_not_exists => TRUE, 
            chunk_time_interval => INTERVAL '1 hour'
        );
    """)
    
    # Create indexes for toxicity analysis
    op.execute("CREATE INDEX IF NOT EXISTS idx_toxicity_symbol_timeframe ON order_flow_toxicity_analysis (symbol, timeframe)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_toxicity_score ON order_flow_toxicity_analysis (toxicity_score)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_toxicity_trend ON order_flow_toxicity_analysis (toxicity_trend)")
    
    # ==================== MARKET MAKER VS TAKER ANALYSIS ====================
    
    # Market maker vs taker analysis table
    op.create_table(
        'market_maker_taker_analysis',
        sa.Column('id', sa.BigInteger(), nullable=False),
        sa.Column('symbol', sa.String(20), nullable=False),
        sa.Column('timeframe', sa.String(10), nullable=False),
        sa.Column('timestamp', TIMESTAMP(timezone=True), nullable=False),
        sa.Column('maker_volume_ratio', NUMERIC(10, 6), nullable=False),  # 0 to 1
        sa.Column('taker_volume_ratio', NUMERIC(10, 6), nullable=False),  # 0 to 1
        sa.Column('maker_buy_volume', NUMERIC(20, 8), nullable=True),
        sa.Column('maker_sell_volume', NUMERIC(20, 8), nullable=True),
        sa.Column('taker_buy_volume', NUMERIC(20, 8), nullable=True),
        sa.Column('taker_sell_volume', NUMERIC(20, 8), nullable=True),
        sa.Column('maker_taker_imbalance', NUMERIC(10, 6), nullable=True),  # -1 to +1
        sa.Column('market_maker_activity', sa.String(20), nullable=True),  # 'high', 'medium', 'low'
        sa.Column('taker_aggression', sa.String(20), nullable=True),  # 'high', 'medium', 'low'
        sa.Column('spread_impact', NUMERIC(10, 6), nullable=True),  # Impact on spread
        sa.Column('liquidity_provision_score', NUMERIC(10, 6), nullable=True),  # 0 to 1
        sa.Column('analysis_metadata', JSONB, nullable=True),
        sa.Column('created_at', TIMESTAMP(timezone=True), server_default=sa.text('NOW()'), nullable=False),
        sa.PrimaryKeyConstraint('id', 'timestamp')
    )
    
    # Create TimescaleDB hypertable for maker/taker analysis
    op.execute("""
        SELECT create_hypertable('market_maker_taker_analysis', 'timestamp', 
            if_not_exists => TRUE, 
            chunk_time_interval => INTERVAL '1 hour'
        );
    """)
    
    # Create indexes for maker/taker analysis
    op.execute("CREATE INDEX IF NOT EXISTS idx_maker_taker_symbol_timeframe ON market_maker_taker_analysis (symbol, timeframe)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_maker_volume_ratio ON market_maker_taker_analysis (maker_volume_ratio)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_taker_volume_ratio ON market_maker_taker_analysis (taker_volume_ratio)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_maker_taker_imbalance ON market_maker_taker_analysis (maker_taker_imbalance)")
    
    # ==================== LARGE ORDER TRACKING ====================
    
    # Large order tracking table
    op.create_table(
        'large_order_tracking',
        sa.Column('id', sa.BigInteger(), nullable=False),
        sa.Column('symbol', sa.String(20), nullable=False),
        sa.Column('timestamp', TIMESTAMP(timezone=True), nullable=False),
        sa.Column('order_id', sa.String(100), nullable=True),  # Exchange order ID if available
        sa.Column('side', sa.String(10), nullable=False),  # 'buy' or 'sell'
        sa.Column('price', NUMERIC(20, 8), nullable=False),
        sa.Column('quantity', NUMERIC(20, 8), nullable=False),
        sa.Column('quote_quantity', NUMERIC(20, 8), nullable=False),
        sa.Column('order_type', sa.String(20), nullable=True),  # 'market', 'limit', 'stop'
        sa.Column('size_category', sa.String(20), nullable=False),  # 'large', 'very_large', 'whale'
        sa.Column('size_percentile', NUMERIC(10, 6), nullable=True),  # 0 to 1
        sa.Column('market_impact', NUMERIC(10, 6), nullable=True),  # 0 to 1
        sa.Column('execution_time', NUMERIC(10, 3), nullable=True),  # Seconds
        sa.Column('fill_ratio', NUMERIC(10, 6), nullable=True),  # 0 to 1
        sa.Column('slippage', NUMERIC(10, 6), nullable=True),  # Percentage
        sa.Column('order_flow_pattern', sa.String(50), nullable=True),  # Pattern classification
        sa.Column('institutional_indicator', sa.Boolean(), nullable=True),  # Likely institutional
        sa.Column('analysis_metadata', JSONB, nullable=True),
        sa.Column('created_at', TIMESTAMP(timezone=True), server_default=sa.text('NOW()'), nullable=False),
        sa.PrimaryKeyConstraint('id', 'timestamp')
    )
    
    # Create TimescaleDB hypertable for large order tracking
    op.execute("""
        SELECT create_hypertable('large_order_tracking', 'timestamp', 
            if_not_exists => TRUE, 
            chunk_time_interval => INTERVAL '30 minutes'
        );
    """)
    
    # Create indexes for large order tracking
    op.execute("CREATE INDEX IF NOT EXISTS idx_large_order_symbol_time ON large_order_tracking (symbol, timestamp)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_large_order_side ON large_order_tracking (side)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_large_order_size_category ON large_order_tracking (size_category)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_large_order_pattern ON large_order_tracking (order_flow_pattern)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_large_order_institutional ON large_order_tracking (institutional_indicator)")
    
    # ==================== ORDER FLOW PATTERNS ====================
    
    # Order flow patterns table
    op.create_table(
        'order_flow_patterns',
        sa.Column('id', sa.BigInteger(), nullable=False),
        sa.Column('symbol', sa.String(20), nullable=False),
        sa.Column('timeframe', sa.String(10), nullable=False),
        sa.Column('timestamp', TIMESTAMP(timezone=True), nullable=False),
        sa.Column('pattern_type', sa.String(50), nullable=False),  # 'absorption', 'distribution', 'accumulation', etc.
        sa.Column('pattern_confidence', NUMERIC(4, 3), nullable=False),  # 0 to 1
        sa.Column('pattern_strength', NUMERIC(4, 3), nullable=True),  # 0 to 1
        sa.Column('volume_profile', JSONB, nullable=True),  # Volume distribution analysis
        sa.Column('price_action', JSONB, nullable=True),  # Price movement during pattern
        sa.Column('order_flow_signature', JSONB, nullable=True),  # Order flow characteristics
        sa.Column('duration_minutes', sa.Integer(), nullable=True),  # Pattern duration
        sa.Column('breakout_direction', sa.String(10), nullable=True),  # 'up', 'down', 'none'
        sa.Column('breakout_strength', NUMERIC(4, 3), nullable=True),  # 0 to 1
        sa.Column('pattern_completion', sa.Boolean(), nullable=True),  # Pattern completed
        sa.Column('analysis_metadata', JSONB, nullable=True),
        sa.Column('created_at', TIMESTAMP(timezone=True), server_default=sa.text('NOW()'), nullable=False),
        sa.PrimaryKeyConstraint('id', 'timestamp')
    )
    
    # Create TimescaleDB hypertable for order flow patterns
    op.execute("""
        SELECT create_hypertable('order_flow_patterns', 'timestamp', 
            if_not_exists => TRUE, 
            chunk_time_interval => INTERVAL '1 hour'
        );
    """)
    
    # Create indexes for order flow patterns
    op.execute("CREATE INDEX IF NOT EXISTS idx_patterns_symbol_timeframe ON order_flow_patterns (symbol, timeframe)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_patterns_type ON order_flow_patterns (pattern_type)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_patterns_confidence ON order_flow_patterns (pattern_confidence)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_patterns_breakout ON order_flow_patterns (breakout_direction)")
    
    # ==================== REAL-TIME ORDER FLOW MONITORING ====================
    
    # Real-time order flow monitoring table
    op.create_table(
        'real_time_order_flow_monitoring',
        sa.Column('id', sa.BigInteger(), nullable=False),
        sa.Column('symbol', sa.String(20), nullable=False),
        sa.Column('timestamp', TIMESTAMP(timezone=True), nullable=False),
        sa.Column('monitoring_type', sa.String(50), nullable=False),  # 'toxicity', 'maker_taker', 'large_orders', 'patterns'
        sa.Column('alert_level', sa.String(20), nullable=False),  # 'low', 'medium', 'high', 'critical'
        sa.Column('alert_message', sa.Text(), nullable=True),
        sa.Column('metric_value', NUMERIC(20, 8), nullable=True),
        sa.Column('threshold_value', NUMERIC(20, 8), nullable=True),
        sa.Column('alert_triggered', sa.Boolean(), nullable=False),
        sa.Column('alert_acknowledged', sa.Boolean(), default=False),
        sa.Column('alert_resolved', sa.Boolean(), default=False),
        sa.Column('resolution_time', TIMESTAMP(timezone=True), nullable=True),
        sa.Column('alert_metadata', JSONB, nullable=True),
        sa.Column('created_at', TIMESTAMP(timezone=True), server_default=sa.text('NOW()'), nullable=False),
        sa.PrimaryKeyConstraint('id', 'timestamp')
    )
    
    # Create TimescaleDB hypertable for real-time monitoring
    op.execute("""
        SELECT create_hypertable('real_time_order_flow_monitoring', 'timestamp', 
            if_not_exists => TRUE, 
            chunk_time_interval => INTERVAL '15 minutes'
        );
    """)
    
    # Create indexes for real-time monitoring
    op.execute("CREATE INDEX IF NOT EXISTS idx_monitoring_symbol_time ON real_time_order_flow_monitoring (symbol, timestamp)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_monitoring_type ON real_time_order_flow_monitoring (monitoring_type)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_monitoring_alert_level ON real_time_order_flow_monitoring (alert_level)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_monitoring_triggered ON real_time_order_flow_monitoring (alert_triggered)")
    
    # ==================== ORDER FLOW AGGREGATES ====================
    
    # Order flow aggregates table for performance optimization
    op.create_table(
        'order_flow_aggregates',
        sa.Column('id', sa.BigInteger(), nullable=False),
        sa.Column('symbol', sa.String(20), nullable=False),
        sa.Column('timeframe', sa.String(10), nullable=False),
        sa.Column('timestamp', TIMESTAMP(timezone=True), nullable=False),
        sa.Column('aggregate_type', sa.String(50), nullable=False),  # 'hourly', 'daily', 'weekly'
        sa.Column('avg_toxicity', NUMERIC(10, 6), nullable=True),
        sa.Column('max_toxicity', NUMERIC(10, 6), nullable=True),
        sa.Column('avg_maker_ratio', NUMERIC(10, 6), nullable=True),
        sa.Column('large_order_count', sa.Integer(), nullable=True),
        sa.Column('pattern_count', sa.Integer(), nullable=True),
        sa.Column('total_volume', NUMERIC(20, 8), nullable=True),
        sa.Column('volume_distribution', JSONB, nullable=True),
        sa.Column('price_impact_stats', JSONB, nullable=True),
        sa.Column('aggregate_metadata', JSONB, nullable=True),
        sa.Column('created_at', TIMESTAMP(timezone=True), server_default=sa.text('NOW()'), nullable=False),
        sa.PrimaryKeyConstraint('id', 'timestamp')
    )
    
    # Create TimescaleDB hypertable for aggregates
    op.execute("""
        SELECT create_hypertable('order_flow_aggregates', 'timestamp', 
            if_not_exists => TRUE, 
            chunk_time_interval => INTERVAL '1 day'
        );
    """)
    
    # Create indexes for aggregates
    op.execute("CREATE INDEX IF NOT EXISTS idx_aggregates_symbol_timeframe ON order_flow_aggregates (symbol, timeframe)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_aggregates_type ON order_flow_aggregates (aggregate_type)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_aggregates_toxicity ON order_flow_aggregates (avg_toxicity)")

def downgrade():
    """Drop advanced order flow analysis tables"""
    
    # Drop aggregates table
    op.drop_table('order_flow_aggregates')
    
    # Drop real-time monitoring table
    op.drop_table('real_time_order_flow_monitoring')
    
    # Drop order flow patterns table
    op.drop_table('order_flow_patterns')
    
    # Drop large order tracking table
    op.drop_table('large_order_tracking')
    
    # Drop market maker taker analysis table
    op.drop_table('market_maker_taker_analysis')
    
    # Drop order flow toxicity analysis table
    op.drop_table('order_flow_toxicity_analysis')
