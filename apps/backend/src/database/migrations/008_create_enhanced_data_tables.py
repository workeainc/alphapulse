"""
Migration: Create Enhanced Data Tables for Advanced Pattern Detection
TimescaleDB schema for storing comprehensive market data and pattern analysis
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB, TIMESTAMP

# revision identifiers, used by Alembic.
revision = '008_create_enhanced_data_tables'
down_revision = '007_create_alphapulse_test_tables'
branch_labels = None
depends_on = None

def upgrade():
    """Create enhanced data tables for advanced pattern detection"""
    
    # Enhanced Market Data Table (TimescaleDB hypertable)
    op.create_table(
        'enhanced_market_data',
        sa.Column('id', sa.BigInteger(), nullable=False),
        sa.Column('symbol', sa.String(20), nullable=False),
        sa.Column('timeframe', sa.String(10), nullable=False),
        sa.Column('timestamp', TIMESTAMP(timezone=True), nullable=False),
        sa.Column('open', sa.Numeric(precision=20, scale=8), nullable=False),
        sa.Column('high', sa.Numeric(precision=20, scale=8), nullable=False),
        sa.Column('low', sa.Numeric(precision=20, scale=8), nullable=False),
        sa.Column('close', sa.Numeric(precision=20, scale=8), nullable=False),
        sa.Column('volume', sa.Numeric(precision=20, scale=8), nullable=False),
        sa.Column('price_change', sa.Numeric(precision=10, scale=6), nullable=True),
        sa.Column('volume_change', sa.Numeric(precision=10, scale=6), nullable=True),
        sa.Column('volatility', sa.Numeric(precision=10, scale=6), nullable=True),
        sa.Column('rsi', sa.Numeric(precision=6, scale=3), nullable=True),
        sa.Column('macd', sa.Numeric(precision=10, scale=6), nullable=True),
        sa.Column('macd_signal', sa.Numeric(precision=10, scale=6), nullable=True),
        sa.Column('bollinger_upper', sa.Numeric(precision=20, scale=8), nullable=True),
        sa.Column('bollinger_lower', sa.Numeric(precision=20, scale=8), nullable=True),
        sa.Column('bollinger_middle', sa.Numeric(precision=20, scale=8), nullable=True),
        sa.Column('atr', sa.Numeric(precision=20, scale=8), nullable=True),
        sa.Column('support_level', sa.Numeric(precision=20, scale=8), nullable=True),
        sa.Column('resistance_level', sa.Numeric(precision=20, scale=8), nullable=True),
        sa.Column('market_sentiment', sa.Numeric(precision=4, scale=3), nullable=True),
        sa.Column('data_quality_score', sa.Numeric(precision=4, scale=3), nullable=True),
        sa.Column('created_at', TIMESTAMP(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('symbol', 'timeframe', 'timestamp', name='uq_enhanced_market_data')
    )
    
    # Create TimescaleDB hypertable
    op.execute("SELECT create_hypertable('enhanced_market_data', 'timestamp', chunk_time_interval => INTERVAL '1 day')")
    
    # Create indexes for performance
    op.create_index('idx_enhanced_market_data_symbol_timeframe', 'enhanced_market_data', ['symbol', 'timeframe'])
    op.create_index('idx_enhanced_market_data_timestamp', 'enhanced_market_data', ['timestamp'])
    op.create_index('idx_enhanced_market_data_symbol_timestamp', 'enhanced_market_data', ['symbol', 'timestamp'])
    
    # Pattern Detection Table
    op.create_table(
        'pattern_detections',
        sa.Column('id', sa.BigInteger(), nullable=False),
        sa.Column('pattern_id', sa.String(50), nullable=False),
        sa.Column('symbol', sa.String(20), nullable=False),
        sa.Column('timeframe', sa.String(10), nullable=False),
        sa.Column('pattern_type', sa.String(50), nullable=False),
        sa.Column('pattern_category', sa.String(20), nullable=False),  # continuation, reversal, bilateral
        sa.Column('direction', sa.String(10), nullable=False),  # long, short, neutral
        sa.Column('confidence', sa.Numeric(precision=4, scale=3), nullable=False),
        sa.Column('strength', sa.String(20), nullable=False),  # weak, moderate, strong
        sa.Column('entry_price', sa.Numeric(precision=20, scale=8), nullable=False),
        sa.Column('stop_loss', sa.Numeric(precision=20, scale=8), nullable=False),
        sa.Column('take_profit', sa.Numeric(precision=20, scale=8), nullable=False),
        sa.Column('risk_reward_ratio', sa.Numeric(precision=6, scale=2), nullable=False),
        sa.Column('pattern_start_time', TIMESTAMP(timezone=True), nullable=False),
        sa.Column('pattern_end_time', TIMESTAMP(timezone=True), nullable=False),
        sa.Column('breakout_price', sa.Numeric(precision=20, scale=8), nullable=True),
        sa.Column('breakout_time', TIMESTAMP(timezone=True), nullable=True),
        sa.Column('volume_confirmation', sa.Boolean(), nullable=True),
        sa.Column('technical_indicators', JSONB, nullable=True),  # Store RSI, MACD, etc.
        sa.Column('market_conditions', JSONB, nullable=True),  # Store market context
        sa.Column('data_points_used', sa.Integer(), nullable=False),
        sa.Column('data_quality_score', sa.Numeric(precision=4, scale=3), nullable=False),
        sa.Column('status', sa.String(20), nullable=False, default='active'),  # active, completed, failed
        sa.Column('created_at', TIMESTAMP(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', TIMESTAMP(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('pattern_id', name='uq_pattern_detections')
    )
    
    # Create indexes for pattern detections
    op.create_index('idx_pattern_detections_symbol_timeframe', 'pattern_detections', ['symbol', 'timeframe'])
    op.create_index('idx_pattern_detections_pattern_type', 'pattern_detections', ['pattern_type'])
    op.create_index('idx_pattern_detections_confidence', 'pattern_detections', ['confidence'])
    op.create_index('idx_pattern_detections_status', 'pattern_detections', ['status'])
    op.create_index('idx_pattern_detections_created_at', 'pattern_detections', ['created_at'])
    
    # Signal History Table
    op.create_table(
        'signal_history',
        sa.Column('id', sa.BigInteger(), nullable=False),
        sa.Column('signal_id', sa.String(50), nullable=False),
        sa.Column('pattern_id', sa.String(50), nullable=True),
        sa.Column('symbol', sa.String(20), nullable=False),
        sa.Column('timeframe', sa.String(10), nullable=False),
        sa.Column('direction', sa.String(10), nullable=False),  # long, short
        sa.Column('signal_type', sa.String(20), nullable=False),  # entry, exit, stop_loss, take_profit
        sa.Column('entry_price', sa.Numeric(precision=20, scale=8), nullable=False),
        sa.Column('exit_price', sa.Numeric(precision=20, scale=8), nullable=True),
        sa.Column('stop_loss', sa.Numeric(precision=20, scale=8), nullable=False),
        sa.Column('take_profit', sa.Numeric(precision=20, scale=8), nullable=False),
        sa.Column('risk_amount', sa.Numeric(precision=10, scale=2), nullable=True),  # Risk in USD
        sa.Column('position_size', sa.Numeric(precision=10, scale=8), nullable=True),  # Position size in crypto
        sa.Column('risk_reward_ratio', sa.Numeric(precision=6, scale=2), nullable=False),
        sa.Column('confidence', sa.Numeric(precision=4, scale=3), nullable=False),
        sa.Column('pattern_type', sa.String(50), nullable=False),
        sa.Column('technical_analysis', JSONB, nullable=True),  # Store technical indicators
        sa.Column('market_sentiment', sa.Numeric(precision=4, scale=3), nullable=True),
        sa.Column('signal_generated_at', TIMESTAMP(timezone=True), nullable=False),
        sa.Column('signal_expires_at', TIMESTAMP(timezone=True), nullable=True),
        sa.Column('executed_at', TIMESTAMP(timezone=True), nullable=True),
        sa.Column('closed_at', TIMESTAMP(timezone=True), nullable=True),
        sa.Column('pnl', sa.Numeric(precision=10, scale=2), nullable=True),  # Profit/Loss in USD
        sa.Column('pnl_percentage', sa.Numeric(precision=6, scale=2), nullable=True),
        sa.Column('status', sa.String(20), nullable=False, default='generated'),  # generated, executed, closed, expired, cancelled
        sa.Column('execution_notes', sa.Text(), nullable=True),
        sa.Column('created_at', TIMESTAMP(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', TIMESTAMP(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('signal_id', name='uq_signal_history')
    )
    
    # Create indexes for signal history
    op.create_index('idx_signal_history_symbol_timeframe', 'signal_history', ['symbol', 'timeframe'])
    op.create_index('idx_signal_history_status', 'signal_history', ['status'])
    op.create_index('idx_signal_history_direction', 'signal_history', ['direction'])
    op.create_index('idx_signal_history_generated_at', 'signal_history', ['signal_generated_at'])
    op.create_index('idx_signal_history_executed_at', 'signal_history', ['executed_at'])
    
    # Performance Metrics Table
    op.create_table(
        'performance_metrics',
        sa.Column('id', sa.BigInteger(), nullable=False),
        sa.Column('symbol', sa.String(20), nullable=False),
        sa.Column('timeframe', sa.String(10), nullable=False),
        sa.Column('pattern_type', sa.String(50), nullable=False),
        sa.Column('period_start', TIMESTAMP(timezone=True), nullable=False),
        sa.Column('period_end', TIMESTAMP(timezone=True), nullable=False),
        sa.Column('total_signals', sa.Integer(), nullable=False, default=0),
        sa.Column('winning_signals', sa.Integer(), nullable=False, default=0),
        sa.Column('losing_signals', sa.Integer(), nullable=False, default=0),
        sa.Column('win_rate', sa.Numeric(precision=5, scale=2), nullable=False, default=0),
        sa.Column('total_pnl', sa.Numeric(precision=15, scale=2), nullable=False, default=0),
        sa.Column('total_pnl_percentage', sa.Numeric(precision=8, scale=2), nullable=False, default=0),
        sa.Column('average_pnl', sa.Numeric(precision=10, scale=2), nullable=False, default=0),
        sa.Column('average_pnl_percentage', sa.Numeric(precision=6, scale=2), nullable=False, default=0),
        sa.Column('max_drawdown', sa.Numeric(precision=8, scale=2), nullable=False, default=0),
        sa.Column('sharpe_ratio', sa.Numeric(precision=6, scale=3), nullable=True),
        sa.Column('profit_factor', sa.Numeric(precision=6, scale=2), nullable=True),
        sa.Column('average_risk_reward', sa.Numeric(precision=6, scale=2), nullable=False, default=0),
        sa.Column('average_confidence', sa.Numeric(precision=4, scale=3), nullable=False, default=0),
        sa.Column('best_signal_pnl', sa.Numeric(precision=10, scale=2), nullable=True),
        sa.Column('worst_signal_pnl', sa.Numeric(precision=10, scale=2), nullable=True),
        sa.Column('long_signals', sa.Integer(), nullable=False, default=0),
        sa.Column('short_signals', sa.Integer(), nullable=False, default=0),
        sa.Column('long_win_rate', sa.Numeric(precision=5, scale=2), nullable=False, default=0),
        sa.Column('short_win_rate', sa.Numeric(precision=5, scale=2), nullable=False, default=0),
        sa.Column('market_conditions', JSONB, nullable=True),  # Store market context during period
        sa.Column('created_at', TIMESTAMP(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', TIMESTAMP(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('symbol', 'timeframe', 'pattern_type', 'period_start', name='uq_performance_metrics')
    )
    
    # Create indexes for performance metrics
    op.create_index('idx_performance_metrics_symbol_timeframe', 'performance_metrics', ['symbol', 'timeframe'])
    op.create_index('idx_performance_metrics_pattern_type', 'performance_metrics', ['pattern_type'])
    op.create_index('idx_performance_metrics_period', 'performance_metrics', ['period_start', 'period_end'])
    op.create_index('idx_performance_metrics_win_rate', 'performance_metrics', ['win_rate'])
    
    # Confidence Scoring Table
    op.create_table(
        'confidence_scores',
        sa.Column('id', sa.BigInteger(), nullable=False),
        sa.Column('pattern_type', sa.String(50), nullable=False),
        sa.Column('symbol', sa.String(20), nullable=False),
        sa.Column('timeframe', sa.String(10), nullable=False),
        sa.Column('market_condition', sa.String(50), nullable=False),  # bull, bear, sideways, volatile
        sa.Column('volatility_level', sa.String(20), nullable=False),  # low, medium, high
        sa.Column('volume_level', sa.String(20), nullable=False),  # low, medium, high
        sa.Column('historical_accuracy', sa.Numeric(precision=5, scale=2), nullable=False),
        sa.Column('sample_size', sa.Integer(), nullable=False),
        sa.Column('confidence_score', sa.Numeric(precision=4, scale=3), nullable=False),
        sa.Column('confidence_factors', JSONB, nullable=True),  # Store contributing factors
        sa.Column('last_updated', TIMESTAMP(timezone=True), nullable=False),
        sa.Column('created_at', TIMESTAMP(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('pattern_type', 'symbol', 'timeframe', 'market_condition', 'volatility_level', 'volume_level', name='uq_confidence_scores')
    )
    
    # Create indexes for confidence scores
    op.create_index('idx_confidence_scores_pattern_type', 'confidence_scores', ['pattern_type'])
    op.create_index('idx_confidence_scores_symbol_timeframe', 'confidence_scores', ['symbol', 'timeframe'])
    op.create_index('idx_confidence_scores_confidence', 'confidence_scores', ['confidence_score'])
    
    # Market Conditions Table
    op.create_table(
        'market_conditions',
        sa.Column('id', sa.BigInteger(), nullable=False),
        sa.Column('symbol', sa.String(20), nullable=False),
        sa.Column('timeframe', sa.String(10), nullable=False),
        sa.Column('timestamp', TIMESTAMP(timezone=True), nullable=False),
        sa.Column('market_regime', sa.String(20), nullable=False),  # bull, bear, sideways, volatile
        sa.Column('volatility_level', sa.String(20), nullable=False),  # low, medium, high
        sa.Column('volume_level', sa.String(20), nullable=False),  # low, medium, high
        sa.Column('trend_strength', sa.Numeric(precision=4, scale=3), nullable=False),
        sa.Column('momentum_score', sa.Numeric(precision=4, scale=3), nullable=False),
        sa.Column('support_resistance_quality', sa.Numeric(precision=4, scale=3), nullable=False),
        sa.Column('market_sentiment', sa.Numeric(precision=4, scale=3), nullable=False),
        sa.Column('technical_indicators', JSONB, nullable=True),  # Store all technical indicators
        sa.Column('market_metrics', JSONB, nullable=True),  # Store additional market metrics
        sa.Column('created_at', TIMESTAMP(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('symbol', 'timeframe', 'timestamp', name='uq_market_conditions')
    )
    
    # Create TimescaleDB hypertable for market conditions
    op.execute("SELECT create_hypertable('market_conditions', 'timestamp', chunk_time_interval => INTERVAL '1 day')")
    
    # Create indexes for market conditions
    op.create_index('idx_market_conditions_symbol_timeframe', 'market_conditions', ['symbol', 'timeframe'])
    op.create_index('idx_market_conditions_timestamp', 'market_conditions', ['timestamp'])
    op.create_index('idx_market_conditions_regime', 'market_conditions', ['market_regime'])

def downgrade():
    """Drop enhanced data tables"""
    op.drop_table('market_conditions')
    op.drop_table('confidence_scores')
    op.drop_table('performance_metrics')
    op.drop_table('signal_history')
    op.drop_table('pattern_detections')
    op.drop_table('enhanced_market_data')
