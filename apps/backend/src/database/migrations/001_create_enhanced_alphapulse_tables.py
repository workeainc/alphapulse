"""Create enhanced AlphaPulse testing tables

Revision ID: 001
Revises: 
Create Date: 2025-08-15 12:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import sqlite

# revision identifiers, used by Alembic.
revision = '001'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create enhanced AlphaPulse testing tables"""
    
    # Create signals table
    op.create_table('signals',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('signal_id', sa.String(length=20), nullable=False),
        sa.Column('timestamp', sa.DateTime(timezone=True), nullable=False),
        sa.Column('symbol', sa.String(length=20), nullable=False),
        sa.Column('timeframe', sa.String(length=10), nullable=False),
        sa.Column('direction', sa.String(length=10), nullable=False),
        sa.Column('confidence', sa.Float(), nullable=False),
        sa.Column('entry_price', sa.Float(), nullable=False),
        sa.Column('tp1', sa.Float(), nullable=True),
        sa.Column('tp2', sa.Float(), nullable=True),
        sa.Column('tp3', sa.Float(), nullable=True),
        sa.Column('tp4', sa.Float(), nullable=True),
        sa.Column('stop_loss', sa.Float(), nullable=True),
        sa.Column('risk_reward_ratio', sa.Float(), nullable=True),
        sa.Column('pattern_type', sa.String(length=50), nullable=True),
        sa.Column('volume_confirmation', sa.Boolean(), nullable=True),
        sa.Column('trend_alignment', sa.Boolean(), nullable=True),
        sa.Column('market_regime', sa.String(length=20), nullable=True),
        sa.Column('indicators', sa.JSON(), nullable=True),
        sa.Column('validation_metrics', sa.JSON(), nullable=True),
        sa.Column('metadata', sa.JSON(), nullable=True),
        sa.Column('outcome', sa.String(length=20), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create logs table
    op.create_table('logs',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('signal_id', sa.String(length=20), nullable=False),
        sa.Column('pattern_type', sa.String(length=50), nullable=False),
        sa.Column('confidence_score', sa.Float(), nullable=False),
        sa.Column('volume_context', sa.JSON(), nullable=True),
        sa.Column('trend_context', sa.JSON(), nullable=True),
        sa.Column('outcome', sa.String(length=20), nullable=True),
        sa.Column('timestamp', sa.DateTime(timezone=True), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create feedback table
    op.create_table('feedback',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('signal_id', sa.String(length=20), nullable=False),
        sa.Column('market_outcome', sa.Float(), nullable=True),
        sa.Column('notes', sa.Text(), nullable=True),
        sa.Column('timestamp', sa.DateTime(timezone=True), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create performance_metrics table
    op.create_table('performance_metrics',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('test_name', sa.String(length=100), nullable=False),
        sa.Column('test_timestamp', sa.DateTime(timezone=True), nullable=False),
        sa.Column('avg_latency_ms', sa.Float(), nullable=False),
        sa.Column('max_latency_ms', sa.Float(), nullable=False),
        sa.Column('min_latency_ms', sa.Float(), nullable=False),
        sa.Column('p95_latency_ms', sa.Float(), nullable=True),
        sa.Column('p99_latency_ms', sa.Float(), nullable=True),
        sa.Column('win_rate', sa.Float(), nullable=True),
        sa.Column('total_signals', sa.Integer(), nullable=False),
        sa.Column('winning_signals', sa.Integer(), nullable=True),
        sa.Column('losing_signals', sa.Integer(), nullable=True),
        sa.Column('filtered_signals', sa.Integer(), nullable=True),
        sa.Column('filter_rate', sa.Float(), nullable=True),
        sa.Column('throughput_signals_per_sec', sa.Float(), nullable=True),
        sa.Column('cpu_usage_percent', sa.Float(), nullable=True),
        sa.Column('memory_usage_mb', sa.Float(), nullable=True),
        sa.Column('test_config', sa.JSON(), nullable=True),
        sa.Column('test_results', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create indexes
    op.create_index('idx_signals_signal_id', 'signals', ['signal_id'], unique=True)
    op.create_index('idx_signals_symbol_timeframe_timestamp', 'signals', ['symbol', 'timeframe', 'timestamp'])
    op.create_index('idx_signals_confidence_outcome', 'signals', ['confidence', 'outcome'])
    op.create_index('idx_logs_signal_id', 'logs', ['signal_id'])
    op.create_index('idx_logs_timestamp', 'logs', ['timestamp'])
    op.create_index('idx_feedback_signal_id', 'feedback', ['signal_id'])
    op.create_index('idx_performance_metrics_test_timestamp', 'performance_metrics', ['test_name', 'test_timestamp'])
    
    # Create foreign key constraints
    op.create_foreign_key('fk_logs_signal_id', 'logs', 'signals', ['signal_id'], ['signal_id'])
    op.create_foreign_key('fk_feedback_signal_id', 'feedback', 'signals', ['signal_id'], ['signal_id'])


def downgrade() -> None:
    """Drop enhanced AlphaPulse testing tables"""
    
    # Drop foreign key constraints
    op.drop_constraint('fk_feedback_signal_id', 'feedback', type_='foreignkey')
    op.drop_constraint('fk_logs_signal_id', 'logs', type_='foreignkey')
    
    # Drop indexes
    op.drop_index('idx_performance_metrics_test_timestamp', table_name='performance_metrics')
    op.drop_index('idx_feedback_signal_id', table_name='feedback')
    op.drop_index('idx_logs_timestamp', table_name='logs')
    op.drop_index('idx_logs_signal_id', table_name='logs')
    op.drop_index('idx_signals_confidence_outcome', table_name='signals')
    op.drop_index('idx_signals_symbol_timeframe_timestamp', table_name='signals')
    op.drop_index('idx_signals_signal_id', table_name='signals')
    
    # Drop tables
    op.drop_table('performance_metrics')
    op.drop_table('feedback')
    op.drop_table('logs')
    op.drop_table('signals')
