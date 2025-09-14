"""Create AlphaPulse testing tables

Revision ID: 007
Revises: 006
Create Date: 2025-08-15 10:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import sqlite

# revision identifiers, used by Alembic.
revision = '007'
down_revision = '006'
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create AlphaPulse testing tables"""
    
    # Create signals table
    op.create_table('signals',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('symbol', sa.String(length=20), nullable=False),
        sa.Column('timeframe', sa.String(length=10), nullable=False),
        sa.Column('direction', sa.String(length=10), nullable=False),  # 'buy'/'sell'
        sa.Column('confidence', sa.Float(), nullable=False),
        sa.Column('tp1', sa.Float(), nullable=True),
        sa.Column('tp2', sa.Float(), nullable=True),
        sa.Column('tp3', sa.Float(), nullable=True),
        sa.Column('tp4', sa.Float(), nullable=True),
        sa.Column('sl', sa.Float(), nullable=True),
        sa.Column('timestamp', sa.DateTime(), nullable=False),
        sa.Column('outcome', sa.String(length=20), nullable=True, default='pending'),  # 'win'/'loss'/'pending'
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create logs table
    op.create_table('logs',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('pattern_type', sa.String(length=50), nullable=False),
        sa.Column('confidence_score', sa.Float(), nullable=False),
        sa.Column('volume_context', sa.JSON(), nullable=True),
        sa.Column('trend_context', sa.JSON(), nullable=True),
        sa.Column('outcome', sa.String(length=20), nullable=True),
        sa.Column('timestamp', sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create feedback table
    op.create_table('feedback',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('signal_id', sa.Integer(), nullable=False),
        sa.Column('market_outcome', sa.Float(), nullable=True),  # PnL
        sa.Column('notes', sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['signal_id'], ['signals.id'], )
    )
    
    # Create indexes for better performance
    op.create_index('idx_signals_symbol_timeframe_timestamp', 'signals', ['symbol', 'timeframe', 'timestamp'])
    op.create_index('idx_logs_timestamp', 'logs', ['timestamp'])
    op.create_index('idx_feedback_signal_id', 'feedback', ['signal_id'])


def downgrade() -> None:
    """Drop AlphaPulse testing tables"""
    
    # Drop indexes
    op.drop_index('idx_feedback_signal_id', table_name='feedback')
    op.drop_index('idx_logs_timestamp', table_name='logs')
    op.drop_index('idx_signals_symbol_timeframe_timestamp', table_name='signals')
    
    # Drop tables
    op.drop_table('feedback')
    op.drop_table('logs')
    op.drop_table('signals')
