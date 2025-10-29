"""
Migration: Create model_accuracy_benchmarks table for accuracy evaluation
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '005_create_accuracy_benchmarks_table'
down_revision = '004_create_latency_metrics_table'
branch_labels = None
depends_on = None


def upgrade():
    """Create model_accuracy_benchmarks table"""
    
    # Create model_accuracy_benchmarks table
    op.create_table(
        'model_accuracy_benchmarks',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('model_id', sa.String(100), nullable=False),
        sa.Column('symbol', sa.String(20), nullable=False),
        sa.Column('strategy_name', sa.String(100), nullable=False),
        
        # ML Classification Metrics
        sa.Column('precision', sa.Float(), nullable=False),
        sa.Column('recall', sa.Float(), nullable=False),
        sa.Column('f1_score', sa.Float(), nullable=False),
        sa.Column('accuracy', sa.Float(), nullable=False),
        sa.Column('roc_auc', sa.Float(), nullable=False),
        
        # Trading Performance Metrics
        sa.Column('win_rate', sa.Float(), nullable=False),
        sa.Column('profit_factor', sa.Float(), nullable=False),
        sa.Column('avg_win', sa.Float(), nullable=False),
        sa.Column('avg_loss', sa.Float(), nullable=False),
        sa.Column('total_return', sa.Float(), nullable=False),
        sa.Column('sharpe_ratio', sa.Float(), nullable=False),
        sa.Column('max_drawdown', sa.Float(), nullable=False),
        
        # Additional Metrics
        sa.Column('total_trades', sa.Integer(), nullable=False),
        sa.Column('winning_trades', sa.Integer(), nullable=False),
        sa.Column('losing_trades', sa.Integer(), nullable=False),
        sa.Column('avg_holding_period', sa.Float(), nullable=False),
        sa.Column('risk_reward_ratio', sa.Float(), nullable=False),
        
        # Test Configuration
        sa.Column('test_period_days', sa.Integer(), nullable=False),
        sa.Column('frozen_test_set', sa.Boolean(), nullable=False, default=True),
        
        # Metadata
        sa.Column('evaluation_date', sa.DateTime(), nullable=False, default=sa.func.now()),
        sa.Column('benchmark_version', sa.String(20), nullable=False, default='v1.0'),
        
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create indexes for performance
    op.create_index('ix_model_accuracy_benchmarks_model_id', 'model_accuracy_benchmarks', ['model_id'])
    op.create_index('ix_model_accuracy_benchmarks_symbol', 'model_accuracy_benchmarks', ['symbol'])
    op.create_index('ix_model_accuracy_benchmarks_strategy_name', 'model_accuracy_benchmarks', ['strategy_name'])
    op.create_index('ix_model_accuracy_benchmarks_evaluation_date', 'model_accuracy_benchmarks', ['evaluation_date'])
    
    # Create composite indexes for common queries
    op.create_index('ix_model_accuracy_benchmarks_model_symbol', 'model_accuracy_benchmarks', ['model_id', 'symbol'])
    op.create_index('ix_model_accuracy_benchmarks_model_strategy', 'model_accuracy_benchmarks', ['model_id', 'strategy_name'])
    op.create_index('ix_model_accuracy_benchmarks_symbol_date', 'model_accuracy_benchmarks', ['symbol', 'evaluation_date'])
    
    # Convert to TimescaleDB hypertable for time-series optimization
    op.execute("SELECT create_hypertable('model_accuracy_benchmarks', 'evaluation_date', if_not_exists => TRUE)")
    
    # Set compression policy for older data
    op.execute("""
        ALTER TABLE model_accuracy_benchmarks SET (
            timescaledb.compress,
            timescaledb.compress_segmentby = 'model_id,symbol',
            timescaledb.compress_orderby = 'evaluation_date DESC'
        )
    """)
    
    # Add compression policy (compress chunks older than 7 days)
    op.execute("SELECT add_compression_policy('model_accuracy_benchmarks', INTERVAL '7 days')")
    
    # Add retention policy (keep data for 1 year)
    op.execute("SELECT add_retention_policy('model_accuracy_benchmarks', INTERVAL '1 year')")


def downgrade():
    """Drop model_accuracy_benchmarks table"""
    
    # Drop indexes
    op.drop_index('ix_model_accuracy_benchmarks_symbol_date', 'model_accuracy_benchmarks')
    op.drop_index('ix_model_accuracy_benchmarks_model_strategy', 'model_accuracy_benchmarks')
    op.drop_index('ix_model_accuracy_benchmarks_model_symbol', 'model_accuracy_benchmarks')
    op.drop_index('ix_model_accuracy_benchmarks_evaluation_date', 'model_accuracy_benchmarks')
    op.drop_index('ix_model_accuracy_benchmarks_strategy_name', 'model_accuracy_benchmarks')
    op.drop_index('ix_model_accuracy_benchmarks_symbol', 'model_accuracy_benchmarks')
    op.drop_index('ix_model_accuracy_benchmarks_model_id', 'model_accuracy_benchmarks')
    
    # Drop table
    op.drop_table('model_accuracy_benchmarks')
