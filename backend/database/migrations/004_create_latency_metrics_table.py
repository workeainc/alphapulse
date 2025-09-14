"""
Migration: Create latency_metrics table for performance tracking
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '004_create_latency_metrics_table'
down_revision = '003_create_active_learning_tables'
branch_labels = None
depends_on = None


def upgrade():
    """Create latency_metrics table"""
    
    # Create latency_metrics table
    op.create_table(
        'latency_metrics',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('model_id', sa.String(100), nullable=True),
        sa.Column('operation_type', sa.String(50), nullable=False),
        sa.Column('fetch_time_ms', sa.Float(), nullable=True),
        sa.Column('preprocess_time_ms', sa.Float(), nullable=True),
        sa.Column('inference_time_ms', sa.Float(), nullable=True),
        sa.Column('postprocess_time_ms', sa.Float(), nullable=True),
        sa.Column('total_latency_ms', sa.Float(), nullable=False),
        sa.Column('symbol', sa.String(20), nullable=True),
        sa.Column('strategy_name', sa.String(100), nullable=True),
        sa.Column('success', sa.Boolean(), nullable=False, default=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('metadata_json', postgresql.JSONB(), nullable=True),
        sa.Column('timestamp', sa.DateTime(), nullable=False, default=sa.func.now()),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create indexes for performance
    op.create_index('ix_latency_metrics_model_id', 'latency_metrics', ['model_id'])
    op.create_index('ix_latency_metrics_operation_type', 'latency_metrics', ['operation_type'])
    op.create_index('ix_latency_metrics_symbol', 'latency_metrics', ['symbol'])
    op.create_index('ix_latency_metrics_strategy_name', 'latency_metrics', ['strategy_name'])
    op.create_index('ix_latency_metrics_timestamp', 'latency_metrics', ['timestamp'])
    
    # Create composite indexes for common queries
    op.create_index('ix_latency_metrics_symbol_timestamp', 'latency_metrics', ['symbol', 'timestamp'])
    op.create_index('ix_latency_metrics_strategy_timestamp', 'latency_metrics', ['strategy_name', 'timestamp'])
    op.create_index('ix_latency_metrics_model_timestamp', 'latency_metrics', ['model_id', 'timestamp'])
    
    # Convert to TimescaleDB hypertable for time-series optimization
    op.execute("SELECT create_hypertable('latency_metrics', 'timestamp', if_not_exists => TRUE)")
    
    # Set compression policy for older data
    op.execute("""
        ALTER TABLE latency_metrics SET (
            timescaledb.compress,
            timescaledb.compress_segmentby = 'operation_type,model_id',
            timescaledb.compress_orderby = 'timestamp DESC'
        )
    """)
    
    # Add compression policy (compress chunks older than 1 day)
    op.execute("SELECT add_compression_policy('latency_metrics', INTERVAL '1 day')")
    
    # Add retention policy (keep data for 30 days)
    op.execute("SELECT add_retention_policy('latency_metrics', INTERVAL '30 days')")


def downgrade():
    """Drop latency_metrics table"""
    
    # Drop indexes
    op.drop_index('ix_latency_metrics_model_timestamp', 'latency_metrics')
    op.drop_index('ix_latency_metrics_strategy_timestamp', 'latency_metrics')
    op.drop_index('ix_latency_metrics_symbol_timestamp', 'latency_metrics')
    op.drop_index('ix_latency_metrics_timestamp', 'latency_metrics')
    op.drop_index('ix_latency_metrics_strategy_name', 'latency_metrics')
    op.drop_index('ix_latency_metrics_symbol', 'latency_metrics')
    op.drop_index('ix_latency_metrics_operation_type', 'latency_metrics')
    op.drop_index('ix_latency_metrics_model_id', 'latency_metrics')
    
    # Drop table
    op.drop_table('latency_metrics')
