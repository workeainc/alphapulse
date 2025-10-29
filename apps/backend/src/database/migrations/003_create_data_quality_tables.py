"""
Migration: Create Data Quality Tables
Tables for tracking data quality metrics and anomalies
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB, TIMESTAMP

# revision identifiers, used by Alembic.
revision = '003_create_data_quality_tables'
down_revision = '002_create_enhanced_data_tables'
branch_labels = None
depends_on = None

def upgrade():
    """Create data quality tables"""
    
    # Data Quality Metrics Table
    op.create_table(
        'data_quality_metrics',
        sa.Column('id', sa.BigInteger(), nullable=False),
        sa.Column('symbol', sa.String(20), nullable=False),
        sa.Column('timeframe', sa.String(10), nullable=False),
        sa.Column('completeness', sa.Numeric(precision=4, scale=3), nullable=False),
        sa.Column('accuracy', sa.Numeric(precision=4, scale=3), nullable=False),
        sa.Column('consistency', sa.Numeric(precision=4, scale=3), nullable=False),
        sa.Column('timeliness', sa.Numeric(precision=4, scale=3), nullable=False),
        sa.Column('validity', sa.Numeric(precision=4, scale=3), nullable=False),
        sa.Column('overall_score', sa.Numeric(precision=4, scale=3), nullable=False),
        sa.Column('timestamp', TIMESTAMP(timezone=True), nullable=False),
        sa.Column('created_at', TIMESTAMP(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('symbol', 'timeframe', 'timestamp', name='uq_data_quality_metrics')
    )
    
    # Create TimescaleDB hypertable for data quality metrics
    op.execute("SELECT create_hypertable('data_quality_metrics', 'timestamp', chunk_time_interval => INTERVAL '1 day')")
    
    # Create indexes for data quality metrics
    op.create_index('idx_data_quality_metrics_symbol_timeframe', 'data_quality_metrics', ['symbol', 'timeframe'])
    op.create_index('idx_data_quality_metrics_timestamp', 'data_quality_metrics', ['timestamp'])
    op.create_index('idx_data_quality_metrics_overall_score', 'data_quality_metrics', ['overall_score'])
    
    # Data Anomalies Table
    op.create_table(
        'data_anomalies',
        sa.Column('id', sa.BigInteger(), nullable=False),
        sa.Column('symbol', sa.String(20), nullable=False),
        sa.Column('timeframe', sa.String(10), nullable=False),
        sa.Column('timestamp', TIMESTAMP(timezone=True), nullable=False),
        sa.Column('anomaly_type', sa.String(50), nullable=False),  # price_spike, volume_spike, missing_data, etc.
        sa.Column('severity', sa.String(20), nullable=False),      # low, medium, high, critical
        sa.Column('description', sa.Text(), nullable=False),
        sa.Column('suggested_action', sa.Text(), nullable=False),
        sa.Column('resolved', sa.Boolean(), nullable=False, default=False),
        sa.Column('resolved_at', TIMESTAMP(timezone=True), nullable=True),
        sa.Column('resolution_notes', sa.Text(), nullable=True),
        sa.Column('created_at', TIMESTAMP(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('symbol', 'timeframe', 'timestamp', 'anomaly_type', name='uq_data_anomalies')
    )
    
    # Create TimescaleDB hypertable for data anomalies
    op.execute("SELECT create_hypertable('data_anomalies', 'timestamp', chunk_time_interval => INTERVAL '1 day')")
    
    # Create indexes for data anomalies
    op.create_index('idx_data_anomalies_symbol_timeframe', 'data_anomalies', ['symbol', 'timeframe'])
    op.create_index('idx_data_anomalies_timestamp', 'data_anomalies', ['timestamp'])
    op.create_index('idx_data_anomalies_type', 'data_anomalies', ['anomaly_type'])
    op.create_index('idx_data_anomalies_severity', 'data_anomalies', ['severity'])
    op.create_index('idx_data_anomalies_resolved', 'data_anomalies', ['resolved'])

def downgrade():
    """Drop data quality tables"""
    op.drop_table('data_anomalies')
    op.drop_table('data_quality_metrics')
