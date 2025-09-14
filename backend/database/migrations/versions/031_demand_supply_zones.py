"""Demand and Supply Zones Analysis

Revision ID: 031
Revises: 030_advanced_order_flow_analysis
Create Date: 2024-01-15 10:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB, NUMERIC
from sqlalchemy.sql import text

# revision identifiers, used by Alembic.
revision = '031_demand_supply_zones'
down_revision = '030_advanced_order_flow_analysis'
branch_labels = None
depends_on = None

def upgrade():
    """Create demand and supply zones analysis tables"""
    
    # Create demand_supply_zones table
    op.create_table(
        'demand_supply_zones',
        sa.Column('id', sa.BigInteger(), nullable=False),
        sa.Column('symbol', sa.String(20), nullable=False),
        sa.Column('timeframe', sa.String(10), nullable=False),
        sa.Column('timestamp', sa.TIMESTAMP(timezone=True), nullable=False),
        sa.Column('zone_type', sa.String(20), nullable=False),  # 'demand' or 'supply'
        sa.Column('zone_start_price', NUMERIC(18, 8), nullable=False),
        sa.Column('zone_end_price', NUMERIC(18, 8), nullable=False),
        sa.Column('zone_volume', NUMERIC(18, 8), nullable=False),
        sa.Column('zone_strength', NUMERIC(10, 6), nullable=False),  # 0 to 1
        sa.Column('zone_confidence', NUMERIC(10, 6), nullable=False),  # 0 to 1
        sa.Column('zone_touches', sa.Integer(), nullable=False),  # Number of price touches
        sa.Column('zone_duration_hours', sa.Integer(), nullable=True),
        sa.Column('zone_breakout_direction', sa.String(10), nullable=True),  # 'up', 'down', 'none'
        sa.Column('zone_breakout_strength', NUMERIC(10, 6), nullable=True),  # 0 to 1
        sa.Column('zone_volume_profile', JSONB, nullable=True),
        sa.Column('zone_order_flow', JSONB, nullable=True),
        sa.Column('zone_metadata', JSONB, nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('id', 'timestamp', name='uq_demand_supply_zones_id_timestamp')
    )
    
    # Create TimescaleDB hypertable (commented out for now)
    # op.execute("SELECT create_hypertable('demand_supply_zones', 'timestamp', if_not_exists => TRUE, chunk_time_interval => INTERVAL '1 hour');")
    
    # Create indexes
    op.execute("CREATE INDEX IF NOT EXISTS idx_demand_supply_zones_symbol_timestamp ON demand_supply_zones (symbol, timestamp)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_demand_supply_zones_zone_type ON demand_supply_zones (zone_type)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_demand_supply_zones_strength ON demand_supply_zones (zone_strength)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_demand_supply_zones_price_range ON demand_supply_zones (zone_start_price, zone_end_price)")
    
    # Create volume_profile_analysis table
    op.create_table(
        'volume_profile_analysis',
        sa.Column('id', sa.BigInteger(), nullable=False),
        sa.Column('symbol', sa.String(20), nullable=False),
        sa.Column('timeframe', sa.String(10), nullable=False),
        sa.Column('timestamp', sa.TIMESTAMP(timezone=True), nullable=False),
        sa.Column('price_level', NUMERIC(18, 8), nullable=False),
        sa.Column('volume_at_level', NUMERIC(18, 8), nullable=False),
        sa.Column('volume_percentage', NUMERIC(10, 6), nullable=False),  # Percentage of total volume
        sa.Column('volume_node_type', sa.String(20), nullable=False),  # 'high', 'medium', 'low'
        sa.Column('volume_concentration', NUMERIC(10, 6), nullable=False),  # 0 to 1
        sa.Column('price_efficiency', NUMERIC(10, 6), nullable=True),  # How efficient price movement is at this level
        sa.Column('volume_trend', sa.String(20), nullable=True),  # 'increasing', 'decreasing', 'stable'
        sa.Column('volume_metadata', JSONB, nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('id', 'timestamp', name='uq_volume_profile_analysis_id_timestamp')
    )
    
    # Create TimescaleDB hypertable (commented out for now)
    # op.execute("SELECT create_hypertable('volume_profile_analysis', 'timestamp', if_not_exists => TRUE, chunk_time_interval => INTERVAL '1 hour');")
    
    # Create indexes
    op.execute("CREATE INDEX IF NOT EXISTS idx_volume_profile_symbol_timestamp ON volume_profile_analysis (symbol, timestamp)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_volume_profile_price_level ON volume_profile_analysis (price_level)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_volume_profile_node_type ON volume_profile_analysis (volume_node_type)")
    
    # Create zone_breakouts table
    op.create_table(
        'zone_breakouts',
        sa.Column('id', sa.BigInteger(), nullable=False),
        sa.Column('symbol', sa.String(20), nullable=False),
        sa.Column('timeframe', sa.String(10), nullable=False),
        sa.Column('timestamp', sa.TIMESTAMP(timezone=True), nullable=False),
        sa.Column('zone_id', sa.BigInteger(), nullable=False),
        sa.Column('breakout_type', sa.String(20), nullable=False),  # 'demand_breakout', 'supply_breakout', 'demand_breakdown', 'supply_breakdown'
        sa.Column('breakout_price', NUMERIC(18, 8), nullable=False),
        sa.Column('breakout_volume', NUMERIC(18, 8), nullable=False),
        sa.Column('breakout_strength', NUMERIC(10, 6), nullable=False),  # 0 to 1
        sa.Column('breakout_confidence', NUMERIC(10, 6), nullable=False),  # 0 to 1
        sa.Column('breakout_volume_ratio', NUMERIC(10, 6), nullable=False),  # Volume vs average
        sa.Column('breakout_momentum', NUMERIC(10, 6), nullable=True),  # Price momentum during breakout
        sa.Column('breakout_retest', sa.Boolean(), nullable=True),  # Whether price retested the zone
        sa.Column('breakout_metadata', JSONB, nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('id', 'timestamp', name='uq_zone_breakouts_id_timestamp'),
        sa.ForeignKeyConstraint(['zone_id'], ['demand_supply_zones.id'], ondelete='CASCADE')
    )
    
    # Create TimescaleDB hypertable (commented out for now)
    # op.execute("SELECT create_hypertable('zone_breakouts', 'timestamp', if_not_exists => TRUE, chunk_time_interval => INTERVAL '1 hour');")
    
    # Create indexes
    op.execute("CREATE INDEX IF NOT EXISTS idx_zone_breakouts_symbol_timestamp ON zone_breakouts (symbol, timestamp)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_zone_breakouts_zone_id ON zone_breakouts (zone_id)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_zone_breakouts_type ON zone_breakouts (breakout_type)")
    
    # Create zone_interactions table
    op.create_table(
        'zone_interactions',
        sa.Column('id', sa.BigInteger(), nullable=False),
        sa.Column('symbol', sa.String(20), nullable=False),
        sa.Column('timeframe', sa.String(10), nullable=False),
        sa.Column('timestamp', sa.TIMESTAMP(timezone=True), nullable=False),
        sa.Column('zone_id', sa.BigInteger(), nullable=False),
        sa.Column('interaction_type', sa.String(20), nullable=False),  # 'touch', 'bounce', 'penetration', 'rejection'
        sa.Column('interaction_price', NUMERIC(18, 8), nullable=False),
        sa.Column('interaction_volume', NUMERIC(18, 8), nullable=False),
        sa.Column('interaction_strength', NUMERIC(10, 6), nullable=False),  # 0 to 1
        sa.Column('interaction_confidence', NUMERIC(10, 6), nullable=False),  # 0 to 1
        sa.Column('interaction_duration_minutes', sa.Integer(), nullable=True),
        sa.Column('interaction_momentum', NUMERIC(10, 6), nullable=True),  # Price momentum during interaction
        sa.Column('interaction_metadata', JSONB, nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('id', 'timestamp', name='uq_zone_interactions_id_timestamp'),
        sa.ForeignKeyConstraint(['zone_id'], ['demand_supply_zones.id'], ondelete='CASCADE')
    )
    
    # Create TimescaleDB hypertable (commented out for now)
    # op.execute("SELECT create_hypertable('zone_interactions', 'timestamp', if_not_exists => TRUE, chunk_time_interval => INTERVAL '1 hour');")
    
    # Create indexes
    op.execute("CREATE INDEX IF NOT EXISTS idx_zone_interactions_symbol_timestamp ON zone_interactions (symbol, timestamp)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_zone_interactions_zone_id ON zone_interactions (zone_id)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_zone_interactions_type ON zone_interactions (interaction_type)")
    
    # Create zone_aggregates table for performance optimization
    op.create_table(
        'zone_aggregates',
        sa.Column('id', sa.BigInteger(), nullable=False),
        sa.Column('symbol', sa.String(20), nullable=False),
        sa.Column('timeframe', sa.String(10), nullable=False),
        sa.Column('timestamp', sa.TIMESTAMP(timezone=True), nullable=False),
        sa.Column('aggregation_period', sa.String(10), nullable=False),  # '1h', '4h', '1d'
        sa.Column('active_demand_zones', sa.Integer(), nullable=False),
        sa.Column('active_supply_zones', sa.Integer(), nullable=False),
        sa.Column('strongest_demand_zone', NUMERIC(18, 8), nullable=True),
        sa.Column('strongest_supply_zone', NUMERIC(18, 8), nullable=True),
        sa.Column('zone_breakout_count', sa.Integer(), nullable=False),
        sa.Column('zone_touch_count', sa.Integer(), nullable=False),
        sa.Column('volume_profile_summary', JSONB, nullable=True),
        sa.Column('zone_analysis_summary', JSONB, nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('id', 'timestamp', name='uq_zone_aggregates_id_timestamp')
    )
    
    # Create TimescaleDB hypertable (commented out for now)
    # op.execute("SELECT create_hypertable('zone_aggregates', 'timestamp', if_not_exists => TRUE, chunk_time_interval => INTERVAL '1 hour');")
    
    # Create indexes
    op.execute("CREATE INDEX IF NOT EXISTS idx_zone_aggregates_symbol_timestamp ON zone_aggregates (symbol, timestamp)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_zone_aggregates_period ON zone_aggregates (aggregation_period)")

def downgrade():
    """Drop demand and supply zones analysis tables"""
    op.drop_table('zone_aggregates')
    op.drop_table('zone_interactions')
    op.drop_table('zone_breakouts')
    op.drop_table('volume_profile_analysis')
    op.drop_table('demand_supply_zones')
