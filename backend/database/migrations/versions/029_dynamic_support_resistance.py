"""
Migration 029: Dynamic Support/Resistance Enhancement
Adds comprehensive dynamic support/resistance analysis capabilities to AlphaPulse
"""

import asyncio
import logging
from sqlalchemy import text
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

logger = logging.getLogger(__name__)

# revision identifiers, used by Alembic
revision = '029_dynamic_support_resistance'
down_revision = '028_market_structure_analysis'
branch_labels = None
depends_on = None

def upgrade():
    """Create dynamic support/resistance enhancement tables"""
    
    # Dynamic support/resistance levels table
    op.create_table(
        'dynamic_support_resistance',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('symbol', sa.String(20), nullable=False),
        sa.Column('timeframe', sa.String(10), nullable=False),
        sa.Column('timestamp', sa.TIMESTAMP(timezone=True), nullable=False),
        
        # Level information
        sa.Column('level_type', sa.String(20), nullable=False),  # 'support', 'resistance', 'dynamic'
        sa.Column('price_level', sa.Numeric(20, 8), nullable=False),
        sa.Column('strength', sa.Numeric(4, 3), nullable=False),  # 0-1 strength score
        sa.Column('confidence', sa.Numeric(4, 3), nullable=False),  # 0-1 confidence
        
        # Touch analysis
        sa.Column('touch_count', sa.Integer(), default=0),
        sa.Column('first_touch_time', sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column('last_touch_time', sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column('touch_points', JSONB, nullable=True),  # Array of touch details
        
        # Volume analysis
        sa.Column('volume_confirmation', sa.Boolean(), nullable=True),
        sa.Column('avg_volume_at_level', sa.Numeric(20, 8), nullable=True),
        sa.Column('volume_spike_ratio', sa.Numeric(8, 4), nullable=True),
        sa.Column('institutional_activity', sa.Boolean(), nullable=True),
        
        # Level characteristics
        sa.Column('level_age_bars', sa.Integer(), nullable=True),  # Age in bars
        sa.Column('level_range', sa.Numeric(8, 4), nullable=True),  # Price range around level
        sa.Column('penetration_count', sa.Integer(), default=0),  # Times level was penetrated
        sa.Column('rejection_count', sa.Integer(), default=0),  # Times price rejected from level
        
        # Status tracking
        sa.Column('is_active', sa.Boolean(), default=True),
        sa.Column('is_broken', sa.Boolean(), default=False),
        sa.Column('break_time', sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column('break_volume', sa.Numeric(20, 8), nullable=True),
        
        # Market context
        sa.Column('market_structure_context', sa.String(50), nullable=True),  # Context from market structure
        sa.Column('trend_alignment', sa.String(20), nullable=True),  # 'with_trend', 'against_trend', 'neutral'
        sa.Column('psychological_level', sa.Boolean(), default=False),  # Round number levels
        
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), server_default=sa.text('NOW()'), nullable=False),
        sa.Column('updated_at', sa.TIMESTAMP(timezone=True), server_default=sa.text('NOW()'), nullable=False),
        
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create indexes for performance
    op.create_index('idx_sr_symbol_timeframe', 'dynamic_support_resistance', ['symbol', 'timeframe'])
    op.create_index('idx_sr_timestamp', 'dynamic_support_resistance', ['timestamp'])
    op.create_index('idx_sr_level_type', 'dynamic_support_resistance', ['level_type'])
    op.create_index('idx_sr_price_level', 'dynamic_support_resistance', ['price_level'])
    op.create_index('idx_sr_strength', 'dynamic_support_resistance', ['strength'])
    op.create_index('idx_sr_active', 'dynamic_support_resistance', ['is_active'])
    op.create_index('idx_sr_broken', 'dynamic_support_resistance', ['is_broken'])
    
    # Volume-weighted support/resistance table
    op.create_table(
        'volume_weighted_levels',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('symbol', sa.String(20), nullable=False),
        sa.Column('timeframe', sa.String(10), nullable=False),
        sa.Column('timestamp', sa.TIMESTAMP(timezone=True), nullable=False),
        
        # Volume-weighted level information
        sa.Column('level_type', sa.String(20), nullable=False),  # 'vwap_support', 'vwap_resistance', 'volume_node'
        sa.Column('price_level', sa.Numeric(20, 8), nullable=False),
        sa.Column('volume_weight', sa.Numeric(20, 8), nullable=False),
        sa.Column('volume_percentage', sa.Numeric(6, 3), nullable=True),  # % of total volume
        
        # Volume analysis
        sa.Column('total_volume_at_level', sa.Numeric(20, 8), nullable=False),
        sa.Column('buy_volume', sa.Numeric(20, 8), nullable=True),
        sa.Column('sell_volume', sa.Numeric(20, 8), nullable=True),
        sa.Column('volume_imbalance', sa.Numeric(8, 4), nullable=True),  # Buy/sell imbalance
        
        # Time-based analysis
        sa.Column('time_spent_at_level', sa.Integer(), nullable=True),  # Time in seconds
        sa.Column('price_acceptance', sa.Boolean(), nullable=True),  # Price accepted at level
        sa.Column('level_efficiency', sa.Numeric(4, 3), nullable=True),  # How efficiently price moved through
        
        # Level validation
        sa.Column('validation_score', sa.Numeric(4, 3), nullable=False),
        sa.Column('statistical_significance', sa.Numeric(8, 6), nullable=True),
        sa.Column('level_reliability', sa.String(20), nullable=True),  # 'high', 'medium', 'low'
        
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), server_default=sa.text('NOW()'), nullable=False),
        
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create indexes for volume-weighted levels
    op.create_index('idx_vwl_symbol_timeframe', 'volume_weighted_levels', ['symbol', 'timeframe'])
    op.create_index('idx_vwl_timestamp', 'volume_weighted_levels', ['timestamp'])
    op.create_index('idx_vwl_level_type', 'volume_weighted_levels', ['level_type'])
    op.create_index('idx_vwl_price_level', 'volume_weighted_levels', ['price_level'])
    op.create_index('idx_vwl_volume_weight', 'volume_weighted_levels', ['volume_weight'])
    op.create_index('idx_vwl_validation_score', 'volume_weighted_levels', ['validation_score'])
    
    # Psychological levels table
    op.create_table(
        'psychological_levels',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('symbol', sa.String(20), nullable=False),
        sa.Column('timeframe', sa.String(10), nullable=False),
        sa.Column('timestamp', sa.TIMESTAMP(timezone=True), nullable=False),
        
        # Psychological level information
        sa.Column('level_type', sa.String(30), nullable=False),  # 'round_number', 'fibonacci', 'previous_high_low'
        sa.Column('price_level', sa.Numeric(20, 8), nullable=False),
        sa.Column('psychological_strength', sa.Numeric(4, 3), nullable=False),
        
        # Level characteristics
        sa.Column('round_number_type', sa.String(20), nullable=True),  # 'major', 'minor', 'micro'
        sa.Column('fibonacci_ratio', sa.Numeric(8, 6), nullable=True),  # For fibonacci levels
        sa.Column('historical_significance', sa.Numeric(4, 3), nullable=True),
        
        # Market behavior
        sa.Column('reaction_count', sa.Integer(), default=0),  # Times market reacted to level
        sa.Column('penetration_difficulty', sa.Numeric(4, 3), nullable=True),
        sa.Column('average_reaction_strength', sa.Numeric(8, 4), nullable=True),
        
        # Validation metrics
        sa.Column('back_test_success_rate', sa.Numeric(6, 3), nullable=True),
        sa.Column('forward_test_accuracy', sa.Numeric(6, 3), nullable=True),
        sa.Column('reliability_score', sa.Numeric(4, 3), nullable=False),
        
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), server_default=sa.text('NOW()'), nullable=False),
        sa.Column('updated_at', sa.TIMESTAMP(timezone=True), server_default=sa.text('NOW()'), nullable=False),
        
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create indexes for psychological levels
    op.create_index('idx_psych_symbol_timeframe', 'psychological_levels', ['symbol', 'timeframe'])
    op.create_index('idx_psych_timestamp', 'psychological_levels', ['timestamp'])
    op.create_index('idx_psych_level_type', 'psychological_levels', ['level_type'])
    op.create_index('idx_psych_price_level', 'psychological_levels', ['price_level'])
    op.create_index('idx_psych_strength', 'psychological_levels', ['psychological_strength'])
    op.create_index('idx_psych_reliability', 'psychological_levels', ['reliability_score'])
    
    # Level interactions table
    op.create_table(
        'level_interactions',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('symbol', sa.String(20), nullable=False),
        sa.Column('timeframe', sa.String(10), nullable=False),
        sa.Column('timestamp', sa.TIMESTAMP(timezone=True), nullable=False),
        
        # Interaction details
        sa.Column('level_id', sa.Integer(), nullable=False),  # Reference to the level
        sa.Column('level_type', sa.String(20), nullable=False),  # Type of level interacted with
        sa.Column('interaction_type', sa.String(30), nullable=False),  # 'touch', 'bounce', 'penetration', 'break'
        
        # Price action
        sa.Column('approach_price', sa.Numeric(20, 8), nullable=False),
        sa.Column('interaction_price', sa.Numeric(20, 8), nullable=False),
        sa.Column('reaction_price', sa.Numeric(20, 8), nullable=True),
        sa.Column('price_distance', sa.Numeric(8, 4), nullable=True),  # Distance to level
        
        # Volume and momentum
        sa.Column('interaction_volume', sa.Numeric(20, 8), nullable=True),
        sa.Column('volume_ratio', sa.Numeric(8, 4), nullable=True),  # vs average volume
        sa.Column('momentum_strength', sa.Numeric(8, 4), nullable=True),
        sa.Column('momentum_direction', sa.String(10), nullable=True),  # 'up', 'down'
        
        # Outcome analysis
        sa.Column('reaction_strength', sa.Numeric(8, 4), nullable=True),  # How strong was the reaction
        sa.Column('reaction_duration', sa.Integer(), nullable=True),  # Duration in bars
        sa.Column('success_probability', sa.Numeric(4, 3), nullable=True),  # Predicted success
        sa.Column('actual_outcome', sa.String(20), nullable=True),  # 'bounce', 'break', 'consolidate'
        
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), server_default=sa.text('NOW()'), nullable=False),
        
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create indexes for level interactions
    op.create_index('idx_interactions_symbol_timeframe', 'level_interactions', ['symbol', 'timeframe'])
    op.create_index('idx_interactions_timestamp', 'level_interactions', ['timestamp'])
    op.create_index('idx_interactions_level_id', 'level_interactions', ['level_id'])
    op.create_index('idx_interactions_type', 'level_interactions', ['interaction_type'])
    op.create_index('idx_interactions_outcome', 'level_interactions', ['actual_outcome'])

def downgrade():
    """Drop dynamic support/resistance enhancement tables"""
    
    # Drop tables in reverse order
    op.drop_table('level_interactions')
    op.drop_table('psychological_levels')
    op.drop_table('volume_weighted_levels')
    op.drop_table('dynamic_support_resistance')

async def create_timescaledb_hypertables():
    """Create TimescaleDB hypertables for time-series optimization"""
    try:
        # Get database connection
        from sqlalchemy import create_engine, text
        
        # Create engine (you'll need to configure this with your database URL)
        engine = create_engine('postgresql://username:password@localhost:5432/alphapulse')
        
        with engine.connect() as conn:
            # Create hypertables
            conn.execute(text("""
                SELECT create_hypertable('dynamic_support_resistance', 'timestamp', if_not_exists => TRUE);
            """))
            
            conn.execute(text("""
                SELECT create_hypertable('volume_weighted_levels', 'timestamp', if_not_exists => TRUE);
            """))
            
            conn.execute(text("""
                SELECT create_hypertable('psychological_levels', 'timestamp', if_not_exists => TRUE);
            """))
            
            conn.execute(text("""
                SELECT create_hypertable('level_interactions', 'timestamp', if_not_exists => TRUE);
            """))
            
            conn.commit()
            
        logger.info("✅ TimescaleDB hypertables for support/resistance created successfully")
        
    except Exception as e:
        logger.error(f"❌ Error creating TimescaleDB hypertables: {e}")
        raise

if __name__ == "__main__":
    # Run the migration
    asyncio.run(create_timescaledb_hypertables())
