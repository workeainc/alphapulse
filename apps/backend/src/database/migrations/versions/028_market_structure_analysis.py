"""
Migration 028: Market Structure Analysis Tables
Adds comprehensive market structure analysis capabilities to AlphaPulse
"""

import asyncio
import logging
from sqlalchemy import text
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

logger = logging.getLogger(__name__)

# revision identifiers, used by Alembic
revision = '028_market_structure_analysis'
down_revision = None  # This is the first migration
branch_labels = None
depends_on = None

def upgrade():
    """Create market structure analysis tables"""
    
    # Market structure analysis table
    op.create_table(
        'market_structure_analysis',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('symbol', sa.String(20), nullable=False),
        sa.Column('timeframe', sa.String(10), nullable=False),
        sa.Column('timestamp', sa.TIMESTAMP(timezone=True), nullable=False),
        
        # Market structure patterns
        sa.Column('higher_highs', JSONB, nullable=True),  # Array of HH points
        sa.Column('lower_highs', JSONB, nullable=True),   # Array of LH points
        sa.Column('higher_lows', JSONB, nullable=True),   # Array of HL points
        sa.Column('lower_lows', JSONB, nullable=True),    # Array of LL points
        
        # Structure analysis
        sa.Column('market_structure_type', sa.String(50), nullable=True),  # 'uptrend', 'downtrend', 'consolidation'
        sa.Column('structure_strength', sa.Numeric(4, 3), nullable=True),  # 0-1 confidence
        sa.Column('structure_breakout', sa.Boolean(), nullable=True),      # True if structure broken
        sa.Column('breakout_direction', sa.String(10), nullable=True),     # 'up', 'down'
        
        # Trend lines
        sa.Column('trend_lines', JSONB, nullable=True),       # Dynamic trend lines
        sa.Column('trend_line_breaks', JSONB, nullable=True), # Trend line breakouts
        
        # Analysis metadata
        sa.Column('analysis_confidence', sa.Numeric(4, 3), nullable=True),  # Overall confidence
        sa.Column('last_swing_high', sa.Numeric(20, 8), nullable=True),     # Last swing high price
        sa.Column('last_swing_low', sa.Numeric(20, 8), nullable=True),      # Last swing low price
        sa.Column('current_structure_phase', sa.String(30), nullable=True), # 'accumulation', 'markup', 'distribution', 'markdown'
        
        # Performance metrics
        sa.Column('structure_duration_bars', sa.Integer(), nullable=True),  # How long current structure has been active
        sa.Column('structure_quality_score', sa.Numeric(4, 3), nullable=True), # Quality of structure formation
        
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), server_default=sa.text('NOW()'), nullable=False),
        
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create indexes for performance
    op.create_index('idx_market_structure_symbol_timeframe', 'market_structure_analysis', ['symbol', 'timeframe'])
    op.create_index('idx_market_structure_timestamp', 'market_structure_analysis', ['timestamp'])
    op.create_index('idx_market_structure_type', 'market_structure_analysis', ['market_structure_type'])
    op.create_index('idx_market_structure_breakout', 'market_structure_analysis', ['structure_breakout'])
    
    # Market structure breakouts table
    op.create_table(
        'market_structure_breakouts',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('symbol', sa.String(20), nullable=False),
        sa.Column('timeframe', sa.String(10), nullable=False),
        sa.Column('timestamp', sa.TIMESTAMP(timezone=True), nullable=False),
        
        # Breakout information
        sa.Column('breakout_type', sa.String(20), nullable=False),  # 'structure_breakout', 'trend_line_break', 'level_break'
        sa.Column('breakout_direction', sa.String(10), nullable=False),  # 'up', 'down'
        sa.Column('breakout_price', sa.Numeric(20, 8), nullable=False),
        sa.Column('breakout_volume', sa.Numeric(20, 8), nullable=True),
        
        # Breakout strength
        sa.Column('breakout_strength', sa.Numeric(4, 3), nullable=True),  # 0-1 strength
        sa.Column('volume_confirmation', sa.Boolean(), nullable=True),    # Volume confirms breakout
        sa.Column('momentum_confirmation', sa.Boolean(), nullable=True),  # Momentum confirms breakout
        
        # Breakout context
        sa.Column('previous_structure', sa.String(50), nullable=True),    # Previous market structure
        sa.Column('breakout_level', sa.Numeric(20, 8), nullable=True),    # Level that was broken
        sa.Column('breakout_distance', sa.Numeric(8, 4), nullable=True),  # Distance from breakout level
        
        # Signal generation
        sa.Column('signal_generated', sa.Boolean(), default=False),       # True if signal was generated
        sa.Column('signal_confidence', sa.Numeric(4, 3), nullable=True),  # Signal confidence
        sa.Column('signal_direction', sa.String(10), nullable=True),      # 'long', 'short'
        
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), server_default=sa.text('NOW()'), nullable=False),
        
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create indexes for breakouts
    op.create_index('idx_breakouts_symbol_timeframe', 'market_structure_breakouts', ['symbol', 'timeframe'])
    op.create_index('idx_breakouts_timestamp', 'market_structure_breakouts', ['timestamp'])
    op.create_index('idx_breakouts_type', 'market_structure_breakouts', ['breakout_type'])
    op.create_index('idx_breakouts_direction', 'market_structure_breakouts', ['breakout_direction'])
    op.create_index('idx_breakouts_signal', 'market_structure_breakouts', ['signal_generated'])
    
    # Trend line analysis table
    op.create_table(
        'trend_line_analysis',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('symbol', sa.String(20), nullable=False),
        sa.Column('timeframe', sa.String(10), nullable=False),
        sa.Column('timestamp', sa.TIMESTAMP(timezone=True), nullable=False),
        
        # Trend line information
        sa.Column('trend_line_type', sa.String(20), nullable=False),  # 'support', 'resistance', 'dynamic'
        sa.Column('trend_line_direction', sa.String(10), nullable=False),  # 'up', 'down'
        sa.Column('start_price', sa.Numeric(20, 8), nullable=False),
        sa.Column('end_price', sa.Numeric(20, 8), nullable=False),
        sa.Column('start_time', sa.TIMESTAMP(timezone=True), nullable=False),
        sa.Column('end_time', sa.TIMESTAMP(timezone=True), nullable=False),
        
        # Trend line validation
        sa.Column('touch_count', sa.Integer(), default=0),  # Number of touches
        sa.Column('touch_points', JSONB, nullable=True),    # Array of touch points
        sa.Column('validation_score', sa.Numeric(4, 3), nullable=True),  # 0-1 validation score
        
        # Trend line status
        sa.Column('trend_line_active', sa.Boolean(), default=True),  # True if trend line is still valid
        sa.Column('trend_line_broken', sa.Boolean(), default=False), # True if trend line was broken
        sa.Column('break_time', sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column('break_price', sa.Numeric(20, 8), nullable=True),
        
        # Trend line strength
        sa.Column('trend_line_strength', sa.Numeric(4, 3), nullable=True),  # 0-1 strength
        sa.Column('slope_angle', sa.Numeric(6, 3), nullable=True),          # Slope angle in degrees
        sa.Column('duration_bars', sa.Integer(), nullable=True),            # Duration in bars
        
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), server_default=sa.text('NOW()'), nullable=False),
        
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create indexes for trend lines
    op.create_index('idx_trend_lines_symbol_timeframe', 'trend_line_analysis', ['symbol', 'timeframe'])
    op.create_index('idx_trend_lines_timestamp', 'trend_line_analysis', ['timestamp'])
    op.create_index('idx_trend_lines_type', 'trend_line_analysis', ['trend_line_type'])
    op.create_index('idx_trend_lines_active', 'trend_line_analysis', ['trend_line_active'])
    op.create_index('idx_trend_lines_broken', 'trend_line_analysis', ['trend_line_broken'])
    
    # Swing points table for storing HH/LH/HL/LL points
    op.create_table(
        'swing_points',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('symbol', sa.String(20), nullable=False),
        sa.Column('timeframe', sa.String(10), nullable=False),
        sa.Column('timestamp', sa.TIMESTAMP(timezone=True), nullable=False),
        
        # Swing point information
        sa.Column('swing_type', sa.String(10), nullable=False),  # 'high', 'low'
        sa.Column('price', sa.Numeric(20, 8), nullable=False),
        sa.Column('volume', sa.Numeric(20, 8), nullable=True),
        
        # Swing point classification
        sa.Column('is_higher_high', sa.Boolean(), nullable=True),  # True if this is a higher high
        sa.Column('is_lower_high', sa.Boolean(), nullable=True),   # True if this is a lower high
        sa.Column('is_higher_low', sa.Boolean(), nullable=True),   # True if this is a higher low
        sa.Column('is_lower_low', sa.Boolean(), nullable=True),    # True if this is a lower low
        
        # Swing point strength
        sa.Column('swing_strength', sa.Numeric(4, 3), nullable=True),  # 0-1 strength
        sa.Column('volume_confirmation', sa.Boolean(), nullable=True), # Volume confirms swing
        sa.Column('momentum_confirmation', sa.Boolean(), nullable=True), # Momentum confirms swing
        
        # Swing point context
        sa.Column('previous_swing_price', sa.Numeric(20, 8), nullable=True),  # Previous swing point price
        sa.Column('swing_distance', sa.Numeric(8, 4), nullable=True),         # Distance from previous swing
        sa.Column('swing_duration_bars', sa.Integer(), nullable=True),        # Duration since previous swing
        
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), server_default=sa.text('NOW()'), nullable=False),
        
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create indexes for swing points
    op.create_index('idx_swing_points_symbol_timeframe', 'swing_points', ['symbol', 'timeframe'])
    op.create_index('idx_swing_points_timestamp', 'swing_points', ['timestamp'])
    op.create_index('idx_swing_points_type', 'swing_points', ['swing_type'])
    op.create_index('idx_swing_points_hh', 'swing_points', ['is_higher_high'])
    op.create_index('idx_swing_points_lh', 'swing_points', ['is_lower_high'])
    op.create_index('idx_swing_points_hl', 'swing_points', ['is_higher_low'])
    op.create_index('idx_swing_points_ll', 'swing_points', ['is_lower_low'])

def downgrade():
    """Drop market structure analysis tables"""
    
    # Drop tables in reverse order
    op.drop_table('swing_points')
    op.drop_table('trend_line_analysis')
    op.drop_table('market_structure_breakouts')
    op.drop_table('market_structure_analysis')

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
                SELECT create_hypertable('market_structure_analysis', 'timestamp', if_not_exists => TRUE);
            """))
            
            conn.execute(text("""
                SELECT create_hypertable('market_structure_breakouts', 'timestamp', if_not_exists => TRUE);
            """))
            
            conn.execute(text("""
                SELECT create_hypertable('trend_line_analysis', 'timestamp', if_not_exists => TRUE);
            """))
            
            conn.execute(text("""
                SELECT create_hypertable('swing_points', 'timestamp', if_not_exists => TRUE);
            """))
            
            conn.commit()
            
        logger.info("✅ TimescaleDB hypertables created successfully")
        
    except Exception as e:
        logger.error(f"❌ Error creating TimescaleDB hypertables: {e}")
        raise

if __name__ == "__main__":
    # Run the migration
    asyncio.run(create_timescaledb_hypertables())
