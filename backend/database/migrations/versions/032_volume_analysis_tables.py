"""Migration 032: Volume Analysis Tables

Revision ID: 032_volume_analysis_tables
Revises: 031_demand_supply_zones
Create Date: 2025-01-23 20:05:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB, TIMESTAMP

# revision identifiers, used by Alembic.
revision = '032_volume_analysis_tables'
down_revision = '031_demand_supply_zones'
branch_labels = None
depends_on = None

def upgrade():
    """Create volume analysis tables"""
    
    # Volume Analysis Table
    volume_analysis_table = """
    CREATE TABLE IF NOT EXISTS volume_analysis (
        id SERIAL,
        symbol VARCHAR(20) NOT NULL,
        timestamp TIMESTAMPTZ NOT NULL,
        timeframe VARCHAR(10) NOT NULL,
        
        -- Volume Metrics
        total_volume DECIMAL(20, 8) NOT NULL,
        volume_ma_short DECIMAL(20, 8),
        volume_ma_long DECIMAL(20, 8),
        volume_ratio DECIMAL(10, 4),
        volume_trend VARCHAR(20),
        
        -- Volume Profile
        poc_level DECIMAL(20, 8),
        value_area_high DECIMAL(20, 8),
        value_area_low DECIMAL(20, 8),
        volume_nodes JSONB,
        volume_gaps JSONB,
        
        -- Volume Positioning
        volume_positioning_score DECIMAL(5, 4),
        volume_confirmation BOOLEAN,
        volume_divergence BOOLEAN,
        volume_climax BOOLEAN,
        
        -- Market Context
        price_level DECIMAL(20, 8) NOT NULL,
        trend_direction VARCHAR(20),
        market_structure VARCHAR(50),
        
        -- Analysis Results
        analysis_confidence DECIMAL(3, 2),
        signal_strength VARCHAR(20),
        trading_implications JSONB,
        
        -- Metadata
        created_at TIMESTAMPTZ DEFAULT NOW(),
        updated_at TIMESTAMPTZ DEFAULT NOW(),
        
        PRIMARY KEY (timestamp, id)
    );
    """
    
    # Create TimescaleDB hypertable
    create_hypertable = """
    SELECT create_hypertable('volume_analysis', 'timestamp', 
                           chunk_time_interval => INTERVAL '1 day',
                           if_not_exists => TRUE);
    """
    
    # Create indexes
    indexes = [
        "CREATE INDEX IF NOT EXISTS idx_volume_analysis_symbol_timestamp ON volume_analysis (symbol, timestamp DESC);",
        "CREATE INDEX IF NOT EXISTS idx_volume_analysis_timeframe ON volume_analysis (timeframe);",
        "CREATE INDEX IF NOT EXISTS idx_volume_analysis_volume_trend ON volume_analysis (volume_trend);",
        "CREATE INDEX IF NOT EXISTS idx_volume_analysis_confidence ON volume_analysis (analysis_confidence DESC);"
    ]
    
    # Execute SQL in correct order
    op.execute(volume_analysis_table)
    op.execute(create_hypertable)
    
    # Wait a moment for table creation to complete
    import time
    time.sleep(1)
    
    for index in indexes:
        op.execute(index)

def downgrade():
    """Drop volume analysis tables"""
    op.execute("DROP TABLE IF EXISTS volume_analysis CASCADE;")
