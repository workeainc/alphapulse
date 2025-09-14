"""
Migration: Enhanced Leverage, Liquidity, and Order Book Analysis Tables
Add comprehensive tables for futures data, liquidation events, and order book analysis
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB, TIMESTAMP, NUMERIC

# revision identifiers, used by Alembic.
revision = '021_enhanced_leverage_liquidity_orderbook'
down_revision = '020_self_training_ml_system'
branch_labels = None
depends_on = None

def upgrade():
    """Create enhanced leverage, liquidity, and order book analysis tables"""
    
    # ==================== ENHANCED ORDER BOOK TABLES ====================
    
    # Enhanced order book snapshots table with delta storage
    op.create_table(
        'enhanced_order_book_snapshots',
        sa.Column('id', sa.BigInteger(), nullable=False),
        sa.Column('symbol', sa.String(20), nullable=False),
        sa.Column('timestamp', TIMESTAMP(timezone=True), nullable=False),
        sa.Column('exchange', sa.String(20), nullable=False),
        sa.Column('bids', JSONB, nullable=False),  # Array of [price, volume] pairs
        sa.Column('asks', JSONB, nullable=False),  # Array of [price, volume] pairs
        sa.Column('spread', NUMERIC(20, 8), nullable=True),
        sa.Column('total_bid_volume', NUMERIC(20, 8), nullable=True),
        sa.Column('total_ask_volume', NUMERIC(20, 8), nullable=True),
        sa.Column('depth_levels', sa.Integer(), nullable=False),
        sa.Column('bid_ask_imbalance', NUMERIC(10, 6), nullable=True),  # -1 to +1
        sa.Column('liquidity_score', NUMERIC(10, 6), nullable=True),  # 0 to 1
        sa.Column('order_flow_toxicity', NUMERIC(10, 6), nullable=True),  # -1 to +1
        sa.Column('depth_pressure', NUMERIC(10, 6), nullable=True),  # -1 to +1
        sa.Column('metadata', JSONB, nullable=True),
        sa.Column('created_at', TIMESTAMP(timezone=True), server_default=sa.text('NOW()'), nullable=False)
    )
    
    # Create TimescaleDB hypertable for order book data
    op.execute("""
        SELECT create_hypertable('enhanced_order_book_snapshots', 'timestamp', 
            if_not_exists => TRUE, 
            chunk_time_interval => INTERVAL '1 hour'
        );
    """)
    
    # Create indexes for enhanced order book snapshots
    op.create_index('idx_enhanced_order_book_symbol_time', 'enhanced_order_book_snapshots', ['symbol', 'timestamp'])
    op.create_index('idx_enhanced_order_book_exchange_time', 'enhanced_order_book_snapshots', ['exchange', 'timestamp'])
    op.create_index('idx_enhanced_order_book_imbalance', 'enhanced_order_book_snapshots', ['bid_ask_imbalance'])
    op.create_index('idx_enhanced_order_book_liquidity', 'enhanced_order_book_snapshots', ['liquidity_score'])
    
    # Order book delta updates table
    op.create_table(
        'order_book_deltas',
        sa.Column('id', sa.BigInteger(), nullable=False),
        sa.Column('symbol', sa.String(20), nullable=False),
        sa.Column('timestamp', TIMESTAMP(timezone=True), nullable=False),
        sa.Column('exchange', sa.String(20), nullable=False),
        sa.Column('bids_delta', JSONB, nullable=False),  # Array of [price, volume] pairs
        sa.Column('asks_delta', JSONB, nullable=False),  # Array of [price, volume] pairs
        sa.Column('sequence_number', sa.BigInteger(), nullable=True),
        sa.Column('delta_size', sa.Integer(), nullable=True),  # Number of changes
        sa.Column('impact_score', NUMERIC(10, 6), nullable=True),  # 0 to 1
        sa.Column('metadata', JSONB, nullable=True),
        sa.Column('created_at', TIMESTAMP(timezone=True), server_default=sa.text('NOW()'), nullable=False)
    )
    
    # Create TimescaleDB hypertable for order book deltas
    op.execute("""
        SELECT create_hypertable('order_book_deltas', 'timestamp', 
            if_not_exists => TRUE, 
            chunk_time_interval => INTERVAL '30 minutes'
        );
    """)
    
    # Create indexes for order book deltas
    op.create_index('idx_order_book_deltas_symbol_time', 'order_book_deltas', ['symbol', 'timestamp'])
    op.create_index('idx_order_book_deltas_sequence', 'order_book_deltas', ['sequence_number'])
    op.create_index('idx_order_book_deltas_impact', 'order_book_deltas', ['impact_score'])
    
    # ==================== LIQUIDATION EVENTS TABLES ====================
    
    # Liquidation events table
    op.create_table(
        'liquidation_events',
        sa.Column('id', sa.BigInteger(), nullable=False),
        sa.Column('symbol', sa.String(20), nullable=False),
        sa.Column('timestamp', TIMESTAMP(timezone=True), nullable=False),
        sa.Column('exchange', sa.String(20), nullable=False),
        sa.Column('side', sa.String(10), nullable=False),  # 'long' or 'short'
        sa.Column('price', NUMERIC(20, 8), nullable=False),
        sa.Column('quantity', NUMERIC(20, 8), nullable=False),
        sa.Column('quote_quantity', NUMERIC(20, 8), nullable=False),
        sa.Column('liquidation_type', sa.String(20), nullable=False),  # 'partial' or 'full'
        sa.Column('leverage_level', sa.Integer(), nullable=True),
        sa.Column('impact_score', NUMERIC(10, 6), nullable=True),  # 0 to 1
        sa.Column('cluster_id', sa.String(50), nullable=True),  # For grouping related liquidations
        sa.Column('metadata', JSONB, nullable=True),
        sa.Column('created_at', TIMESTAMP(timezone=True), server_default=sa.text('NOW()'), nullable=False)
    )
    
    # Create TimescaleDB hypertable for liquidation events
    op.execute("""
        SELECT create_hypertable('liquidation_events', 'timestamp', 
            if_not_exists => TRUE, 
            chunk_time_interval => INTERVAL '1 hour'
        );
    """)
    
    # Create indexes for liquidation events
    op.create_index('idx_liquidation_events_symbol_time', 'liquidation_events', ['symbol', 'timestamp'])
    op.create_index('idx_liquidation_events_side', 'liquidation_events', ['side'])
    op.create_index('idx_liquidation_events_cluster', 'liquidation_events', ['cluster_id'])
    op.create_index('idx_liquidation_events_impact', 'liquidation_events', ['impact_score'])
    
    # Liquidation levels table for risk management
    op.create_table(
        'liquidation_levels',
        sa.Column('id', sa.BigInteger(), nullable=False),
        sa.Column('symbol', sa.String(20), nullable=False),
        sa.Column('timestamp', TIMESTAMP(timezone=True), nullable=False),
        sa.Column('exchange', sa.String(20), nullable=False),
        sa.Column('price_level', NUMERIC(20, 8), nullable=False),
        sa.Column('side', sa.String(10), nullable=False),  # 'long' or 'short'
        sa.Column('quantity', NUMERIC(20, 8), nullable=False),
        sa.Column('risk_score', NUMERIC(10, 6), nullable=True),  # 0 to 1
        sa.Column('distance_from_price', NUMERIC(10, 6), nullable=True),  # Percentage
        sa.Column('metadata', JSONB, nullable=True),
        sa.Column('created_at', TIMESTAMP(timezone=True), server_default=sa.text('NOW()'), nullable=False)
    )
    
    # Create TimescaleDB hypertable for liquidation levels
    op.execute("""
        SELECT create_hypertable('liquidation_levels', 'timestamp', 
            if_not_exists => TRUE, 
            chunk_time_interval => INTERVAL '1 hour'
        );
    """)
    
    # Create indexes for liquidation levels
    op.create_index('idx_liquidation_levels_symbol_time', 'liquidation_levels', ['symbol', 'timestamp'])
    op.create_index('idx_liquidation_levels_price', 'liquidation_levels', ['price_level'])
    op.create_index('idx_liquidation_levels_risk', 'liquidation_levels', ['risk_score'])
    
    # ==================== FUTURES DATA TABLES ====================
    
    # Open interest table
    op.create_table(
        'open_interest',
        sa.Column('id', sa.BigInteger(), nullable=False),
        sa.Column('symbol', sa.String(20), nullable=False),
        sa.Column('timestamp', TIMESTAMP(timezone=True), nullable=False),
        sa.Column('exchange', sa.String(20), nullable=False),
        sa.Column('open_interest', NUMERIC(20, 8), nullable=False),
        sa.Column('open_interest_value', NUMERIC(20, 8), nullable=False),  # In quote currency
        sa.Column('change_24h', NUMERIC(10, 6), nullable=True),  # Percentage change
        sa.Column('change_1h', NUMERIC(10, 6), nullable=True),  # Percentage change
        sa.Column('metadata', JSONB, nullable=True),
        sa.Column('created_at', TIMESTAMP(timezone=True), server_default=sa.text('NOW()'), nullable=False)
    )
    
    # Create TimescaleDB hypertable for open interest
    op.execute("""
        SELECT create_hypertable('open_interest', 'timestamp', 
            if_not_exists => TRUE, 
            chunk_time_interval => INTERVAL '1 hour'
        );
    """)
    
    # Create indexes for open interest
    op.create_index('idx_open_interest_symbol_time', 'open_interest', ['symbol', 'timestamp'])
    op.create_index('idx_open_interest_value', 'open_interest', ['open_interest_value'])
    op.create_index('idx_open_interest_change', 'open_interest', ['change_24h'])
    
    # Enhanced funding rates table
    op.create_table(
        'enhanced_funding_rates',
        sa.Column('id', sa.BigInteger(), nullable=False),
        sa.Column('symbol', sa.String(20), nullable=False),
        sa.Column('timestamp', TIMESTAMP(timezone=True), nullable=False),
        sa.Column('exchange', sa.String(20), nullable=False),
        sa.Column('funding_rate', NUMERIC(10, 8), nullable=False),
        sa.Column('next_funding_time', TIMESTAMP(timezone=True), nullable=True),
        sa.Column('estimated_rate', NUMERIC(10, 8), nullable=True),
        sa.Column('predicted_rate', NUMERIC(10, 8), nullable=True),
        sa.Column('rate_change_1h', NUMERIC(10, 8), nullable=True),
        sa.Column('rate_change_8h', NUMERIC(10, 8), nullable=True),
        sa.Column('funding_impact_score', NUMERIC(10, 6), nullable=True),  # 0 to 1
        sa.Column('metadata', JSONB, nullable=True),
        sa.Column('created_at', TIMESTAMP(timezone=True), server_default=sa.text('NOW()'), nullable=False)
    )
    
    # Create TimescaleDB hypertable for funding rates
    op.execute("""
        SELECT create_hypertable('enhanced_funding_rates', 'timestamp', 
            if_not_exists => TRUE, 
            chunk_time_interval => INTERVAL '1 hour'
        );
    """)
    
    # Create indexes for funding rates
    op.create_index('idx_enhanced_funding_rates_symbol_time', 'enhanced_funding_rates', ['symbol', 'timestamp'])
    op.create_index('idx_enhanced_funding_rates_rate', 'enhanced_funding_rates', ['funding_rate'])
    op.create_index('idx_enhanced_funding_rates_impact', 'enhanced_funding_rates', ['funding_impact_score'])
    
    # ==================== MARKET DEPTH ANALYSIS TABLES ====================
    
    # Market depth analysis table
    op.create_table(
        'market_depth_analysis',
        sa.Column('id', sa.BigInteger(), nullable=False),
        sa.Column('symbol', sa.String(20), nullable=False),
        sa.Column('timestamp', TIMESTAMP(timezone=True), nullable=False),
        sa.Column('exchange', sa.String(20), nullable=False),
        sa.Column('analysis_type', sa.String(50), nullable=False),  # 'liquidity_walls', 'order_clusters', 'imbalance'
        sa.Column('price_level', NUMERIC(20, 8), nullable=False),
        sa.Column('volume_at_level', NUMERIC(20, 8), nullable=False),
        sa.Column('side', sa.String(10), nullable=False),  # 'bid' or 'ask'
        sa.Column('confidence', NUMERIC(4, 3), nullable=True),  # 0 to 1
        sa.Column('strength_score', NUMERIC(10, 6), nullable=True),  # 0 to 1
        sa.Column('distance_from_mid', NUMERIC(10, 6), nullable=True),  # Percentage
        sa.Column('wall_thickness', sa.Integer(), nullable=True),  # Number of levels
        sa.Column('metadata', JSONB, nullable=True),
        sa.Column('created_at', TIMESTAMP(timezone=True), server_default=sa.text('NOW()'), nullable=False)
    )
    
    # Create TimescaleDB hypertable for market depth analysis
    op.execute("""
        SELECT create_hypertable('market_depth_analysis', 'timestamp', 
            if_not_exists => TRUE, 
            chunk_time_interval => INTERVAL '1 hour'
        );
    """)
    
    # Create indexes for market depth analysis
    op.create_index('idx_market_depth_analysis_symbol_time', 'market_depth_analysis', ['symbol', 'timestamp'])
    op.create_index('idx_market_depth_analysis_type', 'market_depth_analysis', ['analysis_type'])
    op.create_index('idx_market_depth_analysis_confidence', 'market_depth_analysis', ['confidence'])
    op.create_index('idx_market_depth_analysis_strength', 'market_depth_analysis', ['strength_score'])
    
    # ==================== ENHANCED TRADES TABLE UPDATES ====================
    
    # Add new columns to existing trades table
    op.add_column('trades', sa.Column('leverage_ratio', NUMERIC(10, 6), nullable=True))
    op.add_column('trades', sa.Column('margin_used', NUMERIC(20, 8), nullable=True))
    op.add_column('trades', sa.Column('liquidation_price', NUMERIC(20, 8), nullable=True))
    op.add_column('trades', sa.Column('risk_score', NUMERIC(10, 6), nullable=True))
    op.add_column('trades', sa.Column('liquidity_score', NUMERIC(10, 6), nullable=True))
    op.add_column('trades', sa.Column('order_book_imbalance', NUMERIC(10, 6), nullable=True))
    op.add_column('trades', sa.Column('funding_rate_at_entry', NUMERIC(10, 8), nullable=True))
    op.add_column('trades', sa.Column('open_interest_at_entry', NUMERIC(20, 8), nullable=True))
    op.add_column('trades', sa.Column('market_depth_analysis', JSONB, nullable=True))
    
    # Create indexes for new trade columns
    op.create_index('idx_trades_leverage_ratio', 'trades', ['leverage_ratio'])
    op.create_index('idx_trades_risk_score', 'trades', ['risk_score'])
    op.create_index('idx_trades_liquidity_score', 'trades', ['liquidity_score'])
    op.create_index('idx_trades_liquidation_price', 'trades', ['liquidation_price'])

def downgrade():
    """Remove enhanced leverage, liquidity, and order book analysis tables"""
    
    # Drop indexes for trades table
    op.drop_index('idx_trades_liquidation_price', 'trades')
    op.drop_index('idx_trades_liquidity_score', 'trades')
    op.drop_index('idx_trades_risk_score', 'trades')
    op.drop_index('idx_trades_leverage_ratio', 'trades')
    
    # Drop new columns from trades table
    op.drop_column('trades', 'market_depth_analysis')
    op.drop_column('trades', 'open_interest_at_entry')
    op.drop_column('trades', 'funding_rate_at_entry')
    op.drop_column('trades', 'order_book_imbalance')
    op.drop_column('trades', 'liquidity_score')
    op.drop_column('trades', 'risk_score')
    op.drop_column('trades', 'liquidation_price')
    op.drop_column('trades', 'margin_used')
    op.drop_column('trades', 'leverage_ratio')
    
    # Drop market depth analysis table
    op.drop_index('idx_market_depth_analysis_strength', 'market_depth_analysis')
    op.drop_index('idx_market_depth_analysis_confidence', 'market_depth_analysis')
    op.drop_index('idx_market_depth_analysis_type', 'market_depth_analysis')
    op.drop_index('idx_market_depth_analysis_symbol_time', 'market_depth_analysis')
    op.drop_table('market_depth_analysis')
    
    # Drop enhanced funding rates table
    op.drop_index('idx_enhanced_funding_rates_impact', 'enhanced_funding_rates')
    op.drop_index('idx_enhanced_funding_rates_rate', 'enhanced_funding_rates')
    op.drop_index('idx_enhanced_funding_rates_symbol_time', 'enhanced_funding_rates')
    op.drop_table('enhanced_funding_rates')
    
    # Drop open interest table
    op.drop_index('idx_open_interest_change', 'open_interest')
    op.drop_index('idx_open_interest_value', 'open_interest')
    op.drop_index('idx_open_interest_symbol_time', 'open_interest')
    op.drop_table('open_interest')
    
    # Drop liquidation levels table
    op.drop_index('idx_liquidation_levels_risk', 'liquidation_levels')
    op.drop_index('idx_liquidation_levels_price', 'liquidation_levels')
    op.drop_index('idx_liquidation_levels_symbol_time', 'liquidation_levels')
    op.drop_table('liquidation_levels')
    
    # Drop liquidation events table
    op.drop_index('idx_liquidation_events_impact', 'liquidation_events')
    op.drop_index('idx_liquidation_events_cluster', 'liquidation_events')
    op.drop_index('idx_liquidation_events_side', 'liquidation_events')
    op.drop_index('idx_liquidation_events_symbol_time', 'liquidation_events')
    op.drop_table('liquidation_events')
    
    # Drop order book deltas table
    op.drop_index('idx_order_book_deltas_impact', 'order_book_deltas')
    op.drop_index('idx_order_book_deltas_sequence', 'order_book_deltas')
    op.drop_index('idx_order_book_deltas_symbol_time', 'order_book_deltas')
    op.drop_table('order_book_deltas')
    
    # Drop enhanced order book snapshots table
    op.drop_index('idx_enhanced_order_book_liquidity', 'enhanced_order_book_snapshots')
    op.drop_index('idx_enhanced_order_book_imbalance', 'enhanced_order_book_snapshots')
    op.drop_index('idx_enhanced_order_book_exchange_time', 'enhanced_order_book_snapshots')
    op.drop_index('idx_enhanced_order_book_symbol_time', 'enhanced_order_book_snapshots')
    op.drop_table('enhanced_order_book_snapshots')
