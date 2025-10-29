#!/usr/bin/env python3
"""
Migration: Create Volume Analysis Tables for Advanced Volume Pattern Detection
TimescaleDB schema for storing comprehensive volume analysis and pattern detection
"""

import asyncio
import logging
import os
import subprocess
import asyncpg
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database connection parameters
DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'database': 'alphapulse',
    'user': 'alpha_emon',
    'password': 'Emon_@17711'
}

async def create_volume_analysis_tables():
    """Create volume analysis tables with TimescaleDB optimizations"""
    
    conn = None
    try:
        # Connect to database
        conn = await asyncpg.connect(**DB_CONFIG)
        logger.info("🔗 Connected to database for volume analysis table creation")
        
        # Create main volume analysis table
        volume_analysis_table = """
    CREATE TABLE IF NOT EXISTS volume_analysis (
        id SERIAL,
        symbol VARCHAR(20) NOT NULL,
        timeframe VARCHAR(10) NOT NULL,
        timestamp TIMESTAMPTZ NOT NULL,
        volume_ratio DECIMAL(6,3) NOT NULL,
        volume_trend VARCHAR(20) NOT NULL,
        volume_positioning_score DECIMAL(3,2) NOT NULL,
        order_book_imbalance DECIMAL(6,3) NOT NULL,
        buy_volume_ratio DECIMAL(3,2) NOT NULL,
        sell_volume_ratio DECIMAL(3,2) NOT NULL,
        volume_breakout BOOLEAN NOT NULL DEFAULT FALSE,
        -- Advanced metrics
        vwap DECIMAL(12,6),
        cumulative_volume_delta DECIMAL(15,3),
        relative_volume DECIMAL(6,3),
        volume_weighted_price DECIMAL(12,6),
        volume_flow_imbalance DECIMAL(8,4),
        -- Pattern detection
        volume_pattern_type VARCHAR(50),
        volume_pattern_strength VARCHAR(20),
        volume_pattern_confidence DECIMAL(3,2),
        volume_analysis TEXT,
        volume_context JSONB,
        created_at TIMESTAMPTZ DEFAULT NOW(),
        PRIMARY KEY (timestamp, id)
    );
    """
    
        # Create volume patterns table
        volume_patterns_table = """
    CREATE TABLE IF NOT EXISTS volume_patterns (
        id SERIAL,
        symbol VARCHAR(20) NOT NULL,
        timeframe VARCHAR(10) NOT NULL,
        timestamp TIMESTAMPTZ NOT NULL,
        pattern_type VARCHAR(50) NOT NULL,
        pattern_direction VARCHAR(10) NOT NULL,
        pattern_strength VARCHAR(20) NOT NULL,
        pattern_confidence DECIMAL(3,2) NOT NULL,
        volume_spike_multiplier DECIMAL(4,2),
        volume_divergence_type VARCHAR(30),
        pattern_description TEXT,
        pattern_metadata JSONB,
        created_at TIMESTAMPTZ DEFAULT NOW(),
        PRIMARY KEY (timestamp, id)
    );
    """
    
        # Create delta profile table for footprint charts
        delta_profile_table = """
    CREATE TABLE IF NOT EXISTS delta_profile (
        id SERIAL,
        symbol VARCHAR(20) NOT NULL,
        timeframe VARCHAR(10) NOT NULL,
        timestamp TIMESTAMPTZ NOT NULL,
        price_level DECIMAL(12,6) NOT NULL,
        price_band_start DECIMAL(12,6) NOT NULL,
        price_band_end DECIMAL(12,6) NOT NULL,
        volume_at_level DECIMAL(15,3) NOT NULL,
        buy_volume DECIMAL(15,3) NOT NULL,
        sell_volume DECIMAL(15,3) NOT NULL,
        delta_imbalance DECIMAL(15,3) NOT NULL,
        volume_density DECIMAL(8,4) NOT NULL,
        is_support BOOLEAN DEFAULT FALSE,
        is_resistance BOOLEAN DEFAULT FALSE,
        node_strength DECIMAL(3,2) DEFAULT 0.0,
        created_at TIMESTAMPTZ DEFAULT NOW(),
        PRIMARY KEY (timestamp, id)
    );
    """
    
        # Create liquidity snapshots table
        liquidity_snapshots_table = """
    CREATE TABLE IF NOT EXISTS liquidity_snapshots (
        id SERIAL,
        symbol VARCHAR(20) NOT NULL,
        timestamp TIMESTAMPTZ NOT NULL,
        bid_depth_0_5 DECIMAL(15,3) NOT NULL,
        bid_depth_1_0 DECIMAL(15,3) NOT NULL,
        bid_depth_2_0 DECIMAL(15,3) NOT NULL,
        ask_depth_0_5 DECIMAL(15,3) NOT NULL,
        ask_depth_1_0 DECIMAL(15,3) NOT NULL,
        ask_depth_2_0 DECIMAL(15,3) NOT NULL,
        bid_ask_ratio DECIMAL(6,3) NOT NULL,
        liquidity_imbalance DECIMAL(8,4) NOT NULL,
        spread_bps DECIMAL(6,2) NOT NULL,
        liquidity_score DECIMAL(3,2) NOT NULL,
        spoofing_detected BOOLEAN DEFAULT FALSE,
        whale_activity BOOLEAN DEFAULT FALSE,
        liquidity_gaps JSONB,
        created_at TIMESTAMPTZ DEFAULT NOW(),
        PRIMARY KEY (timestamp, id)
    );
    """
    
        # Create volume nodes table for high-volume price levels
        volume_nodes_table = """
    CREATE TABLE IF NOT EXISTS volume_nodes (
        id SERIAL,
        symbol VARCHAR(20) NOT NULL,
        timeframe VARCHAR(10) NOT NULL,
        price_level DECIMAL(12,6) NOT NULL,
        node_type VARCHAR(20) NOT NULL, -- 'support', 'resistance', 'accumulation', 'distribution'
        volume_cluster_size DECIMAL(15,3) NOT NULL,
        node_strength DECIMAL(3,2) NOT NULL,
        first_detected TIMESTAMPTZ NOT NULL,
        last_updated TIMESTAMPTZ NOT NULL,
        touch_count INTEGER DEFAULT 1,
        is_active BOOLEAN DEFAULT TRUE,
        decay_factor DECIMAL(3,2) DEFAULT 1.0,
        node_metadata JSONB,
        created_at TIMESTAMPTZ DEFAULT NOW(),
        PRIMARY KEY (symbol, price_level, timeframe)
    );
    """
    
        # Execute table creation
        commands = [volume_analysis_table, volume_patterns_table, delta_profile_table, liquidity_snapshots_table, volume_nodes_table]
        
        for i, command in enumerate(commands, 1):
            try:
                logger.info(f"Creating volume analysis table {i}/{len(commands)}...")
                await conn.execute(command)
                logger.info(f"✅ Volume analysis table {i} created successfully")
            except Exception as e:
                logger.error(f"❌ Error creating table {i}: {e}")
                continue
    
        # Create TimescaleDB hypertables
        hypertable_commands = [
            "SELECT create_hypertable('volume_analysis', 'timestamp', if_not_exists => TRUE, chunk_time_interval => INTERVAL '1 hour');",
            "SELECT create_hypertable('volume_patterns', 'timestamp', if_not_exists => TRUE, chunk_time_interval => INTERVAL '1 hour');",
            "SELECT create_hypertable('delta_profile', 'timestamp', if_not_exists => TRUE, chunk_time_interval => INTERVAL '1 hour');",
            "SELECT create_hypertable('liquidity_snapshots', 'timestamp', if_not_exists => TRUE, chunk_time_interval => INTERVAL '30 minutes');",
            "SELECT create_hypertable('volume_nodes', 'created_at', if_not_exists => TRUE, chunk_time_interval => INTERVAL '1 day');"
        ]
        
        for i, command in enumerate(hypertable_commands, 1):
            try:
                logger.info(f"Creating volume analysis hypertable {i}/{len(hypertable_commands)}...")
                await conn.execute(command)
                logger.info(f"✅ Volume analysis hypertable {i} created successfully")
            except Exception as e:
                logger.warning(f"⚠️ Hypertable {i} creation warning: {e}")
                continue
    
        # Create indexes
        index_commands = [
            "CREATE INDEX IF NOT EXISTS idx_volume_analysis_symbol_timeframe ON volume_analysis (symbol, timeframe, timestamp DESC);",
            "CREATE INDEX IF NOT EXISTS idx_volume_analysis_pattern_type ON volume_analysis (volume_pattern_type, timestamp DESC);",
            "CREATE INDEX IF NOT EXISTS idx_volume_patterns_symbol_type ON volume_patterns (symbol, pattern_type, timestamp DESC);",
            "CREATE INDEX IF NOT EXISTS idx_delta_profile_symbol_price ON delta_profile (symbol, price_level, timestamp DESC);",
            "CREATE INDEX IF NOT EXISTS idx_delta_profile_support_resistance ON delta_profile (symbol, is_support, is_resistance, timestamp DESC);",
            "CREATE INDEX IF NOT EXISTS idx_liquidity_snapshots_symbol ON liquidity_snapshots (symbol, timestamp DESC);",
            "CREATE INDEX IF NOT EXISTS idx_liquidity_snapshots_spoofing ON liquidity_snapshots (symbol, spoofing_detected, timestamp DESC);",
            "CREATE INDEX IF NOT EXISTS idx_volume_nodes_symbol_type ON volume_nodes (symbol, node_type, is_active, price_level);",
            "CREATE INDEX IF NOT EXISTS idx_volume_nodes_strength ON volume_nodes (symbol, node_strength DESC, is_active);"
        ]
        
        for i, command in enumerate(index_commands, 1):
            try:
                logger.info(f"Creating volume analysis index {i}/{len(index_commands)}...")
                await conn.execute(command)
                logger.info(f"✅ Volume analysis index {i} created successfully")
            except Exception as e:
                logger.warning(f"⚠️ Index {i} creation warning: {e}")
                continue
    
            # Add new columns to existing table if they don't exist
        await add_advanced_metrics_columns(conn)
    
        # Create continuous aggregates for performance optimization
        await create_continuous_aggregates(conn)
        
        # Create materialized views for dashboard data
        await create_materialized_views(conn)
        
        logger.info("✅ Volume analysis tables migration completed!")
        return True

async def add_advanced_metrics_columns(conn):
    """Add advanced metrics columns to existing volume_analysis table"""
    try:
        logger.info("🔍 Adding advanced metrics columns to existing table...")
        
        # Columns to add
        new_columns = [
            "ALTER TABLE volume_analysis ADD COLUMN IF NOT EXISTS vwap DECIMAL(12,6);",
            "ALTER TABLE volume_analysis ADD COLUMN IF NOT EXISTS cumulative_volume_delta DECIMAL(15,3);",
            "ALTER TABLE volume_analysis ADD COLUMN IF NOT EXISTS relative_volume DECIMAL(6,3);",
            "ALTER TABLE volume_analysis ADD COLUMN IF NOT EXISTS volume_weighted_price DECIMAL(12,6);",
            "ALTER TABLE volume_analysis ADD COLUMN IF NOT EXISTS volume_flow_imbalance DECIMAL(8,4);"
        ]
        
        for i, command in enumerate(new_columns, 1):
            try:
                logger.info(f"Adding column {i}/{len(new_columns)}...")
                await conn.execute(command)
                logger.info(f"✅ Column {i} added successfully")
            except Exception as e:
                logger.warning(f"⚠️ Column {i} addition warning: {e}")
                continue
        
        logger.info("✅ Advanced metrics columns added successfully")
        
    except Exception as e:
        logger.error(f"❌ Error adding advanced metrics columns: {e}")
    
    return True

async def create_continuous_aggregates(conn):
    """Create continuous aggregates for performance optimization"""
    try:
        logger.info("🔍 Creating continuous aggregates...")
        
        # Continuous aggregate for 5-minute volume statistics
        continuous_aggregate_5m = """
        CREATE MATERIALIZED VIEW IF NOT EXISTS volume_stats_5m
        WITH (timescaledb.continuous) AS
        SELECT 
            time_bucket('5 minutes', timestamp) AS bucket,
            symbol,
            timeframe,
            AVG(volume_ratio) AS avg_volume_ratio,
            AVG(volume_positioning_score) AS avg_volume_positioning,
            AVG(order_book_imbalance) AS avg_order_book_imbalance,
            AVG(vwap) AS avg_vwap,
            AVG(cumulative_volume_delta) AS avg_cvd,
            AVG(relative_volume) AS avg_rvol,
            AVG(volume_flow_imbalance) AS avg_flow_imbalance,
            COUNT(*) AS data_points,
            COUNT(CASE WHEN volume_breakout THEN 1 END) AS breakout_count
        FROM volume_analysis
        GROUP BY bucket, symbol, timeframe
        WITH NO DATA;
        """
        
        # Continuous aggregate for delta profile statistics
        delta_profile_aggregate_5m = """
        CREATE MATERIALIZED VIEW IF NOT EXISTS delta_profile_stats_5m
        WITH (timescaledb.continuous) AS
        SELECT 
            time_bucket('5 minutes', timestamp) AS bucket,
            symbol,
            timeframe,
            price_level,
            SUM(volume_at_level) AS total_volume_at_level,
            SUM(buy_volume) AS total_buy_volume,
            SUM(sell_volume) AS total_sell_volume,
            SUM(delta_imbalance) AS total_delta_imbalance,
            AVG(volume_density) AS avg_volume_density,
            COUNT(CASE WHEN is_support THEN 1 END) AS support_count,
            COUNT(CASE WHEN is_resistance THEN 1 END) AS resistance_count,
            AVG(node_strength) AS avg_node_strength
        FROM delta_profile
        GROUP BY bucket, symbol, timeframe, price_level
        WITH NO DATA;
        """
        
        # Continuous aggregate for liquidity statistics
        liquidity_stats_5m = """
        CREATE MATERIALIZED VIEW IF NOT EXISTS liquidity_stats_5m
        WITH (timescaledb.continuous) AS
        SELECT 
            time_bucket('5 minutes', timestamp) AS bucket,
            symbol,
            AVG(bid_ask_ratio) AS avg_bid_ask_ratio,
            AVG(liquidity_imbalance) AS avg_liquidity_imbalance,
            AVG(spread_bps) AS avg_spread_bps,
            AVG(liquidity_score) AS avg_liquidity_score,
            COUNT(CASE WHEN spoofing_detected THEN 1 END) AS spoofing_count,
            COUNT(CASE WHEN whale_activity THEN 1 END) AS whale_activity_count
        FROM liquidity_snapshots
        GROUP BY bucket, symbol
        WITH NO DATA;
        """
        
        # Continuous aggregate for 1-hour volume statistics
        continuous_aggregate_1h = """
        CREATE MATERIALIZED VIEW IF NOT EXISTS volume_stats_1h
        WITH (timescaledb.continuous) AS
        SELECT 
            time_bucket('1 hour', timestamp) AS bucket,
            symbol,
            timeframe,
            AVG(volume_ratio) AS avg_volume_ratio,
            AVG(volume_positioning_score) AS avg_volume_positioning,
            AVG(order_book_imbalance) AS avg_order_book_imbalance,
            AVG(vwap) AS avg_vwap,
            AVG(cumulative_volume_delta) AS avg_cvd,
            AVG(relative_volume) AS avg_rvol,
            AVG(volume_flow_imbalance) AS avg_flow_imbalance,
            COUNT(*) AS data_points,
            COUNT(CASE WHEN volume_breakout THEN 1 END) AS breakout_count
        FROM volume_analysis
        GROUP BY bucket, symbol, timeframe
        WITH NO DATA;
        """
        
        # Continuous aggregate for daily volume statistics
        continuous_aggregate_1d = """
        CREATE MATERIALIZED VIEW IF NOT EXISTS volume_stats_1d
        WITH (timescaledb.continuous) AS
        SELECT 
            time_bucket('1 day', timestamp) AS bucket,
            symbol,
            timeframe,
            AVG(volume_ratio) AS avg_volume_ratio,
            AVG(volume_positioning_score) AS avg_volume_positioning,
            AVG(order_book_imbalance) AS avg_order_book_imbalance,
            AVG(vwap) AS avg_vwap,
            AVG(cumulative_volume_delta) AS avg_cvd,
            AVG(relative_volume) AS avg_rvol,
            AVG(volume_flow_imbalance) AS avg_flow_imbalance,
            COUNT(*) AS data_points,
            COUNT(CASE WHEN volume_breakout THEN 1 END) AS breakout_count
        FROM volume_analysis
        GROUP BY bucket, symbol, timeframe
        WITH NO DATA;
        """
        
        aggregates = [
            ("5-minute volume stats", continuous_aggregate_5m),
            ("1-hour volume stats", continuous_aggregate_1h),
            ("1-day volume stats", continuous_aggregate_1d),
            ("5-minute delta profile stats", delta_profile_aggregate_5m),
            ("5-minute liquidity stats", liquidity_stats_5m)
        ]
        
        for name, command in aggregates:
            try:
                logger.info(f"Creating {name}...")
                await conn.execute(command)
                logger.info(f"✅ {name} created successfully")
            except Exception as e:
                logger.warning(f"⚠️ {name} creation warning: {e}")
                continue
        
        logger.info("✅ Continuous aggregates created successfully")
        
    except Exception as e:
        logger.error(f"❌ Error creating continuous aggregates: {e}")

async def create_materialized_views(conn):
    """Create materialized views for dashboard data"""
    try:
        logger.info("🔍 Creating materialized views...")
        
        # Materialized view for recent volume analysis
        recent_volume_view = """
        CREATE MATERIALIZED VIEW IF NOT EXISTS recent_volume_analysis AS
        SELECT 
            symbol,
            timeframe,
            timestamp,
            volume_ratio,
            volume_trend,
            volume_positioning_score,
            order_book_imbalance,
            vwap,
            cumulative_volume_delta,
            relative_volume,
            volume_flow_imbalance,
            volume_breakout,
            volume_pattern_type,
            volume_pattern_strength,
            volume_pattern_confidence
        FROM volume_analysis
        WHERE timestamp >= NOW() - INTERVAL '24 hours'
        ORDER BY timestamp DESC;
        """
        
        # Materialized view for volume pattern summary
        pattern_summary_view = """
        CREATE MATERIALIZED VIEW IF NOT EXISTS volume_pattern_summary AS
        SELECT 
            symbol,
            timeframe,
            volume_pattern_type,
            volume_pattern_strength,
            COUNT(*) as pattern_count,
            AVG(volume_pattern_confidence) as avg_confidence,
            MAX(timestamp) as last_occurrence
        FROM volume_analysis
        WHERE volume_pattern_type IS NOT NULL
        AND timestamp >= NOW() - INTERVAL '7 days'
        GROUP BY symbol, timeframe, volume_pattern_type, volume_pattern_strength
        ORDER BY pattern_count DESC;
        """
        
        # Materialized view for volume breakout alerts
        breakout_alerts_view = """
        CREATE MATERIALIZED VIEW IF NOT EXISTS volume_breakout_alerts AS
        SELECT 
            symbol,
            timeframe,
            timestamp,
            volume_ratio,
            vwap,
            cumulative_volume_delta,
            volume_flow_imbalance,
            volume_pattern_type,
            volume_pattern_strength
        FROM volume_analysis
        WHERE volume_breakout = TRUE
        AND timestamp >= NOW() - INTERVAL '1 hour'
        ORDER BY timestamp DESC;
        """
        
        # Materialized view for active volume nodes
        active_nodes_view = """
        CREATE MATERIALIZED VIEW IF NOT EXISTS active_volume_nodes AS
        SELECT 
            symbol,
            timeframe,
            price_level,
            node_type,
            volume_cluster_size,
            node_strength,
            touch_count,
            last_updated
        FROM volume_nodes
        WHERE is_active = TRUE
        AND node_strength >= 0.7
        ORDER BY node_strength DESC, last_updated DESC;
        """
        
        # Materialized view for liquidity alerts
        liquidity_alerts_view = """
        CREATE MATERIALIZED VIEW IF NOT EXISTS liquidity_alerts AS
        SELECT 
            symbol,
            timestamp,
            bid_ask_ratio,
            liquidity_imbalance,
            spread_bps,
            liquidity_score,
            spoofing_detected,
            whale_activity
        FROM liquidity_snapshots
        WHERE (spoofing_detected = TRUE OR whale_activity = TRUE OR liquidity_score < 0.3)
        AND timestamp >= NOW() - INTERVAL '1 hour'
        ORDER BY timestamp DESC;
        """
        
        views = [
            ("Recent volume analysis", recent_volume_view),
            ("Volume pattern summary", pattern_summary_view),
            ("Volume breakout alerts", breakout_alerts_view),
            ("Active volume nodes", active_nodes_view),
            ("Liquidity alerts", liquidity_alerts_view)
        ]
        
        for name, command in views:
            try:
                logger.info(f"Creating {name} view...")
                await conn.execute(command)
                logger.info(f"✅ {name} view created successfully")
            except Exception as e:
                logger.warning(f"⚠️ {name} view creation warning: {e}")
                continue
        
        logger.info("✅ Materialized views created successfully")
        
    except Exception as e:
        logger.error(f"❌ Error creating materialized views: {e}")

async def main():
    """Main migration function"""
    logger.info("🚀 Starting Volume Analysis Tables Migration...")
    
    try:
        success = await create_volume_analysis_tables()
        if success:
            logger.info("✅ Volume Analysis Tables Migration completed successfully!")
        else:
            logger.error("❌ Migration failed")
            
    except Exception as e:
        logger.error(f"❌ Migration failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())