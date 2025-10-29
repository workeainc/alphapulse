-- Enhanced Order Book Integration Migration
-- Creates tables for volume profile analysis and order book integration

-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Volume Profile Analysis Table (TimescaleDB hypertable)
CREATE TABLE IF NOT EXISTS volume_profile_analysis (
    id BIGSERIAL,
    symbol VARCHAR(20) NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    
    -- Volume Profile Data
    poc_price NUMERIC(20,8) NOT NULL,  -- Point of Control price
    poc_volume NUMERIC(20,8) NOT NULL,  -- Point of Control volume
    value_area_high NUMERIC(20,8) NOT NULL,  -- Value Area High
    value_area_low NUMERIC(20,8) NOT NULL,   -- Value Area Low
    value_area_volume NUMERIC(20,8) NOT NULL, -- Value Area volume
    total_volume NUMERIC(20,8) NOT NULL,      -- Total volume analyzed
    
    -- Analysis Quality
    analysis_confidence NUMERIC(4,3) NOT NULL DEFAULT 0.0,
    
    -- Algorithm Inputs (JSONB for flexibility)
    algorithm_inputs JSONB,
    
    -- Metadata
    created_at TIMESTAMPTZ DEFAULT NOW() NOT NULL,
    
    -- Constraints
    CONSTRAINT valid_confidence CHECK (analysis_confidence >= 0 AND analysis_confidence <= 1),
    CONSTRAINT valid_poc_price CHECK (poc_price > 0),
    CONSTRAINT valid_poc_volume CHECK (poc_volume > 0),
    CONSTRAINT valid_value_area CHECK (value_area_high >= value_area_low),
    CONSTRAINT valid_total_volume CHECK (total_volume > 0),
    PRIMARY KEY (id, timestamp)
);

-- Create TimescaleDB hypertable with 1-hour chunks
SELECT create_hypertable('volume_profile_analysis', 'timestamp', chunk_time_interval => INTERVAL '1 hour');

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_volume_profile_symbol_timeframe ON volume_profile_analysis (symbol, timeframe);
CREATE INDEX IF NOT EXISTS idx_volume_profile_timestamp ON volume_profile_analysis (timestamp);
CREATE INDEX IF NOT EXISTS idx_volume_profile_symbol_timestamp ON volume_profile_analysis (symbol, timestamp);
CREATE INDEX IF NOT EXISTS idx_volume_profile_poc_price ON volume_profile_analysis (poc_price);
CREATE INDEX IF NOT EXISTS idx_volume_profile_confidence ON volume_profile_analysis (analysis_confidence);

-- Order Book Levels Table (TimescaleDB hypertable)
CREATE TABLE IF NOT EXISTS order_book_levels (
    id BIGSERIAL,
    symbol VARCHAR(20) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    
    -- Level Information
    level_type VARCHAR(30) NOT NULL,  -- 'high_volume_node', 'low_volume_node', 'liquidity_wall', etc.
    price_level NUMERIC(20,8) NOT NULL,
    volume_at_level NUMERIC(20,8) NOT NULL,
    volume_percentage NUMERIC(6,3) NOT NULL,
    
    -- Volume Breakdown
    bid_volume NUMERIC(20,8) NOT NULL DEFAULT 0,
    ask_volume NUMERIC(20,8) NOT NULL DEFAULT 0,
    volume_imbalance NUMERIC(8,4) NOT NULL DEFAULT 0,
    
    -- Level Strength and Confidence
    level_strength NUMERIC(4,3) NOT NULL DEFAULT 0.0,
    confidence NUMERIC(4,3) NOT NULL DEFAULT 0.0,
    
    -- Activity Tracking
    is_active BOOLEAN DEFAULT TRUE,
    touch_count INTEGER DEFAULT 0,
    last_touch_time TIMESTAMPTZ,
    
    -- Metadata
    created_at TIMESTAMPTZ DEFAULT NOW() NOT NULL,
    
    -- Constraints
    CONSTRAINT valid_level_strength CHECK (level_strength >= 0 AND level_strength <= 1),
    CONSTRAINT valid_confidence CHECK (confidence >= 0 AND confidence <= 1),
    CONSTRAINT valid_volume_percentage CHECK (volume_percentage >= 0 AND volume_percentage <= 100),
    CONSTRAINT valid_price_level CHECK (price_level > 0),
    CONSTRAINT valid_volume_at_level CHECK (volume_at_level >= 0),
    PRIMARY KEY (id, timestamp)
);

-- Create TimescaleDB hypertable with 1-hour chunks
SELECT create_hypertable('order_book_levels', 'timestamp', chunk_time_interval => INTERVAL '1 hour');

-- Create indexes for order book levels
CREATE INDEX IF NOT EXISTS idx_order_book_levels_symbol ON order_book_levels (symbol);
CREATE INDEX IF NOT EXISTS idx_order_book_levels_timestamp ON order_book_levels (timestamp);
CREATE INDEX IF NOT EXISTS idx_order_book_levels_type ON order_book_levels (level_type);
CREATE INDEX IF NOT EXISTS idx_order_book_levels_price ON order_book_levels (price_level);
CREATE INDEX IF NOT EXISTS idx_order_book_levels_strength ON order_book_levels (level_strength);
CREATE INDEX IF NOT EXISTS idx_order_book_levels_active ON order_book_levels (is_active);

-- Market Microstructure Analysis Table (TimescaleDB hypertable)
CREATE TABLE IF NOT EXISTS market_microstructure (
    id BIGSERIAL,
    symbol VARCHAR(20) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    
    -- Order Book Metrics
    bid_ask_imbalance NUMERIC(8,4) NOT NULL DEFAULT 0,
    depth_pressure NUMERIC(8,4) NOT NULL DEFAULT 0,
    liquidity_score NUMERIC(4,3) NOT NULL DEFAULT 0.0,
    
    -- Volume Metrics
    total_bid_volume NUMERIC(20,8) NOT NULL DEFAULT 0,
    total_ask_volume NUMERIC(20,8) NOT NULL DEFAULT 0,
    total_volume NUMERIC(20,8) NOT NULL DEFAULT 0,
    
    -- Price Metrics
    spread NUMERIC(20,8) NOT NULL DEFAULT 0,
    spread_percentage NUMERIC(8,4) NOT NULL DEFAULT 0,
    mid_price NUMERIC(20,8) NOT NULL DEFAULT 0,
    best_bid NUMERIC(20,8) NOT NULL DEFAULT 0,
    best_ask NUMERIC(20,8) NOT NULL DEFAULT 0,
    
    -- Analysis Quality
    analysis_confidence NUMERIC(4,3) NOT NULL DEFAULT 0.0,
    
    -- Raw Data (JSONB for flexibility)
    raw_order_book_data JSONB,
    
    -- Metadata
    created_at TIMESTAMPTZ DEFAULT NOW() NOT NULL,
    
    -- Constraints
    CONSTRAINT valid_liquidity_score CHECK (liquidity_score >= 0 AND liquidity_score <= 1),
    CONSTRAINT valid_analysis_confidence CHECK (analysis_confidence >= 0 AND analysis_confidence <= 1),
    CONSTRAINT valid_spread_percentage CHECK (spread_percentage >= 0),
    CONSTRAINT valid_mid_price CHECK (mid_price > 0),
    CONSTRAINT valid_best_bid CHECK (best_bid > 0),
    CONSTRAINT valid_best_ask CHECK (best_ask > 0),
    PRIMARY KEY (id, timestamp)
);

-- Create TimescaleDB hypertable with 1-hour chunks
SELECT create_hypertable('market_microstructure', 'timestamp', chunk_time_interval => INTERVAL '1 hour');

-- Create indexes for market microstructure
CREATE INDEX IF NOT EXISTS idx_market_microstructure_symbol ON market_microstructure (symbol);
CREATE INDEX IF NOT EXISTS idx_market_microstructure_timestamp ON market_microstructure (timestamp);
CREATE INDEX IF NOT EXISTS idx_market_microstructure_imbalance ON market_microstructure (bid_ask_imbalance);
CREATE INDEX IF NOT EXISTS idx_market_microstructure_liquidity ON market_microstructure (liquidity_score);

-- Continuous Aggregates for Performance Optimization

-- Hourly volume profile summary
CREATE MATERIALIZED VIEW IF NOT EXISTS volume_profile_hourly AS
SELECT 
    time_bucket('1 hour', timestamp) AS bucket,
    symbol,
    timeframe,
    AVG(poc_price) AS avg_poc_price,
    AVG(poc_volume) AS avg_poc_volume,
    AVG(value_area_high) AS avg_value_area_high,
    AVG(value_area_low) AS avg_value_area_low,
    AVG(total_volume) AS avg_total_volume,
    AVG(analysis_confidence) AS avg_analysis_confidence,
    COUNT(*) AS data_points
FROM volume_profile_analysis
GROUP BY bucket, symbol, timeframe;

-- Hourly order book levels summary
CREATE MATERIALIZED VIEW IF NOT EXISTS order_book_levels_hourly AS
SELECT 
    time_bucket('1 hour', timestamp) AS bucket,
    symbol,
    level_type,
    COUNT(*) AS level_count,
    AVG(level_strength) AS avg_level_strength,
    AVG(confidence) AS avg_confidence,
    AVG(volume_at_level) AS avg_volume_at_level,
    AVG(volume_imbalance) AS avg_volume_imbalance
FROM order_book_levels
GROUP BY bucket, symbol, level_type;

-- Hourly market microstructure summary
CREATE MATERIALIZED VIEW IF NOT EXISTS market_microstructure_hourly AS
SELECT 
    time_bucket('1 hour', timestamp) AS bucket,
    symbol,
    AVG(bid_ask_imbalance) AS avg_bid_ask_imbalance,
    AVG(depth_pressure) AS avg_depth_pressure,
    AVG(liquidity_score) AS avg_liquidity_score,
    AVG(total_volume) AS avg_total_volume,
    AVG(spread_percentage) AS avg_spread_percentage,
    AVG(analysis_confidence) AS avg_analysis_confidence,
    COUNT(*) AS data_points
FROM market_microstructure
GROUP BY bucket, symbol;

-- Add compression policies for old data
ALTER TABLE volume_profile_analysis SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'symbol'
);
ALTER TABLE order_book_levels SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'symbol'
);
ALTER TABLE market_microstructure SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'symbol'
);

-- Add compression policies (compress data older than 1 month)
SELECT add_compression_policy('volume_profile_analysis', INTERVAL '1 month');
SELECT add_compression_policy('order_book_levels', INTERVAL '1 month');
SELECT add_compression_policy('market_microstructure', INTERVAL '1 month');

-- Create refresh policies for continuous aggregates
SELECT add_continuous_aggregate_policy('volume_profile_hourly',
    start_offset => INTERVAL '1 hour',
    end_offset => INTERVAL '5 minutes',
    schedule_interval => INTERVAL '5 minutes');

SELECT add_continuous_aggregate_policy('order_book_levels_hourly',
    start_offset => INTERVAL '1 hour',
    end_offset => INTERVAL '5 minutes',
    schedule_interval => INTERVAL '5 minutes');

SELECT add_continuous_aggregate_policy('market_microstructure_hourly',
    start_offset => INTERVAL '1 hour',
    end_offset => INTERVAL '5 minutes',
    schedule_interval => INTERVAL '5 minutes');

-- Grant permissions to alpha_emon user
GRANT ALL PRIVILEGES ON TABLE volume_profile_analysis TO alpha_emon;
GRANT ALL PRIVILEGES ON TABLE order_book_levels TO alpha_emon;
GRANT ALL PRIVILEGES ON TABLE market_microstructure TO alpha_emon;
GRANT ALL PRIVILEGES ON TABLE volume_profile_hourly TO alpha_emon;
GRANT ALL PRIVILEGES ON TABLE order_book_levels_hourly TO alpha_emon;
GRANT ALL PRIVILEGES ON TABLE market_microstructure_hourly TO alpha_emon;

-- Grant sequence privileges
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO alpha_emon;

-- Create function to get latest volume profile for algorithm input
CREATE OR REPLACE FUNCTION get_latest_volume_profile(
    p_symbol VARCHAR(20),
    p_timeframe VARCHAR(10)
)
RETURNS TABLE (
    poc_price NUMERIC(20,8),
    poc_volume NUMERIC(20,8),
    value_area_high NUMERIC(20,8),
    value_area_low NUMERIC(20,8),
    total_volume NUMERIC(20,8),
    analysis_confidence NUMERIC(4,3),
    algorithm_inputs JSONB
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        vpa.poc_price,
        vpa.poc_volume,
        vpa.value_area_high,
        vpa.value_area_low,
        vpa.total_volume,
        vpa.analysis_confidence,
        vpa.algorithm_inputs
    FROM volume_profile_analysis vpa
    WHERE vpa.symbol = p_symbol 
      AND vpa.timeframe = p_timeframe
    ORDER BY vpa.timestamp DESC
    LIMIT 1;
END;
$$ LANGUAGE plpgsql;

-- Grant execute permission on the function
GRANT EXECUTE ON FUNCTION get_latest_volume_profile TO alpha_emon;

-- Create function to get active order book levels
CREATE OR REPLACE FUNCTION get_active_order_book_levels(
    p_symbol VARCHAR(20),
    p_level_type VARCHAR(30) DEFAULT NULL
)
RETURNS TABLE (
    level_type VARCHAR(30),
    price_level NUMERIC(20,8),
    volume_at_level NUMERIC(20,8),
    level_strength NUMERIC(4,3),
    confidence NUMERIC(4,3),
    volume_imbalance NUMERIC(8,4),
    last_touch_time TIMESTAMPTZ
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        obl.level_type,
        obl.price_level,
        obl.volume_at_level,
        obl.level_strength,
        obl.confidence,
        obl.volume_imbalance,
        obl.last_touch_time
    FROM order_book_levels obl
    WHERE obl.symbol = p_symbol 
      AND obl.is_active = TRUE
      AND (p_level_type IS NULL OR obl.level_type = p_level_type)
    ORDER BY obl.level_strength DESC, obl.timestamp DESC;
END;
$$ LANGUAGE plpgsql;

-- Grant execute permission on the function
GRANT EXECUTE ON FUNCTION get_active_order_book_levels TO alpha_emon;

-- Create view for order book integration dashboard
CREATE OR REPLACE VIEW order_book_integration_dashboard AS
SELECT 
    vpa.symbol,
    vpa.timeframe,
    vpa.timestamp,
    vpa.poc_price,
    vpa.poc_volume,
    vpa.value_area_high,
    vpa.value_area_low,
    vpa.analysis_confidence,
    COUNT(obl.id) as active_levels_count,
    COUNT(CASE WHEN obl.level_type = 'high_volume_node' THEN 1 END) as hvn_count,
    COUNT(CASE WHEN obl.level_type = 'low_volume_node' THEN 1 END) as lvn_count,
    COUNT(CASE WHEN obl.level_type = 'liquidity_wall' THEN 1 END) as liquidity_walls_count,
    AVG(obl.level_strength) as avg_level_strength,
    AVG(obl.confidence) as avg_level_confidence
FROM volume_profile_analysis vpa
LEFT JOIN order_book_levels obl ON (
    vpa.symbol = obl.symbol 
    AND obl.timestamp >= vpa.timestamp - INTERVAL '1 hour'
    AND obl.is_active = TRUE
)
WHERE vpa.timestamp >= NOW() - INTERVAL '1 hour'
GROUP BY vpa.symbol, vpa.timeframe, vpa.timestamp, vpa.poc_price, vpa.poc_volume, 
         vpa.value_area_high, vpa.value_area_low, vpa.analysis_confidence
ORDER BY vpa.timestamp DESC;

-- Grant select permission on the view
GRANT SELECT ON order_book_integration_dashboard TO alpha_emon;

-- Final success message
DO $$
BEGIN
    RAISE NOTICE '✅ Enhanced Order Book Integration Schema created successfully!';
    RAISE NOTICE '📊 Tables created: volume_profile_analysis, order_book_levels, market_microstructure';
    RAISE NOTICE '📈 Views created: volume_profile_hourly, order_book_levels_hourly, market_microstructure_hourly, order_book_integration_dashboard';
    RAISE NOTICE '🔧 Functions created: get_latest_volume_profile, get_active_order_book_levels';
    RAISE NOTICE '👤 Permissions granted to alpha_emon user';
    RAISE NOTICE '🚀 Ready for enhanced order book integration!';
END $$;
