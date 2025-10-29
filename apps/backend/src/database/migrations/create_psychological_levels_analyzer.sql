-- Standalone Psychological Levels Analyzer Migration
-- Creates tables for psychological levels analysis

-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Psychological Levels Analysis Table (TimescaleDB hypertable)
CREATE TABLE IF NOT EXISTS psychological_levels_analysis (
    id BIGSERIAL,
    symbol VARCHAR(20) NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    
    -- Current Market State
    current_price NUMERIC(20,8) NOT NULL,
    
    -- Nearest Levels
    nearest_support_price NUMERIC(20,8),
    nearest_resistance_price NUMERIC(20,8),
    
    -- Market Analysis
    market_regime VARCHAR(20) NOT NULL DEFAULT 'unknown',
    analysis_confidence NUMERIC(4,3) NOT NULL DEFAULT 0.0,
    
    -- Algorithm Inputs (JSONB for flexibility)
    algorithm_inputs JSONB,
    
    -- Metadata
    created_at TIMESTAMPTZ DEFAULT NOW() NOT NULL,
    
    -- Constraints
    CONSTRAINT valid_current_price CHECK (current_price > 0),
    CONSTRAINT valid_support_price CHECK (nearest_support_price IS NULL OR nearest_support_price > 0),
    CONSTRAINT valid_resistance_price CHECK (nearest_resistance_price IS NULL OR nearest_resistance_price > 0),
    CONSTRAINT valid_analysis_confidence CHECK (analysis_confidence >= 0 AND analysis_confidence <= 1),
    CONSTRAINT valid_market_regime CHECK (market_regime IN ('trending_up', 'trending_down', 'ranging', 'volatile', 'unknown')),
    PRIMARY KEY (id, timestamp)
);

-- Create TimescaleDB hypertable with 1-hour chunks
SELECT create_hypertable('psychological_levels_analysis', 'timestamp', chunk_time_interval => INTERVAL '1 hour');

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_psychological_analysis_symbol_timeframe ON psychological_levels_analysis (symbol, timeframe);
CREATE INDEX IF NOT EXISTS idx_psychological_analysis_timestamp ON psychological_levels_analysis (timestamp);
CREATE INDEX IF NOT EXISTS idx_psychological_analysis_symbol_timestamp ON psychological_levels_analysis (symbol, timestamp);
CREATE INDEX IF NOT EXISTS idx_psychological_analysis_market_regime ON psychological_levels_analysis (market_regime);
CREATE INDEX IF NOT EXISTS idx_psychological_analysis_confidence ON psychological_levels_analysis (analysis_confidence);

-- Individual Psychological Levels Table (TimescaleDB hypertable)
CREATE TABLE IF NOT EXISTS psychological_levels (
    id BIGSERIAL,
    symbol VARCHAR(20) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    
    -- Level Information
    level_type VARCHAR(30) NOT NULL,  -- 'round_number', 'fibonacci_retracement', 'fibonacci_extension', 'price_milestone', 'percentage_level'
    price_level NUMERIC(20,8) NOT NULL,
    
    -- Level Strength and Confidence
    strength NUMERIC(4,3) NOT NULL DEFAULT 0.0,
    confidence NUMERIC(4,3) NOT NULL DEFAULT 0.0,
    
    -- Activity Tracking
    touch_count INTEGER DEFAULT 0,
    market_context VARCHAR(20),  -- 'support', 'resistance', 'current'
    first_touch_time TIMESTAMPTZ,
    last_touch_time TIMESTAMPTZ,
    
    -- Level Status
    is_active BOOLEAN DEFAULT TRUE,
    is_broken BOOLEAN DEFAULT FALSE,
    break_time TIMESTAMPTZ,
    rejection_count INTEGER DEFAULT 0,
    penetration_count INTEGER DEFAULT 0,
    
    -- Volume Analysis
    volume_at_level NUMERIC(20,8) DEFAULT 0.0,
    
    -- Metadata
    created_at TIMESTAMPTZ DEFAULT NOW() NOT NULL,
    
    -- Constraints
    CONSTRAINT valid_price_level CHECK (price_level > 0),
    CONSTRAINT valid_strength CHECK (strength >= 0 AND strength <= 1),
    CONSTRAINT valid_confidence CHECK (confidence >= 0 AND confidence <= 1),
    CONSTRAINT valid_touch_count CHECK (touch_count >= 0),
    CONSTRAINT valid_rejection_count CHECK (rejection_count >= 0),
    CONSTRAINT valid_penetration_count CHECK (penetration_count >= 0),
    CONSTRAINT valid_volume_at_level CHECK (volume_at_level >= 0),
    CONSTRAINT valid_market_context CHECK (market_context IN ('support', 'resistance', 'current', 'unknown')),
    CONSTRAINT valid_level_type CHECK (level_type IN ('round_number', 'fibonacci_retracement', 'fibonacci_extension', 'golden_ratio', 'major_support_resistance', 'price_milestone', 'percentage_level')),
    PRIMARY KEY (id, timestamp)
);

-- Create TimescaleDB hypertable with 1-hour chunks
SELECT create_hypertable('psychological_levels', 'timestamp', chunk_time_interval => INTERVAL '1 hour');

-- Create indexes for psychological levels
CREATE INDEX IF NOT EXISTS idx_psychological_levels_symbol ON psychological_levels (symbol);
CREATE INDEX IF NOT EXISTS idx_psychological_levels_timestamp ON psychological_levels (timestamp);
CREATE INDEX IF NOT EXISTS idx_psychological_levels_type ON psychological_levels (level_type);
CREATE INDEX IF NOT EXISTS idx_psychological_levels_price ON psychological_levels (price_level);
CREATE INDEX IF NOT EXISTS idx_psychological_levels_strength ON psychological_levels (strength);
CREATE INDEX IF NOT EXISTS idx_psychological_levels_confidence ON psychological_levels (confidence);
CREATE INDEX IF NOT EXISTS idx_psychological_levels_active ON psychological_levels (is_active);
CREATE INDEX IF NOT EXISTS idx_psychological_levels_context ON psychological_levels (market_context);

-- Level Interactions Table (TimescaleDB hypertable)
CREATE TABLE IF NOT EXISTS psychological_level_interactions (
    id BIGSERIAL,
    symbol VARCHAR(20) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    
    -- Interaction Details
    level_price NUMERIC(20,8) NOT NULL,
    level_type VARCHAR(30) NOT NULL,
    interaction_price NUMERIC(20,8) NOT NULL,
    interaction_volume NUMERIC(20,8) NOT NULL,
    reaction_type VARCHAR(20) NOT NULL,  -- 'rejection', 'penetration', 'breakout'
    
    -- Interaction Analysis
    reaction_strength NUMERIC(4,3) NOT NULL DEFAULT 0.0,
    volume_confirmation BOOLEAN DEFAULT FALSE,
    follow_through BOOLEAN DEFAULT FALSE,
    
    -- Metadata
    created_at TIMESTAMPTZ DEFAULT NOW() NOT NULL,
    
    -- Constraints
    CONSTRAINT valid_level_price CHECK (level_price > 0),
    CONSTRAINT valid_interaction_price CHECK (interaction_price > 0),
    CONSTRAINT valid_interaction_volume CHECK (interaction_volume >= 0),
    CONSTRAINT valid_reaction_strength CHECK (reaction_strength >= 0 AND reaction_strength <= 1),
    CONSTRAINT valid_reaction_type CHECK (reaction_type IN ('rejection', 'penetration', 'breakout', 'test')),
    PRIMARY KEY (id, timestamp)
);

-- Create TimescaleDB hypertable with 1-hour chunks
SELECT create_hypertable('psychological_level_interactions', 'timestamp', chunk_time_interval => INTERVAL '1 hour');

-- Create indexes for level interactions
CREATE INDEX IF NOT EXISTS idx_level_interactions_symbol ON psychological_level_interactions (symbol);
CREATE INDEX IF NOT EXISTS idx_level_interactions_timestamp ON psychological_level_interactions (timestamp);
CREATE INDEX IF NOT EXISTS idx_level_interactions_level_price ON psychological_level_interactions (level_price);
CREATE INDEX IF NOT EXISTS idx_level_interactions_reaction_type ON psychological_level_interactions (reaction_type);

-- Continuous Aggregates for Performance Optimization

-- Hourly psychological analysis summary
CREATE MATERIALIZED VIEW IF NOT EXISTS psychological_analysis_hourly AS
SELECT 
    time_bucket('1 hour', timestamp) AS bucket,
    symbol,
    timeframe,
    AVG(current_price) AS avg_current_price,
    AVG(analysis_confidence) AS avg_analysis_confidence,
    mode() WITHIN GROUP (ORDER BY market_regime) AS dominant_market_regime,
    COUNT(*) AS data_points
FROM psychological_levels_analysis
GROUP BY bucket, symbol, timeframe;

-- Hourly psychological levels summary
CREATE MATERIALIZED VIEW IF NOT EXISTS psychological_levels_hourly AS
SELECT 
    time_bucket('1 hour', timestamp) AS bucket,
    symbol,
    level_type,
    COUNT(*) AS level_count,
    AVG(strength) AS avg_strength,
    AVG(confidence) AS avg_confidence,
    AVG(touch_count) AS avg_touch_count,
    COUNT(CASE WHEN is_active THEN 1 END) AS active_levels_count
FROM psychological_levels
GROUP BY bucket, symbol, level_type;

-- Hourly level interactions summary
CREATE MATERIALIZED VIEW IF NOT EXISTS psychological_interactions_hourly AS
SELECT 
    time_bucket('1 hour', timestamp) AS bucket,
    symbol,
    reaction_type,
    COUNT(*) AS interaction_count,
    AVG(reaction_strength) AS avg_reaction_strength,
    COUNT(CASE WHEN volume_confirmation THEN 1 END) AS volume_confirmed_count,
    COUNT(CASE WHEN follow_through THEN 1 END) AS follow_through_count
FROM psychological_level_interactions
GROUP BY bucket, symbol, reaction_type;

-- Add compression policies for old data
ALTER TABLE psychological_levels_analysis SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'symbol'
);
ALTER TABLE psychological_levels SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'symbol'
);
ALTER TABLE psychological_level_interactions SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'symbol'
);

-- Add compression policies (compress data older than 1 month)
SELECT add_compression_policy('psychological_levels_analysis', INTERVAL '1 month');
SELECT add_compression_policy('psychological_levels', INTERVAL '1 month');
SELECT add_compression_policy('psychological_level_interactions', INTERVAL '1 month');

-- Create refresh policies for continuous aggregates
SELECT add_continuous_aggregate_policy('psychological_analysis_hourly',
    start_offset => INTERVAL '1 hour',
    end_offset => INTERVAL '5 minutes',
    schedule_interval => INTERVAL '5 minutes');

SELECT add_continuous_aggregate_policy('psychological_levels_hourly',
    start_offset => INTERVAL '1 hour',
    end_offset => INTERVAL '5 minutes',
    schedule_interval => INTERVAL '5 minutes');

SELECT add_continuous_aggregate_policy('psychological_interactions_hourly',
    start_offset => INTERVAL '1 hour',
    end_offset => INTERVAL '5 minutes',
    schedule_interval => INTERVAL '5 minutes');

-- Grant permissions to alpha_emon user
GRANT ALL PRIVILEGES ON TABLE psychological_levels_analysis TO alpha_emon;
GRANT ALL PRIVILEGES ON TABLE psychological_levels TO alpha_emon;
GRANT ALL PRIVILEGES ON TABLE psychological_level_interactions TO alpha_emon;
GRANT ALL PRIVILEGES ON TABLE psychological_analysis_hourly TO alpha_emon;
GRANT ALL PRIVILEGES ON TABLE psychological_levels_hourly TO alpha_emon;
GRANT ALL PRIVILEGES ON TABLE psychological_interactions_hourly TO alpha_emon;

-- Grant sequence privileges
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO alpha_emon;

-- Create function to get latest psychological analysis
CREATE OR REPLACE FUNCTION get_latest_psychological_analysis(
    p_symbol VARCHAR(20),
    p_timeframe VARCHAR(10)
)
RETURNS TABLE (
    current_price NUMERIC(20,8),
    nearest_support_price NUMERIC(20,8),
    nearest_resistance_price NUMERIC(20,8),
    market_regime VARCHAR(20),
    analysis_confidence NUMERIC(4,3),
    algorithm_inputs JSONB
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        pla.current_price,
        pla.nearest_support_price,
        pla.nearest_resistance_price,
        pla.market_regime,
        pla.analysis_confidence,
        pla.algorithm_inputs
    FROM psychological_levels_analysis pla
    WHERE pla.symbol = p_symbol 
      AND pla.timeframe = p_timeframe
    ORDER BY pla.timestamp DESC
    LIMIT 1;
END;
$$ LANGUAGE plpgsql;

-- Grant execute permission on the function
GRANT EXECUTE ON FUNCTION get_latest_psychological_analysis TO alpha_emon;

-- Create function to get active psychological levels
CREATE OR REPLACE FUNCTION get_active_psychological_levels(
    p_symbol VARCHAR(20),
    p_level_type VARCHAR(30) DEFAULT NULL,
    p_market_context VARCHAR(20) DEFAULT NULL
)
RETURNS TABLE (
    level_type VARCHAR(30),
    price_level NUMERIC(20,8),
    strength NUMERIC(4,3),
    confidence NUMERIC(4,3),
    touch_count INTEGER,
    market_context VARCHAR(20),
    last_touch_time TIMESTAMPTZ
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        pl.level_type,
        pl.price_level,
        pl.strength,
        pl.confidence,
        pl.touch_count,
        pl.market_context,
        pl.last_touch_time
    FROM psychological_levels pl
    WHERE pl.symbol = p_symbol 
      AND pl.is_active = TRUE
      AND (p_level_type IS NULL OR pl.level_type = p_level_type)
      AND (p_market_context IS NULL OR pl.market_context = p_market_context)
    ORDER BY pl.strength DESC, pl.timestamp DESC;
END;
$$ LANGUAGE plpgsql;

-- Grant execute permission on the function
GRANT EXECUTE ON FUNCTION get_active_psychological_levels TO alpha_emon;

-- Create function to get recent level interactions
CREATE OR REPLACE FUNCTION get_recent_level_interactions(
    p_symbol VARCHAR(20),
    p_hours INTEGER DEFAULT 24
)
RETURNS TABLE (
    level_price NUMERIC(20,8),
    level_type VARCHAR(30),
    interaction_price NUMERIC(20,8),
    reaction_type VARCHAR(20),
    reaction_strength NUMERIC(4,3),
    volume_confirmation BOOLEAN,
    timestamp TIMESTAMPTZ
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        pli.level_price,
        pli.level_type,
        pli.interaction_price,
        pli.reaction_type,
        pli.reaction_strength,
        pli.volume_confirmation,
        pli.timestamp
    FROM psychological_level_interactions pli
    WHERE pli.symbol = p_symbol 
      AND pli.timestamp >= NOW() - INTERVAL '1 hour' * p_hours
    ORDER BY pli.timestamp DESC;
END;
$$ LANGUAGE plpgsql;

-- Grant execute permission on the function
GRANT EXECUTE ON FUNCTION get_recent_level_interactions TO alpha_emon;

-- Create view for psychological levels dashboard
CREATE OR REPLACE VIEW psychological_levels_dashboard AS
SELECT 
    pla.symbol,
    pla.timeframe,
    pla.timestamp,
    pla.current_price,
    pla.market_regime,
    pla.analysis_confidence,
    pla.nearest_support_price,
    pla.nearest_resistance_price,
    COUNT(pl.id) as total_levels_count,
    COUNT(CASE WHEN pl.level_type = 'round_number' THEN 1 END) as round_number_count,
    COUNT(CASE WHEN pl.level_type = 'fibonacci_retracement' THEN 1 END) as fibonacci_retracement_count,
    COUNT(CASE WHEN pl.level_type = 'fibonacci_extension' THEN 1 END) as fibonacci_extension_count,
    COUNT(CASE WHEN pl.level_type = 'price_milestone' THEN 1 END) as milestone_count,
    COUNT(CASE WHEN pl.level_type = 'percentage_level' THEN 1 END) as percentage_level_count,
    COUNT(CASE WHEN pl.is_active THEN 1 END) as active_levels_count,
    AVG(pl.strength) as avg_level_strength,
    AVG(pl.confidence) as avg_level_confidence,
    COUNT(pli.id) as recent_interactions_count
FROM psychological_levels_analysis pla
LEFT JOIN psychological_levels pl ON (
    pla.symbol = pl.symbol 
    AND pl.timestamp >= pla.timestamp - INTERVAL '1 hour'
)
LEFT JOIN psychological_level_interactions pli ON (
    pla.symbol = pli.symbol 
    AND pli.timestamp >= pla.timestamp - INTERVAL '1 hour'
)
WHERE pla.timestamp >= NOW() - INTERVAL '1 hour'
GROUP BY pla.symbol, pla.timeframe, pla.timestamp, pla.current_price, pla.market_regime, 
         pla.analysis_confidence, pla.nearest_support_price, pla.nearest_resistance_price
ORDER BY pla.timestamp DESC;

-- Grant select permission on the view
GRANT SELECT ON psychological_levels_dashboard TO alpha_emon;

-- Final success message
DO $$
BEGIN
    RAISE NOTICE '✅ Standalone Psychological Levels Analyzer Schema created successfully!';
    RAISE NOTICE '📊 Tables created: psychological_levels_analysis, psychological_levels, psychological_level_interactions';
    RAISE NOTICE '📈 Views created: psychological_analysis_hourly, psychological_levels_hourly, psychological_interactions_hourly, psychological_levels_dashboard';
    RAISE NOTICE '🔧 Functions created: get_latest_psychological_analysis, get_active_psychological_levels, get_recent_level_interactions';
    RAISE NOTICE '👤 Permissions granted to alpha_emon user';
    RAISE NOTICE '🚀 Ready for standalone psychological levels analysis!';
END $$;
