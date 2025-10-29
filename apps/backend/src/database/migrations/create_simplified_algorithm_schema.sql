-- Simplified Migration for AlphaPlus Algorithm Integration
-- Creates essential tables for the enhanced algorithm implementations

-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Volume Profile Analysis Table (Compatible with existing schema)
CREATE TABLE IF NOT EXISTS volume_profile_analysis (
    id BIGSERIAL,
    symbol VARCHAR(20) NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,  -- Use standard timestamp field
    poc_price NUMERIC(20,8) NOT NULL,
    poc_volume NUMERIC(20,8) NOT NULL,
    value_area_high NUMERIC(20,8) NOT NULL,
    value_area_low NUMERIC(20,8) NOT NULL,
    value_area_volume NUMERIC(20,8) NOT NULL,
    total_volume NUMERIC(20,8) NOT NULL,
    analysis_confidence DECIMAL(5,4) NOT NULL DEFAULT 0.0,  -- Use DECIMAL(5,4) for consistency
    algorithm_inputs JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW() NOT NULL,
    PRIMARY KEY (id, timestamp)  -- Use timestamp for hypertable compatibility
);

-- Create TimescaleDB hypertable
SELECT create_hypertable('volume_profile_analysis', 'timestamp', chunk_time_interval => INTERVAL '1 hour');

-- Add unique constraint for volume profile analysis
ALTER TABLE volume_profile_analysis ADD CONSTRAINT uk_volume_profile_symbol_timeframe_timestamp 
UNIQUE (symbol, timeframe, timestamp);

-- Order Book Levels Table (Compatible with existing schema)
CREATE TABLE IF NOT EXISTS order_book_levels (
    id BIGSERIAL,
    symbol VARCHAR(20) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,  -- Use standard timestamp field
    level_type VARCHAR(30) NOT NULL,
    price_level NUMERIC(20,8) NOT NULL,
    volume_at_level NUMERIC(20,8) NOT NULL,
    volume_percentage NUMERIC(6,3) NOT NULL,
    bid_volume NUMERIC(20,8) NOT NULL DEFAULT 0,
    ask_volume NUMERIC(20,8) NOT NULL DEFAULT 0,
    volume_imbalance NUMERIC(8,4) NOT NULL DEFAULT 0,
    level_strength DECIMAL(5,4) NOT NULL DEFAULT 0.0,  -- Use DECIMAL(5,4) for consistency
    confidence DECIMAL(5,4) NOT NULL DEFAULT 0.0,      -- Use DECIMAL(5,4) for consistency
    is_active BOOLEAN DEFAULT TRUE,
    touch_count INTEGER DEFAULT 0,
    last_touch_time TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW() NOT NULL,
    PRIMARY KEY (id, timestamp)  -- Use timestamp for hypertable compatibility
);

-- Create TimescaleDB hypertable
SELECT create_hypertable('order_book_levels', 'timestamp', chunk_time_interval => INTERVAL '1 hour');

-- Market Microstructure Table (Compatible with existing schema)
CREATE TABLE IF NOT EXISTS market_microstructure (
    id BIGSERIAL,
    symbol VARCHAR(20) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,  -- Use standard timestamp field
    bid_ask_imbalance NUMERIC(8,4) NOT NULL DEFAULT 0,
    depth_pressure NUMERIC(8,4) NOT NULL DEFAULT 0,
    liquidity_score DECIMAL(5,4) NOT NULL DEFAULT 0.0,  -- Use DECIMAL(5,4) for consistency
    total_bid_volume NUMERIC(20,8) NOT NULL DEFAULT 0,
    total_ask_volume NUMERIC(20,8) NOT NULL DEFAULT 0,
    total_volume NUMERIC(20,8) NOT NULL DEFAULT 0,
    spread NUMERIC(20,8) NOT NULL DEFAULT 0,
    spread_percentage NUMERIC(8,4) NOT NULL DEFAULT 0,
    mid_price NUMERIC(20,8) NOT NULL DEFAULT 0,
    best_bid NUMERIC(20,8) NOT NULL DEFAULT 0,
    best_ask NUMERIC(20,8) NOT NULL DEFAULT 0,
    analysis_confidence DECIMAL(5,4) NOT NULL DEFAULT 0.0,  -- Use DECIMAL(5,4) for consistency
    raw_order_book_data JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW() NOT NULL,
    PRIMARY KEY (id, timestamp)  -- Use timestamp for hypertable compatibility
);

-- Create TimescaleDB hypertable
SELECT create_hypertable('market_microstructure', 'timestamp', chunk_time_interval => INTERVAL '1 hour');

-- Psychological Levels Analysis Table (Compatible with existing schema)
CREATE TABLE IF NOT EXISTS psychological_levels_analysis (
    id BIGSERIAL,
    symbol VARCHAR(20) NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,  -- Use standard timestamp field
    current_price NUMERIC(20,8) NOT NULL,
    nearest_support_price NUMERIC(20,8),
    nearest_resistance_price NUMERIC(20,8),
    market_regime VARCHAR(20) NOT NULL DEFAULT 'unknown',
    analysis_confidence DECIMAL(5,4) NOT NULL DEFAULT 0.0,  -- Use DECIMAL(5,4) for consistency
    algorithm_inputs JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW() NOT NULL,
    PRIMARY KEY (id, timestamp)  -- Use timestamp for hypertable compatibility
);

-- Create TimescaleDB hypertable
SELECT create_hypertable('psychological_levels_analysis', 'timestamp', chunk_time_interval => INTERVAL '1 hour');

-- Add unique constraint for psychological levels analysis
ALTER TABLE psychological_levels_analysis ADD CONSTRAINT uk_psychological_analysis_symbol_timeframe_timestamp 
UNIQUE (symbol, timeframe, timestamp);

-- Psychological Levels Table (Compatible with existing schema)
CREATE TABLE IF NOT EXISTS psychological_levels (
    id BIGSERIAL,
    symbol VARCHAR(20) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,  -- Use standard timestamp field
    level_type VARCHAR(30) NOT NULL,
    price_level NUMERIC(20,8) NOT NULL,
    strength DECIMAL(5,4) NOT NULL DEFAULT 0.0,  -- Use DECIMAL(5,4) for consistency
    confidence DECIMAL(5,4) NOT NULL DEFAULT 0.0,  -- Use DECIMAL(5,4) for consistency
    touch_count INTEGER DEFAULT 0,
    market_context VARCHAR(20),
    first_touch_time TIMESTAMPTZ,
    last_touch_time TIMESTAMPTZ,
    is_active BOOLEAN DEFAULT TRUE,
    is_broken BOOLEAN DEFAULT FALSE,
    break_time TIMESTAMPTZ,
    rejection_count INTEGER DEFAULT 0,
    penetration_count INTEGER DEFAULT 0,
    volume_at_level NUMERIC(20,8) DEFAULT 0.0,
    created_at TIMESTAMPTZ DEFAULT NOW() NOT NULL,
    PRIMARY KEY (id, timestamp)  -- Use timestamp for hypertable compatibility
);

-- Create TimescaleDB hypertable
SELECT create_hypertable('psychological_levels', 'timestamp', chunk_time_interval => INTERVAL '1 hour');

-- Add unique constraint for psychological levels
ALTER TABLE psychological_levels ADD CONSTRAINT uk_psychological_levels_symbol_type_price_timestamp 
UNIQUE (symbol, level_type, price_level, timestamp);

-- Psychological Level Interactions Table (Compatible with existing schema)
CREATE TABLE IF NOT EXISTS psychological_level_interactions (
    id BIGSERIAL,
    symbol VARCHAR(20) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,  -- Use standard timestamp field
    level_price NUMERIC(20,8) NOT NULL,
    level_type VARCHAR(30) NOT NULL,
    interaction_price NUMERIC(20,8) NOT NULL,
    interaction_volume NUMERIC(20,8) NOT NULL,
    reaction_type VARCHAR(20) NOT NULL,
    reaction_strength DECIMAL(5,4) NOT NULL DEFAULT 0.0,  -- Use DECIMAL(5,4) for consistency
    volume_confirmation BOOLEAN DEFAULT FALSE,
    follow_through BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT NOW() NOT NULL,
    PRIMARY KEY (id, timestamp)  -- Use timestamp for hypertable compatibility
);

-- Create TimescaleDB hypertable
SELECT create_hypertable('psychological_level_interactions', 'timestamp', chunk_time_interval => INTERVAL '1 hour');

-- Create basic indexes (Updated for compatibility)
CREATE INDEX IF NOT EXISTS idx_volume_profile_symbol_timeframe ON volume_profile_analysis (symbol, timeframe);
CREATE INDEX IF NOT EXISTS idx_volume_profile_timestamp ON volume_profile_analysis (timestamp);
CREATE INDEX IF NOT EXISTS idx_order_book_levels_symbol ON order_book_levels (symbol);
CREATE INDEX IF NOT EXISTS idx_order_book_levels_timestamp ON order_book_levels (timestamp);
CREATE INDEX IF NOT EXISTS idx_market_microstructure_symbol ON market_microstructure (symbol);
CREATE INDEX IF NOT EXISTS idx_market_microstructure_timestamp ON market_microstructure (timestamp);
CREATE INDEX IF NOT EXISTS idx_psychological_analysis_symbol ON psychological_levels_analysis (symbol);
CREATE INDEX IF NOT EXISTS idx_psychological_analysis_timestamp ON psychological_levels_analysis (timestamp);
CREATE INDEX IF NOT EXISTS idx_psychological_levels_symbol ON psychological_levels (symbol);
CREATE INDEX IF NOT EXISTS idx_psychological_levels_timestamp ON psychological_levels (timestamp);
CREATE INDEX IF NOT EXISTS idx_psychological_interactions_symbol ON psychological_level_interactions (symbol);
CREATE INDEX IF NOT EXISTS idx_psychological_interactions_timestamp ON psychological_level_interactions (timestamp);

-- Grant permissions to alpha_emon user
GRANT ALL PRIVILEGES ON TABLE volume_profile_analysis TO alpha_emon;
GRANT ALL PRIVILEGES ON TABLE order_book_levels TO alpha_emon;
GRANT ALL PRIVILEGES ON TABLE market_microstructure TO alpha_emon;
GRANT ALL PRIVILEGES ON TABLE psychological_levels_analysis TO alpha_emon;
GRANT ALL PRIVILEGES ON TABLE psychological_levels TO alpha_emon;
GRANT ALL PRIVILEGES ON TABLE psychological_level_interactions TO alpha_emon;

-- Grant sequence privileges
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO alpha_emon;

-- Final success message
DO $$
BEGIN
    RAISE NOTICE '✅ Simplified AlphaPlus Algorithm Integration Schema created successfully!';
    RAISE NOTICE '📊 Tables created: volume_profile_analysis, order_book_levels, market_microstructure, psychological_levels_analysis, psychological_levels, psychological_level_interactions';
    RAISE NOTICE '👤 Permissions granted to alpha_emon user';
    RAISE NOTICE '🚀 Ready for enhanced algorithm integration!';
END $$;
