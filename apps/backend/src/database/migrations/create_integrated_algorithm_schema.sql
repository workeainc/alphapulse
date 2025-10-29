-- Integrated Algorithm Schema for AlphaPlus TimescaleDB
-- Creates essential tables for enhanced algorithm implementations
-- Handles existing tables gracefully

-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Algorithm Results Table (stores detailed results from each algorithm)
CREATE TABLE IF NOT EXISTS algorithm_results (
    id BIGSERIAL,
    timestamp TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    algorithm_type VARCHAR(50) NOT NULL,
    algorithm_version VARCHAR(20) DEFAULT '1.0',
    result_data JSONB NOT NULL,
    confidence_score NUMERIC(4,3) NOT NULL,
    strength_score NUMERIC(4,3) NOT NULL,
    processing_time_ms INTEGER,
    data_quality_score NUMERIC(4,3) DEFAULT 1.0,
    created_at TIMESTAMPTZ DEFAULT NOW() NOT NULL,
    
    -- Constraints
    CONSTRAINT valid_confidence CHECK (confidence_score >= 0 AND confidence_score <= 1),
    CONSTRAINT valid_strength CHECK (strength_score >= 0 AND strength_score <= 1),
    CONSTRAINT valid_algorithm_type CHECK (algorithm_type IN (
        'support_resistance', 'demand_supply_zones', 'volume_analysis', 
        'pattern_recognition', 'breakout_detection', 'chart_patterns',
        'psychological_levels', 'volume_weighted_levels'
    )),
    PRIMARY KEY (id, timestamp)
);

-- Create TimescaleDB hypertable with 1-hour chunks
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM timescaledb_information.hypertables 
        WHERE hypertable_name = 'algorithm_results'
    ) THEN
        PERFORM create_hypertable('algorithm_results', 'timestamp', chunk_time_interval => INTERVAL '1 hour');
    END IF;
END $$;

-- Signal Confluence Table (stores final signal decisions with algorithm confirmations)
CREATE TABLE IF NOT EXISTS signal_confluence (
    timestamp TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    direction VARCHAR(10) NOT NULL,
    confluence_score NUMERIC(4,3) NOT NULL,
    algorithm_confirmations JSONB NOT NULL,
    ml_confidence NUMERIC(4,3) NOT NULL,
    sde_consensus BOOLEAN DEFAULT FALSE,
    signal_strength VARCHAR(20),
    entry_price NUMERIC(20,8),
    stop_loss NUMERIC(20,8),
    take_profit NUMERIC(20,8),
    risk_reward_ratio NUMERIC(6,3),
    market_regime VARCHAR(20),
    volume_confirmation BOOLEAN DEFAULT FALSE,
    pattern_confirmation BOOLEAN DEFAULT FALSE,
    breakout_confirmation BOOLEAN DEFAULT FALSE,
    support_resistance_confirmation BOOLEAN DEFAULT FALSE,
    demand_supply_confirmation BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT NOW() NOT NULL,
    
    -- Constraints
    CONSTRAINT valid_direction CHECK (direction IN ('BUY', 'SELL', 'NEUTRAL')),
    CONSTRAINT valid_confluence CHECK (confluence_score >= 0 AND confluence_score <= 1),
    CONSTRAINT valid_ml_confidence CHECK (ml_confidence >= 0 AND ml_confidence <= 1),
    CONSTRAINT valid_signal_strength CHECK (signal_strength IN ('weak', 'medium', 'strong')),
    CONSTRAINT valid_market_regime CHECK (market_regime IN ('trending', 'ranging', 'volatile', 'unknown')),
    PRIMARY KEY (timestamp)
);

-- Create TimescaleDB hypertable
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM timescaledb_information.hypertables 
        WHERE hypertable_name = 'signal_confluence'
    ) THEN
        PERFORM create_hypertable('signal_confluence', 'timestamp', chunk_time_interval => INTERVAL '1 hour');
    END IF;
END $$;

-- Algorithm Performance Metrics Table
CREATE TABLE IF NOT EXISTS algorithm_performance (
    id BIGSERIAL PRIMARY KEY,
    algorithm_type VARCHAR(50) NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    total_signals INTEGER DEFAULT 0,
    correct_signals INTEGER DEFAULT 0,
    accuracy_rate NUMERIC(4,3) DEFAULT 0.0,
    avg_confidence NUMERIC(4,3) DEFAULT 0.0,
    avg_strength NUMERIC(4,3) DEFAULT 0.0,
    processing_time_avg_ms INTEGER DEFAULT 0,
    error_count INTEGER DEFAULT 0,
    success_count INTEGER DEFAULT 0,
    last_successful_run TIMESTAMPTZ,
    last_error_message TEXT,
    performance_score NUMERIC(4,3) DEFAULT 0.0,
    created_at TIMESTAMPTZ DEFAULT NOW() NOT NULL,
    updated_at TIMESTAMPTZ DEFAULT NOW() NOT NULL,
    
    -- Constraints
    CONSTRAINT valid_accuracy CHECK (accuracy_rate >= 0 AND accuracy_rate <= 1),
    CONSTRAINT valid_performance_score CHECK (performance_score >= 0 AND performance_score <= 1)
);

-- Create TimescaleDB hypertable
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM timescaledb_information.hypertables 
        WHERE hypertable_name = 'algorithm_performance'
    ) THEN
        PERFORM create_hypertable('algorithm_performance', 'timestamp', chunk_time_interval => INTERVAL '1 day');
    END IF;
END $$;

-- Enhanced Algorithm Performance Monitoring Table
CREATE TABLE IF NOT EXISTS enhanced_algorithm_metrics (
    timestamp TIMESTAMPTZ NOT NULL,
    psychological_levels_processed INTEGER DEFAULT 0,
    volume_weighted_levels_processed INTEGER DEFAULT 0,
    orderbook_analyses_processed INTEGER DEFAULT 0,
    algorithm_integration_cycles INTEGER DEFAULT 0,
    avg_psychological_analysis_time_ms FLOAT DEFAULT 0.0,
    avg_volume_analysis_time_ms FLOAT DEFAULT 0.0,
    avg_orderbook_analysis_time_ms FLOAT DEFAULT 0.0,
    algorithm_success_rate FLOAT DEFAULT 0.0,
    enhanced_signals_generated INTEGER DEFAULT 0,
    algorithm_data_points_collected INTEGER DEFAULT 0,
    PRIMARY KEY (timestamp)
);

-- Convert to hypertable if not already
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM timescaledb_information.hypertables 
        WHERE hypertable_name = 'enhanced_algorithm_metrics'
    ) THEN
        PERFORM create_hypertable('enhanced_algorithm_metrics', 'timestamp');
    END IF;
END $$;

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_algorithm_results_symbol_timeframe ON algorithm_results (symbol, timeframe);
CREATE INDEX IF NOT EXISTS idx_algorithm_results_timestamp ON algorithm_results (timestamp);
CREATE INDEX IF NOT EXISTS idx_algorithm_results_algorithm_type ON algorithm_results (algorithm_type);
CREATE INDEX IF NOT EXISTS idx_algorithm_results_confidence ON algorithm_results (confidence_score);
CREATE INDEX IF NOT EXISTS idx_algorithm_results_strength ON algorithm_results (strength_score);

CREATE INDEX IF NOT EXISTS idx_signal_confluence_symbol_timeframe ON signal_confluence (symbol, timeframe);
CREATE INDEX IF NOT EXISTS idx_signal_confluence_timestamp ON signal_confluence (timestamp);
CREATE INDEX IF NOT EXISTS idx_signal_confluence_direction ON signal_confluence (direction);
CREATE INDEX IF NOT EXISTS idx_signal_confluence_confluence_score ON signal_confluence (confluence_score);
CREATE INDEX IF NOT EXISTS idx_signal_confluence_ml_confidence ON signal_confluence (ml_confidence);

CREATE INDEX IF NOT EXISTS idx_algorithm_performance_type ON algorithm_performance (algorithm_type);
CREATE INDEX IF NOT EXISTS idx_algorithm_performance_symbol ON algorithm_performance (symbol);
CREATE INDEX IF NOT EXISTS idx_algorithm_performance_accuracy ON algorithm_performance (accuracy_rate);
CREATE INDEX IF NOT EXISTS idx_algorithm_performance_score ON algorithm_performance (performance_score);

CREATE INDEX IF NOT EXISTS idx_enhanced_algorithm_metrics_timestamp ON enhanced_algorithm_metrics (timestamp);
CREATE INDEX IF NOT EXISTS idx_enhanced_algorithm_metrics_success_rate ON enhanced_algorithm_metrics (algorithm_success_rate);

-- Grant permissions to alpha_emon user
GRANT ALL PRIVILEGES ON TABLE algorithm_results TO alpha_emon;
GRANT ALL PRIVILEGES ON TABLE signal_confluence TO alpha_emon;
GRANT ALL PRIVILEGES ON TABLE algorithm_performance TO alpha_emon;
GRANT ALL PRIVILEGES ON TABLE enhanced_algorithm_metrics TO alpha_emon;

-- Grant sequence privileges
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO alpha_emon;

-- Final success message
DO $$
BEGIN
    RAISE NOTICE '✅ Integrated Algorithm Schema created successfully!';
    RAISE NOTICE '📊 Tables created: algorithm_results, signal_confluence, algorithm_performance, enhanced_algorithm_metrics';
    RAISE NOTICE '👤 Permissions granted to alpha_emon user';
    RAISE NOTICE '🚀 Ready for enhanced algorithm integration!';
END $$;
