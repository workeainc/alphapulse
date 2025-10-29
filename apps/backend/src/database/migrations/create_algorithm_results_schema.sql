-- Enhanced Algorithm Results Schema for AlphaPlus TimescaleDB
-- Extends existing schema to store results from all 8 major algorithms
-- Connects algorithms to live data pipeline for signal generation

-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Extend existing ohlcv_data table with algorithm results
ALTER TABLE ohlcv_data ADD COLUMN IF NOT EXISTS support_levels JSONB;
ALTER TABLE ohlcv_data ADD COLUMN IF NOT EXISTS resistance_levels JSONB;
ALTER TABLE ohlcv_data ADD COLUMN IF NOT EXISTS psychological_levels JSONB;
ALTER TABLE ohlcv_data ADD COLUMN IF NOT EXISTS volume_patterns JSONB;
ALTER TABLE ohlcv_data ADD COLUMN IF NOT EXISTS chart_patterns JSONB;
ALTER TABLE ohlcv_data ADD COLUMN IF NOT EXISTS demand_zones JSONB;
ALTER TABLE ohlcv_data ADD COLUMN IF NOT EXISTS supply_zones JSONB;
ALTER TABLE ohlcv_data ADD COLUMN IF NOT EXISTS breakout_signals JSONB;
ALTER TABLE ohlcv_data ADD COLUMN IF NOT EXISTS algorithm_confidence NUMERIC(4,3) DEFAULT 0.0;

-- Algorithm Results Hypertable (stores detailed results from each algorithm)
CREATE TABLE IF NOT EXISTS algorithm_results (
    id BIGSERIAL,
    result_timestamp TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    algorithm_type VARCHAR(50) NOT NULL, -- 'support_resistance', 'volume_analysis', 'pattern_recognition', etc.
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
    PRIMARY KEY (id, result_timestamp)
);

-- Create TimescaleDB hypertable with 1-hour chunks
SELECT create_hypertable('algorithm_results', 'result_timestamp', chunk_time_interval => INTERVAL '1 hour');

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_algorithm_results_symbol_timeframe ON algorithm_results (symbol, timeframe);
CREATE INDEX IF NOT EXISTS idx_algorithm_results_timestamp ON algorithm_results (result_timestamp);
CREATE INDEX IF NOT EXISTS idx_algorithm_results_algorithm_type ON algorithm_results (algorithm_type);
CREATE INDEX IF NOT EXISTS idx_algorithm_results_symbol_timestamp ON algorithm_results (symbol, result_timestamp);
CREATE INDEX IF NOT EXISTS idx_algorithm_results_confidence ON algorithm_results (confidence_score);
CREATE INDEX IF NOT EXISTS idx_algorithm_results_strength ON algorithm_results (strength_score);

-- Signal Confluence Table (stores final signal decisions with algorithm confirmations)
CREATE TABLE IF NOT EXISTS signal_confluence (
    id BIGSERIAL PRIMARY KEY,
    signal_timestamp TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    direction VARCHAR(10) NOT NULL, -- 'BUY', 'SELL', 'NEUTRAL'
    confluence_score NUMERIC(4,3) NOT NULL,
    algorithm_confirmations JSONB NOT NULL, -- Which algorithms confirmed the signal
    ml_confidence NUMERIC(4,3) NOT NULL,
    sde_consensus BOOLEAN DEFAULT FALSE,
    signal_strength VARCHAR(20), -- 'weak', 'medium', 'strong'
    entry_price NUMERIC(20,8),
    stop_loss NUMERIC(20,8),
    take_profit NUMERIC(20,8),
    risk_reward_ratio NUMERIC(6,3),
    market_regime VARCHAR(20), -- 'trending', 'ranging', 'volatile'
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
    CONSTRAINT valid_market_regime CHECK (market_regime IN ('trending', 'ranging', 'volatile', 'unknown'))
);

-- Create TimescaleDB hypertable
SELECT create_hypertable('signal_confluence', 'signal_timestamp', chunk_time_interval => INTERVAL '1 hour');

-- Create indexes for signal confluence
CREATE INDEX IF NOT EXISTS idx_signal_confluence_symbol_timeframe ON signal_confluence (symbol, timeframe);
CREATE INDEX IF NOT EXISTS idx_signal_confluence_timestamp ON signal_confluence (signal_timestamp);
CREATE INDEX IF NOT EXISTS idx_signal_confluence_direction ON signal_confluence (direction);
CREATE INDEX IF NOT EXISTS idx_signal_confluence_confluence_score ON signal_confluence (confluence_score);
CREATE INDEX IF NOT EXISTS idx_signal_confluence_ml_confidence ON signal_confluence (ml_confidence);
CREATE INDEX IF NOT EXISTS idx_signal_confluence_sde_consensus ON signal_confluence (sde_consensus);

-- Algorithm Performance Metrics Table (tracks algorithm accuracy and performance)
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
SELECT create_hypertable('algorithm_performance', 'timestamp', chunk_time_interval => INTERVAL '1 day');

-- Create indexes for performance tracking
CREATE INDEX IF NOT EXISTS idx_algorithm_performance_type ON algorithm_performance (algorithm_type);
CREATE INDEX IF NOT EXISTS idx_algorithm_performance_symbol ON algorithm_performance (symbol);
CREATE INDEX IF NOT EXISTS idx_algorithm_performance_accuracy ON algorithm_performance (accuracy_rate);
CREATE INDEX IF NOT EXISTS idx_algorithm_performance_score ON algorithm_performance (performance_score);

-- Continuous Aggregates for Performance Optimization
-- Daily algorithm performance summary
CREATE MATERIALIZED VIEW IF NOT EXISTS algorithm_performance_daily AS
SELECT 
    time_bucket('1 day', timestamp) AS bucket,
    algorithm_type,
    symbol,
    timeframe,
    AVG(accuracy_rate) AS avg_accuracy,
    AVG(avg_confidence) AS avg_confidence,
    AVG(avg_strength) AS avg_strength,
    AVG(performance_score) AS avg_performance_score,
    SUM(total_signals) AS total_signals,
    SUM(correct_signals) AS total_correct_signals,
    AVG(processing_time_avg_ms) AS avg_processing_time_ms
FROM algorithm_performance
GROUP BY bucket, algorithm_type, symbol, timeframe
WITH DATA;

-- Hourly signal confluence summary
CREATE MATERIALIZED VIEW IF NOT EXISTS signal_confluence_hourly AS
SELECT 
    time_bucket('1 hour', signal_timestamp) AS bucket,
    symbol,
    timeframe,
    COUNT(*) AS total_signals,
    COUNT(*) FILTER (WHERE direction = 'BUY') AS buy_signals,
    COUNT(*) FILTER (WHERE direction = 'SELL') AS sell_signals,
    COUNT(*) FILTER (WHERE direction = 'NEUTRAL') AS neutral_signals,
    AVG(confluence_score) AS avg_confluence_score,
    AVG(ml_confidence) AS avg_ml_confidence,
    COUNT(*) FILTER (WHERE sde_consensus = TRUE) AS consensus_signals
FROM signal_confluence
GROUP BY bucket, symbol, timeframe
WITH DATA;

-- Add compression policies for old data
ALTER TABLE algorithm_results SET (timescaledb.compress, timescaledb.compress_segmentby = 'symbol, algorithm_type');
ALTER TABLE signal_confluence SET (timescaledb.compress, timescaledb.compress_segmentby = 'symbol');
ALTER TABLE algorithm_performance SET (timescaledb.compress, timescaledb.compress_segmentby = 'algorithm_type');

-- Add compression policies (compress data older than 1 month)
SELECT add_compression_policy('algorithm_results', INTERVAL '1 month');
SELECT add_compression_policy('signal_confluence', INTERVAL '1 month');
SELECT add_compression_policy('algorithm_performance', INTERVAL '3 months');

-- Create refresh policies for continuous aggregates
SELECT add_continuous_aggregate_policy('algorithm_performance_daily',
    start_offset => INTERVAL '1 day',
    end_offset => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour');

SELECT add_continuous_aggregate_policy('signal_confluence_hourly',
    start_offset => INTERVAL '1 hour',
    end_offset => INTERVAL '5 minutes',
    schedule_interval => INTERVAL '5 minutes');

-- Grant permissions to alpha_emon user
GRANT ALL PRIVILEGES ON TABLE algorithm_results TO alpha_emon;
GRANT ALL PRIVILEGES ON TABLE signal_confluence TO alpha_emon;
GRANT ALL PRIVILEGES ON TABLE algorithm_performance TO alpha_emon;
GRANT ALL PRIVILEGES ON TABLE algorithm_performance_daily TO alpha_emon;
GRANT ALL PRIVILEGES ON TABLE signal_confluence_hourly TO alpha_emon;

-- Grant sequence privileges
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO alpha_emon;

-- Create function to update algorithm performance
CREATE OR REPLACE FUNCTION update_algorithm_performance(
    p_algorithm_type VARCHAR(50),
    p_symbol VARCHAR(20),
    p_timeframe VARCHAR(10),
    p_accuracy_rate NUMERIC(4,3),
    p_avg_confidence NUMERIC(4,3),
    p_avg_strength NUMERIC(4,3),
    p_processing_time_ms INTEGER,
    p_success BOOLEAN,
    p_error_message TEXT DEFAULT NULL
) RETURNS VOID AS $$
BEGIN
    INSERT INTO algorithm_performance (
        algorithm_type, symbol, timeframe, timestamp,
        accuracy_rate, avg_confidence, avg_strength,
        processing_time_avg_ms, success_count, error_count,
        last_successful_run, last_error_message,
        performance_score
    ) VALUES (
        p_algorithm_type, p_symbol, p_timeframe, NOW(),
        p_accuracy_rate, p_avg_confidence, p_avg_strength,
        p_processing_time_ms,
        CASE WHEN p_success THEN 1 ELSE 0 END,
        CASE WHEN NOT p_success THEN 1 ELSE 0 END,
        CASE WHEN p_success THEN NOW() ELSE NULL END,
        p_error_message,
        (p_accuracy_rate + p_avg_confidence + p_avg_strength) / 3.0
    )
    ON CONFLICT (algorithm_type, symbol, timeframe, timestamp) 
    DO UPDATE SET
        accuracy_rate = EXCLUDED.accuracy_rate,
        avg_confidence = EXCLUDED.avg_confidence,
        avg_strength = EXCLUDED.avg_strength,
        processing_time_avg_ms = EXCLUDED.processing_time_avg_ms,
        success_count = algorithm_performance.success_count + EXCLUDED.success_count,
        error_count = algorithm_performance.error_count + EXCLUDED.error_count,
        last_successful_run = EXCLUDED.last_successful_run,
        last_error_message = EXCLUDED.last_error_message,
        performance_score = EXCLUDED.performance_score,
        updated_at = NOW();
END;
$$ LANGUAGE plpgsql;

-- Grant execute permission on the function
GRANT EXECUTE ON FUNCTION update_algorithm_performance TO alpha_emon;

-- Create function to calculate signal confluence score
CREATE OR REPLACE FUNCTION calculate_confluence_score(
    p_symbol VARCHAR(20),
    p_timeframe VARCHAR(10),
    p_timestamp TIMESTAMPTZ
) RETURNS NUMERIC(4,3) AS $$
DECLARE
    confluence_score NUMERIC(4,3) := 0.0;
    algorithm_count INTEGER := 0;
    avg_confidence NUMERIC(4,3) := 0.0;
BEGIN
    -- Count algorithms that provided results in the last hour
    SELECT COUNT(*), AVG(confidence_score)
    INTO algorithm_count, avg_confidence
    FROM algorithm_results
    WHERE symbol = p_symbol 
    AND timeframe = p_timeframe
    AND timestamp >= p_timestamp - INTERVAL '1 hour'
    AND timestamp <= p_timestamp;
    
    -- Calculate confluence score based on algorithm count and average confidence
    IF algorithm_count > 0 THEN
        confluence_score := (algorithm_count::NUMERIC / 8.0) * avg_confidence;
    END IF;
    
    RETURN LEAST(confluence_score, 1.0);
END;
$$ LANGUAGE plpgsql;

-- Grant execute permission on the function
GRANT EXECUTE ON FUNCTION calculate_confluence_score TO alpha_emon;

-- Insert initial performance records for all algorithms
INSERT INTO algorithm_performance (algorithm_type, symbol, timeframe, timestamp, performance_score)
SELECT 
    algorithm_type,
    'BTCUSDT' as symbol,
    '1m' as timeframe,
    NOW() as timestamp,
    0.0 as performance_score
FROM (VALUES 
    ('support_resistance'),
    ('demand_supply_zones'),
    ('volume_analysis'),
    ('pattern_recognition'),
    ('breakout_detection'),
    ('chart_patterns'),
    ('psychological_levels'),
    ('volume_weighted_levels')
) AS t(algorithm_type)
ON CONFLICT DO NOTHING;

-- Create view for algorithm status dashboard
CREATE OR REPLACE VIEW algorithm_status_dashboard AS
SELECT 
    ar.algorithm_type,
    ar.symbol,
    ar.timeframe,
    COUNT(*) as total_runs_last_hour,
    AVG(ar.confidence_score) as avg_confidence,
    AVG(ar.strength_score) as avg_strength,
    AVG(ar.processing_time_ms) as avg_processing_time,
    ap.accuracy_rate,
    ap.performance_score,
    ap.last_successful_run,
    ap.error_count,
    CASE 
        WHEN ap.last_successful_run > NOW() - INTERVAL '1 hour' THEN 'ACTIVE'
        WHEN ap.last_successful_run > NOW() - INTERVAL '6 hours' THEN 'WARNING'
        ELSE 'INACTIVE'
    END as status
FROM algorithm_results ar
LEFT JOIN algorithm_performance ap ON (
    ar.algorithm_type = ap.algorithm_type 
    AND ar.symbol = ap.symbol 
    AND ar.timeframe = ap.timeframe
)
WHERE ar.timestamp >= NOW() - INTERVAL '1 hour'
GROUP BY ar.algorithm_type, ar.symbol, ar.timeframe, 
         ap.accuracy_rate, ap.performance_score, ap.last_successful_run, ap.error_count;

-- Grant select permission on the view
GRANT SELECT ON algorithm_status_dashboard TO alpha_emon;

-- Final success message
DO $$
BEGIN
    RAISE NOTICE '✅ Enhanced Algorithm Results Schema created successfully!';
    RAISE NOTICE '📊 Tables created: algorithm_results, signal_confluence, algorithm_performance';
    RAISE NOTICE '📈 Views created: algorithm_performance_daily, signal_confluence_hourly, algorithm_status_dashboard';
    RAISE NOTICE '🔧 Functions created: update_algorithm_performance, calculate_confluence_score';
    RAISE NOTICE '👤 Permissions granted to alpha_emon user';
    RAISE NOTICE '🚀 Ready for algorithm integration!';
END $$;
