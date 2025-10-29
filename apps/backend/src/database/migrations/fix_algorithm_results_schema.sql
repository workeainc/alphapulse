-- Fixed Algorithm Results Schema for AlphaPlus TimescaleDB
-- Fixes primary key issues for hypertables

-- Signal Confluence Table (fixed primary key)
CREATE TABLE IF NOT EXISTS signal_confluence (
    timestamp TIMESTAMPTZ NOT NULL,
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
    
    -- Fixed primary key to include timestamp for hypertable
    PRIMARY KEY (timestamp, symbol, timeframe),
    
    -- Constraints
    CONSTRAINT valid_direction CHECK (direction IN ('BUY', 'SELL', 'NEUTRAL')),
    CONSTRAINT valid_confluence CHECK (confluence_score >= 0 AND confluence_score <= 1),
    CONSTRAINT valid_ml_confidence CHECK (ml_confidence >= 0 AND ml_confidence <= 1),
    CONSTRAINT valid_signal_strength CHECK (signal_strength IN ('weak', 'medium', 'strong')),
    CONSTRAINT valid_market_regime CHECK (market_regime IN ('trending', 'ranging', 'volatile', 'unknown'))
);

-- Create TimescaleDB hypertable
SELECT create_hypertable('signal_confluence', 'timestamp', chunk_time_interval => INTERVAL '1 hour');

-- Create indexes for signal confluence
CREATE INDEX IF NOT EXISTS idx_signal_confluence_symbol_timeframe ON signal_confluence (symbol, timeframe);
CREATE INDEX IF NOT EXISTS idx_signal_confluence_timestamp ON signal_confluence (timestamp);
CREATE INDEX IF NOT EXISTS idx_signal_confluence_direction ON signal_confluence (direction);
CREATE INDEX IF NOT EXISTS idx_signal_confluence_confluence_score ON signal_confluence (confluence_score);
CREATE INDEX IF NOT EXISTS idx_signal_confluence_ml_confidence ON signal_confluence (ml_confidence);
CREATE INDEX IF NOT EXISTS idx_signal_confluence_sde_consensus ON signal_confluence (sde_consensus);

-- Algorithm Performance Metrics Table (fixed primary key)
CREATE TABLE IF NOT EXISTS algorithm_performance (
    timestamp TIMESTAMPTZ NOT NULL,
    algorithm_type VARCHAR(50) NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
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
    
    -- Fixed primary key to include timestamp for hypertable
    PRIMARY KEY (timestamp, algorithm_type, symbol, timeframe),
    
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

-- Grant permissions to alpha_emon user
GRANT ALL PRIVILEGES ON TABLE signal_confluence TO alpha_emon;
GRANT ALL PRIVILEGES ON TABLE algorithm_performance TO alpha_emon;

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

-- Insert initial performance records for all algorithms
INSERT INTO algorithm_performance (timestamp, algorithm_type, symbol, timeframe, performance_score)
SELECT 
    NOW() as timestamp,
    algorithm_type,
    'BTCUSDT' as symbol,
    '1m' as timeframe,
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

-- Final success message
DO $$
BEGIN
    RAISE NOTICE '✅ Fixed Algorithm Results Schema created successfully!';
    RAISE NOTICE '📊 Tables created: signal_confluence, algorithm_performance';
    RAISE NOTICE '📈 View created: algorithm_status_dashboard';
    RAISE NOTICE '👤 Permissions granted to alpha_emon user';
    RAISE NOTICE '🚀 Ready for algorithm integration!';
END $$;
