-- Enhanced Data Tables for AlphaPlus Phase 1
-- Create comprehensive market data and pattern analysis tables

-- Enhanced Market Data Table (TimescaleDB hypertable)
CREATE TABLE IF NOT EXISTS enhanced_market_data (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    open NUMERIC(20,8) NOT NULL,
    high NUMERIC(20,8) NOT NULL,
    low NUMERIC(20,8) NOT NULL,
    close NUMERIC(20,8) NOT NULL,
    volume NUMERIC(20,8) NOT NULL,
    price_change NUMERIC(10,6),
    volume_change NUMERIC(10,6),
    volatility NUMERIC(10,6),
    rsi NUMERIC(6,3),
    macd NUMERIC(10,6),
    macd_signal NUMERIC(10,6),
    bollinger_upper NUMERIC(20,8),
    bollinger_lower NUMERIC(20,8),
    bollinger_middle NUMERIC(20,8),
    atr NUMERIC(20,8),
    support_level NUMERIC(20,8),
    resistance_level NUMERIC(20,8),
    market_sentiment NUMERIC(4,3),
    data_quality_score NUMERIC(4,3),
    created_at TIMESTAMPTZ DEFAULT NOW() NOT NULL
);

-- Create TimescaleDB hypertable
SELECT create_hypertable('enhanced_market_data', 'timestamp', chunk_time_interval => INTERVAL '1 day');

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_enhanced_market_data_symbol_timeframe ON enhanced_market_data (symbol, timeframe);
CREATE INDEX IF NOT EXISTS idx_enhanced_market_data_timestamp ON enhanced_market_data (timestamp);
CREATE INDEX IF NOT EXISTS idx_enhanced_market_data_symbol_timestamp ON enhanced_market_data (symbol, timestamp);

-- Pattern Detection Table
CREATE TABLE IF NOT EXISTS pattern_detections (
    id BIGSERIAL PRIMARY KEY,
    pattern_id VARCHAR(50) UNIQUE NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    pattern_type VARCHAR(50) NOT NULL,
    pattern_category VARCHAR(20) NOT NULL, -- continuation, reversal, bilateral
    direction VARCHAR(10) NOT NULL, -- long, short, neutral
    confidence NUMERIC(4,3) NOT NULL,
    strength VARCHAR(20) NOT NULL, -- weak, moderate, strong
    entry_price NUMERIC(20,8) NOT NULL,
    stop_loss NUMERIC(20,8) NOT NULL,
    take_profit NUMERIC(20,8) NOT NULL,
    risk_reward_ratio NUMERIC(6,2) NOT NULL,
    pattern_start_time TIMESTAMPTZ NOT NULL,
    pattern_end_time TIMESTAMPTZ NOT NULL,
    breakout_price NUMERIC(20,8),
    breakout_time TIMESTAMPTZ,
    volume_confirmation BOOLEAN,
    technical_indicators JSONB,
    market_conditions JSONB,
    data_points_used INTEGER NOT NULL,
    data_quality_score NUMERIC(4,3) NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'active', -- active, completed, failed
    created_at TIMESTAMPTZ DEFAULT NOW() NOT NULL,
    updated_at TIMESTAMPTZ DEFAULT NOW() NOT NULL
);

-- Create indexes for pattern detections
CREATE INDEX IF NOT EXISTS idx_pattern_detections_symbol_timeframe ON pattern_detections (symbol, timeframe);
CREATE INDEX IF NOT EXISTS idx_pattern_detections_pattern_type ON pattern_detections (pattern_type);
CREATE INDEX IF NOT EXISTS idx_pattern_detections_confidence ON pattern_detections (confidence);
CREATE INDEX IF NOT EXISTS idx_pattern_detections_status ON pattern_detections (status);
CREATE INDEX IF NOT EXISTS idx_pattern_detections_created_at ON pattern_detections (created_at);

-- Signal History Table
CREATE TABLE IF NOT EXISTS signal_history (
    id BIGSERIAL PRIMARY KEY,
    signal_id VARCHAR(50) UNIQUE NOT NULL,
    pattern_id VARCHAR(50),
    symbol VARCHAR(20) NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    direction VARCHAR(10) NOT NULL, -- long, short
    signal_type VARCHAR(20) NOT NULL, -- entry, exit, stop_loss, take_profit
    entry_price NUMERIC(20,8) NOT NULL,
    exit_price NUMERIC(20,8),
    stop_loss NUMERIC(20,8) NOT NULL,
    take_profit NUMERIC(20,8) NOT NULL,
    risk_amount NUMERIC(10,2), -- Risk in USD
    position_size NUMERIC(10,8), -- Position size in crypto
    risk_reward_ratio NUMERIC(6,2) NOT NULL,
    confidence NUMERIC(4,3) NOT NULL,
    pattern_type VARCHAR(50) NOT NULL,
    technical_analysis JSONB,
    market_sentiment NUMERIC(4,3),
    signal_generated_at TIMESTAMPTZ NOT NULL,
    signal_expires_at TIMESTAMPTZ,
    executed_at TIMESTAMPTZ,
    closed_at TIMESTAMPTZ,
    pnl NUMERIC(10,2), -- Profit/Loss in USD
    pnl_percentage NUMERIC(6,2),
    status VARCHAR(20) NOT NULL DEFAULT 'generated', -- generated, executed, closed, expired, cancelled
    execution_notes TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW() NOT NULL,
    updated_at TIMESTAMPTZ DEFAULT NOW() NOT NULL
);

-- Create indexes for signal history
CREATE INDEX IF NOT EXISTS idx_signal_history_symbol_timeframe ON signal_history (symbol, timeframe);
CREATE INDEX IF NOT EXISTS idx_signal_history_status ON signal_history (status);
CREATE INDEX IF NOT EXISTS idx_signal_history_direction ON signal_history (direction);
CREATE INDEX IF NOT EXISTS idx_signal_history_generated_at ON signal_history (signal_generated_at);
CREATE INDEX IF NOT EXISTS idx_signal_history_executed_at ON signal_history (executed_at);

-- Performance Metrics Table
CREATE TABLE IF NOT EXISTS performance_metrics (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    pattern_type VARCHAR(50) NOT NULL,
    period_start TIMESTAMPTZ NOT NULL,
    period_end TIMESTAMPTZ NOT NULL,
    total_signals INTEGER NOT NULL DEFAULT 0,
    winning_signals INTEGER NOT NULL DEFAULT 0,
    losing_signals INTEGER NOT NULL DEFAULT 0,
    win_rate NUMERIC(5,2) NOT NULL DEFAULT 0,
    total_pnl NUMERIC(15,2) NOT NULL DEFAULT 0,
    total_pnl_percentage NUMERIC(8,2) NOT NULL DEFAULT 0,
    average_pnl NUMERIC(10,2) NOT NULL DEFAULT 0,
    average_pnl_percentage NUMERIC(6,2) NOT NULL DEFAULT 0,
    max_drawdown NUMERIC(8,2) NOT NULL DEFAULT 0,
    sharpe_ratio NUMERIC(6,3),
    profit_factor NUMERIC(6,2),
    average_risk_reward NUMERIC(6,2) NOT NULL DEFAULT 0,
    average_confidence NUMERIC(4,3) NOT NULL DEFAULT 0,
    best_signal_pnl NUMERIC(10,2),
    worst_signal_pnl NUMERIC(10,2),
    long_signals INTEGER NOT NULL DEFAULT 0,
    short_signals INTEGER NOT NULL DEFAULT 0,
    long_win_rate NUMERIC(5,2) NOT NULL DEFAULT 0,
    short_win_rate NUMERIC(5,2) NOT NULL DEFAULT 0,
    market_conditions JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW() NOT NULL,
    updated_at TIMESTAMPTZ DEFAULT NOW() NOT NULL,
    UNIQUE(symbol, timeframe, pattern_type, period_start)
);

-- Create indexes for performance metrics
CREATE INDEX IF NOT EXISTS idx_performance_metrics_symbol_timeframe ON performance_metrics (symbol, timeframe);
CREATE INDEX IF NOT EXISTS idx_performance_metrics_pattern_type ON performance_metrics (pattern_type);
CREATE INDEX IF NOT EXISTS idx_performance_metrics_period ON performance_metrics (period_start, period_end);
CREATE INDEX IF NOT EXISTS idx_performance_metrics_win_rate ON performance_metrics (win_rate);

-- Confidence Scoring Table
CREATE TABLE IF NOT EXISTS confidence_scores (
    id BIGSERIAL PRIMARY KEY,
    pattern_type VARCHAR(50) NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    market_condition VARCHAR(50) NOT NULL, -- bull, bear, sideways, volatile
    volatility_level VARCHAR(20) NOT NULL, -- low, medium, high
    volume_level VARCHAR(20) NOT NULL, -- low, medium, high
    historical_accuracy NUMERIC(5,2) NOT NULL,
    sample_size INTEGER NOT NULL,
    confidence_score NUMERIC(4,3) NOT NULL,
    confidence_factors JSONB,
    last_updated TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW() NOT NULL,
    UNIQUE(pattern_type, symbol, timeframe, market_condition, volatility_level, volume_level)
);

-- Create indexes for confidence scores
CREATE INDEX IF NOT EXISTS idx_confidence_scores_pattern_type ON confidence_scores (pattern_type);
CREATE INDEX IF NOT EXISTS idx_confidence_scores_symbol_timeframe ON confidence_scores (symbol, timeframe);
CREATE INDEX IF NOT EXISTS idx_confidence_scores_confidence ON confidence_scores (confidence_score);

-- Market Conditions Table
CREATE TABLE IF NOT EXISTS market_conditions (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    market_regime VARCHAR(20) NOT NULL, -- bull, bear, sideways, volatile
    volatility_level VARCHAR(20) NOT NULL, -- low, medium, high
    volume_level VARCHAR(20) NOT NULL, -- low, medium, high
    trend_strength NUMERIC(4,3) NOT NULL,
    momentum_score NUMERIC(4,3) NOT NULL,
    support_resistance_quality NUMERIC(4,3) NOT NULL,
    market_sentiment NUMERIC(4,3) NOT NULL,
    technical_indicators JSONB,
    market_metrics JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW() NOT NULL,
    UNIQUE(symbol, timeframe, timestamp)
);

-- Create TimescaleDB hypertable for market conditions
SELECT create_hypertable('market_conditions', 'timestamp', chunk_time_interval => INTERVAL '1 day');

-- Create indexes for market conditions
CREATE INDEX IF NOT EXISTS idx_market_conditions_symbol_timeframe ON market_conditions (symbol, timeframe);
CREATE INDEX IF NOT EXISTS idx_market_conditions_timestamp ON market_conditions (timestamp);
CREATE INDEX IF NOT EXISTS idx_market_conditions_regime ON market_conditions (market_regime);

-- Data Quality Metrics Table
CREATE TABLE IF NOT EXISTS data_quality_metrics (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    completeness NUMERIC(4,3) NOT NULL,
    accuracy NUMERIC(4,3) NOT NULL,
    consistency NUMERIC(4,3) NOT NULL,
    timeliness NUMERIC(4,3) NOT NULL,
    validity NUMERIC(4,3) NOT NULL,
    overall_score NUMERIC(4,3) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW() NOT NULL,
    UNIQUE(symbol, timeframe, timestamp)
);

-- Create TimescaleDB hypertable for data quality metrics
SELECT create_hypertable('data_quality_metrics', 'timestamp', chunk_time_interval => INTERVAL '1 day');

-- Create indexes for data quality metrics
CREATE INDEX IF NOT EXISTS idx_data_quality_metrics_symbol_timeframe ON data_quality_metrics (symbol, timeframe);
CREATE INDEX IF NOT EXISTS idx_data_quality_metrics_timestamp ON data_quality_metrics (timestamp);
CREATE INDEX IF NOT EXISTS idx_data_quality_metrics_overall_score ON data_quality_metrics (overall_score);

-- Data Anomalies Table
CREATE TABLE IF NOT EXISTS data_anomalies (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    anomaly_type VARCHAR(50) NOT NULL, -- price_spike, volume_spike, missing_data, etc.
    severity VARCHAR(20) NOT NULL, -- low, medium, high, critical
    description TEXT NOT NULL,
    suggested_action TEXT NOT NULL,
    resolved BOOLEAN NOT NULL DEFAULT FALSE,
    resolved_at TIMESTAMPTZ,
    resolution_notes TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW() NOT NULL,
    UNIQUE(symbol, timeframe, timestamp, anomaly_type)
);

-- Create TimescaleDB hypertable for data anomalies
SELECT create_hypertable('data_anomalies', 'timestamp', chunk_time_interval => INTERVAL '1 day');

-- Create indexes for data anomalies
CREATE INDEX IF NOT EXISTS idx_data_anomalies_symbol_timeframe ON data_anomalies (symbol, timeframe);
CREATE INDEX IF NOT EXISTS idx_data_anomalies_timestamp ON data_anomalies (timestamp);
CREATE INDEX IF NOT EXISTS idx_data_anomalies_type ON data_anomalies (anomaly_type);
CREATE INDEX IF NOT EXISTS idx_data_anomalies_severity ON data_anomalies (severity);
CREATE INDEX IF NOT EXISTS idx_data_anomalies_resolved ON data_anomalies (resolved);

-- Grant permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO alpha_emon;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO alpha_emon;

-- Create a function to update the updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create triggers for updated_at columns
CREATE TRIGGER update_pattern_detections_updated_at BEFORE UPDATE ON pattern_detections FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_signal_history_updated_at BEFORE UPDATE ON signal_history FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_performance_metrics_updated_at BEFORE UPDATE ON performance_metrics FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
