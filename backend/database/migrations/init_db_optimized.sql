-- Optimized Database Schema for AlphaPlus AI Trading System
-- TimescaleDB optimized for high-frequency trading data

-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Market Data Table (TimescaleDB hypertable for time-series data)
CREATE TABLE IF NOT EXISTS market_data (
    id BIGSERIAL,
    symbol VARCHAR(20) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    open DECIMAL(15,8) NOT NULL,
    high DECIMAL(15,8) NOT NULL,
    low DECIMAL(15,8) NOT NULL,
    close DECIMAL(15,8) NOT NULL,
    volume DECIMAL(20,8) NOT NULL,
    price_change DECIMAL(10,4),
    data_points INTEGER DEFAULT 1,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Convert to TimescaleDB hypertable
SELECT create_hypertable('market_data', 'timestamp', if_not_exists => TRUE);

-- Create optimized indexes for market data
CREATE INDEX IF NOT EXISTS idx_market_data_symbol_time ON market_data (symbol, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_market_data_time_symbol ON market_data (timestamp DESC, symbol);
CREATE INDEX IF NOT EXISTS idx_market_data_symbol_volume ON market_data (symbol, volume DESC) WHERE volume > 0;

-- Patterns Table (for storing detected patterns)
CREATE TABLE IF NOT EXISTS patterns (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    pattern_type VARCHAR(50) NOT NULL,
    direction VARCHAR(10) NOT NULL,
    confidence DECIMAL(5,4) NOT NULL,
    strength VARCHAR(20) DEFAULT 'medium',
    entry_price DECIMAL(15,8),
    stop_loss DECIMAL(15,8),
    take_profit DECIMAL(15,8),
    timestamp TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for patterns
CREATE INDEX IF NOT EXISTS idx_patterns_symbol_time ON patterns (symbol, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_patterns_confidence ON patterns (confidence DESC, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_patterns_type_confidence ON patterns (pattern_type, confidence DESC);
CREATE INDEX IF NOT EXISTS idx_patterns_high_confidence ON patterns (confidence DESC, timestamp DESC) WHERE confidence >= 0.8;

-- Optimize existing signals table
DROP TABLE IF EXISTS signals CASCADE;
CREATE TABLE IF NOT EXISTS signals (
    id BIGSERIAL PRIMARY KEY,
    signal_id VARCHAR(100) UNIQUE NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    direction VARCHAR(10) NOT NULL,
    confidence DECIMAL(5,4) NOT NULL,
    entry_price DECIMAL(15,8) NOT NULL,
    stop_loss DECIMAL(15,8),
    take_profit DECIMAL(15,8),
    pattern_type VARCHAR(50),
    risk_reward_ratio DECIMAL(10,4),
    timestamp TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Create optimized indexes for signals
CREATE INDEX IF NOT EXISTS idx_signals_symbol_time ON signals (symbol, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_signals_confidence ON signals (confidence DESC, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_signals_direction ON signals (direction, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_signals_high_confidence ON signals (confidence DESC, timestamp DESC) WHERE confidence >= 0.8;
CREATE INDEX IF NOT EXISTS idx_signals_pattern_type ON signals (pattern_type, timestamp DESC);

-- Optimize existing trades table
DROP TABLE IF EXISTS trades CASCADE;
CREATE TABLE IF NOT EXISTS trades (
    id BIGSERIAL PRIMARY KEY,
    signal_id VARCHAR(100) NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    side VARCHAR(10) NOT NULL,
    entry_price DECIMAL(15,8) NOT NULL,
    exit_price DECIMAL(15,8),
    quantity DECIMAL(15,8) NOT NULL,
    leverage INTEGER DEFAULT 1,
    pnl DECIMAL(15,8),
    pnl_percentage DECIMAL(10,4),
    strategy_name VARCHAR(100) NOT NULL,
    status VARCHAR(20) DEFAULT 'open',
    entry_time TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    exit_time TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for trades
CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades (symbol);
CREATE INDEX IF NOT EXISTS idx_trades_status ON trades (status);
CREATE INDEX IF NOT EXISTS idx_trades_entry_time ON trades (entry_time DESC);
CREATE INDEX IF NOT EXISTS idx_trades_signal_id ON trades (signal_id);
CREATE INDEX IF NOT EXISTS idx_trades_strategy ON trades (strategy_name, entry_time DESC);

-- Performance metrics table (optimized)
DROP TABLE IF EXISTS performance_metrics CASCADE;
CREATE TABLE IF NOT EXISTS performance_metrics (
    id BIGSERIAL PRIMARY KEY,
    var_95 DECIMAL(10,4),
    max_drawdown DECIMAL(10,4),
    sharpe_ratio DECIMAL(10,4),
    sortino_ratio DECIMAL(10,4),
    win_rate DECIMAL(5,4),
    total_trades INTEGER,
    profitable_trades INTEGER,
    timestamp TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for performance metrics
CREATE INDEX IF NOT EXISTS idx_performance_timestamp ON performance_metrics (timestamp DESC);

-- Market Regime Detection Table
CREATE TABLE IF NOT EXISTS market_regimes (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    regime_type VARCHAR(50) NOT NULL, -- 'trending', 'ranging', 'volatile', 'low_volatility'
    confidence DECIMAL(5,4) NOT NULL,
    volatility_level DECIMAL(10,4),
    trend_strength DECIMAL(10,4),
    timestamp TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for market regimes
CREATE INDEX IF NOT EXISTS idx_market_regimes_symbol_time ON market_regimes (symbol, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_market_regimes_type ON market_regimes (regime_type, timestamp DESC);

-- AI Model Performance Tracking
CREATE TABLE IF NOT EXISTS model_performance (
    id BIGSERIAL PRIMARY KEY,
    model_name VARCHAR(100) NOT NULL,
    accuracy DECIMAL(5,4),
    precision DECIMAL(5,4),
    recall DECIMAL(5,4),
    f1_score DECIMAL(5,4),
    total_predictions INTEGER,
    correct_predictions INTEGER,
    timestamp TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for model performance
CREATE INDEX IF NOT EXISTS idx_model_performance_name ON model_performance (model_name, timestamp DESC);

-- Insert sample data for testing
INSERT INTO market_data (symbol, timestamp, open, high, low, close, volume, price_change) VALUES
('BTC/USDT', CURRENT_TIMESTAMP - INTERVAL '1 hour', 50000.0, 50100.0, 49900.0, 50050.0, 1000.0, 0.1),
('ETH/USDT', CURRENT_TIMESTAMP - INTERVAL '1 hour', 3000.0, 3010.0, 2990.0, 3005.0, 500.0, 0.17),
('ADA/USDT', CURRENT_TIMESTAMP - INTERVAL '1 hour', 0.5, 0.51, 0.49, 0.505, 10000.0, 1.0),
('SOL/USDT', CURRENT_TIMESTAMP - INTERVAL '1 hour', 100.0, 101.0, 99.0, 100.5, 2000.0, 0.5);

INSERT INTO patterns (symbol, timeframe, pattern_type, direction, confidence, strength, entry_price, stop_loss, take_profit, timestamp) VALUES
('BTC/USDT', '1m', 'bullish_trend', 'long', 0.85, 'strong', 50050.0, 49050.0, 52550.0, CURRENT_TIMESTAMP - INTERVAL '30 minutes'),
('ETH/USDT', '1m', 'bearish_trend', 'short', 0.78, 'medium', 3005.0, 3065.0, 2855.0, CURRENT_TIMESTAMP - INTERVAL '20 minutes');

INSERT INTO signals (signal_id, symbol, timeframe, direction, confidence, entry_price, stop_loss, take_profit, pattern_type, risk_reward_ratio, timestamp) VALUES
('SIG-BTC-001', 'BTC/USDT', '1m', 'long', 0.85, 50050.0, 49050.0, 52550.0, 'bullish_trend', 2.5, CURRENT_TIMESTAMP - INTERVAL '25 minutes'),
('SIG-ETH-001', 'ETH/USDT', '1m', 'short', 0.78, 3005.0, 3065.0, 2855.0, 'bearish_trend', 2.0, CURRENT_TIMESTAMP - INTERVAL '15 minutes');

INSERT INTO trades (signal_id, symbol, side, entry_price, quantity, strategy_name, status, entry_time) VALUES
('SIG-BTC-001', 'BTC/USDT', 'long', 50050.0, 0.1, 'trend_following', 'open', CURRENT_TIMESTAMP - INTERVAL '20 minutes'),
('SIG-ETH-001', 'ETH/USDT', 'short', 3005.0, 1.0, 'mean_reversion', 'open', CURRENT_TIMESTAMP - INTERVAL '10 minutes');

INSERT INTO performance_metrics (var_95, max_drawdown, sharpe_ratio, sortino_ratio, win_rate, total_trades, profitable_trades) VALUES
(2.5, 8.2, 1.85, 2.1, 0.65, 100, 65);

INSERT INTO market_regimes (symbol, regime_type, confidence, volatility_level, trend_strength, timestamp) VALUES
('BTC/USDT', 'trending', 0.82, 0.15, 0.75, CURRENT_TIMESTAMP - INTERVAL '1 hour'),
('ETH/USDT', 'ranging', 0.78, 0.08, 0.25, CURRENT_TIMESTAMP - INTERVAL '1 hour');

INSERT INTO model_performance (model_name, accuracy, precision, recall, f1_score, total_predictions, correct_predictions) VALUES
('pattern_detector_v1', 0.78, 0.82, 0.75, 0.78, 1000, 780),
('signal_generator_v1', 0.72, 0.75, 0.70, 0.72, 500, 360);

-- Create compression policies for TimescaleDB
-- Compress market data older than 1 day
SELECT add_compression_policy('market_data', INTERVAL '1 day');

-- Create retention policies
-- Keep market data for 30 days
SELECT add_retention_policy('market_data', INTERVAL '30 days');

-- Create continuous aggregates for common queries
-- Hourly market data summary
CREATE MATERIALIZED VIEW IF NOT EXISTS market_data_hourly
WITH (timescaledb.continuous) AS
SELECT 
    symbol,
    time_bucket('1 hour', timestamp) AS bucket,
    AVG(close) as avg_price,
    MAX(high) as max_price,
    MIN(low) as min_price,
    SUM(volume) as total_volume,
    COUNT(*) as data_points
FROM market_data
GROUP BY symbol, bucket;

-- Daily market data summary
CREATE MATERIALIZED VIEW IF NOT EXISTS market_data_daily
WITH (timescaledb.continuous) AS
SELECT 
    symbol,
    time_bucket('1 day', timestamp) AS bucket,
    AVG(close) as avg_price,
    MAX(high) as max_price,
    MIN(low) as min_price,
    SUM(volume) as total_volume,
    COUNT(*) as data_points
FROM market_data
GROUP BY symbol, bucket;

-- Signal performance summary
CREATE MATERIALIZED VIEW IF NOT EXISTS signal_performance_daily
WITH (timescaledb.continuous) AS
SELECT 
    symbol,
    time_bucket('1 day', timestamp) AS bucket,
    COUNT(*) as total_signals,
    AVG(confidence) as avg_confidence,
    COUNT(CASE WHEN confidence >= 0.8 THEN 1 END) as high_confidence_signals,
    COUNT(CASE WHEN direction = 'long' THEN 1 END) as long_signals,
    COUNT(CASE WHEN direction = 'short' THEN 1 END) as short_signals
FROM signals
GROUP BY symbol, bucket;

-- Display table statistics
SELECT 
    schemaname,
    tablename,
    attname,
    n_distinct,
    correlation
FROM pg_stats 
WHERE tablename IN ('market_data', 'patterns', 'signals', 'trades')
ORDER BY tablename, attname;

-- Show created indexes
SELECT 
    tablename,
    indexname,
    indexdef
FROM pg_indexes 
WHERE tablename IN ('market_data', 'patterns', 'signals', 'trades')
ORDER BY tablename, indexname;

-- Display the created data
SELECT 'Market Data' as table_name, COUNT(*) as record_count FROM market_data
UNION ALL
SELECT 'Patterns' as table_name, COUNT(*) as record_count FROM patterns
UNION ALL
SELECT 'Signals' as table_name, COUNT(*) as record_count FROM signals
UNION ALL
SELECT 'Trades' as table_name, COUNT(*) as record_count FROM trades
UNION ALL
SELECT 'Performance Metrics' as table_name, COUNT(*) as record_count FROM performance_metrics
UNION ALL
SELECT 'Market Regimes' as table_name, COUNT(*) as record_count FROM market_regimes
UNION ALL
SELECT 'Model Performance' as table_name, COUNT(*) as record_count FROM model_performance;
