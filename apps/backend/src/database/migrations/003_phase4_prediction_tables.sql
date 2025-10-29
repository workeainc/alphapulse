-- ===== PHASE 4: PREDICTION & CROSS-ASSET CORRELATION TABLES =====
-- Migration: 003_phase4_prediction_tables.sql
-- Description: Add tables for predictive analytics and cross-asset correlation
-- Date: August 21, 2025

-- 1. Sentiment Predictions Table
CREATE TABLE IF NOT EXISTS sentiment_predictions (
    id SERIAL,
    symbol VARCHAR(20) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    time_horizon VARCHAR(10) NOT NULL, -- '1h', '4h', '1d', '1w'
    prediction_probability FLOAT NOT NULL, -- 0.0 to 1.0
    direction VARCHAR(10) NOT NULL, -- 'bullish', 'bearish', 'neutral'
    strength VARCHAR(10) NOT NULL, -- 'strong', 'moderate', 'weak'
    confidence FLOAT NOT NULL, -- 0.0 to 1.0
    sentiment_score FLOAT NOT NULL, -- -1.0 to +1.0
    technical_indicators JSONB, -- RSI, MACD, volume, etc.
    factors JSONB, -- Contribution breakdown
    created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (timestamp, id)
);

-- Create hypertable for time-series optimization
SELECT create_hypertable('sentiment_predictions', 'timestamp', if_not_exists => TRUE);

-- Indexes for fast querying
CREATE INDEX IF NOT EXISTS idx_sentiment_predictions_symbol_time 
ON sentiment_predictions (symbol, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_sentiment_predictions_direction 
ON sentiment_predictions (direction, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_sentiment_predictions_probability 
ON sentiment_predictions (prediction_probability, timestamp DESC);

-- 2. Cross-Asset Sentiment Table
CREATE TABLE IF NOT EXISTS cross_asset_sentiment (
    id SERIAL,
    primary_symbol VARCHAR(20) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    correlations JSONB NOT NULL, -- Correlation matrix between assets
    market_sentiment JSONB NOT NULL, -- Overall market sentiment metrics
    created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (timestamp, id)
);

-- Create hypertable for time-series optimization
SELECT create_hypertable('cross_asset_sentiment', 'timestamp', if_not_exists => TRUE);

-- Indexes for cross-asset analysis
CREATE INDEX IF NOT EXISTS idx_cross_asset_primary_symbol 
ON cross_asset_sentiment (primary_symbol, timestamp DESC);

-- 3. Model Performance Metrics Table
CREATE TABLE IF NOT EXISTS model_performance_metrics (
    id SERIAL,
    symbol VARCHAR(20) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    accuracy FLOAT NOT NULL, -- 0.0 to 1.0
    predicted_probability FLOAT NOT NULL, -- 0.0 to 1.0
    actual_price_change FLOAT NOT NULL, -- Actual price change percentage
    model_version VARCHAR(20) NOT NULL, -- Model version for tracking
    created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (timestamp, id)
);

-- Create hypertable for time-series optimization
SELECT create_hypertable('model_performance_metrics', 'timestamp', if_not_exists => TRUE);

-- Indexes for performance tracking
CREATE INDEX IF NOT EXISTS idx_model_performance_symbol 
ON model_performance_metrics (symbol, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_model_performance_accuracy 
ON model_performance_metrics (accuracy, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_model_performance_version 
ON model_performance_metrics (model_version, timestamp DESC);

-- 4. Enhanced Sentiment Data Table Extensions
-- Add new columns to existing enhanced_sentiment_data table

-- Add prediction confidence column
ALTER TABLE enhanced_sentiment_data 
ADD COLUMN IF NOT EXISTS prediction_confidence FLOAT DEFAULT 0.0;

-- Add cross-asset correlation column
ALTER TABLE enhanced_sentiment_data 
ADD COLUMN IF NOT EXISTS cross_asset_correlation JSONB DEFAULT '{}';

-- Add macro indicators column
ALTER TABLE enhanced_sentiment_data 
ADD COLUMN IF NOT EXISTS macro_indicators JSONB DEFAULT '{}';

-- Add model version tracking
ALTER TABLE enhanced_sentiment_data 
ADD COLUMN IF NOT EXISTS model_version VARCHAR(20) DEFAULT 'v1.0';

-- Add retraining date
ALTER TABLE enhanced_sentiment_data 
ADD COLUMN IF NOT EXISTS retraining_date TIMESTAMPTZ DEFAULT NOW();

-- 5. Views for Easy Querying

-- Latest Predictions View
CREATE OR REPLACE VIEW latest_sentiment_predictions AS
SELECT DISTINCT ON (symbol) 
    symbol,
    timestamp,
    time_horizon,
    prediction_probability,
    direction,
    strength,
    confidence,
    sentiment_score,
    technical_indicators,
    factors
FROM sentiment_predictions 
ORDER BY symbol, timestamp DESC;

-- Model Performance Summary View
CREATE OR REPLACE VIEW model_performance_summary AS
SELECT 
    symbol,
    model_version,
    AVG(accuracy) as average_accuracy,
    COUNT(*) as total_predictions,
    AVG(predicted_probability) as average_predicted_probability,
    AVG(actual_price_change) as average_actual_change,
    MAX(timestamp) as last_updated
FROM model_performance_metrics 
GROUP BY symbol, model_version
ORDER BY symbol, last_updated DESC;

-- Cross-Asset Correlation Summary View
CREATE OR REPLACE VIEW cross_asset_correlation_summary AS
SELECT DISTINCT ON (primary_symbol) 
    primary_symbol,
    timestamp,
    correlations,
    market_sentiment
FROM cross_asset_sentiment 
ORDER BY primary_symbol, timestamp DESC;

-- 6. Continuous Aggregation Policies (TimescaleDB)

-- Create continuous aggregation for hourly prediction accuracy
CREATE MATERIALIZED VIEW IF NOT EXISTS hourly_prediction_accuracy
WITH (timescaledb.continuous) AS
SELECT 
    time_bucket('1 hour', timestamp) AS bucket,
    symbol,
    AVG(accuracy) as avg_accuracy,
    COUNT(*) as prediction_count,
    AVG(predicted_probability) as avg_predicted_prob,
    AVG(actual_price_change) as avg_actual_change
FROM model_performance_metrics 
GROUP BY bucket, symbol;

-- Create continuous aggregation for daily sentiment predictions
CREATE MATERIALIZED VIEW IF NOT EXISTS daily_sentiment_predictions
WITH (timescaledb.continuous) AS
SELECT 
    time_bucket('1 day', timestamp) AS bucket,
    symbol,
    AVG(prediction_probability) as avg_prediction_probability,
    COUNT(*) as prediction_count,
    MODE() WITHIN GROUP (ORDER BY direction) as most_common_direction,
    AVG(confidence) as avg_confidence
FROM sentiment_predictions 
GROUP BY bucket, symbol;

-- 7. Retention Policies

-- Set retention policy for predictions (keep 90 days)
SELECT add_retention_policy('sentiment_predictions', INTERVAL '90 days');

-- Set retention policy for cross-asset data (keep 60 days)
SELECT add_retention_policy('cross_asset_sentiment', INTERVAL '60 days');

-- Set retention policy for model performance (keep 180 days)
SELECT add_retention_policy('model_performance_metrics', INTERVAL '180 days');

-- 8. Compression Policies

-- Enable compression for predictions after 7 days
SELECT add_compression_policy('sentiment_predictions', INTERVAL '7 days');

-- Enable compression for cross-asset data after 7 days
SELECT add_compression_policy('cross_asset_sentiment', INTERVAL '7 days');

-- Enable compression for model performance after 30 days
SELECT add_compression_policy('model_performance_metrics', INTERVAL '30 days');

-- 9. Sample Data Insertion (for testing)

-- Insert sample prediction data
INSERT INTO sentiment_predictions (
    symbol, timestamp, time_horizon, prediction_probability, 
    direction, strength, confidence, sentiment_score, 
    technical_indicators, factors
) VALUES 
('BTC/USDT', NOW() - INTERVAL '1 hour', '4h', 0.75, 'bullish', 'moderate', 0.8, 0.6, 
 '{"rsi": 55.0, "macd": 0.02, "volume": 1.2}', 
 '{"sentiment_contribution": 0.24, "technical_contribution": 0.1, "volume_contribution": 0.04, "confidence_contribution": 0.06}'),
('ETH/USDT', NOW() - INTERVAL '30 minutes', '4h', 0.35, 'bearish', 'weak', 0.6, -0.2, 
 '{"rsi": 45.0, "macd": -0.01, "volume": 0.8}', 
 '{"sentiment_contribution": -0.08, "technical_contribution": -0.1, "volume_contribution": -0.04, "confidence_contribution": 0.02}');

-- Insert sample model performance data
INSERT INTO model_performance_metrics (
    symbol, timestamp, accuracy, predicted_probability, actual_price_change, model_version
) VALUES 
('BTC/USDT', NOW() - INTERVAL '2 hours', 1.0, 0.75, 2.5, 'v1.0'),
('ETH/USDT', NOW() - INTERVAL '1 hour', 0.0, 0.35, -1.2, 'v1.0');

-- Insert sample cross-asset data
INSERT INTO cross_asset_sentiment (
    primary_symbol, timestamp, correlations, market_sentiment
) VALUES 
('BTC/USDT', NOW() - INTERVAL '30 minutes', 
 '{"BTC/USDT_vs_ETH/USDT": 0.8, "BTC/USDT_vs_BNB/USDT": 0.7}', 
 '{"average_sentiment": 0.3, "average_confidence": 0.75, "market_mood": "bullish", "sentiment_volatility": 0.2, "asset_count": 3}');

-- 10. Grant Permissions (if needed)
-- GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO alpha_emon;
-- GRANT USAGE ON ALL SEQUENCES IN SCHEMA public TO alpha_emon;

-- Migration completed successfully
SELECT 'Phase 4 Prediction Tables Migration Completed Successfully' as status;
