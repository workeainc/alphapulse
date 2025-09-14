-- Intelligent Data Collection Tables for AlphaPulse
-- Enhanced data collection with TimescaleDB integration

-- Market Intelligence Table (for BTC dominance, Total2/Total3, market sentiment)
CREATE TABLE IF NOT EXISTS market_intelligence (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    btc_dominance NUMERIC(10,4),
    total2_value NUMERIC(20,8),
    total3_value NUMERIC(20,8),
    market_sentiment_score NUMERIC(4,3),
    news_sentiment_score NUMERIC(4,3),
    volume_positioning_score NUMERIC(4,3),
    fear_greed_index INTEGER,
    market_regime VARCHAR(50), -- 'bullish', 'bearish', 'sideways', 'volatile'
    volatility_index NUMERIC(6,4),
    trend_strength NUMERIC(4,3),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create TimescaleDB hypertable for market_intelligence
SELECT create_hypertable('market_intelligence', 'timestamp', if_not_exists => TRUE);

-- Comprehensive Analysis Table (for intelligent signal generation)
CREATE TABLE IF NOT EXISTS comprehensive_analysis (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    
    -- Pattern Analysis
    pattern_confidence NUMERIC(4,3),
    pattern_type VARCHAR(100),
    pattern_strength VARCHAR(20),
    
    -- Technical Analysis
    technical_confidence NUMERIC(4,3),
    rsi_value NUMERIC(6,3),
    macd_signal VARCHAR(20),
    bollinger_position VARCHAR(20),
    support_level NUMERIC(20,8),
    resistance_level NUMERIC(20,8),
    
    -- Sentiment Analysis
    sentiment_confidence NUMERIC(4,3),
    news_sentiment NUMERIC(4,3),
    social_sentiment NUMERIC(4,3),
    market_sentiment NUMERIC(4,3),
    
    -- Volume Analysis
    volume_confidence NUMERIC(4,3),
    volume_ratio NUMERIC(6,3),
    volume_positioning VARCHAR(20),
    order_book_imbalance NUMERIC(6,3),
    
    -- Market Regime Analysis
    market_regime_confidence NUMERIC(4,3),
    market_regime VARCHAR(50),
    volatility_level VARCHAR(20),
    trend_direction VARCHAR(20),
    
    -- Overall Assessment
    overall_confidence NUMERIC(4,3),
    risk_reward_ratio NUMERIC(6,3),
    safe_entry_detected BOOLEAN,
    
    -- Entry/Exit Levels (only if safe entry detected)
    entry_price NUMERIC(20,8),
    stop_loss NUMERIC(20,8),
    take_profit_1 NUMERIC(20,8), -- 50% of position
    take_profit_2 NUMERIC(20,8), -- 25% of position
    take_profit_3 NUMERIC(20,8), -- 15% of position
    take_profit_4 NUMERIC(20,8), -- 10% of position
    position_size_percentage NUMERIC(5,2),
    
    -- Analysis Details
    analysis_reasoning TEXT,
    no_safe_entry_reasons TEXT[],
    signal_direction VARCHAR(10), -- 'long', 'short', 'neutral'
    signal_strength VARCHAR(20), -- 'weak', 'moderate', 'strong', 'very_strong'
    
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create TimescaleDB hypertable for comprehensive_analysis
SELECT create_hypertable('comprehensive_analysis', 'timestamp', if_not_exists => TRUE);

-- Intelligent Signals Table (final output)
CREATE TABLE IF NOT EXISTS intelligent_signals (
    id SERIAL PRIMARY KEY,
    signal_id VARCHAR(50) UNIQUE NOT NULL,
    analysis_id INTEGER REFERENCES comprehensive_analysis(id),
    symbol VARCHAR(20) NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    
    -- Signal Type
    signal_type VARCHAR(20) NOT NULL, -- 'entry', 'no_safe_entry', 'exit'
    signal_direction VARCHAR(10), -- 'long', 'short', 'neutral'
    signal_strength VARCHAR(20),
    
    -- Confidence and Risk
    confidence_score NUMERIC(4,3),
    risk_reward_ratio NUMERIC(6,3),
    risk_level VARCHAR(20), -- 'low', 'medium', 'high'
    
    -- Entry/Exit Levels
    entry_price NUMERIC(20,8),
    stop_loss NUMERIC(20,8),
    take_profit_1 NUMERIC(20,8),
    take_profit_2 NUMERIC(20,8),
    take_profit_3 NUMERIC(20,8),
    take_profit_4 NUMERIC(20,8),
    position_size_percentage NUMERIC(5,2),
    
    -- Analysis Summary
    pattern_analysis TEXT,
    technical_analysis TEXT,
    sentiment_analysis TEXT,
    volume_analysis TEXT,
    market_regime_analysis TEXT,
    
    -- Reasoning
    entry_reasoning TEXT,
    no_safe_entry_reasons TEXT[],
    
    -- Status
    status VARCHAR(20) DEFAULT 'generated', -- 'generated', 'active', 'completed', 'cancelled'
    pnl NUMERIC(20,8),
    executed_at TIMESTAMPTZ,
    closed_at TIMESTAMPTZ,
    
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create TimescaleDB hypertable for intelligent_signals
SELECT create_hypertable('intelligent_signals', 'timestamp', if_not_exists => TRUE);

-- Market Data Quality Metrics
CREATE TABLE IF NOT EXISTS data_quality_metrics_intelligent (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    data_source VARCHAR(50) NOT NULL,
    symbol VARCHAR(20),
    completeness NUMERIC(4,3),
    accuracy NUMERIC(4,3),
    consistency NUMERIC(4,3),
    timeliness NUMERIC(4,3),
    validity NUMERIC(4,3),
    overall_score NUMERIC(4,3),
    issues_detected TEXT[],
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create TimescaleDB hypertable for data quality metrics
SELECT create_hypertable('data_quality_metrics_intelligent', 'timestamp', if_not_exists => TRUE);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_market_intelligence_timestamp ON market_intelligence (timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_comprehensive_analysis_symbol_timestamp ON comprehensive_analysis (symbol, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_comprehensive_analysis_confidence ON comprehensive_analysis (overall_confidence DESC);
CREATE INDEX IF NOT EXISTS idx_intelligent_signals_symbol_timestamp ON intelligent_signals (symbol, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_intelligent_signals_confidence ON intelligent_signals (confidence_score DESC);
CREATE INDEX IF NOT EXISTS idx_intelligent_signals_status ON intelligent_signals (status);

-- Compression policies for TimescaleDB
SELECT add_compression_policy('market_intelligence', INTERVAL '7 days');
SELECT add_compression_policy('comprehensive_analysis', INTERVAL '7 days');
SELECT add_compression_policy('intelligent_signals', INTERVAL '7 days');
SELECT add_compression_policy('data_quality_metrics_intelligent', INTERVAL '7 days');

-- Retention policies (keep data for 90 days)
SELECT add_retention_policy('market_intelligence', INTERVAL '90 days');
SELECT add_retention_policy('comprehensive_analysis', INTERVAL '90 days');
SELECT add_retention_policy('intelligent_signals', INTERVAL '90 days');
SELECT add_retention_policy('data_quality_metrics_intelligent', INTERVAL '90 days');

-- Grant permissions
GRANT ALL PRIVILEGES ON TABLE market_intelligence TO alpha_emon;
GRANT ALL PRIVILEGES ON TABLE comprehensive_analysis TO alpha_emon;
GRANT ALL PRIVILEGES ON TABLE intelligent_signals TO alpha_emon;
GRANT ALL PRIVILEGES ON TABLE data_quality_metrics_intelligent TO alpha_emon;

GRANT ALL PRIVILEGES ON SEQUENCE market_intelligence_id_seq TO alpha_emon;
GRANT ALL PRIVILEGES ON SEQUENCE comprehensive_analysis_id_seq TO alpha_emon;
GRANT ALL PRIVILEGES ON SEQUENCE intelligent_signals_id_seq TO alpha_emon;
GRANT ALL PRIVILEGES ON SEQUENCE data_quality_metrics_intelligent_id_seq TO alpha_emon;

-- Insert sample market intelligence data
INSERT INTO market_intelligence (timestamp, btc_dominance, total2_value, total3_value, market_sentiment_score, fear_greed_index, market_regime, volatility_index, trend_strength) VALUES
(NOW(), 45.23, 1234567890.12345678, 9876543210.87654321, 0.65, 45, 'sideways', 0.0234, 0.45),
(NOW() - INTERVAL '1 hour', 44.87, 1234567891.12345678, 9876543211.87654321, 0.62, 42, 'sideways', 0.0245, 0.43),
(NOW() - INTERVAL '2 hours', 45.01, 1234567892.12345678, 9876543212.87654321, 0.68, 48, 'bullish', 0.0223, 0.52);

COMMIT;
