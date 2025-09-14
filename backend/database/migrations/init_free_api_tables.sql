-- Free API Data Integration Tables for AlphaPlus
-- Creates tables to store free API data for market analysis and signal generation

-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Free API Market Data Table (TimescaleDB hypertable)
CREATE TABLE IF NOT EXISTS free_api_market_data (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    source VARCHAR(50) NOT NULL, -- binance, coingecko, cryptocompare
    price NUMERIC(20,8) NOT NULL,
    volume_24h NUMERIC(20,8),
    market_cap NUMERIC(20,8),
    price_change_24h NUMERIC(10,6),
    volume_change_24h NUMERIC(10,6),
    market_cap_change_24h NUMERIC(10,6),
    fear_greed_index INTEGER,
    liquidation_events JSONB,
    raw_data JSONB,
    data_quality_score NUMERIC(4,3) DEFAULT 1.0,
    created_at TIMESTAMPTZ DEFAULT NOW() NOT NULL
);

-- Create TimescaleDB hypertable
SELECT create_hypertable('free_api_market_data', 'timestamp', chunk_time_interval => INTERVAL '1 hour');

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_free_api_market_data_symbol_source ON free_api_market_data (symbol, source);
CREATE INDEX IF NOT EXISTS idx_free_api_market_data_timestamp ON free_api_market_data (timestamp);
CREATE INDEX IF NOT EXISTS idx_free_api_market_data_symbol_timestamp ON free_api_market_data (symbol, timestamp);

-- Free API Sentiment Data Table (TimescaleDB hypertable)
CREATE TABLE IF NOT EXISTS free_api_sentiment_data (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    source VARCHAR(50) NOT NULL, -- newsapi, reddit, twitter, telegram, huggingface
    sentiment_type VARCHAR(20) NOT NULL, -- news, social, ai_model
    sentiment_score NUMERIC(4,3) NOT NULL, -- -1.0 to 1.0
    sentiment_label VARCHAR(20) NOT NULL, -- bullish, bearish, neutral
    confidence NUMERIC(4,3) NOT NULL,
    volume INTEGER, -- number of articles/posts analyzed
    keywords JSONB,
    raw_data JSONB,
    data_quality_score NUMERIC(4,3) DEFAULT 1.0,
    created_at TIMESTAMPTZ DEFAULT NOW() NOT NULL
);

-- Create TimescaleDB hypertable
SELECT create_hypertable('free_api_sentiment_data', 'timestamp', chunk_time_interval => INTERVAL '1 hour');

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_free_api_sentiment_data_symbol_source ON free_api_sentiment_data (symbol, source);
CREATE INDEX IF NOT EXISTS idx_free_api_sentiment_data_timestamp ON free_api_sentiment_data (timestamp);
CREATE INDEX IF NOT EXISTS idx_free_api_sentiment_data_sentiment_score ON free_api_sentiment_data (sentiment_score);
CREATE INDEX IF NOT EXISTS idx_free_api_sentiment_data_sentiment_type ON free_api_sentiment_data (sentiment_type);

-- Free API News Data Table (TimescaleDB hypertable)
CREATE TABLE IF NOT EXISTS free_api_news_data (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    source VARCHAR(50) NOT NULL, -- newsapi, reddit, cryptocompare
    title TEXT NOT NULL,
    content TEXT,
    url TEXT,
    published_at TIMESTAMPTZ,
    sentiment_score NUMERIC(4,3),
    sentiment_label VARCHAR(20),
    relevance_score NUMERIC(4,3),
    keywords JSONB,
    raw_data JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW() NOT NULL
);

-- Create TimescaleDB hypertable
SELECT create_hypertable('free_api_news_data', 'timestamp', chunk_time_interval => INTERVAL '1 day');

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_free_api_news_data_symbol_source ON free_api_news_data (symbol, source);
CREATE INDEX IF NOT EXISTS idx_free_api_news_data_timestamp ON free_api_news_data (timestamp);
CREATE INDEX IF NOT EXISTS idx_free_api_news_data_published_at ON free_api_news_data (published_at);
CREATE INDEX IF NOT EXISTS idx_free_api_news_data_relevance_score ON free_api_news_data (relevance_score);

-- Free API Social Media Data Table (TimescaleDB hypertable)
CREATE TABLE IF NOT EXISTS free_api_social_data (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    platform VARCHAR(20) NOT NULL, -- reddit, twitter, telegram
    post_id VARCHAR(100),
    content TEXT NOT NULL,
    author VARCHAR(100),
    engagement_metrics JSONB, -- likes, retweets, comments, etc.
    sentiment_score NUMERIC(4,3),
    sentiment_label VARCHAR(20),
    influence_score NUMERIC(4,3),
    keywords JSONB,
    raw_data JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW() NOT NULL
);

-- Create TimescaleDB hypertable
SELECT create_hypertable('free_api_social_data', 'timestamp', chunk_time_interval => INTERVAL '1 hour');

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_free_api_social_data_symbol_platform ON free_api_social_data (symbol, platform);
CREATE INDEX IF NOT EXISTS idx_free_api_social_data_timestamp ON free_api_social_data (timestamp);
CREATE INDEX IF NOT EXISTS idx_free_api_social_data_sentiment_score ON free_api_social_data (sentiment_score);
CREATE INDEX IF NOT EXISTS idx_free_api_social_data_influence_score ON free_api_social_data (influence_score);

-- Free API Liquidation Events Table (TimescaleDB hypertable)
CREATE TABLE IF NOT EXISTS free_api_liquidation_events (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    source VARCHAR(50) NOT NULL, -- binance, cryptocompare
    liquidation_type VARCHAR(20) NOT NULL, -- long, short
    price NUMERIC(20,8) NOT NULL,
    quantity NUMERIC(20,8) NOT NULL,
    value_usd NUMERIC(20,2) NOT NULL,
    side VARCHAR(10) NOT NULL, -- buy, sell
    raw_data JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW() NOT NULL
);

-- Create TimescaleDB hypertable
SELECT create_hypertable('free_api_liquidation_events', 'timestamp', chunk_time_interval => INTERVAL '1 hour');

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_free_api_liquidation_events_symbol ON free_api_liquidation_events (symbol);
CREATE INDEX IF NOT EXISTS idx_free_api_liquidation_events_timestamp ON free_api_liquidation_events (timestamp);
CREATE INDEX IF NOT EXISTS idx_free_api_liquidation_events_type ON free_api_liquidation_events (liquidation_type);
CREATE INDEX IF NOT EXISTS idx_free_api_liquidation_events_value_usd ON free_api_liquidation_events (value_usd);

-- Free API Data Quality Metrics Table
CREATE TABLE IF NOT EXISTS free_api_data_quality (
    id BIGSERIAL PRIMARY KEY,
    source VARCHAR(50) NOT NULL,
    data_type VARCHAR(20) NOT NULL, -- market, sentiment, news, social, liquidation
    timestamp TIMESTAMPTZ NOT NULL,
    availability_score NUMERIC(4,3) NOT NULL,
    accuracy_score NUMERIC(4,3) NOT NULL,
    completeness_score NUMERIC(4,3) NOT NULL,
    timeliness_score NUMERIC(4,3) NOT NULL,
    overall_score NUMERIC(4,3) NOT NULL,
    error_count INTEGER DEFAULT 0,
    success_count INTEGER DEFAULT 0,
    rate_limit_status VARCHAR(20), -- ok, warning, exceeded
    last_successful_fetch TIMESTAMPTZ,
    last_error_message TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW() NOT NULL
);

-- Create TimescaleDB hypertable
SELECT create_hypertable('free_api_data_quality', 'timestamp', chunk_time_interval => INTERVAL '1 hour');

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_free_api_data_quality_source_type ON free_api_data_quality (source, data_type);
CREATE INDEX IF NOT EXISTS idx_free_api_data_quality_timestamp ON free_api_data_quality (timestamp);
CREATE INDEX IF NOT EXISTS idx_free_api_data_quality_overall_score ON free_api_data_quality (overall_score);

-- Free API Rate Limit Tracking Table
CREATE TABLE IF NOT EXISTS free_api_rate_limits (
    id BIGSERIAL PRIMARY KEY,
    source VARCHAR(50) NOT NULL,
    endpoint VARCHAR(100) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    requests_made INTEGER NOT NULL,
    requests_remaining INTEGER NOT NULL,
    requests_reset_at TIMESTAMPTZ,
    daily_limit INTEGER,
    hourly_limit INTEGER,
    minute_limit INTEGER,
    status VARCHAR(20) NOT NULL, -- ok, warning, exceeded
    created_at TIMESTAMPTZ DEFAULT NOW() NOT NULL
);

-- Create TimescaleDB hypertable
SELECT create_hypertable('free_api_rate_limits', 'timestamp', chunk_time_interval => INTERVAL '1 hour');

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_free_api_rate_limits_source_endpoint ON free_api_rate_limits (source, endpoint);
CREATE INDEX IF NOT EXISTS idx_free_api_rate_limits_timestamp ON free_api_rate_limits (timestamp);
CREATE INDEX IF NOT EXISTS idx_free_api_rate_limits_status ON free_api_rate_limits (status);

-- Grant permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO alpha_emon;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO alpha_emon;

-- Create materialized view for aggregated sentiment data
CREATE MATERIALIZED VIEW IF NOT EXISTS aggregated_sentiment_data AS
SELECT 
    symbol,
    DATE_TRUNC('hour', timestamp) as hour_bucket,
    sentiment_type,
    AVG(sentiment_score) as avg_sentiment_score,
    COUNT(*) as sentiment_count,
    AVG(confidence) as avg_confidence,
    MAX(timestamp) as last_updated
FROM free_api_sentiment_data
WHERE timestamp >= NOW() - INTERVAL '7 days'
GROUP BY symbol, hour_bucket, sentiment_type
ORDER BY symbol, hour_bucket DESC;

-- Create index on materialized view
CREATE INDEX IF NOT EXISTS idx_aggregated_sentiment_data_symbol_hour ON aggregated_sentiment_data (symbol, hour_bucket);

-- Create materialized view for aggregated market data
CREATE MATERIALIZED VIEW IF NOT EXISTS aggregated_market_data AS
SELECT 
    symbol,
    DATE_TRUNC('hour', timestamp) as hour_bucket,
    source,
    AVG(price) as avg_price,
    AVG(volume_24h) as avg_volume_24h,
    AVG(market_cap) as avg_market_cap,
    AVG(price_change_24h) as avg_price_change_24h,
    AVG(fear_greed_index) as avg_fear_greed_index,
    COUNT(*) as data_points,
    MAX(timestamp) as last_updated
FROM free_api_market_data
WHERE timestamp >= NOW() - INTERVAL '7 days'
GROUP BY symbol, hour_bucket, source
ORDER BY symbol, hour_bucket DESC;

-- Create index on materialized view
CREATE INDEX IF NOT EXISTS idx_aggregated_market_data_symbol_hour ON aggregated_market_data (symbol, hour_bucket);

-- Create function to refresh materialized views
CREATE OR REPLACE FUNCTION refresh_free_api_views()
RETURNS void AS $$
BEGIN
    REFRESH MATERIALIZED VIEW aggregated_sentiment_data;
    REFRESH MATERIALIZED VIEW aggregated_market_data;
END;
$$ LANGUAGE plpgsql;

-- Create function to clean old data (keep last 30 days)
CREATE OR REPLACE FUNCTION cleanup_old_free_api_data()
RETURNS void AS $$
BEGIN
    DELETE FROM free_api_market_data WHERE timestamp < NOW() - INTERVAL '30 days';
    DELETE FROM free_api_sentiment_data WHERE timestamp < NOW() - INTERVAL '30 days';
    DELETE FROM free_api_news_data WHERE timestamp < NOW() - INTERVAL '30 days';
    DELETE FROM free_api_social_data WHERE timestamp < NOW() - INTERVAL '30 days';
    DELETE FROM free_api_liquidation_events WHERE timestamp < NOW() - INTERVAL '30 days';
    DELETE FROM free_api_data_quality WHERE timestamp < NOW() - INTERVAL '30 days';
    DELETE FROM free_api_rate_limits WHERE timestamp < NOW() - INTERVAL '7 days';
END;
$$ LANGUAGE plpgsql;
