-- OHLCV Data Hypertable for AlphaPlus TimescaleDB
-- Creates proper hypertables without unique index conflicts

-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- OHLCV Data Table (TimescaleDB hypertable)
CREATE TABLE ohlcv_data (
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    open NUMERIC(20,8) NOT NULL,
    high NUMERIC(20,8) NOT NULL,
    low NUMERIC(20,8) NOT NULL,
    close NUMERIC(20,8) NOT NULL,
    volume NUMERIC(20,8) NOT NULL,
    quote_volume NUMERIC(20,8),
    trades_count INTEGER,
    taker_buy_base_volume NUMERIC(20,8),
    taker_buy_quote_volume NUMERIC(20,8),
    source TEXT DEFAULT 'websocket',
    data_quality_score NUMERIC(4,3) DEFAULT 1.0,
    created_at TIMESTAMPTZ DEFAULT NOW() NOT NULL,
    
    -- Constraints
    CONSTRAINT valid_ohlc CHECK (high >= low AND high >= open AND high >= close AND low <= open AND low <= close),
    CONSTRAINT positive_volume CHECK (volume >= 0),
    CONSTRAINT valid_timeframe CHECK (timeframe IN ('1m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w', '1M'))
);

-- Create TimescaleDB hypertable with 1-hour chunks
SELECT create_hypertable('ohlcv_data', 'timestamp', chunk_time_interval => INTERVAL '1 hour');

-- Create indexes for performance
CREATE INDEX idx_ohlcv_symbol_timeframe ON ohlcv_data (symbol, timeframe);
CREATE INDEX idx_ohlcv_timestamp ON ohlcv_data (timestamp);
CREATE INDEX idx_ohlcv_symbol_timestamp ON ohlcv_data (symbol, timestamp);
CREATE INDEX idx_ohlcv_symbol_timeframe_timestamp ON ohlcv_data (symbol, timeframe, timestamp);

-- Order Book Data Table (TimescaleDB hypertable)
CREATE TABLE order_book_data (
    symbol TEXT NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    bids JSONB NOT NULL,
    asks JSONB NOT NULL,
    best_bid NUMERIC(20,8),
    best_ask NUMERIC(20,8),
    spread NUMERIC(20,8),
    mid_price NUMERIC(20,8),
    source TEXT DEFAULT 'websocket',
    created_at TIMESTAMPTZ DEFAULT NOW() NOT NULL
);

-- Create TimescaleDB hypertable for order book data
SELECT create_hypertable('order_book_data', 'timestamp', chunk_time_interval => INTERVAL '1 hour');

-- Create indexes for order book data
CREATE INDEX idx_orderbook_symbol_timestamp ON order_book_data (symbol, timestamp);
CREATE INDEX idx_orderbook_timestamp ON order_book_data (timestamp);

-- Technical Indicators Table (TimescaleDB hypertable)
CREATE TABLE technical_indicators (
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    indicator_name TEXT NOT NULL,
    indicator_value NUMERIC(20,8) NOT NULL,
    indicator_params JSONB,
    calculation_method TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW() NOT NULL
);

-- Create TimescaleDB hypertable for indicators
SELECT create_hypertable('technical_indicators', 'timestamp', chunk_time_interval => INTERVAL '1 hour');

-- Create indexes for indicators
CREATE INDEX idx_indicators_symbol_timeframe ON technical_indicators (symbol, timeframe);
CREATE INDEX idx_indicators_name ON technical_indicators (indicator_name);
CREATE INDEX idx_indicators_symbol_name_timestamp ON technical_indicators (symbol, indicator_name, timestamp);

-- Support/Resistance Levels Table (regular table, not hypertable)
CREATE TABLE support_resistance_levels (
    id BIGSERIAL PRIMARY KEY,
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    level_type TEXT NOT NULL,
    price_level NUMERIC(20,8) NOT NULL,
    strength NUMERIC(4,3) NOT NULL,
    confidence NUMERIC(4,3) NOT NULL,
    touch_count INTEGER DEFAULT 0,
    first_touch_time TIMESTAMPTZ,
    last_touch_time TIMESTAMPTZ,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT NOW() NOT NULL,
    updated_at TIMESTAMPTZ DEFAULT NOW() NOT NULL
);

-- Create indexes for S/R levels
CREATE INDEX idx_sr_symbol_timeframe ON support_resistance_levels (symbol, timeframe);
CREATE INDEX idx_sr_level_type ON support_resistance_levels (level_type);
CREATE INDEX idx_sr_price_level ON support_resistance_levels (price_level);
CREATE INDEX idx_sr_active ON support_resistance_levels (is_active);

-- Compression Policy (compress data older than 1 month)
ALTER TABLE ohlcv_data SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'symbol'
);

SELECT add_compression_policy('ohlcv_data', INTERVAL '1 month');

-- Data Retention Policy (keep data for 1 year)
SELECT add_retention_policy('ohlcv_data', INTERVAL '1 year');

-- Function to get latest OHLCV data for signal generation
CREATE OR REPLACE FUNCTION get_latest_ohlcv_data(
    p_symbol TEXT,
    p_timeframe TEXT,
    p_periods INTEGER DEFAULT 100
)
RETURNS TABLE (
    timestamp TIMESTAMPTZ,
    open NUMERIC(20,8),
    high NUMERIC(20,8),
    low NUMERIC(20,8),
    close NUMERIC(20,8),
    volume NUMERIC(20,8)
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        o.timestamp,
        o.open,
        o.high,
        o.low,
        o.close,
        o.volume
    FROM ohlcv_data o
    WHERE o.symbol = p_symbol 
      AND o.timeframe = p_timeframe
    ORDER BY o.timestamp DESC
    LIMIT p_periods;
END;
$$ LANGUAGE plpgsql;

-- Function to calculate technical indicators and store them
CREATE OR REPLACE FUNCTION calculate_and_store_indicators(
    p_symbol TEXT,
    p_timeframe TEXT,
    p_periods INTEGER DEFAULT 100
)
RETURNS void AS $$
DECLARE
    sma_20 NUMERIC(20,8);
    sma_50 NUMERIC(20,8);
BEGIN
    -- Calculate SMA 20
    SELECT AVG(close) INTO sma_20
    FROM (
        SELECT close FROM ohlcv_data 
        WHERE symbol = p_symbol AND timeframe = p_timeframe
        ORDER BY timestamp DESC LIMIT 20
    ) t;
    
    -- Calculate SMA 50
    SELECT AVG(close) INTO sma_50
    FROM (
        SELECT close FROM ohlcv_data 
        WHERE symbol = p_symbol AND timeframe = p_timeframe
        ORDER BY timestamp DESC LIMIT 50
    ) t;
    
    -- Store indicators
    INSERT INTO technical_indicators (symbol, timeframe, timestamp, indicator_name, indicator_value, calculation_method)
    VALUES 
        (p_symbol, p_timeframe, NOW(), 'SMA_20', sma_20, 'simple_moving_average'),
        (p_symbol, p_timeframe, NOW(), 'SMA_50', sma_50, 'simple_moving_average')
    ON CONFLICT DO NOTHING;
END;
$$ LANGUAGE plpgsql;
