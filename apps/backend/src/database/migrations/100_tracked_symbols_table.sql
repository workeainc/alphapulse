-- Tracked Symbols Table for Dynamic Symbol Management
-- Stores top 100 Binance symbols (50 futures + 50 spot) with automatic daily updates

-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Tracked Symbols Table
CREATE TABLE IF NOT EXISTS tracked_symbols (
    symbol TEXT PRIMARY KEY,
    market_type TEXT NOT NULL CHECK (market_type IN ('futures', 'spot')),
    base_asset TEXT NOT NULL,  -- e.g., 'BTC', 'ETH'
    quote_asset TEXT NOT NULL,  -- e.g., 'USDT', 'USD'
    volume_24h NUMERIC(20,2) NOT NULL,
    volume_rank INTEGER NOT NULL,
    price_change_24h NUMERIC(10,4),
    last_price NUMERIC(20,8),
    is_active BOOLEAN DEFAULT true,
    last_updated TIMESTAMPTZ DEFAULT NOW(),
    added_at TIMESTAMPTZ DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_tracked_symbols_type ON tracked_symbols (market_type);
CREATE INDEX IF NOT EXISTS idx_tracked_symbols_active ON tracked_symbols (is_active) WHERE is_active = true;
CREATE INDEX IF NOT EXISTS idx_tracked_symbols_rank ON tracked_symbols (volume_rank);
CREATE INDEX IF NOT EXISTS idx_tracked_symbols_volume ON tracked_symbols (volume_24h DESC);

-- Symbol Update History Table (track changes over time)
CREATE TABLE IF NOT EXISTS tracked_symbols_history (
    id BIGSERIAL,
    symbol TEXT NOT NULL,
    market_type TEXT NOT NULL,
    volume_24h NUMERIC(20,2),
    volume_rank INTEGER,
    price_change_24h NUMERIC(10,4),
    snapshot_time TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Convert history to hypertable for efficient time-series queries
SELECT create_hypertable('tracked_symbols_history', 'snapshot_time', if_not_exists => TRUE);

-- Create index for history queries
CREATE INDEX IF NOT EXISTS idx_symbols_history_symbol_time ON tracked_symbols_history (symbol, snapshot_time DESC);

-- Function to get active symbols by type
CREATE OR REPLACE FUNCTION get_active_symbols(p_market_type TEXT DEFAULT NULL)
RETURNS TABLE (
    symbol TEXT,
    market_type TEXT,
    volume_24h NUMERIC,
    volume_rank INTEGER
) AS $$
BEGIN
    IF p_market_type IS NULL THEN
        RETURN QUERY
        SELECT ts.symbol, ts.market_type, ts.volume_24h, ts.volume_rank
        FROM tracked_symbols ts
        WHERE ts.is_active = true
        ORDER BY ts.volume_rank;
    ELSE
        RETURN QUERY
        SELECT ts.symbol, ts.market_type, ts.volume_24h, ts.volume_rank
        FROM tracked_symbols ts
        WHERE ts.is_active = true AND ts.market_type = p_market_type
        ORDER BY ts.volume_rank;
    END IF;
END;
$$ LANGUAGE plpgsql;

-- Function to archive symbol changes to history
CREATE OR REPLACE FUNCTION archive_symbol_snapshot()
RETURNS void AS $$
BEGIN
    INSERT INTO tracked_symbols_history (
        symbol, market_type, volume_24h, volume_rank, price_change_24h, snapshot_time
    )
    SELECT 
        symbol, market_type, volume_24h, volume_rank, price_change_24h, NOW()
    FROM tracked_symbols
    WHERE is_active = true;
END;
$$ LANGUAGE plpgsql;

-- Comments
COMMENT ON TABLE tracked_symbols IS 'Dynamically managed list of top 100 Binance symbols (50 futures + 50 spot)';
COMMENT ON TABLE tracked_symbols_history IS 'Historical snapshots of symbol rankings and volumes';
COMMENT ON FUNCTION get_active_symbols IS 'Get active symbols optionally filtered by market type';
COMMENT ON FUNCTION archive_symbol_snapshot IS 'Create historical snapshot of current symbol list';

