-- Migration: Create enhanced_signals table for signal deduplication
-- This migration ensures the table has proper structure for deduplication logic

-- Create enhanced_signals table if it doesn't exist
CREATE TABLE IF NOT EXISTS enhanced_signals (
    id VARCHAR(255) PRIMARY KEY,
    symbol VARCHAR(50) NOT NULL,
    side VARCHAR(10) NOT NULL,  -- 'buy' or 'sell'
    strategy VARCHAR(100) NOT NULL,
    confidence DECIMAL(5,4) NOT NULL,
    strength DECIMAL(5,4) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    price DECIMAL(20,8) NOT NULL,
    stop_loss DECIMAL(20,8),
    take_profit DECIMAL(20,8),
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create indexes for efficient deduplication queries
CREATE INDEX IF NOT EXISTS idx_enhanced_signals_symbol_side ON enhanced_signals(symbol, side);
CREATE INDEX IF NOT EXISTS idx_enhanced_signals_timestamp ON enhanced_signals(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_enhanced_signals_confidence ON enhanced_signals(confidence DESC);
CREATE INDEX IF NOT EXISTS idx_enhanced_signals_strategy ON enhanced_signals(strategy);

-- Create unique constraint for deduplication (symbol + side + timestamp within 5 minutes)
-- This prevents duplicate signals for the same symbol/side within a short time window
CREATE UNIQUE INDEX IF NOT EXISTS idx_enhanced_signals_deduplication 
ON enhanced_signals(symbol, side, (timestamp::date), (EXTRACT(hour FROM timestamp) * 60 + EXTRACT(minute FROM timestamp)) / 5);

-- Convert to TimescaleDB hypertable for time-series optimization
SELECT create_hypertable('enhanced_signals', 'timestamp', if_not_exists => TRUE);

-- Create retention policy (keep data for 30 days)
SELECT add_retention_policy('enhanced_signals', INTERVAL '30 days', if_not_exists => TRUE);

-- Create compression policy (compress data older than 1 day)
SELECT add_compression_policy('enhanced_signals', INTERVAL '1 day', if_not_exists => TRUE);

-- Add comments for documentation
COMMENT ON TABLE enhanced_signals IS 'Enhanced signals table with deduplication support for AlphaPulse';
COMMENT ON COLUMN enhanced_signals.id IS 'Unique signal identifier';
COMMENT ON COLUMN enhanced_signals.symbol IS 'Trading symbol (e.g., BTCUSDT)';
COMMENT ON COLUMN enhanced_signals.side IS 'Signal direction (buy/sell)';
COMMENT ON COLUMN enhanced_signals.confidence IS 'Signal confidence score (0.0-1.0)';
COMMENT ON COLUMN enhanced_signals.metadata IS 'Additional signal metadata in JSON format';
