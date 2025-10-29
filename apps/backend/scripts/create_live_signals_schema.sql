-- AlphaPulse Live Signals Database Schema
-- TimescaleDB optimized schema for real-time signal generation

-- ============================================================================
-- LIVE SIGNALS TABLE (Active signals with entry proximity)
-- ============================================================================
CREATE TABLE IF NOT EXISTS live_signals (
    id SERIAL PRIMARY KEY,
    signal_id VARCHAR(100) UNIQUE NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    direction VARCHAR(10) NOT NULL, -- 'long' or 'short'
    
    -- Pricing
    entry_price DECIMAL(20, 8) NOT NULL,
    current_price DECIMAL(20, 8) NOT NULL,
    stop_loss DECIMAL(20, 8) NOT NULL,
    take_profit DECIMAL(20, 8) NOT NULL,
    tp1 DECIMAL(20, 8),
    tp2 DECIMAL(20, 8),
    tp3 DECIMAL(20, 8),
    tp4 DECIMAL(20, 8),
    
    -- Confidence & Quality
    confidence DECIMAL(5, 4) NOT NULL,
    quality_score DECIMAL(5, 4) NOT NULL,
    pattern_type VARCHAR(100),
    
    -- Entry Proximity
    entry_proximity_pct DECIMAL(10, 6) NOT NULL, -- Percentage from current to entry
    entry_proximity_status VARCHAR(20) NOT NULL, -- 'imminent', 'soon', 'waiting', 'missed'
    time_to_entry_estimate INT, -- Seconds
    
    -- SDE Consensus (JSONB for flexibility)
    sde_consensus JSONB NOT NULL,
    agreeing_heads INT NOT NULL,
    consensus_score DECIMAL(5, 4) NOT NULL,
    
    -- MTF Analysis
    mtf_analysis JSONB NOT NULL,
    mtf_boost DECIMAL(5, 4) NOT NULL,
    base_confidence DECIMAL(5, 4) NOT NULL,
    
    -- Lifecycle
    status VARCHAR(20) NOT NULL DEFAULT 'active', -- 'pending', 'active', 'filled', 'invalid', 'expired'
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    activated_at TIMESTAMPTZ, -- When it became active (entry proximity met)
    last_validated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    invalidated_at TIMESTAMPTZ,
    invalidation_reason VARCHAR(200),
    
    -- Metadata
    indicators JSONB,
    market_regime VARCHAR(50),
    risk_reward_ratio DECIMAL(10, 4),
    
    CONSTRAINT valid_direction CHECK (direction IN ('long', 'short')),
    CONSTRAINT valid_status CHECK (status IN ('pending', 'active', 'filled', 'invalid', 'expired')),
    CONSTRAINT valid_proximity CHECK (entry_proximity_status IN ('imminent', 'soon', 'waiting', 'missed'))
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_live_signals_symbol ON live_signals(symbol);
CREATE INDEX IF NOT EXISTS idx_live_signals_status ON live_signals(status);
CREATE INDEX IF NOT EXISTS idx_live_signals_proximity ON live_signals(entry_proximity_status);
CREATE INDEX IF NOT EXISTS idx_live_signals_created ON live_signals(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_live_signals_active ON live_signals(symbol, status) WHERE status = 'active';

-- ============================================================================
-- SIGNAL HISTORY TABLE (All generated signals for ML training)
-- ============================================================================
CREATE TABLE IF NOT EXISTS signal_history (
    id SERIAL PRIMARY KEY,
    signal_id VARCHAR(100) UNIQUE NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    direction VARCHAR(10) NOT NULL,
    
    -- Pricing
    entry_price DECIMAL(20, 8) NOT NULL,
    stop_loss DECIMAL(20, 8) NOT NULL,
    take_profit DECIMAL(20, 8) NOT NULL,
    
    -- Signal Quality
    confidence DECIMAL(5, 4) NOT NULL,
    quality_score DECIMAL(5, 4),
    pattern_type VARCHAR(100),
    
    -- SDE & MTF Data
    sde_consensus JSONB,
    mtf_analysis JSONB,
    agreeing_heads INT,
    
    -- Technical Indicators (for ML)
    rsi DECIMAL(10, 4),
    macd DECIMAL(10, 6),
    volume_ratio DECIMAL(10, 4),
    indicators JSONB,
    
    -- Outcome (for ML learning)
    outcome VARCHAR(20) DEFAULT 'pending', -- 'win', 'loss', 'pending', 'breakeven'
    actual_entry_price DECIMAL(20, 8),
    actual_exit_price DECIMAL(20, 8),
    profit_loss_pct DECIMAL(10, 4),
    
    -- Lifecycle
    source VARCHAR(50) NOT NULL, -- 'backtest', 'live', 'manual'
    lifecycle_status VARCHAR(20) DEFAULT 'completed',
    
    -- Timestamps
    signal_timestamp TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMPTZ,
    
    CONSTRAINT valid_outcome CHECK (outcome IN ('win', 'loss', 'pending', 'breakeven', 'cancelled'))
);

-- Convert to hypertable for time-series optimization
SELECT create_hypertable('signal_history', 'signal_timestamp', 
    if_not_exists => TRUE,
    chunk_time_interval => INTERVAL '1 day'
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_signal_history_symbol ON signal_history(symbol);
CREATE INDEX IF NOT EXISTS idx_signal_history_timestamp ON signal_history(signal_timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_signal_history_outcome ON signal_history(outcome);
CREATE INDEX IF NOT EXISTS idx_signal_history_source ON signal_history(source);

-- ============================================================================
-- SIGNAL LIFECYCLE TABLE (Track state transitions)
-- ============================================================================
CREATE TABLE IF NOT EXISTS signal_lifecycle (
    id SERIAL PRIMARY KEY,
    signal_id VARCHAR(100) NOT NULL,
    from_status VARCHAR(20) NOT NULL,
    to_status VARCHAR(20) NOT NULL,
    reason VARCHAR(200),
    current_price DECIMAL(20, 8),
    entry_proximity_pct DECIMAL(10, 6),
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    FOREIGN KEY (signal_id) REFERENCES live_signals(signal_id) ON DELETE CASCADE
);

-- Index for tracking
CREATE INDEX IF NOT EXISTS idx_lifecycle_signal ON signal_lifecycle(signal_id);
CREATE INDEX IF NOT EXISTS idx_lifecycle_timestamp ON signal_lifecycle(timestamp DESC);

-- ============================================================================
-- CURRENT MARKET PRICES TABLE (Cache latest prices)
-- ============================================================================
CREATE TABLE IF NOT EXISTS current_market_prices (
    symbol VARCHAR(20) PRIMARY KEY,
    price DECIMAL(20, 8) NOT NULL,
    volume_24h DECIMAL(30, 8),
    change_24h DECIMAL(10, 4),
    last_updated TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ============================================================================
-- HELPER FUNCTIONS
-- ============================================================================

-- Function to update signal lifecycle
CREATE OR REPLACE FUNCTION update_signal_lifecycle()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.status != OLD.status THEN
        INSERT INTO signal_lifecycle (signal_id, from_status, to_status, reason, current_price, entry_proximity_pct)
        VALUES (NEW.signal_id, OLD.status, NEW.status, NEW.invalidation_reason, NEW.current_price, NEW.entry_proximity_pct);
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger for lifecycle tracking
DROP TRIGGER IF EXISTS track_signal_lifecycle ON live_signals;
CREATE TRIGGER track_signal_lifecycle
    AFTER UPDATE ON live_signals
    FOR EACH ROW
    WHEN (NEW.status IS DISTINCT FROM OLD.status)
    EXECUTE FUNCTION update_signal_lifecycle();

-- Function to auto-invalidate old signals
CREATE OR REPLACE FUNCTION auto_invalidate_old_signals()
RETURNS void AS $$
BEGIN
    UPDATE live_signals
    SET status = 'expired',
        invalidated_at = NOW(),
        invalidation_reason = 'Entry window timeout (30 min)'
    WHERE status IN ('pending', 'active')
      AND created_at < NOW() - INTERVAL '30 minutes';
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- VIEWS FOR QUICK ACCESS
-- ============================================================================

-- Active signals ready for display
CREATE OR REPLACE VIEW active_signals_view AS
SELECT 
    signal_id,
    symbol,
    direction,
    confidence,
    quality_score,
    entry_price,
    current_price,
    stop_loss,
    take_profit,
    entry_proximity_pct,
    entry_proximity_status,
    sde_consensus,
    mtf_analysis,
    agreeing_heads,
    pattern_type,
    timeframe,
    created_at
FROM live_signals
WHERE status = 'active'
  AND entry_proximity_status IN ('imminent', 'soon')
ORDER BY quality_score DESC, confidence DESC;

-- Signal performance for ML
CREATE OR REPLACE VIEW signal_performance_view AS
SELECT 
    symbol,
    direction,
    AVG(confidence) as avg_confidence,
    AVG(quality_score) as avg_quality,
    COUNT(*) FILTER (WHERE outcome = 'win') as wins,
    COUNT(*) FILTER (WHERE outcome = 'loss') as losses,
    COUNT(*) as total,
    AVG(profit_loss_pct) FILTER (WHERE outcome IN ('win', 'loss')) as avg_return
FROM signal_history
WHERE source = 'live'
  AND outcome IN ('win', 'loss')
GROUP BY symbol, direction;

-- ============================================================================
-- INITIAL DATA & CLEANUP
-- ============================================================================

-- Clean up any old test data
TRUNCATE TABLE live_signals CASCADE;
TRUNCATE TABLE signal_lifecycle CASCADE;

COMMENT ON TABLE live_signals IS 'Active trading signals with real-time entry proximity validation';
COMMENT ON TABLE signal_history IS 'Complete signal history for ML training and backtesting';
COMMENT ON TABLE signal_lifecycle IS 'Tracks signal state transitions for debugging and analysis';

