-- Multi-Timeframe Entry Fields Migration
-- Adds MTF entry refinement fields to signal tracking tables
-- Industry standard: Higher TF for trend, Lower TF for precise entry

-- Enable TimescaleDB extension if not already enabled
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Enhanced Signals Table for MTF Entry (Main signals storage)
CREATE TABLE IF NOT EXISTS ai_signals_mtf (
    id BIGSERIAL,
    signal_id UUID DEFAULT gen_random_uuid(),
    
    -- Basic Signal Info
    symbol TEXT NOT NULL,
    direction TEXT NOT NULL CHECK (direction IN ('LONG', 'SHORT', 'FLAT')),
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    -- Multi-Timeframe Configuration
    signal_timeframe TEXT NOT NULL, -- Higher TF for trend (1h, 4h, 1d)
    entry_timeframe TEXT NOT NULL,  -- Lower TF for entry (15m, 1h)
    
    -- Prices (from higher timeframe)
    signal_price NUMERIC(20,8) NOT NULL,
    
    -- Refined Entry (from lower timeframe)
    entry_price NUMERIC(20,8) NOT NULL,
    stop_loss NUMERIC(20,8),
    take_profit_1 NUMERIC(20,8),
    take_profit_2 NUMERIC(20,8),
    take_profit_3 NUMERIC(20,8),
    
    -- Entry Analysis
    entry_strategy TEXT, -- 'EMA_PULLBACK', 'FIBONACCI_RETRACEMENT', 'ORDER_BLOCK', 'BREAK_RETEST'
    entry_pattern TEXT,  -- Pattern detected on entry TF
    entry_confidence NUMERIC(4,3) CHECK (entry_confidence >= 0 AND entry_confidence <= 1),
    fibonacci_level NUMERIC(5,3), -- 0.382, 0.5, 0.618, etc.
    
    -- Signal Quality
    signal_confidence NUMERIC(4,3) NOT NULL CHECK (signal_confidence >= 0 AND signal_confidence <= 1),
    signal_probability NUMERIC(4,3) CHECK (signal_probability >= 0 AND signal_probability <= 1),
    consensus_achieved BOOLEAN NOT NULL,
    consensus_score NUMERIC(4,3),
    agreeing_heads_count INTEGER,
    
    -- Entry Timing Indicators
    atr_entry_tf NUMERIC(20,8),    -- ATR from entry timeframe
    atr_signal_tf NUMERIC(20,8),   -- ATR from signal timeframe
    volume_confirmation BOOLEAN DEFAULT FALSE,
    ema_alignment BOOLEAN DEFAULT FALSE,
    
    -- Market Context
    market_regime TEXT, -- 'TRENDING', 'RANGING', 'VOLATILE', 'BREAKOUT'
    higher_tf_trend TEXT, -- 'BULLISH', 'BEARISH', 'NEUTRAL'
    
    -- Model Head Analysis (JSON for detailed reasoning)
    model_heads_analysis JSONB,
    
    -- Risk Management
    risk_reward_ratio NUMERIC(6,3),
    position_size_suggested NUMERIC(20,8),
    
    -- Performance Tracking
    is_active BOOLEAN DEFAULT TRUE,
    status TEXT DEFAULT 'OPEN', -- 'OPEN', 'CLOSED', 'EXPIRED', 'CANCELLED'
    exit_price NUMERIC(20,8),
    exit_timestamp TIMESTAMPTZ,
    realized_pnl NUMERIC(20,8),
    outcome TEXT, -- 'TP1_HIT', 'TP2_HIT', 'TP3_HIT', 'SL_HIT', 'BREAKEVEN', 'MANUAL_EXIT'
    
    -- Data Quality
    data_quality_score NUMERIC(4,3),
    
    -- Metadata
    created_at TIMESTAMPTZ DEFAULT NOW() NOT NULL,
    updated_at TIMESTAMPTZ DEFAULT NOW() NOT NULL,
    
    -- Primary key includes timestamp for hypertable compatibility
    PRIMARY KEY (timestamp, signal_id),
    
    -- Indexes for performance
    CONSTRAINT valid_timeframes CHECK (
        signal_timeframe IN ('5m', '15m', '30m', '1h', '2h', '4h', '6h', '12h', '1d') AND
        entry_timeframe IN ('1m', '5m', '15m', '30m', '1h', '2h', '4h')
    ),
    CONSTRAINT entry_lower_than_signal CHECK (
        -- Ensure entry timeframe is lower than signal timeframe
        (signal_timeframe = '1d' AND entry_timeframe IN ('15m', '30m', '1h', '2h', '4h')) OR
        (signal_timeframe = '4h' AND entry_timeframe IN ('15m', '30m', '1h')) OR
        (signal_timeframe = '1h' AND entry_timeframe IN ('5m', '15m')) OR
        (signal_timeframe = '15m' AND entry_timeframe IN ('1m', '5m'))
    )
);

-- Create TimescaleDB hypertable for time-series analysis
SELECT create_hypertable('ai_signals_mtf', 'timestamp', chunk_time_interval => INTERVAL '1 day', if_not_exists => TRUE);

-- Create indexes for fast queries
CREATE INDEX IF NOT EXISTS idx_ai_signals_mtf_symbol_timestamp ON ai_signals_mtf (symbol, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_ai_signals_mtf_direction ON ai_signals_mtf (direction);
CREATE INDEX IF NOT EXISTS idx_ai_signals_mtf_timeframes ON ai_signals_mtf (signal_timeframe, entry_timeframe);
CREATE INDEX IF NOT EXISTS idx_ai_signals_mtf_status ON ai_signals_mtf (status, is_active);
CREATE INDEX IF NOT EXISTS idx_ai_signals_mtf_confidence ON ai_signals_mtf (signal_confidence DESC);
CREATE INDEX IF NOT EXISTS idx_ai_signals_mtf_entry_strategy ON ai_signals_mtf (entry_strategy);
CREATE INDEX IF NOT EXISTS idx_ai_signals_mtf_signal_id ON ai_signals_mtf (signal_id);

-- Entry Analysis History (Track all entry refinement attempts)
CREATE TABLE IF NOT EXISTS mtf_entry_analysis_history (
    id BIGSERIAL,
    analysis_id UUID DEFAULT gen_random_uuid(),
    signal_id UUID,
    
    symbol TEXT NOT NULL,
    entry_timeframe TEXT NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    -- Detected Entry Zones
    ema_9_level NUMERIC(20,8),
    ema_21_level NUMERIC(20,8),
    ema_50_level NUMERIC(20,8),
    fibonacci_382 NUMERIC(20,8),
    fibonacci_50 NUMERIC(20,8),
    fibonacci_618 NUMERIC(20,8),
    order_block_high NUMERIC(20,8),
    order_block_low NUMERIC(20,8),
    
    -- Selected Entry
    selected_entry_price NUMERIC(20,8) NOT NULL,
    selected_entry_reason TEXT,
    
    -- Entry Quality Metrics
    distance_from_current_percent NUMERIC(6,3),
    volume_at_entry NUMERIC(20,8),
    volume_vs_ma_ratio NUMERIC(6,3),
    
    -- Candlestick Pattern
    candlestick_pattern TEXT,
    pattern_confidence NUMERIC(4,3),
    
    created_at TIMESTAMPTZ DEFAULT NOW() NOT NULL,
    
    -- Primary key includes timestamp for hypertable compatibility
    PRIMARY KEY (timestamp, analysis_id)
);

-- Create TimescaleDB hypertable
SELECT create_hypertable('mtf_entry_analysis_history', 'timestamp', chunk_time_interval => INTERVAL '1 day', if_not_exists => TRUE);

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_mtf_entry_history_signal ON mtf_entry_analysis_history (signal_id);
CREATE INDEX IF NOT EXISTS idx_mtf_entry_history_symbol ON mtf_entry_analysis_history (symbol, timestamp DESC);

-- MTF Entry Performance Tracking (Aggregate statistics)
CREATE TABLE IF NOT EXISTS mtf_entry_performance (
    symbol TEXT NOT NULL,
    entry_strategy TEXT NOT NULL,
    signal_timeframe TEXT NOT NULL,
    entry_timeframe TEXT NOT NULL,
    
    -- Statistics
    total_signals INTEGER DEFAULT 0,
    winning_signals INTEGER DEFAULT 0,
    losing_signals INTEGER DEFAULT 0,
    win_rate NUMERIC(5,2),
    
    avg_entry_confidence NUMERIC(4,3),
    avg_slippage_percent NUMERIC(6,3),
    avg_time_to_entry_minutes INTEGER,
    
    -- Best/Worst
    best_rr_ratio NUMERIC(6,3),
    worst_rr_ratio NUMERIC(6,3),
    avg_rr_ratio NUMERIC(6,3),
    
    -- Updated
    last_updated TIMESTAMPTZ DEFAULT NOW(),
    
    PRIMARY KEY (symbol, entry_strategy, signal_timeframe, entry_timeframe)
);

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_mtf_performance_strategy ON mtf_entry_performance (entry_strategy);
CREATE INDEX IF NOT EXISTS idx_mtf_performance_win_rate ON mtf_entry_performance (win_rate DESC);

-- Function to update MTF entry performance
CREATE OR REPLACE FUNCTION update_mtf_entry_performance()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.status = 'CLOSED' AND NEW.outcome IS NOT NULL THEN
        INSERT INTO mtf_entry_performance (
            symbol, entry_strategy, signal_timeframe, entry_timeframe,
            total_signals, winning_signals, losing_signals, win_rate,
            avg_entry_confidence, avg_rr_ratio, last_updated
        )
        VALUES (
            NEW.symbol, 
            NEW.entry_strategy, 
            NEW.signal_timeframe, 
            NEW.entry_timeframe,
            1,
            CASE WHEN NEW.outcome LIKE 'TP%' THEN 1 ELSE 0 END,
            CASE WHEN NEW.outcome = 'SL_HIT' THEN 1 ELSE 0 END,
            CASE WHEN NEW.outcome LIKE 'TP%' THEN 100.0 ELSE 0.0 END,
            NEW.entry_confidence,
            NEW.risk_reward_ratio,
            NOW()
        )
        ON CONFLICT (symbol, entry_strategy, signal_timeframe, entry_timeframe)
        DO UPDATE SET
            total_signals = mtf_entry_performance.total_signals + 1,
            winning_signals = mtf_entry_performance.winning_signals + 
                CASE WHEN NEW.outcome LIKE 'TP%' THEN 1 ELSE 0 END,
            losing_signals = mtf_entry_performance.losing_signals + 
                CASE WHEN NEW.outcome = 'SL_HIT' THEN 1 ELSE 0 END,
            win_rate = (mtf_entry_performance.winning_signals + 
                CASE WHEN NEW.outcome LIKE 'TP%' THEN 1 ELSE 0 END) * 100.0 / 
                (mtf_entry_performance.total_signals + 1),
            avg_entry_confidence = (mtf_entry_performance.avg_entry_confidence * mtf_entry_performance.total_signals + NEW.entry_confidence) /
                (mtf_entry_performance.total_signals + 1),
            avg_rr_ratio = (mtf_entry_performance.avg_rr_ratio * mtf_entry_performance.total_signals + NEW.risk_reward_ratio) /
                (mtf_entry_performance.total_signals + 1),
            last_updated = NOW();
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger to auto-update performance
CREATE TRIGGER trigger_update_mtf_performance
AFTER UPDATE ON ai_signals_mtf
FOR EACH ROW
WHEN (NEW.status = 'CLOSED' AND OLD.status != 'CLOSED')
EXECUTE FUNCTION update_mtf_entry_performance();

-- Compression policy (compress data older than 7 days)
ALTER TABLE ai_signals_mtf SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'symbol, direction'
);

SELECT add_compression_policy('ai_signals_mtf', INTERVAL '7 days');

-- Data retention policy (keep signals for 1 year)
SELECT add_retention_policy('ai_signals_mtf', INTERVAL '1 year');

-- Continuous aggregate for daily performance
CREATE MATERIALIZED VIEW IF NOT EXISTS mtf_daily_performance
WITH (timescaledb.continuous) AS
SELECT 
    time_bucket('1 day', timestamp) AS day,
    symbol,
    entry_strategy,
    signal_timeframe,
    entry_timeframe,
    COUNT(*) as total_signals,
    COUNT(*) FILTER (WHERE outcome LIKE 'TP%') as winning_signals,
    COUNT(*) FILTER (WHERE outcome = 'SL_HIT') as losing_signals,
    AVG(entry_confidence) as avg_entry_confidence,
    AVG(risk_reward_ratio) as avg_rr_ratio,
    AVG(
        CASE WHEN outcome LIKE 'TP%' THEN 1.0 
             WHEN outcome = 'SL_HIT' THEN 0.0 
             ELSE NULL 
        END
    ) * 100 as win_rate_percent
FROM ai_signals_mtf
WHERE status = 'CLOSED'
GROUP BY day, symbol, entry_strategy, signal_timeframe, entry_timeframe;

-- Refresh policy for continuous aggregate (refresh every hour)
SELECT add_continuous_aggregate_policy('mtf_daily_performance',
    start_offset => INTERVAL '3 days',
    end_offset => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour');

-- Helpful view for active signals with MTF info
CREATE OR REPLACE VIEW active_mtf_signals AS
SELECT 
    signal_id,
    symbol,
    direction,
    signal_timeframe,
    entry_timeframe,
    entry_price,
    stop_loss,
    take_profit_1,
    take_profit_2,
    take_profit_3,
    entry_strategy,
    entry_pattern,
    entry_confidence,
    signal_confidence,
    risk_reward_ratio,
    market_regime,
    timestamp,
    EXTRACT(EPOCH FROM (NOW() - timestamp))/60 as minutes_since_signal
FROM ai_signals_mtf
WHERE is_active = TRUE AND status = 'OPEN'
ORDER BY timestamp DESC;

-- Helper function to get best entry strategy for a symbol
CREATE OR REPLACE FUNCTION get_best_entry_strategy(p_symbol TEXT, p_signal_tf TEXT DEFAULT '1h')
RETURNS TABLE (
    entry_strategy TEXT,
    entry_timeframe TEXT,
    win_rate NUMERIC(5,2),
    total_signals INTEGER
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        p.entry_strategy,
        p.entry_timeframe,
        p.win_rate,
        p.total_signals
    FROM mtf_entry_performance p
    WHERE p.symbol = p_symbol 
      AND p.signal_timeframe = p_signal_tf
      AND p.total_signals >= 10  -- Minimum sample size
    ORDER BY p.win_rate DESC, p.total_signals DESC
    LIMIT 1;
END;
$$ LANGUAGE plpgsql;

-- Grant permissions (adjust as needed)
-- GRANT ALL ON ai_signals_mtf TO alpha_emon;
-- GRANT ALL ON mtf_entry_analysis_history TO alpha_emon;
-- GRANT ALL ON mtf_entry_performance TO alpha_emon;

