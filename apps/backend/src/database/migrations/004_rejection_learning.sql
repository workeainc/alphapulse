-- Migration: 004_rejection_learning.sql
-- Description: Create tables for learning from rejected signals
-- Purpose: Enable counterfactual learning - learn from the road not taken

-- ============================================================================
-- REJECTED SIGNALS TABLE (Shadow tracking of rejected signals)
-- ============================================================================
CREATE TABLE IF NOT EXISTS rejected_signals (
    id SERIAL PRIMARY KEY,
    shadow_id VARCHAR(100) UNIQUE NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    direction VARCHAR(10), -- What direction it would have been
    
    -- What was rejected
    simulated_entry_price DECIMAL(20, 8),
    simulated_take_profit DECIMAL(20, 8),
    simulated_stop_loss DECIMAL(20, 8),
    
    -- Why rejected
    rejection_reason VARCHAR(50) NOT NULL, -- 'no_consensus', 'historical_performance', 'regime_limit', 'cooldown'
    rejection_stage VARCHAR(50) NOT NULL,  -- Which filter rejected it
    
    -- Consensus data at rejection (IMPORTANT for learning)
    sde_consensus JSONB, -- All 9 head votes even if no consensus
    head_votes JSONB,    -- Individual head directions
    consensus_score DECIMAL(5, 4),
    agreeing_heads INT DEFAULT 0,
    
    -- Technical indicators at rejection
    indicators JSONB,
    market_regime VARCHAR(50),
    
    -- Shadow monitoring (track what WOULD have happened)
    monitoring_status VARCHAR(20) DEFAULT 'monitoring', -- 'monitoring', 'completed', 'expired'
    simulated_entry_time TIMESTAMPTZ NOT NULL,
    monitor_until TIMESTAMPTZ NOT NULL, -- Stop monitoring after 48 hours
    
    -- Simulated outcome (filled when monitoring completes)
    simulated_outcome VARCHAR(20), -- 'would_tp', 'would_sl', 'would_neutral', 'expired'
    simulated_exit_price DECIMAL(20, 8),
    simulated_exit_time TIMESTAMPTZ,
    simulated_profit_pct DECIMAL(10, 4),
    
    -- Learning classification
    learning_outcome VARCHAR(30), -- 'missed_opportunity', 'good_rejection', 'neutral', 'pending'
    learning_processed BOOLEAN DEFAULT FALSE,
    
    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    completed_at TIMESTAMPTZ
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_rejected_signals_symbol ON rejected_signals(symbol);
CREATE INDEX IF NOT EXISTS idx_rejected_signals_status ON rejected_signals(monitoring_status);
CREATE INDEX IF NOT EXISTS idx_rejected_signals_created ON rejected_signals(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_rejected_signals_monitoring ON rejected_signals(monitoring_status, monitor_until) 
    WHERE monitoring_status = 'monitoring';
CREATE INDEX IF NOT EXISTS idx_rejected_signals_learning ON rejected_signals(learning_processed) 
    WHERE learning_processed = FALSE;

-- ============================================================================
-- SCAN HISTORY TABLE (Every scan, whether signal or not)
-- ============================================================================
CREATE TABLE IF NOT EXISTS scan_history (
    id SERIAL PRIMARY KEY,
    scan_id VARCHAR(100) UNIQUE NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    scan_timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    -- Scan result
    result_type VARCHAR(20) NOT NULL, -- 'signal_generated', 'rejected_consensus', 'rejected_quality', 'rejected_limits'
    
    -- If signal generated
    signal_id VARCHAR(100), -- References live_signals if generated
    
    -- If rejected
    shadow_id VARCHAR(100), -- References rejected_signals if rejected
    rejection_reason VARCHAR(50),
    
    -- Consensus data (always captured)
    head_votes JSONB NOT NULL, -- All 9 head votes ALWAYS captured
    consensus_achieved BOOLEAN DEFAULT FALSE,
    consensus_direction VARCHAR(10),
    consensus_confidence DECIMAL(5, 4),
    agreeing_heads INT,
    
    -- Indicators at scan time
    indicators JSONB,
    market_regime VARCHAR(50),
    
    -- For analytics
    price_at_scan DECIMAL(20, 8)
);

-- Convert to hypertable for time-series optimization
SELECT create_hypertable('scan_history', 'scan_timestamp',
    if_not_exists => TRUE,
    chunk_time_interval => INTERVAL '1 day'
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_scan_history_symbol ON scan_history(symbol);
CREATE INDEX IF NOT EXISTS idx_scan_history_timestamp ON scan_history(scan_timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_scan_history_result ON scan_history(result_type);
CREATE INDEX IF NOT EXISTS idx_scan_history_signal ON scan_history(signal_id) WHERE signal_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_scan_history_shadow ON scan_history(shadow_id) WHERE shadow_id IS NOT NULL;

-- ============================================================================
-- REJECTION LEARNING METRICS TABLE
-- ============================================================================
CREATE TABLE IF NOT EXISTS rejection_learning_metrics (
    id SERIAL PRIMARY KEY,
    metric_date DATE NOT NULL,
    
    -- Rejection statistics
    total_scans INT DEFAULT 0,
    signals_generated INT DEFAULT 0,
    signals_rejected INT DEFAULT 0,
    rejection_rate DECIMAL(5, 4),
    
    -- Counterfactual outcomes
    missed_opportunities INT DEFAULT 0,  -- Rejected but would have won
    good_rejections INT DEFAULT 0,       -- Rejected and would have lost
    neutral_rejections INT DEFAULT 0,    -- Rejected, neutral outcome
    
    -- Learning metrics
    rejection_accuracy DECIMAL(5, 4),    -- good_rejections / (good_rejections + missed_opportunities)
    opportunity_cost DECIMAL(10, 4),     -- Sum of profits from missed opportunities
    
    -- Head performance on rejections
    head_rejection_performance JSONB,
    
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    UNIQUE(metric_date)
);

CREATE INDEX IF NOT EXISTS idx_rejection_metrics_date ON rejection_learning_metrics(metric_date DESC);

-- ============================================================================
-- HELPER FUNCTIONS
-- ============================================================================

-- Function to record a rejection
CREATE OR REPLACE FUNCTION record_rejection(
    p_shadow_id VARCHAR(100),
    p_symbol VARCHAR(20),
    p_timeframe VARCHAR(10),
    p_direction VARCHAR(10),
    p_entry_price DECIMAL(20,8),
    p_sde_consensus JSONB,
    p_rejection_reason VARCHAR(50),
    p_indicators JSONB
)
RETURNS VARCHAR(100) AS $$
BEGIN
    INSERT INTO rejected_signals (
        shadow_id, symbol, timeframe, direction,
        simulated_entry_price, sde_consensus,
        rejection_reason, rejection_stage,
        simulated_entry_time, monitor_until,
        indicators
    ) VALUES (
        p_shadow_id, p_symbol, p_timeframe, p_direction,
        p_entry_price, p_sde_consensus,
        p_rejection_reason, 'quality_filter',
        NOW(), NOW() + INTERVAL '48 hours',
        p_indicators
    );
    
    RETURN p_shadow_id;
END;
$$ LANGUAGE plpgsql;

-- Function to complete rejection monitoring
CREATE OR REPLACE FUNCTION complete_rejection_monitoring(
    p_shadow_id VARCHAR(100),
    p_outcome VARCHAR(20),
    p_exit_price DECIMAL(20,8),
    p_profit_pct DECIMAL(10,4)
)
RETURNS BOOLEAN AS $$
BEGIN
    UPDATE rejected_signals
    SET monitoring_status = 'completed',
        simulated_outcome = p_outcome,
        simulated_exit_price = p_exit_price,
        simulated_profit_pct = p_profit_pct,
        simulated_exit_time = NOW(),
        learning_outcome = CASE 
            WHEN p_outcome IN ('would_tp', 'would_neutral') AND p_profit_pct > 1.0 THEN 'missed_opportunity'
            WHEN p_outcome = 'would_sl' THEN 'good_rejection'
            ELSE 'neutral'
        END,
        completed_at = NOW()
    WHERE shadow_id = p_shadow_id;
    
    RETURN TRUE;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- COMMENTS
-- ============================================================================

COMMENT ON TABLE rejected_signals IS 'Tracks rejected signals for counterfactual learning';
COMMENT ON TABLE scan_history IS 'Complete history of all scans (signals + rejections)';
COMMENT ON TABLE rejection_learning_metrics IS 'Daily metrics on rejection accuracy';
COMMENT ON FUNCTION record_rejection IS 'Record a rejected signal for shadow tracking';
COMMENT ON FUNCTION complete_rejection_monitoring IS 'Mark rejection monitoring as complete with outcome';

