-- Migration: 003_learning_state.sql
-- Description: Create learning state tables for self-learning system
-- Purpose: Store learned weights, thresholds, and performance history

-- ============================================================================
-- LEARNING STATE TABLE (Version history of all learned parameters)
-- ============================================================================
CREATE TABLE IF NOT EXISTS learning_state (
    id SERIAL PRIMARY KEY,
    state_type VARCHAR(50) NOT NULL, -- 'head_weights', 'indicator_weights', 'thresholds'
    state_data JSONB NOT NULL,
    version INTEGER NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    performance_metrics JSONB,
    deployment_status VARCHAR(20) DEFAULT 'deployed', -- 'deployed', 'testing', 'archived'
    created_by VARCHAR(50) DEFAULT 'system', -- 'system', 'manual', 'retraining_job'
    notes TEXT,
    
    CONSTRAINT unique_state_version UNIQUE(state_type, version)
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_learning_state_type ON learning_state(state_type);
CREATE INDEX IF NOT EXISTS idx_learning_state_version ON learning_state(state_type, version DESC);
CREATE INDEX IF NOT EXISTS idx_learning_state_created ON learning_state(created_at DESC);

-- ============================================================================
-- ACTIVE LEARNING STATE TABLE (Current deployed parameters)
-- ============================================================================
CREATE TABLE IF NOT EXISTS active_learning_state (
    state_type VARCHAR(50) PRIMARY KEY,
    current_version INTEGER NOT NULL,
    state_data JSONB NOT NULL,
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    previous_version INTEGER,
    deployment_timestamp TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================================================
-- LEARNING EVENTS LOG (Track all learning events)
-- ============================================================================
CREATE TABLE IF NOT EXISTS learning_events (
    id SERIAL PRIMARY KEY,
    event_type VARCHAR(50) NOT NULL, -- 'weight_update', 'threshold_adjust', 'model_retrain'
    event_timestamp TIMESTAMPTZ DEFAULT NOW(),
    signal_id VARCHAR(100), -- Related signal if applicable
    state_type VARCHAR(50),
    old_value JSONB,
    new_value JSONB,
    performance_delta JSONB, -- Change in performance metrics
    triggered_by VARCHAR(50) DEFAULT 'outcome', -- 'outcome', 'daily_job', 'weekly_job', 'manual'
    notes TEXT
);

CREATE INDEX IF NOT EXISTS idx_learning_events_timestamp ON learning_events(event_timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_learning_events_type ON learning_events(event_type);
CREATE INDEX IF NOT EXISTS idx_learning_events_signal ON learning_events(signal_id);

-- ============================================================================
-- INITIAL DATA: Insert default learning states
-- ============================================================================

-- Default 9-head weights (equal distribution initially)
INSERT INTO active_learning_state (state_type, current_version, state_data)
VALUES (
    'head_weights',
    1,
    '{
        "HEAD_A": 0.111,
        "HEAD_B": 0.111,
        "HEAD_C": 0.111,
        "HEAD_D": 0.111,
        "HEAD_E": 0.111,
        "HEAD_F": 0.111,
        "HEAD_G": 0.111,
        "HEAD_H": 0.111,
        "HEAD_I": 0.111
    }'::jsonb
)
ON CONFLICT (state_type) DO NOTHING;

-- Store in version history
INSERT INTO learning_state (state_type, version, state_data, performance_metrics, created_by, notes)
VALUES (
    'head_weights',
    1,
    '{
        "HEAD_A": 0.111,
        "HEAD_B": 0.111,
        "HEAD_C": 0.111,
        "HEAD_D": 0.111,
        "HEAD_E": 0.111,
        "HEAD_F": 0.111,
        "HEAD_G": 0.111,
        "HEAD_H": 0.111,
        "HEAD_I": 0.111
    }'::jsonb,
    '{
        "baseline": true,
        "win_rate": null,
        "signals_analyzed": 0
    }'::jsonb,
    'system',
    'Initial default weights - equal distribution across all 9 heads'
)
ON CONFLICT (state_type, version) DO NOTHING;

-- Default confidence threshold
INSERT INTO active_learning_state (state_type, current_version, state_data)
VALUES (
    'confidence_threshold',
    1,
    '{
        "global_threshold": 0.70,
        "regime_thresholds": {
            "trending_bullish": 0.65,
            "trending_bearish": 0.65,
            "ranging": 0.75,
            "volatile": 0.75,
            "unknown": 0.70
        }
    }'::jsonb
)
ON CONFLICT (state_type) DO NOTHING;

-- Store in version history
INSERT INTO learning_state (state_type, version, state_data, performance_metrics, created_by, notes)
VALUES (
    'confidence_threshold',
    1,
    '{
        "global_threshold": 0.70,
        "regime_thresholds": {
            "trending_bullish": 0.65,
            "trending_bearish": 0.65,
            "ranging": 0.75,
            "volatile": 0.75,
            "unknown": 0.70
        }
    }'::jsonb,
    '{
        "baseline": true
    }'::jsonb,
    'system',
    'Initial thresholds based on system specification'
)
ON CONFLICT (state_type, version) DO NOTHING;

-- Default learning configuration
INSERT INTO active_learning_state (state_type, current_version, state_data)
VALUES (
    'learning_config',
    1,
    '{
        "learning_rate": 0.05,
        "min_outcomes_for_update": 10,
        "min_outcomes_for_retraining": 50,
        "max_weight_change_per_update": 0.20,
        "ema_alpha": 0.05,
        "enable_automatic_updates": true,
        "enable_weekly_retraining": true,
        "enable_regime_adaptation": true
    }'::jsonb
)
ON CONFLICT (state_type) DO NOTHING;

-- Store in version history
INSERT INTO learning_state (state_type, version, state_data, created_by, notes)
VALUES (
    'learning_config',
    1,
    '{
        "learning_rate": 0.05,
        "min_outcomes_for_update": 10,
        "min_outcomes_for_retraining": 50,
        "max_weight_change_per_update": 0.20,
        "ema_alpha": 0.05,
        "enable_automatic_updates": true,
        "enable_weekly_retraining": true,
        "enable_regime_adaptation": true
    }'::jsonb,
    'system',
    'Initial learning configuration with conservative parameters'
)
ON CONFLICT (state_type, version) DO NOTHING;

-- ============================================================================
-- HELPER FUNCTIONS
-- ============================================================================

-- Function to get current head weights
CREATE OR REPLACE FUNCTION get_current_head_weights()
RETURNS JSONB AS $$
BEGIN
    RETURN (
        SELECT state_data 
        FROM active_learning_state 
        WHERE state_type = 'head_weights'
    );
END;
$$ LANGUAGE plpgsql;

-- Function to update head weights with versioning
CREATE OR REPLACE FUNCTION update_head_weights(new_weights JSONB, metrics JSONB)
RETURNS INTEGER AS $$
DECLARE
    new_version INTEGER;
BEGIN
    -- Get next version number
    SELECT COALESCE(MAX(version), 0) + 1 INTO new_version
    FROM learning_state
    WHERE state_type = 'head_weights';
    
    -- Insert new version into history
    INSERT INTO learning_state (state_type, version, state_data, performance_metrics, created_by)
    VALUES ('head_weights', new_version, new_weights, metrics, 'system');
    
    -- Update active state
    UPDATE active_learning_state
    SET current_version = new_version,
        state_data = new_weights,
        updated_at = NOW(),
        previous_version = current_version
    WHERE state_type = 'head_weights';
    
    RETURN new_version;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- COMMENTS
-- ============================================================================

COMMENT ON TABLE learning_state IS 'Version history of all learned parameters';
COMMENT ON TABLE active_learning_state IS 'Currently deployed learning parameters';
COMMENT ON TABLE learning_events IS 'Log of all learning events for audit trail';
COMMENT ON FUNCTION get_current_head_weights() IS 'Retrieve current active head weights';
COMMENT ON FUNCTION update_head_weights(JSONB, JSONB) IS 'Update head weights with automatic versioning';

