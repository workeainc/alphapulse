-- Migration: 002_outcome_tracking_fixed.sql
-- Description: Create outcome tracking tables for Phase 2 (Fixed version)
-- Date: 2024-01-XX

-- First, create the signals table if it doesn't exist
CREATE TABLE IF NOT EXISTS signals (
    signal_id VARCHAR(50) PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    signal_type VARCHAR(10) NOT NULL, -- 'long', 'short'
    entry_price DECIMAL(20,8) NOT NULL,
    confidence_score DECIMAL(5,4) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    metadata JSONB
);

-- Signal outcomes table
CREATE TABLE IF NOT EXISTS signal_outcomes (
    id SERIAL PRIMARY KEY,
    signal_id VARCHAR(50),
    outcome_type VARCHAR(20), -- 'tp_hit', 'sl_hit', 'time_exit', 'manual_close'
    exit_price DECIMAL(20,8),
    exit_timestamp TIMESTAMPTZ,
    realized_pnl DECIMAL(20,8),
    max_adverse_excursion DECIMAL(20,8),
    max_favorable_excursion DECIMAL(20,8),
    time_to_exit INTERVAL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    -- Transactional consistency
    transaction_id UUID,
    consistency_version INTEGER DEFAULT 1,
    audit_trail JSONB,
    -- Complex order types
    order_type VARCHAR(20), -- 'market', 'limit', 'oco', 'bracket'
    partial_fill_details JSONB,
    order_state VARCHAR(20) -- 'pending', 'filled', 'cancelled', 'rejected'
);

-- TP/SL hits table
CREATE TABLE IF NOT EXISTS tp_sl_hits (
    id SERIAL PRIMARY KEY,
    signal_id VARCHAR(50),
    hit_type VARCHAR(20), -- 'take_profit', 'stop_loss', 'partial_tp', 'partial_sl', 'trailing_stop'
    hit_price DECIMAL(20,8),
    hit_timestamp TIMESTAMPTZ,
    precision VARCHAR(10), -- 'exact', 'above', 'below', 'gap'
    hit_delay_ms DECIMAL(10,2),
    partial_fill_amount DECIMAL(20,8),
    remaining_position DECIMAL(20,8),
    hit_confidence DECIMAL(5,4),
    market_conditions JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Model drift events table
CREATE TABLE IF NOT EXISTS model_drift_events (
    id SERIAL PRIMARY KEY,
    model_id TEXT,
    drift_type TEXT, -- 'statistical', 'concept', 'data'
    severity TEXT, -- 'low', 'medium', 'high', 'critical'
    detected_at TIMESTAMPTZ DEFAULT NOW(),
    drift_metrics JSONB,
    triggered_retraining BOOLEAN DEFAULT FALSE,
    resolved_at TIMESTAMPTZ
);

-- Retraining events table
CREATE TABLE IF NOT EXISTS retraining_events (
    id SERIAL PRIMARY KEY,
    model_id TEXT,
    trigger_type TEXT, -- 'drift', 'performance', 'scheduled'
    trigger_metrics JSONB,
    started_at TIMESTAMPTZ DEFAULT NOW(),
    completed_at TIMESTAMPTZ,
    performance_improvement DECIMAL(10,4),
    status TEXT -- 'pending', 'running', 'completed', 'failed'
);

-- Transaction management table
CREATE TABLE IF NOT EXISTS outcome_transactions (
    id UUID PRIMARY KEY,
    signal_id VARCHAR(50),
    transaction_type VARCHAR(50),
    status VARCHAR(20), -- 'pending', 'committed', 'rolled_back'
    created_at TIMESTAMPTZ DEFAULT NOW(),
    committed_at TIMESTAMPTZ,
    rollback_reason TEXT
);

-- Compliance and regulatory tracking table
CREATE TABLE IF NOT EXISTS compliance_events (
    id SERIAL PRIMARY KEY,
    event_type TEXT, -- 'trade_report', 'audit_log', 'regulatory_check'
    regulation TEXT, -- 'mifid_ii', 'sec', 'gdpr', 'ccpa'
    status TEXT, -- 'pending', 'completed', 'failed'
    created_at TIMESTAMPTZ DEFAULT NOW(),
    completed_at TIMESTAMPTZ,
    details JSONB,
    compliance_score DECIMAL(5,2)
);

-- Audit logs table
CREATE TABLE IF NOT EXISTS audit_logs (
    id SERIAL PRIMARY KEY,
    user_id TEXT,
    action TEXT,
    resource_type TEXT,
    resource_id TEXT,
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    ip_address INET,
    user_agent TEXT,
    details JSONB,
    -- Immutable audit trail
    hash_value TEXT,
    previous_hash TEXT
);

-- Regulatory reports table
CREATE TABLE IF NOT EXISTS regulatory_reports (
    id SERIAL PRIMARY KEY,
    report_type TEXT, -- 'trade_report', 'transparency_report'
    regulation TEXT,
    report_date DATE,
    status TEXT, -- 'pending', 'generated', 'submitted', 'acknowledged'
    created_at TIMESTAMPTZ DEFAULT NOW(),
    submitted_at TIMESTAMPTZ,
    report_data JSONB,
    validation_status TEXT
);

-- Data loss recovery tracking table
CREATE TABLE IF NOT EXISTS data_loss_events (
    id SERIAL PRIMARY KEY,
    event_type TEXT, -- 'gap_detected', 'data_corruption', 'recovery_attempt'
    symbol TEXT,
    timeframe TEXT,
    start_timestamp TIMESTAMPTZ,
    end_timestamp TIMESTAMPTZ,
    data_points_missing INTEGER,
    recovery_status TEXT, -- 'pending', 'in_progress', 'completed', 'failed'
    created_at TIMESTAMPTZ DEFAULT NOW(),
    resolved_at TIMESTAMPTZ,
    recovery_method TEXT,
    details JSONB
);

-- User feedback tracking table
CREATE TABLE IF NOT EXISTS user_feedback (
    id SERIAL PRIMARY KEY,
    user_id TEXT,
    signal_id VARCHAR(50),
    feedback_type TEXT, -- 'signal_quality', 'ui_rating', 'accuracy_rating'
    rating INTEGER CHECK (rating >= 1 AND rating <= 5),
    feedback_text TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    session_id TEXT,
    user_agent TEXT,
    ip_address INET
);

-- User satisfaction metrics table
CREATE TABLE IF NOT EXISTS user_satisfaction_metrics (
    id SERIAL PRIMARY KEY,
    user_id TEXT,
    metric_type TEXT, -- 'overall_satisfaction', 'signal_accuracy', 'ui_usability'
    metric_value DECIMAL(5,2),
    sample_size INTEGER,
    calculated_at TIMESTAMPTZ DEFAULT NOW(),
    time_period TEXT -- 'daily', 'weekly', 'monthly'
);

-- Multi-tenancy support table
CREATE TABLE IF NOT EXISTS tenants (
    id SERIAL PRIMARY KEY,
    tenant_id VARCHAR(50) UNIQUE,
    tenant_name VARCHAR(100),
    tenant_type VARCHAR(50), -- 'individual', 'institutional', 'enterprise'
    status VARCHAR(20) DEFAULT 'active', -- 'active', 'suspended', 'inactive'
    created_at TIMESTAMPTZ DEFAULT NOW(),
    config JSONB,
    limits JSONB -- API limits, storage limits, etc.
);

-- Tenant data partitions table
CREATE TABLE IF NOT EXISTS tenant_data_partitions (
    id SERIAL PRIMARY KEY,
    tenant_id VARCHAR(50),
    table_name VARCHAR(100),
    partition_key VARCHAR(100),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create indexes for performance (only if they don't exist)
DO $$
BEGIN
    -- Signal outcomes indexes
    IF NOT EXISTS (SELECT 1 FROM pg_indexes WHERE indexname = 'idx_signal_outcomes_signal_id') THEN
        CREATE INDEX idx_signal_outcomes_signal_id ON signal_outcomes(signal_id);
    END IF;
    
    IF NOT EXISTS (SELECT 1 FROM pg_indexes WHERE indexname = 'idx_signal_outcomes_timestamp') THEN
        CREATE INDEX idx_signal_outcomes_timestamp ON signal_outcomes(exit_timestamp);
    END IF;
    
    IF NOT EXISTS (SELECT 1 FROM pg_indexes WHERE indexname = 'idx_signal_outcomes_outcome_type') THEN
        CREATE INDEX idx_signal_outcomes_outcome_type ON signal_outcomes(outcome_type);
    END IF;
    
    -- TP/SL hits indexes
    IF NOT EXISTS (SELECT 1 FROM pg_indexes WHERE indexname = 'idx_tp_sl_hits_signal_id') THEN
        CREATE INDEX idx_tp_sl_hits_signal_id ON tp_sl_hits(signal_id);
    END IF;
    
    IF NOT EXISTS (SELECT 1 FROM pg_indexes WHERE indexname = 'idx_tp_sl_hits_timestamp') THEN
        CREATE INDEX idx_tp_sl_hits_timestamp ON tp_sl_hits(hit_timestamp);
    END IF;
    
    IF NOT EXISTS (SELECT 1 FROM pg_indexes WHERE indexname = 'idx_tp_sl_hits_hit_type') THEN
        CREATE INDEX idx_tp_sl_hits_hit_type ON tp_sl_hits(hit_type);
    END IF;
    
    -- Model drift indexes
    IF NOT EXISTS (SELECT 1 FROM pg_indexes WHERE indexname = 'idx_model_drift_events_model_id') THEN
        CREATE INDEX idx_model_drift_events_model_id ON model_drift_events(model_id);
    END IF;
    
    IF NOT EXISTS (SELECT 1 FROM pg_indexes WHERE indexname = 'idx_model_drift_events_detected_at') THEN
        CREATE INDEX idx_model_drift_events_detected_at ON model_drift_events(detected_at);
    END IF;
    
    -- Retraining indexes
    IF NOT EXISTS (SELECT 1 FROM pg_indexes WHERE indexname = 'idx_retraining_events_model_id') THEN
        CREATE INDEX idx_retraining_events_model_id ON retraining_events(model_id);
    END IF;
    
    IF NOT EXISTS (SELECT 1 FROM pg_indexes WHERE indexname = 'idx_retraining_events_started_at') THEN
        CREATE INDEX idx_retraining_events_started_at ON retraining_events(started_at);
    END IF;
    
    -- Other indexes
    IF NOT EXISTS (SELECT 1 FROM pg_indexes WHERE indexname = 'idx_outcome_transactions_signal_id') THEN
        CREATE INDEX idx_outcome_transactions_signal_id ON outcome_transactions(signal_id);
    END IF;
    
    IF NOT EXISTS (SELECT 1 FROM pg_indexes WHERE indexname = 'idx_compliance_events_regulation') THEN
        CREATE INDEX idx_compliance_events_regulation ON compliance_events(regulation);
    END IF;
    
    IF NOT EXISTS (SELECT 1 FROM pg_indexes WHERE indexname = 'idx_audit_logs_timestamp') THEN
        CREATE INDEX idx_audit_logs_timestamp ON audit_logs(timestamp);
    END IF;
    
    IF NOT EXISTS (SELECT 1 FROM pg_indexes WHERE indexname = 'idx_user_feedback_user_id') THEN
        CREATE INDEX idx_user_feedback_user_id ON user_feedback(user_id);
    END IF;
    
    IF NOT EXISTS (SELECT 1 FROM pg_indexes WHERE indexname = 'idx_tenants_tenant_id') THEN
        CREATE INDEX idx_tenants_tenant_id ON tenants(tenant_id);
    END IF;
END $$;

-- Create hypertables for time-series data (only if TimescaleDB is available)
DO $$
BEGIN
    -- Check if TimescaleDB extension is available
    IF EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'timescaledb') THEN
        -- Create hypertables
        PERFORM create_hypertable('signal_outcomes', 'exit_timestamp', if_not_exists => TRUE);
        PERFORM create_hypertable('tp_sl_hits', 'hit_timestamp', if_not_exists => TRUE);
        PERFORM create_hypertable('model_drift_events', 'detected_at', if_not_exists => TRUE);
        PERFORM create_hypertable('retraining_events', 'started_at', if_not_exists => TRUE);
        PERFORM create_hypertable('compliance_events', 'created_at', if_not_exists => TRUE);
        PERFORM create_hypertable('audit_logs', 'timestamp', if_not_exists => TRUE);
        PERFORM create_hypertable('data_loss_events', 'created_at', if_not_exists => TRUE);
        PERFORM create_hypertable('user_feedback', 'created_at', if_not_exists => TRUE);
        PERFORM create_hypertable('user_satisfaction_metrics', 'calculated_at', if_not_exists => TRUE);
        
        -- Enable compression on hypertables
        ALTER TABLE signal_outcomes SET (timescaledb.compress, timescaledb.compress_segmentby = 'signal_id, outcome_type');
        ALTER TABLE tp_sl_hits SET (timescaledb.compress, timescaledb.compress_segmentby = 'signal_id, hit_type');
        ALTER TABLE model_drift_events SET (timescaledb.compress, timescaledb.compress_segmentby = 'model_id, drift_type');
        ALTER TABLE retraining_events SET (timescaledb.compress, timescaledb.compress_segmentby = 'model_id, trigger_type');
        ALTER TABLE compliance_events SET (timescaledb.compress, timescaledb.compress_segmentby = 'event_type, regulation');
        ALTER TABLE audit_logs SET (timescaledb.compress, timescaledb.compress_segmentby = 'user_id, action');
        ALTER TABLE data_loss_events SET (timescaledb.compress, timescaledb.compress_segmentby = 'symbol, event_type');
        ALTER TABLE user_feedback SET (timescaledb.compress, timescaledb.compress_segmentby = 'user_id, feedback_type');
        ALTER TABLE user_satisfaction_metrics SET (timescaledb.compress, timescaledb.compress_segmentby = 'user_id, metric_type');
        
        -- Add compression policies
        PERFORM add_compression_policy('signal_outcomes', INTERVAL '7 days');
        PERFORM add_compression_policy('tp_sl_hits', INTERVAL '7 days');
        PERFORM add_compression_policy('model_drift_events', INTERVAL '30 days');
        PERFORM add_compression_policy('retraining_events', INTERVAL '30 days');
        PERFORM add_compression_policy('compliance_events', INTERVAL '90 days');
        PERFORM add_compression_policy('audit_logs', INTERVAL '30 days');
        PERFORM add_compression_policy('data_loss_events', INTERVAL '90 days');
        PERFORM add_compression_policy('user_feedback', INTERVAL '30 days');
        PERFORM add_compression_policy('user_satisfaction_metrics', INTERVAL '30 days');
        
        -- Add retention policies
        PERFORM add_retention_policy('signal_outcomes', INTERVAL '2 years');
        PERFORM add_retention_policy('tp_sl_hits', INTERVAL '2 years');
        PERFORM add_retention_policy('model_drift_events', INTERVAL '5 years');
        PERFORM add_retention_policy('retraining_events', INTERVAL '5 years');
        PERFORM add_retention_policy('compliance_events', INTERVAL '7 years');
        PERFORM add_retention_policy('audit_logs', INTERVAL '7 years');
        PERFORM add_retention_policy('data_loss_events', INTERVAL '5 years');
        PERFORM add_retention_policy('user_feedback', INTERVAL '3 years');
        PERFORM add_retention_policy('user_satisfaction_metrics', INTERVAL '3 years');
    END IF;
END $$;

-- Create views for common queries
CREATE OR REPLACE VIEW outcome_summary AS
SELECT 
    DATE_TRUNC('day', exit_timestamp) as date,
    outcome_type,
    COUNT(*) as count,
    AVG(realized_pnl) as avg_pnl,
    SUM(realized_pnl) as total_pnl,
    AVG(EXTRACT(EPOCH FROM time_to_exit)) as avg_time_to_exit_seconds
FROM signal_outcomes 
GROUP BY DATE_TRUNC('day', exit_timestamp), outcome_type
ORDER BY date DESC, outcome_type;

CREATE OR REPLACE VIEW tp_sl_performance AS
SELECT 
    signal_id,
    hit_type,
    hit_price,
    hit_timestamp,
    precision,
    hit_delay_ms,
    market_conditions
FROM tp_sl_hits
ORDER BY hit_timestamp DESC;

CREATE OR REPLACE VIEW model_drift_summary AS
SELECT 
    model_id,
    drift_type,
    severity,
    COUNT(*) as event_count,
    MAX(detected_at) as last_detected,
    AVG(CAST(drift_metrics->>'confidence' AS DECIMAL)) as avg_confidence
FROM model_drift_events
GROUP BY model_id, drift_type, severity
ORDER BY last_detected DESC;

-- Insert sample data for testing (only if not exists)
INSERT INTO tenants (tenant_id, tenant_name, tenant_type, config, limits) 
SELECT 'default', 'Default Tenant', 'individual', '{"features": ["basic"]}', '{"api_calls_per_hour": 1000, "storage_gb": 10}'
WHERE NOT EXISTS (SELECT 1 FROM tenants WHERE tenant_id = 'default');

INSERT INTO tenants (tenant_id, tenant_name, tenant_type, config, limits) 
SELECT 'premium', 'Premium Tenant', 'institutional', '{"features": ["basic", "advanced", "premium"]}', '{"api_calls_per_hour": 10000, "storage_gb": 100}'
WHERE NOT EXISTS (SELECT 1 FROM tenants WHERE tenant_id = 'premium');

-- Log migration completion
INSERT INTO audit_logs (user_id, action, resource_type, resource_id, details) VALUES
('system', 'migration_executed', 'database', '002_outcome_tracking_fixed', '{"migration": "002_outcome_tracking_fixed.sql", "tables_created": 12, "indexes_created": 15, "views_created": 3, "status": "completed"}');
