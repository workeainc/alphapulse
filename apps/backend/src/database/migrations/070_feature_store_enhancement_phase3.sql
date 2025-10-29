-- =====================================================
-- PHASE 3: FEATURE STORE ENHANCEMENT MIGRATION
-- =====================================================
-- Migration: 070_feature_store_enhancement_phase3.sql
-- Purpose: Enhance feature store with versioned snapshots, lineage tracking, and quality monitoring
-- Database: TimescaleDB (localhost)
-- User: alpha_emon
-- Date: August 29, 2025
-- =====================================================

-- Enable TimescaleDB extension if not already enabled
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- =====================================================
-- 1. FEATURE SNAPSHOTS TABLE
-- =====================================================
CREATE TABLE IF NOT EXISTS feature_snapshots (
    id SERIAL PRIMARY KEY,
    snapshot_id VARCHAR(100) UNIQUE NOT NULL,
    feature_set_name VARCHAR(100) NOT NULL,
    version VARCHAR(20) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    metadata JSONB DEFAULT '{}',
    feature_count INTEGER DEFAULT 0,
    data_points_count INTEGER DEFAULT 0,
    -- Streaming integration
    streaming_data_version VARCHAR(50),
    consistency_hash VARCHAR(64),
    validation_status VARCHAR(20) DEFAULT 'pending',
    -- Performance tracking
    computation_time_ms INTEGER,
    memory_usage_mb INTEGER,
    -- Quality metrics
    quality_score DECIMAL(5,2),
    drift_detected BOOLEAN DEFAULT FALSE,
    -- Indexes for performance
    CONSTRAINT feature_snapshots_version_check CHECK (version ~ '^[0-9]+\.[0-9]+\.[0-9]+$')
);

-- Create TimescaleDB hypertable for time-series optimization
SELECT create_hypertable('feature_snapshots', 'created_at', if_not_exists => TRUE);

-- Add indexes for performance
CREATE INDEX IF NOT EXISTS idx_feature_snapshots_snapshot_id ON feature_snapshots(snapshot_id);
CREATE INDEX IF NOT EXISTS idx_feature_snapshots_feature_set ON feature_snapshots(feature_set_name);
CREATE INDEX IF NOT EXISTS idx_feature_snapshots_version ON feature_snapshots(version);
CREATE INDEX IF NOT EXISTS idx_feature_snapshots_validation ON feature_snapshots(validation_status);
CREATE INDEX IF NOT EXISTS idx_feature_snapshots_quality ON feature_snapshots(quality_score);

-- =====================================================
-- 2. FEATURE LINEAGE TABLE
-- =====================================================
CREATE TABLE IF NOT EXISTS feature_lineage (
    id SERIAL PRIMARY KEY,
    feature_name VARCHAR(100) NOT NULL,
    parent_features JSONB DEFAULT '[]',
    computation_rule TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    version VARCHAR(20) NOT NULL,
    -- Cross-system lineage
    streaming_source VARCHAR(100),
    outcome_tracking_version VARCHAR(50),
    cross_system_consistency BOOLEAN DEFAULT TRUE,
    -- Performance tracking
    computation_complexity VARCHAR(20) DEFAULT 'low', -- low, medium, high
    estimated_computation_time_ms INTEGER,
    -- Quality tracking
    lineage_quality_score DECIMAL(5,2),
    dependency_count INTEGER DEFAULT 0,
    -- Indexes
    CONSTRAINT feature_lineage_version_check CHECK (version ~ '^[0-9]+\.[0-9]+\.[0-9]+$')
);

-- Create TimescaleDB hypertable
SELECT create_hypertable('feature_lineage', 'created_at', if_not_exists => TRUE);

-- Add indexes
CREATE INDEX IF NOT EXISTS idx_feature_lineage_feature_name ON feature_lineage(feature_name);
CREATE INDEX IF NOT EXISTS idx_feature_lineage_version ON feature_lineage(version);
CREATE INDEX IF NOT EXISTS idx_feature_lineage_consistency ON feature_lineage(cross_system_consistency);

-- =====================================================
-- 3. FEATURE CONSISTENCY CHECKS TABLE
-- =====================================================
CREATE TABLE IF NOT EXISTS feature_consistency_checks (
    id SERIAL PRIMARY KEY,
    snapshot_id VARCHAR(100) NOT NULL,
    check_type VARCHAR(50) NOT NULL, -- 'streaming', 'outcome', 'cross_system', 'quality'
    status VARCHAR(20) NOT NULL, -- 'passed', 'failed', 'warning', 'pending'
    check_timestamp TIMESTAMPTZ DEFAULT NOW(),
    details JSONB DEFAULT '{}',
    auto_fixed BOOLEAN DEFAULT FALSE,
    -- Performance tracking
    check_duration_ms INTEGER,
    -- Quality metrics
    confidence_score DECIMAL(5,2),
    -- Foreign key reference
    CONSTRAINT fk_consistency_snapshot FOREIGN KEY (snapshot_id) REFERENCES feature_snapshots(snapshot_id) ON DELETE CASCADE
);

-- Create TimescaleDB hypertable
SELECT create_hypertable('feature_consistency_checks', 'check_timestamp', if_not_exists => TRUE);

-- Add indexes
CREATE INDEX IF NOT EXISTS idx_consistency_snapshot_id ON feature_consistency_checks(snapshot_id);
CREATE INDEX IF NOT EXISTS idx_consistency_check_type ON feature_consistency_checks(check_type);
CREATE INDEX IF NOT EXISTS idx_consistency_status ON feature_consistency_checks(status);
CREATE INDEX IF NOT EXISTS idx_consistency_timestamp ON feature_consistency_checks(check_timestamp);

-- =====================================================
-- 4. FEATURE PERFORMANCE METRICS TABLE
-- =====================================================
CREATE TABLE IF NOT EXISTS feature_performance_metrics (
    id SERIAL PRIMARY KEY,
    feature_name VARCHAR(100) NOT NULL,
    computation_time_ms INTEGER NOT NULL,
    memory_usage_mb INTEGER,
    usage_frequency INTEGER DEFAULT 1,
    performance_score DECIMAL(5,2),
    recorded_at TIMESTAMPTZ DEFAULT NOW(),
    -- Context information
    snapshot_id VARCHAR(100),
    computation_context VARCHAR(100), -- 'training', 'inference', 'validation'
    -- Quality metrics
    accuracy_impact DECIMAL(5,2),
    -- Indexes
    CONSTRAINT fk_performance_snapshot FOREIGN KEY (snapshot_id) REFERENCES feature_snapshots(snapshot_id) ON DELETE SET NULL
);

-- Create TimescaleDB hypertable
SELECT create_hypertable('feature_performance_metrics', 'recorded_at', if_not_exists => TRUE);

-- Add indexes
CREATE INDEX IF NOT EXISTS idx_performance_feature_name ON feature_performance_metrics(feature_name);
CREATE INDEX IF NOT EXISTS idx_performance_score ON feature_performance_metrics(performance_score);
CREATE INDEX IF NOT EXISTS idx_performance_context ON feature_performance_metrics(computation_context);

-- =====================================================
-- 5. FEATURE DOCUMENTATION TABLE
-- =====================================================
CREATE TABLE IF NOT EXISTS feature_documentation (
    id SERIAL PRIMARY KEY,
    feature_name VARCHAR(100) NOT NULL,
    documentation_version VARCHAR(20) NOT NULL,
    content TEXT NOT NULL,
    examples JSONB DEFAULT '[]',
    change_history JSONB DEFAULT '[]',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    -- Quality metrics
    documentation_quality_score DECIMAL(5,2),
    completeness_score DECIMAL(5,2),
    -- Metadata
    author VARCHAR(100),
    tags JSONB DEFAULT '[]',
    -- Indexes
    CONSTRAINT feature_doc_version_check CHECK (documentation_version ~ '^[0-9]+\.[0-9]+\.[0-9]+$')
);

-- Create TimescaleDB hypertable
SELECT create_hypertable('feature_documentation', 'created_at', if_not_exists => TRUE);

-- Add indexes
CREATE INDEX IF NOT EXISTS idx_documentation_feature_name ON feature_documentation(feature_name);
CREATE INDEX IF NOT EXISTS idx_documentation_version ON feature_documentation(documentation_version);
CREATE INDEX IF NOT EXISTS idx_documentation_quality ON feature_documentation(documentation_quality_score);

-- =====================================================
-- 6. FEATURE METADATA TABLE (Extended metadata management)
-- =====================================================
CREATE TABLE IF NOT EXISTS feature_metadata (
    id SERIAL PRIMARY KEY,
    feature_name VARCHAR(100) NOT NULL,
    metadata_type VARCHAR(50) NOT NULL, -- 'schema', 'quality', 'performance', 'business'
    metadata_key VARCHAR(100) NOT NULL,
    metadata_value JSONB NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    -- Version tracking
    version VARCHAR(20),
    -- Quality tracking
    confidence_score DECIMAL(5,2),
    -- Unique constraint
    UNIQUE(feature_name, metadata_type, metadata_key)
);

-- Create TimescaleDB hypertable
SELECT create_hypertable('feature_metadata', 'created_at', if_not_exists => TRUE);

-- Add indexes
CREATE INDEX IF NOT EXISTS idx_metadata_feature_name ON feature_metadata(feature_name);
CREATE INDEX IF NOT EXISTS idx_metadata_type ON feature_metadata(metadata_type);
CREATE INDEX IF NOT EXISTS idx_metadata_key ON feature_metadata(metadata_key);

-- =====================================================
-- 7. FEATURE DRIFT DETECTION TABLE
-- =====================================================
CREATE TABLE IF NOT EXISTS feature_drift_detection (
    id SERIAL PRIMARY KEY,
    feature_name VARCHAR(100) NOT NULL,
    drift_type VARCHAR(50) NOT NULL, -- 'statistical', 'distribution', 'concept'
    drift_score DECIMAL(5,2) NOT NULL,
    threshold_value DECIMAL(5,2) NOT NULL,
    detected_at TIMESTAMPTZ DEFAULT NOW(),
    -- Context information
    baseline_snapshot_id VARCHAR(100),
    current_snapshot_id VARCHAR(100),
    -- Details
    drift_details JSONB DEFAULT '{}',
    severity VARCHAR(20) DEFAULT 'low', -- low, medium, high, critical
    -- Action tracking
    action_taken VARCHAR(100),
    resolved_at TIMESTAMPTZ,
    -- Foreign keys
    CONSTRAINT fk_drift_baseline FOREIGN KEY (baseline_snapshot_id) REFERENCES feature_snapshots(snapshot_id) ON DELETE SET NULL,
    CONSTRAINT fk_drift_current FOREIGN KEY (current_snapshot_id) REFERENCES feature_snapshots(snapshot_id) ON DELETE SET NULL
);

-- Create TimescaleDB hypertable
SELECT create_hypertable('feature_drift_detection', 'detected_at', if_not_exists => TRUE);

-- Add indexes
CREATE INDEX IF NOT EXISTS idx_drift_feature_name ON feature_drift_detection(feature_name);
CREATE INDEX IF NOT EXISTS idx_drift_type ON feature_drift_detection(drift_type);
CREATE INDEX IF NOT EXISTS idx_drift_score ON feature_drift_detection(drift_score);
CREATE INDEX IF NOT EXISTS idx_drift_severity ON feature_drift_detection(severity);

-- =====================================================
-- 8. CREATE VIEWS FOR EASY QUERYING
-- =====================================================

-- View: Feature Snapshot Summary
CREATE OR REPLACE VIEW feature_snapshot_summary AS
SELECT 
    fs.snapshot_id,
    fs.feature_set_name,
    fs.version,
    fs.created_at,
    fs.feature_count,
    fs.data_points_count,
    fs.quality_score,
    fs.validation_status,
    COUNT(fcc.id) as consistency_checks_count,
    COUNT(CASE WHEN fcc.status = 'passed' THEN 1 END) as passed_checks,
    COUNT(CASE WHEN fcc.status = 'failed' THEN 1 END) as failed_checks
FROM feature_snapshots fs
LEFT JOIN feature_consistency_checks fcc ON fs.snapshot_id = fcc.snapshot_id
GROUP BY fs.id, fs.snapshot_id, fs.feature_set_name, fs.version, fs.created_at, 
         fs.feature_count, fs.data_points_count, fs.quality_score, fs.validation_status;

-- View: Feature Performance Summary
CREATE OR REPLACE VIEW feature_performance_summary AS
SELECT 
    fpm.feature_name,
    AVG(fpm.computation_time_ms) as avg_computation_time_ms,
    MAX(fpm.computation_time_ms) as max_computation_time_ms,
    AVG(fpm.memory_usage_mb) as avg_memory_usage_mb,
    SUM(fpm.usage_frequency) as total_usage_count,
    AVG(fpm.performance_score) as avg_performance_score,
    MAX(fpm.recorded_at) as last_used_at
FROM feature_performance_metrics fpm
GROUP BY fpm.feature_name;

-- View: Feature Quality Dashboard
CREATE OR REPLACE VIEW feature_quality_dashboard AS
SELECT 
    fs.feature_set_name,
    fs.version,
    fs.quality_score,
    fs.drift_detected,
    COUNT(fdd.id) as drift_incidents,
    MAX(fdd.drift_score) as max_drift_score,
    MAX(fdd.detected_at) as last_drift_detected
FROM feature_snapshots fs
LEFT JOIN feature_drift_detection fdd ON fs.feature_set_name = fdd.feature_name
GROUP BY fs.feature_set_name, fs.version, fs.quality_score, fs.drift_detected;

-- =====================================================
-- 9. CREATE FUNCTIONS FOR AUTOMATED OPERATIONS
-- =====================================================

-- Function: Generate snapshot ID
CREATE OR REPLACE FUNCTION generate_snapshot_id(feature_set_name VARCHAR, version VARCHAR)
RETURNS VARCHAR AS $$
BEGIN
    RETURN feature_set_name || '_' || version || '_' || EXTRACT(EPOCH FROM NOW())::BIGINT;
END;
$$ LANGUAGE plpgsql;

-- Function: Calculate consistency hash
CREATE OR REPLACE FUNCTION calculate_consistency_hash(feature_data JSONB)
RETURNS VARCHAR AS $$
BEGIN
    RETURN encode(sha256(feature_data::text::bytea), 'hex');
END;
$$ LANGUAGE plpgsql;

-- Function: Update feature documentation timestamp
CREATE OR REPLACE FUNCTION update_feature_documentation_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Function: Update feature metadata timestamp
CREATE OR REPLACE FUNCTION update_feature_metadata_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- =====================================================
-- 10. CREATE TRIGGERS FOR AUTOMATED UPDATES
-- =====================================================

-- Trigger: Update feature documentation timestamp
CREATE TRIGGER trigger_update_feature_documentation_timestamp
    BEFORE UPDATE ON feature_documentation
    FOR EACH ROW
    EXECUTE FUNCTION update_feature_documentation_timestamp();

-- Trigger: Update feature metadata timestamp
CREATE TRIGGER trigger_update_feature_metadata_timestamp
    BEFORE UPDATE ON feature_metadata
    FOR EACH ROW
    EXECUTE FUNCTION update_feature_metadata_timestamp();

-- =====================================================
-- 11. INSERT INITIAL DATA
-- =====================================================

-- Insert default metadata for existing features
INSERT INTO feature_metadata (feature_name, metadata_type, metadata_key, metadata_value, version)
VALUES 
    ('price', 'schema', 'data_type', '"float"', '1.0.0'),
    ('volume', 'schema', 'data_type', '"integer"', '1.0.0'),
    ('rsi', 'schema', 'data_type', '"float"', '1.0.0'),
    ('macd', 'schema', 'data_type', '"float"', '1.0.0'),
    ('bollinger_upper', 'schema', 'data_type', '"float"', '1.0.0'),
    ('bollinger_lower', 'schema', 'data_type', '"float"', '1.0.0')
ON CONFLICT (feature_name, metadata_type, metadata_key) DO NOTHING;

-- =====================================================
-- 12. CREATE INDEXES FOR PERFORMANCE OPTIMIZATION
-- =====================================================

-- Composite indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_snapshots_feature_version ON feature_snapshots(feature_set_name, version);
CREATE INDEX IF NOT EXISTS idx_snapshots_created_quality ON feature_snapshots(created_at, quality_score);
CREATE INDEX IF NOT EXISTS idx_lineage_feature_version ON feature_lineage(feature_name, version);
CREATE INDEX IF NOT EXISTS idx_consistency_snapshot_type ON feature_consistency_checks(snapshot_id, check_type);
CREATE INDEX IF NOT EXISTS idx_performance_feature_time ON feature_performance_metrics(feature_name, recorded_at);
CREATE INDEX IF NOT EXISTS idx_drift_feature_severity ON feature_drift_detection(feature_name, severity);

-- =====================================================
-- MIGRATION COMPLETED
-- =====================================================

-- Log migration completion
INSERT INTO feature_metadata (feature_name, metadata_type, metadata_key, metadata_value, version)
VALUES ('system', 'migration', 'phase3_completed', 
        jsonb_build_object('timestamp', NOW(), 'version', '1.0.0', 'tables_created', 7), 
        '1.0.0')
ON CONFLICT (feature_name, metadata_type, metadata_key) DO UPDATE 
SET metadata_value = EXCLUDED.metadata_value, updated_at = NOW();

-- =====================================================
-- VERIFICATION QUERIES
-- =====================================================

-- Verify tables were created
SELECT table_name, table_type 
FROM information_schema.tables 
WHERE table_schema = 'public' 
AND table_name IN ('feature_snapshots', 'feature_lineage', 'feature_consistency_checks', 
                   'feature_performance_metrics', 'feature_documentation', 'feature_metadata', 
                   'feature_drift_detection')
ORDER BY table_name;

-- Verify hypertables were created
SELECT hypertable_name, num_chunks 
FROM timescaledb_information.hypertables 
WHERE hypertable_name IN ('feature_snapshots', 'feature_lineage', 'feature_consistency_checks', 
                         'feature_performance_metrics', 'feature_documentation', 'feature_metadata', 
                         'feature_drift_detection')
ORDER BY hypertable_name;

-- Verify views were created
SELECT table_name 
FROM information_schema.views 
WHERE table_schema = 'public' 
AND table_name IN ('feature_snapshot_summary', 'feature_performance_summary', 'feature_quality_dashboard')
ORDER BY table_name;

-- =====================================================
-- MIGRATION SUCCESSFULLY COMPLETED
-- =====================================================
