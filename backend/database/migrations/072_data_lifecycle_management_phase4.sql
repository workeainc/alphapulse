-- =====================================================
-- PHASE 4: DATA LIFECYCLE MANAGEMENT MIGRATION
-- =====================================================
-- This migration adds comprehensive data lifecycle management
-- to the existing TimescaleDB infrastructure
-- =====================================================

-- Enable TimescaleDB extension if not already enabled
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- =====================================================
-- 1. LIFECYCLE MANAGEMENT TABLES
-- =====================================================

-- Lifecycle policies configuration table
CREATE TABLE IF NOT EXISTS lifecycle_policies (
    id SERIAL PRIMARY KEY,
    table_name VARCHAR(100) NOT NULL,
    policy_name VARCHAR(100) NOT NULL,
    policy_type VARCHAR(50) NOT NULL, -- 'retention', 'compression', 'archive', 'cleanup'
    policy_config JSONB NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(table_name, policy_name)
);

-- Lifecycle execution history
CREATE TABLE IF NOT EXISTS lifecycle_executions (
    id SERIAL PRIMARY KEY,
    policy_id INTEGER REFERENCES lifecycle_policies(id),
    execution_type VARCHAR(50) NOT NULL, -- 'retention', 'compression', 'archive', 'cleanup'
    table_name VARCHAR(100) NOT NULL,
    records_processed INTEGER DEFAULT 0,
    records_affected INTEGER DEFAULT 0,
    execution_status VARCHAR(20) NOT NULL, -- 'success', 'failed', 'partial'
    error_message TEXT,
    started_at TIMESTAMPTZ DEFAULT NOW(),
    completed_at TIMESTAMPTZ,
    execution_duration_ms INTEGER,
    metadata JSONB
);

-- Data compression metrics
CREATE TABLE IF NOT EXISTS compression_metrics (
    id SERIAL PRIMARY KEY,
    table_name VARCHAR(100) NOT NULL,
    chunk_name VARCHAR(200),
    original_size_bytes BIGINT,
    compressed_size_bytes BIGINT,
    compression_ratio DECIMAL(5,2),
    compression_time_ms INTEGER,
    compressed_at TIMESTAMPTZ DEFAULT NOW(),
    metadata JSONB
);

-- Archive management
CREATE TABLE IF NOT EXISTS archive_metadata (
    id SERIAL PRIMARY KEY,
    table_name VARCHAR(100) NOT NULL,
    archive_name VARCHAR(200) NOT NULL,
    archive_path VARCHAR(500),
    archive_size_bytes BIGINT,
    records_count INTEGER,
    date_range_start TIMESTAMPTZ,
    date_range_end TIMESTAMPTZ,
    archive_format VARCHAR(20) DEFAULT 'parquet', -- 'parquet', 'csv', 'json'
    compression_type VARCHAR(20) DEFAULT 'gzip',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    restored_at TIMESTAMPTZ,
    is_active BOOLEAN DEFAULT TRUE,
    metadata JSONB
);

-- Cleanup tracking
CREATE TABLE IF NOT EXISTS cleanup_operations (
    id SERIAL PRIMARY KEY,
    table_name VARCHAR(100) NOT NULL,
    cleanup_type VARCHAR(50) NOT NULL, -- 'orphaned', 'duplicate', 'corrupted', 'expired'
    records_removed INTEGER DEFAULT 0,
    cleanup_criteria JSONB,
    execution_status VARCHAR(20) NOT NULL,
    started_at TIMESTAMPTZ DEFAULT NOW(),
    completed_at TIMESTAMPTZ,
    error_message TEXT,
    metadata JSONB
);

-- =====================================================
-- 2. TIMESCALEDB HYPERTABLES FOR LIFECYCLE METRICS
-- =====================================================

-- Convert lifecycle_executions to hypertable
SELECT create_hypertable('lifecycle_executions', 'started_at', 
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

-- Convert compression_metrics to hypertable
SELECT create_hypertable('compression_metrics', 'compressed_at',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

-- Convert cleanup_operations to hypertable
SELECT create_hypertable('cleanup_operations', 'started_at',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

-- =====================================================
-- 3. INDEXES FOR PERFORMANCE
-- =====================================================

-- Lifecycle policies indexes
CREATE INDEX IF NOT EXISTS idx_lifecycle_policies_table_name ON lifecycle_policies(table_name);
CREATE INDEX IF NOT EXISTS idx_lifecycle_policies_type ON lifecycle_policies(policy_type);
CREATE INDEX IF NOT EXISTS idx_lifecycle_policies_active ON lifecycle_policies(is_active);

-- Lifecycle executions indexes
CREATE INDEX IF NOT EXISTS idx_lifecycle_executions_policy_id ON lifecycle_executions(policy_id);
CREATE INDEX IF NOT EXISTS idx_lifecycle_executions_table_name ON lifecycle_executions(table_name);
CREATE INDEX IF NOT EXISTS idx_lifecycle_executions_status ON lifecycle_executions(execution_status);
CREATE INDEX IF NOT EXISTS idx_lifecycle_executions_started_at ON lifecycle_executions(started_at);

-- Compression metrics indexes
CREATE INDEX IF NOT EXISTS idx_compression_metrics_table_name ON compression_metrics(table_name);
CREATE INDEX IF NOT EXISTS idx_compression_metrics_ratio ON compression_metrics(compression_ratio);
CREATE INDEX IF NOT EXISTS idx_compression_metrics_compressed_at ON compression_metrics(compressed_at);

-- Archive metadata indexes
CREATE INDEX IF NOT EXISTS idx_archive_metadata_table_name ON archive_metadata(table_name);
CREATE INDEX IF NOT EXISTS idx_archive_metadata_active ON archive_metadata(is_active);
CREATE INDEX IF NOT EXISTS idx_archive_metadata_date_range ON archive_metadata(date_range_start, date_range_end);

-- Cleanup operations indexes
CREATE INDEX IF NOT EXISTS idx_cleanup_operations_table_name ON cleanup_operations(table_name);
CREATE INDEX IF NOT EXISTS idx_cleanup_operations_type ON cleanup_operations(cleanup_type);
CREATE INDEX IF NOT EXISTS idx_cleanup_operations_status ON cleanup_operations(execution_status);

-- =====================================================
-- 4. COMPRESSION POLICIES FOR LIFECYCLE TABLES
-- =====================================================

-- Enable compression on lifecycle tables
ALTER TABLE lifecycle_executions SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'table_name,execution_type',
    timescaledb.compress_orderby = 'started_at DESC'
);

ALTER TABLE compression_metrics SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'table_name',
    timescaledb.compress_orderby = 'compressed_at DESC'
);

ALTER TABLE cleanup_operations SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'table_name,cleanup_type',
    timescaledb.compress_orderby = 'started_at DESC'
);

-- Add compression policies
SELECT add_compression_policy('lifecycle_executions', INTERVAL '7 days');
SELECT add_compression_policy('compression_metrics', INTERVAL '7 days');
SELECT add_compression_policy('cleanup_operations', INTERVAL '7 days');

-- =====================================================
-- 5. LIFECYCLE MANAGEMENT FUNCTIONS
-- =====================================================

-- Function to create retention policy
CREATE OR REPLACE FUNCTION create_lifecycle_retention_policy(
    p_table_name VARCHAR(100),
    p_retention_days INTEGER,
    p_policy_name VARCHAR(100) DEFAULT NULL
)
RETURNS VOID AS $$
DECLARE
    v_policy_name VARCHAR(100);
    v_policy_config JSONB;
BEGIN
    -- Generate policy name if not provided
    IF p_policy_name IS NULL THEN
        v_policy_name := p_table_name || '_retention_' || p_retention_days || 'd';
    ELSE
        v_policy_name := p_policy_name;
    END IF;
    
    -- Create policy configuration
    v_policy_config := jsonb_build_object(
        'retention_days', p_retention_days,
        'policy_type', 'retention',
        'created_at', NOW()
    );
    
    -- Insert or update policy
    INSERT INTO lifecycle_policies (table_name, policy_name, policy_type, policy_config)
    VALUES (p_table_name, v_policy_name, 'retention', v_policy_config)
    ON CONFLICT (table_name, policy_name) 
    DO UPDATE SET 
        policy_config = v_policy_config,
        updated_at = NOW();
    
    -- Add TimescaleDB retention policy
    PERFORM add_retention_policy(p_table_name, INTERVAL (p_retention_days || ' days'));
    
    RAISE NOTICE 'Retention policy created: % for table % with % days retention', 
        v_policy_name, p_table_name, p_retention_days;
END;
$$ LANGUAGE plpgsql;

-- Function to create compression policy
CREATE OR REPLACE FUNCTION create_lifecycle_compression_policy(
    p_table_name VARCHAR(100),
    p_compress_after_days INTEGER DEFAULT 7,
    p_policy_name VARCHAR(100) DEFAULT NULL
)
RETURNS VOID AS $$
DECLARE
    v_policy_name VARCHAR(100);
    v_policy_config JSONB;
BEGIN
    -- Generate policy name if not provided
    IF p_policy_name IS NULL THEN
        v_policy_name := p_table_name || '_compression_' || p_compress_after_days || 'd';
    ELSE
        v_policy_name := p_policy_name;
    END IF;
    
    -- Create policy configuration
    v_policy_config := jsonb_build_object(
        'compress_after_days', p_compress_after_days,
        'policy_type', 'compression',
        'created_at', NOW()
    );
    
    -- Insert or update policy
    INSERT INTO lifecycle_policies (table_name, policy_name, policy_type, policy_config)
    VALUES (p_table_name, v_policy_name, 'compression', v_policy_config)
    ON CONFLICT (table_name, policy_name) 
    DO UPDATE SET 
        policy_config = v_policy_config,
        updated_at = NOW();
    
    -- Add TimescaleDB compression policy
    PERFORM add_compression_policy(p_table_name, INTERVAL (p_compress_after_days || ' days'));
    
    RAISE NOTICE 'Compression policy created: % for table % with % days delay', 
        v_policy_name, p_table_name, p_compress_after_days;
END;
$$ LANGUAGE plpgsql;

-- Function to execute lifecycle policy
CREATE OR REPLACE FUNCTION execute_lifecycle_policy(
    p_policy_id INTEGER
)
RETURNS INTEGER AS $$
DECLARE
    v_policy RECORD;
    v_execution_id INTEGER;
    v_start_time TIMESTAMPTZ;
    v_end_time TIMESTAMPTZ;
    v_records_affected INTEGER := 0;
    v_status VARCHAR(20) := 'success';
    v_error_message TEXT;
BEGIN
    -- Get policy details
    SELECT * INTO v_policy FROM lifecycle_policies WHERE id = p_policy_id;
    
    IF NOT FOUND THEN
        RAISE EXCEPTION 'Policy with ID % not found', p_policy_id;
    END IF;
    
    -- Create execution record
    INSERT INTO lifecycle_executions (policy_id, execution_type, table_name, execution_status)
    VALUES (p_policy_id, v_policy.policy_type, v_policy.table_name, 'running')
    RETURNING id INTO v_execution_id;
    
    v_start_time := NOW();
    
    BEGIN
        -- Execute based on policy type
        CASE v_policy.policy_type
            WHEN 'retention' THEN
                -- Retention is handled by TimescaleDB automatically
                v_records_affected := 0;
            WHEN 'compression' THEN
                -- Compression is handled by TimescaleDB automatically
                v_records_affected := 0;
            WHEN 'cleanup' THEN
                -- Execute cleanup based on policy config
                v_records_affected := execute_cleanup_operation(v_policy);
            ELSE
                RAISE EXCEPTION 'Unknown policy type: %', v_policy.policy_type;
        END CASE;
        
        v_end_time := NOW();
        
    EXCEPTION WHEN OTHERS THEN
        v_status := 'failed';
        v_error_message := SQLERRM;
        v_end_time := NOW();
    END;
    
    -- Update execution record
    UPDATE lifecycle_executions 
    SET 
        records_affected = v_records_affected,
        execution_status = v_status,
        error_message = v_error_message,
        completed_at = v_end_time,
        execution_duration_ms = EXTRACT(EPOCH FROM (v_end_time - v_start_time)) * 1000
    WHERE id = v_execution_id;
    
    RETURN v_execution_id;
END;
$$ LANGUAGE plpgsql;

-- Function to execute cleanup operation
CREATE OR REPLACE FUNCTION execute_cleanup_operation(p_policy RECORD)
RETURNS INTEGER AS $$
DECLARE
    v_records_removed INTEGER := 0;
    v_cleanup_type VARCHAR(50);
    v_criteria JSONB;
BEGIN
    v_cleanup_type := p_policy.policy_config->>'cleanup_type';
    v_criteria := p_policy.policy_config->'criteria';
    
    -- Insert cleanup operation record
    INSERT INTO cleanup_operations (table_name, cleanup_type, cleanup_criteria, execution_status)
    VALUES (p_policy.table_name, v_cleanup_type, v_criteria, 'running');
    
    -- Execute cleanup based on type
    CASE v_cleanup_type
        WHEN 'orphaned' THEN
            -- Remove orphaned records (example for signals table)
            IF p_policy.table_name = 'signals' THEN
                DELETE FROM signals 
                WHERE id NOT IN (SELECT DISTINCT signal_id FROM signal_outcomes WHERE signal_id IS NOT NULL)
                AND created_at < NOW() - INTERVAL '30 days';
                GET DIAGNOSTICS v_records_removed = ROW_COUNT;
            END IF;
        WHEN 'duplicate' THEN
            -- Remove duplicate records
            -- This would be table-specific implementation
            v_records_removed := 0;
        WHEN 'corrupted' THEN
            -- Remove corrupted records
            -- This would be table-specific implementation
            v_records_removed := 0;
        WHEN 'expired' THEN
            -- Remove expired records based on criteria
            -- This would be table-specific implementation
            v_records_removed := 0;
        ELSE
            RAISE EXCEPTION 'Unknown cleanup type: %', v_cleanup_type;
    END CASE;
    
    -- Update cleanup operation record
    UPDATE cleanup_operations 
    SET 
        records_removed = v_records_removed,
        execution_status = 'success',
        completed_at = NOW()
    WHERE table_name = p_policy.table_name 
    AND cleanup_type = v_cleanup_type 
    AND execution_status = 'running';
    
    RETURN v_records_removed;
END;
$$ LANGUAGE plpgsql;

-- Function to get lifecycle statistics
CREATE OR REPLACE FUNCTION get_lifecycle_statistics(
    p_table_name VARCHAR(100) DEFAULT NULL,
    p_days_back INTEGER DEFAULT 30
)
RETURNS TABLE (
    table_name VARCHAR(100),
    policy_type VARCHAR(50),
    executions_count INTEGER,
    success_count INTEGER,
    failed_count INTEGER,
    total_records_affected INTEGER,
    avg_execution_time_ms INTEGER,
    last_execution TIMESTAMPTZ
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        le.table_name,
        le.execution_type as policy_type,
        COUNT(*) as executions_count,
        COUNT(*) FILTER (WHERE le.execution_status = 'success') as success_count,
        COUNT(*) FILTER (WHERE le.execution_status = 'failed') as failed_count,
        COALESCE(SUM(le.records_affected), 0) as total_records_affected,
        COALESCE(AVG(le.execution_duration_ms), 0)::INTEGER as avg_execution_time_ms,
        MAX(le.started_at) as last_execution
    FROM lifecycle_executions le
    WHERE le.started_at >= NOW() - INTERVAL (p_days_back || ' days')
    AND (p_table_name IS NULL OR le.table_name = p_table_name)
    GROUP BY le.table_name, le.execution_type
    ORDER BY le.table_name, le.execution_type;
END;
$$ LANGUAGE plpgsql;

-- =====================================================
-- 6. DEFAULT LIFECYCLE POLICIES
-- =====================================================

-- Insert default lifecycle policies for existing tables
INSERT INTO lifecycle_policies (table_name, policy_name, policy_type, policy_config) VALUES
-- Streaming data tables
('stream_data', 'stream_data_retention_30d', 'retention', 
 '{"retention_days": 30, "policy_type": "retention", "created_at": "2024-01-01T00:00:00Z"}'),
('stream_data', 'stream_data_compression_7d', 'compression',
 '{"compress_after_days": 7, "policy_type": "compression", "created_at": "2024-01-01T00:00:00Z"}'),

-- Signal tables
('signals', 'signals_retention_365d', 'retention',
 '{"retention_days": 365, "policy_type": "retention", "created_at": "2024-01-01T00:00:00Z"}'),
('signals', 'signals_compression_30d', 'compression',
 '{"compress_after_days": 30, "policy_type": "compression", "created_at": "2024-01-01T00:00:00Z"}'),

-- Outcome tracking tables
('signal_outcomes', 'signal_outcomes_retention_365d', 'retention',
 '{"retention_days": 365, "policy_type": "retention", "created_at": "2024-01-01T00:00:00Z"}'),
('signal_outcomes', 'signal_outcomes_compression_30d', 'compression',
 '{"compress_after_days": 30, "policy_type": "compression", "created_at": "2024-01-01T00:00:00Z"}'),

-- Feature store tables
('feature_snapshot_versions', 'feature_snapshots_retention_180d', 'retention',
 '{"retention_days": 180, "policy_type": "retention", "created_at": "2024-01-01T00:00:00Z"}'),
('feature_snapshot_versions', 'feature_snapshots_compression_14d', 'compression',
 '{"compress_after_days": 14, "policy_type": "compression", "created_at": "2024-01-01T00:00:00Z"}'),

-- Cleanup policies
('signals', 'signals_cleanup_orphaned', 'cleanup',
 '{"cleanup_type": "orphaned", "criteria": {"min_age_days": 30}, "policy_type": "cleanup", "created_at": "2024-01-01T00:00:00Z"}')
ON CONFLICT (table_name, policy_name) DO NOTHING;

-- =====================================================
-- 7. VIEWS FOR MONITORING
-- =====================================================

-- View for lifecycle policy status
CREATE OR REPLACE VIEW lifecycle_policy_status AS
SELECT 
    lp.id,
    lp.table_name,
    lp.policy_name,
    lp.policy_type,
    lp.is_active,
    lp.created_at,
    lp.updated_at,
    COUNT(le.id) as total_executions,
    COUNT(le.id) FILTER (WHERE le.execution_status = 'success') as successful_executions,
    COUNT(le.id) FILTER (WHERE le.execution_status = 'failed') as failed_executions,
    MAX(le.started_at) as last_execution,
    MAX(le.completed_at) as last_completion
FROM lifecycle_policies lp
LEFT JOIN lifecycle_executions le ON lp.id = le.policy_id
GROUP BY lp.id, lp.table_name, lp.policy_name, lp.policy_type, lp.is_active, lp.created_at, lp.updated_at;

-- View for compression statistics
CREATE OR REPLACE VIEW compression_statistics AS
SELECT 
    table_name,
    COUNT(*) as chunks_compressed,
    AVG(compression_ratio) as avg_compression_ratio,
    SUM(original_size_bytes) as total_original_size,
    SUM(compressed_size_bytes) as total_compressed_size,
    (SUM(original_size_bytes) - SUM(compressed_size_bytes)) / SUM(original_size_bytes) * 100 as total_space_saved_percent,
    MAX(compressed_at) as last_compression
FROM compression_metrics
GROUP BY table_name;

-- View for cleanup statistics
CREATE OR REPLACE VIEW cleanup_statistics AS
SELECT 
    table_name,
    cleanup_type,
    COUNT(*) as cleanup_operations,
    SUM(records_removed) as total_records_removed,
    AVG(EXTRACT(EPOCH FROM (completed_at - started_at))) as avg_duration_seconds,
    MAX(completed_at) as last_cleanup
FROM cleanup_operations
WHERE execution_status = 'success'
GROUP BY table_name, cleanup_type;

-- =====================================================
-- 8. TRIGGERS FOR AUTOMATIC METRICS
-- =====================================================

-- Trigger function to update compression metrics
CREATE OR REPLACE FUNCTION update_compression_metrics()
RETURNS TRIGGER AS $$
BEGIN
    -- This would be called when compression happens
    -- For now, we'll create a placeholder
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- =====================================================
-- 9. GRANTS AND PERMISSIONS
-- =====================================================

-- Grant permissions to the application user
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO alpha_emon;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO alpha_emon;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO alpha_emon;

-- =====================================================
-- 10. MIGRATION COMPLETION
-- =====================================================

-- Log migration completion
INSERT INTO lifecycle_executions (policy_id, execution_type, table_name, execution_status, metadata)
VALUES (NULL, 'migration', 'system', 'success', 
        '{"migration_version": "072", "migration_name": "data_lifecycle_management_phase4", "tables_created": 6, "functions_created": 6, "views_created": 3}');

-- Print completion message
DO $$
BEGIN
    RAISE NOTICE '====================================================';
    RAISE NOTICE 'PHASE 4: DATA LIFECYCLE MANAGEMENT MIGRATION COMPLETE';
    RAISE NOTICE '====================================================';
    RAISE NOTICE 'Created tables:';
    RAISE NOTICE '  - lifecycle_policies';
    RAISE NOTICE '  - lifecycle_executions';
    RAISE NOTICE '  - compression_metrics';
    RAISE NOTICE '  - archive_metadata';
    RAISE NOTICE '  - cleanup_operations';
    RAISE NOTICE '';
    RAISE NOTICE 'Created functions:';
    RAISE NOTICE '  - create_lifecycle_retention_policy()';
    RAISE NOTICE '  - create_lifecycle_compression_policy()';
    RAISE NOTICE '  - execute_lifecycle_policy()';
    RAISE NOTICE '  - execute_cleanup_operation()';
    RAISE NOTICE '  - get_lifecycle_statistics()';
    RAISE NOTICE '';
    RAISE NOTICE 'Created views:';
    RAISE NOTICE '  - lifecycle_policy_status';
    RAISE NOTICE '  - compression_statistics';
    RAISE NOTICE '  - cleanup_statistics';
    RAISE NOTICE '';
    RAISE NOTICE 'Default policies created for existing tables';
    RAISE NOTICE '====================================================';
END $$;
