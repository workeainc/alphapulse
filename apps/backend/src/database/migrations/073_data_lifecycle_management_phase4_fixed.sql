-- =====================================================
-- PHASE 4: DATA LIFECYCLE MANAGEMENT MIGRATION - FIXED
-- =====================================================
-- This migration fixes the issues from the previous migration
-- =====================================================

-- Fix hypertable creation issues by dropping and recreating tables
DROP TABLE IF EXISTS lifecycle_executions CASCADE;
DROP TABLE IF EXISTS compression_metrics CASCADE;
DROP TABLE IF EXISTS cleanup_operations CASCADE;

-- Recreate lifecycle_executions table without primary key conflict
CREATE TABLE IF NOT EXISTS lifecycle_executions (
    id SERIAL,
    policy_id INTEGER,
    execution_type TEXT NOT NULL,
    table_name TEXT NOT NULL,
    records_processed INTEGER DEFAULT 0,
    records_affected INTEGER DEFAULT 0,
    execution_status TEXT NOT NULL,
    error_message TEXT,
    started_at TIMESTAMPTZ DEFAULT NOW() NOT NULL,
    completed_at TIMESTAMPTZ,
    execution_duration_ms INTEGER,
    metadata JSONB
);

-- Recreate compression_metrics table without primary key conflict
CREATE TABLE IF NOT EXISTS compression_metrics (
    id SERIAL,
    table_name TEXT NOT NULL,
    chunk_name TEXT,
    original_size_bytes BIGINT,
    compressed_size_bytes BIGINT,
    compression_ratio DECIMAL(5,2),
    compression_time_ms INTEGER,
    compressed_at TIMESTAMPTZ DEFAULT NOW() NOT NULL,
    metadata JSONB
);

-- Recreate cleanup_operations table without primary key conflict
CREATE TABLE IF NOT EXISTS cleanup_operations (
    id SERIAL,
    table_name TEXT NOT NULL,
    cleanup_type TEXT NOT NULL,
    records_removed INTEGER DEFAULT 0,
    cleanup_criteria JSONB,
    execution_status TEXT NOT NULL,
    started_at TIMESTAMPTZ DEFAULT NOW() NOT NULL,
    completed_at TIMESTAMPTZ,
    error_message TEXT,
    metadata JSONB
);

-- Create hypertables properly
SELECT create_hypertable('lifecycle_executions', 'started_at', 
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

SELECT create_hypertable('compression_metrics', 'compressed_at',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

SELECT create_hypertable('cleanup_operations', 'started_at',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

-- Add primary keys after hypertable creation
ALTER TABLE lifecycle_executions ADD PRIMARY KEY (id, started_at);
ALTER TABLE compression_metrics ADD PRIMARY KEY (id, compressed_at);
ALTER TABLE cleanup_operations ADD PRIMARY KEY (id, started_at);

-- Add foreign key constraint
ALTER TABLE lifecycle_executions ADD CONSTRAINT fk_lifecycle_executions_policy 
    FOREIGN KEY (policy_id) REFERENCES lifecycle_policies(id);

-- Recreate indexes
CREATE INDEX IF NOT EXISTS idx_lifecycle_executions_policy_id ON lifecycle_executions(policy_id);
CREATE INDEX IF NOT EXISTS idx_lifecycle_executions_table_name ON lifecycle_executions(table_name);
CREATE INDEX IF NOT EXISTS idx_lifecycle_executions_status ON lifecycle_executions(execution_status);
CREATE INDEX IF NOT EXISTS idx_lifecycle_executions_started_at ON lifecycle_executions(started_at);

CREATE INDEX IF NOT EXISTS idx_compression_metrics_table_name ON compression_metrics(table_name);
CREATE INDEX IF NOT EXISTS idx_compression_metrics_ratio ON compression_metrics(compression_ratio);
CREATE INDEX IF NOT EXISTS idx_compression_metrics_compressed_at ON compression_metrics(compressed_at);

CREATE INDEX IF NOT EXISTS idx_cleanup_operations_table_name ON cleanup_operations(table_name);
CREATE INDEX IF NOT EXISTS idx_cleanup_operations_type ON cleanup_operations(cleanup_type);
CREATE INDEX IF NOT EXISTS idx_cleanup_operations_status ON cleanup_operations(execution_status);

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

-- Fix function syntax errors
CREATE OR REPLACE FUNCTION create_lifecycle_retention_policy(
    p_table_name TEXT,
    p_retention_days INTEGER,
    p_policy_name TEXT DEFAULT NULL
)
RETURNS VOID AS $$
DECLARE
    v_policy_name TEXT;
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
    PERFORM add_retention_policy(p_table_name, INTERVAL (p_retention_days::TEXT || ' days'));
    
    RAISE NOTICE 'Retention policy created: % for table % with % days retention', 
        v_policy_name, p_table_name, p_retention_days;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION create_lifecycle_compression_policy(
    p_table_name TEXT,
    p_compress_after_days INTEGER DEFAULT 7,
    p_policy_name TEXT DEFAULT NULL
)
RETURNS VOID AS $$
DECLARE
    v_policy_name TEXT;
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
    PERFORM add_compression_policy(p_table_name, INTERVAL (p_compress_after_days::TEXT || ' days'));
    
    RAISE NOTICE 'Compression policy created: % for table % with % days delay', 
        v_policy_name, p_table_name, p_compress_after_days;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION get_lifecycle_statistics(
    p_table_name TEXT DEFAULT NULL,
    p_days_back INTEGER DEFAULT 30
)
RETURNS TABLE (
    table_name TEXT,
    policy_type TEXT,
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
    WHERE le.started_at >= NOW() - INTERVAL (p_days_back::TEXT || ' days')
    AND (p_table_name IS NULL OR le.table_name = p_table_name)
    GROUP BY le.table_name, le.execution_type
    ORDER BY le.table_name, le.execution_type;
END;
$$ LANGUAGE plpgsql;

-- Log migration completion
INSERT INTO lifecycle_executions (policy_id, execution_type, table_name, execution_status, metadata)
VALUES (NULL, 'migration', 'system', 'success', 
        '{"migration_version": "073", "migration_name": "data_lifecycle_management_phase4_fixed", "fixes_applied": ["hypertable_creation", "function_syntax", "primary_key_constraints"]}');

-- Print completion message
DO $$
BEGIN
    RAISE NOTICE '====================================================';
    RAISE NOTICE 'PHASE 4: DATA LIFECYCLE MANAGEMENT MIGRATION FIXED';
    RAISE NOTICE '====================================================';
    RAISE NOTICE 'Fixed issues:';
    RAISE NOTICE '  - Hypertable creation conflicts resolved';
    RAISE NOTICE '  - Function syntax errors fixed';
    RAISE NOTICE '  - Primary key constraints properly applied';
    RAISE NOTICE '  - Compression policies enabled';
    RAISE NOTICE '';
    RAISE NOTICE 'All lifecycle management components are now operational';
    RAISE NOTICE '====================================================';
END $$;
