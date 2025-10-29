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
    v_interval TEXT;
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
    v_interval := p_retention_days || ' days';
    PERFORM add_retention_policy(p_table_name, INTERVAL v_interval);
    
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
    v_interval TEXT;
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
    v_interval := p_compress_after_days || ' days';
    PERFORM add_compression_policy(p_table_name, INTERVAL v_interval);
    
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
DECLARE
    v_interval TEXT;
BEGIN
    v_interval := p_days_back || ' days';
    
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
    WHERE le.started_at >= NOW() - INTERVAL v_interval
    AND (p_table_name IS NULL OR le.table_name = p_table_name)
    GROUP BY le.table_name, le.execution_type
    ORDER BY le.table_name, le.execution_type;
END;
$$ LANGUAGE plpgsql;

-- Log the fix
INSERT INTO lifecycle_executions (policy_id, execution_type, table_name, execution_status, metadata)
VALUES (NULL, 'migration', 'system', 'success', 
        '{"migration_version": "074", "migration_name": "fix_function_syntax", "fixes_applied": ["function_syntax_errors"]}');

DO $$
BEGIN
    RAISE NOTICE 'Function syntax errors fixed successfully';
END $$;
