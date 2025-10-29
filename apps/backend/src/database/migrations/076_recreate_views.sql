-- Recreate missing views for Phase 4
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

-- Log the view recreation
INSERT INTO lifecycle_executions (policy_id, execution_type, table_name, execution_status, metadata)
VALUES (NULL, 'migration', 'system', 'success', 
        '{"migration_version": "076", "migration_name": "recreate_views", "views_created": 3}');

DO $$
BEGIN
    RAISE NOTICE 'Views recreated successfully for Phase 4';
END $$;
