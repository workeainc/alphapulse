-- Fix duplicated security functions
-- Drop and recreate security functions to resolve duplication

-- Drop existing functions
DROP FUNCTION IF EXISTS log_security_audit(VARCHAR, VARCHAR, VARCHAR, VARCHAR, VARCHAR, JSONB, INET, TEXT, VARCHAR, TEXT, INTEGER, INTEGER, BOOLEAN, TEXT, JSONB);
DROP FUNCTION IF EXISTS log_security_audit(TEXT, TEXT, TEXT, TEXT, TEXT, JSONB, INET, TEXT, TEXT, TEXT, INTEGER, INTEGER, BOOLEAN, TEXT, JSONB);
DROP FUNCTION IF EXISTS check_user_permission(VARCHAR, VARCHAR, VARCHAR, VARCHAR);
DROP FUNCTION IF EXISTS check_user_permission(TEXT, TEXT, TEXT, TEXT);
DROP FUNCTION IF EXISTS log_security_event(VARCHAR, VARCHAR, VARCHAR, VARCHAR, VARCHAR, JSONB, INET, TEXT, VARCHAR);
DROP FUNCTION IF EXISTS log_security_event(TEXT, TEXT, TEXT, TEXT, TEXT, JSONB, INET, TEXT, TEXT);
DROP FUNCTION IF EXISTS rotate_secret(VARCHAR, VARCHAR);
DROP FUNCTION IF EXISTS rotate_secret(TEXT, TEXT);
DROP FUNCTION IF EXISTS get_security_statistics(INTEGER);

-- Recreate functions with proper signatures
CREATE OR REPLACE FUNCTION log_security_audit(
    p_user_id TEXT,
    p_session_id TEXT,
    p_action_type TEXT,
    p_resource_type TEXT,
    p_resource_id TEXT,
    p_action_details JSONB,
    p_ip_address INET,
    p_user_agent TEXT,
    p_request_method TEXT,
    p_request_path TEXT,
    p_response_status INTEGER,
    p_execution_time_ms INTEGER,
    p_success BOOLEAN,
    p_error_message TEXT,
    p_metadata JSONB
) RETURNS BIGINT AS $$
DECLARE
    audit_id BIGINT;
BEGIN
    INSERT INTO security_audit_logs (
        user_id, session_id, action_type, resource_type, resource_id,
        action_details, ip_address, user_agent, request_method, request_path,
        response_status, execution_time_ms, success, error_message, metadata
    ) VALUES (
        p_user_id, p_session_id, p_action_type, p_resource_type, p_resource_id,
        p_action_details, p_ip_address, p_user_agent, p_request_method, p_request_path,
        p_response_status, p_execution_time_ms, p_success, p_error_message, p_metadata
    ) RETURNING id INTO audit_id;
    
    RETURN audit_id;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION check_user_permission(
    p_user_id TEXT,
    p_permission TEXT,
    p_resource_type TEXT DEFAULT NULL,
    p_resource_id TEXT DEFAULT NULL
) RETURNS BOOLEAN AS $$
DECLARE
    has_permission BOOLEAN := false;
    user_role TEXT;
    role_permissions JSONB;
BEGIN
    -- Get user's active role and permissions
    SELECT role_name, permissions INTO user_role, role_permissions
    FROM security_access_control
    WHERE user_id = p_user_id 
      AND is_active = true 
      AND (expires_at IS NULL OR expires_at > NOW())
    ORDER BY created_at DESC
    LIMIT 1;
    
    IF user_role IS NULL THEN
        RETURN false;
    END IF;
    
    -- Check if permission exists in role permissions
    has_permission := (role_permissions ? p_permission);
    
    -- If resource type is specified, check resource-specific permissions
    IF p_resource_type IS NOT NULL AND has_permission THEN
        has_permission := (role_permissions -> p_permission ? p_resource_type);
    END IF;
    
    RETURN has_permission;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION log_security_event(
    p_event_type TEXT,
    p_severity TEXT,
    p_source TEXT,
    p_user_id TEXT,
    p_session_id TEXT,
    p_event_details JSONB,
    p_ip_address INET,
    p_user_agent TEXT,
    p_threat_level TEXT
) RETURNS BIGINT AS $$
DECLARE
    event_id BIGINT;
BEGIN
    INSERT INTO security_events (
        event_type, severity, source, user_id, session_id,
        event_details, ip_address, user_agent, threat_level
    ) VALUES (
        p_event_type, p_severity, p_source, p_user_id, p_session_id,
        p_event_details, p_ip_address, p_user_agent, p_threat_level
    ) RETURNING id INTO event_id;
    
    RETURN event_id;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION rotate_secret(
    p_secret_name TEXT,
    p_new_version TEXT
) RETURNS BOOLEAN AS $$
DECLARE
    rotation_interval INTEGER;
BEGIN
    -- Get current secret metadata
    SELECT key_rotation_interval_days INTO rotation_interval
    FROM security_secrets_metadata
    WHERE secret_name = p_secret_name AND is_active = true;
    
    IF NOT FOUND THEN
        RETURN false;
    END IF;
    
    -- Update secret version and rotation timestamps
    UPDATE security_secrets_metadata
    SET 
        secret_version = p_new_version,
        last_rotated_at = NOW(),
        next_rotation_at = NOW() + (rotation_interval || ' days')::INTERVAL,
        updated_at = NOW()
    WHERE secret_name = p_secret_name;
    
    RETURN true;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION get_security_statistics(
    p_days_back INTEGER DEFAULT 30
) RETURNS JSONB AS $$
DECLARE
    stats JSONB;
BEGIN
    SELECT jsonb_build_object(
        'audit_logs', (
            SELECT jsonb_build_object(
                'total_events', COUNT(*),
                'successful_events', COUNT(*) FILTER (WHERE success = true),
                'failed_events', COUNT(*) FILTER (WHERE success = false),
                'unique_users', COUNT(DISTINCT user_id),
                'unique_ips', COUNT(DISTINCT ip_address)
            )
            FROM security_audit_logs
            WHERE created_at >= NOW() - (p_days_back || ' days')::INTERVAL
        ),
        'security_events', (
            SELECT jsonb_build_object(
                'total_events', COUNT(*),
                'resolved_events', COUNT(*) FILTER (WHERE is_resolved = true),
                'unresolved_events', COUNT(*) FILTER (WHERE is_resolved = false),
                'high_severity', COUNT(*) FILTER (WHERE severity = 'high'),
                'medium_severity', COUNT(*) FILTER (WHERE severity = 'medium'),
                'low_severity', COUNT(*) FILTER (WHERE severity = 'low')
            )
            FROM security_events
            WHERE created_at >= NOW() - (p_days_back || ' days')::INTERVAL
        ),
        'access_control', (
            SELECT jsonb_build_object(
                'active_users', COUNT(DISTINCT user_id),
                'total_roles', COUNT(DISTINCT role_name),
                'expired_access', COUNT(*) FILTER (WHERE expires_at < NOW())
            )
            FROM security_access_control
            WHERE is_active = true
        ),
        'secrets_management', (
            SELECT jsonb_build_object(
                'total_secrets', COUNT(*),
                'active_secrets', COUNT(*) FILTER (WHERE is_active = true),
                'secrets_due_rotation', COUNT(*) FILTER (WHERE next_rotation_at < NOW())
            )
            FROM security_secrets_metadata
        )
    ) INTO stats;
    
    RETURN stats;
END;
$$ LANGUAGE plpgsql;

-- Log the migration execution
INSERT INTO lifecycle_executions (policy_id, execution_type, table_name, execution_status, metadata)
VALUES (NULL, 'migration', 'system', 'success', 
        '{"migration_version": "079", "migration_name": "fix_security_functions", "functions_fixed": 5}');

DO $$
BEGIN
    RAISE NOTICE 'Security functions fixed successfully';
END $$;
