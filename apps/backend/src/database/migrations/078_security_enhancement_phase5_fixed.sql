-- Phase 5: Security Enhancement Database Migration - FIXED
-- Comprehensive security infrastructure for AlphaPlus

-- =====================================================
-- SECURITY TABLES CREATION (FIXED)
-- =====================================================

-- Drop existing tables if they exist (from previous migration)
DROP TABLE IF EXISTS security_audit_logs CASCADE;
DROP TABLE IF EXISTS security_events CASCADE;

-- 1. Security Audit Logs Table (Hypertable) - FIXED
CREATE TABLE IF NOT EXISTS security_audit_logs (
    id BIGSERIAL,
    user_id TEXT,
    session_id TEXT,
    action_type TEXT NOT NULL,
    resource_type TEXT,
    resource_id TEXT,
    action_details JSONB,
    ip_address INET,
    user_agent TEXT,
    request_method TEXT,
    request_path TEXT,
    response_status INTEGER,
    execution_time_ms INTEGER,
    success BOOLEAN DEFAULT true,
    error_message TEXT,
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (id, created_at)
);

-- Convert to hypertable for time-series optimization
SELECT create_hypertable('security_audit_logs', 'created_at', if_not_exists => TRUE);

-- 2. Security Access Control Table
CREATE TABLE IF NOT EXISTS security_access_control (
    id BIGSERIAL PRIMARY KEY,
    user_id TEXT NOT NULL,
    role_name TEXT NOT NULL,
    permissions JSONB NOT NULL,
    resource_scope TEXT,
    is_active BOOLEAN DEFAULT true,
    expires_at TIMESTAMPTZ,
    created_by TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- 3. Security Secrets Metadata Table
CREATE TABLE IF NOT EXISTS security_secrets_metadata (
    id BIGSERIAL PRIMARY KEY,
    secret_name TEXT NOT NULL UNIQUE,
    secret_type TEXT NOT NULL,
    secret_version TEXT NOT NULL,
    encryption_algorithm TEXT,
    key_rotation_interval_days INTEGER DEFAULT 30,
    last_rotated_at TIMESTAMPTZ,
    next_rotation_at TIMESTAMPTZ,
    is_active BOOLEAN DEFAULT true,
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- 4. Security Events Table (Hypertable) - FIXED
CREATE TABLE IF NOT EXISTS security_events (
    id BIGSERIAL,
    event_type TEXT NOT NULL,
    severity TEXT NOT NULL,
    source TEXT,
    user_id TEXT,
    session_id TEXT,
    event_details JSONB,
    ip_address INET,
    user_agent TEXT,
    threat_level TEXT,
    is_resolved BOOLEAN DEFAULT false,
    resolved_at TIMESTAMPTZ,
    resolved_by TEXT,
    resolution_notes TEXT,
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (id, created_at)
);

-- Convert to hypertable for time-series optimization
SELECT create_hypertable('security_events', 'created_at', if_not_exists => TRUE);

-- 5. Security Policies Table
CREATE TABLE IF NOT EXISTS security_policies (
    id BIGSERIAL PRIMARY KEY,
    policy_name TEXT NOT NULL UNIQUE,
    policy_type TEXT NOT NULL,
    policy_config JSONB NOT NULL,
    is_active BOOLEAN DEFAULT true,
    priority INTEGER DEFAULT 0,
    created_by TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- =====================================================
-- INDEXES FOR PERFORMANCE
-- =====================================================

-- Security Audit Logs Indexes
CREATE INDEX IF NOT EXISTS idx_security_audit_logs_user_id ON security_audit_logs(user_id);
CREATE INDEX IF NOT EXISTS idx_security_audit_logs_action_type ON security_audit_logs(action_type);
CREATE INDEX IF NOT EXISTS idx_security_audit_logs_success ON security_audit_logs(success);
CREATE INDEX IF NOT EXISTS idx_security_audit_logs_ip_address ON security_audit_logs(ip_address);

-- Security Access Control Indexes
CREATE INDEX IF NOT EXISTS idx_security_access_control_user_id ON security_access_control(user_id);
CREATE INDEX IF NOT EXISTS idx_security_access_control_role_name ON security_access_control(role_name);
CREATE INDEX IF NOT EXISTS idx_security_access_control_is_active ON security_access_control(is_active);

-- Security Events Indexes
CREATE INDEX IF NOT EXISTS idx_security_events_event_type ON security_events(event_type);
CREATE INDEX IF NOT EXISTS idx_security_events_severity ON security_events(severity);
CREATE INDEX IF NOT EXISTS idx_security_events_user_id ON security_events(user_id);
CREATE INDEX IF NOT EXISTS idx_security_events_is_resolved ON security_events(is_resolved);

-- Security Policies Indexes
CREATE INDEX IF NOT EXISTS idx_security_policies_policy_type ON security_policies(policy_type);
CREATE INDEX IF NOT EXISTS idx_security_policies_is_active ON security_policies(is_active);

-- =====================================================
-- COMPRESSION POLICIES
-- =====================================================

-- Add compression policies for security tables
SELECT add_compression_policy('security_audit_logs', INTERVAL '7 days');
SELECT add_compression_policy('security_events', INTERVAL '7 days');

-- =====================================================
-- RETENTION POLICIES
-- =====================================================

-- Add retention policies for security data
SELECT add_retention_policy('security_audit_logs', INTERVAL '2555 days'); -- 7 years
SELECT add_retention_policy('security_events', INTERVAL '365 days'); -- 1 year

-- =====================================================
-- SECURITY FUNCTIONS (REFRESHED)
-- =====================================================

-- 1. Function to log security audit events
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

-- 2. Function to check user permissions
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

-- 3. Function to log security events
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

-- 4. Function to rotate secrets
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

-- 5. Function to get security statistics
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

-- =====================================================
-- SECURITY VIEWS (REFRESHED)
-- =====================================================

-- 1. Security Audit Summary View
CREATE OR REPLACE VIEW security_audit_summary AS
SELECT 
    DATE_TRUNC('day', created_at) as audit_date,
    action_type,
    COUNT(*) as total_actions,
    COUNT(*) FILTER (WHERE success = true) as successful_actions,
    COUNT(*) FILTER (WHERE success = false) as failed_actions,
    COUNT(DISTINCT user_id) as unique_users,
    COUNT(DISTINCT ip_address) as unique_ips,
    AVG(execution_time_ms) as avg_execution_time_ms
FROM security_audit_logs
GROUP BY DATE_TRUNC('day', created_at), action_type
ORDER BY audit_date DESC, action_type;

-- 2. Security Events Summary View
CREATE OR REPLACE VIEW security_events_summary AS
SELECT 
    DATE_TRUNC('day', created_at) as event_date,
    event_type,
    severity,
    COUNT(*) as total_events,
    COUNT(*) FILTER (WHERE is_resolved = true) as resolved_events,
    COUNT(*) FILTER (WHERE is_resolved = false) as unresolved_events,
    COUNT(DISTINCT user_id) as unique_users,
    COUNT(DISTINCT ip_address) as unique_ips
FROM security_events
GROUP BY DATE_TRUNC('day', created_at), event_type, severity
ORDER BY event_date DESC, event_type, severity;

-- 3. User Access Summary View
CREATE OR REPLACE VIEW user_access_summary AS
SELECT 
    user_id,
    role_name,
    COUNT(*) as access_grants,
    MAX(created_at) as last_granted,
    MAX(expires_at) as latest_expiry,
    BOOL_OR(expires_at < NOW()) as has_expired_access
FROM security_access_control
WHERE is_active = true
GROUP BY user_id, role_name
ORDER BY user_id, role_name;

-- =====================================================
-- DEFAULT SECURITY POLICIES
-- =====================================================

-- Insert default security policies
INSERT INTO security_policies (policy_name, policy_type, policy_config, priority) VALUES
('default_audit_policy', 'audit', '{"enabled": true, "retention_days": 2555, "log_level": "INFO"}', 1),
('default_access_policy', 'access', '{"enabled": true, "session_timeout_minutes": 30, "max_failed_attempts": 5}', 1),
('default_secrets_policy', 'secrets', '{"enabled": true, "rotation_interval_days": 30, "encryption_algorithm": "AES-256"}', 1),
('default_monitoring_policy', 'monitoring', '{"enabled": true, "alert_threshold": 10, "notification_channels": ["email", "slack"]}', 1)
ON CONFLICT (policy_name) DO NOTHING;

-- =====================================================
-- DEFAULT SECURITY ROLES
-- =====================================================

-- Insert default security roles
INSERT INTO security_access_control (user_id, role_name, permissions, resource_scope) VALUES
('system_admin', 'admin', '{"read": true, "write": true, "delete": true, "admin": true}', 'all'),
('system_user', 'user', '{"read": true, "write": false, "delete": false, "admin": false}', 'user_data'),
('system_monitor', 'monitor', '{"read": true, "write": false, "delete": false, "admin": false}', 'monitoring')
ON CONFLICT DO NOTHING;

-- =====================================================
-- MIGRATION LOGGING
-- =====================================================

-- Log the migration execution
INSERT INTO lifecycle_executions (policy_id, execution_type, table_name, execution_status, metadata)
VALUES (NULL, 'migration', 'system', 'success', 
        '{"migration_version": "078", "migration_name": "security_enhancement_phase5_fixed", "tables_created": 5, "functions_created": 5, "views_created": 3}');

-- =====================================================
-- COMPLETION NOTIFICATION
-- =====================================================

DO $$
BEGIN
    RAISE NOTICE 'Phase 5 Security Enhancement migration (FIXED) completed successfully';
    RAISE NOTICE 'Created 5 security tables with proper hypertables and compression';
    RAISE NOTICE 'Created 5 security functions for audit, access control, and monitoring';
    RAISE NOTICE 'Created 3 security views for reporting and analysis';
    RAISE NOTICE 'Applied retention and compression policies for security data';
END $$;
