-- Migration: 060_streaming_infrastructure_phase1.sql
-- Description: Phase 1 Streaming Infrastructure - Core streaming tables and TimescaleDB integration
-- Date: 2024-01-XX
-- Author: AlphaPulse Team

-- Enable TimescaleDB extension if not already enabled
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- ============================================================================
-- STREAMING INFRASTRUCTURE TABLES
-- ============================================================================

-- Stream messages table for Redis Streams integration
CREATE TABLE IF NOT EXISTS stream_messages (
    id SERIAL PRIMARY KEY,
    message_id VARCHAR(100) UNIQUE NOT NULL,
    stream_key VARCHAR(200) NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    data_type VARCHAR(50) NOT NULL, -- 'tick', 'candle', 'signal', 'outcome', 'indicator'
    source VARCHAR(100) NOT NULL,
    partition INTEGER DEFAULT 0,
    priority INTEGER DEFAULT 0,
    data JSONB NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    -- Indexes for performance
    INDEX idx_stream_messages_symbol (symbol),
    INDEX idx_stream_messages_data_type (data_type),
    INDEX idx_stream_messages_timestamp (timestamp),
    INDEX idx_stream_messages_stream_key (stream_key)
);

-- Convert to TimescaleDB hypertable
SELECT create_hypertable('stream_messages', 'timestamp', if_not_exists => TRUE);

-- Set compression policy
SELECT add_compression_policy('stream_messages', INTERVAL '1 day');

-- Set retention policy (keep data for 30 days)
SELECT add_retention_policy('stream_messages', INTERVAL '30 days');

-- ============================================================================
-- NORMALIZED DATA TABLES
-- ============================================================================

-- Normalized data table for validated and processed stream data
CREATE TABLE IF NOT EXISTS normalized_data (
    id SERIAL PRIMARY KEY,
    original_message_id VARCHAR(100) NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    data_type VARCHAR(50) NOT NULL,
    validation_status VARCHAR(20) NOT NULL, -- 'valid', 'invalid', 'duplicate', 'outlier'
    confidence_score DECIMAL(5,4) NOT NULL,
    processing_time_ms DECIMAL(10,2) NOT NULL,
    normalized_data JSONB NOT NULL,
    validation_errors JSONB,
    metadata JSONB,
    timestamp TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    -- Indexes
    INDEX idx_normalized_data_symbol (symbol),
    INDEX idx_normalized_data_status (validation_status),
    INDEX idx_normalized_data_timestamp (timestamp),
    INDEX idx_normalized_data_confidence (confidence_score)
);

-- Convert to TimescaleDB hypertable
SELECT create_hypertable('normalized_data', 'timestamp', if_not_exists => TRUE);

-- Set compression policy
SELECT add_compression_policy('normalized_data', INTERVAL '1 day');

-- Set retention policy (keep data for 90 days)
SELECT add_retention_policy('normalized_data', INTERVAL '90 days');

-- ============================================================================
-- CANDLE DATA TABLES
-- ============================================================================

-- Real-time candles table
CREATE TABLE IF NOT EXISTS realtime_candles (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    timeframe VARCHAR(10) NOT NULL, -- '1m', '5m', '15m', '1h', '4h', '1d'
    open_time TIMESTAMPTZ NOT NULL,
    close_time TIMESTAMPTZ NOT NULL,
    open_price DECIMAL(20,8) NOT NULL,
    high_price DECIMAL(20,8) NOT NULL,
    low_price DECIMAL(20,8) NOT NULL,
    close_price DECIMAL(20,8) NOT NULL,
    volume DECIMAL(20,8) NOT NULL,
    trade_count INTEGER DEFAULT 0,
    vwap DECIMAL(20,8) DEFAULT 0,
    metadata JSONB,
    timestamp TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    -- Unique constraint to prevent duplicates
    UNIQUE(symbol, timeframe, open_time),
    
    -- Indexes
    INDEX idx_realtime_candles_symbol_timeframe (symbol, timeframe),
    INDEX idx_realtime_candles_open_time (open_time),
    INDEX idx_realtime_candles_close_time (close_time)
);

-- Convert to TimescaleDB hypertable
SELECT create_hypertable('realtime_candles', 'open_time', if_not_exists => TRUE);

-- Set compression policy
SELECT add_compression_policy('realtime_candles', INTERVAL '1 day');

-- Set retention policy (keep data for 1 year)
SELECT add_retention_policy('realtime_candles', INTERVAL '1 year');

-- ============================================================================
-- ROLLING STATE TABLES
-- ============================================================================

-- Rolling windows table for in-memory state persistence
CREATE TABLE IF NOT EXISTS rolling_windows (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    data_type VARCHAR(50) NOT NULL, -- 'price', 'volume', 'candle', 'indicator'
    window_size INTEGER NOT NULL,
    data JSONB NOT NULL, -- Array of values
    last_update TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    -- Unique constraint
    UNIQUE(symbol, timeframe, data_type),
    
    -- Indexes
    INDEX idx_rolling_windows_symbol (symbol),
    INDEX idx_rolling_windows_timeframe (timeframe),
    INDEX idx_rolling_windows_last_update (last_update)
);

-- Technical indicators table
CREATE TABLE IF NOT EXISTS technical_indicators (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    indicator_name VARCHAR(50) NOT NULL, -- 'SMA', 'EMA', 'RSI', 'MACD', etc.
    value DECIMAL(20,8) NOT NULL,
    parameters JSONB NOT NULL, -- Indicator parameters
    metadata JSONB,
    timestamp TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    -- Indexes
    INDEX idx_technical_indicators_symbol (symbol),
    INDEX idx_technical_indicators_name (indicator_name),
    INDEX idx_technical_indicators_timestamp (timestamp)
);

-- Convert to TimescaleDB hypertable
SELECT create_hypertable('technical_indicators', 'timestamp', if_not_exists => TRUE);

-- Set compression policy
SELECT add_compression_policy('technical_indicators', INTERVAL '1 day');

-- Set retention policy (keep data for 90 days)
SELECT add_retention_policy('technical_indicators', INTERVAL '90 days');

-- ============================================================================
-- STREAMING METRICS TABLES
-- ============================================================================

-- System metrics table
CREATE TABLE IF NOT EXISTS system_metrics (
    id SERIAL PRIMARY KEY,
    cpu_percent DECIMAL(5,2) NOT NULL,
    memory_percent DECIMAL(5,2) NOT NULL,
    memory_used_mb DECIMAL(10,2) NOT NULL,
    memory_available_mb DECIMAL(10,2) NOT NULL,
    disk_usage_percent DECIMAL(5,2) NOT NULL,
    network_bytes_sent BIGINT NOT NULL,
    network_bytes_recv BIGINT NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Convert to TimescaleDB hypertable
SELECT create_hypertable('system_metrics', 'timestamp', if_not_exists => TRUE);

-- Set compression policy
SELECT add_compression_policy('system_metrics', INTERVAL '1 hour');

-- Set retention policy (keep data for 30 days)
SELECT add_retention_policy('system_metrics', INTERVAL '30 days');

-- Component metrics table
CREATE TABLE IF NOT EXISTS component_metrics (
    id SERIAL PRIMARY KEY,
    component_name VARCHAR(100) NOT NULL,
    metric_name VARCHAR(100) NOT NULL,
    metric_value DECIMAL(20,8) NOT NULL,
    metric_type VARCHAR(20) NOT NULL, -- 'counter', 'gauge', 'histogram'
    metadata JSONB,
    timestamp TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    -- Indexes
    INDEX idx_component_metrics_component (component_name),
    INDEX idx_component_metrics_name (metric_name),
    INDEX idx_component_metrics_timestamp (timestamp)
);

-- Convert to TimescaleDB hypertable
SELECT create_hypertable('component_metrics', 'timestamp', if_not_exists => TRUE);

-- Set compression policy
SELECT add_compression_policy('component_metrics', INTERVAL '1 hour');

-- Set retention policy (keep data for 30 days)
SELECT add_retention_policy('component_metrics', INTERVAL '30 days');

-- Alerts table
CREATE TABLE IF NOT EXISTS streaming_alerts (
    id SERIAL PRIMARY KEY,
    severity VARCHAR(20) NOT NULL, -- 'info', 'warning', 'error', 'critical'
    metric_name VARCHAR(100) NOT NULL,
    metric_value DECIMAL(20,8) NOT NULL,
    threshold_value DECIMAL(20,8) NOT NULL,
    description TEXT NOT NULL,
    resolved BOOLEAN DEFAULT FALSE,
    resolved_at TIMESTAMPTZ,
    timestamp TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    -- Indexes
    INDEX idx_streaming_alerts_severity (severity),
    INDEX idx_streaming_alerts_metric (metric_name),
    INDEX idx_streaming_alerts_resolved (resolved),
    INDEX idx_streaming_alerts_timestamp (timestamp)
);

-- Convert to TimescaleDB hypertable
SELECT create_hypertable('streaming_alerts', 'timestamp', if_not_exists => TRUE);

-- Set compression policy
SELECT add_compression_policy('streaming_alerts', INTERVAL '1 day');

-- Set retention policy (keep data for 90 days)
SELECT add_retention_policy('streaming_alerts', INTERVAL '90 days');

-- ============================================================================
-- PROCESSING RESULTS TABLES
-- ============================================================================

-- Processing results table
CREATE TABLE IF NOT EXISTS processing_results (
    id SERIAL PRIMARY KEY,
    message_id VARCHAR(100) NOT NULL,
    success BOOLEAN NOT NULL,
    processing_time_ms DECIMAL(10,2) NOT NULL,
    components_processed JSONB NOT NULL, -- Array of component names
    errors JSONB, -- Array of error messages
    metadata JSONB,
    timestamp TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    -- Indexes
    INDEX idx_processing_results_message (message_id),
    INDEX idx_processing_results_success (success),
    INDEX idx_processing_results_timestamp (timestamp)
);

-- Convert to TimescaleDB hypertable
SELECT create_hypertable('processing_results', 'timestamp', if_not_exists => TRUE);

-- Set compression policy
SELECT add_compression_policy('processing_results', INTERVAL '1 day');

-- Set retention policy (keep data for 30 days)
SELECT add_retention_policy('processing_results', INTERVAL '30 days');

-- ============================================================================
-- VIEWS FOR MONITORING
-- ============================================================================

-- Real-time streaming status view
CREATE OR REPLACE VIEW streaming_status AS
SELECT 
    'stream_messages' as table_name,
    COUNT(*) as total_records,
    MAX(timestamp) as latest_record,
    NOW() - MAX(timestamp) as age_latest_record
FROM stream_messages
UNION ALL
SELECT 
    'normalized_data' as table_name,
    COUNT(*) as total_records,
    MAX(timestamp) as latest_record,
    NOW() - MAX(timestamp) as age_latest_record
FROM normalized_data
UNION ALL
SELECT 
    'realtime_candles' as table_name,
    COUNT(*) as total_records,
    MAX(open_time) as latest_record,
    NOW() - MAX(open_time) as age_latest_record
FROM realtime_candles
UNION ALL
SELECT 
    'technical_indicators' as table_name,
    COUNT(*) as total_records,
    MAX(timestamp) as latest_record,
    NOW() - MAX(timestamp) as age_latest_record
FROM technical_indicators;

-- System health view
CREATE OR REPLACE VIEW system_health AS
SELECT 
    timestamp,
    cpu_percent,
    memory_percent,
    disk_usage_percent,
    CASE 
        WHEN cpu_percent > 80 THEN 'HIGH'
        WHEN cpu_percent > 60 THEN 'MEDIUM'
        ELSE 'LOW'
    END as cpu_status,
    CASE 
        WHEN memory_percent > 85 THEN 'HIGH'
        WHEN memory_percent > 70 THEN 'MEDIUM'
        ELSE 'LOW'
    END as memory_status,
    CASE 
        WHEN disk_usage_percent > 90 THEN 'HIGH'
        WHEN disk_usage_percent > 80 THEN 'MEDIUM'
        ELSE 'LOW'
    END as disk_status
FROM system_metrics
ORDER BY timestamp DESC
LIMIT 100;

-- Processing performance view
CREATE OR REPLACE VIEW processing_performance AS
SELECT 
    DATE_TRUNC('hour', timestamp) as hour,
    COUNT(*) as total_messages,
    COUNT(CASE WHEN success = true THEN 1 END) as successful_messages,
    COUNT(CASE WHEN success = false THEN 1 END) as failed_messages,
    AVG(processing_time_ms) as avg_processing_time_ms,
    MAX(processing_time_ms) as max_processing_time_ms,
    (COUNT(CASE WHEN success = false THEN 1 END) * 100.0 / COUNT(*)) as error_rate_percent
FROM processing_results
WHERE timestamp >= NOW() - INTERVAL '24 hours'
GROUP BY DATE_TRUNC('hour', timestamp)
ORDER BY hour DESC;

-- ============================================================================
-- FUNCTIONS FOR DATA MANAGEMENT
-- ============================================================================

-- Function to clean up old data
CREATE OR REPLACE FUNCTION cleanup_old_streaming_data()
RETURNS void AS $$
BEGIN
    -- Clean up old stream messages (older than 30 days)
    DELETE FROM stream_messages 
    WHERE timestamp < NOW() - INTERVAL '30 days';
    
    -- Clean up old normalized data (older than 90 days)
    DELETE FROM normalized_data 
    WHERE timestamp < NOW() - INTERVAL '90 days';
    
    -- Clean up old processing results (older than 30 days)
    DELETE FROM processing_results 
    WHERE timestamp < NOW() - INTERVAL '30 days';
    
    -- Clean up resolved alerts (older than 90 days)
    DELETE FROM streaming_alerts 
    WHERE resolved = true AND timestamp < NOW() - INTERVAL '90 days';
    
    RAISE NOTICE 'Cleanup completed at %', NOW();
END;
$$ LANGUAGE plpgsql;

-- Function to get streaming statistics
CREATE OR REPLACE FUNCTION get_streaming_stats()
RETURNS TABLE (
    table_name TEXT,
    total_records BIGINT,
    latest_record TIMESTAMPTZ,
    age_latest_record INTERVAL
) AS $$
BEGIN
    RETURN QUERY
    SELECT * FROM streaming_status;
END;
$$ LANGUAGE plpgsql;

-- Function to get system health summary
CREATE OR REPLACE FUNCTION get_system_health_summary()
RETURNS TABLE (
    cpu_avg DECIMAL,
    memory_avg DECIMAL,
    disk_avg DECIMAL,
    cpu_max DECIMAL,
    memory_max DECIMAL,
    disk_max DECIMAL,
    records_count BIGINT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        AVG(cpu_percent) as cpu_avg,
        AVG(memory_percent) as memory_avg,
        AVG(disk_usage_percent) as disk_avg,
        MAX(cpu_percent) as cpu_max,
        MAX(memory_percent) as memory_max,
        MAX(disk_usage_percent) as disk_max,
        COUNT(*) as records_count
    FROM system_metrics
    WHERE timestamp >= NOW() - INTERVAL '1 hour';
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- INDEXES FOR PERFORMANCE
-- ============================================================================

-- Additional composite indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_stream_messages_symbol_type_time 
ON stream_messages (symbol, data_type, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_normalized_data_symbol_status_time 
ON normalized_data (symbol, validation_status, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_realtime_candles_symbol_timeframe_time 
ON realtime_candles (symbol, timeframe, open_time DESC);

CREATE INDEX IF NOT EXISTS idx_technical_indicators_symbol_name_time 
ON technical_indicators (symbol, indicator_name, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_component_metrics_component_name_time 
ON component_metrics (component_name, metric_name, timestamp DESC);

-- ============================================================================
-- MIGRATION COMPLETION
-- ============================================================================

-- Log migration completion
INSERT INTO migration_log (migration_name, applied_at, status, details)
VALUES (
    '060_streaming_infrastructure_phase1',
    NOW(),
    'SUCCESS',
    'Phase 1 Streaming Infrastructure - Core streaming tables, TimescaleDB integration, and monitoring views created successfully'
);

-- Grant permissions to application user
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO alpha_emon;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO alpha_emon;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO alpha_emon;

-- Print completion message
DO $$
BEGIN
    RAISE NOTICE 'Migration 060_streaming_infrastructure_phase1 completed successfully';
    RAISE NOTICE 'Created tables: stream_messages, normalized_data, realtime_candles, rolling_windows, technical_indicators, system_metrics, component_metrics, streaming_alerts, processing_results';
    RAISE NOTICE 'Created views: streaming_status, system_health, processing_performance';
    RAISE NOTICE 'Created functions: cleanup_old_streaming_data, get_streaming_stats, get_system_health_summary';
    RAISE NOTICE 'TimescaleDB hypertables and compression policies configured';
END $$;
