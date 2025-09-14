# Production Database Initialization Script
# Optimized for TimescaleDB with performance tuning

-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Create optimized database configuration
ALTER SYSTEM SET shared_buffers = '256MB';
ALTER SYSTEM SET effective_cache_size = '1GB';
ALTER SYSTEM SET maintenance_work_mem = '64MB';
ALTER SYSTEM SET checkpoint_completion_target = 0.9;
ALTER SYSTEM SET wal_buffers = '16MB';
ALTER SYSTEM SET default_statistics_target = 100;
ALTER SYSTEM SET random_page_cost = 1.1;
ALTER SYSTEM SET effective_io_concurrency = 200;

-- Create optimized tables for real-time data
CREATE TABLE IF NOT EXISTS market_data (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    open DECIMAL(20,8),
    high DECIMAL(20,8),
    low DECIMAL(20,8),
    close DECIMAL(20,8),
    volume DECIMAL(20,8),
    trade_count INTEGER,
    vwap DECIMAL(20,8),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create hypertable for time-series optimization
SELECT create_hypertable('market_data', 'timestamp', 
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE);

-- Create indexes for performance
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_market_data_symbol_timestamp 
    ON market_data (symbol, timestamp DESC);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_market_data_timestamp 
    ON market_data USING btree (timestamp DESC);

-- Create real-time trade data table
CREATE TABLE IF NOT EXISTS trade_data (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    trade_id BIGINT,
    price DECIMAL(20,8) NOT NULL,
    quantity DECIMAL(20,8) NOT NULL,
    is_buyer_maker BOOLEAN,
    buyer_order_id BIGINT,
    seller_order_id BIGINT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create hypertable for trade data
SELECT create_hypertable('trade_data', 'timestamp', 
    chunk_time_interval => INTERVAL '1 hour',
    if_not_exists => TRUE);

-- Create indexes for trade data
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_trade_data_symbol_timestamp 
    ON trade_data (symbol, timestamp DESC);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_trade_data_timestamp 
    ON trade_data USING btree (timestamp DESC);

-- Create order book data table
CREATE TABLE IF NOT EXISTS order_book_data (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    side VARCHAR(4) NOT NULL CHECK (side IN ('bid', 'ask')),
    price DECIMAL(20,8) NOT NULL,
    quantity DECIMAL(20,8) NOT NULL,
    level INTEGER NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create hypertable for order book data
SELECT create_hypertable('order_book_data', 'timestamp', 
    chunk_time_interval => INTERVAL '1 hour',
    if_not_exists => TRUE);

-- Create indexes for order book data
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_order_book_symbol_timestamp 
    ON order_book_data (symbol, timestamp DESC);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_order_book_symbol_side_level 
    ON order_book_data (symbol, side, level);

-- Create liquidation data table
CREATE TABLE IF NOT EXISTS liquidation_data (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    side VARCHAR(4) NOT NULL CHECK (side IN ('buy', 'sell')),
    quantity DECIMAL(20,8) NOT NULL,
    price DECIMAL(20,8) NOT NULL,
    liquidation_type VARCHAR(20),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create hypertable for liquidation data
SELECT create_hypertable('liquidation_data', 'timestamp', 
    chunk_time_interval => INTERVAL '1 hour',
    if_not_exists => TRUE);

-- Create indexes for liquidation data
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_liquidation_symbol_timestamp 
    ON liquidation_data (symbol, timestamp DESC);

-- Create trading signals table
CREATE TABLE IF NOT EXISTS trading_signals (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    signal_type VARCHAR(20) NOT NULL,
    signal_strength DECIMAL(5,2),
    price DECIMAL(20,8),
    confidence DECIMAL(5,2),
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create hypertable for trading signals
SELECT create_hypertable('trading_signals', 'timestamp', 
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE);

-- Create indexes for trading signals
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_signals_symbol_timestamp 
    ON trading_signals (symbol, timestamp DESC);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_signals_type_timestamp 
    ON trading_signals (signal_type, timestamp DESC);

-- Create performance metrics table
CREATE TABLE IF NOT EXISTS performance_metrics (
    id BIGSERIAL PRIMARY KEY,
    metric_name VARCHAR(50) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    value DECIMAL(20,8),
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create hypertable for performance metrics
SELECT create_hypertable('performance_metrics', 'timestamp', 
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE);

-- Create indexes for performance metrics
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_performance_name_timestamp 
    ON performance_metrics (metric_name, timestamp DESC);

-- Create continuous aggregates for real-time analytics
CREATE MATERIALIZED VIEW IF NOT EXISTS market_data_1min
WITH (timescaledb.continuous) AS
SELECT 
    symbol,
    time_bucket('1 minute', timestamp) AS bucket,
    first(open, timestamp) AS open,
    max(high) AS high,
    min(low) AS low,
    last(close, timestamp) AS close,
    sum(volume) AS volume,
    avg(vwap) AS vwap,
    count(*) AS trade_count
FROM market_data
GROUP BY symbol, bucket;

-- Create refresh policy for continuous aggregates
SELECT add_continuous_aggregate_policy('market_data_1min',
    start_offset => INTERVAL '1 hour',
    end_offset => INTERVAL '1 minute',
    schedule_interval => INTERVAL '1 minute');

-- Create 5-minute aggregates
CREATE MATERIALIZED VIEW IF NOT EXISTS market_data_5min
WITH (timescaledb.continuous) AS
SELECT 
    symbol,
    time_bucket('5 minutes', timestamp) AS bucket,
    first(open, timestamp) AS open,
    max(high) AS high,
    min(low) AS low,
    last(close, timestamp) AS close,
    sum(volume) AS volume,
    avg(vwap) AS vwap,
    count(*) AS trade_count
FROM market_data
GROUP BY symbol, bucket;

-- Create refresh policy for 5-minute aggregates
SELECT add_continuous_aggregate_policy('market_data_5min',
    start_offset => INTERVAL '1 hour',
    end_offset => INTERVAL '5 minutes',
    schedule_interval => INTERVAL '5 minutes');

-- Create 1-hour aggregates
CREATE MATERIALIZED VIEW IF NOT EXISTS market_data_1hour
WITH (timescaledb.continuous) AS
SELECT 
    symbol,
    time_bucket('1 hour', timestamp) AS bucket,
    first(open, timestamp) AS open,
    max(high) AS high,
    min(low) AS low,
    last(close, timestamp) AS close,
    sum(volume) AS volume,
    avg(vwap) AS vwap,
    count(*) AS trade_count
FROM market_data
GROUP BY symbol, bucket;

-- Create refresh policy for 1-hour aggregates
SELECT add_continuous_aggregate_policy('market_data_1hour',
    start_offset => INTERVAL '1 day',
    end_offset => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour');

-- Create data retention policies
SELECT add_retention_policy('market_data', INTERVAL '30 days');
SELECT add_retention_policy('trade_data', INTERVAL '7 days');
SELECT add_retention_policy('order_book_data', INTERVAL '3 days');
SELECT add_retention_policy('liquidation_data', INTERVAL '30 days');
SELECT add_retention_policy('trading_signals', INTERVAL '90 days');
SELECT add_retention_policy('performance_metrics', INTERVAL '90 days');

-- Create compression policies for older data
SELECT add_compression_policy('market_data', INTERVAL '7 days');
SELECT add_compression_policy('trade_data', INTERVAL '1 day');
SELECT add_compression_policy('order_book_data', INTERVAL '1 day');

-- Create user for application
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'alphapulse_app') THEN
        CREATE ROLE alphapulse_app WITH LOGIN PASSWORD 'secure_app_password';
    END IF;
END
$$;

-- Grant permissions
GRANT CONNECT ON DATABASE alphapulse TO alphapulse_app;
GRANT USAGE ON SCHEMA public TO alphapulse_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO alphapulse_app;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO alphapulse_app;

-- Create read-only user for monitoring
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'alphapulse_monitor') THEN
        CREATE ROLE alphapulse_monitor WITH LOGIN PASSWORD 'secure_monitor_password';
    END IF;
END
$$;

-- Grant read-only permissions for monitoring
GRANT CONNECT ON DATABASE alphapulse TO alphapulse_monitor;
GRANT USAGE ON SCHEMA public TO alphapulse_monitor;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO alphapulse_monitor;

-- Create function for health check
CREATE OR REPLACE FUNCTION health_check()
RETURNS TABLE(status text, timestamp timestamptz) AS $$
BEGIN
    RETURN QUERY SELECT 'healthy'::text, NOW();
END;
$$ LANGUAGE plpgsql;

-- Grant execute permission on health check function
GRANT EXECUTE ON FUNCTION health_check() TO alphapulse_app;
GRANT EXECUTE ON FUNCTION health_check() TO alphapulse_monitor;

-- Create function for getting system stats
CREATE OR REPLACE FUNCTION get_system_stats()
RETURNS TABLE(
    total_tables bigint,
    total_size text,
    hypertables bigint,
    continuous_aggregates bigint
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        (SELECT count(*) FROM information_schema.tables WHERE table_schema = 'public')::bigint,
        pg_size_pretty(pg_database_size(current_database()))::text,
        (SELECT count(*) FROM timescaledb_information.hypertables)::bigint,
        (SELECT count(*) FROM timescaledb_information.continuous_aggregates)::bigint;
END;
$$ LANGUAGE plpgsql;

-- Grant execute permission on system stats function
GRANT EXECUTE ON FUNCTION get_system_stats() TO alphapulse_monitor;

-- Insert initial configuration
INSERT INTO performance_metrics (metric_name, timestamp, value, metadata) VALUES
('database_initialized', NOW(), 1, '{"version": "1.0", "timescaledb": true}'),
('tables_created', NOW(), 6, '{"tables": ["market_data", "trade_data", "order_book_data", "liquidation_data", "trading_signals", "performance_metrics"]}'),
('hypertables_created', NOW(), 6, '{"hypertables": ["market_data", "trade_data", "order_book_data", "liquidation_data", "trading_signals", "performance_metrics"]}'),
('continuous_aggregates_created', NOW(), 3, '{"aggregates": ["market_data_1min", "market_data_5min", "market_data_1hour"]}');

-- Create initial admin user (replace with your credentials)
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'alphapulse_admin') THEN
        CREATE ROLE alphapulse_admin WITH LOGIN PASSWORD 'secure_admin_password' SUPERUSER;
    END IF;
END
$$;

-- Log successful initialization
INSERT INTO performance_metrics (metric_name, timestamp, value, metadata) VALUES
('initialization_complete', NOW(), 1, '{"status": "success", "timestamp": NOW()}');

COMMIT;