-- Enhanced Strategy Tables Migration for AlphaPlus
-- Adds support for ensemble learning, performance tracking, and advanced strategy management

-- 1. Enhanced Strategy Configurations Table
CREATE TABLE IF NOT EXISTS enhanced_strategy_configs (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    strategy_id VARCHAR(100) NOT NULL,
    strategy_name VARCHAR(100) NOT NULL,
    strategy_type VARCHAR(50) NOT NULL,
    symbol VARCHAR(20),
    timeframe VARCHAR(10),
    parameters JSONB NOT NULL,
    ensemble_weight FLOAT DEFAULT 1.0,
    is_active BOOLEAN DEFAULT TRUE,
    min_confidence_threshold FLOAT DEFAULT 0.6,
    max_daily_trades INTEGER DEFAULT 10,
    risk_parameters JSONB,
    performance_metrics JSONB,
    market_regime_preferences JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create hypertable for enhanced strategy configs
SELECT create_hypertable('enhanced_strategy_configs', 'timestamp', 
    if_not_exists => TRUE, 
    chunk_time_interval => INTERVAL '1 day'
);

-- Indexes for enhanced strategy configs
CREATE INDEX IF NOT EXISTS idx_enhanced_strategy_configs_strategy_symbol 
ON enhanced_strategy_configs (strategy_id, symbol, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_enhanced_strategy_configs_type_active 
ON enhanced_strategy_configs (strategy_type, is_active, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_enhanced_strategy_configs_ensemble_weight 
ON enhanced_strategy_configs (ensemble_weight DESC, timestamp DESC);

-- 2. Ensemble Learning Results Table
CREATE TABLE IF NOT EXISTS ensemble_predictions (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    predicted_strategy VARCHAR(100) NOT NULL,
    confidence FLOAT NOT NULL,
    predicted_performance FLOAT,
    market_regime VARCHAR(50) NOT NULL,
    features JSONB,
    actual_performance FLOAT,
    prediction_accuracy BOOLEAN,
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create hypertable for ensemble predictions
SELECT create_hypertable('ensemble_predictions', 'timestamp', 
    if_not_exists => TRUE, 
    chunk_time_interval => INTERVAL '1 hour'
);

-- Indexes for ensemble predictions
CREATE INDEX IF NOT EXISTS idx_ensemble_predictions_symbol_time 
ON ensemble_predictions (symbol, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_ensemble_predictions_strategy_confidence 
ON ensemble_predictions (predicted_strategy, confidence DESC, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_ensemble_predictions_regime 
ON ensemble_predictions (market_regime, timestamp DESC);

-- 3. Strategy Performance History Table
CREATE TABLE IF NOT EXISTS strategy_performance_history (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    strategy_name VARCHAR(100) NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    market_regime VARCHAR(50) NOT NULL,
    win_rate FLOAT NOT NULL,
    profit_factor FLOAT NOT NULL,
    max_drawdown FLOAT NOT NULL,
    total_trades INTEGER NOT NULL,
    avg_profit FLOAT NOT NULL,
    sharpe_ratio FLOAT,
    sortino_ratio FLOAT,
    calmar_ratio FLOAT,
    success BOOLEAN NOT NULL,
    performance_metrics JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create hypertable for strategy performance history
SELECT create_hypertable('strategy_performance_history', 'timestamp', 
    if_not_exists => TRUE, 
    chunk_time_interval => INTERVAL '1 hour'
);

-- Indexes for strategy performance history
CREATE INDEX IF NOT EXISTS idx_strategy_performance_history_strategy_symbol 
ON strategy_performance_history (strategy_name, symbol, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_strategy_performance_history_regime 
ON strategy_performance_history (market_regime, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_strategy_performance_history_win_rate 
ON strategy_performance_history (win_rate DESC, timestamp DESC);

-- 4. In-Memory Processing Cache Table
CREATE TABLE IF NOT EXISTS in_memory_cache (
    id SERIAL PRIMARY KEY,
    cache_key VARCHAR(255) UNIQUE NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    data_type VARCHAR(50) NOT NULL,
    cached_data JSONB NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    expires_at TIMESTAMPTZ NOT NULL,
    last_accessed TIMESTAMPTZ DEFAULT NOW(),
    access_count INTEGER DEFAULT 0
);

-- Indexes for in-memory cache
CREATE INDEX IF NOT EXISTS idx_in_memory_cache_key 
ON in_memory_cache (cache_key);

CREATE INDEX IF NOT EXISTS idx_in_memory_cache_symbol_timeframe 
ON in_memory_cache (symbol, timeframe, data_type);

CREATE INDEX IF NOT EXISTS idx_in_memory_cache_expires 
ON in_memory_cache (expires_at);

-- 5. Parallel Execution Tasks Table
CREATE TABLE IF NOT EXISTS parallel_execution_tasks (
    id SERIAL PRIMARY KEY,
    task_id VARCHAR(100) UNIQUE NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    strategy_name VARCHAR(100) NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    priority INTEGER DEFAULT 1,
    status VARCHAR(20) DEFAULT 'pending',
    parameters JSONB,
    result JSONB,
    processing_time_ms FLOAT,
    error_message TEXT,
    worker_id VARCHAR(50),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ
);

-- Create hypertable for parallel execution tasks
SELECT create_hypertable('parallel_execution_tasks', 'timestamp', 
    if_not_exists => TRUE, 
    chunk_time_interval => INTERVAL '1 hour'
);

-- Indexes for parallel execution tasks
CREATE INDEX IF NOT EXISTS idx_parallel_execution_tasks_status 
ON parallel_execution_tasks (status, priority DESC, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_parallel_execution_tasks_strategy 
ON parallel_execution_tasks (strategy_name, symbol, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_parallel_execution_tasks_worker 
ON parallel_execution_tasks (worker_id, timestamp DESC);

-- 6. Market Microstructure Data Table
CREATE TABLE IF NOT EXISTS market_microstructure (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    order_book_imbalance FLOAT,
    bid_ask_spread FLOAT,
    liquidity_score FLOAT,
    volume_profile JSONB,
    support_resistance_levels JSONB,
    market_depth JSONB,
    volatility_regime VARCHAR(50),
    microstructure_features JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create hypertable for market microstructure
SELECT create_hypertable('market_microstructure', 'timestamp', 
    if_not_exists => TRUE, 
    chunk_time_interval => INTERVAL '1 minute'
);

-- Indexes for market microstructure
CREATE INDEX IF NOT EXISTS idx_market_microstructure_symbol_time 
ON market_microstructure (symbol, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_market_microstructure_imbalance 
ON market_microstructure (order_book_imbalance DESC, timestamp DESC);

-- 7. Adaptive Parameter Tuning Table
CREATE TABLE IF NOT EXISTS adaptive_parameter_tuning (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    strategy_name VARCHAR(100) NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    parameter_name VARCHAR(100) NOT NULL,
    old_value FLOAT,
    new_value FLOAT,
    tuning_method VARCHAR(50) NOT NULL,
    performance_improvement FLOAT,
    market_regime VARCHAR(50),
    tuning_metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create hypertable for adaptive parameter tuning
SELECT create_hypertable('adaptive_parameter_tuning', 'timestamp', 
    if_not_exists => TRUE, 
    chunk_time_interval => INTERVAL '1 hour'
);

-- Indexes for adaptive parameter tuning
CREATE INDEX IF NOT EXISTS idx_adaptive_parameter_tuning_strategy 
ON adaptive_parameter_tuning (strategy_name, symbol, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_adaptive_parameter_tuning_improvement 
ON adaptive_parameter_tuning (performance_improvement DESC, timestamp DESC);

-- 8. Risk Clustering Table
CREATE TABLE IF NOT EXISTS risk_clustering (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    cluster_id VARCHAR(100) NOT NULL,
    cluster_type VARCHAR(50) NOT NULL,
    symbols JSONB NOT NULL,
    correlation_matrix JSONB,
    risk_score FLOAT NOT NULL,
    max_exposure FLOAT NOT NULL,
    current_exposure FLOAT NOT NULL,
    cluster_metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create hypertable for risk clustering
SELECT create_hypertable('risk_clustering', 'timestamp', 
    if_not_exists => TRUE, 
    chunk_time_interval => INTERVAL '1 hour'
);

-- Indexes for risk clustering
CREATE INDEX IF NOT EXISTS idx_risk_clustering_cluster_id 
ON risk_clustering (cluster_id, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_risk_clustering_risk_score 
ON risk_clustering (risk_score DESC, timestamp DESC);

-- 9. Slippage Modeling Table
CREATE TABLE IF NOT EXISTS slippage_modeling (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    order_size FLOAT NOT NULL,
    order_type VARCHAR(20) NOT NULL,
    predicted_slippage FLOAT NOT NULL,
    actual_slippage FLOAT,
    market_conditions JSONB,
    order_book_state JSONB,
    slippage_model_version VARCHAR(50),
    accuracy_score FLOAT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create hypertable for slippage modeling
SELECT create_hypertable('slippage_modeling', 'timestamp', 
    if_not_exists => TRUE, 
    chunk_time_interval => INTERVAL '1 minute'
);

-- Indexes for slippage modeling
CREATE INDEX IF NOT EXISTS idx_slippage_modeling_symbol_time 
ON slippage_modeling (symbol, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_slippage_modeling_accuracy 
ON slippage_modeling (accuracy_score DESC, timestamp DESC);

-- 10. Event-Driven Signal Streaming Table
CREATE TABLE IF NOT EXISTS signal_streaming_events (
    id SERIAL PRIMARY KEY,
    event_id VARCHAR(100) UNIQUE NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    event_type VARCHAR(50) NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    strategy_name VARCHAR(100) NOT NULL,
    signal_data JSONB NOT NULL,
    priority INTEGER DEFAULT 1,
    processed BOOLEAN DEFAULT FALSE,
    processing_time_ms FLOAT,
    event_metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create hypertable for signal streaming events
SELECT create_hypertable('signal_streaming_events', 'timestamp', 
    if_not_exists => TRUE, 
    chunk_time_interval => INTERVAL '1 minute'
);

-- Indexes for signal streaming events
CREATE INDEX IF NOT EXISTS idx_signal_streaming_events_type_priority 
ON signal_streaming_events (event_type, priority DESC, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_signal_streaming_events_processed 
ON signal_streaming_events (processed, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_signal_streaming_events_strategy 
ON signal_streaming_events (strategy_name, symbol, timestamp DESC);

-- 11. Pre-aggregated Strategy Views Table
CREATE TABLE IF NOT EXISTS pre_aggregated_strategy_views (
    id SERIAL PRIMARY KEY,
    aggregation_period VARCHAR(20) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    strategy_name VARCHAR(100) NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    total_signals INTEGER NOT NULL,
    successful_signals INTEGER NOT NULL,
    total_pnl FLOAT NOT NULL,
    avg_confidence FLOAT NOT NULL,
    win_rate FLOAT NOT NULL,
    profit_factor FLOAT NOT NULL,
    max_drawdown FLOAT NOT NULL,
    sharpe_ratio FLOAT,
    market_regime_distribution JSONB,
    aggregated_metrics JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create hypertable for pre-aggregated views
SELECT create_hypertable('pre_aggregated_strategy_views', 'timestamp', 
    if_not_exists => TRUE, 
    chunk_time_interval => INTERVAL '1 hour'
);

-- Indexes for pre-aggregated views
CREATE INDEX IF NOT EXISTS idx_pre_aggregated_strategy_views_period 
ON pre_aggregated_strategy_views (aggregation_period, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_pre_aggregated_strategy_views_strategy 
ON pre_aggregated_strategy_views (strategy_name, symbol, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_pre_aggregated_strategy_views_performance 
ON pre_aggregated_strategy_views (win_rate DESC, profit_factor DESC, timestamp DESC);

-- 12. System Performance Metrics Table
CREATE TABLE IF NOT EXISTS system_performance_metrics (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    metric_name VARCHAR(100) NOT NULL,
    metric_value FLOAT NOT NULL,
    metric_unit VARCHAR(20),
    component VARCHAR(50) NOT NULL,
    sub_component VARCHAR(50),
    threshold_value FLOAT,
    alert_level VARCHAR(20),
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create hypertable for system performance metrics
SELECT create_hypertable('system_performance_metrics', 'timestamp', 
    if_not_exists => TRUE, 
    chunk_time_interval => INTERVAL '1 minute'
);

-- Indexes for system performance metrics
CREATE INDEX IF NOT EXISTS idx_system_performance_metrics_component 
ON system_performance_metrics (component, metric_name, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_system_performance_metrics_alert 
ON system_performance_metrics (alert_level, timestamp DESC);

-- Create views for easier querying

-- View for strategy performance summary
CREATE OR REPLACE VIEW strategy_performance_summary AS
SELECT 
    strategy_name,
    symbol,
    timeframe,
    market_regime,
    COUNT(*) as total_periods,
    AVG(win_rate) as avg_win_rate,
    AVG(profit_factor) as avg_profit_factor,
    AVG(sharpe_ratio) as avg_sharpe_ratio,
    MAX(max_drawdown) as max_drawdown,
    SUM(total_trades) as total_trades,
    AVG(avg_profit) as avg_profit
FROM strategy_performance_history
GROUP BY strategy_name, symbol, timeframe, market_regime;

-- View for ensemble prediction accuracy
CREATE OR REPLACE VIEW ensemble_prediction_accuracy AS
SELECT 
    predicted_strategy,
    market_regime,
    COUNT(*) as total_predictions,
    AVG(confidence) as avg_confidence,
    COUNT(CASE WHEN prediction_accuracy = true THEN 1 END) as correct_predictions,
    COUNT(CASE WHEN prediction_accuracy = true THEN 1 END)::float / COUNT(*) as accuracy_rate
FROM ensemble_predictions
WHERE prediction_accuracy IS NOT NULL
GROUP BY predicted_strategy, market_regime;

-- View for system performance overview
CREATE OR REPLACE VIEW system_performance_overview AS
SELECT 
    component,
    metric_name,
    AVG(metric_value) as avg_value,
    MAX(metric_value) as max_value,
    MIN(metric_value) as min_value,
    COUNT(*) as data_points
FROM system_performance_metrics
WHERE timestamp >= NOW() - INTERVAL '1 hour'
GROUP BY component, metric_name;

-- Insert migration record
INSERT INTO database_migrations (migration_name, applied_at, version) 
VALUES ('002_enhanced_strategy_tables', NOW(), '2.0.0')
ON CONFLICT (migration_name) DO NOTHING;

-- Log migration completion
SELECT 'Migration 002_enhanced_strategy_tables completed successfully' as status;
