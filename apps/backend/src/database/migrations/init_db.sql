-- Database Initialization Script for AlphaPlus
-- Creates tables and adds sample data

-- Create trades table
CREATE TABLE IF NOT EXISTS trades (
    id SERIAL PRIMARY KEY,
    signal_id VARCHAR(100) NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    side VARCHAR(10) NOT NULL,
    entry_price DECIMAL(15,8) NOT NULL,
    exit_price DECIMAL(15,8),
    quantity DECIMAL(15,8) NOT NULL,
    leverage INTEGER DEFAULT 1,
    pnl DECIMAL(15,8),
    pnl_percentage DECIMAL(10,4),
    strategy_name VARCHAR(100) NOT NULL,
    status VARCHAR(20) DEFAULT 'open',
    entry_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    exit_time TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create performance_metrics table
CREATE TABLE IF NOT EXISTS performance_metrics (
    id SERIAL PRIMARY KEY,
    var_95 DECIMAL(10,4),
    max_drawdown DECIMAL(10,4),
    sharpe_ratio DECIMAL(10,4),
    sortino_ratio DECIMAL(10,4),
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create signals table
CREATE TABLE IF NOT EXISTS signals (
    id SERIAL PRIMARY KEY,
    signal_id VARCHAR(100) UNIQUE NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    direction VARCHAR(10) NOT NULL,
    confidence DECIMAL(5,4) NOT NULL,
    entry_price DECIMAL(15,8) NOT NULL,
    stop_loss DECIMAL(15,8),
    take_profit DECIMAL(15,8),
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol);
CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status);
CREATE INDEX IF NOT EXISTS idx_trades_entry_time ON trades(entry_time);
CREATE INDEX IF NOT EXISTS idx_signals_symbol ON signals(symbol);
CREATE INDEX IF NOT EXISTS idx_signals_timestamp ON signals(timestamp);

-- Insert sample data
INSERT INTO trades (signal_id, symbol, side, entry_price, exit_price, quantity, pnl, pnl_percentage, strategy_name, status, entry_time, exit_time) VALUES
('SIG001', 'BTC/USDT', 'long', 45000.0, 46500.0, 0.1, 150.0, 3.33, 'mean_reversion', 'closed', CURRENT_TIMESTAMP - INTERVAL '2 days', CURRENT_TIMESTAMP - INTERVAL '1 day'),
('SIG002', 'ETH/USDT', 'short', 2800.0, 2750.0, 1.0, 50.0, 1.79, 'momentum', 'closed', CURRENT_TIMESTAMP - INTERVAL '3 days', CURRENT_TIMESTAMP - INTERVAL '2 days'),
('SIG003', 'BTC/USDT', 'long', 46000.0, NULL, 0.05, NULL, NULL, 'trend_following', 'open', CURRENT_TIMESTAMP - INTERVAL '6 hours', NULL);

INSERT INTO performance_metrics (var_95, max_drawdown, sharpe_ratio, sortino_ratio) VALUES
(2.5, 8.2, 1.85, 2.1);

INSERT INTO signals (signal_id, symbol, timeframe, direction, confidence, entry_price, stop_loss, take_profit) VALUES
('SIG001', 'BTC/USDT', '1h', 'long', 0.85, 45000.0, 44000.0, 47000.0),
('SIG002', 'ETH/USDT', '4h', 'short', 0.78, 2800.0, 2850.0, 2700.0);

-- Display the created data
SELECT 'Trades' as table_name, COUNT(*) as record_count FROM trades
UNION ALL
SELECT 'Performance Metrics' as table_name, COUNT(*) as record_count FROM performance_metrics
UNION ALL
SELECT 'Signals' as table_name, COUNT(*) as record_count FROM signals;
