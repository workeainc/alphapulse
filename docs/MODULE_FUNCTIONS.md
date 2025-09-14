# AlphaPulse Module Functions Reference

## 📂 Backend Structure

### `backend/config.py`
**Purpose**: Central configuration management with TimescaleDB and Pine Script settings

**Key Functions**:
- `Settings` class with all configuration parameters
- TimescaleDB-specific settings (chunk intervals, compression)
- Pine Script integration settings
- Trading parameters and risk management settings
- API keys and external service configurations

### `backend/database/connection.py`
**Purpose**: TimescaleDB connection management and setup

**Key Functions**:
- `TimescaleDBConnection.initialize()` - Setup database connections
- `TimescaleDBConnection.setup_timescaledb()` - Configure TimescaleDB extensions
- `TimescaleDBConnection._create_market_data_hypertable()` - Create time-series tables
- `TimescaleDBConnection._setup_continuous_aggregates()` - Setup materialized views
- `TimescaleDBConnection._setup_compression_policies()` - Configure data compression
- `TimescaleDBConnection.get_async_session()` - Get async database session
- `TimescaleDBConnection.health_check()` - Database connectivity check
- `TimescaleDBConnection.close()` - Close database connections

### `backend/database/models.py`
**Purpose**: SQLAlchemy ORM models for TimescaleDB

**Key Functions**:
- `Trade` model - Trading transactions and performance
- `MarketData` model - OHLCV time-series data
- `SentimentData` model - Market sentiment information
- `Strategy` model - Strategy performance tracking
- `SystemHealth` model - Bot health monitoring
- `Portfolio` model - Portfolio management
- `create_tables()` - Initialize database schema

### `backend/database/queries.py`
**Purpose**: TimescaleDB-specific optimized queries

**Key Functions**:
- `TimescaleQueries.get_market_data_timeframe()` - Get time-bucketed market data
- `TimescaleQueries.get_latest_market_data()` - Get recent market data
- `TimescaleQueries.get_sentiment_summary()` - Aggregate sentiment data
- `TimescaleQueries.get_trade_performance_summary()` - Trading performance stats
- `TimescaleQueries.get_volatility_analysis()` - Volatility calculations
- `TimescaleQueries.get_support_resistance_levels()` - S/R level detection
- `TimescaleQueries.get_market_regime_analysis()` - Market regime classification
- `TimescaleQueries.get_continuous_aggregate_data()` - Use materialized views
- `TimescaleQueries.cleanup_old_data()` - Data retention management

### `backend/core/logger.py`
**Purpose**: Centralized logging configuration

**Key Functions**:
- `setup_logging()` - Configure logging levels and handlers
- `get_logger()` - Get module-specific logger
- `log_trade_event()` - Log trading activities
- `log_system_health()` - Log system status
- `log_performance_metrics()` - Log performance data

### `backend/core/websocket_manager.py`
**Purpose**: Real-time WebSocket communication

**Key Functions**:
- `WebSocketManager.connect()` - Handle client connections
- `WebSocketManager.disconnect()` - Handle client disconnections
- `WebSocketManager.broadcast()` - Send updates to all clients
- `WebSocketManager.send_to_client()` - Send to specific client
- `WebSocketManager.broadcast_trade_update()` - Real-time trade updates
- `WebSocketManager.broadcast_market_data()` - Real-time market data
- `WebSocketManager.broadcast_system_status()` - System health updates

### `backend/core/utils.py`
**Purpose**: General utility functions

**Key Functions**:
- `calculate_atr()` - Average True Range calculation
- `calculate_ema()` - Exponential Moving Average
- `calculate_rsi()` - Relative Strength Index
- `calculate_macd()` - MACD indicator
- `calculate_bollinger_bands()` - Bollinger Bands
- `format_timestamp()` - Timestamp formatting
- `validate_symbol()` - Symbol validation
- `calculate_position_size()` - Position sizing utilities

### `backend/data/fetcher.py`
**Purpose**: Market data fetching from exchanges

**Key Functions**:
- `MarketDataFetcher.fetch_ohlcv()` - Get OHLCV data from exchanges
- `MarketDataFetcher.fetch_current_price()` - Get current prices
- `MarketDataFetcher.fetch_order_book()` - Get order book data
- `MarketDataFetcher.fetch_volume_profile()` - Volume analysis
- `MarketDataFetcher.fetch_historical_data()` - Historical data retrieval
- `MarketDataFetcher.calculate_btc_dominance()` - BTC dominance calculation
- `MarketDataFetcher.get_exchange_status()` - Exchange health check

### `backend/data/updater.py`
**Purpose**: Continuous data updates and synchronization

**Key Functions**:
- `DataUpdater.start_update_loop()` - Start continuous updates
- `DataUpdater.update_market_data()` - Update market data
- `DataUpdater.update_sentiment_data()` - Update sentiment data
- `DataUpdater.sync_with_timescaledb()` - Sync with database
- `DataUpdater.handle_data_gaps()` - Handle missing data
- `DataUpdater.cleanup_old_data()` - Data retention

### `backend/data/indicators.py`
**Purpose**: Technical indicator calculations

**Key Functions**:
- `calculate_ema()` - Exponential Moving Average
- `calculate_sma()` - Simple Moving Average
- `calculate_rsi()` - Relative Strength Index
- `calculate_macd()` - MACD indicator
- `calculate_bollinger_bands()` - Bollinger Bands
- `calculate_atr()` - Average True Range
- `calculate_stochastic()` - Stochastic Oscillator
- `calculate_williams_r()` - Williams %R
- `calculate_adx()` - Average Directional Index
- `calculate_kama()` - Kaufman Adaptive Moving Average

### `backend/data/regime_detection.py`
**Purpose**: Market regime classification

**Key Functions**:
- `MarketRegimeDetector.detect_regime()` - Main regime detection
- `MarketRegimeDetector.calculate_volatility()` - Volatility measurement
- `MarketRegimeDetector.is_trending()` - Trending market detection
- `MarketRegimeDetector.is_ranging()` - Ranging market detection
- `MarketRegimeDetector.is_volatile()` - Volatile market detection
- `MarketRegimeDetector.get_regime_confidence()` - Regime confidence score
- `MarketRegimeDetector.get_regime_transition()` - Regime change detection

### `backend/data/pine_script_input.py`
**Purpose**: Pine Script signal processing

**Key Functions**:
- `PineScriptProcessor.process_signal()` - Process incoming signals
- `PineScriptProcessor._validate_signal_format()` - Signal validation
- `PineScriptProcessor._parse_signal()` - Signal parsing
- `PineScriptProcessor._validate_signal_logic()` - Logic validation
- `PineScriptProcessor.get_active_signals()` - Get active signals
- `PineScriptProcessor.get_signal_summary()` - Signal summary
- `PineScriptWebhookHandler.handle_webhook()` - Webhook processing
- `PineScriptWebhookHandler._extract_signal_from_webhook()` - Extract signal data

### `backend/strategies/base_strategy.py`
**Purpose**: Base strategy class and common functionality

**Key Functions**:
- `BaseStrategy.generate_signals()` - Abstract signal generation
- `BaseStrategy.calculate_indicators()` - Abstract indicator calculation
- `BaseStrategy.validate_signal()` - Signal validation
- `BaseStrategy.calculate_position_size()` - Position sizing
- `BaseStrategy.calculate_stop_loss()` - Stop loss calculation
- `BaseStrategy.calculate_take_profit()` - Take profit calculation
- `BaseStrategy.get_market_regime()` - Market regime detection
- `BaseStrategy.filter_signals_by_timeframe()` - Multi-timeframe filtering
- `BaseStrategy.update_performance_metrics()` - Performance tracking

### `backend/strategies/trend_following.py`
**Purpose**: Trend-following strategy implementation

**Key Functions**:
- `TrendFollowingStrategy.calculate_indicators()` - EMA, MACD, RSI, ATR
- `TrendFollowingStrategy.generate_signals()` - Trend signal generation
- `TrendFollowingStrategy.validate_signal()` - Trend-specific validation
- `TrendFollowingStrategy.get_market_regime()` - Trending regime detection
- `TrendFollowingStrategy._check_ema_crossover()` - EMA crossover detection
- `TrendFollowingStrategy._check_macd_signal()` - MACD signal validation
- `TrendFollowingStrategy._check_volume_confirmation()` - Volume confirmation

### `backend/strategies/mean_reversion.py`
**Purpose**: Mean reversion strategy implementation

**Key Functions**:
- `MeanReversionStrategy.calculate_indicators()` - RSI, BB, Stochastic, Williams %R
- `MeanReversionStrategy.generate_signals()` - Reversion signal generation
- `MeanReversionStrategy.validate_signal()` - Reversion-specific validation
- `MeanReversionStrategy.get_market_regime()` - Ranging regime detection
- `MeanReversionStrategy._check_rsi_conditions()` - RSI condition checking
- `MeanReversionStrategy._check_bollinger_bands()` - BB signal detection
- `MeanReversionStrategy._check_support_resistance()` - S/R level detection

### `backend/strategies/breakout.py`
**Purpose**: Breakout detection strategy

**Key Functions**:
- `BreakoutDetectionStrategy.calculate_indicators()` - Support/Resistance, Volume, ATR
- `BreakoutDetectionStrategy.generate_signals()` - Breakout signal generation
- `BreakoutDetectionStrategy.validate_signal()` - Breakout-specific validation
- `BreakoutDetectionStrategy.get_market_regime()` - Volatile regime detection
- `BreakoutDetectionStrategy._detect_breakout()` - Breakout detection
- `BreakoutDetectionStrategy._check_volume_confirmation()` - Volume confirmation
- `BreakoutDetectionStrategy._filter_false_breakout()` - False breakout filtering

### `backend/strategies/strategy_manager.py`
**Purpose**: Multi-strategy coordination and confirmation

**Key Functions**:
- `StrategyManager.add_strategy()` - Add strategy to manager
- `StrategyManager.remove_strategy()` - Remove strategy
- `StrategyManager.generate_signals()` - Generate signals from all strategies
- `StrategyManager.filter_by_confluence()` - Multi-strategy confirmation
- `StrategyManager.filter_by_timeframe()` - Multi-timeframe filtering
- `StrategyManager.get_strategy_performance()` - Strategy performance tracking
- `StrategyManager.optimize_strategies()` - Strategy optimization
- `StrategyManager.switch_regime()` - Regime-based strategy switching

### `backend/execution/order_manager.py`
**Purpose**: Order execution and management

**Key Functions**:
- `OrderManager.place_order()` - Place new orders
- `OrderManager.modify_order()` - Modify existing orders
- `OrderManager.cancel_order()` - Cancel orders
- `OrderManager.get_order_status()` - Get order status
- `OrderManager.get_open_orders()` - Get open orders
- `OrderManager.execute_market_order()` - Market order execution
- `OrderManager.execute_limit_order()` - Limit order execution
- `OrderManager.handle_order_fill()` - Order fill handling

### `backend/execution/position_sizing.py`
**Purpose**: Position sizing and risk calculation

**Key Functions**:
- `PositionSizer.calculate_position_size()` - Position size calculation
- `PositionSizer.calculate_risk_amount()` - Risk amount calculation
- `PositionSizer.calculate_leverage()` - Leverage calculation
- `PositionSizer.validate_position_size()` - Position size validation
- `PositionSizer.adjust_for_volatility()` - Volatility adjustment
- `PositionSizer.calculate_portfolio_exposure()` - Portfolio exposure
- `PositionSizer.get_max_position_size()` - Maximum position size

### `backend/execution/sl_tp_manager.py`
**Purpose**: Stop loss and take profit management

**Key Functions**:
- `SLTPManager.calculate_dynamic_sl()` - Dynamic stop loss calculation
- `SLTPManager.calculate_take_profit()` - Take profit calculation
- `SLTPManager.update_trailing_stop()` - Trailing stop updates
- `SLTPManager.check_break_even()` - Break-even stop management
- `SLTPManager.modify_sl_tp()` - Modify SL/TP levels
- `SLTPManager.cancel_sl_tp()` - Cancel SL/TP orders
- `SLTPManager.get_sl_tp_status()` - SL/TP status tracking

### `backend/execution/scaling.py`
**Purpose**: Position scaling logic

**Key Functions**:
- `PositionScaler.scale_in()` - Scale into position
- `PositionScaler.scale_out()` - Scale out of position
- `PositionScaler.calculate_scale_levels()` - Calculate scale levels
- `PositionScaler.check_scale_conditions()` - Check scaling conditions
- `PositionScaler.execute_scale_order()` - Execute scale orders
- `PositionScaler.track_scale_performance()` - Scale performance tracking

### `backend/execution/risk_manager.py`
**Purpose**: Risk management and limits

**Key Functions**:
- `RiskManager.check_daily_loss()` - Daily loss limit checking
- `RiskManager.check_consecutive_losses()` - Consecutive loss tracking
- `RiskManager.check_portfolio_exposure()` - Portfolio exposure limits
- `RiskManager.calculate_var()` - Value at Risk calculation
- `RiskManager.calculate_portfolio_beta()` - Portfolio beta calculation
- `RiskManager.enforce_risk_limits()` - Risk limit enforcement
- `RiskManager.get_risk_summary()` - Risk summary reporting

### `backend/ai/trade_filter.py`
**Purpose**: AI-powered trade quality filtering

**Key Functions**:
- `TradeFilter.evaluate_trade_quality()` - Trade quality evaluation
- `TradeFilter.extract_features()` - Feature extraction
- `TradeFilter.predict_probability()` - Success probability prediction
- `TradeFilter.filter_by_historical_patterns()` - Historical pattern matching
- `TradeFilter.update_model()` - Model updates
- `TradeFilter.get_filter_performance()` - Filter performance tracking

### `backend/ai/model_training.py`
**Purpose**: ML model training and management

**Key Functions**:
- `ModelTrainer.prepare_training_data()` - Data preparation
- `ModelTrainer.train_model()` - Model training
- `ModelTrainer.evaluate_model()` - Model evaluation
- `ModelTrainer.save_model()` - Model persistence
- `ModelTrainer.load_model()` - Model loading
- `ModelTrainer.retrain_model()` - Model retraining
- `ModelTrainer.get_model_performance()` - Model performance metrics

### `backend/ai/feature_engineering.py`
**Purpose**: Feature engineering for ML models

**Key Functions**:
- `FeatureEngineer.create_technical_features()` - Technical indicator features
- `FeatureEngineer.create_sentiment_features()` - Sentiment features
- `FeatureEngineer.create_market_features()` - Market regime features
- `FeatureEngineer.create_volume_features()` - Volume analysis features
- `FeatureEngineer.normalize_features()` - Feature normalization
- `FeatureEngineer.select_features()` - Feature selection
- `FeatureEngineer.create_lag_features()` - Lag feature creation

### `backend/ai/sentiment_analysis.py`
**Purpose**: Sentiment analysis and scoring

**Key Functions**:
- `SentimentAnalyzer.analyze_text()` - Text sentiment analysis
- `SentimentAnalyzer.get_twitter_sentiment()` - Twitter sentiment
- `SentimentAnalyzer.get_reddit_sentiment()` - Reddit sentiment
- `SentimentAnalyzer.get_news_sentiment()` - News sentiment
- `SentimentAnalyzer.aggregate_sentiment()` - Sentiment aggregation
- `SentimentAnalyzer.calculate_sentiment_score()` - Sentiment scoring
- `SentimentAnalyzer.detect_sentiment_trends()` - Sentiment trend detection

### `backend/protection/diagnostics.py`
**Purpose**: System diagnostics and health monitoring

**Key Functions**:
- `SystemDiagnostics.check_api_latency()` - API latency monitoring
- `SystemDiagnostics.check_database_health()` - Database health check
- `SystemDiagnostics.check_memory_usage()` - Memory usage monitoring
- `SystemDiagnostics.check_cpu_usage()` - CPU usage monitoring
- `SystemDiagnostics.check_disk_space()` - Disk space monitoring
- `SystemDiagnostics.check_network_connectivity()` - Network connectivity
- `SystemDiagnostics.generate_health_report()` - Health report generation

### `backend/protection/auto_pause.py`
**Purpose**: Automatic pause functionality

**Key Functions**:
- `AutoPause.check_pause_conditions()` - Pause condition checking
- `AutoPause.pause_trading()` - Pause trading operations
- `AutoPause.resume_trading()` - Resume trading operations
- `AutoPause.handle_emergency_stop()` - Emergency stop handling
- `AutoPause.close_all_positions()` - Close all positions
- `AutoPause.get_pause_status()` - Pause status reporting

### `backend/protection/alerting.py`
**Purpose**: Alert and notification system

**Key Functions**:
- `AlertManager.send_telegram_alert()` - Telegram notifications
- `AlertManager.send_discord_alert()` - Discord notifications
- `AlertManager.send_email_alert()` - Email notifications
- `AlertManager.create_alert()` - Create new alert
- `AlertManager.send_trade_alert()` - Trade-specific alerts
- `AlertManager.send_system_alert()` - System alerts
- `AlertManager.send_performance_alert()` - Performance alerts

### `backend/routes/bot_control.py`
**Purpose**: Trading bot control endpoints

**Key Functions**:
- `@app.post("/api/trading/start")` - Start trading bot
- `@app.post("/api/trading/stop")` - Stop trading bot
- `@app.get("/api/trading/status")` - Get trading status
- `@app.post("/api/trading/pause")` - Pause trading
- `@app.post("/api/trading/resume")` - Resume trading
- `@app.get("/api/trading/performance")` - Get performance metrics

### `backend/routes/settings.py`
**Purpose**: Configuration and settings endpoints

**Key Functions**:
- `@app.get("/api/settings")` - Get current settings
- `@app.put("/api/settings")` - Update settings
- `@app.get("/api/settings/strategies")` - Get strategy settings
- `@app.put("/api/settings/strategies")` - Update strategy settings
- `@app.get("/api/settings/risk")` - Get risk settings
- `@app.put("/api/settings/risk")` - Update risk settings

### `backend/routes/data_api.py`
**Purpose**: Data access endpoints

**Key Functions**:
- `@app.get("/api/data/market/{symbol}")` - Get market data
- `@app.get("/api/data/sentiment/{symbol}")` - Get sentiment data
- `@app.get("/api/data/indicators/{symbol}")` - Get technical indicators
- `@app.get("/api/data/history/{symbol}")` - Get historical data
- `@app.get("/api/data/portfolio")` - Get portfolio data
- `@app.get("/api/data/trades")` - Get trade history

## 📂 Frontend Structure

### `frontend/pages/index.js`
**Purpose**: Main dashboard page

**Key Functions**:
- `Dashboard` component - Main dashboard layout
- `useWebSocket()` - WebSocket connection hook
- `useTradingStatus()` - Trading status hook
- `usePortfolioData()` - Portfolio data hook
- `useMarketData()` - Market data hook
- `handleStartTrading()` - Start trading function
- `handleStopTrading()` - Stop trading function

### `frontend/components/ChartLive.js`
**Purpose**: Live trading chart component

**Key Functions**:
- `ChartLive` component - Live chart display
- `updateChartData()` - Update chart data
- `addTradeMarker()` - Add trade markers
- `updateIndicators()` - Update technical indicators
- `handleChartInteraction()` - Chart interaction handling

### `frontend/components/SignalFeed.js`
**Purpose**: Live signal feed component

**Key Functions**:
- `SignalFeed` component - Signal feed display
- `addSignal()` - Add new signal
- `filterSignals()` - Filter signals
- `updateSignalStatus()` - Update signal status
- `handleSignalClick()` - Signal click handling

### `frontend/components/StrategyStatus.js`
**Purpose**: Strategy status component

**Key Functions**:
- `StrategyStatus` component - Strategy status display
- `updateStrategyPerformance()` - Update performance
- `toggleStrategy()` - Toggle strategy on/off
- `showStrategyDetails()` - Show strategy details
- `updateStrategySettings()` - Update strategy settings

### `frontend/components/RiskPanel.js`
**Purpose**: Risk management panel

**Key Functions**:
- `RiskPanel` component - Risk panel display
- `updateRiskMetrics()` - Update risk metrics
- `showRiskAlerts()` - Show risk alerts
- `updateRiskSettings()` - Update risk settings
- `showRiskHistory()` - Show risk history

### `frontend/utils/websocket.js`
**Purpose**: WebSocket connection management

**Key Functions**:
- `connectWebSocket()` - Establish WebSocket connection
- `disconnectWebSocket()` - Disconnect WebSocket
- `sendMessage()` - Send WebSocket message
- `handleMessage()` - Handle incoming messages
- `reconnectWebSocket()` - Reconnect WebSocket
- `getConnectionStatus()` - Get connection status

### `frontend/utils/api.js`
**Purpose**: API communication utilities

**Key Functions**:
- `api.get()` - GET request wrapper
- `api.post()` - POST request wrapper
- `api.put()` - PUT request wrapper
- `api.delete()` - DELETE request wrapper
- `api.setAuthToken()` - Set authentication token
- `api.handleError()` - Error handling
- `api.retryRequest()` - Request retry logic

## 📂 Pine Scripts

### `pine_scripts/alpha_pulse_indicator.pine`
**Purpose**: Main multi-strategy indicator

**Key Features**:
- Multi-strategy signal generation
- Market regime detection
- Signal strength classification
- Webhook integration
- Real-time alerts

### `pine_scripts/ema_trend.pine`
**Purpose**: Trend-following strategy

**Key Features**:
- EMA crossover signals
- MACD confirmation
- Volume analysis
- Trend strength calculation

### `pine_scripts/rsi_reversion.pine`
**Purpose**: Mean reversion strategy

**Key Features**:
- RSI oversold/overbought signals
- Bollinger Bands integration
- Divergence detection
- Stochastic and Williams %R

### `pine_scripts/breakout_detector.pine`
**Purpose**: Breakout detection strategy

**Key Features**:
- Support/resistance breakouts
- Volume confirmation
- False breakout filtering
- Consolidation detection

## 📂 Docker Configuration

### `docker/docker-compose.yml`
**Purpose**: Multi-service orchestration

**Services**:
- PostgreSQL with TimescaleDB
- Redis for caching
- Backend FastAPI service
- Frontend Next.js service
- Celery worker for background tasks
- Celery beat for scheduled tasks
- Nginx reverse proxy (optional)

### `docker/Dockerfile.backend`
**Purpose**: Backend service containerization

**Features**:
- Python 3.11 slim image
- Dependencies installation
- Non-root user setup
- Health checks
- Volume mounting

### `docker/Dockerfile.frontend`
**Purpose**: Frontend service containerization

**Features**:
- Multi-stage build
- Node.js 18 Alpine
- Next.js optimization
- Production-ready image
- Static file serving

## 🔧 Utility Scripts

### `start.py`
**Purpose**: Project management script

**Key Functions**:
- `check_requirements()` - Check system requirements
- `create_env_file()` - Create environment file
- `setup_database()` - Database setup
- `build_services()` - Build Docker services
- `start_services()` - Start all services
- `stop_services()` - Stop all services
- `check_status()` - Check service status
- `view_logs()` - View service logs

This comprehensive function list provides a clear roadmap for implementing each module in the AlphaPulse trading bot system. Each function has a specific responsibility and integrates with the overall system architecture.
