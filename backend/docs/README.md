# AlphaPulse Backend - High-Frequency Trading Signal System

## Overview

AlphaPulse is a high-frequency trading signal system designed for crypto/forex/stocks with the following key features:

- **Target Performance**: <50ms latency, 75-85% signal accuracy, 60-80% low-quality signal filtering
- **Market Regime Detection**: Multi-metric classification with ML integration
- **Real-time Processing**: WebSocket streaming with rolling buffers
- **Advanced Analytics**: Technical indicators, pattern detection, and ML scoring
- **Comprehensive Testing**: 100% test coverage with performance benchmarks

## Architecture

### Directory Structure

```
backend/
├── core/                    # Core trading system components
│   ├── alphapulse_core.py   # Main trading system
│   ├── indicators_engine.py # Technical indicators
│   ├── ml_signal_generator.py # ML-based signal generation
│   ├── market_regime_detector.py # Market regime detection
│   ├── optimized_trading_system.py # Optimized trading logic
│   └── websocket_binance.py # WebSocket connections
├── utils/                   # Utility functions and helpers
│   ├── feature_engineering.py # Feature engineering utilities
│   ├── risk_management.py   # Risk management functions
│   ├── threshold_env.py     # Environment configuration
│   ├── utils.py            # General utilities
│   └── config.py           # Unified configuration
├── services/               # Service layer components
│   ├── data_services.py    # Data processing services
│   ├── monitoring_services.py # System monitoring
│   ├── trading_services.py # Trading execution services
│   ├── pattern_services.py # Pattern detection services
│   └── active_learning_service.py # Active learning
├── database/               # Database models and operations
│   ├── models.py           # SQLAlchemy models
│   ├── queries.py          # Database queries
│   ├── connection.py       # Database connections
│   └── migrations/         # Database migrations
├── tests/                  # Comprehensive test suite
│   ├── test_integration.py # Integration tests
│   ├── test_indicators.py  # Indicator tests
│   ├── test_performance.py # Performance tests
│   ├── test_database.py    # Database tests
│   ├── test_edge_cases.py  # Edge case tests
│   ├── test_utils.py       # Test utilities
│   └── conftest.py         # Test configuration
├── ai/                     # AI and ML components
│   ├── advanced_utils.py   # Advanced AI utilities
│   ├── ml_models.py        # ML model management
│   ├── feature_store.py    # Feature store
│   └── deployment.py       # Model deployment
├── strategies/             # Trading strategies
│   ├── pattern_detectors.py # Pattern detection strategies
│   ├── signal_generators.py # Signal generation strategies
│   ├── trend_analyzers.py  # Trend analysis strategies
│   └── strategy_manager.py # Strategy management
├── execution/              # Trade execution
│   ├── trading_engine.py   # Trading engine
│   ├── order_manager.py    # Order management
│   ├── portfolio_manager.py # Portfolio management
│   └── risk_manager.py     # Risk management
├── scripts/                # Utility scripts
│   ├── run_alphapulse.py   # Main runner
│   ├── run_tests.py        # Test runner
│   ├── setup_database.py   # Database setup
│   └── migrate_data.py     # Data migration
└── docs/                   # Documentation
    ├── README.md           # This file
    ├── model_docs.md       # Model documentation
    ├── performance_baseline.md # Performance baselines
    └── api_docs.md         # API documentation
```

## Key Components

### Core Trading System (`core/alphapulse_core.py`)

The main AlphaPulse trading system that orchestrates all components:

```python
from core.alphapulse_core import AlphaPulseCore

# Initialize the system
alphapulse = AlphaPulseCore(
    symbols=['BTC/USDT', 'ETH/USDT'],
    timeframes=['1m', '15m', '1h'],
    enable_regime_detection=True,
    enable_ml=True,
    target_latency_ms=50.0
)

# Start the system
await alphapulse.start()
```

**Features:**
- Real-time signal generation
- Market regime detection
- Performance monitoring
- Signal validation and filtering
- Redis caching for low latency

### Market Regime Detection (`core/market_regime_detector.py`)

Multi-metric market regime classification:

```python
from core.market_regime_detector import MarketRegimeDetector

detector = MarketRegimeDetector(
    symbol='BTC/USDT',
    timeframe='15m',
    redis_client=redis_client
)

# Update regime with new candle data
regime = await detector.update_regime(candle_data)
# Returns: MarketRegime.STRONG_TREND_BULL, etc.
```

**Regime Types:**
- `STRONG_TREND_BULL`: Strong upward trend
- `STRONG_TREND_BEAR`: Strong downward trend
- `WEAK_TREND`: Weak directional movement
- `RANGING`: Sideways movement
- `VOLATILE_BREAKOUT`: High volatility breakout
- `CHOPPY`: Low-quality, choppy movement

### Technical Indicators (`core/indicators_engine.py`)

Comprehensive technical analysis:

```python
from core.indicators_engine import IndicatorsEngine

engine = IndicatorsEngine()
indicators = await engine.calculate_all(candles)

# Available indicators:
# - RSI (14-period with divergence detection)
# - MACD (8-24-9 with signal line)
# - Bollinger Bands (20-period)
# - ATR (14-period)
# - Volume analysis
# - Trend strength
# - Breakout detection
```

### ML Signal Generation (`core/ml_signal_generator.py`)

Machine learning-based signal generation:

```python
from core.ml_signal_generator import MLSignalGenerator

ml_generator = MLSignalGenerator()
signals = await ml_generator.generate_signals(
    symbol='BTC/USDT',
    timeframe='15m',
    candles=candles,
    indicators=indicators
)
```

## Database Schema

### Core Tables

#### Signals Table
```sql
CREATE TABLE signals (
    id SERIAL PRIMARY KEY,
    signal_id VARCHAR(100) UNIQUE NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    direction VARCHAR(10) NOT NULL,
    confidence FLOAT NOT NULL,
    entry_price FLOAT NOT NULL,
    tp1 FLOAT,
    tp2 FLOAT,
    tp3 FLOAT,
    tp4 FLOAT,
    stop_loss FLOAT,
    risk_reward_ratio FLOAT,
    pattern_type VARCHAR(50),
    volume_confirmation BOOLEAN DEFAULT FALSE,
    trend_alignment BOOLEAN DEFAULT FALSE,
    market_regime VARCHAR(50),
    indicators JSONB,
    validation_metrics JSONB,
    timestamp TIMESTAMP DEFAULT NOW(),
    outcome VARCHAR(20) DEFAULT 'pending'
);
```

#### Market Regimes Table
```sql
CREATE TABLE market_regimes (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    regime_type VARCHAR(50) NOT NULL,
    confidence FLOAT,
    duration_candles INTEGER,
    metrics JSONB,
    timestamp TIMESTAMP DEFAULT NOW()
);
```

#### Performance Metrics Table
```sql
CREATE TABLE performance_metrics (
    id SERIAL PRIMARY KEY,
    test_run_id VARCHAR(100) NOT NULL,
    latency_avg_ms FLOAT,
    latency_max_ms FLOAT,
    throughput_signals_sec FLOAT,
    accuracy FLOAT,
    filter_rate FLOAT,
    timestamp TIMESTAMP DEFAULT NOW()
);
```

## Installation and Setup

### Prerequisites

- Python 3.10+
- PostgreSQL 13+
- Redis 6+
- Required Python packages (see `requirements.txt`)

### Quick Start

1. **Clone and setup environment:**
```bash
git clone <repository>
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

2. **Setup database:**
```bash
python scripts/setup_database.py
```

3. **Run tests:**
```bash
python scripts/run_tests.py
```

4. **Start AlphaPulse:**
```bash
python scripts/run_alphapulse.py
```

### Configuration

Create a `.env` file in the backend directory:

```env
# Database
DATABASE_URL=postgresql://user:password@localhost:5432/alphapulse

# Redis
REDIS_URL=redis://localhost:6379

# Trading
SYMBOLS=BTC/USDT,ETH/USDT
TIMEFRAMES=1m,15m,1h
TARGET_LATENCY_MS=50.0

# Logging
LOG_LEVEL=INFO
DEBUG=false
```

## Testing

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test categories
pytest tests/test_integration.py -v
pytest tests/test_performance.py -v
pytest tests/test_database.py -v

# Run with performance benchmarks
pytest tests/ --benchmark-save=alphapulse
```

### Test Categories

1. **Integration Tests** (`test_integration.py`)
   - End-to-end pipeline testing
   - Component integration
   - Error handling and recovery

2. **Performance Tests** (`test_performance.py`)
   - Latency benchmarks
   - Throughput testing
   - Memory usage monitoring

3. **Database Tests** (`test_database.py`)
   - Schema validation
   - Query performance
   - Data integrity

4. **Edge Case Tests** (`test_edge_cases.py`)
   - Error scenarios
   - Boundary conditions
   - Stress testing

## Performance Targets

### Latency Targets
- **Target**: <50ms average latency
- **Peak**: <100ms maximum latency
- **Measurement**: End-to-end signal processing

### Accuracy Targets
- **Signal Accuracy**: 75-85%
- **Filter Rate**: 60-80% (low-quality signal rejection)
- **Regime Detection**: >80% accuracy

### Throughput Targets
- **Signals per Second**: >10 signals/sec
- **Candles per Second**: >100 candles/sec
- **Memory Usage**: <1GB total

## Monitoring and Metrics

### Performance Metrics

The system tracks comprehensive performance metrics:

```python
# Get performance summary
summary = alphapulse.get_performance_summary()

# Metrics include:
# - uptime_seconds
# - total_signals
# - avg_latency_ms
# - max_latency_ms
# - min_latency_ms
# - throughput_signals_sec
# - target_latency_ms
# - latency_target_met
```

### Market Regime Monitoring

```python
# Get current regimes
regimes = await alphapulse.get_current_regimes()

# Example output:
# {
#     'BTC/USDT_15m': 'strong_trend_bull',
#     'ETH/USDT_15m': 'ranging'
# }
```

### Signal History

```python
# Get recent signals
signals = alphapulse.get_signal_history(limit=100)

# Each signal includes:
# - signal_id, symbol, timeframe
# - direction, confidence, entry_price
# - take_profit levels, stop_loss
# - pattern_type, market_regime
# - indicators, validation_metrics
```

## API Documentation

### Core API

#### AlphaPulseCore

```python
class AlphaPulseCore:
    def __init__(self, symbols, timeframes, enable_regime_detection=True, 
                 enable_ml=True, target_latency_ms=50.0):
        """Initialize AlphaPulse trading system"""
    
    async def start(self):
        """Start the trading system"""
    
    async def stop(self):
        """Stop the trading system"""
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance metrics summary"""
    
    def get_signal_history(self, limit: int = 100) -> List[Signal]:
        """Get recent signal history"""
    
    async def get_current_regimes(self) -> Dict[str, str]:
        """Get current market regimes"""
```

#### Signal Model

```python
@dataclass
class Signal:
    signal_id: str
    symbol: str
    timeframe: str
    direction: str  # 'long' or 'short'
    confidence: float
    entry_price: float
    tp1: float
    tp2: float
    tp3: float
    tp4: float
    stop_loss: float
    risk_reward_ratio: float
    pattern_type: str
    volume_confirmation: bool
    trend_alignment: bool
    market_regime: str
    indicators: Dict[str, float]
    validation_metrics: Dict[str, float]
    timestamp: datetime
    outcome: str = 'pending'
```

## Development

### Code Organization

The codebase follows a clean architecture pattern:

1. **Core Layer**: Business logic and trading algorithms
2. **Service Layer**: External integrations and data processing
3. **Data Layer**: Database models and persistence
4. **Presentation Layer**: APIs and interfaces

### Adding New Features

1. **New Indicators**: Add to `core/indicators_engine.py`
2. **New Strategies**: Add to `strategies/` directory
3. **New Services**: Add to `services/` directory
4. **Database Changes**: Create new migration in `database/migrations/`

### Testing Guidelines

1. **Unit Tests**: Test individual functions and classes
2. **Integration Tests**: Test component interactions
3. **Performance Tests**: Benchmark critical paths
4. **Edge Case Tests**: Test error conditions and boundaries

## Troubleshooting

### Common Issues

1. **Database Connection Errors**
   - Check PostgreSQL is running
   - Verify connection string in `.env`
   - Run `python scripts/setup_database.py`

2. **Redis Connection Errors**
   - Check Redis is running
   - Verify Redis URL in configuration
   - Test with `redis-cli ping`

3. **Performance Issues**
   - Check system resources (CPU, memory)
   - Review performance metrics
   - Adjust `target_latency_ms` if needed

4. **Test Failures**
   - Ensure all dependencies installed
   - Check test database is clean
   - Run tests with `-v` flag for verbose output

### Logging

The system uses structured logging with different levels:

```python
import logging

# Set log level
logging.basicConfig(level=logging.INFO)

# Log examples
logger.info("System started successfully")
logger.warning("High latency detected")
logger.error("Database connection failed")
logger.debug("Processing candle data")
```

## Contributing

1. **Fork the repository**
2. **Create a feature branch**
3. **Add tests for new functionality**
4. **Ensure all tests pass**
5. **Submit a pull request**

### Code Style

- Follow PEP 8 guidelines
- Use type hints for all functions
- Add docstrings for all classes and methods
- Keep functions focused and small

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support and questions:
- Create an issue in the repository
- Check the documentation in `docs/`
- Review the test examples for usage patterns

---

**Last Updated**: August 15, 2025  
**Version**: 2.0.0  
**Status**: Production Ready
