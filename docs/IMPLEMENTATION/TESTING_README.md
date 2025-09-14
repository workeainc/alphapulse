# AlphaPulse Comprehensive Testing Suite

This document provides a complete guide to the AlphaPulse testing infrastructure, including database setup, test execution, and performance validation.

## 🎯 Overview

The AlphaPulse testing suite is designed to validate the high-frequency trading signal system against strict performance and accuracy requirements:

- **Latency**: < 50ms tick-to-signal
- **Throughput**: > 10,000 signals/second
- **Accuracy**: 75-85% win rate
- **Filter Rate**: 60-80% signal filtering
- **CPU Usage**: < 80% peak utilization

## 📁 Project Structure

```
backend/
├── database/
│   ├── db_scanner_simple.py          # Database structure scanner
│   ├── migrations/                   # Alembic migrations
│   │   ├── env.py                   # Alembic environment config
│   │   └── 007_create_alphapulse_test_tables.py
│   └── models.py                    # SQLAlchemy models
├── tests/
│   ├── conftest.py                  # Pytest configuration and fixtures
│   ├── test_unit_indicators.py      # Unit tests for indicators
│   ├── test_integration_pipeline.py # Integration tests
│   ├── test_database_operations.py  # Database tests
│   ├── test_performance_benchmark.py # Performance tests
│   └── test_edge_cases.py           # Edge case tests
├── test_data/
│   └── sample_historical_data.csv   # Sample market data
├── create_test_tables.py            # Database table creation
├── run_comprehensive_tests.py       # Main test runner
├── alembic.ini                      # Alembic configuration
└── TESTING_README.md               # This file
```

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export TEST_DB_URL="sqlite:///test_alphapulse_test.db"
export BINANCE_WS_URL="wss://stream.binance.com:9443/ws/btcusdt@kline_1m"
export REDIS_URL="redis://localhost:6379/0"
```

### 2. Database Setup

```bash
# Create test tables
python create_test_tables.py

# Run database scan
python database/db_scanner_simple.py
```

### 3. Run All Tests

```bash
# Run comprehensive test suite
python run_comprehensive_tests.py
```

### 4. Run Individual Test Categories

```bash
# Unit tests
pytest tests/test_unit_indicators.py -v

# Integration tests
pytest tests/test_integration_pipeline.py -v

# Database tests
pytest tests/test_database_operations.py -v

# Performance tests
pytest tests/test_performance_benchmark.py -v

# Edge case tests
pytest tests/test_edge_cases.py -v
```

## 📊 Test Categories

### 1. Database Structure Scanning (`db_scanner_simple.py`)

**Purpose**: Analyze existing database structure and compare with required schema.

**Features**:
- Scans existing tables, columns, indexes, and foreign keys
- Compares with required AlphaPulse schema
- Generates migration recommendations
- Creates detailed analysis reports

**Usage**:
```python
from database.db_scanner_simple import SimpleDatabaseScanner

scanner = SimpleDatabaseScanner()
report = scanner.generate_report()
scanner.print_summary(report)
```

### 2. Unit Tests (`test_unit_indicators.py`)

**Purpose**: Test individual components in isolation.

**Coverage**:
- Technical indicator calculations (RSI, MACD, Bollinger Bands)
- Signal validation logic
- Pattern detection algorithms
- Performance metrics validation

**Key Tests**:
- RSI calculation accuracy
- Divergence detection
- Breakout strength calculation
- Signal filtering effectiveness
- Latency benchmarks

### 3. Integration Tests (`test_integration_pipeline.py`)

**Purpose**: Test full system integration with realistic data flows.

**Coverage**:
- End-to-end pipeline processing
- WebSocket data handling
- Signal generation and validation
- Multi-symbol processing
- Error recovery mechanisms

**Key Tests**:
- Full pipeline latency (< 50ms)
- Signal accuracy simulation (75-85% win rate)
- Multi-symbol handling
- Error recovery and resilience
- Performance benchmarks

### 4. Database Tests (`test_database_operations.py`)

**Purpose**: Validate database operations and data integrity.

**Coverage**:
- Signal insertion and retrieval
- Bulk operations performance
- Query optimization
- Foreign key constraints
- Concurrent access handling

**Key Tests**:
- Signal insertion performance
- Query latency (< 10ms for indexed queries)
- Bulk insertion throughput
- Data integrity validation
- Concurrent access safety

### 5. Performance Tests (`test_performance_benchmark.py`)

**Purpose**: Validate system performance under load.

**Coverage**:
- Throughput benchmarks
- Latency consistency
- Memory efficiency
- CPU utilization
- Concurrent processing

**Key Tests**:
- 10,000+ signals/second throughput
- < 1ms average latency
- Memory growth < 500MB
- CPU efficiency > 100 signals/CPU%
- Concurrent processing scalability

### 6. Edge Case Tests (`test_edge_cases.py`)

**Purpose**: Test system behavior under extreme conditions.

**Coverage**:
- Low-volume signal rejection
- Multi-symbol handling
- Error recovery
- Extreme market conditions
- Memory pressure scenarios

**Key Tests**:
- Low-volume signal filtering
- Malformed data handling
- Extreme price movements
- Memory pressure resilience
- Timeout handling

## 🗄️ Database Schema

### Required Tables

#### 1. `signals` Table
```sql
CREATE TABLE signals (
    id INTEGER PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    direction VARCHAR(10) NOT NULL,  -- 'buy'/'sell'
    confidence FLOAT NOT NULL,
    tp1 FLOAT,
    tp2 FLOAT,
    tp3 FLOAT,
    tp4 FLOAT,
    sl FLOAT,
    timestamp DATETIME NOT NULL,
    outcome VARCHAR(20) DEFAULT 'pending'  -- 'win'/'loss'/'pending'
);

CREATE INDEX idx_signals_symbol_timeframe_timestamp 
ON signals(symbol, timeframe, timestamp);
```

#### 2. `logs` Table
```sql
CREATE TABLE logs (
    id INTEGER PRIMARY KEY,
    pattern_type VARCHAR(50) NOT NULL,
    confidence_score FLOAT NOT NULL,
    volume_context JSON,
    trend_context JSON,
    outcome VARCHAR(20),
    timestamp DATETIME NOT NULL
);

CREATE INDEX idx_logs_timestamp ON logs(timestamp);
```

#### 3. `feedback` Table
```sql
CREATE TABLE feedback (
    id INTEGER PRIMARY KEY,
    signal_id INTEGER NOT NULL,
    market_outcome FLOAT,  -- PnL
    notes TEXT,
    FOREIGN KEY (signal_id) REFERENCES signals(id)
);

CREATE INDEX idx_feedback_signal_id ON feedback(signal_id);
```

## 📈 Performance Targets

### Latency Requirements
- **RSI Calculation**: < 1ms
- **Signal Generation**: < 1ms
- **Database Query**: < 10ms (indexed)
- **Full Pipeline**: < 50ms

### Throughput Requirements
- **Signal Generation**: > 10,000 signals/sec
- **Tick Processing**: > 15,000 ticks/sec
- **Database Operations**: > 12,000 ops/sec

### Accuracy Requirements
- **Win Rate**: 75-85%
- **Signal Filter Rate**: 60-80%
- **False Positive Rate**: < 15%

### System Requirements
- **Peak CPU**: < 80%
- **Peak Memory**: < 1000 MB
- **Memory Growth**: < 500 MB under load

## 🔧 Configuration

### Environment Variables

```bash
# Database Configuration
TEST_DB_URL="sqlite:///test_alphapulse_test.db"
DATABASE_URL="postgresql://user:pass@localhost:5432/alphapulse"

# WebSocket Configuration
BINANCE_WS_URL="wss://stream.binance.com:9443/ws/btcusdt@kline_1m"

# Redis Configuration
REDIS_URL="redis://localhost:6379/0"

# Test Configuration
PYTEST_ADDOPTS="-v --tb=short --json-report"
```

### Pytest Configuration (`conftest.py`)

The test configuration provides:
- Database session management
- Sample data generation
- Mock WebSocket clients
- Performance monitoring
- Redis mocking

## 📊 Reporting and Metrics

### Test Reports

The comprehensive test runner generates:
- **JSON Reports**: Detailed test results and metrics
- **Performance Charts**: Latency, throughput, and accuracy visualizations
- **System Metrics**: CPU and memory usage graphs
- **Summary Reports**: Executive-level test summaries

### Report Files

```
alphapulse_test_report_YYYYMMDD_HHMMSS.json  # Detailed test report
performance_latency.png                      # Latency performance chart
performance_throughput.png                   # Throughput performance chart
performance_accuracy.png                     # Accuracy metrics chart
system_metrics.png                          # System performance chart
```

### Sample Report Output

```
📊 ALPHAPULSE COMPREHENSIVE TEST SUMMARY
================================================================================
Test Run Duration: 45.2 seconds
Total Tests: 6
Passed: 6
Failed: 0
Success Rate: 100.0%

📋 Test Results:
  ✅ Database Scan: completed
  ✅ Unit Tests: passed
  ✅ Integration Tests: passed
  ✅ Database Tests: passed
  ✅ Performance Tests: passed
  ✅ Edge Case Tests: passed

🎯 Performance Targets:
  Target Latency: < 50ms
  Target Throughput: > 10,000 signals/sec
  Target Accuracy: > 75%
  Target Filter Rate: > 60%

💡 Recommendations:
  1. Monitor system performance in production
  2. Implement continuous integration for automated testing
  3. Set up monitoring and alerting for system health
```

## 🛠️ Troubleshooting

### Common Issues

#### 1. Database Connection Errors
```bash
# Check database URL
echo $TEST_DB_URL

# Verify database exists
python -c "from create_test_tables import create_tables; create_tables()"
```

#### 2. Import Errors
```bash
# Add backend to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Install missing dependencies
pip install -r requirements.txt
```

#### 3. Performance Test Failures
```bash
# Check system resources
htop
free -h

# Run with reduced load
pytest tests/test_performance_benchmark.py -k "test_high_throughput_processing" -v
```

#### 4. Memory Issues
```bash
# Increase memory limits
ulimit -v unlimited

# Force garbage collection
python -c "import gc; gc.collect()"
```

### Debug Mode

Enable debug output:
```bash
export DEBUG=true
pytest tests/ -v -s --tb=long
```

## 🔄 Continuous Integration

### GitHub Actions Example

```yaml
name: AlphaPulse Tests
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
      - name: Run tests
        run: |
          python run_comprehensive_tests.py
      - name: Upload test results
        uses: actions/upload-artifact@v2
        with:
          name: test-results
          path: alphapulse_test_report_*.json
```

## 📚 Additional Resources

### Documentation
- [AlphaPulse Core Documentation](ALPHAPULSE_README.md)
- [Database Models](database/models.py)
- [Alembic Migrations](database/migrations/)

### Performance Tuning
- [Performance Optimization Guide](docs/performance_optimization.md)
- [Database Indexing Strategy](docs/database_indexing.md)
- [Memory Management Best Practices](docs/memory_management.md)

### Monitoring
- [System Health Monitoring](docs/monitoring.md)
- [Alert Configuration](docs/alerts.md)
- [Metrics Collection](docs/metrics.md)

## 🤝 Contributing

### Adding New Tests

1. **Unit Tests**: Add to `tests/test_unit_indicators.py`
2. **Integration Tests**: Add to `tests/test_integration_pipeline.py`
3. **Database Tests**: Add to `tests/test_database_operations.py`
4. **Performance Tests**: Add to `tests/test_performance_benchmark.py`
5. **Edge Case Tests**: Add to `tests/test_edge_cases.py`

### Test Guidelines

- Use descriptive test names
- Include performance assertions
- Add proper error handling
- Document test requirements
- Ensure idempotent execution

### Code Style

- Follow PEP 8 guidelines
- Use type hints
- Add docstrings
- Include error messages
- Use meaningful variable names

## 📞 Support

For questions or issues:
1. Check the troubleshooting section
2. Review test logs and reports
3. Consult the documentation
4. Open an issue with detailed information

---

**Last Updated**: August 15, 2025  
**Version**: 1.0.0  
**Maintainer**: AlphaPulse Development Team
