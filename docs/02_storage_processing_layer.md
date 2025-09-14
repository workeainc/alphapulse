# 🗄️ Storage & Processing Layer

## Overview
The brain's memory — built on TimescaleDB for high-performance time-series handling. This layer manages data storage, processing, and optimization for fast historical backtesting and real-time analysis.

## ✅ Implemented Components

### 1. TimescaleDB Connection
- **File**: `backend/database/connection.py` ✅
- **Features**:
  - Async database connections
  - Connection pooling
  - Health checks
  - Graceful shutdown

### 2. Database Models
- **File**: `backend/database/models.py` ✅
- **Features**:
  - MarketData table with technical indicators
  - Trade history tracking
  - Sentiment data storage
  - Portfolio management

### 3. TimescaleDB Setup
- **File**: `backend/database/connection.py` ✅
- **Features**:
  - Hypertable creation for time-series data
  - Basic indexing on timestamp and symbol
  - Connection string configuration

## 🚧 Partially Implemented

### 4. Data Processing Pipeline
- **Status**: Basic structure exists
- **Needs**: Continuous aggregates and data validation

### 5. Performance Optimization
- **Status**: Basic indexing exists
- **Needs**: Advanced query optimization

## ❌ Not Yet Implemented

### 6. Continuous Aggregates
- **Required**: Auto-generate higher timeframe data
- **Purpose**: Fast multi-timeframe analysis
- **Priority**: Medium

### 7. Advanced Indexing
- **Required**: Custom indexes for backtesting queries
- **Purpose**: Sub-second historical data retrieval
- **Priority**: Medium

### 8. Data Validation
- **Required**: Quality checks and anomaly detection
- **Purpose**: Ensure data integrity
- **Priority**: High

### 9. Data Compression
- **Required**: Automatic data compression policies
- **Purpose**: Storage optimization
- **Priority**: Low

## 🔧 Implementation Tasks

### Immediate (This Week)
1. **Enhanced Database Connection**
   ```python
   # Update: backend/database/connection.py
   class TimescaleDBManager:
       def __init__(self):
           self.connection_pool = None
           self.hypertables = {}
           self.continuous_aggregates = {}
       
       async def initialize_hypertables(self):
           """Initialize TimescaleDB hypertables"""
           async with self.get_connection() as conn:
               # Create MarketData hypertable
               await conn.execute("""
                   SELECT create_hypertable('market_data', 'timestamp', 
                       if_not_exists => TRUE,
                       chunk_time_interval => INTERVAL '1 day'
                   );
               """)
               
               # Create indexes for fast queries
               await conn.execute("""
                   CREATE INDEX IF NOT EXISTS idx_market_data_symbol_timestamp 
                   ON market_data (symbol, timestamp DESC);
               """)
               
               await conn.execute("""
                   CREATE INDEX IF NOT EXISTS idx_market_data_timestamp_symbol 
                   ON market_data (timestamp DESC, symbol);
               """)
   ```

2. **Data Validation System**
   ```python
   # New file: backend/data/validation.py
   class DataValidator:
       def __init__(self):
           self.validation_rules = {
               'price': {'min': 0.000001, 'max': 1000000},
               'volume': {'min': 0, 'max': 1000000000},
               'timestamp': {'max_age_hours': 24}
           }
       
       def validate_market_data(self, data: MarketData) -> ValidationResult:
           """Validate market data entry"""
           errors = []
           
           # Price validation
           if data.price <= 0 or data.price > self.validation_rules['price']['max']:
               errors.append(f"Invalid price: {data.price}")
           
           # Volume validation
           if data.volume < 0 or data.volume > self.validation_rules['volume']['max']:
               errors.append(f"Invalid volume: {data.volume}")
           
           # Timestamp validation
           if data.timestamp < datetime.now() - timedelta(hours=self.validation_rules['timestamp']['max_age_hours']):
               errors.append(f"Data too old: {data.timestamp}")
           
           return ValidationResult(
               valid=len(errors) == 0,
               errors=errors
           )
   ```

### Short Term (Next 2 Weeks)
1. **Continuous Aggregates Implementation**
   - 1-minute to 5-minute aggregation
   - 5-minute to 15-minute aggregation
   - 15-minute to 1-hour aggregation

2. **Advanced Query Optimization**
   - Partition pruning
   - Parallel query execution
   - Query plan optimization

### Medium Term (Next Month)
1. **Data Compression Policies**
   - Automatic compression after retention period
   - Compression ratio monitoring
   - Storage cost optimization

## 📊 Storage Architecture

### Hypertable Structure
```
MarketData Hypertable
├── Chunk 1: 2024-01-01 to 2024-01-02
├── Chunk 2: 2024-01-02 to 2024-01-03
├── Chunk 3: 2024-01-03 to 2024-01-04
└── ... (daily chunks)
```

### Data Flow
```
Raw Data → Validation → Storage → Aggregation → Analysis
   ↓         ↓         ↓         ↓           ↓
Exchange   Quality   Timescale  Continuous  Strategy
  API      Check      DB        Aggregates  Engine
```

## 🗃️ Database Schema

### MarketData Table
```sql
CREATE TABLE market_data (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    open DECIMAL(20,8) NOT NULL,
    high DECIMAL(20,8) NOT NULL,
    low DECIMAL(20,8) NOT NULL,
    close DECIMAL(20,8) NOT NULL,
    volume DECIMAL(20,8) NOT NULL,
    
    -- Technical Indicators
    ema_9 DECIMAL(20,8),
    ema_21 DECIMAL(20,8),
    ema_50 DECIMAL(20,8),
    ema_200 DECIMAL(20,8),
    rsi_14 DECIMAL(10,4),
    macd DECIMAL(20,8),
    macd_signal DECIMAL(20,8),
    macd_histogram DECIMAL(20,8),
    bb_upper DECIMAL(20,8),
    bb_middle DECIMAL(20,8),
    bb_lower DECIMAL(20,8),
    atr_14 DECIMAL(20,8),
    
    -- Market Regime
    trend_direction VARCHAR(10),
    volatility_regime VARCHAR(20),
    
    -- Metadata
    source VARCHAR(50),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Convert to hypertable
SELECT create_hypertable('market_data', 'timestamp', 
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

-- Create indexes
CREATE INDEX idx_market_data_symbol_timestamp ON market_data (symbol, timestamp DESC);
CREATE INDEX idx_market_data_timestamp_symbol ON market_data (timestamp DESC, symbol);
CREATE INDEX idx_market_data_symbol_trend ON market_data (symbol, trend_direction, timestamp DESC);
```

### Continuous Aggregates
```sql
-- 5-minute continuous aggregate
CREATE MATERIALIZED VIEW market_data_5m
WITH (timescaledb.continuous) AS
SELECT 
    time_bucket('5 minutes', timestamp) AS bucket,
    symbol,
    first(open, timestamp) AS open,
    max(high) AS high,
    min(low) AS low,
    last(close, timestamp) AS close,
    sum(volume) AS volume,
    avg(ema_9) AS ema_9,
    avg(ema_21) AS ema_21,
    avg(ema_50) AS ema_50,
    avg(ema_200) AS ema_200,
    avg(rsi_14) AS rsi_14,
    avg(macd) AS macd,
    avg(macd_signal) AS macd_signal,
    avg(macd_histogram) AS macd_histogram,
    avg(bb_upper) AS bb_upper,
    avg(bb_middle) AS bb_middle,
    avg(bb_lower) AS bb_lower,
    avg(atr_14) AS atr_14,
    mode(trend_direction) AS trend_direction,
    mode(volatility_regime) AS volatility_regime
FROM market_data
GROUP BY bucket, symbol;

-- 15-minute continuous aggregate
CREATE MATERIALIZED VIEW market_data_15m
WITH (timescaledb.continuous) AS
SELECT 
    time_bucket('15 minutes', timestamp) AS bucket,
    symbol,
    first(open, timestamp) AS open,
    max(high) AS high,
    min(low) AS low,
    last(close, timestamp) AS close,
    sum(volume) AS volume,
    avg(ema_9) AS ema_9,
    avg(ema_21) AS ema_21,
    avg(ema_50) AS ema_50,
    avg(ema_200) AS ema_200,
    avg(rsi_14) AS rsi_14,
    avg(macd) AS macd,
    avg(macd_signal) AS macd_signal,
    avg(macd_histogram) AS macd_histogram,
    avg(bb_upper) AS bb_upper,
    avg(bb_middle) AS bb_middle,
    avg(bb_lower) AS bb_lower,
    avg(atr_14) AS atr_14,
    mode(trend_direction) AS trend_direction,
    mode(volatility_regime) AS volatility_regime
FROM market_data
GROUP BY bucket, symbol;

-- 1-hour continuous aggregate
CREATE MATERIALIZED VIEW market_data_1h
WITH (timescaledb.continuous) AS
SELECT 
    time_bucket('1 hour', timestamp) AS bucket,
    symbol,
    first(open, timestamp) AS open,
    max(high) AS high,
    min(low) AS low,
    last(close, timestamp) AS close,
    sum(volume) AS volume,
    avg(ema_9) AS ema_9,
    avg(ema_21) AS ema_21,
    avg(ema_50) AS ema_50,
    avg(ema_200) AS ema_200,
    avg(rsi_14) AS rsi_14,
    avg(macd) AS macd,
    avg(macd_signal) AS macd_signal,
    avg(macd_histogram) AS macd_histogram,
    avg(bb_upper) AS bb_upper,
    avg(bb_middle) AS bb_middle,
    avg(bb_lower) AS bb_lower,
    avg(atr_14) AS atr_14,
    mode(trend_direction) AS trend_direction,
    mode(volatility_regime) AS volatility_regime
FROM market_data
GROUP BY bucket, symbol;
```

## 🔍 Query Optimization

### Fast Historical Queries
```python
class OptimizedDataQuerier:
    def __init__(self, db_manager: TimescaleDBManager):
        self.db_manager = db_manager
    
    async def get_historical_data(self, symbol: str, timeframe: str, 
                                 start_time: datetime, end_time: datetime) -> List[MarketData]:
        """Get optimized historical data"""
        async with self.db_manager.get_connection() as conn:
            # Use appropriate continuous aggregate based on timeframe
            if timeframe == "5m":
                table = "market_data_5m"
                time_col = "bucket"
            elif timeframe == "15m":
                table = "market_data_15m"
                time_col = "bucket"
            elif timeframe == "1h":
                table = "market_data_1h"
                time_col = "bucket"
            else:
                table = "market_data"
                time_col = "timestamp"
            
            query = f"""
                SELECT * FROM {table}
                WHERE symbol = $1 
                AND {time_col} BETWEEN $2 AND $3
                ORDER BY {time_col} ASC
            """
            
            result = await conn.execute(query, symbol, start_time, end_time)
            return [MarketData(**row) for row in result.fetchall()]
    
    async def get_latest_data(self, symbol: str, limit: int = 100) -> List[MarketData]:
        """Get latest data with optimized query"""
        async with self.db_manager.get_connection() as conn:
            query = """
                SELECT * FROM market_data
                WHERE symbol = $1
                ORDER BY timestamp DESC
                LIMIT $2
            """
            
            result = await conn.execute(query, symbol, limit)
            return [MarketData(**row) for row in result.fetchall()]
```

### Performance Monitoring
```python
class PerformanceMonitor:
    def __init__(self, db_manager: TimescaleDBManager):
        self.db_manager = db_manager
    
    async def get_query_performance(self) -> Dict:
        """Monitor query performance"""
        async with self.db_manager.get_connection() as conn:
            # Get slow queries
            slow_queries = await conn.execute("""
                SELECT query, calls, total_time, mean_time
                FROM pg_stat_statements
                WHERE mean_time > 1000  -- Queries taking > 1 second
                ORDER BY mean_time DESC
                LIMIT 10
            """)
            
            # Get table statistics
            table_stats = await conn.execute("""
                SELECT schemaname, tablename, n_tup_ins, n_tup_upd, n_tup_del
                FROM pg_stat_user_tables
                WHERE schemaname = 'public'
                ORDER BY n_tup_ins DESC
            """)
            
            return {
                "slow_queries": slow_queries.fetchall(),
                "table_stats": table_stats.fetchall()
            }
    
    async def get_storage_stats(self) -> Dict:
        """Get storage statistics"""
        async with self.db_manager.get_connection() as conn:
            # Get table sizes
            sizes = await conn.execute("""
                SELECT 
                    schemaname,
                    tablename,
                    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size
                FROM pg_tables
                WHERE schemaname = 'public'
                ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC
            """)
            
            # Get compression stats
            compression = await conn.execute("""
                SELECT 
                    hypertable_name,
                    compression_status,
                    before_compression_total_bytes,
                    after_compression_total_bytes
                FROM timescaledb_information.compression_settings
            """)
            
            return {
                "table_sizes": sizes.fetchall(),
                "compression_stats": compression.fetchall()
            }
```

## 📈 Data Processing Pipeline

### Real-Time Data Ingestion
```python
class DataIngestionPipeline:
    def __init__(self, db_manager: TimescaleDBManager, validator: DataValidator):
        self.db_manager = db_manager
        self.validator = validator
        self.batch_size = 1000
        self.batch_buffer = []
    
    async def ingest_market_data(self, data: MarketData):
        """Ingest single market data point"""
        # Validate data
        validation = self.validator.validate_market_data(data)
        if not validation.valid:
            logger.warning(f"Data validation failed: {validation.errors}")
            return False
        
        # Add to batch buffer
        self.batch_buffer.append(data)
        
        # Process batch if full
        if len(self.batch_buffer) >= self.batch_size:
            await self._process_batch()
        
        return True
    
    async def _process_batch(self):
        """Process batch of market data"""
        if not self.batch_buffer:
            return
        
        try:
            async with self.db_manager.get_connection() as conn:
                # Prepare batch insert
                values = []
                for data in self.batch_buffer:
                    values.append(f"('{data.symbol}', '{data.timestamp}', {data.open}, "
                               f"{data.high}, {data.low}, {data.close}, {data.volume}, "
                               f"{data.ema_9}, {data.ema_21}, {data.ema_50}, {data.ema_200}, "
                               f"{data.rsi_14}, {data.macd}, {data.macd_signal}, "
                               f"{data.macd_histogram}, {data.bb_upper}, {data.bb_middle}, "
                               f"{data.bb_lower}, {data.atr_14}, '{data.trend_direction}', "
                               f"'{data.volatility_regime}', '{data.source}')")
                
                # Batch insert
                query = f"""
                    INSERT INTO market_data (
                        symbol, timestamp, open, high, low, close, volume,
                        ema_9, ema_21, ema_50, ema_200, rsi_14, macd,
                        macd_signal, macd_histogram, bb_upper, bb_middle,
                        bb_lower, atr_14, trend_direction, volatility_regime, source
                    ) VALUES {','.join(values)}
                    ON CONFLICT (symbol, timestamp) DO UPDATE SET
                        open = EXCLUDED.open,
                        high = EXCLUDED.high,
                        low = EXCLUDED.low,
                        close = EXCLUDED.close,
                        volume = EXCLUDED.volume
                """
                
                await conn.execute(query)
                logger.info(f"Processed batch of {len(self.batch_buffer)} records")
                
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            # Re-queue failed items
            self.batch_buffer.extend(self.batch_buffer)
        
        finally:
            # Clear buffer
            self.batch_buffer.clear()
    
    async def flush_buffer(self):
        """Flush remaining buffer items"""
        if self.batch_buffer:
            await self._process_batch()
```

## 🔧 Maintenance Tasks

### Data Retention Policy
```python
class DataRetentionManager:
    def __init__(self, db_manager: TimescaleDBManager):
        self.db_manager = db_manager
        self.retention_policies = {
            "1m": timedelta(days=30),      # Keep 1-minute data for 30 days
            "5m": timedelta(days=90),      # Keep 5-minute data for 90 days
            "15m": timedelta(days=180),    # Keep 15-minute data for 180 days
            "1h": timedelta(days=365),     # Keep 1-hour data for 1 year
            "1d": timedelta(days=2555)     # Keep daily data for 7 years
        }
    
    async def apply_retention_policies(self):
        """Apply data retention policies"""
        async with self.db_manager.get_connection() as conn:
            for timeframe, retention in self.retention_policies.items():
                cutoff_date = datetime.now() - retention
                
                if timeframe == "1m":
                    # Delete old 1-minute data
                    await conn.execute("""
                        DELETE FROM market_data 
                        WHERE timestamp < $1
                    """, cutoff_date)
                else:
                    # Delete old continuous aggregate data
                    table = f"market_data_{timeframe}"
                    await conn.execute(f"""
                        DELETE FROM {table}
                        WHERE bucket < $1
                    """, cutoff_date)
                
                logger.info(f"Applied retention policy for {timeframe}: deleted data before {cutoff_date}")
    
    async def compress_old_data(self):
        """Compress old data for storage optimization"""
        async with self.db_manager.get_connection() as conn:
            # Enable compression on hypertables
            await conn.execute("""
                ALTER TABLE market_data SET (
                    timescaledb.compress,
                    timescaledb.compress_segmentby = 'symbol',
                    timescaledb.compress_orderby = 'timestamp DESC'
                );
            """)
            
            # Add compression policy
            await conn.execute("""
                SELECT add_compression_policy('market_data', INTERVAL '7 days');
            """)
            
            logger.info("Compression policies applied to market_data table")
```

## 🚀 Next Steps

1. **Implement continuous aggregates** for multi-timeframe data
2. **Add data validation** pipeline for quality assurance
3. **Create performance monitoring** for query optimization
4. **Set up data retention** and compression policies
5. **Optimize indexes** for backtesting queries

## 📚 Related Documentation

- [Data Collection Layer](./01_data_collection_layer.md)
- [Analysis Layer](./03_analysis_layer.md)
- [Execution Layer](./04_execution_layer.md)
- [Risk Management](./05_risk_management.md)
- [Pine Script Integration](./06_pine_script_integration.md)
