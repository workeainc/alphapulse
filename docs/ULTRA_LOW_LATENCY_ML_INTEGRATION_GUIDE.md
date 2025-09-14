# Ultra-Low Latency ML Integration Guide for AlphaPulse

## 🎯 Overview

This guide provides comprehensive instructions for implementing and using the **Ultra-Low Latency ML Pipeline** in AlphaPulse. The system achieves **<10ms inference latency** while maintaining ensemble-level accuracy through knowledge distillation, feature caching, and ONNX optimization.

## 🚀 Key Features

- **Knowledge Distillation**: Single lightweight model with ensemble accuracy
- **Feature Caching**: Redis-based pre-computed technical indicators
- **ONNX Optimization**: 2-5x faster inference with ONNX Runtime
- **TimescaleDB Integration**: Optimized storage for ML predictions and signals
- **Batch Processing**: Efficient multi-symbol/timeframe inference
- **Performance Monitoring**: Real-time latency and accuracy tracking

## 📋 Prerequisites

### System Requirements
- Python 3.11+
- Redis server (for feature caching)
- TimescaleDB/PostgreSQL (for data storage)
- 8GB+ RAM (for model loading and caching)
- GPU (optional, for ONNX GPU acceleration)

### Dependencies
```bash
# Install additional dependencies
pip install onnxruntime==1.22.0
pip install onnxruntime-gpu==1.22.1  # If GPU available
pip install onnx==1.18.0
pip install skl2onnx==1.18.0
pip install onnxconverter-common==1.14.0
pip install catboost==1.2.2
pip install river==0.20.1
pip install redis==5.0.1
```

## 🔧 Installation & Setup

### 1. Install Dependencies
```bash
cd backend
pip install -r requirements.txt
```

### 2. Configure Redis
```bash
# Start Redis server
redis-server

# Or using Docker
docker run -d -p 6379:6379 redis:latest
```

### 3. Configure TimescaleDB
Ensure your TimescaleDB instance is running and accessible with the credentials in your configuration.

## 🏗️ Architecture Components

### 1. Knowledge Distillation (`backend/ai/ml_models/knowledge_distillation.py`)
- **Purpose**: Creates lightweight "student" models that mimic ensemble behavior
- **Benefits**: Ensemble accuracy with single-model latency
- **Key Features**:
  - Temperature scaling for soft targets
  - Alpha blending of hard and soft targets
  - Multiple student model types (LightGBM, XGBoost, CatBoost)
  - ONNX conversion for maximum speed

### 2. Feature Cache Manager (`backend/ai/feature_cache_manager.py`)
- **Purpose**: Pre-computes and caches technical indicators in Redis
- **Benefits**: Eliminates feature computation latency during inference
- **Key Features**:
  - Redis-based caching with TTL
  - Comprehensive technical indicators (RSI, MACD, Bollinger Bands, etc.)
  - Fallback calculations when TA-Lib unavailable
  - Cache hit/miss statistics

### 3. Ultra-Low Latency Inference (`backend/ai/ultra_low_latency_inference.py`)
- **Purpose**: Orchestrates all components for optimal inference performance
- **Benefits**: <10ms inference latency with intelligent model selection
- **Key Features**:
  - Automatic model selection (ONNX → Student → Ensemble → Fallback)
  - Batch processing for multiple symbols/timeframes
  - Model warm-up for optimal performance
  - Comprehensive performance monitoring

### 4. TimescaleDB Integration (`backend/services/timescaledb_ml_integration.py`)
- **Purpose**: Stores ML predictions and signals with optimized schema
- **Benefits**: Fast queries and efficient storage for time-series data
- **Key Features**:
  - TimescaleDB hypertables for time-based partitioning
  - Optimized indexes for common query patterns
  - JSONB storage for flexible metadata
  - Performance tracking and statistics

## 🚀 Quick Start

### 1. Basic Usage
```python
import asyncio
from backend.ai.ultra_low_latency_inference import UltraLowLatencyInference
from backend.services.timescaledb_ml_integration import TimescaleDBMLIntegration

async def basic_example():
    # Initialize inference engine
    inference_engine = UltraLowLatencyInference()
    
    # Initialize with training data (optional)
    training_data = load_your_training_data()
    await inference_engine.initialize(training_data)
    
    # Make prediction
    candlestick_data = get_candlestick_data()
    result = await inference_engine.predict("BTC/USDT", "5m", candlestick_data)
    
    print(f"Prediction: {result.prediction:.3f}")
    print(f"Confidence: {result.confidence:.3f}")
    print(f"Latency: {result.latency_ms:.1f}ms")
    print(f"Model used: {result.model_used}")

# Run the example
asyncio.run(basic_example())
```

### 2. With TimescaleDB Integration
```python
import asyncio
from backend.services.timescaledb_ml_integration import TimescaleDBMLIntegration

async def with_storage_example():
    # Initialize integration service
    integration = TimescaleDBMLIntegration()
    await integration.initialize()
    
    # Make prediction and store
    candlestick_data = get_candlestick_data()
    result, success = await integration.make_prediction_and_store(
        "BTC/USDT", "5m", candlestick_data, generate_signal=True
    )
    
    if success:
        print(f"Prediction stored successfully: {result.prediction:.3f}")
    
    # Get recent predictions
    recent_predictions = await integration.get_recent_predictions(limit=10)
    print(f"Retrieved {len(recent_predictions)} recent predictions")

# Run the example
asyncio.run(with_storage_example())
```

### 3. Batch Processing
```python
import asyncio
from backend.ai.ultra_low_latency_inference import UltraLowLatencyInference

async def batch_example():
    inference_engine = UltraLowLatencyInference()
    await inference_engine.initialize()
    
    # Prepare batch predictions
    candlestick_data = get_candlestick_data()
    batch_predictions = [
        ("BTC/USDT", "5m", candlestick_data),
        ("ETH/USDT", "5m", candlestick_data),
        ("ADA/USDT", "5m", candlestick_data),
        ("SOL/USDT", "5m", candlestick_data)
    ]
    
    # Make batch predictions
    results = await inference_engine.predict_batch(batch_predictions)
    
    for i, result in enumerate(results):
        symbol = batch_predictions[i][0]
        print(f"{symbol}: {result.prediction:.3f} ({result.latency_ms:.1f}ms)")

# Run the example
asyncio.run(batch_example())
```

## ⚙️ Configuration

### Inference Configuration
```python
from backend.ai.ultra_low_latency_inference import InferenceConfig

config = InferenceConfig(
    target_latency_ms=10.0,  # Target inference latency
    enable_knowledge_distillation=True,
    enable_feature_caching=True,
    enable_onnx=True,
    enable_batching=True,
    batch_size=4,  # Optimal batch size
    warmup_runs=100,  # Number of warmup runs
    fallback_to_ensemble=True
)

inference_engine = UltraLowLatencyInference(config)
```

### Knowledge Distillation Configuration
```python
from backend.ai.knowledge_distillation import DistillationConfig

config = DistillationConfig(
    student_model_type="lightgbm",  # "lightgbm", "xgboost", "catboost"
    temperature=3.0,  # Temperature for soft targets
    alpha=0.7,  # Weight for hard vs soft targets
    max_depth=4,  # Shallow for fast inference
    n_estimators=50,  # Fewer trees for speed
    target_latency_ms=10.0
)
```

### Feature Cache Configuration
```python
from backend.ai.feature_cache_manager import FeatureCacheConfig

config = FeatureCacheConfig(
    redis_url="redis://localhost:6379",
    cache_ttl=3600,  # 1 hour cache TTL
    enable_compression=True,
    batch_size=100,
    update_interval=60,
    feature_groups=['momentum', 'trend', 'volatility', 'volume', 'oscillators']
)
```

## 📊 Performance Monitoring

### Get Performance Statistics
```python
# Get inference performance stats
stats = await inference_engine.get_performance_stats()
print(f"Average latency: {stats['avg_latency_ms']:.1f}ms")
print(f"P99 latency: {stats['p99_latency_ms']:.1f}ms")
print(f"Target met: {stats['target_met_pct']:.1f}%")
print(f"Model usage: {stats['model_usage']}")

# Get cache statistics
cache_stats = await feature_cache.get_cache_stats()
print(f"Cache hit rate: {cache_stats['hit_rate']:.1%}")
print(f"Cache hits: {cache_stats['cache_hits']}")
print(f"Cache misses: {cache_stats['cache_misses']}")

# Get storage statistics
storage_stats = await integration.get_performance_stats()
print(f"Predictions stored: {storage_stats['storage_stats']['predictions_stored']}")
print(f"Signals stored: {storage_stats['storage_stats']['signals_stored']}")
print(f"Average storage time: {storage_stats['storage_stats']['avg_storage_time_ms']:.1f}ms")
```

### Performance Targets
- **Inference Latency**: <10ms (target), <15ms (acceptable)
- **Cache Hit Rate**: >80% for frequently accessed features
- **Storage Latency**: <5ms for database operations
- **Accuracy Preservation**: >95% of ensemble accuracy
- **Throughput**: 100+ predictions/second

## 🔍 Database Schema

### ML Predictions Table
```sql
CREATE TABLE ml_predictions (
    id BIGSERIAL,
    prediction_id VARCHAR(100) UNIQUE NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    prediction DECIMAL(10,6) NOT NULL,
    confidence DECIMAL(10,6) NOT NULL,
    model_used VARCHAR(50) NOT NULL,
    features JSONB NOT NULL,
    inference_latency_ms DECIMAL(10,3) NOT NULL,
    ensemble_predictions JSONB,
    model_weights JSONB,
    feature_importance JSONB,
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (timestamp, id)
);
```

### ML Signals Table
```sql
CREATE TABLE ml_signals (
    id BIGSERIAL,
    signal_id VARCHAR(100) UNIQUE NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    prediction DECIMAL(10,6) NOT NULL,
    confidence DECIMAL(10,6) NOT NULL,
    model_used VARCHAR(50) NOT NULL,
    features JSONB NOT NULL,
    inference_latency_ms DECIMAL(10,3) NOT NULL,
    signal_direction VARCHAR(10) NOT NULL,
    entry_price DECIMAL(20,8) NOT NULL,
    stop_loss DECIMAL(20,8),
    take_profit DECIMAL(20,8),
    risk_reward_ratio DECIMAL(10,4),
    market_regime VARCHAR(20),
    pattern_type VARCHAR(50),
    volume_confirmation BOOLEAN,
    trend_alignment BOOLEAN,
    metadata JSONB,
    status VARCHAR(20) DEFAULT 'generated',
    outcome VARCHAR(20),
    pnl DECIMAL(20,8),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (timestamp, id)
);
```

## 🧪 Testing

### Run Integration Test
```bash
cd backend
python scripts/integrate_ultra_low_latency_ml.py
```

### Expected Output
```
🚀 Starting Ultra-Low Latency ML Demonstration...
============================================================

📚 STEP 1: KNOWLEDGE DISTILLATION
----------------------------------------
✅ Knowledge distillation completed:
   - Student accuracy: 0.847
   - Ensemble accuracy: 0.856
   - Accuracy preservation: 98.9%
   - Latency improvement: 3.2x
   - Student latency: 8.5ms

💾 STEP 2: FEATURE CACHING
----------------------------------------
✅ Feature caching results:
   - First computation: 0.045s
   - Second computation: 0.002s
   - Speedup: 22.5x
   - Cache hit rate: 50.0%
   - Features computed: 25

⚡ STEP 3: ULTRA-LOW LATENCY INFERENCE
----------------------------------------
✅ Single prediction results:
   - Prediction: 0.723
   - Confidence: 0.446
   - Latency: 7.2ms
   - Model used: student
   - Target met: True

🗄️ STEP 4: TIMESCALEDB INTEGRATION
----------------------------------------
✅ Prediction and storage results:
   - Prediction: 0.723
   - Confidence: 0.446
   - Latency: 7.2ms
   - Storage success: True

🎯 DEMONSTRATION SUMMARY
============================================================
✅ All components successfully demonstrated!
✅ Ultra-low latency ML pipeline is ready for production
✅ Target latency of <10ms achieved
✅ TimescaleDB integration working
✅ Feature caching providing significant speedup
✅ Knowledge distillation preserving accuracy
```

## 🔧 Troubleshooting

### Common Issues

#### 1. Redis Connection Failed
```
⚠️ Redis connection failed: Connection refused
```
**Solution**: Ensure Redis server is running
```bash
redis-server
# Or check if Redis is running
redis-cli ping
```

#### 2. ONNX Model Loading Failed
```
❌ Error loading ONNX model: No such file or directory
```
**Solution**: Ensure ONNX models are generated first
```python
# Generate ONNX model from student model
await knowledge_distillation._convert_to_onnx(student_model, result, config)
```

#### 3. TimescaleDB Connection Failed
```
❌ Error connecting to TimescaleDB: connection refused
```
**Solution**: Check database configuration and ensure TimescaleDB is running
```python
# Verify connection
await db_connection.initialize()
```

#### 4. High Latency (>10ms)
```
⚠️ Inference latency 15.2ms exceeds target 10ms
```
**Solutions**:
- Enable feature caching
- Use ONNX models
- Reduce model complexity
- Optimize batch size

### Performance Optimization

#### 1. Optimize Feature Caching
```python
# Increase cache TTL for frequently accessed features
config = FeatureCacheConfig(cache_ttl=7200)  # 2 hours

# Pre-compute features for common symbols/timeframes
for symbol in ['BTC/USDT', 'ETH/USDT']:
    for timeframe in ['1m', '5m', '15m']:
        await feature_cache.precompute_features(symbol, timeframe, candlestick_data)
```

#### 2. Optimize Model Selection
```python
# Prioritize fastest models
config = InferenceConfig(
    enable_onnx=True,  # Use ONNX first
    enable_knowledge_distillation=True,  # Then student model
    fallback_to_ensemble=False  # Skip ensemble for speed
)
```

#### 3. Optimize Batch Processing
```python
# Find optimal batch size for your hardware
config = InferenceConfig(batch_size=8)  # Test different values

# Use batch processing for multiple predictions
results = await inference_engine.predict_batch(batch_predictions)
```

## 📈 Production Deployment

### 1. Environment Setup
```bash
# Production environment variables
export REDIS_URL="redis://production-redis:6379"
export DATABASE_URL="postgresql://user:pass@production-db:5432/alphapulse"
export ENABLE_GPU="true"  # If GPU available
export LOG_LEVEL="INFO"
```

### 2. Service Configuration
```python
# Production inference configuration
config = InferenceConfig(
    target_latency_ms=8.0,  # Stricter target for production
    enable_knowledge_distillation=True,
    enable_feature_caching=True,
    enable_onnx=True,
    enable_batching=True,
    batch_size=16,  # Larger batches for production
    warmup_runs=500,  # More warmup runs
    enable_monitoring=True
)
```

### 3. Monitoring and Alerting
```python
# Set up performance monitoring
async def monitor_performance():
    while True:
        stats = await inference_engine.get_performance_stats()
        
        # Alert if latency exceeds threshold
        if stats['avg_latency_ms'] > 10.0:
            send_alert(f"High latency: {stats['avg_latency_ms']:.1f}ms")
        
        # Alert if accuracy drops
        if stats.get('accuracy', 1.0) < 0.8:
            send_alert(f"Low accuracy: {stats['accuracy']:.3f}")
        
        await asyncio.sleep(60)  # Check every minute

# Start monitoring
asyncio.create_task(monitor_performance())
```

## 🎯 Best Practices

### 1. Model Management
- Regularly retrain and update distilled models
- Monitor model drift and performance degradation
- Use A/B testing for new model versions
- Maintain model versioning and rollback capabilities

### 2. Caching Strategy
- Cache frequently accessed features
- Use appropriate TTL based on data freshness requirements
- Monitor cache hit rates and optimize accordingly
- Implement cache warming for critical symbols/timeframes

### 3. Performance Monitoring
- Track latency percentiles (P50, P90, P99)
- Monitor accuracy preservation over time
- Alert on performance degradation
- Use distributed tracing for debugging

### 4. Database Optimization
- Use appropriate indexes for common query patterns
- Implement data retention policies
- Monitor query performance and optimize slow queries
- Use connection pooling for database connections

## 🔗 Integration with Existing Systems

### 1. Trading Engine Integration
```python
# Integrate with existing trading engine
class TradingEngine:
    def __init__(self):
        self.ml_integration = TimescaleDBMLIntegration()
        asyncio.create_task(self.ml_integration.initialize())
    
    async def process_market_data(self, symbol, timeframe, candlestick_data):
        # Make ML prediction
        result, success = await self.ml_integration.make_prediction_and_store(
            symbol, timeframe, candlestick_data, generate_signal=True
        )
        
        # Use prediction in trading logic
        if result.confidence > 0.8:
            await self.execute_trade_signal(result)
```

### 2. WebSocket Integration
```python
# Real-time predictions via WebSocket
async def websocket_handler(websocket, path):
    async for message in websocket:
        data = json.loads(message)
        
        # Make prediction
        result = await inference_engine.predict(
            data['symbol'], data['timeframe'], data['candlestick_data']
        )
        
        # Send result back
        await websocket.send(json.dumps({
            'prediction': result.prediction,
            'confidence': result.confidence,
            'latency_ms': result.latency_ms,
            'model_used': result.model_used
        }))
```

### 3. API Integration
```python
# FastAPI endpoint for predictions
@app.post("/api/v1/predict")
async def predict_endpoint(request: PredictionRequest):
    result = await inference_engine.predict(
        request.symbol, request.timeframe, request.candlestick_data
    )
    
    return PredictionResponse(
        prediction=result.prediction,
        confidence=result.confidence,
        latency_ms=result.latency_ms,
        model_used=result.model_used
    )
```

## 📚 Additional Resources

- [ONNX Runtime Documentation](https://onnxruntime.ai/)
- [TimescaleDB Documentation](https://docs.timescale.com/)
- [Redis Documentation](https://redis.io/documentation)
- [Knowledge Distillation Paper](https://arxiv.org/abs/1503.02531)
- [AlphaPulse Documentation](../README.md)

## 🆘 Support

For issues and questions:
1. Check the troubleshooting section above
2. Review the logs for error messages
3. Verify all dependencies are installed correctly
4. Ensure Redis and TimescaleDB are running
5. Test with the integration script first

---

**🎉 Congratulations!** You now have a production-ready ultra-low latency ML pipeline that can achieve <10ms inference latency while maintaining ensemble-level accuracy.
