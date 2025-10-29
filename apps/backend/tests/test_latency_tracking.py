#!/usr/bin/env python3
"""
Test script for latency tracking implementation
"""

import asyncio
import time
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import latency tracking components
from src.app.core.latency_tracker import latency_tracker, track_trading_pipeline, TradingPipelineMetrics

async def test_latency_tracking():
    """Test the latency tracking functionality"""
    
    logger.info("🧪 Testing Latency Tracking Implementation")
    
    # Test 1: Basic latency tracking
    logger.info("Test 1: Basic latency tracking")
    
    @track_trading_pipeline(model_id="test_model", symbol="BTCUSDT", strategy_name="test_strategy")
    async def mock_trading_pipeline():
        """Mock trading pipeline for testing"""
        # Simulate different pipeline stages
        await asyncio.sleep(0.1)  # Simulate fetch time
        await asyncio.sleep(0.2)  # Simulate preprocess time
        await asyncio.sleep(0.3)  # Simulate inference time
        await asyncio.sleep(0.05)  # Simulate postprocess time
        return {"signal": "buy", "confidence": 0.85}
    
    # Run the mock pipeline
    result = await mock_trading_pipeline()
    logger.info(f"Pipeline result: {result}")
    
    # Test 2: Error handling
    logger.info("Test 2: Error handling")
    
    @track_trading_pipeline(model_id="test_model_error", symbol="ETHUSDT", strategy_name="test_strategy")
    async def mock_pipeline_with_error():
        """Mock pipeline that raises an error"""
        await asyncio.sleep(0.1)
        raise Exception("Simulated pipeline error")
    
    try:
        await mock_pipeline_with_error()
    except Exception as e:
        logger.info(f"Expected error caught: {e}")
    
    # Test 3: Manual metrics creation
    logger.info("Test 3: Manual metrics creation")
    
    manual_metrics = TradingPipelineMetrics(
        model_id="manual_test",
        symbol="ADAUSDT",
        strategy_name="manual_strategy",
        fetch_time_ms=50.0,
        preprocess_time_ms=25.0,
        inference_time_ms=150.0,
        postprocess_time_ms=10.0,
        total_latency_ms=235.0,
        success=True,
        metadata={"test": True, "manual": True}
    )
    
    await latency_tracker.track_trading_pipeline(manual_metrics)
    
    # Test 4: Get summaries
    logger.info("Test 4: Get summaries")
    
    # Get overall summary
    overall_summary = latency_tracker.get_latency_summary()
    logger.info(f"Overall summary: {overall_summary}")
    
    # Get pipeline summary
    pipeline_summary = latency_tracker.get_pipeline_summary()
    logger.info(f"Pipeline summary: {pipeline_summary}")
    
    # Test 5: Get metrics by strategy
    logger.info("Test 5: Get metrics by strategy")
    
    strategy_metrics = latency_tracker.get_pipeline_metrics_by_strategy("test_strategy", minutes=60)
    logger.info(f"Strategy metrics count: {len(strategy_metrics)}")
    
    if strategy_metrics:
        latest_metric = strategy_metrics[-1]
        logger.info(f"Latest metric: {latest_metric.total_latency_ms:.2f}ms total")
    
    # Test 6: Get metrics by symbol
    logger.info("Test 6: Get metrics by symbol")
    
    symbol_metrics = latency_tracker.get_pipeline_metrics_by_symbol("BTCUSDT", minutes=60)
    logger.info(f"Symbol metrics count: {len(symbol_metrics)}")
    
    if symbol_metrics:
        latest_metric = symbol_metrics[-1]
        logger.info(f"Latest BTCUSDT metric: {latest_metric.total_latency_ms:.2f}ms total")
    
    logger.info("✅ All latency tracking tests completed successfully!")

async def test_database_integration():
    """Test database integration (if available)"""
    
    logger.info("🧪 Testing Database Integration")
    
    try:
        from ..src.database.connection import get_db
        from ..src.database.queries import TimescaleQueries
        
        async with get_db() as session:
            # Test latency metrics summary query
            summary = await TimescaleQueries.get_latency_metrics_summary(session, hours=1)
            logger.info(f"Database summary: {len(summary)} records")
            
            # Test latency trends query
            trends = await TimescaleQueries.get_latency_trends(session, hours=1)
            logger.info(f"Database trends: {len(trends)} records")
            
            # Test high latency operations query
            high_latency = await TimescaleQueries.get_high_latency_operations(session, threshold_ms=100, hours=1)
            logger.info(f"High latency operations: {len(high_latency)} records")
            
        logger.info("✅ Database integration tests completed successfully!")
        
    except ImportError as e:
        logger.warning(f"Database not available: {e}")
    except Exception as e:
        logger.error(f"Database integration test failed: {e}")

async def main():
    """Main test function"""
    
    logger.info("🚀 Starting Latency Tracking Tests")
    
    # Test basic functionality
    await test_latency_tracking()
    
    # Test database integration
    await test_database_integration()
    
    logger.info("🎉 All tests completed!")

if __name__ == "__main__":
    asyncio.run(main())
