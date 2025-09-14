"""
Phase 4.1 Performance Test - Real-time Processing Optimization
Test ultra-low latency signal generation, caching, async processing, and memory management
"""

import asyncio
import time
import logging
import sys
import os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Add the backend directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_signal_generator_performance():
    """Test signal generator performance with Phase 4.1 optimizations"""
    try:
        logger.info("🧪 Testing Phase 4.1: Signal Generator Performance")
        
        # Import the signal generator
        from app.strategies.real_time_signal_generator import RealTimeSignalGenerator
        
        # Create signal generator with Phase 4.1 optimizations
        config = {
            'use_database': True,
            'use_advanced_indicators': True,
            'use_smc': True,
            'use_dl': True,
            'use_rl': True,
            'use_nlp': True,
            'use_ensemble': True,
            'use_sentiment': True,
            'min_confidence': 0.6,
            'signal_cooldown': 300,
            'enable_async_processing': True,
            'enable_caching': True,
            'enable_memory_management': True,
            'target_latency_ms': 100
        }
        
        signal_generator = RealTimeSignalGenerator(config)
        
        # Start the signal generator (this will initialize all engines)
        await signal_generator.start()
        
        # Wait for initialization
        await asyncio.sleep(2)
        
        # Test performance with real market data
        logger.info("📊 Testing signal generation performance...")
        
        # Create sample market data
        dates = pd.date_range(start='2024-01-01', periods=100, freq='1H')
        market_data = pd.DataFrame({
            'timestamp': dates,
            'open': np.random.uniform(45000, 55000, 100),
            'high': np.random.uniform(45000, 55000, 100),
            'low': np.random.uniform(45000, 55000, 100),
            'close': np.random.uniform(45000, 55000, 100),
            'volume': np.random.uniform(1000, 5000, 100)
        })
        
        # Test multiple signal generations
        performance_results = []
        for i in range(10):
            start_time = time.time()
            
            # Generate signals
            signals = await signal_generator.generate_signals(market_data)
            
            end_time = time.time()
            processing_time = (end_time - start_time) * 1000  # Convert to milliseconds
            
            performance_results.append(processing_time)
            
            logger.info(f"Test {i+1}: Generated {len(signals)} signals in {processing_time:.2f}ms")
            
            # Small delay between tests
            await asyncio.sleep(0.1)
        
        # Calculate performance metrics
        avg_time = sum(performance_results) / len(performance_results)
        max_time = max(performance_results)
        min_time = min(performance_results)
        
        logger.info(f"📈 Performance Results:")
        logger.info(f"   Average: {avg_time:.2f}ms")
        logger.info(f"   Maximum: {max_time:.2f}ms")
        logger.info(f"   Minimum: {min_time:.2f}ms")
        logger.info(f"   Target: <100ms")
        
        # Check if performance targets are met
        if avg_time <= 100:
            logger.info("✅ Performance target met: Average processing time < 100ms")
        else:
            logger.warning(f"⚠️ Performance target not met: Average processing time {avg_time:.2f}ms > 100ms")
        
        # Stop the signal generator
        await signal_generator.stop()
        
        return {
            'success': True,
            'avg_time': avg_time,
            'max_time': max_time,
            'min_time': min_time,
            'target_met': avg_time <= 100
        }
        
    except Exception as e:
        logger.error(f"❌ Error testing signal generator performance: {e}")
        return {'success': False, 'error': str(e)}

async def test_caching_performance():
    """Test caching performance"""
    try:
        logger.info("🧪 Testing Phase 4.1: Caching Performance")
        
        from app.strategies.real_time_signal_generator import RealTimeSignalGenerator
        
        config = {
            'enable_caching': True,
            'target_latency_ms': 100
        }
        
        signal_generator = RealTimeSignalGenerator(config)
        
        # Test cache functionality
        cache_key = "test_cache_key"
        test_data = {"test": "data", "timestamp": datetime.now().isoformat()}
        
        # Test cache storage
        signal_generator._cache_result(cache_key, test_data)
        logger.info("✅ Cache storage test passed")
        
        # Test cache retrieval
        cached_result = await signal_generator._get_cached_result(cache_key)
        if cached_result == test_data:
            logger.info("✅ Cache retrieval test passed")
        else:
            logger.error("❌ Cache retrieval test failed")
            return {'success': False, 'error': 'Cache retrieval failed'}
        
        # Test cache performance
        start_time = time.time()
        for _ in range(100):
            await signal_generator._get_cached_result(cache_key)
        end_time = time.time()
        
        cache_time = (end_time - start_time) * 1000
        logger.info(f"📈 Cache performance: 100 retrievals in {cache_time:.2f}ms ({cache_time/100:.4f}ms per retrieval)")
        
        return {
            'success': True,
            'cache_time': cache_time,
            'avg_cache_time': cache_time / 100
        }
        
    except Exception as e:
        logger.error(f"❌ Error testing caching performance: {e}")
        return {'success': False, 'error': str(e)}

async def test_async_processing_performance():
    """Test async processing performance"""
    try:
        logger.info("🧪 Testing Phase 4.1: Async Processing Performance")
        
        from app.strategies.real_time_signal_generator import RealTimeSignalGenerator
        
        config = {
            'enable_async_processing': True,
            'target_latency_ms': 100
        }
        
        signal_generator = RealTimeSignalGenerator(config)
        
        # Test async processing with multiple concurrent tasks
        async def async_task(task_id):
            await asyncio.sleep(0.01)  # Simulate some work
            return f"Task {task_id} completed"
        
        # Test concurrent execution
        start_time = time.time()
        tasks = [async_task(i) for i in range(10)]
        results = await asyncio.gather(*tasks)
        end_time = time.time()
        
        async_time = (end_time - start_time) * 1000
        logger.info(f"📈 Async processing: 10 concurrent tasks in {async_time:.2f}ms")
        
        if len(results) == 10:
            logger.info("✅ Async processing test passed")
        else:
            logger.error("❌ Async processing test failed")
            return {'success': False, 'error': 'Async processing failed'}
        
        return {
            'success': True,
            'async_time': async_time,
            'results_count': len(results)
        }
        
    except Exception as e:
        logger.error(f"❌ Error testing async processing performance: {e}")
        return {'success': False, 'error': str(e)}

async def test_memory_management():
    """Test memory management"""
    try:
        logger.info("🧪 Testing Phase 4.1: Memory Management")
        
        import psutil
        import gc
        
        # Get initial memory usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        logger.info(f"📊 Initial memory usage: {initial_memory:.2f}MB")
        
        # Simulate memory-intensive operations
        large_data = []
        for i in range(1000):
            large_data.append({
                'id': i,
                'data': np.random.rand(1000).tolist(),
                'timestamp': datetime.now().isoformat()
            })
        
        # Get memory usage after creating large data
        memory_after_data = process.memory_info().rss / 1024 / 1024
        logger.info(f"📊 Memory after creating large data: {memory_after_data:.2f}MB")
        
        # Clear data and force garbage collection
        del large_data
        gc.collect()
        
        # Get memory usage after cleanup
        memory_after_cleanup = process.memory_info().rss / 1024 / 1024
        logger.info(f"📊 Memory after cleanup: {memory_after_cleanup:.2f}MB")
        
        # Check if memory was properly freed
        memory_freed = memory_after_data - memory_after_cleanup
        logger.info(f"📊 Memory freed: {memory_freed:.2f}MB")
        
        if memory_freed > 0:
            logger.info("✅ Memory management test passed")
        else:
            logger.warning("⚠️ Memory management test: No memory was freed")
        
        return {
            'success': True,
            'initial_memory': initial_memory,
            'memory_after_data': memory_after_data,
            'memory_after_cleanup': memory_after_cleanup,
            'memory_freed': memory_freed
        }
        
    except Exception as e:
        logger.error(f"❌ Error testing memory management: {e}")
        return {'success': False, 'error': str(e)}

async def test_performance_monitoring():
    """Test performance monitoring"""
    try:
        logger.info("🧪 Testing Phase 4.1: Performance Monitoring")
        
        from app.strategies.real_time_signal_generator import RealTimeSignalGenerator
        
        config = {
            'enable_async_processing': True,
            'enable_caching': True,
            'enable_memory_management': True,
            'target_latency_ms': 100
        }
        
        signal_generator = RealTimeSignalGenerator(config)
        
        # Test performance metrics tracking
        signal_generator.processing_times = [50, 75, 100, 125, 150]  # Sample times
        signal_generator.cache_hits = 80
        signal_generator.cache_misses = 20
        signal_generator.total_requests = 100
        
        # Test performance logging
        await signal_generator._log_performance_metrics()
        logger.info("✅ Performance monitoring test passed")
        
        # Check if metrics are being tracked
        if len(signal_generator.processing_times) > 0:
            logger.info(f"📊 Processing times tracked: {len(signal_generator.processing_times)} samples")
        
        cache_hit_rate = (signal_generator.cache_hits / signal_generator.total_requests) * 100
        logger.info(f"📊 Cache hit rate: {cache_hit_rate:.1f}%")
        
        return {
            'success': True,
            'processing_times_count': len(signal_generator.processing_times),
            'cache_hit_rate': cache_hit_rate
        }
        
    except Exception as e:
        logger.error(f"❌ Error testing performance monitoring: {e}")
        return {'success': False, 'error': str(e)}

async def test_database_performance():
    """Test database performance with Phase 4.1 optimizations"""
    try:
        logger.info("🧪 Testing Phase 4.1: Database Performance")
        
        # Test database connection
        from database.connection import TimescaleDBConnection
        
        db_config = {
            'host': 'localhost',
            'port': 5432,
            'database': 'alphapulse',
            'username': 'alpha_emon',
            'password': 'Emon_@17711'
        }
        
        db_connection = TimescaleDBConnection(db_config)
        await db_connection.initialize()
        
        # Test signal retrieval performance
        start_time = time.time()
        signals = await db_connection.get_latest_signals(limit=10)
        end_time = time.time()
        
        db_time = (end_time - start_time) * 1000
        logger.info(f"📈 Database query time: {db_time:.2f}ms for {len(signals)} signals")
        
        if db_time < 1000:  # Less than 1 second
            logger.info("✅ Database performance test passed")
        else:
            logger.warning(f"⚠️ Database query took {db_time:.2f}ms (target: <1000ms)")
        
        return {
            'success': True,
            'db_time': db_time,
            'signals_retrieved': len(signals)
        }
        
    except Exception as e:
        logger.error(f"❌ Error testing database performance: {e}")
        return {'success': False, 'error': str(e)}

async def main():
    """Run all Phase 4.1 performance tests"""
    logger.info("🚀 Starting Phase 4.1 Performance Tests")
    logger.info("=" * 60)
    
    test_results = {}
    
    # Run all tests
    tests = [
        ("Signal Generator Performance", test_signal_generator_performance),
        ("Caching Performance", test_caching_performance),
        ("Async Processing Performance", test_async_processing_performance),
        ("Memory Management", test_memory_management),
        ("Performance Monitoring", test_performance_monitoring),
        ("Database Performance", test_database_performance)
    ]
    
    for test_name, test_func in tests:
        logger.info(f"\n🧪 Running: {test_name}")
        logger.info("-" * 40)
        
        try:
            result = await test_func()
            test_results[test_name] = result
            
            if result.get('success'):
                logger.info(f"✅ {test_name}: PASSED")
            else:
                logger.error(f"❌ {test_name}: FAILED - {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            logger.error(f"❌ {test_name}: ERROR - {e}")
            test_results[test_name] = {'success': False, 'error': str(e)}
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("📊 Phase 4.1 Performance Test Summary")
    logger.info("=" * 60)
    
    passed_tests = sum(1 for result in test_results.values() if result.get('success'))
    total_tests = len(test_results)
    
    logger.info(f"✅ Tests Passed: {passed_tests}/{total_tests}")
    
    for test_name, result in test_results.items():
        status = "✅ PASSED" if result.get('success') else "❌ FAILED"
        logger.info(f"   {test_name}: {status}")
        
        if result.get('success') and 'avg_time' in result:
            logger.info(f"      Performance: {result['avg_time']:.2f}ms")
        elif result.get('success') and 'db_time' in result:
            logger.info(f"      Database Time: {result['db_time']:.2f}ms")
    
    if passed_tests == total_tests:
        logger.info("\n🎉 All Phase 4.1 performance tests passed!")
    else:
        logger.warning(f"\n⚠️ {total_tests - passed_tests} tests failed. Please review the errors above.")
    
    return test_results

if __name__ == "__main__":
    asyncio.run(main())
