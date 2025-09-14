#!/usr/bin/env python3
"""
Phase 4.2: Memory & CPU Optimization Test Suite
Tests the new memory and CPU optimization features, resource monitoring, and database integration
"""
import asyncio
import logging
import time
import psutil
import gc
import json
from datetime import datetime, timedelta
from app.core.database_manager import DatabaseManager
from app.strategies.real_time_signal_generator import RealTimeSignalGenerator
from ai.feature_engineering import FeatureExtractor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_phase4_2_memory_optimization():
    """Test Phase 4.2 memory optimization features"""
    logger.info("🧠 Testing Phase 4.2 Memory Optimization...")
    
    try:
        # Initialize signal generator
        signal_generator = RealTimeSignalGenerator()
        await signal_generator.start()
        
        # Test memory metrics collection
        memory_metrics = await signal_generator._get_memory_metrics()
        assert memory_metrics is not None
        assert hasattr(memory_metrics, 'total_memory_mb')
        assert hasattr(memory_metrics, 'available_memory_mb')
        assert hasattr(memory_metrics, 'memory_percent')
        assert hasattr(memory_metrics, 'process_memory_mb')
        
        logger.info(f"✅ Memory metrics collected: {memory_metrics}")
        
        # Test memory optimization
        initial_memory = psutil.virtual_memory().percent
        await signal_generator._optimize_memory_usage()
        optimized_memory = psutil.virtual_memory().percent
        
        logger.info(f"✅ Memory optimization applied: {initial_memory:.2f}% -> {optimized_memory:.2f}%")
        
        # Test garbage collection optimization
        gc_stats_before = gc.get_stats()
        await signal_generator._optimize_garbage_collection()
        gc_stats_after = gc.get_stats()
        
        logger.info(f"✅ Garbage collection optimized: {len(gc_stats_before)} -> {len(gc_stats_after)} collections")
        
        # Test cache compression
        cache_size_before = len(signal_generator.result_cache)
        await signal_generator._compress_cache()
        cache_size_after = len(signal_generator.result_cache)
        
        logger.info(f"✅ Cache compression applied: {cache_size_before} -> {cache_size_after} entries")
        
        # Test memory stats
        memory_stats = await signal_generator.get_memory_stats()
        assert memory_stats is not None
        assert 'memory_usage_mb' in memory_stats
        assert 'cache_size' in memory_stats
        assert 'gc_collections' in memory_stats
        
        logger.info(f"✅ Memory stats retrieved: {memory_stats}")
        
        logger.info("🎉 Phase 4.2 Memory Optimization tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Phase 4.2 Memory Optimization test failed: {e}")
        return False

async def test_phase4_2_cpu_optimization():
    """Test Phase 4.2 CPU optimization features"""
    logger.info("⚡ Testing Phase 4.2 CPU Optimization...")
    
    try:
        # Initialize signal generator
        signal_generator = RealTimeSignalGenerator()
        await signal_generator.start()
        
        # Test CPU metrics collection
        cpu_metrics = await signal_generator._get_cpu_metrics()
        assert cpu_metrics is not None
        assert hasattr(cpu_metrics, 'cpu_percent')
        assert hasattr(cpu_metrics, 'timestamp')
        
        logger.info(f"✅ CPU metrics collected: {cpu_metrics}")
        
        # Test CPU optimization
        initial_cpu = psutil.cpu_percent(interval=1)
        await signal_generator._optimize_cpu_usage()
        optimized_cpu = psutil.cpu_percent(interval=1)
        
        logger.info(f"✅ CPU optimization applied: {initial_cpu:.2f}% -> {optimized_cpu:.2f}%")
        
        # Test CPU throttling
        await signal_generator._throttle_cpu_usage()
        throttled_cpu = psutil.cpu_percent(interval=1)
        
        logger.info(f"✅ CPU throttling applied: {throttled_cpu:.2f}%")
        
        # Test resource monitoring
        await signal_generator._monitor_resources()
        alerts = signal_generator.resource_alerts
        
        logger.info(f"✅ Resource monitoring active: {len(alerts)} alerts")
        
        logger.info("🎉 Phase 4.2 CPU Optimization tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Phase 4.2 CPU Optimization test failed: {e}")
        return False

async def test_phase4_2_feature_engineering_optimization():
    """Test Phase 4.2 feature engineering optimization"""
    logger.info("🔧 Testing Phase 4.2 Feature Engineering Optimization...")
    
    try:
        # Initialize feature extractor
        feature_extractor = FeatureExtractor()
        
        # Test memory usage monitoring
        memory_usage = feature_extractor._check_memory_usage()
        assert memory_usage is not None
        assert isinstance(memory_usage, bool)
        
        logger.info(f"✅ Memory usage monitoring: {memory_usage}")
        
        # Test memory optimization
        await feature_extractor._optimize_memory_usage()
        
        # Test cache cleanup
        cache_size_before = len(feature_extractor.feature_cache)
        await feature_extractor._cleanup_feature_cache()
        cache_size_after = len(feature_extractor.feature_cache)
        
        logger.info(f"✅ Feature cache cleanup: {cache_size_before} -> {cache_size_after} entries")
        
        # Test parallel feature extraction
        sample_data = {
            'close': [100, 101, 102, 103, 104],
            'volume': [1000, 1100, 1200, 1300, 1400],
            'high': [102, 103, 104, 105, 106],
            'low': [99, 100, 101, 102, 103]
        }
        
        features_parallel = await feature_extractor._parallel_feature_extraction(sample_data)
        features_sequential = await feature_extractor._extract_features_sequential(sample_data)
        
        assert features_parallel is not None
        assert features_sequential is not None
        
        logger.info(f"✅ Parallel feature extraction: {len(features_parallel)} features")
        logger.info(f"✅ Sequential feature extraction: {len(features_sequential)} features")
        
        # Test Phase 4.2 stats
        stats = feature_extractor.get_phase4_2_stats()
        assert stats is not None
        assert 'total_extractions' in stats
        assert 'cache_hits' in stats
        assert 'cache_misses' in stats
        
        logger.info(f"✅ Phase 4.2 stats: {stats}")
        
        logger.info("🎉 Phase 4.2 Feature Engineering Optimization tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Phase 4.2 Feature Engineering Optimization test failed: {e}")
        return False

async def test_phase4_2_database_integration():
    """Test Phase 4.2 database integration"""
    logger.info("🗄️ Testing Phase 4.2 Database Integration...")
    
    try:
        db_manager = DatabaseManager()
        await db_manager.initialize({
            'host': 'localhost',
            'port': 5432,
            'database': 'alphapulse',
            'username': 'alpha_emon',
            'password': 'Emon_@17711'
        })
        
        async with db_manager.get_connection() as conn:
            # Test Phase 4.2 columns exist
            result = await conn.fetch("""
                SELECT column_name FROM information_schema.columns 
                WHERE table_name = 'enhanced_signals' 
                AND column_name IN (
                    'memory_usage_mb', 'cpu_usage_percent', 'cache_size', 'gc_collections',
                    'gc_objects_freed', 'memory_cleanups', 'cache_hit_rate', 'processing_time_ms',
                    'throughput_per_second', 'latency_percentile_95', 'resource_alerts',
                    'optimization_enabled', 'memory_pressure_level', 'cpu_throttling_applied',
                    'cache_compression_applied', 'gc_optimization_applied', 'phase_4_2_features',
                    'memory_optimization_metadata'
                )
                ORDER BY column_name
            """)
            
            phase4_2_columns = [row['column_name'] for row in result]
            expected_columns = [
                'memory_usage_mb', 'cpu_usage_percent', 'cache_size', 'gc_collections',
                'gc_objects_freed', 'memory_cleanups', 'cache_hit_rate', 'processing_time_ms',
                'throughput_per_second', 'latency_percentile_95', 'resource_alerts',
                'optimization_enabled', 'memory_pressure_level', 'cpu_throttling_applied',
                'cache_compression_applied', 'gc_optimization_applied', 'phase_4_2_features',
                'memory_optimization_metadata'
            ]
            
            assert len(phase4_2_columns) == len(expected_columns)
            logger.info(f"✅ All {len(phase4_2_columns)} Phase 4.2 columns verified")
            
            # Test Phase 4.2 views exist
            result = await conn.fetch("""
                SELECT viewname FROM pg_views 
                WHERE viewname IN ('phase4_2_performance_summary', 'phase4_2_resource_alerts', 'phase4_2_optimization_stats')
                ORDER BY viewname
            """)
            
            views = [row['viewname'] for row in result]
            assert len(views) == 3
            logger.info(f"✅ All Phase 4.2 views verified: {views}")
            
            # Test Phase 4.2 functions exist
            result = await conn.fetch("""
                SELECT proname FROM pg_proc 
                WHERE proname IN ('get_phase4_2_performance_stats', 'update_phase4_2_optimization_metadata', 'log_phase4_2_performance_metrics')
                ORDER BY proname
            """)
            
            functions = [row['proname'] for row in result]
            assert len(functions) == 3
            logger.info(f"✅ All Phase 4.2 functions verified: {functions}")
            
            # Test inserting Phase 4.2 data
            test_signal_id = f"test_phase4_2_{int(time.time())}"
            
            await conn.execute("""
                INSERT INTO enhanced_signals (
                    id, symbol, timestamp, signal_type, confidence, price, 
                    memory_usage_mb, cpu_usage_percent, cache_size, gc_collections,
                    gc_objects_freed, memory_cleanups, cache_hit_rate, processing_time_ms,
                    throughput_per_second, latency_percentile_95, resource_alerts,
                    optimization_enabled, memory_pressure_level, cpu_throttling_applied,
                    cache_compression_applied, gc_optimization_applied, phase_4_2_features,
                    memory_optimization_metadata
                ) VALUES (
                    $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24
                )
            """, (
                test_signal_id, 'BTCUSDT', datetime.now(), 'BUY', 0.85, 50000.0,
                512.5, 45.2, 1000, 5, 1500, 3, 0.85, 75.5,
                13.3, 95.2, json.dumps({'memory_alert': 'Normal', 'cpu_alert': 'Normal'}),
                json.dumps({'memory_optimization': True, 'cpu_optimization': True}),
                0.3, False, False, True, True,
                json.dumps({
                    'optimization_version': '4.2',
                    'memory_optimization_enabled': True,
                    'cpu_optimization_enabled': True,
                    'test_data': True
                })
            ))
            
            logger.info(f"✅ Phase 4.2 test signal inserted: {test_signal_id}")
            
            # Test retrieving Phase 4.2 data
            result = await conn.fetch("""
                SELECT 
                    memory_usage_mb, cpu_usage_percent, cache_size, gc_collections,
                    cache_hit_rate, processing_time_ms, memory_pressure_level,
                    cpu_throttling_applied, cache_compression_applied, gc_optimization_applied
                FROM enhanced_signals 
                WHERE id = $1
            """, test_signal_id)
            
            assert len(result) == 1
            row = result[0]
            assert row['memory_usage_mb'] == 512.5
            assert row['cpu_usage_percent'] == 45.2
            assert row['cache_size'] == 1000
            assert row['gc_collections'] == 5
            assert row['cache_hit_rate'] == 0.85
            assert row['processing_time_ms'] == 75.5
            assert row['memory_pressure_level'] == 0.3
            assert row['cpu_throttling_applied'] == False
            assert row['cache_compression_applied'] == False
            assert row['gc_optimization_applied'] == True
            
            logger.info("✅ Phase 4.2 data retrieval verified")
            
            # Test performance stats function
            result = await conn.fetch("""
                SELECT get_phase4_2_performance_stats(NOW() - INTERVAL '1 hour', NOW())
            """)
            
            assert len(result) == 1
            stats = result[0]['get_phase4_2_performance_stats']
            assert stats is not None
            assert 'total_signals' in stats
            assert 'avg_memory_usage_mb' in stats
            
            logger.info(f"✅ Performance stats function verified: {stats}")
            
            # Cleanup test data
            await conn.execute("DELETE FROM enhanced_signals WHERE id = $1", test_signal_id)
            logger.info("✅ Test data cleaned up")
        
        logger.info("🎉 Phase 4.2 Database Integration tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Phase 4.2 Database Integration test failed: {e}")
        return False

async def test_phase4_2_signal_generation_with_optimization():
    """Test signal generation with Phase 4.2 optimizations"""
    logger.info("🚀 Testing Phase 4.2 Signal Generation with Optimization...")
    
    try:
        # Initialize signal generator
        signal_generator = RealTimeSignalGenerator()
        await signal_generator.start()
        
        # Test signal generation with optimization enabled
        start_time = time.time()
        
        # Generate test market data
        test_market_data = {
            'BTCUSDT': {
                '1h': {
                    'close': [50000, 50100, 50200, 50300, 50400],
                    'volume': [1000, 1100, 1200, 1300, 1400],
                    'high': [50200, 50300, 50400, 50500, 50600],
                    'low': [49900, 50000, 50100, 50200, 50300],
                    'open': [50000, 50100, 50200, 50300, 50400]
                }
            }
        }
        
        # Generate signals with Phase 4.2 optimizations
        signals = await signal_generator.generate_signals(test_market_data)
        
        end_time = time.time()
        processing_time_ms = (end_time - start_time) * 1000
        
        assert signals is not None
        assert len(signals) > 0
        
        logger.info(f"✅ Signals generated with Phase 4.2 optimizations: {len(signals)} signals")
        logger.info(f"✅ Processing time: {processing_time_ms:.2f}ms")
        
        # Check if optimizations were applied
        memory_stats = await signal_generator.get_memory_stats()
        assert memory_stats['memory_optimization_enabled'] == True
        assert memory_stats['cpu_optimization_enabled'] == True
        
        logger.info(f"✅ Optimizations applied: {memory_stats}")
        
        # Test performance metrics
        performance_metrics = signal_generator.performance_monitor.get_metrics() if hasattr(signal_generator, 'performance_monitor') else {}
        if performance_metrics:
            logger.info(f"✅ Performance metrics: {performance_metrics}")
        
        logger.info("🎉 Phase 4.2 Signal Generation with Optimization tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Phase 4.2 Signal Generation with Optimization test failed: {e}")
        return False

async def test_phase4_2_resource_monitoring():
    """Test Phase 4.2 resource monitoring capabilities"""
    logger.info("📊 Testing Phase 4.2 Resource Monitoring...")
    
    try:
        # Initialize signal generator
        signal_generator = RealTimeSignalGenerator()
        await signal_generator.start()
        
        # Test resource monitoring
        await signal_generator._monitor_resources()
        
        # Check resource alerts
        alerts = signal_generator.resource_alerts
        assert isinstance(alerts, list)
        
        logger.info(f"✅ Resource monitoring active: {len(alerts)} alerts")
        
        # Test memory pressure detection
        memory_pressure = await signal_generator._get_memory_metrics()
        assert memory_pressure.memory_pressure_level >= 0
        assert memory_pressure.memory_pressure_level <= 1
        
        logger.info(f"✅ Memory pressure level: {memory_pressure.memory_pressure_level:.3f}")
        
        # Test CPU usage monitoring
        cpu_metrics = await signal_generator._get_cpu_metrics()
        assert cpu_metrics.cpu_percent >= 0
        assert cpu_metrics.cpu_percent <= 100
        
        logger.info(f"✅ CPU usage: {cpu_metrics.cpu_percent:.2f}%")
        
        # Test resource alert checking
        alerts = await signal_generator._check_resource_alerts()
        assert isinstance(alerts, list)
        
        logger.info(f"✅ Resource alerts checked: {len(alerts)} alerts")
        
        logger.info("🎉 Phase 4.2 Resource Monitoring tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Phase 4.2 Resource Monitoring test failed: {e}")
        return False

async def run_phase4_2_tests():
    """Run all Phase 4.2 tests"""
    logger.info("🧪 Starting Phase 4.2 Memory & CPU Optimization Test Suite...")
    
    tests = [
        ("Memory Optimization", test_phase4_2_memory_optimization),
        ("CPU Optimization", test_phase4_2_cpu_optimization),
        ("Feature Engineering Optimization", test_phase4_2_feature_engineering_optimization),
        ("Database Integration", test_phase4_2_database_integration),
        ("Signal Generation with Optimization", test_phase4_2_signal_generation_with_optimization),
        ("Resource Monitoring", test_phase4_2_resource_monitoring)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*60}")
        logger.info(f"Running: {test_name}")
        logger.info(f"{'='*60}")
        
        try:
            result = await test_func()
            results[test_name] = result
            status = "✅ PASSED" if result else "❌ FAILED"
            logger.info(f"{status}: {test_name}")
        except Exception as e:
            logger.error(f"❌ ERROR in {test_name}: {e}")
            results[test_name] = False
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("Phase 4.2 Test Results Summary")
    logger.info(f"{'='*60}")
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{status}: {test_name}")
    
    logger.info(f"\nOverall Result: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All Phase 4.2 Memory & CPU Optimization tests passed!")
        return True
    else:
        logger.error(f"❌ {total - passed} Phase 4.2 tests failed!")
        return False

if __name__ == "__main__":
    asyncio.run(run_phase4_2_tests())
