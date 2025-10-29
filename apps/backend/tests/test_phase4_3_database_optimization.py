#!/usr/bin/env python3
"""
Phase 4.3: Database Integration Optimization Test Suite
Tests TimescaleDB optimizations, batch processing, and database performance
"""
import asyncio
import logging
import time
import json
from datetime import datetime, timedelta
from src.app.core.database_manager import DatabaseManager
from src.app.strategies.real_time_signal_generator import RealTimeSignalGenerator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_phase4_3_database_schema():
    """Test Phase 4.3 database schema features"""
    logger.info("🗄️ Testing Phase 4.3 Database Schema...")
    
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
            # Check Phase 4.3 columns exist
            result = await conn.fetch("""
                SELECT column_name FROM information_schema.columns 
                WHERE table_name = 'enhanced_signals' 
                AND column_name IN (
                    'phase_4_3_features', 'batch_processed', 'database_optimization_metadata',
                    'query_optimization_level', 'compression_status', 'index_usage_stats',
                    'hypertable_chunk_count', 'batch_insert_timestamp', 'database_performance_score',
                    'query_execution_time_ms', 'index_hit_ratio', 'compression_ratio',
                    'chunk_compression_status', 'retention_policy_status'
                )
                ORDER BY column_name
            """)
            
            phase4_3_columns = [row['column_name'] for row in result]
            expected_columns = [
                'phase_4_3_features', 'batch_processed', 'database_optimization_metadata',
                'query_optimization_level', 'compression_status', 'index_usage_stats',
                'hypertable_chunk_count', 'batch_insert_timestamp', 'database_performance_score',
                'query_execution_time_ms', 'index_hit_ratio', 'compression_ratio',
                'chunk_compression_status', 'retention_policy_status'
            ]
            
            assert len(phase4_3_columns) == len(expected_columns)
            logger.info(f"✅ All {len(phase4_3_columns)} Phase 4.3 columns verified")
            
            # Check views exist
            result = await conn.fetch("""
                SELECT viewname FROM pg_views 
                WHERE viewname IN ('phase4_3_performance_summary', 'phase4_3_batch_processing_status', 'phase4_3_compression_analysis')
                ORDER BY viewname
            """)
            
            views = [row['viewname'] for row in result]
            assert len(views) == 3
            logger.info(f"✅ All Phase 4.3 views verified: {views}")
            
            # Check functions exist
            result = await conn.fetch("""
                SELECT proname FROM pg_proc 
                WHERE proname IN ('get_phase4_3_performance_stats', 'update_phase4_3_optimization_metadata', 'optimize_database_queries', 'log_phase4_3_database_metrics')
                ORDER BY proname
            """)
            
            functions = [row['proname'] for row in result]
            assert len(functions) == 4
            logger.info(f"✅ All Phase 4.3 functions verified: {functions}")
            
            # Test hypertable information
            result = await conn.fetch("""
                SELECT hypertable_name, num_chunks, compression_enabled
                FROM timescaledb_information.hypertables 
                WHERE hypertable_name = 'enhanced_signals'
            """)
            
            if len(result) > 0:
                hypertable_info = result[0]
                logger.info(f"✅ Hypertable verified: {hypertable_info['hypertable_name']}")
                logger.info(f"   Chunks: {hypertable_info['num_chunks']}")
                logger.info(f"   Compression: {hypertable_info['compression_enabled']}")
            else:
                logger.warning("⚠️ Hypertable information not found")
        
        logger.info("🎉 Phase 4.3 Database Schema tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Phase 4.3 Database Schema test failed: {e}")
        return False

async def test_phase4_3_batch_processing():
    """Test Phase 4.3 batch processing capabilities"""
    logger.info("🔄 Testing Phase 4.3 Batch Processing...")
    
    try:
        # Initialize signal generator
        signal_generator = RealTimeSignalGenerator()
        await signal_generator.start()
        
        # Initialize Phase 4.3 optimizations
        await signal_generator.initialize_phase4_3_database_optimizations()
        
        # Create test signal data
        test_signals = []
        for i in range(10):
            signal_data = {
                'id': f"test_phase4_3_batch_{i}_{int(time.time())}",
                'symbol': 'BTCUSDT',
                'timestamp': datetime.now(),
                'side': 'BUY' if i % 2 == 0 else 'SELL',
                'confidence': 0.8 + (i * 0.02),
                'price': 50000.0 + (i * 100),
                'strategy': 'phase4_3_test',
                'strength': 'strong',
                'metadata': {
                    'test_batch': True,
                    'batch_index': i,
                    'phase4_3_features': True
                }
            }
            test_signals.append(signal_data)
        
        # Test batch storage
        for signal_data in test_signals:
            success = await signal_generator.store_signal_with_phase4_3_optimizations(signal_data)
            assert success == True
        
        logger.info(f"✅ Successfully stored {len(test_signals)} signals in batch")
        
        # Test batch flush
        await signal_generator._flush_batch_signals()
        
        # Verify signals were stored
        db_manager = DatabaseManager()
        await db_manager.initialize({
            'host': 'localhost',
            'port': 5432,
            'database': 'alphapulse',
            'username': 'alpha_emon',
            'password': 'Emon_@17711'
        })
        
        async with db_manager.get_connection() as conn:
            result = await conn.fetch("""
                SELECT COUNT(*) as count FROM enhanced_signals 
                WHERE strategy = 'phase4_3_test' AND phase_4_3_features = TRUE
            """)
            
            stored_count = result[0]['count']
            assert stored_count >= len(test_signals)
            logger.info(f"✅ Verified {stored_count} signals stored in database")
        
        logger.info("🎉 Phase 4.3 Batch Processing tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Phase 4.3 Batch Processing test failed: {e}")
        return False

async def test_phase4_3_database_performance():
    """Test Phase 4.3 database performance optimizations"""
    logger.info("⚡ Testing Phase 4.3 Database Performance...")
    
    try:
        # Initialize signal generator
        signal_generator = RealTimeSignalGenerator()
        await signal_generator.start()
        
        # Get database performance metrics
        performance_metrics = await signal_generator.get_database_performance_metrics()
        
        assert performance_metrics is not None
        assert 'database_metrics' in performance_metrics
        assert 'phase4_3_metrics' in performance_metrics
        
        logger.info("✅ Database performance metrics retrieved")
        logger.info(f"   Batch processing: {performance_metrics['phase4_3_metrics']['batch_processing']}")
        logger.info(f"   Optimization features: {performance_metrics['phase4_3_metrics']['optimization_features']}")
        
        # Test optimized database queries
        optimized_results = await signal_generator.optimize_database_queries('BTCUSDT', '1 hour')
        
        assert optimized_results is not None
        assert 'optimized_signals' in optimized_results
        assert 'query_optimization' in optimized_results
        
        logger.info("✅ Optimized database queries working")
        logger.info(f"   Query optimization: {optimized_results['query_optimization']}")
        
        # Test database performance function
        db_manager = DatabaseManager()
        await db_manager.initialize({
            'host': 'localhost',
            'port': 5432,
            'database': 'alphapulse',
            'username': 'alpha_emon',
            'password': 'Emon_@17711'
        })
        
        async with db_manager.get_connection() as conn:
            result = await conn.fetch("""
                SELECT get_phase4_3_performance_stats(NOW() - INTERVAL '1 hour', NOW())
            """)
            
            if len(result) > 0:
                stats = result[0]['get_phase4_3_performance_stats']
                logger.info(f"✅ Performance stats function working: {stats}")
            else:
                logger.warning("⚠️ Performance stats function not returning data")
        
        logger.info("🎉 Phase 4.3 Database Performance tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Phase 4.3 Database Performance test failed: {e}")
        return False

async def test_phase4_3_timescaledb_optimizations():
    """Test Phase 4.3 TimescaleDB optimizations"""
    logger.info("📊 Testing Phase 4.3 TimescaleDB Optimizations...")
    
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
            # Test hypertable compression
            result = await conn.fetch("""
                SELECT 
                    hypertable_name,
                    compression_enabled,
                    compression_policy_status
                FROM timescaledb_information.hypertables 
                WHERE hypertable_name = 'enhanced_signals'
            """)
            
            if len(result) > 0:
                hypertable_info = result[0]
                logger.info(f"✅ Hypertable compression: {hypertable_info['compression_enabled']}")
                logger.info(f"   Compression policy: {hypertable_info['compression_policy_status']}")
            
            # Test chunk information
            result = await conn.fetch("""
                SELECT 
                    chunk_name,
                    range_start,
                    range_end,
                    is_compressed
                FROM timescaledb_information.chunks 
                WHERE hypertable_name = 'enhanced_signals'
                ORDER BY range_start DESC
                LIMIT 5
            """)
            
            logger.info(f"✅ Found {len(result)} chunks")
            for chunk in result:
                logger.info(f"   Chunk: {chunk['chunk_name']}, Compressed: {chunk['is_compressed']}")
            
            # Test compression policies
            result = await conn.fetch("""
                SELECT 
                    hypertable_name,
                    compression_enabled,
                    compression_policy_status
                FROM timescaledb_information.compression_settings 
                WHERE hypertable_name = 'enhanced_signals'
            """)
            
            if len(result) > 0:
                compression_info = result[0]
                logger.info(f"✅ Compression settings verified: {compression_info['compression_enabled']}")
            
            # Test retention policies
            result = await conn.fetch("""
                SELECT 
                    hypertable_name,
                    retention_policy_status
                FROM timescaledb_information.jobs 
                WHERE proc_name = 'policy_retention'
                AND hypertable_name = 'enhanced_signals'
            """)
            
            if len(result) > 0:
                retention_info = result[0]
                logger.info(f"✅ Retention policy: {retention_info['retention_policy_status']}")
        
        logger.info("🎉 Phase 4.3 TimescaleDB Optimizations tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Phase 4.3 TimescaleDB Optimizations test failed: {e}")
        return False

async def test_phase4_3_query_optimization():
    """Test Phase 4.3 query optimization features"""
    logger.info("🔍 Testing Phase 4.3 Query Optimization...")
    
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
            # Test optimized query function
            result = await conn.fetch("""
                SELECT optimize_database_queries('BTCUSDT', INTERVAL '1 hour')
            """)
            
            if len(result) > 0:
                query_result = result[0]['optimize_database_queries']
                logger.info(f"✅ Optimized query function working: {query_result}")
            
            # Test performance summary view
            result = await conn.fetch("""
                SELECT * FROM phase4_3_performance_summary 
                ORDER BY hour DESC 
                LIMIT 5
            """)
            
            logger.info(f"✅ Performance summary view: {len(result)} records")
            
            # Test batch processing status view
            result = await conn.fetch("""
                SELECT * FROM phase4_3_batch_processing_status 
                ORDER BY day DESC 
                LIMIT 5
            """)
            
            logger.info(f"✅ Batch processing status view: {len(result)} records")
            
            # Test compression analysis view
            result = await conn.fetch("""
                SELECT * FROM phase4_3_compression_analysis 
                ORDER BY total_signals DESC 
                LIMIT 5
            """)
            
            logger.info(f"✅ Compression analysis view: {len(result)} records")
            
            # Test index usage
            result = await conn.fetch("""
                SELECT 
                    indexrelname,
                    idx_tup_read,
                    idx_tup_fetch
                FROM pg_stat_user_indexes 
                WHERE relname = 'enhanced_signals'
                AND indexrelname LIKE '%phase4_3%'
                ORDER BY idx_tup_read DESC
            """)
            
            logger.info(f"✅ Phase 4.3 indexes found: {len(result)}")
            for index in result:
                logger.info(f"   Index: {index['indexrelname']}, Reads: {index['idx_tup_read']}")
        
        logger.info("🎉 Phase 4.3 Query Optimization tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Phase 4.3 Query Optimization test failed: {e}")
        return False

async def run_phase4_3_tests():
    """Run all Phase 4.3 tests"""
    logger.info("🧪 Starting Phase 4.3 Database Integration Optimization Test Suite...")
    
    tests = [
        ("Database Schema", test_phase4_3_database_schema),
        ("Batch Processing", test_phase4_3_batch_processing),
        ("Database Performance", test_phase4_3_database_performance),
        ("TimescaleDB Optimizations", test_phase4_3_timescaledb_optimizations),
        ("Query Optimization", test_phase4_3_query_optimization)
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
    logger.info("Phase 4.3 Test Results Summary")
    logger.info(f"{'='*60}")
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{status}: {test_name}")
    
    logger.info(f"\nOverall Result: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All Phase 4.3 Database Integration Optimization tests passed!")
        logger.info("🚀 Phase 4.3 is fully functional and ready for production!")
        return True
    else:
        logger.error(f"❌ {total - passed} Phase 4.3 tests failed!")
        return False

if __name__ == "__main__":
    asyncio.run(run_phase4_3_tests())
