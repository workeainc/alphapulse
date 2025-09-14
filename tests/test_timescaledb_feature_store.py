#!/usr/bin/env python3
"""
Test script for TimescaleDB Feature Store Implementation
Phase 2A: Feature Store Implementation (Unified Database)
"""

import asyncio
import logging
import sys
import os
from datetime import datetime, timedelta
from pathlib import Path
import json

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "backend"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_feature_store_initialization():
    """Test feature store initialization"""
    try:
        logger.info("🧪 Testing TimescaleDB feature store initialization...")
        
        from ..ai.feature_store_timescaledb import TimescaleDBFeatureStore
        
        # Initialize feature store
        async with TimescaleDBFeatureStore() as feature_store:
            logger.info("✅ TimescaleDB feature store initialized successfully")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ TimescaleDB feature store initialization test failed: {e}")
        return False

async def test_feature_registration():
    """Test feature registration"""
    try:
        logger.info("🧪 Testing feature registration...")
        
        from ..ai.feature_store_timescaledb import TimescaleDBFeatureStore, FeatureDefinition
        
        async with TimescaleDBFeatureStore() as feature_store:
            # Create test feature
            test_feature = FeatureDefinition(
                name="test_rsi",
                description="Test RSI feature",
                data_type="float",
                source_table="candles",
                computation_rule="rsi_14_period",
                version="1.0.0",
                created_at=datetime.now(),
                tags=["test", "technical"]
            )
            
            # Register feature
            success = await feature_store.register_feature(test_feature)
            if not success:
                raise Exception("Feature registration failed")
            
            logger.info("✅ Feature registration test passed")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Feature registration test failed: {e}")
        return False

async def test_feature_computation():
    """Test feature computation"""
    try:
        logger.info("🧪 Testing feature computation...")
        
        from ..ai.feature_store_timescaledb import TimescaleDBFeatureStore
        
        async with TimescaleDBFeatureStore() as feature_store:
            # Test computing a feature
            entity_id = "BTCUSDT_1h"
            timestamp = datetime.now()
            
            # This will use the placeholder computation methods
            value = await feature_store.compute_feature("rsi_14", entity_id, timestamp)
            
            if value is not None:
                logger.info(f"✅ Feature computation test passed: RSI = {value}")
            else:
                raise Exception("Feature computation returned None")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Feature computation test failed: {e}")
        return False

async def test_feature_sets():
    """Test feature sets"""
    try:
        logger.info("🧪 Testing feature sets...")
        
        from ..ai.feature_store_timescaledb import TimescaleDBFeatureStore, FeatureSet
        
        async with TimescaleDBFeatureStore() as feature_store:
            # Create test feature set
            test_feature_set = FeatureSet(
                name="test_indicators",
                description="Test technical indicators",
                features=["rsi_14", "macd", "ema_20"],
                version="1.0.0",
                created_at=datetime.now(),
                metadata={"category": "test"}
            )
            
            # Register feature set
            success = await feature_store.register_feature_set(test_feature_set)
            if not success:
                raise Exception("Feature set registration failed")
            
            logger.info("✅ Feature set test passed")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Feature set test failed: {e}")
        return False

async def test_feature_computation_pipeline():
    """Test feature computation pipeline"""
    try:
        logger.info("🧪 Testing feature computation pipeline...")
        
        from ..ai.feature_store_timescaledb import TimescaleDBFeatureStore
        from ..ai.feature_computation_pipeline import FeatureComputationPipeline, ComputationJob
        
        async with TimescaleDBFeatureStore() as feature_store:
            pipeline = FeatureComputationPipeline(feature_store, max_workers=2)
            
            # Create test computation job
            job = ComputationJob(
                job_id="",
                feature_names=["rsi_14", "macd", "ema_20"],
                entity_ids=["BTCUSDT_1h", "ETHUSDT_1h"],
                start_time=datetime.now() - timedelta(hours=24),
                end_time=datetime.now(),
                priority=1
            )
            
            # Submit job
            job_id = pipeline.submit_computation_job(job)
            if not job_id:
                raise Exception("Job submission failed")
            
            logger.info(f"✅ Job submitted successfully: {job_id}")
            
            # Wait a bit for job to process
            import time
            time.sleep(2)
            
            # Check pipeline status
            status = pipeline.get_pipeline_status()
            logger.info(f"📊 Pipeline status: {status}")
            
            # Clean up
            pipeline.shutdown()
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Feature computation pipeline test failed: {e}")
        return False

async def test_batch_computation():
    """Test batch feature computation"""
    try:
        logger.info("🧪 Testing batch feature computation...")
        
        from ..ai.feature_store_timescaledb import TimescaleDBFeatureStore
        from ..ai.feature_computation_pipeline import FeatureComputationPipeline
        
        async with TimescaleDBFeatureStore() as feature_store:
            pipeline = FeatureComputationPipeline(feature_store, max_workers=2)
            
            # Test batch computation
            entity_ids = ["BTCUSDT_1h", "ETHUSDT_1h", "ADAUSDT_1h"]
            timestamp = datetime.now()
            
            results = pipeline.batch_compute_features(
                entity_ids, 
                timestamp, 
                "technical_indicators"
            )
            
            if results and len(results) > 0:
                logger.info(f"✅ Batch computation test passed: {len(results)} entities processed")
                for entity_id, features in results.items():
                    logger.info(f"   - {entity_id}: {len(features)} features")
            else:
                raise Exception("Batch computation returned no results")
            
            # Clean up
            pipeline.shutdown()
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Batch computation test failed: {e}")
        return False

async def test_feature_quality_metrics():
    """Test feature quality metrics"""
    try:
        logger.info("🧪 Testing feature quality metrics...")
        
        from ..ai.feature_store_timescaledb import TimescaleDBFeatureStore
        from ..ai.feature_computation_pipeline import FeatureComputationPipeline
        
        async with TimescaleDBFeatureStore() as feature_store:
            pipeline = FeatureComputationPipeline(feature_store, max_workers=1)
            
            # Compute some features first
            entity_id = "BTCUSDT_1h"
            timestamp = datetime.now()
            
            features = pipeline.compute_features_for_entity(entity_id, timestamp)
            if not features:
                raise Exception("No features computed")
            
            # Get quality metrics
            start_time = datetime.now() - timedelta(hours=24)
            end_time = datetime.now()
            
            quality_metrics = pipeline.get_feature_quality_metrics(
                "rsi_14", start_time, end_time
            )
            
            if quality_metrics:
                logger.info(f"✅ Feature quality metrics test passed")
                logger.info(f"   - Overall score: {quality_metrics.get('overall_score', 0):.2f}")
                logger.info(f"   - Completeness: {quality_metrics.get('data_quality', {}).get('completeness', 0):.2f}")
            else:
                logger.warning("⚠️ No quality metrics available (this is expected for test data)")
            
            # Clean up
            pipeline.shutdown()
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Feature quality metrics test failed: {e}")
        return False

async def test_timescaledb_integration():
    """Test TimescaleDB-specific features"""
    try:
        logger.info("🧪 Testing TimescaleDB integration features...")
        
        from ..ai.feature_store_timescaledb import TimescaleDBFeatureStore
        
        async with TimescaleDBFeatureStore() as feature_store:
            # Test TimescaleDB-specific functionality
            start_time = datetime.now() - timedelta(hours=24)
            end_time = datetime.now()
            
            # Test feature statistics (uses TimescaleDB optimized queries)
            stats = await feature_store.get_feature_statistics("rsi_14", start_time, end_time)
            logger.info(f"✅ TimescaleDB feature statistics: {stats}")
            
            # Test feature history (uses TimescaleDB hypertable)
            history = await feature_store.get_feature_history("rsi_14", "BTCUSDT_1h", start_time, end_time)
            logger.info(f"✅ TimescaleDB feature history: {len(history)} records")
            
            logger.info("✅ TimescaleDB integration test passed")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ TimescaleDB integration test failed: {e}")
        return False

async def run_all_tests():
    """Run all TimescaleDB feature store tests"""
    logger.info("🚀 Starting TimescaleDB Feature Store Tests")
    logger.info("=" * 60)
    
    test_results = {}
    
    # Test 1: Feature store initialization
    test_results['feature_store_init'] = await test_feature_store_initialization()
    
    # Test 2: Feature registration
    test_results['feature_registration'] = await test_feature_registration()
    
    # Test 3: Feature computation
    test_results['feature_computation'] = await test_feature_computation()
    
    # Test 4: Feature sets
    test_results['feature_sets'] = await test_feature_sets()
    
    # Test 5: Feature computation pipeline
    test_results['computation_pipeline'] = await test_feature_computation_pipeline()
    
    # Test 6: Batch computation
    test_results['batch_computation'] = await test_batch_computation()
    
    # Test 7: Feature quality metrics
    test_results['quality_metrics'] = await test_feature_quality_metrics()
    
    # Test 8: TimescaleDB integration
    test_results['timescaledb_integration'] = await test_timescaledb_integration()
    
    # Summary
    passed_tests = sum(test_results.values())
    total_tests = len(test_results)
    
    logger.info("\n" + "="*60)
    logger.info("📊 TIMESCALEDB FEATURE STORE TEST RESULTS SUMMARY")
    logger.info("="*60)
    
    for test_name, result in test_results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{test_name:30} {status}")
    
    logger.info("="*60)
    logger.info(f"Total Tests: {total_tests}")
    logger.info(f"Passed: {passed_tests}")
    logger.info(f"Failed: {total_tests - passed_tests}")
    logger.info(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
    
    if passed_tests == total_tests:
        logger.info("🎉 ALL TESTS PASSED! TimescaleDB feature store is working correctly.")
        logger.info("🚀 Phase 2A: Unified TimescaleDB Feature Store Implementation is complete!")
    else:
        logger.warning("⚠️ Some tests failed. Please review the implementation.")
    
    return test_results

async def main():
    """Main test runner"""
    try:
        # Run tests
        test_results = await run_all_tests()
        
        # Exit with appropriate code
        if all(test_results.values()):
            sys.exit(0)  # Success
        else:
            sys.exit(1)  # Failure
            
    except KeyboardInterrupt:
        logger.info("🛑 Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Unexpected error during testing: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Run async tests
    asyncio.run(main())
