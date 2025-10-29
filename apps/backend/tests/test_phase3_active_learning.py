#!/usr/bin/env python3
"""
Test script for Phase 3 - Priority 7: Active Learning Loop
Tests the Active Learning Service functionality including low-confidence prediction capture,
manual labeling, and integration with retrain queue.
"""

import asyncio
import logging
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from sqlalchemy import text

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import the active learning service
try:
    from src.app.services.active_learning_service import ActiveLearningService, ActiveLearningItem, PredictionLabel
    ACTIVE_LEARNING_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Active learning service not available: {e}")
    ACTIVE_LEARNING_AVAILABLE = False

# Import database components
try:
    from ..src.database.connection_simple import SimpleTimescaleDBConnection
    from ..src.database.data_versioning_dao import DataVersioningDAO
    DATABASE_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Database components not available: {e}")
    DATABASE_AVAILABLE = False


def generate_test_features() -> Dict[str, Any]:
    """Generate test features for prediction"""
    return {
        'rsi': np.random.uniform(30, 70),
        'macd': np.random.uniform(-0.1, 0.1),
        'bb_position': np.random.uniform(0, 1),
        'volume_ratio': np.random.uniform(0.5, 2.0),
        'price_change': np.random.uniform(-0.05, 0.05),
        'volatility': np.random.uniform(0.01, 0.05)
    }


def generate_test_market_data() -> Dict[str, Any]:
    """Generate test market data"""
    return {
        'open': 100.0 + np.random.uniform(-5, 5),
        'high': 105.0 + np.random.uniform(-5, 5),
        'low': 95.0 + np.random.uniform(-5, 5),
        'close': 100.0 + np.random.uniform(-5, 5),
        'volume': 1000000 + np.random.uniform(-100000, 100000),
        'timestamp': datetime.now().isoformat()
    }


async def test_database_connection():
    """Test database connection and active learning table"""
    logger.info("🧪 Testing database connection...")
    
    if not DATABASE_AVAILABLE:
        logger.warning("❌ Database components not available, skipping test")
        return False
    
    try:
        db_connection = SimpleTimescaleDBConnection()
        session_factory = await db_connection.get_async_session()
        async with session_factory as session:
            # Test if active_learning_queue table exists
            result = await session.execute(text("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_name = 'active_learning_queue'
            """))
            
            table_exists = result.fetchone() is not None
            
            if table_exists:
                logger.info("✅ Active learning queue table exists")
                
                # Test if views exist
                result = await session.execute(text("""
                    SELECT table_name 
                    FROM information_schema.views 
                    WHERE table_name LIKE 'active_learning_%'
                """))
                
                views = result.fetchall()
                logger.info(f"✅ Found {len(views)} active learning views")
                
                return True
            else:
                logger.error("❌ Active learning queue table does not exist")
                return False
                
    except Exception as e:
        logger.error(f"❌ Database connection test failed: {e}")
        return False


async def test_active_learning_service_initialization():
    """Test active learning service initialization"""
    logger.info("🧪 Testing active learning service initialization...")
    
    if not ACTIVE_LEARNING_AVAILABLE:
        logger.warning("❌ Active learning service not available, skipping test")
        return False
    
    try:
        # Test with default parameters
        service = ActiveLearningService()
        
        # Check configuration
        assert service.confidence_low == 0.45
        assert service.confidence_high == 0.55
        assert service.max_queue_size == 1000
        assert service.auto_cleanup_days == 30
        assert not service.is_running
        
        logger.info("✅ Active learning service initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Active learning service initialization failed: {e}")
        return False


async def test_low_confidence_capture():
    """Test capturing low-confidence predictions"""
    logger.info("🧪 Testing low-confidence prediction capture...")
    
    if not ACTIVE_LEARNING_AVAILABLE or not DATABASE_AVAILABLE:
        logger.warning("❌ Required components not available, skipping test")
        return False
    
    try:
        service = ActiveLearningService()
        await service.start()
        
        # Test cases with different confidence levels
        test_cases = [
            # (confidence, should_be_captured, description)
            (0.44, False, "Below low confidence range"),
            (0.45, True, "At low confidence boundary"),
            (0.50, True, "Middle of low confidence range"),
            (0.55, True, "At high confidence boundary"),
            (0.56, False, "Above high confidence range"),
            (0.48, True, "High priority range"),
            (0.52, True, "High priority range"),
            (0.46, True, "Medium priority range"),
            (0.54, True, "Medium priority range")
        ]
        
        captured_count = 0
        
        for confidence, should_capture, description in test_cases:
            queue_id = await service.capture_low_confidence_prediction(
                signal_id=1,
                symbol="BTCUSDT",
                timeframe="1h",
                prediction_confidence=confidence,
                predicted_label="BUY",
                predicted_probability=confidence,
                features=generate_test_features(),
                market_data=generate_test_market_data(),
                model_id="test_model_v1",
                timestamp=datetime.now()
            )
            
            if should_capture and queue_id is not None:
                captured_count += 1
                logger.info(f"✅ Captured prediction with confidence {confidence:.3f} ({description})")
            elif not should_capture and queue_id is None:
                logger.info(f"✅ Correctly skipped prediction with confidence {confidence:.3f} ({description})")
            else:
                logger.error(f"❌ Incorrect capture behavior for confidence {confidence:.3f} ({description})")
                return False
        
        logger.info(f"✅ Low-confidence capture test completed. Captured {captured_count} predictions.")
        
        await service.stop()
        return True
        
    except Exception as e:
        logger.error(f"❌ Low-confidence capture test failed: {e}")
        return False


async def test_manual_labeling():
    """Test manual labeling functionality"""
    logger.info("🧪 Testing manual labeling functionality...")
    
    if not ACTIVE_LEARNING_AVAILABLE or not DATABASE_AVAILABLE:
        logger.warning("❌ Required components not available, skipping test")
        return False
    
    try:
        service = ActiveLearningService()
        await service.start()
        
        # First, capture a low-confidence prediction
        queue_id = await service.capture_low_confidence_prediction(
            signal_id=2,
            symbol="ETHUSDT",
            timeframe="4h",
            prediction_confidence=0.49,
            predicted_label="SELL",
            predicted_probability=0.49,
            features=generate_test_features(),
            market_data=generate_test_market_data(),
            model_id="test_model_v2",
            timestamp=datetime.now()
        )
        
        if queue_id is None:
            logger.error("❌ Could not capture prediction for labeling test")
            return False
        
        logger.info(f"📝 Created test item with queue_id: {queue_id}")
        
        # Test labeling with valid label
        logger.info(f"🔍 Attempting to label item {queue_id} as BUY...")
        success = await service.label_item(
            queue_id=queue_id,
            manual_label="BUY",
            labeled_by="test_user",
            labeling_notes="Test labeling"
        )
        
        if success:
            logger.info("✅ Manual labeling successful")
        else:
            logger.error("❌ Manual labeling failed")
            return False
        
        # Test labeling with invalid label
        success = await service.label_item(
            queue_id=queue_id,
            manual_label="INVALID",
            labeled_by="test_user"
        )
        
        if not success:
            logger.info("✅ Correctly rejected invalid label")
        else:
            logger.error("❌ Should have rejected invalid label")
            return False
        
        await service.stop()
        return True
        
    except Exception as e:
        logger.error(f"❌ Manual labeling test failed: {e}")
        return False


async def test_statistics():
    """Test statistics functionality"""
    logger.info("🧪 Testing statistics functionality...")
    
    if not ACTIVE_LEARNING_AVAILABLE or not DATABASE_AVAILABLE:
        logger.warning("❌ Required components not available, skipping test")
        return False
    
    try:
        service = ActiveLearningService()
        await service.start()
        
        # Capture some test predictions
        for i in range(5):
            await service.capture_low_confidence_prediction(
                signal_id=10 + i,
                symbol=f"TEST{i}",
                timeframe="1h",
                prediction_confidence=0.45 + (i * 0.02),
                predicted_label="BUY" if i % 2 == 0 else "SELL",
                predicted_probability=0.45 + (i * 0.02),
                features=generate_test_features(),
                market_data=generate_test_market_data(),
                model_id=f"test_model_v{i}",
                timestamp=datetime.now()
            )
        
        # Get statistics
        stats = await service.get_statistics()
        
        # Verify statistics structure
        assert hasattr(stats, 'total_items')
        assert hasattr(stats, 'pending_items')
        assert hasattr(stats, 'labeled_items')
        assert hasattr(stats, 'avg_confidence')
        assert hasattr(stats, 'label_distribution')
        assert hasattr(stats, 'model_distribution')
        
        logger.info(f"✅ Statistics test completed. Total items: {stats.total_items}")
        
        await service.stop()
        return True
        
    except Exception as e:
        logger.error(f"❌ Statistics test failed: {e}")
        return False


async def test_pending_items_retrieval():
    """Test retrieving pending items"""
    logger.info("🧪 Testing pending items retrieval...")
    
    if not ACTIVE_LEARNING_AVAILABLE or not DATABASE_AVAILABLE:
        logger.warning("❌ Required components not available, skipping test")
        return False
    
    try:
        service = ActiveLearningService()
        await service.start()
        
        # Capture some test predictions
        for i in range(3):
            await service.capture_low_confidence_prediction(
                signal_id=20 + i,
                symbol="BTCUSDT",
                timeframe="1h",
                prediction_confidence=0.48 + (i * 0.02),
                predicted_label="BUY",
                predicted_probability=0.48 + (i * 0.02),
                features=generate_test_features(),
                market_data=generate_test_market_data(),
                model_id="test_model_v3",
                timestamp=datetime.now()
            )
        
        # Get pending items
        items = await service.get_pending_items(limit=10)
        
        if len(items) > 0:
            logger.info(f"✅ Retrieved {len(items)} pending items")
            
            # Verify item structure
            item = items[0]
            assert hasattr(item, 'id')
            assert hasattr(item, 'symbol')
            assert hasattr(item, 'prediction_confidence')
            assert hasattr(item, 'predicted_label')
            assert hasattr(item, 'status')
            
            logger.info(f"✅ Item structure verified. Sample item ID: {item.id}")
        else:
            logger.warning("⚠️ No pending items found")
        
        await service.stop()
        return True
        
    except Exception as e:
        logger.error(f"❌ Pending items retrieval test failed: {e}")
        return False


async def test_skip_functionality():
    """Test skipping items"""
    logger.info("🧪 Testing skip functionality...")
    
    if not ACTIVE_LEARNING_AVAILABLE or not DATABASE_AVAILABLE:
        logger.warning("❌ Required components not available, skipping test")
        return False
    
    try:
        service = ActiveLearningService()
        await service.start()
        
        # Capture a test prediction
        queue_id = await service.capture_low_confidence_prediction(
            signal_id=30,
            symbol="ADAUSDT",
            timeframe="1h",
            prediction_confidence=0.47,
            predicted_label="HOLD",
            predicted_probability=0.47,
            features=generate_test_features(),
            market_data=generate_test_market_data(),
            model_id="test_model_v4",
            timestamp=datetime.now()
        )
        
        if queue_id is None:
            logger.error("❌ Could not capture prediction for skip test")
            return False
        
        # Test skipping with reason
        success = await service.skip_item(
            queue_id=queue_id,
            reason="Test skip functionality"
        )
        
        if success:
            logger.info("✅ Skip functionality successful")
        else:
            logger.error("❌ Skip functionality failed")
            return False
        
        await service.stop()
        return True
        
    except Exception as e:
        logger.error(f"❌ Skip functionality test failed: {e}")
        return False


async def test_service_stats():
    """Test service statistics"""
    logger.info("🧪 Testing service statistics...")
    
    if not ACTIVE_LEARNING_AVAILABLE:
        logger.warning("❌ Active learning service not available, skipping test")
        return False
    
    try:
        service = ActiveLearningService()
        
        # Get service stats before starting
        stats = service.get_service_stats()
        
        # Verify stats structure
        assert 'service_running' in stats
        assert 'confidence_range' in stats
        assert 'max_queue_size' in stats
        assert 'items_captured' in stats
        assert 'items_labeled' in stats
        
        logger.info(f"✅ Service stats test completed. Confidence range: {stats['confidence_range']}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Service stats test failed: {e}")
        return False


async def cleanup_test_data():
    """Clean up test data from database"""
    logger.info("🧹 Cleaning up test data...")
    
    if not DATABASE_AVAILABLE:
        logger.warning("❌ Database not available, skipping cleanup")
        return
    
    try:
        db_connection = SimpleTimescaleDBConnection()
        session_factory = await db_connection.get_async_session()
        async with session_factory as session:
            # Delete test items from active learning queue
            result = await session.execute(text("""
                DELETE FROM active_learning_queue 
                WHERE model_id LIKE 'test_model_%'
            """))
            
            deleted_count = result.rowcount
            logger.info(f"✅ Cleaned up {deleted_count} test items")
            
    except Exception as e:
        logger.error(f"❌ Cleanup failed: {e}")


async def main():
    """Main test function"""
    logger.info("🚀 Starting Phase 3 - Priority 7: Active Learning Loop Tests")
    logger.info("=" * 80)
    
    # Track test results
    test_results = {}
    
    # Run tests
    tests = [
        ("Database Connection", test_database_connection),
        ("Service Initialization", test_active_learning_service_initialization),
        ("Low Confidence Capture", test_low_confidence_capture),
        ("Manual Labeling", test_manual_labeling),
        ("Statistics", test_statistics),
        ("Pending Items Retrieval", test_pending_items_retrieval),
        ("Skip Functionality", test_skip_functionality),
        ("Service Stats", test_service_stats),
    ]
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*60}")
        logger.info(f"Running: {test_name}")
        logger.info(f"{'='*60}")
        
        try:
            result = await test_func()
            test_results[test_name] = result
            status = "✅ PASSED" if result else "❌ FAILED"
            logger.info(f"{status}: {test_name}")
        except Exception as e:
            logger.error(f"❌ ERROR in {test_name}: {e}")
            test_results[test_name] = False
    
    # Summary
    logger.info(f"\n{'='*80}")
    logger.info("📋 TEST SUMMARY")
    logger.info(f"{'='*80}")
    
    passed = sum(1 for result in test_results.values() if result)
    total = len(test_results)
    
    for test_name, result in test_results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{status}: {test_name}")
    
    logger.info(f"\n📊 Overall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        logger.info("🎉 All Active Learning Loop tests passed!")
    else:
        logger.warning(f"⚠️ {total - passed} tests failed")
    
    # Cleanup
    await cleanup_test_data()
    
    return passed == total


if __name__ == "__main__":
    # Run the tests
    success = asyncio.run(main())
    exit(0 if success else 1)
