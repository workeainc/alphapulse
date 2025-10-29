#!/usr/bin/env python3
"""
Minimal Test for Consolidated Retraining System
Tests core functionality without problematic imports
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "backend"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_core_consolidated_functionality():
    """Test core consolidated functionality without problematic imports"""
    try:
        logger.info("🧪 Testing core consolidated functionality...")
        
        # Test 1: Direct import of consolidated classes
        logger.info("📦 Test 1: Direct imports...")
        try:
            # Import directly from the files to avoid problematic dependencies
            from ..src.ai.retraining.orchestrator import RetrainingOrchestrator
            from ..src.ai.retraining.data_service import RetrainingDataService
            from ..src.ai.retraining.trigger_service import AutoRetrainTriggerService
            from ..src.ai.retraining.drift_monitor import DriftDetectionMonitor
            logger.info("✅ Direct imports successful")
        except Exception as e:
            logger.error(f"❌ Direct imports failed: {e}")
            return False
        
        # Test 2: Service instantiation
        logger.info("🔧 Test 2: Service instantiation...")
        try:
            orchestrator = RetrainingOrchestrator()
            data_service = RetrainingDataService()
            trigger_service = AutoRetrainTriggerService()
            drift_monitor = DriftDetectionMonitor()
            logger.info("✅ All services instantiated")
        except Exception as e:
            logger.error(f"❌ Service instantiation failed: {e}")
            return False
        
        # Test 3: Basic status methods
        logger.info("⚡ Test 3: Basic status methods...")
        try:
            # Test orchestrator status
            status = orchestrator.get_current_status()
            logger.info(f"   - Orchestrator status: {status}")
            
            # Test trigger service status
            trigger_status = trigger_service.get_current_status()
            logger.info(f"   - Trigger service status: {trigger_status}")
            
            # Test drift monitor status
            drift_status = drift_monitor.get_current_status()
            logger.info(f"   - Drift monitor status: {drift_status}")
            
            logger.info("✅ Basic status methods working")
        except Exception as e:
            logger.error(f"❌ Basic status methods failed: {e}")
            return False
        
        # Test 4: Orchestration summary
        logger.info("📊 Test 4: Orchestration summary...")
        try:
            summary = orchestrator.get_orchestration_summary()
            logger.info(f"   - Orchestrator summary: {summary['status']}")
            logger.info(f"   - Execution stats: {len(summary['execution_stats'])} metrics")
            logger.info(f"   - Subsystems: {len(summary['subsystems'])} systems")
            logger.info("✅ Orchestration summary working")
        except Exception as e:
            logger.error(f"❌ Orchestration summary failed: {e}")
            return False
        
        # Test 5: Data service cache stats
        logger.info("🗄️ Test 5: Data service cache...")
        try:
            cache_stats = await data_service.get_cache_stats()
            logger.info(f"   - Cache size: {cache_stats['cache_size']} entries")
            logger.info("✅ Data service cache working")
        except Exception as e:
            logger.error(f"❌ Data service cache failed: {e}")
            return False
        
        logger.info("🎉 All core functionality tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Core functionality test failed: {e}")
        return False

async def test_import_structure():
    """Test the import structure of the consolidated package"""
    try:
        logger.info("📦 Testing import structure...")
        
        # Test package import
        import ai.retraining
        logger.info("✅ Package import successful")
        
        # Test __all__ exports
        expected_exports = [
            'RetrainingOrchestrator',
            'RetrainingDataService', 
            'AutoRetrainTriggerService',
            'DriftDetectionMonitor'
        ]
        
        for export in expected_exports:
            if hasattr(ai.retraining, export):
                logger.info(f"   - {export}: ✅ Available")
            else:
                logger.error(f"   - {export}: ❌ Missing")
                return False
        
        logger.info("✅ Import structure test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Import structure test failed: {e}")
        return False

async def main():
    """Main test function"""
    logger.info("🚀 Starting Minimal Consolidated Retraining System Test")
    
    test_results = {}
    
    # Test 1: Import structure
    import_success = await test_import_structure()
    test_results['import_structure'] = import_success
    
    if not import_success:
        logger.error("❌ Import structure failed - stopping tests")
        return False
    
    # Test 2: Core functionality
    core_success = await test_core_consolidated_functionality()
    test_results['core_functionality'] = core_success
    
    # Summary
    logger.info("📋 Test Results Summary:")
    for test_name, success in test_results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        logger.info(f"   - {test_name}: {status}")
    
    total_tests = len(test_results)
    passed_tests = sum(test_results.values())
    
    logger.info(f"📊 Overall Results: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        logger.info("🎉 All tests passed! Consolidated system core functionality is working.")
        return True
    else:
        logger.error(f"❌ {total_tests - passed_tests} tests failed")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    
    if success:
        logger.info("✅ Minimal test completed successfully")
        sys.exit(0)
    else:
        logger.error("❌ Minimal test failed")
        sys.exit(1)
