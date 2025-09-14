#!/usr/bin/env python3
"""
Simple Test for Consolidated Retraining System
Tests the new consolidated package without database dependencies
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

async def test_consolidated_imports():
    """Test importing the consolidated retraining system"""
    try:
        logger.info("📦 Testing consolidated imports...")
        
        # Test imports
        from ..ai.retraining import (
            RetrainingOrchestrator,
            RetrainingDataService,
            AutoRetrainTriggerService,
            DriftDetectionMonitor
        )
        logger.info("✅ All consolidated imports successful")
        
        return True, {
            'RetrainingOrchestrator': RetrainingOrchestrator,
            'RetrainingDataService': RetrainingDataService,
            'AutoRetrainTriggerService': AutoRetrainTriggerService,
            'DriftDetectionMonitor': DriftDetectionMonitor
        }
        
    except Exception as e:
        logger.error(f"❌ Consolidated imports failed: {e}")
        return False, None

async def test_service_instantiation(classes):
    """Test service instantiation"""
    try:
        logger.info("🔧 Testing service instantiation...")
        
        services = {}
        
        # Test each service
        for name, cls in classes.items():
            try:
                service = cls()
                services[name] = service
                logger.info(f"   - {name}: ✅ Instantiated")
            except Exception as e:
                logger.error(f"   - {name}: ❌ Failed - {e}")
                return False, None
        
        logger.info("✅ All services instantiated successfully")
        return True, services
        
    except Exception as e:
        logger.error(f"❌ Service instantiation failed: {e}")
        return False, None

async def test_basic_functionality(services):
    """Test basic functionality of services"""
    try:
        logger.info("⚡ Testing basic functionality...")
        
        # Test orchestrator
        orchestrator = services['RetrainingOrchestrator']
        status = orchestrator.get_current_status()
        logger.info(f"   - Orchestrator status: {status}")
        
        # Test data service
        data_service = services['RetrainingDataService']
        cache_stats = await data_service.get_cache_stats()
        logger.info(f"   - Data service cache: {cache_stats['cache_size']} entries")
        
        # Test trigger service
        trigger_service = services['AutoRetrainTriggerService']
        trigger_status = trigger_service.get_current_status()
        logger.info(f"   - Trigger service status: {trigger_status}")
        
        # Test drift monitor
        drift_monitor = services['DriftDetectionMonitor']
        drift_status = drift_monitor.get_current_status()
        logger.info(f"   - Drift monitor status: {drift_status}")
        
        logger.info("✅ Basic functionality tests passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Basic functionality test failed: {e}")
        return False

async def test_orchestration_summary(services):
    """Test orchestration summary"""
    try:
        logger.info("📊 Testing orchestration summary...")
        
        orchestrator = services['RetrainingOrchestrator']
        summary = orchestrator.get_orchestration_summary()
        
        logger.info(f"   - Orchestrator status: {summary['status']}")
        logger.info(f"   - Execution stats: {len(summary['execution_stats'])} metrics")
        logger.info(f"   - Subsystems: {len(summary['subsystems'])} systems")
        
        # Log some key metrics
        for metric, value in summary['execution_stats'].items():
            logger.info(f"   - {metric}: {value}")
        
        logger.info("✅ Orchestration summary test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Orchestration summary test failed: {e}")
        return False

async def test_hard_example_integration():
    """Test hard example integration with consolidated system"""
    try:
        logger.info("🔗 Testing hard example integration...")
        
        from app.services.hard_example_integration_service import HardExampleIntegrationService
        
        # Create integration service
        integration_service = HardExampleIntegrationService()
        logger.info("✅ Hard example integration service created")
        
        # Check if AI services are available
        if hasattr(integration_service, 'model_orchestrator') and integration_service.model_orchestrator:
            logger.info("✅ AI services integrated successfully")
        else:
            logger.warning("⚠️ AI services not available (expected in test environment)")
        
        logger.info("✅ Hard example integration test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Hard example integration test failed: {e}")
        return False

async def main():
    """Main test function"""
    logger.info("🚀 Starting Simple Consolidated Retraining System Test")
    
    test_results = {}
    
    # Test 1: Consolidated imports
    import_success, classes = await test_consolidated_imports()
    test_results['consolidated_imports'] = import_success
    
    if not import_success:
        logger.error("❌ Consolidated imports failed - stopping tests")
        return False
    
    # Test 2: Service instantiation
    instantiation_success, services = await test_service_instantiation(classes)
    test_results['service_instantiation'] = instantiation_success
    
    if not instantiation_success:
        logger.error("❌ Service instantiation failed - stopping tests")
        return False
    
    # Test 3: Basic functionality
    functionality_success = await test_basic_functionality(services)
    test_results['basic_functionality'] = functionality_success
    
    # Test 4: Orchestration summary
    summary_success = await test_orchestration_summary(services)
    test_results['orchestration_summary'] = summary_success
    
    # Test 5: Hard example integration
    integration_success = await test_hard_example_integration()
    test_results['hard_example_integration'] = integration_success
    
    # Summary
    logger.info("📋 Test Results Summary:")
    for test_name, success in test_results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        logger.info(f"   - {test_name}: {status}")
    
    total_tests = len(test_results)
    passed_tests = sum(test_results.values())
    
    logger.info(f"📊 Overall Results: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        logger.info("🎉 All tests passed! Consolidated system is working correctly.")
        return True
    else:
        logger.error(f"❌ {total_tests - passed_tests} tests failed")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    
    if success:
        logger.info("✅ Simple test completed successfully")
        sys.exit(0)
    else:
        logger.error("❌ Simple test failed")
        sys.exit(1)
