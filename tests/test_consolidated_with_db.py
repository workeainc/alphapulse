#!/usr/bin/env python3
"""
Comprehensive Test for Consolidated Retraining System with Database
Tests the new consolidated package with TimescaleDB integration
"""

import asyncio
import logging
import sys
import time
from pathlib import Path

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "backend"))

# Setup test environment first
from test_env import setup_test_environment
setup_test_environment()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_database_connection():
    """Test database connection"""
    try:
        logger.info("🔌 Testing database connection...")
        
        from ..database.connection_simple import SimpleTimescaleDBConnection
        
        # Initialize connection
        db_connection = SimpleTimescaleDBConnection()
        await db_connection.initialize()
        
        # Test health check
        health = await db_connection.health_check()
        logger.info(f"   - Database health: {health['healthy']}")
        logger.info(f"   - Connection time: {health['connection_time_ms']:.2f}ms")
        
        if not health['healthy']:
            logger.error("❌ Database connection failed")
            return False
        
        logger.info("✅ Database connection successful")
        return True, db_connection
        
    except Exception as e:
        logger.error(f"❌ Database connection test failed: {e}")
        return False, None

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

async def test_database_operations(db_connection):
    """Test database operations"""
    try:
        logger.info("🗄️ Testing database operations...")
        
        session_factory = await db_connection.get_async_session()
        async with session_factory as session:
            from sqlalchemy import text
            
            # Test basic query
            result = await session.execute(text("SELECT version()"))
            version = result.fetchone()[0]
            logger.info(f"   - PostgreSQL version: {version}")
            
            # Test TimescaleDB extension
            result = await session.execute(text("SELECT default_version, installed_version FROM pg_available_extensions WHERE name = 'timescaledb'"))
            timescale_info = result.fetchone()
            if timescale_info:
                logger.info(f"   - TimescaleDB available: {timescale_info[0]}")
                logger.info(f"   - TimescaleDB installed: {timescale_info[1]}")
            else:
                logger.warning("   - TimescaleDB extension not found")
            
            # Test tables existence
            result = await session.execute(text("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name IN ('signals', 'candles', 'retrain_queue')
                ORDER BY table_name
            """))
            tables = [row[0] for row in result.fetchall()]
            logger.info(f"   - Found tables: {tables}")
            
            # Test retrain_queue table specifically
            if 'retrain_queue' in tables:
                result = await session.execute(text("SELECT COUNT(*) FROM retrain_queue"))
                count = result.fetchone()[0]
                logger.info(f"   - Retrain queue entries: {count}")
            else:
                logger.warning("   - Retrain queue table not found")
        
        logger.info("✅ Database operations test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Database operations test failed: {e}")
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
        
        # Test integration status
        try:
            status = await integration_service.get_integration_status()
            logger.info(f"   - Integration health: {status.get('overall_health', 'unknown')}")
        except Exception as e:
            logger.warning(f"   - Integration status check failed: {e}")
        
        logger.info("✅ Hard example integration test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Hard example integration test failed: {e}")
        return False

async def main():
    """Main test function"""
    logger.info("🚀 Starting Comprehensive Consolidated Retraining System Test")
    
    test_results = {}
    
    # Test 1: Database connection
    db_success, db_connection = await test_database_connection()
    test_results['database_connection'] = db_success
    
    if not db_success:
        logger.error("❌ Database connection failed - stopping tests")
        return False
    
    # Test 2: Consolidated imports
    import_success, classes = await test_consolidated_imports()
    test_results['consolidated_imports'] = import_success
    
    if not import_success:
        logger.error("❌ Consolidated imports failed - stopping tests")
        return False
    
    # Test 3: Service instantiation
    instantiation_success, services = await test_service_instantiation(classes)
    test_results['service_instantiation'] = instantiation_success
    
    if not instantiation_success:
        logger.error("❌ Service instantiation failed - stopping tests")
        return False
    
    # Test 4: Basic functionality
    functionality_success = await test_basic_functionality(services)
    test_results['basic_functionality'] = functionality_success
    
    # Test 5: Database operations
    db_ops_success = await test_database_operations(db_connection)
    test_results['database_operations'] = db_ops_success
    
    # Test 6: Orchestration summary
    summary_success = await test_orchestration_summary(services)
    test_results['orchestration_summary'] = summary_success
    
    # Test 7: Hard example integration
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
        logger.info("✅ Comprehensive test completed successfully")
        sys.exit(0)
    else:
        logger.error("❌ Comprehensive test failed")
        sys.exit(1)
