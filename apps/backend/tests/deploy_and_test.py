"""
Deployment and Test Script for AlphaPulse
Sets up database tables and runs comprehensive tests
"""

import asyncio
import logging
import os
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def setup_database():
    """Setup database tables"""
    try:
        logger.info("🚀 Setting up database tables...")
        
        # Import and run database setup
        from setup_database_tables import setup_database_tables
        await setup_database_tables()
        
        logger.info("✅ Database setup completed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Database setup failed: {e}")
        return False

async def run_tests():
    """Run comprehensive tests"""
    try:
        logger.info("🧪 Running comprehensive tests...")
        
        # Import and run tests
        from test_intelligent_signal_generator import main as run_tests
        await run_tests()
        
        logger.info("✅ Tests completed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Tests failed: {e}")
        return False

async def check_system_health():
    """Check overall system health"""
    try:
        logger.info("🏥 Checking system health...")
        
        # Import components to check if they can be loaded
        try:
            from src.app.signals.intelligent_signal_generator import IntelligentSignalGenerator
            logger.info("✅ IntelligentSignalGenerator can be imported")
        except Exception as e:
            logger.error(f"❌ IntelligentSignalGenerator import failed: {e}")
            return False
        
        try:
            from src.app.data_collection.enhanced_data_collection_manager import EnhancedDataCollectionManager
            logger.info("✅ EnhancedDataCollectionManager can be imported")
        except Exception as e:
            logger.error(f"❌ EnhancedDataCollectionManager import failed: {e}")
            return False
        
        try:
            from src.app.analysis.intelligent_analysis_engine import IntelligentAnalysisEngine
            logger.info("✅ IntelligentAnalysisEngine can be imported")
        except Exception as e:
            logger.error(f"❌ IntelligentAnalysisEngine import failed: {e}")
            return False
        
        logger.info("✅ All components can be imported successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ System health check failed: {e}")
        return False

async def main():
    """Main deployment function"""
    logger.info("🚀 Starting AlphaPulse deployment and testing...")
    
    # Step 1: Setup database
    logger.info("\n" + "="*50)
    logger.info("STEP 1: DATABASE SETUP")
    logger.info("="*50)
    
    db_success = await setup_database()
    if not db_success:
        logger.error("❌ Database setup failed. Cannot proceed.")
        sys.exit(1)
    
    # Step 2: Check system health
    logger.info("\n" + "="*50)
    logger.info("STEP 2: SYSTEM HEALTH CHECK")
    logger.info("="*50)
    
    health_success = await check_system_health()
    if not health_success:
        logger.error("❌ System health check failed. Cannot proceed.")
        sys.exit(1)
    
    # Step 3: Run comprehensive tests
    logger.info("\n" + "="*50)
    logger.info("STEP 3: COMPREHENSIVE TESTING")
    logger.info("="*50)
    
    test_success = await run_tests()
    if not test_success:
        logger.warning("⚠️ Some tests failed. Please check the logs.")
    
    # Final summary
    logger.info("\n" + "="*50)
    logger.info("🎉 DEPLOYMENT SUMMARY")
    logger.info("="*50)
    logger.info(f"Database Setup: {'✅ SUCCESS' if db_success else '❌ FAILED'}")
    logger.info(f"System Health: {'✅ SUCCESS' if health_success else '❌ FAILED'}")
    logger.info(f"Comprehensive Tests: {'✅ SUCCESS' if test_success else '⚠️ PARTIAL'}")
    
    if all([db_success, health_success, test_success]):
        logger.info("\n🎉 AlphaPulse deployment completed successfully!")
        logger.info("🚀 The system is ready for production use.")
    else:
        logger.warning("\n⚠️ Deployment completed with some issues.")
        logger.info("Please review the logs above and address any failures.")
    
    logger.info("="*50)

if __name__ == "__main__":
    asyncio.run(main())
