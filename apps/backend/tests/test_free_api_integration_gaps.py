#!/usr/bin/env python3
"""
Free API Integration Test
Tests the Free API services and identifies gaps
"""

import asyncio
import sys
import os
from pathlib import Path
import logging
from datetime import datetime
import json

# Add backend to path
backend_path = Path(__file__).parent
sys.path.insert(0, str(backend_path))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_free_api_manager():
    """Test Free API Manager functionality"""
    logger.info("🔍 Testing Free API Manager...")
    
    try:
        from src.services.free_api_manager import FreeAPIManager
        manager = FreeAPIManager()
        logger.info("✅ Free API Manager imported successfully")
        
        # Test market data collection
        logger.info("📈 Testing market data collection...")
        market_data = await manager.get_market_data('BTC')
        logger.info(f"Market data result: {market_data.get('success', False) if market_data else False}")
        
        # Test sentiment analysis
        logger.info("😊 Testing sentiment analysis...")
        sentiment = await manager.get_sentiment_analysis('BTC')
        logger.info(f"Sentiment analysis result: {sentiment.get('success', False) if sentiment else False}")
        
        # Test news data
        logger.info("📰 Testing news data collection...")
        news = await manager.get_news_data('BTC')
        logger.info(f"News data result: {news.get('success', False) if news else False}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error testing Free API Manager: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_database_service():
    """Test Free API Database Service"""
    logger.info("🗄️ Testing Free API Database Service...")
    
    try:
        from src.services.free_api_database_service import FreeAPIDatabaseService
        import asyncpg
        
        # Create database connection
        conn = await asyncpg.connect('postgresql://alpha_emon:Emon_%4017711@postgres:5432/alphapulse')
        
        # Initialize database service
        db_service = FreeAPIDatabaseService(conn)
        logger.info("✅ Free API Database Service initialized")
        
        # Test table access
        tables = ['free_api_market_data', 'free_api_sentiment_data', 'free_api_news_data', 
                 'free_api_social_data', 'free_api_liquidation_events', 'free_api_data_quality', 
                 'free_api_rate_limits']
        
        for table in tables:
            count = await conn.fetchval(f'SELECT COUNT(*) FROM {table}')
            logger.info(f"✅ Table {table}: {count} records")
        
        await conn.close()
        return True
        
    except Exception as e:
        logger.error(f"❌ Error testing Database Service: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_sde_integration():
    """Test SDE Integration Service"""
    logger.info("🔗 Testing SDE Integration Service...")
    
    try:
        from src.services.free_api_sde_integration_service import FreeAPISDEIntegrationService
        from src.services.free_api_database_service import FreeAPIDatabaseService
        from src.services.free_api_manager import FreeAPIManager
        import asyncpg
        
        # Create connections
        conn = await asyncpg.connect('postgresql://alpha_emon:Emon_%4017711@postgres:5432/alphapulse')
        
        # Initialize services
        db_service = FreeAPIDatabaseService(conn)
        manager = FreeAPIManager()
        sde_service = FreeAPISDEIntegrationService(db_service, manager)
        
        logger.info("✅ SDE Integration Service initialized")
        
        # Test data processing
        logger.info("🔄 Testing data processing...")
        result = await sde_service.process_market_data('BTC')
        logger.info(f"Data processing result: {result}")
        
        await conn.close()
        return True
        
    except Exception as e:
        logger.error(f"❌ Error testing SDE Integration: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_data_pipeline():
    """Test Free API Data Pipeline"""
    logger.info("🚰 Testing Free API Data Pipeline...")
    
    try:
        from src.services.free_api_data_pipeline import FreeAPIDataPipeline
        from src.services.free_api_database_service import FreeAPIDatabaseService
        from src.services.free_api_manager import FreeAPIManager
        import asyncpg
        
        # Create connections
        conn = await asyncpg.connect('postgresql://alpha_emon:Emon_%4017711@postgres:5432/alphapulse')
        
        # Initialize services
        db_service = FreeAPIDatabaseService(conn)
        manager = FreeAPIManager()
        pipeline = FreeAPIDataPipeline(db_service, manager)
        
        logger.info("✅ Free API Data Pipeline initialized")
        
        # Test pipeline execution
        logger.info("🔄 Testing pipeline execution...")
        result = await pipeline.run_pipeline(['BTC'])
        logger.info(f"Pipeline execution result: {result}")
        
        await conn.close()
        return True
        
    except Exception as e:
        logger.error(f"❌ Error testing Data Pipeline: {e}")
        import traceback
        traceback.print_exc()
        return False

async def identify_gaps():
    """Identify gaps in the implementation"""
    logger.info("🔍 Identifying implementation gaps...")
    
    gaps = []
    
    try:
        # Check if all required services exist
        required_services = [
            'services.free_api_manager',
            'services.free_api_database_service', 
            'services.free_api_sde_integration_service',
            'services.free_api_data_pipeline'
        ]
        
        for service in required_services:
            try:
                __import__(service)
                logger.info(f"✅ {service} exists")
            except ImportError as e:
                gaps.append(f"Missing service: {service} - {e}")
                logger.warning(f"❌ {service} missing: {e}")
        
        # Check database tables
        import asyncpg
        conn = await asyncpg.connect('postgresql://alpha_emon:Emon_%4017711@postgres:5432/alphapulse')
        
        # Check for TimescaleDB hypertables
        hypertables = await conn.fetch("SELECT hypertable_name FROM timescaledb_information.hypertables WHERE hypertable_name LIKE 'free_api_%'")
        if len(hypertables) < 7:
            gaps.append(f"Missing hypertables: Expected 7, found {len(hypertables)}")
        
        # Check for indexes
        indexes = await conn.fetch("SELECT indexname FROM pg_indexes WHERE tablename LIKE 'free_api_%'")
        if len(indexes) < 10:  # Expected minimum number of indexes
            gaps.append(f"Missing indexes: Expected at least 10, found {len(indexes)}")
        
        await conn.close()
        
    except Exception as e:
        gaps.append(f"Error checking gaps: {e}")
        logger.error(f"❌ Error identifying gaps: {e}")
    
    return gaps

async def main():
    """Main test function"""
    logger.info("🚀 Starting Free API Integration Tests")
    
    results = {}
    
    # Test 1: Free API Manager
    results['free_api_manager'] = await test_free_api_manager()
    
    # Test 2: Database Service
    results['database_service'] = await test_database_service()
    
    # Test 3: SDE Integration
    results['sde_integration'] = await test_sde_integration()
    
    # Test 4: Data Pipeline
    results['data_pipeline'] = await test_data_pipeline()
    
    # Test 5: Identify gaps
    gaps = await identify_gaps()
    results['gaps'] = gaps
    
    # Summary
    logger.info("📊 Test Results Summary:")
    for test_name, result in results.items():
        if test_name == 'gaps':
            logger.info(f"🔍 Gaps identified: {len(result)}")
            for gap in result:
                logger.warning(f"  - {gap}")
        else:
            status = "✅ PASS" if result else "❌ FAIL"
            logger.info(f"  {test_name}: {status}")
    
    # Overall result
    passed_tests = sum(1 for k, v in results.items() if k != 'gaps' and v)
    total_tests = len([k for k in results.keys() if k != 'gaps'])
    
    logger.info(f"🎯 Overall: {passed_tests}/{total_tests} tests passed")
    
    if gaps:
        logger.warning(f"⚠️ {len(gaps)} gaps identified that need attention")
    
    return results

if __name__ == "__main__":
    asyncio.run(main())
