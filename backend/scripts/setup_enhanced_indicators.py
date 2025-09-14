#!/usr/bin/env python3
"""
Enhanced Indicators Setup Script
Quick setup for AlphaPlus enhanced technical indicators system
"""

import asyncio
import logging
import sys
import os
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

from database.connection import TimescaleDBConnection
from services.enhanced_indicators_integration import EnhancedIndicatorsIntegration
import redis.asyncio as redis

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def setup_enhanced_indicators():
    """Setup enhanced indicators system"""
    
    logger.info("🚀 Setting up Enhanced Technical Indicators System...")
    
    try:
        # Initialize database connection
        db_config = {
            'host': 'localhost',
            'port': 5432,
            'database': 'alphapulse',
            'username': 'alpha_emon',
            'password': 'Emon_@17711',
            'pool_size': 5,
            'max_overflow': 10
        }
        
        db_connection = TimescaleDBConnection(db_config)
        await db_connection.initialize()
        
        # 1. Test database connection
        logger.info("📊 Testing database connection...")
        async with db_connection.async_session() as db_session:
            # Test basic query
            from sqlalchemy import text
            result = await db_session.execute(text("SELECT 1"))
            await db_session.commit()
            logger.info("✅ Database connection successful")
        
        # 2. Test Redis connection
        logger.info("🔴 Testing Redis connection...")
        try:
            redis_client = redis.Redis(host='localhost', port=6379, db=0)
            await redis_client.ping()
            logger.info("✅ Redis connection successful")
        except Exception as e:
            logger.warning(f"⚠️ Redis connection failed: {e}")
            logger.info("📝 Continuing without Redis (caching will be disabled)")
            redis_client = None
        
        # 3. Initialize enhanced indicators integration
        logger.info("⚡ Initializing enhanced indicators integration...")
        async with db_connection.async_session() as db_session:
            indicators_integration = EnhancedIndicatorsIntegration(
                db_session=db_session,
                redis_client=redis_client,
                enable_enhanced=True
            )
            logger.info("✅ Enhanced indicators integration initialized")
        
        # 4. Test with sample data
        logger.info("🧪 Testing with sample data...")
        import pandas as pd
        import numpy as np
        
        # Create sample market data
        dates = pd.date_range(start='2024-01-01', periods=100, freq='1H')
        sample_data = pd.DataFrame({
            'timestamp': dates,
            'open': np.random.uniform(100, 110, 100),
            'high': np.random.uniform(105, 115, 100),
            'low': np.random.uniform(95, 105, 100),
            'close': np.random.uniform(100, 110, 100),
            'volume': np.random.uniform(1000, 10000, 100)
        })
        
        # Calculate indicators
        async with db_connection.async_session() as db_session:
            indicators_integration = EnhancedIndicatorsIntegration(
                db_session=db_session,
                redis_client=redis_client,
                enable_enhanced=True
            )
            
            result = await indicators_integration.calculate_indicators(
                df=sample_data,
                symbol="BTC/USDT",
                timeframe="1h"
            )
            
            logger.info("✅ Sample indicator calculation successful")
            logger.info(f"📈 Sample RSI: {result.rsi:.2f}")
            logger.info(f"📈 Sample MACD: {result.macd:.6f}")
            logger.info(f"📈 Sample VWAP: {result.vwap:.2f}")
        
        # 5. Get performance statistics
        stats = indicators_integration.get_performance_stats()
        logger.info("📊 Performance Statistics:")
        logger.info(f"   Enhanced Usage Rate: {stats['enhanced_usage_rate']:.2%}")
        logger.info(f"   Cache Hit Rate: {stats['cache_hit_rate']:.2%}")
        logger.info(f"   Avg Enhanced Time: {stats['avg_enhanced_time_ms']:.2f}ms")
        
        # 6. Test historical data retrieval
        logger.info("📚 Testing historical data retrieval...")
        async with db_connection.async_session() as db_session:
            indicators_integration = EnhancedIndicatorsIntegration(
                db_session=db_session,
                redis_client=redis_client,
                enable_enhanced=True
            )
            
            historical_data = await indicators_integration.get_indicators_from_timescaledb(
                symbol="BTC/USDT",
                timeframe="1h",
                hours_back=24,
                use_aggregates=True
            )
            
            logger.info(f"✅ Historical data retrieval successful (rows: {len(historical_data)})")
        
        logger.info("🎉 Enhanced Indicators System setup completed successfully!")
        logger.info("📖 See docs/ENHANCED_INDICATORS_IMPLEMENTATION_GUIDE.md for usage instructions")
        
    except Exception as e:
        logger.error(f"❌ Setup failed: {e}")
        logger.error("🔧 Please check the implementation guide for troubleshooting steps")
        return False
    
    return True

async def check_dependencies():
    """Check if all required dependencies are installed"""
    
    logger.info("🔍 Checking dependencies...")
    
    required_packages = [
        'polars',
        'pandas',
        'numpy',
        'redis',
        'sqlalchemy',
        'asyncpg'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            logger.info(f"✅ {package}")
        except ImportError:
            logger.error(f"❌ {package} - MISSING")
            missing_packages.append(package)
    
    if missing_packages:
        logger.error(f"❌ Missing packages: {', '.join(missing_packages)}")
        logger.info("📦 Install missing packages with:")
        logger.info("pip install -r backend/requirements_enhanced_indicators.txt")
        return False
    
    logger.info("✅ All dependencies are installed")
    return True

async def check_database_tables():
    """Check if required database tables exist"""
    
    logger.info("🗄️ Checking database tables...")
    
    try:
        # Initialize database connection
        db_config = {
            'host': 'localhost',
            'port': 5432,
            'database': 'alphapulse',
            'username': 'alpha_emon',
            'password': 'Emon_@17711',
            'pool_size': 5,
            'max_overflow': 10
        }
        
        db_connection = TimescaleDBConnection(db_config)
        await db_connection.initialize()
        
        async with db_connection.async_session() as db_session:
            from sqlalchemy import text
            
            # Check if enhanced_market_data table exists
            result = await db_session.execute(text("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'enhanced_market_data'
                );
            """))
            table_exists = result.scalar()
            
            if table_exists:
                logger.info("✅ enhanced_market_data table exists")
            else:
                logger.error("❌ enhanced_market_data table missing")
                logger.info("📝 Run database migrations: alembic upgrade head")
                return False
            
            # Check if continuous aggregates exist
            result = await db_session.execute(text("""
                SELECT COUNT(*) FROM timescaledb_information.continuous_aggregates 
                WHERE view_name LIKE '%enhanced_market_data%';
            """))
            aggregate_count = result.scalar()
            
            if aggregate_count > 0:
                logger.info(f"✅ {aggregate_count} continuous aggregates found")
            else:
                logger.warning("⚠️ No continuous aggregates found")
                logger.info("📝 Run migration 009_create_enhanced_indicators_aggregates")
            
            await db_session.commit()
            
    except Exception as e:
        logger.error(f"❌ Database check failed: {e}")
        return False
    
    return True

async def main():
    """Main setup function"""
    
    logger.info("🎯 AlphaPlus Enhanced Indicators Setup")
    logger.info("=" * 50)
    
    # Check dependencies
    if not await check_dependencies():
        return False
    
    # Check database tables
    if not await check_database_tables():
        return False
    
    # Setup enhanced indicators
    if not await setup_enhanced_indicators():
        return False
    
    logger.info("=" * 50)
    logger.info("🎉 Setup completed successfully!")
    logger.info("🚀 Your enhanced indicators system is ready to use!")
    
    return True

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
