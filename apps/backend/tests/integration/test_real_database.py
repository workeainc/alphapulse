#!/usr/bin/env python3
"""
Test Real PostgreSQL Connection and Run Live Signal Generation
Tests the complete pipeline with real TimescaleDB
"""

import asyncio
import logging
import sys
import os

# Add backend to path
sys.path.append('backend')

from src.data.realtime_data_pipeline import RealTimeDataPipeline
from src.ai.sde_database_integration import SDEDatabaseIntegration, SignalGenerationRequest
from src.strategies.dynamic_support_resistance_analyzer import DynamicSupportResistanceAnalyzer
from src.data.volume_analyzer import VolumeAnalyzer
from src.core.websocket_binance import BinanceWebSocketClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_real_database_connection():
    """Test connection to real PostgreSQL database"""
    logger.info("🔌 Testing real PostgreSQL connection...")
    
    # Use the user's PostgreSQL credentials
    db_url = "postgresql://alpha_emon:Emon_%4017711@localhost:5432/alphapulse"
    
    try:
        import asyncpg
        conn = await asyncpg.connect(db_url)
        
        # Test basic query
        result = await conn.fetchval("SELECT version()")
        logger.info(f"✅ PostgreSQL connection successful!")
        logger.info(f"📊 PostgreSQL version: {result}")
        
        # Test TimescaleDB
        timescale_version = await conn.fetchval("SELECT extversion FROM pg_extension WHERE extname = 'timescaledb'")
        if timescale_version:
            logger.info(f"📊 TimescaleDB version: {timescale_version}")
        
        # Test hypertables
        hypertables = await conn.fetch("""
            SELECT hypertable_name, num_dimensions, num_chunks 
            FROM timescaledb_information.hypertables 
            WHERE hypertable_schema = 'public'
        """)
        
        logger.info("📊 Created hypertables:")
        for table in hypertables:
            logger.info(f"  - {table['hypertable_name']}: {table['num_dimensions']} dimensions, {table['num_chunks']} chunks")
        
        # Test tables
        tables = await conn.fetch("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_name IN ('ohlcv_data', 'order_book_data', 'technical_indicators', 'support_resistance_levels')
        """)
        
        logger.info("📊 Created tables:")
        for table in tables:
            logger.info(f"  - {table['table_name']}")
        
        await conn.close()
        return True
        
    except Exception as e:
        logger.error(f"❌ Database connection failed: {e}")
        return False

async def test_real_data_pipeline():
    """Test real data pipeline with PostgreSQL"""
    logger.info("🔄 Testing real data pipeline...")
    
    db_url = "postgresql://alpha_emon:Emon_%4017711@localhost:5432/alphapulse"
    
    try:
        # Initialize data pipeline
        pipeline = RealTimeDataPipeline(db_url=db_url)
        await pipeline.initialize()
        
        # Test inserting sample data
        sample_message = {
            'type': 'kline',
            'symbol': 'BTCUSDT',
            'timeframe': '1m',
            'timestamp': '2025-01-15T20:00:00+00:00',
            'open': 45000.0,
            'high': 45100.0,
            'low': 44900.0,
            'close': 45050.0,
            'volume': 100.5,
            'quote_volume': 4525000.0,
            'trades': 150
        }
        
        # Process message
        success = await pipeline.process_websocket_message(sample_message)
        if success:
            logger.info("✅ Sample data inserted successfully!")
        
        # Calculate indicators
        await pipeline.calculate_technical_indicators('BTCUSDT', '1m')
        
        # Get pipeline stats
        stats = pipeline.get_pipeline_stats()
        logger.info(f"📊 Pipeline stats: {stats}")
        
        await pipeline.close()
        return True
        
    except Exception as e:
        logger.error(f"❌ Data pipeline test failed: {e}")
        return False

async def test_real_sde_integration():
    """Test real SDE integration with PostgreSQL"""
    logger.info("🤖 Testing real SDE integration...")
    
    db_url = "postgresql://alpha_emon:Emon_%4017711@localhost:5432/alphapulse"
    
    try:
        # Initialize SDE integration
        sde_integration = SDEDatabaseIntegration(db_url=db_url)
        await sde_integration.initialize()
        
        # Create signal generation request
        request = SignalGenerationRequest(
            symbol="BTCUSDT",
            timeframe="1m",
            market_data={
                'current_price': 45000.0,
                'indicators': {
                    'sma_20': 44800.0,
                    'sma_50': 44500.0,
                    'rsi_14': 35.2,
                    'macd': 0.85
                }
            },
            analysis_results={
                'sentiment_analysis': {
                    'overall_sentiment': 0.3,
                    'confidence': 0.8
                }
            },
            timestamp='2025-01-15T20:00:00+00:00'
        )
        
        # Generate signal
        result = await sde_integration.generate_signal(request)
        
        if result:
            logger.info("✅ SDE signal generated successfully!")
            logger.info(f"📊 Signal ID: {result.signal_id}")
            logger.info(f"📊 Direction: {result.direction}")
            logger.info(f"📊 Confidence: {result.confidence}")
            logger.info(f"📊 Strength: {result.strength}")
        
        # Get integration stats
        stats = sde_integration.get_integration_stats()
        logger.info(f"📊 SDE Integration stats: {stats}")
        
        await sde_integration.close()
        return True
        
    except Exception as e:
        logger.error(f"❌ SDE integration test failed: {e}")
        return False

async def run_live_signal_generation_test():
    """Run the complete live signal generation test with real database"""
    logger.info("🚀 Running Live Signal Generation Test with Real Database")
    logger.info("=" * 60)
    
    try:
        # Test database connection
        if not await test_real_database_connection():
            logger.error("❌ Database connection test failed")
            return False
        
        # Test data pipeline
        if not await test_real_data_pipeline():
            logger.error("❌ Data pipeline test failed")
            return False
        
        # Test SDE integration
        if not await test_real_sde_integration():
            logger.error("❌ SDE integration test failed")
            return False
        
        logger.info("🎉 All tests passed! Real database integration is working!")
        logger.info("💡 You can now run the live signal generation test with real PostgreSQL!")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Live signal generation test failed: {e}")
        return False

async def main():
    """Main function"""
    logger.info("🚀 Starting Real Database Integration Test")
    logger.info("=" * 60)
    
    success = await run_live_signal_generation_test()
    
    if success:
        logger.info("🎉 Real database integration test completed successfully!")
        logger.info("📊 Database Details:")
        logger.info("   Host: localhost")
        logger.info("   Port: 5432")
        logger.info("   Database: alphapulse")
        logger.info("   Username: alpha_emon")
        logger.info("   Password: Emon_@17711")
        logger.info("   TimescaleDB: Enabled")
    else:
        logger.error("❌ Real database integration test failed!")

if __name__ == "__main__":
    asyncio.run(main())
