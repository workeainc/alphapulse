#!/usr/bin/env python3
"""
Simple Data Collection Test
Test the enhanced market intelligence data collection
"""

import asyncio
import logging
import sys
import os
import asyncpg
import ccxt

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Database configuration
DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'database': 'alphapulse',
    'user': 'alpha_emon',
    'password': 'Emon_@17711'
}

async def test_data_collection():
    """Test data collection"""
    try:
        logger.info("🔄 Testing enhanced market intelligence data collection...")
        
        # Create database connection pool
        db_pool = await asyncpg.create_pool(**DB_CONFIG)
        logger.info("✅ Database connection pool created")
        
        # Initialize exchange
        exchange = ccxt.binance({
            'apiKey': '',
            'secret': '',
            'sandbox': True,
            'enableRateLimit': True
        })
        logger.info("✅ Exchange initialized")
        
        # Import the collector using importlib
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "collector_module", 
            os.path.join(os.path.dirname(__file__), 'data', 'enhanced_market_intelligence_collector.py')
        )
        collector_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(collector_module)
        
        # Initialize collector
        collector = collector_module.EnhancedMarketIntelligenceCollector(db_pool, exchange)
        await collector.initialize()
        logger.info("✅ Enhanced Market Intelligence Collector initialized")
        
        # Collect comprehensive market intelligence
        result = await collector.collect_comprehensive_market_intelligence()
        
        if result and 'market_intelligence' in result:
            market_intelligence = result['market_intelligence']
            logger.info(f"✅ Market intelligence collected successfully")
            logger.info(f"   - BTC Dominance: {market_intelligence.btc_dominance:.2f}%")
            logger.info(f"   - Total2 Value: ${market_intelligence.total2_value:,.0f}")
            logger.info(f"   - Total3 Value: ${market_intelligence.total3_value:,.0f}")
            logger.info(f"   - Market Regime: {market_intelligence.market_regime}")
            logger.info(f"   - Fear & Greed Index: {market_intelligence.fear_greed_index}")
            logger.info(f"   - Composite Market Strength: {market_intelligence.composite_market_strength:.3f}")
        else:
            logger.error("❌ Market intelligence collection failed")
            return False
        
        if result and 'inflow_outflow' in result:
            inflow_outflow = result['inflow_outflow']
            logger.info(f"✅ Inflow/outflow analysis collected for {len(inflow_outflow)} symbols")
            
            for data in inflow_outflow[:3]:  # Show first 3 symbols
                logger.info(f"   - {data.symbol}: {data.flow_direction} ({data.flow_strength}) - Confidence: {data.flow_confidence:.3f}")
        else:
            logger.error("❌ Inflow/outflow analysis failed")
            return False
        
        # Check data storage
        async with db_pool.acquire() as conn:
            # Check enhanced market intelligence data
            count = await conn.fetchval("""
                SELECT COUNT(*) FROM enhanced_market_intelligence 
                WHERE timestamp > NOW() - INTERVAL '1 hour'
            """)
            logger.info(f"✅ Enhanced market intelligence records in last hour: {count}")
            
            # Check inflow/outflow data
            count = await conn.fetchval("""
                SELECT COUNT(*) FROM inflow_outflow_analysis 
                WHERE timestamp > NOW() - INTERVAL '1 hour'
            """)
            logger.info(f"✅ Inflow/outflow records in last hour: {count}")
        
        # Cleanup
        await collector.close()
        await db_pool.close()
        logger.info("✅ Cleanup completed")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Data collection test error: {e}")
        return False

async def main():
    """Main function"""
    try:
        success = await test_data_collection()
        
        if success:
            logger.info("🎉 Data collection test PASSED!")
            return 0
        else:
            logger.error("💥 Data collection test FAILED!")
            return 1
            
    except Exception as e:
        logger.error(f"💥 Test crashed: {e}")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
