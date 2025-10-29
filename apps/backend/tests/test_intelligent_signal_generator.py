"""
Test script for Intelligent Signal Generator
Verifies all components are working correctly
"""

import asyncio
import asyncpg
import logging
import os
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_database_connection():
    """Test database connection and table existence"""
    try:
        # Connect to database
        conn = await asyncpg.connect(
            host=os.getenv('DB_HOST', 'postgres'),
            port=int(os.getenv('DB_PORT', 5432)),
            database=os.getenv('DB_NAME', 'alphapulse'),
            user=os.getenv('DB_USER', 'alpha_emon'),
            password=os.getenv('DB_PASSWORD', 'Emon_@17711')
        )
        
        logger.info("✅ Database connection successful")
        
        # Check if required tables exist
        required_tables = [
            'market_intelligence',
            'volume_analysis', 
            'candles',
            'price_action_ml_predictions',
            'market_regime_data'
        ]
        
        for table in required_tables:
            result = await conn.fetchval(f"""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = $1
                );
            """, table)
            
            if result:
                logger.info(f"✅ Table {table} exists")
            else:
                logger.warning(f"⚠️ Table {table} does not exist")
        
        # Check if tables have data
        for table in required_tables:
            count = await conn.fetchval(f"SELECT COUNT(*) FROM {table}")
            logger.info(f"📊 Table {table} has {count} rows")
        
        await conn.close()
        return True
        
    except Exception as e:
        logger.error(f"❌ Database connection failed: {e}")
        return False

async def test_intelligent_signal_generator():
    """Test the intelligent signal generator"""
    try:
        # Import components
        from src.app.signals.intelligent_signal_generator import IntelligentSignalGenerator
        import ccxt
        
        # Connect to database
        conn = await asyncpg.connect(
            host=os.getenv('DB_HOST', 'postgres'),
            port=int(os.getenv('DB_PORT', 5432)),
            database=os.getenv('DB_NAME', 'alphapulse'),
            user=os.getenv('DB_USER', 'alpha_emon'),
            password=os.getenv('DB_PASSWORD', 'Emon_@17711')
        )
        
        # Initialize exchange
        exchange = ccxt.binance({
            'sandbox': False,
            'enableRateLimit': True,
        })
        
        # Create signal generator
        signal_generator = IntelligentSignalGenerator(conn, exchange)
        logger.info("✅ IntelligentSignalGenerator initialized")
        
        # Test signal generation
        symbol = "BTC/USDT"
        timeframe = "1h"
        
        logger.info(f"🔄 Generating signal for {symbol} on {timeframe} timeframe...")
        signal = await signal_generator.generate_intelligent_signal(symbol, timeframe)
        
        if signal:
            logger.info(f"✅ Signal generated successfully!")
            logger.info(f"   Symbol: {signal.symbol}")
            logger.info(f"   Direction: {signal.signal_direction}")
            logger.info(f"   Confidence: {signal.confidence_score:.3f}")
            logger.info(f"   Signal Type: {signal.signal_type}")
            logger.info(f"   Entry Price: {signal.entry_price}")
            logger.info(f"   Stop Loss: {signal.stop_loss}")
            logger.info(f"   Take Profit 1: {signal.take_profit_1}")
        else:
            logger.warning("⚠️ No signal generated (this might be normal if conditions aren't met)")
        
        await conn.close()
        return True
        
    except Exception as e:
        logger.error(f"❌ Intelligent signal generator test failed: {e}")
        return False

async def test_data_collection_components():
    """Test data collection components"""
    try:
        # Import components
        from src.app.data_collection.enhanced_data_collection_manager import EnhancedDataCollectionManager
        from src.app.data_collection.market_intelligence_collector import MarketIntelligenceCollector
        from src.app.data_collection.volume_positioning_analyzer import VolumePositioningAnalyzer
        import ccxt
        
        # Connect to database
        conn = await asyncpg.connect(
            host=os.getenv('DB_HOST', 'postgres'),
            port=int(os.getenv('DB_PORT', 5432)),
            database=os.getenv('DB_NAME', 'alphapulse'),
            user=os.getenv('DB_USER', 'alpha_emon'),
            password=os.getenv('DB_PASSWORD', 'Emon_@17711')
        )
        
        # Initialize exchange
        exchange = ccxt.binance({
            'sandbox': False,
            'enableRateLimit': True,
        })
        
        # Test market intelligence collector
        market_collector = MarketIntelligenceCollector(conn, exchange)
        market_data = await market_collector.collect_market_intelligence()
        
        if market_data:
            logger.info("✅ Market intelligence collection successful")
            logger.info(f"   BTC Dominance: {market_data.btc_dominance}")
            logger.info(f"   Market Regime: {market_data.market_regime}")
            logger.info(f"   Fear/Greed Index: {market_data.fear_greed_index}")
        else:
            logger.warning("⚠️ Market intelligence collection returned None")
        
        # Test volume positioning analyzer
        volume_analyzer = VolumePositioningAnalyzer(conn, exchange)
        volume_data = await volume_analyzer.analyze_volume_positioning("BTC/USDT", "1h")
        
        if volume_data:
            logger.info("✅ Volume positioning analysis successful")
            logger.info(f"   Volume Ratio: {volume_data.volume_ratio}")
            logger.info(f"   Volume Trend: {volume_data.volume_trend}")
            logger.info(f"   Positioning Score: {volume_data.volume_positioning_score}")
        else:
            logger.warning("⚠️ Volume positioning analysis returned None")
        
        await conn.close()
        return True
        
    except Exception as e:
        logger.error(f"❌ Data collection components test failed: {e}")
        return False

async def test_analysis_engine():
    """Test the intelligent analysis engine"""
    try:
        # Import components
        from src.app.analysis.intelligent_analysis_engine import IntelligentAnalysisEngine
        import ccxt
        
        # Connect to database
        conn = await asyncpg.connect(
            host=os.getenv('DB_HOST', 'postgres'),
            port=int(os.getenv('DB_PORT', 5432)),
            database=os.getenv('DB_NAME', 'alphapulse'),
            user=os.getenv('DB_USER', 'alpha_emon'),
            password=os.getenv('DB_PASSWORD', 'Emon_@17711')
        )
        
        # Initialize exchange
        exchange = ccxt.binance({
            'sandbox': False,
            'enableRateLimit': True,
        })
        
        # Create analysis engine
        analysis_engine = IntelligentAnalysisEngine(conn, exchange)
        await analysis_engine.initialize()
        logger.info("✅ IntelligentAnalysisEngine initialized")
        
        # Test technical analysis
        symbol = "BTC/USDT"
        timeframe = "1h"
        
        technical_result = await analysis_engine.get_technical_analysis(symbol, timeframe)
        if technical_result:
            logger.info("✅ Technical analysis successful")
            logger.info(f"   RSI: {technical_result.get('rsi', 'N/A')}")
            logger.info(f"   MACD Signal: {technical_result.get('macd_signal', 'N/A')}")
            logger.info(f"   Confidence: {technical_result.get('confidence', 'N/A')}")
        else:
            logger.warning("⚠️ Technical analysis returned None")
        
        # Test sentiment analysis
        sentiment_result = await analysis_engine.get_sentiment_analysis(symbol)
        if sentiment_result:
            logger.info("✅ Sentiment analysis successful")
            logger.info(f"   Sentiment Score: {sentiment_result.get('sentiment_score', 'N/A')}")
            logger.info(f"   News Impact: {sentiment_result.get('news_impact', 'N/A')}")
        else:
            logger.warning("⚠️ Sentiment analysis returned None")
        
        await conn.close()
        return True
        
    except Exception as e:
        logger.error(f"❌ Analysis engine test failed: {e}")
        return False

async def main():
    """Run all tests"""
    logger.info("🚀 Starting comprehensive AlphaPulse system tests...")
    
    # Test database connection
    logger.info("\n📊 Testing database connection...")
    db_success = await test_database_connection()
    
    if not db_success:
        logger.error("❌ Database connection failed. Cannot proceed with other tests.")
        return
    
    # Test data collection components
    logger.info("\n📈 Testing data collection components...")
    data_collection_success = await test_data_collection_components()
    
    # Test analysis engine
    logger.info("\n🔍 Testing analysis engine...")
    analysis_success = await test_analysis_engine()
    
    # Test intelligent signal generator
    logger.info("\n🎯 Testing intelligent signal generator...")
    signal_generator_success = await test_intelligent_signal_generator()
    
    # Summary
    logger.info("\n" + "="*50)
    logger.info("📋 TEST SUMMARY")
    logger.info("="*50)
    logger.info(f"Database Connection: {'✅ PASS' if db_success else '❌ FAIL'}")
    logger.info(f"Data Collection: {'✅ PASS' if data_collection_success else '❌ FAIL'}")
    logger.info(f"Analysis Engine: {'✅ PASS' if analysis_success else '❌ FAIL'}")
    logger.info(f"Signal Generator: {'✅ PASS' if signal_generator_success else '❌ FAIL'}")
    
    if all([db_success, data_collection_success, analysis_success, signal_generator_success]):
        logger.info("\n🎉 All tests passed! AlphaPulse system is ready.")
    else:
        logger.warning("\n⚠️ Some tests failed. Please check the logs above.")
    
    logger.info("="*50)

if __name__ == "__main__":
    asyncio.run(main())
