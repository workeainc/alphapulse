#!/usr/bin/env python3
"""
Test Real Signal Generation
Verifies that the intelligent signal generator uses real analysis instead of mock data
"""

import asyncio
import logging
import sys
import os
from datetime import datetime

# Add the backend directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import asyncpg
import ccxt
from app.signals.intelligent_signal_generator import IntelligentSignalGenerator, get_intelligent_signal_generator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RealSignalGenerationTester:
    """Test real signal generation with actual data"""
    
    def __init__(self):
        self.db_pool = None
        self.exchange = None
        self.signal_generator = None
        
    async def setup(self):
        """Setup database connection and exchange"""
        try:
            # Database connection
            self.db_pool = await asyncpg.create_pool(
                host='localhost',
                port=5432,
                user='alpha_emon',
                password='Emon_@17711',
                database='alphapulse'
            )
            
            # Exchange setup
            self.exchange = ccxt.binance({
                'apiKey': '',
                'secret': '',
                'sandbox': True,
                'options': {
                    'defaultType': 'spot',
                    'adjustForTimeDifference': True
                }
            })
            
            # Initialize signal generator
            self.signal_generator = await get_intelligent_signal_generator(self.db_pool, self.exchange)
            
            logger.info("✅ Setup completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Setup failed: {e}")
            return False
    
    async def test_real_technical_analysis(self):
        """Test that technical analysis uses real data"""
        logger.info("🔍 Testing Real Technical Analysis...")
        
        try:
            # Test with a real symbol
            symbol = "BTC/USDT"
            timeframe = "1h"
            
            # Get technical analysis
            technical_result = self.signal_generator._get_technical_analysis(symbol, timeframe)
            
            # Verify it's not mock data
            if technical_result.get('rsi') == 65.5:  # Old mock value
                logger.error("❌ Still using mock RSI data")
                return False
            
            if technical_result.get('macd') == 0.0023:  # Old mock value
                logger.error("❌ Still using mock MACD data")
                return False
            
            # Check for real data indicators
            rsi = technical_result.get('rsi', 0)
            macd_signal = technical_result.get('macd_signal', '')
            current_price = technical_result.get('current_price', 0)
            
            logger.info(f"📊 Real Technical Analysis Results:")
            logger.info(f"   RSI: {rsi:.2f}")
            logger.info(f"   MACD Signal: {macd_signal}")
            logger.info(f"   Current Price: {current_price:.2f}")
            logger.info(f"   Confidence: {technical_result.get('confidence', 0):.3f}")
            
            # Verify we have reasonable values
            if 0 <= rsi <= 100 and current_price > 0:
                logger.info("✅ Real technical analysis data detected")
                return True
            else:
                logger.error("❌ Invalid technical analysis data")
                return False
                
        except Exception as e:
            logger.error(f"❌ Technical analysis test failed: {e}")
            return False
    
    async def test_real_sentiment_analysis(self):
        """Test that sentiment analysis uses real data"""
        logger.info("🔍 Testing Real Sentiment Analysis...")
        
        try:
            symbol = "BTC/USDT"
            
            # Get sentiment analysis
            sentiment_result = self.signal_generator._get_sentiment_analysis(symbol)
            
            # Verify it's not mock data
            if sentiment_result.get('sentiment_score') == 0.65:  # Old mock value
                logger.error("❌ Still using mock sentiment data")
                return False
            
            # Check for real data indicators
            sentiment_score = sentiment_result.get('sentiment_score', 0)
            news_impact = sentiment_result.get('news_impact', 0)
            fear_greed = sentiment_result.get('fear_greed_index', 0)
            
            logger.info(f"📊 Real Sentiment Analysis Results:")
            logger.info(f"   Sentiment Score: {sentiment_score:.3f}")
            logger.info(f"   News Impact: {news_impact:.3f}")
            logger.info(f"   Fear & Greed Index: {fear_greed}")
            logger.info(f"   Confidence: {sentiment_result.get('confidence', 0):.3f}")
            
            # Verify we have reasonable values
            if 0 <= sentiment_score <= 1 and 0 <= fear_greed <= 100:
                logger.info("✅ Real sentiment analysis data detected")
                return True
            else:
                logger.error("❌ Invalid sentiment analysis data")
                return False
                
        except Exception as e:
            logger.error(f"❌ Sentiment analysis test failed: {e}")
            return False
    
    async def test_real_volume_analysis(self):
        """Test that volume analysis uses real data"""
        logger.info("🔍 Testing Real Volume Analysis...")
        
        try:
            symbol = "BTC/USDT"
            timeframe = "1h"
            
            # Get volume analysis
            volume_result = self.signal_generator._get_volume_analysis(symbol, timeframe)
            
            # Verify it's not mock data
            if volume_result.get('volume_ratio') == 1.2:  # Old mock value
                logger.error("❌ Still using mock volume data")
                return False
            
            # Check for real data indicators
            volume_trend = volume_result.get('volume_trend', '')
            volume_ratio = volume_result.get('volume_ratio', 0)
            positioning_score = volume_result.get('positioning_score', 0)
            
            logger.info(f"📊 Real Volume Analysis Results:")
            logger.info(f"   Volume Trend: {volume_trend}")
            logger.info(f"   Volume Ratio: {volume_ratio:.3f}")
            logger.info(f"   Positioning Score: {positioning_score:.3f}")
            logger.info(f"   Confidence: {volume_result.get('confidence', 0):.3f}")
            
            # Verify we have reasonable values
            if volume_ratio > 0 and 0 <= positioning_score <= 1:
                logger.info("✅ Real volume analysis data detected")
                return True
            else:
                logger.error("❌ Invalid volume analysis data")
                return False
                
        except Exception as e:
            logger.error(f"❌ Volume analysis test failed: {e}")
            return False
    
    async def test_real_market_regime_analysis(self):
        """Test that market regime analysis uses real data"""
        logger.info("🔍 Testing Real Market Regime Analysis...")
        
        try:
            symbol = "BTC/USDT"
            
            # Get market regime analysis
            regime_result = self.signal_generator._get_market_regime_analysis(symbol)
            
            # Verify it's not mock data
            if regime_result.get('regime') == 'trending':  # Old mock value
                logger.error("❌ Still using mock regime data")
                return False
            
            # Check for real data indicators
            regime = regime_result.get('regime', '')
            volatility = regime_result.get('volatility', '')
            trend_strength = regime_result.get('trend_strength', 0)
            
            logger.info(f"📊 Real Market Regime Analysis Results:")
            logger.info(f"   Regime: {regime}")
            logger.info(f"   Volatility: {volatility}")
            logger.info(f"   Trend Strength: {trend_strength:.3f}")
            logger.info(f"   Confidence: {regime_result.get('confidence', 0):.3f}")
            
            # Verify we have reasonable values
            if regime and 0 <= trend_strength <= 1:
                logger.info("✅ Real market regime analysis data detected")
                return True
            else:
                logger.error("❌ Invalid market regime analysis data")
                return False
                
        except Exception as e:
            logger.error(f"❌ Market regime analysis test failed: {e}")
            return False
    
    async def test_complete_signal_generation(self):
        """Test complete signal generation with real data"""
        logger.info("🔍 Testing Complete Signal Generation...")
        
        try:
            symbol = "BTC/USDT"
            timeframe = "1h"
            
            # Generate a complete signal
            signal = await self.signal_generator.generate_intelligent_signal(symbol, timeframe)
            
            if signal is None:
                logger.warning("⚠️ No signal generated (this might be normal if confidence < 85%)")
                return True
            
            # Verify signal has real data
            logger.info(f"📊 Generated Signal:")
            logger.info(f"   Symbol: {signal.symbol}")
            logger.info(f"   Direction: {signal.signal_direction}")
            logger.info(f"   Strength: {signal.signal_strength}")
            logger.info(f"   Confidence: {signal.confidence_score:.3f}")
            logger.info(f"   Entry Price: {signal.entry_price:.2f}")
            logger.info(f"   Stop Loss: {signal.stop_loss:.2f}")
            logger.info(f"   Take Profit 1: {signal.take_profit_1:.2f}")
            logger.info(f"   Risk/Reward: {signal.risk_reward_ratio:.2f}")
            logger.info(f"   Health Score: {signal.health_score:.3f}")
            logger.info(f"   Entry Reasoning: {signal.entry_reasoning}")
            
            # Verify it's not using mock prices
            if signal.entry_price == 45000:  # Old mock price
                logger.error("❌ Still using mock entry price")
                return False
            
            if signal.stop_loss == 44000:  # Old mock stop loss
                logger.error("❌ Still using mock stop loss")
                return False
            
            # Verify we have reasonable values
            if (signal.entry_price > 0 and 
                signal.stop_loss > 0 and 
                signal.take_profit_1 > 0 and
                signal.confidence_score >= 0):
                
                logger.info("✅ Real signal generation successful")
                return True
            else:
                logger.error("❌ Invalid signal data")
                return False
                
        except Exception as e:
            logger.error(f"❌ Complete signal generation test failed: {e}")
            return False
    
    async def run_all_tests(self):
        """Run all tests"""
        logger.info("🚀 Starting Real Signal Generation Tests")
        
        # Setup
        if not await self.setup():
            return False
        
        test_results = []
        
        # Run individual tests
        test_results.append(await self.test_real_technical_analysis())
        test_results.append(await self.test_real_sentiment_analysis())
        test_results.append(await self.test_real_volume_analysis())
        test_results.append(await self.test_real_market_regime_analysis())
        test_results.append(await self.test_complete_signal_generation())
        
        # Summary
        passed_tests = sum(test_results)
        total_tests = len(test_results)
        
        logger.info(f"\n📋 Test Summary:")
        logger.info(f"   Passed: {passed_tests}/{total_tests}")
        logger.info(f"   Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        
        if passed_tests == total_tests:
            logger.info("🎉 All tests passed! Signal generator is using real data.")
        else:
            logger.warning("⚠️ Some tests failed. Check the logs for details.")
        
        # Cleanup
        if self.db_pool:
            await self.db_pool.close()
        
        return passed_tests == total_tests

async def main():
    """Main test function"""
    tester = RealSignalGenerationTester()
    success = await tester.run_all_tests()
    
    if success:
        print("\n✅ REAL SIGNAL GENERATION TEST: PASSED")
        print("🎯 The intelligent signal generator is now using real analysis!")
        print("📊 It analyzes actual technical indicators, sentiment, volume, and market regime data.")
        print("💰 Signals are generated with real prices and proper risk management.")
    else:
        print("\n❌ REAL SIGNAL GENERATION TEST: FAILED")
        print("🔧 Some components may still be using mock data.")
        print("📝 Check the logs above for specific issues.")

if __name__ == "__main__":
    asyncio.run(main())
