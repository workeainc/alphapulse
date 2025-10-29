#!/usr/bin/env python3
"""
Enhanced Signal Generator Integration Test
Tests the integration of additional ML models and analysis types into the IntelligentSignalGenerator
"""

import asyncio
import logging
import sys
import os
from datetime import datetime
from typing import Dict, Any, List

# Add the backend directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import asyncpg
import ccxt

from src.app.signals.intelligent_signal_generator import IntelligentSignalGenerator, IntelligentSignal

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EnhancedSignalGeneratorTester:
    """Test the enhanced signal generator with additional ML models"""
    
    def __init__(self):
        self.db_pool = None
        self.exchange = None
        self.signal_generator = None
        self.test_results = {}
        
    async def initialize(self) -> bool:
        """Initialize the test environment"""
        try:
            logger.info("🔄 Initializing Enhanced Signal Generator Test...")
            
            # Initialize database connection
            self.db_pool = await asyncpg.create_pool(
                host='localhost',
                port=5432,
                database='alphapulse',
                user='alpha_emon',
                password='Emon_@17711',
                min_size=5,
                max_size=20
            )
            
            # Initialize exchange with safe configuration
            from safe_exchange_config import get_safe_exchange_for_testing
            self.exchange = get_safe_exchange_for_testing()
            
            # Initialize signal generator
            self.signal_generator = IntelligentSignalGenerator(self.db_pool, self.exchange)
            await self.signal_generator.initialize()
            
            logger.info("✅ Test environment initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize test environment: {e}")
            return False
    
    async def test_ensemble_models_integration(self) -> bool:
        """Test integration of new ensemble models"""
        try:
            logger.info("🧪 Testing ensemble models integration...")
            
            # Test symbol and timeframe
            symbol = 'BTC/USDT'
            timeframe = '1h'
            
            # Generate a test signal
            signal = await self.signal_generator.generate_signal(symbol, timeframe)
            
            if not signal:
                logger.error("❌ No signal generated")
                return False
            
            # Check if ensemble votes include new models
            if not signal.ensemble_votes:
                logger.error("❌ No ensemble votes found")
                return False
            
            # Verify all expected ensemble models are present
            expected_models = [
                'technical_ml', 'price_action_ml', 'sentiment_score', 'market_regime',
                'catboost_models', 'drift_detection', 'chart_pattern_ml', 
                'candlestick_ml', 'volume_ml'
            ]
            
            missing_models = []
            for model in expected_models:
                if model not in signal.ensemble_votes:
                    missing_models.append(model)
            
            if missing_models:
                logger.error(f"❌ Missing ensemble models: {missing_models}")
                return False
            
            # Check ensemble vote structure
            for model, vote_data in signal.ensemble_votes.items():
                if not isinstance(vote_data, dict):
                    logger.error(f"❌ Invalid vote structure for {model}")
                    return False
                
                required_fields = ['vote_confidence', 'vote_direction', 'model_weight']
                for field in required_fields:
                    if field not in vote_data:
                        logger.error(f"❌ Missing field {field} in {model} vote")
                        return False
            
            logger.info("✅ Ensemble models integration test passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Ensemble models integration test failed: {e}")
            return False
    
    async def test_health_score_enhancement(self) -> bool:
        """Test enhanced health score calculation"""
        try:
            logger.info("🧪 Testing enhanced health score calculation...")
            
            # Generate a test signal
            signal = await self.signal_generator.generate_signal('ETH/USDT', '1h')
            
            if not signal:
                logger.error("❌ No signal generated for health score test")
                return False
            
            # Check health score
            if signal.health_score < 0.0 or signal.health_score > 1.0:
                logger.error(f"❌ Invalid health score: {signal.health_score}")
                return False
            
            # Check confidence breakdown
            if not signal.confidence_breakdown:
                logger.error("❌ No confidence breakdown found")
                return False
            
            # Verify confidence breakdown structure
            expected_components = [
                'pattern_analysis', 'technical_analysis', 'sentiment_analysis',
                'volume_analysis', 'market_regime_analysis', 'risk_reward_ratio'
            ]
            
            for component in expected_components:
                if component not in signal.confidence_breakdown:
                    logger.warning(f"⚠️ Missing confidence component: {component}")
            
            logger.info(f"✅ Health score: {signal.health_score:.3f}")
            logger.info("✅ Enhanced health score calculation test passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Health score enhancement test failed: {e}")
            return False
    
    async def test_ml_models_availability(self) -> bool:
        """Test ML models availability and fallback mechanisms"""
        try:
            logger.info("🧪 Testing ML models availability...")
            
            # Test CatBoost prediction
            catboost_score = await self.signal_generator._get_catboost_prediction('BTC/USDT', '1h')
            if not isinstance(catboost_score, float) or catboost_score < 0.0 or catboost_score > 1.0:
                logger.error(f"❌ Invalid CatBoost score: {catboost_score}")
                return False
            
            # Test drift detection
            drift_score = await self.signal_generator._get_drift_detection_score('BTC/USDT', '1h')
            if not isinstance(drift_score, float) or drift_score < 0.0 or drift_score > 1.0:
                logger.error(f"❌ Invalid drift detection score: {drift_score}")
                return False
            
            # Test chart pattern recognition
            pattern_score = await self.signal_generator._get_chart_pattern_score('BTC/USDT', '1h')
            if not isinstance(pattern_score, float) or pattern_score < 0.0 or pattern_score > 1.0:
                logger.error(f"❌ Invalid chart pattern score: {pattern_score}")
                return False
            
            # Test candlestick pattern analysis
            candlestick_score = await self.signal_generator._get_candlestick_pattern_score('BTC/USDT', '1h')
            if not isinstance(candlestick_score, float) or candlestick_score < 0.0 or candlestick_score > 1.0:
                logger.error(f"❌ Invalid candlestick pattern score: {candlestick_score}")
                return False
            
            # Test volume analysis
            volume_score = await self.signal_generator._get_volume_analysis_score('BTC/USDT', '1h')
            if not isinstance(volume_score, float) or volume_score < 0.0 or volume_score > 1.0:
                logger.error(f"❌ Invalid volume analysis score: {volume_score}")
                return False
            
            logger.info(f"✅ CatBoost Score: {catboost_score:.3f}")
            logger.info(f"✅ Drift Detection Score: {drift_score:.3f}")
            logger.info(f"✅ Chart Pattern Score: {pattern_score:.3f}")
            logger.info(f"✅ Candlestick Pattern Score: {candlestick_score:.3f}")
            logger.info(f"✅ Volume Analysis Score: {volume_score:.3f}")
            logger.info("✅ ML models availability test passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ ML models availability test failed: {e}")
            return False
    
    async def test_health_components(self) -> bool:
        """Test health component calculations"""
        try:
            logger.info("🧪 Testing health component calculations...")
            
            # Test ML model health
            ml_health = await self.signal_generator._get_ml_model_health_score('BTC/USDT', '1h')
            if not isinstance(ml_health, float) or ml_health < 0.0 or ml_health > 1.0:
                logger.error(f"❌ Invalid ML model health score: {ml_health}")
                return False
            
            # Test pattern health
            pattern_health = await self.signal_generator._get_pattern_health_score('BTC/USDT', '1h')
            if not isinstance(pattern_health, float) or pattern_health < 0.0 or pattern_health > 1.0:
                logger.error(f"❌ Invalid pattern health score: {pattern_health}")
                return False
            
            # Test volume health
            volume_health = await self.signal_generator._get_volume_health_score('BTC/USDT', '1h')
            if not isinstance(volume_health, float) or volume_health < 0.0 or volume_health > 1.0:
                logger.error(f"❌ Invalid volume health score: {volume_health}")
                return False
            
            logger.info(f"✅ ML Model Health: {ml_health:.3f}")
            logger.info(f"✅ Pattern Health: {pattern_health:.3f}")
            logger.info(f"✅ Volume Health: {volume_health:.3f}")
            logger.info("✅ Health component calculations test passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Health component calculations test failed: {e}")
            return False
    
    async def test_data_retrieval_methods(self) -> bool:
        """Test data retrieval methods for ML models"""
        try:
            logger.info("🧪 Testing data retrieval methods...")
            
            # Test market data retrieval
            market_data = await self.signal_generator._get_market_data_for_prediction('BTC/USDT', '1h')
            if market_data is not None:
                logger.info(f"✅ Market data retrieved: {len(market_data.get('data', []))} records")
            
            # Test drift detection data
            drift_data = await self.signal_generator._get_recent_data_for_drift_detection('BTC/USDT', '1h')
            if drift_data is not None:
                logger.info(f"✅ Drift detection data retrieved: {len(drift_data.get('data', []))} records")
            
            # Test candlestick data
            candlestick_data = await self.signal_generator._get_candlestick_data('BTC/USDT', '1h', 20)
            if candlestick_data is not None:
                logger.info(f"✅ Candlestick data retrieved: {len(candlestick_data)} records")
            
            # Test volume data
            volume_data = await self.signal_generator._get_volume_data('BTC/USDT', '1h')
            if volume_data is not None:
                logger.info(f"✅ Volume data retrieved: {len(volume_data.get('data', []))} records")
            
            logger.info("✅ Data retrieval methods test passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Data retrieval methods test failed: {e}")
            return False
    
    async def test_signal_generation_with_enhancements(self) -> bool:
        """Test complete signal generation with all enhancements"""
        try:
            logger.info("🧪 Testing complete signal generation with enhancements...")
            
            # Test multiple symbols
            test_symbols = ['BTC/USDT', 'ETH/USDT', 'ADA/USDT']
            test_timeframes = ['15m', '1h', '4h']
            
            signals_generated = 0
            high_confidence_signals = 0
            
            for symbol in test_symbols:
                for timeframe in test_timeframes:
                    try:
                        signal = await self.signal_generator.generate_signal(symbol, timeframe)
                        
                        if signal:
                            signals_generated += 1
                            
                            # Check if signal meets confidence threshold
                            if signal.confidence_score >= 0.85:
                                high_confidence_signals += 1
                                logger.info(f"✅ High confidence signal: {symbol} {timeframe} - {signal.confidence_score:.1%}")
                            
                            # Verify all enhancements are present
                            if not hasattr(signal, 'health_score'):
                                logger.error(f"❌ Missing health_score in signal for {symbol}")
                                return False
                            
                            if not hasattr(signal, 'ensemble_votes'):
                                logger.error(f"❌ Missing ensemble_votes in signal for {symbol}")
                                return False
                            
                            if not hasattr(signal, 'confidence_breakdown'):
                                logger.error(f"❌ Missing confidence_breakdown in signal for {symbol}")
                                return False
                            
                    except Exception as e:
                        logger.warning(f"⚠️ Error generating signal for {symbol} {timeframe}: {e}")
                        continue
            
            logger.info(f"✅ Generated {signals_generated} signals")
            logger.info(f"✅ High confidence signals: {high_confidence_signals}")
            logger.info("✅ Complete signal generation test passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Complete signal generation test failed: {e}")
            return False
    
    async def run_all_tests(self):
        """Run all enhancement tests"""
        logger.info("🚀 Starting Enhanced Signal Generator Integration Tests...")
        
        if not await self.initialize():
            logger.error("❌ Failed to initialize test system")
            return
        
        tests = [
            ('Ensemble Models Integration', self.test_ensemble_models_integration),
            ('Health Score Enhancement', self.test_health_score_enhancement),
            ('ML Models Availability', self.test_ml_models_availability),
            ('Health Components', self.test_health_components),
            ('Data Retrieval Methods', self.test_data_retrieval_methods),
            ('Complete Signal Generation', self.test_signal_generation_with_enhancements)
        ]
        
        for test_name, test_func in tests:
            logger.info(f"\n{'='*60}")
            logger.info(f"Running: {test_name}")
            logger.info(f"{'='*60}")
            try:
                result = await test_func()
                self.test_results[test_name.lower().replace(' ', '_')] = result
            except Exception as e:
                logger.error(f"❌ {test_name} test failed with exception: {e}")
                self.test_results[test_name.lower().replace(' ', '_')] = False
        
        self.print_results()
        await self.cleanup()
    
    def print_results(self):
        """Print test results summary"""
        logger.info(f"\n{'='*60}")
        logger.info("ENHANCED SIGNAL GENERATOR INTEGRATION TEST RESULTS")
        logger.info(f"{'='*60}")
        
        passed = 0
        total = len(self.test_results)
        
        for test_name, result in self.test_results.items():
            status = "✅ PASSED" if result else "❌ FAILED"
            logger.info(f"{test_name.replace('_', ' ').title()}: {status}")
            if result:
                passed += 1
        
        logger.info(f"\nOverall Results: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
        
        if passed == total:
            logger.info("🎉 ALL TESTS PASSED! Enhanced Signal Generator integration successful!")
        else:
            logger.warning(f"⚠️ {total - passed} tests failed. Please review the implementation.")
    
    async def cleanup(self):
        """Clean up test resources"""
        try:
            if self.signal_generator:
                await self.signal_generator.close()
            
            if self.db_pool:
                await self.db_pool.close()
            
            logger.info("✅ Test cleanup completed")
            
        except Exception as e:
            logger.error(f"❌ Error during cleanup: {e}")

async def main():
    """Main test execution"""
    tester = EnhancedSignalGeneratorTester()
    await tester.run_all_tests()

if __name__ == "__main__":
    asyncio.run(main())
