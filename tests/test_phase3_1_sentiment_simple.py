#!/usr/bin/env python3
"""
Phase 3.1: Sentiment Analysis Integration - Simplified Test Script
Tests enhanced sentiment service and signal generator integration without database
"""

import asyncio
import logging
import sys
import os
from datetime import datetime
from typing import Dict, Any

# Add the backend directory to the path
sys.path.append(os.path.dirname(__file__))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Phase3_1SimpleTester:
    """Simplified test for Phase 3.1: Sentiment Analysis Integration"""
    
    def __init__(self):
        self.sentiment_service = None
        self.signal_generator = None
        self.test_results = {}
    
    async def run_all_tests(self):
        """Run all simplified Phase 3.1 tests"""
        logger.info("🚀 Starting Phase 3.1: Sentiment Analysis Integration - Simplified Tests")
        
        try:
            # Test 1: Enhanced Sentiment Service
            await self.test_enhanced_sentiment_service()
            
            # Test 2: Signal Generator Integration
            await self.test_signal_generator_integration()
            
            # Test 3: Sentiment Analysis Methods
            await self.test_sentiment_analysis_methods()
            
            # Print results
            self.print_test_results()
            
        except Exception as e:
            logger.error(f"❌ Phase 3.1 simplified tests failed: {e}")
            return False
        
        return True
    
    async def test_enhanced_sentiment_service(self):
        """Test enhanced sentiment service functionality"""
        logger.info("📊 Testing Enhanced Sentiment Service...")
        
        try:
            # Import sentiment service
            from app.services.sentiment_service import SentimentService
            
            # Initialize service
            self.sentiment_service = SentimentService()
            
            # Test basic sentiment analysis
            basic_sentiment = await self.sentiment_service.get_sentiment("BTC/USDT")
            logger.info(f"✅ Basic sentiment analysis: {basic_sentiment.get('sentiment_score', 0):.3f}")
            
            # Test enhanced sentiment analysis
            enhanced_sentiment = await self.sentiment_service.get_enhanced_sentiment("BTC/USDT")
            
            # Verify enhanced features
            required_features = [
                'trend_analysis', 'volatility_metrics', 'momentum_indicators',
                'correlation_metrics', 'enhanced_confidence', 'sentiment_strength',
                'prediction_confidence', 'phase_3_1_features'
            ]
            
            missing_features = [feature for feature in required_features 
                              if feature not in enhanced_sentiment]
            
            if missing_features:
                raise Exception(f"Missing enhanced features: {missing_features}")
            
            # Verify trend analysis
            trend_analysis = enhanced_sentiment.get('trend_analysis', {})
            if 'trend' not in trend_analysis or 'trend_strength' not in trend_analysis:
                raise Exception("Invalid trend analysis structure")
            
            # Verify momentum indicators
            momentum = enhanced_sentiment.get('momentum_indicators', {})
            if 'momentum_direction' not in momentum or 'momentum_strength' not in momentum:
                raise Exception("Invalid momentum indicators structure")
            
            # Verify volatility metrics
            volatility = enhanced_sentiment.get('volatility_metrics', {})
            if 'volatility_rank' not in volatility or 'stability_score' not in volatility:
                raise Exception("Invalid volatility metrics structure")
            
            # Verify correlation metrics
            correlation = enhanced_sentiment.get('correlation_metrics', {})
            if 'correlation_strength' not in correlation or 'predictive_power' not in correlation:
                raise Exception("Invalid correlation metrics structure")
            
            logger.info(f"✅ Enhanced sentiment analysis: trend={trend_analysis.get('trend')}, "
                       f"momentum={momentum.get('momentum_direction')}, "
                       f"volatility={volatility.get('volatility_rank')}")
            
            # Test performance metrics
            performance = self.sentiment_service.get_performance_metrics()
            logger.info(f"✅ Performance metrics: {performance.get('total_analyses', 0)} analyses")
            
            self.test_results['enhanced_sentiment_service'] = '✅ PASSED'
            logger.info("✅ Enhanced sentiment service test passed")
            
        except Exception as e:
            logger.error(f"❌ Enhanced sentiment service test failed: {e}")
            self.test_results['enhanced_sentiment_service'] = f'❌ FAILED: {e}'
    
    async def test_signal_generator_integration(self):
        """Test signal generator integration with enhanced sentiment"""
        logger.info("📊 Testing Signal Generator Integration...")
        
        try:
            # Import signal generator
            from app.strategies.real_time_signal_generator import RealTimeSignalGenerator
            
            # Initialize signal generator
            config = {
                'use_sentiment': True,
                'min_confidence': 0.6,
                'min_strength': 0.6
            }
            
            self.signal_generator = RealTimeSignalGenerator(config)
            
            # Initialize sentiment service manually
            await self.signal_generator.initialize_sentiment_service()
            
            # Test sentiment analysis method
            sentiment_signals = await self.signal_generator.analyze_enhanced_sentiment_predictions("BTC/USDT")
            
            # Verify sentiment signals structure
            required_fields = [
                'sentiment_bias', 'sentiment_score', 'sentiment_confidence',
                'sentiment_strength', 'high_confidence_sentiment', 'trend_analysis',
                'momentum_indicators', 'volatility_metrics', 'correlation_metrics',
                'enhanced_confidence', 'prediction_confidence', 'phase_3_1_features'
            ]
            
            missing_fields = [field for field in required_fields 
                            if field not in sentiment_signals]
            
            if missing_fields:
                raise Exception(f"Missing sentiment signal fields: {missing_fields}")
            
            # Verify sentiment bias values
            valid_biases = ['bullish', 'bearish', 'neutral']
            if sentiment_signals.get('sentiment_bias') not in valid_biases:
                raise Exception(f"Invalid sentiment bias: {sentiment_signals.get('sentiment_bias')}")
            
            # Verify phase 3.1 features flag
            if not sentiment_signals.get('phase_3_1_features'):
                raise Exception("Phase 3.1 features not enabled")
            
            logger.info(f"✅ Sentiment signals: bias={sentiment_signals.get('sentiment_bias')}, "
                       f"score={sentiment_signals.get('sentiment_score'):.3f}, "
                       f"confidence={sentiment_signals.get('enhanced_confidence'):.3f}")
            
            self.test_results['signal_generator_integration'] = '✅ PASSED'
            logger.info("✅ Signal generator integration test passed")
            
        except Exception as e:
            logger.error(f"❌ Signal generator integration test failed: {e}")
            self.test_results['signal_generator_integration'] = f'❌ FAILED: {e}'
    
    async def test_sentiment_analysis_methods(self):
        """Test individual sentiment analysis methods"""
        logger.info("📊 Testing Sentiment Analysis Methods...")
        
        try:
            if not self.sentiment_service:
                raise Exception("Sentiment service not initialized")
            
            # Test trend calculation
            trend_analysis = self.sentiment_service._calculate_sentiment_trend("BTC/USDT")
            if not isinstance(trend_analysis, dict):
                raise Exception("Trend analysis should return a dictionary")
            
            if 'trend' not in trend_analysis or 'trend_strength' not in trend_analysis:
                raise Exception("Invalid trend analysis structure")
            
            logger.info(f"✅ Trend analysis: {trend_analysis.get('trend')} (strength: {trend_analysis.get('trend_strength'):.3f})")
            
            # Test volatility calculation
            volatility_metrics = self.sentiment_service._calculate_sentiment_volatility("BTC/USDT")
            if not isinstance(volatility_metrics, dict):
                raise Exception("Volatility metrics should return a dictionary")
            
            if 'volatility_rank' not in volatility_metrics or 'stability_score' not in volatility_metrics:
                raise Exception("Invalid volatility metrics structure")
            
            logger.info(f"✅ Volatility metrics: {volatility_metrics.get('volatility_rank')} (stability: {volatility_metrics.get('stability_score'):.3f})")
            
            # Test momentum calculation
            momentum_indicators = self.sentiment_service._calculate_sentiment_momentum("BTC/USDT")
            if not isinstance(momentum_indicators, dict):
                raise Exception("Momentum indicators should return a dictionary")
            
            if 'momentum_direction' not in momentum_indicators or 'momentum_strength' not in momentum_indicators:
                raise Exception("Invalid momentum indicators structure")
            
            logger.info(f"✅ Momentum indicators: {momentum_indicators.get('momentum_direction')} (strength: {momentum_indicators.get('momentum_strength')})")
            
            # Test correlation calculation
            correlation_metrics = self.sentiment_service._calculate_sentiment_correlation("BTC/USDT")
            if not isinstance(correlation_metrics, dict):
                raise Exception("Correlation metrics should return a dictionary")
            
            if 'correlation_strength' not in correlation_metrics or 'predictive_power' not in correlation_metrics:
                raise Exception("Invalid correlation metrics structure")
            
            logger.info(f"✅ Correlation metrics: {correlation_metrics.get('correlation_strength')} (predictive power: {correlation_metrics.get('predictive_power'):.3f})")
            
            # Test enhanced confidence calculation
            base_sentiment = {'confidence': 0.6}
            enhanced_confidence = self.sentiment_service._calculate_enhanced_confidence(
                base_sentiment, trend_analysis, volatility_metrics
            )
            
            if not isinstance(enhanced_confidence, float):
                raise Exception("Enhanced confidence should return a float")
            
            if enhanced_confidence < 0 or enhanced_confidence > 1:
                raise Exception("Enhanced confidence should be between 0 and 1")
            
            logger.info(f"✅ Enhanced confidence calculation: {enhanced_confidence:.3f}")
            
            # Test sentiment strength calculation
            sentiment_strength = self.sentiment_service._calculate_sentiment_strength(
                base_sentiment, momentum_indicators
            )
            
            if not isinstance(sentiment_strength, float):
                raise Exception("Sentiment strength should return a float")
            
            if sentiment_strength < 0 or sentiment_strength > 1:
                raise Exception("Sentiment strength should be between 0 and 1")
            
            logger.info(f"✅ Sentiment strength calculation: {sentiment_strength:.3f}")
            
            # Test prediction confidence calculation
            prediction_confidence = self.sentiment_service._calculate_prediction_confidence(
                base_sentiment, correlation_metrics
            )
            
            if not isinstance(prediction_confidence, float):
                raise Exception("Prediction confidence should return a float")
            
            if prediction_confidence < 0 or prediction_confidence > 1:
                raise Exception("Prediction confidence should be between 0 and 1")
            
            logger.info(f"✅ Prediction confidence calculation: {prediction_confidence:.3f}")
            
            self.test_results['sentiment_analysis_methods'] = '✅ PASSED'
            logger.info("✅ Sentiment analysis methods test passed")
            
        except Exception as e:
            logger.error(f"❌ Sentiment analysis methods test failed: {e}")
            self.test_results['sentiment_analysis_methods'] = f'❌ FAILED: {e}'
    
    def print_test_results(self):
        """Print test results summary"""
        logger.info("\n" + "="*60)
        logger.info("📋 PHASE 3.1: SENTIMENT ANALYSIS INTEGRATION - SIMPLIFIED TEST RESULTS")
        logger.info("="*60)
        
        passed = 0
        total = len(self.test_results)
        
        for test_name, result in self.test_results.items():
            status = "✅ PASSED" if result.startswith("✅") else "❌ FAILED"
            logger.info(f"{test_name:.<40} {status}")
            if result.startswith("✅"):
                passed += 1
        
        logger.info("-"*60)
        logger.info(f"TOTAL: {passed}/{total} tests passed")
        
        if passed == total:
            logger.info("🎉 ALL PHASE 3.1 SIMPLIFIED TESTS PASSED! Sentiment Analysis Integration successful!")
        else:
            logger.info("⚠️ Some tests failed. Please review the errors above.")
        
        logger.info("="*60)

async def main():
    """Main test function"""
    tester = Phase3_1SimpleTester()
    success = await tester.run_all_tests()
    
    if success:
        logger.info("🚀 Phase 3.1: Sentiment Analysis Integration - Simplified tests completed successfully!")
        return 0
    else:
        logger.error("❌ Phase 3.1: Sentiment Analysis Integration - Simplified tests failed!")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
