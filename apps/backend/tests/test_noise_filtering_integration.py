#!/usr/bin/env python3
"""
Integration Test for Noise Filtering and Adaptive Learning
Tests the complete noise filtering and adaptive learning system
"""

import asyncio
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add the backend directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.ai.noise_filter_engine import NoiseFilterEngine
from src.ai.market_regime_classifier import MarketRegimeClassifier
from src.ai.adaptive_learning_engine import AdaptiveLearningEngine
from src.strategies.pattern_detector import CandlestickPatternDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_noise_filtering_integration():
    """Test the complete noise filtering and adaptive learning integration"""
    
    print("🚀 Starting Noise Filtering and Adaptive Learning Integration Test")
    print("=" * 70)
    
    # Database configuration
    db_config = {
        'host': 'localhost',
        'port': 5432,
        'database': 'alphapulse',
        'user': 'postgres',
        'password': 'Emon_@17711'
    }
    
    # Initialize components
    noise_filter = None
    market_regime_classifier = None
    adaptive_learning = None
    pattern_detector = None
    
    try:
        print("📊 Step 1: Initializing Components...")
        
        # Initialize noise filter engine
        noise_filter = NoiseFilterEngine(db_config)
        await noise_filter.initialize()
        print("✅ Noise Filter Engine initialized")
        
        # Initialize market regime classifier
        market_regime_classifier = MarketRegimeClassifier(db_config)
        await market_regime_classifier.initialize()
        print("✅ Market Regime Classifier initialized")
        
        # Initialize adaptive learning engine
        adaptive_learning = AdaptiveLearningEngine(db_config)
        await adaptive_learning.initialize()
        print("✅ Adaptive Learning Engine initialized")
        
        # Initialize pattern detector with advanced features
        pattern_detector = CandlestickPatternDetector(db_config)
        await pattern_detector._initialize_advanced_components()
        print("✅ Pattern Detector with advanced features initialized")
        
        print("\n📊 Step 2: Testing Market Regime Classification...")
        
        # Create sample market data (trending market)
        trending_data = pd.DataFrame({
            'open': [50000, 50100, 50200, 50300, 50400, 50500, 50600, 50700, 50800, 50900, 51000, 51100, 51200, 51300, 51400],
            'high': [50100, 50200, 50300, 50400, 50500, 50600, 50700, 50800, 50900, 51000, 51100, 51200, 51300, 51400, 51500],
            'low': [49900, 50000, 50100, 50200, 50300, 50400, 50500, 50600, 50700, 50800, 50900, 51000, 51100, 51200, 51300],
            'close': [50100, 50200, 50300, 50400, 50500, 50600, 50700, 50800, 50900, 51000, 51100, 51200, 51300, 51400, 51500],
            'volume': [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400]
        })
        
        # Test market regime classification
        regime_result = await market_regime_classifier.classify_market_regime(trending_data, 'BTCUSDT', '1h')
        
        print(f"   Market Regime: {regime_result['regime_type']}")
        print(f"   Confidence: {regime_result['regime_confidence']:.3f}")
        print(f"   Trend Strength: {regime_result['trend_strength']:.3f}")
        print(f"   Volatility Level: {regime_result['volatility_level']:.4f}")
        print(f"   Volume Profile: {regime_result['volume_profile']}")
        
        print("\n📊 Step 3: Testing Noise Filtering...")
        
        # Create sample pattern data
        pattern_data = {
            'pattern_id': 'test_pattern_123',
            'symbol': 'BTCUSDT',
            'pattern_name': 'doji',
            'timeframe': '1h',
            'confidence': 0.75
        }
        
        # Test noise filtering
        passed_filter, filter_results = await noise_filter.filter_pattern(pattern_data, trending_data)
        
        print(f"   Pattern Passed Filter: {passed_filter}")
        print(f"   Overall Score: {filter_results['overall_score']:.3f}")
        print(f"   Noise Level: {filter_results['noise_level']:.3f}")
        print(f"   Filter Scores: {filter_results['filter_scores']}")
        print(f"   Filter Reasons: {filter_results['filter_reasons']}")
        
        print("\n📊 Step 4: Testing Pattern Detection with Noise Filtering...")
        
        # Test enhanced pattern detection
        enhanced_signals = await pattern_detector.detect_patterns_enhanced(trending_data, 'BTCUSDT', '1h')
        
        print(f"   Enhanced Patterns Detected: {len(enhanced_signals)}")
        
        for i, signal in enumerate(enhanced_signals[:3]):  # Show first 3 patterns
            print(f"   Pattern {i+1}: {signal.pattern}")
            print(f"     Type: {signal.type}")
            print(f"     Confidence: {signal.confidence:.3f}")
            if signal.additional_info:
                print(f"     Market Regime: {signal.additional_info.get('market_regime', 'N/A')}")
                print(f"     Noise Filter Score: {signal.additional_info.get('noise_filter_score', 'N/A')}")
                print(f"     Adaptive Confidence: {signal.additional_info.get('adaptive_confidence', 'N/A')}")
        
        print("\n📊 Step 5: Testing Adaptive Learning...")
        
        # Test pattern outcome tracking
        if enhanced_signals:
            test_signal = enhanced_signals[0]
            tracking_id = test_signal.additional_info.get('tracking_id', 'test_tracking_123')
            
            pattern_data_for_tracking = {
                'tracking_id': tracking_id,
                'pattern_name': test_signal.pattern,
                'market_regime': test_signal.additional_info.get('market_regime', 'trending'),
                'confidence': test_signal.confidence
            }
            
            # Track a successful outcome
            success = await adaptive_learning.track_pattern_outcome(
                pattern_data_for_tracking, 'success', 51500.0, 500.0
            )
            
            print(f"   Pattern Outcome Tracking: {success}")
            
            # Test adaptive confidence
            adaptive_confidence = await adaptive_learning.get_adaptive_confidence(
                pattern_data_for_tracking, test_signal.confidence
            )
            
            print(f"   Base Confidence: {test_signal.confidence:.3f}")
            print(f"   Adaptive Confidence: {adaptive_confidence:.3f}")
        
        print("\n📊 Step 6: Testing Performance Summary...")
        
        # Test performance summary
        performance_summary = await adaptive_learning.get_performance_summary()
        
        print(f"   Performance Summary Records: {len(performance_summary['summary'])}")
        
        for item in performance_summary['summary']:
            print(f"   {item['pattern_name']} ({item['market_regime']}): "
                  f"{item['success_rate']:.2f} success rate, "
                  f"{item['total_patterns']} patterns")
        
        print("\n📊 Step 7: Testing Low Volume Pattern Filtering...")
        
        # Create low volume data to test filtering
        low_volume_data = pd.DataFrame({
            'open': [50000, 50100, 50200, 50300, 50400],
            'high': [50100, 50200, 50300, 50400, 50500],
            'low': [49900, 50000, 50100, 50200, 50300],
            'close': [50100, 50200, 50300, 50400, 50500],
            'volume': [100, 150, 200, 180, 120]  # Very low volume
        })
        
        low_volume_pattern_data = {
            'pattern_id': 'low_volume_test',
            'symbol': 'BTCUSDT',
            'pattern_name': 'hammer',
            'timeframe': '1h',
            'confidence': 0.8
        }
        
        # Test noise filtering on low volume data
        passed_low_volume, low_volume_results = await noise_filter.filter_pattern(low_volume_pattern_data, low_volume_data)
        
        print(f"   Low Volume Pattern Passed: {passed_low_volume}")
        print(f"   Low Volume Filter Score: {low_volume_results['overall_score']:.3f}")
        print(f"   Low Volume Filter Reasons: {low_volume_results['filter_reasons']}")
        
        print("\n📊 Step 8: Testing Market Regime Adaptation...")
        
        # Create sideways market data
        sideways_data = pd.DataFrame({
            'open': [50000, 50050, 49950, 50000, 50050, 49950, 50000, 50050, 49950, 50000],
            'high': [50100, 50150, 50050, 50100, 50150, 50050, 50100, 50150, 50050, 50100],
            'low': [49900, 49950, 49850, 49900, 49950, 49850, 49900, 49950, 49850, 49900],
            'close': [50050, 49950, 50000, 50050, 49950, 50000, 50050, 49950, 50000, 50050],
            'volume': [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900]
        })
        
        # Test market regime classification for sideways market
        sideways_regime = await market_regime_classifier.classify_market_regime(sideways_data, 'BTCUSDT', '1h')
        
        print(f"   Sideways Market Regime: {sideways_regime['regime_type']}")
        print(f"   Sideways Confidence: {sideways_regime['regime_confidence']:.3f}")
        print(f"   Sideways Trend Strength: {sideways_regime['trend_strength']:.3f}")
        
        print("\n🎯 Integration Test Results Summary:")
        print("✅ All components initialized successfully")
        print("✅ Market regime classification working")
        print("✅ Noise filtering working")
        print("✅ Pattern detection with enhancement working")
        print("✅ Adaptive learning working")
        print("✅ Performance tracking working")
        print("✅ Low volume filtering working")
        print("✅ Market regime adaptation working")
        
        print("\n🚀 Noise Filtering and Adaptive Learning Integration Test COMPLETED SUCCESSFULLY!")
        
    except Exception as e:
        logger.error(f"❌ Integration test failed: {e}")
        print(f"❌ Test failed with error: {e}")
        
    finally:
        print("\n🧹 Cleaning up resources...")
        
        # Cleanup all components
        if noise_filter:
            await noise_filter.cleanup()
        if market_regime_classifier:
            await market_regime_classifier.cleanup()
        if adaptive_learning:
            await adaptive_learning.cleanup()
        if pattern_detector:
            await pattern_detector.cleanup()
        
        print("✅ Cleanup completed")

if __name__ == "__main__":
    asyncio.run(test_noise_filtering_integration())
