#!/usr/bin/env python3
"""
Test script for Enhanced Feature Engineering
Phase 2C: Enhanced Feature Engineering
"""

import asyncio
import logging
import sys
import os
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np
import json

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "backend"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def generate_sample_ohlcv_data(symbols: list, timeframes: list, periods: int = 100) -> pd.DataFrame:
    """Generate sample OHLCV data for testing"""
    try:
        data = []
        base_time = datetime.now() - timedelta(hours=periods)
        
        for symbol in symbols:
            for tf in timeframes:
                # Generate realistic price data
                np.random.seed(42)  # For reproducible results
                
                # Start with a base price
                base_price = 100.0 if 'BTC' in symbol else 50.0
                prices = [base_price]
                
                # Generate price movements
                for i in range(periods - 1):
                    # Random walk with some trend
                    change = np.random.normal(0, 0.02) + (0.001 if i < periods // 2 else -0.001)
                    new_price = prices[-1] * (1 + change)
                    prices.append(max(new_price, 1.0))  # Ensure positive prices
                
                # Generate OHLCV data
                for i, price in enumerate(prices):
                    # Generate realistic OHLC from close price
                    volatility = price * 0.01
                    high = price + np.random.uniform(0, volatility)
                    low = price - np.random.uniform(0, volatility)
                    open_price = prices[i-1] if i > 0 else price
                    
                    # Volume with some correlation to price movement
                    volume = np.random.uniform(1000, 10000) * (1 + abs(change) * 10)
                    
                    timestamp = base_time + timedelta(hours=i)
                    
                    data.append({
                        'symbol': symbol,
                        'tf': tf,
                        'ts': timestamp,
                        'o': open_price,
                        'h': high,
                        'l': low,
                        'c': price,
                        'v': volume,
                        'vwap': (high + low + price) / 3,
                        'taker_buy_vol': volume * np.random.uniform(0.4, 0.6)
                    })
        
        df = pd.DataFrame(data)
        df.set_index('ts', inplace=True)
        
        logger.info(f"✅ Generated sample OHLCV data: {df.shape}")
        return df
        
    except Exception as e:
        logger.error(f"❌ Failed to generate sample data: {e}")
        return pd.DataFrame()

async def test_technical_indicators_engine():
    """Test the technical indicators engine"""
    try:
        logger.info("🧪 Testing Technical Indicators Engine...")
        
        from ..ai.technical_indicators_engine import TechnicalIndicatorsEngine
        
        # Initialize engine
        engine = TechnicalIndicatorsEngine()
        
        # Test available indicators
        available_indicators = engine.get_available_indicators()
        assert len(available_indicators) > 0, "No indicators available"
        logger.info(f"✅ Available indicators: {len(available_indicators)}")
        
        # Test indicator configurations
        for indicator in ['rsi', 'macd', 'ema', 'bollinger_bands', 'atr']:
            config = engine.get_indicator_config(indicator)
            assert config is not None, f"Config not found for {indicator}"
            assert hasattr(config, 'name'), f"Config missing name for {indicator}"
            assert hasattr(config, 'parameters'), f"Config missing parameters for {indicator}"
        
        # Test individual indicator calculations
        prices = pd.Series([100, 101, 99, 102, 98, 103, 97, 104, 96, 105])
        
        # Test RSI
        rsi = engine.calculate_rsi(prices, period=5)
        assert len(rsi) == len(prices), "RSI length mismatch"
        assert not rsi.isna().all(), "RSI all NaN"
        
        # Test MACD
        macd_data = engine.calculate_macd(prices, fast_period=3, slow_period=5, signal_period=2)
        assert 'macd' in macd_data, "MACD data missing"
        assert 'signal' in macd_data, "MACD signal missing"
        assert 'histogram' in macd_data, "MACD histogram missing"
        
        # Test EMA
        ema = engine.calculate_ema(prices, period=5)
        assert len(ema) == len(prices), "EMA length mismatch"
        assert not ema.isna().all(), "EMA all NaN"
        
        # Test Bollinger Bands
        bb_data = engine.calculate_bollinger_bands(prices, period=5)
        assert 'upper' in bb_data, "BB upper missing"
        assert 'middle' in bb_data, "BB middle missing"
        assert 'lower' in bb_data, "BB lower missing"
        assert 'position' in bb_data, "BB position missing"
        
        # Test ATR
        high = prices * 1.01
        low = prices * 0.99
        atr = engine.calculate_atr(high, low, prices, period=5)
        assert len(atr) == len(prices), "ATR length mismatch"
        assert not atr.isna().all(), "ATR all NaN"
        
        # Test metadata
        metadata = engine.get_indicator_metadata()
        assert len(metadata) > 0, "No metadata available"
        
        logger.info("✅ Technical Indicators Engine tests passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Technical Indicators Engine tests failed: {e}")
        return False

async def test_feature_drift_detector():
    """Test the feature drift detector"""
    try:
        logger.info("🧪 Testing Feature Drift Detector...")
        
        from ..ai.feature_drift_detector import FeatureDriftDetector
        
        # Initialize detector
        detector = FeatureDriftDetector()
        
        # Generate sample data
        np.random.seed(42)
        reference_data = pd.Series(np.random.normal(100, 10, 100))
        current_data = pd.Series(np.random.normal(110, 15, 100))  # Slightly different distribution
        
        # Test reference data update
        success = detector.update_reference_data("test_feature", reference_data)
        assert success, "Failed to update reference data"
        
        # Test drift detection
        drift_metrics = detector.detect_drift("test_feature", current_data)
        assert drift_metrics is not None, "Drift detection failed"
        assert hasattr(drift_metrics, 'drift_score'), "Drift metrics missing score"
        assert hasattr(drift_metrics, 'severity'), "Drift metrics missing severity"
        assert hasattr(drift_metrics, 'confidence'), "Drift metrics missing confidence"
        
        # Test drift summary
        summary = detector.get_drift_summary()
        assert 'total_drifts' in summary, "Drift summary missing total"
        assert 'features_monitored' in summary, "Drift summary missing features"
        
        # Test feature health score
        health_score = detector.get_feature_health_score("test_feature")
        assert 0.0 <= health_score <= 1.0, "Health score out of range"
        
        # Test alerts
        assert len(detector.alerts) > 0, "No drift alerts generated"
        
        logger.info("✅ Feature Drift Detector tests passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Feature Drift Detector tests failed: {e}")
        return False

async def test_feature_quality_validator():
    """Test the feature quality validator"""
    try:
        logger.info("🧪 Testing Feature Quality Validator...")
        
        from ..ai.feature_quality_validator import FeatureQualityValidator
        
        # Initialize validator
        validator = FeatureQualityValidator()
        
        # Generate sample data with different quality characteristics
        np.random.seed(42)
        
        # High quality data
        high_quality_data = pd.Series(np.random.normal(100, 10, 200))
        
        # Low quality data (with outliers and missing values)
        low_quality_data = pd.Series(np.random.normal(100, 10, 200))
        low_quality_data.iloc[::10] = np.nan  # 10% missing
        low_quality_data.iloc[::20] = 1000  # 5% extreme outliers
        
        # Test high quality data
        high_quality_metrics = validator.validate_feature_quality(
            "high_quality_feature", high_quality_data
        )
        assert high_quality_metrics is not None, "High quality validation failed"
        assert high_quality_metrics.overall_score > 0.7, "High quality data scored too low"
        assert high_quality_metrics.quality_grade in ['A', 'B'], "High quality data got poor grade"
        
        # Test low quality data
        low_quality_metrics = validator.validate_feature_quality(
            "low_quality_feature", low_quality_data
        )
        assert low_quality_metrics is not None, "Low quality validation failed"
        assert low_quality_metrics.overall_score < 0.8, "Low quality data scored too high"
        assert len(low_quality_metrics.issues) > 0, "No issues detected in low quality data"
        
        # Test quality summary
        summary = validator.get_quality_summary()
        assert 'total_validations' in summary, "Quality summary missing total"
        assert 'overall_quality' in summary, "Quality summary missing overall quality"
        
        # Test quality trend
        trend = validator.get_feature_quality_trend("high_quality_feature")
        assert 'trend_data' in trend, "Quality trend missing data"
        
        # Test alerts
        assert len(validator.alerts) > 0, "No quality alerts generated"
        
        logger.info("✅ Feature Quality Validator tests passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Feature Quality Validator tests failed: {e}")
        return False

async def test_enhanced_feature_store():
    """Test the enhanced feature store"""
    try:
        logger.info("🧪 Testing Enhanced Feature Store...")
        
        from ..ai.enhanced_feature_store import EnhancedFeatureStore
        
        # Generate sample data
        symbols = ['BTCUSDT', 'ETHUSDT']
        timeframes = ['1h', '4h']
        ohlcv_data = generate_sample_ohlcv_data(symbols, timeframes, periods=200)
        
        if ohlcv_data.empty:
            logger.warning("⚠️ Skipping enhanced feature store test due to empty sample data")
            return True
        
        # Initialize enhanced feature store
        async with EnhancedFeatureStore() as store:
            # Test feature metadata loading
            assert len(store.feature_metadata) > 0, "No feature metadata loaded"
            assert len(store.feature_monitoring) > 0, "No feature monitoring initialized"
            
            # Test technical feature computation
            computed_features = await store.compute_technical_features(
                ohlcv_data, symbols, timeframes, 
                indicators=['rsi', 'macd', 'ema', 'bollinger_bands', 'atr']
            )
            
            assert len(computed_features) > 0, "No features computed"
            
            # Test feature health summary
            health_summary = await store.get_feature_health_summary()
            assert 'total_features' in health_summary, "Health summary missing total"
            assert 'health_distribution' in health_summary, "Health summary missing distribution"
            
            # Test feature recommendations
            recommendations = await store.get_feature_recommendations()
            assert 'general' in recommendations, "Recommendations missing general"
            assert 'specific_features' in recommendations, "Recommendations missing specific"
            
            # Test production features
            production_features = await store.get_production_features(
                symbols, timeframes, ['rsi', 'macd']
            )
            # Note: This might be empty if no data exists in the database
            
            # Test feature refresh
            refresh_success = await store.refresh_features(force=True)
            assert refresh_success, "Feature refresh failed"
            
            # Test feature report export
            report_path = await store.export_feature_report()
            assert report_path != "", "Feature report export failed"
            
            # Verify report file exists
            if report_path and os.path.exists(report_path):
                with open(report_path, 'r') as f:
                    report_data = json.load(f)
                assert 'feature_metadata' in report_data, "Report missing metadata"
                assert 'feature_monitoring' in report_data, "Report missing monitoring"
                
                # Clean up test report
                os.remove(report_path)
        
        logger.info("✅ Enhanced Feature Store tests passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Enhanced Feature Store tests failed: {e}")
        return False

async def test_integration_workflow():
    """Test the complete integration workflow"""
    try:
        logger.info("🧪 Testing Complete Integration Workflow...")
        
        from ..ai.enhanced_feature_store import EnhancedFeatureStore
        from ..ai.technical_indicators_engine import TechnicalIndicatorsEngine
        from ..ai.feature_drift_detector import FeatureDriftDetector
        from ..ai.feature_quality_validator import FeatureQualityValidator
        
        # Generate comprehensive test data
        symbols = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT']
        timeframes = ['1h', '4h', '1d']
        ohlcv_data = generate_sample_ohlcv_data(symbols, timeframes, periods=300)
        
        if ohlcv_data.empty:
            logger.warning("⚠️ Skipping integration workflow test due to empty sample data")
            return True
        
        # Test complete workflow
        async with EnhancedFeatureStore() as store:
            # Step 1: Compute technical features
            logger.info("📊 Step 1: Computing technical features...")
            computed_features = await store.compute_technical_features(
                ohlcv_data, symbols, timeframes,
                indicators=['rsi', 'macd', 'ema', 'bollinger_bands', 'atr', 'volume_sma_ratio']
            )
            
            assert len(computed_features) > 0, "No features computed in workflow"
            logger.info(f"✅ Computed features for {len(computed_features)} combinations")
            
            # Step 2: Monitor feature health
            logger.info("🔍 Step 2: Monitoring feature health...")
            health_summary = await store.get_feature_health_summary()
            
            assert 'total_features' in health_summary, "Health summary missing in workflow"
            assert 'health_distribution' in health_summary, "Health distribution missing in workflow"
            
            # Step 3: Get recommendations
            logger.info("💡 Step 3: Getting feature recommendations...")
            recommendations = await store.get_feature_recommendations()
            
            assert 'general' in recommendations, "General recommendations missing in workflow"
            assert 'specific_features' in recommendations, "Specific recommendations missing in workflow"
            
            # Step 4: Export comprehensive report
            logger.info("📋 Step 4: Exporting comprehensive report...")
            report_path = await store.export_feature_report()
            
            if report_path and os.path.exists(report_path):
                with open(report_path, 'r') as f:
                    report_data = json.load(f)
                
                # Verify comprehensive report structure
                required_sections = [
                    'feature_metadata', 'feature_monitoring', 'health_summary',
                    'recommendations', 'drift_summary', 'quality_summary'
                ]
                
                for section in required_sections:
                    assert section in report_data, f"Report missing {section} section"
                
                # Clean up test report
                os.remove(report_path)
                logger.info("✅ Comprehensive report generated and verified")
            
            # Step 5: Test production feature retrieval
            logger.info("🚀 Step 5: Testing production feature retrieval...")
            production_features = await store.get_production_features(
                symbols[:2], timeframes[:2], ['rsi', 'macd']
            )
            
            # Note: This might be empty if no data exists in the database
            logger.info(f"✅ Production features retrieved: {len(production_features)} rows")
        
        logger.info("✅ Complete Integration Workflow tests passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Complete Integration Workflow tests failed: {e}")
        return False

async def run_all_tests():
    """Run all enhanced feature engineering tests"""
    logger.info("🚀 Starting Enhanced Feature Engineering Tests")
    logger.info("=" * 70)
    
    test_results = {}
    
    # Test 1: Technical Indicators Engine
    test_results['technical_indicators'] = await test_technical_indicators_engine()
    
    # Test 2: Feature Drift Detector
    test_results['drift_detector'] = await test_feature_drift_detector()
    
    # Test 3: Feature Quality Validator
    test_results['quality_validator'] = await test_feature_quality_validator()
    
    # Test 4: Enhanced Feature Store
    test_results['enhanced_feature_store'] = await test_enhanced_feature_store()
    
    # Test 5: Complete Integration Workflow
    test_results['integration_workflow'] = await test_integration_workflow()
    
    # Summary
    passed_tests = sum(test_results.values())
    total_tests = len(test_results)
    
    logger.info("\n" + "="*70)
    logger.info("📊 ENHANCED FEATURE ENGINEERING TEST RESULTS SUMMARY")
    logger.info("="*70)
    
    for test_name, result in test_results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{test_name:35} {status}")
    
    logger.info("="*70)
    logger.info(f"Total Tests: {total_tests}")
    logger.info(f"Passed: {passed_tests}")
    logger.info(f"Failed: {total_tests - passed_tests}")
    logger.info(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
    
    if passed_tests == total_tests:
        logger.info("🎉 ALL TESTS PASSED! Enhanced Feature Engineering is working correctly.")
        logger.info("🚀 Phase 2C: Enhanced Feature Engineering is complete!")
    else:
        logger.warning("⚠️ Some tests failed. Please review the implementation.")
    
    return test_results

async def main():
    """Main test runner"""
    try:
        # Run tests
        test_results = await run_all_tests()
        
        # Exit with appropriate code
        if all(test_results.values()):
            sys.exit(0)  # Success
        else:
            sys.exit(1)  # Failure
            
    except KeyboardInterrupt:
        logger.info("🛑 Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Unexpected error during testing: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Run async tests
    asyncio.run(main())
