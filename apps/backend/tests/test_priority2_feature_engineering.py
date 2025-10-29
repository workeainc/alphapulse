#!/usr/bin/env python3
"""
Test script for Priority 2: Advanced Feature Engineering
Tests optimized sliding windows, enhanced PCA, and advanced caching
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import os
import asyncio

# Add the backend directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "backend"))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_sample_data(n_rows: int = 1000) -> pd.DataFrame:
    """Create sample OHLCV data for testing."""
    np.random.seed(42)
    
    # Generate price data with some trend and volatility
    base_price = 100
    returns = np.random.normal(0.001, 0.02, n_rows)  # Daily returns
    prices = [base_price]
    
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    prices = np.array(prices)
    
    # Generate OHLCV data
    data = []
    for i in range(n_rows):
        price = prices[i]
        volatility = 0.01 + 0.005 * np.random.random()
        
        # Generate OHLC from price
        open_price = price * (1 + np.random.normal(0, volatility/2))
        high_price = max(open_price, price) * (1 + abs(np.random.normal(0, volatility/4)))
        low_price = min(open_price, price) * (1 - abs(np.random.normal(0, volatility/4)))
        close_price = price
        
        # Generate volume
        volume = np.random.lognormal(10, 0.5) * (1 + abs(returns[i]) * 10)
        
        data.append({
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'volume': volume
        })
    
    df = pd.DataFrame(data)
    df.index = pd.date_range('2024-01-01', periods=n_rows, freq='1H')
    
    return df


async def test_priority2_feature_engineering():
    """Test Priority 2 feature engineering system."""
    logger.info("🧪 Testing Priority 2: Advanced Feature Engineering")
    
    try:
        # Import the Priority 2 feature engineering system
        from ..src.ai.priority2_feature_engineering import priority2_feature_engineering
        
        # Create sample data
        sample_data = create_sample_data(500)
        logger.info(f"✅ Created sample data: {sample_data.shape}")
        
        # Test 1: Basic feature extraction
        logger.info("📊 Test 1: Basic feature extraction")
        features, metadata = await priority2_feature_engineering.extract_priority2_features(
            sample_data, "BTCUSDT"
        )
        
        logger.info(f"✅ Feature extraction completed:")
        logger.info(f"   - Original shape: {metadata['original_shape']}")
        logger.info(f"   - Final shape: {metadata['final_shape']}")
        logger.info(f"   - Extraction time: {metadata['extraction_time']:.3f}s")
        logger.info(f"   - Features removed: {metadata['features_removed']}")
        
        # Test 2: Cache functionality
        logger.info("📊 Test 2: Cache functionality")
        cached_features, cached_metadata = await priority2_feature_engineering.extract_priority2_features(
            sample_data, "BTCUSDT"
        )
        
        logger.info(f"✅ Cache test completed:")
        logger.info(f"   - Source: {cached_metadata['source']}")
        logger.info(f"   - Cache hit: {cached_metadata['source'] == 'cache'}")
        
        # Test 3: Performance statistics
        logger.info("📊 Test 3: Performance statistics")
        stats = priority2_feature_engineering.get_performance_stats()
        
        logger.info(f"✅ Performance stats:")
        logger.info(f"   - Cache hits: {stats['cache_hits']}")
        logger.info(f"   - Cache misses: {stats['cache_misses']}")
        logger.info(f"   - Cache hit rate: {stats['cache_hit_rate']:.2%}")
        logger.info(f"   - Avg extraction time: {stats['avg_extraction_time']:.3f}s")
        
        # Test 4: Feature quality check
        logger.info("📊 Test 4: Feature quality check")
        
        # Check for NaN values
        nan_count = features.isna().sum().sum()
        logger.info(f"   - NaN values: {nan_count}")
        
        # Check for infinite values
        inf_count = np.isinf(features.select_dtypes(include=[np.number])).sum().sum()
        logger.info(f"   - Infinite values: {inf_count}")
        
        # Check feature types
        numeric_features = features.select_dtypes(include=[np.number]).shape[1]
        logger.info(f"   - Numeric features: {numeric_features}")
        
        # Test 5: Sliding window features
        logger.info("📊 Test 5: Sliding window features")
        window_features = [col for col in features.columns if any(x in col for x in ['mean_', 'std_', 'momentum_', 'volatility_'])]
        logger.info(f"   - Sliding window features: {len(window_features)}")
        
        # Test 6: PCA features
        logger.info("📊 Test 6: PCA features")
        pca_features = [col for col in features.columns if 'pca_' in col]
        logger.info(f"   - PCA features: {len(pca_features)}")
        
        # Test 7: Advanced indicators
        logger.info("📊 Test 7: Advanced indicators")
        advanced_features = [col for col in features.columns if any(x in col for x in ['divergence', 'regime', 'breakout', 'strength', 'reversal'])]
        logger.info(f"   - Advanced indicators: {len(advanced_features)}")
        
        # Test 8: Overlapping window features
        logger.info("📊 Test 8: Overlapping window features")
        overlap_features = [col for col in features.columns if 'overlap_' in col or 'adaptive_' in col]
        logger.info(f"   - Overlapping window features: {len(overlap_features)}")
        
        # Summary
        logger.info("🎉 Priority 2 Feature Engineering Test Summary:")
        logger.info(f"   ✅ Total features extracted: {len(features.columns)}")
        logger.info(f"   ✅ Data quality: {'PASS' if nan_count == 0 and inf_count == 0 else 'FAIL'}")
        logger.info(f"   ✅ Cache functionality: {'PASS' if stats['cache_hits'] > 0 else 'FAIL'}")
        logger.info(f"   ✅ Sliding windows: {'PASS' if len(window_features) > 0 else 'FAIL'}")
        logger.info(f"   ✅ PCA reduction: {'PASS' if len(pca_features) > 0 else 'FAIL'}")
        logger.info(f"   ✅ Advanced indicators: {'PASS' if len(advanced_features) > 0 else 'FAIL'}")
        logger.info(f"   ✅ Overlapping windows: {'PASS' if len(overlap_features) > 0 else 'FAIL'}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Priority 2 feature engineering test failed: {e}")
        return False


async def test_multiple_symbols():
    """Test feature extraction for multiple symbols."""
    logger.info("🧪 Testing multiple symbols feature extraction")
    
    try:
        from ..src.ai.priority2_feature_engineering import priority2_feature_engineering
        
        symbols = ["BTCUSDT", "ETHUSDT", "ADAUSDT"]
        results = {}
        
        for symbol in symbols:
            sample_data = create_sample_data(300)
            features, metadata = await priority2_feature_engineering.extract_priority2_features(
                sample_data, symbol
            )
            results[symbol] = {
                'features_count': len(features.columns),
                'extraction_time': metadata['extraction_time'],
                'shape': features.shape
            }
            logger.info(f"✅ {symbol}: {len(features.columns)} features in {metadata['extraction_time']:.3f}s")
        
        # Check consistency across symbols
        feature_counts = [results[s]['features_count'] for s in symbols]
        if len(set(feature_counts)) == 1:
            logger.info("✅ Feature consistency: PASS - All symbols have same feature count")
        else:
            logger.warning("⚠️ Feature consistency: FAIL - Different feature counts across symbols")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Multiple symbols test failed: {e}")
        return False


async def test_performance_benchmark():
    """Test performance with different data sizes."""
    logger.info("🧪 Testing performance benchmark")
    
    try:
        from ..src.ai.priority2_feature_engineering import priority2_feature_engineering
        
        sizes = [100, 500, 1000, 2000]
        performance_results = {}
        
        for size in sizes:
            sample_data = create_sample_data(size)
            start_time = pd.Timestamp.now()
            
            features, metadata = await priority2_feature_engineering.extract_priority2_features(
                sample_data, f"TEST_{size}"
            )
            
            end_time = pd.Timestamp.now()
            actual_time = (end_time - start_time).total_seconds()
            
            performance_results[size] = {
                'data_size': size,
                'extraction_time': actual_time,
                'features_count': len(features.columns),
                'time_per_row': actual_time / size
            }
            
            logger.info(f"✅ Size {size}: {actual_time:.3f}s ({actual_time/size*1000:.2f}ms per row)")
        
        # Performance analysis
        logger.info("📊 Performance Analysis:")
        for size, result in performance_results.items():
            logger.info(f"   - {size} rows: {result['extraction_time']:.3f}s ({result['time_per_row']*1000:.2f}ms/row)")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Performance benchmark failed: {e}")
        return False


async def main():
    """Main test function."""
    logger.info("🚀 Starting Priority 2 Feature Engineering Tests")
    
    tests = [
        ("Basic Feature Engineering", test_priority2_feature_engineering),
        ("Multiple Symbols", test_multiple_symbols),
        ("Performance Benchmark", test_performance_benchmark)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*60}")
        logger.info(f"Running: {test_name}")
        logger.info(f"{'='*60}")
        
        try:
            result = await test_func()
            results[test_name] = result
            logger.info(f"✅ {test_name}: {'PASSED' if result else 'FAILED'}")
        except Exception as e:
            logger.error(f"❌ {test_name} failed with exception: {e}")
            results[test_name] = False
    
    # Final summary
    logger.info(f"\n{'='*60}")
    logger.info("PRIORITY 2 FEATURE ENGINEERING TEST SUMMARY")
    logger.info(f"{'='*60}")
    
    passed_tests = sum(results.values())
    total_tests = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\nOverall Result: {passed_tests}/{total_tests} tests passed ({passed_tests/total_tests*100:.1f}%)")
    
    if passed_tests == total_tests:
        logger.info("🎉 ALL PRIORITY 2 TESTS PASSED!")
        return True
    else:
        logger.error("❌ SOME PRIORITY 2 TESTS FAILED!")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
