#!/usr/bin/env python3
"""
Simple SMC (Smart Money Concepts) Test
Validates SMC pattern detection without complex dependencies
"""

import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

def test_smc_basic():
    """Basic SMC pattern detection test"""
    print("🧪 Testing SMC Basic Pattern Detection...")
    
    # Generate simple test data
    np.random.seed(42)
    base_price = 45000
    dates = pd.date_range(start='2025-01-01', periods=50, freq='1h')
    
    # Create bullish order block pattern
    prices = []
    for i in range(50):
        if i < 40:
            # Normal trading
            price = base_price + np.random.normal(0, 100)
        elif i < 45:
            # Strong bullish move (order block)
            price = base_price + 500 + np.random.normal(0, 50)
        else:
            # Consolidation after the move
            price = base_price + 400 + np.random.normal(0, 30)
        prices.append(price)
    
    # Create OHLCV data
    df = pd.DataFrame({
        'timestamp': dates,
        'open': prices,
        'high': [p + abs(np.random.normal(0, 20)) for p in prices],
        'low': [p - abs(np.random.normal(0, 20)) for p in prices],
        'close': [p + np.random.normal(0, 10) for p in prices],
        'volume': [np.random.randint(1000, 10000) for _ in prices]
    })
    
    # Ensure OHLC relationships are valid
    df['high'] = df[['open', 'high', 'close']].max(axis=1)
    df['low'] = df[['open', 'low', 'close']].min(axis=1)
    
    print(f"✅ Test data generated: {len(df)} candles")
    print(f"   Price range: {df['low'].min():.2f} - {df['high'].max():.2f}")
    print(f"   Volume range: {df['volume'].min()} - {df['volume'].max()}")
    
    # Test basic pattern detection logic
    try:
        # Test order block detection logic
        recent_df = df.tail(30)
        highs = recent_df['high'].values
        lows = recent_df['low'].values
        opens = recent_df['open'].values
        closes = recent_df['close'].values
        volumes = recent_df['volume'].values
        
        order_blocks_found = 0
        
        for i in range(5, len(recent_df) - 5):
            # Calculate move strength
            body_size = abs(closes[i] - opens[i])
            total_range = highs[i] - lows[i]
            body_ratio = body_size / total_range if total_range > 0 else 0
            
            # Volume confirmation
            avg_volume = np.mean(volumes[max(0, i-5):i+5])
            volume_ratio = volumes[i] / avg_volume if avg_volume > 0 else 1.0
            
            # Check for strong move (potential order block)
            if body_ratio > 0.6 and volume_ratio > 1.2:
                order_blocks_found += 1
                print(f"   Order block found at index {i}: body_ratio={body_ratio:.2f}, volume_ratio={volume_ratio:.2f}")
        
        print(f"✅ Order blocks detected: {order_blocks_found}")
        
        # Test fair value gap detection
        fair_value_gaps_found = 0
        
        for i in range(1, len(recent_df) - 1):
            current_low = lows[i]
            prev_high = highs[i-1]
            
            # Bullish Fair Value Gap: Current low > Previous high
            if current_low > prev_high:
                gap_size = current_low - prev_high
                gap_ratio = gap_size / prev_high if prev_high > 0 else 0
                
                if gap_ratio > 0.001:  # Minimum gap size (0.1%)
                    fair_value_gaps_found += 1
                    print(f"   Fair value gap found at index {i}: gap_size={gap_size:.2f}, gap_ratio={gap_ratio:.4f}")
        
        print(f"✅ Fair value gaps detected: {fair_value_gaps_found}")
        
        # Test liquidity sweep detection
        support_levels = []
        for i in range(2, len(lows) - 2):
            if lows[i] < lows[i-1] and lows[i] < lows[i-2] and lows[i] < lows[i+1] and lows[i] < lows[i+2]:
                support_levels.append(lows[i])
        
        support_levels = sorted(list(set(support_levels)))[:3]  # Top 3 support levels
        print(f"✅ Support levels identified: {len(support_levels)}")
        
        liquidity_sweeps_found = 0
        for i in range(5, len(recent_df) - 5):
            current_low = lows[i]
            current_close = closes[i]
            
            for support_level in support_levels:
                if current_low < support_level * 0.995:  # Sweep below support
                    # Check for reversal in next few candles
                    for j in range(i + 1, min(i + 5, len(closes))):
                        if closes[j] > support_level:
                            liquidity_sweeps_found += 1
                            print(f"   Liquidity sweep found at index {i}: support={support_level:.2f}, sweep_low={current_low:.2f}")
                            break
                    break  # Only count one sweep per support level
        
        print(f"✅ Liquidity sweeps detected: {liquidity_sweeps_found}")
        
        # Overall results
        total_patterns = order_blocks_found + fair_value_gaps_found + liquidity_sweeps_found
        print(f"✅ Total SMC patterns detected: {total_patterns}")
        
        if total_patterns > 0:
            print("🎉 SMC Basic Test: SUCCESS")
            return True
        else:
            print("⚠️ No SMC patterns detected")
            return False
            
    except Exception as e:
        print(f"❌ SMC Basic Test failed: {e}")
        return False

def test_performance():
    """Test SMC pattern detection performance"""
    print("🧪 Testing SMC Performance...")
    
    try:
        import time
        
        # Generate test data
        np.random.seed(42)
        base_price = 45000
        dates = pd.date_range(start='2025-01-01', periods=100, freq='1h')
        
        prices = [base_price + np.random.normal(0, 100) for _ in range(100)]
        
        df = pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': [p + abs(np.random.normal(0, 20)) for p in prices],
            'low': [p - abs(np.random.normal(0, 20)) for p in prices],
            'close': [p + np.random.normal(0, 10) for p in prices],
            'volume': [np.random.randint(1000, 10000) for _ in prices]
        })
        
        df['high'] = df[['open', 'high', 'close']].max(axis=1)
        df['low'] = df[['open', 'low', 'close']].min(axis=1)
        
        # Test performance
        start_time = time.time()
        
        # Run pattern detection multiple times
        for _ in range(10):
            recent_df = df.tail(30)
            highs = recent_df['high'].values
            lows = recent_df['low'].values
            opens = recent_df['open'].values
            closes = recent_df['close'].values
            volumes = recent_df['volume'].values
            
            # Order block detection
            for i in range(5, len(recent_df) - 5):
                body_size = abs(closes[i] - opens[i])
                total_range = highs[i] - lows[i]
                body_ratio = body_size / total_range if total_range > 0 else 0
                
                avg_volume = np.mean(volumes[max(0, i-5):i+5])
                volume_ratio = volumes[i] / avg_volume if avg_volume > 0 else 1.0
                
                if body_ratio > 0.6 and volume_ratio > 1.2:
                    pass  # Order block found
        
        end_time = time.time()
        processing_time = (end_time - start_time) * 1000
        
        print(f"✅ Performance test completed")
        print(f"   Average processing time: {processing_time / 10:.2f}ms per detection")
        
        if processing_time / 10 < 100:  # Should be under 100ms
            print("✅ Performance meets requirements (<100ms)")
            return True
        else:
            print(f"⚠️ Performance exceeds requirements: {processing_time / 10:.2f}ms")
            return False
            
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        return False

def main():
    """Main test execution"""
    print("🚀 Starting SMC Simple Test Suite")
    print("=" * 50)
    
    test_results = []
    
    # Run tests
    test_results.append(test_smc_basic())
    test_results.append(test_performance())
    
    # Summary
    passed_tests = sum(test_results)
    total_tests = len(test_results)
    
    print("=" * 50)
    print(f"📊 Test Results Summary:")
    print(f"   Passed: {passed_tests}/{total_tests}")
    print(f"   Success Rate: {passed_tests/total_tests*100:.1f}%")
    
    if passed_tests == total_tests:
        print("🎉 All SMC simple tests passed!")
        return 0
    else:
        print(f"⚠️ {total_tests - passed_tests} tests failed")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
