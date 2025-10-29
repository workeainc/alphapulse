#!/usr/bin/env python3
"""
Comprehensive Tests for Market Regime Detection Module
Unit tests, integration tests, and performance benchmarks
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List
import asyncio
import time
import json
import tempfile
import os

from market_regime_detector import (
    MarketRegimeDetector, 
    MarketRegime, 
    RegimeMetrics, 
    RegimeState
)
from backtest_regime import RegimeBacktester

class TestMarketRegimeDetector:
    """Unit tests for MarketRegimeDetector"""
    
    @pytest.fixture
    def detector(self):
        """Create a test detector instance"""
        return MarketRegimeDetector(
            symbol='BTC/USDT',
            timeframe='15m',
            lookback_period=5,
            min_regime_duration=3,
            hysteresis_threshold=0.15,
            enable_ml=False  # Disable ML for unit tests
        )
    
    @pytest.fixture
    def sample_indicators(self):
        """Sample technical indicators"""
        return {
            'adx': 35.0,
            'bb_upper': 52000.0,
            'bb_lower': 48000.0,
            'bb_middle': 50000.0,
            'atr': 1500.0,
            'rsi': 65.0,
            'volume_sma': 1000000.0
        }
    
    @pytest.fixture
    def sample_candlestick(self):
        """Sample candlestick data"""
        return {
            'open': 50000.0,
            'high': 51000.0,
            'low': 49000.0,
            'close': 50500.0,
            'volume': 1500000.0
        }
    
    def test_initialization(self, detector):
        """Test detector initialization"""
        assert detector.symbol == 'BTC/USDT'
        assert detector.timeframe == '15m'
        assert detector.current_regime == MarketRegime.RANGING
        assert detector.regime_confidence == 0.5
        assert detector.regime_duration == 0
        assert len(detector.regime_scores) == 0
        assert len(detector.metric_history) == 0
    
    def test_calculate_bb_width(self, detector):
        """Test Bollinger Band width calculation"""
        bb_width = detector.calculate_bb_width(52000.0, 48000.0, 50000.0)
        expected_width = (52000.0 - 48000.0) / 50000.0
        assert bb_width == expected_width
        assert bb_width == 0.08  # 8% width
    
    def test_calculate_breakout_strength(self, detector):
        """Test breakout strength calculation"""
        strength = detector.calculate_breakout_strength(1500000.0, 1500.0, 35.0)
        
        # Expected calculation:
        # volume_multiplier = 1500000 / 1000000 = 1.5
        # atr_volatility = 1500 / 1000 = 1.5
        # adx_component = 1.0 (since ADX > 25)
        # strength = (1.5 * 0.6 + 1.5 * 0.3 + 1.0 * 0.1) * 100 = 145.0
        
        assert strength == 145.0
    
    def test_calculate_ma_slope(self, detector):
        """Test MA slope calculation"""
        # Add some price history
        prices = [50000.0, 50100.0, 50200.0, 50300.0, 50400.0, 50500.0]
        for price in prices:
            detector.price_history.append(price)
        
        slope = detector.calculate_ma_slope(prices)
        assert slope > 0  # Positive slope for upward trend
    
    def test_classify_regime_rule_based_strong_trend_bull(self, detector, sample_indicators):
        """Test rule-based classification for strong trend bull"""
        # Modify indicators for strong trend bull
        sample_indicators['adx'] = 40.0  # Strong trend
        sample_indicators['rsi'] = 70.0  # Overbought in trend
        
        metrics = RegimeMetrics(
            adx=40.0,
            ma_slope=0.0002,  # Bullish slope
            bb_width=0.06,
            atr=1500.0,
            rsi=70.0,
            volume_ratio=1.5,
            breakout_strength=80.0,
            price_momentum=0.01,
            volatility_score=0.03
        )
        
        regime, confidence = detector.classify_regime_rule_based(metrics)
        assert regime == MarketRegime.STRONG_TREND_BULL
        assert confidence > 0.8
    
    def test_classify_regime_rule_based_strong_trend_bear(self, detector):
        """Test rule-based classification for strong trend bear"""
        metrics = RegimeMetrics(
            adx=40.0,
            ma_slope=-0.0002,  # Bearish slope
            bb_width=0.06,
            atr=1500.0,
            rsi=30.0,  # Oversold
            volume_ratio=1.5,
            breakout_strength=80.0,
            price_momentum=-0.01,
            volatility_score=0.03
        )
        
        regime, confidence = detector.classify_regime_rule_based(metrics)
        assert regime == MarketRegime.STRONG_TREND_BEAR
        assert confidence > 0.8
    
    def test_classify_regime_rule_based_ranging(self, detector):
        """Test rule-based classification for ranging market"""
        metrics = RegimeMetrics(
            adx=20.0,  # Weak trend
            ma_slope=0.00005,  # Small slope
            bb_width=0.04,  # Narrow bands
            atr=1000.0,
            rsi=50.0,  # Neutral
            volume_ratio=1.0,
            breakout_strength=50.0,
            price_momentum=0.001,
            volatility_score=0.02
        )
        
        regime, confidence = detector.classify_regime_rule_based(metrics)
        assert regime == MarketRegime.RANGING
        assert confidence > 0.7
    
    def test_classify_regime_rule_based_volatile_breakout(self, detector):
        """Test rule-based classification for volatile breakout"""
        metrics = RegimeMetrics(
            adx=30.0,
            ma_slope=0.0001,
            bb_width=0.08,  # Wide bands
            atr=2000.0,
            rsi=55.0,
            volume_ratio=2.0,  # High volume
            breakout_strength=75.0,  # High breakout strength
            price_momentum=0.02,
            volatility_score=0.04
        )
        
        regime, confidence = detector.classify_regime_rule_based(metrics)
        assert regime == MarketRegime.VOLATILE_BREAKOUT
        assert confidence > 0.7
    
    def test_apply_smoothing(self, detector):
        """Test regime smoothing"""
        # Add some regime scores
        detector.regime_scores.extend([0.8, 0.7, 0.9])
        
        regime = MarketRegime.STRONG_TREND_BULL
        confidence = 0.85
        
        smoothed_regime, smoothed_confidence = detector.apply_smoothing(regime, confidence)
        
        # Should maintain the same regime due to smoothing
        assert smoothed_regime == MarketRegime.STRONG_TREND_BULL
        assert smoothed_confidence > 0.7
    
    def test_check_regime_change(self, detector):
        """Test regime change validation"""
        # Test minimum duration check
        detector.regime_duration = 2  # Less than min_regime_duration (3)
        should_change = detector.check_regime_change(MarketRegime.STRONG_TREND_BULL, 0.9)
        assert not should_change
        
        # Test hysteresis check
        detector.regime_duration = 5
        detector.regime_confidence = 0.8
        should_change = detector.check_regime_change(MarketRegime.STRONG_TREND_BULL, 0.85)
        assert not should_change  # Not enough confidence increase
        
        # Test valid change
        should_change = detector.check_regime_change(MarketRegime.STRONG_TREND_BULL, 0.95)
        assert should_change
    
    def test_update_regime(self, detector, sample_indicators, sample_candlestick):
        """Test regime update with real data"""
        # Add some price history
        for i in range(10):
            detector.price_history.append(50000.0 + i * 100)
        
        regime_state = detector.update_regime(sample_indicators, sample_candlestick)
        
        assert isinstance(regime_state, RegimeState)
        assert regime_state.regime in MarketRegime
        assert 0.0 <= regime_state.confidence <= 1.0
        assert regime_state.duration_candles >= 0
        assert isinstance(regime_state.metrics, RegimeMetrics)
        assert 0.0 <= regime_state.stability_score <= 1.0
    
    def test_should_filter_signal(self, detector):
        """Test signal filtering based on regime"""
        # Test choppy regime - higher threshold
        detector.current_regime = MarketRegime.CHOPPY
        assert detector.should_filter_signal(0.80)  # Should filter
        assert not detector.should_filter_signal(0.90)  # Should not filter
        
        # Test strong trend - lower threshold
        detector.current_regime = MarketRegime.STRONG_TREND_BULL
        assert detector.should_filter_signal(0.60)  # Should filter
        assert not detector.should_filter_signal(0.70)  # Should not filter
        
        # Test volatile breakout
        detector.current_regime = MarketRegime.VOLATILE_BREAKOUT
        assert detector.should_filter_signal(0.70)  # Should filter
        assert not detector.should_filter_signal(0.80)  # Should not filter
    
    def test_performance_metrics(self, detector):
        """Test performance metrics collection"""
        # Simulate some updates
        for i in range(5):
            detector.update_count += 1
            detector.avg_latency_ms = (detector.avg_latency_ms * (i) + 25.0) / (i + 1)
        
        metrics = detector.get_performance_metrics()
        
        assert metrics['update_count'] == 5
        assert metrics['avg_latency_ms'] == 25.0
        assert metrics['current_regime'] == detector.current_regime.value
        assert metrics['regime_confidence'] == detector.regime_confidence
        assert metrics['stability_score'] == detector.stability_score
        assert metrics['regime_duration'] == detector.regime_duration

class TestRegimeBacktester:
    """Integration tests for RegimeBacktester"""
    
    @pytest.fixture
    def sample_data_path(self):
        """Create temporary sample data file"""
        # Generate sample historical data
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='15min')
        np.random.seed(42)
        
        data = []
        base_price = 50000.0
        
        for i, date in enumerate(dates):
            # Simulate realistic price movement
            price_change = np.sin(i * 0.01) * 1000 + np.random.normal(0, 200)
            current_price = base_price + price_change
            
            data.append({
                'timestamp': date,
                'open': current_price - 50,
                'high': current_price + 100,
                'low': current_price - 100,
                'close': current_price + 50,
                'volume': 1000000 + np.random.randint(-200000, 200000)
            })
        
        df = pd.DataFrame(data)
        
        # Create temporary file
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        df.to_csv(temp_file.name, index=False)
        temp_file.close()
        
        yield temp_file.name
        
        # Cleanup
        os.unlink(temp_file.name)
    
    @pytest.fixture
    def backtester(self, sample_data_path):
        """Create backtester instance"""
        return RegimeBacktester(
            data_path=sample_data_path,
            symbol='BTC/USDT',
            timeframe='15m',
            train_ratio=0.7,
            random_state=42
        )
    
    def test_load_data(self, backtester):
        """Test data loading and preprocessing"""
        success = backtester.load_data()
        assert success
        
        assert backtester.data is not None
        assert len(backtester.data) > 0
        assert len(backtester.train_data) > 0
        assert len(backtester.test_data) > 0
        
        # Check required columns
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            assert col in backtester.data.columns
        
        # Check indicator columns
        indicator_columns = ['rsi', 'macd', 'bb_upper', 'bb_lower', 'bb_middle', 'adx', 'atr']
        for col in indicator_columns:
            assert col in backtester.data.columns
    
    def test_calculate_indicators(self, backtester):
        """Test technical indicator calculations"""
        backtester.load_data()
        
        # Check RSI values
        assert 'rsi' in backtester.data.columns
        assert backtester.data['rsi'].min() >= 0
        assert backtester.data['rsi'].max() <= 100
        
        # Check MACD values
        assert 'macd' in backtester.data.columns
        assert 'macd_signal' in backtester.data.columns
        assert 'macd_histogram' in backtester.data.columns
        
        # Check Bollinger Bands
        assert 'bb_upper' in backtester.data.columns
        assert 'bb_lower' in backtester.data.columns
        assert 'bb_middle' in backtester.data.columns
        
        # Check ADX
        assert 'adx' in backtester.data.columns
        assert backtester.data['adx'].min() >= 0
        
        # Check ATR
        assert 'atr' in backtester.data.columns
        assert backtester.data['atr'].min() >= 0
    
    def test_generate_synthetic_labels(self, backtester):
        """Test synthetic label generation"""
        backtester.load_data()
        labels = backtester.generate_synthetic_labels()
        
        assert len(labels) == len(backtester.data)
        assert labels.min() >= 0
        assert labels.max() <= 5  # 6 regimes (0-5)
        
        # Check label distribution
        label_counts = labels.value_counts()
        assert len(label_counts) > 0  # Should have some labels
    
    def test_backtest_regime_detector(self, backtester):
        """Test regime detector backtesting"""
        backtester.load_data()
        
        # Test with default thresholds
        thresholds = {
            'adx_trend': 25.0,
            'adx_strong_trend': 35.0,
            'ma_slope_bull': 0.0001,
            'ma_slope_bear': -0.0001,
            'bb_width_volatile': 0.05,
            'bb_width_breakout': 0.07,
            'rsi_overbought': 60.0,
            'rsi_oversold': 40.0,
            'volume_ratio_high': 1.5,
            'breakout_strength_high': 70.0
        }
        
        result = backtester.backtest_regime_detector(thresholds, enable_ml=False)
        
        assert result is not None
        assert 0.0 <= result.accuracy <= 1.0
        assert 0.0 <= result.stability_score <= 1.0
        assert result.avg_regime_duration >= 0
        assert result.regime_changes >= 0
        assert 0.0 <= result.signal_filter_rate <= 1.0
        assert 0.0 <= result.win_rate <= 1.0
        assert result.latency_ms >= 0
        assert isinstance(result.thresholds, dict)
        assert isinstance(result.regime_distribution, dict)
    
    def test_optimize_thresholds(self, backtester):
        """Test threshold optimization"""
        backtester.load_data()
        
        # Run optimization with fewer trials for testing
        result = backtester.optimize_thresholds(n_trials=10)
        
        assert result is not None
        assert isinstance(result.best_thresholds, dict)
        assert 0.0 <= result.best_accuracy <= 1.0
        assert 0.0 <= result.best_stability <= 1.0
        assert len(result.optimization_history) > 0
        assert isinstance(result.model_performance, dict)
        
        # Check that optimization improved performance
        assert result.best_accuracy > 0.0
        assert result.best_stability > 0.0
    
    def test_train_ml_model(self, backtester):
        """Test ML model training"""
        backtester.load_data()
        
        success = backtester.train_ml_model()
        assert success
        
        # Check that model files were created
        model_path = f"models/regime_detector_BTC_USDT_15m"
        assert os.path.exists(f"{model_path}_model.pkl")
        assert os.path.exists(f"{model_path}_scaler.pkl")
        
        # Cleanup
        if os.path.exists(f"{model_path}_model.pkl"):
            os.remove(f"{model_path}_model.pkl")
        if os.path.exists(f"{model_path}_scaler.pkl"):
            os.remove(f"{model_path}_scaler.pkl")

class TestPerformanceBenchmarks:
    """Performance benchmarks for regime detection"""
    
    @pytest.fixture
    def performance_detector(self):
        """Create detector for performance testing"""
        return MarketRegimeDetector(
            symbol='BTC/USDT',
            timeframe='15m',
            lookback_period=10,
            min_regime_duration=5,
            hysteresis_threshold=0.2,
            enable_ml=False  # Disable ML for performance tests
        )
    
    def test_latency_benchmark(self, performance_detector):
        """Test that regime updates complete within 50ms"""
        sample_indicators = {
            'adx': 35.0,
            'bb_upper': 52000.0,
            'bb_lower': 48000.0,
            'bb_middle': 50000.0,
            'atr': 1500.0,
            'rsi': 65.0,
            'volume_sma': 1000000.0
        }
        
        sample_candlestick = {
            'open': 50000.0,
            'high': 51000.0,
            'low': 49000.0,
            'close': 50500.0,
            'volume': 1500000.0
        }
        
        # Add price history
        for i in range(20):
            performance_detector.price_history.append(50000.0 + i * 100)
        
        # Measure latency for multiple updates
        latencies = []
        for _ in range(100):
            start_time = time.perf_counter()
            regime_state = performance_detector.update_regime(sample_indicators, sample_candlestick)
            end_time = time.perf_counter()
            
            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)
            
            assert isinstance(regime_state, RegimeState)
        
        avg_latency = np.mean(latencies)
        max_latency = np.max(latencies)
        
        print(f"Average latency: {avg_latency:.2f} ms")
        print(f"Maximum latency: {max_latency:.2f} ms")
        
        # Performance requirements
        assert avg_latency < 50.0, f"Average latency {avg_latency:.2f}ms exceeds 50ms limit"
        assert max_latency < 100.0, f"Maximum latency {max_latency:.2f}ms exceeds 100ms limit"
    
    def test_throughput_benchmark(self, performance_detector):
        """Test throughput (updates per second)"""
        sample_indicators = {
            'adx': 35.0,
            'bb_upper': 52000.0,
            'bb_lower': 48000.0,
            'bb_middle': 50000.0,
            'atr': 1500.0,
            'rsi': 65.0,
            'volume_sma': 1000000.0
        }
        
        sample_candlestick = {
            'open': 50000.0,
            'high': 51000.0,
            'low': 49000.0,
            'close': 50500.0,
            'volume': 1500000.0
        }
        
        # Add price history
        for i in range(20):
            performance_detector.price_history.append(50000.0 + i * 100)
        
        # Measure throughput
        start_time = time.perf_counter()
        updates = 1000
        
        for _ in range(updates):
            regime_state = performance_detector.update_regime(sample_indicators, sample_candlestick)
            assert isinstance(regime_state, RegimeState)
        
        end_time = time.perf_counter()
        duration = end_time - start_time
        throughput = updates / duration
        
        print(f"Throughput: {throughput:.0f} updates/second")
        
        # Should handle at least 100 updates per second
        assert throughput > 100, f"Throughput {throughput:.0f} updates/sec below 100 updates/sec requirement"
    
    def test_memory_usage_benchmark(self, performance_detector):
        """Test memory usage during extended operation"""
        import psutil
        import gc
        
        sample_indicators = {
            'adx': 35.0,
            'bb_upper': 52000.0,
            'bb_lower': 48000.0,
            'bb_middle': 50000.0,
            'atr': 1500.0,
            'rsi': 65.0,
            'volume_sma': 1000000.0
        }
        
        sample_candlestick = {
            'open': 50000.0,
            'high': 51000.0,
            'low': 49000.0,
            'close': 50500.0,
            'volume': 1500000.0
        }
        
        # Add price history
        for i in range(20):
            performance_detector.price_history.append(50000.0 + i * 100)
        
        # Measure initial memory
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Run many updates
        for _ in range(10000):
            regime_state = performance_detector.update_regime(sample_indicators, sample_candlestick)
        
        # Force garbage collection
        gc.collect()
        
        # Measure final memory
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        print(f"Initial memory: {initial_memory:.1f} MB")
        print(f"Final memory: {final_memory:.1f} MB")
        print(f"Memory increase: {memory_increase:.1f} MB")
        
        # Memory increase should be reasonable (< 100 MB)
        assert memory_increase < 100, f"Memory increase {memory_increase:.1f}MB exceeds 100MB limit"

class TestIntegrationScenarios:
    """Integration test scenarios"""
    
    @pytest.fixture
    def integration_detector(self):
        """Create detector for integration testing"""
        return MarketRegimeDetector(
            symbol='BTC/USDT',
            timeframe='15m',
            lookback_period=10,
            min_regime_duration=5,
            hysteresis_threshold=0.2,
            enable_ml=True
        )
    
    def test_regime_stability(self, integration_detector):
        """Test that regimes remain stable under normal conditions"""
        sample_indicators = {
            'adx': 35.0,
            'bb_upper': 52000.0,
            'bb_lower': 48000.0,
            'bb_middle': 50000.0,
            'atr': 1500.0,
            'rsi': 65.0,
            'volume_sma': 1000000.0
        }
        
        sample_candlestick = {
            'open': 50000.0,
            'high': 51000.0,
            'low': 49000.0,
            'close': 50500.0,
            'volume': 1500000.0
        }
        
        # Add price history
        for i in range(20):
            integration_detector.price_history.append(50000.0 + i * 100)
        
        # Run multiple updates with similar data
        regimes = []
        for _ in range(20):
            regime_state = integration_detector.update_regime(sample_indicators, sample_candlestick)
            regimes.append(regime_state.regime)
        
        # Check stability - should not change regimes too frequently
        regime_changes = sum(1 for i in range(1, len(regimes)) if regimes[i] != regimes[i-1])
        change_rate = regime_changes / len(regimes)
        
        print(f"Regime change rate: {change_rate:.2f}")
        
        # Should be stable (low change rate)
        assert change_rate < 0.3, f"Regime change rate {change_rate:.2f} too high"
    
    def test_regime_transitions(self, integration_detector):
        """Test regime transitions under changing market conditions"""
        # Add price history
        for i in range(20):
            integration_detector.price_history.append(50000.0 + i * 100)
        
        # Test transition from ranging to trending
        # Start with ranging conditions
        ranging_indicators = {
            'adx': 20.0,
            'bb_upper': 50200.0,
            'bb_lower': 49800.0,
            'bb_middle': 50000.0,
            'atr': 1000.0,
            'rsi': 50.0,
            'volume_sma': 1000000.0
        }
        
        ranging_candlestick = {
            'open': 50000.0,
            'high': 50100.0,
            'low': 49900.0,
            'close': 50050.0,
            'volume': 1000000.0
        }
        
        # Update with ranging data
        for _ in range(10):
            regime_state = integration_detector.update_regime(ranging_indicators, ranging_candlestick)
        
        initial_regime = regime_state.regime
        
        # Switch to trending conditions
        trending_indicators = {
            'adx': 40.0,
            'bb_upper': 52000.0,
            'bb_lower': 48000.0,
            'bb_middle': 50000.0,
            'atr': 1500.0,
            'rsi': 70.0,
            'volume_sma': 1000000.0
        }
        
        trending_candlestick = {
            'open': 50000.0,
            'high': 51000.0,
            'low': 49000.0,
            'close': 50500.0,
            'volume': 1500000.0
        }
        
        # Update with trending data
        for _ in range(15):
            regime_state = integration_detector.update_regime(trending_indicators, trending_candlestick)
        
        final_regime = regime_state.regime
        
        # Should transition to a different regime
        assert final_regime != initial_regime, "Regime should have transitioned"
        assert final_regime in [MarketRegime.STRONG_TREND_BULL, MarketRegime.WEAK_TREND], f"Expected trending regime, got {final_regime}"

def main():
    """Run all tests"""
    pytest.main([__file__, "-v", "--tb=short"])

if __name__ == "__main__":
    main()
