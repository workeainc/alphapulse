#!/usr/bin/env python3
"""
Consolidated Indicator Tests for AlphaPulse
Tests for technical indicators and pattern detection
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Any
import asyncio

# Import the modules to test
from ..core.indicators_engine import IndicatorsEngine
from ..utils.feature_engineering import FeatureEngineer

class TestIndicatorsEngine:
    """Test suite for IndicatorsEngine"""
    
    @pytest.fixture
    def sample_candles(self):
        """Generate sample candle data for testing"""
        candles = []
        base_price = 100.0
        
        for i in range(200):
            # Create realistic price movement
            price_change = np.random.normal(0, 0.5)  # 0.5% volatility
            base_price *= (1 + price_change / 100)
            
            high = base_price * (1 + abs(np.random.normal(0, 0.002)))
            low = base_price * (1 - abs(np.random.normal(0, 0.002)))
            open_price = base_price * (1 + np.random.normal(0, 0.001))
            close_price = base_price * (1 + np.random.normal(0, 0.001))
            
            candles.append({
                'timestamp': datetime.now() - timedelta(minutes=i),
                'open': open_price,
                'high': high,
                'low': low,
                'close': close_price,
                'volume': 1000 + np.random.randint(-100, 100),
                'symbol': 'BTC/USDT',
                'timeframe': '1m'
            })
        
        return candles[::-1]  # Reverse to get chronological order
    
    @pytest.fixture
    def indicators_engine(self):
        """Create IndicatorsEngine instance"""
        return IndicatorsEngine()
    
    @pytest.mark.asyncio
    async def test_rsi_calculation(self, indicators_engine, sample_candles):
        """Test RSI calculation"""
        # Calculate RSI
        rsi = await indicators_engine.calculate_rsi(sample_candles, period=14)
        
        # Verify RSI values are within valid range [0, 100]
        assert 0 <= rsi <= 100, f"RSI value {rsi} is outside valid range [0, 100]"
        
        # Verify RSI is calculated for recent data
        assert not np.isnan(rsi), "RSI should not be NaN"
        
        # Test with different periods
        rsi_20 = await indicators_engine.calculate_rsi(sample_candles, period=20)
        assert 0 <= rsi_20 <= 100, f"RSI-20 value {rsi_20} is outside valid range"
    
    @pytest.mark.asyncio
    async def test_macd_calculation(self, indicators_engine, sample_candles):
        """Test MACD calculation"""
        # Calculate MACD
        macd, signal, histogram = await indicators_engine.calculate_macd(sample_candles)
        
        # Verify MACD components are calculated
        assert not np.isnan(macd), "MACD should not be NaN"
        assert not np.isnan(signal), "MACD signal should not be NaN"
        assert not np.isnan(histogram), "MACD histogram should not be NaN"
        
        # Verify histogram = MACD - signal
        assert abs(histogram - (macd - signal)) < 1e-10, "MACD histogram should equal MACD - signal"
    
    @pytest.mark.asyncio
    async def test_bollinger_bands(self, indicators_engine, sample_candles):
        """Test Bollinger Bands calculation"""
        # Calculate Bollinger Bands
        upper, middle, lower = await indicators_engine.calculate_bollinger_bands(sample_candles, period=20)
        
        # Verify bands are calculated
        assert not np.isnan(upper), "Upper band should not be NaN"
        assert not np.isnan(middle), "Middle band should not be NaN"
        assert not np.isnan(lower), "Lower band should not be NaN"
        
        # Verify band relationships
        assert upper >= middle, "Upper band should be >= middle band"
        assert middle >= lower, "Middle band should be >= lower band"
        assert upper >= lower, "Upper band should be >= lower band"
        
        # Verify band width is reasonable
        band_width = (upper - lower) / middle
        assert 0 < band_width < 1, f"Band width {band_width} should be between 0 and 1"
    
    @pytest.mark.asyncio
    async def test_atr_calculation(self, indicators_engine, sample_candles):
        """Test Average True Range calculation"""
        # Calculate ATR
        atr = await indicators_engine.calculate_atr(sample_candles, period=14)
        
        # Verify ATR is positive
        assert atr > 0, f"ATR should be positive, got {atr}"
        assert not np.isnan(atr), "ATR should not be NaN"
        
        # Test with different periods
        atr_20 = await indicators_engine.calculate_atr(sample_candles, period=20)
        assert atr_20 > 0, f"ATR-20 should be positive, got {atr_20}"
    
    @pytest.mark.asyncio
    async def test_volume_analysis(self, indicators_engine, sample_candles):
        """Test volume analysis functions"""
        # Calculate volume metrics
        volume_sma = await indicators_engine.calculate_volume_sma(sample_candles, period=20)
        volume_ratio = await indicators_engine.calculate_volume_ratio(sample_candles, period=20)
        
        # Verify volume metrics
        assert volume_sma > 0, f"Volume SMA should be positive, got {volume_sma}"
        assert volume_ratio > 0, f"Volume ratio should be positive, got {volume_ratio}"
        assert not np.isnan(volume_sma), "Volume SMA should not be NaN"
        assert not np.isnan(volume_ratio), "Volume ratio should not be NaN"
    
    @pytest.mark.asyncio
    async def test_trend_analysis(self, indicators_engine, sample_candles):
        """Test trend analysis functions"""
        # Calculate trend metrics
        trend_strength = await indicators_engine.calculate_trend_strength(sample_candles, period=20)
        trend_direction = await indicators_engine.calculate_trend_direction(sample_candles, period=20)
        
        # Verify trend metrics
        assert 0 <= trend_strength <= 1, f"Trend strength should be between 0 and 1, got {trend_strength}"
        assert trend_direction in [-1, 0, 1], f"Trend direction should be -1, 0, or 1, got {trend_direction}"
        assert not np.isnan(trend_strength), "Trend strength should not be NaN"
    
    @pytest.mark.asyncio
    async def test_breakout_detection(self, indicators_engine, sample_candles):
        """Test breakout detection"""
        # Calculate breakout metrics
        breakout_strength = await indicators_engine.calculate_breakout_strength(sample_candles)
        breakout_direction = await indicators_engine.calculate_breakout_direction(sample_candles)
        
        # Verify breakout metrics
        assert 0 <= breakout_strength <= 1, f"Breakout strength should be between 0 and 1, got {breakout_strength}"
        assert breakout_direction in [-1, 0, 1], f"Breakout direction should be -1, 0, or 1, got {breakout_direction}"
        assert not np.isnan(breakout_strength), "Breakout strength should not be NaN"
    
    @pytest.mark.asyncio
    async def test_all_indicators(self, indicators_engine, sample_candles):
        """Test calculation of all indicators at once"""
        # Calculate all indicators
        indicators = await indicators_engine.calculate_all(sample_candles)
        
        # Verify all expected indicators are present
        expected_indicators = [
            'rsi', 'macd', 'macd_signal', 'macd_histogram',
            'bb_upper', 'bb_middle', 'bb_lower', 'bb_width',
            'atr', 'volume_sma', 'volume_ratio', 'trend_strength',
            'trend_direction', 'breakout_strength', 'breakout_direction'
        ]
        
        for indicator in expected_indicators:
            assert indicator in indicators, f"Indicator {indicator} should be present"
            assert not np.isnan(indicators[indicator]), f"Indicator {indicator} should not be NaN"
    
    @pytest.mark.asyncio
    async def test_edge_cases(self, indicators_engine):
        """Test edge cases and error handling"""
        # Test with insufficient data
        with pytest.raises(ValueError):
            await indicators_engine.calculate_rsi([], period=14)
        
        # Test with minimal data
        minimal_candles = [
            {'open': 100, 'high': 101, 'low': 99, 'close': 100.5, 'volume': 1000}
        ] * 15  # Just enough for RSI-14
        
        rsi = await indicators_engine.calculate_rsi(minimal_candles, period=14)
        assert not np.isnan(rsi), "RSI should be calculated even with minimal data"
    
    @pytest.mark.asyncio
    async def test_performance(self, indicators_engine, sample_candles):
        """Test performance of indicator calculations"""
        import time
        
        # Time the calculation of all indicators
        start_time = time.time()
        indicators = await indicators_engine.calculate_all(sample_candles)
        end_time = time.time()
        
        calculation_time = end_time - start_time
        
        # Should complete within reasonable time (adjust threshold as needed)
        assert calculation_time < 1.0, f"Indicator calculation took {calculation_time:.3f}s, should be < 1.0s"
        
        print(f"Indicator calculation time: {calculation_time:.3f}s")

class TestFeatureEngineering:
    """Test suite for FeatureEngineer"""
    
    @pytest.fixture
    def sample_candles(self):
        """Generate sample candle data for testing"""
        candles = []
        base_price = 100.0
        
        for i in range(200):
            # Create realistic price movement
            price_change = np.random.normal(0, 0.5)
            base_price *= (1 + price_change / 100)
            
            high = base_price * (1 + abs(np.random.normal(0, 0.002)))
            low = base_price * (1 - abs(np.random.normal(0, 0.002)))
            open_price = base_price * (1 + np.random.normal(0, 0.001))
            close_price = base_price * (1 + np.random.normal(0, 0.001))
            
            candles.append({
                'timestamp': datetime.now() - timedelta(minutes=i),
                'open': open_price,
                'high': high,
                'low': low,
                'close': close_price,
                'volume': 1000 + np.random.randint(-100, 100),
                'symbol': 'BTC/USDT',
                'timeframe': '1m'
            })
        
        return candles[::-1]
    
    @pytest.fixture
    def feature_engineer(self):
        """Create FeatureEngineer instance"""
        return FeatureEngineer()
    
    def test_feature_engineering(self, feature_engineer, sample_candles):
        """Test feature engineering functionality"""
        # Engineer features
        feature_set = feature_engineer.engineer_features(sample_candles)
        
        # Verify feature set structure
        assert hasattr(feature_set, 'timestamp'), "FeatureSet should have timestamp"
        assert hasattr(feature_set, 'symbol'), "FeatureSet should have symbol"
        assert hasattr(feature_set, 'timeframe'), "FeatureSet should have timeframe"
        assert hasattr(feature_set, 'features'), "FeatureSet should have features"
        assert hasattr(feature_set, 'metadata'), "FeatureSet should have metadata"
        
        # Verify feature set content
        assert feature_set.symbol == 'BTC/USDT', f"Symbol should be BTC/USDT, got {feature_set.symbol}"
        assert feature_set.timeframe == '1m', f"Timeframe should be 1m, got {feature_set.timeframe}"
        assert isinstance(feature_set.features, dict), "Features should be a dictionary"
        assert len(feature_set.features) > 0, "Features dictionary should not be empty"
        
        # Verify key features are present
        expected_features = ['close', 'volume', 'rsi_14', 'macd', 'atr_14']
        for feature in expected_features:
            if feature in feature_set.features:
                assert not np.isnan(feature_set.features[feature]), f"Feature {feature} should not be NaN"
    
    def test_price_features(self, feature_engineer, sample_candles):
        """Test price-based feature extraction"""
        df = pd.DataFrame(sample_candles)
        features = feature_engineer._price_features(df)
        
        # Verify price features
        assert 'close' in features, "Price features should include 'close'"
        assert 'open' in features, "Price features should include 'open'"
        assert 'high' in features, "Price features should include 'high'"
        assert 'low' in features, "Price features should include 'low'"
        
        # Verify moving averages
        for period in feature_engineer.lookback_periods:
            if len(df) >= period:
                assert f'sma_{period}' in features, f"Price features should include SMA-{period}"
                assert f'ema_{period}' in features, f"Price features should include EMA-{period}"
    
    def test_volume_features(self, feature_engineer, sample_candles):
        """Test volume-based feature extraction"""
        df = pd.DataFrame(sample_candles)
        features = feature_engineer._volume_features(df)
        
        # Verify volume features
        assert 'volume' in features, "Volume features should include 'volume'"
        
        # Verify volume metrics
        for period in feature_engineer.lookback_periods:
            if len(df) >= period:
                assert f'volume_sma_{period}' in features, f"Volume features should include volume SMA-{period}"
                assert f'volume_ratio_{period}' in features, f"Volume features should include volume ratio-{period}"
    
    def test_technical_features(self, feature_engineer, sample_candles):
        """Test technical indicator feature extraction"""
        df = pd.DataFrame(sample_candles)
        features = feature_engineer._technical_features(df)
        
        # Verify technical indicators
        assert 'rsi_14' in features, "Technical features should include RSI-14"
        assert 'rsi_20' in features, "Technical features should include RSI-20"
        
        if len(df) >= 26:  # MACD requires at least 26 periods
            assert 'macd' in features, "Technical features should include MACD"
            assert 'macd_signal' in features, "Technical features should include MACD signal"
            assert 'macd_histogram' in features, "Technical features should include MACD histogram"
        
        # Verify Bollinger Bands
        for period in [20, 50]:
            if len(df) >= period:
                assert f'bb_upper_{period}' in features, f"Technical features should include BB upper-{period}"
                assert f'bb_lower_{period}' in features, f"Technical features should include BB lower-{period}"
                assert f'bb_width_{period}' in features, f"Technical features should include BB width-{period}"
    
    def test_volatility_features(self, feature_engineer, sample_candles):
        """Test volatility-based feature extraction"""
        df = pd.DataFrame(sample_candles)
        features = feature_engineer._volatility_features(df)
        
        # Verify volatility features
        for period in feature_engineer.lookback_periods:
            if len(df) >= period:
                assert f'volatility_{period}' in features, f"Volatility features should include volatility-{period}"
                assert f'volatility_annualized_{period}' in features, f"Volatility features should include annualized volatility-{period}"
    
    def test_momentum_features(self, feature_engineer, sample_candles):
        """Test momentum-based feature extraction"""
        df = pd.DataFrame(sample_candles)
        features = feature_engineer._momentum_features(df)
        
        # Verify momentum features
        for period in feature_engineer.lookback_periods:
            if len(df) >= period:
                assert f'momentum_{period}' in features, f"Momentum features should include momentum-{period}"
        
        # Verify rate of change
        for period in [10, 20, 50]:
            if len(df) >= period:
                assert f'roc_{period}' in features, f"Momentum features should include ROC-{period}"
    
    def test_pattern_features(self, feature_engineer, sample_candles):
        """Test pattern-based feature extraction"""
        df = pd.DataFrame(sample_candles)
        features = feature_engineer._pattern_features(df)
        
        # Verify pattern features
        assert 'doji' in features, "Pattern features should include doji"
        assert 'hammer' in features, "Pattern features should include hammer"
        assert 'shooting_star' in features, "Pattern features should include shooting star"
        assert 'engulfing' in features, "Pattern features should include engulfing"
        
        # Verify support/resistance
        for period in [20, 50]:
            if len(df) >= period:
                assert f'support_{period}' in features, f"Pattern features should include support-{period}"
                assert f'resistance_{period}' in features, f"Pattern features should include resistance-{period}"
    
    def test_microstructure_features(self, feature_engineer, sample_candles):
        """Test market microstructure feature extraction"""
        df = pd.DataFrame(sample_candles)
        features = feature_engineer._microstructure_features(df)
        
        # Verify microstructure features
        assert 'spread_proxy' in features, "Microstructure features should include spread proxy"
        
        # Verify efficiency and profile features
        for period in [20, 50]:
            if len(df) >= period:
                assert f'price_efficiency_{period}' in features, f"Microstructure features should include price efficiency-{period}"
                assert f'volume_profile_{period}' in features, f"Microstructure features should include volume profile-{period}"
    
    def test_edge_cases_feature_engineering(self, feature_engineer):
        """Test edge cases in feature engineering"""
        # Test with insufficient data
        with pytest.raises(ValueError):
            feature_engineer.engineer_features([])
        
        # Test with minimal data
        minimal_candles = [
            {'timestamp': datetime.now(), 'open': 100, 'high': 101, 'low': 99, 'close': 100.5, 'volume': 1000, 'symbol': 'BTC/USDT', 'timeframe': '1m'}
        ] * 50  # Just enough for basic features
        
        feature_set = feature_engineer.engineer_features(minimal_candles)
        assert len(feature_set.features) > 0, "Should generate some features even with minimal data"
    
    def test_performance_feature_engineering(self, feature_engineer, sample_candles):
        """Test performance of feature engineering"""
        import time
        
        # Time the feature engineering
        start_time = time.time()
        feature_set = feature_engineer.engineer_features(sample_candles)
        end_time = time.time()
        
        engineering_time = end_time - start_time
        
        # Should complete within reasonable time
        assert engineering_time < 2.0, f"Feature engineering took {engineering_time:.3f}s, should be < 2.0s"
        
        print(f"Feature engineering time: {engineering_time:.3f}s")
        print(f"Generated {len(feature_set.features)} features")

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
