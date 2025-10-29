#!/usr/bin/env python3
"""
Unit tests for Priority2FeatureEngineering
Tests feature extraction, caching, and statistical calculations
"""

import pytest
import asyncio
import logging
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import sys
import os
import time

# Add backend to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ..src.ai.priority2_feature_engineering import Priority2FeatureEngineering, CacheEntry


class TestPriority2FeatureEngineering:
    """Test Priority2FeatureEngineering functionality"""
    
    def setup_method(self):
        """Setup test environment"""
        self.fe = Priority2FeatureEngineering()
    
    def teardown_method(self):
        """Cleanup after tests"""
        # Clean up local cache
        self.fe._local_cache.clear()
    
    def test_initialization(self):
        """Test Priority2FeatureEngineering initialization"""
        assert isinstance(self.fe.cache_dir, type(self.fe.cache_dir))
        assert isinstance(self.fe.stats, dict)
        assert "cache_hits" in self.fe.stats
        assert "cache_misses" in self.fe.stats
        assert "extraction_times" in self.fe.stats
        assert "total_requests" in self.fe.stats
        assert isinstance(self.fe._local_cache, dict)
        assert self.fe._cache_cleanup_interval == 300
    
    def test_calculate_skewness_valid_data(self):
        """Test skewness calculation with valid data"""
        # Create test data
        data = np.random.randn(100, 10)
        
        result = self.fe._calculate_skewness(data)
        
        assert isinstance(result, np.ndarray)
        assert len(result) == data.shape[0]
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))
    
    def test_calculate_skewness_empty_data(self):
        """Test skewness calculation with empty data"""
        empty_data = np.array([])
        
        result = self.fe._calculate_skewness(empty_data)
        
        assert isinstance(result, np.ndarray)
        assert len(result) == 0
    
    def test_calculate_skewness_single_row(self):
        """Test skewness calculation with single row data"""
        single_row_data = np.random.randn(1, 10)
        
        result = self.fe._calculate_skewness(single_row_data)
        
        assert isinstance(result, np.ndarray)
        assert len(result) == 1
        assert not np.isnan(result[0])
        assert not np.isinf(result[0])
    
    def test_calculate_skewness_caching(self):
        """Test skewness calculation caching"""
        data = np.random.randn(50, 10)
        
        # First calculation
        result1 = self.fe._calculate_skewness(data)
        
        # Second calculation with same data (should use cache)
        result2 = self.fe._calculate_skewness(data)
        
        # Results should be identical
        np.testing.assert_array_almost_equal(result1, result2)
        
        # Check that cache was used
        assert len(self.fe._local_cache) > 0
    
    def test_calculate_kurtosis_valid_data(self):
        """Test kurtosis calculation with valid data"""
        # Create test data
        data = np.random.randn(100, 10)
        
        result = self.fe._calculate_kurtosis(data)
        
        assert isinstance(result, np.ndarray)
        assert len(result) == data.shape[0]
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))
    
    def test_calculate_kurtosis_empty_data(self):
        """Test kurtosis calculation with empty data"""
        empty_data = np.array([])
        
        result = self.fe._calculate_kurtosis(empty_data)
        
        assert isinstance(result, np.ndarray)
        assert len(result) == 0
    
    def test_calculate_kurtosis_caching(self):
        """Test kurtosis calculation caching"""
        data = np.random.randn(50, 10)
        
        # First calculation
        result1 = self.fe._calculate_kurtosis(data)
        
        # Second calculation with same data (should use cache)
        result2 = self.fe._calculate_kurtosis(data)
        
        # Results should be identical
        np.testing.assert_array_almost_equal(result1, result2)
        
        # Check that cache was used
        assert len(self.fe._local_cache) > 0
    
    def test_calculate_rolling_statistics(self):
        """Test rolling statistics calculation"""
        # Create test data
        data = np.random.randn(100, 10)
        window_size = 20
        
        result = self.fe._calculate_rolling_statistics(data, window_size)
        
        assert isinstance(result, dict)
        assert "rolling_mean" in result
        assert "rolling_std" in result
        assert "rolling_skew" in result
        assert "rolling_kurt" in result
        
        # Check shapes
        assert result["rolling_mean"].shape == data.shape
        assert result["rolling_std"].shape == data.shape
        assert result["rolling_skew"].shape == data.shape
        assert result["rolling_kurt"].shape == data.shape
    
    def test_calculate_rolling_statistics_small_window(self):
        """Test rolling statistics with small window"""
        data = np.random.randn(10, 5)
        window_size = 5
        
        result = self.fe._calculate_rolling_statistics(data, window_size)
        
        assert isinstance(result, dict)
        assert "rolling_mean" in result
        assert "rolling_std" in result
        assert "rolling_skew" in result
        assert "rolling_kurt" in result
    
    def test_extract_technical_features(self):
        """Test technical feature extraction"""
        # Create test data
        data = pd.DataFrame({
            'open': np.random.rand(100) * 100,
            'high': np.random.rand(100) * 100,
            'low': np.random.rand(100) * 100,
            'close': np.random.rand(100) * 100,
            'volume': np.random.rand(100) * 1000
        })
        
        features = self.fe._extract_technical_features(data)
        
        assert isinstance(features, dict)
        assert "price_features" in features
        assert "volume_features" in features
        assert "volatility_features" in features
        assert "momentum_features" in features
        
        # Check that features are numpy arrays
        for feature_type, feature_data in features.items():
            assert isinstance(feature_data, dict)
            for feature_name, feature_values in feature_data.items():
                assert isinstance(feature_values, np.ndarray)
    
    def test_extract_technical_features_empty_data(self):
        """Test technical feature extraction with empty data"""
        empty_data = pd.DataFrame()
        
        features = self.fe._extract_technical_features(empty_data)
        
        assert isinstance(features, dict)
        assert "price_features" in features
        assert "volume_features" in features
        assert "volatility_features" in features
        assert "momentum_features" in features
    
    def test_extract_statistical_features(self):
        """Test statistical feature extraction"""
        # Create test data
        data = np.random.randn(100, 10)
        
        features = self.fe._extract_statistical_features(data)
        
        assert isinstance(features, dict)
        assert "mean" in features
        assert "std" in features
        assert "skewness" in features
        assert "kurtosis" in features
        assert "min" in features
        assert "max" in features
        assert "median" in features
        
        # Check that features are numpy arrays
        for feature_name, feature_values in features.items():
            assert isinstance(feature_values, np.ndarray)
            assert len(feature_values) == data.shape[0]
    
    def test_extract_statistical_features_empty_data(self):
        """Test statistical feature extraction with empty data"""
        empty_data = np.array([])
        
        features = self.fe._extract_statistical_features(empty_data)
        
        assert isinstance(features, dict)
        assert "mean" in features
        assert "std" in features
        assert "skewness" in features
        assert "kurtosis" in features
        assert "min" in features
        assert "max" in features
        assert "median" in features
    
    def test_extract_frequency_domain_features(self):
        """Test frequency domain feature extraction"""
        # Create test data
        data = np.random.randn(100, 10)
        
        features = self.fe._extract_frequency_domain_features(data)
        
        assert isinstance(features, dict)
        assert "fft_magnitude" in features
        assert "fft_phase" in features
        assert "power_spectral_density" in features
        
        # Check that features are numpy arrays
        for feature_name, feature_values in features.items():
            assert isinstance(feature_values, np.ndarray)
    
    def test_extract_frequency_domain_features_empty_data(self):
        """Test frequency domain feature extraction with empty data"""
        empty_data = np.array([])
        
        features = self.fe._extract_frequency_domain_features(empty_data)
        
        assert isinstance(features, dict)
        assert "fft_magnitude" in features
        assert "fft_phase" in features
        assert "power_spectral_density" in features
    
    def test_extract_pattern_features(self):
        """Test pattern feature extraction"""
        # Create test data
        data = pd.DataFrame({
            'open': np.random.rand(100) * 100,
            'high': np.random.rand(100) * 100,
            'low': np.random.rand(100) * 100,
            'close': np.random.rand(100) * 100,
            'volume': np.random.rand(100) * 1000
        })
        
        features = self.fe._extract_pattern_features(data)
        
        assert isinstance(features, dict)
        assert "candlestick_patterns" in features
        assert "support_resistance" in features
        assert "trend_patterns" in features
        
        # Check that features are numpy arrays
        for feature_name, feature_values in features.items():
            assert isinstance(feature_values, np.ndarray)
    
    def test_extract_pattern_features_empty_data(self):
        """Test pattern feature extraction with empty data"""
        empty_data = pd.DataFrame()
        
        features = self.fe._extract_pattern_features(empty_data)
        
        assert isinstance(features, dict)
        assert "candlestick_patterns" in features
        assert "support_resistance" in features
        assert "trend_patterns" in features
    
    @pytest.mark.asyncio
    async def test_extract_features_success(self):
        """Test successful feature extraction"""
        # Create test data
        data = pd.DataFrame({
            'open': np.random.rand(100) * 100,
            'high': np.random.rand(100) * 100,
            'low': np.random.rand(100) * 100,
            'close': np.random.rand(100) * 100,
            'volume': np.random.rand(100) * 1000
        })
        
        features = await self.fe.extract_features(data, "BTC", "15m")
        
        assert isinstance(features, dict)
        assert "technical_features" in features
        assert "statistical_features" in features
        assert "frequency_features" in features
        assert "pattern_features" in features
        assert "metadata" in features
        
        # Check metadata
        metadata = features["metadata"]
        assert metadata["symbol"] == "BTC"
        assert metadata["timeframe"] == "15m"
        assert "extraction_time" in metadata
        assert "feature_count" in metadata
    
    @pytest.mark.asyncio
    async def test_extract_features_empty_data(self):
        """Test feature extraction with empty data"""
        empty_data = pd.DataFrame()
        
        features = await self.fe.extract_features(empty_data, "BTC", "15m")
        
        assert isinstance(features, dict)
        assert "technical_features" in features
        assert "statistical_features" in features
        assert "frequency_features" in features
        assert "pattern_features" in features
        assert "metadata" in features
    
    @pytest.mark.asyncio
    async def test_extract_features_caching(self):
        """Test feature extraction caching"""
        # Create test data
        data = pd.DataFrame({
            'open': np.random.rand(50) * 100,
            'high': np.random.rand(50) * 100,
            'low': np.random.rand(50) * 100,
            'close': np.random.rand(50) * 100,
            'volume': np.random.rand(50) * 1000
        })
        
        # First extraction
        features1 = await self.fe.extract_features(data, "BTC", "15m")
        
        # Second extraction with same data (should use cache)
        features2 = await self.fe.extract_features(data, "BTC", "15m")
        
        # Results should be identical
        assert features1["metadata"]["symbol"] == features2["metadata"]["symbol"]
        assert features1["metadata"]["timeframe"] == features2["metadata"]["timeframe"]
        
        # Check that cache was used
        assert self.fe.stats["cache_hits"] > 0
    
    def test_get_extraction_stats(self):
        """Test getting extraction statistics"""
        # Update some stats
        self.fe.stats["total_requests"] = 10
        self.fe.stats["cache_hits"] = 5
        self.fe.stats["cache_misses"] = 5
        self.fe.stats["extraction_times"] = [0.1, 0.2, 0.3]
        
        stats = self.fe.get_extraction_stats()
        
        assert isinstance(stats, dict)
        assert stats["total_requests"] == 10
        assert stats["cache_hits"] == 5
        assert stats["cache_misses"] == 5
        assert "cache_hit_rate" in stats
        assert "avg_extraction_time" in stats
        assert "min_extraction_time" in stats
        assert "max_extraction_time" in stats
    
    def test_reset_stats(self):
        """Test statistics reset"""
        # Update some stats
        self.fe.stats["total_requests"] = 10
        self.fe.stats["cache_hits"] = 5
        self.fe.stats["extraction_times"] = [0.1, 0.2, 0.3]
        
        self.fe.reset_stats()
        
        assert self.fe.stats["total_requests"] == 0
        assert self.fe.stats["cache_hits"] == 0
        assert self.fe.stats["cache_misses"] == 0
        assert self.fe.stats["extraction_times"] == []
    
    def test_cleanup_expired_cache(self):
        """Test cleanup of expired cache entries"""
        # Add expired and valid cache entries
        expired_entry = CacheEntry("expired_data", datetime.now() - timedelta(hours=2))
        valid_entry = CacheEntry("valid_data", datetime.now())
        
        self.fe._local_cache["expired_key"] = expired_entry
        self.fe._local_cache["valid_key"] = valid_entry
        
        # Run cleanup
        self.fe._cleanup_expired_cache()
        
        # Expired entry should be removed, valid entry should remain
        assert "expired_key" not in self.fe._local_cache
        assert "valid_key" in self.fe._local_cache


class TestCacheEntry:
    """Test CacheEntry dataclass"""
    
    def test_cache_entry_creation(self):
        """Test CacheEntry creation"""
        data = {"test": "data"}
        timestamp = datetime.now()
        ttl = 1800  # 30 minutes
        
        entry = CacheEntry(data, timestamp, ttl)
        
        assert entry.data == data
        assert entry.timestamp == timestamp
        assert entry.ttl == ttl
    
    def test_cache_entry_default_ttl(self):
        """Test CacheEntry with default TTL"""
        data = {"test": "data"}
        timestamp = datetime.now()
        
        entry = CacheEntry(data, timestamp)
        
        assert entry.data == data
        assert entry.timestamp == timestamp
        assert entry.ttl == 3600  # Default 1 hour
    
    def test_cache_entry_is_expired_fresh(self):
        """Test CacheEntry expiration check with fresh entry"""
        data = {"test": "data"}
        timestamp = datetime.now()
        
        entry = CacheEntry(data, timestamp, ttl=3600)
        
        assert not entry.is_expired()
    
    def test_cache_entry_is_expired_old(self):
        """Test CacheEntry expiration check with old entry"""
        data = {"test": "data"}
        timestamp = datetime.now() - timedelta(hours=2)
        
        entry = CacheEntry(data, timestamp, ttl=3600)
        
        assert entry.is_expired()
    
    def test_cache_entry_is_expired_boundary(self):
        """Test CacheEntry expiration check at boundary"""
        data = {"test": "data"}
        timestamp = datetime.now() - timedelta(hours=1)
        
        entry = CacheEntry(data, timestamp, ttl=3600)
        
        # Should be expired (1 hour ago + 1 hour TTL = expired)
        assert entry.is_expired()


class TestPriority2FeatureEngineeringIntegration:
    """Integration tests for Priority2FeatureEngineering"""
    
    @pytest.mark.asyncio
    async def test_full_feature_extraction_workflow(self):
        """Test complete feature extraction workflow"""
        fe = Priority2FeatureEngineering()
        
        try:
            # Create test data
            data = pd.DataFrame({
                'open': np.random.rand(100) * 100,
                'high': np.random.rand(100) * 100,
                'low': np.random.rand(100) * 100,
                'close': np.random.rand(100) * 100,
                'volume': np.random.rand(100) * 1000
            })
            
            # Extract features
            features = await fe.extract_features(data, "BTC", "15m")
            
            assert isinstance(features, dict)
            assert "technical_features" in features
            assert "statistical_features" in features
            assert "frequency_features" in features
            assert "pattern_features" in features
            assert "metadata" in features
            
            # Check metadata
            metadata = features["metadata"]
            assert metadata["symbol"] == "BTC"
            assert metadata["timeframe"] == "15m"
            assert metadata["feature_count"] > 0
            
            # Check that stats were updated
            stats = fe.get_extraction_stats()
            assert stats["total_requests"] == 1
            assert stats["cache_misses"] == 1
            
            # Extract same features again to test caching
            features2 = await fe.extract_features(data, "BTC", "15m")
            
            assert isinstance(features2, dict)
            assert features2["metadata"]["symbol"] == features["metadata"]["symbol"]
            
            # Check that cache was used
            stats = fe.get_extraction_stats()
            assert stats["cache_hits"] == 1
            
        finally:
            fe._local_cache.clear()
    
    @pytest.mark.asyncio
    async def test_concurrent_feature_extraction(self):
        """Test concurrent feature extraction"""
        fe = Priority2FeatureEngineering()
        
        try:
            # Create multiple test datasets
            datasets = []
            for i in range(3):
                data = pd.DataFrame({
                    'open': np.random.rand(50) * 100,
                    'high': np.random.rand(50) * 100,
                    'low': np.random.rand(50) * 100,
                    'close': np.random.rand(50) * 100,
                    'volume': np.random.rand(50) * 1000
                })
                datasets.append(data)
            
            # Extract features concurrently
            tasks = []
            for i, data in enumerate(datasets):
                task = fe.extract_features(data, f"SYMBOL_{i}", "15m")
                tasks.append(task)
            
            results = await asyncio.gather(*tasks)
            
            # All tasks should complete successfully
            assert len(results) == 3
            for result in results:
                assert isinstance(result, dict)
                assert "metadata" in result
                assert "technical_features" in result
            
            # Check stats
            stats = fe.get_extraction_stats()
            assert stats["total_requests"] == 3
            
        finally:
            fe._local_cache.clear()
    
    @pytest.mark.asyncio
    async def test_error_handling_in_extraction(self):
        """Test error handling during feature extraction"""
        fe = Priority2FeatureEngineering()
        
        try:
            # Test with empty data
            empty_data = pd.DataFrame()
            
            features = await fe.extract_features(empty_data, "BTC", "15m")
            
            # Should still return a valid structure
            assert isinstance(features, dict)
            assert "metadata" in features
            assert "technical_features" in features
            assert "statistical_features" in features
            assert "frequency_features" in features
            assert "pattern_features" in features
            
            # Test with invalid data
            invalid_data = pd.DataFrame({
                'open': [100, 101],
                'high': [105, 106]
                # Missing required columns
            })
            
            features = await fe.extract_features(invalid_data, "BTC", "15m")
            
            # Should still return a valid structure
            assert isinstance(features, dict)
            assert "metadata" in features
            
        finally:
            fe._local_cache.clear()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
