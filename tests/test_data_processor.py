#!/usr/bin/env python3
"""
Unit tests for OptimizedDataProcessor
Tests data processing, caching, and performance optimization
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

from data.optimized_data_processor import OptimizedDataProcessor, OptimizedDataChunk


class TestOptimizedDataProcessor:
    """Test OptimizedDataProcessor functionality"""
    
    def setup_method(self):
        """Setup test environment"""
        self.processor = OptimizedDataProcessor(max_workers=2, chunk_size=100)
    
    def teardown_method(self):
        """Cleanup after tests"""
        # Clean up any cached data
        self.processor.data_cache.clear()
        self.processor.indicator_cache.clear()
        self.processor.pattern_cache.clear()
    
    def test_initialization(self):
        """Test OptimizedDataProcessor initialization"""
        assert self.processor.max_workers == 2
        assert self.processor.chunk_size == 100
        assert isinstance(self.processor.data_cache, dict)
        assert isinstance(self.processor.indicator_cache, dict)
        assert isinstance(self.processor.pattern_cache, dict)
        assert self.processor.cache_ttl == 300
        assert isinstance(self.processor.stats, dict)
        assert "total_chunks_processed" in self.processor.stats
    
    def test_create_test_data(self):
        """Test creation of test data"""
        data = self.processor._create_test_data(100)
        
        assert isinstance(data, pd.DataFrame)
        assert len(data) == 100
        assert all(col in data.columns for col in ['open', 'high', 'low', 'close', 'volume'])
        assert all(data['high'] >= data['low'])
        assert all(data['high'] >= data['open'])
        assert all(data['high'] >= data['close'])
        assert all(data['low'] <= data['open'])
        assert all(data['low'] <= data['close'])
    
    def test_validate_input_data_valid(self):
        """Test input data validation with valid data"""
        data = self.processor._create_test_data(50)
        
        is_valid = self.processor._validate_input_data(data, "BTC", "15m")
        
        assert is_valid == True
    
    def test_validate_input_data_invalid_symbol(self):
        """Test input data validation with invalid symbol"""
        data = self.processor._create_test_data(50)
        
        is_valid = self.processor._validate_input_data(data, "", "15m")
        
        assert is_valid == False
    
    def test_validate_input_data_invalid_timeframe(self):
        """Test input data validation with invalid timeframe"""
        data = self.processor._create_test_data(50)
        
        is_valid = self.processor._validate_input_data(data, "BTC", "invalid")
        
        assert is_valid == False
    
    def test_validate_input_data_empty_dataframe(self):
        """Test input data validation with empty dataframe"""
        empty_data = pd.DataFrame()
        
        is_valid = self.processor._validate_input_data(empty_data, "BTC", "15m")
        
        assert is_valid == False
    
    def test_validate_input_data_missing_columns(self):
        """Test input data validation with missing columns"""
        incomplete_data = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [105, 106, 107]
            # Missing low, close, volume
        })
        
        is_valid = self.processor._validate_input_data(incomplete_data, "BTC", "15m")
        
        assert is_valid == False
    
    def test_apply_data_preconditions_valid(self):
        """Test data preconditions with valid data"""
        data = self.processor._create_test_data(100)
        
        preconditions_met = self.processor._apply_data_preconditions(data)
        
        assert preconditions_met == True
    
    def test_apply_data_preconditions_insufficient_data(self):
        """Test data preconditions with insufficient data"""
        data = self.processor._create_test_data(10)  # Less than minimum required
        
        preconditions_met = self.processor._apply_data_preconditions(data)
        
        assert preconditions_met == False
    
    def test_create_processing_context(self):
        """Test processing context creation"""
        context = self.processor._create_processing_context("BTC", "15m")
        
        with context as ctx:
            assert ctx.symbol == "BTC"
            assert ctx.timeframe == "15m"
            assert hasattr(ctx, 'start_time')
            assert isinstance(ctx.start_time, datetime)
    
    def test_calculate_data_hash(self):
        """Test data hash calculation"""
        data1 = self.processor._create_test_data(50)
        data2 = self.processor._create_test_data(50)
        
        hash1 = self.processor._calculate_data_hash(data1)
        hash2 = self.processor._calculate_data_hash(data2)
        
        assert isinstance(hash1, str)
        assert isinstance(hash2, str)
        assert hash1 != hash2  # Different data should have different hashes
        
        # Same data should have same hash
        hash1_again = self.processor._calculate_data_hash(data1)
        assert hash1 == hash1_again
    
    def test_cache_management(self):
        """Test cache management functionality"""
        # Test cache cleanup
        self.processor.data_cache = {
            "old_key": {"timestamp": time.time() - 400, "data": "old_data"},
            "new_key": {"timestamp": time.time(), "data": "new_data"}
        }
        
        self.processor._cleanup_expired_cache()
        
        # Old key should be removed, new key should remain
        assert "old_key" not in self.processor.data_cache
        assert "new_key" in self.processor.data_cache
    
    def test_get_cache_entry_valid(self):
        """Test getting valid cache entry"""
        cache_key = "test_key"
        test_data = {"test": "data"}
        
        # Add entry to cache
        self.processor.data_cache[cache_key] = {
            "timestamp": time.time(),
            "data": test_data
        }
        
        cached_data = self.processor._get_cache_entry(cache_key)
        
        assert cached_data == test_data
    
    def test_get_cache_entry_expired(self):
        """Test getting expired cache entry"""
        cache_key = "test_key"
        test_data = {"test": "data"}
        
        # Add expired entry to cache
        self.processor.data_cache[cache_key] = {
            "timestamp": time.time() - 400,  # Expired
            "data": test_data
        }
        
        cached_data = self.processor._get_cache_entry(cache_key)
        
        assert cached_data is None
        # Expired entry should be removed
        assert cache_key not in self.processor.data_cache
    
    def test_get_cache_entry_nonexistent(self):
        """Test getting nonexistent cache entry"""
        cached_data = self.processor._get_cache_entry("nonexistent_key")
        
        assert cached_data is None
    
    def test_set_cache_entry(self):
        """Test setting cache entry"""
        cache_key = "test_key"
        test_data = {"test": "data"}
        
        self.processor._set_cache_entry(cache_key, test_data)
        
        assert cache_key in self.processor.data_cache
        assert self.processor.data_cache[cache_key]["data"] == test_data
        assert "timestamp" in self.processor.data_cache[cache_key]
    
    def test_calculate_technical_indicators(self):
        """Test technical indicators calculation"""
        data = self.processor._create_test_data(100)
        
        indicators = self.processor._calculate_technical_indicators(data)
        
        assert isinstance(indicators, dict)
        assert "sma_20" in indicators
        assert "ema_12" in indicators
        assert "rsi" in indicators
        assert "macd" in indicators
        assert "bollinger_bands" in indicators
        
        # Check that indicators are numpy arrays
        for indicator_name, indicator_data in indicators.items():
            assert isinstance(indicator_data, np.ndarray)
    
    def test_detect_patterns(self):
        """Test pattern detection"""
        data = self.processor._create_test_data(100)
        indicators = self.processor._calculate_technical_indicators(data)
        
        patterns = self.processor._detect_patterns(data, indicators)
        
        assert isinstance(patterns, list)
        # Patterns should be detected (even if empty)
        assert len(patterns) >= 0
    
    @pytest.mark.asyncio
    async def test_process_data_chunk_optimized_success(self):
        """Test successful data chunk processing"""
        data = self.processor._create_test_data(100)
        
        result = await self.processor.process_data_chunk_optimized(
            symbol="BTC",
            data=data,
            timeframe="15m"
        )
        
        assert isinstance(result, OptimizedDataChunk)
        assert result.symbol == "BTC"
        assert result.timeframe == "15m"
        assert result.data.equals(data)
        assert isinstance(result.indicators, dict)
        assert isinstance(result.patterns, list)
        assert isinstance(result.timestamp, datetime)
        assert result.processing_time_ms > 0
    
    @pytest.mark.asyncio
    async def test_process_data_chunk_optimized_invalid_input(self):
        """Test data chunk processing with invalid input"""
        invalid_data = pd.DataFrame()  # Empty dataframe
        
        result = await self.processor.process_data_chunk_optimized(
            symbol="BTC",
            data=invalid_data,
            timeframe="15m"
        )
        
        assert result is None
    
    @pytest.mark.asyncio
    async def test_process_data_chunk_optimized_cache_hit(self):
        """Test data chunk processing with cache hit"""
        data = self.processor._create_test_data(50)
        
        # Process first time
        result1 = await self.processor.process_data_chunk_optimized(
            symbol="BTC",
            data=data,
            timeframe="15m"
        )
        
        # Process same data again (should hit cache)
        result2 = await self.processor.process_data_chunk_optimized(
            symbol="BTC",
            data=data,
            timeframe="15m"
        )
        
        assert result1 is not None
        assert result2 is not None
        assert result1.symbol == result2.symbol
        assert result1.timeframe == result2.timeframe
        # Cache hit should be faster
        assert result2.processing_time_ms <= result1.processing_time_ms
    
    def test_get_processing_stats(self):
        """Test getting processing statistics"""
        # Update some stats
        self.processor.stats["total_chunks_processed"] = 10
        self.processor.stats["cache_hits"] = 5
        self.processor.stats["avg_processing_time_ms"] = 25.5
        
        stats = self.processor.get_processing_stats()
        
        assert isinstance(stats, dict)
        assert stats["total_chunks_processed"] == 10
        assert stats["cache_hits"] == 5
        assert stats["avg_processing_time_ms"] == 25.5
        assert "cache_hit_rate" in stats
        assert "total_processing_time_ms" in stats
    
    def test_reset_stats(self):
        """Test statistics reset"""
        # Update some stats
        self.processor.stats["total_chunks_processed"] = 10
        self.processor.stats["cache_hits"] = 5
        
        self.processor.reset_stats()
        
        assert self.processor.stats["total_chunks_processed"] == 0
        assert self.processor.stats["cache_hits"] == 0
        assert self.processor.stats["avg_processing_time_ms"] == 0.0
    
    def test_cleanup_resources(self):
        """Test resource cleanup"""
        # Add some test data to caches
        self.processor.data_cache["test"] = {"data": "test"}
        self.processor.indicator_cache["test"] = {"data": "test"}
        self.processor.pattern_cache["test"] = {"data": "test"}
        
        self.processor.cleanup_resources()
        
        # Caches should be cleared
        assert len(self.processor.data_cache) == 0
        assert len(self.processor.indicator_cache) == 0
        assert len(self.processor.pattern_cache) == 0


class TestOptimizedDataChunk:
    """Test OptimizedDataChunk dataclass"""
    
    def test_optimized_data_chunk_creation(self):
        """Test OptimizedDataChunk creation"""
        data = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [105, 106, 107],
            'low': [95, 96, 97],
            'close': [103, 104, 105],
            'volume': [1000, 1100, 1200]
        })
        
        indicators = {"sma_20": np.array([100.5, 101.0, 101.5])}
        patterns = [{"pattern": "doji", "confidence": 0.8}]
        timestamp = datetime.now()
        
        chunk = OptimizedDataChunk(
            symbol="BTC",
            timeframe="15m",
            data=data,
            indicators=indicators,
            patterns=patterns,
            timestamp=timestamp,
            processing_time_ms=25.5,
            cache_hit=True,
            data_hash="abc123"
        )
        
        assert chunk.symbol == "BTC"
        assert chunk.timeframe == "15m"
        assert chunk.data.equals(data)
        assert chunk.indicators == indicators
        assert chunk.patterns == patterns
        assert chunk.timestamp == timestamp
        assert chunk.processing_time_ms == 25.5
        assert chunk.cache_hit == True
        assert chunk.data_hash == "abc123"
    
    def test_optimized_data_chunk_defaults(self):
        """Test OptimizedDataChunk with default values"""
        data = pd.DataFrame({'open': [100], 'high': [105], 'low': [95], 'close': [103], 'volume': [1000]})
        
        chunk = OptimizedDataChunk(
            symbol="BTC",
            timeframe="15m",
            data=data,
            indicators={},
            patterns=[],
            timestamp=datetime.now()
        )
        
        assert chunk.processing_time_ms == 0.0
        assert chunk.cache_hit == False
        assert chunk.data_hash == ""


class TestOptimizedDataProcessorIntegration:
    """Integration tests for OptimizedDataProcessor"""
    
    @pytest.mark.asyncio
    async def test_full_data_processing_workflow(self):
        """Test complete data processing workflow"""
        processor = OptimizedDataProcessor(max_workers=2, chunk_size=50)
        
        try:
            # Create test data
            data = processor._create_test_data(100)
            
            # Process data chunk
            result = await processor.process_data_chunk_optimized(
                symbol="BTC",
                data=data,
                timeframe="15m"
            )
            
            assert result is not None
            assert result.symbol == "BTC"
            assert result.timeframe == "15m"
            assert len(result.data) == 100
            assert len(result.indicators) > 0
            assert isinstance(result.patterns, list)
            assert result.processing_time_ms > 0
            
            # Check that stats were updated
            stats = processor.get_processing_stats()
            assert stats["total_chunks_processed"] == 1
            assert stats["data_points_processed"] == 100
            
            # Process same data again to test caching
            result2 = await processor.process_data_chunk_optimized(
                symbol="BTC",
                data=data,
                timeframe="15m"
            )
            
            assert result2 is not None
            assert result2.cache_hit == True
            
            # Check cache hit rate
            stats = processor.get_processing_stats()
            assert stats["cache_hits"] == 1
            
        finally:
            processor.cleanup_resources()
    
    @pytest.mark.asyncio
    async def test_concurrent_data_processing(self):
        """Test concurrent data processing"""
        processor = OptimizedDataProcessor(max_workers=4, chunk_size=25)
        
        try:
            # Create multiple test datasets
            datasets = [
                processor._create_test_data(50) for _ in range(4)
            ]
            
            # Process all datasets concurrently
            tasks = []
            for i, data in enumerate(datasets):
                task = processor.process_data_chunk_optimized(
                    symbol=f"SYMBOL_{i}",
                    data=data,
                    timeframe="15m"
                )
                tasks.append(task)
            
            results = await asyncio.gather(*tasks)
            
            # All tasks should complete successfully
            assert len(results) == 4
            for result in results:
                assert result is not None
                assert isinstance(result, OptimizedDataChunk)
            
            # Check stats
            stats = processor.get_processing_stats()
            assert stats["total_chunks_processed"] == 4
            assert stats["data_points_processed"] == 200  # 4 * 50
            
        finally:
            processor.cleanup_resources()
    
    @pytest.mark.asyncio
    async def test_error_handling_in_processing(self):
        """Test error handling during processing"""
        processor = OptimizedDataProcessor(max_workers=1, chunk_size=10)
        
        try:
            # Test with invalid data
            invalid_data = pd.DataFrame({
                'open': [100, 101],
                'high': [105, 106]
                # Missing required columns
            })
            
            result = await processor.process_data_chunk_optimized(
                symbol="BTC",
                data=invalid_data,
                timeframe="15m"
            )
            
            # Should return None for invalid data
            assert result is None
            
            # Test with empty data
            empty_data = pd.DataFrame()
            
            result = await processor.process_data_chunk_optimized(
                symbol="BTC",
                data=empty_data,
                timeframe="15m"
            )
            
            # Should return None for empty data
            assert result is None
            
        finally:
            processor.cleanup_resources()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
