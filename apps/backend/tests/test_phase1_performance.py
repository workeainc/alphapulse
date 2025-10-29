"""
Phase 1 Performance Tests for Enhanced Pattern Detection
Tests vectorized detection, sliding windows, and async parallelization
"""

import pytest
import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from typing import List, Dict
import time
import logging

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from backend.strategies.ultra_fast_pattern_detector import UltraFastPatternDetector, vectorized_doji_detection, vectorized_hammer_detection, vectorized_engulfing_detection
from backend.strategies.sliding_window_buffer import SlidingWindowBuffer, AsyncSlidingWindowBuffer
from backend.strategies.async_pattern_detector import AsyncPatternDetector, PatternDetectionOrchestrator

logger = logging.getLogger(__name__)

class TestPhase1Performance:
    """Test suite for Phase 1 performance optimizations"""
    
    @pytest.fixture
    def sample_candles(self):
        """Generate sample candlestick data for testing"""
        np.random.seed(42)
        n_candles = 1000
        
        # Generate realistic OHLCV data
        base_price = 50000.0
        prices = []
        opens = []
        highs = []
        lows = []
        closes = []
        volumes = []
        timestamps = []
        
        current_price = base_price
        for i in range(n_candles):
            # Generate price movement
            price_change = np.random.normal(0, 0.02)  # 2% volatility
            current_price *= (1 + price_change)
            
            # Generate OHLC
            open_price = current_price
            high_price = open_price * (1 + abs(np.random.normal(0, 0.01)))
            low_price = open_price * (1 - abs(np.random.normal(0, 0.01)))
            close_price = np.random.uniform(low_price, high_price)
            
            # Ensure proper OHLC relationship
            high_price = max(open_price, close_price, high_price)
            low_price = min(open_price, close_price, low_price)
            
            # Generate volume
            volume = np.random.uniform(1000, 100000)
            
            # Generate timestamp
            timestamp = datetime.now(timezone.utc) - timedelta(minutes=n_candles-i)
            
            opens.append(open_price)
            highs.append(high_price)
            lows.append(low_price)
            closes.append(close_price)
            volumes.append(volume)
            timestamps.append(timestamp)
            
            current_price = close_price
        
        return [
            {
                'open': opens[i],
                'high': highs[i],
                'low': lows[i],
                'close': closes[i],
                'volume': volumes[i],
                'timestamp': timestamps[i]
            }
            for i in range(n_candles)
        ]
    
    @pytest.fixture
    def ultra_fast_detector(self):
        """Initialize ultra-fast pattern detector"""
        return UltraFastPatternDetector()
    
    @pytest.fixture
    def sliding_buffer(self):
        """Initialize sliding window buffer"""
        return SlidingWindowBuffer(max_size=1000)
    
    @pytest.fixture
    def async_detector(self):
        """Initialize async pattern detector"""
        return AsyncPatternDetector(max_concurrent_detections=5)
    
    def test_vectorized_detection_performance(self, sample_candles):
        """Test vectorized pattern detection performance"""
        # Extract OHLCV arrays
        opens = np.array([c['open'] for c in sample_candles])
        highs = np.array([c['high'] for c in sample_candles])
        lows = np.array([c['low'] for c in sample_candles])
        closes = np.array([c['close'] for c in sample_candles])
        
        # Test vectorized doji detection
        start_time = time.time()
        doji_results = vectorized_doji_detection(opens, highs, lows, closes)
        doji_time = time.time() - start_time
        
        # Test vectorized hammer detection
        start_time = time.time()
        hammer_results = vectorized_hammer_detection(opens, highs, lows, closes)
        hammer_time = time.time() - start_time
        
        # Test vectorized engulfing detection
        start_time = time.time()
        engulfing_results = vectorized_engulfing_detection(opens, highs, lows, closes)
        engulfing_time = time.time() - start_time
        
        # Performance assertions
        assert doji_time < 0.1, f"Doji detection took {doji_time:.4f}s, expected < 0.1s"
        assert hammer_time < 0.1, f"Hammer detection took {hammer_time:.4f}s, expected < 0.1s"
        assert engulfing_time < 0.1, f"Engulfing detection took {engulfing_time:.4f}s, expected < 0.1s"
        
        # Verify results
        assert len(doji_results) == len(sample_candles)
        assert len(hammer_results) == len(sample_candles)
        assert len(engulfing_results) == len(sample_candles)
        
        logger.info(f"✅ Vectorized detection performance: Doji={doji_time:.4f}s, Hammer={hammer_time:.4f}s, Engulfing={engulfing_time:.4f}s")
    
    def test_sliding_window_buffer_performance(self, sliding_buffer, sample_candles):
        """Test sliding window buffer performance"""
        symbol = "BTC/USDT"
        timeframe = "1m"
        
        # Test adding candles
        start_time = time.time()
        for candle in sample_candles:
            sliding_buffer.add_candle(symbol, timeframe, candle)
        add_time = time.time() - start_time
        
        # Test retrieving candles
        start_time = time.time()
        recent_candles = sliding_buffer.get_recent_candles(symbol, timeframe, 100)
        retrieve_time = time.time() - start_time
        
        # Test getting OHLCV arrays
        start_time = time.time()
        opens, highs, lows, closes, volumes = sliding_buffer.get_ohlcv_arrays(symbol, timeframe)
        array_time = time.time() - start_time
        
        # Performance assertions
        assert add_time < 1.0, f"Adding candles took {add_time:.4f}s, expected < 1.0s"
        assert retrieve_time < 0.01, f"Retrieving candles took {retrieve_time:.4f}s, expected < 0.01s"
        assert array_time < 0.01, f"Getting OHLCV arrays took {array_time:.4f}s, expected < 0.01s"
        
        # Verify results
        assert len(recent_candles) == 100
        assert len(opens) == len(sample_candles)
        assert len(highs) == len(sample_candles)
        assert len(lows) == len(sample_candles)
        assert len(closes) == len(sample_candles)
        assert len(volumes) == len(sample_candles)
        
        # Test buffer stats
        stats = sliding_buffer.get_buffer_stats(symbol, timeframe)
        assert stats['size'] == len(sample_candles)
        assert stats['time_span_hours'] > 0
        
        logger.info(f"✅ Sliding window buffer performance: Add={add_time:.4f}s, Retrieve={retrieve_time:.4f}s, Arrays={array_time:.4f}s")
    
    @pytest.mark.asyncio
    async def test_async_detection_performance(self, async_detector, sample_candles):
        """Test async pattern detection performance"""
        symbol = "BTC/USDT"
        timeframes = ['1m', '5m', '15m']
        
        # Prepare candles data for multiple timeframes
        candles_data = {}
        for timeframe in timeframes:
            # Simulate different timeframe data
            timeframe_candles = sample_candles[::int(timeframe.replace('m', ''))]  # Sample every N minutes
            candles_data[timeframe] = timeframe_candles[:100]  # Limit to 100 candles per timeframe
        
        # Test multi-timeframe detection
        start_time = time.time()
        results = await async_detector.detect_patterns_multi_timeframe(
            symbol, timeframes, candles_data
        )
        detection_time = time.time() - start_time
        
        # Performance assertions
        assert detection_time < 2.0, f"Multi-timeframe detection took {detection_time:.4f}s, expected < 2.0s"
        
        # Verify results
        assert len(results) == len(timeframes)
        for result in results:
            assert result.symbol == symbol
            assert result.timeframe in timeframes
            assert result.detection_latency_ms < 1000  # Less than 1 second per timeframe
            assert result.confidence_score >= 0.0
            assert result.confidence_score <= 1.0
        
        # Test performance summary
        summary = await async_detector.get_performance_summary()
        assert 'async_detector_stats' in summary
        assert 'buffer_stats' in summary
        assert 'ultra_fast_detector_stats' in summary
        
        logger.info(f"✅ Async detection performance: Multi-timeframe={detection_time:.4f}s")
    
    @pytest.mark.asyncio
    async def test_bulk_detection_performance(self, sample_candles):
        """Test bulk pattern detection across multiple symbols"""
        orchestrator = PatternDetectionOrchestrator(max_concurrent_symbols=3)
        
        # Prepare data for multiple symbols
        symbols_data = {}
        symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'ADA/USDT', 'SOL/USDT']
        timeframes = ['1m', '5m', '15m']
        
        for symbol in symbols:
            candles_data = {}
            for timeframe in timeframes:
                # Simulate different data for each symbol
                timeframe_candles = sample_candles[::int(timeframe.replace('m', ''))]
                candles_data[timeframe] = timeframe_candles[:50]  # 50 candles per timeframe
            symbols_data[symbol] = candles_data
        
        # Test bulk detection
        start_time = time.time()
        results = await orchestrator.detect_patterns_bulk(symbols_data, timeframes)
        bulk_time = time.time() - start_time
        
        # Performance assertions
        assert bulk_time < 5.0, f"Bulk detection took {bulk_time:.4f}s, expected < 5.0s"
        
        # Verify results
        assert len(results) == len(symbols)
        for symbol, symbol_results in results.items():
            assert symbol in symbols
            assert len(symbol_results) == len(timeframes)
        
        logger.info(f"✅ Bulk detection performance: {len(symbols)} symbols in {bulk_time:.4f}s")
    
    def test_memory_efficiency(self, sliding_buffer, sample_candles):
        """Test memory efficiency of sliding window buffer"""
        symbol = "BTC/USDT"
        timeframe = "1m"
        
        # Add candles and monitor memory usage
        initial_stats = sliding_buffer.get_global_stats()
        
        for candle in sample_candles:
            sliding_buffer.add_candle(symbol, timeframe, candle)
        
        final_stats = sliding_buffer.get_global_stats()
        
        # Memory efficiency assertions
        assert final_stats['memory_usage_mb'] < 50, f"Memory usage {final_stats['memory_usage_mb']:.2f}MB, expected < 50MB"
        assert final_stats['cache_hit_rate'] >= 0.0
        assert final_stats['cache_hit_rate'] <= 1.0
        
        # Test buffer size limits
        assert final_stats['total_candles'] <= 1000  # Max buffer size
        
        logger.info(f"✅ Memory efficiency: {final_stats['memory_usage_mb']:.2f}MB for {final_stats['total_candles']} candles")
    
    @pytest.mark.asyncio
    async def test_concurrent_access(self):
        """Test concurrent access to async components"""
        async_buffer = AsyncSlidingWindowBuffer(max_size=1000)
        symbol = "BTC/USDT"
        timeframe = "1m"
        
        # Create sample candle
        candle = {
            'open': 50000.0,
            'high': 50100.0,
            'low': 49900.0,
            'close': 50050.0,
            'volume': 1000,
            'timestamp': datetime.now(timezone.utc)
        }
        
        # Test concurrent operations
        async def add_candles():
            for i in range(100):
                await async_buffer.add_candle_async(symbol, timeframe, candle)
        
        async def get_candles():
            for i in range(100):
                await async_buffer.get_recent_candles_async(symbol, timeframe, 10)
        
        # Run concurrent operations
        start_time = time.time()
        await asyncio.gather(add_candles(), get_candles())
        concurrent_time = time.time() - start_time
        
        # Performance assertions
        assert concurrent_time < 2.0, f"Concurrent operations took {concurrent_time:.4f}s, expected < 2.0s"
        
        # Verify data integrity
        stats = await async_buffer.get_buffer_stats_async(symbol, timeframe)
        assert stats['size'] > 0
        
        logger.info(f"✅ Concurrent access performance: {concurrent_time:.4f}s")
    
    def test_latency_benchmarks(self, ultra_fast_detector, sample_candles):
        """Test latency benchmarks for different operations"""
        symbol = "BTC/USDT"
        timeframe = "1m"
        
        # Test single pattern detection latency
        start_time = time.time()
        results = asyncio.run(ultra_fast_detector.detect_patterns_async(
            symbol, timeframe, sample_candles[:100]  # Test with 100 candles
        ))
        single_latency = time.time() - start_time
        
        # Test batch pattern detection latency
        start_time = time.time()
        batch_results = []
        for i in range(0, len(sample_candles), 100):
            batch = sample_candles[i:i+100]
            batch_result = asyncio.run(ultra_fast_detector.detect_patterns_async(
                symbol, timeframe, batch
            ))
            batch_results.extend(batch_result)
        batch_latency = time.time() - start_time
        
        # Latency assertions
        assert single_latency < 0.5, f"Single detection latency {single_latency:.4f}s, expected < 0.5s"
        assert batch_latency < 2.0, f"Batch detection latency {batch_latency:.4f}s, expected < 2.0s"
        
        # Verify throughput
        throughput = len(batch_results) / batch_latency
        assert throughput > 50, f"Throughput {throughput:.1f} patterns/sec, expected > 50"
        
        logger.info(f"✅ Latency benchmarks: Single={single_latency:.4f}s, Batch={batch_latency:.4f}s, Throughput={throughput:.1f} patterns/sec")
    
    def test_accuracy_validation(self, ultra_fast_detector, sample_candles):
        """Test accuracy of pattern detection"""
        symbol = "BTC/USDT"
        timeframe = "1m"
        
        # Test with known patterns (create artificial patterns)
        test_candles = sample_candles[:50].copy()
        
        # Create a doji pattern
        test_candles[25] = {
            'open': 50000.0,
            'high': 50010.0,
            'low': 49990.0,
            'close': 50000.0,  # Same as open
            'volume': 1000,
            'timestamp': datetime.now(timezone.utc)
        }
        
        # Create a hammer pattern
        test_candles[30] = {
            'open': 50000.0,
            'high': 50005.0,
            'low': 49950.0,  # Long lower shadow
            'close': 50002.0,
            'volume': 1000,
            'timestamp': datetime.now(timezone.utc)
        }
        
        # Detect patterns
        results = asyncio.run(ultra_fast_detector.detect_patterns_async(
            symbol, timeframe, test_candles
        ))
        
        # Verify pattern detection
        pattern_names = [r.pattern_name for r in results]
        assert 'doji' in pattern_names or len([r for r in results if 'doji' in r.pattern_name.lower()]) > 0
        assert 'hammer' in pattern_names or len([r for r in results if 'hammer' in r.pattern_name.lower()]) > 0
        
        # Verify confidence scores
        for result in results:
            assert 0.0 <= result.confidence <= 1.0
            assert result.strength in ['weak', 'moderate', 'strong']
            assert result.direction in ['bullish', 'bearish', 'neutral']
        
        logger.info(f"✅ Accuracy validation: Detected {len(results)} patterns with valid confidence scores")

if __name__ == "__main__":
    # Run performance tests
    pytest.main([__file__, "-v", "-s"])
