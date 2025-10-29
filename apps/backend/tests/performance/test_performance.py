"""
Performance Tests for AlphaPulse

This module provides comprehensive performance testing including latency,
throughput, memory usage, and system resource monitoring.
"""

import pytest
import asyncio
import time
import psutil
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any
import logging
import gc
import os
import sys

# Update import paths for new structure
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'backend'))

from ..src.core.alphapulse_core import AlphaPulseCore
from ..src.core.indicators_engine import IndicatorsEngine
from ..src.core.ml_signal_generator import MLSignalGenerator
from ..src.services.data_services import DataService
from ..src.utils.utils import PerformanceMonitor, Cache
from ..src.database.models import Signal, Log, PerformanceMetrics
from ..src.database.connection import get_session

logger = logging.getLogger(__name__)


class TestPerformance:
    """Performance testing suite for AlphaPulse."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test environment."""
        self.monitor = PerformanceMonitor()
        self.cache = Cache(default_ttl=300)
        self.test_data = self._generate_test_data()
        
        # Performance targets
        self.latency_target = 50  # milliseconds
        self.throughput_target = 10000  # signals per second
        self.memory_target = 100  # MB
        self.cpu_target = 80  # percent
        
        yield
        
        # Cleanup
        self.cache.clear()
        gc.collect()
    
    def _generate_test_data(self) -> pd.DataFrame:
        """Generate test candle data."""
        np.random.seed(42)
        n_candles = 1000
        
        timestamps = pd.date_range(
            start=datetime.now() - timedelta(hours=n_candles),
            periods=n_candles,
            freq='1min'
        )
        
        # Generate realistic price data
        base_price = 50000
        returns = np.random.normal(0, 0.001, n_candles)
        prices = base_price * np.exp(np.cumsum(returns))
        
        data = pd.DataFrame({
            'timestamp': timestamps,
            'open': prices * (1 + np.random.normal(0, 0.0005, n_candles)),
            'high': prices * (1 + np.abs(np.random.normal(0, 0.001, n_candles))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.001, n_candles))),
            'close': prices,
            'volume': np.random.lognormal(10, 1, n_candles)
        })
        
        return data
    
    @pytest.mark.asyncio
    async def test_latency_targets(self):
        """Test that latency targets are met."""
        self.monitor.start_timer("signal_generation")
        
        # Initialize components
        indicators_engine = IndicatorsEngine()
        ml_generator = MLSignalGenerator()
        
        # Process test data
        for i in range(100):
            candle = self.test_data.iloc[i].to_dict()
            
            # Calculate indicators
            indicators = await indicators_engine.calculate_all_indicators([candle])
            
            # Generate signal
            signal = await ml_generator.generate_signal(indicators)
            
            if signal:
                break
        
        latency = self.monitor.end_timer("signal_generation") * 1000  # Convert to milliseconds
        
        logger.info(f"Signal generation latency: {latency:.2f}ms")
        assert latency < self.latency_target, f"Latency {latency}ms exceeds target {self.latency_target}ms"
    
    @pytest.mark.asyncio
    async def test_throughput_targets(self):
        """Test that throughput targets are met."""
        self.monitor.start_timer("throughput_test")
        
        # Initialize components
        indicators_engine = IndicatorsEngine()
        ml_generator = MLSignalGenerator()
        
        signals_generated = 0
        start_time = time.time()
        
        # Process data for 1 second
        while time.time() - start_time < 1.0:
            for i in range(min(100, len(self.test_data))):
                candle = self.test_data.iloc[i].to_dict()
                
                # Calculate indicators
                indicators = await indicators_engine.calculate_all_indicators([candle])
                
                # Generate signal
                signal = await ml_generator.generate_signal(indicators)
                
                if signal:
                    signals_generated += 1
        
        duration = time.time() - start_time
        throughput = signals_generated / duration
        
        logger.info(f"Throughput: {throughput:.2f} signals/second")
        assert throughput >= self.throughput_target, f"Throughput {throughput} below target {self.throughput_target}"
    
    @pytest.mark.asyncio
    async def test_memory_usage(self):
        """Test memory usage stays within targets."""
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        # Initialize components
        indicators_engine = IndicatorsEngine()
        ml_generator = MLSignalGenerator()
        data_service = DataService()
        
        # Process large dataset
        for i in range(0, len(self.test_data), 10):
            batch = self.test_data.iloc[i:i+10].to_dict('records')
            
            # Calculate indicators
            indicators = await indicators_engine.calculate_all_indicators(batch)
            
            # Generate signals
            for indicator_set in indicators:
                signal = await ml_generator.generate_signal(indicator_set)
                if signal:
                    await data_service.store_data(signal)
        
        # Force garbage collection
        gc.collect()
        
        final_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        logger.info(f"Memory usage: {final_memory:.2f}MB (increase: {memory_increase:.2f}MB)")
        assert memory_increase < self.memory_target, f"Memory increase {memory_increase}MB exceeds target {self.memory_target}MB"
    
    @pytest.mark.asyncio
    async def test_cpu_usage(self):
        """Test CPU usage stays within targets."""
        # Monitor CPU usage during intensive operations
        cpu_percentages = []
        
        indicators_engine = IndicatorsEngine()
        ml_generator = MLSignalGenerator()
        
        # Process data with CPU monitoring
        for i in range(0, len(self.test_data), 5):
            batch = self.test_data.iloc[i:i+5].to_dict('records')
            
            # Monitor CPU during processing
            cpu_start = psutil.cpu_percent(interval=0.1)
            
            # Calculate indicators
            indicators = await indicators_engine.calculate_all_indicators(batch)
            
            # Generate signals
            for indicator_set in indicators:
                signal = await ml_generator.generate_signal(indicator_set)
            
            cpu_end = psutil.cpu_percent(interval=0.1)
            cpu_percentages.append((cpu_start + cpu_end) / 2)
        
        avg_cpu = np.mean(cpu_percentages)
        max_cpu = np.max(cpu_percentages)
        
        logger.info(f"Average CPU usage: {avg_cpu:.2f}%, Max CPU usage: {max_cpu:.2f}%")
        assert avg_cpu < self.cpu_target, f"Average CPU usage {avg_cpu}% exceeds target {self.cpu_target}%"
    
    @pytest.mark.asyncio
    async def test_cache_performance(self):
        """Test cache performance and hit rates."""
        # Test cache operations
        test_keys = [f"test_key_{i}" for i in range(1000)]
        test_values = [f"test_value_{i}" for i in range(1000)]
        
        # Measure cache write performance
        self.monitor.start_timer("cache_writes")
        for key, value in zip(test_keys, test_values):
            self.cache.set(key, value)
        write_time = self.monitor.end_timer("cache_writes")
        
        # Measure cache read performance
        self.monitor.start_timer("cache_reads")
        hits = 0
        for key in test_keys:
            if self.cache.get(key):
                hits += 1
        read_time = self.monitor.end_timer("cache_reads")
        
        hit_rate = hits / len(test_keys)
        write_throughput = len(test_keys) / write_time
        read_throughput = len(test_keys) / read_time
        
        logger.info(f"Cache hit rate: {hit_rate:.2%}")
        logger.info(f"Cache write throughput: {write_throughput:.0f} ops/sec")
        logger.info(f"Cache read throughput: {read_throughput:.0f} ops/sec")
        
        assert hit_rate > 0.95, f"Cache hit rate {hit_rate:.2%} below target 95%"
        assert write_throughput > 10000, f"Cache write throughput {write_throughput:.0f} below target 10000"
        assert read_throughput > 50000, f"Cache read throughput {read_throughput:.0f} below target 50000"
    
    @pytest.mark.asyncio
    async def test_database_performance(self):
        """Test database performance for signal storage and retrieval."""
        async with get_session() as session:
            # Test signal insertion performance
            self.monitor.start_timer("signal_insertion")
            
            signals = []
            for i in range(100):
                signal = Signal(
                    signal_id=f"test_signal_{i}",
                    symbol="BTC/USDT",
                    timeframe="1m",
                    direction="long",
                    confidence=0.8,
                    entry_price=50000.0,
                    tp1=51000.0,
                    tp2=52000.0,
                    tp3=53000.0,
                    tp4=54000.0,
                    stop_loss=49000.0,
                    risk_reward_ratio=2.0,
                    pattern_type="breakout",
                    volume_confirmation=True,
                    trend_alignment=True,
                    market_regime="trending",
                    indicators={"rsi": 65.0, "macd": 0.5},
                    validation_metrics={"accuracy": 0.8, "filter_rate": 0.7},
                    timestamp=datetime.now()
                )
                signals.append(signal)
            
            session.add_all(signals)
            await session.commit()
            
            insertion_time = self.monitor.end_timer("signal_insertion")
            
            # Test signal query performance
            self.monitor.start_timer("signal_query")
            
            query = await session.execute(
                "SELECT * FROM signals WHERE symbol = 'BTC/USDT' AND confidence > 0.7"
            )
            results = query.fetchall()
            
            query_time = self.monitor.end_timer("signal_query")
            
            insertion_throughput = len(signals) / insertion_time
            query_latency = query_time * 1000  # Convert to milliseconds
            
            logger.info(f"Signal insertion throughput: {insertion_throughput:.0f} signals/sec")
            logger.info(f"Signal query latency: {query_latency:.2f}ms")
            logger.info(f"Query returned {len(results)} signals")
            
            assert insertion_throughput > 100, f"Insertion throughput {insertion_throughput:.0f} below target 100"
            assert query_latency < 10, f"Query latency {query_latency:.2f}ms exceeds target 10ms"
            
            # Cleanup
            for signal in signals:
                await session.delete(signal)
            await session.commit()
    
    @pytest.mark.asyncio
    async def test_end_to_end_performance(self):
        """Test end-to-end performance of the complete pipeline."""
        self.monitor.start_timer("end_to_end")
        
        # Initialize AlphaPulse core
        core = AlphaPulseCore(
            symbols=["BTC/USDT"],
            timeframes=["1m"],
            risk_per_trade=0.02,
            max_positions=5
        )
        
        # Process test data through complete pipeline
        signals_generated = 0
        signals_validated = 0
        
        for i in range(0, len(self.test_data), 10):
            batch = self.test_data.iloc[i:i+10].to_dict('records')
            
            # Process through pipeline
            for candle in batch:
                # Generate signal
                signal = await core._generate_signals(candle)
                
                if signal:
                    signals_generated += 1
                    
                    # Validate signal
                    is_valid = await core._validate_signals(signal)
                    if is_valid:
                        signals_validated += 1
                        
                        # Store signal
                        await core._store_signal(signal)
        
        total_time = self.monitor.end_timer("end_to_end")
        
        # Calculate performance metrics
        throughput = signals_generated / total_time
        validation_rate = signals_validated / signals_generated if signals_generated > 0 else 0
        avg_latency = total_time / signals_generated * 1000 if signals_generated > 0 else 0
        
        logger.info(f"End-to-end performance:")
        logger.info(f"  Total time: {total_time:.2f}s")
        logger.info(f"  Signals generated: {signals_generated}")
        logger.info(f"  Signals validated: {signals_validated}")
        logger.info(f"  Throughput: {throughput:.2f} signals/sec")
        logger.info(f"  Validation rate: {validation_rate:.2%}")
        logger.info(f"  Average latency: {avg_latency:.2f}ms")
        
        # Performance assertions
        assert throughput >= 100, f"Throughput {throughput:.2f} below target 100 signals/sec"
        assert avg_latency < self.latency_target, f"Average latency {avg_latency:.2f}ms exceeds target {self.latency_target}ms"
        assert validation_rate > 0.6, f"Validation rate {validation_rate:.2%} below target 60%"
    
    @pytest.mark.asyncio
    async def test_memory_leaks(self):
        """Test for memory leaks during extended operation."""
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        # Initialize components
        indicators_engine = IndicatorsEngine()
        ml_generator = MLSignalGenerator()
        
        # Run extended operation
        for cycle in range(10):
            logger.info(f"Memory test cycle {cycle + 1}/10")
            
            # Process data
            for i in range(0, len(self.test_data), 20):
                batch = self.test_data.iloc[i:i+20].to_dict('records')
                
                # Calculate indicators
                indicators = await indicators_engine.calculate_all_indicators(batch)
                
                # Generate signals
                for indicator_set in indicators:
                    signal = await ml_generator.generate_signal(indicator_set)
            
            # Force garbage collection
            gc.collect()
            
            # Check memory usage
            current_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            memory_increase = current_memory - initial_memory
            
            logger.info(f"Cycle {cycle + 1}: Memory usage {current_memory:.2f}MB (increase: {memory_increase:.2f}MB)")
            
            # Check for excessive memory growth
            if cycle > 0:
                assert memory_increase < 50, f"Memory leak detected: {memory_increase:.2f}MB increase"
        
        final_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        total_increase = final_memory - initial_memory
        
        logger.info(f"Final memory usage: {final_memory:.2f}MB (total increase: {total_increase:.2f}MB)")
        assert total_increase < 50, f"Total memory increase {total_increase:.2f}MB exceeds limit 50MB"
    
    @pytest.mark.asyncio
    async def test_concurrent_performance(self):
        """Test performance under concurrent load."""
        self.monitor.start_timer("concurrent_test")
        
        # Initialize components
        indicators_engine = IndicatorsEngine()
        ml_generator = MLSignalGenerator()
        
        # Define concurrent task
        async def process_batch(batch_id: int):
            batch = self.test_data.iloc[batch_id*100:(batch_id+1)*100].to_dict('records')
            signals = 0
            
            for candle in batch:
                indicators = await indicators_engine.calculate_all_indicators([candle])
                signal = await ml_generator.generate_signal(indicators)
                if signal:
                    signals += 1
            
            return signals
        
        # Run concurrent tasks
        tasks = [process_batch(i) for i in range(10)]
        results = await asyncio.gather(*tasks)
        
        total_time = self.monitor.end_timer("concurrent_test")
        total_signals = sum(results)
        
        concurrent_throughput = total_signals / total_time
        
        logger.info(f"Concurrent performance:")
        logger.info(f"  Total time: {total_time:.2f}s")
        logger.info(f"  Total signals: {total_signals}")
        logger.info(f"  Concurrent throughput: {concurrent_throughput:.2f} signals/sec")
        
        # Performance assertions
        assert concurrent_throughput >= 500, f"Concurrent throughput {concurrent_throughput:.2f} below target 500"
        assert total_time < 10, f"Concurrent processing time {total_time:.2f}s exceeds target 10s"
    
    def test_system_metrics(self):
        """Test system resource monitoring."""
        # Get system metrics
        system_metrics = self.monitor.get_system_metrics()
        
        logger.info(f"System metrics: {system_metrics}")
        
        # Validate metrics
        assert 'cpu_percent' in system_metrics
        assert 'memory_percent' in system_metrics
        assert 'memory_available' in system_metrics
        assert 'disk_percent' in system_metrics
        assert 'disk_free' in system_metrics
        
        # Check reasonable ranges
        assert 0 <= system_metrics['cpu_percent'] <= 100
        assert 0 <= system_metrics['memory_percent'] <= 100
        assert system_metrics['memory_available'] > 0
        assert 0 <= system_metrics['disk_percent'] <= 100
        assert system_metrics['disk_free'] > 0


class TestPerformanceBenchmarks:
    """Performance benchmarking tests."""
    
    @pytest.mark.benchmark
    def test_indicator_calculation_benchmark(self, benchmark):
        """Benchmark indicator calculation performance."""
        indicators_engine = IndicatorsEngine()
        
        # Generate test data
        test_data = pd.DataFrame({
            'open': np.random.random(1000) * 50000,
            'high': np.random.random(1000) * 50000,
            'low': np.random.random(1000) * 50000,
            'close': np.random.random(1000) * 50000,
            'volume': np.random.random(1000) * 1000000
        }).to_dict('records')
        
        def calculate_indicators():
            return asyncio.run(indicators_engine.calculate_all_indicators(test_data))
        
        result = benchmark(calculate_indicators)
        assert result is not None
    
    @pytest.mark.benchmark
    def test_signal_generation_benchmark(self, benchmark):
        """Benchmark signal generation performance."""
        ml_generator = MLSignalGenerator()
        
        # Generate test indicators
        test_indicators = {
            'rsi': 65.0,
            'macd': 0.5,
            'bollinger_bands': {'upper': 51000, 'middle': 50000, 'lower': 49000},
            'atr': 500.0,
            'adx': 25.0,
            'volume_sma': 1000000,
            'current_volume': 1200000
        }
        
        def generate_signal():
            return asyncio.run(ml_generator.generate_signal(test_indicators))
        
        result = benchmark(generate_signal)
        assert result is not None
    
    @pytest.mark.benchmark
    def test_cache_operations_benchmark(self, benchmark):
        """Benchmark cache operations."""
        cache = Cache()
        
        def cache_operations():
            for i in range(1000):
                cache.set(f"key_{i}", f"value_{i}")
                cache.get(f"key_{i}")
            cache.clear()
        
        result = benchmark(cache_operations)
        assert result is not None


if __name__ == "__main__":
    # Run performance tests
    pytest.main([__file__, "-v", "--benchmark-save=alphapulse"])
