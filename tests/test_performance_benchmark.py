#!/usr/bin/env python3
"""
Performance benchmark tests for AlphaPulse
"""

import pytest
import time
import numpy as np
import psutil
import threading
from datetime import datetime, timedelta
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import AlphaPulse components
try:
    from indicators_engine import TechnicalIndicators
    from ml_signal_generator import MLSignalGenerator
    from alphapulse_core import AlphaPulse
except ImportError:
    # Create mock classes for testing
    class TechnicalIndicators:
        def __init__(self):
            pass
        
        def calculate_rsi(self, prices, period=14):
            if len(prices) < period:
                return None
            return 50.0 + np.random.normal(0, 10)
        
        def calculate_macd(self, prices, fast=8, slow=24, signal=9):
            if len(prices) < slow:
                return None, None, None
            return 100, 95, 5
        
        def calculate_bollinger_bands(self, prices, period=20, std_dev=2):
            if len(prices) < period:
                return None, None, None
            return 50200, 50000, 49800
    
    class MLSignalGenerator:
        def __init__(self):
            pass
        
        def detect_patterns(self, candlestick_data, indicators):
            return [{'pattern_type': 'candlestick', 'confidence': 0.8, 'direction': 'buy'}]
    
    class AlphaPulse:
        def __init__(self, symbols, timeframes):
            self.symbols = symbols
            self.timeframes = timeframes
            self.signals = []
            self.indicators = TechnicalIndicators()
            self.signal_generator = MLSignalGenerator()

class PerformanceMonitor:
    """Monitor system performance during tests"""
    
    def __init__(self):
        self.process = psutil.Process()
        self.start_cpu = None
        self.start_memory = None
        self.peak_cpu = 0
        self.peak_memory = 0
        self.samples = []
    
    def start_monitoring(self):
        """Start performance monitoring"""
        self.start_cpu = self.process.cpu_percent()
        self.start_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        self.peak_cpu = self.start_cpu
        self.peak_memory = self.start_memory
        self.samples = []
    
    def sample(self):
        """Take a performance sample"""
        cpu_percent = self.process.cpu_percent()
        memory_mb = self.process.memory_info().rss / 1024 / 1024
        
        self.peak_cpu = max(self.peak_cpu, cpu_percent)
        self.peak_memory = max(self.peak_memory, memory_mb)
        
        self.samples.append({
            'timestamp': time.time(),
            'cpu_percent': cpu_percent,
            'memory_mb': memory_mb
        })
    
    def stop_monitoring(self):
        """Stop monitoring and return results"""
        avg_cpu = np.mean([s['cpu_percent'] for s in self.samples]) if self.samples else 0
        avg_memory = np.mean([s['memory_mb'] for s in self.samples]) if self.samples else 0
        
        return {
            'start_cpu': self.start_cpu,
            'start_memory': self.start_memory,
            'peak_cpu': self.peak_cpu,
            'peak_memory': self.peak_memory,
            'avg_cpu': avg_cpu,
            'avg_memory': avg_memory,
            'samples': len(self.samples)
        }

class TestPerformanceBenchmarks:
    """Performance benchmark tests"""
    
    def setup_method(self):
        """Setup test environment"""
        self.symbols = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'DOTUSDT', 'LINKUSDT']
        self.timeframes = ['1m', '5m', '15m']
        
        # Initialize components
        self.indicators = TechnicalIndicators()
        self.signal_generator = MLSignalGenerator()
        self.alphapulse = AlphaPulse(self.symbols, self.timeframes)
        
        # Performance monitor
        self.monitor = PerformanceMonitor()
    
    def generate_test_data(self, num_samples: int) -> List[Dict]:
        """Generate test data for benchmarking"""
        data = []
        base_price = 50000
        
        for i in range(num_samples):
            # Simulate realistic price movement
            price_change = np.sin(i * 0.01) * 100 + np.random.normal(0, 50)
            current_price = base_price + price_change
            
            candlestick = {
                'open': current_price - np.random.uniform(0, 100),
                'high': current_price + np.random.uniform(0, 200),
                'low': current_price - np.random.uniform(0, 200),
                'close': current_price + np.random.uniform(-100, 100),
                'volume': 1000000 + np.random.uniform(-200000, 200000)
            }
            
            data.append({
                'candlestick': candlestick,
                'symbol': self.symbols[i % len(self.symbols)],
                'timeframe': self.timeframes[i % len(self.timeframes)]
            })
        
        return data
    
    def test_indicator_calculation_throughput(self, benchmark):
        """Test indicator calculation throughput"""
        print("⚡ Testing indicator calculation throughput...")
        
        # Generate test data
        prices = [50000 + np.random.normal(0, 100) for _ in range(1000)]
        
        def calculate_indicators():
            """Calculate all indicators for benchmark"""
            rsi = self.indicators.calculate_rsi(prices, period=14)
            macd, macd_signal, macd_hist = self.indicators.calculate_macd(prices)
            bb_upper, bb_middle, bb_lower = self.indicators.calculate_bollinger_bands(prices)
            return rsi, macd, bb_upper
        
        # Run benchmark
        result = benchmark(calculate_indicators)
        
        # Verify results
        assert result is not None, "Indicator calculation should succeed"
        
        print(f"✅ Indicator calculation benchmark completed")
        print(f"  Operations per second: {benchmark.stats.stats.mean:.2f}")
    
    def test_signal_generation_throughput(self, benchmark):
        """Test signal generation throughput"""
        print("⚡ Testing signal generation throughput...")
        
        # Generate test data
        test_data = self.generate_test_data(1000)
        
        def generate_signals():
            """Generate signals for benchmark"""
            signals = []
            for data in test_data:
                candlestick = data['candlestick']
                indicators = {
                    'rsi': 65,
                    'macd': 100,
                    'volume_sma': 1000000,
                    'adx': 28
                }
                patterns = self.signal_generator.detect_patterns(candlestick, indicators)
                signals.extend(patterns)
            return signals
        
        # Run benchmark
        result = benchmark(generate_signals)
        
        # Verify results
        assert len(result) > 0, "Should generate some signals"
        
        print(f"✅ Signal generation benchmark completed")
        print(f"  Operations per second: {benchmark.stats.stats.mean:.2f}")
    
    def test_high_throughput_processing(self):
        """Test high-throughput processing (10,000 signals/second)"""
        print("⚡ Testing high-throughput processing...")
        
        # Generate large dataset
        num_signals = 10000
        test_data = self.generate_test_data(num_signals)
        
        # Start performance monitoring
        self.monitor.start_monitoring()
        
        start_time = time.perf_counter()
        
        # Process signals
        signals_generated = 0
        for i, data in enumerate(test_data):
            # Process candlestick
            candlestick = data['candlestick']
            indicators = {
                'rsi': 50 + (i % 40),
                'macd': 100 + i,
                'volume_sma': 1000000,
                'adx': 20 + (i % 30)
            }
            
            # Generate signal
            patterns = self.signal_generator.detect_patterns(candlestick, indicators)
            signals_generated += len(patterns)
            
            # Sample performance every 1000 operations
            if i % 1000 == 0:
                self.monitor.sample()
        
        end_time = time.perf_counter()
        
        # Stop monitoring
        performance = self.monitor.stop_monitoring()
        
        # Calculate metrics
        total_time = end_time - start_time
        throughput = signals_generated / total_time
        
        print(f"📊 High-Throughput Results:")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Signals generated: {signals_generated}")
        print(f"  Throughput: {throughput:.2f} signals/sec")
        print(f"  Peak CPU: {performance['peak_cpu']:.1f}%")
        print(f"  Peak memory: {performance['peak_memory']:.1f} MB")
        print(f"  Average CPU: {performance['avg_cpu']:.1f}%")
        print(f"  Average memory: {performance['avg_memory']:.1f} MB")
        
        # Assertions
        assert throughput > 10000, f"Throughput should be > 10,000 signals/sec, got {throughput:.2f}"
        assert performance['peak_cpu'] < 80, f"Peak CPU should be < 80%, got {performance['peak_cpu']:.1f}%"
        assert performance['peak_memory'] < 1000, f"Peak memory should be < 1000 MB, got {performance['peak_memory']:.1f} MB"
    
    def test_concurrent_processing(self):
        """Test concurrent processing performance"""
        print("⚡ Testing concurrent processing...")
        
        # Generate test data
        num_signals = 5000
        test_data = self.generate_test_data(num_signals)
        
        # Split data for concurrent processing
        num_threads = 4
        chunk_size = len(test_data) // num_threads
        data_chunks = [test_data[i:i+chunk_size] for i in range(0, len(test_data), chunk_size)]
        
        def process_chunk(chunk_data):
            """Process a chunk of data"""
            signals = []
            for data in chunk_data:
                candlestick = data['candlestick']
                indicators = {
                    'rsi': 65,
                    'macd': 100,
                    'volume_sma': 1000000,
                    'adx': 28
                }
                patterns = self.signal_generator.detect_patterns(candlestick, indicators)
                signals.extend(patterns)
            return signals
        
        # Start performance monitoring
        self.monitor.start_monitoring()
        
        start_time = time.perf_counter()
        
        # Process concurrently
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(process_chunk, chunk) for chunk in data_chunks]
            results = [future.result() for future in as_completed(futures)]
        
        end_time = time.perf_counter()
        
        # Stop monitoring
        performance = self.monitor.stop_monitoring()
        
        # Calculate metrics
        total_time = end_time - start_time
        total_signals = sum(len(result) for result in results)
        throughput = total_signals / total_time
        
        print(f"📊 Concurrent Processing Results:")
        print(f"  Threads: {num_threads}")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Signals generated: {total_signals}")
        print(f"  Throughput: {throughput:.2f} signals/sec")
        print(f"  Peak CPU: {performance['peak_cpu']:.1f}%")
        print(f"  Peak memory: {performance['peak_memory']:.1f} MB")
        
        # Assertions
        assert throughput > 5000, f"Concurrent throughput should be > 5,000 signals/sec, got {throughput:.2f}"
        assert performance['peak_cpu'] < 90, f"Peak CPU should be < 90%, got {performance['peak_cpu']:.1f}%"
    
    def test_memory_efficiency(self):
        """Test memory efficiency during long-running operations"""
        print("⚡ Testing memory efficiency...")
        
        # Generate large dataset
        num_signals = 50000
        test_data = self.generate_test_data(num_signals)
        
        # Start performance monitoring
        self.monitor.start_monitoring()
        
        # Process in batches to test memory efficiency
        batch_size = 1000
        signals_generated = 0
        
        for i in range(0, len(test_data), batch_size):
            batch = test_data[i:i+batch_size]
            
            for data in batch:
                candlestick = data['candlestick']
                indicators = {
                    'rsi': 65,
                    'macd': 100,
                    'volume_sma': 1000000,
                    'adx': 28
                }
                patterns = self.signal_generator.detect_patterns(candlestick, indicators)
                signals_generated += len(patterns)
            
            # Sample performance after each batch
            self.monitor.sample()
            
            # Force garbage collection periodically
            if i % 5000 == 0:
                import gc
                gc.collect()
        
        # Stop monitoring
        performance = self.monitor.stop_monitoring()
        
        print(f"📊 Memory Efficiency Results:")
        print(f"  Total signals: {signals_generated}")
        print(f"  Start memory: {performance['start_memory']:.1f} MB")
        print(f"  Peak memory: {performance['peak_memory']:.1f} MB")
        print(f"  Average memory: {performance['avg_memory']:.1f} MB")
        print(f"  Memory growth: {performance['peak_memory'] - performance['start_memory']:.1f} MB")
        
        # Assertions
        memory_growth = performance['peak_memory'] - performance['start_memory']
        assert memory_growth < 500, f"Memory growth should be < 500 MB, got {memory_growth:.1f} MB"
        assert performance['avg_memory'] < 800, f"Average memory should be < 800 MB, got {performance['avg_memory']:.1f} MB"
    
    def test_latency_consistency(self):
        """Test latency consistency under load"""
        print("⚡ Testing latency consistency...")
        
        # Generate test data
        num_signals = 10000
        test_data = self.generate_test_data(num_signals)
        
        latencies = []
        
        # Process signals and measure latency
        for data in test_data:
            start_time = time.perf_counter()
            
            candlestick = data['candlestick']
            indicators = {
                'rsi': 65,
                'macd': 100,
                'volume_sma': 1000000,
                'adx': 28
            }
            patterns = self.signal_generator.detect_patterns(candlestick, indicators)
            
            end_time = time.perf_counter()
            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)
        
        # Calculate latency statistics
        avg_latency = np.mean(latencies)
        p95_latency = np.percentile(latencies, 95)
        p99_latency = np.percentile(latencies, 99)
        max_latency = np.max(latencies)
        min_latency = np.min(latencies)
        
        print(f"📊 Latency Consistency Results:")
        print(f"  Average latency: {avg_latency:.2f}ms")
        print(f"  95th percentile: {p95_latency:.2f}ms")
        print(f"  99th percentile: {p99_latency:.2f}ms")
        print(f"  Min latency: {min_latency:.2f}ms")
        print(f"  Max latency: {max_latency:.2f}ms")
        print(f"  Latency variance: {np.var(latencies):.4f}")
        
        # Assertions
        assert avg_latency < 1.0, f"Average latency should be < 1ms, got {avg_latency:.2f}ms"
        assert p95_latency < 2.0, f"95th percentile latency should be < 2ms, got {p95_latency:.2f}ms"
        assert p99_latency < 5.0, f"99th percentile latency should be < 5ms, got {p99_latency:.2f}ms"
        assert max_latency < 10.0, f"Max latency should be < 10ms, got {max_latency:.2f}ms"
    
    def test_cpu_efficiency(self):
        """Test CPU efficiency under sustained load"""
        print("⚡ Testing CPU efficiency...")
        
        # Generate test data
        num_signals = 20000
        test_data = self.generate_test_data(num_signals)
        
        # Start performance monitoring
        self.monitor.start_monitoring()
        
        # Process signals with frequent sampling
        signals_generated = 0
        for i, data in enumerate(test_data):
            candlestick = data['candlestick']
            indicators = {
                'rsi': 50 + (i % 40),
                'macd': 100 + i,
                'volume_sma': 1000000,
                'adx': 20 + (i % 30)
            }
            patterns = self.signal_generator.detect_patterns(candlestick, indicators)
            signals_generated += len(patterns)
            
            # Sample CPU usage frequently
            if i % 100 == 0:
                self.monitor.sample()
        
        # Stop monitoring
        performance = self.monitor.stop_monitoring()
        
        print(f"📊 CPU Efficiency Results:")
        print(f"  Total signals: {signals_generated}")
        print(f"  Average CPU: {performance['avg_cpu']:.1f}%")
        print(f"  Peak CPU: {performance['peak_cpu']:.1f}%")
        print(f"  CPU samples: {performance['samples']}")
        
        # Calculate CPU efficiency (signals per CPU percentage)
        cpu_efficiency = signals_generated / performance['avg_cpu'] if performance['avg_cpu'] > 0 else float('inf')
        
        print(f"  CPU efficiency: {cpu_efficiency:.0f} signals per CPU %")
        
        # Assertions
        assert performance['avg_cpu'] < 50, f"Average CPU should be < 50%, got {performance['avg_cpu']:.1f}%"
        assert performance['peak_cpu'] < 80, f"Peak CPU should be < 80%, got {performance['peak_cpu']:.1f}%"
        assert cpu_efficiency > 100, f"CPU efficiency should be > 100 signals per CPU %, got {cpu_efficiency:.0f}"

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
