#!/usr/bin/env python3
"""
Detailed Benchmark Scenarios for AlphaPulse
Phase 2: Single/multi-symbol testing with advanced performance metrics
"""

import time
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import psutil
import gc

# Update import paths for new structure
from ..ai.profiling_framework import ProfilingFramework, ProfilingResult, PerformanceMetrics

logger = logging.getLogger(__name__)

@dataclass
class BenchmarkScenario:
    """Configuration for a benchmark scenario"""
    name: str
    description: str
    symbols: List[str]
    timeframes: List[str]
    data_size: int
    iterations: int
    warmup_iterations: int
    parallel_workers: int
    scenario_type: str  # 'single_symbol', 'multi_symbol', 'scaling_test'
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class BenchmarkResult:
    """Results from a benchmark scenario"""
    scenario: BenchmarkScenario
    total_time: float
    avg_time_per_iteration: float
    throughput: float
    memory_usage: Dict[str, float]
    cpu_usage: Dict[str, float]
    scaling_efficiency: Optional[float] = None
    bottleneck_analysis: Dict[str, Any] = field(default_factory=dict)
    profiling_results: List[ProfilingResult] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class PerformanceTarget:
    """Performance targets for different scenarios"""
    scenario_type: str
    target_time_ms: float
    target_throughput: float
    max_memory_mb: float
    max_cpu_percent: float
    description: str

class BenchmarkScenarios:
    """Comprehensive benchmark scenarios for AlphaPulse trading system"""
    
    def __init__(self, profiling_framework: ProfilingFramework):
        self.profiler = profiling_framework
        self.results: List[BenchmarkResult] = []
        
        # Performance targets
        self.performance_targets = {
            'single_symbol': PerformanceTarget(
                scenario_type='single_symbol',
                target_time_ms=100.0,
                target_throughput=10.0,
                max_memory_mb=100.0,
                max_cpu_percent=70.0,
                description="Single symbol, one timeframe, 1 year data"
            ),
            'multi_symbol': PerformanceTarget(
                scenario_type='multi_symbol',
                target_time_ms=1000.0,
                target_throughput=1.0,
                max_memory_mb=500.0,
                max_cpu_percent=80.0,
                description="10-20 symbols, multiple timeframes"
            ),
            'scaling_test': PerformanceTarget(
                scenario_type='scaling_test',
                target_time_ms=5000.0,
                target_throughput=0.2,
                max_memory_mb=1000.0,
                max_cpu_percent=90.0,
                description="Large-scale performance test"
            )
        }
        
        # Predefined scenarios
        self.scenarios = self._initialize_scenarios()
        
        logger.info("🚀 Benchmark Scenarios initialized")
        logger.info(f"   Available scenarios: {len(self.scenarios)}")
    
    def _initialize_scenarios(self) -> Dict[str, BenchmarkScenario]:
        """Initialize predefined benchmark scenarios"""
        scenarios = {}
        
        # Case A: Single Symbol
        scenarios['single_symbol_basic'] = BenchmarkScenario(
            name="Single Symbol Basic",
            description="One contract, one timeframe, 1 year data",
            symbols=["BTCUSDT"],
            timeframes=["1h"],
            data_size=8760,
            iterations=50,
            warmup_iterations=10,
            parallel_workers=1,
            scenario_type="single_symbol"
        )
        
        # Case B: Multi-Symbol Parallel
        scenarios['multi_symbol_small'] = BenchmarkScenario(
            name="Multi-Symbol Small",
            description="10 symbols, parallel processing",
            symbols=["BTCUSDT", "ETHUSDT", "ADAUSDT", "DOTUSDT", "LINKUSDT", 
                    "LTCUSDT", "BCHUSDT", "XRPUSDT", "EOSUSDT", "TRXUSDT"],
            timeframes=["1h"],
            data_size=8760,
            iterations=20,
            warmup_iterations=5,
            parallel_workers=4,
            scenario_type="multi_symbol"
        )
        
        return scenarios
    
    def generate_test_data(self, symbol: str, timeframe: str, size: int) -> pd.DataFrame:
        """Generate realistic test data for benchmarking"""
        np.random.seed(hash(symbol) % 2**32)
        
        end_date = datetime.now()
        start_date = end_date - timedelta(hours=size)
        dates = pd.date_range(start=start_date, end=end_date, periods=size)
        
        base_price = 50000 if "BTC" in symbol else 3000 if "ETH" in symbol else 100
        volatility = 0.02
        
        price_changes = np.random.normal(0, volatility/np.sqrt(24), size)
        prices = [base_price]
        
        for change in price_changes[1:]:
            new_price = prices[-1] * (1 + change)
            prices.append(max(new_price, base_price * 0.1))
        
        data = []
        for i, (date, price) in enumerate(zip(dates, prices)):
            intra_volatility = price * 0.005
            
            open_price = price
            high_price = price + np.random.uniform(0, intra_volatility)
            low_price = price - np.random.uniform(0, intra_volatility)
            close_price = price + np.random.uniform(-intra_volatility/2, intra_volatility/2)
            
            high_price = max(high_price, open_price, close_price)
            low_price = min(low_price, open_price, close_price)
            
            base_volume = 1000
            volume = base_volume + np.random.uniform(-200, 800)
            volume = max(volume, 100)
            
            data.append({
                'timestamp': date,
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': volume
            })
        
        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        return df
    
    def simulate_pattern_detection(self, data: pd.DataFrame, symbol: str, timeframe: str) -> Dict[str, Any]:
        """Simulate pattern detection workload"""
        result = {
            'symbol': symbol,
            'timeframe': timeframe,
            'patterns_found': 0,
            'processing_time': 0.0,
            'data_points': len(data)
        }
        
        start_time = time.time()
        
        # Simulate pattern detection algorithm
        for i in range(20, len(data)):
            # Calculate technical indicators
            sma_20 = data['close'].iloc[i-20:i].mean()
            sma_50 = data['close'].iloc[i-50:i].mean() if i >= 50 else sma_20
            
            if i >= 14:
                delta = data['close'].iloc[i-14:i+1].diff()
                gain = delta.where(delta > 0, 0).mean()
                loss = -delta.where(delta < 0, 0).mean()
                if loss != 0:
                    rs = gain / loss
                    rsi = 100 - (100 / (1 + rs))
                else:
                    rsi = 100
            
            if i >= 26:
                ema_12 = data['close'].iloc[i-12:i+1].ewm(span=12).mean().iloc[-1]
                ema_26 = data['close'].iloc[i-26:i+1].ewm(span=26).mean().iloc[-1]
                macd = ema_12 - ema_26
            
            if i % 100 == 0:
                result['patterns_found'] += 1
        
        result['processing_time'] = time.time() - start_time
        return result
    
    def run_single_symbol_benchmark(self, scenario: BenchmarkScenario) -> BenchmarkResult:
        """Run single symbol benchmark"""
        logger.info(f"🏃 Running single symbol benchmark: {scenario.name}")
        
        self.profiler.start_system_monitoring(interval=0.1)
        
        symbol = scenario.symbols[0]
        timeframe = scenario.timeframes[0]
        data = self.generate_test_data(symbol, timeframe, scenario.data_size)
        
        # Warmup
        for _ in range(scenario.warmup_iterations):
            self.simulate_pattern_detection(data, symbol, timeframe)
        
        gc.collect()
        
        # Benchmark iterations
        start_time = time.time()
        results = []
        
        for i in range(scenario.iterations):
            iteration_start = time.time()
            
            self.profiler.start_cprofile(f"single_symbol_iteration_{i}")
            result = self.simulate_pattern_detection(data, symbol, timeframe)
            profiling_result = self.profiler.stop_cprofile(f"single_symbol_iteration_{i}")
            
            iteration_time = time.time() - iteration_start
            results.append({
                'iteration': i,
                'processing_time': result['processing_time'],
                'total_time': iteration_time,
                'patterns_found': result['patterns_found']
            })
        
        total_time = time.time() - start_time
        self.profiler.stop_system_monitoring()
        
        # Calculate metrics
        avg_time = total_time / scenario.iterations
        throughput = scenario.iterations / total_time
        
        current_metrics = self.profiler.get_current_metrics()
        memory_usage = {
            'current': current_metrics.memory_usage if current_metrics else 0.0,
            'peak': current_metrics.memory_peak if current_metrics else 0.0
        }
        cpu_usage = {
            'current': current_metrics.cpu_usage if current_metrics else 0.0,
            'average': sum(m.cpu_usage for m in self.profiler.performance_history[-10:]) / 10 if self.profiler.performance_history else 0.0
        }
        
        benchmark_result = BenchmarkResult(
            scenario=scenario,
            total_time=total_time,
            avg_time_per_iteration=avg_time,
            throughput=throughput,
            memory_usage=memory_usage,
            cpu_usage=cpu_usage,
            profiling_results=self.profiler.profiling_results[-scenario.iterations:]
        )
        
        self.results.append(benchmark_result)
        
        logger.info(f"✅ Single symbol benchmark completed")
        logger.info(f"   Total time: {total_time:.4f}s")
        logger.info(f"   Throughput: {throughput:.2f} ops/sec")
        
        return benchmark_result
    
    def run_multi_symbol_benchmark(self, scenario: BenchmarkScenario) -> BenchmarkResult:
        """Run multi-symbol parallel benchmark"""
        logger.info(f"🏃 Running multi-symbol benchmark: {scenario.name}")
        
        self.profiler.start_system_monitoring(interval=0.1)
        
        # Generate test data for all symbols
        test_data = {}
        for symbol in scenario.symbols:
            for timeframe in scenario.timeframes:
                key = f"{symbol}_{timeframe}"
                test_data[key] = self.generate_test_data(symbol, timeframe, scenario.data_size)
        
        # Warmup
        for _ in range(scenario.warmup_iterations):
            for key, data in test_data.items():
                symbol, timeframe = key.split('_', 1)
                self.simulate_pattern_detection(data, symbol, timeframe)
        
        gc.collect()
        
        # Benchmark iterations
        start_time = time.time()
        results = []
        
        def process_symbol_data(key: str, data: pd.DataFrame, iteration: int):
            symbol, timeframe = key.split('_', 1)
            self.profiler.start_cprofile(f"multi_symbol_{key}_iteration_{iteration}")
            result = self.simulate_pattern_detection(data, symbol, timeframe)
            profiling_result = self.profiler.stop_cprofile(f"multi_symbol_{key}_iteration_{iteration}")
            return {
                'symbol': symbol,
                'timeframe': timeframe,
                'processing_time': result['processing_time'],
                'patterns_found': result['patterns_found']
            }
        
        for i in range(scenario.iterations):
            iteration_start = time.time()
            
            with ThreadPoolExecutor(max_workers=scenario.parallel_workers) as executor:
                futures = []
                for key, data in test_data.items():
                    future = executor.submit(process_symbol_data, key, data, i)
                    futures.append(future)
                
                iteration_results = []
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        iteration_results.append(result)
                    except Exception as e:
                        logger.error(f"Error in parallel processing: {e}")
            
            iteration_time = time.time() - iteration_start
            results.append({
                'iteration': i,
                'total_time': iteration_time,
                'symbol_results': iteration_results
            })
        
        total_time = time.time() - start_time
        self.profiler.stop_system_monitoring()
        
        # Calculate metrics
        avg_time = total_time / scenario.iterations
        throughput = scenario.iterations / total_time
        
        current_metrics = self.profiler.get_current_metrics()
        memory_usage = {
            'current': current_metrics.memory_usage if current_metrics else 0.0,
            'peak': current_metrics.memory_peak if current_metrics else 0.0
        }
        cpu_usage = {
            'current': current_metrics.cpu_usage if current_metrics else 0.0,
            'average': sum(m.cpu_usage for m in self.profiler.performance_history[-10:]) / 10 if self.profiler.performance_history else 0.0
        }
        
        # Calculate scaling efficiency
        total_symbols = len(scenario.symbols) * len(scenario.timeframes)
        theoretical_time = avg_time * total_symbols
        actual_time = avg_time
        scaling_efficiency = theoretical_time / actual_time if actual_time > 0 else 0.0
        
        benchmark_result = BenchmarkResult(
            scenario=scenario,
            total_time=total_time,
            avg_time_per_iteration=avg_time,
            throughput=throughput,
            memory_usage=memory_usage,
            cpu_usage=cpu_usage,
            scaling_efficiency=scaling_efficiency,
            profiling_results=self.profiler.profiling_results[-scenario.iterations * total_symbols:]
        )
        
        self.results.append(benchmark_result)
        
        logger.info(f"✅ Multi-symbol benchmark completed")
        logger.info(f"   Total time: {total_time:.4f}s")
        logger.info(f"   Scaling efficiency: {scaling_efficiency:.2f}x")
        
        return benchmark_result
    
    def run_benchmark_scenario(self, scenario_name: str) -> Optional[BenchmarkResult]:
        """Run a specific benchmark scenario"""
        if scenario_name not in self.scenarios:
            logger.error(f"Scenario '{scenario_name}' not found")
            return None
        
        scenario = self.scenarios[scenario_name]
        
        try:
            if scenario.scenario_type == "single_symbol":
                return self.run_single_symbol_benchmark(scenario)
            elif scenario.scenario_type == "multi_symbol":
                return self.run_multi_symbol_benchmark(scenario)
            else:
                logger.error(f"Unknown scenario type: {scenario.scenario_type}")
                return None
                
        except Exception as e:
            logger.error(f"Error running benchmark scenario '{scenario_name}': {e}")
            return None
    
    def run_all_scenarios(self) -> List[BenchmarkResult]:
        """Run all benchmark scenarios"""
        logger.info("🚀 Running all benchmark scenarios...")
        
        results = []
        for scenario_name in self.scenarios.keys():
            try:
                result = self.run_benchmark_scenario(scenario_name)
                if result:
                    results.append(result)
            except Exception as e:
                logger.error(f"Error running scenario '{scenario_name}': {e}")
        
        logger.info(f"✅ Completed {len(results)} benchmark scenarios")
        return results
    
    def generate_benchmark_report(self) -> str:
        """Generate comprehensive benchmark report"""
        if not self.results:
            return "No benchmark results available"
        
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("ALPHAPULSE BENCHMARK REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"Generated: {datetime.now()}")
        report_lines.append(f"Total scenarios: {len(self.results)}")
        report_lines.append("")
        
        # Summary table
        report_lines.append("Benchmark Results Summary:")
        report_lines.append("-" * 80)
        report_lines.append(f"{'Scenario':<25} {'Type':<15} {'Time (ms)':<12} {'Throughput':<12} {'Memory (MB)':<12} {'CPU %':<8}")
        report_lines.append("-" * 80)
        
        for result in self.results:
            time_ms = result.avg_time_per_iteration * 1000
            memory = result.memory_usage['current']
            cpu = result.cpu_usage['current']
            
            report_lines.append(
                f"{result.scenario.name:<25} {result.scenario.scenario_type:<15} "
                f"{time_ms:<12.2f} {result.throughput:<12.2f} {memory:<12.1f} {cpu:<8.1f}"
            )
        
        report_lines.append("")
        
        # Detailed results
        for result in self.results:
            report_lines.append(f"Detailed Results - {result.scenario.name}:")
            report_lines.append("-" * 40)
            report_lines.append(f"  Description: {result.scenario.description}")
            report_lines.append(f"  Total time: {result.total_time:.4f}s")
            report_lines.append(f"  Avg time per iteration: {result.avg_time_per_iteration:.4f}s")
            report_lines.append(f"  Throughput: {result.throughput:.2f} ops/sec")
            report_lines.append(f"  Memory usage: {result.memory_usage['current']:.1f} MB (peak: {result.memory_usage['peak']:.1f} MB)")
            report_lines.append(f"  CPU usage: {result.cpu_usage['current']:.1f}% (avg: {result.cpu_usage['average']:.1f}%)")
            
            if result.scaling_efficiency:
                report_lines.append(f"  Scaling efficiency: {result.scaling_efficiency:.2f}x")
            
            report_lines.append("")
        
        report_lines.append("=" * 80)
        
        report = "\n".join(report_lines)
        
        # Save report to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = Path("benchmark_results") / f"benchmark_report_{timestamp}.txt"
        report_file.parent.mkdir(exist_ok=True)
        
        with open(report_file, 'w') as f:
            f.write(report)
        
        logger.info(f"📊 Generated benchmark report: {report_file}")
        return str(report_file)

# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    profiler = ProfilingFramework()
    benchmark_scenarios = BenchmarkScenarios(profiler)
    
    # Run specific scenario
    result = benchmark_scenarios.run_benchmark_scenario("single_symbol_basic")
    
    # Run all scenarios
    results = benchmark_scenarios.run_all_scenarios()
    
    # Generate report
    report_file = benchmark_scenarios.generate_benchmark_report()
    
    print(f"Benchmark complete! Report saved to: {report_file}")
