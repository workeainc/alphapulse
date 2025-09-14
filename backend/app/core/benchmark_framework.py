"""
Benchmark Framework for Performance Testing
Tests single vs. multi-symbol scenarios and measures performance metrics
"""

import time
import logging
import asyncio
import threading
import psutil
import os
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import statistics
import json
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class BenchmarkConfig:
    """Configuration for benchmark tests"""
    test_name: str
    scenario: str  # 'single_symbol', 'multi_symbol', 'cold_start', 'warm_start'
    iterations: int = 3
    warmup_iterations: int = 1
    timeout_seconds: float = 300.0  # 5 minutes
    parallel_workers: int = 4
    memory_tracking: bool = True
    cpu_tracking: bool = True
    detailed_logging: bool = False

@dataclass
class BenchmarkMetrics:
    """Performance metrics collected during benchmark"""
    timestamp: datetime
    execution_time_ms: float
    memory_usage_mb: float
    cpu_utilization: float
    throughput_ops_per_sec: float
    error_count: int = 0
    success_count: int = 0
    additional_metrics: Dict[str, Any] = field(default_factory=dict)

class BenchmarkFramework:
    """Comprehensive benchmark framework for performance testing"""
    
    def __init__(self, output_dir: str = "results/benchmark_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        self.benchmark_results: List[BenchmarkResult] = []
        self.performance_baselines: List[PerformanceBaseline] = []
        
        # System monitoring
        self._process = psutil.Process(os.getpid())
        self._baseline_memory = self._process.memory_info().rss / 1024 / 1024  # MB
        self._baseline_cpu = 0.0
        
        # Import from performance_profiling
        from app.core.performance_profiling import BenchmarkResult, PerformanceBaseline
        self.BenchmarkResult = BenchmarkResult
        self.PerformanceBaseline = PerformanceBaseline
    
    async def run_benchmark(self, config: BenchmarkConfig, 
                           test_function: Callable, *args, **kwargs):
        """Run a benchmark test with the specified configuration"""
        try:
            self.logger.info(f"🚀 Starting benchmark: {config.test_name} ({config.scenario})")
            self.logger.info(f"   - Iterations: {config.iterations}")
            self.logger.info(f"   - Warmup iterations: {config.warmup_iterations}")
            self.logger.info(f"   - Parallel workers: {config.parallel_workers}")
            
            # Warmup phase
            if config.warmup_iterations > 0:
                await self._run_warmup(config, test_function, *args, **kwargs)
            
            # Main benchmark phase
            metrics_list = []
            for iteration in range(config.iterations):
                self.logger.info(f"📊 Running iteration {iteration + 1}/{config.iterations}")
                
                metrics = await self._run_single_iteration(config, test_function, *args, **kwargs)
                metrics_list.append(metrics)
                
                # Small delay between iterations
                await asyncio.sleep(0.1)
            
            # Calculate aggregate results
            benchmark_result = self._calculate_aggregate_results(config, metrics_list)
            
            # Store result
            self.benchmark_results.append(benchmark_result)
            
            # Save detailed results
            self._save_benchmark_results(config, metrics_list, benchmark_result)
            
            self.logger.info(f"✅ Benchmark completed: {benchmark_result.execution_time_ms:.2f}ms avg")
            return benchmark_result
            
        except Exception as e:
            self.logger.error(f"❌ Benchmark failed: {e}")
            raise
    
    async def _run_warmup(self, config: BenchmarkConfig, test_function: Callable, *args, **kwargs):
        """Run warmup iterations to stabilize performance"""
        try:
            self.logger.info("🔥 Running warmup iterations...")
            
            for i in range(config.warmup_iterations):
                await self._run_single_iteration(config, test_function, *args, **kwargs)
                await asyncio.sleep(0.05)  # Short delay between warmup runs
            
            self.logger.info("✅ Warmup completed")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Warmup iteration {i + 1} failed: {e}")
    
    async def _run_single_iteration(self, config: BenchmarkConfig, test_function: Callable, 
                                   *args, **kwargs) -> BenchmarkMetrics:
        """Run a single benchmark iteration"""
        try:
            # Start monitoring
            start_time = time.time()
            start_memory = self._process.memory_info().rss / 1024 / 1024  # MB
            start_cpu = self._process.cpu_percent()
            
            # Execute test function
            if asyncio.iscoroutinefunction(test_function):
                result = await test_function(*args, **kwargs)
            else:
                # Run sync function in thread pool
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, test_function, *args, **kwargs)
            
            # End monitoring
            end_time = time.time()
            end_memory = self._process.memory_info().rss / 1024 / 1024  # MB
            end_cpu = self._process.cpu_percent()
            
            # Calculate metrics
            execution_time = (end_time - start_time) * 1000  # Convert to ms
            memory_usage = end_memory - self._baseline_memory
            cpu_utilization = (start_cpu + end_cpu) / 2  # Average CPU usage
            
            # Calculate throughput (if result provides operation count)
            throughput = self._calculate_throughput(result, execution_time)
            
            metrics = BenchmarkMetrics(
                timestamp=datetime.now(),
                execution_time_ms=execution_time,
                memory_usage_mb=memory_usage,
                cpu_utilization=cpu_utilization,
                throughput_ops_per_sec=throughput,
                success_count=1,
                error_count=0,
                additional_metrics={'result_type': type(result).__name__}
            )
            
            if config.detailed_logging:
                self.logger.info(f"   Iteration metrics: {execution_time:.2f}ms, "
                               f"{memory_usage:.2f}MB, {cpu_utilization:.1f}% CPU")
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"❌ Iteration failed: {e}")
            
            # Return error metrics
            return BenchmarkMetrics(
                timestamp=datetime.now(),
                execution_time_ms=0.0,
                memory_usage_mb=0.0,
                cpu_utilization=0.0,
                throughput_ops_per_sec=0.0,
                success_count=0,
                error_count=1,
                additional_metrics={'error': str(e)}
            )
    
    def _calculate_throughput(self, result: Any, execution_time_ms: float) -> float:
        """Calculate throughput based on result and execution time"""
        try:
            if execution_time_ms <= 0:
                return 0.0
            
            # Try to extract operation count from result
            if hasattr(result, 'operation_count'):
                return result.operation_count / (execution_time_ms / 1000)
            elif hasattr(result, 'pattern_count'):
                return result.pattern_count / (execution_time_ms / 1000)
            elif hasattr(result, 'row_count'):
                return result.row_count / (execution_time_ms / 1000)
            elif isinstance(result, (list, tuple)):
                return len(result) / (execution_time_ms / 1000)
            elif isinstance(result, dict) and 'count' in result:
                return result['count'] / (execution_time_ms / 1000)
            else:
                # Default: assume 1 operation per execution
                return 1000 / execution_time_ms  # ops per second
                
        except Exception:
            return 0.0
    
    def _calculate_aggregate_results(self, config: BenchmarkConfig, 
                                   metrics_list: List[BenchmarkMetrics]):
        """Calculate aggregate results from multiple iterations"""
        try:
            # Filter successful iterations
            successful_metrics = [m for m in metrics_list if m.success_count > 0]
            
            if not successful_metrics:
                raise Exception("No successful iterations to aggregate")
            
            # Calculate statistics
            execution_times = [m.execution_time_ms for m in successful_metrics]
            memory_usages = [m.memory_usage_mb for m in successful_metrics]
            cpu_utilizations = [m.cpu_utilization for m in successful_metrics]
            throughputs = [m.throughput_ops_per_sec for m in successful_metrics]
            
            # Create benchmark result
            benchmark_id = f"benchmark_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
            
            result = self.BenchmarkResult(
                benchmark_id=benchmark_id,
                timestamp=datetime.now(),
                test_name=config.test_name,
                scenario=config.scenario,
                execution_time_ms=statistics.mean(execution_times),
                memory_usage_mb=statistics.mean(memory_usages),
                cpu_utilization=statistics.mean(cpu_utilizations),
                throughput_ops_per_sec=statistics.mean(throughputs),
                metadata={
                    'iterations': len(successful_metrics),
                    'total_iterations': len(metrics_list),
                    'execution_time_stats': {
                        'mean': statistics.mean(execution_times),
                        'median': statistics.median(execution_times),
                        'min': min(execution_times),
                        'max': max(execution_times),
                        'stdev': statistics.stdev(execution_times) if len(execution_times) > 1 else 0
                    },
                    'memory_stats': {
                        'mean': statistics.mean(memory_usages),
                        'median': statistics.median(memory_usages),
                        'min': min(memory_usages),
                        'max': max(memory_usages),
                        'stdev': statistics.stdev(memory_usages) if len(memory_usages) > 1 else 0
                    },
                    'cpu_stats': {
                        'mean': statistics.mean(cpu_utilizations),
                        'median': statistics.median(cpu_utilizations),
                        'min': min(cpu_utilizations),
                        'max': max(cpu_utilizations),
                        'stdev': statistics.stdev(cpu_utilizations) if len(cpu_utilizations) > 1 else 0
                    },
                    'throughput_stats': {
                        'mean': statistics.mean(throughputs),
                        'median': statistics.median(throughputs),
                        'min': min(throughputs),
                        'max': max(throughputs),
                        'stdev': statistics.stdev(throughputs) if len(throughputs) > 1 else 0
                    }
                }
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error calculating aggregate results: {e}")
            raise
    
    def _save_benchmark_results(self, config: BenchmarkConfig, 
                               metrics_list: List[BenchmarkMetrics], 
                               benchmark_result):
        """Save detailed benchmark results to files"""
        try:
            # Create timestamp for filenames
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Save summary results
            summary_path = self.output_dir / f"{config.test_name}_{config.scenario}_{timestamp}_summary.json"
            with open(summary_path, 'w') as f:
                json.dump({
                    'benchmark_id': benchmark_result.benchmark_id,
                    'test_name': benchmark_result.test_name,
                    'scenario': benchmark_result.scenario,
                    'timestamp': benchmark_result.timestamp.isoformat(),
                    'execution_time_ms': benchmark_result.execution_time_ms,
                    'memory_usage_mb': benchmark_result.memory_usage_mb,
                    'cpu_utilization': benchmark_result.cpu_utilization,
                    'throughput_ops_per_sec': benchmark_result.throughput_ops_per_sec,
                    'metadata': benchmark_result.metadata
                }, f, indent=2, default=str)
            
            # Save detailed metrics
            detailed_path = self.output_dir / f"{config.test_name}_{config.scenario}_{timestamp}_detailed.json"
            with open(detailed_path, 'w') as f:
                json.dump({
                    'config': {
                        'test_name': config.test_name,
                        'scenario': config.scenario,
                        'iterations': config.iterations,
                        'warmup_iterations': config.warmup_iterations,
                        'parallel_workers': config.parallel_workers
                    },
                    'individual_metrics': [
                        {
                            'timestamp': m.timestamp.isoformat(),
                            'execution_time_ms': m.execution_time_ms,
                            'memory_usage_mb': m.memory_usage_mb,
                            'cpu_utilization': m.cpu_utilization,
                            'throughput_ops_per_sec': m.throughput_ops_per_sec,
                            'success_count': m.success_count,
                            'error_count': m.error_count,
                            'additional_metrics': m.additional_metrics
                        }
                        for m in metrics_list
                    ]
                }, f, indent=2, default=str)
            
            self.logger.info(f"💾 Benchmark results saved:")
            self.logger.info(f"   - Summary: {summary_path}")
            self.logger.info(f"   - Detailed: {detailed_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving benchmark results: {e}")
    
    async def run_multi_symbol_benchmark(self, config: BenchmarkConfig, 
                                        test_function: Callable, 
                                        symbol_count: int, *args, **kwargs):
        """Run benchmark with multiple symbols to test scaling"""
        try:
            self.logger.info(f"🚀 Starting multi-symbol benchmark: {symbol_count} symbols")
            
            # Create test data with multiple symbols
            test_args = self._prepare_multi_symbol_test_data(symbol_count, *args)
            
            # Run benchmark
            result = await self.run_benchmark(config, test_function, *test_args, **kwargs)
            
            # Add scaling information
            result.metadata['scaling'] = {
                'symbol_count': symbol_count,
                'scaling_factor': symbol_count,
                'efficiency': result.throughput_ops_per_sec / symbol_count if symbol_count > 0 else 0
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Multi-symbol benchmark failed: {e}")
            raise
    
    def _prepare_multi_symbol_test_data(self, symbol_count: int, *args) -> tuple:
        """Prepare test data for multi-symbol scenarios"""
        try:
            # This is a placeholder - actual implementation would depend on your data structure
            # For now, we'll create a simple list of symbol identifiers
            
            symbols = [f"SYMBOL_{i:03d}" for i in range(symbol_count)]
            
            # If args contains data, we might need to replicate it for each symbol
            if args:
                # Simple approach: create a list of (symbol, *args) tuples
                test_args = [(symbol, *args) for symbol in symbols]
                return (test_args,)
            else:
                return (symbols,)
                
        except Exception as e:
            self.logger.error(f"Error preparing multi-symbol test data: {e}")
            return args
    
    async def run_cold_start_benchmark(self, config: BenchmarkConfig, 
                                      test_function: Callable, *args, **kwargs):
        """Run benchmark with cold start (no warmup)"""
        try:
            self.logger.info("❄️ Running cold start benchmark")
            
            # Create cold start config
            cold_config = BenchmarkConfig(
                test_name=f"{config.test_name}_cold_start",
                scenario="cold_start",
                iterations=config.iterations,
                warmup_iterations=0,  # No warmup for cold start
                timeout_seconds=config.timeout_seconds,
                parallel_workers=config.parallel_workers,
                memory_tracking=config.memory_tracking,
                cpu_tracking=config.cpu_tracking,
                detailed_logging=config.detailed_logging
            )
            
            # Run benchmark
            result = await self.run_benchmark(cold_config, test_function, *args, **kwargs)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Cold start benchmark failed: {e}")
            raise
    
    def get_benchmark_summary(self) -> Dict[str, Any]:
        """Get summary of all benchmark results"""
        try:
            if not self.benchmark_results:
                return {"message": "No benchmark results available"}
            
            # Group by test name and scenario
            test_summary = {}
            for result in self.benchmark_results:
                key = f"{result.test_name}_{result.scenario}"
                if key not in test_summary:
                    test_summary[key] = []
                test_summary[key].append(result)
            
            # Calculate statistics for each test
            summary = {}
            for test_key, results in test_summary.items():
                execution_times = [r.execution_time_ms for r in results]
                memory_usages = [r.memory_usage_mb for r in results]
                cpu_utilizations = [r.cpu_utilization for r in results]
                throughputs = [r.throughput_ops_per_sec for r in results]
                
                summary[test_key] = {
                    'total_runs': len(results),
                    'latest_run': results[-1].timestamp.isoformat(),
                    'execution_time': {
                        'avg_ms': statistics.mean(execution_times),
                        'min_ms': min(execution_times),
                        'max_ms': max(execution_times)
                    },
                    'memory_usage': {
                        'avg_mb': statistics.mean(memory_usages),
                        'min_mb': min(memory_usages),
                        'max_mb': max(memory_usages)
                    },
                    'cpu_utilization': {
                        'avg_percent': statistics.mean(cpu_utilizations),
                        'min_percent': min(cpu_utilizations),
                        'max_percent': max(cpu_utilizations)
                    },
                    'throughput': {
                        'avg_ops_per_sec': statistics.mean(throughputs),
                        'min_ops_per_sec': min(throughputs),
                        'max_ops_per_sec': max(throughputs)
                    }
                }
            
            return {
                'total_benchmarks': len(self.benchmark_results),
                'test_summary': summary,
                'recent_runs': [
                    {
                        'benchmark_id': r.benchmark_id,
                        'test_name': r.test_name,
                        'scenario': r.scenario,
                        'execution_time_ms': r.execution_time_ms,
                        'timestamp': r.timestamp.isoformat()
                    }
                    for r in sorted(self.benchmark_results, key=lambda x: x.timestamp, reverse=True)[:10]
                ]
            }
            
        except Exception as e:
            self.logger.error(f"Error getting benchmark summary: {e}")
            return {"error": str(e)}
    
    def clear_benchmark_history(self):
        """Clear benchmark history"""
        self.benchmark_results.clear()
        self.logger.info("🗑️ Benchmark history cleared")

# Global benchmark framework instance
benchmark_framework = BenchmarkFramework()

def get_benchmark_framework() -> BenchmarkFramework:
    """Get the global benchmark framework instance"""
    return benchmark_framework
