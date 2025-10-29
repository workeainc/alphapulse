"""
Performance Profiling Framework
Integrates cProfile and line_profiler for systematic performance analysis
"""

import cProfile
import pstats
import io
import time
import logging
import asyncio
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import json
import functools
import threading
from contextlib import contextmanager

logger = logging.getLogger(__name__)

@dataclass
class ProfilingResult:
    """Result of a profiling session"""
    session_id: str
    timestamp: datetime
    function_name: str
    total_time: float
    total_calls: int
    primitive_calls: int
    cumulative_time: float
    per_call_time: float
    cumulative_per_call: float
    filename: str
    line_number: int
    profile_data: Dict[str, Any] = field(default_factory=dict)

@dataclass
class BenchmarkResult:
    """Result of a benchmark test"""
    benchmark_id: str
    timestamp: datetime
    test_name: str
    scenario: str  # 'single_symbol', 'multi_symbol', 'cold_start', etc.
    execution_time_ms: float
    memory_usage_mb: float
    cpu_utilization: float
    throughput_ops_per_sec: float
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PerformanceBaseline:
    """Performance baseline for regression testing"""
    baseline_id: str
    timestamp: datetime
    test_name: str
    scenario: str
    expected_execution_time_ms: float
    expected_memory_usage_mb: float
    expected_cpu_utilization: float
    expected_throughput_ops_per_sec: float
    tolerance_percentage: float = 10.0  # 10% tolerance by default

class PerformanceProfiler:
    """Comprehensive performance profiling framework"""
    
    def __init__(self, output_dir: str = "results/performance_profiles"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        self.profiling_results: List[ProfilingResult] = []
        self.benchmark_results: List[BenchmarkResult] = []
        self.performance_baselines: List[PerformanceBaseline] = []
        
        # Profiling state
        self._current_profiler = None
        self._profiling_active = False
        self._profile_lock = threading.Lock()
    
    def profile_function(self, output_file: str = None, sort_by: str = 'cumulative'):
        """Decorator to profile a function using cProfile"""
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                return self._profile_sync_function(func, args, kwargs, output_file, sort_by)
            
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                return await self._profile_async_function(func, args, kwargs, output_file, sort_by)
            
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            else:
                return sync_wrapper
        return decorator
    
    def _profile_sync_function(self, func: Callable, args: tuple, kwargs: dict, 
                              output_file: str, sort_by: str):
        """Profile a synchronous function"""
        try:
            # Create profiler
            profiler = cProfile.Profile()
            profiler.enable()
            
            start_time = time.time()
            result = func(*args, **kwargs)
            execution_time = (time.time() - start_time) * 1000  # Convert to ms
            
            profiler.disable()
            
            # Process profiling results
            self._process_profiling_result(profiler, func.__name__, execution_time, 
                                        output_file, sort_by)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error profiling function {func.__name__}: {e}")
            return func(*args, **kwargs)
    
    async def _profile_async_function(self, func: Callable, args: tuple, kwargs: dict,
                                    output_file: str, sort_by: str):
        """Profile an asynchronous function"""
        try:
            # Create profiler
            profiler = cProfile.Profile()
            profiler.enable()
            
            start_time = time.time()
            result = await func(*args, **kwargs)
            execution_time = (time.time() - start_time) * 1000  # Convert to ms
            
            profiler.disable()
            
            # Process profiling results
            self._process_profiling_result(profiler, func.__name__, execution_time,
                                        output_file, sort_by)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error profiling async function {func.__name__}: {e}")
            return await func(*args, **kwargs)
    
    def _process_profiling_result(self, profiler: cProfile.Profile, function_name: str,
                                execution_time: float, output_file: str, sort_by: str):
        """Process and store profiling results"""
        try:
            # Create stats object
            stats = pstats.Stats(profiler)
            
            # Sort by specified criteria
            if sort_by == 'cumulative':
                stats.sort_stats('cumulative')
            elif sort_by == 'time':
                stats.sort_stats('time')
            elif sort_by == 'calls':
                stats.sort_stats('calls')
            else:
                stats.sort_stats('cumulative')
            
            # Get top functions
            top_functions = []
            for func, (cc, nc, tt, ct, callers) in stats.stats.items():
                filename, line_number, function = func
                top_functions.append({
                    'function': function,
                    'filename': filename,
                    'line_number': line_number,
                    'total_calls': cc,
                    'primitive_calls': nc,
                    'total_time': tt,
                    'cumulative_time': ct,
                    'per_call_time': tt / cc if cc > 0 else 0,
                    'cumulative_per_call': ct / cc if cc > 0 else 0
                })
            
            # Create profiling result
            session_id = f"profile_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
            result = ProfilingResult(
                session_id=session_id,
                timestamp=datetime.now(),
                function_name=function_name,
                total_time=execution_time,
                total_calls=top_functions[0]['total_calls'] if top_functions else 0,
                primitive_calls=top_functions[0]['primitive_calls'] if top_functions else 0,
                cumulative_time=top_functions[0]['cumulative_time'] if top_functions else 0,
                per_call_time=top_functions[0]['per_call_time'] if top_functions else 0,
                cumulative_per_call=top_functions[0]['cumulative_per_call'] if top_functions else 0,
                filename=top_functions[0]['filename'] if top_functions else '',
                line_number=top_functions[0]['line_number'] if top_functions else 0,
                profile_data={'top_functions': top_functions}
            )
            
            # Store result
            self.profiling_results.append(result)
            
            # Save detailed profile if output file specified
            if output_file:
                self._save_detailed_profile(profiler, output_file, sort_by)
            
            # Log summary
            self.logger.info(f"📊 Profiled {function_name}: {execution_time:.2f}ms, "
                           f"Top function: {top_functions[0]['function'] if top_functions else 'N/A'}")
            
        except Exception as e:
            self.logger.error(f"Error processing profiling result: {e}")
    
    def _save_detailed_profile(self, profiler: cProfile.Profile, output_file: str, sort_by: str):
        """Save detailed profiling output to file"""
        try:
            # Create output path
            output_path = self.output_dir / f"{output_file}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.prof"
            
            # Save raw profile data
            profiler.dump_stats(str(output_path))
            
            # Create human-readable summary
            stats = pstats.Stats(profiler)
            if sort_by == 'cumulative':
                stats.sort_stats('cumulative')
            elif sort_by == 'time':
                stats.sort_stats('time')
            elif sort_by == 'calls':
                stats.sort_stats('calls')
            
            # Capture stats output
            stats_output = io.StringIO()
            stats.print_stats(20)  # Top 20 functions
            stats_content = stats_output.getvalue()
            stats_output.close()
            
            # Save human-readable summary
            summary_path = output_path.with_suffix('.txt')
            with open(summary_path, 'w') as f:
                f.write(f"Performance Profile Summary\n")
                f.write(f"Generated: {datetime.now()}\n")
                f.write(f"Sort by: {sort_by}\n")
                f.write(f"\n{stats_content}")
            
            self.logger.info(f"💾 Detailed profile saved to {output_path}")
            self.logger.info(f"📝 Summary saved to {summary_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving detailed profile: {e}")
    
    @contextmanager
    def profile_context(self, context_name: str, output_file: str = None):
        """Context manager for profiling code blocks"""
        try:
            profiler = cProfile.Profile()
            profiler.enable()
            start_time = time.time()
            
            yield profiler
            
        finally:
            profiler.disable()
            execution_time = (time.time() - start_time) * 1000
            
            # Process results
            self._process_profiling_result(profiler, context_name, execution_time,
                                        output_file, 'cumulative')
    
    def start_profiling(self):
        """Start continuous profiling"""
        with self._profile_lock:
            if self._profiling_active:
                self.logger.warning("Profiling already active")
                return
            
            self._current_profiler = cProfile.Profile()
            self._current_profiler.enable()
            self._profiling_active = True
            self.logger.info("🚀 Continuous profiling started")
    
    def stop_profiling(self, output_file: str = None):
        """Stop continuous profiling and save results"""
        with self._profile_lock:
            if not self._profiling_active:
                self.logger.warning("No profiling active")
                return None
            
            self._current_profiler.disable()
            execution_time = time.time() * 1000  # Approximate
            
            # Process results
            self._process_profiling_result(self._current_profiler, "continuous_session",
                                        execution_time, output_file, 'cumulative')
            
            self._profiling_active = False
            self._current_profiler = None
            
            self.logger.info("⏹️ Continuous profiling stopped")
            return self.profiling_results[-1] if self.profiling_results else None
    
    def get_profiling_summary(self) -> Dict[str, Any]:
        """Get summary of all profiling results"""
        try:
            if not self.profiling_results:
                return {"message": "No profiling results available"}
            
            # Calculate statistics
            total_sessions = len(self.profiling_results)
            avg_execution_time = sum(r.total_time for r in self.profiling_results) / total_sessions
            max_execution_time = max(r.total_time for r in self.profiling_results)
            min_execution_time = min(r.total_time for r in self.profiling_results)
            
            # Group by function
            function_stats = {}
            for result in self.profiling_results:
                if result.function_name not in function_stats:
                    function_stats[result.function_name] = {
                        'total_calls': 0,
                        'total_time': 0.0,
                        'avg_time': 0.0,
                        'min_time': float('inf'),
                        'max_time': 0.0
                    }
                
                stats = function_stats[result.function_name]
                stats['total_calls'] += 1
                stats['total_time'] += result.total_time
                stats['min_time'] = min(stats['min_time'], result.total_time)
                stats['max_time'] = max(stats['max_time'], result.total_time)
                stats['avg_time'] = stats['total_time'] / stats['total_calls']
            
            return {
                "total_sessions": total_sessions,
                "overall_stats": {
                    "avg_execution_time_ms": avg_execution_time,
                    "max_execution_time_ms": max_execution_time,
                    "min_execution_time_ms": min_execution_time
                },
                "function_breakdown": function_stats,
                "recent_sessions": [
                    {
                        "session_id": r.session_id,
                        "function": r.function_name,
                        "execution_time_ms": r.total_time,
                        "timestamp": r.timestamp.isoformat()
                    }
                    for r in sorted(self.profiling_results, key=lambda x: x.timestamp, reverse=True)[:10]
                ]
            }
            
        except Exception as e:
            self.logger.error(f"Error getting profiling summary: {e}")
            return {"error": str(e)}
    
    def clear_profiling_history(self):
        """Clear profiling history"""
        self.profiling_results.clear()
        self.logger.info("🗑️ Profiling history cleared")

# Global profiler instance
performance_profiler = PerformanceProfiler()

def get_performance_profiler() -> PerformanceProfiler:
    """Get the global performance profiler instance"""
    return performance_profiler
