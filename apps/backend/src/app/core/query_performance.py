"""
Query Performance Monitoring Framework
Provides timing decorators, latency tracking, and performance diagnostics
"""

import time
import logging
import functools
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
import statistics
import asyncio

logger = logging.getLogger(__name__)

@dataclass
class QueryMetrics:
    """Metrics for a single query execution"""
    query_name: str
    execution_time_ms: float
    timestamp: datetime
    success: bool
    error_message: Optional[str] = None
    rows_affected: Optional[int] = None
    parameters: Optional[Dict] = None

@dataclass
class PerformanceStats:
    """Aggregated performance statistics"""
    query_name: str
    total_executions: int = 0
    successful_executions: int = 0
    failed_executions: int = 0
    total_time_ms: float = 0.0
    avg_time_ms: float = 0.0
    min_time_ms: float = float('inf')
    max_time_ms: float = 0.0
    p95_time_ms: float = 0.0
    p99_time_ms: float = 0.0
    recent_times: deque = field(default_factory=lambda: deque(maxlen=100))
    last_execution: Optional[datetime] = None
    error_rate: float = 0.0

class QueryPerformanceMonitor:
    """Monitor and track query performance metrics"""
    
    def __init__(self, history_size: int = 1000):
        self.history_size = history_size
        self.query_metrics: List[QueryMetrics] = []
        self.performance_stats: Dict[str, PerformanceStats] = defaultdict(
            lambda: PerformanceStats(query_name="")
        )
        self.slow_query_threshold_ms = 200  # Default threshold
        self.critical_query_threshold_ms = 1000  # Critical threshold
        
    def log_slow_queries(self, threshold_ms: int = 200):
        """Decorator to log slow queries"""
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                start_time = time.time()
                success = True
                error_message = None
                rows_affected = None
                
                try:
                    result = await func(*args, **kwargs)
                    # Try to get row count if it's a database operation
                    if hasattr(result, 'rowcount'):
                        rows_affected = result.rowcount
                    return result
                except Exception as e:
                    success = False
                    error_message = str(e)
                    raise
                finally:
                    elapsed_ms = (time.time() - start_time) * 1000
                    self._record_query_metrics(
                        func.__name__, elapsed_ms, success, error_message, rows_affected
                    )
                    
                    if elapsed_ms > threshold_ms:
                        logger.warning(
                            f"Slow query detected: {func.__name__} took {elapsed_ms:.2f} ms "
                            f"(threshold: {threshold_ms} ms)"
                        )
            
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                start_time = time.time()
                success = True
                error_message = None
                rows_affected = None
                
                try:
                    result = func(*args, **kwargs)
                    # Try to get row count if it's a database operation
                    if hasattr(result, 'rowcount'):
                        rows_affected = result.rowcount
                    return result
                except Exception as e:
                    success = False
                    error_message = str(e)
                    raise
                finally:
                    elapsed_ms = (time.time() - start_time) * 1000
                    self._record_query_metrics(
                        func.__name__, elapsed_ms, success, error_message, rows_affected
                    )
                    
                    if elapsed_ms > threshold_ms:
                        logger.warning(
                            f"Slow query detected: {func.__name__} took {elapsed_ms:.2f} ms "
                            f"(threshold: {threshold_ms} ms)"
                        )
            
            # Return async wrapper if function is async, sync wrapper otherwise
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            else:
                return sync_wrapper
        
        return decorator
    
    def _record_query_metrics(self, query_name: str, execution_time_ms: float, 
                            success: bool, error_message: Optional[str], 
                            rows_affected: Optional[int]):
        """Record metrics for a query execution"""
        metrics = QueryMetrics(
            query_name=query_name,
            execution_time_ms=execution_time_ms,
            timestamp=datetime.now(),
            success=success,
            error_message=error_message,
            rows_affected=rows_affected
        )
        
        # Add to history
        self.query_metrics.append(metrics)
        if len(self.query_metrics) > self.history_size:
            self.query_metrics.pop(0)
        
        # Update statistics
        stats = self.performance_stats[query_name]
        stats.query_name = query_name
        stats.total_executions += 1
        stats.total_time_ms += execution_time_ms
        stats.last_execution = metrics.timestamp
        
        if success:
            stats.successful_executions += 1
        else:
            stats.failed_executions += 1
        
        # Update timing statistics
        stats.recent_times.append(execution_time_ms)
        stats.min_time_ms = min(stats.min_time_ms, execution_time_ms)
        stats.max_time_ms = max(stats.max_time_ms, execution_time_ms)
        
        if stats.total_executions > 0:
            stats.avg_time_ms = stats.total_time_ms / stats.total_executions
            stats.error_rate = stats.failed_executions / stats.total_executions
        
        # Calculate percentiles
        if len(stats.recent_times) >= 5:
            times_list = list(stats.recent_times)
            stats.p95_time_ms = statistics.quantiles(times_list, n=20)[18]  # 95th percentile
            stats.p99_time_ms = statistics.quantiles(times_list, n=100)[98]  # 99th percentile
    
    def get_slow_queries(self, threshold_ms: Optional[int] = None) -> List[QueryMetrics]:
        """Get queries that exceeded the threshold"""
        if threshold_ms is None:
            threshold_ms = self.slow_query_threshold_ms
        
        return [
            metrics for metrics in self.query_metrics
            if metrics.execution_time_ms > threshold_ms
        ]
    
    def get_critical_queries(self) -> List[QueryMetrics]:
        """Get queries that exceeded the critical threshold"""
        return self.get_slow_queries(self.critical_query_threshold_ms)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get overall performance summary"""
        total_queries = len(self.query_metrics)
        if total_queries == 0:
            return {"message": "No queries recorded yet"}
        
        slow_queries = self.get_slow_queries()
        critical_queries = self.get_critical_queries()
        
        all_times = [m.execution_time_ms for m in self.query_metrics]
        
        return {
            "total_queries": total_queries,
            "slow_queries_count": len(slow_queries),
            "critical_queries_count": len(critical_queries),
            "avg_execution_time_ms": statistics.mean(all_times),
            "median_execution_time_ms": statistics.median(all_times),
            "p95_execution_time_ms": statistics.quantiles(all_times, n=20)[18] if len(all_times) >= 5 else 0,
            "p99_execution_time_ms": statistics.quantiles(all_times, n=100)[98] if len(all_times) >= 5 else 0,
            "slow_query_threshold_ms": self.slow_query_threshold_ms,
            "critical_query_threshold_ms": self.critical_query_threshold_ms
        }
    
    def get_query_statistics(self, query_name: str) -> Optional[PerformanceStats]:
        """Get detailed statistics for a specific query"""
        return self.performance_stats.get(query_name)
    
    def clear_history(self):
        """Clear query history"""
        self.query_metrics.clear()
        self.performance_stats.clear()

# Global instance
query_monitor = QueryPerformanceMonitor()

# Convenience decorator
def log_slow_queries(threshold_ms: int = 200):
    """Decorator to log slow queries using the global monitor"""
    return query_monitor.log_slow_queries(threshold_ms)
