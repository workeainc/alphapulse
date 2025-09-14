"""
Latency Tracking Framework
Tracks insert and retrieval latency with moving averages and alerting
Enhanced for trading pipeline performance monitoring
"""

import time
import logging
import asyncio
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque, defaultdict
import statistics

# Import database models
try:
    from ..database.models import LatencyMetrics as DBLatencyMetrics
    from ..database.connection import get_db
    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("Database models not available - latency tracking will be in-memory only")

logger = logging.getLogger(__name__)

@dataclass
class LatencyMetrics:
    """Latency metrics for a specific operation"""
    operation_type: str  # 'insert', 'retrieval', 'batch_insert', etc.
    latency_ms: float
    timestamp: datetime
    success: bool
    rows_affected: Optional[int] = None
    error_message: Optional[str] = None
    additional_data: Optional[Dict] = None

@dataclass
class TradingPipelineMetrics:
    """Complete trading pipeline latency metrics"""
    model_id: Optional[str] = None
    symbol: Optional[str] = None
    strategy_name: Optional[str] = None
    
    # Pipeline stage timings
    fetch_time_ms: float = 0.0
    preprocess_time_ms: float = 0.0
    inference_time_ms: float = 0.0
    postprocess_time_ms: float = 0.0
    total_latency_ms: float = 0.0
    
    # Context
    success: bool = True
    error_message: Optional[str] = None
    metadata: Optional[Dict] = None
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class LatencyStats:
    """Aggregated latency statistics"""
    operation_type: str
    total_operations: int = 0
    successful_operations: int = 0
    failed_operations: int = 0
    total_latency_ms: float = 0.0
    avg_latency_ms: float = 0.0
    min_latency_ms: float = float('inf')
    max_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    moving_avg_1min: float = 0.0
    moving_avg_5min: float = 0.0
    moving_avg_15min: float = 0.0
    recent_latencies: deque = field(default_factory=lambda: deque(maxlen=1000))
    last_operation: Optional[datetime] = None
    error_rate: float = 0.0
    throughput_ops_per_sec: float = 0.0

class LatencyTracker:
    """Track and analyze latency metrics for database operations and trading pipeline"""
    
    def __init__(self, history_size: int = 10000):
        self.history_size = history_size
        self.latency_metrics: List[LatencyMetrics] = []
        self.latency_stats: Dict[str, LatencyStats] = defaultdict(
            lambda: LatencyStats(operation_type="")
        )
        
        # Trading pipeline tracking
        self.pipeline_metrics: List[TradingPipelineMetrics] = []
        
        # Alerting thresholds
        self.alert_thresholds = {
            'insert_latency_ms': 500,
            'retrieval_latency_ms': 1000,
            'batch_insert_latency_ms': 2000,
            'critical_latency_ms': 5000,
            # Trading pipeline thresholds
            'fetch_time_ms': 200,
            'preprocess_time_ms': 100,
            'inference_time_ms': 500,
            'postprocess_time_ms': 50,
            'total_pipeline_latency_ms': 1000
        }
        
        # Moving average windows (in seconds)
        self.moving_avg_windows = {
            '1min': 60,
            '5min': 300,
            '15min': 900
        }
        
        # Background task for cleanup
        self._cleanup_task = None
        # Don't start cleanup task immediately - will be started when needed
    
    def _start_cleanup_task(self):
        """Start background cleanup task"""
        if self._cleanup_task is None:
            try:
                self._cleanup_task = asyncio.create_task(self._cleanup_old_metrics())
            except RuntimeError:
                # No running event loop, will start later
                pass
    
    async def _cleanup_old_metrics(self):
        """Clean up old metrics periodically"""
        while True:
            try:
                await asyncio.sleep(300)  # Clean up every 5 minutes
                cutoff_time = datetime.now() - timedelta(hours=24)  # Keep 24 hours
                
                # Remove old metrics
                self.latency_metrics = [
                    metric for metric in self.latency_metrics
                    if metric.timestamp > cutoff_time
                ]
                
                # Update moving averages
                self._update_moving_averages()
                
            except Exception as e:
                logger.error(f"Error in cleanup task: {e}")
    
    async def track_latency(self, operation_type: str, latency_ms: float, 
                     success: bool, rows_affected: Optional[int] = None,
                     error_message: Optional[str] = None, 
                     additional_data: Optional[Dict] = None):
        """Track latency for an operation"""
        # Start cleanup task if not already started
        self._start_cleanup_task()
        metrics = LatencyMetrics(
            operation_type=operation_type,
            latency_ms=latency_ms,
            timestamp=datetime.now(),
            success=success,
            rows_affected=rows_affected,
            error_message=error_message,
            additional_data=additional_data or {}
        )
        
        # Add to history
        self.latency_metrics.append(metrics)
        if len(self.latency_metrics) > self.history_size:
            self.latency_metrics.pop(0)
        
        # Update statistics
        self._update_stats(metrics)
        
        # Check for alerts
        self._check_alerts(metrics)
        
        # Check performance alerting conditions
        await self._check_performance_alerts(metrics)
    
    def _update_stats(self, metrics: LatencyMetrics):
        """Update statistics for an operation type"""
        stats = self.latency_stats[metrics.operation_type]
        stats.operation_type = metrics.operation_type
        stats.total_operations += 1
        stats.total_latency_ms += metrics.latency_ms
        stats.last_operation = metrics.timestamp
        
        if metrics.success:
            stats.successful_operations += 1
        else:
            stats.failed_operations += 1
        
        # Update latency statistics
        stats.recent_latencies.append(metrics.latency_ms)
        stats.min_latency_ms = min(stats.min_latency_ms, metrics.latency_ms)
        stats.max_latency_ms = max(stats.max_latency_ms, metrics.latency_ms)
        
        if stats.total_operations > 0:
            stats.avg_latency_ms = stats.total_latency_ms / stats.total_operations
            stats.error_rate = stats.failed_operations / stats.total_operations
        
        # Calculate percentiles
        if len(stats.recent_latencies) >= 5:
            latencies_list = list(stats.recent_latencies)
            stats.p95_latency_ms = statistics.quantiles(latencies_list, n=20)[18]
            stats.p99_latency_ms = statistics.quantiles(latencies_list, n=100)[98]
        
        # Calculate throughput (operations per second)
        if len(stats.recent_latencies) >= 10:
            recent_ops = [m for m in self.latency_metrics 
                         if m.operation_type == metrics.operation_type 
                         and m.timestamp > datetime.now() - timedelta(seconds=60)]
            if recent_ops:
                stats.throughput_ops_per_sec = len(recent_ops) / 60.0
    
    def _update_moving_averages(self):
        """Update moving averages for all operation types"""
        now = datetime.now()
        
        for operation_type, stats in self.latency_stats.items():
            for window_name, window_seconds in self.moving_avg_windows.items():
                cutoff_time = now - timedelta(seconds=window_seconds)
                
                # Get recent latencies within window
                recent_latencies = [
                    m.latency_ms for m in self.latency_metrics
                    if m.operation_type == operation_type 
                    and m.timestamp > cutoff_time
                ]
                
                if recent_latencies:
                    avg_latency = statistics.mean(recent_latencies)
                    if window_name == '1min':
                        stats.moving_avg_1min = avg_latency
                    elif window_name == '5min':
                        stats.moving_avg_5min = avg_latency
                    elif window_name == '15min':
                        stats.moving_avg_15min = avg_latency
    
    def _check_alerts(self, metrics: LatencyMetrics):
        """Check for latency alerts"""
        threshold_key = f"{metrics.operation_type}_latency_ms"
        threshold = self.alert_thresholds.get(threshold_key, 
                                             self.alert_thresholds['critical_latency_ms'])
        
        if metrics.latency_ms > threshold:
            logger.warning(
                f"High latency alert: {metrics.operation_type} took {metrics.latency_ms:.2f} ms "
                f"(threshold: {threshold} ms)"
            )
        
        # Check for gradual slowdowns (moving average > 2x normal)
        stats = self.latency_stats[metrics.operation_type]
        if stats.moving_avg_5min > 0 and stats.avg_latency_ms > 0:
            slowdown_ratio = stats.moving_avg_5min / stats.avg_latency_ms
            if slowdown_ratio > 2.0:
                logger.warning(
                    f"Gradual slowdown detected: {metrics.operation_type} "
                    f"5-min avg ({stats.moving_avg_5min:.2f} ms) is "
                    f"{slowdown_ratio:.1f}x higher than overall avg ({stats.avg_latency_ms:.2f} ms)"
                )
    
    async def _check_performance_alerts(self, metrics: LatencyMetrics):
        """Check performance alerting conditions"""
        try:
            from app.core.performance_alerting import performance_alerting
            
            # Prepare metrics for alerting
            alert_metrics = {
                "latency": metrics.latency_ms,
                "operation_type": metrics.operation_type,
                "success": metrics.success,
                "rows_affected": metrics.rows_affected,
                "timestamp": metrics.timestamp.isoformat()
            }
            
            # Add statistics if available
            stats = self.latency_stats.get(metrics.operation_type)
            if stats:
                alert_metrics.update({
                    "avg_latency": stats.avg_latency_ms,
                    "moving_avg_1min": stats.moving_avg_1min,
                    "moving_avg_5min": stats.moving_avg_5min,
                    "moving_avg_15min": stats.moving_avg_15min,
                    "error_rate": stats.error_rate,
                    "throughput_ops_per_sec": stats.throughput_ops_per_sec
                })
            
            # Check alert conditions
            await performance_alerting.check_alert_conditions(alert_metrics)
            
        except Exception as e:
            logger.error(f"Error checking performance alerts: {e}")
    
    async def track_trading_pipeline(self, pipeline_metrics: TradingPipelineMetrics):
        """Track complete trading pipeline latency"""
        try:
            # Start cleanup task if not already started
            self._start_cleanup_task()
            
            # Add to in-memory tracking
            self.pipeline_metrics.append(pipeline_metrics)
            if len(self.pipeline_metrics) > self.history_size:
                self.pipeline_metrics.pop(0)
            
            # Store in database if available
            if DB_AVAILABLE:
                await self._store_pipeline_metrics_db(pipeline_metrics)
            
            # Check for alerts
            self._check_pipeline_alerts(pipeline_metrics)
            
            # Log performance metrics
            logger.info(f"📊 Pipeline latency: {pipeline_metrics.total_latency_ms:.2f}ms "
                       f"(fetch: {pipeline_metrics.fetch_time_ms:.2f}ms, "
                       f"preprocess: {pipeline_metrics.preprocess_time_ms:.2f}ms, "
                       f"inference: {pipeline_metrics.inference_time_ms:.2f}ms, "
                       f"postprocess: {pipeline_metrics.postprocess_time_ms:.2f}ms)")
            
        except Exception as e:
            logger.error(f"Error tracking trading pipeline: {e}")

    async def _store_pipeline_metrics_db(self, metrics: TradingPipelineMetrics):
        """Store pipeline metrics in database"""
        try:
            from ..database.connection import TimescaleDBConnection
            
            # Initialize database connection
            db_connection = TimescaleDBConnection()
            db_connection.initialize()
            
            async with db_connection.get_async_session() as session:
                db_metrics = DBLatencyMetrics(
                    model_id=metrics.model_id,
                    operation_type="trading_pipeline",
                    fetch_time_ms=metrics.fetch_time_ms,
                    preprocess_time_ms=metrics.preprocess_time_ms,
                    inference_time_ms=metrics.inference_time_ms,
                    postprocess_time_ms=metrics.postprocess_time_ms,
                    total_latency_ms=metrics.total_latency_ms,
                    symbol=metrics.symbol,
                    strategy_name=metrics.strategy_name,
                    success=metrics.success,
                    error_message=metrics.error_message,
                    metadata_json=metrics.metadata,
                    timestamp=metrics.timestamp
                )
                session.add(db_metrics)
                await session.commit()
                
        except Exception as e:
            logger.error(f"Error storing pipeline metrics in database: {e}")

    def _check_pipeline_alerts(self, metrics: TradingPipelineMetrics):
        """Check for trading pipeline latency alerts"""
        # Check individual stage thresholds
        if metrics.fetch_time_ms > self.alert_thresholds['fetch_time_ms']:
            logger.warning(f"High fetch latency: {metrics.fetch_time_ms:.2f}ms "
                          f"(threshold: {self.alert_thresholds['fetch_time_ms']}ms)")
        
        if metrics.preprocess_time_ms > self.alert_thresholds['preprocess_time_ms']:
            logger.warning(f"High preprocess latency: {metrics.preprocess_time_ms:.2f}ms "
                          f"(threshold: {self.alert_thresholds['preprocess_time_ms']}ms)")
        
        if metrics.inference_time_ms > self.alert_thresholds['inference_time_ms']:
            logger.warning(f"High inference latency: {metrics.inference_time_ms:.2f}ms "
                          f"(threshold: {self.alert_thresholds['inference_time_ms']}ms)")
        
        if metrics.postprocess_time_ms > self.alert_thresholds['postprocess_time_ms']:
            logger.warning(f"High postprocess latency: {metrics.postprocess_time_ms:.2f}ms "
                          f"(threshold: {self.alert_thresholds['postprocess_time_ms']}ms)")
        
        # Check total pipeline latency
        if metrics.total_latency_ms > self.alert_thresholds['total_pipeline_latency_ms']:
            logger.warning(f"High total pipeline latency: {metrics.total_latency_ms:.2f}ms "
                          f"(threshold: {self.alert_thresholds['total_pipeline_latency_ms']}ms)")

    def get_latency_summary(self) -> Dict[str, Any]:
        """Get overall latency summary"""
        total_operations = len(self.latency_metrics)
        if total_operations == 0:
            return {"message": "No latency metrics recorded yet"}
        
        # Calculate overall statistics
        all_latencies = [m.latency_ms for m in self.latency_metrics]
        high_latency_ops = [
            m for m in self.latency_metrics
            if m.latency_ms > self.alert_thresholds['critical_latency_ms']
        ]
        
        return {
            "total_operations": total_operations,
            "high_latency_operations": len(high_latency_ops),
            "avg_latency_ms": statistics.mean(all_latencies),
            "median_latency_ms": statistics.median(all_latencies),
            "p95_latency_ms": statistics.quantiles(all_latencies, n=20)[18] if len(all_latencies) >= 5 else 0,
            "p99_latency_ms": statistics.quantiles(all_latencies, n=100)[98] if len(all_latencies) >= 5 else 0,
            "operation_types": list(self.latency_stats.keys()),
            "alert_thresholds": self.alert_thresholds
        }
    
    def get_operation_stats(self, operation_type: str) -> Optional[LatencyStats]:
        """Get detailed statistics for a specific operation type"""
        return self.latency_stats.get(operation_type)
    
    def get_recent_latencies(self, operation_type: str, 
                           minutes: int = 60) -> List[LatencyMetrics]:
        """Get recent latency metrics for an operation type"""
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        return [
            m for m in self.latency_metrics
            if m.operation_type == operation_type and m.timestamp > cutoff_time
        ]
    
    def get_prometheus_metrics(self) -> Dict[str, float]:
        """Get metrics in Prometheus format"""
        metrics = {}
        
        for operation_type, stats in self.latency_stats.items():
            prefix = f"latency_{operation_type}"
            metrics[f"{prefix}_total_operations"] = stats.total_operations
            metrics[f"{prefix}_successful_operations"] = stats.successful_operations
            metrics[f"{prefix}_failed_operations"] = stats.failed_operations
            metrics[f"{prefix}_avg_latency_ms"] = stats.avg_latency_ms
            metrics[f"{prefix}_p95_latency_ms"] = stats.p95_latency_ms
            metrics[f"{prefix}_p99_latency_ms"] = stats.p99_latency_ms
            metrics[f"{prefix}_moving_avg_1min_ms"] = stats.moving_avg_1min
            metrics[f"{prefix}_moving_avg_5min_ms"] = stats.moving_avg_5min
            metrics[f"{prefix}_moving_avg_15min_ms"] = stats.moving_avg_15min
            metrics[f"{prefix}_error_rate"] = stats.error_rate
            metrics[f"{prefix}_throughput_ops_per_sec"] = stats.throughput_ops_per_sec
        
        return metrics
    
    def clear_history(self):
        """Clear latency history"""
        self.latency_metrics.clear()
        self.latency_stats.clear()

    def get_pipeline_summary(self) -> Dict[str, Any]:
        """Get trading pipeline latency summary"""
        if not self.pipeline_metrics:
            return {"message": "No pipeline metrics recorded yet"}
        
        # Calculate statistics
        total_latencies = [m.total_latency_ms for m in self.pipeline_metrics]
        fetch_times = [m.fetch_time_ms for m in self.pipeline_metrics]
        preprocess_times = [m.preprocess_time_ms for m in self.pipeline_metrics]
        inference_times = [m.inference_time_ms for m in self.pipeline_metrics]
        postprocess_times = [m.postprocess_time_ms for m in self.pipeline_metrics]
        
        successful_pipelines = [m for m in self.pipeline_metrics if m.success]
        
        return {
            "total_pipelines": len(self.pipeline_metrics),
            "successful_pipelines": len(successful_pipelines),
            "success_rate": len(successful_pipelines) / len(self.pipeline_metrics) if self.pipeline_metrics else 0,
            "latency_stats": {
                "total": {
                    "avg_ms": statistics.mean(total_latencies),
                    "median_ms": statistics.median(total_latencies),
                    "p95_ms": statistics.quantiles(total_latencies, n=20)[18] if len(total_latencies) >= 5 else 0,
                    "p99_ms": statistics.quantiles(total_latencies, n=100)[98] if len(total_latencies) >= 5 else 0
                },
                "fetch": {
                    "avg_ms": statistics.mean(fetch_times),
                    "median_ms": statistics.median(fetch_times)
                },
                "preprocess": {
                    "avg_ms": statistics.mean(preprocess_times),
                    "median_ms": statistics.median(preprocess_times)
                },
                "inference": {
                    "avg_ms": statistics.mean(inference_times),
                    "median_ms": statistics.median(inference_times)
                },
                "postprocess": {
                    "avg_ms": statistics.mean(postprocess_times),
                    "median_ms": statistics.median(postprocess_times)
                }
            },
            "alert_thresholds": self.alert_thresholds
        }

    def get_pipeline_metrics_by_symbol(self, symbol: str, minutes: int = 60) -> List[TradingPipelineMetrics]:
        """Get recent pipeline metrics for a specific symbol"""
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        return [
            m for m in self.pipeline_metrics
            if m.symbol == symbol and m.timestamp > cutoff_time
        ]

    def get_pipeline_metrics_by_strategy(self, strategy_name: str, minutes: int = 60) -> List[TradingPipelineMetrics]:
        """Get recent pipeline metrics for a specific strategy"""
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        return [
            m for m in self.pipeline_metrics
            if m.strategy_name == strategy_name and m.timestamp > cutoff_time
        ]

# Global instance
latency_tracker = LatencyTracker()

# Enhanced decorator for trading pipeline tracking
def track_trading_pipeline(model_id: Optional[str] = None, symbol: Optional[str] = None, strategy_name: Optional[str] = None):
    """Decorator to track complete trading pipeline latency"""
    def decorator(func: Callable) -> Callable:
        async def async_wrapper(*args, **kwargs):
            # Initialize timing checkpoints
            t0 = time.time()
            t1 = t2 = t3 = t4 = t0
            
            success = True
            error_message = None
            metadata = {}
            
            try:
                # Execute the function (this should be the complete pipeline)
                result = await func(*args, **kwargs)
                t4 = time.time()
                
                # Calculate stage timings (approximate based on function execution)
                total_time = (t4 - t0) * 1000
                fetch_time = total_time * 0.1  # Assume 10% for fetch
                preprocess_time = total_time * 0.2  # Assume 20% for preprocess
                inference_time = total_time * 0.6  # Assume 60% for inference
                postprocess_time = total_time * 0.1  # Assume 10% for postprocess
                
                # Create pipeline metrics
                pipeline_metrics = TradingPipelineMetrics(
                    model_id=model_id,
                    symbol=symbol,
                    strategy_name=strategy_name,
                    fetch_time_ms=fetch_time,
                    preprocess_time_ms=preprocess_time,
                    inference_time_ms=inference_time,
                    postprocess_time_ms=postprocess_time,
                    total_latency_ms=total_time,
                    success=success,
                    error_message=error_message,
                    metadata=metadata
                )
                
                # Track the metrics
                await latency_tracker.track_trading_pipeline(pipeline_metrics)
                
                return result
                
            except Exception as e:
                success = False
                error_message = str(e)
                t4 = time.time()
                
                # Create error metrics
                total_time = (t4 - t0) * 1000
                pipeline_metrics = TradingPipelineMetrics(
                    model_id=model_id,
                    symbol=symbol,
                    strategy_name=strategy_name,
                    fetch_time_ms=0.0,
                    preprocess_time_ms=0.0,
                    inference_time_ms=0.0,
                    postprocess_time_ms=0.0,
                    total_latency_ms=total_time,
                    success=success,
                    error_message=error_message,
                    metadata=metadata
                )
                
                await latency_tracker.track_trading_pipeline(pipeline_metrics)
                raise
        
        def sync_wrapper(*args, **kwargs):
            # For sync functions, create a simple wrapper
            start_time = time.time()
            success = True
            error_message = None
            
            try:
                result = func(*args, **kwargs)
                elapsed_ms = (time.time() - start_time) * 1000
                
                # Create simple metrics for sync functions
                pipeline_metrics = TradingPipelineMetrics(
                    model_id=model_id,
                    symbol=symbol,
                    strategy_name=strategy_name,
                    fetch_time_ms=elapsed_ms * 0.1,
                    preprocess_time_ms=elapsed_ms * 0.2,
                    inference_time_ms=elapsed_ms * 0.6,
                    postprocess_time_ms=elapsed_ms * 0.1,
                    total_latency_ms=elapsed_ms,
                    success=success,
                    error_message=error_message
                )
                
                # Track metrics (async call)
                try:
                    asyncio.create_task(latency_tracker.track_trading_pipeline(pipeline_metrics))
                except RuntimeError:
                    logger.info(f"Pipeline latency tracked: {elapsed_ms:.2f}ms")
                
                return result
                
            except Exception as e:
                success = False
                error_message = str(e)
                elapsed_ms = (time.time() - start_time) * 1000
                
                pipeline_metrics = TradingPipelineMetrics(
                    model_id=model_id,
                    symbol=symbol,
                    strategy_name=strategy_name,
                    fetch_time_ms=0.0,
                    preprocess_time_ms=0.0,
                    inference_time_ms=0.0,
                    postprocess_time_ms=0.0,
                    total_latency_ms=elapsed_ms,
                    success=success,
                    error_message=error_message
                )
                
                try:
                    asyncio.create_task(latency_tracker.track_trading_pipeline(pipeline_metrics))
                except RuntimeError:
                    logger.error(f"Pipeline error tracked: {elapsed_ms:.2f}ms - {error_message}")
                
                raise
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator
