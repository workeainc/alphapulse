"""
Stream Metrics for AlphaPulse
Performance metrics tracking and monitoring for streaming infrastructure
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from collections import defaultdict, deque
import psutil
import threading

# Import existing components
try:
    from .stream_buffer import StreamBuffer, stream_buffer
    from .stream_normalizer import StreamNormalizer, stream_normalizer
    from .candle_builder import CandleBuilder, candle_builder
    from .rolling_state_manager import RollingStateManager, rolling_state_manager
    from .stream_processor import StreamProcessor, stream_processor
    from ..database.connection import TimescaleDBConnection
    from ..core.config import settings
except ImportError:
    # Fallback for standalone testing
    try:
        from stream_buffer import StreamBuffer, stream_buffer
        from stream_normalizer import StreamNormalizer, stream_normalizer
        from candle_builder import CandleBuilder, candle_builder
        from rolling_state_manager import RollingStateManager, rolling_state_manager
        from stream_processor import StreamProcessor, stream_processor
        from database.connection import TimescaleDBConnection
        from core.config import settings
    except ImportError:
        # Minimal fallback classes for testing
        from dataclasses import dataclass
        from datetime import datetime, timezone
        from typing import Dict, Any, Optional, List
        from collections import deque
        
        @dataclass
        class StreamBuffer:
            def __init__(self, config=None): 
                self.metrics = type('obj', (object,), {
                    'messages_received': 0,
                    'messages_processed': 0,
                    'messages_failed': 0,
                    'avg_processing_time_ms': 0.0,
                    'throughput_mps': 0.0,
                    'buffer_size': 0,
                    'last_message_time': None,
                    'error_count': 0,
                    'reconnection_count': 0
                })()
                self.is_connected = False
                self.is_running = False
                self.message_buffer = []
            async def initialize(self): pass
            async def shutdown(self): pass
            def get_metrics(self):
                return {
                    'messages_received': 0,
                    'messages_processed': 0,
                    'messages_failed': 0,
                    'avg_processing_time_ms': 0.0,
                    'throughput_mps': 0.0,
                    'buffer_size': 0,
                    'last_message_time': None,
                    'error_count': 0,
                    'reconnection_count': 0,
                    'is_connected': False,
                    'is_running': False,
                    'message_buffer_size': 0
                }
        
        @dataclass
        class StreamNormalizer:
            def __init__(self, config=None): pass
            async def initialize(self): pass
            async def shutdown(self): pass
        
        @dataclass
        class CandleBuilder:
            def __init__(self, config=None): 
                self.timeframes = ['1m', '5m', '15m', '1h', '4h', '1d']
            async def initialize(self): pass
            async def shutdown(self): pass
        
        @dataclass
        class RollingStateManager:
            def __init__(self, config=None): pass
            async def initialize(self): pass
            async def shutdown(self): pass
        
        @dataclass
        class StreamProcessor:
            def __init__(self, config=None): pass
            async def initialize(self): pass
            async def shutdown(self): pass
        
        @dataclass
        class TimescaleDBConnection:
            def __init__(self): pass
            async def initialize(self): pass
            async def close(self): pass
        
        class settings:
            DATABASE_HOST = 'localhost'
            DATABASE_PORT = 5432
            DATABASE_NAME = 'alphapulse'
            DATABASE_USER = 'alpha_emon'
            DATABASE_PASSWORD = 'Emon_@17711'
            TIMESCALEDB_HOST = 'localhost'
            TIMESCALEDB_PORT = 5432
            TIMESCALEDB_DATABASE = 'alphapulse'
            TIMESCALEDB_USERNAME = 'alpha_emon'
            TIMESCALEDB_PASSWORD = 'Emon_@17711'
        
        # Create fallback instances
        stream_buffer = StreamBuffer()
        stream_normalizer = StreamNormalizer()
        candle_builder = CandleBuilder()
        rolling_state_manager = RollingStateManager()
        stream_processor = StreamProcessor()

logger = logging.getLogger(__name__)

@dataclass
class SystemMetrics:
    """System performance metrics"""
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    memory_available_mb: float
    disk_usage_percent: float
    network_bytes_sent: int
    network_bytes_recv: int
    timestamp: datetime

@dataclass
class StreamMetricsData:
    """Comprehensive stream metrics data"""
    timestamp: datetime
    system_metrics: SystemMetrics
    component_metrics: Dict[str, Dict[str, Any]]
    aggregated_metrics: Dict[str, Any]
    alerts: List[str] = field(default_factory=list)

@dataclass
class MetricsThreshold:
    """Metrics threshold configuration"""
    metric_name: str
    threshold_value: float
    comparison: str  # 'gt', 'lt', 'eq', 'gte', 'lte'
    severity: str  # 'info', 'warning', 'error', 'critical'
    description: str

class StreamMetrics:
    """
    Stream metrics collection and monitoring
    
    Features:
    - Real-time metrics collection from all components
    - System performance monitoring
    - Threshold-based alerting
    - Historical metrics storage
    - TimescaleDB integration
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Metrics collection settings
        self.collection_interval = self.config.get('collection_interval', 5.0)  # seconds
        self.retention_hours = self.config.get('retention_hours', 24)
        self.max_history_size = self.config.get('max_history_size', 10000)
        
        # Alerting settings
        self.enable_alerts = self.config.get('enable_alerts', True)
        self.alert_thresholds = self.config.get('alert_thresholds', {
            'cpu_percent': MetricsThreshold('cpu_percent', 80.0, 'gt', 'warning', 'High CPU usage'),
            'memory_percent': MetricsThreshold('memory_percent', 85.0, 'gt', 'warning', 'High memory usage'),
            'processing_latency_ms': MetricsThreshold('processing_latency_ms', 100.0, 'gt', 'error', 'High processing latency'),
            'error_rate': MetricsThreshold('error_rate', 0.05, 'gt', 'error', 'High error rate'),
            'queue_size': MetricsThreshold('queue_size', 1000, 'gt', 'warning', 'Large processing queue')
        })
        
        # Performance settings
        self.batch_size = self.config.get('batch_size', 100)
        self.db_flush_interval = self.config.get('db_flush_interval', 30.0)  # seconds
        
        # State management
        self.is_running = False
        self.metrics_history = deque(maxlen=self.max_history_size)
        self.alert_history = deque(maxlen=1000)
        self.last_system_metrics = None
        
        # Component references
        self.stream_buffer = stream_buffer
        self.stream_normalizer = stream_normalizer
        self.candle_builder = candle_builder
        self.rolling_state_manager = rolling_state_manager
        self.stream_processor = stream_processor
        
        # TimescaleDB integration
        self.timescaledb = None
        self.db_batch = []
        self.db_flush_task = None
        
        # Alert callbacks
        self.alert_callbacks = []
        
        # Background tasks
        self.collection_task = None
        self.system_monitor_task = None
        
        logger.info("StreamMetrics initialized")
    
    async def initialize(self):
        """Initialize the metrics collector"""
        try:
            # Initialize TimescaleDB connection
            await self._initialize_timescaledb()
            
            # Start background tasks
            await self._start_background_tasks()
            
            self.is_running = True
            logger.info("✅ StreamMetrics initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize StreamMetrics: {e}")
            raise
    
    async def _initialize_timescaledb(self):
        """Initialize TimescaleDB connection"""
        try:
            self.timescaledb = TimescaleDBConnection({
                'host': settings.TIMESCALEDB_HOST,
                'port': settings.TIMESCALEDB_PORT,
                'database': settings.TIMESCALEDB_DATABASE,
                'username': settings.TIMESCALEDB_USERNAME,
                'password': settings.TIMESCALEDB_PASSWORD,
                'pool_size': 5,
                'max_overflow': 10
            })
            
            await self.timescaledb.initialize()
            logger.info("✅ TimescaleDB connection initialized for stream metrics")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize TimescaleDB: {e}")
            self.timescaledb = None
    
    async def _start_background_tasks(self):
        """Start background metrics collection tasks"""
        # Start metrics collection
        self.collection_task = asyncio.create_task(self._metrics_collection_loop())
        
        # Start system monitoring
        self.system_monitor_task = asyncio.create_task(self._system_monitoring_loop())
        
        # Start DB flush task
        if self.timescaledb:
            self.db_flush_task = asyncio.create_task(self._db_flush_loop())
        
        logger.info("✅ Background tasks started")
    
    async def _metrics_collection_loop(self):
        """Periodic metrics collection loop"""
        while self.is_running:
            try:
                await self._collect_metrics()
                await asyncio.sleep(self.collection_interval)
            except Exception as e:
                logger.error(f"Metrics collection error: {e}")
    
    async def _system_monitoring_loop(self):
        """System monitoring loop"""
        while self.is_running:
            try:
                await self._collect_system_metrics()
                await asyncio.sleep(10)  # System metrics every 10 seconds
            except Exception as e:
                logger.error(f"System monitoring error: {e}")
    
    async def _db_flush_loop(self):
        """Periodic database flush loop"""
        while self.is_running:
            try:
                if self.db_batch:
                    await self._flush_db_batch()
                await asyncio.sleep(self.db_flush_interval)
            except Exception as e:
                logger.error(f"DB flush error: {e}")
    
    async def _collect_metrics(self):
        """Collect metrics from all components"""
        try:
            timestamp = datetime.now(timezone.utc)
            
            # Collect component metrics
            component_metrics = {
                'stream_buffer': self.stream_buffer.get_metrics() if self.stream_buffer else {},
                'stream_normalizer': self.stream_normalizer.get_metrics() if self.stream_normalizer else {},
                'candle_builder': self.candle_builder.get_metrics() if self.candle_builder else {},
                'rolling_state_manager': self.rolling_state_manager.get_metrics() if self.rolling_state_manager else {},
                'stream_processor': self.stream_processor.get_metrics() if self.stream_processor else {}
            }
            
            # Calculate aggregated metrics
            aggregated_metrics = await self._calculate_aggregated_metrics(component_metrics)
            
            # Check for alerts
            alerts = await self._check_alerts(aggregated_metrics)
            
            # Create metrics data
            metrics_data = StreamMetricsData(
                timestamp=timestamp,
                system_metrics=self.last_system_metrics,
                component_metrics=component_metrics,
                aggregated_metrics=aggregated_metrics,
                alerts=alerts
            )
            
            # Store in history
            self.metrics_history.append(metrics_data)
            
            # Add to DB batch
            if self.timescaledb:
                self.db_batch.append(metrics_data)
                if len(self.db_batch) >= self.batch_size:
                    await self._flush_db_batch()
            
            # Trigger alerts
            if alerts:
                await self._trigger_alerts(alerts)
            
            logger.debug(f"Collected metrics: {len(component_metrics)} components, {len(alerts)} alerts")
            
        except Exception as e:
            logger.error(f"Error collecting metrics: {e}")
    
    async def _collect_system_metrics(self):
        """Collect system performance metrics"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_used_mb = memory.used / (1024 * 1024)
            memory_available_mb = memory.available / (1024 * 1024)
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_usage_percent = disk.percent
            
            # Network usage
            network = psutil.net_io_counters()
            network_bytes_sent = network.bytes_sent
            network_bytes_recv = network.bytes_recv
            
            self.last_system_metrics = SystemMetrics(
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                memory_used_mb=memory_used_mb,
                memory_available_mb=memory_available_mb,
                disk_usage_percent=disk_usage_percent,
                network_bytes_sent=network_bytes_sent,
                network_bytes_recv=network_bytes_recv,
                timestamp=datetime.now(timezone.utc)
            )
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
    
    async def _calculate_aggregated_metrics(self, component_metrics: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate aggregated metrics from component metrics"""
        try:
            aggregated = {
                'total_messages_processed': 0,
                'total_messages_failed': 0,
                'avg_processing_latency_ms': 0.0,
                'total_throughput_mps': 0.0,
                'total_memory_usage_mb': 0.0,
                'total_active_connections': 0,
                'total_error_rate': 0.0
            }
            
            # Aggregate message counts
            for component, metrics in component_metrics.items():
                if 'messages_processed' in metrics:
                    aggregated['total_messages_processed'] += metrics['messages_processed']
                if 'messages_failed' in metrics:
                    aggregated['total_messages_failed'] += metrics['messages_failed']
                if 'avg_processing_time_ms' in metrics:
                    # Weighted average of processing times
                    weight = metrics.get('messages_processed', 1)
                    aggregated['avg_processing_latency_ms'] += metrics['avg_processing_time_ms'] * weight
                if 'throughput_mps' in metrics:
                    aggregated['total_throughput_mps'] += metrics['throughput_mps']
                if 'memory_usage_mb' in metrics:
                    aggregated['total_memory_usage_mb'] += metrics['memory_usage_mb']
                if 'is_connected' in metrics and metrics['is_connected']:
                    aggregated['total_active_connections'] += 1
            
            # Calculate error rate
            total_messages = aggregated['total_messages_processed'] + aggregated['total_messages_failed']
            if total_messages > 0:
                aggregated['total_error_rate'] = aggregated['total_messages_failed'] / total_messages
            
            # Normalize processing latency
            if aggregated['total_messages_processed'] > 0:
                aggregated['avg_processing_latency_ms'] /= aggregated['total_messages_processed']
            
            return aggregated
            
        except Exception as e:
            logger.error(f"Error calculating aggregated metrics: {e}")
            return {}
    
    async def _check_alerts(self, aggregated_metrics: Dict[str, Any]) -> List[str]:
        """Check for threshold-based alerts"""
        alerts = []
        
        try:
            for threshold in self.alert_thresholds.values():
                metric_value = aggregated_metrics.get(threshold.metric_name)
                
                if metric_value is None:
                    continue
                
                # Check threshold
                triggered = False
                if threshold.comparison == 'gt' and metric_value > threshold.threshold_value:
                    triggered = True
                elif threshold.comparison == 'lt' and metric_value < threshold.threshold_value:
                    triggered = True
                elif threshold.comparison == 'eq' and metric_value == threshold.threshold_value:
                    triggered = True
                elif threshold.comparison == 'gte' and metric_value >= threshold.threshold_value:
                    triggered = True
                elif threshold.comparison == 'lte' and metric_value <= threshold.threshold_value:
                    triggered = True
                
                if triggered:
                    alert_message = f"{threshold.severity.upper()}: {threshold.description} - {threshold.metric_name}={metric_value}"
                    alerts.append(alert_message)
                    
                    # Store in alert history
                    self.alert_history.append({
                        'timestamp': datetime.now(timezone.utc),
                        'severity': threshold.severity,
                        'metric': threshold.metric_name,
                        'value': metric_value,
                        'threshold': threshold.threshold_value,
                        'description': threshold.description
                    })
            
        except Exception as e:
            logger.error(f"Error checking alerts: {e}")
        
        return alerts
    
    async def _trigger_alerts(self, alerts: List[str]):
        """Trigger alert callbacks"""
        for alert in alerts:
            for callback in self.alert_callbacks:
                try:
                    await callback(alert)
                except Exception as e:
                    logger.error(f"Alert callback error: {e}")
    
    async def _flush_db_batch(self):
        """Flush metrics data to TimescaleDB"""
        if not self.db_batch or not self.timescaledb:
            return
        
        try:
            batch = self.db_batch.copy()
            self.db_batch.clear()
            
            # Insert batch into TimescaleDB
            async with self.timescaledb.async_session() as session:
                for metrics_data in batch:
                    await self._insert_metrics_data(session, metrics_data)
                await session.commit()
            
            logger.debug(f"Flushed {len(batch)} metrics records to TimescaleDB")
            
        except Exception as e:
            logger.error(f"DB flush failed: {e}")
            # Re-add to batch for retry
            self.db_batch.extend(batch)
    
    async def _insert_metrics_data(self, session, metrics_data: StreamMetricsData):
        """Insert metrics data into TimescaleDB"""
        # Implementation depends on your metrics table schema
        pass
    
    def add_alert_callback(self, callback: Callable):
        """Add alert callback"""
        self.alert_callbacks.append(callback)
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current metrics snapshot"""
        try:
            if not self.metrics_history:
                return {}
            
            latest = self.metrics_history[-1]
            return {
                'timestamp': latest.timestamp.isoformat(),
                'system_metrics': {
                    'cpu_percent': latest.system_metrics.cpu_percent if latest.system_metrics else 0.0,
                    'memory_percent': latest.system_metrics.memory_percent if latest.system_metrics else 0.0,
                    'memory_used_mb': latest.system_metrics.memory_used_mb if latest.system_metrics else 0.0,
                    'disk_usage_percent': latest.system_metrics.disk_usage_percent if latest.system_metrics else 0.0
                },
                'component_metrics': latest.component_metrics,
                'aggregated_metrics': latest.aggregated_metrics,
                'alerts': latest.alerts
            }
        except Exception as e:
            logger.error(f"Error getting current metrics: {e}")
            return {}
    
    def get_metrics_history(self, hours: int = 1) -> List[Dict[str, Any]]:
        """Get metrics history for the specified hours"""
        try:
            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
            history = []
            
            for metrics_data in self.metrics_history:
                if metrics_data.timestamp >= cutoff_time:
                    history.append({
                        'timestamp': metrics_data.timestamp.isoformat(),
                        'system_metrics': {
                            'cpu_percent': metrics_data.system_metrics.cpu_percent if metrics_data.system_metrics else 0.0,
                            'memory_percent': metrics_data.system_metrics.memory_percent if metrics_data.system_metrics else 0.0
                        },
                        'aggregated_metrics': metrics_data.aggregated_metrics,
                        'alerts': metrics_data.alerts
                    })
            
            return history
            
        except Exception as e:
            logger.error(f"Error getting metrics history: {e}")
            return []
    
    def get_alert_history(self, hours: int = 1) -> List[Dict[str, Any]]:
        """Get alert history for the specified hours"""
        try:
            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
            alerts = []
            
            for alert in self.alert_history:
                if alert['timestamp'] >= cutoff_time:
                    alerts.append(alert)
            
            return alerts
            
        except Exception as e:
            logger.error(f"Error getting alert history: {e}")
            return []
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get stream metrics collector metrics"""
        return {
            'is_running': self.is_running,
            'metrics_history_size': len(self.metrics_history),
            'alert_history_size': len(self.alert_history),
            'collection_interval': self.collection_interval,
            'db_batch_size': len(self.db_batch),
            'alert_callbacks': len(self.alert_callbacks)
        }
    
    async def collect_system_metrics(self) -> Dict[str, Any]:
        """Collect system-level metrics"""
        try:
            # Get system metrics using psutil
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            system_metrics = {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_used_mb': memory.used / (1024 * 1024),
                'memory_total_mb': memory.total / (1024 * 1024),
                'disk_usage_percent': disk.percent,
                'disk_free_gb': disk.free / (1024 * 1024 * 1024),
                'disk_total_gb': disk.total / (1024 * 1024 * 1024),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
            return system_metrics
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
            return {
                'cpu_percent': 0.0,
                'memory_percent': 0.0,
                'memory_used_mb': 0.0,
                'memory_total_mb': 0.0,
                'disk_usage_percent': 0.0,
                'disk_free_gb': 0.0,
                'disk_total_gb': 0.0,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'error': str(e)
            }
    
    async def collect_component_metrics(self) -> Dict[str, Any]:
        """Collect metrics from all streaming components"""
        try:
            component_metrics = {}
            
            # Collect metrics from each component
            if self.stream_buffer:
                component_metrics['stream_buffer'] = self.stream_buffer.get_metrics()
            
            if self.stream_normalizer:
                component_metrics['stream_normalizer'] = self.stream_normalizer.get_metrics()
            
            if self.candle_builder:
                component_metrics['candle_builder'] = self.candle_builder.get_metrics()
            
            if self.rolling_state_manager:
                component_metrics['rolling_state_manager'] = self.rolling_state_manager.get_metrics()
            
            if self.stream_processor:
                component_metrics['stream_processor'] = self.stream_processor.get_metrics()
            
            return component_metrics
            
        except Exception as e:
            logger.error(f"Error collecting component metrics: {e}")
            return {
                'stream_buffer': {},
                'stream_normalizer': {},
                'candle_builder': {},
                'rolling_state_manager': {},
                'stream_processor': {},
                'error': str(e)
            }
    
    async def shutdown(self):
        """Shutdown the metrics collector"""
        self.is_running = False
        
        # Cancel background tasks
        if self.collection_task:
            self.collection_task.cancel()
        if self.system_monitor_task:
            self.system_monitor_task.cancel()
        if self.db_flush_task:
            self.db_flush_task.cancel()
        
        # Flush remaining data
        if self.db_batch and self.timescaledb:
            await self._flush_db_batch()
        
        # Close TimescaleDB connection
        if self.timescaledb:
            await self.timescaledb.close()
        
        logger.info("🛑 StreamMetrics shutdown complete")

# Global instance
stream_metrics = StreamMetrics()
