"""
Monitoring & Auto-scaling for AlphaPulse
Phase 5: Prometheus monitoring and auto-scaling capabilities
"""

import asyncio
import logging
import time
import psutil
import threading
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
import json

# Prometheus imports
try:
    from prometheus_client import Counter, Gauge, Histogram, Summary, start_http_server
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    print("Prometheus not available - using simplified metrics")

logger = logging.getLogger(__name__)

@dataclass
class SystemMetrics:
    """System performance metrics"""
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    disk_usage_percent: float = 0.0
    network_io_bytes: int = 0
    active_connections: int = 0
    queue_size: int = 0
    processing_latency_ms: float = 0.0
    throughput_mps: float = 0.0
    error_rate: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class ScalingDecision:
    """Auto-scaling decision"""
    action: str  # 'scale_up', 'scale_down', 'maintain'
    reason: str
    current_instances: int
    target_instances: int
    confidence: float
    timestamp: datetime = field(default_factory=datetime.now)

class PrometheusMetrics:
    """Prometheus metrics collector"""
    
    def __init__(self, port: int = 8000):
        self.port = port
        self.is_running = False
        
        # Define metrics
        if PROMETHEUS_AVAILABLE:
            self.metrics = {
                'signals_processed_total': Counter('signals_processed_total', 'Total signals processed'),
                'signals_processed_duration': Histogram('signals_processed_duration_seconds', 'Signal processing duration'),
                'system_cpu_usage': Gauge('system_cpu_usage_percent', 'CPU usage percentage'),
                'system_memory_usage': Gauge('system_memory_usage_percent', 'Memory usage percentage'),
                'system_disk_usage': Gauge('system_disk_usage_percent', 'Disk usage percentage'),
                'active_connections': Gauge('active_connections', 'Number of active connections'),
                'queue_size': Gauge('queue_size', 'Current queue size'),
                'processing_latency': Histogram('processing_latency_seconds', 'Processing latency'),
                'error_rate': Gauge('error_rate', 'Error rate percentage'),
                'scaling_events': Counter('scaling_events_total', 'Total scaling events', ['action'])
            }
        else:
            self.metrics = {}
        
        logger.info("Prometheus Metrics initialized")
    
    def start(self):
        """Start Prometheus metrics server"""
        if not PROMETHEUS_AVAILABLE:
            logger.warning("Prometheus not available - metrics disabled")
            return
        
        try:
            start_http_server(self.port)
            self.is_running = True
            logger.info(f"🚀 Prometheus metrics server started on port {self.port}")
        except Exception as e:
            logger.error(f"Error starting Prometheus server: {e}")
    
    def record_signal_processed(self, duration: float):
        """Record signal processing metrics"""
        if 'signals_processed_total' in self.metrics:
            self.metrics['signals_processed_total'].inc()
            self.metrics['signals_processed_duration'].observe(duration)
    
    def update_system_metrics(self, metrics: SystemMetrics):
        """Update system metrics"""
        if not PROMETHEUS_AVAILABLE:
            return
        
        try:
            self.metrics['system_cpu_usage'].set(metrics.cpu_percent)
            self.metrics['system_memory_usage'].set(metrics.memory_percent)
            self.metrics['system_disk_usage'].set(metrics.disk_usage_percent)
            self.metrics['active_connections'].set(metrics.active_connections)
            self.metrics['queue_size'].set(metrics.queue_size)
            self.metrics['processing_latency'].observe(metrics.processing_latency_ms / 1000)
            self.metrics['error_rate'].set(metrics.error_rate)
        except Exception as e:
            logger.error(f"Error updating Prometheus metrics: {e}")
    
    def record_scaling_event(self, action: str):
        """Record scaling event"""
        if 'scaling_events' in self.metrics:
            self.metrics['scaling_events'].labels(action=action).inc()

class SystemMonitor:
    """System performance monitor"""
    
    def __init__(self, collection_interval: int = 5):
        self.collection_interval = collection_interval
        self.is_running = False
        self.metrics_history = deque(maxlen=1000)
        self.monitoring_callbacks = []
        
        # Performance thresholds
        self.thresholds = {
            'cpu_warning': 70.0,
            'cpu_critical': 90.0,
            'memory_warning': 80.0,
            'memory_critical': 95.0,
            'disk_warning': 85.0,
            'disk_critical': 95.0,
            'latency_warning': 100.0,  # ms
            'latency_critical': 500.0,  # ms
            'error_rate_warning': 5.0,  # %
            'error_rate_critical': 10.0  # %
        }
        
        logger.info("System Monitor initialized")
    
    async def start(self):
        """Start system monitoring"""
        self.is_running = True
        asyncio.create_task(self._monitoring_loop())
        logger.info("🚀 System Monitor started")
    
    async def stop(self):
        """Stop system monitoring"""
        self.is_running = False
        logger.info("🛑 System Monitor stopped")
    
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.is_running:
            try:
                metrics = self._collect_system_metrics()
                self.metrics_history.append(metrics)
                
                # Check thresholds and trigger alerts
                await self._check_thresholds(metrics)
                
                # Notify callbacks
                for callback in self.monitoring_callbacks:
                    try:
                        await callback(metrics)
                    except Exception as e:
                        logger.error(f"Error in monitoring callback: {e}")
                
                await asyncio.sleep(self.collection_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(self.collection_interval)
    
    def _collect_system_metrics(self) -> SystemMetrics:
        """Collect current system metrics"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_usage_percent = (disk.used / disk.total) * 100
            
            # Network I/O
            network = psutil.net_io_counters()
            network_io_bytes = network.bytes_sent + network.bytes_recv
            
            # Simplified metrics for demonstration
            active_connections = len(psutil.net_connections())
            queue_size = len(self.metrics_history) if self.metrics_history else 0
            
            # Calculate processing latency (simplified)
            processing_latency_ms = 50.0  # Placeholder
            
            # Calculate throughput (simplified)
            throughput_mps = 100.0  # Placeholder
            
            # Calculate error rate (simplified)
            error_rate = 0.5  # Placeholder
            
            return SystemMetrics(
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                disk_usage_percent=disk_usage_percent,
                network_io_bytes=network_io_bytes,
                active_connections=active_connections,
                queue_size=queue_size,
                processing_latency_ms=processing_latency_ms,
                throughput_mps=throughput_mps,
                error_rate=error_rate
            )
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
            return SystemMetrics()
    
    async def _check_thresholds(self, metrics: SystemMetrics):
        """Check metrics against thresholds and trigger alerts"""
        alerts = []
        
        # CPU checks
        if metrics.cpu_percent >= self.thresholds['cpu_critical']:
            alerts.append(f"CRITICAL: CPU usage {metrics.cpu_percent:.1f}%")
        elif metrics.cpu_percent >= self.thresholds['cpu_warning']:
            alerts.append(f"WARNING: CPU usage {metrics.cpu_percent:.1f}%")
        
        # Memory checks
        if metrics.memory_percent >= self.thresholds['memory_critical']:
            alerts.append(f"CRITICAL: Memory usage {metrics.memory_percent:.1f}%")
        elif metrics.memory_percent >= self.thresholds['memory_warning']:
            alerts.append(f"WARNING: Memory usage {metrics.memory_percent:.1f}%")
        
        # Disk checks
        if metrics.disk_usage_percent >= self.thresholds['disk_critical']:
            alerts.append(f"CRITICAL: Disk usage {metrics.disk_usage_percent:.1f}%")
        elif metrics.disk_usage_percent >= self.thresholds['disk_warning']:
            alerts.append(f"WARNING: Disk usage {metrics.disk_usage_percent:.1f}%")
        
        # Latency checks
        if metrics.processing_latency_ms >= self.thresholds['latency_critical']:
            alerts.append(f"CRITICAL: Processing latency {metrics.processing_latency_ms:.1f}ms")
        elif metrics.processing_latency_ms >= self.thresholds['latency_warning']:
            alerts.append(f"WARNING: Processing latency {metrics.processing_latency_ms:.1f}ms")
        
        # Error rate checks
        if metrics.error_rate >= self.thresholds['error_rate_critical']:
            alerts.append(f"CRITICAL: Error rate {metrics.error_rate:.1f}%")
        elif metrics.error_rate >= self.thresholds['error_rate_warning']:
            alerts.append(f"WARNING: Error rate {metrics.error_rate:.1f}%")
        
        # Log alerts
        for alert in alerts:
            logger.warning(f"🚨 {alert}")
    
    def add_monitoring_callback(self, callback: Callable[[SystemMetrics], None]):
        """Add monitoring callback"""
        self.monitoring_callbacks.append(callback)
    
    def get_recent_metrics(self, minutes: int = 5) -> List[SystemMetrics]:
        """Get recent metrics from the last N minutes"""
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        return [m for m in self.metrics_history if m.timestamp >= cutoff_time]
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get metrics summary"""
        if not self.metrics_history:
            return {}
        
        recent_metrics = self.get_recent_metrics(5)
        if not recent_metrics:
            return {}
        
        return {
            'avg_cpu_percent': sum(m.cpu_percent for m in recent_metrics) / len(recent_metrics),
            'avg_memory_percent': sum(m.memory_percent for m in recent_metrics) / len(recent_metrics),
            'avg_disk_usage_percent': sum(m.disk_usage_percent for m in recent_metrics) / len(recent_metrics),
            'avg_processing_latency_ms': sum(m.processing_latency_ms for m in recent_metrics) / len(recent_metrics),
            'avg_error_rate': sum(m.error_rate for m in recent_metrics) / len(recent_metrics),
            'total_metrics_collected': len(self.metrics_history),
            'monitoring_active': self.is_running
        }

class AutoScaler:
    """Auto-scaling controller"""
    
    def __init__(self, 
                 min_instances: int = 1,
                 max_instances: int = 10,
                 scale_up_threshold: float = 80.0,
                 scale_down_threshold: float = 30.0):
        
        self.min_instances = min_instances
        self.max_instances = max_instances
        self.scale_up_threshold = scale_up_threshold
        self.scale_down_threshold = scale_down_threshold
        
        self.current_instances = min_instances
        self.scaling_history = deque(maxlen=100)
        self.is_running = False
        
        # Scaling cooldown
        self.last_scale_time = datetime.now()
        self.scale_cooldown_seconds = 300  # 5 minutes
        
        logger.info("AutoScaler initialized")
    
    async def start(self):
        """Start auto-scaling"""
        self.is_running = True
        logger.info("🚀 AutoScaler started")
    
    async def stop(self):
        """Stop auto-scaling"""
        self.is_running = False
        logger.info("🛑 AutoScaler stopped")
    
    async def evaluate_scaling(self, metrics: SystemMetrics) -> Optional[ScalingDecision]:
        """Evaluate if scaling is needed"""
        if not self.is_running:
            return None
        
        # Check cooldown period
        if (datetime.now() - self.last_scale_time).total_seconds() < self.scale_cooldown_seconds:
            return None
        
        # Calculate load score (weighted average of key metrics)
        load_score = (
            metrics.cpu_percent * 0.4 +
            metrics.memory_percent * 0.3 +
            (metrics.processing_latency_ms / 100) * 0.2 +
            metrics.error_rate * 0.1
        )
        
        decision = None
        
        # Scale up decision
        if (load_score > self.scale_up_threshold and 
            self.current_instances < self.max_instances):
            
            target_instances = min(self.current_instances + 1, self.max_instances)
            decision = ScalingDecision(
                action='scale_up',
                reason=f"High load score: {load_score:.1f} > {self.scale_up_threshold}",
                current_instances=self.current_instances,
                target_instances=target_instances,
                confidence=min(load_score / 100, 1.0)
            )
            
            self.current_instances = target_instances
            self.last_scale_time = datetime.now()
        
        # Scale down decision
        elif (load_score < self.scale_down_threshold and 
              self.current_instances > self.min_instances):
            
            target_instances = max(self.current_instances - 1, self.min_instances)
            decision = ScalingDecision(
                action='scale_down',
                reason=f"Low load score: {load_score:.1f} < {self.scale_down_threshold}",
                current_instances=self.current_instances,
                target_instances=target_instances,
                confidence=min((100 - load_score) / 100, 1.0)
            )
            
            self.current_instances = target_instances
            self.last_scale_time = datetime.now()
        
        # Maintain current scaling
        else:
            decision = ScalingDecision(
                action='maintain',
                reason=f"Load score within range: {load_score:.1f}",
                current_instances=self.current_instances,
                target_instances=self.current_instances,
                confidence=0.5
            )
        
        # Record scaling decision
        self.scaling_history.append(decision)
        
        return decision
    
    async def execute_scaling(self, decision: ScalingDecision):
        """Execute scaling decision"""
        if decision.action == 'maintain':
            return
        
        try:
            if decision.action == 'scale_up':
                logger.info(f"🔄 Scaling UP: {decision.current_instances} -> {decision.target_instances}")
                # Here you would implement actual scaling logic
                # For example, starting new containers, instances, etc.
                
            elif decision.action == 'scale_down':
                logger.info(f"🔄 Scaling DOWN: {decision.current_instances} -> {decision.target_instances}")
                # Here you would implement actual scaling logic
                # For example, stopping containers, instances, etc.
            
            # Record scaling event in Prometheus
            if hasattr(self, 'prometheus_metrics'):
                self.prometheus_metrics.record_scaling_event(decision.action)
                
        except Exception as e:
            logger.error(f"Error executing scaling decision: {e}")
    
    def get_scaling_summary(self) -> Dict[str, Any]:
        """Get scaling summary"""
        recent_decisions = [d for d in self.scaling_history 
                           if (datetime.now() - d.timestamp).total_seconds() < 3600]
        
        scale_up_count = len([d for d in recent_decisions if d.action == 'scale_up'])
        scale_down_count = len([d for d in recent_decisions if d.action == 'scale_down'])
        
        return {
            'current_instances': self.current_instances,
            'min_instances': self.min_instances,
            'max_instances': self.max_instances,
            'scale_up_threshold': self.scale_up_threshold,
            'scale_down_threshold': self.scale_down_threshold,
            'recent_scale_ups': scale_up_count,
            'recent_scale_downs': scale_down_count,
            'total_scaling_decisions': len(self.scaling_history),
            'autoscaler_active': self.is_running
        }

class MonitoringAutoScaling:
    """Main monitoring and auto-scaling orchestrator"""
    
    def __init__(self, prometheus_port: int = 8000):
        self.prometheus_metrics = PrometheusMetrics(prometheus_port)
        self.system_monitor = SystemMonitor()
        self.auto_scaler = AutoScaler()
        
        self.is_running = False
        
        logger.info("Monitoring & Auto-scaling system initialized")
    
    async def start(self):
        """Start monitoring and auto-scaling"""
        try:
            # Start Prometheus metrics server
            self.prometheus_metrics.start()
            
            # Start system monitoring
            await self.system_monitor.start()
            
            # Start auto-scaler
            await self.auto_scaler.start()
            
            # Connect monitoring to auto-scaling
            self.system_monitor.add_monitoring_callback(self._handle_metrics_update)
            
            self.is_running = True
            logger.info("🚀 Monitoring & Auto-scaling system started")
            
        except Exception as e:
            logger.error(f"Error starting monitoring system: {e}")
            await self.stop()
            raise
    
    async def stop(self):
        """Stop monitoring and auto-scaling"""
        self.is_running = False
        
        await self.system_monitor.stop()
        await self.auto_scaler.stop()
        
        logger.info("🛑 Monitoring & Auto-scaling system stopped")
    
    async def _handle_metrics_update(self, metrics: SystemMetrics):
        """Handle metrics updates for auto-scaling"""
        try:
            # Update Prometheus metrics
            self.prometheus_metrics.update_system_metrics(metrics)
            
            # Evaluate scaling
            scaling_decision = await self.auto_scaler.evaluate_scaling(metrics)
            
            if scaling_decision and scaling_decision.action != 'maintain':
                await self.auto_scaler.execute_scaling(scaling_decision)
                
        except Exception as e:
            logger.error(f"Error handling metrics update: {e}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            'monitoring_active': self.is_running,
            'prometheus_available': PROMETHEUS_AVAILABLE,
            'system_metrics': self.system_monitor.get_metrics_summary(),
            'scaling_status': self.auto_scaler.get_scaling_summary(),
            'prometheus_port': self.prometheus_metrics.port
        }

# Global monitoring and auto-scaling instance
monitoring_autoscaling = MonitoringAutoScaling()
