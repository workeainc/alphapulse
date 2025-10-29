import asyncio
import json
import logging
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, asdict
from enum import Enum
import psutil

logger = logging.getLogger(__name__)

class MetricType(Enum):
    PATTERNS_PER_SECOND = "patterns_per_second"
    MEMORY_USAGE = "memory_usage"
    CPU_USAGE = "cpu_usage"
    STORAGE_EFFICIENCY = "storage_efficiency"
    BATCH_SIZE = "batch_size"
    ACTIVE_WORKERS = "active_workers"
    DATABASE_HEALTH = "database_health"
    SYSTEM_HEALTH = "system_health"

@dataclass
class DashboardMetric:
    timestamp: datetime
    metric_type: MetricType
    value: float
    unit: str
    metadata: Dict[str, Any]

@dataclass
class DashboardAlert:
    id: str
    severity: str
    message: str
    timestamp: datetime
    acknowledged: bool
    metadata: Dict[str, Any]

@dataclass
class DashboardWidget:
    id: str
    title: str
    widget_type: str
    data: Dict[str, Any]
    position: Dict[str, int]
    size: Dict[str, int]
    refresh_interval: int

class RealTimeAnalyticsDashboard:
    """
    Real-time analytics dashboard for monitoring pattern processing performance
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.metrics_history: Dict[MetricType, List[DashboardMetric]] = {
            metric_type: [] for metric_type in MetricType
        }
        self.alerts: List[DashboardAlert] = []
        self.widgets: Dict[str, DashboardWidget] = {}
        self.connected_clients: Set[str] = set()
        self.dashboard_state = {}
        
        # Configuration
        self.metrics_retention_hours = config.get('metrics_retention_hours', 24)
        self.update_interval = config.get('update_interval', 1)  # seconds
        self.max_metrics_per_type = config.get('max_metrics_per_type', 1000)
        self.enable_real_time_updates = config.get('enable_real_time_updates', True)
        
        # Performance tracking
        self.stats = {
            'total_metrics_collected': 0,
            'total_alerts_generated': 0,
            'active_connections': 0,
            'last_update': None,
            'dashboard_uptime': 0
        }
        
        # Initialize default widgets
        self._initialize_default_widgets()
        
        logger.info("RealTimeAnalyticsDashboard initialized with config: %s", config)
    
    def _initialize_default_widgets(self):
        """Initialize default dashboard widgets"""
        default_widgets = [
            {
                'id': 'performance_overview',
                'title': 'Performance Overview',
                'widget_type': 'gauge',
                'position': {'x': 0, 'y': 0},
                'size': {'width': 300, 'height': 200},
                'refresh_interval': 5
            },
            {
                'id': 'patterns_throughput',
                'title': 'Patterns Throughput',
                'widget_type': 'line_chart',
                'position': {'x': 320, 'y': 0},
                'size': {'width': 400, 'height': 200},
                'refresh_interval': 2
            },
            {
                'id': 'system_resources',
                'title': 'System Resources',
                'widget_type': 'bar_chart',
                'position': {'x': 0, 'y': 220},
                'size': {'width': 300, 'height': 200},
                'refresh_interval': 5
            },
            {
                'id': 'storage_metrics',
                'title': 'Storage Metrics',
                'widget_type': 'pie_chart',
                'position': {'x': 320, 'y': 220},
                'size': {'width': 300, 'height': 200},
                'refresh_interval': 10
            },
            {
                'id': 'active_alerts',
                'title': 'Active Alerts',
                'widget_type': 'alert_list',
                'position': {'x': 640, 'y': 0},
                'size': {'width': 300, 'height': 420},
                'refresh_interval': 3
            }
        ]
        
        for widget_config in default_widgets:
            widget = DashboardWidget(
                id=widget_config['id'],
                title=widget_config['title'],
                widget_type=widget_config['widget_type'],
                data={},
                position=widget_config['position'],
                size=widget_config['size'],
                refresh_interval=widget_config['refresh_interval']
            )
            self.widgets[widget.id] = widget
    
    async def start(self):
        """Start the analytics dashboard"""
        logger.info("Starting real-time analytics dashboard...")
        
        # Start background tasks
        asyncio.create_task(self._metrics_collector())
        asyncio.create_task(self._dashboard_updater())
        asyncio.create_task(self._cleanup_old_metrics())
        
        if self.enable_real_time_updates:
            asyncio.create_task(self._real_time_broadcaster())
        
        self.stats['last_update'] = datetime.now(timezone.utc)
        logger.info("Real-time analytics dashboard started successfully")
    
    async def _metrics_collector(self):
        """Collect system metrics periodically"""
        while True:
            try:
                await self._collect_system_metrics()
                await asyncio.sleep(self.update_interval)
                
            except Exception as e:
                logger.error("Error in metrics collector: %s", e)
                await asyncio.sleep(5)
    
    async def _collect_system_metrics(self):
        """Collect current system metrics"""
        current_time = datetime.now(timezone.utc)
        
        # System metrics
        try:
            memory = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent(interval=0.1)
            
            # Memory usage
            memory_metric = DashboardMetric(
                timestamp=current_time,
                metric_type=MetricType.MEMORY_USAGE,
                value=memory.percent,
                unit='%',
                metadata={'total_gb': round(memory.total / (1024**3), 2),
                         'available_gb': round(memory.available / (1024**3), 2)}
            )
            
            # CPU usage
            cpu_metric = DashboardMetric(
                timestamp=current_time,
                metric_type=MetricType.CPU_USAGE,
                value=cpu_percent,
                unit='%',
                metadata={'cpu_count': psutil.cpu_count()}
            )
            
            self._add_metric(memory_metric)
            self._add_metric(cpu_metric)
            
        except Exception as e:
            logger.warning("Could not collect system metrics: %s", e)
        
        # Update dashboard state
        self.dashboard_state['last_metrics_update'] = current_time.isoformat()
        self.stats['last_update'] = current_time
    
    def _add_metric(self, metric: DashboardMetric):
        """Add a new metric to the history"""
        metric_type = metric.metric_type
        self.metrics_history[metric_type].append(metric)
        
        # Keep only the most recent metrics
        if len(self.metrics_history[metric_type]) > self.max_metrics_per_type:
            self.metrics_history[metric_type] = self.metrics_history[metric_type][-self.max_metrics_per_type:]
        
        self.stats['total_metrics_collected'] += 1
    
    def update_pattern_metrics(self, pattern_stats: Dict[str, Any]):
        """Update pattern processing metrics from external source"""
        current_time = datetime.now(timezone.utc)
        
        # Patterns per second
        if 'patterns_per_second' in pattern_stats:
            pps_metric = DashboardMetric(
                timestamp=current_time,
                metric_type=MetricType.PATTERNS_PER_SECOND,
                value=pattern_stats['patterns_per_second'],
                unit='patterns/s',
                metadata={'total_patterns': pattern_stats.get('total_patterns', 0)}
            )
            self._add_metric(pps_metric)
        
        # Storage efficiency
        if 'storage_efficiency' in pattern_stats:
            storage_metric = DashboardMetric(
                timestamp=current_time,
                metric_type=MetricType.STORAGE_EFFICIENCY,
                value=pattern_stats['storage_efficiency'],
                unit='%',
                metadata={'storage_method': pattern_stats.get('storage_method', 'unknown')}
            )
            self._add_metric(storage_metric)
        
        # Batch size
        if 'current_batch_size' in pattern_stats:
            batch_metric = DashboardMetric(
                timestamp=current_time,
                metric_type=MetricType.BATCH_SIZE,
                value=pattern_stats['current_batch_size'],
                unit='patterns',
                metadata={'batch_method': pattern_stats.get('batch_method', 'unknown')}
            )
            self._add_metric(batch_metric)
        
        # Active workers
        if 'active_workers' in pattern_stats:
            workers_metric = DashboardMetric(
                timestamp=current_time,
                metric_type=MetricType.ACTIVE_WORKERS,
                value=pattern_stats['active_workers'],
                unit='workers',
                metadata={'max_workers': pattern_stats.get('max_workers', 0)}
            )
            self._add_metric(workers_metric)
    
    def update_database_health(self, health_data: Dict[str, Any]):
        """Update database health metrics"""
        current_time = datetime.now(timezone.utc)
        
        health_metric = DashboardMetric(
            timestamp=current_time,
            metric_type=MetricType.DATABASE_HEALTH,
            value=health_data.get('health_score', 0),
            unit='score',
            metadata={
                'connection_status': health_data.get('connection_status', 'unknown'),
                'response_time_ms': health_data.get('response_time_ms', 0),
                'active_connections': health_data.get('active_connections', 0)
            }
        )
        
        self._add_metric(health_metric)
    
    def update_system_health(self, health_data: Dict[str, Any]):
        """Update overall system health metrics"""
        current_time = datetime.now(timezone.utc)
        
        health_metric = DashboardMetric(
            timestamp=current_time,
            metric_type=MetricType.SYSTEM_HEALTH,
            value=health_data.get('overall_health_score', 0),
            unit='score',
            metadata={
                'component_count': health_data.get('component_count', 0),
                'healthy_components': health_data.get('healthy_components', 0),
                'critical_issues': health_data.get('critical_issues', 0)
            }
        )
        
        self._add_metric(health_metric)
    
    async def _dashboard_updater(self):
        """Update dashboard widgets with latest data"""
        while True:
            try:
                await self._update_all_widgets()
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error("Error in dashboard updater: %s", e)
                await asyncio.sleep(5)
    
    async def _update_all_widgets(self):
        """Update all dashboard widgets with current data"""
        for widget_id, widget in self.widgets.items():
            try:
                await self._update_widget(widget)
            except Exception as e:
                logger.error("Error updating widget %s: %s", widget_id, e)
    
    async def _update_widget(self, widget: DashboardWidget):
        """Update a specific widget with current data"""
        if widget.widget_type == 'gauge':
            widget.data = await self._get_performance_overview_data()
        elif widget.widget_type == 'line_chart':
            widget.data = await self._get_patterns_throughput_data()
        elif widget.widget_type == 'bar_chart':
            widget.data = await self._get_system_resources_data()
        elif widget.widget_type == 'pie_chart':
            widget.data = await self._get_storage_metrics_data()
        elif widget.widget_type == 'alert_list':
            widget.data = await self._get_active_alerts_data()
    
    async def _get_performance_overview_data(self) -> Dict[str, Any]:
        """Get data for performance overview widget"""
        # Calculate overall performance score
        recent_metrics = self._get_recent_metrics(5)  # Last 5 minutes
        
        if not recent_metrics:
            return {'score': 0, 'status': 'No Data'}
        
        # Calculate weighted average of key metrics
        pps_metrics = [m for m in recent_metrics if m.metric_type == MetricType.PATTERNS_PER_SECOND]
        storage_metrics = [m for m in recent_metrics if m.metric_type == MetricType.STORAGE_EFFICIENCY]
        health_metrics = [m for m in recent_metrics if m.metric_type == MetricType.SYSTEM_HEALTH]
        
        score = 0
        if pps_metrics:
            avg_pps = sum(m.value for m in pps_metrics) / len(pps_metrics)
            score += min(avg_pps / 1000, 1.0) * 40  # Max 40 points for throughput
        
        if storage_metrics:
            avg_storage = sum(m.value for m in storage_metrics) / len(storage_metrics)
            score += (avg_storage / 100) * 30  # Max 30 points for storage efficiency
        
        if health_metrics:
            avg_health = sum(m.value for m in health_metrics) / len(health_metrics)
            score += (avg_health / 100) * 30  # Max 30 points for system health
        
        # Determine status
        if score >= 80:
            status = 'Excellent'
        elif score >= 60:
            status = 'Good'
        elif score >= 40:
            status = 'Fair'
        else:
            status = 'Poor'
        
        return {
            'score': round(score, 1),
            'status': status,
            'max_score': 100,
            'last_updated': datetime.now(timezone.utc).isoformat()
        }
    
    async def _get_patterns_throughput_data(self) -> Dict[str, Any]:
        """Get data for patterns throughput chart"""
        pps_metrics = self.metrics_history[MetricType.PATTERNS_PER_SECOND]
        
        if not pps_metrics:
            return {'labels': [], 'data': [], 'status': 'No Data'}
        
        # Get last 50 data points
        recent_metrics = pps_metrics[-50:] if len(pps_metrics) > 50 else pps_metrics
        
        labels = [m.timestamp.strftime('%H:%M:%S') for m in recent_metrics]
        data = [m.value for m in recent_metrics]
        
        return {
            'labels': labels,
            'data': data,
            'status': 'Active',
            'current_value': data[-1] if data else 0,
            'average_value': sum(data) / len(data) if data else 0,
            'max_value': max(data) if data else 0
        }
    
    async def _get_system_resources_data(self) -> Dict[str, Any]:
        """Get data for system resources chart"""
        memory_metrics = self.metrics_history[MetricType.MEMORY_USAGE]
        cpu_metrics = self.metrics_history[MetricType.CPU_USAGE]
        
        current_memory = memory_metrics[-1].value if memory_metrics else 0
        current_cpu = cpu_metrics[-1].value if cpu_metrics else 0
        
        return {
            'labels': ['Memory', 'CPU'],
            'data': [current_memory, current_cpu],
            'max_values': [100, 100],
            'units': ['%', '%'],
            'status': 'Active',
            'last_updated': datetime.now(timezone.utc).isoformat()
        }
    
    async def _get_storage_metrics_data(self) -> Dict[str, Any]:
        """Get data for storage metrics chart"""
        storage_metrics = self.metrics_history[MetricType.STORAGE_EFFICIENCY]
        
        if not storage_metrics:
            return {'labels': [], 'data': [], 'status': 'No Data'}
        
        # Group by storage method if available
        method_data = {}
        for metric in storage_metrics[-20:]:  # Last 20 metrics
            method = metric.metadata.get('storage_method', 'Unknown')
            if method not in method_data:
                method_data[method] = []
            method_data[method].append(metric.value)
        
        labels = list(method_data.keys())
        data = [sum(values) / len(values) for values in method_data.values()]
        
        return {
            'labels': labels,
            'data': data,
            'status': 'Active',
            'total_methods': len(labels),
            'last_updated': datetime.now(timezone.utc).isoformat()
        }
    
    async def _get_active_alerts_data(self) -> Dict[str, Any]:
        """Get data for active alerts widget"""
        active_alerts = [alert for alert in self.alerts if not alert.acknowledged]
        
        return {
            'alerts': [asdict(alert) for alert in active_alerts],
            'total_active': len(active_alerts),
            'critical_count': len([a for a in active_alerts if a.severity == 'critical']),
            'warning_count': len([a for a in active_alerts if a.severity == 'warning']),
            'last_updated': datetime.now(timezone.utc).isoformat()
        }
    
    def _get_recent_metrics(self, minutes: int) -> List[DashboardMetric]:
        """Get metrics from the last N minutes"""
        cutoff_time = datetime.now(timezone.utc) - timedelta(minutes=minutes)
        
        all_metrics = []
        for metric_list in self.metrics_history.values():
            recent_metrics = [m for m in metric_list if m.timestamp >= cutoff_time]
            all_metrics.extend(recent_metrics)
        
        return all_metrics
    
    async def _real_time_broadcaster(self):
        """Broadcast real-time updates to connected clients"""
        while True:
            try:
                if self.connected_clients:
                    update_data = await self._prepare_real_time_update()
                    await self._broadcast_to_clients(update_data)
                
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error("Error in real-time broadcaster: %s", e)
                await asyncio.sleep(5)
    
    async def _prepare_real_time_update(self) -> Dict[str, Any]:
        """Prepare real-time update data for clients"""
        return {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'dashboard_state': self.dashboard_state,
            'widgets': {wid: asdict(widget) for wid, widget in self.widgets.items()},
            'stats': self.stats,
            'recent_alerts': [asdict(alert) for alert in self.alerts[-5:]]  # Last 5 alerts
        }
    
    async def _broadcast_to_clients(self, data: Dict[str, Any]):
        """Broadcast data to all connected clients"""
        # This is a placeholder - in real implementation, you'd send via WebSocket
        logger.debug("Broadcasting update to %d connected clients", len(self.connected_clients))
    
    async def _cleanup_old_metrics(self):
        """Clean up old metrics based on retention policy"""
        while True:
            try:
                cutoff_time = datetime.now(timezone.utc) - timedelta(hours=self.metrics_retention_hours)
                
                for metric_type in MetricType:
                    self.metrics_history[metric_type] = [
                        m for m in self.metrics_history[metric_type]
                        if m.timestamp >= cutoff_time
                    ]
                
                await asyncio.sleep(300)  # Clean up every 5 minutes
                
            except Exception as e:
                logger.error("Error in metrics cleanup: %s", e)
                await asyncio.sleep(60)
    
    def add_alert(self, severity: str, message: str, metadata: Dict[str, Any] = None):
        """Add a new alert to the dashboard"""
        alert = DashboardAlert(
            id=str(len(self.alerts) + 1),
            severity=severity,
            message=message,
            timestamp=datetime.now(timezone.utc),
            acknowledged=False,
            metadata=metadata or {}
        )
        
        self.alerts.append(alert)
        self.stats['total_alerts_generated'] += 1
        
        logger.info("New alert added: %s - %s", severity, message)
    
    def acknowledge_alert(self, alert_id: str):
        """Acknowledge an alert"""
        for alert in self.alerts:
            if alert.id == alert_id:
                alert.acknowledged = True
                logger.info("Alert %s acknowledged", alert_id)
                break
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get complete dashboard data for API consumption"""
        return {
            'widgets': {wid: asdict(widget) for wid, widget in self.widgets.items()},
            'metrics_summary': {
                metric_type.name: {
                    'count': len(metrics),
                    'latest_value': metrics[-1].value if metrics else None,
                    'latest_timestamp': metrics[-1].timestamp.isoformat() if metrics else None
                }
                for metric_type, metrics in self.metrics_history.items()
            },
            'alerts': {
                'total': len(self.alerts),
                'active': len([a for a in self.alerts if not a.acknowledged]),
                'recent': [asdict(alert) for alert in self.alerts[-10:]]
            },
            'stats': self.stats,
            'dashboard_state': self.dashboard_state,
            'last_updated': datetime.now(timezone.utc).isoformat()
        }
    
    def get_metrics_export(self, metric_type: MetricType = None, 
                          hours: int = 24) -> Dict[str, Any]:
        """Export metrics data for external analysis"""
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
        
        if metric_type:
            metrics = [
                m for m in self.metrics_history[metric_type]
                if m.timestamp >= cutoff_time
            ]
        else:
            metrics = self._get_recent_metrics(hours * 60)
        
        return {
            'export_timestamp': datetime.now(timezone.utc).isoformat(),
            'time_range_hours': hours,
            'metric_type': metric_type.name if metric_type else 'all',
            'metrics_count': len(metrics),
            'data': [asdict(m) for m in metrics]
        }
    
    async def stop(self):
        """Stop the analytics dashboard"""
        logger.info("Stopping real-time analytics dashboard...")
        # Cleanup would go here
        logger.info("Real-time analytics dashboard stopped")
