#!/usr/bin/env python3
"""
Resilience Monitoring and Metrics Dashboard for AlphaPulse
Provides real-time monitoring, alerting, and visualization of resilience features
"""

import asyncio
import logging
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, asdict
from enum import Enum
import threading
import json
from collections import defaultdict, deque
import weakref

from src.app.core.resilience import get_resilience_manager
from src.app.core.rate_limiting import get_rate_limit_manager
from src.app.core.graceful_shutdown import get_shutdown_manager

logger = logging.getLogger(__name__)

class MetricType(Enum):
    """Types of metrics"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"

class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class MetricPoint:
    """A single metric data point"""
    timestamp: datetime
    value: float
    labels: Dict[str, str]
    metadata: Dict[str, Any]

@dataclass
class Alert:
    """Alert information"""
    id: str
    severity: AlertSeverity
    message: str
    timestamp: datetime
    source: str
    metric_name: str
    threshold: float
    current_value: float
    resolved: bool = False
    resolved_at: Optional[datetime] = None

@dataclass
class AlertRule:
    """Alert rule configuration"""
    name: str
    metric_name: str
    condition: str  # ">", "<", ">=", "<=", "==", "!="
    threshold: float
    severity: AlertSeverity
    description: str
    enabled: bool = True
    cooldown: float = 300.0  # 5 minutes default

class ResilienceMetrics:
    """Collects and manages resilience metrics"""
    
    def __init__(self, retention_hours: int = 24):
        self.retention_hours = retention_hours
        self.logger = logging.getLogger(__name__)
        
        # Metric storage
        self.metrics: Dict[str, List[MetricPoint]] = defaultdict(list)
        self.metric_locks: Dict[str, threading.Lock] = defaultdict(threading.Lock)
        
        # Metric definitions
        self.metric_definitions = {
            # Retry metrics
            "retry_attempts_total": {"type": MetricType.COUNTER, "description": "Total retry attempts"},
            "retry_successes_total": {"type": MetricType.COUNTER, "description": "Total successful retries"},
            "retry_failures_total": {"type": MetricType.COUNTER, "description": "Total retry failures"},
            "retry_delay_avg": {"type": MetricType.GAUGE, "description": "Average retry delay"},
            "retry_delay_max": {"type": MetricType.GAUGE, "description": "Maximum retry delay"},
            
            # Circuit breaker metrics
            "circuit_breaker_opens_total": {"type": MetricType.COUNTER, "description": "Total circuit breaker opens"},
            "circuit_breaker_closes_total": {"type": MetricType.COUNTER, "description": "Total circuit breaker closes"},
            "circuit_breaker_failure_rate": {"type": MetricType.GAUGE, "description": "Current failure rate"},
            "circuit_breaker_state": {"type": MetricType.GAUGE, "description": "Current circuit breaker state"},
            
            # Timeout metrics
            "timeout_occurrences_total": {"type": MetricType.COUNTER, "description": "Total timeout occurrences"},
            "operation_duration_avg": {"type": MetricType.GAUGE, "description": "Average operation duration"},
            "operation_duration_max": {"type": MetricType.GAUGE, "description": "Maximum operation duration"},
            
            # Dead letter queue metrics
            "dlq_size": {"type": MetricType.GAUGE, "description": "Current dead letter queue size"},
            "dlq_operations_total": {"type": MetricType.COUNTER, "description": "Total operations in DLQ"},
            "dlq_retries_total": {"type": MetricType.COUNTER, "description": "Total DLQ retry attempts"},
            
            # Rate limiting metrics
            "rate_limit_requests_total": {"type": MetricType.COUNTER, "description": "Total rate limit requests"},
            "rate_limit_allowed_total": {"type": MetricType.COUNTER, "description": "Total allowed requests"},
            "rate_limit_blocked_total": {"type": MetricType.COUNTER, "description": "Total blocked requests"},
            "rate_limit_queue_size": {"type": MetricType.GAUGE, "description": "Current rate limit queue size"},
            
            # Connection metrics
            "db_connections_active": {"type": MetricType.GAUGE, "description": "Active database connections"},
            "db_connections_idle": {"type": MetricType.GAUGE, "description": "Idle database connections"},
            "db_health_check_duration": {"type": MetricType.GAUGE, "description": "Database health check duration"},
            "db_connection_errors_total": {"type": MetricType.COUNTER, "description": "Total connection errors"},
            
            # Shutdown metrics
            "shutdown_duration": {"type": MetricType.GAUGE, "description": "Shutdown duration"},
            "shutdown_tasks_completed": {"type": MetricType.COUNTER, "description": "Completed shutdown tasks"},
            "shutdown_tasks_failed": {"type": MetricType.COUNTER, "description": "Failed shutdown tasks"},
        }
        
        # Start cleanup task
        self._cleanup_task = None
        self._start_cleanup_task()
    
    def _start_cleanup_task(self):
        """Start background cleanup task"""
        async def cleanup_loop():
            while True:
                try:
                    await self._cleanup_old_metrics()
                    await asyncio.sleep(3600)  # Clean up every hour
                except Exception as e:
                    self.logger.error(f"❌ Error in cleanup loop: {e}")
                    await asyncio.sleep(60)
        
        self._cleanup_task = asyncio.create_task(cleanup_loop())
    
    async def _cleanup_old_metrics(self):
        """Remove old metrics outside retention window"""
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=self.retention_hours)
        
        for metric_name, metric_lock in self.metric_locks.items():
            with metric_lock:
                if metric_name in self.metrics:
                    # Remove old data points
                    self.metrics[metric_name] = [
                        point for point in self.metrics[metric_name]
                        if point.timestamp > cutoff_time
                    ]
        
        self.logger.debug(f"🧹 Cleaned up metrics older than {self.retention_hours} hours")
    
    def record_metric(self, name: str, value: float, labels: Dict[str, str] = None, metadata: Dict[str, Any] = None):
        """Record a metric value"""
        if name not in self.metric_definitions:
            self.logger.warning(f"⚠️ Unknown metric: {name}")
            return
        
        labels = labels or {}
        metadata = metadata or {}
        
        metric_point = MetricPoint(
            timestamp=datetime.now(timezone.utc),
            value=value,
            labels=labels,
            metadata=metadata
        )
        
        with self.metric_locks[name]:
            self.metrics[name].append(metric_point)
    
    def increment_counter(self, name: str, labels: Dict[str, str] = None, metadata: Dict[str, Any] = None):
        """Increment a counter metric"""
        current_value = self.get_metric_value(name, labels) or 0
        self.record_metric(name, current_value + 1, labels, metadata)
    
    def set_gauge(self, name: str, value: float, labels: Dict[str, str] = None, metadata: Dict[str, Any] = None):
        """Set a gauge metric value"""
        self.record_metric(name, value, labels, metadata)
    
    def record_timer(self, name: str, duration: float, labels: Dict[str, str] = None, metadata: Dict[str, Any] = None):
        """Record a timer metric"""
        self.record_metric(name, duration, labels, metadata)
    
    def get_metric_value(self, name: str, labels: Dict[str, str] = None) -> Optional[float]:
        """Get the current value of a metric"""
        if name not in self.metrics:
            return None
        
        labels = labels or {}
        
        with self.metric_locks[name]:
            # Find the most recent data point with matching labels
            for point in reversed(self.metrics[name]):
                if point.labels == labels:
                    return point.value
        
        return None
    
    def get_metric_history(self, name: str, hours: int = 1, labels: Dict[str, str] = None) -> List[MetricPoint]:
        """Get metric history for the specified time period"""
        if name not in self.metrics:
            return []
        
        labels = labels or {}
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
        
        with self.metric_locks[name]:
            return [
                point for point in self.metrics[name]
                if point.timestamp > cutoff_time and point.labels == labels
            ]
    
    def get_metric_summary(self, name: str, hours: int = 1, labels: Dict[str, str] = None) -> Dict[str, Any]:
        """Get a summary of metric statistics"""
        history = self.get_metric_history(name, hours, labels)
        
        if not history:
            return {
                "count": 0,
                "min": 0,
                "max": 0,
                "avg": 0,
                "sum": 0
            }
        
        values = [point.value for point in history]
        
        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "avg": sum(values) / len(values),
            "sum": sum(values)
        }
    
    def get_all_metrics_summary(self, hours: int = 1) -> Dict[str, Any]:
        """Get summary of all metrics"""
        summary = {}
        
        for metric_name in self.metric_definitions:
            summary[metric_name] = self.get_metric_summary(metric_name, hours)
        
        return summary

class AlertManager:
    """Manages alerting rules and notifications"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.alert_rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        
        # Default alert rules
        self._setup_default_rules()
    
    def _setup_default_rules(self):
        """Setup default alert rules"""
        default_rules = [
            AlertRule(
                name="high_failure_rate",
                metric_name="circuit_breaker_failure_rate",
                condition=">",
                threshold=0.8,
                severity=AlertSeverity.CRITICAL,
                description="Circuit breaker failure rate is too high"
            ),
            AlertRule(
                name="high_retry_rate",
                metric_name="retry_attempts_total",
                condition=">",
                threshold=100,
                severity=AlertSeverity.WARNING,
                description="High number of retry attempts"
            ),
            AlertRule(
                name="queue_overflow",
                metric_name="rate_limit_queue_size",
                condition=">",
                threshold=0.9,
                severity=AlertSeverity.ERROR,
                description="Rate limit queue is nearly full"
            ),
            AlertRule(
                name="db_connection_issues",
                metric_name="db_connection_errors_total",
                condition=">",
                threshold=10,
                severity=AlertSeverity.ERROR,
                description="High number of database connection errors"
            ),
            AlertRule(
                name="shutdown_failures",
                metric_name="shutdown_tasks_failed",
                condition=">",
                threshold=0,
                severity=AlertSeverity.CRITICAL,
                description="Shutdown tasks are failing"
            )
        ]
        
        for rule in default_rules:
            self.add_alert_rule(rule)
    
    def add_alert_rule(self, rule: AlertRule):
        """Add an alert rule"""
        self.alert_rules[rule.name] = rule
        self.logger.info(f"✅ Added alert rule: {rule.name}")
    
    def remove_alert_rule(self, rule_name: str):
        """Remove an alert rule"""
        if rule_name in self.alert_rules:
            del self.alert_rules[rule_name]
            self.logger.info(f"✅ Removed alert rule: {rule_name}")
    
    def check_alerts(self, metrics: ResilienceMetrics):
        """Check all alert rules against current metrics"""
        current_time = datetime.now(timezone.utc)
        
        for rule_name, rule in self.alert_rules.items():
            if not rule.enabled:
                continue
            
            # Get current metric value
            current_value = metrics.get_metric_value(rule.metric_name)
            if current_value is None:
                continue
            
            # Check if alert should be triggered
            should_alert = self._evaluate_condition(
                current_value, rule.condition, rule.threshold
            )
            
            if should_alert:
                self._trigger_alert(rule, current_value, current_time)
            else:
                self._resolve_alert(rule_name, current_time)
    
    def _evaluate_condition(self, value: float, condition: str, threshold: float) -> bool:
        """Evaluate if a condition is met"""
        if condition == ">":
            return value > threshold
        elif condition == "<":
            return value < threshold
        elif condition == ">=":
            return value >= threshold
        elif condition == "<=":
            return value <= threshold
        elif condition == "==":
            return value == threshold
        elif condition == "!=":
            return value != threshold
        else:
            return False
    
    def _trigger_alert(self, rule: AlertRule, current_value: float, timestamp: datetime):
        """Trigger a new alert"""
        alert_id = f"{rule.name}_{int(timestamp.timestamp())}"
        
        # Check if alert is already active
        if rule.name in self.active_alerts:
            active_alert = self.active_alerts[rule.name]
            
            # Check cooldown
            time_since_last = (timestamp - active_alert.timestamp).total_seconds()
            if time_since_last < rule.cooldown:
                return  # Still in cooldown period
        
        # Create new alert
        alert = Alert(
            id=alert_id,
            severity=rule.severity,
            message=rule.description,
            timestamp=timestamp,
            source=rule.metric_name,
            metric_name=rule.metric_name,
            threshold=rule.threshold,
            current_value=current_value
        )
        
        self.active_alerts[rule.name] = alert
        self.alert_history.append(alert)
        
        # Log alert
        self.logger.warning(f"🚨 ALERT [{rule.severity.value.upper()}] {rule.name}: {rule.description} "
                           f"(Value: {current_value}, Threshold: {rule.threshold})")
        
        # Keep only last 1000 alerts in history
        if len(self.alert_history) > 1000:
            self.alert_history = self.alert_history[-1000:]
    
    def _resolve_alert(self, rule_name: str, timestamp: datetime):
        """Resolve an alert"""
        if rule_name in self.active_alerts:
            alert = self.active_alerts[rule_name]
            alert.resolved = True
            alert.resolved_at = timestamp
            
            self.logger.info(f"✅ Alert resolved: {rule_name}")
            
            # Remove from active alerts
            del self.active_alerts[rule_name]
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all currently active alerts"""
        return list(self.active_alerts.values())
    
    def get_alert_history(self, hours: int = 24) -> List[Alert]:
        """Get alert history for the specified time period"""
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
        
        return [
            alert for alert in self.alert_history
            if alert.timestamp > cutoff_time
        ]
    
    def get_alert_summary(self) -> Dict[str, Any]:
        """Get summary of alert statistics"""
        active_alerts = self.get_active_alerts()
        
        severity_counts = defaultdict(int)
        for alert in active_alerts:
            severity_counts[alert.severity.value] += 1
        
        return {
            "total_active": len(active_alerts),
            "by_severity": dict(severity_counts),
            "critical_count": severity_counts[AlertSeverity.CRITICAL.value],
            "error_count": severity_counts[AlertSeverity.ERROR.value],
            "warning_count": severity_counts[AlertSeverity.WARNING.value],
            "info_count": severity_counts[AlertSeverity.INFO.value]
        }

class ResilienceDashboard:
    """Main dashboard for resilience monitoring"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.metrics = ResilienceMetrics()
        self.alert_manager = AlertManager()
        
        # Dashboard state
        self.last_update = datetime.now(timezone.utc)
        self.update_interval = 5.0  # Update every 5 seconds
        
        # Start monitoring task
        self._monitoring_task = None
        self._start_monitoring()
    
    def _start_monitoring(self):
        """Start background monitoring task"""
        async def monitoring_loop():
            while True:
                try:
                    await self._collect_metrics()
                    await self._check_alerts()
                    await asyncio.sleep(self.update_interval)
                except Exception as e:
                    self.logger.error(f"❌ Error in monitoring loop: {e}")
                    await asyncio.sleep(10)
        
        self._monitoring_task = asyncio.create_task(monitoring_loop())
    
    async def _collect_metrics(self):
        """Collect metrics from all resilience components"""
        try:
            # Collect resilience metrics
            await self._collect_resilience_metrics()
            
            # Collect rate limiting metrics
            await self._collect_rate_limiting_metrics()
            
            # Collect shutdown metrics
            await self._collect_shutdown_metrics()
            
            # Update last update time
            self.last_update = datetime.now(timezone.utc)
            
        except Exception as e:
            self.logger.error(f"❌ Error collecting metrics: {e}")
    
    async def _collect_resilience_metrics(self):
        """Collect metrics from resilience manager"""
        try:
            resilience_manager = get_resilience_manager()
            
            # Get resilience status
            status = resilience_manager.get_resilience_status()
            
            # Record retry metrics
            self.metrics.set_gauge("retry_attempts_total", status.get('total_retry_attempts', 0))
            self.metrics.set_gauge("retry_successes_total", status.get('total_retry_successes', 0))
            self.metrics.set_gauge("retry_failures_total", status.get('total_retry_failures', 0))
            
            # Record circuit breaker metrics
            circuit_breakers = status.get('circuit_breakers', {})
            for cb_name, cb_status in circuit_breakers.items():
                self.metrics.set_gauge("circuit_breaker_state", 
                                    1 if cb_status.get('state') == 'open' else 0,
                                    {"name": cb_name})
                
                if cb_status.get('state') == 'open':
                    self.metrics.increment_counter("circuit_breaker_opens_total", {"name": cb_name})
                elif cb_status.get('state') == 'closed':
                    self.metrics.increment_counter("circuit_breaker_closes_total", {"name": cb_name})
            
            # Record DLQ metrics
            dlq_status = status.get('dead_letter_queue', {})
            self.metrics.set_gauge("dlq_size", dlq_status.get('queue_size', 0))
            self.metrics.set_gauge("dlq_operations_total", dlq_status.get('total_operations', 0))
            
        except Exception as e:
            self.logger.error(f"❌ Error collecting resilience metrics: {e}")
    
    async def _collect_rate_limiting_metrics(self):
        """Collect metrics from rate limiting manager"""
        try:
            rate_limit_manager = get_rate_limit_manager()
            
            # Get global stats
            global_stats = rate_limit_manager.get_global_stats()
            
            self.metrics.set_gauge("rate_limit_requests_total", global_stats.get('total_requests', 0))
            self.metrics.set_gauge("rate_limit_allowed_total", global_stats.get('total_allowed', 0))
            self.metrics.set_gauge("rate_limit_blocked_total", global_stats.get('total_blocked', 0))
            
            # Get individual rate limiter stats
            for limiter_name in rate_limit_manager.rate_limiters:
                stats = rate_limit_manager.get_rate_limit_stats(limiter_name)
                self.metrics.set_gauge("rate_limit_queue_size", 
                                    stats.get('queue_utilization', 0) / 100,
                                    {"limiter": limiter_name})
            
        except Exception as e:
            self.logger.error(f"❌ Error collecting rate limiting metrics: {e}")
    
    async def _collect_shutdown_metrics(self):
        """Collect metrics from shutdown manager"""
        try:
            shutdown_manager = get_shutdown_manager()
            
            # Get shutdown status
            status = shutdown_manager.get_shutdown_status()
            
            self.metrics.set_gauge("shutdown_duration", status.get('shutdown_duration', 0))
            self.metrics.set_gauge("shutdown_tasks_completed", status.get('completed_tasks', 0))
            self.metrics.set_gauge("shutdown_tasks_failed", status.get('failed_tasks', 0))
            
        except Exception as e:
            self.logger.error(f"❌ Error collecting shutdown metrics: {e}")
    
    async def _check_alerts(self):
        """Check all alert rules"""
        try:
            self.alert_manager.check_alerts(self.metrics)
        except Exception as e:
            self.logger.error(f"❌ Error checking alerts: {e}")
    
    def get_dashboard_data(self, hours: int = 1) -> Dict[str, Any]:
        """Get comprehensive dashboard data"""
        return {
            "last_update": self.last_update.isoformat(),
            "metrics_summary": self.metrics.get_all_metrics_summary(hours),
            "alerts_summary": self.alert_manager.get_alert_summary(),
            "active_alerts": [asdict(alert) for alert in self.alert_manager.get_active_alerts()],
            "alert_history": [asdict(alert) for alert in self.alert_manager.get_alert_history(hours)],
            "system_health": self._get_system_health(),
            "performance_metrics": self._get_performance_metrics(hours)
        }
    
    def _get_system_health(self) -> Dict[str, Any]:
        """Calculate overall system health score"""
        try:
            # Get key metrics
            failure_rate = self.metrics.get_metric_value("circuit_breaker_failure_rate") or 0
            retry_rate = self.metrics.get_metric_value("retry_attempts_total") or 0
            queue_utilization = self.metrics.get_metric_value("rate_limit_queue_size") or 0
            db_errors = self.metrics.get_metric_value("db_connection_errors_total") or 0
            
            # Calculate health score (0-100, higher is better)
            health_score = 100
            
            # Deduct points for various issues
            if failure_rate > 0.5:
                health_score -= 30
            elif failure_rate > 0.2:
                health_score -= 15
            
            if retry_rate > 100:
                health_score -= 20
            elif retry_rate > 50:
                health_score -= 10
            
            if queue_utilization > 0.8:
                health_score -= 20
            elif queue_utilization > 0.5:
                health_score -= 10
            
            if db_errors > 20:
                health_score -= 25
            elif db_errors > 10:
                health_score -= 15
            
            # Ensure score is within bounds
            health_score = max(0, min(100, health_score))
            
            # Determine health status
            if health_score >= 80:
                status = "healthy"
            elif health_score >= 60:
                status = "degraded"
            elif health_score >= 40:
                status = "unhealthy"
            else:
                status = "critical"
            
            return {
                "score": health_score,
                "status": status,
                "failure_rate": failure_rate,
                "retry_rate": retry_rate,
                "queue_utilization": queue_utilization,
                "db_errors": db_errors
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating system health: {e}")
            return {
                "score": 0,
                "status": "unknown",
                "error": str(e)
            }
    
    def _get_performance_metrics(self, hours: int) -> Dict[str, Any]:
        """Get performance-related metrics"""
        try:
            # Get operation duration metrics
            duration_summary = self.metrics.get_metric_summary("operation_duration_avg", hours)
            
            # Get retry efficiency
            retry_summary = self.metrics.get_metric_summary("retry_attempts_total", hours)
            retry_successes = self.metrics.get_metric_summary("retry_successes_total", hours)
            
            # Calculate retry success rate
            retry_success_rate = 0
            if retry_summary["sum"] > 0:
                retry_success_rate = (retry_successes["sum"] / retry_summary["sum"]) * 100
            
            return {
                "operation_duration": duration_summary,
                "retry_efficiency": {
                    "total_attempts": retry_summary["sum"],
                    "successful_retries": retry_successes["sum"],
                    "success_rate_percent": retry_success_rate
                },
                "throughput": {
                    "requests_per_minute": self._calculate_throughput(hours)
                }
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating performance metrics: {e}")
            return {"error": str(e)}
    
    def _calculate_throughput(self, hours: int) -> float:
        """Calculate requests per minute throughput"""
        try:
            # Get rate limit requests for the last hour
            requests_summary = self.metrics.get_metric_summary("rate_limit_requests_total", hours)
            
            # Calculate requests per minute
            if hours > 0:
                return requests_summary["sum"] / (hours * 60)
            return 0
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating throughput: {e}")
            return 0

# Global dashboard instance
_resilience_dashboard = None

def get_resilience_dashboard() -> ResilienceDashboard:
    """Get the global resilience dashboard instance"""
    global _resilience_dashboard
    if _resilience_dashboard is None:
        _resilience_dashboard = ResilienceDashboard()
    return _resilience_dashboard

def get_dashboard_data(hours: int = 1) -> Dict[str, Any]:
    """Get dashboard data (convenience function)"""
    dashboard = get_resilience_dashboard()
    return dashboard.get_dashboard_data(hours)

async def start_monitoring():
    """Start the resilience monitoring system"""
    dashboard = get_resilience_dashboard()
    # Monitoring is started automatically when dashboard is created
    return dashboard
