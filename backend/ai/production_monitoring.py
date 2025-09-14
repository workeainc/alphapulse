"""
Production Monitoring System for AlphaPulse
Phase 5C: Production Features & Monitoring

Implements:
1. Health checks and system diagnostics
2. Performance metrics collection
3. Alerting and notification system
4. Resource monitoring and optimization
5. Operational dashboards and reporting
"""

import asyncio
import logging
import time
import psutil
import os
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import json
import hashlib
import threading
from dataclasses import dataclass, asdict
from enum import Enum

# Local imports
from ..core.prefect_config import prefect_settings
from .advanced_logging_system import redis_logger, EventType, LogLevel

logger = logging.getLogger(__name__)

class HealthStatus(Enum):
    """Health status enumeration"""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"

class AlertLevel(Enum):
    """Alert level enumeration"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class SystemMetrics:
    """System performance metrics"""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    disk_percent: float
    network_io: Dict[str, float]
    disk_io: Dict[str, float]
    process_count: int
    load_average: Optional[float] = None

@dataclass
class PipelineMetrics:
    """Pipeline performance metrics"""
    timestamp: datetime
    pipeline_type: str
    execution_time: float
    success: bool
    data_size: int
    model_performance: Dict[str, float]
    resource_usage: Dict[str, float]

@dataclass
class HealthCheck:
    """Health check result"""
    component: str
    status: HealthStatus
    message: str
    timestamp: datetime
    details: Optional[Dict[str, Any]] = None

class ProductionMonitoring:
    """
    Production monitoring system for model retraining pipeline
    Provides comprehensive health checks, metrics, and alerting
    """
    
    def __init__(self):
        self.is_running = False
        self.metrics_history = []
        self.health_checks = []
        self.alerts = []
        
        # Monitoring configuration
        self.monitoring_config = {
            'metrics_collection_interval': 30,  # seconds
            'health_check_interval': 60,  # seconds
            'metrics_retention_hours': 24,
            'alert_thresholds': {
                'cpu_critical': 90,
                'cpu_warning': 80,
                'memory_critical': 95,
                'memory_warning': 85,
                'latency_critical': 200,  # 200ms p95 threshold
                'latency_warning': 100,   # 100ms p95 threshold
                'disk_critical': 95,
                'disk_warning': 85,
                'pipeline_failure_rate_critical': 0.2,  # 20%
                'pipeline_failure_rate_warning': 0.1,  # 10%
                'execution_time_critical': 3600,  # 1 hour
                'execution_time_warning': 1800,  # 30 minutes
            }
        }
        
        # Performance tracking
        self.pipeline_executions = {
            'weekly_quick': [],
            'monthly_full': [],
            'nightly_incremental': []
        }
        
        # Latency monitoring
        self.latency_history = {
            'inference': [],
            'api': [],
            'database': [],
            'feature_computation': []
        }
        
        # Alert channels
        self.alert_channels = {
            'email': False,
            'slack': False,
            'webhook': False,
            'redis': True  # Use Redis for internal alerting
        }
        
        # Monitoring threads
        self.metrics_thread = None
        self.health_thread = None
        
        logger.info("🚀 Production Monitoring System initialized")
    
    async def start(self):
        """Start the monitoring system"""
        if self.is_running:
            logger.warning("Monitoring system is already running")
            return
        
        try:
            self.is_running = True
            
            # Start metrics collection
            self.metrics_thread = threading.Thread(
                target=self._run_metrics_collection,
                daemon=True
            )
            self.metrics_thread.start()
            
            # Start health checks
            self.health_thread = threading.Thread(
                target=self._run_health_checks,
                daemon=True
            )
            self.health_thread.start()
            
            logger.info("✅ Production Monitoring System started")
            
        except Exception as e:
            logger.error(f"❌ Failed to start monitoring system: {e}")
            raise
    
    async def stop(self):
        """Stop the monitoring system"""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # Wait for threads to finish
        if self.metrics_thread and self.metrics_thread.is_alive():
            self.metrics_thread.join(timeout=5)
        
        if self.health_thread and self.health_thread.is_alive():
            self.health_thread.join(timeout=5)
        
        logger.info("🛑 Production Monitoring System stopped")
    
    def _run_metrics_collection(self):
        """Run metrics collection in background thread"""
        while self.is_running:
            try:
                metrics = self._collect_system_metrics()
                self.metrics_history.append(metrics)
                
                # Clean up old metrics
                self._cleanup_old_metrics()
                
                # Check for threshold violations
                self._check_threshold_violations(metrics)
                
                time.sleep(self.monitoring_config['metrics_collection_interval'])
                
            except Exception as e:
                logger.error(f"Error in metrics collection: {e}")
                time.sleep(5)
    
    def _run_health_checks(self):
        """Run health checks in background thread"""
        while self.is_running:
            try:
                health_checks = self._perform_health_checks()
                self.health_checks.extend(health_checks)
                
                # Clean up old health checks
                self._cleanup_old_health_checks()
                
                time.sleep(self.monitoring_config['health_check_interval'])
                
            except Exception as e:
                logger.error(f"Error in health checks: {e}")
                time.sleep(5)
    
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
            disk_percent = disk.percent
            
            # Network I/O
            network_io = psutil.net_io_counters()
            network_metrics = {
                'bytes_sent': network_io.bytes_sent,
                'bytes_recv': network_io.bytes_recv,
                'packets_sent': network_io.packets_sent,
                'packets_recv': network_io.packets_recv
            }
            
            # Disk I/O
            disk_io = psutil.disk_io_counters()
            disk_metrics = {
                'read_bytes': disk_io.read_bytes if disk_io else 0,
                'write_bytes': disk_io.write_bytes if disk_io else 0,
                'read_count': disk_io.read_count if disk_io else 0,
                'write_count': disk_io.write_count if disk_io else 0
            }
            
            # Process count
            process_count = len(psutil.pids())
            
            # Load average (Unix-like systems)
            load_average = None
            try:
                load_average = os.getloadavg()[0] if hasattr(os, 'getloadavg') else None
            except (OSError, AttributeError):
                pass
            
            return SystemMetrics(
                timestamp=datetime.now(),
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                disk_percent=disk_percent,
                network_io=network_metrics,
                disk_io=disk_metrics,
                process_count=process_count,
                load_average=load_average
            )
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
            return SystemMetrics(
                timestamp=datetime.now(),
                cpu_percent=0.0,
                memory_percent=0.0,
                disk_percent=0.0,
                network_io={},
                disk_io={},
                process_count=0
            )
    
    def _perform_health_checks(self) -> List[HealthCheck]:
        """Perform comprehensive health checks"""
        health_checks = []
        
        try:
            # System health checks
            health_checks.extend(self._check_system_health())
            
            # Pipeline health checks
            health_checks.extend(self._check_pipeline_health())
            
            # Database health checks
            health_checks.extend(self._check_database_health())
            
            # External service health checks
            health_checks.extend(self._check_external_services())
            
        except Exception as e:
            logger.error(f"Error performing health checks: {e}")
            health_checks.append(HealthCheck(
                component="monitoring_system",
                status=HealthStatus.CRITICAL,
                message=f"Health check system error: {e}",
                timestamp=datetime.now()
            ))
        
        return health_checks
    
    def _check_system_health(self) -> List[HealthCheck]:
        """Check system health metrics"""
        checks = []
        
        if not self.metrics_history:
            return checks
        
        latest_metrics = self.metrics_history[-1]
        thresholds = self.monitoring_config['alert_thresholds']
        
        # CPU health check
        if latest_metrics.cpu_percent >= thresholds['cpu_critical']:
            status = HealthStatus.CRITICAL
        elif latest_metrics.cpu_percent >= thresholds['cpu_warning']:
            status = HealthStatus.WARNING
        else:
            status = HealthStatus.HEALTHY
        
        checks.append(HealthCheck(
            component="cpu",
            status=status,
            message=f"CPU usage: {latest_metrics.cpu_percent:.1f}%",
            timestamp=datetime.now(),
            details={'cpu_percent': latest_metrics.cpu_percent}
        ))
        
        # Memory health check
        if latest_metrics.memory_percent >= thresholds['memory_critical']:
            status = HealthStatus.CRITICAL
        elif latest_metrics.memory_percent >= thresholds['memory_warning']:
            status = HealthStatus.WARNING
        else:
            status = HealthStatus.HEALTHY
        
        checks.append(HealthCheck(
            component="memory",
            status=status,
            message=f"Memory usage: {latest_metrics.memory_percent:.1f}%",
            timestamp=datetime.now(),
            details={'memory_percent': latest_metrics.memory_percent}
        ))
        
        # Disk health check
        if latest_metrics.disk_percent >= thresholds['disk_critical']:
            status = HealthStatus.CRITICAL
        elif latest_metrics.disk_percent >= thresholds['disk_warning']:
            status = HealthStatus.WARNING
        else:
            status = HealthStatus.HEALTHY
        
        checks.append(HealthCheck(
            component="disk",
            status=status,
            message=f"Disk usage: {latest_metrics.disk_percent:.1f}%",
            timestamp=datetime.now(),
            details={'disk_percent': latest_metrics.disk_percent}
        ))
        
        return checks
    
    def _check_pipeline_health(self) -> List[HealthCheck]:
        """Check pipeline health and performance"""
        checks = []
        
        for pipeline_type, executions in self.pipeline_executions.items():
            if not executions:
                checks.append(HealthCheck(
                    component=f"pipeline_{pipeline_type}",
                    status=HealthStatus.WARNING,
                    message=f"No recent executions for {pipeline_type}",
                    timestamp=datetime.now()
                ))
                continue
            
            # Calculate failure rate
            recent_executions = [
                ex for ex in executions 
                if ex['timestamp'] > datetime.now() - timedelta(hours=24)
            ]
            
            if not recent_executions:
                checks.append(HealthCheck(
                    component=f"pipeline_{pipeline_type}",
                    status=HealthStatus.WARNING,
                    message=f"No executions in last 24h for {pipeline_type}",
                    timestamp=datetime.now()
                ))
                continue
            
            failure_rate = 1 - sum(1 for ex in recent_executions if ex['success']) / len(recent_executions)
            thresholds = self.monitoring_config['alert_thresholds']
            
            if failure_rate >= thresholds['pipeline_failure_rate_critical']:
                status = HealthStatus.CRITICAL
            elif failure_rate >= thresholds['pipeline_failure_rate_warning']:
                status = HealthStatus.WARNING
            else:
                status = HealthStatus.HEALTHY
            
            checks.append(HealthCheck(
                component=f"pipeline_{pipeline_type}",
                status=status,
                message=f"Failure rate: {failure_rate:.1%} for {pipeline_type}",
                timestamp=datetime.now(),
                details={
                    'failure_rate': failure_rate,
                    'total_executions': len(recent_executions),
                    'successful_executions': sum(1 for ex in recent_executions if ex['success'])
                }
            ))
            
            # Check execution time
            avg_execution_time = sum(ex['execution_time'] for ex in recent_executions) / len(recent_executions)
            
            if avg_execution_time >= thresholds['execution_time_critical']:
                status = HealthStatus.CRITICAL
            elif avg_execution_time >= thresholds['execution_time_warning']:
                status = HealthStatus.WARNING
            else:
                status = HealthStatus.HEALTHY
            
            checks.append(HealthCheck(
                component=f"pipeline_{pipeline_type}_performance",
                status=status,
                message=f"Avg execution time: {avg_execution_time:.1f}s for {pipeline_type}",
                timestamp=datetime.now(),
                details={'avg_execution_time': avg_execution_time}
            ))
        
        return checks
    
    def _check_database_health(self) -> List[HealthCheck]:
        """Check database connectivity and health"""
        checks = []
        
        # TODO: Implement actual database health checks
        # For now, return basic health status
        
        checks.append(HealthCheck(
            component="database",
            status=HealthStatus.HEALTHY,
            message="Database connectivity check not implemented",
            timestamp=datetime.now()
        ))
        
        return checks
    
    def _check_external_services(self) -> List[HealthCheck]:
        """Check external service health"""
        checks = []
        
        # TODO: Implement external service health checks
        # For now, return basic health status
        
        checks.append(HealthCheck(
            component="external_services",
            status=HealthStatus.HEALTHY,
            message="External service checks not implemented",
            timestamp=datetime.now()
        ))
        
        return checks
    
    def _check_threshold_violations(self, metrics: SystemMetrics):
        """Check for threshold violations and trigger alerts"""
        thresholds = self.monitoring_config['alert_thresholds']
        
        # CPU threshold check
        if metrics.cpu_percent >= thresholds['cpu_warning']:
            level = AlertLevel.CRITICAL if metrics.cpu_percent >= thresholds['cpu_critical'] else AlertLevel.WARNING
            # Schedule alert sending in background
            asyncio.create_task(self._send_alert(
                level=level,
                component="cpu",
                message=f"High CPU usage: {metrics.cpu_percent:.1f}%",
                details={'cpu_percent': metrics.cpu_percent}
            ))
        
        # Memory threshold check
        if metrics.memory_percent >= thresholds['memory_warning']:
            level = AlertLevel.CRITICAL if metrics.memory_percent >= thresholds['memory_critical'] else AlertLevel.WARNING
            # Schedule alert sending in background
            asyncio.create_task(self._send_alert(
                level=level,
                component="memory",
                message=f"High memory usage: {metrics.memory_percent:.1f}%",
                details={'memory_percent': metrics.memory_percent}
            ))
        
        # Disk threshold check
        if metrics.disk_percent >= thresholds['disk_warning']:
            level = AlertLevel.CRITICAL if metrics.disk_percent >= thresholds['disk_critical'] else AlertLevel.WARNING
            # Schedule alert sending in background
            asyncio.create_task(self._send_alert(
                level=level,
                component="disk",
                message=f"High disk usage: {metrics.disk_percent:.1f}%",
                details={'disk_percent': metrics.disk_percent}
            ))
    
    async def _send_alert(self, level: AlertLevel, component: str, message: str, details: Optional[Dict[str, Any]] = None):
        """Send alert through configured channels"""
        alert = {
            'level': level.value,
            'component': component,
            'message': message,
            'timestamp': datetime.now().isoformat(),
            'details': details or {}
        }
        
        self.alerts.append(alert)
        
        # Send to Redis if enabled
        if self.alert_channels['redis']:
            try:
                await redis_logger.log(
                    event_type=EventType.SYSTEM_ALERT,
                    data=alert,
                    log_level=LogLevel.ERROR if level in [AlertLevel.ERROR, AlertLevel.CRITICAL] else LogLevel.WARNING
                )
            except Exception as e:
                logger.error(f"Failed to send alert to Redis: {e}")
        
        # TODO: Implement other alert channels (email, Slack, webhook)
        
        logger.warning(f"🚨 ALERT [{level.value.upper()}] {component}: {message}")
    
    def record_pipeline_execution(self, pipeline_type: str, execution_data: Dict[str, Any]):
        """Record pipeline execution for monitoring"""
        execution_record = {
            'timestamp': datetime.now(),
            'pipeline_type': pipeline_type,
            'execution_time': execution_data.get('execution_time', 0),
            'success': execution_data.get('success', False),
            'data_size': execution_data.get('data_size', 0),
            'model_performance': execution_data.get('model_performance', {}),
            'resource_usage': execution_data.get('resource_usage', {})
        }
        
        if pipeline_type in self.pipeline_executions:
            self.pipeline_executions[pipeline_type].append(execution_record)
            
            # Keep only recent executions (last 100)
            if len(self.pipeline_executions[pipeline_type]) > 100:
                self.pipeline_executions[pipeline_type] = self.pipeline_executions[pipeline_type][-100:]
    
    def _cleanup_old_metrics(self):
        """Clean up old metrics data"""
        cutoff_time = datetime.now() - timedelta(hours=self.monitoring_config['metrics_retention_hours'])
        self.metrics_history = [
            m for m in self.metrics_history 
            if m.timestamp > cutoff_time
        ]
    
    def _cleanup_old_health_checks(self):
        """Clean up old health check data"""
        cutoff_time = datetime.now() - timedelta(hours=self.monitoring_config['metrics_retention_hours'])
        self.health_checks = [
            h for h in self.health_checks 
            if h.timestamp > cutoff_time
        ]
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        try:
            latest_metrics = self.metrics_history[-1] if self.metrics_history else None
            latest_health = self.health_checks[-10:] if self.health_checks else []  # Last 10 health checks
            
            # Calculate overall health status
            overall_status = HealthStatus.HEALTHY
            if any(h.status == HealthStatus.CRITICAL for h in latest_health):
                overall_status = HealthStatus.CRITICAL
            elif any(h.status == HealthStatus.WARNING for h in latest_health):
                overall_status = HealthStatus.WARNING
            
            return {
                'overall_status': overall_status.value,
                'timestamp': datetime.now().isoformat(),
                'system_metrics': asdict(latest_metrics) if latest_metrics else None,
                'recent_health_checks': [asdict(h) for h in latest_health],
                'pipeline_summary': {
                    pipeline_type: {
                        'total_executions': len(executions),
                        'recent_executions': len([e for e in executions if e['timestamp'] > datetime.now() - timedelta(hours=24)]),
                        'success_rate': sum(1 for e in executions if e['success']) / len(executions) if executions else 0
                    }
                    for pipeline_type, executions in self.pipeline_executions.items()
                },
                'alerts_summary': {
                    'total_alerts': len(self.alerts),
                    'critical_alerts': len([a for a in self.alerts if a['level'] == 'critical']),
                    'recent_alerts': [a for a in self.alerts[-10:]]  # Last 10 alerts
                },
                'monitoring_config': self.monitoring_config
            }
            
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return {
                'overall_status': 'unknown',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def record_latency(self, latency_type: str, latency_ms: float, 
                       timestamp: datetime = None, metadata: Dict[str, Any] = None):
        """Record latency measurement for monitoring"""
        try:
            timestamp = timestamp or datetime.now()
            
            if latency_type not in self.latency_history:
                logger.warning(f"⚠️ Unknown latency type: {latency_type}")
                return
            
            # Record latency with metadata
            latency_record = {
                'timestamp': timestamp,
                'latency_ms': latency_ms,
                'metadata': metadata or {}
            }
            
            self.latency_history[latency_type].append(latency_record)
            
            # Keep only recent latency data (last 1000 measurements)
            if len(self.latency_history[latency_type]) > 1000:
                self.latency_history[latency_type] = self.latency_history[latency_type][-1000:]
            
            # Check for latency drift
            self._check_latency_drift(latency_type, latency_ms)
            
            logger.debug(f"✅ Recorded {latency_type} latency: {latency_ms:.2f}ms")
            
        except Exception as e:
            logger.error(f"❌ Failed to record latency: {e}")
    
    def _check_latency_drift(self, latency_type: str, current_latency: float):
        """Check for latency drift and trigger alerts"""
        try:
            if not self.latency_history[latency_type]:
                return
            
            # Calculate p95 latency from recent measurements
            recent_latencies = [
                record['latency_ms'] for record in self.latency_history[latency_type][-100:]
            ]
            
            if len(recent_latencies) < 10:  # Need at least 10 measurements
                return
            
            # Calculate p95 latency
            p95_latency = np.percentile(recent_latencies, 95)
            
            # Check thresholds
            thresholds = self.monitoring_config['alert_thresholds']
            
            if p95_latency >= thresholds['latency_critical']:
                self._send_latency_alert('critical', latency_type, p95_latency, current_latency)
            elif p95_latency >= thresholds['latency_warning']:
                self._send_latency_alert('warning', latency_type, p95_latency, current_latency)
            
        except Exception as e:
            logger.error(f"❌ Failed to check latency drift: {e}")
    
    def _send_latency_alert(self, level: str, latency_type: str, p95_latency: float, current_latency: float):
        """Send latency drift alert"""
        try:
            message = f"Latency drift detected in {latency_type}: p95={p95_latency:.2f}ms, current={current_latency:.2f}ms"
            
            if level == 'critical':
                alert_level = AlertLevel.CRITICAL
                message += " - CRITICAL: Model size/complexity review required"
            else:
                alert_level = AlertLevel.WARNING
                message += " - WARNING: Monitor closely"
            
            # Send alert
            asyncio.create_task(self._send_alert(
                alert_level, 
                f"latency_{latency_type}", 
                message,
                {
                    'latency_type': latency_type,
                    'p95_latency': p95_latency,
                    'current_latency': current_latency,
                    'threshold': self.monitoring_config['alert_thresholds'][f'latency_{level}']
                }
            ))
            
            logger.warning(f"🚨 Latency alert [{level.upper()}]: {message}")
            
        except Exception as e:
            logger.error(f"❌ Failed to send latency alert: {e}")
    
    def get_latency_summary(self, latency_type: str = None, hours: int = 24) -> Dict[str, Any]:
        """Get latency summary for specified time period"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            if latency_type:
                # Single latency type summary
                if latency_type not in self.latency_history:
                    return {'error': f'Unknown latency type: {latency_type}'}
                
                latency_data = [
                    record for record in self.latency_history[latency_type]
                    if record['timestamp'] >= cutoff_time
                ]
                
                if not latency_data:
                    return {
                        'latency_type': latency_type,
                        'period_hours': hours,
                        'data_points': 0,
                        'message': 'No latency data available for specified period'
                    }
                
                latencies = [record['latency_ms'] for record in latency_data]
                
                return {
                    'latency_type': latency_type,
                    'period_hours': hours,
                    'data_points': len(latencies),
                    'statistics': {
                        'min': np.min(latencies),
                        'max': np.max(latencies),
                        'mean': np.mean(latencies),
                        'median': np.median(latencies),
                        'p95': np.percentile(latencies, 95),
                        'p99': np.percentile(latencies, 99),
                        'std': np.std(latencies)
                    },
                    'drift_analysis': {
                        'p95_exceeds_warning': np.percentile(latencies, 95) >= self.monitoring_config['alert_thresholds']['latency_warning'],
                        'p95_exceeds_critical': np.percentile(latencies, 95) >= self.monitoring_config['alert_thresholds']['latency_critical'],
                        'trend': self._calculate_latency_trend(latency_data)
                    }
                }
            else:
                # All latency types summary
                summary = {}
                for lt in self.latency_history.keys():
                    summary[lt] = self.get_latency_summary(lt, hours)
                
                return {
                    'period_hours': hours,
                    'latency_types': summary
                }
            
        except Exception as e:
            logger.error(f"❌ Failed to get latency summary: {e}")
            return {'error': str(e)}
    
    def _calculate_latency_trend(self, latency_data: List[Dict]) -> str:
        """Calculate latency trend (improving, stable, degrading)"""
        try:
            if len(latency_data) < 10:
                return 'insufficient_data'
            
            # Split data into two halves
            mid_point = len(latency_data) // 2
            first_half = [record['latency_ms'] for record in latency_data[:mid_point]]
            second_half = [record['latency_ms'] for record in latency_data[mid_point:]]
            
            first_mean = np.mean(first_half)
            second_mean = np.mean(second_half)
            
            # Calculate change percentage
            change_pct = (second_mean - first_mean) / first_mean
            
            if change_pct < -0.05:  # 5% improvement
                return 'improving'
            elif change_pct > 0.05:  # 5% degradation
                return 'degrading'
            else:
                return 'stable'
                
        except Exception as e:
            logger.error(f"❌ Failed to calculate latency trend: {e}")
            return 'unknown'
    
    def get_performance_report(self, hours: int = 24) -> Dict[str, Any]:
        """Get performance report for specified time period"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            # Filter metrics for time period
            period_metrics = [
                m for m in self.metrics_history 
                if m.timestamp > cutoff_time
            ]
            
            if not period_metrics:
                return {'error': 'No metrics available for specified period'}
            
            # Calculate averages
            avg_cpu = sum(m.cpu_percent for m in period_metrics) / len(period_metrics)
            avg_memory = sum(m.memory_percent for m in period_metrics) / len(period_metrics)
            avg_disk = sum(m.disk_percent for m in period_metrics) / len(period_metrics)
            
            # Calculate pipeline performance
            pipeline_performance = {}
            for pipeline_type, executions in self.pipeline_executions.items():
                period_executions = [
                    e for e in executions 
                    if e['timestamp'] > cutoff_time
                ]
                
                if period_executions:
                    pipeline_performance[pipeline_type] = {
                        'total_executions': len(period_executions),
                        'success_rate': sum(1 for e in period_executions if e['success']) / len(period_executions),
                        'avg_execution_time': sum(e['execution_time'] for e in period_executions) / len(period_executions),
                        'total_data_processed': sum(e['data_size'] for e in period_executions)
                    }
            
            return {
                'period_hours': hours,
                'start_time': cutoff_time.isoformat(),
                'end_time': datetime.now().isoformat(),
                'system_performance': {
                    'avg_cpu_percent': avg_cpu,
                    'avg_memory_percent': avg_memory,
                    'avg_disk_percent': avg_disk,
                    'metrics_count': len(period_metrics)
                },
                'pipeline_performance': pipeline_performance,
                'health_summary': {
                    'total_checks': len([h for h in self.health_checks if h.timestamp > cutoff_time]),
                    'healthy_checks': len([h for h in self.health_checks if h.timestamp > cutoff_time and h.status == HealthStatus.HEALTHY]),
                    'warning_checks': len([h for h in self.health_checks if h.timestamp > cutoff_time and h.status == HealthStatus.WARNING]),
                    'critical_checks': len([h for h in self.health_checks if h.timestamp > cutoff_time and h.status == HealthStatus.CRITICAL])
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating performance report: {e}")
            return {'error': str(e)}

# Global monitoring instance
production_monitoring = ProductionMonitoring()
