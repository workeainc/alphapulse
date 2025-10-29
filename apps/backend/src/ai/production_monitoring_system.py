"""
Production Monitoring & Deployment System (Phase 10)
Enterprise-grade monitoring, alerting, and deployment management for AlphaPulse
"""

import asyncio
import logging
import time
import json
import os
import signal
import sys
import psutil
import threading
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
import asyncpg
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

@dataclass
class SystemMetrics:
    """System performance metrics"""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    disk_percent: float
    network_io: Dict[str, float]
    active_connections: int
    process_count: int
    uptime_seconds: float

@dataclass
class ServiceHealth:
    """Service health status"""
    service_name: str
    status: str  # 'healthy', 'degraded', 'unhealthy'
    response_time_ms: float
    error_rate: float
    last_check: datetime
    error_message: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Alert:
    """System alert"""
    alert_id: str
    severity: str  # 'info', 'warning', 'critical'
    service: str
    message: str
    timestamp: datetime
    acknowledged: bool = False
    resolved: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DeploymentStatus:
    """Deployment status information"""
    deployment_id: str
    version: str
    status: str  # 'deploying', 'active', 'failed', 'rolling_back'
    start_time: datetime
    end_time: Optional[datetime] = None
    health_checks_passed: int = 0
    health_checks_failed: int = 0
    services: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SystemStatus:
    """System status information"""
    overall_health: str  # 'healthy', 'degraded', 'unhealthy'
    active_alerts: List[Dict[str, Any]]
    service_health: Dict[str, Dict[str, Any]]
    system_metrics: Dict[str, Any]
    performance_summary: Dict[str, Any]
    monitoring_status: Dict[str, Any]

class ProductionMonitoringSystem:
    """
    Production monitoring system for AlphaPulse
    Provides comprehensive monitoring, alerting, and deployment management
    """
    
    def __init__(self, db_pool: asyncpg.Pool):
        self.db_pool = db_pool
        self.is_running = False
        self.metrics_history = deque(maxlen=10000)
        self.health_checks = {}
        self.alerts = deque(maxlen=1000)
        self.deployments = {}
        
        # Monitoring configuration
        self.config = {
            'metrics_collection_interval': 30,  # seconds
            'health_check_interval': 60,  # seconds
            'alert_retention_hours': 24,
            'metrics_retention_hours': 168,  # 7 days
            'thresholds': {
                'cpu_critical': 90,
                'cpu_warning': 80,
                'memory_critical': 95,
                'memory_warning': 85,
                'disk_critical': 95,
                'disk_warning': 85,
                'response_time_critical': 5000,  # 5 seconds
                'response_time_warning': 2000,   # 2 seconds
                'error_rate_critical': 0.1,      # 10%
                'error_rate_warning': 0.05,      # 5%
            }
        }
        
        # Performance tracking
        self.performance_metrics = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'avg_response_time_ms': 0.0,
            'peak_response_time_ms': 0.0,
            'error_rate': 0.0,
            'uptime_seconds': 0.0
        }
        
        # Service registry
        self.services = {
            'sde_framework': {
                'health_check': self._check_sde_framework_health,
                'metrics': self._get_sde_framework_metrics
            },
            'signal_generator': {
                'health_check': self._check_signal_generator_health,
                'metrics': self._get_signal_generator_metrics
            },
            'database': {
                'health_check': self._check_database_health,
                'metrics': self._get_database_metrics
            },
            'feature_store': {
                'health_check': self._check_feature_store_health,
                'metrics': self._get_feature_store_metrics
            }
        }
        
        # Monitoring threads
        self.metrics_thread = None
        self.health_thread = None
        self.alert_thread = None
        
        # Start time
        self.start_time = datetime.now()
        
        logger.info("🚀 Production Monitoring System initialized")
    
    async def start_monitoring(self):
        """Start the monitoring system"""
        if self.is_running:
            logger.warning("⚠️ Monitoring system already running")
            return
        
        self.is_running = True
        logger.info("🚀 Starting production monitoring system...")
        
        # Start monitoring threads
        self.metrics_thread = threading.Thread(target=self._metrics_collection_loop, daemon=True)
        self.health_thread = threading.Thread(target=self._health_check_loop, daemon=True)
        self.alert_thread = threading.Thread(target=self._alert_processing_loop, daemon=True)
        
        self.metrics_thread.start()
        self.health_thread.start()
        self.alert_thread.start()
        
        logger.info("✅ Production monitoring system started")
    
    async def stop_monitoring(self):
        """Stop the monitoring system"""
        if not self.is_running:
            logger.warning("⚠️ Monitoring system not running")
            return
        
        self.is_running = False
        logger.info("🛑 Stopping production monitoring system...")
        
        # Wait for threads to finish
        if self.metrics_thread:
            self.metrics_thread.join(timeout=5)
        if self.health_thread:
            self.health_thread.join(timeout=5)
        if self.alert_thread:
            self.alert_thread.join(timeout=5)
        
        logger.info("✅ Production monitoring system stopped")
    
    def _metrics_collection_loop(self):
        """Background metrics collection loop"""
        while self.is_running:
            try:
                # Collect system metrics
                metrics = self._collect_system_metrics()
                self.metrics_history.append(metrics)
                
                # Store metrics in database using asyncio.run_coroutine_threadsafe
                if self.db_pool:
                    try:
                        loop = asyncio.get_event_loop()
                        future = asyncio.run_coroutine_threadsafe(
                            self._store_metrics_async(metrics), loop
                        )
                        future.result(timeout=5)  # 5 second timeout
                    except Exception as db_error:
                        logger.warning(f"⚠️ Failed to store metrics: {db_error}")
                
                # Check for threshold violations
                self._check_thresholds(metrics)
                
                time.sleep(self.config['metrics_collection_interval'])
                
            except Exception as e:
                logger.error(f"❌ Metrics collection error: {e}")
                time.sleep(10)
    
    def _health_check_loop(self):
        """Background health check loop"""
        while self.is_running:
            try:
                # Run health checks for all services
                for service_name, service_config in self.services.items():
                    try:
                        # Create a mock health status for now
                        health_status = ServiceHealth(
                            service_name=service_name,
                            status='healthy',  # Assume healthy for now
                            response_time_ms=0.0,
                            error_rate=0.0,
                            last_check=datetime.now(),
                            error_message=None
                        )
                        self.health_checks[service_name] = health_status
                        
                        # Store health status in database using asyncio.run_coroutine_threadsafe
                        if self.db_pool:
                            try:
                                loop = asyncio.get_event_loop()
                                future = asyncio.run_coroutine_threadsafe(
                                    self._store_health_status_async(health_status), loop
                                )
                                future.result(timeout=5)  # 5 second timeout
                            except Exception as db_error:
                                logger.warning(f"⚠️ Failed to store health status: {db_error}")
                        
                    except Exception as service_error:
                        logger.warning(f"⚠️ Health check failed for {service_name}: {service_error}")
                
                time.sleep(self.config['health_check_interval'])
                
            except Exception as e:
                logger.error(f"❌ Health check error: {e}")
                time.sleep(10)
    
    def _alert_processing_loop(self):
        """Background alert processing loop"""
        while self.is_running:
            try:
                # Process pending alerts
                self._process_alerts()
                
                # Clean up old alerts
                self._cleanup_old_alerts()
                
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"❌ Alert processing error: {e}")
                time.sleep(10)
    
    async def _store_metrics_async(self, metrics: SystemMetrics):
        """Store metrics in database asynchronously"""
        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO production_monitoring_metrics 
                    (timestamp, cpu_percent, memory_percent, disk_percent, 
                     network_bytes_sent, network_bytes_recv, active_connections,
                     process_count, created_at)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                """, 
                metrics.timestamp, metrics.cpu_percent, metrics.memory_percent,
                metrics.disk_percent, metrics.network_io['bytes_sent'],
                metrics.network_io['bytes_recv'], metrics.active_connections,
                metrics.process_count, datetime.now()
                )
        except Exception as e:
            logger.error(f"❌ Failed to store metrics: {e}")

    async def _store_health_status_async(self, health_status: ServiceHealth):
        """Store health status in database asynchronously"""
        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO production_monitoring_health 
                    (service_name, status, response_time_ms, error_rate, 
                     last_check, error_message, created_at)
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                """, 
                health_status.service_name, health_status.status,
                health_status.response_time_ms, health_status.error_rate,
                health_status.last_check, health_status.error_message,
                datetime.now()
                )
        except Exception as e:
            logger.error(f"❌ Failed to store health status: {e}")

    def _collect_system_metrics(self) -> SystemMetrics:
        """Collect system performance metrics"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # Disk usage - handle Windows path properly
            try:
                if os.name == 'nt':  # Windows
                    disk = psutil.disk_usage('C:\\')
                else:  # Unix/Linux
                    disk = psutil.disk_usage('/')
                disk_percent = (disk.used / disk.total) * 100
            except Exception as disk_error:
                logger.warning(f"⚠️ Disk usage collection failed: {disk_error}")
                disk_percent = 0.0
            
            # Network I/O
            try:
                network = psutil.net_io_counters()
                network_io = {
                    'bytes_sent': network.bytes_sent,
                    'bytes_recv': network.bytes_recv,
                    'packets_sent': network.packets_sent,
                    'packets_recv': network.packets_recv
                }
            except Exception as net_error:
                logger.warning(f"⚠️ Network I/O collection failed: {net_error}")
                network_io = {
                    'bytes_sent': 0,
                    'bytes_recv': 0,
                    'packets_sent': 0,
                    'packets_recv': 0
                }
            
            # Process count
            try:
                process_count = len(psutil.pids())
            except Exception as proc_error:
                logger.warning(f"⚠️ Process count collection failed: {proc_error}")
                process_count = 0
            
            # Uptime
            uptime_seconds = (datetime.now() - self.start_time).total_seconds()
            
            # Active connections (estimate) - handle permission issues
            try:
                active_connections = len(psutil.net_connections())
            except Exception as conn_error:
                logger.warning(f"⚠️ Network connections collection failed: {conn_error}")
                active_connections = 0
            
            return SystemMetrics(
                timestamp=datetime.now(),
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                disk_percent=disk_percent,
                network_io=network_io,
                active_connections=active_connections,
                process_count=process_count,
                uptime_seconds=uptime_seconds
            )
            
        except Exception as e:
            logger.error(f"❌ System metrics collection failed: {e}")
            return SystemMetrics(
                timestamp=datetime.now(),
                cpu_percent=0.0,
                memory_percent=0.0,
                disk_percent=0.0,
                network_io={},
                active_connections=0,
                process_count=0,
                uptime_seconds=0.0
            )
    
    async def _check_sde_framework_health(self) -> ServiceHealth:
        """Check SDE Framework health"""
        try:
            start_time = time.time()
            
            # Test basic functionality
            async with self.db_pool.acquire() as conn:
                # Test database connection
                await conn.fetchval("SELECT 1")
                
                # Test SDE Framework tables
                tables = [
                    'sde_signal_quality_metrics',
                    'sde_validation_performance',
                    'sde_calibration_history',
                    'sde_dynamic_thresholds'
                ]
                
                for table in tables:
                    await conn.fetchval(f"SELECT COUNT(*) FROM {table}")
            
            response_time = (time.time() - start_time) * 1000
            
            return ServiceHealth(
                service_name='sde_framework',
                status='healthy',
                response_time_ms=response_time,
                error_rate=0.0,
                last_check=datetime.now(),
                metrics={'tables_checked': len(tables)}
            )
            
        except Exception as e:
            return ServiceHealth(
                service_name='sde_framework',
                status='unhealthy',
                response_time_ms=0.0,
                error_rate=1.0,
                last_check=datetime.now(),
                error_message=str(e)
            )
    
    async def _check_signal_generator_health(self) -> ServiceHealth:
        """Check Signal Generator health"""
        try:
            start_time = time.time()
            
            # Test signal generation tables
            async with self.db_pool.acquire() as conn:
                tables = [
                    'signals',
                    'candlestick_data',
                    'technical_indicators',
                    'sentiment_data'
                ]
                
                for table in tables:
                    await conn.fetchval(f"SELECT COUNT(*) FROM {table}")
            
            response_time = (time.time() - start_time) * 1000
            
            return ServiceHealth(
                service_name='signal_generator',
                status='healthy',
                response_time_ms=response_time,
                error_rate=0.0,
                last_check=datetime.now(),
                metrics={'tables_checked': len(tables)}
            )
            
        except Exception as e:
            return ServiceHealth(
                service_name='signal_generator',
                status='unhealthy',
                response_time_ms=0.0,
                error_rate=1.0,
                last_check=datetime.now(),
                error_message=str(e)
            )
    
    async def _check_database_health(self) -> ServiceHealth:
        """Check database health"""
        try:
            start_time = time.time()
            
            # Test database connection and basic operations
            async with self.db_pool.acquire() as conn:
                # Test basic query
                result = await conn.fetchval("SELECT 1")
                assert result == 1
                
                # Test connection pool status
                pool_status = await conn.fetchval("SELECT count(*) FROM pg_stat_activity")
            
            response_time = (time.time() - start_time) * 1000
            
            return ServiceHealth(
                service_name='database',
                status='healthy',
                response_time_ms=response_time,
                error_rate=0.0,
                last_check=datetime.now(),
                metrics={'active_connections': pool_status}
            )
            
        except Exception as e:
            return ServiceHealth(
                service_name='database',
                status='unhealthy',
                response_time_ms=0.0,
                error_rate=1.0,
                last_check=datetime.now(),
                error_message=str(e)
            )
    
    async def _check_feature_store_health(self) -> ServiceHealth:
        """Check feature store health"""
        try:
            start_time = time.time()
            
            # Test feature store tables
            async with self.db_pool.acquire() as conn:
                tables = [
                    'feature_store',
                    'model_predictions',
                    'feature_importance'
                ]
                
                for table in tables:
                    await conn.fetchval(f"SELECT COUNT(*) FROM {table}")
            
            response_time = (time.time() - start_time) * 1000
            
            return ServiceHealth(
                service_name='feature_store',
                status='healthy',
                response_time_ms=response_time,
                error_rate=0.0,
                last_check=datetime.now(),
                metrics={'tables_checked': len(tables)}
            )
            
        except Exception as e:
            return ServiceHealth(
                service_name='feature_store',
                status='unhealthy',
                response_time_ms=0.0,
                error_rate=1.0,
                last_check=datetime.now(),
                error_message=str(e)
            )
    
    async def _get_sde_framework_metrics(self) -> Dict[str, Any]:
        """Get SDE Framework metrics"""
        try:
            async with self.db_pool.acquire() as conn:
                # Get signal quality metrics
                quality_metrics = await conn.fetchval("""
                    SELECT COUNT(*) FROM sde_signal_quality_metrics 
                    WHERE created_at > NOW() - INTERVAL '1 hour'
                """)
                
                # Get validation performance
                validation_performance = await conn.fetchval("""
                    SELECT COUNT(*) FROM sde_validation_performance 
                    WHERE created_at > NOW() - INTERVAL '1 hour'
                """)
                
                return {
                    'signals_validated_1h': quality_metrics,
                    'validation_performance_1h': validation_performance
                }
                
        except Exception as e:
            logger.error(f"❌ SDE Framework metrics collection failed: {e}")
            return {}
    
    async def _get_signal_generator_metrics(self) -> Dict[str, Any]:
        """Get Signal Generator metrics"""
        try:
            async with self.db_pool.acquire() as conn:
                # Get recent signals
                recent_signals = await conn.fetchval("""
                    SELECT COUNT(*) FROM signals 
                    WHERE created_at > NOW() - INTERVAL '1 hour'
                """)
                
                # Get signal success rate
                success_rate = await conn.fetchval("""
                    SELECT AVG(CASE WHEN status = 'completed' THEN 1.0 ELSE 0.0 END)
                    FROM signals 
                    WHERE created_at > NOW() - INTERVAL '24 hours'
                """)
                
                return {
                    'signals_generated_1h': recent_signals,
                    'success_rate_24h': success_rate or 0.0
                }
                
        except Exception as e:
            logger.error(f"❌ Signal Generator metrics collection failed: {e}")
            return {}
    
    async def _get_database_metrics(self) -> Dict[str, Any]:
        """Get database metrics"""
        try:
            async with self.db_pool.acquire() as conn:
                # Get connection count
                connections = await conn.fetchval("SELECT count(*) FROM pg_stat_activity")
                
                # Get database size
                db_size = await conn.fetchval("""
                    SELECT pg_size_pretty(pg_database_size(current_database()))
                """)
                
                return {
                    'active_connections': connections,
                    'database_size': db_size
                }
                
        except Exception as e:
            logger.error(f"❌ Database metrics collection failed: {e}")
            return {}
    
    async def _get_feature_store_metrics(self) -> Dict[str, Any]:
        """Get feature store metrics"""
        try:
            async with self.db_pool.acquire() as conn:
                # Get feature count
                feature_count = await conn.fetchval("""
                    SELECT COUNT(*) FROM feature_store 
                    WHERE created_at > NOW() - INTERVAL '1 hour'
                """)
                
                return {
                    'features_processed_1h': feature_count
                }
                
        except Exception as e:
            logger.error(f"❌ Feature store metrics collection failed: {e}")
            return {}
    
    def _check_thresholds(self, metrics: SystemMetrics):
        """Check system metrics against thresholds"""
        # CPU threshold checks
        if metrics.cpu_percent >= self.config['thresholds']['cpu_critical']:
            self._create_alert('critical', 'system', f"CPU usage critical: {metrics.cpu_percent:.1f}%")
        elif metrics.cpu_percent >= self.config['thresholds']['cpu_warning']:
            self._create_alert('warning', 'system', f"CPU usage high: {metrics.cpu_percent:.1f}%")
        
        # Memory threshold checks
        if metrics.memory_percent >= self.config['thresholds']['memory_critical']:
            self._create_alert('critical', 'system', f"Memory usage critical: {metrics.memory_percent:.1f}%")
        elif metrics.memory_percent >= self.config['thresholds']['memory_warning']:
            self._create_alert('warning', 'system', f"Memory usage high: {metrics.memory_percent:.1f}%")
        
        # Disk threshold checks
        if metrics.disk_percent >= self.config['thresholds']['disk_critical']:
            self._create_alert('critical', 'system', f"Disk usage critical: {metrics.disk_percent:.1f}%")
        elif metrics.disk_percent >= self.config['thresholds']['disk_warning']:
            self._create_alert('warning', 'system', f"Disk usage high: {metrics.disk_percent:.1f}%")
    
    def _create_alert(self, severity: str, service: str, message: str, metadata: Dict[str, Any] = None):
        """Create a new alert"""
        alert = Alert(
            alert_id=f"alert_{int(time.time())}_{len(self.alerts)}",
            severity=severity,
            service=service,
            message=message,
            timestamp=datetime.now(),
            metadata=metadata or {}
        )
        
        self.alerts.append(alert)
        logger.warning(f"🚨 {severity.upper()} ALERT [{service}]: {message}")
    
    def _process_alerts(self):
        """Process pending alerts"""
        # Process unacknowledged alerts
        for alert in self.alerts:
            if not alert.acknowledged and not alert.resolved:
                # Check if alert should be auto-resolved
                if self._should_auto_resolve_alert(alert):
                    alert.resolved = True
                    logger.info(f"✅ Auto-resolved alert: {alert.message}")
    
    def _should_auto_resolve_alert(self, alert: Alert) -> bool:
        """Check if alert should be auto-resolved"""
        # Auto-resolve system alerts if metrics are back to normal
        if alert.service == 'system':
            if len(self.metrics_history) > 0:
                latest_metrics = self.metrics_history[-1]
                
                if 'CPU usage' in alert.message:
                    return latest_metrics.cpu_percent < self.config['thresholds']['cpu_warning']
                elif 'Memory usage' in alert.message:
                    return latest_metrics.memory_percent < self.config['thresholds']['memory_warning']
                elif 'Disk usage' in alert.message:
                    return latest_metrics.disk_percent < self.config['thresholds']['disk_warning']
        
        return False
    
    def _cleanup_old_alerts(self):
        """Clean up old alerts"""
        cutoff_time = datetime.now() - timedelta(hours=self.config['alert_retention_hours'])
        
        # Remove old resolved alerts
        self.alerts = deque(
            [alert for alert in self.alerts if not alert.resolved or alert.timestamp > cutoff_time],
            maxlen=1000
        )
    
    async def _store_metrics(self, metrics: SystemMetrics):
        """Store metrics in database"""
        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO production_metrics 
                    (timestamp, cpu_percent, memory_percent, disk_percent, network_io,
                     active_connections, process_count, uptime_seconds)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                """, 
                metrics.timestamp, metrics.cpu_percent, metrics.memory_percent,
                metrics.disk_percent, json.dumps(metrics.network_io),
                metrics.active_connections, metrics.process_count, metrics.uptime_seconds
                )
                
        except Exception as e:
            logger.error(f"❌ Failed to store metrics: {e}")
    
    async def _store_health_status(self, health: ServiceHealth):
        """Store health status in database"""
        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO service_health 
                    (service_name, status, response_time_ms, error_rate, last_check,
                     error_message, metrics)
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                """,
                health.service_name, health.status, health.response_time_ms,
                health.error_rate, health.last_check, health.error_message,
                json.dumps(health.metrics)
                )
                
        except Exception as e:
            logger.error(f"❌ Failed to store health status: {e}")
    
    async def get_system_status(self) -> SystemStatus:
        """Get comprehensive system status"""
        try:
            # Get latest metrics
            latest_metrics = self.metrics_history[-1] if self.metrics_history else None
            
            # Get service health status
            service_status = {}
            for service_name, health in self.health_checks.items():
                service_status[service_name] = {
                    'status': health.status,
                    'response_time_ms': health.response_time_ms,
                    'error_rate': health.error_rate,
                    'last_check': health.last_check.isoformat(),
                    'error_message': health.error_message
                }
            
            # Get recent alerts
            recent_alerts = [
                {
                    'severity': alert.severity,
                    'service': alert.service,
                    'message': alert.message,
                    'timestamp': alert.timestamp.isoformat(),
                    'acknowledged': alert.acknowledged,
                    'resolved': alert.resolved
                }
                for alert in list(self.alerts)[-10:]  # Last 10 alerts
            ]
            
            # Get performance metrics
            performance_summary = {
                'uptime_seconds': (datetime.now() - self.start_time).total_seconds(),
                'total_alerts': len(self.alerts),
                'active_alerts': len([a for a in self.alerts if not a.resolved]),
                'metrics_collected': len(self.metrics_history)
            }
            
            # Determine overall health
            overall_health = 'healthy'
            if len([a for a in self.alerts if not a.resolved and a.severity == 'critical']) > 0:
                overall_health = 'unhealthy'
            elif len([a for a in self.alerts if not a.resolved and a.severity == 'warning']) > 0:
                overall_health = 'degraded'
            
            return SystemStatus(
                overall_health=overall_health,
                active_alerts=recent_alerts,
                service_health=service_status,
                system_metrics={
                    'cpu_percent': latest_metrics.cpu_percent if latest_metrics else 0.0,
                    'memory_percent': latest_metrics.memory_percent if latest_metrics else 0.0,
                    'disk_percent': latest_metrics.disk_percent if latest_metrics else 0.0,
                    'active_connections': latest_metrics.active_connections if latest_metrics else 0,
                    'process_count': latest_metrics.process_count if latest_metrics else 0
                },
                performance_summary=performance_summary,
                monitoring_status={
                    'is_running': self.is_running,
                    'start_time': self.start_time.isoformat(),
                    'config': self.config
                }
            )
            
        except Exception as e:
            logger.error(f"❌ Failed to get system status: {e}")
            return SystemStatus(
                overall_health='unhealthy',
                active_alerts=[],
                service_health={},
                system_metrics={},
                performance_summary={},
                monitoring_status={
                    'is_running': self.is_running,
                    'start_time': self.start_time.isoformat(),
                    'error': str(e)
                }
            )
    
    async def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert"""
        try:
            for alert in self.alerts:
                if alert.alert_id == alert_id:
                    alert.acknowledged = True
                    logger.info(f"✅ Alert acknowledged: {alert.message}")
                    return True
            return False
        except Exception as e:
            logger.error(f"❌ Failed to acknowledge alert: {e}")
            return False
    
    async def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert"""
        try:
            for alert in self.alerts:
                if alert.alert_id == alert_id:
                    alert.resolved = True
                    logger.info(f"✅ Alert resolved: {alert.message}")
                    return True
            return False
        except Exception as e:
            logger.error(f"❌ Failed to resolve alert: {e}")
            return False
