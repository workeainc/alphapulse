"""
Production Deployment Dashboard for AlphaPulse
Comprehensive dashboard for monitoring production deployments, system health, and alerts
"""

import asyncio
import logging
import json
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import asyncpg
import aiohttp
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import importlib.util

# Import production config
spec = importlib.util.spec_from_file_location('production', 'config/production.py')
production_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(production_module)
production_config = production_module.production_config

logger = logging.getLogger(__name__)

@dataclass
class DashboardMetrics:
    """Dashboard metrics"""
    total_deployments: int = 0
    active_deployments: int = 0
    failed_deployments: int = 0
    successful_deployments: int = 0
    total_services: int = 0
    healthy_services: int = 0
    unhealthy_services: int = 0
    active_alerts: int = 0
    critical_alerts: int = 0
    system_cpu_percent: float = 0.0
    system_memory_percent: float = 0.0
    system_disk_percent: float = 0.0
    average_response_time: float = 0.0
    error_rate: float = 0.0

class ProductionDeploymentDashboard:
    """
    Production deployment dashboard with real-time monitoring and alerting
    """
    
    def __init__(self, db_pool: asyncpg.Pool):
        self.db_pool = db_pool
        self.is_running = False
        
        # FastAPI app
        self.app = FastAPI(title="AlphaPulse Production Dashboard", version="1.0.0")
        self.setup_routes()
        
        # WebSocket connections
        self.websocket_connections: List[WebSocket] = []
        
        # Dashboard metrics
        self.metrics = DashboardMetrics()
        
        # Background tasks
        self.background_tasks: List[asyncio.Task] = []
        
        # Notification queue
        self.notification_queue: List[Dict[str, Any]] = []
        
        logger.info("Production Deployment Dashboard initialized")
    
    def setup_routes(self):
        """Setup FastAPI routes"""
        
        @self.app.get("/", response_class=HTMLResponse)
        async def dashboard_home():
            """Dashboard home page"""
            return self.get_dashboard_html()
        
        @self.app.get("/api/metrics")
        async def get_metrics():
            """Get current dashboard metrics"""
            return await self.get_current_metrics()
        
        @self.app.get("/api/deployments")
        async def get_deployments():
            """Get deployment summary"""
            return await self.get_deployment_summary()
        
        @self.app.get("/api/services")
        async def get_services():
            """Get service health summary"""
            return await self.get_service_health_summary()
        
        @self.app.get("/api/alerts")
        async def get_alerts():
            """Get active alerts"""
            return await self.get_active_alerts()
        
        @self.app.get("/api/system")
        async def get_system_health():
            """Get system health metrics"""
            return await self.get_system_health_metrics()
        
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket endpoint for real-time updates and notifications"""
            await websocket.accept()
            self.websocket_connections.append(websocket)
            
            try:
                while True:
                    # Send real-time updates
                    metrics = await self.get_current_metrics()
                    await websocket.send_text(json.dumps({
                        "type": "metrics_update",
                        "data": metrics,
                        "timestamp": datetime.now().isoformat()
                    }))
                    await asyncio.sleep(5)  # Update every 5 seconds
            except WebSocketDisconnect:
                self.websocket_connections.remove(websocket)
        
        @self.app.websocket("/ws/notifications")
        async def notification_websocket(websocket: WebSocket):
            """WebSocket endpoint for real-time notifications"""
            await websocket.accept()
            self.websocket_connections.append(websocket)
            
            try:
                while True:
                    # Keep connection alive
                    await websocket.receive_text()
            except WebSocketDisconnect:
                self.websocket_connections.remove(websocket)
    
    def get_dashboard_html(self) -> str:
        """Get dashboard HTML"""
        return """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>AlphaPulse Production Dashboard</title>
            <style>
                body {
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    margin: 0;
                    padding: 20px;
                    background-color: #f5f5f5;
                }
                .header {
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 20px;
                    border-radius: 10px;
                    margin-bottom: 20px;
                    text-align: center;
                }
                .metrics-grid {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                    gap: 20px;
                    margin-bottom: 20px;
                }
                .metric-card {
                    background: white;
                    padding: 20px;
                    border-radius: 10px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                }
                .metric-title {
                    font-size: 18px;
                    font-weight: bold;
                    margin-bottom: 10px;
                    color: #333;
                }
                .metric-value {
                    font-size: 32px;
                    font-weight: bold;
                    color: #667eea;
                }
                .metric-subtitle {
                    font-size: 14px;
                    color: #666;
                    margin-top: 5px;
                }
                .status-healthy { color: #28a745; }
                .status-warning { color: #ffc107; }
                .status-critical { color: #dc3545; }
                .deployments-table {
                    background: white;
                    border-radius: 10px;
                    padding: 20px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                }
                table {
                    width: 100%;
                    border-collapse: collapse;
                }
                th, td {
                    padding: 12px;
                    text-align: left;
                    border-bottom: 1px solid #ddd;
                }
                th {
                    background-color: #f8f9fa;
                    font-weight: bold;
                }
                .refresh-btn {
                    background: #667eea;
                    color: white;
                    border: none;
                    padding: 10px 20px;
                    border-radius: 5px;
                    cursor: pointer;
                    margin-bottom: 20px;
                }
                .refresh-btn:hover {
                    background: #5a6fd8;
                }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🚀 AlphaPulse Production Dashboard</h1>
                <p>Real-time monitoring and deployment management</p>
            </div>
            
            <button class="refresh-btn" onclick="refreshDashboard()">🔄 Refresh Dashboard</button>
            
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-title">Deployments</div>
                    <div class="metric-value" id="total-deployments">-</div>
                    <div class="metric-subtitle">Total Deployments</div>
                </div>
                <div class="metric-card">
                    <div class="metric-title">Active Deployments</div>
                    <div class="metric-value status-healthy" id="active-deployments">-</div>
                    <div class="metric-subtitle">Currently Running</div>
                </div>
                <div class="metric-card">
                    <div class="metric-title">Services</div>
                    <div class="metric-value" id="total-services">-</div>
                    <div class="metric-subtitle">Total Services</div>
                </div>
                <div class="metric-card">
                    <div class="metric-title">Healthy Services</div>
                    <div class="metric-value status-healthy" id="healthy-services">-</div>
                    <div class="metric-subtitle">Operating Normally</div>
                </div>
                <div class="metric-card">
                    <div class="metric-title">System CPU</div>
                    <div class="metric-value" id="system-cpu">-</div>
                    <div class="metric-subtitle">Current Usage</div>
                </div>
                <div class="metric-card">
                    <div class="metric-title">System Memory</div>
                    <div class="metric-value" id="system-memory">-</div>
                    <div class="metric-subtitle">Current Usage</div>
                </div>
                <div class="metric-card">
                    <div class="metric-title">Active Alerts</div>
                    <div class="metric-value status-warning" id="active-alerts">-</div>
                    <div class="metric-subtitle">Requiring Attention</div>
                </div>
                <div class="metric-card">
                    <div class="metric-title">Response Time</div>
                    <div class="metric-value" id="response-time">-</div>
                    <div class="metric-subtitle">Average (ms)</div>
                </div>
            </div>
            
            <div class="deployments-table">
                <h2>Recent Deployments</h2>
                <table id="deployments-table">
                    <thead>
                        <tr>
                            <th>Deployment ID</th>
                            <th>Version</th>
                            <th>Environment</th>
                            <th>Status</th>
                            <th>Start Time</th>
                            <th>Duration</th>
                            <th>Health Checks</th>
                        </tr>
                    </thead>
                    <tbody id="deployments-body">
                        <tr><td colspan="7">Loading...</td></tr>
                    </tbody>
                </table>
            </div>
            
            <script>
                function refreshDashboard() {
                    loadMetrics();
                    loadDeployments();
                }
                
                async function loadMetrics() {
                    try {
                        const response = await fetch('/api/metrics');
                        const metrics = await response.json();
                        
                        document.getElementById('total-deployments').textContent = metrics.total_deployments;
                        document.getElementById('active-deployments').textContent = metrics.active_deployments;
                        document.getElementById('total-services').textContent = metrics.total_services;
                        document.getElementById('healthy-services').textContent = metrics.healthy_services;
                        document.getElementById('system-cpu').textContent = metrics.system_cpu_percent.toFixed(1) + '%';
                        document.getElementById('system-memory').textContent = metrics.system_memory_percent.toFixed(1) + '%';
                        document.getElementById('active-alerts').textContent = metrics.active_alerts;
                        document.getElementById('response-time').textContent = metrics.average_response_time.toFixed(0) + 'ms';
                        
                        // Update status colors
                        updateStatusColors(metrics);
                    } catch (error) {
                        console.error('Error loading metrics:', error);
                    }
                }
                
                async function loadDeployments() {
                    try {
                        const response = await fetch('/api/deployments');
                        const deployments = await response.json();
                        
                        const tbody = document.getElementById('deployments-body');
                        tbody.innerHTML = '';
                        
                        deployments.forEach(deployment => {
                            const row = document.createElement('tr');
                            row.innerHTML = `
                                <td>${deployment.deployment_id}</td>
                                <td>${deployment.version}</td>
                                <td>${deployment.environment}</td>
                                <td><span class="status-${deployment.deployment_status}">${deployment.deployment_status}</span></td>
                                <td>${new Date(deployment.start_time).toLocaleString()}</td>
                                <td>${deployment.deployment_duration ? Math.round(deployment.deployment_duration) + 's' : '-'}</td>
                                <td>${deployment.health_checks_passed}/${deployment.health_checks_passed + deployment.health_checks_failed}</td>
                            `;
                            tbody.appendChild(row);
                        });
                    } catch (error) {
                        console.error('Error loading deployments:', error);
                    }
                }
                
                function updateStatusColors(metrics) {
                    // Update CPU status
                    const cpuElement = document.getElementById('system-cpu');
                    if (metrics.system_cpu_percent > 80) {
                        cpuElement.className = 'metric-value status-critical';
                    } else if (metrics.system_cpu_percent > 60) {
                        cpuElement.className = 'metric-value status-warning';
                    } else {
                        cpuElement.className = 'metric-value status-healthy';
                    }
                    
                    // Update memory status
                    const memoryElement = document.getElementById('system-memory');
                    if (metrics.system_memory_percent > 85) {
                        memoryElement.className = 'metric-value status-critical';
                    } else if (metrics.system_memory_percent > 70) {
                        memoryElement.className = 'metric-value status-warning';
                    } else {
                        memoryElement.className = 'metric-value status-healthy';
                    }
                }
                
                // Load initial data
                refreshDashboard();
                
                // Auto-refresh every 30 seconds
                setInterval(refreshDashboard, 30000);
            </script>
        </body>
        </html>
        """
    
    async def get_current_metrics(self) -> Dict[str, Any]:
        """Get current dashboard metrics"""
        try:
            async with self.db_pool.acquire() as conn:
                # Get deployment metrics
                deployment_stats = await conn.fetchrow("""
                    SELECT 
                        COUNT(*) as total_deployments,
                        COUNT(CASE WHEN end_time IS NULL THEN 1 END) as active_deployments,
                        COUNT(CASE WHEN health_checks_failed > 0 THEN 1 END) as failed_deployments,
                        COUNT(CASE WHEN health_checks_failed = 0 THEN 1 END) as successful_deployments
                    FROM deployment_metrics
                """)
                
                # Get service health metrics
                service_stats = await conn.fetchrow("""
                    SELECT 
                        COUNT(*) as total_services,
                        COUNT(CASE WHEN status = 'healthy' THEN 1 END) as healthy_services,
                        COUNT(CASE WHEN status != 'healthy' THEN 1 END) as unhealthy_services
                    FROM service_health
                    WHERE last_check >= NOW() - INTERVAL '1 hour'
                """)
                
                # Get alert metrics
                alert_stats = await conn.fetchrow("""
                    SELECT 
                        COUNT(*) as active_alerts,
                        COUNT(CASE WHEN severity = 'critical' THEN 1 END) as critical_alerts
                    FROM deployment_alerts
                    WHERE status = 'active'
                """)
                
                # Get latest system metrics
                system_metrics = await conn.fetchrow("""
                    SELECT 
                        cpu_percent,
                        memory_percent,
                        disk_percent,
                        response_time_avg,
                        error_rate
                    FROM system_health_metrics
                    ORDER BY timestamp DESC
                    LIMIT 1
                """)
                
                # Update dashboard metrics
                self.metrics.total_deployments = deployment_stats['total_deployments'] or 0
                self.metrics.active_deployments = deployment_stats['active_deployments'] or 0
                self.metrics.failed_deployments = deployment_stats['failed_deployments'] or 0
                self.metrics.successful_deployments = deployment_stats['successful_deployments'] or 0
                self.metrics.total_services = service_stats['total_services'] or 0
                self.metrics.healthy_services = service_stats['healthy_services'] or 0
                self.metrics.unhealthy_services = service_stats['unhealthy_services'] or 0
                self.metrics.active_alerts = alert_stats['active_alerts'] or 0
                self.metrics.critical_alerts = alert_stats['critical_alerts'] or 0
                
                if system_metrics:
                    self.metrics.system_cpu_percent = system_metrics['cpu_percent'] or 0.0
                    self.metrics.system_memory_percent = system_metrics['memory_percent'] or 0.0
                    self.metrics.system_disk_percent = system_metrics['disk_percent'] or 0.0
                    self.metrics.average_response_time = system_metrics['response_time_avg'] or 0.0
                    self.metrics.error_rate = system_metrics['error_rate'] or 0.0
                
                return {
                    "total_deployments": self.metrics.total_deployments,
                    "active_deployments": self.metrics.active_deployments,
                    "failed_deployments": self.metrics.failed_deployments,
                    "successful_deployments": self.metrics.successful_deployments,
                    "total_services": self.metrics.total_services,
                    "healthy_services": self.metrics.healthy_services,
                    "unhealthy_services": self.metrics.unhealthy_services,
                    "active_alerts": self.metrics.active_alerts,
                    "critical_alerts": self.metrics.critical_alerts,
                    "system_cpu_percent": self.metrics.system_cpu_percent,
                    "system_memory_percent": self.metrics.system_memory_percent,
                    "system_disk_percent": self.metrics.system_disk_percent,
                    "average_response_time": self.metrics.average_response_time,
                    "error_rate": self.metrics.error_rate,
                    "timestamp": datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Error getting current metrics: {e}")
            return {}
    
    async def get_deployment_summary(self) -> List[Dict[str, Any]]:
        """Get deployment summary"""
        try:
            async with self.db_pool.acquire() as conn:
                deployments = await conn.fetch("""
                    SELECT * FROM deployment_summary
                    ORDER BY start_time DESC
                    LIMIT 20
                """)
                
                return [dict(deployment) for deployment in deployments]
                
        except Exception as e:
            logger.error(f"Error getting deployment summary: {e}")
            return []
    
    async def get_service_health_summary(self) -> List[Dict[str, Any]]:
        """Get service health summary"""
        try:
            async with self.db_pool.acquire() as conn:
                services = await conn.fetch("""
                    SELECT * FROM service_health_summary
                    ORDER BY last_check DESC
                    LIMIT 50
                """)
                
                return [dict(service) for service in services]
                
        except Exception as e:
            logger.error(f"Error getting service health summary: {e}")
            return []
    
    async def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get active alerts"""
        try:
            async with self.db_pool.acquire() as conn:
                alerts = await conn.fetch("""
                    SELECT * FROM active_alerts
                    ORDER BY created_at DESC
                    LIMIT 20
                """)
                
                return [dict(alert) for alert in alerts]
                
        except Exception as e:
            logger.error(f"Error getting active alerts: {e}")
            return []
    
    async def get_system_health_metrics(self) -> Dict[str, Any]:
        """Get system health metrics"""
        try:
            async with self.db_pool.acquire() as conn:
                metrics = await conn.fetchrow("""
                    SELECT 
                        cpu_percent,
                        memory_percent,
                        disk_percent,
                        network_io_in,
                        network_io_out,
                        active_connections,
                        total_requests,
                        error_rate,
                        response_time_avg,
                        timestamp
                    FROM system_health_metrics
                    ORDER BY timestamp DESC
                    LIMIT 1
                """)
                
                if metrics:
                    return dict(metrics)
                else:
                    return {}
                    
        except Exception as e:
            logger.error(f"Error getting system health metrics: {e}")
            return {}
    
    async def start(self):
        """Start the dashboard"""
        if self.is_running:
            logger.warning("Dashboard already running")
            return
        
        self.is_running = True
        logger.info("Starting production deployment dashboard...")
        
        # Start background tasks
        self.background_tasks = [
            asyncio.create_task(self._metrics_update_loop()),
            asyncio.create_task(self._websocket_broadcast_loop())
        ]
        
        logger.info("Production deployment dashboard started")
    
    async def stop(self):
        """Stop the dashboard"""
        if not self.is_running:
            logger.warning("Dashboard not running")
            return
        
        self.is_running = False
        logger.info("Stopping production deployment dashboard...")
        
        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*self.background_tasks, return_exceptions=True)
        
        logger.info("Production deployment dashboard stopped")
    
    async def _metrics_update_loop(self):
        """Metrics update loop"""
        while self.is_running:
            try:
                # Update metrics
                await self.get_current_metrics()
                await asyncio.sleep(30)  # Update every 30 seconds
                
            except Exception as e:
                logger.error(f"Metrics update error: {e}")
                await asyncio.sleep(30)
    
    async def _websocket_broadcast_loop(self):
        """WebSocket broadcast loop"""
        while self.is_running:
            try:
                if self.websocket_connections:
                    # Get current metrics
                    metrics = await self.get_current_metrics()
                    
                    # Broadcast to all connected clients
                    for websocket in self.websocket_connections[:]:  # Copy list to avoid modification during iteration
                        try:
                            await websocket.send_text(json.dumps(metrics))
                        except Exception as e:
                            logger.error(f"WebSocket broadcast error: {e}")
                            self.websocket_connections.remove(websocket)
                
                await asyncio.sleep(5)  # Broadcast every 5 seconds
                
            except Exception as e:
                logger.error(f"WebSocket broadcast loop error: {e}")
                await asyncio.sleep(5)
    
    def get_app(self) -> FastAPI:
        """Get FastAPI app"""
        return self.app
    
    # Notification Methods
    async def broadcast_notification(self, notification_type: str, data: Dict[str, Any]):
        """Broadcast notification to all connected WebSocket clients"""
        try:
            message = {
                "type": notification_type,
                "data": data,
                "timestamp": datetime.now().isoformat()
            }
            
            # Add to notification queue
            self.notification_queue.append(message)
            
            # Keep only last 100 notifications
            if len(self.notification_queue) > 100:
                self.notification_queue = self.notification_queue[-100:]
            
            # Broadcast to all connected clients
            for websocket in self.websocket_connections[:]:
                try:
                    await websocket.send_text(json.dumps(message))
                except Exception as e:
                    logger.error(f"Failed to send notification to WebSocket: {e}")
                    self.websocket_connections.remove(websocket)
                    
        except Exception as e:
            logger.error(f"Error broadcasting notification: {e}")
    
    async def send_signal_notification(self, symbol: str, direction: str, confidence: float, price: float):
        """Send signal notification"""
        await self.broadcast_notification("signal", {
            "symbol": symbol,
            "direction": direction,
            "confidence": confidence,
            "price": price,
            "message": f"{symbol} {direction} Signal - {confidence}% Confidence"
        })
    
    async def send_tp_notification(self, symbol: str, tp_level: int, price: float):
        """Send take profit notification"""
        await self.broadcast_notification("tp_hit", {
            "symbol": symbol,
            "tp_level": tp_level,
            "price": price,
            "message": f"{symbol} TP{tp_level} Hit at ${price}"
        })
    
    async def send_sl_notification(self, symbol: str, price: float):
        """Send stop loss notification"""
        await self.broadcast_notification("sl_hit", {
            "symbol": symbol,
            "price": price,
            "message": f"{symbol} Stop Loss Hit at ${price}"
        })
    
    async def send_system_alert(self, message: str, priority: str = "medium"):
        """Send system alert notification"""
        await self.broadcast_notification("system_alert", {
            "message": message,
            "priority": priority
        })
    
    async def send_market_update(self, condition: str, volatility: float):
        """Send market update notification"""
        await self.broadcast_notification("market_update", {
            "condition": condition,
            "volatility": volatility,
            "message": f"Market condition changed to {condition}"
        })
    
    def get_notification_queue(self) -> List[Dict[str, Any]]:
        """Get notification queue"""
        return self.notification_queue.copy()
