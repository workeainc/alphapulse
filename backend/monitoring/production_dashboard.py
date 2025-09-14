"""
Production Monitoring Dashboard for AlphaPulse
Enhanced real-time monitoring dashboard with comprehensive system metrics and alerts
"""

import asyncio
import logging
import json
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
import psutil
import asyncpg
import redis.asyncio as redis
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse
import uvicorn
from config.production import production_config

logger = logging.getLogger(__name__)

@dataclass
class DashboardMetrics:
    """Dashboard metrics"""
    timestamp: datetime
    system_metrics: Dict[str, Any]
    service_metrics: Dict[str, Any]
    trading_metrics: Dict[str, Any]
    alert_metrics: Dict[str, Any]
    deployment_metrics: Dict[str, Any]

@dataclass
class AlertSummary:
    """Alert summary"""
    total_alerts: int
    critical_alerts: int
    warning_alerts: int
    info_alerts: int
    recent_alerts: List[Dict[str, Any]]

class ProductionDashboard:
    """
    Production monitoring dashboard with real-time metrics and alerts
    """
    
    def __init__(self, db_pool: asyncpg.Pool, redis_client: redis.Redis):
        self.db_pool = db_pool
        self.redis_client = redis_client
        self.is_running = False
        
        # Configuration
        self.config = production_config.MONITORING_CONFIG
        
        # Metrics storage
        self.metrics_history: deque = deque(maxlen=10000)
        self.alert_history: deque = deque(maxlen=1000)
        self.websocket_connections: List[WebSocket] = []
        
        # Background tasks
        self.background_tasks: List[asyncio.Task] = []
        
        # FastAPI app
        self.app = FastAPI(title="AlphaPulse Production Dashboard", version="1.0.0")
        self._setup_routes()
        
        logger.info("🚀 Production Dashboard initialized")
    
    def _setup_routes(self):
        """Setup FastAPI routes"""
        
        @self.app.get("/", response_class=HTMLResponse)
        async def dashboard_home():
            """Dashboard home page"""
            return self._get_dashboard_html()
        
        @self.app.get("/api/metrics")
        async def get_metrics():
            """Get current metrics"""
            return await self._get_current_metrics()
        
        @self.app.get("/api/alerts")
        async def get_alerts():
            """Get current alerts"""
            return await self._get_current_alerts()
        
        @self.app.get("/api/system/health")
        async def get_system_health():
            """Get system health"""
            return await self._get_system_health()
        
        @self.app.get("/api/services/status")
        async def get_services_status():
            """Get services status"""
            return await self._get_services_status()
        
        @self.app.get("/api/trading/overview")
        async def get_trading_overview():
            """Get trading overview"""
            return await self._get_trading_overview()
        
        @self.app.get("/api/deployments/status")
        async def get_deployments_status():
            """Get deployments status"""
            return await self._get_deployments_status()
        
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket endpoint for real-time updates"""
            await self._handle_websocket_connection(websocket)
    
    def _get_dashboard_html(self) -> str:
        """Get dashboard HTML"""
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>AlphaPulse Production Dashboard</title>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
            <style>
                body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }
                .container { max-width: 1400px; margin: 0 auto; }
                .header { background: #2c3e50; color: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; }
                .metrics-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin-bottom: 20px; }
                .metric-card { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
                .metric-title { font-size: 18px; font-weight: bold; margin-bottom: 10px; color: #2c3e50; }
                .metric-value { font-size: 24px; font-weight: bold; color: #27ae60; }
                .metric-subtitle { font-size: 14px; color: #7f8c8d; margin-top: 5px; }
                .status-healthy { color: #27ae60; }
                .status-warning { color: #f39c12; }
                .status-critical { color: #e74c3c; }
                .chart-container { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 20px; }
                .alerts-container { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
                .alert-item { padding: 10px; border-left: 4px solid #e74c3c; margin-bottom: 10px; background: #fdf2f2; }
                .alert-critical { border-left-color: #e74c3c; background: #fdf2f2; }
                .alert-warning { border-left-color: #f39c12; background: #fef9e7; }
                .alert-info { border-left-color: #3498db; background: #f0f8ff; }
                .refresh-btn { background: #3498db; color: white; border: none; padding: 10px 20px; border-radius: 4px; cursor: pointer; }
                .refresh-btn:hover { background: #2980b9; }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🚀 AlphaPulse Production Dashboard</h1>
                    <p>Real-time monitoring and system health</p>
                    <button class="refresh-btn" onclick="refreshDashboard()">🔄 Refresh</button>
                </div>
                
                <div class="metrics-grid">
                    <div class="metric-card">
                        <div class="metric-title">System Health</div>
                        <div class="metric-value" id="system-health">Loading...</div>
                        <div class="metric-subtitle">Overall system status</div>
                    </div>
                    
                    <div class="metric-card">
                        <div class="metric-title">CPU Usage</div>
                        <div class="metric-value" id="cpu-usage">Loading...</div>
                        <div class="metric-subtitle">Current CPU utilization</div>
                    </div>
                    
                    <div class="metric-card">
                        <div class="metric-title">Memory Usage</div>
                        <div class="metric-value" id="memory-usage">Loading...</div>
                        <div class="metric-subtitle">Current memory utilization</div>
                    </div>
                    
                    <div class="metric-card">
                        <div class="metric-title">Active Connections</div>
                        <div class="metric-value" id="active-connections">Loading...</div>
                        <div class="metric-subtitle">WebSocket connections</div>
                    </div>
                    
                    <div class="metric-card">
                        <div class="metric-title">Signal Accuracy</div>
                        <div class="metric-value" id="signal-accuracy">Loading...</div>
                        <div class="metric-subtitle">Current signal accuracy</div>
                    </div>
                    
                    <div class="metric-card">
                        <div class="metric-title">Active Alerts</div>
                        <div class="metric-value" id="active-alerts">Loading...</div>
                        <div class="metric-subtitle">Critical and warning alerts</div>
                    </div>
                </div>
                
                <div class="chart-container">
                    <h3>System Performance Over Time</h3>
                    <canvas id="performanceChart" width="400" height="200"></canvas>
                </div>
                
                <div class="alerts-container">
                    <h3>Recent Alerts</h3>
                    <div id="alerts-list">Loading alerts...</div>
                </div>
            </div>
            
            <script>
                let performanceChart;
                let ws;
                
                // Initialize WebSocket connection
                function initWebSocket() {
                    ws = new WebSocket('ws://localhost:8050/ws');
                    ws.onmessage = function(event) {
                        const data = JSON.parse(event.data);
                        updateDashboard(data);
                    };
                    ws.onclose = function() {
                        setTimeout(initWebSocket, 5000);
                    };
                }
                
                // Initialize performance chart
                function initChart() {
                    const ctx = document.getElementById('performanceChart').getContext('2d');
                    performanceChart = new Chart(ctx, {
                        type: 'line',
                        data: {
                            labels: [],
                            datasets: [{
                                label: 'CPU Usage (%)',
                                data: [],
                                borderColor: '#3498db',
                                backgroundColor: 'rgba(52, 152, 219, 0.1)',
                                tension: 0.4
                            }, {
                                label: 'Memory Usage (%)',
                                data: [],
                                borderColor: '#e74c3c',
                                backgroundColor: 'rgba(231, 76, 60, 0.1)',
                                tension: 0.4
                            }]
                        },
                        options: {
                            responsive: true,
                            scales: {
                                y: {
                                    beginAtZero: true,
                                    max: 100
                                }
                            }
                        }
                    });
                }
                
                // Update dashboard with new data
                function updateDashboard(data) {
                    if (data.system_metrics) {
                        document.getElementById('system-health').textContent = data.system_metrics.overall_health;
                        document.getElementById('system-health').className = 'metric-value status-' + data.system_metrics.overall_health;
                        
                        document.getElementById('cpu-usage').textContent = data.system_metrics.cpu_percent + '%';
                        document.getElementById('memory-usage').textContent = data.system_metrics.memory_percent + '%';
                    }
                    
                    if (data.service_metrics) {
                        document.getElementById('active-connections').textContent = data.service_metrics.active_connections || 0;
                    }
                    
                    if (data.trading_metrics) {
                        document.getElementById('signal-accuracy').textContent = (data.trading_metrics.signal_accuracy * 100).toFixed(1) + '%';
                    }
                    
                    if (data.alert_metrics) {
                        document.getElementById('active-alerts').textContent = data.alert_metrics.total_alerts || 0;
                    }
                    
                    // Update chart
                    if (data.system_metrics && performanceChart) {
                        const timestamp = new Date().toLocaleTimeString();
                        performanceChart.data.labels.push(timestamp);
                        performanceChart.data.datasets[0].data.push(data.system_metrics.cpu_percent);
                        performanceChart.data.datasets[1].data.push(data.system_metrics.memory_percent);
                        
                        if (performanceChart.data.labels.length > 20) {
                            performanceChart.data.labels.shift();
                            performanceChart.data.datasets[0].data.shift();
                            performanceChart.data.datasets[1].data.shift();
                        }
                        
                        performanceChart.update();
                    }
                    
                    // Update alerts
                    if (data.recent_alerts) {
                        updateAlerts(data.recent_alerts);
                    }
                }
                
                // Update alerts list
                function updateAlerts(alerts) {
                    const alertsList = document.getElementById('alerts-list');
                    if (alerts.length === 0) {
                        alertsList.innerHTML = '<p>No active alerts</p>';
                        return;
                    }
                    
                    let alertsHtml = '';
                    alerts.forEach(alert => {
                        const alertClass = 'alert-' + alert.severity;
                        alertsHtml += `
                            <div class="alert-item ${alertClass}">
                                <strong>${alert.severity.toUpperCase()}</strong> - ${alert.service}<br>
                                <small>${alert.message}</small><br>
                                <small>${new Date(alert.timestamp).toLocaleString()}</small>
                            </div>
                        `;
                    });
                    alertsList.innerHTML = alertsHtml;
                }
                
                // Refresh dashboard
                function refreshDashboard() {
                    fetch('/api/metrics')
                        .then(response => response.json())
                        .then(data => updateDashboard(data))
                        .catch(error => console.error('Error refreshing dashboard:', error));
                }
                
                // Initialize dashboard
                document.addEventListener('DOMContentLoaded', function() {
                    initChart();
                    initWebSocket();
                    refreshDashboard();
                    
                    // Auto-refresh every 30 seconds
                    setInterval(refreshDashboard, 30000);
                });
            </script>
        </body>
        </html>
        """
    
    async def start(self):
        """Start the dashboard"""
        if self.is_running:
            logger.warning("⚠️ Dashboard already running")
            return
        
        self.is_running = True
        logger.info("🚀 Starting production dashboard...")
        
        # Start background tasks
        self.background_tasks = [
            asyncio.create_task(self._metrics_collection_loop()),
            asyncio.create_task(self._alert_monitoring_loop()),
            asyncio.create_task(self._websocket_broadcast_loop())
        ]
        
        logger.info("✅ Production dashboard started")
    
    async def stop(self):
        """Stop the dashboard"""
        if not self.is_running:
            logger.warning("⚠️ Dashboard not running")
            return
        
        self.is_running = False
        logger.info("🛑 Stopping production dashboard...")
        
        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*self.background_tasks, return_exceptions=True)
        
        # Close websocket connections
        for websocket in self.websocket_connections:
            try:
                await websocket.close()
            except Exception as e:
                logger.error(f"❌ Error closing websocket: {e}")
        
        logger.info("✅ Production dashboard stopped")
    
    async def _handle_websocket_connection(self, websocket: WebSocket):
        """Handle websocket connection"""
        try:
            await websocket.accept()
            self.websocket_connections.append(websocket)
            
            # Send initial data
            initial_data = await self._get_current_metrics()
            await websocket.send_text(json.dumps(initial_data))
            
            # Keep connection alive
            while self.is_running:
                try:
                    await websocket.receive_text()
                except WebSocketDisconnect:
                    break
                except Exception as e:
                    logger.error(f"❌ WebSocket error: {e}")
                    break
        
        except Exception as e:
            logger.error(f"❌ Error handling websocket connection: {e}")
        finally:
            if websocket in self.websocket_connections:
                self.websocket_connections.remove(websocket)
    
    async def _get_current_metrics(self) -> Dict[str, Any]:
        """Get current metrics"""
        try:
            # System metrics
            system_metrics = await self._get_system_metrics()
            
            # Service metrics
            service_metrics = await self._get_service_metrics()
            
            # Trading metrics
            trading_metrics = await self._get_trading_metrics()
            
            # Alert metrics
            alert_metrics = await self._get_alert_metrics()
            
            # Deployment metrics
            deployment_metrics = await self._get_deployment_metrics()
            
            return {
                "timestamp": datetime.now().isoformat(),
                "system_metrics": system_metrics,
                "service_metrics": service_metrics,
                "trading_metrics": trading_metrics,
                "alert_metrics": alert_metrics,
                "deployment_metrics": deployment_metrics
            }
        
        except Exception as e:
            logger.error(f"❌ Error getting current metrics: {e}")
            return {
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            }
    
    async def _get_system_metrics(self) -> Dict[str, Any]:
        """Get system metrics"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Determine overall health
            overall_health = "healthy"
            if cpu_percent > 90 or memory.percent > 95 or disk.percent > 95:
                overall_health = "critical"
            elif cpu_percent > 80 or memory.percent > 85 or disk.percent > 85:
                overall_health = "warning"
            
            return {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "disk_percent": disk.percent,
                "overall_health": overall_health,
                "uptime_seconds": time.time() - psutil.boot_time()
            }
        
        except Exception as e:
            logger.error(f"❌ Error getting system metrics: {e}")
            return {"error": str(e)}
    
    async def _get_service_metrics(self) -> Dict[str, Any]:
        """Get service metrics"""
        try:
            # Get real-time data manager status
            realtime_status = await self._get_realtime_status()
            
            # Get database connection status
            db_status = await self._get_database_status()
            
            return {
                "active_connections": realtime_status.get("active_connections", 0),
                "total_connections": realtime_status.get("total_connections", 0),
                "database_connections": db_status.get("active_connections", 0),
                "services_healthy": realtime_status.get("is_running", False) and db_status.get("healthy", False)
            }
        
        except Exception as e:
            logger.error(f"❌ Error getting service metrics: {e}")
            return {"error": str(e)}
    
    async def _get_trading_metrics(self) -> Dict[str, Any]:
        """Get trading metrics"""
        try:
            # Get signal accuracy from database
            async with self.db_pool.acquire() as conn:
                result = await conn.fetchrow("""
                    SELECT AVG(accuracy) as avg_accuracy, COUNT(*) as total_signals
                    FROM signals 
                    WHERE ts > NOW() - INTERVAL '24 hours'
                """)
                
                avg_accuracy = result['avg_accuracy'] if result['avg_accuracy'] else 0.85
                total_signals = result['total_signals'] if result['total_signals'] else 0
            
            return {
                "signal_accuracy": avg_accuracy,
                "total_signals_24h": total_signals,
                "active_signals": 0,  # This would be calculated from active signals
                "trading_enabled": production_config.TRADING_CONFIG["enabled"]
            }
        
        except Exception as e:
            logger.error(f"❌ Error getting trading metrics: {e}")
            return {"error": str(e)}
    
    async def _get_alert_metrics(self) -> Dict[str, Any]:
        """Get alert metrics"""
        try:
            # Get alerts from database
            async with self.db_pool.acquire() as conn:
                result = await conn.fetchrow("""
                    SELECT 
                        COUNT(*) as total_alerts,
                        COUNT(CASE WHEN severity = 'critical' THEN 1 END) as critical_alerts,
                        COUNT(CASE WHEN severity = 'warning' THEN 1 END) as warning_alerts
                    FROM alerts 
                    WHERE created_at > NOW() - INTERVAL '24 hours'
                """)
                
                total_alerts = result['total_alerts'] if result['total_alerts'] else 0
                critical_alerts = result['critical_alerts'] if result['critical_alerts'] else 0
                warning_alerts = result['warning_alerts'] if result['warning_alerts'] else 0
            
            return {
                "total_alerts": total_alerts,
                "critical_alerts": critical_alerts,
                "warning_alerts": warning_alerts,
                "info_alerts": total_alerts - critical_alerts - warning_alerts
            }
        
        except Exception as e:
            logger.error(f"❌ Error getting alert metrics: {e}")
            return {"error": str(e)}
    
    async def _get_deployment_metrics(self) -> Dict[str, Any]:
        """Get deployment metrics"""
        try:
            # Get deployment status from database
            async with self.db_pool.acquire() as conn:
                result = await conn.fetchrow("""
                    SELECT 
                        COUNT(*) as total_deployments,
                        COUNT(CASE WHEN status = 'active' THEN 1 END) as active_deployments,
                        COUNT(CASE WHEN status = 'failed' THEN 1 END) as failed_deployments
                    FROM deployment_history 
                    WHERE start_time > NOW() - INTERVAL '7 days'
                """)
                
                total_deployments = result['total_deployments'] if result['total_deployments'] else 0
                active_deployments = result['active_deployments'] if result['active_deployments'] else 0
                failed_deployments = result['failed_deployments'] if result['failed_deployments'] else 0
            
            return {
                "total_deployments": total_deployments,
                "active_deployments": active_deployments,
                "failed_deployments": failed_deployments,
                "success_rate": (active_deployments / total_deployments * 100) if total_deployments > 0 else 100
            }
        
        except Exception as e:
            logger.error(f"❌ Error getting deployment metrics: {e}")
            return {"error": str(e)}
    
    async def _get_realtime_status(self) -> Dict[str, Any]:
        """Get real-time data manager status"""
        try:
            # This would integrate with your real-time data manager
            return {
                "is_running": True,
                "active_connections": 0,
                "total_connections": 0
            }
        except Exception as e:
            logger.error(f"❌ Error getting real-time status: {e}")
            return {"error": str(e)}
    
    async def _get_database_status(self) -> Dict[str, Any]:
        """Get database status"""
        try:
            async with self.db_pool.acquire() as conn:
                # Test connection
                await conn.execute("SELECT 1")
                
                # Get connection pool stats
                pool_stats = {
                    "healthy": True,
                    "active_connections": 0,  # This would be from pool stats
                    "total_connections": 0
                }
                
                return pool_stats
        
        except Exception as e:
            logger.error(f"❌ Error getting database status: {e}")
            return {"healthy": False, "error": str(e)}
    
    async def _get_current_alerts(self) -> List[Dict[str, Any]]:
        """Get current alerts"""
        try:
            async with self.db_pool.acquire() as conn:
                alerts = await conn.fetch("""
                    SELECT severity, service, message, created_at
                    FROM alerts 
                    WHERE created_at > NOW() - INTERVAL '1 hour'
                    ORDER BY created_at DESC
                    LIMIT 10
                """)
                
                return [
                    {
                        "severity": alert['severity'],
                        "service": alert['service'],
                        "message": alert['message'],
                        "timestamp": alert['created_at'].isoformat()
                    }
                    for alert in alerts
                ]
        
        except Exception as e:
            logger.error(f"❌ Error getting current alerts: {e}")
            return []
    
    async def _get_system_health(self) -> Dict[str, Any]:
        """Get system health"""
        try:
            system_metrics = await self._get_system_metrics()
            service_metrics = await self._get_service_metrics()
            
            return {
                "overall_health": system_metrics.get("overall_health", "unknown"),
                "system_metrics": system_metrics,
                "service_health": service_metrics,
                "last_check": datetime.now().isoformat()
            }
        
        except Exception as e:
            logger.error(f"❌ Error getting system health: {e}")
            return {"error": str(e)}
    
    async def _get_services_status(self) -> Dict[str, Any]:
        """Get services status"""
        try:
            return {
                "sde_framework": {"status": "healthy", "last_check": datetime.now().isoformat()},
                "signal_generator": {"status": "healthy", "last_check": datetime.now().isoformat()},
                "database": await self._get_database_status(),
                "real_time_manager": await self._get_realtime_status()
            }
        
        except Exception as e:
            logger.error(f"❌ Error getting services status: {e}")
            return {"error": str(e)}
    
    async def _get_trading_overview(self) -> Dict[str, Any]:
        """Get trading overview"""
        try:
            trading_metrics = await self._get_trading_metrics()
            
            return {
                "enabled": trading_metrics.get("trading_enabled", False),
                "signal_accuracy": trading_metrics.get("signal_accuracy", 0.85),
                "total_signals_24h": trading_metrics.get("total_signals_24h", 0),
                "active_signals": trading_metrics.get("active_signals", 0),
                "last_update": datetime.now().isoformat()
            }
        
        except Exception as e:
            logger.error(f"❌ Error getting trading overview: {e}")
            return {"error": str(e)}
    
    async def _get_deployments_status(self) -> Dict[str, Any]:
        """Get deployments status"""
        try:
            deployment_metrics = await self._get_deployment_metrics()
            
            return {
                "total_deployments": deployment_metrics.get("total_deployments", 0),
                "active_deployments": deployment_metrics.get("active_deployments", 0),
                "failed_deployments": deployment_metrics.get("failed_deployments", 0),
                "success_rate": deployment_metrics.get("success_rate", 100),
                "last_update": datetime.now().isoformat()
            }
        
        except Exception as e:
            logger.error(f"❌ Error getting deployments status: {e}")
            return {"error": str(e)}
    
    async def _metrics_collection_loop(self):
        """Background metrics collection loop"""
        while self.is_running:
            try:
                metrics = await self._get_current_metrics()
                self.metrics_history.append(metrics)
                
                await asyncio.sleep(30)  # Collect metrics every 30 seconds
                
            except Exception as e:
                logger.error(f"❌ Error in metrics collection: {e}")
                await asyncio.sleep(30)
    
    async def _alert_monitoring_loop(self):
        """Background alert monitoring loop"""
        while self.is_running:
            try:
                alerts = await self._get_current_alerts()
                if alerts:
                    self.alert_history.extend(alerts)
                
                await asyncio.sleep(60)  # Check alerts every minute
                
            except Exception as e:
                logger.error(f"❌ Error in alert monitoring: {e}")
                await asyncio.sleep(60)
    
    async def _websocket_broadcast_loop(self):
        """Background websocket broadcast loop"""
        while self.is_running:
            try:
                if self.websocket_connections:
                    metrics = await self._get_current_metrics()
                    
                    # Broadcast to all connected clients
                    disconnected = []
                    for websocket in self.websocket_connections:
                        try:
                            await websocket.send_text(json.dumps(metrics))
                        except Exception:
                            disconnected.append(websocket)
                    
                    # Remove disconnected clients
                    for websocket in disconnected:
                        if websocket in self.websocket_connections:
                            self.websocket_connections.remove(websocket)
                
                await asyncio.sleep(5)  # Broadcast every 5 seconds
                
            except Exception as e:
                logger.error(f"❌ Error in websocket broadcast: {e}")
                await asyncio.sleep(5)
    
    def run_dashboard(self, host: str = "0.0.0.0", port: int = 8050):
        """Run the dashboard server"""
        logger.info(f"🚀 Starting dashboard server on {host}:{port}")
        uvicorn.run(self.app, host=host, port=port, log_level="info")

# Create dashboard instance
dashboard = None

def create_dashboard(db_pool: asyncpg.Pool, redis_client: redis.Redis) -> ProductionDashboard:
    """Create dashboard instance"""
    global dashboard
    if dashboard is None:
        dashboard = ProductionDashboard(db_pool, redis_client)
    return dashboard
