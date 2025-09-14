#!/usr/bin/env python3
"""
Resilience Dashboard Service for AlphaPulse
Provides HTTP API endpoints for monitoring resilience metrics and alerts
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import json

from app.core.resilience_monitoring import (
    get_resilience_dashboard, 
    start_monitoring,
    AlertRule, AlertSeverity
)

logger = logging.getLogger(__name__)

class ResilienceDashboardService:
    """FastAPI service for resilience monitoring dashboard"""
    
    def __init__(self):
        self.app = FastAPI(
            title="AlphaPulse Resilience Dashboard",
            description="Real-time monitoring and alerting for resilience features",
            version="1.0.0"
        )
        
        self.dashboard = None
        self.setup_routes()
        
        # Start monitoring on startup
        self.app.add_event_handler("startup", self.startup_event)
        self.app.add_event_handler("shutdown", self.shutdown_event)
    
    def setup_routes(self):
        """Setup API routes"""
        
        @self.app.get("/", response_class=HTMLResponse)
        async def get_dashboard():
            """Get the main dashboard HTML"""
            return self._get_dashboard_html()
        
        @self.app.get("/api/health")
        async def health_check():
            """Health check endpoint"""
            return {"status": "healthy", "timestamp": datetime.now(timezone.utc).isoformat()}
        
        @self.app.get("/api/dashboard")
        async def get_dashboard_data(hours: int = 1):
            """Get comprehensive dashboard data"""
            try:
                if self.dashboard is None:
                    raise HTTPException(status_code=503, detail="Dashboard not initialized")
                
                data = self.dashboard.get_dashboard_data(hours)
                return data
            except Exception as e:
                logger.error(f"❌ Error getting dashboard data: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/metrics")
        async def get_metrics(hours: int = 1, metric_name: Optional[str] = None):
            """Get metrics data"""
            try:
                if self.dashboard is None:
                    raise HTTPException(status_code=503, detail="Dashboard not initialized")
                
                if metric_name:
                    # Get specific metric
                    summary = self.dashboard.metrics.get_metric_summary(metric_name, hours)
                    history = self.dashboard.metrics.get_metric_history(metric_name, hours)
                    return {
                        "metric_name": metric_name,
                        "summary": summary,
                        "history": [
                            {
                                "timestamp": point.timestamp.isoformat(),
                                "value": point.value,
                                "labels": point.labels,
                                "metadata": point.metadata
                            }
                            for point in history
                        ]
                    }
                else:
                    # Get all metrics
                    return self.dashboard.metrics.get_all_metrics_summary(hours)
                    
            except Exception as e:
                logger.error(f"❌ Error getting metrics: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/alerts")
        async def get_alerts(active_only: bool = False, hours: int = 24):
            """Get alerts data"""
            try:
                if self.dashboard is None:
                    raise HTTPException(status_code=503, detail="Dashboard not initialized")
                
                if active_only:
                    alerts = self.dashboard.alert_manager.get_active_alerts()
                else:
                    alerts = self.dashboard.alert_manager.get_alert_history(hours)
                
                return [
                    {
                        "id": alert.id,
                        "severity": alert.severity.value,
                        "message": alert.message,
                        "timestamp": alert.timestamp.isoformat(),
                        "source": alert.source,
                        "metric_name": alert.metric_name,
                        "threshold": alert.threshold,
                        "current_value": alert.current_value,
                        "resolved": alert.resolved,
                        "resolved_at": alert.resolved_at.isoformat() if alert.resolved_at else None
                    }
                    for alert in alerts
                ]
                
            except Exception as e:
                logger.error(f"❌ Error getting alerts: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/alerts/summary")
        async def get_alerts_summary():
            """Get alerts summary"""
            try:
                if self.dashboard is None:
                    raise HTTPException(status_code=503, detail="Dashboard not initialized")
                
                return self.dashboard.alert_manager.get_alert_summary()
                
            except Exception as e:
                logger.error(f"❌ Error getting alerts summary: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/api/alerts/rules")
        async def add_alert_rule(rule_data: Dict[str, Any]):
            """Add a new alert rule"""
            try:
                if self.dashboard is None:
                    raise HTTPException(status_code=503, detail="Dashboard not initialized")
                
                # Validate rule data
                required_fields = ["name", "metric_name", "condition", "threshold", "severity", "description"]
                for field in required_fields:
                    if field not in rule_data:
                        raise HTTPException(status_code=400, detail=f"Missing required field: {field}")
                
                # Create alert rule
                rule = AlertRule(
                    name=rule_data["name"],
                    metric_name=rule_data["metric_name"],
                    condition=rule_data["condition"],
                    threshold=float(rule_data["threshold"]),
                    severity=AlertSeverity(rule_data["severity"]),
                    description=rule_data["description"],
                    enabled=rule_data.get("enabled", True),
                    cooldown=float(rule_data.get("cooldown", 300.0))
                )
                
                self.dashboard.alert_manager.add_alert_rule(rule)
                
                return {"message": f"Alert rule '{rule.name}' added successfully", "rule": rule_data}
                
            except Exception as e:
                logger.error(f"❌ Error adding alert rule: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.delete("/api/alerts/rules/{rule_name}")
        async def remove_alert_rule(rule_name: str):
            """Remove an alert rule"""
            try:
                if self.dashboard is None:
                    raise HTTPException(status_code=503, detail="Dashboard not initialized")
                
                self.dashboard.alert_manager.remove_alert_rule(rule_name)
                
                return {"message": f"Alert rule '{rule_name}' removed successfully"}
                
            except Exception as e:
                logger.error(f"❌ Error removing alert rule: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/system/health")
        async def get_system_health():
            """Get system health status"""
            try:
                if self.dashboard is None:
                    raise HTTPException(status_code=503, detail="Dashboard not initialized")
                
                return self.dashboard._get_system_health()
                
            except Exception as e:
                logger.error(f"❌ Error getting system health: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/system/performance")
        async def get_performance_metrics(hours: int = 1):
            """Get performance metrics"""
            try:
                if self.dashboard is None:
                    raise HTTPException(status_code=503, detail="Dashboard not initialized")
                
                return self.dashboard._get_performance_metrics(hours)
                
            except Exception as e:
                logger.error(f"❌ Error getting performance metrics: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/api/system/refresh")
        async def refresh_metrics(background_tasks: BackgroundTasks):
            """Manually refresh metrics collection"""
            try:
                if self.dashboard is None:
                    raise HTTPException(status_code=503, detail="Dashboard not initialized")
                
                # Trigger metrics collection in background
                background_tasks.add_task(self.dashboard._collect_metrics)
                background_tasks.add_task(self.dashboard._check_alerts)
                
                return {"message": "Metrics refresh initiated", "timestamp": datetime.now(timezone.utc).isoformat()}
                
            except Exception as e:
                logger.error(f"❌ Error refreshing metrics: {e}")
                raise HTTPException(status_code=500, detail=str(e))
    
    async def startup_event(self):
        """Startup event handler"""
        try:
            logger.info("🚀 Starting Resilience Dashboard Service...")
            
            # Start monitoring
            self.dashboard = await start_monitoring()
            
            logger.info("✅ Resilience Dashboard Service started successfully")
            
        except Exception as e:
            logger.error(f"❌ Error starting Resilience Dashboard Service: {e}")
            raise
    
    async def shutdown_event(self):
        """Shutdown event handler"""
        try:
            logger.info("🔄 Shutting down Resilience Dashboard Service...")
            
            if self.dashboard and self.dashboard._monitoring_task:
                self.dashboard._monitoring_task.cancel()
                try:
                    await self.dashboard._monitoring_task
                except asyncio.CancelledError:
                    pass
            
            logger.info("✅ Resilience Dashboard Service shut down successfully")
            
        except Exception as e:
            logger.error(f"❌ Error shutting down Resilience Dashboard Service: {e}")
    
    def _get_dashboard_html(self) -> str:
        """Generate the dashboard HTML"""
        return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AlphaPulse Resilience Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
            min-height: 100vh;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }
        
        .header {
            text-align: center;
            margin-bottom: 30px;
            color: white;
        }
        
        .header h1 {
            font-size: 2.5rem;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        
        .header p {
            font-size: 1.1rem;
            opacity: 0.9;
        }
        
        .dashboard-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .card {
            background: white;
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        
        .card:hover {
            transform: translateY(-5px);
            box-shadow: 0 15px 40px rgba(0,0,0,0.15);
        }
        
        .card-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 2px solid #f0f0f0;
        }
        
        .card-title {
            font-size: 1.2rem;
            font-weight: 600;
            color: #2c3e50;
        }
        
        .card-value {
            font-size: 2rem;
            font-weight: 700;
            margin: 10px 0;
        }
        
        .status-healthy { color: #27ae60; }
        .status-degraded { color: #f39c12; }
        .status-unhealthy { color: #e74c3c; }
        .status-critical { color: #c0392b; }
        
        .metric-item {
            display: flex;
            justify-content: space-between;
            margin: 8px 0;
            padding: 5px 0;
            border-bottom: 1px solid #f0f0f0;
        }
        
        .metric-label {
            font-weight: 500;
            color: #7f8c8d;
        }
        
        .metric-value {
            font-weight: 600;
            color: #2c3e50;
        }
        
        .alerts-section {
            background: white;
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        
        .alert-item {
            padding: 15px;
            margin: 10px 0;
            border-radius: 10px;
            border-left: 5px solid;
        }
        
        .alert-critical { 
            background: #fdf2f2; 
            border-left-color: #c0392b;
            color: #c0392b;
        }
        
        .alert-error { 
            background: #fef5e7; 
            border-left-color: #e74c3c;
            color: #e74c3c;
        }
        
        .alert-warning { 
            background: #fef9e7; 
            border-left-color: #f39c12;
            color: #f39c12;
        }
        
        .alert-info { 
            background: #e8f4fd; 
            border-left-color: #3498db;
            color: #3498db;
        }
        
        .refresh-btn {
            background: #3498db;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 8px;
            cursor: pointer;
            font-size: 1rem;
            transition: background 0.3s ease;
        }
        
        .refresh-btn:hover {
            background: #2980b9;
        }
        
        .loading {
            text-align: center;
            padding: 40px;
            color: #7f8c8d;
        }
        
        .error {
            background: #fdf2f2;
            color: #c0392b;
            padding: 15px;
            border-radius: 8px;
            margin: 10px 0;
        }
        
        @media (max-width: 768px) {
            .container {
                padding: 10px;
            }
            
            .header h1 {
                font-size: 2rem;
            }
            
            .dashboard-grid {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🛡️ AlphaPulse Resilience Dashboard</h1>
            <p>Real-time monitoring and alerting for your trading bot's resilience features</p>
        </div>
        
        <div style="text-align: center; margin-bottom: 20px;">
            <button class="refresh-btn" onclick="refreshDashboard()">🔄 Refresh Dashboard</button>
            <span style="margin-left: 20px; color: white; opacity: 0.8;" id="lastUpdate"></span>
        </div>
        
        <div id="dashboardContent">
            <div class="loading">Loading dashboard data...</div>
        </div>
    </div>
    
    <script>
        let dashboardData = null;
        let charts = {};
        
        async function loadDashboard() {
            try {
                const response = await fetch('/api/dashboard');
                if (!response.ok) {
                    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                }
                
                dashboardData = await response.json();
                updateDashboard();
                
            } catch (error) {
                console.error('Error loading dashboard:', error);
                document.getElementById('dashboardContent').innerHTML = 
                    `<div class="error">Error loading dashboard: ${error.message}</div>`;
            }
        }
        
        function updateDashboard() {
            if (!dashboardData) return;
            
            const content = document.getElementById('dashboardContent');
            
            // Update last update time
            const lastUpdate = new Date(dashboardData.last_update);
            document.getElementById('lastUpdate').textContent = 
                `Last updated: ${lastUpdate.toLocaleString()}`;
            
            // Build dashboard content
            let html = '';
            
            // System Health Card
            const health = dashboardData.system_health;
            html += `
                <div class="dashboard-grid">
                    <div class="card">
                        <div class="card-header">
                            <span class="card-title">🏥 System Health</span>
                            <span class="card-value status-${health.status}">${health.score}/100</span>
                        </div>
                        <div class="metric-item">
                            <span class="metric-label">Status:</span>
                            <span class="metric-value status-${health.status}">${health.status.toUpperCase()}</span>
                        </div>
                        <div class="metric-item">
                            <span class="metric-label">Failure Rate:</span>
                            <span class="metric-value">${(health.failure_rate * 100).toFixed(1)}%</span>
                        </div>
                        <div class="metric-item">
                            <span class="metric-label">Retry Rate:</span>
                            <span class="metric-value">${health.retry_rate}</span>
                        </div>
                        <div class="metric-item">
                            <span class="metric-label">Queue Utilization:</span>
                            <span class="metric-value">${(health.queue_utilization * 100).toFixed(1)}%</span>
                        </div>
                    </div>
            `;
            
            // Alerts Summary Card
            const alerts = dashboardData.alerts_summary;
            html += `
                    <div class="card">
                        <div class="card-header">
                            <span class="card-title">🚨 Active Alerts</span>
                            <span class="card-value">${alerts.total_active}</span>
                        </div>
                        <div class="metric-item">
                            <span class="metric-label">Critical:</span>
                            <span class="metric-value status-critical">${alerts.critical_count}</span>
                        </div>
                        <div class="metric-item">
                            <span class="metric-label">Errors:</span>
                            <span class="metric-value status-unhealthy">${alerts.error_count}</span>
                        </div>
                        <div class="metric-item">
                            <span class="metric-label">Warnings:</span>
                            <span class="metric-value status-degraded">${alerts.warning_count}</span>
                        </div>
                        <div class="metric-item">
                            <span class="metric-label">Info:</span>
                            <span class="metric-value status-healthy">${alerts.info_count}</span>
                        </div>
                    </div>
            `;
            
            // Performance Metrics Card
            const performance = dashboardData.performance_metrics;
            if (performance.retry_efficiency) {
                html += `
                    <div class="card">
                        <div class="card-header">
                            <span class="card-title">⚡ Performance</span>
                        </div>
                        <div class="metric-item">
                            <span class="metric-label">Retry Success Rate:</span>
                            <span class="metric-value">${performance.retry_efficiency.success_rate_percent.toFixed(1)}%</span>
                        </div>
                        <div class="metric-item">
                            <span class="metric-label">Total Retries:</span>
                            <span class="metric-value">${performance.retry_efficiency.total_attempts}</span>
                        </div>
                        <div class="metric-item">
                            <span class="metric-label">Throughput:</span>
                            <span class="metric-value">${performance.throughput.requests_per_minute.toFixed(1)} req/min</span>
                        </div>
                    </div>
                `;
            }
            
            html += '</div>';
            
            // Active Alerts Section
            if (dashboardData.active_alerts && dashboardData.active_alerts.length > 0) {
                html += `
                    <div class="alerts-section">
                        <h3 style="margin-bottom: 20px; color: #2c3e50;">🚨 Active Alerts</h3>
                `;
                
                dashboardData.active_alerts.forEach(alert => {
                    html += `
                        <div class="alert-item alert-${alert.severity}">
                            <strong>${alert.severity.toUpperCase()}:</strong> ${alert.message}<br>
                            <small>Metric: ${alert.metric_name} | Current: ${alert.current_value} | Threshold: ${alert.threshold}</small>
                        </div>
                    `;
                });
                
                html += '</div>';
            }
            
            // Metrics Summary Section
            html += `
                <div class="alerts-section">
                    <h3 style="margin-bottom: 20px; color: #2c3e50;">📊 Metrics Summary (Last Hour)</h3>
                    <div class="dashboard-grid">
            `;
            
            const metrics = dashboardData.metrics_summary;
            Object.entries(metrics).forEach(([metricName, summary]) => {
                if (summary.count > 0) {
                    html += `
                        <div class="card">
                            <div class="card-header">
                                <span class="card-title">${metricName.replace(/_/g, ' ').toUpperCase()}</span>
                            </div>
                            <div class="metric-item">
                                <span class="metric-label">Count:</span>
                                <span class="metric-value">${summary.count}</span>
                            </div>
                            <div class="metric-item">
                                <span class="metric-label">Average:</span>
                                <span class="metric-value">${summary.avg.toFixed(2)}</span>
                            </div>
                            <div class="metric-item">
                                <span class="metric-label">Max:</span>
                                <span class="metric-value">${summary.max.toFixed(2)}</span>
                            </div>
                        </div>
                    `;
                }
            });
            
            html += '</div></div>';
            
            content.innerHTML = html;
        }
        
        function refreshDashboard() {
            loadDashboard();
        }
        
        // Auto-refresh every 30 seconds
        setInterval(loadDashboard, 30000);
        
        // Load dashboard on page load
        document.addEventListener('DOMContentLoaded', loadDashboard);
    </script>
</body>
</html>
        """

# Global service instance
_resilience_dashboard_service = None

def get_resilience_dashboard_service() -> ResilienceDashboardService:
    """Get the global resilience dashboard service instance"""
    global _resilience_dashboard_service
    if _resilience_dashboard_service is None:
        _resilience_dashboard_service = ResilienceDashboardService()
    return _resilience_dashboard_service

def get_resilience_app() -> FastAPI:
    """Get the FastAPI app for the resilience dashboard"""
    service = get_resilience_dashboard_service()
    return service.app
