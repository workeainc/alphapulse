#!/usr/bin/env python3
"""
Multi-Region Resilience Dashboard Service for AlphaPulse
Provides web interface for monitoring and controlling multi-region operations
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse

from app.core.multi_region_resilience import (
    get_multi_region_manager,
    FailoverStrategy,
    LoadBalancingStrategy
)

logger = logging.getLogger(__name__)

class MultiRegionDashboardService:
    """FastAPI service for multi-region resilience dashboard"""
    
    def __init__(self):
        self.app = FastAPI(
            title="AlphaPulse Multi-Region Resilience",
            description="Cross-region failover, load balancing, and disaster recovery",
            version="1.0.0"
        )
        
        self.multi_region_manager = None
        self.setup_routes()
    
    def setup_routes(self):
        """Setup API routes"""
        
        @self.app.get("/", response_class=HTMLResponse)
        async def get_multi_region_dashboard():
            """Get the main multi-region dashboard HTML"""
            return self._get_multi_region_dashboard_html()
        
        @self.app.get("/api/health")
        async def health_check():
            """Health check endpoint"""
            return {"status": "healthy", "timestamp": datetime.now(timezone.utc).isoformat()}
        
        @self.app.get("/api/status")
        async def get_status():
            """Get multi-region status"""
            try:
                manager = get_multi_region_manager()
                return manager.get_current_status()
            except Exception as e:
                logger.error(f"❌ Error getting multi-region status: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/regions")
        async def get_regions():
            """Get all regions and their health"""
            try:
                manager = get_multi_region_manager()
                return {
                    "regions": [
                        {
                            "name": region.name,
                            "endpoint": region.endpoint,
                            "weight": region.weight,
                            "failover_priority": region.failover_priority,
                            "max_connections": region.max_connections,
                            "timeout": region.timeout
                        }
                        for region in manager.regions
                    ],
                    "health_status": manager.health_monitor.get_all_health_status()
                }
            except Exception as e:
                logger.error(f"❌ Error getting regions: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/failover/history")
        async def get_failover_history(limit: int = 20):
            """Get failover history"""
            try:
                manager = get_multi_region_manager()
                return {
                    "failover_history": manager.get_failover_history(limit)
                }
            except Exception as e:
                logger.error(f"❌ Error getting failover history: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/api/failover/force/{target_region}")
        async def force_failover(target_region: str):
            """Force a failover to a specific region"""
            try:
                manager = get_multi_region_manager()
                success = await manager.force_failover(target_region)
                
                if success:
                    return {
                        "message": f"Failover to {target_region} successful",
                        "success": True
                    }
                else:
                    raise HTTPException(status_code=400, detail=f"Failover to {target_region} failed")
                    
            except Exception as e:
                logger.error(f"❌ Error forcing failover: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/strategies")
        async def get_strategies():
            """Get available failover and load balancing strategies"""
            return {
                "failover_strategies": [
                    {"value": strategy.value, "name": strategy.name.replace("_", " ").title()}
                    for strategy in FailoverStrategy
                ],
                "load_balancing_strategies": [
                    {"value": strategy.value, "name": strategy.name.replace("_", " ").title()}
                    for strategy in LoadBalancingStrategy
                ]
            }
    
    def _get_multi_region_dashboard_html(self) -> str:
        """Generate the multi-region dashboard HTML"""
        return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AlphaPulse Multi-Region Resilience</title>
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
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
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
        
        .region-item {
            padding: 15px;
            margin: 10px 0;
            border-radius: 10px;
            border-left: 5px solid #3498db;
            background: #f8f9fa;
            transition: all 0.3s ease;
        }
        
        .region-item.primary {
            border-left-color: #27ae60;
            background: #f0f9f0;
        }
        
        .region-item.unhealthy {
            border-left-color: #e74c3c;
            background: #fdf2f2;
        }
        
        .region-item.degraded {
            border-left-color: #f39c12;
            background: #fef9e7;
        }
        
        .region-name {
            font-weight: 600;
            color: #2c3e50;
            margin-bottom: 5px;
        }
        
        .region-status {
            display: inline-block;
            padding: 4px 8px;
            border-radius: 12px;
            font-size: 0.8rem;
            font-weight: 500;
            text-transform: uppercase;
        }
        
        .status-healthy {
            background: #d4edda;
            color: #155724;
        }
        
        .status-unhealthy {
            background: #f8d7da;
            color: #721c24;
        }
        
        .status-degraded {
            background: #fff3cd;
            color: #856404;
        }
        
        .region-metrics {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
            margin: 10px 0;
            font-size: 0.9rem;
        }
        
        .metric-item {
            display: flex;
            justify-content: space-between;
            padding: 5px 0;
        }
        
        .metric-label {
            color: #7f8c8d;
        }
        
        .metric-value {
            font-weight: 500;
            color: #2c3e50;
        }
        
        .failover-btn {
            background: #e74c3c;
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 0.9rem;
            transition: background 0.3s ease;
            margin-top: 10px;
        }
        
        .failover-btn:hover {
            background: #c0392b;
        }
        
        .failover-btn:disabled {
            background: #bdc3c7;
            cursor: not-allowed;
        }
        
        .status-section {
            background: white;
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        
        .status-item {
            padding: 10px;
            margin: 5px 0;
            border-radius: 8px;
            background: #f8f9fa;
            border-left: 4px solid #3498db;
        }
        
        .status-success {
            border-left-color: #27ae60;
            background: #f0f9f0;
        }
        
        .status-failure {
            border-left-color: #e74c3c;
            background: #fdf2f2;
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
            margin-bottom: 20px;
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
        
        .primary-indicator {
            background: #27ae60;
            color: white;
            padding: 4px 8px;
            border-radius: 12px;
            font-size: 0.8rem;
            font-weight: 500;
            margin-left: 10px;
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
            <h1>🌍 AlphaPulse Multi-Region Resilience</h1>
            <p>Cross-region failover, load balancing, and disaster recovery for your trading bot</p>
        </div>
        
        <div style="text-align: center; margin-bottom: 20px;">
            <button class="refresh-btn" onclick="refreshDashboard()">🔄 Refresh Dashboard</button>
            <span style="margin-left: 20px; color: white; opacity: 0.8;" id="lastUpdate"></span>
        </div>
        
        <div id="dashboardContent">
            <div class="loading">Loading multi-region dashboard...</div>
        </div>
    </div>
    
    <script>
        let dashboardData = null;
        
        async function loadDashboard() {
            try {
                const response = await fetch('/api/status');
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
            const lastUpdate = new Date();
            document.getElementById('lastUpdate').textContent = 
                `Last updated: ${lastUpdate.toLocaleString()}`;
            
            // Build dashboard content
            let html = '';
            
            // Current Status Section
            html += `
                <div class="status-section">
                    <h3 style="margin-bottom: 20px; color: #2c3e50;">📊 Current Multi-Region Status</h3>
                    <div class="dashboard-grid">
                        <div class="card">
                            <div class="card-header">
                                <span class="card-title">🏆 Primary Region</span>
                            </div>
                            <div style="font-size: 1.5rem; font-weight: 600; color: #27ae60;">
                                ${dashboardData.current_primary || 'None'}
                            </div>
                            <small style="color: #7f8c8d;">Current active primary region</small>
                        </div>
                        
                        <div class="card">
                            <div class="card-header">
                                <span class="card-title">🔄 Failover Strategy</span>
                            </div>
                            <div style="font-size: 1.2rem; font-weight: 600; color: #2c3e50;">
                                ${dashboardData.failover_strategy.replace('_', ' ').toUpperCase()}
                            </div>
                            <small style="color: #7f8c8d;">Current failover approach</small>
                        </div>
                        
                        <div class="card">
                            <div class="card-header">
                                <span class="card-title">🌍 Region Health</span>
                            </div>
                            <div style="font-size: 1.5rem; font-weight: 600; color: #27ae60;">
                                ${dashboardData.healthy_regions}/${dashboardData.total_regions}
                            </div>
                            <small style="color: #7f8c8d;">Healthy regions available</small>
                        </div>
                    </div>
                </div>
            `;
            
            // Region Health Section
            html += `
                <div class="status-section">
                    <h3 style="margin-bottom: 20px; color: #2c3e50;">🏥 Region Health Status</h3>
                    <div class="dashboard-grid">
            `;
            
            const regionHealth = dashboardData.region_health || [];
            regionHealth.forEach(region => {
                const isPrimary = region.region_name === dashboardData.current_primary;
                const statusClass = `region-item ${isPrimary ? 'primary' : ''} ${region.status === 'unhealthy' ? 'unhealthy' : region.status === 'degraded' ? 'degraded' : ''}`;
                const statusClass2 = `region-status status-${region.status}`;
                
                html += `
                    <div class="${statusClass}">
                        <div class="region-name">
                            ${region.region_name}
                            ${isPrimary ? '<span class="primary-indicator">PRIMARY</span>' : ''}
                        </div>
                        <div class="${statusClass2}">${region.status}</div>
                        
                        <div class="region-metrics">
                            <div class="metric-item">
                                <span class="metric-label">Response Time:</span>
                                <span class="metric-value">${(region.response_time * 1000).toFixed(1)}ms</span>
                            </div>
                            <div class="metric-item">
                                <span class="metric-label">Error Rate:</span>
                                <span class="metric-value">${(region.error_rate * 100).toFixed(1)}%</span>
                            </div>
                            <div class="metric-item">
                                <span class="metric-label">Last Check:</span>
                                <span class="metric-value">${new Date(region.last_check).toLocaleTimeString()}</span>
                            </div>
                            <div class="metric-item">
                                <span class="metric-label">Available:</span>
                                <span class="metric-value">${region.is_available ? '✅ Yes' : '❌ No'}</span>
                            </div>
                        </div>
                        
                        ${!isPrimary && region.is_available ? `
                            <button class="failover-btn" onclick="forceFailover('${region.region_name}')">
                                🚀 Make Primary
                            </button>
                        ` : ''}
                    </div>
                `;
            });
            
            html += '</div></div>';
            
            // Recent Failovers Section
            const recentFailovers = dashboardData.recent_failovers || [];
            if (recentFailovers.length > 0) {
                html += `
                    <div class="status-section">
                        <h3 style="margin-bottom: 20px; color: #2c3e50;">📈 Recent Failover Events</h3>
                `;
                
                recentFailovers.forEach(failover => {
                    const statusClass = failover.success ? 'status-success' : 'status-failure';
                    const statusIcon = failover.success ? '✅' : '❌';
                    
                    html += `
                        <div class="status-item ${statusClass}">
                            <strong>${statusIcon} Failover Event</strong><br>
                            <small>
                                From: ${failover.from_region} → To: ${failover.to_region} | 
                                Reason: ${failover.reason} | 
                                Time: ${new Date(failover.timestamp).toLocaleString()}
                            </small>
                        </div>
                    `;
                });
                
                html += '</div>';
            }
            
            content.innerHTML = html;
        }
        
        async function forceFailover(targetRegion) {
            try {
                const response = await fetch(`/api/failover/force/${targetRegion}`, {
                    method: 'POST'
                });
                
                if (!response.ok) {
                    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                }
                
                const result = await response.json();
                
                if (result.success) {
                    alert(`✅ Failover to ${targetRegion} successful!`);
                    // Refresh dashboard after a short delay
                    setTimeout(() => {
                        refreshDashboard();
                    }, 2000);
                } else {
                    alert(`❌ Failover failed: ${result.message}`);
                }
                
            } catch (error) {
                console.error('Error forcing failover:', error);
                alert(`❌ Failover error: ${error.message}`);
            }
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
_multi_region_service = None

def get_multi_region_service() -> MultiRegionDashboardService:
    """Get the global multi-region dashboard service instance"""
    global _multi_region_service
    if _multi_region_service is None:
        _multi_region_service = MultiRegionDashboardService()
    return _multi_region_service

def get_multi_region_app() -> FastAPI:
    """Get the FastAPI app for multi-region resilience"""
    service = get_multi_region_service()
    return service.app
