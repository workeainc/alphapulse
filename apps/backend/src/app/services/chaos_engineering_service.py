#!/usr/bin/env python3
"""
Chaos Engineering Service for AlphaPulse
Provides web interface and API for running chaos experiments
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse

from src.app.core.chaos_engineering import (
    get_chaos_runner, 
    run_chaos_experiment,
    get_chaos_status,
    ChaosExperiment, ChaosType
)

logger = logging.getLogger(__name__)

class ChaosEngineeringService:
    """FastAPI service for chaos engineering"""
    
    def __init__(self):
        self.app = FastAPI(
            title="AlphaPulse Chaos Engineering",
            description="Automated failure injection and resilience validation",
            version="1.0.0"
        )
        
        self.chaos_runner = None
        self.setup_routes()
    
    def setup_routes(self):
        """Setup API routes"""
        
        @self.app.get("/", response_class=HTMLResponse)
        async def get_chaos_dashboard():
            """Get the main chaos engineering dashboard HTML"""
            return self._get_chaos_dashboard_html()
        
        @self.app.get("/api/health")
        async def health_check():
            """Health check endpoint"""
            return {"status": "healthy", "timestamp": datetime.now(timezone.utc).isoformat()}
        
        @self.app.get("/api/status")
        async def get_status():
            """Get chaos engineering status"""
            try:
                return get_chaos_status()
            except Exception as e:
                logger.error(f"❌ Error getting chaos status: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/experiments")
        async def get_experiments():
            """Get all experiments"""
            try:
                runner = get_chaos_runner()
                return {
                    "predefined": runner.get_predefined_experiments(),
                    "active": runner.get_all_experiments_status(),
                    "history": runner.get_experiment_history(20)
                }
            except Exception as e:
                logger.error(f"❌ Error getting experiments: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/api/experiments/{experiment_name}/run")
        async def run_experiment(experiment_name: str, background_tasks: BackgroundTasks):
            """Run a chaos experiment"""
            try:
                # Run experiment in background
                experiment_id = await run_chaos_experiment(experiment_name)
                
                return {
                    "message": f"Chaos experiment '{experiment_name}' started",
                    "experiment_id": experiment_id,
                    "status": "running"
                }
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))
            except Exception as e:
                logger.error(f"❌ Error running experiment: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/experiments/{experiment_id}")
        async def get_experiment_status(experiment_id: str):
            """Get status of a specific experiment"""
            try:
                runner = get_chaos_runner()
                status = runner.get_experiment_status(experiment_id)
                
                if status is None:
                    raise HTTPException(status_code=404, detail="Experiment not found")
                
                return status
            except Exception as e:
                logger.error(f"❌ Error getting experiment status: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/experiments/{experiment_id}/result")
        async def get_experiment_result(experiment_id: str):
            """Get result of a completed experiment"""
            try:
                runner = get_chaos_runner()
                status = runner.get_experiment_status(experiment_id)
                
                if status is None:
                    raise HTTPException(status_code=404, detail="Experiment not found")
                
                if status["state"] != "completed":
                    raise HTTPException(status_code=400, detail="Experiment not completed")
                
                return status["result"]
            except Exception as e:
                logger.error(f"❌ Error getting experiment result: {e}")
                raise HTTPException(status_code=500, detail=str(e))
    
    def _get_chaos_dashboard_html(self) -> str:
        """Generate the chaos engineering dashboard HTML"""
        return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AlphaPulse Chaos Engineering</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
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
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
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
        
        .experiment-item {
            padding: 15px;
            margin: 10px 0;
            border-radius: 10px;
            border-left: 5px solid #e74c3c;
            background: #fdf2f2;
        }
        
        .experiment-name {
            font-weight: 600;
            color: #c0392b;
            margin-bottom: 5px;
        }
        
        .experiment-description {
            color: #7f8c8d;
            font-size: 0.9rem;
            margin-bottom: 10px;
        }
        
        .experiment-params {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
            font-size: 0.8rem;
        }
        
        .param-item {
            display: flex;
            justify-content: space-between;
        }
        
        .run-btn {
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
        
        .run-btn:hover {
            background: #c0392b;
        }
        
        .run-btn:disabled {
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
        
        .status-running {
            border-left-color: #f39c12;
            background: #fef9e7;
        }
        
        .status-completed {
            border-left-color: #27ae60;
            background: #f0f9f0;
        }
        
        .status-failed {
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
            <h1>🧪 AlphaPulse Chaos Engineering</h1>
            <p>Automated failure injection and resilience validation for your trading bot</p>
        </div>
        
        <div style="text-align: center; margin-bottom: 20px;">
            <button class="refresh-btn" onclick="refreshDashboard()">🔄 Refresh Dashboard</button>
            <span style="margin-left: 20px; color: white; opacity: 0.8;" id="lastUpdate"></span>
        </div>
        
        <div id="dashboardContent">
            <div class="loading">Loading chaos engineering dashboard...</div>
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
            
            // Predefined Experiments Section
            html += `
                <div class="status-section">
                    <h3 style="margin-bottom: 20px; color: #2c3e50;">🧪 Predefined Chaos Experiments</h3>
                    <div class="dashboard-grid">
            `;
            
            const predefined = dashboardData.predefined_experiments || [];
            predefined.forEach(experiment => {
                html += `
                    <div class="card">
                        <div class="experiment-item">
                            <div class="experiment-name">${experiment.name}</div>
                            <div class="experiment-description">${experiment.description}</div>
                            <div class="experiment-params">
                                <div class="param-item">
                                    <span>Type:</span>
                                    <span>${experiment.chaos_type}</span>
                                </div>
                                <div class="param-item">
                                    <span>Duration:</span>
                                    <span>${experiment.duration}s</span>
                                </div>
                                <div class="param-item">
                                    <span>Intensity:</span>
                                    <span>${(experiment.intensity * 100).toFixed(0)}%</span>
                                </div>
                                <div class="param-item">
                                    <span>Targets:</span>
                                    <span>${experiment.target_services.join(', ')}</span>
                                </div>
                            </div>
                            <button class="run-btn" onclick="runExperiment('${experiment.name}')">
                                🚀 Run Experiment
                            </button>
                        </div>
                    </div>
                `;
            });
            
            html += '</div></div>';
            
            // Active Experiments Section
            const active = dashboardData.active_experiments || [];
            if (active.length > 0) {
                html += `
                    <div class="status-section">
                        <h3 style="margin-bottom: 20px; color: #2c3e50;">🔄 Active Experiments</h3>
                `;
                
                active.forEach(exp => {
                    const statusClass = `status-${exp.state}`;
                    html += `
                        <div class="status-item ${statusClass}">
                            <strong>${exp.name}</strong><br>
                            <small>ID: ${exp.exp_id} | State: ${exp.state} | Started: ${exp.start_time}</small>
                        </div>
                    `;
                });
                
                html += '</div>';
            }
            
            // Recent History Section
            const history = dashboardData.recent_history || [];
            if (history.length > 0) {
                html += `
                    <div class="status-section">
                        <h3 style="margin-bottom: 20px; color: #2c3e50;">📊 Recent Experiment History</h3>
                `;
                
                history.forEach(result => {
                    const statusClass = `status-${result.state}`;
                    const score = result.resilience_score || 0;
                    const scoreColor = score > 0.7 ? '#27ae60' : score > 0.4 ? '#f39c12' : '#e74c3c';
                    
                    html += `
                        <div class="status-item ${statusClass}">
                            <strong>${result.experiment_name}</strong><br>
                            <small>
                                State: ${result.state} | 
                                Success: ${result.success ? '✅' : '❌'} | 
                                Resilience Score: <span style="color: ${scoreColor}">${(score * 100).toFixed(1)}%</span> | 
                                Completed: ${result.end_time || 'N/A'}
                            </small>
                        </div>
                    `;
                });
                
                html += '</div>';
            }
            
            content.innerHTML = html;
        }
        
        async function runExperiment(experimentName) {
            try {
                const button = event.target;
                button.disabled = true;
                button.textContent = '🔄 Running...';
                
                const response = await fetch(`/api/experiments/${experimentName}/run`, {
                    method: 'POST'
                });
                
                if (!response.ok) {
                    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                }
                
                const result = await response.json();
                
                button.textContent = '✅ Started';
                button.style.background = '#27ae60';
                
                // Refresh dashboard after a short delay
                setTimeout(() => {
                    refreshDashboard();
                }, 2000);
                
            } catch (error) {
                console.error('Error running experiment:', error);
                
                const button = event.target;
                button.disabled = false;
                button.textContent = '❌ Failed';
                button.style.background = '#e74c3c';
                
                // Reset button after delay
                setTimeout(() => {
                    button.textContent = '🚀 Run Experiment';
                    button.style.background = '#e74c3c';
                    button.disabled = false;
                }, 3000);
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
_chaos_service = None

def get_chaos_service() -> ChaosEngineeringService:
    """Get the global chaos engineering service instance"""
    global _chaos_service
    if _chaos_service is None:
        _chaos_service = ChaosEngineeringService()
    return _chaos_service

def get_chaos_app() -> FastAPI:
    """Get the FastAPI app for chaos engineering"""
    service = get_chaos_service()
    return service.app
