"""
Advanced Analytics Dashboard Service
Provides web interface and API for advanced analytics and ML features
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import sys
import os

# Add the backend directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from app.core.advanced_analytics import (
    AdvancedAnalytics, AnalysisConfig, AnalysisType, get_analytics
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AnalyticsDashboardService:
    def __init__(self):
        self.app = FastAPI(
            title="AlphaPulse Advanced Analytics Dashboard",
            description="Advanced analytics and machine learning dashboard for AlphaPulse",
            version="1.0.0"
        )
        self.analytics = get_analytics()
        self.setup_routes()
        
    def setup_routes(self):
        """Setup API routes"""
        
        @self.app.get("/", response_class=HTMLResponse)
        async def get_dashboard():
            """Serve the analytics dashboard"""
            return self._get_dashboard_html()
        
        @self.app.get("/api/health")
        async def health_check():
            """Health check endpoint"""
            return {
                "status": "healthy",
                "service": "Advanced Analytics Dashboard",
                "timestamp": datetime.now().isoformat()
            }
        
        @self.app.get("/api/analytics/status")
        async def get_analytics_status():
            """Get analytics service status"""
            try:
                history = await self.analytics.get_analysis_history(limit=10)
                return {
                    "status": "operational",
                    "total_analyses": len(history),
                    "recent_analyses": len(history),
                    "last_analysis": history[-1].timestamp.isoformat() if history else None
                }
            except Exception as e:
                logger.error(f"Error getting analytics status: {e}")
                return {
                    "status": "error",
                    "error": str(e)
                }
        
        @self.app.post("/api/analytics/run")
        async def run_analysis(analysis_request: Dict[str, Any]):
            """Run a new analysis"""
            try:
                analysis_type = AnalysisType(analysis_request.get('analysis_type'))
                parameters = analysis_request.get('parameters', {})
                
                config = AnalysisConfig(
                    analysis_type=analysis_type,
                    parameters=parameters
                )
                
                data = analysis_request.get('data', {})
                
                # Run analysis in background
                result = await self.analytics.run_analysis(config, data)
                
                return {
                    "status": "success",
                    "analysis_id": result.analysis_id,
                    "analysis_type": result.analysis_type.value,
                    "timestamp": result.timestamp.isoformat(),
                    "confidence": result.confidence,
                    "metrics": result.metrics,
                    "insights": result.insights,
                    "recommendations": result.recommendations
                }
                
            except Exception as e:
                logger.error(f"Error running analysis: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/analytics/history")
        async def get_analysis_history(limit: int = 50):
            """Get analysis history"""
            try:
                history = await self.analytics.get_analysis_history(limit=limit)
                return {
                    "analyses": [
                        {
                            "analysis_id": result.analysis_id,
                            "analysis_type": result.analysis_type.value,
                            "timestamp": result.timestamp.isoformat(),
                            "confidence": result.confidence,
                            "metrics": result.metrics,
                            "insights": result.insights[:3],  # Limit insights for list view
                            "recommendations": result.recommendations[:3]  # Limit recommendations
                        }
                        for result in history
                    ]
                }
            except Exception as e:
                logger.error(f"Error getting analysis history: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/analytics/types")
        async def get_analysis_types():
            """Get available analysis types"""
            return {
                "analysis_types": [
                    {
                        "value": analysis_type.value,
                        "name": analysis_type.name,
                        "description": self._get_analysis_description(analysis_type)
                    }
                    for analysis_type in AnalysisType
                ]
            }
        
        @self.app.post("/api/analytics/pattern-analysis")
        async def run_pattern_analysis(patterns: List[Dict[str, Any]]):
            """Run pattern analysis with sample data"""
            try:
                config = AnalysisConfig(
                    analysis_type=AnalysisType.PATTERN_ANALYSIS
                )
                
                result = await self.analytics.run_analysis(config, patterns)
                
                return {
                    "status": "success",
                    "analysis_id": result.analysis_id,
                    "metrics": result.metrics,
                    "insights": result.insights,
                    "recommendations": result.recommendations,
                    "confidence": result.confidence
                }
                
            except Exception as e:
                logger.error(f"Error running pattern analysis: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/api/analytics/performance-optimization")
        async def run_performance_optimization(data: Dict[str, Any]):
            """Run performance optimization analysis"""
            try:
                config = AnalysisConfig(
                    analysis_type=AnalysisType.PERFORMANCE_OPTIMIZATION
                )
                
                result = await self.analytics.run_analysis(config, data)
                
                return {
                    "status": "success",
                    "analysis_id": result.analysis_id,
                    "metrics": result.metrics,
                    "insights": result.insights,
                    "recommendations": result.recommendations,
                    "confidence": result.confidence,
                    "optimizations": result.data.get('optimizations', {})
                }
                
            except Exception as e:
                logger.error(f"Error running performance optimization: {e}")
                raise HTTPException(status_code=500, detail=str(e))
    
    def _get_analysis_description(self, analysis_type: AnalysisType) -> str:
        """Get description for analysis type"""
        descriptions = {
            AnalysisType.PATTERN_ANALYSIS: "Analyze trading patterns for insights and performance metrics",
            AnalysisType.PREDICTIVE_MODELING: "Build and train machine learning models for predictions",
            AnalysisType.ANOMALY_DETECTION: "Detect anomalies in trading data and system behavior",
            AnalysisType.PERFORMANCE_OPTIMIZATION: "Optimize system performance and trading strategies"
        }
        return descriptions.get(analysis_type, "Advanced analytics analysis")
    
    def _get_dashboard_html(self) -> str:
        """Generate the analytics dashboard HTML"""
        return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AlphaPulse Advanced Analytics Dashboard</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            color: #333;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }
        
        .header {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 30px;
            margin-bottom: 30px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        }
        
        .header h1 {
            color: #2c3e50;
            font-size: 2.5em;
            margin-bottom: 10px;
            text-align: center;
        }
        
        .header p {
            color: #7f8c8d;
            text-align: center;
            font-size: 1.1em;
        }
        
        .dashboard-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 30px;
            margin-bottom: 30px;
        }
        
        .card {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 25px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        
        .card:hover {
            transform: translateY(-5px);
            box-shadow: 0 12px 40px rgba(0, 0, 0, 0.15);
        }
        
        .card h3 {
            color: #2c3e50;
            margin-bottom: 20px;
            font-size: 1.4em;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
        }
        
        .status-indicator {
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 8px;
        }
        
        .status-healthy { background-color: #27ae60; }
        .status-warning { background-color: #f39c12; }
        .status-error { background-color: #e74c3c; }
        
        .metric {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 10px 0;
            border-bottom: 1px solid #ecf0f1;
        }
        
        .metric:last-child {
            border-bottom: none;
        }
        
        .metric-label {
            font-weight: 600;
            color: #34495e;
        }
        
        .metric-value {
            font-weight: bold;
            color: #2c3e50;
        }
        
        .btn {
            background: linear-gradient(135deg, #3498db, #2980b9);
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 8px;
            cursor: pointer;
            font-size: 1em;
            font-weight: 600;
            transition: all 0.3s ease;
            margin: 5px;
        }
        
        .btn:hover {
            background: linear-gradient(135deg, #2980b9, #1f5f8b);
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(52, 152, 219, 0.3);
        }
        
        .btn-secondary {
            background: linear-gradient(135deg, #95a5a6, #7f8c8d);
        }
        
        .btn-secondary:hover {
            background: linear-gradient(135deg, #7f8c8d, #6c7b7d);
        }
        
        .analysis-form {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 25px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
            margin-bottom: 30px;
        }
        
        .form-group {
            margin-bottom: 20px;
        }
        
        .form-group label {
            display: block;
            margin-bottom: 8px;
            font-weight: 600;
            color: #2c3e50;
        }
        
        .form-group select,
        .form-group input,
        .form-group textarea {
            width: 100%;
            padding: 12px;
            border: 2px solid #ecf0f1;
            border-radius: 8px;
            font-size: 1em;
            transition: border-color 0.3s ease;
        }
        
        .form-group select:focus,
        .form-group input:focus,
        .form-group textarea:focus {
            outline: none;
            border-color: #3498db;
        }
        
        .results-section {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 25px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        }
        
        .insight-item {
            background: #f8f9fa;
            border-left: 4px solid #3498db;
            padding: 15px;
            margin: 10px 0;
            border-radius: 0 8px 8px 0;
        }
        
        .recommendation-item {
            background: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 15px;
            margin: 10px 0;
            border-radius: 0 8px 8px 0;
        }
        
        .loading {
            text-align: center;
            padding: 20px;
            color: #7f8c8d;
        }
        
        .error {
            background: #f8d7da;
            border-left: 4px solid #dc3545;
            padding: 15px;
            margin: 10px 0;
            border-radius: 0 8px 8px 0;
            color: #721c24;
        }
        
        @media (max-width: 768px) {
            .dashboard-grid {
                grid-template-columns: 1fr;
            }
            
            .header h1 {
                font-size: 2em;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 AlphaPulse Advanced Analytics</h1>
            <p>Advanced analytics and machine learning dashboard for pattern analysis and optimization</p>
        </div>
        
        <div class="dashboard-grid">
            <div class="card">
                <h3>📊 Analytics Status</h3>
                <div id="analytics-status">
                    <div class="loading">Loading status...</div>
                </div>
            </div>
            
            <div class="card">
                <h3>📈 Recent Analyses</h3>
                <div id="recent-analyses">
                    <div class="loading">Loading analyses...</div>
                </div>
            </div>
            
            <div class="card">
                <h3>🎯 Quick Actions</h3>
                <button class="btn" onclick="runPatternAnalysis()">Run Pattern Analysis</button>
                <button class="btn" onclick="runPerformanceOptimization()">Performance Optimization</button>
                <button class="btn btn-secondary" onclick="loadAnalysisHistory()">View History</button>
            </div>
            
            <div class="card">
                <h3>⚙️ Analysis Types</h3>
                <div id="analysis-types">
                    <div class="loading">Loading types...</div>
                </div>
            </div>
        </div>
        
        <div class="analysis-form">
            <h3>🔬 Run Custom Analysis</h3>
            <div class="form-group">
                <label for="analysis-type">Analysis Type:</label>
                <select id="analysis-type">
                    <option value="">Select analysis type...</option>
                </select>
            </div>
            <div class="form-group">
                <label for="analysis-data">Data (JSON):</label>
                <textarea id="analysis-data" rows="6" placeholder='{"patterns": [], "metrics": {}}'></textarea>
            </div>
            <button class="btn" onclick="runCustomAnalysis()">Run Analysis</button>
        </div>
        
        <div class="results-section" id="results-section" style="display: none;">
            <h3>📋 Analysis Results</h3>
            <div id="analysis-results"></div>
        </div>
    </div>
    
    <script>
        // Global variables
        let analysisTypes = [];
        
        // Initialize dashboard
        document.addEventListener('DOMContentLoaded', function() {
            loadDashboard();
        });
        
        async function loadDashboard() {
            await Promise.all([
                loadAnalyticsStatus(),
                loadRecentAnalyses(),
                loadAnalysisTypes()
            ]);
        }
        
        async function loadAnalyticsStatus() {
            try {
                const response = await fetch('/api/analytics/status');
                const data = await response.json();
                
                const statusHtml = `
                    <div class="metric">
                        <span class="metric-label">Status:</span>
                        <span class="metric-value">
                            <span class="status-indicator status-${data.status === 'operational' ? 'healthy' : 'error'}"></span>
                            ${data.status}
                        </span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Total Analyses:</span>
                        <span class="metric-value">${data.total_analyses || 0}</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Recent Analyses:</span>
                        <span class="metric-value">${data.recent_analyses || 0}</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Last Analysis:</span>
                        <span class="metric-value">${data.last_analysis ? new Date(data.last_analysis).toLocaleString() : 'None'}</span>
                    </div>
                `;
                
                document.getElementById('analytics-status').innerHTML = statusHtml;
            } catch (error) {
                document.getElementById('analytics-status').innerHTML = `
                    <div class="error">Error loading status: ${error.message}</div>
                `;
            }
        }
        
        async function loadRecentAnalyses() {
            try {
                const response = await fetch('/api/analytics/history?limit=5');
                const data = await response.json();
                
                if (data.analyses.length === 0) {
                    document.getElementById('recent-analyses').innerHTML = '<p>No analyses run yet.</p>';
                    return;
                }
                
                const analysesHtml = data.analyses.map(analysis => `
                    <div class="metric">
                        <span class="metric-label">${analysis.analysis_type}:</span>
                        <span class="metric-value">${(analysis.confidence * 100).toFixed(1)}%</span>
                    </div>
                `).join('');
                
                document.getElementById('recent-analyses').innerHTML = analysesHtml;
            } catch (error) {
                document.getElementById('recent-analyses').innerHTML = `
                    <div class="error">Error loading analyses: ${error.message}</div>
                `;
            }
        }
        
        async function loadAnalysisTypes() {
            try {
                const response = await fetch('/api/analytics/types');
                const data = await response.json();
                analysisTypes = data.analysis_types;
                
                const typesHtml = data.analysis_types.map(type => `
                    <div class="metric">
                        <span class="metric-label">${type.name}:</span>
                        <span class="metric-value">${type.description}</span>
                    </div>
                `).join('');
                
                document.getElementById('analysis-types').innerHTML = typesHtml;
                
                // Populate select dropdown
                const select = document.getElementById('analysis-type');
                select.innerHTML = '<option value="">Select analysis type...</option>';
                data.analysis_types.forEach(type => {
                    const option = document.createElement('option');
                    option.value = type.value;
                    option.textContent = type.name;
                    select.appendChild(option);
                });
            } catch (error) {
                document.getElementById('analysis-types').innerHTML = `
                    <div class="error">Error loading types: ${error.message}</div>
                `;
            }
        }
        
        async function runPatternAnalysis() {
            const sampleData = [
                {
                    "type": "breakout",
                    "success": true,
                    "profit": 150.50,
                    "timestamp": new Date().toISOString()
                },
                {
                    "type": "breakout",
                    "success": false,
                    "profit": -50.25,
                    "timestamp": new Date().toISOString()
                },
                {
                    "type": "reversal",
                    "success": true,
                    "profit": 200.00,
                    "timestamp": new Date().toISOString()
                }
            ];
            
            await runAnalysis('pattern_analysis', sampleData);
        }
        
        async function runPerformanceOptimization() {
            const sampleData = {
                "patterns": [
                    {"type": "breakout", "size": 1200},
                    {"type": "reversal", "size": 800}
                ],
                "performance_metrics": {
                    "query_time": 0.15,
                    "storage_usage": 1024
                }
            };
            
            await runAnalysis('performance_optimization', sampleData);
        }
        
        async function runCustomAnalysis() {
            const analysisType = document.getElementById('analysis-type').value;
            const dataText = document.getElementById('analysis-data').value;
            
            if (!analysisType) {
                alert('Please select an analysis type');
                return;
            }
            
            let data;
            try {
                data = dataText ? JSON.parse(dataText) : {};
            } catch (error) {
                alert('Invalid JSON data');
                return;
            }
            
            await runAnalysis(analysisType, data);
        }
        
        async function runAnalysis(analysisType, data) {
            const resultsSection = document.getElementById('results-section');
            const resultsDiv = document.getElementById('analysis-results');
            
            resultsSection.style.display = 'block';
            resultsDiv.innerHTML = '<div class="loading">Running analysis...</div>';
            
            try {
                const response = await fetch('/api/analytics/run', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        analysis_type: analysisType,
                        data: data
                    })
                });
                
                const result = await response.json();
                
                if (result.status === 'success') {
                    const resultsHtml = `
                        <div class="metric">
                            <span class="metric-label">Analysis ID:</span>
                            <span class="metric-value">${result.analysis_id}</span>
                        </div>
                        <div class="metric">
                            <span class="metric-label">Confidence:</span>
                            <span class="metric-value">${(result.confidence * 100).toFixed(1)}%</span>
                        </div>
                        <div class="metric">
                            <span class="metric-label">Timestamp:</span>
                            <span class="metric-value">${new Date(result.timestamp).toLocaleString()}</span>
                        </div>
                        
                        <h4>📊 Metrics:</h4>
                        ${Object.entries(result.metrics).map(([key, value]) => `
                            <div class="metric">
                                <span class="metric-label">${key}:</span>
                                <span class="metric-value">${typeof value === 'number' ? value.toFixed(2) : value}</span>
                            </div>
                        `).join('')}
                        
                        <h4>💡 Insights:</h4>
                        ${result.insights.map(insight => `
                            <div class="insight-item">${insight}</div>
                        `).join('')}
                        
                        <h4>🎯 Recommendations:</h4>
                        ${result.recommendations.map(rec => `
                            <div class="recommendation-item">${rec}</div>
                        `).join('')}
                    `;
                    
                    resultsDiv.innerHTML = resultsHtml;
                } else {
                    resultsDiv.innerHTML = `<div class="error">Analysis failed: ${result.error || 'Unknown error'}</div>`;
                }
            } catch (error) {
                resultsDiv.innerHTML = `<div class="error">Error running analysis: ${error.message}</div>`;
            }
        }
        
        async function loadAnalysisHistory() {
            const resultsSection = document.getElementById('results-section');
            const resultsDiv = document.getElementById('analysis-results');
            
            resultsSection.style.display = 'block';
            resultsDiv.innerHTML = '<div class="loading">Loading history...</div>';
            
            try {
                const response = await fetch('/api/analytics/history?limit=20');
                const data = await response.json();
                
                if (data.analyses.length === 0) {
                    resultsDiv.innerHTML = '<p>No analysis history available.</p>';
                    return;
                }
                
                const historyHtml = `
                    <h4>📋 Analysis History:</h4>
                    ${data.analyses.map(analysis => `
                        <div class="insight-item">
                            <strong>${analysis.analysis_type}</strong> - ${new Date(analysis.timestamp).toLocaleString()}
                            <br>
                            <small>Confidence: ${(analysis.confidence * 100).toFixed(1)}%</small>
                            <br>
                            <small>Insights: ${analysis.insights.join(', ')}</small>
                        </div>
                    `).join('')}
                `;
                
                resultsDiv.innerHTML = historyHtml;
            } catch (error) {
                resultsDiv.innerHTML = `<div class="error">Error loading history: ${error.message}</div>`;
            }
        }
    </script>
</body>
</html>
        """

# Create service instance
analytics_dashboard_service = AnalyticsDashboardService()
app = analytics_dashboard_service.app
