#!/usr/bin/env python3
"""
Quick Start Example: FastAPI Dashboard with AlphaPulse Services
This shows how to quickly integrate your existing services with a web dashboard
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, Any
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import uvicorn

# Import your existing services
from app.services.pattern_storage_service import PatternStorageService
from app.services.advanced_retrieval_optimizer import AdvancedRetrievalOptimizer
from app.services.predictive_optimizer import PredictiveOptimizer
from app.services.trading_engine import TradingEngine
from app.services.risk_manager import RiskManager

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QuickStartDashboard:
    """Quick start dashboard integrating with existing AlphaPulse services"""
    
    def __init__(self):
        self.app = FastAPI(title="AlphaPulse Quick Start", version="1.0.0")
        self.connected_clients = []
        
        # Initialize your existing services
        self.pattern_storage = PatternStorageService()
        self.retrieval_optimizer = AdvancedRetrievalOptimizer()
        self.predictive_optimizer = PredictiveOptimizer({})
        self.trading_engine = TradingEngine()
        self.risk_manager = RiskManager()
        
        # Setup routes
        self._setup_routes()
        
    def _setup_routes(self):
        """Setup FastAPI routes"""
        
        @self.app.get("/")
        async def dashboard():
            """Main dashboard page"""
            html = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>AlphaPulse Dashboard</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 20px; }
                    .card { border: 1px solid #ddd; padding: 20px; margin: 10px; border-radius: 8px; }
                    .metric { display: flex; justify-content: space-between; margin: 10px 0; }
                    .status { padding: 5px 10px; border-radius: 4px; color: white; }
                    .healthy { background: #28a745; }
                    .warning { background: #ffc107; color: black; }
                    .error { background: #dc3545; }
                </style>
            </head>
            <body>
                <h1>🚀 AlphaPulse Quick Start Dashboard</h1>
                
                <div class="card">
                    <h2>📊 Pattern Storage Service</h2>
                    <div id="pattern-storage-status">Loading...</div>
                </div>
                
                <div class="card">
                    <h2>🔧 Retrieval Optimizer</h2>
                    <div id="retrieval-optimizer-status">Loading...</div>
                </div>
                
                <div class="card">
                    <h2>🤖 Predictive Optimizer</h2>
                    <div id="predictive-optimizer-status">Loading...</div>
                </div>
                
                <div class="card">
                    <h2>📈 Trading Engine</h2>
                    <div id="trading-engine-status">Loading...</div>
                </div>
                
                <div class="card">
                    <h2>⚠️ Risk Manager</h2>
                    <div id="risk-manager-status">Loading...</div>
                </div>
                
                <script>
                    // Simple auto-refresh every 10 seconds
                    setInterval(async () => {
                        try {
                            const response = await fetch('/api/status');
                            const data = await response.json();
                            
                            // Update pattern storage
                            document.getElementById('pattern-storage-status').innerHTML = 
                                `<div class="metric">
                                    <span>Status: <span class="status ${data.pattern_storage.status}">${data.pattern_storage.status}</span></span>
                                    <span>Patterns: ${data.pattern_storage.total_patterns.toLocaleString()}</span>
                                </div>
                                <div class="metric">
                                    <span>Throughput: ${data.pattern_storage.patterns_per_second}/sec</span>
                                    <span>Last Update: ${new Date(data.pattern_storage.timestamp).toLocaleTimeString()}</span>
                                </div>`;
                            
                            // Update other services similarly...
                            
                        } catch (error) {
                            console.error('Error updating dashboard:', error);
                        }
                    }, 10000);
                    
                    // Initial load
                    fetch('/api/status').then(r => r.json()).then(data => {
                        // Initial update
                    });
                </script>
            </body>
            </html>
            """
            return HTMLResponse(content=html)
        
        @self.app.get("/api/status")
        async def get_status():
            """Get status of all services"""
            try:
                return {
                    "pattern_storage": await self._get_pattern_storage_status(),
                    "retrieval_optimizer": await self._get_retrieval_optimizer_status(),
                    "predictive_optimizer": await self._get_predictive_optimizer_status(),
                    "trading_engine": await self._get_trading_engine_status(),
                    "risk_manager": await self._get_risk_manager_status(),
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            except Exception as e:
                logger.error(f"Error getting status: {e}")
                return {"error": str(e)}
        
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket for real-time updates"""
            await websocket.accept()
            self.connected_clients.append(websocket)
            
            try:
                while True:
                    # Send status updates every 5 seconds
                    status = await self._get_comprehensive_status()
                    await websocket.send_text(str(status))
                    await asyncio.sleep(5)
            except WebSocketDisconnect:
                self.connected_clients.remove(websocket)
    
    async def _get_pattern_storage_status(self) -> Dict[str, Any]:
        """Get pattern storage service status"""
        try:
            stats = self.pattern_storage.performance_stats
            return {
                "status": "healthy",
                "total_patterns": stats.get("total_patterns_stored", 0),
                "patterns_per_second": stats.get("last_batch_performance", {}).get("patterns_per_second", 0),
                "last_batch_time": stats.get("last_batch_performance", {}).get("insertion_time", 0),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting pattern storage status: {e}")
            return {
                "status": "error",
                "total_patterns": 0,
                "patterns_per_second": 0,
                "last_batch_time": 0,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
    
    async def _get_retrieval_optimizer_status(self) -> Dict[str, Any]:
        """Get retrieval optimizer status"""
        try:
            # Mock status for now - integrate with your service
            return {
                "status": "healthy",
                "queries_optimized": 150,
                "avg_improvement": 25.5,
                "last_optimization": datetime.now(timezone.utc).isoformat(),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting retrieval optimizer status: {e}")
            return {"status": "error", "timestamp": datetime.now(timezone.utc).isoformat()}
    
    async def _get_predictive_optimizer_status(self) -> Dict[str, Any]:
        """Get predictive optimizer status"""
        try:
            # Mock status for now - integrate with your service
            return {
                "status": "healthy",
                "predictions_made": 89,
                "accuracy": 87.2,
                "last_training": datetime.now(timezone.utc).isoformat(),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting predictive optimizer status: {e}")
            return {"status": "error", "timestamp": datetime.now(timezone.utc).isoformat()}
    
    async def _get_trading_engine_status(self) -> Dict[str, Any]:
        """Get trading engine status"""
        try:
            # Mock status for now - integrate with your service
            return {
                "status": "healthy",
                "active_positions": 3,
                "total_positions": 10,
                "daily_pnl": 1250.50,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting trading engine status: {e}")
            return {"status": "error", "timestamp": datetime.now(timezone.utc).isoformat()}
    
    async def _get_risk_manager_status(self) -> Dict[str, Any]:
        """Get risk manager status"""
        try:
            # Mock status for now - integrate with your service
            return {
                "status": "healthy",
                "risk_level": "medium",
                "max_drawdown": 8.5,
                "current_drawdown": 3.2,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting risk manager status: {e}")
            return {"status": "error", "timestamp": datetime.now(timezone.utc).isoformat()}
    
    async def _get_comprehensive_status(self) -> Dict[str, Any]:
        """Get comprehensive status of all services"""
        try:
            return {
                "pattern_storage": await self._get_pattern_storage_status(),
                "retrieval_optimizer": await self._get_retrieval_optimizer_status(),
                "predictive_optimizer": await self._get_predictive_optimizer_status(),
                "trading_engine": await self._get_trading_engine_status(),
                "risk_manager": await self._get_risk_manager_status(),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting comprehensive status: {e}")
            return {"error": str(e)}
    
    async def start(self, host: str = "0.0.0.0", port: int = 8000):
        """Start the dashboard server"""
        config = uvicorn.Config(
            app=self.app,
            host=host,
            port=port,
            log_level="info"
        )
        server = uvicorn.Server(config)
        await server.serve()

# Quick start function
async def main():
    """Quick start the dashboard"""
    try:
        logger.info("🚀 Starting AlphaPulse Quick Start Dashboard...")
        
        dashboard = QuickStartDashboard()
        await dashboard.start()
        
    except KeyboardInterrupt:
        logger.info("🛑 Dashboard stopped by user")
    except Exception as e:
        logger.error(f"❌ Failed to start dashboard: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
