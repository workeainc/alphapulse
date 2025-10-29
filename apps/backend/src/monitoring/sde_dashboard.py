"""
SDE Real-Time Monitoring Dashboard
Provides real-time monitoring and analytics for the SDE framework
"""

import asyncio
import logging
import json
import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import asyncpg
import aiohttp
from aiohttp import web
import websockets
import uuid

logger = logging.getLogger(__name__)

@dataclass
class SystemHealth:
    overall_health: float
    database_health: float
    model_health: float
    data_health: float
    api_health: float
    last_update: datetime
    status: str  # 'healthy', 'warning', 'critical'

@dataclass
class SignalMetrics:
    total_signals: int
    active_signals: int
    signals_today: int
    avg_confidence: float
    avg_confluence: float
    win_rate: float
    profit_factor: float
    last_signal_time: Optional[datetime]

@dataclass
class ModelPerformance:
    model_name: str
    win_rate: float
    avg_confidence: float
    total_signals: int
    recent_signals: int
    calibration_score: float
    drift_score: float
    status: str  # 'active', 'degraded', 'failed'

@dataclass
class RealTimeSignal:
    signal_id: str
    symbol: str
    timeframe: str
    direction: str
    confidence: float
    confluence_score: float
    entry_price: float
    stop_loss: float
    tp1_price: float
    tp2_price: float
    tp3_price: float
    tp4_price: float
    timestamp: datetime
    status: str  # 'active', 'tp1_hit', 'tp2_hit', 'tp3_hit', 'tp4_hit', 'stopped'

class SDEDashboard:
    """Real-time monitoring dashboard for SDE framework"""
    
    def __init__(self, db_pool: asyncpg.Pool):
        self.db_pool = db_pool
        self.websocket_clients = set()
        self.dashboard_data = {}
        self.update_interval = 5  # seconds
        
    async def start_dashboard(self, host: str = 'localhost', port: int = 8080):
        """Start the dashboard web server"""
        app = web.Application()
        
        # Add routes
        app.router.add_get('/', self.dashboard_handler)
        app.router.add_get('/api/health', self.health_api_handler)
        app.router.add_get('/api/signals', self.signals_api_handler)
        app.router.add_get('/api/performance', self.performance_api_handler)
        app.router.add_get('/api/metrics', self.metrics_api_handler)
        app.router.add_get('/ws', self.websocket_handler)
        
        # Start background tasks
        asyncio.create_task(self.update_dashboard_data())
        asyncio.create_task(self.broadcast_updates())
        
        # Start server
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, host, port)
        await site.start()
        
        logger.info(f"🚀 SDE Dashboard started on http://{host}:{port}")
        
    async def dashboard_handler(self, request):
        """Serve the main dashboard HTML"""
        html_content = self._get_dashboard_html()
        return web.Response(text=html_content, content_type='text/html')
    
    async def health_api_handler(self, request):
        """API endpoint for system health"""
        health_data = await self.get_system_health()
        return web.json_response(asdict(health_data))
    
    async def signals_api_handler(self, request):
        """API endpoint for signal metrics"""
        signals_data = await self.get_signal_metrics()
        return web.json_response(asdict(signals_data))
    
    async def performance_api_handler(self, request):
        """API endpoint for model performance"""
        performance_data = await self.get_model_performance()
        return web.json_response([asdict(model) for model in performance_data])
    
    async def metrics_api_handler(self, request):
        """API endpoint for all metrics"""
        all_metrics = {
            'health': asdict(await self.get_system_health()),
            'signals': asdict(await self.get_signal_metrics()),
            'performance': [asdict(model) for model in await self.get_model_performance()],
            'recent_signals': await self.get_recent_signals(),
            'timestamp': datetime.now().isoformat()
        }
        return web.json_response(all_metrics)
    
    async def websocket_handler(self, request):
        """WebSocket handler for real-time updates"""
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        
        self.websocket_clients.add(ws)
        
        try:
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    # Handle client messages
                    data = json.loads(msg.data)
                    await self.handle_websocket_message(ws, data)
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    logger.error(f"WebSocket error: {ws.exception()}")
        finally:
            self.websocket_clients.discard(ws)
        
        return ws
    
    async def handle_websocket_message(self, ws, data):
        """Handle WebSocket messages from clients"""
        message_type = data.get('type')
        
        if message_type == 'subscribe':
            # Client wants to subscribe to specific updates
            channels = data.get('channels', [])
            await ws.send_json({
                'type': 'subscribed',
                'channels': channels,
                'timestamp': datetime.now().isoformat()
            })
        elif message_type == 'ping':
            # Respond to ping
            await ws.send_json({
                'type': 'pong',
                'timestamp': datetime.now().isoformat()
            })
    
    async def update_dashboard_data(self):
        """Background task to update dashboard data"""
        while True:
            try:
                # Update all dashboard data
                self.dashboard_data = {
                    'health': await self.get_system_health(),
                    'signals': await self.get_signal_metrics(),
                    'performance': await self.get_model_performance(),
                    'recent_signals': await self.get_recent_signals(),
                    'timestamp': datetime.now().isoformat()
                }
                
                await asyncio.sleep(self.update_interval)
                
            except Exception as e:
                logger.error(f"❌ Dashboard data update failed: {e}")
                await asyncio.sleep(self.update_interval)
    
    async def broadcast_updates(self):
        """Background task to broadcast updates to WebSocket clients"""
        while True:
            try:
                if self.websocket_clients:
                    # Prepare update message
                    update_message = {
                        'type': 'update',
                        'data': self.dashboard_data,
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    # Broadcast to all connected clients
                    disconnected_clients = set()
                    for ws in self.websocket_clients:
                        try:
                            await ws.send_json(update_message)
                        except Exception as e:
                            logger.error(f"❌ Failed to send to WebSocket client: {e}")
                            disconnected_clients.add(ws)
                    
                    # Remove disconnected clients
                    self.websocket_clients -= disconnected_clients
                
                await asyncio.sleep(self.update_interval)
                
            except Exception as e:
                logger.error(f"❌ Broadcast update failed: {e}")
                await asyncio.sleep(self.update_interval)
    
    async def get_system_health(self) -> SystemHealth:
        """Get overall system health metrics"""
        try:
            # Check database health
            db_health = await self._check_database_health()
            
            # Check model health
            model_health = await self._check_model_health()
            
            # Check data health
            data_health = await self._check_data_health()
            
            # Check API health
            api_health = await self._check_api_health()
            
            # Calculate overall health
            overall_health = (db_health + model_health + data_health + api_health) / 4
            
            # Determine status
            if overall_health >= 0.8:
                status = 'healthy'
            elif overall_health >= 0.6:
                status = 'warning'
            else:
                status = 'critical'
            
            return SystemHealth(
                overall_health=overall_health,
                database_health=db_health,
                model_health=model_health,
                data_health=data_health,
                api_health=api_health,
                last_update=datetime.now(),
                status=status
            )
            
        except Exception as e:
            logger.error(f"❌ Failed to get system health: {e}")
            return SystemHealth(
                overall_health=0.0,
                database_health=0.0,
                model_health=0.0,
                data_health=0.0,
                api_health=0.0,
                last_update=datetime.now(),
                status='critical'
            )
    
    async def get_signal_metrics(self) -> SignalMetrics:
        """Get signal performance metrics"""
        try:
            async with self.db_pool.acquire() as conn:
                # Get total signals
                total_signals = await conn.fetchval("""
                    SELECT COUNT(*) FROM sde_signal_history
                """)
                
                # Get active signals
                active_signals = await conn.fetchval("""
                    SELECT COUNT(*) FROM sde_signal_history 
                    WHERE status = 'active'
                """)
                
                # Get signals today
                signals_today = await conn.fetchval("""
                    SELECT COUNT(*) FROM sde_signal_history 
                    WHERE DATE(signal_timestamp) = CURRENT_DATE
                """)
                
                # Get average confidence
                avg_confidence = await conn.fetchval("""
                    SELECT AVG(confidence) FROM sde_signal_history 
                    WHERE signal_timestamp >= NOW() - INTERVAL '7 days'
                """) or 0.0
                
                # Get average confluence
                avg_confluence = await conn.fetchval("""
                    SELECT AVG(confluence_score) FROM sde_confluence_scores 
                    WHERE timestamp >= NOW() - INTERVAL '7 days'
                """) or 0.0
                
                # Get win rate
                win_rate = await conn.fetchval("""
                    SELECT 
                        CASE 
                            WHEN COUNT(*) > 0 THEN 
                                COUNT(CASE WHEN actual_outcome = 1 THEN 1 END)::float / COUNT(*)
                            ELSE 0.0 
                        END
                    FROM sde_signal_history 
                    WHERE actual_outcome IS NOT NULL
                    AND signal_timestamp >= NOW() - INTERVAL '30 days'
                """) or 0.0
                
                # Get profit factor
                profit_factor = await conn.fetchval("""
                    SELECT 
                        CASE 
                            WHEN SUM(CASE WHEN actual_outcome = 0 THEN 1 ELSE 0 END) > 0 THEN
                                SUM(CASE WHEN actual_outcome = 1 THEN 1 ELSE 0 END)::float / 
                                SUM(CASE WHEN actual_outcome = 0 THEN 1 ELSE 0 END)
                            ELSE 0.0 
                        END
                    FROM sde_signal_history 
                    WHERE actual_outcome IS NOT NULL
                    AND signal_timestamp >= NOW() - INTERVAL '30 days'
                """) or 0.0
                
                # Get last signal time
                last_signal_time = await conn.fetchval("""
                    SELECT MAX(signal_timestamp) FROM sde_signal_history
                """)
                
                return SignalMetrics(
                    total_signals=total_signals or 0,
                    active_signals=active_signals or 0,
                    signals_today=signals_today or 0,
                    avg_confidence=avg_confidence,
                    avg_confluence=avg_confluence,
                    win_rate=win_rate,
                    profit_factor=profit_factor,
                    last_signal_time=last_signal_time
                )
                
        except Exception as e:
            logger.error(f"❌ Failed to get signal metrics: {e}")
            return SignalMetrics(
                total_signals=0,
                active_signals=0,
                signals_today=0,
                avg_confidence=0.0,
                avg_confluence=0.0,
                win_rate=0.0,
                profit_factor=0.0,
                last_signal_time=None
            )
    
    async def get_model_performance(self) -> List[ModelPerformance]:
        """Get performance metrics for all models"""
        try:
            async with self.db_pool.acquire() as conn:
                # Get performance data for each model
                performance_data = await conn.fetch("""
                    SELECT 
                        model_name,
                        AVG(win_rate) as avg_win_rate,
                        AVG(avg_confidence) as avg_confidence,
                        SUM(total_signals) as total_signals,
                        COUNT(*) as recent_periods,
                        AVG(calibration_score) as avg_calibration,
                        AVG(drift_score) as avg_drift
                    FROM sde_model_performance
                    WHERE period_end >= NOW() - INTERVAL '7 days'
                    GROUP BY model_name
                """)
                
                models = []
                for row in performance_data:
                    # Determine model status
                    if row['avg_win_rate'] >= 0.6 and row['avg_calibration'] >= 0.8:
                        status = 'active'
                    elif row['avg_win_rate'] >= 0.5 and row['avg_calibration'] >= 0.6:
                        status = 'degraded'
                    else:
                        status = 'failed'
                    
                    models.append(ModelPerformance(
                        model_name=row['model_name'],
                        win_rate=row['avg_win_rate'] or 0.0,
                        avg_confidence=row['avg_confidence'] or 0.0,
                        total_signals=row['total_signals'] or 0,
                        recent_signals=row['recent_periods'] or 0,
                        calibration_score=row['avg_calibration'] or 0.0,
                        drift_score=row['avg_drift'] or 0.0,
                        status=status
                    ))
                
                return models
                
        except Exception as e:
            logger.error(f"❌ Failed to get model performance: {e}")
            return []
    
    async def get_recent_signals(self) -> List[Dict[str, Any]]:
        """Get recent signals for dashboard display"""
        try:
            async with self.db_pool.acquire() as conn:
                signals = await conn.fetch("""
                    SELECT 
                        signal_id, symbol, timeframe, direction, confidence,
                        confluence_score, entry_price, stop_loss,
                        tp1_price, tp2_price, tp3_price, tp4_price,
                        signal_timestamp, status
                    FROM sde_signal_history
                    WHERE signal_timestamp >= NOW() - INTERVAL '24 hours'
                    ORDER BY signal_timestamp DESC
                    LIMIT 20
                """)
                
                return [
                    {
                        'signal_id': row['signal_id'],
                        'symbol': row['symbol'],
                        'timeframe': row['timeframe'],
                        'direction': row['direction'],
                        'confidence': float(row['confidence']),
                        'confluence_score': float(row['confluence_score']) if row['confluence_score'] else 0.0,
                        'entry_price': float(row['entry_price']),
                        'stop_loss': float(row['stop_loss']),
                        'tp1_price': float(row['tp1_price']) if row['tp1_price'] else 0.0,
                        'tp2_price': float(row['tp2_price']) if row['tp2_price'] else 0.0,
                        'tp3_price': float(row['tp3_price']) if row['tp3_price'] else 0.0,
                        'tp4_price': float(row['tp4_price']) if row['tp4_price'] else 0.0,
                        'timestamp': row['signal_timestamp'].isoformat(),
                        'status': row['status']
                    }
                    for row in signals
                ]
                
        except Exception as e:
            logger.error(f"❌ Failed to get recent signals: {e}")
            return []
    
    async def _check_database_health(self) -> float:
        """Check database connectivity and performance"""
        try:
            async with self.db_pool.acquire() as conn:
                # Test basic query
                start_time = datetime.now()
                await conn.fetchval("SELECT 1")
                query_time = (datetime.now() - start_time).total_seconds()
                
                # Health score based on query time
                if query_time < 0.1:
                    return 1.0
                elif query_time < 0.5:
                    return 0.8
                elif query_time < 1.0:
                    return 0.6
                else:
                    return 0.3
                    
        except Exception as e:
            logger.error(f"❌ Database health check failed: {e}")
            return 0.0
    
    async def _check_model_health(self) -> float:
        """Check model performance and availability"""
        try:
            async with self.db_pool.acquire() as conn:
                # Check recent model performance
                recent_performance = await conn.fetch("""
                    SELECT AVG(win_rate) as avg_win_rate
                    FROM sde_model_performance
                    WHERE period_end >= NOW() - INTERVAL '1 day'
                """)
                
                if recent_performance and recent_performance[0]['avg_win_rate']:
                    win_rate = recent_performance[0]['avg_win_rate']
                    if win_rate >= 0.6:
                        return 1.0
                    elif win_rate >= 0.5:
                        return 0.8
                    elif win_rate >= 0.4:
                        return 0.6
                    else:
                        return 0.3
                else:
                    return 0.5  # No recent data
                    
        except Exception as e:
            logger.error(f"❌ Model health check failed: {e}")
            return 0.0
    
    async def _check_data_health(self) -> float:
        """Check data quality and freshness"""
        try:
            async with self.db_pool.acquire() as conn:
                # Check recent data availability
                recent_data = await conn.fetchval("""
                    SELECT COUNT(*) FROM market_data 
                    WHERE timestamp >= NOW() - INTERVAL '5 minutes'
                """)
                
                if recent_data and recent_data > 100:
                    return 1.0
                elif recent_data and recent_data > 50:
                    return 0.8
                elif recent_data and recent_data > 10:
                    return 0.6
                else:
                    return 0.3
                    
        except Exception as e:
            logger.error(f"❌ Data health check failed: {e}")
            return 0.5  # Assume moderate health if check fails
    
    async def _check_api_health(self) -> float:
        """Check API endpoints health"""
        try:
            # For now, assume API is healthy
            # In a real implementation, you would check actual API endpoints
            return 1.0
        except Exception as e:
            logger.error(f"❌ API health check failed: {e}")
            return 0.5
    
    def _get_dashboard_html(self) -> str:
        """Generate dashboard HTML"""
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>SDE Framework Dashboard</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
                .container { max-width: 1200px; margin: 0 auto; }
                .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px; margin-bottom: 20px; }
                .metrics-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-bottom: 20px; }
                .metric-card { background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
                .metric-title { font-size: 14px; color: #666; margin-bottom: 10px; }
                .metric-value { font-size: 24px; font-weight: bold; margin-bottom: 5px; }
                .metric-change { font-size: 12px; }
                .positive { color: #28a745; }
                .negative { color: #dc3545; }
                .neutral { color: #6c757d; }
                .status-healthy { color: #28a745; }
                .status-warning { color: #ffc107; }
                .status-critical { color: #dc3545; }
                .signals-table { background: white; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); overflow: hidden; }
                .signals-table table { width: 100%; border-collapse: collapse; }
                .signals-table th, .signals-table td { padding: 12px; text-align: left; border-bottom: 1px solid #eee; }
                .signals-table th { background-color: #f8f9fa; font-weight: 600; }
                .refresh-btn { background: #007bff; color: white; border: none; padding: 10px 20px; border-radius: 5px; cursor: pointer; }
                .refresh-btn:hover { background: #0056b3; }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🚀 SDE Framework Dashboard</h1>
                    <p>Real-time monitoring and analytics</p>
                </div>
                
                <div class="metrics-grid" id="metrics-grid">
                    <!-- Metrics will be populated by JavaScript -->
                </div>
                
                <div class="signals-table">
                    <h3>Recent Signals</h3>
                    <table id="signals-table">
                        <thead>
                            <tr>
                                <th>Symbol</th>
                                <th>Direction</th>
                                <th>Confidence</th>
                                <th>Entry</th>
                                <th>Stop</th>
                                <th>Status</th>
                                <th>Time</th>
                            </tr>
                        </thead>
                        <tbody id="signals-tbody">
                            <!-- Signals will be populated by JavaScript -->
                        </tbody>
                    </table>
                </div>
                
                <button class="refresh-btn" onclick="refreshData()">Refresh Data</button>
            </div>
            
            <script>
                let ws = null;
                
                function connectWebSocket() {
                    ws = new WebSocket('ws://localhost:8080/ws');
                    ws.onopen = function() {
                        console.log('WebSocket connected');
                        ws.send(JSON.stringify({type: 'subscribe', channels: ['all']}));
                    };
                    ws.onmessage = function(event) {
                        const data = JSON.parse(event.data);
                        if (data.type === 'update') {
                            updateDashboard(data.data);
                        }
                    };
                    ws.onclose = function() {
                        console.log('WebSocket disconnected');
                        setTimeout(connectWebSocket, 5000);
                    };
                }
                
                function updateDashboard(data) {
                    updateMetrics(data);
                    updateSignals(data.recent_signals);
                }
                
                function updateMetrics(data) {
                    const metricsGrid = document.getElementById('metrics-grid');
                    metricsGrid.innerHTML = `
                        <div class="metric-card">
                            <div class="metric-title">System Health</div>
                            <div class="metric-value status-${data.health.status}">${(data.health.overall_health * 100).toFixed(1)}%</div>
                            <div class="metric-change">Status: ${data.health.status}</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-title">Active Signals</div>
                            <div class="metric-value">${data.signals.active_signals}</div>
                            <div class="metric-change">Total: ${data.signals.total_signals}</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-title">Win Rate</div>
                            <div class="metric-value">${(data.signals.win_rate * 100).toFixed(1)}%</div>
                            <div class="metric-change">Profit Factor: ${data.signals.profit_factor.toFixed(2)}</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-title">Avg Confidence</div>
                            <div class="metric-value">${(data.signals.avg_confidence * 100).toFixed(1)}%</div>
                            <div class="metric-change">Confluence: ${(data.signals.avg_confluence * 10).toFixed(1)}/10</div>
                        </div>
                    `;
                }
                
                function updateSignals(signals) {
                    const tbody = document.getElementById('signals-tbody');
                    tbody.innerHTML = signals.map(signal => `
                        <tr>
                            <td>${signal.symbol}</td>
                            <td>${signal.direction}</td>
                            <td>${(signal.confidence * 100).toFixed(1)}%</td>
                            <td>${signal.entry_price.toFixed(4)}</td>
                            <td>${signal.stop_loss.toFixed(4)}</td>
                            <td>${signal.status}</td>
                            <td>${new Date(signal.timestamp).toLocaleTimeString()}</td>
                        </tr>
                    `).join('');
                }
                
                async function refreshData() {
                    try {
                        const response = await fetch('/api/metrics');
                        const data = await response.json();
                        updateDashboard(data);
                    } catch (error) {
                        console.error('Failed to refresh data:', error);
                    }
                }
                
                // Initialize
                connectWebSocket();
                refreshData();
                
                // Auto-refresh every 30 seconds
                setInterval(refreshData, 30000);
            </script>
        </body>
        </html>
        """
