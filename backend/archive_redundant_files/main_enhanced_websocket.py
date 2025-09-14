#!/usr/bin/env python3
"""
Enhanced AlphaPulse Main Application with WebSocket Integration
Real-time trading system with TimescaleDB integration and performance optimizations
Enhanced with real-time signal processing and notification system
"""

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Local imports
from .services.enhanced_websocket_service import EnhancedWebSocketService
from .signals.intelligent_signal_generator import IntelligentSignalGenerator
from ..core.websocket_enhanced import EnhancedBinanceWebSocketClient, EnhancedWebSocketManager
from ..database.connection import TimescaleDBConnection
from ..database.models import Signal
from ..utils.config import get_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global service instances
websocket_service: Optional[EnhancedWebSocketService] = None
binance_websocket_manager: Optional[EnhancedWebSocketManager] = None
db_connection: Optional[TimescaleDBConnection] = None
signal_generator: Optional[IntelligentSignalGenerator] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global websocket_service, binance_websocket_manager, db_connection, signal_generator
    
    # Startup
    logger.info("🚀 Starting Enhanced AlphaPulse Application...")
    
    try:
        # Initialize database connection
        db_connection = TimescaleDBConnection()
        await db_connection.initialize()
        logger.info("✅ Database connection initialized")
        
        # Initialize enhanced WebSocket service
        websocket_service = EnhancedWebSocketService()
        await websocket_service.initialize()
        logger.info("✅ Enhanced WebSocket service initialized")
        
        # Initialize Binance WebSocket manager
        binance_websocket_manager = EnhancedWebSocketManager(max_connections=3)
        await binance_websocket_manager.start()
        logger.info("✅ Binance WebSocket manager started")
        
        # Initialize intelligent signal generator
        signal_generator = IntelligentSignalGenerator(db_connection.pool, None)  # Exchange will be set later
        await signal_generator.initialize()
        logger.info("✅ Intelligent signal generator initialized")
        
        # Start background tasks
        asyncio.create_task(_background_data_processor())
        asyncio.create_task(_system_monitor())
        asyncio.create_task(_real_time_signal_processor())
        asyncio.create_task(_notification_manager())
        
        logger.info("✅ Enhanced AlphaPulse Application started successfully")
        
    except Exception as e:
        logger.error(f"❌ Failed to start application: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("🔄 Shutting down Enhanced AlphaPulse Application...")
    
    try:
        if signal_generator:
            await signal_generator.stop_signal_generation()
        
        if websocket_service:
            await websocket_service.shutdown()
        
        if binance_websocket_manager:
            await binance_websocket_manager.stop()
        
        if db_connection:
            await db_connection.close()
        
        logger.info("✅ Enhanced AlphaPulse Application shutdown complete")
        
    except Exception as e:
        logger.error(f"❌ Error during shutdown: {e}")

# Create FastAPI app
app = FastAPI(
    title="Enhanced AlphaPulse Trading System",
    description="Real-time trading system with enhanced WebSocket integration, TimescaleDB, and real-time signal processing",
    version="3.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.signal_subscribers: List[WebSocket] = []
        self.notification_subscribers: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"✅ New WebSocket connection. Total: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        if websocket in self.signal_subscribers:
            self.signal_subscribers.remove(websocket)
        if websocket in self.notification_subscribers:
            self.notification_subscribers.remove(websocket)
        logger.info(f"❌ WebSocket disconnected. Total: {len(self.active_connections)}")

    async def send_personal_message(self, message: str, websocket: WebSocket):
        try:
            await websocket.send_text(message)
        except Exception as e:
            logger.error(f"❌ Error sending personal message: {e}")
            self.disconnect(websocket)

    async def broadcast(self, message: str):
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception as e:
                logger.error(f"❌ Error broadcasting message: {e}")
                disconnected.append(connection)
        
        # Remove disconnected connections
        for connection in disconnected:
            self.disconnect(connection)

    async def broadcast_signal(self, signal_data: Dict[str, Any]):
        """Broadcast signal to signal subscribers"""
        message = {
            "type": "signal",
            "data": signal_data,
            "timestamp": datetime.now().isoformat()
        }
        
        disconnected = []
        for connection in self.signal_subscribers:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"❌ Error broadcasting signal: {e}")
                disconnected.append(connection)
        
        # Remove disconnected connections
        for connection in disconnected:
            self.disconnect(connection)

    async def broadcast_notification(self, notification_data: Dict[str, Any]):
        """Broadcast notification to notification subscribers"""
        message = {
            "type": "notification",
            "data": notification_data,
            "timestamp": datetime.now().isoformat()
        }
        
        disconnected = []
        for connection in self.notification_subscribers:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"❌ Error broadcasting notification: {e}")
                disconnected.append(connection)
        
        # Remove disconnected connections
        for connection in disconnected:
            self.disconnect(connection)

manager = ConnectionManager()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # Handle incoming messages
            await handle_websocket_message(websocket, data)
    except WebSocketDisconnect:
        manager.disconnect(websocket)

@app.websocket("/ws/signals")
async def signals_websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    manager.signal_subscribers.append(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # Handle signal-specific messages
            await handle_signal_message(websocket, data)
    except WebSocketDisconnect:
        manager.disconnect(websocket)

@app.websocket("/ws/notifications")
async def notifications_websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    manager.notification_subscribers.append(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # Handle notification-specific messages
            await handle_notification_message(websocket, data)
    except WebSocketDisconnect:
        manager.disconnect(websocket)

async def handle_websocket_message(websocket: WebSocket, data: str):
    """Handle incoming WebSocket messages"""
    try:
        # Parse message
        import json
        message = json.loads(data)
        
        if message.get("type") == "subscribe_signals":
            manager.signal_subscribers.append(websocket)
            await manager.send_personal_message(
                json.dumps({"type": "subscribed", "channel": "signals"}),
                websocket
            )
        
        elif message.get("type") == "subscribe_notifications":
            manager.notification_subscribers.append(websocket)
            await manager.send_personal_message(
                json.dumps({"type": "subscribed", "channel": "notifications"}),
                websocket
            )
        
        elif message.get("type") == "ping":
            await manager.send_personal_message(
                json.dumps({"type": "pong", "timestamp": datetime.now().isoformat()}),
                websocket
            )
        
    except Exception as e:
        logger.error(f"❌ Error handling WebSocket message: {e}")

async def handle_signal_message(websocket: WebSocket, data: str):
    """Handle signal-specific WebSocket messages"""
    try:
        import json
        message = json.loads(data)
        
        if message.get("type") == "request_signals":
            # Send latest signals
            await send_latest_signals(websocket)
        
    except Exception as e:
        logger.error(f"❌ Error handling signal message: {e}")

async def handle_notification_message(websocket: WebSocket, data: str):
    """Handle notification-specific WebSocket messages"""
    try:
        import json
        message = json.loads(data)
        
        if message.get("type") == "request_notifications":
            # Send latest notifications
            await send_latest_notifications(websocket)
        
    except Exception as e:
        logger.error(f"❌ Error handling notification message: {e}")

async def send_latest_signals(websocket: WebSocket):
    """Send latest signals to WebSocket client"""
    try:
        if not db_connection:
            return
        
        async with db_connection.pool.acquire() as conn:
            # Get latest active signals
            rows = await conn.fetch("""
                SELECT * FROM signals 
                WHERE is_active = true 
                ORDER BY ts DESC 
                LIMIT 10
            """)
            
            signals = []
            for row in rows:
                signal_data = {
                    "signal_id": row['signal_id'],
                    "symbol": row['symbol'],
                    "direction": row['direction'],
                    "confidence": row['confidence'],
                    "health_score": row['health_score'],
                    "entry_price": row['entry_price'],
                    "stop_loss": row['stop_loss'],
                    "take_profit": row['tp1'],
                    "timestamp": row['ts'].isoformat(),
                    "status": row['outcome']
                }
                signals.append(signal_data)
            
            await manager.send_personal_message(
                json.dumps({
                    "type": "latest_signals",
                    "data": signals,
                    "timestamp": datetime.now().isoformat()
                }),
                websocket
            )
    
    except Exception as e:
        logger.error(f"❌ Error sending latest signals: {e}")

async def send_latest_notifications(websocket: WebSocket):
    """Send latest notifications to WebSocket client"""
    try:
        if not db_connection:
            return
        
        async with db_connection.pool.acquire() as conn:
            # Get latest notifications
            rows = await conn.fetch("""
                SELECT * FROM signal_notifications 
                ORDER BY sent_at DESC 
                LIMIT 10
            """)
            
            notifications = []
            for row in rows:
                notification_data = {
                    "signal_id": row['signal_id'],
                    "notification_type": row['notification_type'],
                    "channel": row['channel'],
                    "delivery_status": row['delivery_status'],
                    "sent_at": row['sent_at'].isoformat()
                }
                notifications.append(notification_data)
            
            await manager.send_personal_message(
                json.dumps({
                    "type": "latest_notifications",
                    "data": notifications,
                    "timestamp": datetime.now().isoformat()
                }),
                websocket
            )
    
    except Exception as e:
        logger.error(f"❌ Error sending latest notifications: {e}")

async def _background_data_processor():
    """Background task for processing real-time data"""
    global signal_generator
    
    try:
        logger.info("🔄 Starting background data processor...")
        
        while True:
            try:
                # Process real-time data and trigger signal generation
                if signal_generator and signal_generator.is_running:
                    # The signal generator runs its own loop
                    pass
                else:
                    # Start signal generation if not running
                    if signal_generator:
                        await signal_generator.start_signal_generation()
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"❌ Error in background data processor: {e}")
                await asyncio.sleep(60)
    
    except asyncio.CancelledError:
        logger.info("Background data processor cancelled")
    except Exception as e:
        logger.error(f"❌ Background data processor failed: {e}")

async def _real_time_signal_processor():
    """Background task for real-time signal processing"""
    try:
        logger.info("🔄 Starting real-time signal processor...")
        
        while True:
            try:
                # Process real-time signal queue
                await process_real_time_signals()
                
                # Check for expired signals
                await check_expired_signals()
                
                await asyncio.sleep(10)  # Process every 10 seconds
                
            except Exception as e:
                logger.error(f"❌ Error in real-time signal processor: {e}")
                await asyncio.sleep(30)
    
    except asyncio.CancelledError:
        logger.info("Real-time signal processor cancelled")
    except Exception as e:
        logger.error(f"❌ Real-time signal processor failed: {e}")

async def process_real_time_signals():
    """Process signals in the real-time queue"""
    try:
        if not db_connection:
            return
        
        async with db_connection.pool.acquire() as conn:
            # Get pending signals ordered by priority
            rows = await conn.fetch("""
                SELECT * FROM real_time_signal_queue 
                WHERE status = 'pending' 
                ORDER BY priority DESC, confidence DESC 
                LIMIT 5
            """)
            
            for row in rows:
                # Process each signal
                await process_signal(row)
                
    except Exception as e:
        logger.error(f"❌ Error processing real-time signals: {e}")

async def process_signal(queue_row):
    """Process a single signal from the queue"""
    try:
        # Mark as processed
        async with db_connection.pool.acquire() as conn:
            await conn.execute("""
                UPDATE real_time_signal_queue 
                SET status = 'processed', processed_at = NOW()
                WHERE id = $1
            """, queue_row['id'])
        
        # Broadcast signal to subscribers
        signal_data = {
            "signal_id": queue_row['signal_id'],
            "symbol": queue_row['symbol'],
            "priority": queue_row['priority'],
            "confidence": queue_row['confidence'],
            "health_score": queue_row['health_score'],
            "status": "processed"
        }
        
        await manager.broadcast_signal(signal_data)
        logger.info(f"✅ Processed signal: {queue_row['signal_id']}")
        
    except Exception as e:
        logger.error(f"❌ Error processing signal: {e}")

async def check_expired_signals():
    """Check and expire old signals"""
    try:
        if not db_connection:
            return
        
        async with db_connection.pool.acquire() as conn:
            # Find expired signals
            expired_rows = await conn.fetch("""
                SELECT signal_id, symbol, expires_at 
                FROM signals 
                WHERE is_active = true 
                AND expires_at < NOW()
            """)
            
            for row in expired_rows:
                # Mark signal as expired
                await conn.execute("""
                    UPDATE signals 
                    SET is_active = false, 
                        cancelled_reason = 'expired',
                        outcome = 'expired'
                    WHERE signal_id = $1
                """, row['signal_id'])
                
                # Send notification
                notification_data = {
                    "type": "signal_expired",
                    "signal_id": row['signal_id'],
                    "symbol": row['symbol'],
                    "expired_at": row['expires_at'].isoformat()
                }
                
                await manager.broadcast_notification(notification_data)
                logger.info(f"⏰ Signal expired: {row['signal_id']}")
        
    except Exception as e:
        logger.error(f"❌ Error checking expired signals: {e}")

async def _notification_manager():
    """Background task for managing notifications"""
    try:
        logger.info("🔄 Starting notification manager...")
        
        while True:
            try:
                # Check for new notifications
                await check_new_notifications()
                
                await asyncio.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                logger.error(f"❌ Error in notification manager: {e}")
                await asyncio.sleep(30)
    
    except asyncio.CancelledError:
        logger.info("Notification manager cancelled")
    except Exception as e:
        logger.error(f"❌ Notification manager failed: {e}")

async def check_new_notifications():
    """Check for new notifications and broadcast them"""
    try:
        if not db_connection:
            return
        
        async with db_connection.pool.acquire() as conn:
            # Get new notifications
            rows = await conn.fetch("""
                SELECT * FROM signal_notifications 
                WHERE delivery_status = 'sent' 
                AND sent_at > NOW() - INTERVAL '1 minute'
                ORDER BY sent_at DESC
            """)
            
            for row in rows:
                notification_data = {
                    "signal_id": row['signal_id'],
                    "notification_type": row['notification_type'],
                    "channel": row['channel'],
                    "delivery_status": row['delivery_status'],
                    "sent_at": row['sent_at'].isoformat()
                }
                
                await manager.broadcast_notification(notification_data)
        
    except Exception as e:
        logger.error(f"❌ Error checking new notifications: {e}")

async def _system_monitor():
    """Background task for system monitoring"""
    try:
        logger.info("🔄 Starting system monitor...")
        
        while True:
            try:
                # Update system metrics
                await update_system_metrics()
                
                await asyncio.sleep(60)  # Update every minute
                
            except Exception as e:
                logger.error(f"❌ Error in system monitor: {e}")
                await asyncio.sleep(120)
    
    except asyncio.CancelledError:
        logger.info("System monitor cancelled")
    except Exception as e:
        logger.error(f"❌ System monitor failed: {e}")

async def update_system_metrics():
    """Update system performance metrics"""
    try:
        if not db_connection:
            return
        
        async with db_connection.pool.acquire() as conn:
            # Get system metrics
            active_signals = await conn.fetchval("""
                SELECT COUNT(*) FROM signals WHERE is_active = true
            """)
            
            total_signals_24h = await conn.fetchval("""
                SELECT COUNT(*) FROM signals 
                WHERE ts > NOW() - INTERVAL '24 hours'
            """)
            
            avg_confidence = await conn.fetchval("""
                SELECT AVG(confidence) FROM signals 
                WHERE ts > NOW() - INTERVAL '24 hours'
            """)
            
            # Broadcast metrics
            metrics_data = {
                "active_signals": active_signals or 0,
                "total_signals_24h": total_signals_24h or 0,
                "avg_confidence": round(avg_confidence or 0, 2),
                "timestamp": datetime.now().isoformat()
            }
            
            await manager.broadcast(json.dumps({
                "type": "system_metrics",
                "data": metrics_data
            }))
        
    except Exception as e:
        logger.error(f"❌ Error updating system metrics: {e}")

# API Routes
@app.get("/", response_class=HTMLResponse)
async def root():
    """Main dashboard page"""
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Enhanced AlphaPulse Dashboard</title>
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 0;
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
            }
            .header {
                text-align: center;
                margin-bottom: 30px;
            }
            .header h1 {
                font-size: 2.5em;
                margin: 0;
                text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
            }
            .header p {
                font-size: 1.2em;
                opacity: 0.9;
            }
            .dashboard-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 20px;
                margin-bottom: 30px;
            }
            .card {
                background: rgba(255, 255, 255, 0.1);
                backdrop-filter: blur(10px);
                border-radius: 15px;
                padding: 20px;
                border: 1px solid rgba(255, 255, 255, 0.2);
            }
            .card h3 {
                margin-top: 0;
                color: #fff;
                border-bottom: 2px solid rgba(255, 255, 255, 0.3);
                padding-bottom: 10px;
            }
            .metric {
                display: flex;
                justify-content: space-between;
                margin: 10px 0;
                padding: 8px 0;
                border-bottom: 1px solid rgba(255, 255, 255, 0.1);
            }
            .metric-value {
                font-weight: bold;
                color: #4ade80;
            }
            .status-indicator {
                display: inline-block;
                width: 12px;
                height: 12px;
                border-radius: 50%;
                margin-right: 8px;
            }
            .status-connected { background-color: #4ade80; }
            .status-disconnected { background-color: #ef4444; }
            .status-connecting { background-color: #f59e0b; }
            
            #connection-status {
                text-align: center;
                padding: 10px;
                border-radius: 8px;
                margin-bottom: 20px;
                font-weight: bold;
            }
            
            .signals-list {
                max-height: 300px;
                overflow-y: auto;
            }
            .signal-item {
                background: rgba(255, 255, 255, 0.05);
                margin: 5px 0;
                padding: 10px;
                border-radius: 8px;
                border-left: 4px solid #4ade80;
            }
            .signal-buy { border-left-color: #4ade80; }
            .signal-sell { border-left-color: #ef4444; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🚀 Enhanced AlphaPulse Dashboard</h1>
                <p>Real-time trading system with TimescaleDB integration</p>
                <div id="connection-status">🟡 Connecting to WebSocket...</div>
            </div>
            
            <div class="dashboard-grid">
                <div class="card">
                    <h3>📊 System Status</h3>
                    <div class="metric">
                        <span>WebSocket Status</span>
                        <span id="websocket-status">--</span>
                    </div>
                    <div class="metric">
                        <span>Active Clients</span>
                        <span class="metric-value" id="active-clients">--</span>
                    </div>
                    <div class="metric">
                        <span>Messages Sent</span>
                        <span class="metric-value" id="messages-sent">--</span>
                    </div>
                    <div class="metric">
                        <span>Avg Latency</span>
                        <span class="metric-value" id="avg-latency">--</span>
                    </div>
                </div>
                
                <div class="card">
                    <h3>📈 Performance Metrics</h3>
                    <div class="metric">
                        <span>Signals (24h)</span>
                        <span class="metric-value" id="signals-24h">--</span>
                    </div>
                    <div class="metric">
                        <span>Win Rate (7d)</span>
                        <span class="metric-value" id="win-rate">--</span>
                    </div>
                    <div class="metric">
                        <span>Avg Confidence</span>
                        <span class="metric-value" id="avg-confidence">--</span>
                    </div>
                    <div class="metric">
                        <span>Total Signals (7d)</span>
                        <span class="metric-value" id="total-signals">--</span>
                    </div>
                </div>
                
                <div class="card">
                    <h3>🔌 Binance Connection</h3>
                    <div class="metric">
                        <span>Connection Status</span>
                        <span id="binance-status">--</span>
                    </div>
                    <div class="metric">
                        <span>Messages Received</span>
                        <span class="metric-value" id="messages-received">--</span>
                    </div>
                    <div class="metric">
                        <span>Processing Latency</span>
                        <span class="metric-value" id="processing-latency">--</span>
                    </div>
                    <div class="metric">
                        <span>Errors</span>
                        <span class="metric-value" id="errors-count">--</span>
                    </div>
                </div>
            </div>
            
            <div class="card">
                <h3>📊 Latest Signals</h3>
                <div id="latest-signals" class="signals-list">
                    <p>Loading signals...</p>
                </div>
            </div>
        </div>
        
        <script>
            let ws = null;
            let reconnectAttempts = 0;
            const maxReconnectAttempts = 5;
            
            function connectWebSocket() {
                const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                const wsUrl = `${protocol}//${window.location.host}/ws`;
                
                console.log('🔌 Connecting to WebSocket:', wsUrl);
                
                ws = new WebSocket(wsUrl);
                
                ws.onopen = function() {
                    console.log('✅ WebSocket connected');
                    updateConnectionStatus('connected', '🟢 Connected');
                    reconnectAttempts = 0;
                    
                    // Subscribe to signals
                    ws.send(JSON.stringify({
                        type: 'subscribe',
                        subscription: 'signals'
                    }));
                };
                
                ws.onmessage = function(event) {
                    try {
                        const data = JSON.parse(event.data);
                        handleWebSocketMessage(data);
                    } catch (error) {
                        console.error('❌ Error parsing WebSocket message:', error);
                    }
                };
                
                ws.onclose = function() {
                    console.log('🔌 WebSocket disconnected');
                    updateConnectionStatus('disconnected', '🔴 Disconnected');
                    
                    if (reconnectAttempts < maxReconnectAttempts) {
                        reconnectAttempts++;
                        console.log(`🔄 Reconnecting... (attempt ${reconnectAttempts})`);
                        setTimeout(connectWebSocket, 2000 * reconnectAttempts);
                    }
                };
                
                ws.onerror = function(error) {
                    console.error('❌ WebSocket error:', error);
                    updateConnectionStatus('error', '🔴 Connection Error');
                };
            }
            
            function handleWebSocketMessage(data) {
                switch (data.type) {
                    case 'initial_data':
                        updateDashboard(data.data);
                        break;
                    case 'dashboard_update':
                        updateDashboard(data.data);
                        break;
                    case 'signal_update':
                        addNewSignal(data.data);
                        break;
                    case 'pong':
                        console.log('🏓 Received pong');
                        break;
                    default:
                        console.log('📨 Unknown message type:', data.type);
                }
            }
            
            function updateDashboard(data) {
                // Update system status
                if (data.system_status) {
                    document.getElementById('active-clients').textContent = data.system_status.active_clients || '--';
                    document.getElementById('messages-sent').textContent = data.system_status.messages_sent || '--';
                    document.getElementById('avg-latency').textContent = 
                        data.system_status.avg_latency_ms ? `${data.system_status.avg_latency_ms.toFixed(2)}ms` : '--';
                }
                
                // Update performance metrics
                if (data.performance_metrics) {
                    document.getElementById('signals-24h').textContent = data.performance_metrics.signals_24h || '--';
                    document.getElementById('win-rate').textContent = 
                        data.performance_metrics.win_rate_7d ? `${data.performance_metrics.win_rate_7d}%` : '--';
                    document.getElementById('avg-confidence').textContent = 
                        data.performance_metrics.avg_confidence_7d ? data.performance_metrics.avg_confidence_7d.toFixed(3) : '--';
                    document.getElementById('total-signals').textContent = data.performance_metrics.total_signals_7d || '--';
                }
                
                // Update latest signals
                if (data.latest_signals) {
                    updateSignalsList(data.latest_signals);
                }
            }
            
            function updateSignalsList(signals) {
                const container = document.getElementById('latest-signals');
                
                if (signals.length === 0) {
                    container.innerHTML = '<p>No signals available</p>';
                    return;
                }
                
                const signalsHtml = signals.map(signal => `
                    <div class="signal-item signal-${signal.direction}">
                        <div class="metric">
                            <span>${signal.symbol} (${signal.timeframe})</span>
                            <span class="metric-value">${signal.direction.toUpperCase()}</span>
                        </div>
                        <div class="metric">
                            <span>Price: $${signal.entry_price}</span>
                            <span>Confidence: ${(signal.confidence * 100).toFixed(1)}%</span>
                        </div>
                        <div class="metric">
                            <span>${new Date(signal.timestamp).toLocaleString()}</span>
                            <span>${signal.pattern_type || 'N/A'}</span>
                        </div>
                    </div>
                `).join('');
                
                container.innerHTML = signalsHtml;
            }
            
            function addNewSignal(signal) {
                const container = document.getElementById('latest-signals');
                const signalHtml = `
                    <div class="signal-item signal-${signal.direction}" style="animation: fadeIn 0.5s ease-in;">
                        <div class="metric">
                            <span>${signal.symbol} (${signal.timeframe})</span>
                            <span class="metric-value">${signal.direction.toUpperCase()}</span>
                        </div>
                        <div class="metric">
                            <span>Price: $${signal.entry_price}</span>
                            <span>Confidence: ${(signal.confidence * 100).toFixed(1)}%</span>
                        </div>
                        <div class="metric">
                            <span>${new Date(signal.timestamp).toLocaleString()}</span>
                            <span>${signal.pattern_type || 'N/A'}</span>
                        </div>
                    </div>
                `;
                
                container.insertAdjacentHTML('afterbegin', signalHtml);
                
                // Keep only last 10 signals
                const signals = container.querySelectorAll('.signal-item');
                if (signals.length > 10) {
                    signals[signals.length - 1].remove();
                }
            }
            
            function updateConnectionStatus(status, message) {
                const statusEl = document.getElementById('connection-status');
                statusEl.textContent = message;
                statusEl.className = `status-${status}`;
            }
            
            // Start WebSocket connection
            connectWebSocket();
            
            // Send ping every 30 seconds
            setInterval(() => {
                if (ws && ws.readyState === WebSocket.OPEN) {
                    ws.send(JSON.stringify({ type: 'ping' }));
                }
            }, 30000);
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.get("/api/status")
async def get_system_status():
    """Get comprehensive system status"""
    try:
        status = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "system": {
                "status": "operational",
                "uptime": time.time(),
                "version": "2.0.0"
            }
        }
        
        # Add WebSocket service metrics
        if websocket_service:
            status["websocket_service"] = websocket_service.get_service_metrics()
        
        # Add Binance WebSocket manager metrics
        if binance_websocket_manager:
            status["binance_websocket"] = binance_websocket_manager.get_manager_metrics()
        
        # Add database status
        if db_connection:
            status["database"] = {
                "status": "connected",
                "connection_pool_size": db_connection.pool_size if hasattr(db_connection, 'pool_size') else 'N/A'
            }
        
        return status
        
    except Exception as e:
        logger.error(f"❌ Error getting system status: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve system status")

@app.get("/api/signals")
async def get_signals(
    symbol: Optional[str] = Query(None, description="Filter by symbol"),
    timeframe: Optional[str] = Query(None, description="Filter by timeframe"),
    limit: int = Query(100, description="Number of signals to return", ge=1, le=1000)
):
    """Get signals from TimescaleDB"""
    try:
        if not db_connection:
            raise HTTPException(status_code=503, detail="Database not available")
        
        # Build query
        query = """
            SELECT 
                signal_id, timestamp, symbol, timeframe, direction, 
                confidence, entry_price, stop_loss, tp1, pattern_type,
                indicators, metadata, outcome
            FROM signals 
            WHERE 1=1
        """
        params = {}
        
        if symbol:
            query += " AND symbol = :symbol"
            params['symbol'] = symbol
        
        if timeframe:
            query += " AND timeframe = :timeframe"
            params['timeframe'] = timeframe
        
        query += " ORDER BY timestamp DESC LIMIT :limit"
        params['limit'] = limit
        
        # Execute query
        async with db_connection.async_session() as session:
            from sqlalchemy import text
            result = await session.execute(text(query), params)
            rows = result.fetchall()
            
            # Convert to dict format
            signals = []
            for row in rows:
                signal_dict = {
                    'signal_id': row.signal_id,
                    'timestamp': row.timestamp.isoformat(),
                    'symbol': row.symbol,
                    'timeframe': row.timeframe,
                    'direction': row.direction,
                    'confidence': float(row.confidence),
                    'entry_price': float(row.entry_price),
                    'stop_loss': float(row.stop_loss) if row.stop_loss else None,
                    'tp1': float(row.tp1) if row.tp1 else None,
                    'pattern_type': row.pattern_type,
                    'indicators': row.indicators,
                    'metadata': row.metadata,
                    'outcome': row.outcome
                }
                signals.append(signal_dict)
            
            return {"signals": signals, "count": len(signals)}
            
    except Exception as e:
        logger.error(f"❌ Error getting signals: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve signals")

@app.get("/api/signals/latest")
async def get_latest_signals(limit: int = Query(10, description="Number of latest signals", ge=1, le=50)):
    """Get latest signals"""
    return await get_signals(limit=limit)

@app.get("/api/patterns/latest")
async def get_latest_patterns(limit: int = Query(10, description="Number of latest patterns", ge=1, le=50)):
    """Get latest patterns"""
    try:
        if not db_connection:
            raise HTTPException(status_code=503, detail="Database not available")
        
        query = """
            SELECT 
                id, timestamp, symbol, timeframe, pattern_type, 
                confidence, entry_price, stop_loss, tp1, metadata
            FROM enhanced_patterns 
            ORDER BY timestamp DESC 
            LIMIT :limit
        """
        
        async with db_connection.async_session() as session:
            from sqlalchemy import text
            result = await session.execute(text(query), {"limit": limit})
            rows = result.fetchall()
            
            patterns = []
            for row in rows:
                pattern_dict = {
                    'id': row.id,
                    'timestamp': row.timestamp.isoformat(),
                    'symbol': row.symbol,
                    'timeframe': row.timeframe,
                    'pattern_type': row.pattern_type,
                    'confidence': float(row.confidence),
                    'entry_price': float(row.entry_price),
                    'stop_loss': float(row.stop_loss) if row.stop_loss else None,
                    'tp1': float(row.tp1) if row.tp1 else None,
                    'metadata': row.metadata
                }
                patterns.append(pattern_dict)
            
            return {"patterns": patterns, "count": len(patterns)}
            
    except Exception as e:
        logger.error(f"❌ Error getting patterns: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve patterns")

@app.get("/api/market/status")
async def get_market_status():
    """Get market status and metrics"""
    try:
        if not db_connection:
            raise HTTPException(status_code=503, detail="Database not available")
        
        # Get basic market metrics
        query = """
            SELECT 
                COUNT(DISTINCT symbol) as active_symbols,
                COUNT(*) as total_signals,
                AVG(confidence) as avg_confidence
            FROM signals 
            WHERE timestamp >= NOW() - INTERVAL '24 hours'
        """
        
        async with db_connection.async_session() as session:
            from sqlalchemy import text
            result = await session.execute(text(query))
            row = result.fetchone()
            
            market_status = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "active_symbols": row.active_symbols if row.active_symbols else 0,
                "total_signals_24h": row.total_signals if row.total_signals else 0,
                "average_confidence": float(row.avg_confidence) if row.avg_confidence else 0.0,
                "system_status": "operational"
            }
            
            return market_status
            
    except Exception as e:
        logger.error(f"❌ Error getting market status: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve market status")

@app.get("/api/ai/performance")
async def get_ai_performance():
    """Get AI system performance metrics"""
    try:
        if not db_connection:
            raise HTTPException(status_code=503, detail="Database not available")
        
        # Get AI performance metrics
        query = """
            SELECT 
                COUNT(*) as total_predictions,
                AVG(confidence) as avg_confidence,
                COUNT(CASE WHEN outcome = 'win' THEN 1 END) as wins,
                COUNT(CASE WHEN outcome = 'loss' THEN 1 END) as losses
            FROM signals 
            WHERE timestamp >= NOW() - INTERVAL '7 days'
            AND outcome IS NOT NULL
        """
        
        async with db_connection.async_session() as session:
            from sqlalchemy import text
            result = await session.execute(text(query))
            row = result.fetchone()
            
            total = row.total_predictions if row.total_predictions else 0
            wins = row.wins if row.wins else 0
            losses = row.losses if row.losses else 0
            
            win_rate = (wins / total * 100) if total > 0 else 0
            
            ai_performance = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "total_predictions_7d": total,
                "wins": wins,
                "losses": losses,
                "win_rate_percent": round(win_rate, 2),
                "average_confidence": float(row.avg_confidence) if row.avg_confidence else 0.0
            }
            
            return ai_performance
            
    except Exception as e:
        logger.error(f"❌ Error getting AI performance: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve AI performance")

@app.get("/api/test/phase3")
async def test_phase3():
    """Test endpoint for phase 3"""
    return {
        "status": "success",
        "message": "Phase 3 test endpoint working",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": "2.0.0"
    }


@app.get("/api/performance")
async def get_performance_metrics():
    """Get performance metrics from TimescaleDB"""
    try:
        if not db_connection:
            raise HTTPException(status_code=503, detail="Database not available")
        
        # Get various performance metrics
        queries = {
            "signals_24h": """
                SELECT COUNT(*) as count
                FROM signals 
                WHERE timestamp >= NOW() - INTERVAL '24 hours'
            """,
            "signals_7d": """
                SELECT COUNT(*) as count
                FROM signals 
                WHERE timestamp >= NOW() - INTERVAL '7 days'
            """,
            "performance_7d": """
                SELECT 
                    COUNT(*) as total_signals,
                    COUNT(CASE WHEN outcome = 'win' THEN 1 END) as wins,
                    COUNT(CASE WHEN outcome = 'loss' THEN 1 END) as losses,
                    AVG(confidence) as avg_confidence,
                    AVG(CASE WHEN outcome = 'win' THEN 1 ELSE 0 END) * 100 as win_rate
                FROM signals 
                WHERE outcome IN ('win', 'loss')
                AND timestamp >= NOW() - INTERVAL '7 days'
            """,
            "top_symbols": """
                SELECT symbol, COUNT(*) as signal_count
                FROM signals 
                WHERE timestamp >= NOW() - INTERVAL '7 days'
                GROUP BY symbol
                ORDER BY signal_count DESC
                LIMIT 5
            """
        }
        
        async with db_connection.async_session() as session:
            from sqlalchemy import text
            
            results = {}
            
            for key, query in queries.items():
                result = await session.execute(text(query))
                if key == "top_symbols":
                    rows = result.fetchall()
                    results[key] = [{"symbol": row.symbol, "count": row.signal_count} for row in rows]
                else:
                    row = result.fetchone()
                    if row:
                        results[key] = dict(row._mapping)
            
            return {
                "metrics": results,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
    except Exception as e:
        logger.error(f"❌ Error retrieving performance metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve performance metrics")

if __name__ == "__main__":
    uvicorn.run(
        "main_enhanced_websocket:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
