#!/usr/bin/env python3
"""
Enhanced WebSocket Service for AlphaPulse Dashboard
Real-time updates with TimescaleDB integration and performance optimizations
"""

import asyncio
import json
import logging
import time
import json
from typing import Dict, List, Optional, Any, Set
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, asdict
from fastapi import WebSocket, WebSocketDisconnect
import redis.asyncio as aioredis
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

# Local imports
from src.database.connection import TimescaleDBConnection
from src.database.models import Signal
from src.core.websocket_enhanced import SignalData

logger = logging.getLogger(__name__)

@dataclass
class DashboardUpdate:
    """Dashboard update data structure"""
    timestamp: datetime
    update_type: str
    data: Dict[str, Any]
    priority: int = 1  # 1=low, 5=high

@dataclass
class ClientInfo:
    """Client connection information"""
    websocket: WebSocket
    client_id: str
    connected_at: datetime
    last_activity: datetime
    subscriptions: Set[str]
    user_agent: Optional[str] = None

class EnhancedWebSocketService:
    """
    Enhanced WebSocket service for real-time dashboard updates
    - TimescaleDB integration for signal queries
    - Redis pub/sub for efficient broadcasting
    - Delta updates to reduce bandwidth
    - Client management and subscriptions
    """
    
    def __init__(self, 
                 redis_url: str = "redis://localhost:6379",
                 update_interval: float = 3.0,
                 max_clients: int = 1000):
        """
        Initialize enhanced WebSocket service
        
        Args:
            redis_url: Redis connection URL
            update_interval: Update interval in seconds
            max_clients: Maximum number of connected clients
        """
        self.redis_url = redis_url
        self.update_interval = update_interval
        self.max_clients = max_clients
        
        # Client management
        self.clients: Dict[str, ClientInfo] = {}
        self.client_counter = 0
        
        # Database connection
        self.db_connection = None
        
        # Redis for broadcasting
        self.redis_client = None
        self.redis_pubsub = None
        
        # Update management
        self.update_queue = asyncio.Queue(maxsize=10000)
        self.last_updates = {}  # Cache for delta updates
        self.update_tasks = []
        
        # Performance tracking
        self.metrics = {
            'total_clients': 0,
            'active_clients': 0,
            'messages_sent': 0,
            'messages_dropped': 0,
            'avg_latency_ms': 0.0,
            'last_update_time': None
        }
        
        logger.info("Enhanced WebSocket Service initialized")
    
    async def initialize(self):
        """Initialize database and Redis connections"""
        try:
            # Initialize TimescaleDB connection (without table creation)
            self.db_connection = TimescaleDBConnection()
            await self.db_connection.initialize(create_tables=False)
            
            # Initialize Redis
            self.redis_client = await aioredis.from_url(self.redis_url)
            self.redis_pubsub = self.redis_client.pubsub()
            
            # Subscribe to signal updates
            await self.redis_pubsub.subscribe('alphapulse_signals')
            
            # Start background tasks
            await self._start_background_tasks()
            
            logger.info("✅ Enhanced WebSocket Service initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize WebSocket service: {e}")
            raise
    
    async def connect_client(self, websocket: WebSocket) -> str:
        """Accept new client connection"""
        await websocket.accept()
        
        # Generate client ID
        self.client_counter += 1
        client_id = f"client_{self.client_counter}_{int(time.time())}"
        
        # Create client info
        client_info = ClientInfo(
            websocket=websocket,
            client_id=client_id,
            connected_at=datetime.now(timezone.utc),
            last_activity=datetime.now(timezone.utc),
            subscriptions=set(),
            user_agent=websocket.headers.get('user-agent')
        )
        
        # Add to clients
        self.clients[client_id] = client_info
        self.metrics['total_clients'] += 1
        self.metrics['active_clients'] += 1
        
        logger.info(f"✅ Client {client_id} connected (total: {self.metrics['active_clients']})")
        
        # Send initial data
        await self._send_initial_data(client_id)
        
        return client_id
    
    async def disconnect_client(self, client_id: str):
        """Disconnect client"""
        if client_id in self.clients:
            client_info = self.clients[client_id]
            
            # Close WebSocket
            try:
                await client_info.websocket.close()
            except Exception as e:
                logger.warning(f"Error closing WebSocket for {client_id}: {e}")
            
            # Remove from clients
            del self.clients[client_id]
            self.metrics['active_clients'] -= 1
            
            logger.info(f"🔌 Client {client_id} disconnected (active: {self.metrics['active_clients']})")
    
    async def handle_client_message(self, client_id: str, message: str):
        """Handle incoming client message"""
        try:
            data = json.loads(message)
            client_info = self.clients[client_id]
            
            # Update last activity
            client_info.last_activity = datetime.now(timezone.utc)
            
            # Handle different message types
            message_type = data.get('type')
            
            if message_type == 'subscribe':
                await self._handle_subscription(client_id, data)
            elif message_type == 'unsubscribe':
                await self._handle_unsubscription(client_id, data)
            elif message_type == 'ping':
                await self._send_pong(client_id)
            elif message_type == 'get_signals':
                await self._handle_signal_request(client_id, data)
            else:
                logger.warning(f"Unknown message type from {client_id}: {message_type}")
                
        except Exception as e:
            logger.error(f"❌ Error handling message from {client_id}: {e}")
    
    async def _handle_subscription(self, client_id: str, data: Dict):
        """Handle client subscription"""
        client_info = self.clients[client_id]
        subscription = data.get('subscription')
        
        if subscription:
            client_info.subscriptions.add(subscription)
            logger.info(f"📡 Client {client_id} subscribed to {subscription}")
    
    async def _handle_unsubscription(self, client_id: str, data: Dict):
        """Handle client unsubscription"""
        client_info = self.clients[client_id]
        subscription = data.get('subscription')
        
        if subscription in client_info.subscriptions:
            client_info.subscriptions.remove(subscription)
            logger.info(f"📡 Client {client_id} unsubscribed from {subscription}")
    
    async def _send_pong(self, client_id: str):
        """Send pong response"""
        await self._send_to_client(client_id, {
            'type': 'pong',
            'timestamp': datetime.now(timezone.utc).isoformat()
        })
    
    async def _handle_signal_request(self, client_id: str, data: Dict):
        """Handle signal data request"""
        try:
            # Extract request parameters
            symbol = data.get('symbol')
            timeframe = data.get('timeframe')
            limit = data.get('limit', 100)
            
            # Query signals from TimescaleDB
            signals = await self._get_signals_from_db(symbol, timeframe, limit)
            
            # Send response
            await self._send_to_client(client_id, {
                'type': 'signals_response',
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'data': signals
            })
            
        except Exception as e:
            logger.error(f"❌ Error handling signal request: {e}")
            await self._send_to_client(client_id, {
                'type': 'error',
                'message': 'Failed to retrieve signals'
            })
    
    async def _get_signals_from_db(self, symbol: Optional[str], timeframe: Optional[str], limit: int) -> List[Dict]:
        """Get signals from TimescaleDB"""
        try:
            if not self.db_connection:
                return []
            
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
            async with self.db_connection.async_session() as session:
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
                
                return signals
                
        except Exception as e:
            logger.error(f"❌ Error querying signals from database: {e}")
            return []
    
    async def _send_initial_data(self, client_id: str):
        """Send initial data to new client"""
        try:
            # Get system status
            system_status = await self._get_system_status()
            
            # Get recent signals
            recent_signals = await self._get_signals_from_db(None, None, 50)
            
            # Get performance metrics
            performance_metrics = await self._get_performance_metrics()
            
            initial_data = {
                'type': 'initial_data',
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'data': {
                    'system_status': system_status,
                    'recent_signals': recent_signals,
                    'performance_metrics': performance_metrics
                }
            }
            
            await self._send_to_client(client_id, initial_data)
            
        except Exception as e:
            logger.error(f"❌ Error sending initial data to {client_id}: {e}")
    
    async def _get_system_status(self) -> Dict[str, Any]:
        """Get system status information"""
        return {
            'status': 'operational',
            'uptime': time.time(),
            'active_clients': self.metrics['active_clients'],
            'total_clients': self.metrics['total_clients'],
            'messages_sent': self.metrics['messages_sent'],
            'avg_latency_ms': self.metrics['avg_latency_ms']
        }
    
    async def _get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics from TimescaleDB"""
        try:
            if not self.db_connection:
                return {}
            
            # Get signal count in last 24 hours
            query_24h = """
                SELECT COUNT(*) as count
                FROM signals 
                WHERE timestamp >= NOW() - INTERVAL '24 hours'
            """
            
            # Get signal performance
            query_performance = """
                SELECT 
                    COUNT(*) as total_signals,
                    COUNT(CASE WHEN outcome = 'win' THEN 1 END) as wins,
                    COUNT(CASE WHEN outcome = 'loss' THEN 1 END) as losses,
                    AVG(confidence) as avg_confidence
                FROM signals 
                WHERE outcome IN ('win', 'loss')
                AND timestamp >= NOW() - INTERVAL '7 days'
            """
            
            async with self.db_connection.async_session() as session:
                # Get 24h count
                result_24h = await session.execute(text(query_24h))
                count_24h = result_24h.scalar()
                
                # Get performance metrics
                result_perf = await session.execute(text(query_performance))
                row = result_perf.fetchone()
                
                if row:
                    win_rate = (row.wins / (row.wins + row.losses)) * 100 if (row.wins + row.losses) > 0 else 0
                    
                    return {
                        'signals_24h': count_24h,
                        'total_signals_7d': row.total_signals,
                        'wins_7d': row.wins,
                        'losses_7d': row.losses,
                        'win_rate_7d': round(win_rate, 2),
                        'avg_confidence_7d': round(float(row.avg_confidence), 3) if row.avg_confidence else 0
                    }
                
                return {}
                
        except Exception as e:
            logger.error(f"❌ Error getting performance metrics: {e}")
            return {}
    
    async def _start_background_tasks(self):
        """Start background processing tasks"""
        self.update_tasks = [
            asyncio.create_task(self._update_broadcaster()),
            asyncio.create_task(self._redis_listener()),
            asyncio.create_task(self._client_cleanup()),
            asyncio.create_task(self._metrics_updater())
        ]
        
        logger.info("✅ Background tasks started")
    
    async def _update_broadcaster(self):
        """Broadcast periodic updates to all clients"""
        while True:
            try:
                await asyncio.sleep(self.update_interval)
                
                if not self.clients:
                    continue
                
                # Prepare update data
                update_data = await self._prepare_update_data()
                
                # Broadcast to all clients
                await self._broadcast_to_all(update_data)
                
                # Update metrics
                self.metrics['last_update_time'] = datetime.now(timezone.utc)
                
            except Exception as e:
                logger.error(f"❌ Error in update broadcaster: {e}")
    
    async def _prepare_update_data(self) -> Dict[str, Any]:
        """Prepare update data for broadcasting"""
        try:
            # Get real-time metrics
            performance_metrics = await self._get_performance_metrics()
            
            # Get latest signals
            latest_signals = await self._get_signals_from_db(None, None, 10)
            
            update_data = {
                'type': 'dashboard_update',
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'data': {
                    'performance_metrics': performance_metrics,
                    'latest_signals': latest_signals,
                    'active_clients': self.metrics['active_clients'],
                    'system_status': await self._get_system_status()
                }
            }
            
            return update_data
            
        except Exception as e:
            logger.error(f"❌ Error preparing update data: {e}")
            return {
                'type': 'dashboard_update',
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'data': {}
            }
    
    async def _redis_listener(self):
        """Listen for Redis pub/sub messages"""
        while True:
            try:
                if not self.redis_pubsub:
                    await asyncio.sleep(1)
                    continue
                
                message = await self.redis_pubsub.get_message(timeout=1.0)
                
                if message and message['type'] == 'message':
                    # Parse signal data
                    signal_data = json.loads(message['data'])
                    
                    # Create real-time signal update
                    signal_update = {
                        'type': 'signal_update',
                        'timestamp': datetime.now(timezone.utc).isoformat(),
                        'data': signal_data
                    }
                    
                    # Broadcast to subscribed clients
                    await self._broadcast_to_subscribers('signals', signal_update)
                
            except Exception as e:
                logger.error(f"❌ Error in Redis listener: {e}")
                await asyncio.sleep(1)
    
    async def _client_cleanup(self):
        """Clean up inactive clients"""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute
                
                current_time = datetime.now(timezone.utc)
                inactive_clients = []
                
                for client_id, client_info in self.clients.items():
                    # Check if client has been inactive for more than 5 minutes
                    if (current_time - client_info.last_activity).seconds > 300:
                        inactive_clients.append(client_id)
                
                # Disconnect inactive clients
                for client_id in inactive_clients:
                    await self.disconnect_client(client_id)
                    logger.info(f"🔌 Disconnected inactive client {client_id}")
                
            except Exception as e:
                logger.error(f"❌ Error in client cleanup: {e}")
    
    async def _metrics_updater(self):
        """Update service metrics"""
        while True:
            try:
                await asyncio.sleep(30)  # Update every 30 seconds
                
                # Calculate average latency
                if self.metrics['messages_sent'] > 0:
                    # This would be calculated from actual message send times
                    pass
                
                logger.info(f"📊 WebSocket Service Metrics: "
                          f"Active={self.metrics['active_clients']}, "
                          f"Total={self.metrics['total_clients']}, "
                          f"Sent={self.metrics['messages_sent']}, "
                          f"Dropped={self.metrics['messages_dropped']}")
                
            except Exception as e:
                logger.error(f"❌ Error updating metrics: {e}")
    
    async def _broadcast_to_all(self, data: Dict[str, Any]):
        """Broadcast data to all connected clients"""
        if not self.clients:
            return
        
        # Convert to JSON once
        try:
            json_data = json.dumps(data)
        except Exception as e:
            logger.error(f"❌ Error serializing broadcast data: {e}")
            return
        
        # Send to all clients
        disconnected_clients = []
        
        for client_id, client_info in self.clients.items():
            try:
                await client_info.websocket.send_text(json_data)
                self.metrics['messages_sent'] += 1
                
            except Exception as e:
                logger.warning(f"⚠️ Failed to send to client {client_id}: {e}")
                disconnected_clients.append(client_id)
        
        # Clean up disconnected clients
        for client_id in disconnected_clients:
            await self.disconnect_client(client_id)
    
    async def _broadcast_to_subscribers(self, subscription: str, data: Dict[str, Any]):
        """Broadcast data to clients subscribed to specific topics"""
        if not self.clients:
            return
        
        # Convert to JSON once
        try:
            json_data = json.dumps(data)
        except Exception as e:
            logger.error(f"❌ Error serializing subscription data: {e}")
            return
        
        # Send to subscribed clients
        disconnected_clients = []
        
        for client_id, client_info in self.clients.items():
            if subscription in client_info.subscriptions:
                try:
                    await client_info.websocket.send_text(json_data)
                    self.metrics['messages_sent'] += 1
                    
                except Exception as e:
                    logger.warning(f"⚠️ Failed to send to subscribed client {client_id}: {e}")
                    disconnected_clients.append(client_id)
        
        # Clean up disconnected clients
        for client_id in disconnected_clients:
            await self.disconnect_client(client_id)
    
    async def _send_to_client(self, client_id: str, data: Dict[str, Any]):
        """Send data to specific client"""
        if client_id not in self.clients:
            return
        
        try:
            client_info = self.clients[client_id]
            json_data = json.dumps(data)
            await client_info.websocket.send_text(json_data)
            self.metrics['messages_sent'] += 1
            
        except Exception as e:
            logger.error(f"❌ Error sending to client {client_id}: {e}")
            await self.disconnect_client(client_id)
    
    def get_service_metrics(self) -> Dict[str, Any]:
        """Get service metrics"""
        return {
            'active_clients': self.metrics['active_clients'],
            'total_clients': self.metrics['total_clients'],
            'messages_sent': self.metrics['messages_sent'],
            'messages_dropped': self.metrics['messages_dropped'],
            'avg_latency_ms': self.metrics['avg_latency_ms'],
            'last_update_time': self.metrics['last_update_time'].isoformat() if self.metrics['last_update_time'] else None
        }
    
    async def shutdown(self):
        """Shutdown the service"""
        logger.info("🔄 Shutting down Enhanced WebSocket Service...")
        
        # Stop background tasks
        for task in self.update_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*self.update_tasks, return_exceptions=True)
        
        # Disconnect all clients
        for client_id in list(self.clients.keys()):
            await self.disconnect_client(client_id)
        
        # Close Redis connection
        if self.redis_pubsub:
            await self.redis_pubsub.close()
        
        if self.redis_client:
            await self.redis_client.close()
        
        logger.info("✅ Enhanced WebSocket Service shutdown complete")
