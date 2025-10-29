#!/usr/bin/env python3
"""
Enhanced WebSocket Service for AlphaPlus
Ultra-low latency real-time data delivery with cache integration
"""

import asyncio
import json
import logging
import time
from typing import Dict, List, Optional, Any, Set
from datetime import datetime, timedelta
from dataclasses import dataclass
from fastapi import WebSocket, WebSocketDisconnect
from collections import defaultdict
from collections import deque

# Import existing components
from .enhanced_cache_manager import EnhancedCacheManager
from .enhanced_data_pipeline import EnhancedDataPipeline

logger = logging.getLogger(__name__)

@dataclass
class WebSocketConnection:
    """WebSocket connection metadata"""
    websocket: WebSocket
    client_id: str
    subscribed_symbols: Set[str]
    subscribed_timeframes: Set[str]
    connected_at: datetime
    last_activity: datetime
    message_count: int = 0

class EnhancedWebSocketService:
    """
    Enhanced WebSocket service with cache integration
    Provides ultra-low latency real-time data delivery
    """
    
    def __init__(self, 
                 cache_manager: EnhancedCacheManager,
                 data_pipeline: EnhancedDataPipeline,
                 symbols: List[str] = None,
                 timeframes: List[str] = None):
        """
        Initialize enhanced WebSocket service
        
        Args:
            cache_manager: Cache manager for data retrieval
            data_pipeline: Data pipeline for processing
            symbols: Available trading symbols
            timeframes: Available timeframes
        """
        self.cache_manager = cache_manager
        self.data_pipeline = data_pipeline
        self.symbols = symbols or ['BTC/USDT', 'ETH/USDT', 'ADA/USDT', 'SOL/USDT', 'BNB/USDT', 'XRP/USDT']
        self.timeframes = timeframes or ['1m', '5m', '15m', '1h', '4h']
        
        # WebSocket connections management
        self.active_connections: Dict[str, WebSocketConnection] = {}
        self.connection_counter = 0
        
        # Performance tracking
        self.stats = {
            'total_connections': 0,
            'active_connections': 0,
            'total_messages_sent': 0,
            'avg_latency_ms': 0.0,
            'errors_count': 0
        }
        
        # Message delivery tracking
        self.latency_times = []
        self.message_times = deque(maxlen=1000)
        
        # Broadcast queues for different data types
        self.market_data_queue = asyncio.Queue()
        self.signal_queue = asyncio.Queue()
        self.pattern_queue = asyncio.Queue()
        
        # Service state
        self.is_running = False
        self.broadcast_tasks = []
        
        logger.info("🚀 Enhanced WebSocket Service initialized")
    
    async def start(self):
        """Start the WebSocket service"""
        try:
            self.is_running = True
            logger.info("🚀 Starting Enhanced WebSocket Service...")
            
            # Start broadcast tasks
            tasks = [
                asyncio.create_task(self._market_data_broadcast()),
                asyncio.create_task(self._signal_broadcast()),
                asyncio.create_task(self._pattern_broadcast()),
                asyncio.create_task(self._connection_monitoring()),
                asyncio.create_task(self._performance_monitoring())
            ]
            
            self.broadcast_tasks = tasks
            
            logger.info("✅ Enhanced WebSocket Service started")
            
        except Exception as e:
            logger.error(f"❌ Error starting WebSocket service: {e}")
            raise
    
    async def stop(self):
        """Stop the WebSocket service"""
        try:
            self.is_running = False
            logger.info("🛑 Stopping Enhanced WebSocket Service...")
            
            # Cancel broadcast tasks
            for task in self.broadcast_tasks:
                task.cancel()
            
            # Close all connections
            for connection in list(self.active_connections.values()):
                await self._close_connection(connection)
            
            logger.info("✅ Enhanced WebSocket Service stopped")
            
        except Exception as e:
            logger.error(f"❌ Error stopping WebSocket service: {e}")
    
    async def handle_connection(self, websocket: WebSocket, client_id: str = None):
        """Handle new WebSocket connection"""
        try:
            await websocket.accept()
            
            # Generate client ID if not provided
            if not client_id:
                self.connection_counter += 1
                client_id = f"client_{self.connection_counter}_{int(time.time())}"
            
            # Create connection object
            connection = WebSocketConnection(
                websocket=websocket,
                client_id=client_id,
                subscribed_symbols=set(),
                subscribed_timeframes=set(),
                connected_at=datetime.now(),
                last_activity=datetime.now()
            )
            
            # Store connection
            self.active_connections[client_id] = connection
            self.stats['total_connections'] += 1
            self.stats['active_connections'] += 1
            
            logger.info(f"🔌 New WebSocket connection: {client_id}")
            
            # Send welcome message
            welcome_message = {
                'type': 'connection_established',
                'client_id': client_id,
                'timestamp': datetime.now().isoformat(),
                'available_symbols': self.symbols,
                'available_timeframes': self.timeframes,
                'message': 'Connected to AlphaPlus Enhanced WebSocket Service'
            }
            
            await websocket.send_text(json.dumps(welcome_message))
            
            # Handle client messages
            await self._handle_client_messages(connection)
            
        except WebSocketDisconnect:
            logger.info(f"🔌 WebSocket disconnected: {client_id}")
            await self._remove_connection(client_id)
        except Exception as e:
            logger.error(f"❌ Error handling WebSocket connection: {e}")
            await self._remove_connection(client_id)
    
    async def _handle_client_messages(self, connection: WebSocketConnection):
        """Handle messages from WebSocket client"""
        try:
            while self.is_running and connection.client_id in self.active_connections:
                try:
                    # Receive message from client
                    message = await connection.websocket.receive_text()
                    connection.last_activity = datetime.now()
                    connection.message_count += 1
                    
                    # Parse message
                    data = json.loads(message)
                    message_type = data.get('type', 'unknown')
                    
                    # Handle different message types
                    if message_type == 'subscribe':
                        await self._handle_subscribe(connection, data)
                    elif message_type == 'unsubscribe':
                        await self._handle_unsubscribe(connection, data)
                    elif message_type == 'ping':
                        await self._handle_ping(connection)
                    elif message_type == 'request_data':
                        await self._handle_data_request(connection, data)
                    else:
                        await self._send_error(connection, f"Unknown message type: {message_type}")
                
                except WebSocketDisconnect:
                    break
                except json.JSONDecodeError:
                    await self._send_error(connection, "Invalid JSON message")
                except Exception as e:
                    logger.error(f"❌ Error handling client message: {e}")
                    await self._send_error(connection, f"Internal error: {str(e)}")
            
        except Exception as e:
            logger.error(f"❌ Error in client message handler: {e}")
    
    async def _handle_subscribe(self, connection: WebSocketConnection, data: Dict):
        """Handle subscription request"""
        try:
            symbols = data.get('symbols', [])
            timeframes = data.get('timeframes', [])
            
            # Validate symbols and timeframes
            valid_symbols = [s for s in symbols if s in self.symbols]
            valid_timeframes = [tf for tf in timeframes if tf in self.timeframes]
            
            # Update subscriptions
            connection.subscribed_symbols.update(valid_symbols)
            connection.subscribed_timeframes.update(valid_timeframes)
            
            # Send confirmation
            response = {
                'type': 'subscription_confirmed',
                'symbols': valid_symbols,
                'timeframes': valid_timeframes,
                'timestamp': datetime.now().isoformat()
            }
            
            await connection.websocket.send_text(json.dumps(response))
            
            logger.info(f"📡 Client {connection.client_id} subscribed to {len(valid_symbols)} symbols, {len(valid_timeframes)} timeframes")
            
        except Exception as e:
            logger.error(f"❌ Error handling subscription: {e}")
            await self._send_error(connection, f"Subscription error: {str(e)}")
    
    async def _handle_unsubscribe(self, connection: WebSocketConnection, data: Dict):
        """Handle unsubscription request"""
        try:
            symbols = data.get('symbols', [])
            timeframes = data.get('timeframes', [])
            
            # Remove subscriptions
            connection.subscribed_symbols.difference_update(symbols)
            connection.subscribed_timeframes.difference_update(timeframes)
            
            # Send confirmation
            response = {
                'type': 'unsubscription_confirmed',
                'symbols': symbols,
                'timeframes': timeframes,
                'timestamp': datetime.now().isoformat()
            }
            
            await connection.websocket.send_text(json.dumps(response))
            
            logger.info(f"📡 Client {connection.client_id} unsubscribed from {len(symbols)} symbols, {len(timeframes)} timeframes")
            
        except Exception as e:
            logger.error(f"❌ Error handling unsubscription: {e}")
            await self._send_error(connection, f"Unsubscription error: {str(e)}")
    
    async def _handle_ping(self, connection: WebSocketConnection):
        """Handle ping message"""
        try:
            response = {
                'type': 'pong',
                'timestamp': datetime.now().isoformat()
            }
            
            await connection.websocket.send_text(json.dumps(response))
            
        except Exception as e:
            logger.error(f"❌ Error handling ping: {e}")
    
    async def _handle_data_request(self, connection: WebSocketConnection, data: Dict):
        """Handle data request"""
        try:
            symbol = data.get('symbol')
            timeframe = data.get('timeframe')
            data_type = data.get('data_type', 'market_data')
            
            if not symbol or not timeframe:
                await self._send_error(connection, "Missing symbol or timeframe")
                return
            
            # Get data from cache
            if data_type == 'market_data':
                cached_data = await self.cache_manager.get_market_data(symbol, timeframe, limit=100)
            elif data_type == 'signals':
                cached_data = await self.cache_manager.get_signals(symbol, timeframe, limit=50)
            elif data_type == 'real_time':
                cached_data = await self.cache_manager.get_real_time_data(symbol, timeframe)
            else:
                await self._send_error(connection, f"Unknown data type: {data_type}")
                return
            
            # Send response
            response = {
                'type': 'data_response',
                'data_type': data_type,
                'symbol': symbol,
                'timeframe': timeframe,
                'data': cached_data or [],
                'timestamp': datetime.now().isoformat()
            }
            
            await connection.websocket.send_text(json.dumps(response))
            
        except Exception as e:
            logger.error(f"❌ Error handling data request: {e}")
            await self._send_error(connection, f"Data request error: {str(e)}")
    
    async def _send_error(self, connection: WebSocketConnection, error_message: str):
        """Send error message to client"""
        try:
            error_response = {
                'type': 'error',
                'message': error_message,
                'timestamp': datetime.now().isoformat()
            }
            
            await connection.websocket.send_text(json.dumps(error_response))
            
        except Exception as e:
            logger.error(f"❌ Error sending error message: {e}")
    
    async def _market_data_broadcast(self):
        """Broadcast market data updates"""
        try:
            while self.is_running:
                try:
                    # Get market data from queue
                    data = await asyncio.wait_for(self.market_data_queue.get(), timeout=1.0)
                    
                    # Broadcast to subscribed clients
                    await self._broadcast_market_data(data)
                    
                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    logger.error(f"❌ Error in market data broadcast: {e}")
                    self.stats['errors_count'] += 1
            
        except Exception as e:
            logger.error(f"❌ Error in market data broadcast: {e}")
    
    async def _signal_broadcast(self):
        """Broadcast signal updates"""
        try:
            while self.is_running:
                try:
                    # Get signal from queue
                    signal = await asyncio.wait_for(self.signal_queue.get(), timeout=1.0)
                    
                    # Broadcast to subscribed clients
                    await self._broadcast_signal(signal)
                    
                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    logger.error(f"❌ Error in signal broadcast: {e}")
                    self.stats['errors_count'] += 1
            
        except Exception as e:
            logger.error(f"❌ Error in signal broadcast: {e}")
    
    async def _pattern_broadcast(self):
        """Broadcast pattern updates"""
        try:
            while self.is_running:
                try:
                    # Get pattern from queue
                    pattern = await asyncio.wait_for(self.pattern_queue.get(), timeout=1.0)
                    
                    # Broadcast to subscribed clients
                    await self._broadcast_pattern(pattern)
                    
                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    logger.error(f"❌ Error in pattern broadcast: {e}")
                    self.stats['errors_count'] += 1
            
        except Exception as e:
            logger.error(f"❌ Error in pattern broadcast: {e}")
    
    async def _broadcast_market_data(self, data: Dict):
        """Broadcast market data to subscribed clients"""
        try:
            symbol = data['symbol']
            timeframe = data['timeframe']
            
            message = {
                'type': 'market_data_update',
                'symbol': symbol,
                'timeframe': timeframe,
                'data': data,
                'timestamp': datetime.now().isoformat()
            }
            
            await self._broadcast_message(message, symbol, timeframe)
            
        except Exception as e:
            logger.error(f"❌ Error broadcasting market data: {e}")
    
    async def _broadcast_signal(self, signal: Dict):
        """Broadcast signal to subscribed clients"""
        try:
            symbol = signal['symbol']
            timeframe = signal['timeframe']
            
            message = {
                'type': 'signal_update',
                'symbol': symbol,
                'timeframe': timeframe,
                'signal': signal,
                'timestamp': datetime.now().isoformat()
            }
            
            await self._broadcast_message(message, symbol, timeframe)
            
        except Exception as e:
            logger.error(f"❌ Error broadcasting signal: {e}")
    
    async def _broadcast_pattern(self, pattern: Dict):
        """Broadcast pattern to subscribed clients"""
        try:
            symbol = pattern['symbol']
            timeframe = pattern['timeframe']
            
            message = {
                'type': 'pattern_update',
                'symbol': symbol,
                'timeframe': timeframe,
                'pattern': pattern,
                'timestamp': datetime.now().isoformat()
            }
            
            await self._broadcast_message(message, symbol, timeframe)
            
        except Exception as e:
            logger.error(f"❌ Error broadcasting pattern: {e}")
    
    async def _broadcast_message(self, message: Dict, symbol: str, timeframe: str):
        """Broadcast message to subscribed clients"""
        try:
            start_time = time.time()
            message_json = json.dumps(message)
            
            # Find subscribed clients
            subscribed_clients = []
            for connection in self.active_connections.values():
                if (symbol in connection.subscribed_symbols and 
                    timeframe in connection.subscribed_timeframes):
                    subscribed_clients.append(connection)
            
            # Send to subscribed clients
            for connection in subscribed_clients:
                try:
                    await connection.websocket.send_text(message_json)
                    connection.last_activity = datetime.now()
                    connection.message_count += 1
                    self.stats['total_messages_sent'] += 1
                    
                except Exception as e:
                    logger.error(f"❌ Error sending to client {connection.client_id}: {e}")
                    # Mark connection for removal
                    await self._remove_connection(connection.client_id)
            
            # Record latency
            latency = (time.time() - start_time) * 1000
            self.latency_times.append(latency)
            self.message_times.append(time.time())
            
            # Update average latency
            if self.latency_times:
                self.stats['avg_latency_ms'] = sum(self.latency_times) / len(self.latency_times)
            
        except Exception as e:
            logger.error(f"❌ Error broadcasting message: {e}")
    
    async def _connection_monitoring(self):
        """Monitor and clean up inactive connections"""
        try:
            while self.is_running:
                try:
                    current_time = datetime.now()
                    inactive_connections = []
                    
                    # Check for inactive connections
                    for connection in self.active_connections.values():
                        if current_time - connection.last_activity > timedelta(minutes=5):
                            inactive_connections.append(connection.client_id)
                    
                    # Remove inactive connections
                    for client_id in inactive_connections:
                        logger.info(f"🔌 Removing inactive connection: {client_id}")
                        await self._remove_connection(client_id)
                    
                    # Update stats
                    self.stats['active_connections'] = len(self.active_connections)
                    
                    # Wait before next check
                    await asyncio.sleep(60)  # 1 minute
                    
                except Exception as e:
                    logger.error(f"❌ Error in connection monitoring: {e}")
                    await asyncio.sleep(60)
            
        except Exception as e:
            logger.error(f"❌ Error in connection monitoring: {e}")
    
    async def _performance_monitoring(self):
        """Monitor WebSocket service performance"""
        try:
            while self.is_running:
                try:
                    # Log performance metrics
                    logger.info(f"📊 WebSocket Performance - "
                              f"Active Connections: {self.stats['active_connections']}, "
                              f"Total Messages: {self.stats['total_messages_sent']}, "
                              f"Avg Latency: {self.stats['avg_latency_ms']:.2f}ms, "
                              f"Errors: {self.stats['errors_count']}")
                    
                    # Wait before next monitoring cycle
                    await asyncio.sleep(60)  # 1 minute
                    
                except Exception as e:
                    logger.error(f"❌ Error in performance monitoring: {e}")
                    await asyncio.sleep(60)
            
        except Exception as e:
            logger.error(f"❌ Error in performance monitoring: {e}")
    
    async def _remove_connection(self, client_id: str):
        """Remove WebSocket connection"""
        try:
            if client_id in self.active_connections:
                connection = self.active_connections[client_id]
                
                # Close WebSocket
                try:
                    await connection.websocket.close()
                except Exception:
                    pass
                
                # Remove from active connections
                del self.active_connections[client_id]
                self.stats['active_connections'] = len(self.active_connections)
                
                logger.info(f"🔌 Removed connection: {client_id}")
            
        except Exception as e:
            logger.error(f"❌ Error removing connection: {e}")
    
    async def _close_connection(self, connection: WebSocketConnection):
        """Close WebSocket connection"""
        try:
            await connection.websocket.close()
        except Exception:
            pass
    
    async def queue_market_data(self, data: Dict):
        """Queue market data for broadcast"""
        try:
            await self.market_data_queue.put(data)
        except Exception as e:
            logger.error(f"❌ Error queuing market data: {e}")
    
    async def queue_signal(self, signal: Dict):
        """Queue signal for broadcast"""
        try:
            await self.signal_queue.put(signal)
        except Exception as e:
            logger.error(f"❌ Error queuing signal: {e}")
    
    async def queue_pattern(self, pattern: Dict):
        """Queue pattern for broadcast"""
        try:
            await self.pattern_queue.put(pattern)
        except Exception as e:
            logger.error(f"❌ Error queuing pattern: {e}")
    
    async def get_service_stats(self) -> Dict[str, Any]:
        """Get WebSocket service statistics"""
        try:
            return {
                'service_running': self.is_running,
                'total_connections': self.stats['total_connections'],
                'active_connections': self.stats['active_connections'],
                'total_messages_sent': self.stats['total_messages_sent'],
                'avg_latency_ms': round(self.stats['avg_latency_ms'], 2),
                'errors_count': self.stats['errors_count'],
                'available_symbols': self.symbols,
                'available_timeframes': self.timeframes,
                'queue_sizes': {
                    'market_data': self.market_data_queue.qsize(),
                    'signals': self.signal_queue.qsize(),
                    'patterns': self.pattern_queue.qsize()
                }
            }
        except Exception as e:
            logger.error(f"❌ Error getting service stats: {e}")
            return {}
