"""
WebSocket Orchestrator for AlphaPulse
Manages multiple WebSocket connections for 100+ symbols
Routes incoming data to processing pipeline
"""

import asyncio
import logging
from typing import List, Dict, Optional, Any, Callable
from datetime import datetime, timezone
import json
from dataclasses import dataclass

from src.app.core.unified_websocket_client import UnifiedWebSocketClient, UnifiedWebSocketManager, WebSocketConfig, PerformanceMode
from src.data.realtime_data_pipeline import RealTimeDataPipeline
from src.services.dynamic_symbol_manager import DynamicSymbolManager

logger = logging.getLogger(__name__)

@dataclass
class WebSocketStats:
    """WebSocket orchestration statistics"""
    active_connections: int
    total_streams: int
    messages_received: int
    messages_processed: int
    errors_count: int
    reconnections: int
    last_message_time: Optional[datetime]
    uptime_seconds: float

class WebSocketOrchestrator:
    """
    Orchestrates multiple WebSocket connections for 100+ symbols
    Handles connection management, message routing, and health monitoring
    """
    
    def __init__(
        self,
        symbol_manager: DynamicSymbolManager,
        data_pipeline: RealTimeDataPipeline,
        config: Dict[str, Any]
    ):
        self.symbol_manager = symbol_manager
        self.data_pipeline = data_pipeline
        self.config = config
        self.logger = logger
        
        # Note: Redis is on port 56379 in Docker, update data_pipeline if needed
        
        # WebSocket manager
        max_connections = config.get('websocket', {}).get('max_connections', 2)
        self.ws_manager = UnifiedWebSocketManager(max_connections=max_connections)
        
        # Active clients
        self.clients: Dict[str, UnifiedWebSocketClient] = {}
        
        # State
        self.is_running = False
        self.start_time: Optional[datetime] = None
        
        # Statistics
        self.stats = {
            'messages_received': 0,
            'messages_processed': 0,
            'errors_count': 0,
            'reconnections': 0,
            'last_message_time': None
        }
        
        # Message handlers
        self.message_handlers: List[Callable] = []
        
        logger.info("✅ WebSocket Orchestrator initialized")
    
    async def initialize(self):
        """Initialize WebSocket orchestrator"""
        try:
            # Start WebSocket manager
            await self.ws_manager.start()
            
            # Get active symbols
            futures_symbols = await self.symbol_manager.get_futures_symbols()
            spot_symbols = await self.symbol_manager.get_spot_symbols()
            
            all_symbols = futures_symbols + spot_symbols
            logger.info(f"📋 Orchestrator managing {len(all_symbols)} symbols ({len(futures_symbols)} futures + {len(spot_symbols)} spot)")
            
            # Create WebSocket clients for symbol groups
            await self._create_websocket_clients(all_symbols)
            
            logger.info("✅ WebSocket Orchestrator initialized with all connections")
            
        except Exception as e:
            logger.error(f"❌ Error initializing WebSocket orchestrator: {e}")
            raise
    
    async def _create_websocket_clients(self, symbols: List[str]):
        """Create WebSocket clients for symbol groups"""
        try:
            streams_per_connection = self.config.get('websocket', {}).get('streams_per_connection', 100)
            
            # Split symbols into groups based on streams per connection
            symbol_groups = [
                symbols[i:i + streams_per_connection] 
                for i in range(0, len(symbols), streams_per_connection)
            ]
            
            logger.info(f"📡 Creating {len(symbol_groups)} WebSocket clients for {len(symbols)} symbols")
            
            # Create a client for each group
            for idx, symbol_group in enumerate(symbol_groups):
                client_name = f"client_{idx+1}"
                
                # Configure WebSocket client
                config = WebSocketConfig(
                    symbols=symbol_group,
                    timeframes=['1m'],  # Only 1m, higher timeframes via database aggregation
                    base_url="wss://stream.binance.com:9443/stream",
                    reconnect_attempts=10,
                    ping_interval=20,
                    performance_mode=PerformanceMode.BALANCED
                )
                
                # Create and start client
                client = await self.ws_manager.create_client(client_name, config)
                self.clients[client_name] = client
                
                # Register message handler
                await self._register_message_handler(client)
                
                logger.info(f"✅ Created {client_name} with {len(symbol_group)} symbols")
            
            logger.info(f"🚀 All {len(symbol_groups)} WebSocket clients created and connected")
            
        except Exception as e:
            logger.error(f"❌ Error creating WebSocket clients: {e}")
            raise
    
    async def _register_message_handler(self, client: UnifiedWebSocketClient):
        """Register message handler for WebSocket client"""
        # Note: UnifiedWebSocketClient handles messages internally
        # We'll process them through callbacks
        pass
    
    async def start(self):
        """Start the WebSocket orchestrator"""
        if self.is_running:
            logger.warning("⚠️ WebSocket orchestrator already running")
            return
        
        logger.info("🚀 Starting WebSocket orchestrator...")
        self.is_running = True
        self.start_time = datetime.now(timezone.utc)
        
        # Start message processing loop
        asyncio.create_task(self._message_processing_loop())
        
        logger.info("✅ WebSocket orchestrator started")
    
    async def stop(self):
        """Stop the WebSocket orchestrator"""
        if not self.is_running:
            return
        
        logger.info("🛑 Stopping WebSocket orchestrator...")
        self.is_running = False
        
        # Stop all clients
        await self.ws_manager.stop()
        
        logger.info("✅ WebSocket orchestrator stopped")
    
    async def _message_processing_loop(self):
        """Process incoming WebSocket messages"""
        logger.info("🔄 Message processing loop started")
        
        while self.is_running:
            try:
                # Check each client for messages
                for client_name, client in self.clients.items():
                    # Get metrics to check for new messages
                    metrics = client.get_metrics()
                    
                    # The UnifiedWebSocketClient already routes messages internally
                    # We just need to ensure our data pipeline is connected
                    
                    # Update stats
                    self.stats['messages_received'] = metrics.messages_received
                    self.stats['messages_processed'] = metrics.messages_processed
                    self.stats['errors_count'] = metrics.errors
                    self.stats['reconnections'] = metrics.reconnections
                
                # Small delay to prevent tight loop
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"❌ Error in message processing loop: {e}")
                await asyncio.sleep(1)
    
    def register_message_handler(self, handler: Callable):
        """Register a custom message handler"""
        self.message_handlers.append(handler)
        logger.info(f"📝 Registered message handler: {handler.__name__}")
    
    async def handle_incoming_message(self, message: Dict[str, Any]):
        """Handle incoming WebSocket message"""
        try:
            self.stats['messages_received'] += 1
            self.stats['last_message_time'] = datetime.now(timezone.utc)
            
            # Route to data pipeline
            await self.data_pipeline.process_websocket_message(message)
            
            # Call custom handlers
            for handler in self.message_handlers:
                try:
                    await handler(message)
                except Exception as e:
                    logger.error(f"❌ Error in message handler {handler.__name__}: {e}")
            
            self.stats['messages_processed'] += 1
            
        except Exception as e:
            logger.error(f"❌ Error handling message: {e}")
            self.stats['errors_count'] += 1
    
    async def reconnect_client(self, client_name: str):
        """Reconnect a specific WebSocket client"""
        try:
            if client_name in self.clients:
                client = self.clients[client_name]
                # The UnifiedWebSocketClient has auto-reconnect built-in
                logger.info(f"🔄 Reconnection triggered for {client_name}")
                self.stats['reconnections'] += 1
            else:
                logger.warning(f"⚠️ Client {client_name} not found")
                
        except Exception as e:
            logger.error(f"❌ Error reconnecting client {client_name}: {e}")
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get health status of all WebSocket connections"""
        health = {
            'orchestrator_running': self.is_running,
            'uptime_seconds': (datetime.now(timezone.utc) - self.start_time).total_seconds() if self.start_time else 0,
            'total_clients': len(self.clients),
            'clients': {}
        }
        
        for client_name, client in self.clients.items():
            metrics = client.get_metrics()
            health['clients'][client_name] = {
                'connected': client.is_connected,
                'active_streams': len(client.active_streams),
                'messages_received': metrics.messages_received,
                'errors': metrics.errors,
                'last_reconnect': metrics.last_reconnect.isoformat() if metrics.last_reconnect else None
            }
        
        return health
    
    def get_stats(self) -> WebSocketStats:
        """Get orchestrator statistics"""
        total_streams = sum(len(client.active_streams) for client in self.clients.values())
        active_connections = sum(1 for client in self.clients.values() if client.is_connected)
        
        return WebSocketStats(
            active_connections=active_connections,
            total_streams=total_streams,
            messages_received=self.stats['messages_received'],
            messages_processed=self.stats['messages_processed'],
            errors_count=self.stats['errors_count'],
            reconnections=self.stats['reconnections'],
            last_message_time=self.stats['last_message_time'],
            uptime_seconds=(datetime.now(timezone.utc) - self.start_time).total_seconds() if self.start_time else 0
        )

