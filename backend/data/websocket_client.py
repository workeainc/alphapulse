"""
WebSocket Client for Real-Time Market Data Streaming
AlphaPulse Trading Bot - Phase 1 Implementation
"""

import asyncio
import json
import logging
import websockets
from typing import Dict, List, Callable, Optional
from datetime import datetime
import pandas as pd

logger = logging.getLogger(__name__)

class BinanceWebSocketClient:
    """
    Real-time WebSocket client for Binance market data
    Handles multiple symbol subscriptions and candlestick streaming
    """
    
    def __init__(self):
        self.ws = None
        self.callbacks = {}
        self.subscriptions = {}
        self.is_connected = False
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 5
        self.base_url = "wss://stream.binance.com:9443/ws"
        
    async def connect(self):
        """Establish WebSocket connection to Binance"""
        try:
            self.ws = await websockets.connect(self.base_url)
            self.is_connected = True
            self.reconnect_attempts = 0
            logger.info("✅ WebSocket connected to Binance")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to connect to WebSocket: {e}")
            return False
    
    async def disconnect(self):
        """Close WebSocket connection"""
        if self.ws:
            await self.ws.close()
            self.is_connected = False
            logger.info("🔌 WebSocket disconnected")
    
    async def subscribe_candlesticks(self, symbol: str, interval: str, callback: Callable):
        """
        Subscribe to real-time candlestick data
        
        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            interval: Timeframe ('1m', '5m', '15m', '1h', '4h', '1d')
            callback: Function to handle incoming data
        """
        if not self.is_connected:
            await self.connect()
        
        stream_name = f"{symbol.lower()}@kline_{interval}"
        subscription_message = {
            "method": "SUBSCRIBE",
            "params": [stream_name],
            "id": len(self.subscriptions) + 1
        }
        
        try:
            await self.ws.send(json.dumps(subscription_message))
            self.subscriptions[stream_name] = {
                'symbol': symbol,
                'interval': interval,
                'callback': callback,
                'active': True
            }
            self.callbacks[stream_name] = callback
            logger.info(f"📡 Subscribed to {stream_name}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to subscribe to {stream_name}: {e}")
            return False
    
    async def unsubscribe_candlesticks(self, symbol: str, interval: str):
        """Unsubscribe from candlestick stream"""
        stream_name = f"{symbol.lower()}@kline_{interval}"
        if stream_name in self.subscriptions:
            subscription_message = {
                "method": "UNSUBSCRIBE",
                "params": [stream_name],
                "id": len(self.subscriptions) + 1
            }
            
            try:
                await self.ws.send(json.dumps(subscription_message))
                del self.subscriptions[stream_name]
                del self.callbacks[stream_name]
                logger.info(f"📡 Unsubscribed from {stream_name}")
                return True
            except Exception as e:
                logger.error(f"❌ Failed to unsubscribe from {stream_name}: {e}")
                return False
        return False
    
    async def handle_message(self, message: str):
        """Process incoming WebSocket messages"""
        try:
            data = json.loads(message)
            
            # Handle subscription confirmations
            if 'result' in data:
                logger.info(f"📡 Subscription confirmed: {data}")
                return
            
            # Handle candlestick data
            if 'k' in data:
                await self._process_candlestick(data)
            
        except json.JSONDecodeError as e:
            logger.error(f"❌ Failed to parse message: {e}")
        except Exception as e:
            logger.error(f"❌ Error processing message: {e}")
    
    async def _process_candlestick(self, data: Dict):
        """Process candlestick data and trigger callbacks"""
        try:
            kline = data['k']
            stream_name = f"{data['s'].lower()}@kline_{kline['i']}"
            
            if stream_name in self.callbacks:
                # Convert to standardized format
                candlestick_data = {
                    'symbol': data['s'],
                    'interval': kline['i'],
                    'open_time': datetime.fromtimestamp(kline['t'] / 1000),
                    'close_time': datetime.fromtimestamp(kline['T'] / 1000),
                    'open': float(kline['o']),
                    'high': float(kline['h']),
                    'low': float(kline['l']),
                    'close': float(kline['c']),
                    'volume': float(kline['v']),
                    'is_complete': kline['x']
                }
                
                # Trigger callback
                callback = self.callbacks[stream_name]
                await callback(candlestick_data)
                
        except Exception as e:
            logger.error(f"❌ Error processing candlestick: {e}")
    
    async def listen(self):
        """Main listening loop for WebSocket messages"""
        if not self.is_connected:
            await self.connect()
        
        try:
            async for message in self.ws:
                await self.handle_message(message)
        except websockets.exceptions.ConnectionClosed:
            logger.warning("⚠️ WebSocket connection closed, attempting to reconnect...")
            await self._reconnect()
        except Exception as e:
            logger.error(f"❌ Error in WebSocket listener: {e}")
            await self._reconnect()
    
    async def _reconnect(self):
        """Attempt to reconnect to WebSocket"""
        if self.reconnect_attempts < self.max_reconnect_attempts:
            self.reconnect_attempts += 1
            logger.info(f"🔄 Reconnection attempt {self.reconnect_attempts}/{self.max_reconnect_attempts}")
            
            await asyncio.sleep(2 ** self.reconnect_attempts)  # Exponential backoff
            
            if await self.connect():
                # Resubscribe to all active streams
                for stream_name, subscription in self.subscriptions.items():
                    if subscription['active']:
                        await self.subscribe_candlesticks(
                            subscription['symbol'],
                            subscription['interval'],
                            subscription['callback']
                        )
        else:
            logger.error("❌ Max reconnection attempts reached")
    
    async def get_all_subscriptions(self) -> List[Dict]:
        """Get list of all active subscriptions"""
        return [
            {
                'stream': stream_name,
                'symbol': sub['symbol'],
                'interval': sub['interval'],
                'active': sub['active']
            }
            for stream_name, sub in self.subscriptions.items()
        ]

class WebSocketManager:
    """
    High-level manager for multiple WebSocket connections
    Handles connection pooling and load balancing
    """
    
    def __init__(self):
        self.clients = {}
        self.running = False
        
    async def add_client(self, name: str, client: BinanceWebSocketClient):
        """Add a WebSocket client to the manager"""
        self.clients[name] = client
        logger.info(f"➕ Added WebSocket client: {name}")
    
    async def remove_client(self, name: str):
        """Remove a WebSocket client from the manager"""
        if name in self.clients:
            await self.clients[name].disconnect()
            del self.clients[name]
            logger.info(f"➖ Removed WebSocket client: {name}")
    
    async def start_all(self):
        """Start all WebSocket clients"""
        self.running = True
        tasks = []
        
        for name, client in self.clients.items():
            task = asyncio.create_task(client.listen())
            tasks.append(task)
            logger.info(f"🚀 Started WebSocket client: {name}")
        
        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            logger.error(f"❌ Error in WebSocket manager: {e}")
        finally:
            self.running = False
    
    async def stop_all(self):
        """Stop all WebSocket clients"""
        self.running = False
        for name, client in self.clients.items():
            await client.disconnect()
        logger.info("🛑 Stopped all WebSocket clients")

# Example usage and testing
async def example_callback(candlestick_data: Dict):
    """Example callback function for candlestick data"""
    print(f"🕯️ New candlestick: {candlestick_data['symbol']} {candlestick_data['interval']}")
    print(f"   Open: {candlestick_data['open']}, Close: {candlestick_data['close']}")
    print(f"   Volume: {candlestick_data['volume']}")

async def main():
    """Example main function for testing"""
    client = BinanceWebSocketClient()
    
    # Connect and subscribe
    await client.connect()
    await client.subscribe_candlesticks('BTCUSDT', '1m', example_callback)
    await client.subscribe_candlesticks('ETHUSDT', '5m', example_callback)
    
    # Start listening
    try:
        await client.listen()
    except KeyboardInterrupt:
        print("\n🛑 Stopping WebSocket client...")
        await client.disconnect()

if __name__ == "__main__":
    asyncio.run(main())
