#!/usr/bin/env python3
"""
Binance WebSocket Client for AlphaPulse
Real-time market data streaming from Binance WebSocket API
"""

import asyncio
import json
import logging
import time
from typing import Dict, List, Optional, Callable, AsyncGenerator
from datetime import datetime, timezone
import websockets
import aiohttp
from urllib.parse import urlencode

logger = logging.getLogger(__name__)

class BinanceWebSocketClient:
    """
    Real-time WebSocket client for Binance market data
    Handles multiple symbol subscriptions and candlestick streaming
    """
    
    def __init__(self, 
                 symbols: List[str] = None,
                 timeframes: List[str] = None,
                 base_url: str = "wss://stream.binance.com:9443/ws",
                 enable_liquidations: bool = True,
                 enable_orderbook: bool = True,
                 enable_trades: bool = True):
        """
        Initialize Binance WebSocket client
        
        Args:
            symbols: List of trading pairs (e.g., ["BTCUSDT", "ETHUSDT"])
            timeframes: List of timeframes (e.g., ["1m", "5m", "15m", "1h"])
            base_url: Binance WebSocket base URL
            enable_liquidations: Enable liquidation event streaming
            enable_orderbook: Enable order book depth streaming
            enable_trades: Enable trade data streaming
        """
        self.symbols = symbols or ["BTCUSDT", "ETHUSDT"]
        self.timeframes = timeframes or ["1m", "5m", "15m", "1h"]
        self.base_url = base_url
        self.enable_liquidations = enable_liquidations
        self.enable_orderbook = enable_orderbook
        self.enable_trades = enable_trades
        
        # WebSocket connection
        self.websocket = None
        self.is_connected = False
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 5
        self.reconnect_delay = 1  # Start with 1 second
        
        # Stream management
        self.active_streams = set()
        self.stream_callbacks = {}
        
        # Performance tracking
        self.messages_received = 0
        self.last_message_time = None
        self.connection_start_time = None
        
        # Heartbeat tracking
        self.last_heartbeat = None
        self.heartbeat_interval = 30  # 30 seconds
        
        # Real-time data buffers
        self.liquidation_buffer = []
        self.orderbook_buffer = {}
        self.trade_buffer = []
        
        logger.info(f"Binance WebSocket client initialized for {len(self.symbols)} symbols with enhanced real-time features")
    
    async def connect(self):
        """Establish WebSocket connection to Binance with enhanced streams"""
        try:
            # Build stream names for all symbols and timeframes
            stream_names = []
            
            # Add candlestick streams
            for symbol in self.symbols:
                for timeframe in self.timeframes:
                    stream_name = f"{symbol.lower()}@kline_{timeframe}"
                    stream_names.append(stream_name)
            
            # Add order book depth streams
            if self.enable_orderbook:
                for symbol in self.symbols:
                    stream_name = f"{symbol.lower()}@depth20@100ms"
                    stream_names.append(stream_name)
            
            # Add trade streams
            if self.enable_trades:
                for symbol in self.symbols:
                    stream_name = f"{symbol.lower()}@trade"
                    stream_names.append(stream_name)
            
            # Add liquidation streams (futures only)
            if self.enable_liquidations:
                for symbol in self.symbols:
                    # Check if symbol supports futures
                    if symbol.endswith('USDT'):
                        stream_name = f"{symbol.lower()}@forceOrder"
                        stream_names.append(stream_name)
            
            # Create combined stream URL
            combined_streams = "/".join(stream_names)
            ws_url = f"{self.base_url}/{combined_streams}"
            
            logger.info(f"Connecting to Binance WebSocket: {ws_url}")
            
            # Connect to WebSocket
            self.websocket = await websockets.connect(ws_url)
            self.is_connected = True
            self.reconnect_attempts = 0
            self.reconnect_delay = 1
            self.connection_start_time = time.time()
            self.last_heartbeat = time.time()
            
            # Store active streams
            self.active_streams = set(stream_names)
            
            logger.info(f"✅ Connected to Binance WebSocket with {len(stream_names)} enhanced streams")
            logger.info(f"📊 Streams: Candlesticks, OrderBook: {self.enable_orderbook}, Trades: {self.enable_trades}, Liquidations: {self.enable_liquidations}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to connect to Binance WebSocket: {e}")
            await self._handle_connection_error()
            return False
    
    async def disconnect(self):
        """Close WebSocket connection"""
        if self.websocket:
            await self.websocket.close()
            self.is_connected = False
            self.websocket = None
            logger.info("🔌 Disconnected from Binance WebSocket")
    
    async def listen(self) -> AsyncGenerator[dict, None]:
        """
        Listen to WebSocket messages and yield parsed data
        
        Yields:
            dict: Parsed candlestick data
        """
        if not self.is_connected:
            await self.connect()
        
        try:
            async for message in self.websocket:
                if not self.is_connected:
                    break
                
                try:
                    # Parse message
                    data = json.loads(message)
                    
                    # Update stats
                    self.messages_received += 1
                    self.last_message_time = time.time()
                    
                    # Handle different message types
                    if 'e' in data:  # Event message
                        if data['e'] == 'kline':  # Candlestick data
                            yield self._parse_kline_data(data)
                        elif data['e'] == '24hrTicker':  # 24hr ticker
                            yield self._parse_ticker_data(data)
                        elif data['e'] == 'trade':  # Trade data
                            trade_data = self._parse_trade_data(data)
                            if trade_data:
                                self.trade_buffer.append(trade_data)
                                # Keep only last 1000 trades
                                if len(self.trade_buffer) > 1000:
                                    self.trade_buffer = self.trade_buffer[-1000:]
                            yield trade_data
                        elif data['e'] == 'depthUpdate':  # Order book update
                            orderbook_data = self._parse_depth_data(data)
                            if orderbook_data:
                                symbol = orderbook_data.get('symbol')
                                if symbol:
                                    self.orderbook_buffer[symbol] = orderbook_data
                            yield orderbook_data
                        elif data['e'] == 'forceOrder':  # Liquidation event
                            liquidation_data = self._parse_liquidation_data(data)
                            if liquidation_data:
                                self.liquidation_buffer.append(liquidation_data)
                                # Keep only last 100 liquidations
                                if len(self.liquidation_buffer) > 100:
                                    self.liquidation_buffer = self.liquidation_buffer[-100:]
                            yield liquidation_data
                        else:
                            logger.debug(f"Unhandled event type: {data['e']}")
                    
                    # Handle ping/pong for connection health
                    elif 'ping' in data:
                        await self._send_pong()
                    
                    # Handle subscription confirmation
                    elif 'result' in data:
                        logger.info(f"Subscription result: {data}")
                    
                    # Handle error messages
                    elif 'error' in data:
                        logger.error(f"WebSocket error: {data['error']}")
                    
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse WebSocket message: {e}")
                except Exception as e:
                    logger.error(f"Error processing WebSocket message: {e}")
                
                # Check connection health
                await self._check_connection_health()
                
        except websockets.exceptions.ConnectionClosed:
            logger.warning("WebSocket connection closed")
            await self._handle_connection_error()
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
            await self._handle_connection_error()
    
    def _parse_kline_data(self, data: dict) -> dict:
        """Parse kline (candlestick) data with robust volume parsing"""
        try:
            kline = data['k']
            
            # Robust volume parsing with fallbacks
            def safe_float(value, default=0.0):
                """Safely convert to float with error handling"""
                try:
                    if isinstance(value, str):
                        return float(value) if value else default
                    return float(value) if value is not None else default
                except (ValueError, TypeError):
                    logger.warning(f"Failed to parse volume value: {value}, using default: {default}")
                    return default
            
            def safe_int(value, default=0):
                """Safely convert to int with error handling"""
                try:
                    if isinstance(value, str):
                        return int(float(value)) if value else default
                    return int(value) if value is not None else default
                except (ValueError, TypeError):
                    logger.warning(f"Failed to parse int value: {value}, using default: {default}")
                    return default
            
            return {
                'type': 'kline',
                'symbol': data['s'],
                'timeframe': kline['i'],
                'timestamp': datetime.fromtimestamp(kline['t'] / 1000, tz=timezone.utc),
                'open': safe_float(kline['o']),
                'high': safe_float(kline['h']),
                'low': safe_float(kline['l']),
                'close': safe_float(kline['c']),
                'volume': safe_float(kline['v']),
                'quote_volume': safe_float(kline['q']),
                'trades': safe_int(kline['n']),
                'is_complete': kline['x'],
                'quote_asset_volume': safe_float(kline['Q']),
                'taker_buy_base_volume': safe_float(kline['V']),
                'taker_buy_quote_volume': safe_float(kline['Q'])
            }
        except Exception as e:
            logger.error(f"Error parsing kline data: {e}")
            return None
    
    def _parse_ticker_data(self, data: dict) -> dict:
        """Parse 24hr ticker data"""
        try:
            return {
                'type': 'ticker',
                'symbol': data['s'],
                'price_change': float(data['p']),
                'price_change_percent': float(data['P']),
                'weighted_avg_price': float(data['w']),
                'prev_close_price': float(data['x']),
                'last_price': float(data['c']),
                'last_qty': float(data['Q']),
                'bid_price': float(data['b']),
                'ask_price': float(data['a']),
                'open_price': float(data['o']),
                'high_price': float(data['h']),
                'low_price': float(data['l']),
                'volume': float(data['v']),
                'quote_volume': float(data['q']),
                'open_time': datetime.fromtimestamp(data['O'] / 1000),
                'close_time': datetime.fromtimestamp(data['C'] / 1000),
                'first_id': int(data['F']),
                'last_id': int(data['L']),
                'count': int(data['n'])
            }
        except Exception as e:
            logger.error(f"Error parsing ticker data: {e}")
            return None
    
    def _parse_trade_data(self, data: dict) -> dict:
        """Parse trade data with proper field handling"""
        try:
            # Handle different trade data formats
            trade_data = {
                'type': 'trade',
                'symbol': data.get('s', ''),
                'trade_id': int(data.get('t', 0)),
                'price': float(data.get('p', 0)),
                'quantity': float(data.get('q', 0)),
                'trade_time': datetime.fromtimestamp(data.get('T', 0) / 1000),
                'is_buyer_maker': data.get('m', False),
                'ignore': data.get('M', False)
            }
            
            # Add optional fields if they exist
            if 'b' in data:
                trade_data['buyer_order_id'] = int(data['b'])
            if 'a' in data:
                trade_data['seller_order_id'] = int(data['a'])
            
            return trade_data
            
        except Exception as e:
            logger.error(f"Error parsing trade data: {e}")
            return None
    
    def _parse_depth_data(self, data: dict) -> dict:
        """Parse order book depth data with enhanced metrics"""
        try:
            bids = [[float(price), float(qty)] for price, qty in data['b']]
            asks = [[float(price), float(qty)] for price, qty in data['a']]
            
            # Calculate order book metrics
            best_bid = bids[0][0] if bids else 0
            best_ask = asks[0][0] if asks else 0
            spread = best_ask - best_bid if best_bid and best_ask else 0
            spread_percentage = (spread / best_bid * 100) if best_bid else 0
            
            # Calculate volume metrics
            total_bid_volume = sum(bid[1] for bid in bids)
            total_ask_volume = sum(ask[1] for ask in asks)
            volume_imbalance = (total_bid_volume - total_ask_volume) / (total_bid_volume + total_ask_volume) if (total_bid_volume + total_ask_volume) > 0 else 0
            
            return {
                'type': 'depth',
                'symbol': data['s'],
                'event_time': datetime.fromtimestamp(data['E'] / 1000),
                'first_update_id': int(data['U']),
                'final_update_id': int(data['u']),
                'bids': bids,
                'asks': asks,
                'best_bid': best_bid,
                'best_ask': best_ask,
                'spread': spread,
                'spread_percentage': spread_percentage,
                'total_bid_volume': total_bid_volume,
                'total_ask_volume': total_ask_volume,
                'volume_imbalance': volume_imbalance,
                'bid_count': len(bids),
                'ask_count': len(asks)
            }
        except Exception as e:
            logger.error(f"Error parsing depth data: {e}")
            return None
    
    def _parse_liquidation_data(self, data: dict) -> dict:
        """Parse liquidation event data"""
        try:
            order = data['o']
            return {
                'type': 'liquidation',
                'symbol': order['s'],
                'timestamp': datetime.fromtimestamp(order['T'] / 1000, tz=timezone.utc),
                'side': order['S'],  # BUY or SELL
                'order_type': order['o'],  # LIMIT, MARKET, etc.
                'time_in_force': order['f'],  # GTC, IOC, FOK
                'quantity': float(order['q']),
                'price': float(order['p']),
                'average_price': float(order['ap']),
                'order_status': order['X'],  # NEW, FILLED, etc.
                'last_filled_quantity': float(order['z']),
                'filled_accumulated_quantity': float(order['z']),
                'trade_time': datetime.fromtimestamp(order['T'] / 1000),
                'trade_id': order['t'],
                'is_buyer_maker': order['m'],
                'is_isolated': order['V']
            }
        except Exception as e:
            logger.error(f"Error parsing liquidation data: {e}")
            return None
    
    def get_recent_liquidations(self, limit: int = 10) -> List[dict]:
        """Get recent liquidation events"""
        return self.liquidation_buffer[-limit:] if self.liquidation_buffer else []
    
    def get_recent_trades(self, limit: int = 100) -> List[dict]:
        """Get recent trade events"""
        return self.trade_buffer[-limit:] if self.trade_buffer else []
    
    def get_orderbook_snapshot(self, symbol: str) -> Optional[dict]:
        """Get current order book snapshot for a symbol"""
        return self.orderbook_buffer.get(symbol)
    
    def get_all_orderbook_snapshots(self) -> Dict[str, dict]:
        """Get all current order book snapshots"""
        return self.orderbook_buffer.copy()
    
    def get_real_time_stats(self) -> dict:
        """Get real-time statistics"""
        return {
            'messages_received': self.messages_received,
            'is_connected': self.is_connected,
            'active_streams': len(self.active_streams),
            'liquidation_count': len(self.liquidation_buffer),
            'trade_count': len(self.trade_buffer),
            'orderbook_symbols': len(self.orderbook_buffer),
            'connection_uptime': time.time() - self.connection_start_time if self.connection_start_time else 0,
            'last_message_time': self.last_message_time
        }

    async def _send_pong(self):
        """Send pong response to ping"""
        try:
            pong_message = json.dumps({"pong": int(time.time() * 1000)})
            await self.websocket.send(pong_message)
            self.last_heartbeat = time.time()
        except Exception as e:
            logger.error(f"Error sending pong: {e}")
    
    async def _check_connection_health(self):
        """Check connection health and reconnect if needed"""
        current_time = time.time()
        
        # Check heartbeat
        if self.last_heartbeat and (current_time - self.last_heartbeat) > self.heartbeat_interval:
            logger.warning("No heartbeat received, checking connection...")
            try:
                await self.websocket.ping()
                self.last_heartbeat = current_time
            except Exception as e:
                logger.error(f"Connection health check failed: {e}")
                await self._handle_connection_error()
    
    async def _handle_connection_error(self):
        """Handle connection errors and attempt reconnection"""
        self.is_connected = False
        
        if self.reconnect_attempts < self.max_reconnect_attempts:
            self.reconnect_attempts += 1
            logger.info(f"Attempting to reconnect ({self.reconnect_attempts}/{self.max_reconnect_attempts})...")
            
            # Exponential backoff
            await asyncio.sleep(self.reconnect_delay)
            self.reconnect_delay = min(self.reconnect_delay * 2, 60)  # Max 60 seconds
            
            # Attempt reconnection
            await self.connect()
        else:
            logger.error("Max reconnection attempts reached")
            raise ConnectionError("Failed to reconnect to Binance WebSocket")
    
    async def subscribe_to_stream(self, stream_name: str, callback: Callable = None):
        """Subscribe to a specific stream"""
        try:
            if not self.is_connected:
                await self.connect()
            
            # Add to active streams
            self.active_streams.add(stream_name)
            
            if callback:
                self.stream_callbacks[stream_name] = callback
            
            logger.info(f"Subscribed to stream: {stream_name}")
            
        except Exception as e:
            logger.error(f"Error subscribing to stream {stream_name}: {e}")
    
    async def unsubscribe_from_stream(self, stream_name: str):
        """Unsubscribe from a specific stream"""
        try:
            if stream_name in self.active_streams:
                self.active_streams.remove(stream_name)
            
            if stream_name in self.stream_callbacks:
                del self.stream_callbacks[stream_name]
            
            logger.info(f"Unsubscribed from stream: {stream_name}")
            
        except Exception as e:
            logger.error(f"Error unsubscribing from stream {stream_name}: {e}")
    
    def get_connection_stats(self) -> dict:
        """Get connection statistics"""
        return {
            'is_connected': self.is_connected,
            'messages_received': self.messages_received,
            'last_message_time': self.last_message_time,
            'connection_start_time': self.connection_start_time,
            'active_streams': len(self.active_streams),
            'reconnect_attempts': self.reconnect_attempts
        }
    
    def get_active_streams(self) -> List[str]:
        """Get list of active streams"""
        return list(self.active_streams)


class BinanceWebSocketManager:
    """
    Manager for multiple Binance WebSocket connections
    Handles connection pooling and load balancing
    """
    
    def __init__(self, max_connections: int = 5):
        """
        Initialize WebSocket manager
        
        Args:
            max_connections: Maximum number of concurrent connections
        """
        self.max_connections = max_connections
        self.connections = []
        self.connection_pool = asyncio.Queue(maxsize=max_connections)
        self.is_running = False
        
        logger.info(f"Binance WebSocket Manager initialized with {max_connections} max connections")
    
    async def start(self):
        """Start the WebSocket manager"""
        self.is_running = True
        logger.info("Starting Binance WebSocket Manager...")
    
    async def stop(self):
        """Stop the WebSocket manager"""
        self.is_running = False
        
        # Close all connections
        for connection in self.connections:
            await connection.disconnect()
        
        logger.info("Binance WebSocket Manager stopped")
    
    async def get_connection(self) -> BinanceWebSocketClient:
        """Get an available WebSocket connection"""
        try:
            # Try to get from pool
            connection = await asyncio.wait_for(self.connection_pool.get(), timeout=5.0)
            return connection
        except asyncio.TimeoutError:
            # Create new connection if pool is empty
            if len(self.connections) < self.max_connections:
                connection = BinanceWebSocketClient()
                await connection.connect()
                self.connections.append(connection)
                return connection
            else:
                raise Exception("No available WebSocket connections")
    
    async def return_connection(self, connection: BinanceWebSocketClient):
        """Return a connection to the pool"""
        try:
            await self.connection_pool.put(connection)
        except asyncio.QueueFull:
            # Pool is full, close the connection
            await connection.disconnect()
            if connection in self.connections:
                self.connections.remove(connection)


# Example usage
async def main():
    """Example usage of Binance WebSocket client"""
    # Initialize client
    client = BinanceWebSocketClient(
        symbols=["BTCUSDT", "ETHUSDT"],
        timeframes=["1m", "5m"]
    )
    
    try:
        # Connect and listen
        await client.connect()
        
        async for message in client.listen():
            if message and message['type'] == 'kline':
                print(f"📊 {message['symbol']} {message['timeframe']}: "
                      f"O:{message['open']:.2f} H:{message['high']:.2f} "
                      f"L:{message['low']:.2f} C:{message['close']:.2f} "
                      f"V:{message['volume']:.2f}")
            
            # Process for 60 seconds
            if time.time() - client.connection_start_time > 60:
                break
                
    except KeyboardInterrupt:
        print("\n🛑 Stopping WebSocket client...")
    finally:
        await client.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
