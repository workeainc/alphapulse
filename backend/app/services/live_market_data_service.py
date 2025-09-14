"""
Live Market Data Service for AlphaPulse
Handles real-time market data collection and integration with exchanges
"""

import asyncio
import logging
import ccxt
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import asyncpg
from dataclasses import dataclass
import json
from collections import defaultdict, deque
import time

logger = logging.getLogger(__name__)

@dataclass
class LiveMarketData:
    """Live market data structure"""
    symbol: str
    price: float
    volume: float
    bid: float
    ask: float
    spread: float
    high_24h: float
    low_24h: float
    change_24h: float
    change_percent_24h: float
    market_cap: float
    circulating_supply: float
    total_supply: float
    max_supply: float
    timestamp: datetime

@dataclass
class OrderBookData:
    """Order book data structure"""
    symbol: str
    side: str  # 'bid' or 'ask'
    price: float
    volume: float
    order_count: Optional[int]
    timestamp: datetime

@dataclass
class TradeExecution:
    """Trade execution structure"""
    signal_id: str
    symbol: str
    side: str  # 'buy' or 'sell'
    order_type: str  # 'market', 'limit', 'stop'
    quantity: float
    price: float
    executed_price: Optional[float]
    status: str  # 'pending', 'filled', 'cancelled', 'rejected'
    exchange_order_id: Optional[str]
    exchange_trade_id: Optional[str]
    commission: Optional[float]
    commission_asset: Optional[str]
    executed_at: Optional[datetime]

class LiveMarketDataService:
    """
    Live market data service
    Handles real-time data collection and trade execution
    """
    
    def __init__(self, db_pool: asyncpg.Pool, exchange_credentials: Optional[Dict] = None):
        self.db_pool = db_pool
        self.exchange_credentials = exchange_credentials or {}
        
        # Initialize exchange connections
        self.exchanges = {}
        self.initialize_exchanges()
        
        # Data buffers
        self.market_data_buffer = defaultdict(lambda: deque(maxlen=1000))
        self.order_book_buffer = defaultdict(lambda: deque(maxlen=500))
        
        # Trading pairs
        self.trading_pairs = [
            'BTC/USDT', 'ETH/USDT', 'ADA/USDT', 'SOL/USDT', 
            'BNB/USDT', 'XRP/USDT', 'DOT/USDT', 'LINK/USDT'
        ]
        
        # Service state
        self.is_running = False
        self.collection_tasks = []
        self.websocket_connections = {}
        
        # Performance tracking
        self.performance_stats = {
            'data_points_collected': 0,
            'last_update': None,
            'avg_latency_ms': 0,
            'errors_count': 0
        }
        
        logger.info("Live Market Data Service initialized")
    
    def initialize_exchanges(self):
        """Initialize exchange connections"""
        try:
            logger.info("🔄 Initializing exchange connections...")
            
            # Initialize Binance (main exchange) - async version
            self.exchanges['binance'] = ccxt.binance({
                'apiKey': self.exchange_credentials.get('binance_api_key'),
                'secret': self.exchange_credentials.get('binance_secret'),
                'sandbox': self.exchange_credentials.get('sandbox', True),
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'spot'
                }
            })
            
            # Initialize Bybit (backup exchange) - async version
            self.exchanges['bybit'] = ccxt.bybit({
                'apiKey': self.exchange_credentials.get('bybit_api_key'),
                'secret': self.exchange_credentials.get('bybit_secret'),
                'sandbox': self.exchange_credentials.get('sandbox', True),
                'enableRateLimit': True
            })
            
            logger.info(f"✅ Exchange connections initialized: {list(self.exchanges.keys())}")
            
        except Exception as e:
            logger.error(f"❌ Error initializing exchanges: {e}")
            # Initialize with empty dict to prevent KeyError
            self.exchanges = {}
    
    async def start_data_collection(self):
        """Start live market data collection"""
        try:
            logger.info("🚀 Starting live market data collection...")
            self.is_running = True
            
            # Start market data collection
            market_data_task = asyncio.create_task(self._collect_market_data())
            self.collection_tasks.append(market_data_task)
            
            # Start order book collection
            order_book_task = asyncio.create_task(self._collect_order_book_data())
            self.collection_tasks.append(order_book_task)
            
            logger.info("✅ Live market data collection started")
            
        except Exception as e:
            logger.error(f"❌ Error starting data collection: {e}")
            self.is_running = False
    
    async def stop_data_collection(self):
        """Stop live market data collection"""
        try:
            logger.info("🛑 Stopping live market data collection...")
            self.is_running = False
            
            # Cancel all tasks
            for task in self.collection_tasks:
                task.cancel()
            
            # Wait for tasks to complete
            await asyncio.gather(*self.collection_tasks, return_exceptions=True)
            self.collection_tasks.clear()
            
            # Close exchange connections
            await self._close_exchanges()
            
            logger.info("✅ Live market data collection stopped")
            
        except Exception as e:
            logger.error(f"❌ Error stopping data collection: {e}")
    
    async def _close_exchanges(self):
        """Close exchange connections"""
        try:
            for exchange_name, exchange in self.exchanges.items():
                if hasattr(exchange, 'close'):
                    await exchange.close()
                logger.info(f"✅ Closed {exchange_name} connection")
        except Exception as e:
            logger.error(f"❌ Error closing exchanges: {e}")
    
    async def _collect_market_data(self):
        """Collect live market data"""
        while self.is_running:
            try:
                start_time = time.time()
                
                for symbol in self.trading_pairs:
                    # Get ticker data from Binance
                    ticker = await self._get_ticker_data(symbol)
                    if ticker:
                        # Create market data object
                        market_data = LiveMarketData(
                            symbol=symbol,
                            price=float(ticker['last']),
                            volume=float(ticker['baseVolume']),
                            bid=float(ticker['bid']),
                            ask=float(ticker['ask']),
                            spread=float(ticker['ask']) - float(ticker['bid']),
                            high_24h=float(ticker['high']),
                            low_24h=float(ticker['low']),
                            change_24h=float(ticker['change']),
                            change_percent_24h=float(ticker['percentage']),
                            market_cap=float(ticker.get('marketCap', 0)),
                            circulating_supply=float(ticker.get('circulatingSupply', 0)),
                            total_supply=float(ticker.get('totalSupply', 0)),
                            max_supply=float(ticker.get('maxSupply', 0)),
                            timestamp=datetime.now()
                        )
                        
                        # Store in buffer
                        self.market_data_buffer[symbol].append(market_data)
                        
                        # Store in database
                        await self._store_market_data(market_data)
                
                # Update performance stats
                latency = (time.time() - start_time) * 1000
                self.performance_stats['data_points_collected'] += len(self.trading_pairs)
                self.performance_stats['last_update'] = datetime.now()
                self.performance_stats['avg_latency_ms'] = (
                    (self.performance_stats['avg_latency_ms'] * 0.9) + (latency * 0.1)
                )
                
                # Wait before next collection
                await asyncio.sleep(5)  # 5-second intervals
                
            except Exception as e:
                logger.error(f"❌ Error collecting market data: {e}")
                self.performance_stats['errors_count'] += 1
                await asyncio.sleep(10)  # Wait longer on error
    
    async def _collect_order_book_data(self):
        """Collect order book data"""
        while self.is_running:
            try:
                for symbol in self.trading_pairs:
                    # Get order book from Binance
                    order_book = await self._get_order_book_data(symbol)
                    if order_book:
                        # Process bids
                        for bid in order_book['bids'][:10]:  # Top 10 bids
                            bid_data = OrderBookData(
                                symbol=symbol,
                                side='bid',
                                price=float(bid[0]),
                                volume=float(bid[1]),
                                order_count=None,
                                timestamp=datetime.now()
                            )
                            self.order_book_buffer[symbol].append(bid_data)
                            await self._store_order_book_data(bid_data)
                        
                        # Process asks
                        for ask in order_book['asks'][:10]:  # Top 10 asks
                            ask_data = OrderBookData(
                                symbol=symbol,
                                side='ask',
                                price=float(ask[0]),
                                volume=float(ask[1]),
                                order_count=None,
                                timestamp=datetime.now()
                            )
                            self.order_book_buffer[symbol].append(ask_data)
                            await self._store_order_book_data(ask_data)
                
                # Wait before next collection
                await asyncio.sleep(10)  # 10-second intervals
                
            except Exception as e:
                logger.error(f"❌ Error collecting order book data: {e}")
                await asyncio.sleep(15)  # Wait longer on error
    
    async def _get_ticker_data(self, symbol: str) -> Optional[Dict]:
        """Get ticker data from exchange"""
        try:
            if 'binance' not in self.exchanges:
                logger.error(f"Binance exchange not found. Available exchanges: {list(self.exchanges.keys())}")
                return None
            
            exchange = self.exchanges['binance']
            if not exchange:
                logger.error("Binance exchange object is None")
                return None
                
            # Use synchronous method (no await needed)
            ticker = exchange.fetch_ticker(symbol)
            return ticker
        except Exception as e:
            logger.error(f"Error getting ticker for {symbol}: {e}")
            return None
    
    async def _get_order_book_data(self, symbol: str) -> Optional[Dict]:
        """Get order book data from exchange"""
        try:
            if 'binance' not in self.exchanges:
                logger.error(f"Binance exchange not found. Available exchanges: {list(self.exchanges.keys())}")
                return None
            
            exchange = self.exchanges['binance']
            if not exchange:
                logger.error("Binance exchange object is None")
                return None
                
            # Use synchronous method (no await needed)
            order_book = exchange.fetch_order_book(symbol, limit=10)
            return order_book
        except Exception as e:
            logger.error(f"Error getting order book for {symbol}: {e}")
            return None
    
    async def _store_market_data(self, market_data: LiveMarketData):
        """Store market data in database"""
        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO live_market_data (
                        timestamp, symbol, price, volume, bid, ask, spread,
                        high_24h, low_24h, change_24h, change_percent_24h,
                        market_cap, circulating_supply, total_supply, max_supply
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15)
                """, market_data.timestamp, market_data.symbol, market_data.price,
                     market_data.volume, market_data.bid, market_data.ask, market_data.spread,
                     market_data.high_24h, market_data.low_24h, market_data.change_24h,
                     market_data.change_percent_24h, market_data.market_cap,
                     market_data.circulating_supply, market_data.total_supply, market_data.max_supply)
        except Exception as e:
            logger.error(f"Error storing market data: {e}")
    
    async def _store_order_book_data(self, order_book_data: OrderBookData):
        """Store order book data in database"""
        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO order_book_data (
                        timestamp, symbol, side, price, volume, order_count
                    ) VALUES ($1, $2, $3, $4, $5, $6)
                """, order_book_data.timestamp, order_book_data.symbol,
                     order_book_data.side, order_book_data.price,
                     order_book_data.volume, order_book_data.order_count)
        except Exception as e:
            logger.error(f"Error storing order book data: {e}")
    
    async def execute_trade(self, trade: TradeExecution) -> bool:
        """Execute a trade on the exchange"""
        try:
            logger.info(f"🔄 Executing trade: {trade.symbol} {trade.side} {trade.quantity}")
            
            # Execute trade on Binance
            exchange = self.exchanges['binance']
            
            if trade.order_type == 'market':
                order = exchange.create_market_order(
                    symbol=trade.symbol,
                    side=trade.side,
                    amount=trade.quantity
                )
            else:
                order = exchange.create_limit_order(
                    symbol=trade.symbol,
                    side=trade.side,
                    amount=trade.quantity,
                    price=trade.price
                )
            
            # Update trade execution with exchange data
            trade.exchange_order_id = order['id']
            trade.executed_price = float(order.get('price', trade.price))
            trade.status = order['status']
            trade.executed_at = datetime.now()
            
            # Store trade execution in database
            await self._store_trade_execution(trade)
            
            logger.info(f"✅ Trade executed successfully: {order['id']}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error executing trade: {e}")
            trade.status = 'rejected'
            trade.executed_at = datetime.now()
            await self._store_trade_execution(trade)
            return False
    
    async def _store_trade_execution(self, trade: TradeExecution):
        """Store trade execution in database"""
        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO trade_executions (
                        signal_id, symbol, side, order_type, quantity, price,
                        executed_price, status, exchange_order_id, exchange_trade_id,
                        commission, commission_asset, executed_at
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
                """, trade.signal_id, trade.symbol, trade.side, trade.order_type,
                     trade.quantity, trade.price, trade.executed_price, trade.status,
                     trade.exchange_order_id, trade.exchange_trade_id,
                     trade.commission, trade.commission_asset, trade.executed_at)
        except Exception as e:
            logger.error(f"Error storing trade execution: {e}")
    
    async def get_latest_market_data(self, symbol: str) -> Optional[LiveMarketData]:
        """Get latest market data for a symbol"""
        try:
            # First try to get from buffer
            if symbol in self.market_data_buffer and self.market_data_buffer[symbol]:
                return self.market_data_buffer[symbol][-1]
            
            # If buffer is empty, try to fetch directly from exchange
            logger.info(f"Buffer empty for {symbol}, fetching directly from exchange...")
            ticker = await self._get_ticker_data(symbol)
            if ticker:
                market_data = LiveMarketData(
                    symbol=symbol,
                    price=float(ticker['last']),
                    volume=float(ticker['baseVolume']),
                    bid=float(ticker['bid']),
                    ask=float(ticker['ask']),
                    spread=float(ticker['ask']) - float(ticker['bid']),
                    high_24h=float(ticker['high']),
                    low_24h=float(ticker['low']),
                    change_24h=float(ticker['change']),
                    change_percent_24h=float(ticker['percentage']),
                    market_cap=float(ticker.get('marketCap', 0)),
                    circulating_supply=float(ticker.get('circulatingSupply', 0)),
                    total_supply=float(ticker.get('totalSupply', 0)),
                    max_supply=float(ticker.get('maxSupply', 0)),
                    timestamp=datetime.now()
                )
                # Store in buffer for future use
                self.market_data_buffer[symbol].append(market_data)
                return market_data
            
            return None
        except Exception as e:
            logger.error(f"Error getting latest market data: {e}")
            return None
    
    async def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        return {
            **self.performance_stats,
            'buffer_sizes': {
                symbol: len(buffer) for symbol, buffer in self.market_data_buffer.items()
            },
            'active_exchanges': list(self.exchanges.keys())
        }
    
    async def validate_data_quality(self, symbol: str) -> Dict[str, Any]:
        """Validate data quality for a symbol"""
        try:
            latest_data = await self.get_latest_market_data(symbol)
            if not latest_data:
                return {'valid': False, 'reason': 'No data available'}
            
            # Check data freshness
            data_age = (datetime.now() - latest_data.timestamp).total_seconds()
            if data_age > 60:  # Data older than 1 minute
                return {'valid': False, 'reason': f'Data too old: {data_age:.1f}s'}
            
            # Check price validity
            if latest_data.price <= 0:
                return {'valid': False, 'reason': 'Invalid price'}
            
            # Check spread validity
            if latest_data.spread < 0:
                return {'valid': False, 'reason': 'Invalid spread'}
            
            return {
                'valid': True,
                'data_age_seconds': data_age,
                'price': latest_data.price,
                'spread': latest_data.spread,
                'volume': latest_data.volume
            }
            
        except Exception as e:
            logger.error(f"Error validating data quality: {e}")
            return {'valid': False, 'reason': str(e)}

# Global instance
live_market_data_service = None

async def get_live_market_data_service(db_pool: asyncpg.Pool, exchange_credentials: Optional[Dict] = None) -> LiveMarketDataService:
    """Get or create global live market data service instance"""
    global live_market_data_service
    if live_market_data_service is None:
        live_market_data_service = LiveMarketDataService(db_pool, exchange_credentials)
    return live_market_data_service
