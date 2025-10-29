"""
Enhanced CCXT Integration Service for AlphaPulse
Provides unified access to multiple exchanges for real-time market data
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Tuple, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass
from collections import defaultdict, deque
import json

# Import CCXT if available
try:
    import ccxt.async_support as ccxt
    CCXT_AVAILABLE = True
except ImportError:
    CCXT_AVAILABLE = False
    logging.warning("CCXT not available - using mock data")

logger = logging.getLogger(__name__)

@dataclass
class OrderBookSnapshot:
    """Order book snapshot data"""
    symbol: str
    timestamp: datetime
    exchange: str
    bids: List[List[float]]  # [price, volume] pairs
    asks: List[List[float]]  # [price, volume] pairs
    spread: float
    total_bid_volume: float
    total_ask_volume: float
    depth_levels: int

@dataclass
class LiquidationEvent:
    """Liquidation event data"""
    symbol: str
    timestamp: datetime
    exchange: str
    side: str  # 'long' or 'short'
    price: float
    quantity: float
    quote_quantity: float
    liquidation_type: str  # 'partial' or 'full'

@dataclass
class MarketDataTick:
    """Real-time market data tick"""
    symbol: str
    timestamp: datetime
    price: float
    volume: float
    bid: Optional[float]
    ask: Optional[float]
    bid_volume: Optional[float]
    ask_volume: Optional[float]
    exchange: str
    data_type: str  # 'tick', 'order_book', 'trade'

@dataclass
class OnChainEvent:
    """On-chain blockchain event data"""
    timestamp: datetime
    chain: str
    tx_hash: str
    from_address: Optional[str]
    to_address: Optional[str]
    value: Optional[float]
    gas_used: Optional[int]
    event_type: str
    symbol: Optional[str]
    block_number: Optional[int]
    metadata: Dict[str, Any]

@dataclass
class FundingRate:
    """Funding rate data for perpetual futures"""
    symbol: str
    exchange: str
    funding_rate: float
    timestamp: datetime
    next_funding_time: Optional[datetime] = None
    estimated_rate: Optional[float] = None
    metadata: Dict[str, Any] = None

@dataclass
class OpenInterest:
    """Open interest data for futures"""
    symbol: str
    exchange: str
    open_interest: float
    open_interest_value: float  # In quote currency
    timestamp: datetime
    metadata: Dict[str, Any] = None

@dataclass
class OrderBookDelta:
    """Order book delta update for WebSocket streaming"""
    symbol: str
    timestamp: datetime
    exchange: str
    bids_delta: List[List[float]]  # [price, volume] pairs to update
    asks_delta: List[List[float]]  # [price, volume] pairs to update
    sequence_number: Optional[int] = None
    metadata: Dict[str, Any] = None

@dataclass
class LiquidationLevel:
    """Liquidation level data for risk management"""
    symbol: str
    exchange: str
    price_level: float
    side: str  # 'long' or 'short'
    quantity: float
    timestamp: datetime
    metadata: Dict[str, Any] = None

class CCXTIntegrationService:
    """Enhanced CCXT integration service for real-time data"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logger
        
        # Exchange configuration
        self.exchanges: Dict[str, Any] = {}
        self.exchange_configs = self.config.get('exchanges', {
            'binance': {
                'apiKey': '',
                'secret': '',
                'sandbox': True,
                'options': {'defaultType': 'spot'}
            },
            'okx': {
                'apiKey': '',
                'secret': '',
                'password': '',
                'sandbox': True
            },
            'bybit': {
                'apiKey': '',
                'secret': '',
                'sandbox': True
            }
        })
        
        # Data configuration
        self.symbols = self.config.get('symbols', ['BTC/USDT', 'ETH/USDT', 'ADA/USDT'])
        self.timeframes = self.config.get('timeframes', ['1m', '5m', '15m', '1h'])
        self.order_book_depth = self.config.get('order_book_depth', 100)
        
        # Rate limiting
        self.rate_limits = defaultdict(lambda: {'last_call': 0, 'call_count': 0})
        self.max_calls_per_minute = self.config.get('max_calls_per_minute', 1200)
        
        # Data storage
        self.order_book_cache = defaultdict(lambda: defaultdict(dict))  # exchange -> symbol -> order_book
        self.liquidation_cache = defaultdict(list)  # symbol -> liquidation_events
        self.market_data_cache = defaultdict(lambda: deque(maxlen=1000))  # symbol -> market_data
        
        # Enhanced data storage for futures and WebSocket
        self.funding_rate_cache = defaultdict(lambda: deque(maxlen=100))  # symbol -> funding_rates
        self.open_interest_cache = defaultdict(lambda: deque(maxlen=100))  # symbol -> open_interest
        self.liquidation_levels_cache = defaultdict(list)  # symbol -> liquidation_levels
        self.order_book_delta_cache = defaultdict(lambda: deque(maxlen=1000))  # symbol -> deltas
        
        # WebSocket connections
        self.websocket_connections = {}  # exchange -> websocket_connection
        self.websocket_enabled = self.config.get('websocket_enabled', True)
        
        # Delta processing
        self.delta_processing_enabled = self.config.get('delta_processing_enabled', True)
        self.sequence_numbers = defaultdict(int)  # symbol -> sequence_number
        
        # Callbacks
        self.data_callbacks = defaultdict(list)  # event_type -> [callback]
        
        # Performance tracking
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'rate_limit_hits': 0,
            'last_request_time': None
        }
        
        # Initialize exchanges
        self._initialize_exchanges()
    
    def _initialize_exchanges(self):
        """Initialize exchange connections"""
        if not CCXT_AVAILABLE:
            self.logger.warning("CCXT not available - using mock mode")
            return
        
        try:
            for exchange_name, config in self.exchange_configs.items():
                if hasattr(ccxt, exchange_name):
                    exchange_class = getattr(ccxt, exchange_name)
                    self.exchanges[exchange_name] = exchange_class(config)
                    self.logger.info(f"Initialized {exchange_name} exchange")
                else:
                    self.logger.warning(f"Exchange {exchange_name} not supported by CCXT")
                    
        except Exception as e:
            self.logger.error(f"Failed to initialize exchanges: {e}")
    
    async def initialize(self):
        """Initialize the service"""
        try:
            self.logger.info("Initializing CCXT Integration Service...")
            
            # Test exchange connections
            for exchange_name, exchange in self.exchanges.items():
                try:
                    await exchange.load_markets()
                    self.logger.info(f"✅ {exchange_name} markets loaded successfully")
                except Exception as e:
                    self.logger.error(f"❌ Failed to load {exchange_name} markets: {e}")
            
            self.logger.info("CCXT Integration Service initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize CCXT Integration Service: {e}")
            raise
    
    async def _check_rate_limit(self, exchange_name: str) -> bool:
        """Check if we can make a request to the exchange"""
        current_time = time.time()
        rate_limit = self.rate_limits[exchange_name]
        
        # Reset counter if minute has passed
        if current_time - rate_limit['last_call'] >= 60:
            rate_limit['call_count'] = 0
            rate_limit['last_call'] = current_time
        
        # Check if we're under the limit
        if rate_limit['call_count'] < self.max_calls_per_minute:
            rate_limit['call_count'] += 1
            return True
        
        self.stats['rate_limit_hits'] += 1
        return False
    
    async def _wait_for_rate_limit(self, exchange_name: str):
        """Wait until we can make a request"""
        while not await self._check_rate_limit(exchange_name):
            wait_time = 60 - (time.time() - self.rate_limits[exchange_name]['last_call'])
            if wait_time > 0:
                await asyncio.sleep(min(wait_time, 1.0))
    
    async def get_order_book(self, symbol: str, exchange_name: str = 'binance') -> Optional[OrderBookSnapshot]:
        """Get order book from exchange"""
        if not CCXT_AVAILABLE:
            return self._get_mock_order_book(symbol, exchange_name)
        
        try:
            exchange = self.exchanges.get(exchange_name)
            if not exchange:
                self.logger.error(f"Exchange {exchange_name} not available")
                return None
            
            # Check rate limit
            await self._wait_for_rate_limit(exchange_name)
            
            # Fetch order book
            order_book = await exchange.fetch_order_book(symbol, self.order_book_depth)
            
            # Process order book
            bids = order_book['bids'][:self.order_book_depth]
            asks = order_book['asks'][:self.order_book_depth]
            
            # Calculate metrics
            spread = asks[0][0] - bids[0][0] if asks and bids else 0
            total_bid_volume = sum(bid[1] for bid in bids)
            total_ask_volume = sum(ask[1] for ask in asks)
            
            # Create snapshot
            snapshot = OrderBookSnapshot(
                symbol=symbol,
                timestamp=datetime.fromtimestamp(order_book['timestamp'] / 1000),
                exchange=exchange_name,
                bids=bids,
                asks=asks,
                spread=spread,
                total_bid_volume=total_bid_volume,
                total_ask_volume=total_ask_volume,
                depth_levels=len(bids) + len(asks)
            )
            
            # Cache the order book
            self.order_book_cache[exchange_name][symbol] = snapshot
            
            # Update statistics
            self.stats['total_requests'] += 1
            self.stats['successful_requests'] += 1
            self.stats['last_request_time'] = time.time()
            
            # Trigger callbacks
            await self._trigger_callbacks('order_book', snapshot)
            
            return snapshot
            
        except Exception as e:
            self.logger.error(f"Error fetching order book for {symbol} from {exchange_name}: {e}")
            self.stats['total_requests'] += 1
            self.stats['failed_requests'] += 1
            return None
    
    async def get_liquidation_events(self, symbol: str, exchange_name: str = 'binance') -> List[LiquidationEvent]:
        """Get liquidation events from exchange (if supported)"""
        if not CCXT_AVAILABLE:
            return self._get_mock_liquidation_events(symbol, exchange_name)
        
        try:
            exchange = self.exchanges.get(exchange_name)
            if not exchange:
                self.logger.error(f"Exchange {exchange_name} not available")
                return []
            
            # Check rate limit
            await self._wait_for_rate_limit(exchange_name)
            
            # Try to fetch liquidations (not all exchanges support this)
            try:
                if hasattr(exchange, 'fetch_liquidations'):
                    liquidations = await exchange.fetch_liquidations(symbol)
                    
                    events = []
                    for liq in liquidations:
                        event = LiquidationEvent(
                            symbol=symbol,
                            timestamp=datetime.fromtimestamp(liq['timestamp'] / 1000),
                            exchange=exchange_name,
                            side=liq.get('side', 'unknown'),
                            price=liq['price'],
                            quantity=liq['amount'],
                            quote_quantity=liq['price'] * liq['amount'],
                            liquidation_type=liq.get('type', 'full')
                        )
                        events.append(event)
                    
                    # Cache liquidation events
                    self.liquidation_cache[symbol].extend(events)
                    
                    # Update statistics
                    self.stats['total_requests'] += 1
                    self.stats['successful_requests'] += 1
                    
                    # Trigger callbacks
                    await self._trigger_callbacks('liquidation', events)
                    
                    return events
                    
                else:
                    self.logger.debug(f"Exchange {exchange_name} does not support liquidation fetching")
                    return []
                    
            except Exception as e:
                self.logger.debug(f"Liquidation fetching not supported for {exchange_name}: {e}")
                return []
                
        except Exception as e:
            self.logger.error(f"Error fetching liquidations for {symbol} from {exchange_name}: {e}")
            self.stats['total_requests'] += 1
            self.stats['failed_requests'] += 1
            return []
    
    async def get_market_data(self, symbol: str, exchange_name: str = 'binance') -> Optional[MarketDataTick]:
        """Get current market data from exchange"""
        if not CCXT_AVAILABLE:
            return self._get_mock_market_data(symbol, exchange_name)
        
        try:
            exchange = self.exchanges.get(exchange_name)
            if not exchange:
                self.logger.error(f"Exchange {exchange_name} not available")
                return None
            
            # Check rate limit
            await self._wait_for_rate_limit(exchange_name)
            
            # Fetch ticker
            ticker = await exchange.fetch_ticker(symbol)
            
            # Create market data tick
            tick = MarketDataTick(
                symbol=symbol,
                timestamp=datetime.fromtimestamp(ticker['timestamp'] / 1000),
                price=ticker['last'],
                volume=ticker['baseVolume'],
                bid=ticker.get('bid'),
                ask=ticker.get('ask'),
                bid_volume=ticker.get('bidVolume'),
                ask_volume=ticker.get('askVolume'),
                exchange=exchange_name,
                data_type='tick'
            )
            
            # Cache market data
            self.market_data_cache[symbol].append(tick)
            
            # Update statistics
            self.stats['total_requests'] += 1
            self.stats['successful_requests'] += 1
            self.stats['last_request_time'] = time.time()
            
            # Trigger callbacks
            await self._trigger_callbacks('market_data', tick)
            
            return tick
            
        except Exception as e:
            self.logger.error(f"Error fetching market data for {symbol} from {exchange_name}: {e}")
            self.stats['total_requests'] += 1
            self.stats['failed_requests'] += 1
            return None
    
    async def get_ohlcv(self, symbol: str, timeframe: str = '1m', 
                        limit: int = 100, exchange_name: str = 'binance') -> List[Dict[str, Any]]:
        """Get OHLCV data from exchange"""
        if not CCXT_AVAILABLE:
            return self._get_mock_ohlcv(symbol, timeframe, limit, exchange_name)
        
        try:
            exchange = self.exchanges.get(exchange_name)
            if not exchange:
                self.logger.error(f"Exchange {exchange_name} not available")
                return []
            
            # Check rate limit
            await self._wait_for_rate_limit(exchange_name)
            
            # Fetch OHLCV
            ohlcv = await exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            
            # Convert to standard format
            data = []
            for candle in ohlcv:
                data.append({
                    'timestamp': datetime.fromtimestamp(candle[0] / 1000),
                    'open': candle[1],
                    'high': candle[2],
                    'low': candle[3],
                    'close': candle[4],
                    'volume': candle[5]
                })
            
            # Update statistics
            self.stats['total_requests'] += 1
            self.stats['successful_requests'] += 1
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error fetching OHLCV for {symbol} from {exchange_name}: {e}")
            self.stats['total_requests'] += 1
            self.stats['failed_requests'] += 1
            return []
    
    async def get_on_chain_events(self, chain: str = 'ethereum', symbol: str = None, 
                                 limit: int = 100) -> List[OnChainEvent]:
        """Get on-chain events from blockchain APIs"""
        try:
            # For now, use mock data since we don't have API keys configured
            # In production, this would integrate with Alchemy, Etherscan, or similar APIs
            return self._get_mock_on_chain_events(chain, symbol, limit)
            
        except Exception as e:
            self.logger.error(f"Error fetching on-chain events for {chain}: {e}")
            return []
    
    async def get_on_chain_events_alchemy(self, chain: str = 'ethereum', symbol: str = None, 
                                        limit: int = 100) -> List[OnChainEvent]:
        """Get on-chain events using Alchemy API (requires API key)"""
        try:
            # This is a placeholder for Alchemy API integration
            # You would need to add your Alchemy API key to the config
            alchemy_api_key = self.config.get('alchemy_api_key')
            if not alchemy_api_key:
                self.logger.warning("Alchemy API key not configured, using mock data")
                return self._get_mock_on_chain_events(chain, symbol, limit)
            
            # Alchemy API endpoint
            url = f'https://eth-mainnet.g.alchemy.com/v2/{alchemy_api_key}'
            
            # For now, return mock data
            # TODO: Implement actual Alchemy API calls
            return self._get_mock_on_chain_events(chain, symbol, limit)
            
        except Exception as e:
            self.logger.error(f"Error fetching on-chain events from Alchemy: {e}")
            return []
    
    def add_callback(self, event_type: str, callback: Callable):
        """Add callback for data events"""
        self.data_callbacks[event_type].append(callback)
        self.logger.info(f"Added callback for {event_type} events")
    
    async def _trigger_callbacks(self, event_type: str, data: Any):
        """Trigger callbacks for data events"""
        callbacks = self.data_callbacks.get(event_type, [])
        
        for callback in callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(data)
                else:
                    callback(data)
            except Exception as e:
                self.logger.error(f"Error in {event_type} callback: {e}")
    
    def get_cached_order_book(self, symbol: str, exchange_name: str = 'binance') -> Optional[OrderBookSnapshot]:
        """Get cached order book"""
        return self.order_book_cache.get(exchange_name, {}).get(symbol)
    
    def get_cached_liquidation_events(self, symbol: str) -> List[LiquidationEvent]:
        """Get cached liquidation events"""
        return self.liquidation_cache.get(symbol, [])
    
    def get_cached_market_data(self, symbol: str) -> List[MarketDataTick]:
        """Get cached market data"""
        return list(self.market_data_cache.get(symbol, []))
    
    def get_service_statistics(self) -> Dict[str, Any]:
        """Get service statistics"""
        return {
            'stats': self.stats,
            'exchanges': list(self.exchanges.keys()),
            'symbols': self.symbols,
            'cache_sizes': {
                'order_books': sum(len(symbols) for symbols in self.order_book_cache.values()),
                'liquidation_events': sum(len(events) for events in self.liquidation_cache.values()),
                'market_data': sum(len(data) for data in self.market_data_cache.values())
            }
        }
    
    # Mock data methods for testing
    def _get_mock_order_book(self, symbol: str, exchange_name: str) -> OrderBookSnapshot:
        """Generate mock order book data"""
        import random
        
        base_price = 50000.0 if 'BTC' in symbol else 3000.0 if 'ETH' in symbol else 1.0
        
        bids = [[base_price * (1 - i * 0.001), random.uniform(0.1, 10.0)] for i in range(50)]
        asks = [[base_price * (1 + i * 0.001), random.uniform(0.1, 10.0)] for i in range(50)]
        
        return OrderBookSnapshot(
            symbol=symbol,
            timestamp=datetime.now(),
            exchange=exchange_name,
            bids=bids,
            asks=asks,
            spread=base_price * 0.002,
            total_bid_volume=sum(bid[1] for bid in bids),
            total_ask_volume=sum(ask[1] for ask in asks),
            depth_levels=100
        )
    
    def _get_mock_liquidation_events(self, symbol: str, exchange_name: str) -> List[LiquidationEvent]:
        """Generate mock liquidation events"""
        import random
        
        events = []
        base_price = 50000.0 if 'BTC' in symbol else 3000.0 if 'ETH' in symbol else 1.0
        
        for i in range(random.randint(0, 5)):
            event = LiquidationEvent(
                symbol=symbol,
                timestamp=datetime.now() - timedelta(minutes=random.randint(1, 60)),
                exchange=exchange_name,
                side=random.choice(['long', 'short']),
                price=base_price * random.uniform(0.95, 1.05),
                quantity=random.uniform(0.1, 5.0),
                quote_quantity=random.uniform(1000, 100000),
                liquidation_type=random.choice(['partial', 'full'])
            )
            events.append(event)
        
        return events
    
    def _get_mock_market_data(self, symbol: str, exchange_name: str) -> MarketDataTick:
        """Generate mock market data"""
        import random
        
        base_price = 50000.0 if 'BTC' in symbol else 3000.0 if 'ETH' in symbol else 1.0
        price = base_price * random.uniform(0.98, 1.02)
        
        return MarketDataTick(
            symbol=symbol,
            timestamp=datetime.now(),
            price=price,
            volume=random.uniform(100, 10000),
            bid=price * 0.999,
            ask=price * 1.001,
            bid_volume=random.uniform(50, 5000),
            ask_volume=random.uniform(50, 5000),
            exchange=exchange_name,
            data_type='tick'
        )
    
    def _get_mock_ohlcv(self, symbol: str, timeframe: str, limit: int, exchange_name: str) -> List[Dict[str, Any]]:
        """Generate mock OHLCV data"""
        import random
        
        base_price = 50000.0 if 'BTC' in symbol else 3000.0 if 'ETH' in symbol else 1.0
        data = []
        
        for i in range(limit):
            timestamp = datetime.now() - timedelta(minutes=i)
            open_price = base_price * random.uniform(0.95, 1.05)
            high_price = open_price * random.uniform(1.0, 1.02)
            low_price = open_price * random.uniform(0.98, 1.0)
            close_price = random.uniform(low_price, high_price)
            volume = random.uniform(100, 10000)
            
            data.append({
                'timestamp': timestamp,
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': volume
            })
        
        return data
    
    def _get_mock_on_chain_events(self, chain: str, symbol: str = None, limit: int = 100) -> List[OnChainEvent]:
        """Generate mock on-chain events for testing"""
        import random
        
        events = []
        base_value = 100.0 if symbol else 50.0
        
        for i in range(limit):
            # Generate realistic blockchain addresses
            from_addr = f"0x{''.join(random.choices('0123456789abcdef', k=40))}"
            to_addr = f"0x{''.join(random.choices('0123456789abcdef', k=40))}"
            tx_hash = f"0x{''.join(random.choices('0123456789abcdef', k=64))}"
            
            event = OnChainEvent(
                timestamp=datetime.now() - timedelta(minutes=random.randint(1, 60)),
                chain=chain,
                tx_hash=tx_hash,
                from_address=from_addr,
                to_address=to_addr,
                value=base_value * random.uniform(0.1, 10.0),
                gas_used=random.randint(21000, 100000),
                event_type=random.choice(['transfer', 'swap', 'liquidity', 'stake']),
                symbol=symbol,
                block_number=random.randint(18000000, 19000000),
                metadata={
                    'gas_price': random.randint(10, 100),
                    'confirmations': random.randint(1, 12),
                    'fee': random.uniform(0.001, 0.1)
                }
            )
            events.append(event)
        
        return events
    
    # ===== Week 7.3 Phase 1: Funding Rate Methods =====
    
    async def get_funding_rates(self, symbol: str, exchange_name: str = 'okx') -> Optional[FundingRate]:
        """Get funding rates from exchange (prioritizing OKX for better rate limits)"""
        try:
            if not CCXT_AVAILABLE:
                self.logger.warning("CCXT not available - using mock funding rate")
                return self._get_mock_funding_rate(symbol, exchange_name)
            
            # Check rate limiting
            if not self._check_rate_limit(exchange_name):
                self.logger.warning(f"Rate limit hit for {exchange_name}, using cached data")
                return self._get_cached_funding_rate(symbol, exchange_name)
            
            exchange = self.exchanges.get(exchange_name)
            if not exchange:
                self.logger.error(f"Exchange {exchange_name} not initialized")
                return None
            
            # Fetch funding rate from exchange
            try:
                # For OKX, use the funding rate endpoint
                if exchange_name == 'okx':
                    # OKX has better rate limits and funding rate access
                    funding_data = await exchange.fetch_funding_rate(symbol)
                    
                    return FundingRate(
                        symbol=symbol,
                        exchange=exchange_name,
                        funding_rate=funding_data.get('fundingRate', 0.0),
                        timestamp=datetime.fromtimestamp(funding_data.get('timestamp', 0) / 1000),
                        next_funding_time=datetime.fromtimestamp(funding_data.get('nextFundingTime', 0) / 1000) if funding_data.get('nextFundingTime') else None,
                        estimated_rate=funding_data.get('estimatedRate'),
                        metadata={
                            'raw_data': funding_data,
                            'fetch_time': datetime.now().isoformat()
                        }
                    )
                else:
                    # Fallback to other exchanges
                    funding_data = await exchange.fetch_funding_rate(symbol)
                    
                    return FundingRate(
                        symbol=symbol,
                        exchange=exchange_name,
                        funding_rate=funding_data.get('fundingRate', 0.0),
                        timestamp=datetime.fromtimestamp(funding_data.get('timestamp', 0) / 1000),
                        next_funding_time=datetime.fromtimestamp(funding_data.get('nextFundingTime', 0) / 1000) if funding_data.get('nextFundingTime') else None,
                        estimated_rate=funding_data.get('estimatedRate'),
                        metadata={
                            'raw_data': funding_data,
                            'fetch_time': datetime.now().isoformat()
                        }
                    )
                    
            except Exception as e:
                self.logger.error(f"Error fetching funding rate from {exchange_name}: {e}")
                # Fallback to mock data
                return self._get_mock_funding_rate(symbol, exchange_name)
            
        except Exception as e:
            self.logger.error(f"Error in get_funding_rates: {e}")
            return None
    
    def _get_mock_funding_rate(self, symbol: str, exchange_name: str) -> FundingRate:
        """Generate mock funding rate data for testing"""
        import random
        
        # Generate realistic funding rates (-0.1% to +0.1%)
        funding_rate = random.uniform(-0.001, 0.001)
        
        # Next funding time (every 8 hours for most exchanges)
        next_funding = datetime.now() + timedelta(hours=random.randint(1, 8))
        
        return FundingRate(
            symbol=symbol,
            exchange=exchange_name,
            funding_rate=funding_rate,
            timestamp=datetime.now(),
            next_funding_time=next_funding,
            estimated_rate=funding_rate * random.uniform(0.8, 1.2),
            metadata={
                'mock': True,
                'generated_at': datetime.now().isoformat()
            }
        )
    
    def _get_cached_funding_rate(self, symbol: str, exchange_name: str) -> Optional[FundingRate]:
        """Get cached funding rate data"""
        # This would integrate with Redis cache in production
        # For now, return None to trigger mock data
        return None
    
    def _check_rate_limit(self, exchange_name: str) -> bool:
        """Check if we're within rate limits for an exchange"""
        current_time = time.time()
        rate_info = self.rate_limits[exchange_name]
        
        # Reset counter if more than 1 minute has passed
        if current_time - rate_info['last_call'] > 60:
            rate_info['last_call'] = current_time
            rate_info['call_count'] = 0
        
        # Check if we're within limits
        if rate_info['call_count'] >= self.max_calls_per_minute:
            self.stats['rate_limit_hits'] += 1
            return False
        
        rate_info['call_count'] += 1
        return True
    
    async def close(self):
        """Close all exchange connections"""
        try:
            for exchange_name, exchange in self.exchanges.items():
                try:
                    await exchange.close()
                    self.logger.info(f"Closed {exchange_name} connection")
                except Exception as e:
                    self.logger.error(f"Error closing {exchange_name} connection: {e}")
            
            self.logger.info("CCXT Integration Service closed successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to close CCXT Integration Service: {e}")
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()
    
    # ==================== ENHANCED FUTURES DATA COLLECTION ====================
    
    async def get_open_interest(self, symbol: str, exchange_name: str = 'binance') -> Optional[OpenInterest]:
        """Get open interest data for futures trading"""
        if not CCXT_AVAILABLE:
            return self._get_mock_open_interest(symbol, exchange_name)
        
        try:
            exchange = self.exchanges.get(exchange_name)
            if not exchange:
                self.logger.error(f"Exchange {exchange_name} not available")
                return None
            
            # Check rate limit
            await self._wait_for_rate_limit(exchange_name)
            
            # Try to fetch open interest
            try:
                if hasattr(exchange, 'fetch_open_interest'):
                    oi_data = await exchange.fetch_open_interest(symbol)
                    
                    open_interest = OpenInterest(
                        symbol=symbol,
                        exchange=exchange_name,
                        open_interest=oi_data.get('openInterestAmount', 0.0),
                        open_interest_value=oi_data.get('openInterestValue', 0.0),
                        timestamp=datetime.fromtimestamp(oi_data.get('timestamp', 0) / 1000),
                        metadata={
                            'raw_data': oi_data,
                            'fetch_time': datetime.now().isoformat()
                        }
                    )
                    
                    # Cache open interest data
                    self.open_interest_cache[symbol].append(open_interest)
                    
                    # Update statistics
                    self.stats['total_requests'] += 1
                    self.stats['successful_requests'] += 1
                    
                    # Trigger callbacks
                    await self._trigger_callbacks('open_interest', open_interest)
                    
                    return open_interest
                    
                else:
                    self.logger.debug(f"Exchange {exchange_name} does not support open interest fetching")
                    return self._get_mock_open_interest(symbol, exchange_name)
                    
            except Exception as e:
                self.logger.debug(f"Open interest fetching not supported for {exchange_name}: {e}")
                return self._get_mock_open_interest(symbol, exchange_name)
                
        except Exception as e:
            self.logger.error(f"Error fetching open interest for {symbol} from {exchange_name}: {e}")
            self.stats['total_requests'] += 1
            self.stats['failed_requests'] += 1
            return self._get_mock_open_interest(symbol, exchange_name)
    
    def _get_mock_open_interest(self, symbol: str, exchange_name: str) -> OpenInterest:
        """Generate mock open interest data for testing"""
        import random
        
        # Generate realistic open interest values
        oi_amount = random.uniform(1000, 100000)
        oi_value = oi_amount * random.uniform(20000, 50000)  # Assuming BTC price range
        
        return OpenInterest(
            symbol=symbol,
            exchange=exchange_name,
            open_interest=oi_amount,
            open_interest_value=oi_value,
            timestamp=datetime.now(),
            metadata={
                'mock': True,
                'generated_at': datetime.now().isoformat()
            }
        )
    
    async def get_liquidation_levels(self, symbol: str, exchange_name: str = 'binance') -> List[LiquidationLevel]:
        """Get liquidation levels for risk management"""
        if not CCXT_AVAILABLE:
            return self._get_mock_liquidation_levels(symbol, exchange_name)
        
        try:
            exchange = self.exchanges.get(exchange_name)
            if not exchange:
                self.logger.error(f"Exchange {exchange_name} not available")
                return []
            
            # Check rate limit
            await self._wait_for_rate_limit(exchange_name)
            
            # Try to fetch liquidation levels (not all exchanges support this)
            try:
                if hasattr(exchange, 'fetch_liquidation_levels'):
                    levels_data = await exchange.fetch_liquidation_levels(symbol)
                    
                    levels = []
                    for level_data in levels_data:
                        level = LiquidationLevel(
                            symbol=symbol,
                            exchange=exchange_name,
                            price_level=level_data.get('price', 0.0),
                            side=level_data.get('side', 'unknown'),
                            quantity=level_data.get('quantity', 0.0),
                            timestamp=datetime.fromtimestamp(level_data.get('timestamp', 0) / 1000),
                            metadata={
                                'raw_data': level_data,
                                'fetch_time': datetime.now().isoformat()
                            }
                        )
                        levels.append(level)
                    
                    # Cache liquidation levels
                    self.liquidation_levels_cache[symbol].extend(levels)
                    
                    # Update statistics
                    self.stats['total_requests'] += 1
                    self.stats['successful_requests'] += 1
                    
                    # Trigger callbacks
                    await self._trigger_callbacks('liquidation_levels', levels)
                    
                    return levels
                    
                else:
                    self.logger.debug(f"Exchange {exchange_name} does not support liquidation levels fetching")
                    return self._get_mock_liquidation_levels(symbol, exchange_name)
                    
            except Exception as e:
                self.logger.debug(f"Liquidation levels fetching not supported for {exchange_name}: {e}")
                return self._get_mock_liquidation_levels(symbol, exchange_name)
                
        except Exception as e:
            self.logger.error(f"Error fetching liquidation levels for {symbol} from {exchange_name}: {e}")
            self.stats['total_requests'] += 1
            self.stats['failed_requests'] += 1
            return self._get_mock_liquidation_levels(symbol, exchange_name)
    
    def _get_mock_liquidation_levels(self, symbol: str, exchange_name: str) -> List[LiquidationLevel]:
        """Generate mock liquidation levels for testing"""
        import random
        
        levels = []
        current_price = random.uniform(20000, 50000)  # Assuming BTC price range
        
        # Generate liquidation levels around current price
        for i in range(random.randint(5, 15)):
            side = random.choice(['long', 'short'])
            if side == 'long':
                price_level = current_price * (1 + random.uniform(0.01, 0.05))  # Above current price
            else:
                price_level = current_price * (1 - random.uniform(0.01, 0.05))  # Below current price
            
            level = LiquidationLevel(
                symbol=symbol,
                exchange=exchange_name,
                price_level=price_level,
                side=side,
                quantity=random.uniform(10, 1000),
                timestamp=datetime.now(),
                metadata={
                    'mock': True,
                    'generated_at': datetime.now().isoformat()
                }
            )
            levels.append(level)
        
        return levels
    
    # ==================== WEBSOCKET DELTA STREAMING ====================
    
    async def start_websocket_streaming(self, symbols: List[str] = None, exchanges: List[str] = None):
        """Start WebSocket streaming for real-time data"""
        if not self.websocket_enabled:
            self.logger.info("WebSocket streaming disabled")
            return
        
        symbols = symbols or self.symbols
        exchanges = exchanges or list(self.exchanges.keys())
        
        try:
            for exchange_name in exchanges:
                if exchange_name in self.exchanges:
                    await self._start_exchange_websocket(exchange_name, symbols)
                    
        except Exception as e:
            self.logger.error(f"Error starting WebSocket streaming: {e}")
    
    async def _start_exchange_websocket(self, exchange_name: str, symbols: List[str]):
        """Start WebSocket connection for specific exchange"""
        try:
            exchange = self.exchanges.get(exchange_name)
            if not exchange or not hasattr(exchange, 'watch_order_book'):
                self.logger.warning(f"Exchange {exchange_name} does not support WebSocket order book")
                return
            
            self.logger.info(f"Starting WebSocket streaming for {exchange_name}")
            
            # Start order book streaming for each symbol
            for symbol in symbols:
                asyncio.create_task(self._stream_order_book_deltas(exchange_name, symbol))
                
        except Exception as e:
            self.logger.error(f"Error starting WebSocket for {exchange_name}: {e}")
    
    async def _stream_order_book_deltas(self, exchange_name: str, symbol: str):
        """Stream order book deltas via WebSocket"""
        try:
            exchange = self.exchanges.get(exchange_name)
            if not exchange:
                return
            
            self.logger.info(f"Starting order book delta streaming for {symbol} on {exchange_name}")
            
            while True:
                try:
                    # Watch order book with delta updates
                    order_book = await exchange.watch_order_book(symbol, limit=self.order_book_depth)
                    
                    # Process delta update
                    if self.delta_processing_enabled:
                        delta = await self._process_order_book_delta(exchange_name, symbol, order_book)
                        if delta:
                            # Cache delta
                            self.order_book_delta_cache[symbol].append(delta)
                            
                            # Trigger callbacks
                            await self._trigger_callbacks('order_book_delta', delta)
                    
                    # Small delay to prevent overwhelming
                    await asyncio.sleep(0.1)
                    
                except Exception as e:
                    self.logger.error(f"Error in order book delta streaming for {symbol}: {e}")
                    await asyncio.sleep(5)  # Wait before retrying
                    
        except Exception as e:
            self.logger.error(f"Error starting order book delta streaming for {symbol}: {e}")
    
    async def _process_order_book_delta(self, exchange_name: str, symbol: str, order_book: Dict) -> Optional[OrderBookDelta]:
        """Process order book delta update"""
        try:
            # Get previous order book for comparison
            previous_order_book = self.order_book_cache[exchange_name].get(symbol)
            
            if not previous_order_book:
                # First update, store full order book
                self.order_book_cache[exchange_name][symbol] = order_book
                return None
            
            # Calculate deltas
            bids_delta = []
            asks_delta = []
            
            # Compare bids
            current_bids = {price: volume for price, volume in order_book.get('bids', [])}
            previous_bids = {price: volume for price, volume in previous_order_book.get('bids', [])}
            
            for price, volume in current_bids.items():
                if price not in previous_bids or abs(volume - previous_bids[price]) > 0.000001:
                    bids_delta.append([price, volume])
            
            for price, volume in previous_bids.items():
                if price not in current_bids:
                    bids_delta.append([price, 0])  # Remove order
            
            # Compare asks
            current_asks = {price: volume for price, volume in order_book.get('asks', [])}
            previous_asks = {price: volume for price, volume in previous_order_book.get('asks', [])}
            
            for price, volume in current_asks.items():
                if price not in previous_asks or abs(volume - previous_asks[price]) > 0.000001:
                    asks_delta.append([price, volume])
            
            for price, volume in previous_asks.items():
                if price not in current_asks:
                    asks_delta.append([price, 0])  # Remove order
            
            # Update sequence number
            self.sequence_numbers[symbol] += 1
            
            # Create delta object
            delta = OrderBookDelta(
                symbol=symbol,
                timestamp=datetime.fromtimestamp(order_book.get('timestamp', 0) / 1000),
                exchange=exchange_name,
                bids_delta=bids_delta,
                asks_delta=asks_delta,
                sequence_number=self.sequence_numbers[symbol],
                metadata={
                    'raw_order_book': order_book,
                    'processed_at': datetime.now().isoformat()
                }
            )
            
            # Update cached order book
            self.order_book_cache[exchange_name][symbol] = order_book
            
            return delta
            
        except Exception as e:
            self.logger.error(f"Error processing order book delta: {e}")
            return None
    
    async def get_order_book_deltas(self, symbol: str, limit: int = 100) -> List[OrderBookDelta]:
        """Get recent order book deltas for a symbol"""
        return list(self.order_book_delta_cache[symbol])[-limit:]
    
    async def get_funding_rates_history(self, symbol: str, limit: int = 100) -> List[FundingRate]:
        """Get recent funding rates history for a symbol"""
        return list(self.funding_rate_cache[symbol])[-limit:]
    
    async def get_open_interest_history(self, symbol: str, limit: int = 100) -> List[OpenInterest]:
        """Get recent open interest history for a symbol"""
        return list(self.open_interest_cache[symbol])[-limit:]
