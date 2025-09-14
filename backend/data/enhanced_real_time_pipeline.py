"""
Enhanced Real-Time Data Pipeline for AlphaPulse
Integrates CCXT, order books, liquidation data, and real-time signal generation
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Callable, Tuple
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass
from collections import defaultdict, deque
import json
import numpy as np
import pandas as pd

# Import our components
try:
    from .ccxt_integration_service import CCXTIntegrationService, OrderBookSnapshot, LiquidationEvent, MarketDataTick
    from .social_sentiment_service import SocialSentimentService, SocialPost, SentimentAnalysis, ClusteredSentiment
    from .etl_processor import ETLProcessor, ETLResult, DataQualityMetrics
    from ..database.connection import TimescaleDBConnection
    from ..core.trading_engine import TradingEngine
    from ..strategies.strategy_manager import StrategyManager
    from ..execution.order_manager import OrderManager
    from ..app.services.predictive_analytics_service import PredictiveAnalyticsService
    LiquidationPrediction = None
    OrderBookForecast = None
except ImportError as e:
    logging.warning(f"Some imports not available: {e}")
    CCXTIntegrationService = None
    SocialSentimentService = None
    ETLProcessor = None
    TimescaleDBConnection = None
    TradingEngine = None
    StrategyManager = None
    OrderManager = None
    PredictiveAnalyticsService = None

logger = logging.getLogger(__name__)

@dataclass
class RealTimeDataPoint:
    """Real-time data point with multiple data types"""
    symbol: str
    timestamp: datetime
    exchange: str
    price: float
    volume: float
    order_book: Optional[OrderBookSnapshot]
    liquidation_events: List[LiquidationEvent]
    market_data: Optional[MarketDataTick]
    metadata: Dict[str, Any]

@dataclass
class MarketDepthAnalysis:
    """Market depth analysis result"""
    symbol: str
    timestamp: datetime
    exchange: str
    analysis_type: str
    price_level: float
    volume_at_level: float
    side: str
    confidence: float
    metadata: Dict[str, Any]

@dataclass
class SMCOrderBlock:
    """Smart Money Concepts Order Block"""
    symbol: str
    timestamp: datetime
    timeframe: str
    block_type: str  # 'bullish' or 'bearish'
    high: float
    low: float
    open: float
    close: float
    volume: float
    strength: float  # 0.0 to 1.0
    confidence: float
    fair_value_gaps: List[Dict[str, Any]]
    metadata: Dict[str, Any]

@dataclass
class SMCFairValueGap:
    """Smart Money Concepts Fair Value Gap"""
    symbol: str
    timestamp: datetime
    timeframe: str
    gap_type: str  # 'bullish' or 'bearish'
    high: float
    low: float
    gap_size: float
    fill_probability: float
    strength: float
    confidence: float
    metadata: Dict[str, Any]

@dataclass
class SMCLiquiditySweep:
    """Smart Money Concepts Liquidity Sweep"""
    symbol: str
    timestamp: datetime
    timeframe: str
    sweep_type: str  # 'bullish' or 'bearish'
    price_level: float
    volume: float
    sweep_strength: float
    reversal_probability: float
    confidence: float
    metadata: Dict[str, Any]

@dataclass
class SMCMarketStructure:
    """Smart Money Concepts Market Structure"""
    symbol: str
    timestamp: datetime
    timeframe: str
    structure_type: str  # 'BOS', 'CHoCH', 'Liquidity', 'OrderBlock'
    price_level: float
    direction: str  # 'bullish' or 'bearish'
    strength: float
    confidence: float
    metadata: Dict[str, Any]

class EnhancedRealTimePipeline:
    """Enhanced real-time data pipeline with order book and liquidation analysis"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logger
        
        # Pipeline configuration
        self.update_frequency = self.config.get('update_frequency', 1.0)  # seconds
        self.symbols = self.config.get('symbols', ['BTC/USDT', 'ETH/USDT', 'ADA/USDT'])
        self.exchanges = self.config.get('exchanges', ['binance', 'okx', 'bybit'])
        self.analysis_enabled = self.config.get('analysis_enabled', True)
        self.signal_generation_enabled = self.config.get('signal_generation_enabled', True)
        
        # Component references
        self.ccxt_service = None
        self.social_sentiment_service = None
        self.etl_processor = None
        self.db_connection = None
        self.trading_engine = None
        self.strategy_manager = None
        self.order_manager = None
        self.funding_rate_strategy = None  # Week 7.3 Phase 3: Funding Rate Strategy
        self.predictive_signal = None  # Week 7.4: Predictive Signal Optimization
        self.predictive_analytics_service = None  # ML-based predictive analytics
        
        # Data buffers with enhanced performance
        self.data_buffers = defaultdict(lambda: defaultdict(deque))  # symbol -> data_type -> data
        self.buffer_sizes = {
            'order_book': 100,
            'liquidation': 1000,
            'market_data': 1000,
            'analysis': 500,
            'on_chain': 500,  # On-chain events buffer
            'social_sentiment': 500,  # Social sentiment buffer
            'funding_rate': 100,  # Funding rate buffer for Week 7.3 Phase 1
            'leverage_metrics': 200,  # Enhanced leverage metrics
            'liquidity_analysis': 300,  # Liquidity analysis results
            'order_book_deltas': 500,  # Order book delta updates
            'ml_predictions': 50,  # ML predictions buffer
            'liquidation_levels': 200,  # Liquidation level tracking
        }
        
        # Performance optimization settings
        self.micro_batch_size = self.config.get('micro_batch_size', 10)
        self.micro_batch_timeout = self.config.get('micro_batch_timeout', 0.1)  # seconds
        self.parallel_processing = self.config.get('parallel_processing', True)
        self.memory_cache_enabled = self.config.get('memory_cache_enabled', True)
        self.delta_storage_enabled = self.config.get('delta_storage_enabled', True)
        
        # Memory cache for ultra-low latency
        self.memory_cache = {}
        self.cache_ttl = 5.0  # seconds
        self.cache_cleanup_interval = 60.0  # seconds
        
        # Performance metrics
        self.performance_metrics = {
            'total_updates': 0,
            'batch_updates': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'processing_times': deque(maxlen=1000),
            'latency_measurements': deque(maxlen=1000),
        }
        
        # Analysis state
        self.liquidity_walls = defaultdict(list)  # symbol -> liquidity_walls
        self.order_clusters = defaultdict(list)   # symbol -> order_clusters
        self.imbalance_metrics = defaultdict(dict)  # symbol -> metrics
        self.on_chain_metrics = defaultdict(dict)  # symbol -> on-chain metrics
        self.social_sentiment_metrics = defaultdict(dict)  # symbol -> social sentiment metrics
        self.funding_rate_metrics = defaultdict(dict)  # symbol -> funding rate metrics for Week 7.3 Phase 1
        
        # Week 7.3 Phase 2: Advanced Funding Rate Analytics
        self.funding_rate_correlations = defaultdict(dict)  # symbol -> correlation_matrix
        self.funding_rate_patterns = defaultdict(list)  # symbol -> patterns
        self.funding_rate_volatility = defaultdict(dict)  # symbol -> exchange -> volatility
        self.arbitrage_opportunities = defaultdict(list)  # symbol -> opportunities
        
        # Performance tracking
        self.stats = {
            'total_data_points': 0,
            'order_book_updates': 0,
            'liquidation_events': 0,
            'market_data_updates': 0,
            'on_chain_events': 0,
            'social_sentiment_updates': 0,
            'funding_rate_updates': 0,  # Week 7.3 Phase 1
            'correlations_calculated': 0,  # Week 7.3 Phase 2
            'patterns_identified': 0,  # Week 7.3 Phase 2
            'volatility_analyses': 0,  # Week 7.3 Phase 2
            'arbitrage_opportunities': 0,  # Week 7.3 Phase 2
            'etl_batches_processed': 0,
            'signals_generated': 0,
            'analysis_completed': 0,
            'last_update': None,
            'processing_times': deque(maxlen=100)
        }
        
        # Pipeline state
        self.is_running = False
        self.pipeline_task = None
        
        # Callbacks
        self.data_callbacks = defaultdict(list)  # event_type -> [callback]
        self.signal_callbacks = defaultdict(list)  # signal_type -> [callback]
        
        # Initialize components
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize pipeline components"""
        try:
            # Initialize CCXT service
            if CCXTIntegrationService:
                ccxt_config = {
                    'exchanges': {ex: {} for ex in self.exchanges},
                    'symbols': self.symbols,
                    'order_book_depth': 100
                }
                self.ccxt_service = CCXTIntegrationService(ccxt_config)
                self.logger.info("CCXT service initialized")
            
            # Initialize database connection
            if TimescaleDBConnection:
                db_config = self.config.get('database', {})
                self.db_connection = TimescaleDBConnection(db_config)
                self.logger.info("Database connection initialized")
            
            # Initialize trading components
            if TradingEngine:
                self.trading_engine = TradingEngine()
                self.logger.info("Trading engine initialized")
            
            if StrategyManager:
                self.strategy_manager = StrategyManager()
                self.logger.info("Strategy manager initialized")
            
            if OrderManager:
                self.order_manager = OrderManager()
                self.logger.info("Order manager initialized")
            
            # Initialize funding rate strategy
            try:
                from ..strategies.funding_rate_strategy import FundingRateStrategy
                strategy_config = {
                    'min_confidence': 0.7,
                    'max_risk_score': 0.8,
                    'position_sizing_factor': 0.1,
                    'risk_reward_ratio': 2.0,
                    'max_position_size': 0.2
                }
                self.funding_rate_strategy = FundingRateStrategy(strategy_config)
                self.logger.info("Funding rate strategy initialized")
            except Exception as e:
                self.logger.warning(f"Failed to initialize funding rate strategy: {e}")
            
            # Initialize predictive signal service
            try:
                from ..ml.predictive_signal import PredictiveSignal
                predictor_config = {
                    'confidence_threshold': 0.7,
                    'prune_threshold': 0.1,
                    'min_data_points': 50,
                    'prediction_window': 24
                }
                self.predictive_signal = PredictiveSignal(predictor_config)
                self.logger.info("Predictive signal service initialized")
            except Exception as e:
                self.logger.warning(f"Failed to initialize predictive signal service: {e}")
            
            # Initialize strategy configuration manager (Week 8)
            try:
                from ..strategies.strategy_config import StrategyConfigManager
                self.strategy_config = StrategyConfigManager(self.db_connection)
                self.logger.info("Strategy configuration manager initialized")
            except Exception as e:
                self.logger.warning(f"Failed to initialize strategy configuration manager: {e}")
                self.strategy_config = None
            
            # Initialize performance tracker (Week 8)
            try:
                from ..monitoring.performance_tracker import PerformanceTracker
                self.performance_tracker = PerformanceTracker(self.db_connection)
                self.logger.info("Performance tracker initialized")
            except Exception as e:
                self.logger.warning(f"Failed to initialize performance tracker: {e}")
                self.performance_tracker = None
                
        except Exception as e:
            self.logger.error(f"Failed to initialize components: {e}")
    
    async def initialize(self):
        """Initialize the pipeline"""
        try:
            self.logger.info("Initializing Enhanced Real-Time Pipeline...")
            
            # Initialize CCXT service
            if self.ccxt_service:
                await self.ccxt_service.initialize()
                
                # Add callbacks for data events
                self.ccxt_service.add_callback('order_book', self._on_order_book_update)
                self.ccxt_service.add_callback('liquidation', self._on_liquidation_update)
                self.ccxt_service.add_callback('market_data', self._on_market_data_update)
            
            # Initialize database connection
            if self.db_connection:
                await self.db_connection.initialize()
            
            # Initialize trading components
            if self.trading_engine:
                await self.trading_engine.initialize()
            
            if self.strategy_manager:
                await self.strategy_manager.initialize()
            
            if self.order_manager:
                await self.order_manager.initialize()
            
            # Initialize ML predictive analytics service
            if PredictiveAnalyticsService:
                self.predictive_analytics_service = PredictiveAnalyticsService(self.config)
                await self.predictive_analytics_service.initialize()
                self.logger.info("✅ Predictive Analytics Service initialized")
            
            self.logger.info("Enhanced Real-Time Pipeline initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize pipeline: {e}")
            raise
    
    async def start(self):
        """Start the real-time pipeline"""
        if self.is_running:
            self.logger.warning("Pipeline is already running")
            return
        
        try:
            self.logger.info("Starting Enhanced Real-Time Pipeline...")
            self.is_running = True
            
            # Start the main pipeline loop with performance optimizations
            self.pipeline_task = asyncio.create_task(self._pipeline_loop())
            
            # Start performance monitoring tasks
            if self.memory_cache_enabled:
                self.cache_cleanup_task = asyncio.create_task(self._cache_cleanup_loop())
            
            # Start micro-batching processor
            self.micro_batch_task = asyncio.create_task(self._micro_batch_processor())
            
            self.logger.info("✅ Enhanced Real-Time Pipeline started successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to start pipeline: {e}")
            self.is_running = False
            raise
    
    async def stop(self):
        """Stop the real-time pipeline"""
        if not self.is_running:
            self.logger.warning("Pipeline is not running")
            return
        
        try:
            self.logger.info("Stopping Enhanced Real-Time Pipeline...")
            self.is_running = False
            
            # Cancel the pipeline task
            if self.pipeline_task:
                self.pipeline_task.cancel()
                try:
                    await self.pipeline_task
                except asyncio.CancelledError:
                    pass
            
            self.logger.info("✅ Enhanced Real-Time Pipeline stopped successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to stop pipeline: {e}")
    
    async def _pipeline_loop(self):
        """Main pipeline loop"""
        try:
            while self.is_running:
                start_time = time.time()
                
                # Process all symbols
                for symbol in self.symbols:
                    try:
                        await self._process_symbol(symbol)
                    except Exception as e:
                        self.logger.error(f"Error processing symbol {symbol}: {e}")
                
                # Update statistics
                self.stats['last_update'] = datetime.now()
                processing_time = time.time() - start_time
                self.stats['processing_times'].append(processing_time)
                
                # Wait for next update
                await asyncio.sleep(self.update_frequency)
                
        except asyncio.CancelledError:
            self.logger.info("Pipeline loop cancelled")
        except Exception as e:
            self.logger.error(f"Pipeline loop error: {e}")
            self.is_running = False
    
    async def _process_symbol(self, symbol: str):
        """Process data for a single symbol"""
        try:
            # Fetch data from all exchanges
            for exchange in self.exchanges:
                # Get order book
                order_book = await self.ccxt_service.get_order_book(symbol, exchange)
                if order_book:
                    await self._process_order_book(order_book)
                
                # Get liquidation events
                liquidations = await self.ccxt_service.get_liquidation_events(symbol, exchange)
                if liquidations:
                    await self._process_liquidation_events(liquidations)
                
                # Get market data
                market_data = await self.ccxt_service.get_market_data(symbol, exchange)
                if market_data:
                    await self._process_market_data(market_data)
            
            # Get on-chain events (fetch once per symbol, not per exchange)
            if exchange == self.exchanges[0]:  # Only fetch once per symbol
                on_chain_events = await self.ccxt_service.get_on_chain_events('ethereum', symbol, 50)
                if on_chain_events:
                    await self._process_on_chain_events(on_chain_events)
            
            # Get social sentiment data (fetch once per symbol)
            if exchange == self.exchanges[0] and self.social_sentiment_service:
                sentiment_summary = await self.social_sentiment_service.process_symbol_sentiment(symbol)
                if sentiment_summary:
                    await self._process_social_sentiment(symbol, sentiment_summary)
            
            # Get funding rate data (fetch once per symbol, prioritizing OKX)
            if exchange == self.exchanges[0]:  # Only fetch once per symbol
                await self._process_funding_rate(symbol, 'okx')  # Use OKX for better rate limits
            
            # Process data through ETL pipeline (once per symbol)
            if exchange == self.exchanges[0] and self.etl_processor:
                await self._process_etl_pipeline(symbol)
            
            # Perform market depth analysis
            if self.analysis_enabled:
                await self._analyze_market_depth(symbol)
            
            # Perform ML-based predictions
            if self.predictive_analytics_service:
                await self._perform_ml_predictions(symbol)
            
            # Generate signals if enabled
            if self.signal_generation_enabled:
                await self._generate_signals(symbol)
                
        except Exception as e:
            self.logger.error(f"Error processing symbol {symbol}: {e}")
    
    async def _process_order_book(self, order_book: OrderBookSnapshot):
        """Process order book update"""
        try:
            # Store in buffer
            self.data_buffers[order_book.symbol]['order_book'].append(order_book)
            
            # Maintain buffer size
            buffer = self.data_buffers[order_book.symbol]['order_book']
            if len(buffer) > self.buffer_sizes['order_book']:
                buffer.popleft()
            
            # Save to database
            if self.db_connection:
                order_book_data = {
                    'symbol': order_book.symbol,
                    'timestamp': order_book.timestamp,
                    'exchange': order_book.exchange,
                    'bids': order_book.bids,
                    'asks': order_book.asks,
                    'spread': order_book.spread,
                    'total_bid_volume': order_book.total_bid_volume,
                    'total_ask_volume': order_book.total_ask_volume,
                    'depth_levels': order_book.depth_levels
                }
                await self.db_connection.save_order_book_snapshot(order_book_data)
            
            # Update statistics
            self.stats['order_book_updates'] += 1
            self.stats['total_data_points'] += 1
            
            # Trigger callbacks
            await self._trigger_callbacks('order_book', order_book)
            
        except Exception as e:
            self.logger.error(f"Error processing order book: {e}")
    
    async def _process_liquidation_events(self, liquidations: List[LiquidationEvent]):
        """Process liquidation events"""
        try:
            if not liquidations:
                return
            
            symbol = liquidations[0].symbol
            
            # Store in buffer
            self.data_buffers[symbol]['liquidation'].extend(liquidations)
            
            # Maintain buffer size
            buffer = self.data_buffers[symbol]['liquidation']
            if len(buffer) > self.buffer_sizes['liquidation']:
                excess = len(buffer) - self.buffer_sizes['liquidation']
                for _ in range(excess):
                    buffer.popleft()
            
            # Save to database
            if self.db_connection:
                for liquidation in liquidations:
                    liquidation_data = {
                        'symbol': liquidation.symbol,
                        'timestamp': liquidation.timestamp,
                        'exchange': liquidation.exchange,
                        'side': liquidation.side,
                        'price': liquidation.price,
                        'quantity': liquidation.quantity,
                        'quote_quantity': liquidation.quote_quantity,
                        'liquidation_type': liquidation.liquidation_type
                    }
                    await self.db_connection.save_liquidation_event(liquidation_data)
            
            # Update statistics
            self.stats['liquidation_events'] += len(liquidations)
            self.stats['total_data_points'] += len(liquidations)
            
            # Trigger callbacks
            await self._trigger_callbacks('liquidation', liquidations)
            
        except Exception as e:
            self.logger.error(f"Error processing liquidation events: {e}")
    
    async def _process_market_data(self, market_data: MarketDataTick):
        """Process market data update"""
        try:
            # Store in buffer
            self.data_buffers[market_data.symbol]['market_data'].append(market_data)
            
            # Maintain buffer size
            buffer = self.data_buffers[market_data.symbol]['market_data']
            if len(buffer) > self.buffer_sizes['market_data']:
                buffer.popleft()
            
            # Save to database
            if self.db_connection:
                market_data_dict = {
                    'symbol': market_data.symbol,
                    'timestamp': market_data.timestamp,
                    'price': market_data.price,
                    'volume': market_data.volume,
                    'bid': market_data.bid,
                    'ask': market_data.ask,
                    'bid_volume': market_data.bid_volume,
                    'ask_volume': market_data.ask_volume,
                    'exchange': market_data.exchange,
                    'data_type': market_data.data_type
                }
                await self.db_connection.save_real_time_market_data(market_data_dict)
            
            # Update statistics
            self.stats['market_data_updates'] += 1
            self.stats['total_data_points'] += 1
            
            # Trigger callbacks
            await self._trigger_callbacks('market_data', market_data)
            
        except Exception as e:
            self.logger.error(f"Error processing market data: {e}")
    
    async def _process_on_chain_events(self, on_chain_events: List[Any]):
        """Process on-chain events"""
        try:
            if not on_chain_events:
                return
            
            for event in on_chain_events:
                # Store in buffer
                symbol = event.symbol or 'ETH'  # Default to ETH if no symbol
                self.data_buffers[symbol]['on_chain'].append(event)
                
                # Maintain buffer size
                buffer = self.data_buffers[symbol]['on_chain']
                if len(buffer) > self.buffer_sizes['on_chain']:
                    buffer.popleft()
                
                # Save to database
                if self.db_connection:
                    on_chain_data = {
                        'timestamp': event.timestamp,
                        'chain': event.chain,
                        'tx_hash': event.tx_hash,
                        'from_address': event.from_address,
                        'to_address': event.to_address,
                        'value': event.value,
                        'gas_used': event.gas_used,
                        'event_type': event.event_type,
                        'symbol': event.symbol,
                        'block_number': event.block_number,
                        'metadata': event.metadata
                    }
                    await self.db_connection.save_on_chain_event(on_chain_data)
                
                # Update on-chain metrics
                await self._update_on_chain_metrics(symbol, event)
            
            # Update statistics
            self.stats['total_data_points'] += len(on_chain_events)
            self.stats['on_chain_events'] += len(on_chain_events)
            
            # Trigger callbacks
            await self._trigger_callbacks('on_chain', on_chain_events)
            
        except Exception as e:
            self.logger.error(f"Error processing on-chain events: {e}")
    
    async def _update_on_chain_metrics(self, symbol: str, event: Any):
        """Update on-chain metrics for a symbol"""
        try:
            if symbol not in self.on_chain_metrics:
                self.on_chain_metrics[symbol] = {
                    'total_transactions': 0,
                    'total_volume': 0.0,
                    'large_transfers': 0,
                    'whale_activity': 0,
                    'last_update': None
                }
            
            metrics = self.on_chain_metrics[symbol]
            metrics['total_transactions'] += 1
            metrics['total_volume'] += event.value or 0.0
            metrics['last_update'] = datetime.now()
            
            # Detect large transfers (whale activity)
            if event.value and event.value > 1000:  # Threshold for large transfer
                metrics['large_transfers'] += 1
                metrics['whale_activity'] += 1
            
            # Store updated metrics
            self.on_chain_metrics[symbol] = metrics
            
        except Exception as e:
            self.logger.error(f"Error updating on-chain metrics: {e}")
    
    async def _process_social_sentiment(self, symbol: str, sentiment_summary: Dict[str, Any]):
        """Process social sentiment data"""
        try:
            # Store sentiment summary in buffer
            self.data_buffers[symbol]['social_sentiment'].append(sentiment_summary)
            
            # Maintain buffer size
            buffer = self.data_buffers[symbol]['social_sentiment']
            if len(buffer) > self.buffer_sizes['social_sentiment']:
                buffer.popleft()
            
            # Update social sentiment metrics
            self.social_sentiment_metrics[symbol] = sentiment_summary
            
            # Update statistics
            self.stats['social_sentiment_updates'] += 1
            self.stats['total_data_points'] += 1
            
            # Trigger callbacks
            await self._trigger_callbacks('social_sentiment', {
                'symbol': symbol,
                'summary': sentiment_summary,
                'timestamp': datetime.now()
            })
            
        except Exception as e:
            self.logger.error(f"Error processing social sentiment: {e}")
    
    # ===== Week 7.3 Phase 1: Funding Rate Processing =====
    
    async def _process_funding_rate(self, symbol: str, exchange: str = 'okx'):
        """Process funding rate data for leverage signal generation"""
        try:
            if not self.ccxt_service:
                self.logger.warning("CCXT service not available for funding rate processing")
                return
            
            # Fetch funding rate from exchange (prioritizing OKX for better rate limits)
            funding_rate = await self.ccxt_service.get_funding_rates(symbol, exchange)
            if not funding_rate:
                self.logger.warning(f"Failed to fetch funding rate for {symbol} from {exchange}")
                return
            
            # Store in buffer
            self.data_buffers[symbol]['funding_rate'].append(funding_rate)
            
            # Maintain buffer size
            buffer = self.data_buffers[symbol]['funding_rate']
            if len(buffer) > self.buffer_sizes.get('funding_rate', 100):
                buffer.popleft()
            
            # Save to database
            if self.db_connection:
                await self.db_connection.save_funding_rate(
                    symbol=symbol,
                    exchange=exchange,
                    funding_rate=funding_rate.funding_rate,
                    next_funding_time=funding_rate.next_funding_time,
                    estimated_rate=funding_rate.estimated_rate,
                    metadata=funding_rate.metadata
                )
            
            # Update funding rate metrics
            await self._update_funding_rate_metrics(symbol, funding_rate)
            
            # Week 7.3 Phase 2: Advanced Analytics
            await self._update_funding_rate_volatility(symbol, exchange)
            await self._identify_funding_rate_patterns(symbol, exchange)
            await self._update_funding_rate_correlations(symbol)
            await self._check_arbitrage_opportunities(symbol)
            
            # Week 7.3 Phase 3: Strategy Integration
            await self._generate_funding_rate_signals(symbol)
            
            # Week 7.4: Predictive Signal Optimization
            await self._process_predictive_signals(symbol, funding_rate)
            
            # Week 8: Strategy Configuration and Performance Tracking
            await self._process_configured_funding_rate_signals(symbol, funding_rate)
            
            # Update statistics
            self.stats['funding_rate_updates'] += 1
            self.stats['total_data_points'] += 1
            
            # Trigger callbacks
            await self._trigger_callbacks('funding_rate', {
                'symbol': symbol,
                'funding_rate': funding_rate,
                'timestamp': datetime.now()
            })
            
            self.logger.info(f"Funding rate processed for {symbol}: {funding_rate.funding_rate:.6f}")
            
        except Exception as e:
            self.logger.error(f"Error processing funding rate for {symbol}: {e}")
    
    async def _update_funding_rate_metrics(self, symbol: str, funding_rate):
        """Update funding rate metrics for a symbol"""
        try:
            if symbol not in self.funding_rate_metrics:
                self.funding_rate_metrics[symbol] = {
                    'current_rate': 0.0,
                    'average_rate': 0.0,
                    'rate_trend': 'neutral',
                    'leverage_signal': 'neutral',
                    'last_update': None,
                    'rate_history': deque(maxlen=100)
                }
            
            metrics = self.funding_rate_metrics[symbol]
            metrics['current_rate'] = funding_rate.funding_rate
            metrics['last_update'] = datetime.now()
            
            # Add to rate history
            metrics['rate_history'].append(funding_rate.funding_rate)
            
            # Calculate average rate
            if metrics['rate_history']:
                metrics['average_rate'] = sum(metrics['rate_history']) / len(metrics['rate_history'])
            
            # Determine rate trend
            if len(metrics['rate_history']) >= 3:
                recent_rates = list(metrics['rate_history'])[-3:]
                if all(r > 0 for r in recent_rates):
                    metrics['rate_trend'] = 'increasing'
                elif all(r < 0 for r in recent_rates):
                    metrics['rate_trend'] = 'decreasing'
                else:
                    metrics['rate_trend'] = 'volatile'
            
            # Generate leverage signal based on funding rate
            if abs(funding_rate.funding_rate) > 0.0005:  # 0.05% threshold
                if funding_rate.funding_rate > 0:
                    metrics['leverage_signal'] = 'long_squeeze_risk'
                else:
                    metrics['leverage_signal'] = 'short_squeeze_risk'
            else:
                metrics['leverage_signal'] = 'neutral'
            
            # Store updated metrics
            self.funding_rate_metrics[symbol] = metrics
            
        except Exception as e:
            self.logger.error(f"Error updating funding rate metrics: {e}")
    
    # ===== Week 7.3 Phase 2: Advanced Funding Rate Analytics Methods =====
    
    async def _update_funding_rate_correlations(self, symbol: str):
        """Update cross-exchange funding rate correlations"""
        try:
            # Get all funding rate data from the buffer
            funding_rate_buffer = self.data_buffers[symbol]['funding_rate']
            if len(funding_rate_buffer) < 20:  # Need enough data points
                return
            
            # Group data by exchange
            exchange_data = {}
            min_timestamp = datetime.now(timezone.utc) - timedelta(hours=24)
            
            for data_point in funding_rate_buffer:
                # Ensure timestamp is timezone-aware for comparison
                timestamp = data_point.timestamp
                if timestamp.tzinfo is None:
                    timestamp = timestamp.replace(tzinfo=timezone.utc)
                
                if timestamp >= min_timestamp:
                    exchange = data_point.exchange
                    if exchange not in exchange_data:
                        exchange_data[exchange] = {'rates': [], 'timestamps': []}
                    
                    exchange_data[exchange]['rates'].append(data_point.funding_rate)
                    exchange_data[exchange]['timestamps'].append(data_point.timestamp)
            
            if len(exchange_data) < 2:
                return
            
            # Prepare correlation data
            correlation_data = {}
            for exchange, data in exchange_data.items():
                if len(data['rates']) >= 10:  # Minimum data points for correlation
                    correlation_data[exchange] = pd.Series(data['rates'], index=data['timestamps'])
            
            if len(correlation_data) < 2:
                return
            
            # Calculate correlations
            df = pd.DataFrame(correlation_data)
            correlation_matrix = df.corr()
            
            # Store correlation matrix
            self.funding_rate_correlations[symbol] = correlation_matrix
            
            # Generate correlation insights
            await self._generate_correlation_insights(symbol, correlation_matrix)
            
            self.stats['correlations_calculated'] += 1
            
        except Exception as e:
            self.logger.error(f"Error updating funding rate correlations: {e}")
    
    async def _generate_correlation_insights(self, symbol: str, correlation_matrix: pd.DataFrame):
        """Generate insights from correlation matrix"""
        try:
            exchanges = correlation_matrix.index.tolist()
            
            for i, exchange1 in enumerate(exchanges):
                for j, exchange2 in enumerate(exchanges[i+1:], i+1):
                    correlation = correlation_matrix.iloc[i, j]
                    
                    if abs(correlation) >= 0.7:  # High correlation threshold
                        # High correlation detected
                        insight = {
                            'type': 'high_correlation',
                            'symbol': symbol,
                            'exchange1': exchange1,
                            'exchange2': exchange2,
                            'correlation': correlation,
                            'timestamp': datetime.now(),
                            'threshold': 0.7
                        }
                        
                        # Trigger callback
                        await self._trigger_callbacks('correlation_insight', insight)
                        
                        self.logger.info(f"High correlation detected for {symbol}: {exchange1} vs {exchange2} = {correlation:.4f}")
            
        except Exception as e:
            self.logger.error(f"Error generating correlation insights: {e}")
    
    async def _update_funding_rate_volatility(self, symbol: str, exchange: str):
        """Update funding rate volatility analysis"""
        try:
            # Get funding rate data for the specific exchange
            funding_rate_buffer = self.data_buffers[symbol]['funding_rate']
            rates = [data.funding_rate for data in funding_rate_buffer if data.exchange == exchange]
            
            if len(rates) < 20:  # Minimum data points for volatility
                return
            
            rates_array = np.array(rates)
            
            # Calculate volatility measures
            historical_vol = np.std(rates_array)
            
            if len(rates) >= 50:
                rolling_vol = np.std(rates_array[-50:])
            else:
                rolling_vol = historical_vol
            
            # Calculate percentiles for risk assessment
            percentile_95 = np.percentile(rates_array, 95)
            percentile_99 = np.percentile(rates_array, 99)
            
            # Determine volatility type
            if historical_vol > 0.001:  # High volatility threshold
                vol_type = 'high_volatility'
            elif historical_vol < 0.0005:  # Low volatility threshold
                vol_type = 'low_volatility'
            else:
                vol_type = 'moderate_volatility'
            
            volatility_analysis = {
                'symbol': symbol,
                'exchange': exchange,
                'volatility': historical_vol,
                'rolling_volatility': rolling_vol,
                'volatility_type': vol_type,
                'percentile_95': percentile_95,
                'percentile_99': percentile_99,
                'timestamp': datetime.now(),
                'data_points': len(rates)
            }
            
            # Store volatility analysis
            self.funding_rate_volatility[symbol][exchange] = volatility_analysis
            
            # Trigger callback for high volatility
            if vol_type == 'high_volatility':
                await self._trigger_callbacks('high_volatility', volatility_analysis)
            
            self.stats['volatility_analyses'] += 1
            
        except Exception as e:
            self.logger.error(f"Error updating funding rate volatility: {e}")
    
    async def _identify_funding_rate_patterns(self, symbol: str, exchange: str):
        """Identify funding rate patterns"""
        try:
            # Get funding rate data for the specific exchange
            funding_rate_buffer = self.data_buffers[symbol]['funding_rate']
            rates_data = [data for data in funding_rate_buffer if data.exchange == exchange]
            
            if len(rates_data) < 30:  # Minimum data points for pattern recognition
                return
            
            # Extract rates and timestamps
            rates = [data.funding_rate for data in rates_data]
            timestamps = [data.timestamp for data in rates_data]
            
            # Convert to pandas for analysis
            df = pd.DataFrame({
                'rate': rates,
                'timestamp': timestamps
            })
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')
            
            # Calculate moving averages
            df['ma_short'] = df['rate'].rolling(window=5).mean()
            df['ma_long'] = df['rate'].rolling(window=20).mean()
            
            patterns = []
            
            # Trend pattern
            if len(df) >= 20:
                recent_rates = df['rate'].tail(20)
                trend_slope = np.polyfit(range(len(recent_rates)), recent_rates, 1)[0]
                
                if abs(trend_slope) > 0.0001:  # Significant trend
                    pattern_type = 'trending_up' if trend_slope > 0 else 'trending_down'
                    confidence = min(abs(trend_slope) * 10000, 0.95)  # Scale confidence
                    
                    pattern = {
                        'symbol': symbol,
                        'exchange': exchange,
                        'pattern_type': pattern_type,
                        'confidence': confidence,
                        'duration_hours': 20,
                        'start_time': df['timestamp'].iloc[-20],
                        'end_time': df['timestamp'].iloc[-1],
                        'trend_slope': trend_slope,
                        'pattern_method': 'linear_regression'
                    }
                    patterns.append(pattern)
            
            # Mean reversion pattern
            if len(df) >= 30:
                recent_rates = df['rate'].tail(30)
                mean_rate = recent_rates.mean()
                std_rate = recent_rates.std()
                
                # Check if current rate is far from mean
                current_rate = recent_rates.iloc[-1]
                z_score = abs(current_rate - mean_rate) / std_rate
                
                if z_score > 2:  # Significant deviation
                    pattern_type = 'mean_reverting'
                    confidence = min(z_score / 4, 0.95)  # Scale confidence
                    
                    pattern = {
                        'symbol': symbol,
                        'exchange': exchange,
                        'pattern_type': pattern_type,
                        'confidence': confidence,
                        'duration_hours': 30,
                        'start_time': df['timestamp'].iloc[-30],
                        'end_time': df['timestamp'].iloc[-1],
                        'z_score': z_score,
                        'mean_rate': mean_rate,
                        'current_rate': current_rate,
                        'pattern_method': 'z_score_analysis'
                    }
                    patterns.append(pattern)
            
            # Store patterns
            if patterns:
                self.funding_rate_patterns[symbol].extend(patterns)
                
                # Keep only recent patterns
                cutoff_time = datetime.now(timezone.utc) - timedelta(hours=48)
                self.funding_rate_patterns[symbol] = [
                    p for p in self.funding_rate_patterns[symbol] 
                    if p['end_time'] >= cutoff_time
                ]
                
                # Trigger callbacks for high-confidence patterns
                for pattern in patterns:
                    if pattern['confidence'] >= 0.8:  # High confidence threshold
                        await self._trigger_callbacks('pattern_detected', pattern)
                
                self.stats['patterns_identified'] += len(patterns)
            
        except Exception as e:
            self.logger.error(f"Error identifying funding rate patterns: {e}")
    
    async def _check_arbitrage_opportunities(self, symbol: str):
        """Check for funding rate arbitrage opportunities"""
        try:
            # Get all funding rate data and group by exchange
            funding_rate_buffer = self.data_buffers[symbol]['funding_rate']
            if len(funding_rate_buffer) < 10:  # Need enough data points
                return
            
            # Group data by exchange and get latest rates
            exchange_latest_rates = {}
            for data_point in funding_rate_buffer:
                exchange = data_point.exchange
                if exchange not in exchange_latest_rates or data_point.timestamp > exchange_latest_rates[exchange]['timestamp']:
                    exchange_latest_rates[exchange] = {
                        'rate': data_point.funding_rate,
                        'timestamp': data_point.timestamp
                    }
            
            if len(exchange_latest_rates) < 2:
                return
            
            # Get latest rates from all exchanges
            latest_rates = {exchange: data['rate'] for exchange, data in exchange_latest_rates.items()}
            
            # Find arbitrage opportunities
            opportunities = []
            exchanges_list = list(latest_rates.keys())
            
            for i, exchange1 in enumerate(exchanges_list):
                for exchange2 in exchanges_list[i+1:]:
                    if exchange1 in latest_rates and exchange2 in latest_rates:
                        rate1 = latest_rates[exchange1]
                        rate2 = latest_rates[exchange2]
                        
                        rate_difference = abs(rate1 - rate2)
                        
                        if rate_difference >= 0.0005:  # Minimum difference threshold
                            # Calculate potential profit (simplified)
                            potential_profit = rate_difference * 100  # Percentage terms
                            
                            # Determine long/short positions
                            if rate1 > rate2:
                                long_exchange = exchange1
                                short_exchange = exchange2
                            else:
                                long_exchange = exchange2
                                short_exchange = exchange1
                            
                            # Calculate risk score and confidence
                            risk_score = self._calculate_arbitrage_risk(symbol, exchange1, exchange2)
                            confidence = self._calculate_arbitrage_confidence(rate_difference, risk_score)
                            
                            opportunity = {
                                'symbol': symbol,
                                'long_exchange': long_exchange,
                                'short_exchange': short_exchange,
                                'rate_difference': rate_difference,
                                'potential_profit': potential_profit,
                                'risk_score': risk_score,
                                'confidence': confidence,
                                'timestamp': datetime.now(),
                                'rate1': rate1,
                                'rate2': rate2,
                                'threshold': 0.0005
                            }
                            
                            opportunities.append(opportunity)
            
            # Trigger callbacks for high-confidence opportunities
            for opportunity in opportunities:
                if opportunity['confidence'] >= 0.7:  # High confidence threshold
                    await self._trigger_callbacks('arbitrage_opportunity', opportunity)
            
            if opportunities:
                self.arbitrage_opportunities[symbol].extend(opportunities)
                self.stats['arbitrage_opportunities'] += len(opportunities)
            
        except Exception as e:
            self.logger.error(f"Error checking arbitrage opportunities: {e}")
    
    def _calculate_arbitrage_risk(self, symbol: str, exchange1: str, exchange2: str) -> float:
        """Calculate risk score for arbitrage opportunity"""
        try:
            risk_score = 0.5  # Base risk score
            
            # Volatility risk
            if symbol in self.funding_rate_volatility:
                vol1 = self.funding_rate_volatility[symbol].get(exchange1, None)
                vol2 = self.funding_rate_volatility[symbol].get(exchange2, None)
                
                if vol1 and vol2:
                    avg_volatility = (vol1['volatility'] + vol2['volatility']) / 2
                    if avg_volatility > 0.001:  # High volatility threshold
                        risk_score += 0.3  # High volatility increases risk
            
            # Correlation risk
            if symbol in self.funding_rate_correlations:
                corr_matrix = self.funding_rate_correlations[symbol]
                if exchange1 in corr_matrix.index and exchange2 in corr_matrix.columns:
                    correlation = corr_matrix.loc[exchange1, exchange2]
                    if abs(correlation) < 0.5:  # Low correlation threshold
                        risk_score += 0.2  # Low correlation increases risk
            
            return min(1.0, risk_score)
            
        except Exception as e:
            self.logger.error(f"Error calculating arbitrage risk: {e}")
            return 0.5  # Default risk score
    
    def _calculate_arbitrage_confidence(self, rate_difference: float, risk_score: float) -> float:
        """Calculate confidence for arbitrage opportunity"""
        try:
            # Base confidence based on rate difference
            base_confidence = min(rate_difference * 2000, 0.9)  # Scale rate difference
            
            # Adjust for risk
            risk_adjustment = 1.0 - risk_score
            confidence = base_confidence * risk_adjustment
            
            return max(0.1, min(0.95, confidence))
            
        except Exception as e:
            self.logger.error(f"Error calculating arbitrage confidence: {e}")
            return 0.5  # Default confidence
    
    # ===== Week 7.3 Phase 3: Funding Rate Strategy Integration =====
    
    async def _generate_funding_rate_signals(self, symbol: str):
        """Generate trading signals based on funding rate analytics"""
        try:
            if not self.funding_rate_strategy:
                return
            
            # Prepare analytics data for strategy
            analytics_data = {
                'correlations': self.funding_rate_correlations.get(symbol, {}),
                'patterns': self.funding_rate_patterns.get(symbol, []),
                'volatility': self.funding_rate_volatility.get(symbol, {}),
                'arbitrage': self.arbitrage_opportunities.get(symbol, [])
            }
            
            # Generate signals using the strategy service
            signals = await self.funding_rate_strategy.generate_signals(symbol, analytics_data)
            
            if signals:
                # Store signals in pipeline for tracking
                if 'signals' not in self.data_buffers[symbol]:
                    self.data_buffers[symbol]['signals'] = deque(maxlen=100)
                
                for signal in signals:
                    self.data_buffers[symbol]['signals'].append(signal)
                
                # Trigger callbacks for high-confidence signals
                for signal in signals:
                    if signal.confidence >= 0.8:  # High confidence threshold
                        await self._trigger_callbacks('funding_rate_signal', {
                            'symbol': symbol,
                            'signal': signal,
                            'timestamp': datetime.now()
                        })
                
                self.logger.info(f"Generated {len(signals)} funding rate signals for {symbol}")
            
        except Exception as e:
            self.logger.error(f"Error generating funding rate signals: {e}")
    
    # ===== Week 7.4: Predictive Signal Optimization =====
    
    async def _process_predictive_signals(self, symbol: str, funding_rate: Any):
        """Process predictive signals for funding rate data"""
        try:
            if not self.predictive_signal:
                return
            
            # Convert funding rate data to DataFrame for prediction
            if hasattr(funding_rate, 'funding_rate') and hasattr(funding_rate, 'timestamp'):
                # Create DataFrame from funding rate data
                data = pd.DataFrame({
                    'timestamp': [funding_rate.timestamp],
                    'funding_rate': [funding_rate.funding_rate],
                    'price': [50000.0]  # Mock price for demonstration
                })
                
                # Prune low-impact data
                pruning_result = self.predictive_signal.prune_data(data, 'funding_rate')
                
                if pruning_result.retention_rate > 0:
                    # Check if we should generate a signal based on prediction
                    should_generate, confidence = await self.predictive_signal.should_generate_signal(
                        symbol, 'funding_rate', pruning_result.retained_data
                    )
                    
                    if should_generate:
                        # Get prediction details
                        prediction = await self.predictive_signal.predict_signal(
                            symbol, 'funding_rate', pruning_result.retained_data
                        )
                        
                        if prediction and self.db_connection:
                            # Save prediction to database
                            prediction_data = {
                                'timestamp': prediction.timestamp,
                                'symbol': prediction.symbol,
                                'signal_type': prediction.signal_type,
                                'confidence': prediction.confidence,
                                'predicted_pnl': prediction.predicted_pnl,
                                'features': prediction.features,
                                'metadata': prediction.metadata
                            }
                            
                            await self.db_connection.save_signal_prediction(prediction_data)
                            
                            # Trigger callback for high-confidence predictions
                            if prediction.confidence >= 0.8:
                                await self._trigger_callbacks('predictive_signal', {
                                    'symbol': symbol,
                                    'prediction': prediction,
                                    'pruning_result': {
                                        'original_count': pruning_result.original_count,
                                        'retention_rate': pruning_result.retention_rate
                                    }
                                })
                                
                                self.logger.info(f"Predictive signal generated for {symbol}: confidence={prediction.confidence:.3f}")
                
                # Update statistics
                self.stats['data_pruned'] = self.stats.get('data_pruned', 0) + pruning_result.pruned_count
                self.stats['predictive_signals'] = self.stats.get('predictive_signals', 0) + 1
                
        except Exception as e:
            self.logger.error(f"Error processing predictive signals: {e}")
    
    # ===== Week 8: Strategy Configuration and Performance Tracking =====
    
    async def _process_configured_funding_rate_signals(self, symbol: str, funding_rate: Any):
        """Process funding rate signals with strategy configuration and performance tracking"""
        try:
            if not self.strategy_config or not self.performance_tracker:
                return
            
            # Load strategy configuration
            config = await self.strategy_config.load_config('funding_rate', symbol)
            
            # Create signal data
            signal_data = {
                'symbol': symbol,
                'funding_rate': funding_rate.funding_rate,
                'price': getattr(funding_rate, 'price', 50000.0),  # Default price if not available
                'timestamp': funding_rate.timestamp,
                'exchange': getattr(funding_rate, 'exchange', 'okx')
            }
            
            # Validate signal against configuration
            validation_result = self.strategy_config.validate_signal(signal_data, config)
            
            if not validation_result['valid']:
                self.logger.warning(f"Signal validation failed for {symbol}: {validation_result['errors']}")
                return
            
            # Apply configuration-based signal generation
            if abs(funding_rate.funding_rate) >= config.funding_rate_threshold:
                # Determine signal type based on funding rate
                signal_type = 'buy' if funding_rate.funding_rate > 0 else 'sell'
                
                # Calculate confidence based on funding rate magnitude
                confidence = min(0.95, abs(funding_rate.funding_rate) * 1000)  # Scale to 0-1
                
                # Create configured signal
                configured_signal = {
                    'symbol': symbol,
                    'type': signal_type,
                    'confidence': confidence,
                    'strategy_id': 'funding_rate',
                    'funding_rate': funding_rate.funding_rate,
                    'price': signal_data['price'],
                    'timestamp': datetime.now(timezone.utc),
                    'risk_params': {
                        'take_profit_pct': config.take_profit_pct,
                        'stop_loss_pct': config.stop_loss_pct,
                        'max_leverage': config.max_leverage,
                        'position_size_pct': config.position_size_pct,
                        'timeout_hours': config.timeout_hours
                    },
                    'metadata': {
                        'config_source': 'strategy_config',
                        'validation_warnings': validation_result['warnings'],
                        'adjusted_params': validation_result['adjusted_params']
                    }
                }
                
                # Track signal generation performance
                start_time = time.time()
                
                # Store signal in database
                if self.db_connection:
                    await self.db_connection.save_signal_prediction({
                        'timestamp': configured_signal['timestamp'],
                        'symbol': symbol,
                        'signal_type': f"funding_rate_{signal_type}",
                        'confidence': confidence,
                        'predicted_pnl': 0.0,  # Will be calculated after execution
                        'features': {
                            'funding_rate': funding_rate.funding_rate,
                            'price': signal_data['price'],
                            'confidence': confidence
                        },
                        'metadata': configured_signal['metadata']
                    })
                
                # Track execution performance
                execution_time = time.time() - start_time
                execution_metrics = {
                    'pnl': 0.0,  # Placeholder for actual PnL
                    'execution_time': execution_time,
                    'position_size': config.position_size_pct,
                    'metadata': {
                        'signal_type': 'funding_rate',
                        'confidence': confidence,
                        'validation_result': validation_result
                    }
                }
                
                await self.performance_tracker.track_execution(symbol, 'funding_rate', execution_metrics)
                
                # Trigger callbacks for configured signals
                await self._trigger_callbacks('configured_signal', {
                    'symbol': symbol,
                    'signal': configured_signal,
                    'config': config,
                    'validation': validation_result,
                    'execution_time': execution_time
                })
                
                self.logger.info(f"Configured funding rate signal generated for {symbol}: "
                               f"type={signal_type}, confidence={confidence:.3f}, "
                               f"execution_time={execution_time*1000:.2f}ms")
                
                # Update statistics
                self.stats['configured_signals'] = self.stats.get('configured_signals', 0) + 1
                self.stats['total_data_points'] += 1
            
        except Exception as e:
            self.logger.error(f"Error processing configured funding rate signals: {e}")
    
    async def _process_etl_pipeline(self, symbol: str):
        """Process data through ETL pipeline"""
        try:
            # Collect data from all buffers for ETL processing
            etl_data = []
            
            # Add market data
            market_data_buffer = self.data_buffers[symbol]['market_data']
            if market_data_buffer:
                etl_data.extend([{
                    'symbol': data.symbol,
                    'timestamp': data.timestamp,
                    'price': data.price,
                    'volume': data.volume,
                    'bid': data.bid,
                    'ask': data.ask,
                    'bid_volume': data.bid_volume,
                    'ask_volume': data.ask_volume,
                    'exchange': data.exchange,
                    'data_type': data.data_type
                } for data in market_data_buffer])
            
            # Add order book data
            order_book_buffer = self.data_buffers[symbol]['order_book']
            if order_book_buffer:
                latest_order_book = order_book_buffer[-1]
                etl_data.append({
                    'symbol': latest_order_book.symbol,
                    'timestamp': latest_order_book.timestamp,
                    'price': (latest_order_book.bids[0][0] + latest_order_book.asks[0][0]) / 2,
                    'volume': latest_order_book.total_bid_volume + latest_order_book.total_ask_volume,
                    'bid': latest_order_book.bids[0][0],
                    'ask': latest_order_book.asks[0][0],
                    'bid_volume': latest_order_book.total_bid_volume,
                    'ask_volume': latest_order_book.total_ask_volume,
                    'exchange': latest_order_book.exchange,
                    'data_type': 'order_book'
                })
            
            # Process through ETL if we have data
            if etl_data and len(etl_data) >= 10:  # Minimum batch size
                etl_result = await self.etl_processor.process_market_data(etl_data)
                
                if etl_result.output_count > 0:
                    # Update statistics
                    self.stats['etl_batches_processed'] += 1
                    self.stats['total_data_points'] += 1
                    
                    # Store ETL result in buffer
                    self.data_buffers[symbol]['etl_processed'] = etl_result
                    
                    # Trigger callbacks
                    await self._trigger_callbacks('etl_processed', {
                        'symbol': symbol,
                        'result': etl_result,
                        'timestamp': datetime.now()
                    })
                    
                    self.logger.info(f"ETL processing completed for {symbol}: {etl_result.input_count} -> {etl_result.output_count} records")
            
        except Exception as e:
            self.logger.error(f"Error in ETL processing for {symbol}: {e}")
    
    async def _analyze_market_depth(self, symbol: str):
        """Analyze market depth for a symbol"""
        try:
            # Get recent order book data
            order_book_buffer = self.data_buffers[symbol]['order_book']
            if len(order_book_buffer) < 2:
                return
            
            latest_order_book = order_book_buffer[-1]
            
            # Analyze liquidity walls
            liquidity_walls = await self._detect_liquidity_walls(latest_order_book)
            
            # Analyze order clusters
            order_clusters = await self._detect_order_clusters(latest_order_book)
            
            # Calculate imbalance metrics
            imbalance_metrics = await self._calculate_imbalance_metrics(latest_order_book)
            
            # Store analysis results
            analysis_results = []
            
            # Store liquidity walls
            for wall in liquidity_walls:
                analysis = MarketDepthAnalysis(
                    symbol=symbol,
                    timestamp=datetime.now(),
                    exchange=latest_order_book.exchange,
                    analysis_type='liquidity_walls',
                    price_level=wall['price'],
                    volume_at_level=wall['volume'],
                    side=wall['side'],
                    confidence=wall['confidence'],
                    metadata=wall
                )
                analysis_results.append(analysis)
                
                # Save to database
                if self.db_connection:
                    analysis_data = {
                        'symbol': analysis.symbol,
                        'timestamp': analysis.timestamp,
                        'exchange': analysis.exchange,
                        'analysis_type': analysis.analysis_type,
                        'price_level': analysis.price_level,
                        'volume_at_level': analysis.volume_at_level,
                        'side': analysis.side,
                        'confidence': analysis.confidence,
                        'metadata': analysis.metadata
                    }
                    await self.db_connection.save_market_depth_analysis(analysis_data)
            
            # Store in buffer
            self.data_buffers[symbol]['analysis'].extend(analysis_results)
            
            # Maintain buffer size
            buffer = self.data_buffers[symbol]['analysis']
            if len(buffer) > self.buffer_sizes['analysis']:
                excess = len(buffer) - self.buffer_sizes['analysis']
                for _ in range(excess):
                    buffer.popleft()
            
            # Update statistics
            self.stats['analysis_completed'] += 1
            
            # Trigger callbacks
            await self._trigger_callbacks('market_depth_analysis', analysis_results)
            
        except Exception as e:
            self.logger.error(f"Error analyzing market depth for {symbol}: {e}")
    
    async def _perform_ml_predictions(self, symbol: str):
        """Perform ML-based predictions for liquidations and order book forecasting"""
        try:
            if not self.predictive_analytics_service:
                return
            
            # Collect market data for prediction
            market_data = await self._collect_market_data_for_prediction(symbol)
            
            if not market_data:
                return
            
            # Perform liquidation prediction
            liquidation_prediction = await self.predictive_analytics_service.predict_liquidations(symbol, market_data)
            
            if liquidation_prediction:
                # Store prediction in buffer
                self.data_buffers[symbol]['ml_predictions'].append({
                    'type': 'liquidation_prediction',
                    'timestamp': liquidation_prediction.timestamp,
                    'data': liquidation_prediction,
                    'metadata': {
                        'model_confidence': liquidation_prediction.confidence_score,
                        'risk_level': liquidation_prediction.risk_level,
                        'probability': liquidation_prediction.liquidation_probability
                    }
                })
                
                # Store in database with enhanced ML columns
                if self.db_connection:
                    await self._store_ml_prediction(symbol, liquidation_prediction)
                
                # Log high-risk predictions
                if liquidation_prediction.liquidation_probability > 0.7:
                    self.logger.warning(
                        f"High liquidation risk for {symbol}: "
                        f"{liquidation_prediction.liquidation_probability:.2%} "
                        f"(confidence: {liquidation_prediction.confidence_score:.2%})"
                    )
                
                # Update performance metrics
                self.performance_metrics['ml_predictions'] = self.performance_metrics.get('ml_predictions', 0) + 1
                
                # Store in memory cache for ultra-low latency access
                if self.memory_cache_enabled:
                    cache_key = f"ml_prediction:{symbol}:liquidation"
                    self.memory_cache[cache_key] = {
                        'prediction': liquidation_prediction.liquidation_probability,
                        'confidence': liquidation_prediction.confidence_score,
                        'risk_level': liquidation_prediction.risk_level,
                        'timestamp': liquidation_prediction.timestamp.isoformat(),
                        'ttl': time.time() + 300  # 5 minutes TTL
                    }
            
            # Perform order book forecasting
            try:
                order_book_forecast = await self.predictive_analytics_service.forecast_order_book(symbol, market_data)
                
                if order_book_forecast:
                    # Store forecast in buffer
                    self.data_buffers[symbol]['ml_predictions'].append({
                        'type': 'order_book_forecast',
                        'timestamp': order_book_forecast.timestamp,
                        'data': order_book_forecast,
                        'metadata': {
                            'confidence': order_book_forecast.confidence_score,
                            'predicted_spread': order_book_forecast.predicted_spread,
                            'predicted_imbalance': order_book_forecast.predicted_imbalance
                        }
                    })
                    
                    # Store in memory cache
                    if self.memory_cache_enabled:
                        cache_key = f"ml_prediction:{symbol}:order_book"
                        self.memory_cache[cache_key] = {
                            'predicted_spread': order_book_forecast.predicted_spread,
                            'predicted_imbalance': order_book_forecast.predicted_imbalance,
                            'confidence': order_book_forecast.confidence_score,
                            'timestamp': order_book_forecast.timestamp.isoformat(),
                            'ttl': time.time() + 60  # 1 minute TTL for order book predictions
                        }
            
            except Exception as e:
                self.logger.debug(f"Order book forecasting not available for {symbol}: {e}")
            
            # Maintain buffer sizes
            prediction_buffer = self.data_buffers[symbol]['ml_predictions']
            max_predictions = self.buffer_sizes.get('ml_predictions', 50)
            if len(prediction_buffer) > max_predictions:
                excess = len(prediction_buffer) - max_predictions
                for _ in range(excess):
                    prediction_buffer.popleft()
            
            # Update statistics
            self.stats['ml_predictions'] = self.stats.get('ml_predictions', 0) + 1
            
        except Exception as e:
            self.logger.error(f"Error performing ML predictions for {symbol}: {e}")
    
    async def _collect_market_data_for_prediction(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Collect and prepare market data for ML prediction"""
        try:
            market_data = {}
            
            # Get latest order book data
            order_book_buffer = self.data_buffers[symbol]['order_book']
            if order_book_buffer:
                latest_order_book = order_book_buffer[-1]
                market_data.update({
                    'bid_price': latest_order_book.bid_price,
                    'ask_price': latest_order_book.ask_price,
                    'spread': latest_order_book.spread,
                    'bid_volume': latest_order_book.bid_volume,
                    'ask_volume': latest_order_book.ask_volume,
                    'liquidity_imbalance': getattr(latest_order_book, 'liquidity_imbalance', 0),
                    'depth_pressure': getattr(latest_order_book, 'depth_pressure', 0),
                    'order_flow_toxicity': getattr(latest_order_book, 'order_flow_toxicity', 0)
                })
            
            # Get latest market data
            market_data_buffer = self.data_buffers[symbol]['market_data']
            if market_data_buffer:
                latest_market_data = market_data_buffer[-1]
                market_data.update({
                    'current_price': latest_market_data.price,
                    'volume_24h': latest_market_data.volume_24h,
                    'price_change_24h': latest_market_data.price_change_24h,
                    'high_24h': latest_market_data.high_24h,
                    'low_24h': latest_market_data.low_24h
                })
            
            # Get recent liquidation events for context
            liquidation_buffer = self.data_buffers[symbol]['liquidations']
            recent_liquidations = list(liquidation_buffer)[-10:] if liquidation_buffer else []
            
            if recent_liquidations:
                total_liquidation_volume = sum(liq.size for liq in recent_liquidations)
                avg_liquidation_impact = np.mean([liq.impact_score for liq in recent_liquidations if hasattr(liq, 'impact_score')])
                market_data.update({
                    'recent_liquidation_volume': total_liquidation_volume,
                    'avg_liquidation_impact': avg_liquidation_impact,
                    'liquidation_count_recent': len(recent_liquidations)
                })
            
            # Calculate volatility from recent price data
            if 'current_price' in market_data and 'high_24h' in market_data and 'low_24h' in market_data:
                price_range = market_data['high_24h'] - market_data['low_24h']
                volatility = price_range / market_data['current_price'] if market_data['current_price'] > 0 else 0
                market_data['historical_volatility'] = volatility
            
            return market_data if market_data else None
            
        except Exception as e:
            self.logger.error(f"Error collecting market data for prediction: {e}")
            return None
    
    async def _store_ml_prediction(self, symbol: str, prediction: Any):
        """Store ML prediction in database using enhanced ML columns"""
        try:
            if not self.db_connection:
                return
            
            # Store in liquidation_events table with prediction metadata
            prediction_data = {
                'symbol': symbol,
                'timestamp': prediction.timestamp,
                'prediction_probability': prediction.liquidation_probability,
                'prediction_confidence': prediction.confidence_score,
                'prediction_model_version': getattr(prediction, 'model_version', 'v1.0'),
                'prediction_features': {
                    'factors': prediction.factors,
                    'risk_level': prediction.risk_level,
                    'expected_volume': prediction.expected_liquidation_volume,
                    'prediction_horizon': prediction.prediction_horizon
                }
            }
            
            # Save prediction for performance tracking
            await self.db_connection.save_ml_prediction(prediction_data)
            
        except Exception as e:
            self.logger.error(f"Error storing ML prediction: {e}")
    
    async def _detect_liquidity_walls(self, order_book: OrderBookSnapshot) -> List[Dict[str, Any]]:
        """Detect liquidity walls in order book"""
        try:
            walls = []
            
            # Analyze bid side
            bid_walls = self._find_volume_clusters(order_book.bids, 'bid')
            walls.extend(bid_walls)
            
            # Analyze ask side
            ask_walls = self._find_volume_clusters(order_book.asks, 'ask')
            walls.extend(ask_walls)
            
            return walls
            
        except Exception as e:
            self.logger.error(f"Error detecting liquidity walls: {e}")
            return []
    
    def _find_volume_clusters(self, orders: List[List[float]], side: str) -> List[Dict[str, Any]]:
        """Find volume clusters in order book side"""
        try:
            clusters = []
            
            if len(orders) < 3:
                return clusters
            
            # Group orders by price proximity
            price_groups = []
            current_group = [orders[0]]
            
            for i in range(1, len(orders)):
                current_price = orders[i][0]
                prev_price = orders[i-1][0]
                
                # Check if prices are close (within 0.1%)
                if side == 'bid':
                    price_diff = (prev_price - current_price) / prev_price
                else:
                    price_diff = (current_price - prev_price) / prev_price
                
                if price_diff <= 0.001:  # 0.1%
                    current_group.append(orders[i])
                else:
                    if len(current_group) >= 3:  # Minimum cluster size
                        price_groups.append(current_group)
                    current_group = [orders[i]]
            
            # Add last group if it's large enough
            if len(current_group) >= 3:
                price_groups.append(current_group)
            
            # Analyze each group
            for group in price_groups:
                total_volume = sum(order[1] for order in group)
                avg_price = sum(order[0] * order[1] for order in group) / total_volume
                
                # Calculate confidence based on volume and group size
                confidence = min(1.0, (total_volume / 1000) * (len(group) / 10))
                
                if confidence > 0.3:  # Minimum confidence threshold
                    clusters.append({
                        'price': avg_price,
                        'volume': total_volume,
                        'side': side,
                        'confidence': confidence,
                        'order_count': len(group),
                        'price_range': (min(order[0] for order in group), max(order[0] for order in group))
                    })
            
            return clusters
            
        except Exception as e:
            self.logger.error(f"Error finding volume clusters: {e}")
            return []
    
    async def _detect_order_clusters(self, order_book: OrderBookSnapshot) -> List[Dict[str, Any]]:
        """Detect order clusters in order book"""
        try:
            clusters = []
            
            # Analyze bid side
            bid_clusters = self._find_order_clusters(order_book.bids, 'bid')
            clusters.extend(bid_clusters)
            
            # Analyze ask side
            ask_clusters = self._find_order_clusters(order_book.asks, 'ask')
            clusters.extend(ask_clusters)
            
            return clusters
            
        except Exception as e:
            self.logger.error(f"Error detecting order clusters: {e}")
            return []
    
    def _find_order_clusters(self, orders: List[List[float]], side: str) -> List[Dict[str, Any]]:
        """Find order clusters in order book side"""
        try:
            clusters = []
            
            if len(orders) < 5:
                return clusters
            
            # Look for patterns in order sizes
            order_sizes = [order[1] for order in orders]
            mean_size = np.mean(order_sizes)
            std_size = np.std(order_sizes)
            
            # Find orders that are significantly larger than average
            large_orders = []
            for i, order in enumerate(orders):
                if order[1] > mean_size + 2 * std_size:
                    large_orders.append({
                        'index': i,
                        'price': order[0],
                        'volume': order[1],
                        'size_ratio': order[1] / mean_size
                    })
            
            # Group nearby large orders
            if len(large_orders) >= 2:
                current_cluster = [large_orders[0]]
                
                for i in range(1, len(large_orders)):
                    current_order = large_orders[i]
                    prev_order = large_orders[i-1]
                    
                    # Check if orders are close in price
                    price_diff = abs(current_order['price'] - prev_order['price']) / prev_order['price']
                    
                    if price_diff <= 0.002:  # 0.2%
                        current_cluster.append(current_order)
                    else:
                        if len(current_cluster) >= 2:
                            clusters.append({
                                'side': side,
                                'orders': current_cluster,
                                'price_range': (min(o['price'] for o in current_cluster), 
                                              max(o['price'] for o in current_cluster)),
                                'total_volume': sum(o['volume'] for o in current_cluster),
                                'confidence': min(1.0, len(current_cluster) / 5)
                            })
                        current_cluster = [current_order]
                
                # Add last cluster if it's large enough
                if len(current_cluster) >= 2:
                    clusters.append({
                        'side': side,
                        'orders': current_cluster,
                        'price_range': (min(o['price'] for o in current_cluster), 
                                      max(o['price'] for o in current_cluster)),
                        'total_volume': sum(o['volume'] for o in current_cluster),
                        'confidence': min(1.0, len(current_cluster) / 5)
                    })
            
            return clusters
            
        except Exception as e:
            self.logger.error(f"Error finding order clusters: {e}")
            return []
    
    async def _calculate_imbalance_metrics(self, order_book: OrderBookSnapshot) -> Dict[str, Any]:
        """Calculate order book imbalance metrics"""
        try:
            # Calculate bid/ask volume imbalance
            total_bid_volume = order_book.total_bid_volume
            total_ask_volume = order_book.total_ask_volume
            
            volume_imbalance = (total_bid_volume - total_ask_volume) / (total_bid_volume + total_ask_volume)
            
            # Calculate price pressure
            bid_pressure = sum(bid[1] * (1 / (i + 1)) for i, bid in enumerate(order_book.bids))
            ask_pressure = sum(ask[1] * (1 / (i + 1)) for i, ask in enumerate(order_book.asks))
            
            pressure_imbalance = (bid_pressure - ask_pressure) / (bid_pressure + ask_pressure)
            
            # Calculate spread metrics
            spread_ratio = order_book.spread / order_book.bids[0][0] if order_book.bids else 0
            
            # Calculate depth metrics
            bid_depth = len(order_book.bids)
            ask_depth = len(order_book.asks)
            depth_imbalance = (bid_depth - ask_depth) / (bid_depth + ask_depth)
            
            return {
                'volume_imbalance': volume_imbalance,
                'pressure_imbalance': pressure_imbalance,
                'spread_ratio': spread_ratio,
                'depth_imbalance': depth_imbalance,
                'total_bid_volume': total_bid_volume,
                'total_ask_volume': total_ask_volume,
                'bid_depth': bid_depth,
                'ask_depth': ask_depth
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating imbalance metrics: {e}")
            return {}
    
    async def _generate_signals(self, symbol: str):
        """Generate trading signals based on real-time data"""
        try:
            if not self.strategy_manager:
                return
            
            # Get recent analysis data
            analysis_buffer = self.data_buffers[symbol]['analysis']
            if len(analysis_buffer) < 5:
                return
            
            # Get recent market data
            market_data_buffer = self.data_buffers[symbol]['market_data']
            if len(market_data_buffer) < 10:
                return
            
            # Analyze recent data for signal generation
            signal = await self._analyze_for_signals(symbol, analysis_buffer, market_data_buffer)
            
            if signal:
                # Update statistics
                self.stats['signals_generated'] += 1
                
                # Trigger callbacks
                await self._trigger_callbacks('signal', signal)
                
                # Save signal to database
                if self.db_connection:
                    signal_data = {
                        'id': f"signal_{int(time.time())}_{symbol.replace('/', '_')}",
                        'symbol': symbol,
                        'side': signal['side'],
                        'strategy': 'real_time_analysis',
                        'confidence': signal['confidence'],
                        'strength': signal['strength'],
                        'timestamp': datetime.now(),
                        'price': signal['price'],
                        'metadata': signal
                    }
                    await self.db_connection.save_signal(signal_data)
                
        except Exception as e:
            self.logger.error(f"Error generating signals for {symbol}: {e}")
    
    async def _analyze_for_signals(self, symbol: str, analysis_buffer: deque, 
                                  market_data_buffer: deque) -> Optional[Dict[str, Any]]:
        """Analyze data for signal generation"""
        try:
            # Get latest analysis
            latest_analysis = analysis_buffer[-1] if analysis_buffer else None
            if not latest_analysis:
                return None
            
            # Get latest market data
            latest_market_data = market_data_buffer[-1] if market_data_buffer else None
            if not latest_market_data:
                return None
            
            # Simple signal generation logic
            signal = None
            
            # Check for strong liquidity walls
            if latest_analysis.analysis_type == 'liquidity_walls':
                if latest_analysis.confidence > 0.7:
                    if latest_analysis.side == 'bid' and latest_analysis.price_level > latest_market_data.price:
                        # Strong bid wall above current price - bullish signal
                        signal = {
                            'side': 'buy',
                            'confidence': latest_analysis.confidence,
                            'strength': 'strong',
                            'price': latest_market_data.price,
                            'reason': 'strong_bid_wall',
                            'metadata': {
                                'wall_price': latest_analysis.price_level,
                                'wall_volume': latest_analysis.volume_at_level
                            }
                        }
                    elif latest_analysis.side == 'ask' and latest_analysis.price_level < latest_market_data.price:
                        # Strong ask wall below current price - bearish signal
                        signal = {
                            'side': 'sell',
                            'confidence': latest_analysis.confidence,
                            'strength': 'strong',
                            'price': latest_market_data.price,
                            'reason': 'strong_ask_wall',
                            'metadata': {
                                'wall_price': latest_analysis.price_level,
                                'wall_volume': latest_analysis.volume_at_level
                            }
                        }
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Error analyzing for signals: {e}")
            return None
    
    # Callback methods for CCXT service
    async def _on_order_book_update(self, order_book: OrderBookSnapshot):
        """Handle order book updates from CCXT service"""
        await self._process_order_book(order_book)
    
    async def _on_liquidation_update(self, liquidations: List[LiquidationEvent]):
        """Handle liquidation updates from CCXT service"""
        await self._process_liquidation_events(liquidations)
    
    async def _on_market_data_update(self, market_data: MarketDataTick):
        """Handle market data updates from CCXT service"""
        await self._process_market_data(market_data)
    
    # Public methods
    def add_callback(self, event_type: str, callback: Callable):
        """Add callback for pipeline events"""
        self.data_callbacks[event_type].append(callback)
        self.logger.info(f"Added callback for {event_type} events")
    
    def add_signal_callback(self, signal_type: str, callback: Callable):
        """Add callback for signal events"""
        self.signal_callbacks[signal_type].append(callback)
        self.logger.info(f"Added signal callback for {signal_type}")
    
    async def _trigger_callbacks(self, event_type: str, data: Any):
        """Trigger callbacks for pipeline events"""
        callbacks = self.data_callbacks.get(event_type, [])
        
        for callback in callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(data)
                else:
                    callback(data)
            except Exception as e:
                self.logger.error(f"Error in {event_type} callback: {e}")
    
    def get_pipeline_statistics(self) -> Dict[str, Any]:
        """Get pipeline statistics"""
        return {
            'stats': self.stats,
            'buffer_sizes': {
                symbol: {data_type: len(buffer) for data_type, buffer in buffers.items()}
                for symbol, buffers in self.data_buffers.items()
            },
            'is_running': self.is_running,
            'symbols': self.symbols,
            'exchanges': self.exchanges
        }
    
    def get_symbol_data(self, symbol: str, data_type: str = None) -> Dict[str, Any]:
        """Get data for a specific symbol"""
        if symbol not in self.data_buffers:
            return {}
        
        if data_type:
            return {data_type: list(self.data_buffers[symbol][data_type])}
        else:
            return {
                data_type: list(buffer) 
                for data_type, buffer in self.data_buffers[symbol].items()
            }
    
    async def close(self):
        """Close the pipeline"""
        try:
            # Stop the pipeline
            await self.stop()
            
            # Close components
            if self.ccxt_service:
                await self.ccxt_service.close()
            
            if self.db_connection:
                await self.db_connection.close()
            
            if self.trading_engine:
                await self.trading_engine.close()
            
            if self.strategy_manager:
                await self.strategy_manager.close()
            
            if self.order_manager:
                await self.order_manager.close()
            
            self.logger.info("Enhanced Real-Time Pipeline closed successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to close pipeline: {e}")
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()

    # Performance Optimization Methods
    async def _micro_batch_processor(self):
        """Process data in micro-batches for improved performance"""
        while self.is_running:
            try:
                start_time = time.time()
                batch_data = {}
                
                # Collect data for micro-batching
                for symbol in self.symbols:
                    symbol_batch = {}
                    for data_type in ['order_book', 'market_data', 'liquidation']:
                        if symbol in self.data_buffers and data_type in self.data_buffers[symbol]:
                            buffer = self.data_buffers[symbol][data_type]
                            if len(buffer) >= self.micro_batch_size:
                                symbol_batch[data_type] = [buffer.popleft() for _ in range(self.micro_batch_size)]
                    
                    if symbol_batch:
                        batch_data[symbol] = symbol_batch
                
                # Process batch if we have data
                if batch_data:
                    await self._process_micro_batch(batch_data)
                    self.performance_metrics['batch_updates'] += 1
                
                # Measure processing time
                processing_time = time.time() - start_time
                self.performance_metrics['processing_times'].append(processing_time)
                
                # Wait for next batch cycle
                await asyncio.sleep(self.micro_batch_timeout)
                
            except Exception as e:
                self.logger.error(f"Error in micro-batch processor: {e}")
                await asyncio.sleep(1.0)

    async def _process_micro_batch(self, batch_data: Dict[str, Dict[str, List]]):
        """Process a micro-batch of data"""
        try:
            start_time = time.time()
            
            # Process in parallel if enabled
            if self.parallel_processing:
                tasks = []
                for symbol, data_types in batch_data.items():
                    for data_type, data_list in data_types.items():
                        task = asyncio.create_task(
                            self._process_data_batch(symbol, data_type, data_list)
                        )
                        tasks.append(task)
                
                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)
            else:
                # Sequential processing
                for symbol, data_types in batch_data.items():
                    for data_type, data_list in data_types.items():
                        await self._process_data_batch(symbol, data_type, data_list)
            
            # Update performance metrics
            processing_time = time.time() - start_time
            self.performance_metrics['latency_measurements'].append(processing_time)
            
        except Exception as e:
            self.logger.error(f"Error processing micro-batch: {e}")

    async def _process_data_batch(self, symbol: str, data_type: str, data_list: List):
        """Process a batch of data for a specific symbol and type"""
        try:
            if data_type == 'order_book':
                for order_book in data_list:
                    await self._process_order_book(order_book)
            elif data_type == 'market_data':
                for market_data in data_list:
                    await self._process_market_data(market_data)
            elif data_type == 'liquidation':
                await self._process_liquidation_events(data_list)
            
            # Update cache if enabled
            if self.memory_cache_enabled:
                cache_key = f"{symbol}_{data_type}"
                self.memory_cache[cache_key] = {
                    'data': data_list[-1] if data_list else None,
                    'timestamp': time.time(),
                    'count': len(data_list)
                }
                
        except Exception as e:
            self.logger.error(f"Error processing {data_type} batch for {symbol}: {e}")

    async def _cache_cleanup_loop(self):
        """Clean up expired cache entries"""
        while self.is_running:
            try:
                current_time = time.time()
                expired_keys = []
                
                for key, value in self.memory_cache.items():
                    if current_time - value['timestamp'] > self.cache_ttl:
                        expired_keys.append(key)
                
                for key in expired_keys:
                    del self.memory_cache[key]
                
                if expired_keys:
                    self.logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
                
                await asyncio.sleep(self.cache_cleanup_interval)
                
            except Exception as e:
                self.logger.error(f"Error in cache cleanup: {e}")
                await asyncio.sleep(10.0)

    def get_cached_data(self, symbol: str, data_type: str):
        """Get cached data for a symbol and data type"""
        if not self.memory_cache_enabled:
            return None
        
        cache_key = f"{symbol}_{data_type}"
        cached_data = self.memory_cache.get(cache_key)
        
        if cached_data and time.time() - cached_data['timestamp'] <= self.cache_ttl:
            self.performance_metrics['cache_hits'] += 1
            return cached_data['data']
        else:
            self.performance_metrics['cache_misses'] += 1
            return None

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get detailed performance metrics"""
        processing_times = list(self.performance_metrics['processing_times'])
        latency_measurements = list(self.performance_metrics['latency_measurements'])
        
        return {
            'total_updates': self.performance_metrics['total_updates'],
            'batch_updates': self.performance_metrics['batch_updates'],
            'cache_hits': self.performance_metrics['cache_hits'],
            'cache_misses': self.performance_metrics['cache_misses'],
            'cache_hit_rate': (
                self.performance_metrics['cache_hits'] / 
                (self.performance_metrics['cache_hits'] + self.performance_metrics['cache_misses'])
                if (self.performance_metrics['cache_hits'] + self.performance_metrics['cache_misses']) > 0 
                else 0
            ),
            'avg_processing_time': np.mean(processing_times) if processing_times else 0,
            'avg_latency': np.mean(latency_measurements) if latency_measurements else 0,
            'p95_latency': np.percentile(latency_measurements, 95) if latency_measurements else 0,
            'p99_latency': np.percentile(latency_measurements, 99) if latency_measurements else 0,
            'cache_size': len(self.memory_cache),
            'micro_batch_size': self.micro_batch_size,
            'micro_batch_timeout': self.micro_batch_timeout,
        }

    async def optimize_performance(self):
        """Dynamically optimize performance based on metrics"""
        try:
            metrics = self.get_performance_metrics()
            
            # Adjust micro-batch size based on latency
            if metrics['avg_latency'] > 0.05:  # 50ms threshold
                self.micro_batch_size = min(self.micro_batch_size + 2, 20)
                self.logger.info(f"Increased micro-batch size to {self.micro_batch_size}")
            elif metrics['avg_latency'] < 0.01:  # 10ms threshold
                self.micro_batch_size = max(self.micro_batch_size - 1, 5)
                self.logger.info(f"Decreased micro-batch size to {self.micro_batch_size}")
            
            # Adjust cache TTL based on hit rate
            if metrics['cache_hit_rate'] < 0.7:
                self.cache_ttl = min(self.cache_ttl + 1.0, 10.0)
                self.logger.info(f"Increased cache TTL to {self.cache_ttl}s")
            elif metrics['cache_hit_rate'] > 0.9:
                self.cache_ttl = max(self.cache_ttl - 0.5, 2.0)
                self.logger.info(f"Decreased cache TTL to {self.cache_ttl}s")
            
            # Log optimization results
            self.logger.info(f"Performance optimization completed: {metrics}")
            
        except Exception as e:
            self.logger.error(f"Error in performance optimization: {e}")

    # ===== SMART MONEY CONCEPTS (SMC) ANALYZER =====
    
    def analyze_smc_patterns(self, df: pd.DataFrame, symbol: str, timeframe: str) -> Dict[str, Any]:
        """
        Analyze Smart Money Concepts patterns including Order Blocks, Fair Value Gaps, and Liquidity Sweeps
        
        Args:
            df: DataFrame with OHLCV data
            symbol: Trading symbol
            timeframe: Timeframe for analysis
            
        Returns:
            Dictionary containing SMC analysis results
        """
        try:
            if len(df) < 50:  # Need sufficient data for SMC analysis
                return {
                    'order_blocks': [],
                    'fair_value_gaps': [],
                    'liquidity_sweeps': [],
                    'market_structure': [],
                    'confidence': 0.0
                }
            
            results = {
                'order_blocks': self._detect_smc_order_blocks(df, symbol, timeframe),
                'fair_value_gaps': self._detect_smc_fair_value_gaps(df, symbol, timeframe),
                'liquidity_sweeps': self._detect_smc_liquidity_sweeps(df, symbol, timeframe),
                'market_structure': self._detect_smc_market_structure(df, symbol, timeframe),
                'confidence': 0.0
            }
            
            # Calculate overall confidence
            total_patterns = (
                len(results['order_blocks']) + 
                len(results['fair_value_gaps']) + 
                len(results['liquidity_sweeps']) + 
                len(results['market_structure'])
            )
            
            if total_patterns > 0:
                avg_confidence = np.mean([
                    np.mean([ob.confidence for ob in results['order_blocks']]) if results['order_blocks'] else 0,
                    np.mean([fvg.confidence for fvg in results['fair_value_gaps']]) if results['fair_value_gaps'] else 0,
                    np.mean([ls.confidence for ls in results['liquidity_sweeps']]) if results['liquidity_sweeps'] else 0,
                    np.mean([ms.confidence for ms in results['market_structure']]) if results['market_structure'] else 0
                ])
                results['confidence'] = avg_confidence
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in SMC pattern analysis: {e}")
            return {
                'order_blocks': [],
                'fair_value_gaps': [],
                'liquidity_sweeps': [],
                'market_structure': [],
                'confidence': 0.0
            }
    
    def _detect_smc_order_blocks(self, df: pd.DataFrame, symbol: str, timeframe: str) -> List[SMCOrderBlock]:
        """
        Detect SMC Order Blocks (institutional order placement zones)
        
        Order Blocks: Strong moves followed by consolidation where smart money places orders
        """
        try:
            order_blocks = []
            
            # Get recent price action
            recent_df = df.tail(30)
            highs = recent_df['high'].values
            lows = recent_df['low'].values
            opens = recent_df['open'].values
            closes = recent_df['close'].values
            volumes = recent_df['volume'].values
            
            # Find strong moves (potential order blocks)
            for i in range(5, len(recent_df) - 5):
                # Calculate move strength
                body_size = abs(closes[i] - opens[i])
                total_range = highs[i] - lows[i]
                body_ratio = body_size / total_range if total_range > 0 else 0
                
                # Volume confirmation
                avg_volume = np.mean(volumes[max(0, i-5):i+5])
                volume_ratio = volumes[i] / avg_volume if avg_volume > 0 else 1.0
                
                # Check for strong move (potential order block)
                if body_ratio > 0.6 and volume_ratio > 1.2:  # Strong body with high volume
                    # Determine block type
                    if closes[i] > opens[i]:  # Bullish order block
                        block_type = 'bullish'
                        # Look for consolidation after the move
                        consolidation_found = False
                        for j in range(i + 1, min(i + 10, len(closes))):
                            if abs(closes[j] - opens[j]) / (highs[j] - lows[j]) < 0.3:  # Small body
                                consolidation_found = True
                                break
                        
                        if consolidation_found:
                            # Calculate order block strength
                            strength = min(1.0, body_ratio * volume_ratio / 2.0)
                            confidence = min(1.0, strength * 0.8)
                            
                            # Find fair value gaps within the order block
                            fair_value_gaps = self._find_fair_value_gaps_in_range(
                                df, lows[i], highs[i], i, i + 5
                            )
                            
                            order_block = SMCOrderBlock(
                                symbol=symbol,
                                timestamp=df.index[i] if hasattr(df.index[i], 'to_pydatetime') else datetime.now(),
                                timeframe=timeframe,
                                block_type=block_type,
                                high=highs[i],
                                low=lows[i],
                                open=opens[i],
                                close=closes[i],
                                volume=volumes[i],
                                strength=strength,
                                confidence=confidence,
                                fair_value_gaps=fair_value_gaps,
                                metadata={
                                    'body_ratio': body_ratio,
                                    'volume_ratio': volume_ratio,
                                    'consolidation_found': consolidation_found
                                }
                            )
                            order_blocks.append(order_block)
                    
                    elif closes[i] < opens[i]:  # Bearish order block
                        block_type = 'bearish'
                        # Look for consolidation after the move
                        consolidation_found = False
                        for j in range(i + 1, min(i + 10, len(closes))):
                            if abs(closes[j] - opens[j]) / (highs[j] - lows[j]) < 0.3:  # Small body
                                consolidation_found = True
                                break
                        
                        if consolidation_found:
                            # Calculate order block strength
                            strength = min(1.0, body_ratio * volume_ratio / 2.0)
                            confidence = min(1.0, strength * 0.8)
                            
                            # Find fair value gaps within the order block
                            fair_value_gaps = self._find_fair_value_gaps_in_range(
                                df, lows[i], highs[i], i, i + 5
                            )
                            
                            order_block = SMCOrderBlock(
                                symbol=symbol,
                                timestamp=df.index[i] if hasattr(df.index[i], 'to_pydatetime') else datetime.now(),
                                timeframe=timeframe,
                                block_type=block_type,
                                high=highs[i],
                                low=lows[i],
                                open=opens[i],
                                close=closes[i],
                                volume=volumes[i],
                                strength=strength,
                                confidence=confidence,
                                fair_value_gaps=fair_value_gaps,
                                metadata={
                                    'body_ratio': body_ratio,
                                    'volume_ratio': volume_ratio,
                                    'consolidation_found': consolidation_found
                                }
                            )
                            order_blocks.append(order_block)
            
            return order_blocks
            
        except Exception as e:
            self.logger.error(f"Error detecting SMC Order Blocks: {e}")
            return []
    
    def _detect_smc_fair_value_gaps(self, df: pd.DataFrame, symbol: str, timeframe: str) -> List[SMCFairValueGap]:
        """
        Detect SMC Fair Value Gaps (imbalances that get filled)
        
        Fair Value Gaps: Price gaps that represent imbalances in supply/demand
        """
        try:
            fair_value_gaps = []
            
            # Get recent price action
            recent_df = df.tail(20)
            highs = recent_df['high'].values
            lows = recent_df['low'].values
            opens = recent_df['open'].values
            closes = recent_df['close'].values
            
            # Find fair value gaps
            for i in range(1, len(recent_df) - 1):
                current_high = highs[i]
                current_low = lows[i]
                prev_high = highs[i-1]
                prev_low = lows[i-1]
                next_high = highs[i+1]
                next_low = lows[i+1]
                
                # Bullish Fair Value Gap: Current low > Previous high
                if current_low > prev_high:
                    gap_size = current_low - prev_high
                    gap_ratio = gap_size / prev_high if prev_high > 0 else 0
                    
                    if gap_ratio > 0.001:  # Minimum gap size (0.1%)
                        # Calculate fill probability based on gap size and market conditions
                        fill_probability = max(0.1, 1.0 - gap_ratio * 10)  # Smaller gaps more likely to fill
                        strength = min(1.0, gap_ratio * 100)  # Larger gaps = stronger signal
                        
                        fair_value_gap = SMCFairValueGap(
                            symbol=symbol,
                            timestamp=df.index[i] if hasattr(df.index[i], 'to_pydatetime') else datetime.now(),
                            timeframe=timeframe,
                            gap_type='bullish',
                            high=current_high,
                            low=current_low,
                            gap_size=gap_size,
                            fill_probability=fill_probability,
                            strength=strength,
                            confidence=strength,  # Use strength as confidence
                            metadata={
                                'gap_ratio': gap_ratio,
                                'prev_high': prev_high,
                                'current_low': current_low
                            }
                        )
                        fair_value_gaps.append(fair_value_gap)
                
                # Bearish Fair Value Gap: Current high < Previous low
                elif current_high < prev_low:
                    gap_size = prev_low - current_high
                    gap_ratio = gap_size / prev_low if prev_low > 0 else 0
                    
                    if gap_ratio > 0.001:  # Minimum gap size (0.1%)
                        # Calculate fill probability based on gap size and market conditions
                        fill_probability = max(0.1, 1.0 - gap_ratio * 10)  # Smaller gaps more likely to fill
                        strength = min(1.0, gap_ratio * 100)  # Larger gaps = stronger signal
                        
                        fair_value_gap = SMCFairValueGap(
                            symbol=symbol,
                            timestamp=df.index[i] if hasattr(df.index[i], 'to_pydatetime') else datetime.now(),
                            timeframe=timeframe,
                            gap_type='bearish',
                            high=current_high,
                            low=current_low,
                            gap_size=gap_size,
                            fill_probability=fill_probability,
                            strength=strength,
                            confidence=strength,  # Use strength as confidence
                            metadata={
                                'gap_ratio': gap_ratio,
                                'prev_low': prev_low,
                                'current_high': current_high
                            }
                        )
                        fair_value_gaps.append(fair_value_gap)
            
            return fair_value_gaps
            
        except Exception as e:
            self.logger.error(f"Error detecting SMC Fair Value Gaps: {e}")
            return []
    
    def _detect_smc_liquidity_sweeps(self, df: pd.DataFrame, symbol: str, timeframe: str) -> List[SMCLiquiditySweep]:
        """
        Detect SMC Liquidity Sweeps (stop hunting before reversals)
        
        Liquidity Sweeps: Price moves beyond support/resistance to trigger stops before reversing
        """
        try:
            liquidity_sweeps = []
            
            # Get recent price action
            recent_df = df.tail(30)
            highs = recent_df['high'].values
            lows = recent_df['low'].values
            opens = recent_df['open'].values
            closes = recent_df['close'].values
            volumes = recent_df['volume'].values
            
            # Find support and resistance levels
            support_levels = self._find_support_levels(lows)
            resistance_levels = self._find_resistance_levels(highs)
            
            # Check for liquidity sweeps
            for i in range(5, len(recent_df) - 5):
                current_high = highs[i]
                current_low = lows[i]
                current_close = closes[i]
                
                # Check for bullish liquidity sweep (sweep below support then reversal)
                for support_level in support_levels:
                    if current_low < support_level * 0.995:  # Sweep below support
                        # Check for reversal in next few candles
                        reversal_found = False
                        for j in range(i + 1, min(i + 5, len(closes))):
                            if closes[j] > support_level:
                                reversal_found = True
                                break
                        
                        if reversal_found:
                            # Calculate sweep strength
                            sweep_depth = (support_level - current_low) / support_level
                            sweep_strength = min(1.0, sweep_depth * 20)  # Deeper sweep = stronger
                            reversal_probability = 0.7  # Base probability for successful sweep
                            
                            liquidity_sweep = SMCLiquiditySweep(
                                symbol=symbol,
                                timestamp=df.index[i] if hasattr(df.index[i], 'to_pydatetime') else datetime.now(),
                                timeframe=timeframe,
                                sweep_type='bullish',
                                price_level=support_level,
                                volume=volumes[i],
                                sweep_strength=sweep_strength,
                                reversal_probability=reversal_probability,
                                confidence=sweep_strength,  # Use sweep strength as confidence
                                metadata={
                                    'sweep_depth': sweep_depth,
                                    'support_level': support_level,
                                    'reversal_found': reversal_found
                                }
                            )
                            liquidity_sweeps.append(liquidity_sweep)
                            break  # Only count one sweep per support level
                
                # Check for bearish liquidity sweep (sweep above resistance then reversal)
                for resistance_level in resistance_levels:
                    if current_high > resistance_level * 1.005:  # Sweep above resistance
                        # Check for reversal in next few candles
                        reversal_found = False
                        for j in range(i + 1, min(i + 5, len(closes))):
                            if closes[j] < resistance_level:
                                reversal_found = True
                                break
                        
                        if reversal_found:
                            # Calculate sweep strength
                            sweep_height = (current_high - resistance_level) / resistance_level
                            sweep_strength = min(1.0, sweep_height * 20)  # Higher sweep = stronger
                            reversal_probability = 0.7  # Base probability for successful sweep
                            
                            liquidity_sweep = SMCLiquiditySweep(
                                symbol=symbol,
                                timestamp=df.index[i] if hasattr(df.index[i], 'to_pydatetime') else datetime.now(),
                                timeframe=timeframe,
                                sweep_type='bearish',
                                price_level=resistance_level,
                                volume=volumes[i],
                                sweep_strength=sweep_strength,
                                reversal_probability=reversal_probability,
                                confidence=sweep_strength,  # Use sweep strength as confidence
                                metadata={
                                    'sweep_height': sweep_height,
                                    'resistance_level': resistance_level,
                                    'reversal_found': reversal_found
                                }
                            )
                            liquidity_sweeps.append(liquidity_sweep)
                            break  # Only count one sweep per resistance level
            
            return liquidity_sweeps
            
        except Exception as e:
            self.logger.error(f"Error detecting SMC Liquidity Sweeps: {e}")
            return []
    
    def _detect_smc_market_structure(self, df: pd.DataFrame, symbol: str, timeframe: str) -> List[SMCMarketStructure]:
        """
        Detect SMC Market Structure (BOS, CHoCH, etc.)
        
        Market Structure: Break of Structure (BOS) and Change of Character (CHoCH)
        """
        try:
            market_structures = []
            
            # Get recent price action
            recent_df = df.tail(50)
            highs = recent_df['high'].values
            lows = recent_df['low'].values
            closes = recent_df['close'].values
            
            # Find swing highs and lows
            swing_highs = self._find_swing_highs(highs)
            swing_lows = self._find_swing_lows(lows)
            
            # Detect Break of Structure (BOS)
            for i in range(10, len(recent_df) - 5):
                current_high = highs[i]
                current_low = lows[i]
                current_close = closes[i]
                
                # Bullish BOS: Break above previous swing high
                for swing_high in swing_highs:
                    if swing_high < current_high and swing_high < current_close:
                        # Check if this is a significant break
                        break_size = (current_close - swing_high) / swing_high
                        if break_size > 0.005:  # 0.5% break
                            strength = min(1.0, break_size * 100)
                            confidence = min(1.0, strength * 0.8)
                            
                            market_structure = SMCMarketStructure(
                                symbol=symbol,
                                timestamp=df.index[i] if hasattr(df.index[i], 'to_pydatetime') else datetime.now(),
                                timeframe=timeframe,
                                structure_type='BOS',
                                price_level=swing_high,
                                direction='bullish',
                                strength=strength,
                                confidence=confidence,
                                metadata={
                                    'break_size': break_size,
                                    'swing_high': swing_high,
                                    'current_close': current_close
                                }
                            )
                            market_structures.append(market_structure)
                            break
                
                # Bearish BOS: Break below previous swing low
                for swing_low in swing_lows:
                    if swing_low > current_low and swing_low > current_close:
                        # Check if this is a significant break
                        break_size = (swing_low - current_close) / swing_low
                        if break_size > 0.005:  # 0.5% break
                            strength = min(1.0, break_size * 100)
                            confidence = min(1.0, strength * 0.8)
                            
                            market_structure = SMCMarketStructure(
                                symbol=symbol,
                                timestamp=df.index[i] if hasattr(df.index[i], 'to_pydatetime') else datetime.now(),
                                timeframe=timeframe,
                                structure_type='BOS',
                                price_level=swing_low,
                                direction='bearish',
                                strength=strength,
                                confidence=confidence,
                                metadata={
                                    'break_size': break_size,
                                    'swing_low': swing_low,
                                    'current_close': current_close
                                }
                            )
                            market_structures.append(market_structure)
                            break
            
            return market_structures
            
        except Exception as e:
            self.logger.error(f"Error detecting SMC Market Structure: {e}")
            return []
    
    def _find_fair_value_gaps_in_range(self, df: pd.DataFrame, low: float, high: float, start_idx: int, end_idx: int) -> List[Dict[str, Any]]:
        """Find fair value gaps within a specific price range and time range"""
        try:
            gaps = []
            recent_df = df.iloc[start_idx:end_idx]
            
            for i in range(1, len(recent_df)):
                current_low = recent_df['low'].iloc[i]
                prev_high = recent_df['high'].iloc[i-1]
                
                # Check for gaps within the order block range
                if low <= current_low <= high and low <= prev_high <= high:
                    if current_low > prev_high:  # Bullish gap
                        gap_size = current_low - prev_high
                        gaps.append({
                            'type': 'bullish',
                            'size': gap_size,
                            'high': prev_high,
                            'low': current_low
                        })
            
            return gaps
            
        except Exception as e:
            self.logger.error(f"Error finding fair value gaps in range: {e}")
            return []
    
    def _find_support_levels(self, lows: np.ndarray) -> List[float]:
        """Find support levels from price lows"""
        try:
            levels = []
            for i in range(2, len(lows) - 2):
                if lows[i] < lows[i-1] and lows[i] < lows[i-2] and lows[i] < lows[i+1] and lows[i] < lows[i+2]:
                    levels.append(lows[i])
            return sorted(list(set(levels)))[:5]  # Return top 5 support levels
        except Exception as e:
            self.logger.error(f"Error finding support levels: {e}")
            return []
    
    def _find_resistance_levels(self, highs: np.ndarray) -> List[float]:
        """Find resistance levels from price highs"""
        try:
            levels = []
            for i in range(2, len(highs) - 2):
                if highs[i] > highs[i-1] and highs[i] > highs[i-2] and highs[i] > highs[i+1] and highs[i] > highs[i+2]:
                    levels.append(highs[i])
            return sorted(list(set(levels)))[:5]  # Return top 5 resistance levels
        except Exception as e:
            self.logger.error(f"Error finding resistance levels: {e}")
            return []
    
    def _find_swing_highs(self, highs: np.ndarray) -> List[float]:
        """Find swing highs"""
        try:
            swing_highs = []
            for i in range(2, len(highs) - 2):
                if highs[i] > highs[i-1] and highs[i] > highs[i-2] and highs[i] > highs[i+1] and highs[i] > highs[i+2]:
                    swing_highs.append(highs[i])
            return swing_highs
        except Exception as e:
            self.logger.error(f"Error finding swing highs: {e}")
            return []
    
    def _find_swing_lows(self, lows: np.ndarray) -> List[float]:
        """Find swing lows"""
        try:
            swing_lows = []
            for i in range(2, len(lows) - 2):
                if lows[i] < lows[i-1] and lows[i] < lows[i-2] and lows[i] < lows[i+1] and lows[i] < lows[i+2]:
                    swing_lows.append(lows[i])
            return swing_lows
        except Exception as e:
            self.logger.error(f"Error finding swing lows: {e}")
            return []
