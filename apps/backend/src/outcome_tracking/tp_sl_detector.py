"""
Take Profit/Stop Loss Detector for AlphaPulse
Precision detection of TP/SL hits with partial position tracking
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from enum import Enum
import uuid
import json

# Import existing components
try:
    from ..src.database.connection import TimescaleDBConnection
    from ..src.core.config import settings
except ImportError:
    try:
        from src.database.connection import TimescaleDBConnection
        from src.core.config import settings
    except ImportError:
        # Fallback classes for testing
        class TimescaleDBConnection:
            def __init__(self, config=None):
                self.config = config or {}
                self.is_initialized = False
            async def initialize(self): self.is_initialized = True
            async def shutdown(self): self.is_initialized = False
            async def close(self): self.is_initialized = False
        
        class settings:
            TIMESCALEDB_HOST = 'localhost'
            TIMESCALEDB_PORT = 5432
            TIMESCALEDB_DATABASE = 'alphapulse'
            TIMESCALEDB_USERNAME = 'alpha_emon'
            TIMESCALEDB_PASSWORD = 'Emon_@17711'

logger = logging.getLogger(__name__)

class HitType(Enum):
    """TP/SL hit types"""
    TAKE_PROFIT = 'take_profit'
    STOP_LOSS = 'stop_loss'
    PARTIAL_TP = 'partial_tp'
    PARTIAL_SL = 'partial_sl'
    TRAILING_STOP = 'trailing_stop'

class HitPrecision(Enum):
    """Hit precision levels"""
    EXACT = 'exact'  # Price exactly at TP/SL level
    ABOVE = 'above'  # Price above TP level
    BELOW = 'below'  # Price below SL level
    GAP = 'gap'      # Price gapped through level

@dataclass
class TPSLHit:
    """TP/SL hit data structure"""
    signal_id: str
    hit_type: HitType
    hit_price: float
    hit_timestamp: datetime
    precision: HitPrecision
    hit_delay_ms: float
    partial_fill_amount: Optional[float] = None
    remaining_position: Optional[float] = None
    hit_confidence: float = 1.0
    market_conditions: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

@dataclass
class TPSLConfig:
    """TP/SL configuration"""
    take_profit_price: Optional[float] = None
    stop_loss_price: Optional[float] = None
    partial_tp_prices: List[float] = field(default_factory=list)
    partial_sl_prices: List[float] = field(default_factory=list)
    trailing_stop_config: Optional[Dict[str, Any]] = None
    hit_tolerance: float = 0.001  # 0.1% tolerance for hit detection
    min_hit_duration: timedelta = timedelta(milliseconds=100)  # Minimum time price must stay at level
    enable_partial_fills: bool = True
    enable_trailing_stops: bool = False

class TPSLDetector:
    """
    Take Profit/Stop Loss detector with precision timing
    
    Features:
    - Precision TP/SL hit detection
    - Partial position tracking
    - Trailing stop support
    - Hit delay measurement
    - Market condition analysis
    - TimescaleDB integration
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Detection settings
        self.hit_tolerance = self.config.get('hit_tolerance', 0.001)
        self.min_hit_duration = self.config.get('min_hit_duration', timedelta(milliseconds=100))
        self.enable_partial_fills = self.config.get('enable_partial_fills', True)
        self.enable_trailing_stops = self.config.get('enable_trailing_stops', False)
        
        # Performance settings
        self.detection_interval = self.config.get('detection_interval', 0.1)  # seconds
        self.price_history_size = self.config.get('price_history_size', 1000)
        
        # State management
        self.is_running = False
        self.active_positions = {}  # signal_id -> position_data
        self.price_history = {}  # symbol -> price_history
        self.hit_history = []
        self.detection_task = None
        
        # Component references
        self.timescaledb = None
        
        # Callbacks
        self.hit_callbacks = []
        self.alert_callbacks = []
        
        logger.info("TPSLDetector initialized")
    
    async def initialize(self):
        """Initialize the TP/SL detector"""
        try:
            # Initialize TimescaleDB connection
            await self._initialize_timescaledb()
            
            # Start detection loop
            await self._start_detection_loop()
            
            self.is_running = True
            logger.info("✅ TPSLDetector initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize TPSLDetector: {e}")
            raise
    
    async def _initialize_timescaledb(self):
        """Initialize TimescaleDB connection"""
        try:
            self.timescaledb = TimescaleDBConnection({
                'host': settings.TIMESCALEDB_HOST,
                'port': settings.TIMESCALEDB_PORT,
                'database': settings.TIMESCALEDB_DATABASE,
                'username': settings.TIMESCALEDB_USERNAME,
                'password': settings.TIMESCALEDB_PASSWORD,
                'pool_size': 5,
                'max_overflow': 10
            })
            
            await self.timescaledb.initialize()
            logger.info("✅ TimescaleDB connection initialized for TP/SL detector")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize TimescaleDB: {e}")
            self.timescaledb = None
    
    async def _start_detection_loop(self):
        """Start the detection loop"""
        self.detection_task = asyncio.create_task(self._detection_worker())
        logger.info("✅ Detection loop started")
    
    async def _detection_worker(self):
        """Main detection worker loop"""
        while self.is_running:
            try:
                # Process active positions
                await self._process_active_positions()
                
                # Sleep for detection interval
                await asyncio.sleep(self.detection_interval)
                
            except Exception as e:
                logger.error(f"Detection worker error: {e}")
                await asyncio.sleep(1)  # Wait before retrying
    
    async def _process_active_positions(self):
        """Process all active positions for TP/SL detection"""
        current_time = datetime.now(timezone.utc)
        
        for signal_id, position_data in list(self.active_positions.items()):
            try:
                # Get current price
                current_price = await self._get_current_price(position_data['symbol'])
                
                if not current_price:
                    continue
                
                # Update price history
                await self._update_price_history(position_data['symbol'], current_price, current_time)
                
                # Check for TP/SL hits
                hits = await self._check_tpsl_hits(signal_id, position_data, current_price, current_time)
                
                for hit in hits:
                    await self._record_hit(hit)
                    
                    # Update position based on hit
                    await self._update_position_after_hit(signal_id, position_data, hit)
                    
                    # Check if position is fully closed
                    if position_data.get('remaining_position', 0) <= 0:
                        del self.active_positions[signal_id]
                        logger.info(f"✅ Position {signal_id} fully closed")
                
            except Exception as e:
                logger.error(f"Error processing position {signal_id}: {e}")
    
    async def _get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price for symbol"""
        try:
            # This would integrate with streaming data
            # For now, return a mock price
            return 100.0  # Mock price
            
        except Exception as e:
            logger.error(f"Error getting current price for {symbol}: {e}")
            return None
    
    async def _update_price_history(self, symbol: str, price: float, timestamp: datetime):
        """Update price history for symbol"""
        try:
            if symbol not in self.price_history:
                self.price_history[symbol] = []
            
            self.price_history[symbol].append({
                'price': price,
                'timestamp': timestamp
            })
            
            # Keep only recent history
            if len(self.price_history[symbol]) > self.price_history_size:
                self.price_history[symbol] = self.price_history[symbol][-self.price_history_size:]
                
        except Exception as e:
            logger.error(f"Error updating price history for {symbol}: {e}")
    
    async def _check_tpsl_hits(self, signal_id: str, position_data: Dict[str, Any], 
                              current_price: float, current_time: datetime) -> List[TPSLHit]:
        """Check for TP/SL hits"""
        hits = []
        
        try:
            config = position_data.get('tpsl_config', TPSLConfig())
            position_size = position_data.get('position_size', 1.0)
            remaining_position = position_data.get('remaining_position', position_size)
            
            if remaining_position <= 0:
                return hits
            
            # Check take profit hits
            if config.take_profit_price:
                hit = await self._check_tp_hit(
                    signal_id, position_data, current_price, current_time, 
                    config.take_profit_price, remaining_position
                )
                if hit:
                    hits.append(hit)
            
            # Check stop loss hits
            if config.stop_loss_price:
                hit = await self._check_sl_hit(
                    signal_id, position_data, current_price, current_time,
                    config.stop_loss_price, remaining_position
                )
                if hit:
                    hits.append(hit)
            
            # Check partial TP hits
            if self.enable_partial_fills and config.partial_tp_prices:
                for tp_price in config.partial_tp_prices:
                    hit = await self._check_partial_tp_hit(
                        signal_id, position_data, current_price, current_time,
                        tp_price, remaining_position
                    )
                    if hit:
                        hits.append(hit)
            
            # Check partial SL hits
            if self.enable_partial_fills and config.partial_sl_prices:
                for sl_price in config.partial_sl_prices:
                    hit = await self._check_partial_sl_hit(
                        signal_id, position_data, current_price, current_time,
                        sl_price, remaining_position
                    )
                    if hit:
                        hits.append(hit)
            
            # Check trailing stop hits
            if self.enable_trailing_stops and config.trailing_stop_config:
                hit = await self._check_trailing_stop_hit(
                    signal_id, position_data, current_price, current_time
                )
                if hit:
                    hits.append(hit)
            
            return hits
            
        except Exception as e:
            logger.error(f"Error checking TP/SL hits for signal {signal_id}: {e}")
            return hits
    
    async def _check_tp_hit(self, signal_id: str, position_data: Dict[str, Any], 
                           current_price: float, current_time: datetime,
                           tp_price: float, remaining_position: float) -> Optional[TPSLHit]:
        """Check for take profit hit"""
        try:
            signal_type = position_data.get('signal_type', 'long')
            
            # Check if price hit TP level
            if signal_type == 'long' and current_price >= tp_price * (1 - self.hit_tolerance):
                # Verify hit duration
                if await self._verify_hit_duration(position_data['symbol'], tp_price, current_time):
                    hit_delay = await self._calculate_hit_delay(signal_id, tp_price, current_time)
                    precision = await self._determine_hit_precision(current_price, tp_price)
                    
                    return TPSLHit(
                        signal_id=signal_id,
                        hit_type=HitType.TAKE_PROFIT,
                        hit_price=tp_price,
                        hit_timestamp=current_time,
                        precision=precision,
                        hit_delay_ms=hit_delay,
                        partial_fill_amount=remaining_position,
                        remaining_position=0.0,
                        hit_confidence=1.0,
                        market_conditions=await self._analyze_market_conditions(position_data['symbol'])
                    )
            
            elif signal_type == 'short' and current_price <= tp_price * (1 + self.hit_tolerance):
                # Verify hit duration
                if await self._verify_hit_duration(position_data['symbol'], tp_price, current_time):
                    hit_delay = await self._calculate_hit_delay(signal_id, tp_price, current_time)
                    precision = await self._determine_hit_precision(current_price, tp_price)
                    
                    return TPSLHit(
                        signal_id=signal_id,
                        hit_type=HitType.TAKE_PROFIT,
                        hit_price=tp_price,
                        hit_timestamp=current_time,
                        precision=precision,
                        hit_delay_ms=hit_delay,
                        partial_fill_amount=remaining_position,
                        remaining_position=0.0,
                        hit_confidence=1.0,
                        market_conditions=await self._analyze_market_conditions(position_data['symbol'])
                    )
            
            return None
            
        except Exception as e:
            logger.error(f"Error checking TP hit for signal {signal_id}: {e}")
            return None
    
    async def _check_sl_hit(self, signal_id: str, position_data: Dict[str, Any], 
                           current_price: float, current_time: datetime,
                           sl_price: float, remaining_position: float) -> Optional[TPSLHit]:
        """Check for stop loss hit"""
        try:
            signal_type = position_data.get('signal_type', 'long')
            
            # Check if price hit SL level
            if signal_type == 'long' and current_price <= sl_price * (1 + self.hit_tolerance):
                # Verify hit duration
                if await self._verify_hit_duration(position_data['symbol'], sl_price, current_time):
                    hit_delay = await self._calculate_hit_delay(signal_id, sl_price, current_time)
                    precision = await self._determine_hit_precision(current_price, sl_price)
                    
                    return TPSLHit(
                        signal_id=signal_id,
                        hit_type=HitType.STOP_LOSS,
                        hit_price=sl_price,
                        hit_timestamp=current_time,
                        precision=precision,
                        hit_delay_ms=hit_delay,
                        partial_fill_amount=remaining_position,
                        remaining_position=0.0,
                        hit_confidence=1.0,
                        market_conditions=await self._analyze_market_conditions(position_data['symbol'])
                    )
            
            elif signal_type == 'short' and current_price >= sl_price * (1 - self.hit_tolerance):
                # Verify hit duration
                if await self._verify_hit_duration(position_data['symbol'], sl_price, current_time):
                    hit_delay = await self._calculate_hit_delay(signal_id, sl_price, current_time)
                    precision = await self._determine_hit_precision(current_price, sl_price)
                    
                    return TPSLHit(
                        signal_id=signal_id,
                        hit_type=HitType.STOP_LOSS,
                        hit_price=sl_price,
                        hit_timestamp=current_time,
                        precision=precision,
                        hit_delay_ms=hit_delay,
                        partial_fill_amount=remaining_position,
                        remaining_position=0.0,
                        hit_confidence=1.0,
                        market_conditions=await self._analyze_market_conditions(position_data['symbol'])
                    )
            
            return None
            
        except Exception as e:
            logger.error(f"Error checking SL hit for signal {signal_id}: {e}")
            return None
    
    async def _check_partial_tp_hit(self, signal_id: str, position_data: Dict[str, Any], 
                                   current_price: float, current_time: datetime,
                                   tp_price: float, remaining_position: float) -> Optional[TPSLHit]:
        """Check for partial take profit hit"""
        try:
            # Check if this partial TP was already hit
            partial_tps_hit = position_data.get('partial_tps_hit', [])
            if tp_price in partial_tps_hit:
                return None
            
            signal_type = position_data.get('signal_type', 'long')
            partial_size = position_data.get('partial_size', remaining_position * 0.5)
            
            # Check if price hit partial TP level
            if signal_type == 'long' and current_price >= tp_price * (1 - self.hit_tolerance):
                if await self._verify_hit_duration(position_data['symbol'], tp_price, current_time):
                    hit_delay = await self._calculate_hit_delay(signal_id, tp_price, current_time)
                    precision = await self._determine_hit_precision(current_price, tp_price)
                    
                    # Mark this partial TP as hit
                    partial_tps_hit.append(tp_price)
                    position_data['partial_tps_hit'] = partial_tps_hit
                    
                    return TPSLHit(
                        signal_id=signal_id,
                        hit_type=HitType.PARTIAL_TP,
                        hit_price=tp_price,
                        hit_timestamp=current_time,
                        precision=precision,
                        hit_delay_ms=hit_delay,
                        partial_fill_amount=partial_size,
                        remaining_position=remaining_position - partial_size,
                        hit_confidence=1.0,
                        market_conditions=await self._analyze_market_conditions(position_data['symbol'])
                    )
            
            return None
            
        except Exception as e:
            logger.error(f"Error checking partial TP hit for signal {signal_id}: {e}")
            return None
    
    async def _check_partial_sl_hit(self, signal_id: str, position_data: Dict[str, Any], 
                                   current_price: float, current_time: datetime,
                                   sl_price: float, remaining_position: float) -> Optional[TPSLHit]:
        """Check for partial stop loss hit"""
        try:
            # Check if this partial SL was already hit
            partial_sls_hit = position_data.get('partial_sls_hit', [])
            if sl_price in partial_sls_hit:
                return None
            
            signal_type = position_data.get('signal_type', 'long')
            partial_size = position_data.get('partial_size', remaining_position * 0.5)
            
            # Check if price hit partial SL level
            if signal_type == 'long' and current_price <= sl_price * (1 + self.hit_tolerance):
                if await self._verify_hit_duration(position_data['symbol'], sl_price, current_time):
                    hit_delay = await self._calculate_hit_delay(signal_id, sl_price, current_time)
                    precision = await self._determine_hit_precision(current_price, sl_price)
                    
                    # Mark this partial SL as hit
                    partial_sls_hit.append(sl_price)
                    position_data['partial_sls_hit'] = partial_sls_hit
                    
                    return TPSLHit(
                        signal_id=signal_id,
                        hit_type=HitType.PARTIAL_SL,
                        hit_price=sl_price,
                        hit_timestamp=current_time,
                        precision=precision,
                        hit_delay_ms=hit_delay,
                        partial_fill_amount=partial_size,
                        remaining_position=remaining_position - partial_size,
                        hit_confidence=1.0,
                        market_conditions=await self._analyze_market_conditions(position_data['symbol'])
                    )
            
            return None
            
        except Exception as e:
            logger.error(f"Error checking partial SL hit for signal {signal_id}: {e}")
            return None
    
    async def _check_trailing_stop_hit(self, signal_id: str, position_data: Dict[str, Any], 
                                      current_price: float, current_time: datetime) -> Optional[TPSLHit]:
        """Check for trailing stop hit"""
        try:
            trailing_config = position_data.get('tpsl_config', {}).get('trailing_stop_config')
            if not trailing_config:
                return None
            
            # Calculate current trailing stop level
            trailing_level = await self._calculate_trailing_stop_level(
                position_data['symbol'], position_data, trailing_config
            )
            
            if trailing_level:
                signal_type = position_data.get('signal_type', 'long')
                
                # Check if price hit trailing stop
                if signal_type == 'long' and current_price <= trailing_level * (1 + self.hit_tolerance):
                    if await self._verify_hit_duration(position_data['symbol'], trailing_level, current_time):
                        hit_delay = await self._calculate_hit_delay(signal_id, trailing_level, current_time)
                        precision = await self._determine_hit_precision(current_price, trailing_level)
                        
                        return TPSLHit(
                            signal_id=signal_id,
                            hit_type=HitType.TRAILING_STOP,
                            hit_price=trailing_level,
                            hit_timestamp=current_time,
                            precision=precision,
                            hit_delay_ms=hit_delay,
                            partial_fill_amount=position_data.get('remaining_position', 0),
                            remaining_position=0.0,
                            hit_confidence=1.0,
                            market_conditions=await self._analyze_market_conditions(position_data['symbol'])
                        )
            
            return None
            
        except Exception as e:
            logger.error(f"Error checking trailing stop hit for signal {signal_id}: {e}")
            return None
    
    async def _verify_hit_duration(self, symbol: str, price_level: float, current_time: datetime) -> bool:
        """Verify that price stayed at level for minimum duration"""
        try:
            if symbol not in self.price_history:
                return False
            
            # Check recent price history
            recent_prices = self.price_history[symbol][-10:]  # Last 10 price points
            
            hit_start_time = None
            for price_data in recent_prices:
                if abs(price_data['price'] - price_level) <= price_level * self.hit_tolerance:
                    if hit_start_time is None:
                        hit_start_time = price_data['timestamp']
                else:
                    hit_start_time = None
            
            if hit_start_time:
                duration = current_time - hit_start_time
                return duration >= self.min_hit_duration
            
            return False
            
        except Exception as e:
            logger.error(f"Error verifying hit duration: {e}")
            return False
    
    async def _calculate_hit_delay(self, signal_id: str, price_level: float, current_time: datetime) -> float:
        """Calculate delay between signal and hit"""
        try:
            # This would calculate the actual delay
            # For now, return a mock delay
            return 150.0  # Mock 150ms delay
            
        except Exception as e:
            logger.error(f"Error calculating hit delay: {e}")
            return 0.0
    
    async def _determine_hit_precision(self, current_price: float, target_price: float) -> HitPrecision:
        """Determine hit precision"""
        try:
            tolerance = target_price * self.hit_tolerance
            
            if abs(current_price - target_price) <= tolerance * 0.1:
                return HitPrecision.EXACT
            elif current_price > target_price:
                return HitPrecision.ABOVE
            elif current_price < target_price:
                return HitPrecision.BELOW
            else:
                return HitPrecision.GAP
                
        except Exception as e:
            logger.error(f"Error determining hit precision: {e}")
            return HitPrecision.EXACT
    
    async def _analyze_market_conditions(self, symbol: str) -> Dict[str, Any]:
        """Analyze market conditions at hit time"""
        try:
            if symbol not in self.price_history:
                return {}
            
            recent_prices = self.price_history[symbol][-20:]  # Last 20 price points
            
            if len(recent_prices) < 10:
                return {}
            
            prices = [p['price'] for p in recent_prices]
            
            # Calculate volatility
            price_changes = [abs(prices[i] - prices[i-1]) for i in range(1, len(prices))]
            volatility = sum(price_changes) / len(price_changes) if price_changes else 0
            
            # Calculate trend
            trend = (prices[-1] - prices[0]) / prices[0] if prices[0] != 0 else 0
            
            return {
                'volatility': volatility,
                'trend': trend,
                'price_range': max(prices) - min(prices),
                'avg_price': sum(prices) / len(prices),
                'sample_size': len(prices)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing market conditions: {e}")
            return {}
    
    async def _calculate_trailing_stop_level(self, symbol: str, position_data: Dict[str, Any], 
                                           trailing_config: Dict[str, Any]) -> Optional[float]:
        """Calculate current trailing stop level"""
        try:
            # This would implement trailing stop logic
            # For now, return None (no trailing stop)
            return None
            
        except Exception as e:
            logger.error(f"Error calculating trailing stop level: {e}")
            return None
    
    async def _record_hit(self, hit: TPSLHit):
        """Record TP/SL hit to database"""
        try:
            if not self.timescaledb:
                logger.warning("No database connection available")
                return
            
            # Insert into TimescaleDB
            async with self.timescaledb.get_session() as session:
                await session.execute("""
                    INSERT INTO tp_sl_hits (
                        signal_id, hit_type, hit_price, hit_timestamp, precision,
                        hit_delay_ms, partial_fill_amount, remaining_position,
                        hit_confidence, market_conditions, created_at
                    ) VALUES (
                        :signal_id, :hit_type, :hit_price, :hit_timestamp, :precision,
                        :hit_delay_ms, :partial_fill_amount, :remaining_position,
                        :hit_confidence, :market_conditions, :created_at
                    )
                """, {
                    'signal_id': hit.signal_id,
                    'hit_type': hit.hit_type.value,
                    'hit_price': hit.hit_price,
                    'hit_timestamp': hit.hit_timestamp,
                    'precision': hit.precision.value,
                    'hit_delay_ms': hit.hit_delay_ms,
                    'partial_fill_amount': hit.partial_fill_amount,
                    'remaining_position': hit.remaining_position,
                    'hit_confidence': hit.hit_confidence,
                    'market_conditions': json.dumps(hit.market_conditions),
                    'created_at': hit.created_at
                })
                await session.commit()
            
            # Add to history
            self.hit_history.append(hit)
            
            # Trigger callbacks
            await self._trigger_hit_callbacks(hit)
            
            logger.info(f"✅ Recorded {hit.hit_type.value} hit for signal {hit.signal_id}")
            
        except Exception as e:
            logger.error(f"Error recording hit: {e}")
            raise
    
    async def _update_position_after_hit(self, signal_id: str, position_data: Dict[str, Any], hit: TPSLHit):
        """Update position after TP/SL hit"""
        try:
            # Update remaining position
            position_data['remaining_position'] = hit.remaining_position
            
            # Update hit history
            if 'hit_history' not in position_data:
                position_data['hit_history'] = []
            position_data['hit_history'].append(hit)
            
            logger.info(f"✅ Updated position {signal_id} after {hit.hit_type.value} hit")
            
        except Exception as e:
            logger.error(f"Error updating position after hit: {e}")
    
    async def _trigger_hit_callbacks(self, hit: TPSLHit):
        """Trigger hit callbacks"""
        for callback in self.hit_callbacks:
            try:
                await callback(hit)
            except Exception as e:
                logger.error(f"Hit callback error: {e}")
    
    async def track_position(self, position_data: Dict[str, Any]):
        """
        Start tracking a position for TP/SL detection
        
        Args:
            position_data: Position data including symbol, entry_price, tp_price, sl_price, etc.
        """
        try:
            signal_id = position_data.get('signal_id')
            if not signal_id:
                raise ValueError("Signal ID is required")
            
            # Add to active positions
            self.active_positions[signal_id] = {
                **position_data,
                'partial_tps_hit': [],
                'partial_sls_hit': [],
                'hit_history': []
            }
            
            logger.info(f"✅ Started tracking position {signal_id} for TP/SL detection")
            
        except Exception as e:
            logger.error(f"Error tracking position: {e}")
            raise
    
    async def stop_tracking_position(self, signal_id: str, reason: str = 'manual_stop'):
        """Stop tracking a position"""
        try:
            if signal_id in self.active_positions:
                del self.active_positions[signal_id]
                logger.info(f"✅ Stopped tracking position {signal_id}: {reason}")
            
        except Exception as e:
            logger.error(f"Error stopping tracking for position {signal_id}: {e}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get TP/SL detection metrics"""
        return {
            'is_running': self.is_running,
            'active_positions_count': len(self.active_positions),
            'total_hits': len(self.hit_history),
            'hit_types': {
                hit_type.value: len([h for h in self.hit_history if h.hit_type == hit_type])
                for hit_type in HitType
            },
            'avg_hit_delay_ms': sum(h.hit_delay_ms for h in self.hit_history) / len(self.hit_history) if self.hit_history else 0,
            'last_updated': datetime.now(timezone.utc).isoformat()
        }
    
    def add_hit_callback(self, callback):
        """Add hit callback"""
        self.hit_callbacks.append(callback)
    
    def add_alert_callback(self, callback):
        """Add alert callback"""
        self.alert_callbacks.append(callback)
    
    async def shutdown(self):
        """Shutdown the TP/SL detector"""
        self.is_running = False
        
        # Cancel detection task
        if self.detection_task:
            self.detection_task.cancel()
        
        # Close TimescaleDB connection
        if self.timescaledb:
            await self.timescaledb.close()
        
        logger.info("🛑 TPSLDetector shutdown complete")

# Global instance
tp_sl_detector = TPSLDetector()
