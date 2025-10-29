"""
Outcome Tracker for AlphaPulse
Main system for tracking signal outcomes and performance
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
    from ..src.streaming.stream_processor import StreamProcessor
except ImportError:
    try:
        from src.database.connection import TimescaleDBConnection
        from src.core.config import settings
        from src.streaming.stream_processor import StreamProcessor
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
        
        class StreamProcessor:
            def __init__(self, config=None):
                self.config = config or {}
                self.is_initialized = False
            async def initialize(self): self.is_initialized = True
            async def shutdown(self): self.is_initialized = False

logger = logging.getLogger(__name__)

class OutcomeType(Enum):
    """Signal outcome types"""
    TP_HIT = 'tp_hit'
    SL_HIT = 'sl_hit'
    TIME_EXIT = 'time_exit'
    MANUAL_CLOSE = 'manual_close'
    PARTIAL_FILL = 'partial_fill'
    ORDER_CANCELLED = 'order_cancelled'
    ORDER_REJECTED = 'order_rejected'

class OrderState(Enum):
    """Order states"""
    PENDING = 'pending'
    FILLED = 'filled'
    PARTIALLY_FILLED = 'partially_filled'
    CANCELLED = 'cancelled'
    REJECTED = 'rejected'
    EXPIRED = 'expired'

@dataclass
class SignalOutcome:
    """Signal outcome data structure"""
    signal_id: str
    outcome_type: OutcomeType
    exit_price: float
    exit_timestamp: datetime
    realized_pnl: float
    max_adverse_excursion: float
    max_favorable_excursion: float
    time_to_exit: timedelta
    transaction_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    consistency_version: int = 1
    audit_trail: Dict[str, Any] = field(default_factory=dict)
    order_type: str = 'market'
    partial_fill_details: Dict[str, Any] = field(default_factory=dict)
    order_state: OrderState = OrderState.FILLED
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

@dataclass
class OutcomeMetrics:
    """Outcome tracking metrics"""
    total_signals: int = 0
    successful_outcomes: int = 0
    failed_outcomes: int = 0
    total_pnl: float = 0.0
    avg_pnl: float = 0.0
    win_rate: float = 0.0
    avg_time_to_exit: timedelta = timedelta(0)
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    last_updated: Optional[datetime] = None

class OutcomeTracker:
    """
    Main outcome tracker for monitoring signal performance
    
    Features:
    - Real-time signal outcome tracking
    - Take profit/stop loss detection
    - Performance metrics calculation
    - Transactional consistency
    - Audit trail management
    - TimescaleDB integration
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Tracking settings
        self.enable_real_time_tracking = self.config.get('enable_real_time_tracking', True)
        self.enable_audit_trail = self.config.get('enable_audit_trail', True)
        self.enable_transactional_consistency = self.config.get('enable_transactional_consistency', True)
        
        # Performance settings
        self.tracking_interval = self.config.get('tracking_interval', 1.0)  # seconds
        self.max_tracking_duration = self.config.get('max_tracking_duration', 24 * 60 * 60)  # 24 hours
        self.batch_size = self.config.get('batch_size', 100)
        
        # State management
        self.is_running = False
        self.metrics = OutcomeMetrics()
        self.active_signals = {}  # signal_id -> signal_data
        self.outcome_history = []
        self.tracking_task = None
        
        # Component references
        self.timescaledb = None
        self.stream_processor = None
        
        # Callbacks
        self.outcome_callbacks = []
        self.alert_callbacks = []
        
        logger.info("OutcomeTracker initialized")
    
    async def initialize(self):
        """Initialize the outcome tracker"""
        try:
            # Initialize TimescaleDB connection
            await self._initialize_timescaledb()
            
            # Initialize stream processor connection
            await self._initialize_stream_processor()
            
            # Start tracking loop
            if self.enable_real_time_tracking:
                await self._start_tracking_loop()
            
            self.is_running = True
            logger.info("✅ OutcomeTracker initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize OutcomeTracker: {e}")
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
            logger.info("✅ TimescaleDB connection initialized for outcome tracker")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize TimescaleDB: {e}")
            self.timescaledb = None
    
    async def _initialize_stream_processor(self):
        """Initialize stream processor connection"""
        try:
            # Get reference to existing stream processor
            self.stream_processor = StreamProcessor(self.config)
            logger.info("✅ Stream processor connection initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize stream processor: {e}")
            self.stream_processor = None
    
    async def _start_tracking_loop(self):
        """Start the real-time tracking loop"""
        self.tracking_task = asyncio.create_task(self._tracking_worker())
        logger.info("✅ Tracking loop started")
    
    async def _tracking_worker(self):
        """Main tracking worker loop"""
        while self.is_running:
            try:
                # Process active signals
                await self._process_active_signals()
                
                # Update metrics
                await self._update_metrics()
                
                # Sleep for tracking interval
                await asyncio.sleep(self.tracking_interval)
                
            except Exception as e:
                logger.error(f"Tracking worker error: {e}")
                await asyncio.sleep(5)  # Wait before retrying
    
    async def _process_active_signals(self):
        """Process all active signals for outcome detection"""
        current_time = datetime.now(timezone.utc)
        
        for signal_id, signal_data in list(self.active_signals.items()):
            try:
                # Check if signal has expired
                if (current_time - signal_data['created_at']).total_seconds() > self.max_tracking_duration:
                    await self._handle_signal_expiry(signal_id, signal_data)
                    continue
                
                # Check for outcome conditions
                outcome = await self._check_outcome_conditions(signal_id, signal_data, current_time)
                
                if outcome:
                    await self._record_outcome(outcome)
                    del self.active_signals[signal_id]
                
            except Exception as e:
                logger.error(f"Error processing signal {signal_id}: {e}")
    
    async def _check_outcome_conditions(self, signal_id: str, signal_data: Dict[str, Any], current_time: datetime) -> Optional[SignalOutcome]:
        """Check if signal has met outcome conditions"""
        try:
            # Get current price (this would come from streaming data)
            current_price = await self._get_current_price(signal_data['symbol'])
            
            if not current_price:
                return None
            
            signal_type = signal_data.get('signal_type', 'long')
            entry_price = signal_data['entry_price']
            tp_price = signal_data.get('tp_price')
            sl_price = signal_data.get('sl_price')
            
            # Check take profit
            if tp_price:
                if (signal_type == 'long' and current_price >= tp_price) or \
                   (signal_type == 'short' and current_price <= tp_price):
                    return await self._create_outcome(
                        signal_id, signal_data, OutcomeType.TP_HIT, 
                        current_price, current_time
                    )
            
            # Check stop loss
            if sl_price:
                if (signal_type == 'long' and current_price <= sl_price) or \
                   (signal_type == 'short' and current_price >= sl_price):
                    return await self._create_outcome(
                        signal_id, signal_data, OutcomeType.SL_HIT, 
                        current_price, current_time
                    )
            
            # Check time-based exit
            max_hold_time = signal_data.get('max_hold_time', timedelta(hours=24))
            if (current_time - signal_data['created_at']) > max_hold_time:
                return await self._create_outcome(
                    signal_id, signal_data, OutcomeType.TIME_EXIT, 
                    current_price, current_time
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error checking outcome conditions for signal {signal_id}: {e}")
            return None
    
    async def _get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price for symbol"""
        try:
            # This would integrate with streaming data
            # For now, return a mock price
            return 100.0  # Mock price
            
        except Exception as e:
            logger.error(f"Error getting current price for {symbol}: {e}")
            return None
    
    async def _create_outcome(self, signal_id: str, signal_data: Dict[str, Any], 
                            outcome_type: OutcomeType, exit_price: float, 
                            exit_timestamp: datetime) -> SignalOutcome:
        """Create outcome record"""
        try:
            # Calculate P&L
            entry_price = signal_data['entry_price']
            signal_type = signal_data.get('signal_type', 'long')
            
            if signal_type == 'long':
                realized_pnl = exit_price - entry_price
            else:
                realized_pnl = entry_price - exit_price
            
            # Calculate time to exit
            time_to_exit = exit_timestamp - signal_data['created_at']
            
            # Calculate max excursions
            max_adverse_excursion = signal_data.get('max_adverse_excursion', 0.0)
            max_favorable_excursion = signal_data.get('max_favorable_excursion', 0.0)
            
            # Create audit trail
            audit_trail = {
                'outcome_type': outcome_type.value,
                'exit_price': exit_price,
                'exit_timestamp': exit_timestamp.isoformat(),
                'signal_data': signal_data,
                'processing_time': time.time()
            }
            
            return SignalOutcome(
                signal_id=signal_id,
                outcome_type=outcome_type,
                exit_price=exit_price,
                exit_timestamp=exit_timestamp,
                realized_pnl=realized_pnl,
                max_adverse_excursion=max_adverse_excursion,
                max_favorable_excursion=max_favorable_excursion,
                time_to_exit=time_to_exit,
                audit_trail=audit_trail
            )
            
        except Exception as e:
            logger.error(f"Error creating outcome for signal {signal_id}: {e}")
            raise
    
    async def _record_outcome(self, outcome: SignalOutcome):
        """Record outcome to database"""
        try:
            if not self.timescaledb:
                logger.warning("No database connection available")
                return
            
            # Insert into TimescaleDB
            async with self.timescaledb.get_session() as session:
                await session.execute("""
                    INSERT INTO signal_outcomes (
                        signal_id, outcome_type, exit_price, exit_timestamp,
                        realized_pnl, max_adverse_excursion, max_favorable_excursion,
                        time_to_exit, transaction_id, consistency_version, audit_trail,
                        order_type, partial_fill_details, order_state, created_at
                    ) VALUES (
                        :signal_id, :outcome_type, :exit_price, :exit_timestamp,
                        :realized_pnl, :max_adverse_excursion, :max_favorable_excursion,
                        :time_to_exit, :transaction_id, :consistency_version, :audit_trail,
                        :order_type, :partial_fill_details, :order_state, :created_at
                    )
                """, {
                    'signal_id': outcome.signal_id,
                    'outcome_type': outcome.outcome_type.value,
                    'exit_price': outcome.exit_price,
                    'exit_timestamp': outcome.exit_timestamp,
                    'realized_pnl': outcome.realized_pnl,
                    'max_adverse_excursion': outcome.max_adverse_excursion,
                    'max_favorable_excursion': outcome.max_favorable_excursion,
                    'time_to_exit': outcome.time_to_exit,
                    'transaction_id': outcome.transaction_id,
                    'consistency_version': outcome.consistency_version,
                    'audit_trail': json.dumps(outcome.audit_trail),
                    'order_type': outcome.order_type,
                    'partial_fill_details': json.dumps(outcome.partial_fill_details),
                    'order_state': outcome.order_state.value,
                    'created_at': outcome.created_at
                })
                await session.commit()
            
            # Add to history
            self.outcome_history.append(outcome)
            
            # Trigger callbacks
            await self._trigger_outcome_callbacks(outcome)
            
            logger.info(f"✅ Recorded outcome for signal {outcome.signal_id}: {outcome.outcome_type.value}")
            
        except Exception as e:
            logger.error(f"Error recording outcome: {e}")
            raise
    
    async def _handle_signal_expiry(self, signal_id: str, signal_data: Dict[str, Any]):
        """Handle signal expiry"""
        try:
            logger.warning(f"Signal {signal_id} expired - removing from tracking")
            del self.active_signals[signal_id]
            
            # Trigger alert callbacks
            await self._trigger_alert_callbacks('signal_expiry', {
                'signal_id': signal_id,
                'signal_data': signal_data
            })
            
        except Exception as e:
            logger.error(f"Error handling signal expiry for {signal_id}: {e}")
    
    async def _update_metrics(self):
        """Update outcome metrics"""
        try:
            if not self.outcome_history:
                return
            
            # Calculate metrics
            total_signals = len(self.outcome_history)
            successful_outcomes = len([o for o in self.outcome_history if o.realized_pnl > 0])
            total_pnl = sum(o.realized_pnl for o in self.outcome_history)
            
            self.metrics.total_signals = total_signals
            self.metrics.successful_outcomes = successful_outcomes
            self.metrics.failed_outcomes = total_signals - successful_outcomes
            self.metrics.total_pnl = total_pnl
            self.metrics.avg_pnl = total_pnl / total_signals if total_signals > 0 else 0.0
            self.metrics.win_rate = successful_outcomes / total_signals if total_signals > 0 else 0.0
            self.metrics.last_updated = datetime.now(timezone.utc)
            
            # Calculate average time to exit
            if self.outcome_history:
                total_time = sum(o.time_to_exit for o in self.outcome_history)
                self.metrics.avg_time_to_exit = total_time / len(self.outcome_history)
            
        except Exception as e:
            logger.error(f"Error updating metrics: {e}")
    
    async def _trigger_outcome_callbacks(self, outcome: SignalOutcome):
        """Trigger outcome callbacks"""
        for callback in self.outcome_callbacks:
            try:
                await callback(outcome)
            except Exception as e:
                logger.error(f"Outcome callback error: {e}")
    
    async def _trigger_alert_callbacks(self, alert_type: str, alert_data: Dict[str, Any]):
        """Trigger alert callbacks"""
        for callback in self.alert_callbacks:
            try:
                await callback(alert_type, alert_data)
            except Exception as e:
                logger.error(f"Alert callback error: {e}")
    
    async def track_signal(self, signal_data: Dict[str, Any]):
        """
        Start tracking a new signal
        
        Args:
            signal_data: Signal data including symbol, entry_price, tp_price, sl_price, etc.
        """
        try:
            signal_id = signal_data.get('signal_id')
            if not signal_id:
                raise ValueError("Signal ID is required")
            
            # Add to active signals
            self.active_signals[signal_id] = {
                **signal_data,
                'created_at': datetime.now(timezone.utc),
                'max_adverse_excursion': 0.0,
                'max_favorable_excursion': 0.0
            }
            
            logger.info(f"✅ Started tracking signal {signal_id}")
            
        except Exception as e:
            logger.error(f"Error tracking signal: {e}")
            raise
    
    async def stop_tracking_signal(self, signal_id: str, reason: str = 'manual_stop'):
        """Stop tracking a signal"""
        try:
            if signal_id in self.active_signals:
                signal_data = self.active_signals[signal_id]
                del self.active_signals[signal_id]
                
                logger.info(f"✅ Stopped tracking signal {signal_id}: {reason}")
                
                # Trigger alert callbacks
                await self._trigger_alert_callbacks('manual_stop', {
                    'signal_id': signal_id,
                    'reason': reason,
                    'signal_data': signal_data
                })
            
        except Exception as e:
            logger.error(f"Error stopping tracking for signal {signal_id}: {e}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get outcome tracking metrics"""
        return {
            'is_running': self.is_running,
            'active_signals_count': len(self.active_signals),
            'total_outcomes': self.metrics.total_signals,
            'successful_outcomes': self.metrics.successful_outcomes,
            'failed_outcomes': self.metrics.failed_outcomes,
            'total_pnl': self.metrics.total_pnl,
            'avg_pnl': self.metrics.avg_pnl,
            'win_rate': self.metrics.win_rate,
            'avg_time_to_exit': str(self.metrics.avg_time_to_exit),
            'last_updated': self.metrics.last_updated.isoformat() if self.metrics.last_updated else None
        }
    
    def add_outcome_callback(self, callback):
        """Add outcome callback"""
        self.outcome_callbacks.append(callback)
    
    def add_alert_callback(self, callback):
        """Add alert callback"""
        self.alert_callbacks.append(callback)
    
    async def shutdown(self):
        """Shutdown the outcome tracker"""
        self.is_running = False
        
        # Cancel tracking task
        if self.tracking_task:
            self.tracking_task.cancel()
        
        # Close TimescaleDB connection
        if self.timescaledb:
            await self.timescaledb.close()
        
        logger.info("🛑 OutcomeTracker shutdown complete")

# Global instance
outcome_tracker = OutcomeTracker()
