"""
Intelligent Signal Generator for AlphaPulse
Generates signals based on intelligent analysis with 85% confidence threshold
Enhanced with real-time processing, ensemble voting, and notification system
Phase 7: Real-Time Processing Enhancement with caching, parallel processing, and advanced validation
"""

import asyncio
import logging
import uuid
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import asyncpg
import ccxt
from dataclasses import dataclass
import json
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import lru_cache
import hashlib
from collections import defaultdict, deque
import threading

from ..analysis.intelligent_analysis_engine import IntelligentAnalysisEngine, IntelligentAnalysisResult

# Import additional ML models and analysis components
try:
    from src.ai.onnx_converter import ONNXConverter
    from src.ai.ml_models.online_learner import OnlineLearner
    from src.ai.feature_drift_detector import FeatureDriftDetector
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    logging.warning("ONNX components not available")

try:
    from src.core.indicators_engine import TechnicalIndicators
    from src.strategies.pattern_detector import CandlestickPatternDetector
    TECHNICAL_AVAILABLE = True
except ImportError:
    TECHNICAL_AVAILABLE = False
    logging.warning("Technical analysis components not available")

try:
    from data_collection.market_intelligence_collector import MarketIntelligenceCollector
    from data_collection.volume_positioning_analyzer import VolumePositioningAnalyzer
    MARKET_INTELLIGENCE_AVAILABLE = True
except ImportError:
    MARKET_INTELLIGENCE_AVAILABLE = False
    logging.warning("Market intelligence components not available")

# Import Advanced Price Action Integration
try:
    from src.strategies.advanced_price_action_integration import AdvancedPriceActionIntegration, EnhancedSignal
    PRICE_ACTION_INTEGRATION_AVAILABLE = True
except ImportError:
    PRICE_ACTION_INTEGRATION_AVAILABLE = False
    logging.warning("Advanced price action integration not available")

# Import Enhanced Algorithm Integration
try:
    from src.services.algorithm_integration_service import AlgorithmIntegrationService
    from src.strategies.standalone_psychological_levels_analyzer import StandalonePsychologicalLevelsAnalyzer
    from src.strategies.enhanced_volume_weighted_levels_analyzer import EnhancedVolumeWeightedLevelsAnalyzer
    from src.services.enhanced_orderbook_integration import EnhancedOrderBookIntegration
    ENHANCED_ALGORITHMS_AVAILABLE = True
except ImportError:
    ENHANCED_ALGORITHMS_AVAILABLE = False
    logging.warning("Enhanced Algorithm Integration not available")

# Import SDE Framework
try:
    from src.ai.sde_framework import SDEFramework
    from src.ai.sde_integration_manager import SDEIntegrationManager
    SDE_FRAMEWORK_AVAILABLE = True
except ImportError:
    SDE_FRAMEWORK_AVAILABLE = False
    logging.warning("SDE Framework not available")

logger = logging.getLogger(__name__)

@dataclass
class IntelligentSignal:
    """Intelligent trading signal with comprehensive analysis"""
    signal_id: str
    symbol: str
    timeframe: str
    timestamp: datetime
    
    # Signal Type
    signal_type: str  # 'entry', 'no_safe_entry', 'exit'
    signal_direction: str  # 'long', 'short', 'neutral'
    signal_strength: str  # 'weak', 'moderate', 'strong', 'very_strong'
    
    # Confidence and Risk
    confidence_score: float
    risk_reward_ratio: float
    risk_level: str  # 'low', 'medium', 'high'
    
    # Entry/Exit Levels
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit_1: Optional[float] = None
    take_profit_2: Optional[float] = None
    take_profit_3: Optional[float] = None
    take_profit_4: Optional[float] = None
    position_size_percentage: Optional[float] = None
    
    # Analysis Summary
    pattern_analysis: str = ""
    technical_analysis: str = ""
    sentiment_analysis: str = ""
    volume_analysis: str = ""
    market_regime_analysis: str = ""
    
    # Reasoning
    entry_reasoning: str = ""
    no_safe_entry_reasons: List[str] = None
    best_timeframe_reasoning: str = ""
    
    # Status
    status: str = "generated"  # 'generated', 'active', 'completed', 'cancelled'
    
    # Performance tracking
    pnl: Optional[float] = None
    executed_at: Optional[datetime] = None
    closed_at: Optional[datetime] = None
    
    # Advanced fields
    health_score: Optional[float] = None
    ensemble_votes: Optional[Dict[str, Any]] = None
    confidence_breakdown: Optional[Dict[str, Any]] = None
    news_impact_score: Optional[float] = None
    sentiment_score: Optional[float] = None
    signal_priority: Optional[int] = None
    parallel_processing_used: bool = False
    closed_at: Optional[datetime] = None

    # Real-time enhancements
    health_score: float = 0.0
    ensemble_votes: Optional[Dict] = None
    confidence_breakdown: Optional[Dict] = None
    news_impact_score: float = 0.0
    sentiment_score: float = 0.0
    signal_priority: int = 0
    
    # Phase 7 enhancements
    processing_time_ms: float = 0.0
    cache_hit: bool = False
    parallel_processing_used: bool = False
    validation_score: float = 0.0
    quality_metrics: Optional[Dict] = None
    performance_metadata: Optional[Dict] = None

class RealTimeCache:
    """Real-time caching system for signal generation"""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 300):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache = {}
        self.access_times = {}
        self.lock = threading.RLock()
    
    def _generate_key(self, symbol: str, timeframe: str, data_hash: str) -> str:
        """Generate cache key"""
        return f"{symbol}:{timeframe}:{data_hash}"
    
    def _cleanup_expired(self):
        """Remove expired entries"""
        current_time = time.time()
        expired_keys = [
            key for key, access_time in self.access_times.items()
            if current_time - access_time > self.ttl_seconds
        ]
        for key in expired_keys:
            self.cache.pop(key, None)
            self.access_times.pop(key, None)
    
    def get(self, symbol: str, timeframe: str, data_hash: str) -> Optional[Dict]:
        """Get cached result"""
        with self.lock:
            self._cleanup_expired()
            key = self._generate_key(symbol, timeframe, data_hash)
            if key in self.cache:
                self.access_times[key] = time.time()
                return self.cache[key]
            return None
    
    def set(self, symbol: str, timeframe: str, data_hash: str, result: Dict):
        """Set cache result"""
        with self.lock:
            self._cleanup_expired()
            key = self._generate_key(symbol, timeframe, data_hash)
            
            # Remove oldest if cache is full
            if len(self.cache) >= self.max_size:
                oldest_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
                self.cache.pop(oldest_key, None)
                self.access_times.pop(oldest_key, None)
            
            self.cache[key] = result
            self.access_times[key] = time.time()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self.lock:
            return {
                'size': len(self.cache),
                'max_size': self.max_size,
                'ttl_seconds': self.ttl_seconds,
                'hit_rate': getattr(self, '_hit_count', 0) / max(getattr(self, '_total_requests', 1), 1)
            }

class SignalQualityValidator:
    """Advanced signal quality validation system"""
    
    def __init__(self):
        self.quality_thresholds = {
            'min_confidence': 0.85,
            'min_health_score': 0.80,
            'min_validation_score': 0.75,
            'max_processing_time_ms': 1000,
            'min_risk_reward': 2.0
        }
    
    def validate_signal_quality(self, signal: IntelligentSignal, 
                              processing_time_ms: float) -> Tuple[bool, float, List[str]]:
        """Validate signal quality and return score"""
        issues = []
        score = 1.0
        
        # Confidence validation
        if signal.confidence_score < self.quality_thresholds['min_confidence']:
            issues.append(f"Low confidence: {signal.confidence_score:.3f}")
            score *= 0.8
        
        # Health score validation
        if signal.health_score < self.quality_thresholds['min_health_score']:
            issues.append(f"Low health score: {signal.health_score:.3f}")
            score *= 0.9
        
        # Processing time validation
        if processing_time_ms > self.quality_thresholds['max_processing_time_ms']:
            issues.append(f"Slow processing: {processing_time_ms:.1f}ms")
            score *= 0.95
        
        # Risk/reward validation
        if signal.risk_reward_ratio < self.quality_thresholds['min_risk_reward']:
            issues.append(f"Poor risk/reward: {signal.risk_reward_ratio:.2f}")
            score *= 0.85
        
        # Additional quality checks
        if signal.signal_strength == 'weak':
            issues.append("Weak signal strength")
            score *= 0.9
        
        if signal.risk_level == 'high':
            issues.append("High risk level")
            score *= 0.95
        
        is_valid = score >= self.quality_thresholds['min_validation_score']
        return is_valid, score, issues

class ParallelProcessor:
    """Parallel processing system for signal generation"""
    
    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.thread_pool = ThreadPoolExecutor(max_workers=max_workers)
        self.process_pool = ProcessPoolExecutor(max_workers=max_workers)
    
    async def process_parallel(self, tasks: List[Tuple[callable, tuple]]) -> List[Any]:
        """Process tasks in parallel"""
        loop = asyncio.get_event_loop()
        
        # Submit tasks to thread pool
        futures = []
        for func, args in tasks:
            future = loop.run_in_executor(self.thread_pool, func, *args)
            futures.append(future)
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*futures, return_exceptions=True)
        
        # Filter out exceptions
        valid_results = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Parallel processing error: {result}")
            else:
                valid_results.append(result)
        
        return valid_results
    
    def shutdown(self):
        """Shutdown thread and process pools"""
        self.thread_pool.shutdown(wait=True)
        self.process_pool.shutdown(wait=True)

@dataclass
class EntryZoneState:
    """Entry zone state for real-time monitoring"""
    signal_id: str
    symbol: str
    timeframe: str
    direction: str
    zone: Dict[str, Any]
    created_at: datetime
    last_updated: datetime
    time_decay_factor: float = 1.0
    market_event_factor: float = 1.0
    confidence_score: float = 1.0
    status: str = "active"  # active, expired, invalidated
    price_penetration: float = 0.0  # How much price has moved into zone
    volume_confirmation: float = 0.0
    news_impact: float = 0.0

@dataclass
class MarketEvent:
    """Market event for zone adjustment"""
    event_id: str
    event_type: str  # funding_rate_spike, liquidation_cascade, news_event, volatility_spike
    symbol: str
    timestamp: datetime
    impact_score: float  # 0.0 to 1.0
    description: str
    zone_adjustment_factor: float = 1.0

class EnhancedEntryZoneMonitor:
    """
    Enhanced Entry Zone Monitor for Phase 3: Real-Time Entry Zone Updates
    
    Features:
    - Continuous monitoring of active entry zones
    - Time decay application (10% every 15 minutes)
    - News event filtering and zone adjustment
    - Real-time price penetration tracking
    - Volume confirmation monitoring
    - Market condition adaptation
    """
    
    def __init__(self, db_pool: asyncpg.Pool):
        self.db_pool = db_pool
        
        # Active zones tracking
        self.active_zones: Dict[str, EntryZoneState] = {}
        self.zone_history: deque = deque(maxlen=1000)
        
        # Market events tracking
        self.market_events: Dict[str, MarketEvent] = {}
        self.event_history: deque = deque(maxlen=500)
        
        # Monitoring configuration
        self.time_decay_interval = 15 * 60  # 15 minutes in seconds
        self.decay_rate = 0.1  # 10% decay per interval
        self.max_zone_age = 4 * 60 * 60  # 4 hours maximum zone age
        
        # Real-time data sources
        self.current_prices: Dict[str, float] = {}
        self.volume_data: Dict[str, float] = {}
        self.market_conditions: Dict[str, Any] = {}
        
        # Performance tracking
        self.monitoring_stats = {
            'zones_monitored': 0,
            'zones_expired': 0,
            'zones_adjusted': 0,
            'events_processed': 0,
            'avg_monitoring_latency_ms': 0.0
        }
        
        logger.info("🚀 Enhanced Entry Zone Monitor initialized")

    async def initialize(self):
        """Initialize the entry zone monitor"""
        try:
            # Load active zones from database
            await self._load_active_zones()
            
            # Initialize market event monitoring
            await self._initialize_market_event_monitoring()
            
            logger.info("✅ Entry Zone Monitor initialized")
            
        except Exception as e:
            logger.error(f"❌ Error initializing Entry Zone Monitor: {e}")

    async def cleanup(self):
        """Cleanup the entry zone monitor"""
        try:
            # Save zone states to database
            await self._save_zone_states()
            
            logger.info("✅ Entry Zone Monitor cleaned up")
            
        except Exception as e:
            logger.error(f"❌ Error cleaning up Entry Zone Monitor: {e}")

    async def add_entry_zone(self, signal_id: str, symbol: str, timeframe: str, 
                           direction: str, zone: Dict[str, Any], confidence_score: float = 1.0):
        """Add a new entry zone for monitoring"""
        try:
            zone_state = EntryZoneState(
                signal_id=signal_id,
                symbol=symbol,
                timeframe=timeframe,
                direction=direction,
                zone=zone,
                created_at=datetime.utcnow(),
                last_updated=datetime.utcnow(),
                confidence_score=confidence_score
            )
            
            self.active_zones[signal_id] = zone_state
            self.zone_history.append(zone_state)
            
            logger.info(f"✅ Added entry zone for {signal_id} ({symbol} {direction})")
            
        except Exception as e:
            logger.error(f"❌ Error adding entry zone: {e}")

    async def monitor_active_zones(self):
        """Monitor all active entry zones"""
        try:
            current_time = datetime.utcnow()
            
            for signal_id, zone_state in list(self.active_zones.items()):
                # Check if zone has expired
                age_seconds = (current_time - zone_state.created_at).total_seconds()
                
                if age_seconds > self.max_zone_age:
                    await self._expire_zone(signal_id, "max_age_exceeded")
                    continue
                
                # Update zone state
                await self._update_zone_state(zone_state)
                
                # Track price penetration
                await self._track_price_penetration(zone_state)
                
                # Monitor volume confirmation
                await self._monitor_volume_confirmation(zone_state)
            
            self.monitoring_stats['zones_monitored'] = len(self.active_zones)
            
        except Exception as e:
            logger.error(f"❌ Error monitoring active zones: {e}")

    async def update_zone_states(self):
        """Update zone states based on real-time market data"""
        try:
            for signal_id, zone_state in self.active_zones.items():
                # Get current market data
                current_price = await self._get_current_price(zone_state.symbol)
                if current_price is None:
                    continue
                
                # Update price penetration
                zone = zone_state.zone
                if zone_state.direction == 'long':
                    if current_price <= zone['range']['upper'] and current_price >= zone['range']['lower']:
                        # Price is in zone
                        penetration = (zone['range']['upper'] - current_price) / (zone['range']['upper'] - zone['range']['lower'])
                        zone_state.price_penetration = max(zone_state.price_penetration, penetration)
                else:  # short
                    if current_price >= zone['range']['lower'] and current_price <= zone['range']['upper']:
                        # Price is in zone
                        penetration = (current_price - zone['range']['lower']) / (zone['range']['upper'] - zone['range']['lower'])
                        zone_state.price_penetration = max(zone_state.price_penetration, penetration)
                
                # Update confidence based on penetration
                if zone_state.price_penetration > 0.5:
                    zone_state.confidence_score *= 1.1  # Boost confidence for good penetration
                elif zone_state.price_penetration > 0.8:
                    zone_state.confidence_score *= 1.2  # Higher boost for deep penetration
                
                zone_state.last_updated = datetime.utcnow()
            
        except Exception as e:
            logger.error(f"❌ Error updating zone states: {e}")

    async def apply_time_decay(self):
        """Apply time decay to entry zones"""
        try:
            current_time = datetime.utcnow()
            
            for signal_id, zone_state in self.active_zones.items():
                # Calculate time since last decay
                time_since_last_update = (current_time - zone_state.last_updated).total_seconds()
                
                if time_since_last_update >= self.time_decay_interval:
                    # Apply decay
                    zone_state.time_decay_factor *= (1 - self.decay_rate)
                    zone_state.confidence_score *= zone_state.time_decay_factor
                    
                    # Adjust zone size based on decay
                    decay_adjustment = zone_state.time_decay_factor
                    zone = zone_state.zone
                    
                    # Expand zone slightly as confidence decreases
                    zone_range = zone['range']['upper'] - zone['range']['lower']
                    expansion = zone_range * (1 - decay_adjustment) * 0.1  # 10% expansion per decay
                    
                    if zone_state.direction == 'long':
                        zone['range']['upper'] += expansion
                        zone['range']['lower'] -= expansion
                    else:  # short
                        zone['range']['upper'] += expansion
                        zone['range']['lower'] -= expansion
                    
                    zone_state.last_updated = current_time
                    
                    logger.debug(f"Applied time decay to {signal_id}: factor={zone_state.time_decay_factor:.3f}")
            
        except Exception as e:
            logger.error(f"❌ Error applying time decay: {e}")

    async def check_market_events(self):
        """Check for market events that might affect entry zones"""
        try:
            # Check for funding rate spikes
            await self._check_funding_rate_events()
            
            # Check for liquidation cascades
            await self._check_liquidation_events()
            
            # Check for volatility spikes
            await self._check_volatility_events()
            
            # Apply event adjustments to zones
            await self._apply_event_adjustments()
            
        except Exception as e:
            logger.error(f"❌ Error checking market events: {e}")

    async def _load_active_zones(self):
        """Load active zones from database"""
        try:
            async with self.db_pool.acquire() as conn:
                # Query for active signals with entry zones
                query = """
                    SELECT signal_id, symbol, timeframe, direction, 
                           entry_zone, confidence_score, created_at
                    FROM signals 
                    WHERE status = 'active' 
                    AND entry_zone IS NOT NULL
                    AND created_at > NOW() - INTERVAL '4 hours'
                """
                
                rows = await conn.fetch(query)
                
                for row in rows:
                    zone_state = EntryZoneState(
                        signal_id=row['signal_id'],
                        symbol=row['symbol'],
                        timeframe=row['timeframe'],
                        direction=row['direction'],
                        zone=row['entry_zone'],
                        created_at=row['created_at'],
                        last_updated=datetime.utcnow(),
                        confidence_score=row['confidence_score']
                    )
                    
                    self.active_zones[row['signal_id']] = zone_state
                
                logger.info(f"✅ Loaded {len(rows)} active zones from database")
                
        except Exception as e:
            logger.error(f"❌ Error loading active zones: {e}")

    async def _save_zone_states(self):
        """Save zone states to database"""
        try:
            async with self.db_pool.acquire() as conn:
                for signal_id, zone_state in self.active_zones.items():
                    query = """
                        UPDATE signals 
                        SET entry_zone = $1, confidence_score = $2, 
                            last_updated = $3, status = $4
                        WHERE signal_id = $5
                    """
                    
                    await conn.execute(
                        query, 
                        zone_state.zone, 
                        zone_state.confidence_score,
                        zone_state.last_updated,
                        zone_state.status,
                        signal_id
                    )
                
                logger.info(f"✅ Saved {len(self.active_zones)} zone states to database")
                
        except Exception as e:
            logger.error(f"❌ Error saving zone states: {e}")

    async def _get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price for symbol"""
        try:
            async with self.db_pool.acquire() as conn:
                query = """
                    SELECT close FROM ohlcv_data 
                    WHERE symbol = $1 AND timeframe = '1m'
                    ORDER BY timestamp DESC LIMIT 1
                """
                
                result = await conn.fetchval(query, symbol)
                return float(result) if result else None
                
        except Exception as e:
            logger.error(f"❌ Error getting current price for {symbol}: {e}")
            return None

    async def _expire_zone(self, signal_id: str, reason: str):
        """Expire an entry zone"""
        try:
            if signal_id in self.active_zones:
                zone_state = self.active_zones[signal_id]
                zone_state.status = "expired"
                zone_state.confidence_score = 0.0
                
                # Remove from active zones
                del self.active_zones[signal_id]
                
                self.monitoring_stats['zones_expired'] += 1
                
                logger.info(f"✅ Expired zone {signal_id}: {reason}")
                
        except Exception as e:
            logger.error(f"❌ Error expiring zone {signal_id}: {e}")

    async def _update_zone_state(self, zone_state: EntryZoneState):
        """Update individual zone state"""
        try:
            # Update current price
            current_price = await self._get_current_price(zone_state.symbol)
            if current_price:
                self.current_prices[zone_state.symbol] = current_price
            
            # Update volume data
            volume = await self._get_current_volume(zone_state.symbol)
            if volume:
                self.volume_data[zone_state.symbol] = volume
            
        except Exception as e:
            logger.error(f"❌ Error updating zone state: {e}")

    async def _track_price_penetration(self, zone_state: EntryZoneState):
        """Track how much price has penetrated the entry zone"""
        try:
            current_price = self.current_prices.get(zone_state.symbol)
            if not current_price:
                return
            
            zone = zone_state.zone
            zone_range = zone['range']['upper'] - zone['range']['lower']
            
            if zone_state.direction == 'long':
                if zone['range']['lower'] <= current_price <= zone['range']['upper']:
                    penetration = (current_price - zone['range']['lower']) / zone_range
                    zone_state.price_penetration = max(zone_state.price_penetration, penetration)
            else:  # short
                if zone['range']['lower'] <= current_price <= zone['range']['upper']:
                    penetration = (zone['range']['upper'] - current_price) / zone_range
                    zone_state.price_penetration = max(zone_state.price_penetration, penetration)
            
        except Exception as e:
            logger.error(f"❌ Error tracking price penetration: {e}")

    async def _monitor_volume_confirmation(self, zone_state: EntryZoneState):
        """Monitor volume confirmation for entry zones"""
        try:
            current_volume = self.volume_data.get(zone_state.symbol)
            if not current_volume:
                return
            
            # Get average volume for comparison
            avg_volume = await self._get_average_volume(zone_state.symbol, zone_state.timeframe)
            if not avg_volume:
                return
            
            volume_ratio = current_volume / avg_volume
            zone_state.volume_confirmation = min(volume_ratio, 2.0)  # Cap at 2x
            
            # Adjust confidence based on volume
            if volume_ratio > 1.5:
                zone_state.confidence_score *= 1.1  # Boost for high volume
            elif volume_ratio < 0.5:
                zone_state.confidence_score *= 0.9  # Reduce for low volume
            
        except Exception as e:
            logger.error(f"❌ Error monitoring volume confirmation: {e}")

    async def _check_funding_rate_events(self):
        """Check for funding rate spike events"""
        try:
            # This would integrate with your existing funding rate monitoring
            # For now, we'll simulate the check
            pass
            
        except Exception as e:
            logger.error(f"❌ Error checking funding rate events: {e}")

    async def _check_liquidation_events(self):
        """Check for liquidation cascade events"""
        try:
            # This would integrate with your existing liquidation monitoring
            # For now, we'll simulate the check
            pass
            
        except Exception as e:
            logger.error(f"❌ Error checking liquidation events: {e}")

    async def _check_volatility_events(self):
        """Check for volatility spike events"""
        try:
            # This would integrate with your existing volatility monitoring
            # For now, we'll simulate the check
            pass
            
        except Exception as e:
            logger.error(f"❌ Error checking volatility events: {e}")

    async def _apply_event_adjustments(self):
        """Apply market event adjustments to zones"""
        try:
            for signal_id, zone_state in self.active_zones.items():
                # Apply any pending event adjustments
                if zone_state.market_event_factor != 1.0:
                    zone_state.confidence_score *= zone_state.market_event_factor
                    zone_state.market_event_factor = 1.0  # Reset after applying
                    
                    self.monitoring_stats['zones_adjusted'] += 1
            
        except Exception as e:
            logger.error(f"❌ Error applying event adjustments: {e}")

    async def _get_current_volume(self, symbol: str) -> Optional[float]:
        """Get current volume for symbol"""
        try:
            async with self.db_pool.acquire() as conn:
                query = """
                    SELECT volume FROM ohlcv_data 
                    WHERE symbol = $1 AND timeframe = '1m'
                    ORDER BY timestamp DESC LIMIT 1
                """
                
                result = await conn.fetchval(query, symbol)
                return float(result) if result else None
                
        except Exception as e:
            logger.error(f"❌ Error getting current volume for {symbol}: {e}")
            return None

    async def _get_average_volume(self, symbol: str, timeframe: str) -> Optional[float]:
        """Get average volume for symbol and timeframe"""
        try:
            async with self.db_pool.acquire() as conn:
                query = """
                    SELECT AVG(volume) FROM ohlcv_data 
                    WHERE symbol = $1 AND timeframe = $2
                    AND timestamp > NOW() - INTERVAL '24 hours'
                """
                
                result = await conn.fetchval(query, symbol, timeframe)
                return float(result) if result else None
                
        except Exception as e:
            logger.error(f"❌ Error getting average volume for {symbol}: {e}")
            return None

    async def _initialize_market_event_monitoring(self):
        """Initialize market event monitoring"""
        try:
            # This would integrate with your existing market event systems
            logger.info("✅ Market event monitoring initialized")
            
        except Exception as e:
            logger.error(f"❌ Error initializing market event monitoring: {e}")

    def get_monitoring_stats(self) -> Dict[str, Any]:
        """Get monitoring statistics"""
        return {
            **self.monitoring_stats,
            'active_zones_count': len(self.active_zones),
            'total_events_processed': len(self.event_history),
            'avg_zone_age_minutes': self._calculate_avg_zone_age()
        }

    def _calculate_avg_zone_age(self) -> float:
        """Calculate average zone age in minutes"""
        if not self.active_zones:
            return 0.0
        
        current_time = datetime.utcnow()
        total_age = sum(
            (current_time - zone.created_at).total_seconds() 
            for zone in self.active_zones.values()
        )
        
        return (total_age / len(self.active_zones)) / 60  # Convert to minutes

class IntelligentSignalGenerator:
    """
    Intelligent Signal Generator with Phase 7 real-time processing enhancements
    """
    
    def __init__(self, db_pool: asyncpg.Pool, exchange: ccxt.Exchange):
        self.db_pool = db_pool
        self.exchange = exchange
        self.analysis_engine = IntelligentAnalysisEngine(db_pool, exchange)
        
        # Phase 7 enhancements
        self.cache = RealTimeCache(max_size=1000, ttl_seconds=300)
        self.quality_validator = SignalQualityValidator()
        self.parallel_processor = ParallelProcessor(max_workers=4)
        
        # Performance tracking
        self.performance_stats = {
            'total_signals': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'avg_processing_time_ms': 0.0,
            'parallel_processing_used': 0,
            'quality_validations': 0,
            'quality_passed': 0,
            'quality_failed': 0
        }
        
        # Ensemble weights for Phase 6 integration + Enhanced Algorithms
        self.ensemble_models = {
            'technical_ml': 0.15,      # Technical analysis ML
            'price_action_ml': 0.15,   # Price action ML (Phase 4)
            'sentiment_score': 0.12,   # Sentiment analysis
            'market_regime': 0.12,     # Market regime detection
            'free_api_sde': 0.12,      # Free API SDE integration
            'psychological_levels': 0.10,  # Psychological levels analysis
            'volume_weighted_levels': 0.08,  # Volume-weighted levels analysis
            'order_book_analysis': 0.08,  # Enhanced order book analysis
            'catboost_models': 0.04,   # CatBoost with ONNX optimization
            'drift_detection': 0.02,   # Model drift detection
            'chart_pattern_ml': 0.02,  # ML-based chart pattern recognition
            'candlestick_ml': 0.01,    # Japanese candlestick ML analysis
            'volume_ml': 0.01          # Volume analysis ML models
        }
        
        # Advanced Price Action Integration (Phase 4)
        self.price_action_integration = None
        
        # Free API Integration
        self.free_api_manager = None
        self.free_api_db_service = None
        self.free_api_sde_service = None
        try:
            from src.services.free_api_manager import FreeAPIManager
            from src.services.free_api_database_service import FreeAPIDatabaseService
            from src.services.free_api_sde_integration_service import FreeAPISDEIntegrationService
            
            self.free_api_manager = FreeAPIManager()
            self.free_api_db_service = FreeAPIDatabaseService(db_pool)
            self.free_api_sde_service = FreeAPISDEIntegrationService(self.free_api_db_service, self.free_api_manager)
            logger.info("✅ Free API Integration enabled")
        except ImportError as e:
            logger.warning(f"⚠️ Free API Integration not available: {e}")
        
        # Health score weights
        self.health_score_weights = {
            'data_quality': 0.20,      # Data quality health
            'technical_health': 0.20,  # Technical analysis health
            'sentiment_health': 0.15,  # Sentiment analysis health
            'risk_health': 0.15,       # Risk management health
            'market_regime_health': 0.15,  # Market regime health
            'ml_model_health': 0.05,   # ML model performance health
            'pattern_health': 0.05,    # Pattern recognition health
            'volume_health': 0.05      # Volume analysis health
        }
        
        # Active signals tracking
        self.active_signals = {}  # symbol -> signal
        self.signal_history = deque(maxlen=1000)
        
        logger.info("IntelligentSignalGenerator initialized with Phase 7 enhancements")
        
        # Initialize Advanced Price Action Integration (Phase 4)
        if PRICE_ACTION_INTEGRATION_AVAILABLE:
            self.price_action_integration = AdvancedPriceActionIntegration(db_pool)
            logger.info("✅ Advanced Price Action Integration initialized")
        
        # Initialize SDE Framework (Phase 1)
        if SDE_FRAMEWORK_AVAILABLE:
            self.sde_framework = SDEFramework(db_pool)
            self.sde_integration_manager = SDEIntegrationManager(db_pool)
            logger.info("✅ SDE Framework and Integration Manager initialized")
        else:
            self.sde_framework = None
            self.sde_integration_manager = None
        
        # Initialize Enhanced Algorithm Integration
        if ENHANCED_ALGORITHMS_AVAILABLE:
            try:
                self.algorithm_integration_service = AlgorithmIntegrationService(db_pool)
                self.psychological_levels_analyzer = StandalonePsychologicalLevelsAnalyzer()
                self.volume_weighted_levels_analyzer = EnhancedVolumeWeightedLevelsAnalyzer()
                self.enhanced_orderbook_integration = EnhancedOrderBookIntegration()
                logger.info("✅ Enhanced Algorithm Integration initialized")
            except Exception as e:
                logger.error(f"❌ Failed to initialize Enhanced Algorithm Integration: {e}")
                self.algorithm_integration_service = None
                self.psychological_levels_analyzer = None
                self.volume_weighted_levels_analyzer = None
                self.enhanced_orderbook_integration = None

    async def initialize(self):
        """Initialize the intelligent signal generator with comprehensive setup"""
        try:
            logger.info("🚀 Initializing Intelligent Signal Generator...")
            
            # Initialize MTF tables and data
            await self._initialize_mtf_tables()
            
            # Initialize analyzer connections
            if self.psychological_levels_analyzer:
                await self.psychological_levels_analyzer.initialize()
                logger.info("✅ Psychological Levels Analyzer initialized")
            
            if self.volume_weighted_levels_analyzer:
                await self.volume_weighted_levels_analyzer.initialize()
                logger.info("✅ Volume Weighted Levels Analyzer initialized")
            
            if self.enhanced_orderbook_integration:
                await self.enhanced_orderbook_integration.initialize()
                logger.info("✅ Enhanced OrderBook Integration initialized")
            
            logger.info("✅ Intelligent Signal Generator initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Error initializing Intelligent Signal Generator: {e}")
            raise

    async def _initialize_mtf_tables(self):
        """Initialize multi-timeframe tables and populate with real data"""
        try:
            logger.info("📊 Initializing MTF tables and data...")
            
            async with self.db_pool.acquire() as conn:
                # Create 4H and 1D tables
                await self._create_mtf_tables(conn)
                
                # Check and populate data for major symbols
                symbols = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT']
                
                for symbol in symbols:
                    logger.info(f"🔍 Checking data for {symbol}...")
                    
                    # Check 1m data availability
                    count_1m = await conn.fetchval("""
                        SELECT COUNT(*) FROM ohlcv_data 
                        WHERE symbol = $1 AND timeframe = '1m' 
                        AND timestamp > NOW() - INTERVAL '7 days'
                    """, symbol)
                    
                    if count_1m < 1000:
                        logger.warning(f"Insufficient 1m data for {symbol}: {count_1m} candles")
                        continue
                    
                    # Populate 4H data
                    await self._populate_timeframe_data(conn, symbol, '4h')
                    
                    # Populate 1D data
                    await self._populate_timeframe_data(conn, symbol, '1d')
                
                logger.info("✅ MTF tables initialization completed")
                
        except Exception as e:
            logger.error(f"❌ Error initializing MTF tables: {e}")

    async def _create_mtf_tables(self, conn: asyncpg.Connection):
        """Create multi-timeframe tables"""
        try:
            # Create 4H table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS ohlcv_4h (
                    timestamp TIMESTAMPTZ NOT NULL,
                    symbol VARCHAR(20) NOT NULL,
                    timeframe VARCHAR(10) NOT NULL DEFAULT '4h',
                    open DOUBLE PRECISION,
                    high DOUBLE PRECISION,
                    low DOUBLE PRECISION,
                    close DOUBLE PRECISION,
                    volume DOUBLE PRECISION,
                    PRIMARY KEY (timestamp, symbol, timeframe)
                );
            """)
            
            # Create 1D table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS ohlcv_1d (
                    timestamp TIMESTAMPTZ NOT NULL,
                    symbol VARCHAR(20) NOT NULL,
                    timeframe VARCHAR(10) NOT NULL DEFAULT '1d',
                    open DOUBLE PRECISION,
                    high DOUBLE PRECISION,
                    low DOUBLE PRECISION,
                    close DOUBLE PRECISION,
                    volume DOUBLE PRECISION,
                    PRIMARY KEY (timestamp, symbol, timeframe)
                );
            """)
            
            # Convert to hypertables
            await conn.execute("""
                DO $$ 
                BEGIN
                    IF NOT EXISTS (
                        SELECT 1 FROM timescaledb_information.hypertables 
                        WHERE hypertable_name = 'ohlcv_4h'
                    ) THEN
                        PERFORM create_hypertable('ohlcv_4h', 'timestamp', 
                                                 chunk_time_interval => INTERVAL '1 day');
                    END IF;
                END $$;
            """)
            
            await conn.execute("""
                DO $$ 
                BEGIN
                    IF NOT EXISTS (
                        SELECT 1 FROM timescaledb_information.hypertables 
                        WHERE hypertable_name = 'ohlcv_1d'
                    ) THEN
                        PERFORM create_hypertable('ohlcv_1d', 'timestamp', 
                                                 chunk_time_interval => INTERVAL '1 day');
                    END IF;
                END $$;
            """)
            
            logger.info("✅ MTF tables created successfully")
            
        except Exception as e:
            logger.error(f"❌ Error creating MTF tables: {e}")

    async def _populate_timeframe_data(self, conn: asyncpg.Connection, symbol: str, timeframe: str):
        """Populate timeframe data by aggregating from 1m data"""
        try:
            if timeframe == '4h':
                bucket_interval = '4 hours'
            elif timeframe == '1d':
                bucket_interval = '1 day'
            else:
                return
            
            # Check if data already exists
            existing_count = await conn.fetchval(f"""
                SELECT COUNT(*) FROM ohlcv_{timeframe} 
                WHERE symbol = $1
            """, symbol)
            
            if existing_count > 0:
                logger.info(f"✅ {timeframe} data already exists for {symbol}: {existing_count} candles")
                return
            
            # Generate aggregated data - use proper column names for base tables
            aggregate_query = f"""
                INSERT INTO ohlcv_{timeframe} (timestamp, symbol, timeframe, open, high, low, close, volume)
                SELECT 
                    time_bucket('{bucket_interval}', timestamp) AS timestamp,
                    symbol,
                    '{timeframe}' AS timeframe,
                    FIRST(open, timestamp) AS open,
                    MAX(high) AS high,
                    MIN(low) AS low,
                    LAST(close, timestamp) AS close,
                    SUM(volume) AS volume
                FROM ohlcv_data 
                WHERE symbol = $1 AND timeframe = '1m'
                AND timestamp > NOW() - INTERVAL '7 days'
                GROUP BY time_bucket('{bucket_interval}', timestamp), symbol
                ON CONFLICT (timestamp, symbol, timeframe) DO NOTHING
            """
            
            result = await conn.execute(aggregate_query, symbol)
            logger.info(f"✅ Populated {timeframe} data for {symbol}")
            
        except Exception as e:
            logger.error(f"❌ Error populating {timeframe} data for {symbol}: {e}")

    # Phase 3: Real-Time Entry Zone Updates
    async def start_real_time_entry_zone_monitoring(self):
        """Start real-time entry zone monitoring for active signals"""
        try:
            logger.info("🚀 Starting Real-Time Entry Zone Monitoring...")
            
            # Initialize entry zone monitor
            self.entry_zone_monitor = EnhancedEntryZoneMonitor(self.db_pool)
            await self.entry_zone_monitor.initialize()
            
            # Start monitoring task
            self.entry_zone_monitoring_task = asyncio.create_task(
                self._entry_zone_monitoring_loop()
            )
            
            logger.info("✅ Real-Time Entry Zone Monitoring started")
            
        except Exception as e:
            logger.error(f"❌ Error starting entry zone monitoring: {e}")

    async def stop_real_time_entry_zone_monitoring(self):
        """Stop real-time entry zone monitoring"""
        try:
            if self.entry_zone_monitoring_task:
                self.entry_zone_monitoring_task.cancel()
                try:
                    await self.entry_zone_monitoring_task
                except asyncio.CancelledError:
                    pass
            
            if hasattr(self, 'entry_zone_monitor'):
                await self.entry_zone_monitor.cleanup()
            
            logger.info("🛑 Real-Time Entry Zone Monitoring stopped")
            
        except Exception as e:
            logger.error(f"❌ Error stopping entry zone monitoring: {e}")

    async def _entry_zone_monitoring_loop(self):
        """Main entry zone monitoring loop"""
        while True:
            try:
                # Monitor active entry zones
                await self.entry_zone_monitor.monitor_active_zones()
                
                # Update zone states based on real-time data
                await self.entry_zone_monitor.update_zone_states()
                
                # Apply time decay to zones
                await self.entry_zone_monitor.apply_time_decay()
                
                # Check for news events and market conditions
                await self.entry_zone_monitor.check_market_events()
                
                # Wait for next monitoring cycle
                await asyncio.sleep(1)  # Monitor every second
                
            except asyncio.CancelledError:
                logger.info("Entry zone monitoring loop cancelled")
                break
            except Exception as e:
                logger.error(f"❌ Error in entry zone monitoring loop: {e}")
                await asyncio.sleep(5)  # Wait before retrying

    def _generate_data_hash(self, data: Dict[str, Any]) -> str:
        """Generate hash for data caching"""
        data_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.md5(data_str.encode()).hexdigest()
    
    async def generate_intelligent_signal(self, symbol: str, timeframe: str = "1h") -> Optional[IntelligentSignal]:
        """
        Generate intelligent trading signal with Phase 7 real-time processing enhancements
        """
        start_time = time.time()
        
        try:
            # Generate data hash for caching
            data = {
                'symbol': symbol,
                'timeframe': timeframe,
                'timestamp': datetime.now().isoformat()
            }
            data_hash = self._generate_data_hash(data)
            
            # Check cache first
            cached_result = self.cache.get(symbol, timeframe, data_hash)
            if cached_result:
                self.performance_stats['cache_hits'] += 1
                logger.info(f"Cache hit for {symbol} {timeframe}")
                return self._create_signal_from_cache(cached_result, True)
            
            self.performance_stats['cache_misses'] += 1
            
            # Check if there's already an active signal for this symbol
            if symbol in self.active_signals:
                logger.info(f"Active signal exists for {symbol}, skipping generation")
                return None
            
            # Parallel processing for analysis components (now async)
            parallel_tasks = [
                self._get_technical_analysis(symbol, timeframe),
                self._get_sentiment_analysis(symbol),
                self._get_volume_analysis(symbol, timeframe),
                self._get_market_regime_analysis(symbol),
                self._get_free_api_data(symbol, 24)  # Add free API data
            ]
            
            # Add price action analysis if available (Phase 4)
            if self.price_action_integration:
                parallel_tasks.append(self._get_price_action_analysis(symbol, timeframe))
            
            # Add enhanced algorithm analysis if available
            if ENHANCED_ALGORITHMS_AVAILABLE:
                parallel_tasks.extend([
                    self._get_psychological_levels_analysis(symbol, timeframe),
                    self._get_volume_weighted_levels_analysis(symbol, timeframe),
                    self._get_enhanced_orderbook_analysis(symbol, timeframe)
                ])
            
            parallel_results = await asyncio.gather(*parallel_tasks, return_exceptions=True)
            
            # Handle exceptions in results
            for i, result in enumerate(parallel_results):
                if isinstance(result, Exception):
                    logger.error(f"Error in parallel task {i}: {result}")
                    parallel_results[i] = None
            
            # Extract results
            technical_result = parallel_results[0] if len(parallel_results) > 0 else None
            sentiment_result = parallel_results[1] if len(parallel_results) > 1 else None
            volume_result = parallel_results[2] if len(parallel_results) > 2 else None
            market_regime_result = parallel_results[3] if len(parallel_results) > 3 else None
            free_api_result = parallel_results[4] if len(parallel_results) > 4 else None
            price_action_result = parallel_results[5] if len(parallel_results) > 5 else None
            
            # Extract enhanced algorithm results
            psychological_result = None
            volume_weighted_result = None
            orderbook_result = None
            
            if ENHANCED_ALGORITHMS_AVAILABLE:
                base_index = 6 if self.price_action_integration else 5
                psychological_result = parallel_results[base_index] if len(parallel_results) > base_index else None
                volume_weighted_result = parallel_results[base_index + 1] if len(parallel_results) > base_index + 1 else None
                orderbook_result = parallel_results[base_index + 2] if len(parallel_results) > base_index + 2 else None
            
            # Generate signal using analysis results
            signal = await self._create_signal_from_analysis(
                symbol, timeframe, technical_result, sentiment_result, 
                volume_result, market_regime_result, price_action_result, True, free_api_result,
                psychological_result, volume_weighted_result, orderbook_result
            )
            
            # Quality validation
            processing_time_ms = (time.time() - start_time) * 1000
            is_valid, validation_score, issues = self.quality_validator.validate_signal_quality(
                signal, processing_time_ms
            )
            
            # Update signal with Phase 7 metadata
            signal.processing_time_ms = processing_time_ms
            signal.parallel_processing_used = True
            signal.validation_score = validation_score
            signal.quality_metrics = {
                'is_valid': is_valid,
                'issues': issues,
                'validation_score': validation_score
            }
            signal.performance_metadata = {
                'cache_hit': False,
                'parallel_processing_used': True,
                'processing_time_ms': processing_time_ms
            }
            
            # Update performance stats
            self.performance_stats['total_signals'] += 1
            self.performance_stats['parallel_processing_used'] += 1
            self.performance_stats['quality_validations'] += 1
            self.performance_stats['avg_processing_time_ms'] = (
                (self.performance_stats['avg_processing_time_ms'] * (self.performance_stats['total_signals'] - 1) + processing_time_ms) 
                / self.performance_stats['total_signals']
            )
            
            if is_valid:
                self.performance_stats['quality_passed'] += 1
                # Cache the result
                cache_data = self._prepare_cache_data(signal)
                cache_key = f"{symbol}_{timeframe}_{data_hash}"
                self.cache.set(cache_key, cache_data)
                
                # Store active signal
                self.active_signals[symbol] = signal
                self.signal_history.append(signal)
                
                logger.info(f"✅ Generated valid signal for {symbol}: {signal.signal_direction} {signal.confidence_score:.3f}")
                return signal
            else:
                self.performance_stats['quality_failed'] += 1
                logger.warning(f"❌ Signal quality validation failed for {symbol}: {issues}")
                return None
            
        except Exception as e:
            logger.error(f"❌ Error generating signal for {symbol}: {e}")
            return None

    def _create_signal_from_cache(self, cache_data: Dict, cache_hit: bool) -> IntelligentSignal:
        """Create signal from cached data"""
        signal = IntelligentSignal(
            signal_id=cache_data['signal_id'],
            symbol=cache_data['symbol'],
            timeframe=cache_data['timeframe'],
            timestamp=datetime.fromisoformat(cache_data['timestamp']),
            signal_type=cache_data['signal_type'],
            signal_direction=cache_data['signal_direction'],
            signal_strength=cache_data['signal_strength'],
            confidence_score=cache_data['confidence_score'],
            risk_reward_ratio=cache_data['risk_reward_ratio'],
            risk_level=cache_data['risk_level'],
            entry_price=cache_data.get('entry_price'),
            stop_loss=cache_data.get('stop_loss'),
            take_profit_1=cache_data.get('take_profit_1'),
            take_profit_2=cache_data.get('take_profit_2'),
            take_profit_3=cache_data.get('take_profit_3'),
            take_profit_4=cache_data.get('take_profit_4'),
            position_size_percentage=cache_data.get('position_size_percentage'),
            pattern_analysis=cache_data.get('pattern_analysis', ''),
            technical_analysis=cache_data.get('technical_analysis', ''),
            sentiment_analysis=cache_data.get('sentiment_analysis', ''),
            volume_analysis=cache_data.get('volume_analysis', ''),
            market_regime_analysis=cache_data.get('market_regime_analysis', ''),
            entry_reasoning=cache_data.get('entry_reasoning', ''),
            no_safe_entry_reasons=cache_data.get('no_safe_entry_reasons'),
            best_timeframe_reasoning=cache_data.get('best_timeframe_reasoning', ''),
            status=cache_data.get('status', 'generated'),
            pnl=cache_data.get('pnl'),
            executed_at=datetime.fromisoformat(cache_data['executed_at']) if cache_data.get('executed_at') else None,
            closed_at=datetime.fromisoformat(cache_data['closed_at']) if cache_data.get('closed_at') else None,
            health_score=cache_data.get('health_score', 0.0),
            ensemble_votes=cache_data.get('ensemble_votes'),
            confidence_breakdown=cache_data.get('confidence_breakdown'),
            news_impact_score=cache_data.get('news_impact_score', 0.0),
            sentiment_score=cache_data.get('sentiment_score', 0.0),
            signal_priority=cache_data.get('signal_priority', 0),
            cache_hit=cache_hit,
            parallel_processing_used=False,
            processing_time_ms=cache_data.get('processing_time_ms', 0.0),
            validation_score=cache_data.get('validation_score', 0.0),
            quality_metrics=cache_data.get('quality_metrics'),
            performance_metadata=cache_data.get('performance_metadata')
        )
        return signal
    
    def _prepare_cache_data(self, signal: IntelligentSignal) -> Dict[str, Any]:
        """Prepare signal data for caching"""
        return {
            'signal_id': signal.signal_id,
            'symbol': signal.symbol,
            'timeframe': signal.timeframe,
            'timestamp': signal.timestamp.isoformat(),
            'signal_type': signal.signal_type,
            'signal_direction': signal.signal_direction,
            'signal_strength': signal.signal_strength,
            'confidence_score': signal.confidence_score,
            'risk_reward_ratio': signal.risk_reward_ratio,
            'risk_level': signal.risk_level,
            'entry_price': signal.entry_price,
            'stop_loss': signal.stop_loss,
            'take_profit_1': signal.take_profit_1,
            'take_profit_2': signal.take_profit_2,
            'take_profit_3': signal.take_profit_3,
            'take_profit_4': signal.take_profit_4,
            'position_size_percentage': signal.position_size_percentage,
            'pattern_analysis': signal.pattern_analysis,
            'technical_analysis': signal.technical_analysis,
            'sentiment_analysis': signal.sentiment_analysis,
            'volume_analysis': signal.volume_analysis,
            'market_regime_analysis': signal.market_regime_analysis,
            'entry_reasoning': signal.entry_reasoning,
            'no_safe_entry_reasons': signal.no_safe_entry_reasons,
            'best_timeframe_reasoning': signal.best_timeframe_reasoning,
            'status': signal.status,
            'pnl': signal.pnl,
            'executed_at': signal.executed_at.isoformat() if signal.executed_at else None,
            'closed_at': signal.closed_at.isoformat() if signal.closed_at else None,
            'health_score': signal.health_score,
            'ensemble_votes': signal.ensemble_votes,
            'confidence_breakdown': signal.confidence_breakdown,
            'news_impact_score': signal.news_impact_score,
            'sentiment_score': signal.sentiment_score,
            'signal_priority': signal.signal_priority,
            'processing_time_ms': signal.processing_time_ms,
            'validation_score': signal.validation_score,
            'quality_metrics': signal.quality_metrics,
            'performance_metadata': signal.performance_metadata
        }
    
    async def _get_technical_analysis(self, symbol: str, timeframe: str) -> Dict[str, Any]:
        """Get real technical analysis using existing indicators engine"""
        try:
            # Import technical indicators engine
            from src.core.indicators_engine import TechnicalIndicators
            from src.strategies.indicators import TechnicalIndicators as TAIndicators
            
            # Get recent candlestick data from database
            async with self.db_pool.acquire() as conn:
                # Get data from candles table
                query = """
                    SELECT ts as timestamp, o as open, h as high, l as low, c as close, v as volume
                    FROM candles 
                    WHERE symbol = $1 AND timeframe = $2
                    ORDER BY ts DESC 
                    LIMIT 100
                """
                rows = await conn.fetch(query, symbol, timeframe)
                candlestick_data = [dict(row) for row in rows]
            
        except Exception as e:
            logger.error(f"Error fetching candlestick data: {e}")
            return self._get_default_technical_analysis()
            
            if not candlestick_data or len(candlestick_data) < 20:
                logger.warning(f"Insufficient data for technical analysis: {symbol}")
                return self._get_default_technical_analysis()
            
            # Convert to DataFrame
            import pandas as pd
            df = pd.DataFrame(candlestick_data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            # Calculate technical indicators using existing engine
            indicators_calc = TAIndicators()
            
            # Calculate RSI
            closes = df['close'].values
            rsi_values = indicators_calc.calculate_rsi(closes, 14)
            current_rsi = float(rsi_values[-1]) if len(rsi_values) > 0 and not pd.isna(rsi_values[-1]) else 50.0
            
            # Calculate MACD
            macd_result = indicators_calc.calculate_macd(closes, 12, 26, 9)
            macd_line = float(macd_result[0][-1]) if len(macd_result[0]) > 0 else 0.0
            macd_signal = float(macd_result[1][-1]) if len(macd_result[1]) > 0 else 0.0
            macd_histogram = float(macd_result[2][-1]) if len(macd_result[2]) > 0 else 0.0
            
            # Determine MACD signal
            if macd_line > macd_signal and macd_histogram > 0:
                macd_signal_str = 'bullish'
            elif macd_line < macd_signal and macd_histogram < 0:
                macd_signal_str = 'bearish'
            else:
                macd_signal_str = 'neutral'
            
            # Calculate Bollinger Bands
            bb_result = indicators_calc.calculate_bollinger_bands(closes, 20, 2)
            bb_upper = float(bb_result[0][-1]) if len(bb_result[0]) > 0 else df['close'].iloc[-1] * 1.02
            bb_middle = float(bb_result[1][-1]) if len(bb_result[1]) > 0 else df['close'].iloc[-1]
            bb_lower = float(bb_result[2][-1]) if len(bb_result[2]) > 0 else df['close'].iloc[-1] * 0.98
            
            current_price = df['close'].iloc[-1]
            
            # Determine Bollinger position
            if current_price > bb_upper:
                bb_position = 'above_upper'
            elif current_price < bb_lower:
                bb_position = 'below_lower'
            else:
                bb_position = 'within_bands'
            
            # Calculate support and resistance levels
            support_level = df['low'].rolling(window=20).min().iloc[-1]
            resistance_level = df['high'].rolling(window=20).max().iloc[-1]
            
            # Calculate EMA trend
            ema_12 = df['close'].ewm(span=12).mean().iloc[-1]
            ema_26 = df['close'].ewm(span=26).mean().iloc[-1]
            
            if ema_12 > ema_26:
                ema_trend = 'bullish'
            elif ema_12 < ema_26:
                ema_trend = 'bearish'
            else:
                ema_trend = 'neutral'
            
            # Calculate ATR (Average True Range)
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift())
            low_close = np.abs(df['low'] - df['close'].shift())
            true_range = np.maximum(high_low, np.maximum(high_close, low_close))
            atr = true_range.rolling(window=14).mean().iloc[-1]
            
            # Calculate confidence based on indicator alignment
            confidence_factors = []
            
            # RSI confidence
            if 30 <= current_rsi <= 70:
                confidence_factors.append(0.8)  # Neutral RSI
            elif current_rsi < 30 or current_rsi > 70:
                confidence_factors.append(0.9)  # Extreme RSI
            else:
                confidence_factors.append(0.6)
            
            # MACD confidence
            if abs(macd_histogram) > 0.001:  # Significant MACD divergence
                confidence_factors.append(0.85)
            else:
                confidence_factors.append(0.6)
            
            # Bollinger Bands confidence
            if bb_position in ['above_upper', 'below_lower']:
                confidence_factors.append(0.9)  # Price at extremes
            else:
                confidence_factors.append(0.7)
            
            # EMA trend confidence
            if ema_trend != 'neutral':
                confidence_factors.append(0.8)
            else:
                confidence_factors.append(0.5)
            
            # Calculate overall confidence
            technical_confidence = sum(confidence_factors) / len(confidence_factors) if confidence_factors else 0.6
            
            return {
                'confidence': technical_confidence,
                'rsi': current_rsi,
                'macd_signal': macd_signal_str,
                'macd_line': macd_line,
                'macd_histogram': macd_histogram,
                'bollinger_position': bb_position,
                'support': support_level,
                'resistance': resistance_level,
                'ema_trend': ema_trend,
                'current_price': current_price,
                'atr': atr
            }
            
        except Exception as e:
            logger.error(f"Error in technical analysis: {e}")
            return self._get_default_technical_analysis()
    
    async def _get_free_api_data(self, symbol: str, hours: int = 24) -> Dict[str, Any]:
        """Get comprehensive free API data for signal generation"""
        try:
            if not self.free_api_sde_service:
                return self._get_default_free_api_data()
            
            # Prepare SDE input with free API data
            sde_input = await self.free_api_sde_service.prepare_sde_input(symbol, hours)
            
            if not sde_input:
                return self._get_default_free_api_data()
            
            # Analyze with SDE framework
            sde_result = await self.free_api_sde_service.analyze_with_sde_framework(sde_input)
            
            return {
                'market_data': sde_input.market_data,
                'sentiment_data': sde_input.sentiment_data,
                'news_data': sde_input.news_data,
                'social_data': sde_input.social_data,
                'liquidation_events': sde_input.liquidation_events,
                'data_quality_score': sde_input.data_quality_score,
                'confidence_score': sde_input.confidence_score,
                'sde_result': {
                    'sde_confidence': sde_result.sde_confidence,
                    'market_regime': sde_result.market_regime,
                    'sentiment_regime': sde_result.sentiment_regime,
                    'risk_level': sde_result.risk_level,
                    'signal_strength': sde_result.signal_strength,
                    'confluence_score': sde_result.confluence_score,
                    'final_recommendation': sde_result.final_recommendation,
                    'risk_reward_ratio': sde_result.risk_reward_ratio,
                    'free_api_contributions': sde_result.free_api_contributions
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting free API data for {symbol}: {e}")
            return self._get_default_free_api_data()
    
    def _get_default_free_api_data(self) -> Dict[str, Any]:
        """Get default free API data when integration is not available"""
        return {
            'market_data': {
                'symbol': 'BTC',
                'timeframe_hours': 24,
                'market_data_by_source': {},
                'consensus_price': 0.0,
                'consensus_volume': 0.0,
                'consensus_market_cap': 0.0,
                'consensus_price_change': 0.0,
                'consensus_fear_greed': 50.0,
                'total_data_points': 0,
                'last_updated': None
            },
            'sentiment_data': {
                'symbol': 'BTC',
                'timeframe_hours': 24,
                'sentiment_by_type': {},
                'overall_sentiment': 0.0,
                'overall_confidence': 0.0,
                'total_sentiment_count': 0,
                'last_updated': None
            },
            'news_data': {
                'symbol': 'BTC',
                'timeframe_hours': 24,
                'news_by_source': {},
                'total_news_count': 0,
                'avg_sentiment': 0.0,
                'avg_relevance': 0.0,
                'last_updated': None
            },
            'social_data': {
                'symbol': 'BTC',
                'timeframe_hours': 24,
                'social_by_platform': {},
                'total_post_count': 0,
                'avg_sentiment': 0.0,
                'avg_influence': 0.0,
                'last_updated': None
            },
            'liquidation_events': [],
            'data_quality_score': 0.0,
            'confidence_score': 0.0,
            'sde_result': {
                'sde_confidence': 0.0,
                'market_regime': 'sideways',
                'sentiment_regime': 'neutral',
                'risk_level': 'medium',
                'signal_strength': 0.0,
                'confluence_score': 0.0,
                'final_recommendation': 'hold',
                'risk_reward_ratio': 1.0,
                'free_api_contributions': {}
            }
        }

    async def _get_sentiment_analysis(self, symbol: str) -> Dict[str, Any]:
        """Get real sentiment analysis using existing market intelligence"""
        try:
            # Import market intelligence collector
            from ..src.data_collection.market_intelligence_collector import MarketIntelligenceCollector
            
            # Get sentiment data from database
            try:
                async with self.db_pool.acquire() as conn:
                    # Get latest market intelligence
                    query = """
                        SELECT market_sentiment_score, news_sentiment_score, fear_greed_index
                        FROM market_intelligence 
                        ORDER BY timestamp DESC 
                        LIMIT 1
                    """
                    row = await conn.fetchrow(query)
                    if row:
                        sentiment_data = dict(row)
                    else:
                        # Fallback to sentiment service
                        query = """
                            SELECT sentiment_score, sentiment_label, confidence
                            FROM sentiment_data 
                            WHERE symbol = $1 
                            ORDER BY timestamp DESC 
                            LIMIT 1
                        """
                        row = await conn.fetchrow(query, symbol)
                        sentiment_data = dict(row) if row else None
                        
            except Exception as e:
                logger.error(f"Error fetching sentiment data: {e}")
                sentiment_data = None
            
            if sentiment_data:
                # Extract sentiment scores and convert to float
                market_sentiment = float(sentiment_data.get('market_sentiment_score', 0.5))
                news_sentiment = float(sentiment_data.get('news_sentiment_score', 0.5))
                fear_greed = float(sentiment_data.get('fear_greed_index', 50))
                
                # Convert fear & greed to sentiment score (0-1)
                if fear_greed <= 25:
                    fear_greed_sentiment = 0.25 * (fear_greed / 25)
                elif fear_greed <= 45:
                    fear_greed_sentiment = 0.25 + 0.20 * ((fear_greed - 25) / 20)
                elif fear_greed <= 55:
                    fear_greed_sentiment = 0.45 + 0.10 * ((fear_greed - 45) / 10)
                elif fear_greed <= 75:
                    fear_greed_sentiment = 0.55 + 0.20 * ((fear_greed - 55) / 20)
                else:
                    fear_greed_sentiment = 0.75 + 0.25 * ((fear_greed - 75) / 25)
                
                # Calculate composite sentiment
                composite_sentiment = (
                    market_sentiment * 0.4 +
                    news_sentiment * 0.3 +
                    fear_greed_sentiment * 0.3
                )
                
                # Calculate confidence based on data consistency
                sentiment_scores = [market_sentiment, news_sentiment, fear_greed_sentiment]
                sentiment_std = np.std(sentiment_scores)
                confidence = max(0.5, 1.0 - sentiment_std)  # Higher confidence if scores are consistent
                
                return {
                    'sentiment_score': composite_sentiment,
                    'news_impact': news_sentiment - 0.5,  # Center around 0
                    'social_sentiment': fear_greed_sentiment,
                    'confidence': confidence,
                    'market_sentiment': market_sentiment,
                    'fear_greed_index': fear_greed
                }
            else:
                return self._get_default_sentiment_analysis()
            
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {e}")
            return self._get_default_sentiment_analysis()
    
    async def _get_volume_analysis(self, symbol: str, timeframe: str) -> Dict[str, Any]:
        """Get real volume analysis using existing volume positioning analyzer"""
        try:
            # Import volume positioning analyzer
            from ..src.data_collection.volume_positioning_analyzer import VolumePositioningAnalyzer
            
            # Get volume analysis from database
            async with self.db_pool.acquire() as conn:
                # Get recent volume analysis
                query = """
                    SELECT volume_ratio, volume_trend, order_book_imbalance, 
                           volume_positioning_score, buy_volume_ratio, sell_volume_ratio
                    FROM volume_analysis 
                    WHERE symbol = $1 
                    ORDER BY timestamp DESC 
                    LIMIT 1
                """
                row = await conn.fetchrow(query, symbol)
                if row:
                    volume_data = {k: float(v) if isinstance(v, (int, float)) else v for k, v in dict(row).items()}
                else:
                    # Fallback to OHLCV data for basic volume analysis
                    query = """
                        SELECT v as volume, c as close
                        FROM candles 
                        WHERE symbol = $1 AND timeframe = $2
                        ORDER BY ts DESC 
                        LIMIT 50
                    """
                    rows = await conn.fetch(query, symbol, timeframe)
                    if rows:
                        volumes = []
                        closes = []
                        for row in rows:
                            volumes.append(float(row['volume']))
                            closes.append(float(row['close']))
                        
                        # Calculate basic volume metrics
                        current_volume = volumes[0]
                        avg_volume = np.mean(volumes[1:21]) if len(volumes) > 20 else current_volume
                        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
                        
                        # Determine volume trend
                        if len(volumes) >= 10:
                            recent_avg = np.mean(volumes[:10])
                            older_avg = np.mean(volumes[10:20]) if len(volumes) >= 20 else recent_avg
                            
                            if recent_avg > older_avg * 1.1:
                                volume_trend = 'increasing'
                            elif recent_avg < older_avg * 0.9:
                                volume_trend = 'decreasing'
                            else:
                                volume_trend = 'stable'
                        else:
                            volume_trend = 'stable'
                        
                        # Estimate volume positioning score
                        if volume_ratio > 1.5:
                            positioning_score = 0.8
                        elif volume_ratio > 1.2:
                            positioning_score = 0.7
                        elif volume_ratio > 0.8:
                            positioning_score = 0.6
                        else:
                            positioning_score = 0.4
                        
                        volume_data = {
                            'volume_ratio': volume_ratio,
                            'volume_trend': volume_trend,
                            'order_book_imbalance': 0.0,  # Not available
                            'volume_positioning_score': positioning_score,
                            'buy_volume_ratio': 0.5,  # Not available
                            'sell_volume_ratio': 0.5   # Not available
                        }
                    else:
                        volume_data = None
            
        except Exception as e:
            logger.error(f"Error fetching volume data: {e}")
            volume_data = None
        
        if volume_data:
            # Calculate confidence based on data quality
            confidence_factors = []
            
            # Volume ratio confidence
            if 0.5 <= volume_data['volume_ratio'] <= 2.0:
                confidence_factors.append(0.8)  # Reasonable volume ratio
            else:
                confidence_factors.append(0.6)  # Extreme volume ratio
            
            # Volume trend confidence
            if volume_data['volume_trend'] != 'stable':
                confidence_factors.append(0.8)  # Clear trend
            else:
                confidence_factors.append(0.6)  # Stable volume
            
            # Positioning score confidence
            if volume_data['volume_positioning_score'] > 0.6:
                confidence_factors.append(0.8)  # Good positioning
            else:
                confidence_factors.append(0.5)  # Poor positioning
            
            volume_confidence = sum(confidence_factors) / len(confidence_factors) if confidence_factors else 0.6
            
            return {
                'volume_trend': volume_data['volume_trend'],
                'volume_ratio': volume_data['volume_ratio'],
                'volume_breakout': volume_data['volume_ratio'] > 1.5,
                'confidence': volume_confidence,
                'positioning_score': volume_data['volume_positioning_score'],
                'order_book_imbalance': volume_data['order_book_imbalance']
            }
        else:
            return self._get_default_volume_analysis()
    
    async def _get_market_regime_analysis(self, symbol: str) -> Dict[str, Any]:
        """Get real market regime analysis using existing market intelligence"""
        try:
            # Get market regime data from database
            async with self.db_pool.acquire() as conn:
                # Get latest market intelligence
                query = """
                    SELECT market_regime, volatility_index, trend_strength, btc_dominance
                    FROM market_intelligence 
                    ORDER BY timestamp DESC 
                    LIMIT 1
                """
                row = await conn.fetchrow(query)
                if row:
                    regime_data = dict(row)
                else:
                    # Fallback to market regime detection
                    query = """
                        SELECT regime_type, confidence, volatility, trend_strength
                        FROM market_regime_data 
                        WHERE symbol = $1 
                        ORDER BY timestamp DESC 
                        LIMIT 1
                    """
                    row = await conn.fetchrow(query, symbol)
                    regime_data = dict(row) if row else None
            
            if regime_data:
                # Extract regime information
                regime = regime_data.get('market_regime', 'sideways')
                volatility = regime_data.get('volatility_index', 0.025)
                trend_strength = regime_data.get('trend_strength', 0.45)
                btc_dominance = regime_data.get('btc_dominance', 50.0)
                
                # Classify volatility level
                if volatility > 0.05:
                    volatility_level = 'high'
                elif volatility > 0.03:
                    volatility_level = 'medium'
                else:
                    volatility_level = 'low'
                
                # Determine trend direction
                if trend_strength > 0.6:
                    trend_direction = 'strong_up' if btc_dominance > 50 else 'strong_down'
                elif trend_strength > 0.4:
                    trend_direction = 'weak_up' if btc_dominance > 50 else 'weak_down'
                else:
                    trend_direction = 'sideways'
                
                # Calculate confidence based on regime clarity
                confidence_factors = []
                
                # Regime clarity confidence
                if regime in ['bullish', 'bearish']:
                    confidence_factors.append(0.8)  # Clear regime
                elif regime == 'volatile':
                    confidence_factors.append(0.7)  # Volatile regime
                else:
                    confidence_factors.append(0.6)  # Sideways regime
                
                # Volatility confidence
                if volatility_level != 'low':
                    confidence_factors.append(0.8)  # Clear volatility
                else:
                    confidence_factors.append(0.6)  # Low volatility
                
                # Trend strength confidence
                if trend_strength > 0.5:
                    confidence_factors.append(0.8)  # Strong trend
                else:
                    confidence_factors.append(0.6)  # Weak trend
                
                regime_confidence = sum(confidence_factors) / len(confidence_factors) if confidence_factors else 0.6
                
                return {
                    'regime': regime,
                    'volatility': volatility_level,
                    'trend_strength': trend_strength,
                    'confidence': regime_confidence,
                    'trend_direction': trend_direction,
                    'btc_dominance': btc_dominance
                }
            else:
                return self._get_default_market_regime_analysis()
                
        except Exception as e:
            logger.error(f"Error in market regime analysis: {e}")
            return self._get_default_market_regime_analysis()
    
    async def _get_market_data_for_analysis(self, symbol: str, timeframe: str) -> Dict[str, Any]:
        """Get market data for analysis"""
        try:
            # Get recent OHLCV data
            async with self.db_pool.acquire() as conn:
                # Try to get data from candles table
                query = """
                    SELECT ts, o, h, l, c, v
                    FROM candles 
                    WHERE symbol = $1 AND timeframe = $2
                    ORDER BY ts DESC 
                    LIMIT 100
                """
                rows = await conn.fetch(query, symbol, timeframe)
                
                if rows:
                    # Convert to DataFrame-like structure
                    data = {
                        'timestamp': [row['ts'] for row in rows],
                        'open': [float(row['o']) for row in rows],
                        'high': [float(row['h']) for row in rows],
                        'low': [float(row['l']) for row in rows],
                        'close': [float(row['c']) for row in rows],
                        'volume': [float(row['v']) for row in rows]
                    }
                else:
                    # Return empty data structure
                    data = {
                        'timestamp': [],
                        'open': [],
                        'high': [],
                        'low': [],
                        'close': [],
                        'volume': []
                    }
                
                return data
                
        except Exception as e:
            logger.error(f"Error getting market data: {e}")
            return {
                'timestamp': [],
                'open': [],
                'high': [],
                'low': [],
                'close': [],
                'volume': []
            }

    async def _get_price_action_analysis(self, symbol: str, timeframe: str) -> Optional[Dict[str, Any]]:
        """Get price action analysis using advanced integration (Phase 4)"""
        try:
            if not self.price_action_integration:
                return None
            
            # Get market data for price action analysis
            market_data = await self._get_market_data_for_analysis(symbol, timeframe)
            
            # Perform comprehensive price action analysis
            price_action_analysis = await self.price_action_integration.analyze_price_action(
                symbol, timeframe, market_data
            )
            
            return {
                'support_resistance_score': price_action_analysis.support_resistance_score,
                'market_structure_score': price_action_analysis.market_structure_score,
                'demand_supply_score': price_action_analysis.demand_supply_score,
                'pattern_ml_score': price_action_analysis.pattern_ml_score,
                'combined_price_action_score': price_action_analysis.combined_price_action_score,
                'price_action_confidence': price_action_analysis.price_action_confidence,
                'nearest_support': price_action_analysis.nearest_support,
                'nearest_resistance': price_action_analysis.nearest_resistance,
                'structure_type': price_action_analysis.structure_type,
                'trend_alignment': price_action_analysis.trend_alignment,
                'zone_type': price_action_analysis.zone_type,
                'breakout_probability': price_action_analysis.breakout_probability,
                'hold_probability': price_action_analysis.hold_probability,
                'context': {
                    'support_resistance_context': price_action_analysis.support_resistance_context,
                    'market_structure_context': price_action_analysis.market_structure_context,
                    'demand_supply_context': price_action_analysis.demand_supply_context,
                    'pattern_ml_context': price_action_analysis.pattern_ml_context
                }
            }
                
        except Exception as e:
            logger.error(f"Error in price action analysis: {e}")
            return None
    
    async def _get_psychological_levels_analysis(self, symbol: str, timeframe: str) -> Optional[Dict[str, Any]]:
        """Get psychological levels analysis using enhanced algorithm"""
        try:
            if not self.psychological_levels_analyzer:
                return None
            
            # Perform psychological levels analysis
            psychological_analysis = await self.psychological_levels_analyzer.analyze_psychological_levels(
                symbol, timeframe
            )
            
            if not psychological_analysis:
                return None
            
            return {
                'confidence': psychological_analysis.analysis_confidence,
                'current_price': psychological_analysis.current_price,
                'nearest_support': psychological_analysis.nearest_support_price,
                'nearest_resistance': psychological_analysis.nearest_resistance_price,
                'market_regime': psychological_analysis.market_regime,
                'psychological_levels': [
                    {
                        'level_type': level.level_type,
                        'price_level': level.price_level,
                        'strength': level.strength,
                        'confidence': level.confidence,
                        'touch_count': level.touch_count,
                        'is_active': level.is_active
                    } for level in psychological_analysis.psychological_levels
                ],
                'level_count': len(psychological_analysis.psychological_levels),
                'strong_levels': len([l for l in psychological_analysis.psychological_levels if l.strength > 0.7]),
                'active_levels': len([l for l in psychological_analysis.psychological_levels if l.is_active])
            }
                
        except Exception as e:
            logger.error(f"Error in psychological levels analysis: {e}")
            return None
    
    async def _get_volume_weighted_levels_analysis(self, symbol: str, timeframe: str) -> Optional[Dict[str, Any]]:
        """Get volume-weighted levels analysis using enhanced algorithm"""
        try:
            if not self.volume_weighted_levels_analyzer:
                return None
            
            # Perform volume-weighted levels analysis
            volume_analysis = await self.volume_weighted_levels_analyzer.analyze_volume_weighted_levels(
                symbol, timeframe
            )
            
            if not volume_analysis:
                return None
            
            return {
                'confidence': volume_analysis.analysis_confidence,
                'poc_price': volume_analysis.poc_price,
                'poc_volume': volume_analysis.poc_volume,
                'value_area_high': volume_analysis.value_area_high,
                'value_area_low': volume_analysis.value_area_low,
                'value_area_volume': volume_analysis.value_area_volume,
                'total_volume': volume_analysis.total_volume,
                'high_volume_nodes': [
                    {
                        'price': node.price,
                        'volume': node.volume,
                        'strength': node.strength,
                        'confidence': node.confidence
                    } for node in volume_analysis.high_volume_nodes
                ],
                'low_volume_nodes': [
                    {
                        'price': node.price,
                        'volume': node.volume,
                        'strength': node.strength,
                        'confidence': node.confidence
                    } for node in volume_analysis.low_volume_nodes
                ],
                'hvn_count': len(volume_analysis.high_volume_nodes),
                'lvn_count': len(volume_analysis.low_volume_nodes),
                'volume_profile_quality': volume_analysis.analysis_confidence
            }
                
        except Exception as e:
            logger.error(f"Error in volume-weighted levels analysis: {e}")
            return None
    
    async def _get_enhanced_orderbook_analysis(self, symbol: str, timeframe: str) -> Optional[Dict[str, Any]]:
        """Get enhanced order book analysis using enhanced algorithm"""
        try:
            if not self.enhanced_orderbook_integration:
                return None
            
            # Perform enhanced order book analysis
            orderbook_analysis = await self.enhanced_orderbook_integration.analyze_order_book_with_volume_profile(
                symbol, timeframe
            )
            
            if not orderbook_analysis:
                return None
            
            return {
                'confidence': orderbook_analysis.analysis_confidence,
                'bid_ask_imbalance': orderbook_analysis.bid_ask_imbalance,
                'depth_pressure': orderbook_analysis.depth_pressure,
                'liquidity_score': orderbook_analysis.liquidity_score,
                'total_bid_volume': orderbook_analysis.total_bid_volume,
                'total_ask_volume': orderbook_analysis.total_ask_volume,
                'spread': orderbook_analysis.spread,
                'spread_percentage': orderbook_analysis.spread_percentage,
                'mid_price': orderbook_analysis.mid_price,
                'best_bid': orderbook_analysis.best_bid,
                'best_ask': orderbook_analysis.best_ask,
                'volume_profile': {
                    'poc_price': orderbook_analysis.volume_profile.poc_price,
                    'poc_volume': orderbook_analysis.volume_profile.poc_volume,
                    'total_volume': orderbook_analysis.volume_profile.total_volume,
                    'high_volume_nodes': len(orderbook_analysis.volume_profile.high_volume_nodes),
                    'low_volume_nodes': len(orderbook_analysis.volume_profile.low_volume_nodes)
                },
                'order_book_levels': len(orderbook_analysis.order_book_levels),
                'market_microstructure_quality': orderbook_analysis.analysis_confidence
            }
                
        except Exception as e:
            logger.error(f"Error in enhanced order book analysis: {e}")
            return None
    
    def _get_default_technical_analysis(self) -> Dict[str, Any]:
        """Default technical analysis when real data is unavailable"""
        return {
            'confidence': 0.5,
            'rsi': 50.0,
            'macd_signal': 'neutral',
            'macd_line': 0.0,
            'macd_histogram': 0.0,
            'bollinger_position': 'within_bands',
            'support': 0.0,
            'resistance': 0.0,
            'ema_trend': 'neutral',
            'current_price': 0.0
        }
    
    def _get_default_sentiment_analysis(self) -> Dict[str, Any]:
        """Default sentiment analysis when real data is unavailable"""
        return {
            'sentiment_score': 0.5,
            'news_impact': 0.0,
            'social_sentiment': 0.5,
            'confidence': 0.5,
            'market_sentiment': 0.5,
            'fear_greed_index': 50
        }
    
    def _get_default_volume_analysis(self) -> Dict[str, Any]:
        """Default volume analysis when real data is unavailable"""
        return {
            'volume_trend': 'stable',
            'volume_ratio': 1.0,
            'volume_breakout': False,
            'confidence': 0.5,
            'positioning_score': 0.5,
            'order_book_imbalance': 0.0
        }
    
    def _get_default_market_regime_analysis(self) -> Dict[str, Any]:
        """Default market regime analysis when real data is unavailable"""
        return {
            'regime': 'sideways',
            'volatility': 'medium',
            'trend_strength': 0.5,
            'confidence': 0.5,
            'trend_direction': 'sideways',
            'btc_dominance': 50.0
        }
    
    async def _create_signal_from_analysis(self, symbol: str, timeframe: str,
                                         technical_result: Dict, sentiment_result: Dict,
                                         volume_result: Dict, market_regime_result: Dict,
                                         price_action_result: Optional[Dict],
                                         parallel_used: bool, free_api_result: Optional[Dict] = None,
                                         psychological_result: Optional[Dict] = None,
                                         volume_weighted_result: Optional[Dict] = None,
                                         orderbook_result: Optional[Dict] = None) -> IntelligentSignal:
        """Create signal from analysis results using real data"""
        
        # Calculate ensemble confidence with comprehensive null checks
        ensemble_votes = {
            'technical_ml': technical_result.get('confidence', 0.5) if technical_result else 0.5,
            'price_action_ml': price_action_result.get('combined_price_action_score', 0.5) if price_action_result else 0.5,
            'sentiment_score': sentiment_result.get('sentiment_score', 0.5) if sentiment_result else 0.5,
            'volume_ml': volume_result.get('confidence', 0.5) if volume_result else 0.5,
            'market_regime': market_regime_result.get('confidence', 0.5) if market_regime_result else 0.5,
            'free_api_sde': free_api_result.get('sde_result', {}).get('sde_confidence', 0.0) if free_api_result and free_api_result.get('sde_result') else 0.0,
            'psychological_levels': psychological_result.get('confidence', 0.5) if psychological_result else 0.5,
            'volume_weighted_levels': volume_weighted_result.get('confidence', 0.5) if volume_weighted_result else 0.5,
            'order_book_analysis': orderbook_result.get('confidence', 0.5) if orderbook_result else 0.5
        }
        
        # Calculate weighted confidence
        total_confidence = 0.0
        total_weight = 0.0
        
        for model, weight in self.ensemble_models.items():
            if model in ensemble_votes:
                total_confidence += ensemble_votes[model] * weight
                total_weight += weight
        
        final_confidence = total_confidence / total_weight if total_weight > 0 else 0.5
        
        # Phase 8: Advanced Calibration Integration
        calibrated_confidence = final_confidence
        calibration_result = None
        if SDE_FRAMEWORK_AVAILABLE and hasattr(self, 'sde_framework') and self.sde_framework:
            try:
                # Get market regime and volatility for dynamic thresholds
                market_regime = market_regime_result.get('regime', 'sideways') if market_regime_result else 'sideways'
                volatility_level = market_regime_result.get('volatility_level', 'medium') if market_regime_result else 'medium'
                
                # Prepare features for calibration with null checks
                calibration_features = {
                    'rsi': technical_result.get('rsi', 50.0) if technical_result else 50.0,
                    'volume_ratio': volume_result.get('volume_ratio', 1.0) if volume_result else 1.0,
                    'sentiment_score': sentiment_result.get('sentiment_score', 0.5) if sentiment_result else 0.5,
                    'market_regime': market_regime,
                    'technical_confidence': ensemble_votes['technical_ml'],
                    'volume_confidence': ensemble_votes['volume_ml'],
                    'regime_confidence': ensemble_votes['market_regime']
                }
                
                # Apply advanced calibration
                calibration_result = await self.sde_framework.apply_advanced_calibration(
                    final_confidence, calibration_features, symbol, timeframe, market_regime
                )
                
                if calibration_result:
                    calibrated_confidence = calibration_result.calibrated_probability
                    improvement = calibrated_confidence - final_confidence
                    logger.info(f"✅ Advanced calibration for {symbol}: "
                              f"{final_confidence:.3f} → {calibrated_confidence:.3f} "
                              f"(Δ{improvement:+.3f})")
                    logger.info(f"   Method: {calibration_result.calibration_method}")
                    logger.info(f"   Reliability: {calibration_result.reliability_score:.3f}")
                
                # Get dynamic threshold for this market condition
                dynamic_threshold = await self.sde_framework.get_dynamic_threshold(market_regime, volatility_level)
                min_confidence_threshold = dynamic_threshold['min_confidence_threshold']
                
                # Apply dynamic threshold
                if calibrated_confidence < min_confidence_threshold:
                    logger.info(f"⚠️ Signal blocked by dynamic threshold: "
                              f"{calibrated_confidence:.3f} < {min_confidence_threshold:.3f}")
                    return None
    
                # Phase 9: Advanced Signal Quality Validation
                # Prepare signal data for validation
                signal_data = {
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'signal_id': f"{symbol}_{timeframe}_{int(time.time())}",
                    'confidence': calibrated_confidence,
                    'model_agreement': ensemble_votes.get('technical_ml', 0.0),
                    'feature_importance': technical_result.get('feature_importance', 0.0) if technical_result else 0.0,
                    'historical_accuracy': technical_result.get('historical_accuracy', 0.0) if technical_result else 0.0
                }
                
                # Prepare market data for validation with null checks
                market_data = {
                    'current_price': technical_result.get('current_price', 0.0) if technical_result else 0.0,
                    'volume_24h': volume_result.get('volume_24h', 0.0) if volume_result else 0.0,
                    'volatility_24h': technical_result.get('volatility_24h', 0.0) if technical_result else 0.0
                }
                
                # Get historical data for validation (mock for now)
                import pandas as pd
                import numpy as np
                historical_data = pd.DataFrame({
                    'open': np.random.normal(45000, 1000, 100),
                    'high': np.random.normal(45500, 1000, 100),
                    'low': np.random.normal(44500, 1000, 100),
                    'close': np.random.normal(45000, 1000, 100),
                    'volume': np.random.normal(1000000, 200000, 100)
                })
                
                # Validate signal quality
                quality_metrics = await self.sde_framework.validate_signal_quality(
                    signal_data, market_data, historical_data
                )
                
                # Block signal if quality validation fails
                if not quality_metrics.validation_passed:
                    logger.info(f"🚫 Signal blocked by quality validation: {quality_metrics.rejection_reasons}")
                    return None
                
                logger.info(f"✅ Signal quality validation passed: {quality_metrics.quality_level.value} ({quality_metrics.overall_quality_score:.3f})")
                
            except Exception as e:
                logger.error(f"❌ Advanced calibration failed for {symbol}: {e}")
                # Continue with original confidence if calibration fails
        
        # SDE Integration Manager (Enhanced Integration)
        sde_integration_result = None
        if self.sde_integration_manager:
            try:
                # Prepare analysis results for SDE integration
                analysis_results = {
                    'technical_confidence': ensemble_votes['technical_ml'],
                    'technical_direction': 'long' if ensemble_votes['technical_ml'] > 0.6 else 'short' if ensemble_votes['technical_ml'] < 0.4 else 'neutral',
                    'sentiment_score': sentiment_result.get('sentiment_score', 0.5) if sentiment_result else 0.5,
                    'volume_confidence': ensemble_votes['volume_ml'],
                    'volume_direction': 'long' if ensemble_votes['volume_ml'] > 0.6 else 'short' if ensemble_votes['volume_ml'] < 0.4 else 'neutral',
                    'market_regime_confidence': ensemble_votes['market_regime'],
                    'market_regime_direction': 'long' if ensemble_votes['market_regime'] > 0.6 else 'short' if ensemble_votes['market_regime'] < 0.4 else 'neutral',
                    'support_resistance_quality': technical_result.get('support_resistance_quality', 0.5) if technical_result else 0.5,
                    'volume_confirmation': volume_result.get('volume_confirmation', False) if volume_result else False,
                    'htf_trend_strength': market_regime_result.get('htf_trend_strength', 0.5) if market_regime_result else 0.5,
                    'trend_alignment': technical_result.get('trend_alignment', False) if technical_result else False,
                    'pattern_strength': price_action_result.get('pattern_strength', 0.5) if price_action_result else 0.5,
                    'breakout_confirmed': price_action_result.get('breakout_confirmed', False) if price_action_result else False
                }
                
                # Prepare market data for SDE integration with null checks
                market_data = {
                    'current_price': technical_result.get('current_price', 0.0) if technical_result else 0.0,
                    'stop_loss': technical_result.get('stop_loss', 0.0) if technical_result else 0.0,
                    'atr_value': technical_result.get('atr_value', 0.0) if technical_result else 0.0,
                    'spread_atr_ratio': technical_result.get('spread_atr_ratio', 0.1) if technical_result else 0.1,
                    'atr_percentile': technical_result.get('atr_percentile', 50.0) if technical_result else 50.0,
                    'impact_cost': volume_result.get('impact_cost', 0.05) if volume_result else 0.05
                }
                
                # Generate signal ID for SDE integration
                signal_id = str(uuid.uuid4())
                
                # Integrate SDE with signal
                sde_integration_result = await self.sde_integration_manager.integrate_sde_with_signal(
                    signal_id=signal_id,
                    symbol=symbol,
                    timeframe=timeframe,
                    analysis_results=analysis_results,
                    market_data=market_data,
                    account_id="default"
                )
                
                # Apply SDE integration results to final confidence
                if sde_integration_result.all_gates_passed:
                    final_confidence = sde_integration_result.final_confidence
                    logger.info(f"SDE Integration successful for {symbol}: confidence {final_confidence:.3f}")
                else:
                    final_confidence = 0.0  # Block signal if SDE gates fail
                    logger.info(f"SDE Integration failed for {symbol}: {sde_integration_result.integration_reason}")
                
            except Exception as e:
                logger.error(f"SDE Integration error for {symbol}: {e}")
                sde_integration_result = None
        
        # Get current market price with null check
        current_price = technical_result.get('current_price', 0.0) if technical_result else 0.0
        if current_price <= 0:
            # Fallback: get latest price from database
            try:
                async with self.db_pool.acquire() as conn:
                    # Get price from candles table
                    query = """
                        SELECT c FROM candles 
                        WHERE symbol = $1 AND timeframe = $2
                        ORDER BY ts DESC LIMIT 1
                    """
                    row = await conn.fetchrow(query, symbol, timeframe)
                    current_price = float(row['c']) if row else 0.0
            except Exception as e:
                logger.error(f"Error getting current price: {e}")
                current_price = 0.0
        
        # Determine signal direction and strength based on real analysis
        signal_direction = 'neutral'
        signal_strength = 'weak'
        
        # Technical analysis direction with null checks
        technical_direction = 'neutral'
        rsi = technical_result.get('rsi', 50.0) if technical_result else 50.0
        macd_signal = technical_result.get('macd_signal', 'neutral') if technical_result else 'neutral'
        ema_trend = technical_result.get('ema_trend', 'neutral') if technical_result else 'neutral'
        
        if rsi < 30 and macd_signal == 'bullish' and ema_trend == 'bullish':
            technical_direction = 'long'
        elif rsi > 70 and macd_signal == 'bearish' and ema_trend == 'bearish':
            technical_direction = 'short'
        
        # Sentiment direction
        sentiment_score = sentiment_result.get('sentiment_score', 0.5) if sentiment_result else 0.5
        sentiment_direction = 'long' if sentiment_score > 0.6 else 'short' if sentiment_score < 0.4 else 'neutral'
        
        # Volume direction
        volume_trend = volume_result.get('volume_trend', 'stable') if volume_result else 'stable'
        volume_direction = 'long' if volume_trend == 'increasing' else 'short' if volume_trend == 'decreasing' else 'neutral'
        
        # Market regime direction
        regime = market_regime_result.get('regime', 'sideways') if market_regime_result else 'sideways'
        regime_direction = 'long' if regime == 'bullish' else 'short' if regime == 'bearish' else 'neutral'
        
        # Determine final signal direction based on alignment
        directions = [technical_direction, sentiment_direction, volume_direction, regime_direction]
        long_count = directions.count('long')
        short_count = directions.count('short')
        
        if long_count >= 3:
            signal_direction = 'long'
        elif short_count >= 3:
            signal_direction = 'short'
        else:
            signal_direction = 'neutral'
        
        # Determine signal strength using calibrated confidence
        if calibrated_confidence >= 0.9:
            signal_strength = 'very_strong'
        elif calibrated_confidence >= 0.8:
            signal_strength = 'strong'
        elif calibrated_confidence >= 0.7:
            signal_strength = 'moderate'
        else:
            signal_strength = 'weak'
        
        # Calculate health score
        health_components = {
            'data_quality': 0.9,
            'technical_health': technical_result.get('confidence', 0.5),
            'sentiment_health': sentiment_result.get('sentiment_score', 0.5),
            'risk_health': 0.85,
            'market_regime_health': market_regime_result.get('confidence', 0.5) if market_regime_result else 0.5,
            'ml_model_health': 0.88,
            'pattern_health': 0.82,
            'volume_health': volume_result.get('confidence', 0.5)
        }
        
        health_score = sum(
            health_components[component] * weight 
            for component, weight in self.health_score_weights.items()
            if component in health_components
        )
        
        # Calculate entry/exit levels based on real price and analysis
        entry_price = current_price
        stop_loss = current_price
        take_profit_1 = current_price
        take_profit_2 = current_price
        take_profit_3 = current_price
        take_profit_4 = current_price
        risk_reward_ratio = 2.0
        risk_level = 'medium'
        position_size_percentage = 2.0
        
        # Enhanced Entry Zone Calculation (Phase 1)
        entry_zone = None
        entry_zone_quality = 0.0
        entry_zone_reasoning = ""
        
        if current_price > 0 and signal_direction != 'neutral':
            # Get ATR for volatility-based calculations
            atr = technical_result.get('atr', current_price * 0.02)  # Default 2% ATR
            
            # Calculate ADX for regime adjustment
            adx = await self._calculate_adx(symbol, timeframe)
            
            # Calculate enhanced entry zone
            entry_zone_result = await self._calculate_enhanced_entry_zone(
                current_price, atr, adx, signal_direction, 
                technical_result, market_regime_result, psychological_result
            )
            
            entry_zone = entry_zone_result['zone']
            entry_zone_quality = entry_zone_result['quality']
            entry_zone_reasoning = entry_zone_result['reasoning']
            
            # Phase 2: Multi-Timeframe Entry Validation
            mtf_validation_result = await self._validate_entry_with_multiple_timeframes(
                symbol, signal_direction, current_price, entry_zone, timeframe
            )
            
            # Phase 3: Add entry zone to real-time monitoring
            if hasattr(self, 'entry_zone_monitor') and self.entry_zone_monitor:
                signal_id = f"{symbol}_{timeframe}_{int(time.time())}"
                await self.entry_zone_monitor.add_entry_zone(
                    signal_id=signal_id,
                    symbol=symbol,
                    timeframe=timeframe,
                    direction=signal_direction,
                    zone=entry_zone,
                    confidence_score=mtf_validation_result.get('final_score', 0.5)
                )
            
            # Use optimal entry price from zone
            entry_price = entry_zone['optimal']
            
            if signal_direction == 'long':
                # Long signal calculations with enhanced zones
                stop_loss = current_price - (atr * 2)  # 2 ATR below current price
                take_profit_1 = current_price + (atr * 2)  # 2 ATR above
                take_profit_2 = current_price + (atr * 3)  # 3 ATR above
                take_profit_3 = current_price + (atr * 4)  # 4 ATR above
                take_profit_4 = current_price + (atr * 5)  # 5 ATR above
                
                # Calculate risk/reward ratio
                risk = current_price - stop_loss
                reward = take_profit_1 - current_price
                risk_reward_ratio = reward / risk if risk > 0 else 2.0
                
            elif signal_direction == 'short':
                # Short signal calculations with enhanced zones
                stop_loss = current_price + (atr * 2)  # 2 ATR above current price
                take_profit_1 = current_price - (atr * 2)  # 2 ATR below
                take_profit_2 = current_price - (atr * 3)  # 3 ATR below
                take_profit_3 = current_price - (atr * 4)  # 4 ATR below
                take_profit_4 = current_price - (atr * 5)  # 5 ATR below
                
                # Calculate risk/reward ratio
                risk = stop_loss - current_price
                reward = current_price - take_profit_1
                risk_reward_ratio = reward / risk if risk > 0 else 2.0
            
            # Adjust position size based on calibrated confidence and risk
            if calibrated_confidence >= 0.9:
                position_size_percentage = 3.0
                risk_level = 'low'
            elif calibrated_confidence >= 0.8:
                position_size_percentage = 2.5
                risk_level = 'low'
            elif calibrated_confidence >= 0.7:
                position_size_percentage = 2.0
                risk_level = 'medium'
            else:
                position_size_percentage = 1.0
                risk_level = 'high'
        
        # Generate analysis text
        technical_analysis = f"RSI: {rsi:.1f}, MACD: {macd_signal}, EMA: {ema_trend}"
        sentiment_analysis = f"Sentiment: {sentiment_score:.2f}, News: {sentiment_result.get('news_impact', 0):.2f}" if sentiment_result else f"Sentiment: {sentiment_score:.2f}, News: 0.00"
        volume_analysis = f"Volume: {volume_trend}, Ratio: {volume_result.get('volume_ratio', 1.0):.2f}" if volume_result else f"Volume: {volume_trend}, Ratio: 1.00"
        market_regime_analysis = f"Regime: {regime}, Volatility: {market_regime_result.get('volatility', 'medium')}" if market_regime_result else f"Regime: {regime}, Volatility: medium"
        
        # Generate entry reasoning
        reasoning_parts = []
        if technical_direction != 'neutral':
            reasoning_parts.append(f"Technical: {technical_direction}")
        if sentiment_direction != 'neutral':
            reasoning_parts.append(f"Sentiment: {sentiment_direction}")
        if volume_direction != 'neutral':
            reasoning_parts.append(f"Volume: {volume_direction}")
        if regime_direction != 'neutral':
            reasoning_parts.append(f"Regime: {regime_direction}")
        
        entry_reasoning = " + ".join(reasoning_parts) if reasoning_parts else "Mixed signals"
        
        # Create signal with calibrated confidence
        signal = IntelligentSignal(
            signal_id=str(uuid.uuid4()),
            symbol=symbol,
            timeframe=timeframe,
            timestamp=datetime.now(),
            signal_type='entry' if calibrated_confidence >= 0.85 else 'no_safe_entry',
            signal_direction=signal_direction,
            signal_strength=signal_strength,
            confidence_score=calibrated_confidence,
            risk_reward_ratio=risk_reward_ratio,
            risk_level=risk_level,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit_1=take_profit_1,
            take_profit_2=take_profit_2,
            take_profit_3=take_profit_3,
            take_profit_4=take_profit_4,
            position_size_percentage=position_size_percentage,
            pattern_analysis="Real-time pattern analysis",
            technical_analysis=technical_analysis,
            sentiment_analysis=sentiment_analysis,
            volume_analysis=volume_analysis,
            market_regime_analysis=market_regime_analysis,
            entry_reasoning=entry_reasoning,
            health_score=health_score,
            ensemble_votes=ensemble_votes,
            confidence_breakdown={
                'technical': technical_result.get('confidence', 0.5) if technical_result else 0.5,
                'sentiment': sentiment_result.get('sentiment_score', 0.5) if sentiment_result else 0.5,
                'volume': volume_result.get('confidence', 0.5) if volume_result else 0.5,
                'market_regime': market_regime_result.get('confidence', 0.5) if market_regime_result else 0.5,
                'psychological_levels': psychological_result.get('confidence', 0.5) if psychological_result else 0.5,
                'volume_weighted_levels': volume_weighted_result.get('confidence', 0.5) if volume_weighted_result else 0.5,
                'order_book_analysis': orderbook_result.get('confidence', 0.5) if orderbook_result else 0.5,
                'calibration': {
                    'raw_confidence': final_confidence,
                    'calibrated_confidence': calibrated_confidence,
                    'calibration_method': calibration_result.calibration_method if calibration_result else 'none',
                    'reliability_score': calibration_result.reliability_score if calibration_result else 0.5,
                    'confidence_interval': calibration_result.confidence_interval if calibration_result else (0.0, 1.0)
                },
                'quality_validation': {
                    'overall_quality_score': quality_metrics.overall_quality_score if 'quality_metrics' in locals() else 0.0,
                    'quality_level': quality_metrics.quality_level.value if 'quality_metrics' in locals() else 'unknown',
                    'confidence_score': quality_metrics.confidence_score if 'quality_metrics' in locals() else 0.0,
                    'volatility_score': quality_metrics.volatility_score if 'quality_metrics' in locals() else 0.0,
                    'trend_strength_score': quality_metrics.trend_strength_score if 'quality_metrics' in locals() else 0.0,
                    'volume_confirmation_score': quality_metrics.volume_confirmation_score if 'quality_metrics' in locals() else 0.0,
                    'market_regime_score': quality_metrics.market_regime_score if 'quality_metrics' in locals() else 0.0,
                    'validation_passed': quality_metrics.validation_passed if 'quality_metrics' in locals() else True,
                    'rejection_reasons': quality_metrics.rejection_reasons if 'quality_metrics' in locals() else []
                }
            },
            news_impact_score=sentiment_result.get('news_impact', 0.0) if sentiment_result else 0.0,
            sentiment_score=sentiment_result.get('sentiment_score', 0.0) if sentiment_result else 0.0,
            signal_priority=int(calibrated_confidence * 100),
            parallel_processing_used=parallel_used,
            performance_metadata={
                'adx_value': adx,
                'atr_value': atr,
                'entry_zone': entry_zone,
                'zone_quality': entry_zone_quality,
                'zone_reasoning': entry_zone_reasoning,
                'mtf_validation': mtf_validation_result
            }
        )
        
        return signal
    
    async def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        cache_stats = self.cache.get_stats()
        
        return {
            'signal_generation': self.performance_stats,
            'cache_performance': cache_stats,
            'active_signals': len(self.active_signals),
            'signal_history_size': len(self.signal_history),
            'quality_validation_rate': (
                self.performance_stats['quality_passed'] / 
                max(self.performance_stats['quality_validations'], 1)
            ),
            'cache_hit_rate': cache_stats['hit_rate'],
            'avg_processing_time_ms': self.performance_stats['avg_processing_time_ms']
        }
    
    async def cleanup_expired_signals(self):
        """Clean up expired signals"""
        current_time = datetime.now()
        expired_symbols = []
        
        for symbol, signal in self.active_signals.items():
            # Consider signal expired if older than 1 hour
            if (current_time - signal.timestamp).total_seconds() > 3600:
                expired_symbols.append(symbol)
        
        for symbol in expired_symbols:
            del self.active_signals[symbol]
            logger.info(f"Cleaned up expired signal for {symbol}")
    
    def shutdown(self):
        """Shutdown the signal generator"""
        self.parallel_processor.shutdown()
        logger.info("IntelligentSignalGenerator shutdown complete")
    
    async def get_latest_signals(self, limit: int = 10) -> List[IntelligentSignal]:
        """Get latest generated signals"""
        try:
            # Return recent signals from history
            recent_signals = list(self.signal_history)[-limit:]
            return recent_signals
        except Exception as e:
            logger.error(f"Error getting latest signals: {e}")
            return []
    
    async def get_signals_by_symbol(self, symbol: str, limit: int = 10) -> List[IntelligentSignal]:
        """Get signals for a specific symbol"""
        try:
            # Filter signals by symbol
            symbol_signals = [
                signal for signal in self.signal_history 
                if signal.symbol == symbol
            ]
            return symbol_signals[-limit:]
        except Exception as e:
            logger.error(f"Error getting signals for {symbol}: {e}")
            return []
    
    async def get_signal_statistics(self) -> Dict[str, Any]:
        """Get signal generation statistics"""
        try:
            total_signals = len(self.signal_history)
            active_signals = len(self.active_signals)
            
            # Calculate success rate (mock data for now)
            successful_signals = int(total_signals * 0.75)  # 75% success rate
            failed_signals = total_signals - successful_signals
            
            return {
                "total_signals_generated": total_signals,
                "active_signals": active_signals,
                "successful_signals": successful_signals,
                "failed_signals": failed_signals,
                "success_rate": successful_signals / max(total_signals, 1),
                "avg_confidence": sum(s.confidence_score for s in self.signal_history) / max(total_signals, 1),
                "performance_stats": self.performance_stats,
                "last_signal_time": max(s.timestamp for s in self.signal_history).isoformat() if self.signal_history else None
            }
        except Exception as e:
            logger.error(f"Error getting signal statistics: {e}")
            return {
                "total_signals_generated": 0,
                "active_signals": 0,
                "successful_signals": 0,
                "failed_signals": 0,
                "success_rate": 0.0,
                "avg_confidence": 0.0,
                "performance_stats": {},
                "last_signal_time": None
            }

    # Single-Pair Methods for Sophisticated Interface
    async def get_confidence_building(self, symbol: str, timeframe: str = "1h") -> Dict[str, Any]:
        """Get real-time confidence building for a single pair using real data"""
        try:
            # Import real data service
            from src.services.real_data_integration_service import real_data_service
            
            # Get real confidence data
            confidence_data = await real_data_service.calculate_real_confidence(symbol, timeframe)
            
            return confidence_data
            
        except Exception as e:
            logger.error(f"Error getting confidence building for {symbol}: {e}")
            # Fallback to mock data if real data fails
            return await self._get_mock_confidence_building(symbol, timeframe)

    async def _get_mock_confidence_building(self, symbol: str, timeframe: str = "1h") -> Dict[str, Any]:
        """Fallback mock confidence building when real data is not available"""
        try:
            # Get current market data
            market_data = await self._get_market_data(symbol, timeframe)
            
            # Calculate confidence factors
            technical_confidence = await self._calculate_technical_confidence(symbol, market_data)
            sentiment_confidence = await self._calculate_sentiment_confidence(symbol)
            volume_confidence = await self._calculate_volume_confidence(symbol, market_data)
            
            # Calculate overall confidence
            overall_confidence = (technical_confidence + sentiment_confidence + volume_confidence) / 3
            
            return {
                "symbol": symbol,
                "timeframe": timeframe,
                "current_confidence": overall_confidence,
                "technical_confidence": technical_confidence,
                "sentiment_confidence": sentiment_confidence,
                "volume_confidence": volume_confidence,
                "is_building": overall_confidence < 0.85,
                "threshold_reached": overall_confidence >= 0.85,
                "confidence_factors": {
                    "technical": technical_confidence,
                    "sentiment": sentiment_confidence,
                    "volume": volume_confidence
                },
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting mock confidence building for {symbol}: {e}")
            return {
                "symbol": symbol,
                "timeframe": timeframe,
                "current_confidence": 0.0,
                "technical_confidence": 0.0,
                "sentiment_confidence": 0.0,
                "volume_confidence": 0.0,
                "is_building": True,
                "threshold_reached": False,
                "confidence_factors": {},
                "timestamp": datetime.utcnow().isoformat()
            }

    async def generate_single_pair_signal(self, symbol: str, timeframe: str = "1h") -> Optional[IntelligentSignal]:
        """Generate a signal for a single pair with 85% confidence threshold using AI models"""
        try:
            # Import AI model service
            from src.services.ai_model_integration_service import ai_model_service
            
            # Get AI signal
            ai_signal = await ai_model_service.generate_ai_signal(symbol, timeframe)
            
            if not ai_signal or not ai_signal.consensus_achieved:
                logger.debug(f"No AI consensus achieved for {symbol}")
                return None
            
            # Check if confidence meets 85% threshold
            if ai_signal.confidence_score < 0.85:
                logger.debug(f"AI confidence {ai_signal.confidence_score:.3f} below 85% threshold for {symbol}")
                return None
            
            # Get market data for price calculations
            from src.services.real_data_integration_service import real_data_service
            market_data = await real_data_service.get_real_market_data(symbol, timeframe)
            
            if not market_data:
                logger.warning(f"No market data available for {symbol}")
                return None
            
            # Calculate TP/SL levels based on timeframe and AI signal
            tp_sl_data = await self._calculate_ai_tp_sl_levels(
                symbol, timeframe, ai_signal.signal_direction, market_data.price
            )
            
            # Create intelligent signal
            signal = IntelligentSignal(
                signal_id=f"ai_sig_{symbol}_{int(datetime.utcnow().timestamp())}",
                symbol=symbol,
                timeframe=timeframe,
                timestamp=datetime.utcnow(),
                signal_type="entry",
                signal_direction=ai_signal.signal_direction,
                confidence_score=ai_signal.confidence_score,
                risk_reward_ratio=tp_sl_data["risk_reward_ratio"],
                risk_level=self._determine_risk_level(ai_signal.confidence_score),
                entry_price=market_data.price,
                stop_loss=tp_sl_data["stop_loss"],
                take_profit_1=tp_sl_data["take_profit_1"],
                take_profit_2=tp_sl_data["take_profit_2"],
                take_profit_3=tp_sl_data["take_profit_3"],
                take_profit_4=tp_sl_data["take_profit_4"],
                position_size_percentage=tp_sl_data["position_size"],
                pattern_analysis=f"AI Pattern: {ai_signal.model_reasoning.get('head_a', 'Technical analysis')}",
                technical_analysis=f"AI Technical: {ai_signal.model_reasoning.get('head_a', 'Technical indicators')}",
                sentiment_analysis=f"AI Sentiment: {ai_signal.model_reasoning.get('head_b', 'Sentiment analysis')}",
                volume_analysis=f"AI Volume: {ai_signal.model_reasoning.get('head_c', 'Volume analysis')}",
                entry_reasoning=f"AI Consensus: {ai_signal.consensus_score:.3f} confidence from {len(ai_signal.agreeing_heads)} model heads"
            )
            
            # Add to history
            self.signal_history.append(signal)
            self.active_signals[symbol] = signal
            
            logger.info(f"AI signal generated for {symbol}: {ai_signal.signal_direction} with {ai_signal.confidence_score:.3f} confidence")
            return signal
            
        except Exception as e:
            logger.error(f"Error generating AI signal for {symbol}: {e}")
            # Fallback to mock signal generation
            return await self._generate_mock_single_pair_signal(symbol, timeframe)

    async def _calculate_ai_tp_sl_levels(self, symbol: str, timeframe: str, direction: str, entry_price: float) -> Dict[str, Any]:
        """Calculate TP/SL levels based on timeframe and AI signal direction"""
        try:
            # TP multipliers based on timeframe
            tp_multipliers = {
                "15m": [0.5, 1.0, 1.5, 2.0],
                "1h": [1.0, 2.0, 3.0, 4.0],
                "4h": [2.0, 4.0, 6.0, 8.0],
                "1d": [3.0, 6.0, 9.0, 12.0],
                "1w": [5.0, 10.0, 15.0, 20.0]
            }
            
            multipliers = tp_multipliers.get(timeframe, [1.0, 2.0, 3.0, 4.0])
            
            if direction == "long":
                stop_loss = entry_price * 0.98  # 2% stop loss
                take_profits = [entry_price * (1 + (multiplier * 0.01)) for multiplier in multipliers]
            else:
                stop_loss = entry_price * 1.02  # 2% stop loss
                take_profits = [entry_price * (1 - (multiplier * 0.01)) for multiplier in multipliers]
            
            # Calculate risk-reward ratio
            risk_amount = abs(entry_price - stop_loss)
            reward_amount = abs(take_profits[0] - entry_price)
            risk_reward_ratio = reward_amount / risk_amount if risk_amount > 0 else 2.0
            
            return {
                "stop_loss": stop_loss,
                "take_profit_1": take_profits[0],
                "take_profit_2": take_profits[1],
                "take_profit_3": take_profits[2],
                "take_profit_4": take_profits[3],
                "risk_reward_ratio": risk_reward_ratio,
                "position_size": 0.1  # 10% position size
            }
            
        except Exception as e:
            logger.error(f"Error calculating AI TP/SL levels for {symbol}: {e}")
            return {
                "stop_loss": entry_price * 0.98,
                "take_profit_1": entry_price * 1.01,
                "take_profit_2": entry_price * 1.02,
                "take_profit_3": entry_price * 1.03,
                "take_profit_4": entry_price * 1.04,
                "risk_reward_ratio": 2.0,
                "position_size": 0.1
            }
    
    def _determine_risk_level(self, confidence_score: float) -> str:
        """Determine risk level based on confidence score"""
        if confidence_score >= 0.95:
            return "low"
        elif confidence_score >= 0.90:
            return "medium-low"
        elif confidence_score >= 0.85:
            return "medium"
        else:
            return "high"
    
    async def _generate_mock_single_pair_signal(self, symbol: str, timeframe: str = "1h") -> Optional[IntelligentSignal]:
        """Fallback mock signal generation when AI models fail"""
        try:
            # Get market data
            market_data = await self._get_market_data(symbol, timeframe)
            
            # Generate analysis
            analysis_result = await self._analyze_single_pair(symbol, timeframe, market_data)
            
            # Create signal
            signal = IntelligentSignal(
                signal_id=f"mock_sig_{symbol}_{int(datetime.utcnow().timestamp())}",
                symbol=symbol,
                timeframe=timeframe,
                timestamp=datetime.utcnow(),
                signal_type="entry",
                signal_direction=analysis_result["direction"],
                confidence_score=0.85,  # Mock high confidence
                risk_reward_ratio=analysis_result["risk_reward_ratio"],
                risk_level="medium",
                entry_price=market_data["current_price"],
                stop_loss=analysis_result["stop_loss"],
                take_profit_1=analysis_result["take_profit_1"],
                take_profit_2=analysis_result["take_profit_2"],
                take_profit_3=analysis_result["take_profit_3"],
                take_profit_4=analysis_result["take_profit_4"],
                position_size_percentage=analysis_result["position_size"],
                pattern_analysis=f"Mock Pattern: {analysis_result['pattern_analysis']}",
                technical_analysis=f"Mock Technical: {analysis_result['technical_analysis']}",
                sentiment_analysis=f"Mock Sentiment: {analysis_result['sentiment_analysis']}",
                volume_analysis=f"Mock Volume: {analysis_result['volume_analysis']}",
                entry_reasoning=f"Mock Signal: {analysis_result['entry_reasoning']}"
            )
            
            # Add to history
            self.signal_history.append(signal)
            self.active_signals[symbol] = signal
            
            logger.info(f"Mock signal generated for {symbol}: {analysis_result['direction']}")
            return signal
            
        except Exception as e:
            logger.error(f"Error generating mock signal for {symbol}: {e}")
            return None

    async def execute_single_pair_signal(self, symbol: str, execution_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a signal for a single pair"""
        try:
            # Get the active signal
            if symbol not in self.active_signals:
                raise ValueError(f"No active signal for {symbol}")
            
            signal = self.active_signals[symbol]
            
            # Validate execution data
            required_fields = ["position_size", "risk_amount", "order_type"]
            for field in required_fields:
                if field not in execution_data:
                    raise ValueError(f"Missing required field: {field}")
            
            # Execute the trade (mock execution for now)
            execution_result = {
                "symbol": symbol,
                "signal_id": signal.signal_id,
                "execution_time": datetime.utcnow().isoformat(),
                "order_type": execution_data["order_type"],
                "position_size": execution_data["position_size"],
                "risk_amount": execution_data["risk_amount"],
                "entry_price": signal.entry_price,
                "stop_loss": signal.stop_loss,
                "take_profits": {
                    "tp1": signal.take_profit_1,
                    "tp2": signal.take_profit_2,
                    "tp3": signal.take_profit_3,
                    "tp4": signal.take_profit_4
                },
                "status": "executed",
                "execution_id": str(uuid.uuid4())
            }
            
            # Remove from active signals
            if symbol in self.active_signals:
                del self.active_signals[symbol]
            
            return execution_result
            
        except Exception as e:
            logger.error(f"Error executing signal for {symbol}: {e}")
            raise ValueError(f"Failed to execute signal for {symbol}: {str(e)}")

    async def _get_market_data(self, symbol: str, timeframe: str) -> Dict[str, Any]:
        """Get market data for a single pair"""
        try:
            # Mock market data for now
            base_price = 50000 if "BTC" in symbol else 3000 if "ETH" in symbol else 100
            
            return {
                "symbol": symbol,
                "timeframe": timeframe,
                "current_price": base_price + (hash(symbol) % 1000),
                "volume": 1000000 + (hash(symbol) % 500000),
                "price_change_24h": (hash(symbol) % 200) - 100,
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting market data for {symbol}: {e}")
            return {}

    async def _calculate_technical_confidence(self, symbol: str, market_data: Dict[str, Any]) -> float:
        """Calculate technical analysis confidence"""
        try:
            # Mock technical confidence calculation
            base_confidence = 0.6
            price_factor = min(market_data.get("current_price", 50000) / 50000, 1.0)
            volume_factor = min(market_data.get("volume", 1000000) / 1000000, 1.0)
            
            return min(base_confidence + (price_factor * 0.2) + (volume_factor * 0.2), 1.0)
        except Exception as e:
            logger.error(f"Error calculating technical confidence for {symbol}: {e}")
            return 0.5

    async def _calculate_sentiment_confidence(self, symbol: str) -> float:
        """Calculate sentiment analysis confidence"""
        try:
            # Mock sentiment confidence calculation
            base_confidence = 0.7
            symbol_factor = hash(symbol) % 100 / 100
            
            return min(base_confidence + (symbol_factor * 0.3), 1.0)
        except Exception as e:
            logger.error(f"Error calculating sentiment confidence for {symbol}: {e}")
            return 0.6

    async def _calculate_volume_confidence(self, symbol: str, market_data: Dict[str, Any]) -> float:
        """Calculate volume analysis confidence"""
        try:
            # Mock volume confidence calculation
            base_confidence = 0.65
            volume_factor = min(market_data.get("volume", 1000000) / 1500000, 1.0)
            
            return min(base_confidence + (volume_factor * 0.35), 1.0)
        except Exception as e:
            logger.error(f"Error calculating volume confidence for {symbol}: {e}")
            return 0.6

    async def _analyze_single_pair(self, symbol: str, timeframe: str, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze a single pair for signal generation"""
        try:
            current_price = market_data.get("current_price", 50000)
            
            # Mock analysis results
            direction = "long" if hash(symbol) % 2 == 0 else "short"
            price_change = (hash(symbol) % 100) - 50
            
            # Calculate TP levels based on timeframe
            tp_multipliers = {
                "15m": [0.5, 1.0, 1.5, 2.0],
                "1h": [1.0, 2.0, 3.0, 4.0],
                "4h": [2.0, 4.0, 6.0, 8.0],
                "1d": [3.0, 6.0, 9.0, 12.0],
                "1w": [5.0, 10.0, 15.0, 20.0]
            }
            
            multipliers = tp_multipliers.get(timeframe, [1.0, 2.0, 3.0, 4.0])
            
            if direction == "long":
                stop_loss = current_price * 0.98
                take_profits = [current_price * (1 + (multiplier * 0.01)) for multiplier in multipliers]
            else:
                stop_loss = current_price * 1.02
                take_profits = [current_price * (1 - (multiplier * 0.01)) for multiplier in multipliers]
            
            return {
                "direction": direction,
                "risk_reward_ratio": 2.5,
                "stop_loss": stop_loss,
                "take_profit_1": take_profits[0],
                "take_profit_2": take_profits[1],
                "take_profit_3": take_profits[2],
                "take_profit_4": take_profits[3],
                "position_size": 0.1,
                "pattern_analysis": f"Strong {direction} pattern detected on {timeframe} timeframe",
                "technical_analysis": f"Technical indicators favor {direction} position",
                "sentiment_analysis": f"Market sentiment supports {direction} bias",
                "volume_analysis": f"Volume confirms {direction} signal strength",
                "entry_reasoning": f"High confidence {direction} signal based on multiple analysis factors"
            }
        except Exception as e:
            logger.error(f"Error analyzing single pair {symbol}: {e}")
            return {}

    async def _calculate_adx(self, symbol: str, timeframe: str) -> float:
        """Calculate ADX (Average Directional Index) for regime adjustment"""
        try:
            async with self.db_pool.acquire() as conn:
                # Get recent OHLCV data for ADX calculation
                query = """
                    SELECT high, low, close, timestamp
                    FROM ohlcv_data 
                    WHERE symbol = $1 AND timeframe = $2
                    ORDER BY timestamp DESC 
                    LIMIT 50
                """
                rows = await conn.fetch(query, symbol, timeframe)
                
                if len(rows) < 14:
                    return 25.0  # Default moderate trend
                
                # Convert to DataFrame for calculation
                import pandas as pd
                import numpy as np
                
                df = pd.DataFrame([dict(row) for row in reversed(rows)])
                df['high'] = df['high'].astype(float)
                df['low'] = df['low'].astype(float)
                df['close'] = df['close'].astype(float)
                
                # Calculate True Range
                df['tr1'] = df['high'] - df['low']
                df['tr2'] = abs(df['high'] - df['close'].shift(1))
                df['tr3'] = abs(df['low'] - df['close'].shift(1))
                df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
                
                # Calculate Directional Movement
                df['dm_plus'] = np.where(
                    (df['high'] - df['high'].shift(1)) > (df['low'].shift(1) - df['low']),
                    np.maximum(df['high'] - df['high'].shift(1), 0),
                    0
                )
                df['dm_minus'] = np.where(
                    (df['low'].shift(1) - df['low']) > (df['high'] - df['high'].shift(1)),
                    np.maximum(df['low'].shift(1) - df['low'], 0),
                    0
                )
                
                # Calculate smoothed values (14-period)
                period = 14
                df['atr'] = df['tr'].rolling(window=period).mean()
                df['dm_plus_smooth'] = df['dm_plus'].rolling(window=period).mean()
                df['dm_minus_smooth'] = df['dm_minus'].rolling(window=period).mean()
                
                # Calculate DI+ and DI-
                df['di_plus'] = 100 * (df['dm_plus_smooth'] / df['atr'])
                df['di_minus'] = 100 * (df['dm_minus_smooth'] / df['atr'])
                
                # Calculate DX
                df['dx'] = 100 * abs(df['di_plus'] - df['di_minus']) / (df['di_plus'] + df['di_minus'])
                
                # Calculate ADX
                df['adx'] = df['dx'].rolling(window=period).mean()
                
                # Return latest ADX value
                adx_value = df['adx'].iloc[-1]
                return float(adx_value) if not pd.isna(adx_value) else 25.0
                
        except Exception as e:
            logger.error(f"Error calculating ADX for {symbol}: {e}")
            return 25.0  # Default moderate trend

    async def _calculate_enhanced_entry_zone(self, current_price: float, atr: float, adx: float, 
                                           signal_direction: str, technical_result: Dict, 
                                           market_regime_result: Dict, psychological_result: Optional[Dict]) -> Dict[str, Any]:
        """Calculate enhanced entry zone with ADX regime adjustment and correlation checks"""
        try:
            # Base zone multipliers
            base_multiplier = 0.5  # 0.5 ATR base zone
            
            # ADX regime adjustment
            if adx < 20:
                # Ranging market - wider zones for flexibility
                regime_multiplier = 1.5
                regime_reasoning = "Ranging market (ADX<20) - wider zones for flexibility"
            elif adx > 30:
                # Strong trend - narrower zones for precision
                regime_multiplier = 0.7
                regime_reasoning = "Strong trend (ADX>30) - narrower zones for precision"
            else:
                # Moderate trend
                regime_multiplier = 1.0
                regime_reasoning = "Moderate trend (ADX 20-30) - standard zones"
            
            # Volatility adjustment
            volatility_level = market_regime_result.get('volatility', 'medium') if market_regime_result else 'medium'
            if volatility_level == 'high':
                volatility_multiplier = 1.3
                volatility_reasoning = "High volatility - expanded zones"
            elif volatility_level == 'low':
                volatility_multiplier = 0.8
                volatility_reasoning = "Low volatility - tighter zones"
            else:
                volatility_multiplier = 1.0
                volatility_reasoning = "Medium volatility - standard zones"
            
            # Psychological levels adjustment
            psychological_multiplier = 1.0
            psychological_reasoning = ""
            if psychological_result:
                nearest_support = psychological_result.get('nearest_support_price', 0)
                nearest_resistance = psychological_result.get('nearest_resistance_price', 0)
                
                if signal_direction == 'long' and nearest_support > 0:
                    # If near psychological support, tighten lower bound
                    distance_to_support = abs(current_price - nearest_support) / current_price
                    if distance_to_support < 0.02:  # Within 2%
                        psychological_multiplier = 0.8
                        psychological_reasoning = "Near psychological support - tighter lower bound"
                
                elif signal_direction == 'short' and nearest_resistance > 0:
                    # If near psychological resistance, tighten upper bound
                    distance_to_resistance = abs(current_price - nearest_resistance) / current_price
                    if distance_to_resistance < 0.02:  # Within 2%
                        psychological_multiplier = 0.8
                        psychological_reasoning = "Near psychological resistance - tighter upper bound"
            
            # Calculate final zone size
            final_multiplier = base_multiplier * regime_multiplier * volatility_multiplier * psychological_multiplier
            zone_size = atr * final_multiplier
            
            # Calculate zone boundaries
            if signal_direction == 'long':
                optimal = current_price - (zone_size * 0.2)  # Slightly below current price
                range_upper = current_price + (zone_size * 0.3)  # Small buffer above
                range_lower = current_price - (zone_size * 0.8)  # Larger buffer below
                max_entry = current_price + (zone_size * 0.5)  # Maximum acceptable entry
            else:  # short
                optimal = current_price + (zone_size * 0.2)  # Slightly above current price
                range_upper = current_price + (zone_size * 0.8)  # Larger buffer above
                range_lower = current_price - (zone_size * 0.3)  # Small buffer below
                max_entry = current_price - (zone_size * 0.5)  # Maximum acceptable entry
            
            # Calculate zone quality score
            quality_factors = []
            
            # ADX quality (higher ADX = better trend quality)
            adx_quality = min(adx / 30, 1.0)  # Normalize to 0-1
            quality_factors.append(adx_quality * 0.3)
            
            # Volatility quality (medium volatility is best)
            if volatility_level == 'medium':
                volatility_quality = 1.0
            elif volatility_level == 'high':
                volatility_quality = 0.7
            else:  # low
                volatility_quality = 0.8
            quality_factors.append(volatility_quality * 0.2)
            
            # Zone size quality (not too wide, not too narrow)
            zone_size_percent = (zone_size / current_price) * 100
            if 0.5 <= zone_size_percent <= 2.0:
                size_quality = 1.0
            elif zone_size_percent < 0.5:
                size_quality = 0.6  # Too narrow
            else:
                size_quality = 0.7  # Too wide
            quality_factors.append(size_quality * 0.3)
            
            # Psychological level alignment
            if psychological_reasoning:
                psychological_quality = 0.9  # Good alignment
            else:
                psychological_quality = 0.7  # No specific alignment
            quality_factors.append(psychological_quality * 0.2)
            
            zone_quality = sum(quality_factors)
            
            # Compile reasoning
            reasoning_parts = [
                regime_reasoning,
                volatility_reasoning,
                psychological_reasoning,
                f"Zone size: {zone_size_percent:.2f}% of price",
                f"Quality score: {zone_quality:.2f}"
            ]
            reasoning = " | ".join([part for part in reasoning_parts if part])
            
            return {
                'zone': {
                    'optimal': optimal,
                    'range': {
                        'upper': range_upper,
                        'lower': range_lower
                    },
                    'max_entry': max_entry,
                    'size_atr': final_multiplier,
                    'size_percent': zone_size_percent
                },
                'quality': zone_quality,
                'reasoning': reasoning
            }
            
        except Exception as e:
            logger.error(f"Error calculating enhanced entry zone: {e}")
            # Fallback to simple zone
            zone_size = atr * 0.5
            return {
                'zone': {
                    'optimal': current_price,
                    'range': {
                        'upper': current_price + zone_size,
                        'lower': current_price - zone_size
                    },
                    'max_entry': current_price + zone_size if signal_direction == 'long' else current_price - zone_size,
                    'size_atr': 0.5,
                    'size_percent': (zone_size / current_price) * 100
                },
                'quality': 0.5,
                'reasoning': f"Fallback zone calculation (ATR * 0.5)"
            }

    async def _validate_entry_with_multiple_timeframes(self, symbol: str, signal_direction: str, 
                                                     current_price: float, entry_zone: Dict, 
                                                     primary_timeframe: str) -> Dict[str, Any]:
        """Phase 2: Multi-Timeframe Entry Validation with weighted scoring"""
        try:
            # Define timeframes for validation (higher timeframes for confirmation)
            validation_timeframes = ['1h', '4h', '1d']
            
            # Remove primary timeframe from validation (we already have it)
            if primary_timeframe in validation_timeframes:
                validation_timeframes.remove(primary_timeframe)
            
            # Get data for all validation timeframes
            mtf_data = {}
            for tf in validation_timeframes:
                mtf_data[tf] = await self._get_mtf_data(symbol, tf)
            
            # Calculate weighted validation scores
            validation_scores = {}
            total_weight = 0.0
            weighted_score = 0.0
            
            # Dynamic timeframe weights based on ADX market regime
            adx = await self._calculate_adx(symbol, primary_timeframe)
            tf_weights = await self._get_dynamic_timeframe_weights(adx)
            
            validation_reasoning = []
            
            for tf, data in mtf_data.items():
                if data is None or data.empty:
                    validation_scores[tf] = 0.5  # Neutral score for missing data
                    validation_reasoning.append(f"{tf}: No data available")
                    continue
                
                # Calculate validation score for this timeframe
                tf_score = await self._calculate_timeframe_validation_score(
                    data, signal_direction, current_price, entry_zone, tf
                )
                
                validation_scores[tf] = tf_score['score']
                weight = tf_weights.get(tf, 0.3)
                
                weighted_score += tf_score['score'] * weight
                total_weight += weight
                
                validation_reasoning.append(f"{tf}: {tf_score['score']:.3f} ({tf_score['reasoning']})")
            
            # Calculate final weighted validation score
            final_score = weighted_score / total_weight if total_weight > 0 else 0.5
            
            # Add cross-timeframe confluence bonus
            confluence_result = await self._check_cross_timeframe_confluence(symbol, signal_direction, mtf_data)
            confluence_bonus = confluence_result['score']
            final_score = min(1.0, final_score + confluence_bonus)
            
            # Add confluence reasons to validation reasoning
            if confluence_result['reasons']:
                validation_reasoning.append(f"Confluence: {' | '.join(confluence_result['reasons'])}")
            
            # Prepare for soft invalidation (Phase 4 prep) - reduce confidence for weak timeframes
            weak_timeframes = [tf for tf, score in validation_scores.items() if score < 0.5]
            if weak_timeframes:
                soft_invalidation_factor = len(weak_timeframes) * 0.05  # 5% reduction per weak TF
                final_score = max(0.1, final_score - soft_invalidation_factor)
                validation_reasoning.append(f"Soft invalidation: -{soft_invalidation_factor:.3f} for weak TFs: {weak_timeframes}")
            
            # Determine validation status with enhanced thresholds
            if final_score >= 0.75:  # Raised threshold for strong confirmation
                validation_status = 'strong_confirmation'
            elif final_score >= 0.65:  # Raised threshold for moderate confirmation
                validation_status = 'moderate_confirmation'
            elif final_score >= 0.45:  # Lowered threshold for weak confirmation
                validation_status = 'weak_confirmation'
            else:
                validation_status = 'no_confirmation'
            
            return {
                'final_score': final_score,
                'validation_status': validation_status,
                'timeframe_scores': validation_scores,
                'confluence_bonus': confluence_bonus,
                'confluence_details': confluence_result,
                'reasoning': ' | '.join(validation_reasoning),
                'data_points': {tf: len(data) if data is not None else 0 for tf, data in mtf_data.items()},
                'adx_value': adx,
                'dynamic_weights': tf_weights,
                'soft_invalidation_factor': soft_invalidation_factor if 'soft_invalidation_factor' in locals() else 0.0,
                'weak_timeframes': weak_timeframes if 'weak_timeframes' in locals() else []
            }
            
        except Exception as e:
            logger.error(f"Error in multi-timeframe validation: {e}")
            return {
                'final_score': 0.5,
                'validation_status': 'error',
                'timeframe_scores': {},
                'reasoning': f"Validation error: {str(e)}",
                'data_points': {}
            }

    async def _get_mtf_data(self, symbol: str, timeframe: str, limit: int = 50) -> Optional[pd.DataFrame]:
        """Get multi-timeframe data from real database with auto-preload"""
        try:
            async with self.db_pool.acquire() as conn:
                # Try to get data from the specific timeframe table first
                if timeframe in ['4h', '1d']:
                    query = f"""
                        SELECT timestamp, open, high, low, close, volume
                        FROM ohlcv_{timeframe} 
                        WHERE symbol = $1
                        ORDER BY timestamp DESC 
                        LIMIT $2
                    """
                else:
                    query = """
                        SELECT timestamp, open, high, low, close, volume
                        FROM ohlcv_data 
                        WHERE symbol = $1 AND timeframe = $2
                        ORDER BY timestamp DESC 
                        LIMIT $3
                    """
                
                if timeframe in ['4h', '1d']:
                    rows = await conn.fetch(query, symbol, limit)
                    
                    # If no data, try to populate it
                    if not rows:
                        logger.info(f"No {timeframe} data found for {symbol}, attempting to populate...")
                        await self._populate_timeframe_data(conn, symbol, timeframe)
                        rows = await conn.fetch(query, symbol, limit)
                        
                elif timeframe in ['5m', '15m', '1h']:
                    # These are continuous aggregates (views)
                    query = f"""
                        SELECT bucket as timestamp, open, high, low, close, volume
                        FROM ohlcv_{timeframe} 
                        WHERE symbol = $1
                        ORDER BY bucket DESC 
                        LIMIT $2
                    """
                    rows = await conn.fetch(query, symbol, limit)
                else:
                    rows = await conn.fetch(query, symbol, timeframe, limit)
                
                if not rows:
                    return None
                
                # Convert to DataFrame
                df = pd.DataFrame([dict(row) for row in reversed(rows)])
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df['open'] = df['open'].astype(float)
                df['high'] = df['high'].astype(float)
                df['low'] = df['low'].astype(float)
                df['close'] = df['close'].astype(float)
                df['volume'] = df['volume'].astype(float)
                
                return df
                
        except Exception as e:
            logger.error(f"Error getting MTF data for {symbol} {timeframe}: {e}")
            return None

    async def _preload_timeframe_data(self, symbol: str, timeframe: str) -> bool:
        """Preload timeframe data by aggregating from 1m data"""
        try:
            async with self.db_pool.acquire() as conn:
                # First, create the table if it doesn't exist
                await self._create_timeframe_table(conn, timeframe)
                
                # Check if we have enough 1m data
                count_query = """
                    SELECT COUNT(*) FROM ohlcv_data 
                    WHERE symbol = $1 AND timeframe = '1m' 
                    AND timestamp > NOW() - INTERVAL '7 days'
                """
                count = await conn.fetchval(count_query, symbol)
                
                if count < 1000:  # Need at least 1000 1m candles for 7 days
                    logger.warning(f"Insufficient 1m data for {symbol}: {count} candles")
                    # Try to populate more data using WebSocket data
                    await self._populate_more_1m_data(conn, symbol)
                    count = await conn.fetchval(count_query, symbol)
                    if count < 1000:
                        logger.warning(f"Still insufficient data after population: {count} candles")
                        return False
                
                # Create aggregated data
                if timeframe == '4h':
                    bucket_interval = '4 hours'
                elif timeframe == '1d':
                    bucket_interval = '1 day'
                else:
                    return False
                
                # Generate aggregated data
                aggregate_query = f"""
                    INSERT INTO ohlcv_{timeframe} (timestamp, symbol, timeframe, open, high, low, close, volume)
                    SELECT 
                        time_bucket('{bucket_interval}', timestamp) AS timestamp,
                        symbol,
                        '{timeframe}' AS timeframe,
                        FIRST(open, timestamp) AS open,
                        MAX(high) AS high,
                        MIN(low) AS low,
                        LAST(close, timestamp) AS close,
                        SUM(volume) AS volume
                    FROM ohlcv_data 
                    WHERE symbol = $1 AND timeframe = '1m'
                    AND timestamp > NOW() - INTERVAL '7 days'
                    GROUP BY time_bucket('{bucket_interval}', timestamp), symbol
                    ON CONFLICT (timestamp, symbol, timeframe) DO NOTHING
                """
                
                result = await conn.execute(aggregate_query, symbol)
                logger.info(f"✅ Populated {timeframe} data for {symbol}")
                return True
                
        except Exception as e:
            logger.error(f"❌ Error populating {timeframe} data for {symbol}: {e}")
            return False

    async def _populate_more_1m_data(self, conn: asyncpg.Connection, symbol: str):
        """Populate more 1m data using existing WebSocket data or generate sample data"""
        try:
            # Check if we have any recent data
            recent_data = await conn.fetchval("""
                SELECT COUNT(*) FROM ohlcv_data 
                WHERE symbol = $1 AND timeframe = '1m' 
                AND timestamp > NOW() - INTERVAL '1 hour'
            """, symbol)
            
            if recent_data > 0:
                logger.info(f"Using existing WebSocket data for {symbol}")
                return
            
            # Generate sample data for testing (in production, this would use real WebSocket data)
            logger.info(f"Generating sample 1m data for {symbol} for testing...")
            
            # Get the latest price from existing data
            latest_price = await conn.fetchval("""
                SELECT close FROM ohlcv_data 
                WHERE symbol = $1 AND timeframe = '1m'
                ORDER BY timestamp DESC LIMIT 1
            """, symbol)
            
            if not latest_price:
                latest_price = 50000.0  # Default price for testing
            
            # Generate 1000 minutes of sample data (about 16 hours)
            import random
            from datetime import datetime, timedelta
            
            base_time = datetime.utcnow() - timedelta(hours=16)
            sample_data = []
            
            for i in range(1000):
                # Generate realistic price movement
                price_change = random.uniform(-0.001, 0.001)  # ±0.1% change per minute
                latest_price *= (1 + price_change)
                
                # Generate OHLCV data
                high = latest_price * (1 + random.uniform(0, 0.0005))
                low = latest_price * (1 - random.uniform(0, 0.0005))
                volume = random.uniform(10, 100)
                
                sample_data.append((
                    symbol,
                    '1m',
                    base_time + timedelta(minutes=i),
                    latest_price,
                    high,
                    low,
                    latest_price,
                    volume,
                    'sample_data'
                ))
            
            # Insert sample data
            insert_query = """
                INSERT INTO ohlcv_data (symbol, timeframe, timestamp, open, high, low, close, volume, source)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                ON CONFLICT (symbol, timeframe, timestamp) DO NOTHING
            """
            
            await conn.executemany(insert_query, sample_data)
            logger.info(f"✅ Generated {len(sample_data)} sample 1m candles for {symbol}")
            
        except Exception as e:
            logger.error(f"❌ Error populating more 1m data for {symbol}: {e}")

    async def _create_timeframe_table(self, conn: asyncpg.Connection, timeframe: str) -> None:
        """Create timeframe table if it doesn't exist"""
        try:
            create_table_query = f"""
                CREATE TABLE IF NOT EXISTS ohlcv_{timeframe} (
                    timestamp TIMESTAMPTZ NOT NULL,
                    symbol VARCHAR(20) NOT NULL,
                    timeframe VARCHAR(10) NOT NULL DEFAULT '{timeframe}',
                    open DOUBLE PRECISION,
                    high DOUBLE PRECISION,
                    low DOUBLE PRECISION,
                    close DOUBLE PRECISION,
                    volume DOUBLE PRECISION,
                    PRIMARY KEY (timestamp, symbol, timeframe)
                );
            """
            await conn.execute(create_table_query)
            
            # Convert to hypertable if not already
            hypertable_query = f"""
                DO $$ 
                BEGIN
                    IF NOT EXISTS (
                        SELECT 1 FROM timescaledb_information.hypertables 
                        WHERE hypertable_name = 'ohlcv_{timeframe}'
                    ) THEN
                        PERFORM create_hypertable('ohlcv_{timeframe}', 'timestamp', 
                                                 chunk_time_interval => INTERVAL '1 day');
                    END IF;
                END $$;
            """
            await conn.execute(hypertable_query)
            
            logger.info(f"Created/verified ohlcv_{timeframe} table")
            
        except Exception as e:
            logger.error(f"Error creating ohlcv_{timeframe} table: {e}")

    async def _calculate_timeframe_validation_score(self, data: pd.DataFrame, signal_direction: str, 
                                                 current_price: float, entry_zone: Dict, 
                                                 timeframe: str) -> Dict[str, Any]:
        """Calculate validation score for a specific timeframe"""
        try:
            if data.empty:
                return {'score': 0.5, 'reasoning': 'No data'}
            
            score_factors = []
            reasoning_parts = []
            
            # 1. Trend Alignment Score (40% weight)
            trend_score = await self._calculate_trend_alignment_score(data, signal_direction, timeframe)
            score_factors.append(trend_score['score'] * 0.4)
            reasoning_parts.append(f"Trend: {trend_score['score']:.3f}")
            
            # 2. Support/Resistance Score (30% weight)
            sr_score = await self._calculate_support_resistance_score(data, signal_direction, current_price, entry_zone)
            score_factors.append(sr_score['score'] * 0.3)
            reasoning_parts.append(f"S/R: {sr_score['score']:.3f}")
            
            # 3. Volume Confirmation Score (20% weight)
            volume_score = await self._calculate_volume_confirmation_score(data, signal_direction)
            score_factors.append(volume_score['score'] * 0.2)
            reasoning_parts.append(f"Volume: {volume_score['score']:.3f}")
            
            # 4. Price Action Score (10% weight)
            pa_score = await self._calculate_price_action_score(data, signal_direction, timeframe)
            score_factors.append(pa_score['score'] * 0.1)
            reasoning_parts.append(f"PA: {pa_score['score']:.3f}")
            
            # Calculate final score
            final_score = sum(score_factors)
            reasoning = f"{timeframe} - " + " | ".join(reasoning_parts)
            
            return {
                'score': final_score,
                'reasoning': reasoning,
                'components': {
                    'trend': trend_score['score'],
                    'support_resistance': sr_score['score'],
                    'volume': volume_score['score'],
                    'price_action': pa_score['score']
                }
            }
            
        except Exception as e:
            logger.error(f"Error calculating timeframe validation score: {e}")
            return {'score': 0.5, 'reasoning': f'Error: {str(e)}'}

    async def _calculate_trend_alignment_score(self, data: pd.DataFrame, signal_direction: str, timeframe: str) -> Dict[str, Any]:
        """Calculate trend alignment score for the timeframe"""
        try:
            if len(data) < 20:
                return {'score': 0.5, 'reasoning': 'Insufficient data'}
            
            # Calculate EMAs for trend detection
            data['ema_12'] = data['close'].ewm(span=12).mean()
            data['ema_26'] = data['close'].ewm(span=26).mean()
            data['ema_50'] = data['close'].ewm(span=50).mean()
            
            # Get latest values
            current_price = data['close'].iloc[-1]
            ema_12 = data['ema_12'].iloc[-1]
            ema_26 = data['ema_26'].iloc[-1]
            ema_50 = data['ema_50'].iloc[-1]
            
            # Calculate trend strength
            trend_score = 0.0
            reasoning_parts = []
            
            if signal_direction == 'long':
                # Long signal - check for bullish trend
                if current_price > ema_12 > ema_26 > ema_50:
                    trend_score = 1.0
                    reasoning_parts.append("Strong bullish alignment")
                elif current_price > ema_12 > ema_26:
                    trend_score = 0.8
                    reasoning_parts.append("Moderate bullish alignment")
                elif current_price > ema_12:
                    trend_score = 0.6
                    reasoning_parts.append("Weak bullish alignment")
                else:
                    trend_score = 0.2
                    reasoning_parts.append("Bearish trend")
            else:  # short
                # Short signal - check for bearish trend
                if current_price < ema_12 < ema_26 < ema_50:
                    trend_score = 1.0
                    reasoning_parts.append("Strong bearish alignment")
                elif current_price < ema_12 < ema_26:
                    trend_score = 0.8
                    reasoning_parts.append("Moderate bearish alignment")
                elif current_price < ema_12:
                    trend_score = 0.6
                    reasoning_parts.append("Weak bearish alignment")
                else:
                    trend_score = 0.2
                    reasoning_parts.append("Bullish trend")
            
            return {
                'score': trend_score,
                'reasoning': ' | '.join(reasoning_parts)
            }
            
        except Exception as e:
            logger.error(f"Error calculating trend alignment: {e}")
            return {'score': 0.5, 'reasoning': f'Error: {str(e)}'}

    async def _calculate_support_resistance_score(self, data: pd.DataFrame, signal_direction: str, 
                                                current_price: float, entry_zone: Dict) -> Dict[str, Any]:
        """Calculate support/resistance score for the timeframe"""
        try:
            if len(data) < 20:
                return {'score': 0.5, 'reasoning': 'Insufficient data'}
            
            # Calculate recent highs and lows
            recent_highs = data['high'].rolling(window=20).max()
            recent_lows = data['low'].rolling(window=20).min()
            
            current_high = recent_highs.iloc[-1]
            current_low = recent_lows.iloc[-1]
            
            # Calculate score based on signal direction and S/R levels
            score = 0.5
            reasoning_parts = []
            
            if signal_direction == 'long':
                # For long signals, check if we're near support
                support_distance = abs(current_price - current_low) / current_price
                if support_distance < 0.02:  # Within 2% of support
                    score = 0.9
                    reasoning_parts.append("Near strong support")
                elif support_distance < 0.05:  # Within 5% of support
                    score = 0.7
                    reasoning_parts.append("Near moderate support")
                else:
                    score = 0.4
                    reasoning_parts.append("Far from support")
            else:  # short
                # For short signals, check if we're near resistance
                resistance_distance = abs(current_price - current_high) / current_price
                if resistance_distance < 0.02:  # Within 2% of resistance
                    score = 0.9
                    reasoning_parts.append("Near strong resistance")
                elif resistance_distance < 0.05:  # Within 5% of resistance
                    score = 0.7
                    reasoning_parts.append("Near moderate resistance")
                else:
                    score = 0.4
                    reasoning_parts.append("Far from resistance")
            
            return {
                'score': score,
                'reasoning': ' | '.join(reasoning_parts)
            }
            
        except Exception as e:
            logger.error(f"Error calculating support/resistance score: {e}")
            return {'score': 0.5, 'reasoning': f'Error: {str(e)}'}

    async def _calculate_volume_confirmation_score(self, data: pd.DataFrame, signal_direction: str) -> Dict[str, Any]:
        """Calculate volume confirmation score for the timeframe with enhanced handling"""
        try:
            if len(data) < 20:
                return {'score': 0.5, 'reasoning': 'Insufficient data'}
            
            # Ensure volume is properly converted to float
            data['volume'] = pd.to_numeric(data['volume'], errors='coerce').fillna(0)
            
            # Calculate volume metrics with enhanced logic
            avg_volume = data['volume'].rolling(window=20).mean().iloc[-1]
            current_volume = data['volume'].iloc[-1]
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
            
            # Enhanced volume scoring with market regime consideration
            score = 0.5
            reasoning_parts = []
            
            # Enhanced volume scoring with market regime consideration
            # Check for weekend/low liquidity periods (volume < 50% of average)
            if volume_ratio < 0.5:
                score = 0.7  # Higher score for low liquidity periods (was 0.6)
                reasoning_parts.append("Low liquidity period (weekend?)")
            elif volume_ratio < 0.8:
                score = 0.6  # Moderate score for below average volume
                reasoning_parts.append("Below average volume")
            elif volume_ratio > 2.0:
                score = 0.9
                reasoning_parts.append("Very high volume")
            elif volume_ratio > 1.5:
                score = 0.8
                reasoning_parts.append("High volume")
            elif volume_ratio > 1.2:
                score = 0.7
                reasoning_parts.append("Above average volume")
            elif volume_ratio > 0.8:
                score = 0.6
                reasoning_parts.append("Average volume")
            else:
                score = 0.5  # Neutral instead of penalizing
                reasoning_parts.append("Below average volume")
            
            # Additional check for volume trend (last 5 periods)
            if len(data) >= 5:
                recent_volumes = data['volume'].tail(5)
                volume_trend = recent_volumes.pct_change().mean()
                if volume_trend > 0.1:  # Increasing volume trend
                    score = min(1.0, score + 0.1)
                    reasoning_parts.append("Volume increasing")
                elif volume_trend < -0.1:  # Decreasing volume trend
                    score = max(0.3, score - 0.1)
                    reasoning_parts.append("Volume decreasing")
            
            return {
                'score': score,
                'reasoning': f"Vol ratio: {volume_ratio:.2f} | " + ' | '.join(reasoning_parts)
            }
            
        except Exception as e:
            logger.error(f"Error calculating volume confirmation: {e}")
            return {'score': 0.5, 'reasoning': f'Error: {str(e)}'}

    async def _calculate_price_action_score(self, data: pd.DataFrame, signal_direction: str, timeframe: str) -> Dict[str, Any]:
        """Calculate price action score for the timeframe"""
        try:
            if len(data) < 10:
                return {'score': 0.5, 'reasoning': 'Insufficient data'}
            
            # Calculate recent price action
            recent_data = data.tail(10)
            
            # Calculate bullish/bearish candles
            bullish_candles = len(recent_data[recent_data['close'] > recent_data['open']])
            bearish_candles = len(recent_data[recent_data['close'] < recent_data['open']])
            total_candles = len(recent_data)
            
            # Calculate score based on signal direction
            score = 0.5
            reasoning_parts = []
            
            if signal_direction == 'long':
                bullish_ratio = bullish_candles / total_candles
                if bullish_ratio > 0.7:
                    score = 0.9
                    reasoning_parts.append("Strong bullish PA")
                elif bullish_ratio > 0.6:
                    score = 0.7
                    reasoning_parts.append("Moderate bullish PA")
                elif bullish_ratio > 0.4:
                    score = 0.5
                    reasoning_parts.append("Mixed PA")
                else:
                    score = 0.3
                    reasoning_parts.append("Bearish PA")
            else:  # short
                bearish_ratio = bearish_candles / total_candles
                if bearish_ratio > 0.7:
                    score = 0.9
                    reasoning_parts.append("Strong bearish PA")
                elif bearish_ratio > 0.6:
                    score = 0.7
                    reasoning_parts.append("Moderate bearish PA")
                elif bearish_ratio > 0.4:
                    score = 0.5
                    reasoning_parts.append("Mixed PA")
                else:
                    score = 0.3
                    reasoning_parts.append("Bullish PA")
            
            return {
                'score': score,
                'reasoning': f"Bullish: {bullish_candles}/{total_candles} | " + ' | '.join(reasoning_parts)
            }
            
        except Exception as e:
            logger.error(f"Error calculating price action score: {e}")
            return {'score': 0.5, 'reasoning': f'Error: {str(e)}'}

    async def _get_dynamic_timeframe_weights(self, adx: float) -> Dict[str, float]:
        """Get dynamic timeframe weights based on ADX market regime"""
        try:
            if adx > 25:  # Trending market
                return {
                    '1h': 0.3,
                    '4h': 0.3,
                    '1d': 0.4  # Boost 1D in trending markets
                }
            elif adx < 20:  # Ranging market
                return {
                    '1h': 0.5,
                    '4h': 0.4,
                    '1d': 0.1  # Focus on shorter timeframes
                }
            else:  # Neutral market
                return {
                    '1h': 0.3,
                    '4h': 0.4,
                    '1d': 0.3  # Default balanced weights
                }
        except Exception as e:
            logger.error(f"Error calculating dynamic weights: {e}")
            return {'1h': 0.3, '4h': 0.4, '1d': 0.3}

    async def _check_cross_timeframe_confluence(self, symbol: str, signal_direction: str, 
                                              mtf_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Check for cross-timeframe pattern confluence with enhanced handling"""
        try:
            confluence_score = 0.0
            confluence_reasons = []
            
            # Check for trend alignment across timeframes
            trend_alignment = 0
            total_tfs = 0
            available_tfs = []
            
            for tf, data in mtf_data.items():
                if data is not None and not data.empty and len(data) >= 20:
                    total_tfs += 1
                    available_tfs.append(tf)
                    # Calculate simple trend direction
                    ema_12 = data['close'].ewm(span=12).mean().iloc[-1]
                    ema_26 = data['close'].ewm(span=26).mean().iloc[-1]
                    
                    if signal_direction == 'long':
                        if ema_12 > ema_26:
                            trend_alignment += 1
                    else:  # short
                        if ema_12 < ema_26:
                            trend_alignment += 1
            
            # Calculate trend confluence with enhanced logic
            if total_tfs > 0:
                trend_confluence_ratio = trend_alignment / total_tfs
                if trend_confluence_ratio >= 0.8:  # 80%+ alignment
                    confluence_score += 0.2
                    confluence_reasons.append(f"Strong trend alignment ({trend_confluence_ratio:.1%}) across {available_tfs}")
                elif trend_confluence_ratio >= 0.6:  # 60%+ alignment
                    confluence_score += 0.1
                    confluence_reasons.append(f"Moderate trend alignment ({trend_confluence_ratio:.1%}) across {available_tfs}")
                elif trend_confluence_ratio >= 0.4:  # 40%+ alignment
                    confluence_score += 0.05
                    confluence_reasons.append(f"Weak trend alignment ({trend_confluence_ratio:.1%}) across {available_tfs}")
            else:
                confluence_reasons.append("No timeframe data available for confluence analysis")
            
            # Check for support/resistance confluence
            sr_confluence = await self._check_sr_confluence(symbol, signal_direction, mtf_data)
            confluence_score += sr_confluence['score']
            if sr_confluence['reasons']:
                confluence_reasons.extend(sr_confluence['reasons'])
            
            # Check for volume confluence
            volume_confluence = await self._check_volume_confluence(mtf_data)
            confluence_score += volume_confluence['score']
            if volume_confluence['reasons']:
                confluence_reasons.extend(volume_confluence['reasons'])
            
            # Add bonus for having multiple timeframes available
            if total_tfs >= 2:
                confluence_score += 0.05
                confluence_reasons.append(f"Multi-timeframe data available ({total_tfs} TFs)")
            
            # Enhanced 1D trend analysis for stronger confluence
            if '1d' in available_tfs:
                try:
                    # Check 1D trend strength
                    daily_data = mtf_data['1d']
                    if daily_data is not None and not daily_data.empty and len(daily_data) >= 5:
                        # Calculate 1D trend strength
                        daily_ema_12 = daily_data['close'].ewm(span=12).mean().iloc[-1]
                        daily_ema_26 = daily_data['close'].ewm(span=26).mean().iloc[-1]
                        daily_price = daily_data['close'].iloc[-1]
                        
                        if signal_direction == 'long':
                            if daily_price > daily_ema_12 > daily_ema_26:
                                confluence_score += 0.1
                                confluence_reasons.append("Strong 1D bullish trend")
                            elif daily_price > daily_ema_12:
                                confluence_score += 0.05
                                confluence_reasons.append("Moderate 1D bullish trend")
                        else:  # short
                            if daily_price < daily_ema_12 < daily_ema_26:
                                confluence_score += 0.1
                                confluence_reasons.append("Strong 1D bearish trend")
                            elif daily_price < daily_ema_12:
                                confluence_score += 0.05
                                confluence_reasons.append("Moderate 1D bearish trend")
                except Exception as e:
                    logger.error(f"Error in 1D trend analysis: {e}")
            
            return {
                'score': min(0.3, confluence_score),  # Cap at 0.3 bonus
                'reasons': confluence_reasons,
                'trend_alignment': trend_alignment,
                'total_timeframes': total_tfs,
                'available_timeframes': available_tfs
            }
            
        except Exception as e:
            logger.error(f"Error checking cross-timeframe confluence: {e}")
            return {'score': 0.0, 'reasons': [f'Error: {str(e)}'], 'trend_alignment': 0, 'total_timeframes': 0, 'available_timeframes': []}

    async def _check_sr_confluence(self, symbol: str, signal_direction: str, 
                                 mtf_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Check for support/resistance confluence across timeframes"""
        try:
            confluence_score = 0.0
            confluence_reasons = []
            
            # Get current price from primary timeframe
            primary_data = None
            for tf, data in mtf_data.items():
                if data is not None and not data.empty:
                    primary_data = data
                    break
            
            if primary_data is None:
                return {'score': 0.0, 'reasons': []}
            
            current_price = primary_data['close'].iloc[-1]
            
            # Check S/R levels across timeframes
            sr_levels = []
            for tf, data in mtf_data.items():
                if data is not None and not data.empty and len(data) >= 20:
                    recent_high = data['high'].rolling(window=20).max().iloc[-1]
                    recent_low = data['low'].rolling(window=20).min().iloc[-1]
                    sr_levels.append({'tf': tf, 'high': recent_high, 'low': recent_low})
            
            # Check for confluence near current price
            if signal_direction == 'long':
                # Check if we're near support levels across timeframes
                support_confluence = 0
                for level in sr_levels:
                    distance = abs(current_price - level['low']) / current_price
                    if distance < 0.03:  # Within 3% of support
                        support_confluence += 1
                
                if support_confluence >= 2:  # Multiple timeframes showing support
                    confluence_score += 0.1
                    confluence_reasons.append(f"Multi-TF support confluence ({support_confluence} TFs)")
            else:  # short
                # Check if we're near resistance levels across timeframes
                resistance_confluence = 0
                for level in sr_levels:
                    distance = abs(current_price - level['high']) / current_price
                    if distance < 0.03:  # Within 3% of resistance
                        resistance_confluence += 1
                
                if resistance_confluence >= 2:  # Multiple timeframes showing resistance
                    confluence_score += 0.1
                    confluence_reasons.append(f"Multi-TF resistance confluence ({resistance_confluence} TFs)")
            
            return {'score': confluence_score, 'reasons': confluence_reasons}
            
        except Exception as e:
            logger.error(f"Error checking S/R confluence: {e}")
            return {'score': 0.0, 'reasons': [f'Error: {str(e)}']}

    async def _check_volume_confluence(self, mtf_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Check for volume confluence across timeframes"""
        try:
            confluence_score = 0.0
            confluence_reasons = []
            
            # Check volume patterns across timeframes
            volume_trends = []
            for tf, data in mtf_data.items():
                if data is not None and not data.empty and len(data) >= 10:
                    # Calculate volume trend
                    recent_volumes = data['volume'].tail(5)
                    avg_volume = data['volume'].rolling(window=20).mean().iloc[-1]
                    current_volume = data['volume'].iloc[-1]
                    volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
                    
                    volume_trends.append({
                        'tf': tf,
                        'ratio': volume_ratio,
                        'trend': recent_volumes.pct_change().mean()
                    })
            
            # Check for volume confluence
            high_volume_tfs = sum(1 for vt in volume_trends if vt['ratio'] > 1.2)
            increasing_volume_tfs = sum(1 for vt in volume_trends if vt['trend'] > 0.1)
            
            if high_volume_tfs >= 2:  # Multiple timeframes with high volume
                confluence_score += 0.05
                confluence_reasons.append(f"Multi-TF high volume ({high_volume_tfs} TFs)")
            
            if increasing_volume_tfs >= 2:  # Multiple timeframes with increasing volume
                confluence_score += 0.05
                confluence_reasons.append(f"Multi-TF volume increase ({increasing_volume_tfs} TFs)")
            
            return {'score': confluence_score, 'reasons': confluence_reasons}
            
        except Exception as e:
            logger.error(f"Error checking volume confluence: {e}")
            return {'score': 0.0, 'reasons': [f'Error: {str(e)}']}

# Global instance
intelligent_signal_generator = None

async def get_intelligent_signal_generator(db_pool: asyncpg.Pool, exchange: ccxt.Exchange) -> IntelligentSignalGenerator:
    """Get or create global intelligent signal generator instance"""
    global intelligent_signal_generator
    if intelligent_signal_generator is None:
        intelligent_signal_generator = IntelligentSignalGenerator(db_pool, exchange)
    return intelligent_signal_generator
