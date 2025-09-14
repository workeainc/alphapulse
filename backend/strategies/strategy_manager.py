"""
Enhanced Strategy Manager for AlphaPlus
Manages all trading strategies and signal generation
"""

import asyncio
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum
import uuid

from .pattern_detector import CandlestickPatternDetector
from .volume_enhanced_pattern_detector import VolumeEnhancedPatternDetector
from .ml_pattern_detector import MLPatternDetector

logger = logging.getLogger(__name__)

class StrategyType(Enum):
    """Strategy types"""
    PATTERN_BASED = "pattern_based"
    ML_BASED = "ml_based"
    VOLUME_BASED = "volume_based"
    HYBRID = "hybrid"

class SignalStrength(Enum):
    """Signal strength levels"""
    WEAK = "weak"
    MEDIUM = "medium"
    STRONG = "strong"
    VERY_STRONG = "very_strong"

@dataclass
class TradingSignal:
    """Trading signal structure"""
    id: str
    symbol: str
    side: str  # 'buy' or 'sell'
    strategy: str
    confidence: float
    strength: SignalStrength
    timestamp: datetime
    price: float
    quantity: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    risk_amount: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class StrategyConfig:
    """Strategy configuration"""
    name: str
    type: StrategyType
    enabled: bool = True
    min_confidence: float = 0.7
    max_risk_per_trade: float = 0.02
    symbols: List[str] = None
    timeframes: List[str] = None
    parameters: Dict[str, Any] = None

class StrategyManager:
    """Enhanced strategy manager"""
    
    def __init__(self):
        self.logger = logger
        
        # Strategy instances
        self.pattern_detector = CandlestickPatternDetector()
        self.volume_detector = VolumeEnhancedPatternDetector()
        self.ml_detector = MLPatternDetector()
        
        # Strategy configurations
        self.strategies: Dict[str, StrategyConfig] = {}
        self.active_signals: Dict[str, TradingSignal] = {}
        self.signal_history: List[TradingSignal] = []
        
        # Monitoring state
        self.monitoring_active = False
        self.monitoring_task = None
        
        # Performance tracking
        self.signals_generated = 0
        self.signals_executed = 0
        self.signals_expired = 0
        
        # Initialize default strategies
        self._initialize_default_strategies()
    
    def _initialize_default_strategies(self):
        """Initialize default trading strategies"""
        default_strategies = [
            StrategyConfig(
                name="candlestick_patterns",
                type=StrategyType.PATTERN_BASED,
                min_confidence=0.75,
                symbols=["BTCUSDT", "ETHUSDT", "BNBUSDT"],
                timeframes=["1h", "4h", "1d"],
                parameters={"min_pattern_strength": 0.8}
            ),
            StrategyConfig(
                name="volume_analysis",
                type=StrategyType.VOLUME_BASED,
                min_confidence=0.7,
                symbols=["BTCUSDT", "ETHUSDT"],
                timeframes=["1h", "4h"],
                parameters={"volume_threshold": 1.5}
            ),
            StrategyConfig(
                name="ml_patterns",
                type=StrategyType.ML_BASED,
                min_confidence=0.8,
                symbols=["BTCUSDT", "ETHUSDT"],
                timeframes=["1h", "4h"],
                parameters={"model_version": "latest"}
            )
        ]
        
        for strategy in default_strategies:
            self.add_strategy(strategy)
    
    def add_strategy(self, config: StrategyConfig):
        """Add a new strategy"""
        try:
            self.strategies[config.name] = config
            self.logger.info(f"Added strategy: {config.name}")
            
        except Exception as e:
            self.logger.error(f"Error adding strategy {config.name}: {e}")
    
    def remove_strategy(self, strategy_name: str):
        """Remove a strategy"""
        try:
            if strategy_name in self.strategies:
                del self.strategies[strategy_name]
                self.logger.info(f"Removed strategy: {strategy_name}")
            else:
                self.logger.warning(f"Strategy not found: {strategy_name}")
                
            except Exception as e:
            self.logger.error(f"Error removing strategy {strategy_name}: {e}")
    
    async def initialize(self):
        """Initialize the strategy manager"""
        try:
            self.logger.info("Initializing Strategy Manager...")
            
            # Initialize strategy components
            await self.pattern_detector.initialize()
            await self.volume_detector.initialize()
            await self.ml_detector.initialize()
            
            self.logger.info("Strategy Manager initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Strategy Manager: {e}")
            raise
    
    async def start_monitoring(self):
        """Start strategy monitoring"""
        if self.monitoring_active:
            self.logger.warning("Monitoring already active")
            return
        
        try:
            self.logger.info("Starting strategy monitoring...")
            self.monitoring_active = True
            
            # Start monitoring task
            self.monitoring_task = asyncio.create_task(self._monitoring_loop())
            
            self.logger.info("Strategy monitoring started")
                
            except Exception as e:
            self.logger.error(f"Failed to start monitoring: {e}")
            self.monitoring_active = False
            raise
    
    async def stop_monitoring(self):
        """Stop strategy monitoring"""
        try:
            self.logger.info("Stopping strategy monitoring...")
            
            self.monitoring_active = False
            
            if self.monitoring_task:
                self.monitoring_task.cancel()
                try:
                    await self.monitoring_task
                except asyncio.CancelledError:
                    pass
                self.monitoring_task = None
            
            self.logger.info("Strategy monitoring stopped")
                
            except Exception as e:
            self.logger.error(f"Error stopping monitoring: {e}")
    
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        try:
            while self.monitoring_active:
                # Generate signals from all active strategies
                await self._generate_signals()
                
                # Clean up expired signals
                await self._cleanup_expired_signals()
                
                # Wait before next iteration
                await asyncio.sleep(60)  # Check every minute
                
        except asyncio.CancelledError:
            self.logger.info("Monitoring loop cancelled")
        except Exception as e:
            self.logger.error(f"Error in monitoring loop: {e}")
            self.monitoring_active = False
    
    async def _generate_signals(self):
        """Generate signals from all active strategies"""
        try:
            for strategy_name, config in self.strategies.items():
                if not config.enabled:
                continue
            
                signals = await self._generate_strategy_signals(config)
                
                for signal in signals:
                    if signal.confidence >= config.min_confidence:
                        await self._add_signal(signal)
                        
        except Exception as e:
            self.logger.error(f"Error generating signals: {e}")
    
    async def _generate_strategy_signals(self, config: StrategyConfig) -> List[TradingSignal]:
        """Generate signals for a specific strategy"""
        try:
            signals = []
            
            if config.type == StrategyType.PATTERN_BASED:
                signals = await self._generate_pattern_signals(config)
            elif config.type == StrategyType.VOLUME_BASED:
                signals = await self._generate_volume_signals(config)
            elif config.type == StrategyType.ML_BASED:
                signals = await self._generate_ml_signals(config)
            elif config.type == StrategyType.HYBRID:
                signals = await self._generate_hybrid_signals(config)
            
            return signals
                
            except Exception as e:
            self.logger.error(f"Error generating signals for {config.name}: {e}")
            return []
    
    async def _generate_pattern_signals(self, config: StrategyConfig) -> List[TradingSignal]:
        """Generate pattern-based signals"""
        try:
            signals = []
            
            for symbol in config.symbols or []:
                for timeframe in config.timeframes or []:
                    # Get pattern analysis
                    patterns = await self.pattern_detector.analyze_symbol(
                        symbol, timeframe, config.parameters
                    )
                    
                    for pattern in patterns:
                        if pattern.get('confidence', 0) >= config.min_confidence:
                            signal = TradingSignal(
                                id=str(uuid.uuid4()),
                                symbol=symbol,
                                side=pattern.get('side', 'buy'),
                                strategy=config.name,
                                confidence=pattern.get('confidence', 0),
                                strength=self._get_signal_strength(pattern.get('confidence', 0)),
                                timestamp=datetime.now(timezone.utc),
                                price=pattern.get('price', 0),
                                stop_loss=pattern.get('stop_loss'),
                                take_profit=pattern.get('take_profit'),
                                metadata=pattern
                            )
                            signals.append(signal)
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Error generating pattern signals: {e}")
            return []
    
    async def _generate_volume_signals(self, config: StrategyConfig) -> List[TradingSignal]:
        """Generate volume-based signals"""
        try:
            signals = []
            
            for symbol in config.symbols or []:
                for timeframe in config.timeframes or []:
                    # Get volume analysis
                    volume_analysis = await self.volume_detector.analyze_symbol(
                        symbol, timeframe, config.parameters
                    )
                    
                    for analysis in volume_analysis:
                        if analysis.get('confidence', 0) >= config.min_confidence:
                            signal = TradingSignal(
                                id=str(uuid.uuid4()),
                                symbol=symbol,
                                side=analysis.get('side', 'buy'),
                                strategy=config.name,
                                confidence=analysis.get('confidence', 0),
                                strength=self._get_signal_strength(analysis.get('confidence', 0)),
                                timestamp=datetime.now(timezone.utc),
                                price=analysis.get('price', 0),
                                stop_loss=analysis.get('stop_loss'),
                                take_profit=analysis.get('take_profit'),
                                metadata=analysis
                            )
                            signals.append(signal)
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Error generating volume signals: {e}")
            return []
    
    async def _generate_ml_signals(self, config: StrategyConfig) -> List[TradingSignal]:
        """Generate ML-based signals"""
        try:
            signals = []
            
            for symbol in config.symbols or []:
                for timeframe in config.timeframes or []:
                    # Get ML predictions
                    predictions = await self.ml_detector.predict_symbol(
                        symbol, timeframe, config.parameters
                    )
                    
                    for prediction in predictions:
                        if prediction.get('confidence', 0) >= config.min_confidence:
                            signal = TradingSignal(
                                id=str(uuid.uuid4()),
                                symbol=symbol,
                                side=prediction.get('side', 'buy'),
                                strategy=config.name,
                                confidence=prediction.get('confidence', 0),
                                strength=self._get_signal_strength(prediction.get('confidence', 0)),
                                timestamp=datetime.now(timezone.utc),
                                price=prediction.get('price', 0),
                                stop_loss=prediction.get('stop_loss'),
                                take_profit=prediction.get('take_profit'),
                                metadata=prediction
                            )
                            signals.append(signal)
            
            return signals
                
            except Exception as e:
            self.logger.error(f"Error generating ML signals: {e}")
            return []
    
    async def _generate_hybrid_signals(self, config: StrategyConfig) -> List[TradingSignal]:
        """Generate hybrid signals combining multiple approaches"""
        try:
            # Get signals from all strategy types
            pattern_signals = await self._generate_pattern_signals(config)
            volume_signals = await self._generate_volume_signals(config)
            ml_signals = await self._generate_ml_signals(config)
            
            # Combine and rank signals
            all_signals = pattern_signals + volume_signals + ml_signals
            
            # Group by symbol and timeframe
            signal_groups = {}
            for signal in all_signals:
                key = (signal.symbol, signal.side)
                if key not in signal_groups:
                    signal_groups[key] = []
                signal_groups[key].append(signal)
            
            # Generate hybrid signals with enhanced confidence
            hybrid_signals = []
            for (symbol, side), signals in signal_groups.items():
                if len(signals) >= 2:  # At least 2 strategies agree
                    # Calculate combined confidence
                    total_confidence = sum(s.confidence for s in signals)
                    avg_confidence = total_confidence / len(signals)
                    
                    # Boost confidence for agreement
                    boosted_confidence = min(0.95, avg_confidence * 1.1)
                    
                    if boosted_confidence >= config.min_confidence:
                        # Use the signal with highest confidence as base
                        base_signal = max(signals, key=lambda x: x.confidence)
                        
                        hybrid_signal = TradingSignal(
                            id=str(uuid.uuid4()),
                            symbol=symbol,
                            side=side,
                            strategy=f"hybrid_{config.name}",
                            confidence=boosted_confidence,
                            strength=self._get_signal_strength(boosted_confidence),
                            timestamp=datetime.now(timezone.utc),
                            price=base_signal.price,
                            stop_loss=base_signal.stop_loss,
                            take_profit=base_signal.take_profit,
                            metadata={
                                'base_strategies': [s.strategy for s in signals],
                                'individual_confidences': [s.confidence for s in signals],
                                'agreement_count': len(signals)
                            }
                        )
                        hybrid_signals.append(hybrid_signal)
            
            return hybrid_signals
            
        except Exception as e:
            self.logger.error(f"Error generating hybrid signals: {e}")
            return []
    
    def _get_signal_strength(self, confidence: float) -> SignalStrength:
        """Determine signal strength based on confidence"""
        if confidence >= 0.9:
            return SignalStrength.VERY_STRONG
        elif confidence >= 0.8:
            return SignalStrength.STRONG
        elif confidence >= 0.7:
            return SignalStrength.MEDIUM
        else:
            return SignalStrength.WEAK
    
    async def _add_signal(self, signal: TradingSignal):
        """Add a new signal"""
        try:
            # Check if we already have a signal for this symbol/side
            key = (signal.symbol, signal.side)
            
            # Remove existing signal if new one has higher confidence
            existing_signals = [s for s in self.active_signals.values() if (s.symbol, s.side) == key]
            
            for existing in existing_signals:
                if signal.confidence > existing.confidence:
                    del self.active_signals[existing.id]
                    self.signals_expired += 1
            
            # Add new signal
            self.active_signals[signal.id] = signal
            self.signal_history.append(signal)
            self.signals_generated += 1
            
            self.logger.info(f"Added signal: {signal.symbol} {signal.side} "
                           f"({signal.strategy}) - Confidence: {signal.confidence:.2f}")
            
            except Exception as e:
            self.logger.error(f"Error adding signal: {e}")
    
    async def _cleanup_expired_signals(self):
        """Clean up expired signals"""
        try:
            current_time = datetime.now(timezone.utc)
            expired_signals = []
            
            for signal_id, signal in self.active_signals.items():
                # Signal expires after 5 minutes
                if (current_time - signal.timestamp).total_seconds() > 300:
                    expired_signals.append(signal_id)
            
            for signal_id in expired_signals:
                del self.active_signals[signal_id]
                self.signals_expired += 1
            
            if expired_signals:
                self.logger.info(f"Cleaned up {len(expired_signals)} expired signals")
            
        except Exception as e:
            self.logger.error(f"Error cleaning up expired signals: {e}")
    
    async def get_active_signals(self) -> List[TradingSignal]:
        """Get all active trading signals"""
        return list(self.active_signals.values())
    
    async def get_signal_by_id(self, signal_id: str) -> Optional[TradingSignal]:
        """Get a specific signal by ID"""
        return self.active_signals.get(signal_id)
    
    async def get_signals_by_symbol(self, symbol: str) -> List[TradingSignal]:
        """Get all signals for a specific symbol"""
        return [s for s in self.active_signals.values() if s.symbol == symbol]
    
    async def get_signals_by_strategy(self, strategy: str) -> List[TradingSignal]:
        """Get all signals from a specific strategy"""
        return [s for s in self.active_signals.values() if s.strategy == strategy]
    
    async def mark_signal_executed(self, signal_id: str):
        """Mark a signal as executed"""
        try:
            if signal_id in self.active_signals:
                signal = self.active_signals[signal_id]
                del self.active_signals[signal_id]
                self.signals_executed += 1
                
                self.logger.info(f"Signal executed: {signal.symbol} {signal.side}")
            
        except Exception as e:
            self.logger.error(f"Error marking signal executed: {e}")
    
    async def get_performance_summary(self) -> Dict[str, Any]:
        """Get strategy performance summary"""
        try:
            return {
                'total_signals_generated': self.signals_generated,
                'signals_executed': self.signals_executed,
                'signals_expired': self.signals_expired,
                'active_signals': len(self.active_signals),
                'strategies': {
                    name: {
                        'enabled': config.enabled,
                        'type': config.type.value,
                        'min_confidence': config.min_confidence
                    }
                    for name, config in self.strategies.items()
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error getting performance summary: {e}")
            return {}
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check for strategy manager"""
        try:
        return {
                'status': 'healthy' if self.monitoring_active else 'inactive',
                'monitoring_active': self.monitoring_active,
                'active_signals': len(self.active_signals),
                'strategies_count': len(self.strategies),
                'enabled_strategies': len([s for s in self.strategies.values() if s.enabled])
            }
            
        except Exception as e:
            self.logger.error(f"Error in health check: {e}")
            return {'status': 'error', 'error': str(e)}
