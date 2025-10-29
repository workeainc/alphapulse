"""
Funding Rate Strategy Service for AlphaPulse
Week 7.3 Phase 3: Funding Rate Strategy Integration

Features:
- Strategy generation based on funding rate analytics
- Risk-adjusted position sizing
- Multi-timeframe signal aggregation
- Performance tracking and optimization

Author: AlphaPulse Team
Date: 2025
"""

import asyncio
import logging
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass
from collections import defaultdict, deque
from enum import Enum

logger = logging.getLogger(__name__)

class SignalStrength(Enum):
    """Signal strength levels"""
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    EXTREME = "extreme"

class SignalDirection(Enum):
    """Signal direction"""
    LONG = "long"
    SHORT = "short"
    NEUTRAL = "neutral"

@dataclass
class FundingRateSignal:
    """Funding rate trading signal"""
    symbol: str
    direction: SignalDirection
    strength: SignalStrength
    confidence: float
    entry_price: float
    target_price: float
    stop_loss: float
    position_size: float
    risk_score: float
    signal_type: str
    timestamp: datetime
    metadata: Dict[str, Any]

@dataclass
class StrategyPerformance:
    """Strategy performance metrics"""
    symbol: str
    total_signals: int
    successful_signals: int
    failed_signals: int
    win_rate: float
    average_profit: float
    average_loss: float
    profit_factor: float
    max_drawdown: float
    sharpe_ratio: float
    last_update: datetime
    metadata: Dict[str, Any]

class FundingRateStrategy:
    """Funding rate strategy service"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logger
        
        # Strategy configuration
        self.min_confidence = self.config.get('min_confidence', 0.7)
        self.max_risk_score = self.config.get('max_risk_score', 0.8)
        self.position_sizing_factor = self.config.get('position_sizing_factor', 0.1)
        self.risk_reward_ratio = self.config.get('risk_reward_ratio', 2.0)
        self.max_position_size = self.config.get('max_position_size', 0.2)
        
        # Signal thresholds
        self.correlation_threshold = self.config.get('correlation_threshold', 0.7)
        self.volatility_threshold = self.config.get('volatility_threshold', 0.001)
        self.pattern_confidence_threshold = self.config.get('pattern_confidence_threshold', 0.8)
        self.arbitrage_min_difference = self.config.get('arbitrage_min_difference', 0.0005)
        
        # Data storage
        self.signals = defaultdict(list)  # symbol -> signals
        self.performance = defaultdict(dict)  # symbol -> performance
        self.active_positions = defaultdict(dict)  # symbol -> position
        
        # Performance tracking
        self.stats = {
            'signals_generated': 0,
            'signals_executed': 0,
            'successful_trades': 0,
            'failed_trades': 0,
            'total_pnl': 0.0,
            'last_update': None
        }
        
        # Callbacks
        self.strategy_callbacks = defaultdict(list)  # event_type -> [callback]
        
        # Initialize performance tracking
        self._initialize_performance_tracking()
    
    def _initialize_performance_tracking(self):
        """Initialize performance tracking for symbols"""
        try:
            # Initialize performance for common symbols
            common_symbols = ['BTC/USDT', 'ETH/USDT', 'ADA/USDT', 'SOL/USDT']
            
            for symbol in common_symbols:
                self.performance[symbol] = StrategyPerformance(
                    symbol=symbol,
                    total_signals=0,
                    successful_signals=0,
                    failed_signals=0,
                    win_rate=0.0,
                    average_profit=0.0,
                    average_loss=0.0,
                    profit_factor=0.0,
                    max_drawdown=0.0,
                    sharpe_ratio=0.0,
                    last_update=datetime.now(timezone.utc),
                    metadata={}
                )
            
            self.logger.info("Performance tracking initialized for funding rate strategies")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize performance tracking: {e}")
    
    async def generate_signals(self, symbol: str, analytics_data: Dict[str, Any]) -> List[FundingRateSignal]:
        """Generate trading signals based on funding rate analytics"""
        try:
            signals = []
            
            # Extract analytics components
            correlations = analytics_data.get('correlations', {})
            patterns = analytics_data.get('patterns', [])
            volatility = analytics_data.get('volatility', {})
            arbitrage = analytics_data.get('arbitrage', [])
            
            # Generate correlation-based signals
            correlation_signals = await self._generate_correlation_signals(symbol, correlations)
            signals.extend(correlation_signals)
            
            # Generate pattern-based signals
            pattern_signals = await self._generate_pattern_signals(symbol, patterns)
            signals.extend(pattern_signals)
            
            # Generate volatility-based signals
            volatility_signals = await self._generate_volatility_signals(symbol, volatility)
            signals.extend(volatility_signals)
            
            # Generate arbitrage signals
            arbitrage_signals = await self._generate_arbitrage_signals(symbol, arbitrage)
            signals.extend(arbitrage_signals)
            
            # Filter and rank signals
            filtered_signals = self._filter_signals(signals)
            ranked_signals = self._rank_signals(filtered_signals)
            
            # Store signals
            if ranked_signals:
                self.signals[symbol].extend(ranked_signals)
                
                # Keep only recent signals
                cutoff_time = datetime.now(timezone.utc) - timedelta(hours=24)
                self.signals[symbol] = [
                    s for s in self.signals[symbol] 
                    if s.timestamp >= cutoff_time
                ]
                
                # Update statistics
                self.stats['signals_generated'] += len(ranked_signals)
                self.stats['last_update'] = datetime.now(timezone.utc)
                
                # Trigger callbacks for high-confidence signals
                for signal in ranked_signals:
                    if signal.confidence >= self.min_confidence:
                        await self._trigger_callbacks('signal_generated', signal)
            
            return ranked_signals
            
        except Exception as e:
            self.logger.error(f"Error generating signals for {symbol}: {e}")
            return []
    
    async def _generate_correlation_signals(self, symbol: str, correlations: Dict[str, Any]) -> List[FundingRateSignal]:
        """Generate signals based on correlation analysis"""
        try:
            signals = []
            
            if not correlations:
                return signals
            
            # Check for high correlation breakdowns (potential divergence opportunities)
            for exchange1, corr_data in correlations.items():
                for exchange2, correlation in corr_data.items():
                    if exchange1 != exchange2 and abs(correlation) >= self.correlation_threshold:
                        # High correlation detected - look for breakdowns
                        if correlation < 0.5:  # Correlation breakdown
                            # Generate mean reversion signal
                            signal = FundingRateSignal(
                                symbol=symbol,
                                direction=SignalDirection.NEUTRAL,
                                strength=SignalStrength.MODERATE,
                                confidence=min(abs(correlation - 0.5) * 2, 0.9),
                                entry_price=0.0,  # Will be set by execution
                                target_price=0.0,
                                stop_loss=0.0,
                                position_size=0.0,
                                risk_score=0.6,
                                signal_type='correlation_breakdown',
                                timestamp=datetime.now(timezone.utc),
                                metadata={
                                    'exchange1': exchange1,
                                    'exchange2': exchange2,
                                    'correlation': correlation,
                                    'threshold': self.correlation_threshold
                                }
                            )
                            signals.append(signal)
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Error generating correlation signals: {e}")
            return []
    
    async def _generate_pattern_signals(self, symbol: str, patterns: List[Dict[str, Any]]) -> List[FundingRateSignal]:
        """Generate signals based on pattern recognition"""
        try:
            signals = []
            
            for pattern in patterns:
                if pattern['confidence'] >= self.pattern_confidence_threshold:
                    # Determine signal direction based on pattern type
                    if pattern['pattern_type'] == 'trending_up':
                        direction = SignalDirection.LONG
                        strength = self._calculate_pattern_strength(pattern['confidence'])
                    elif pattern['pattern_type'] == 'trending_down':
                        direction = SignalDirection.SHORT
                        strength = self._calculate_pattern_strength(pattern['confidence'])
                    elif pattern['pattern_type'] == 'mean_reverting':
                        # Mean reversion - opposite direction of current trend
                        direction = SignalDirection.NEUTRAL  # Will be determined by current rate
                        strength = self._calculate_pattern_strength(pattern['confidence'])
                    else:
                        continue
                    
                    signal = FundingRateSignal(
                        symbol=symbol,
                        direction=direction,
                        strength=strength,
                        confidence=pattern['confidence'],
                        entry_price=0.0,
                        target_price=0.0,
                        stop_loss=0.0,
                        position_size=0.0,
                        risk_score=self._calculate_pattern_risk(pattern),
                        signal_type=f"pattern_{pattern['pattern_type']}",
                        timestamp=datetime.now(timezone.utc),
                        metadata=pattern
                    )
                    signals.append(signal)
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Error generating pattern signals: {e}")
            return []
    
    async def _generate_volatility_signals(self, symbol: str, volatility: Dict[str, Any]) -> List[FundingRateSignal]:
        """Generate signals based on volatility analysis"""
        try:
            signals = []
            
            for exchange, vol_data in volatility.items():
                if vol_data['volatility_type'] == 'high_volatility':
                    # High volatility - potential breakout opportunity
                    signal = FundingRateSignal(
                        symbol=symbol,
                        direction=SignalDirection.NEUTRAL,  # Direction will be determined by price action
                        strength=SignalStrength.STRONG,
                        confidence=min(vol_data['volatility'] * 1000, 0.9),
                        entry_price=0.0,
                        target_price=0.0,
                        stop_loss=0.0,
                        position_size=0.0,
                        risk_score=0.7,  # Higher risk due to volatility
                        signal_type='volatility_breakout',
                        timestamp=datetime.now(timezone.utc),
                        metadata={
                            'exchange': exchange,
                            'volatility': vol_data['volatility'],
                            'volatility_type': vol_data['volatility_type']
                        }
                    )
                    signals.append(signal)
                
                elif vol_data['volatility_type'] == 'low_volatility':
                    # Low volatility - potential range trading opportunity
                    signal = FundingRateSignal(
                        symbol=symbol,
                        direction=SignalDirection.NEUTRAL,
                        strength=SignalStrength.WEAK,
                        confidence=0.6,
                        entry_price=0.0,
                        target_price=0.0,
                        stop_loss=0.0,
                        position_size=0.0,
                        risk_score=0.4,  # Lower risk due to low volatility
                        signal_type='range_trading',
                        timestamp=datetime.now(timezone.utc),
                        metadata={
                            'exchange': exchange,
                            'volatility': vol_data['volatility'],
                            'volatility_type': vol_data['volatility_type']
                        }
                    )
                    signals.append(signal)
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Error generating volatility signals: {e}")
            return []
    
    async def _generate_arbitrage_signals(self, symbol: str, arbitrage: List[Dict[str, Any]]) -> List[FundingRateSignal]:
        """Generate signals based on arbitrage opportunities"""
        try:
            signals = []
            
            for opportunity in arbitrage:
                if opportunity['confidence'] >= self.min_confidence:
                    # Arbitrage opportunity - long one exchange, short another
                    signal = FundingRateSignal(
                        symbol=symbol,
                        direction=SignalDirection.NEUTRAL,  # Will be both long and short
                        strength=SignalStrength.STRONG,
                        confidence=opportunity['confidence'],
                        entry_price=0.0,
                        target_price=0.0,
                        stop_loss=0.0,
                        position_size=0.0,
                        risk_score=opportunity['risk_score'],
                        signal_type='arbitrage',
                        timestamp=datetime.now(timezone.utc),
                        metadata={
                            'long_exchange': opportunity['long_exchange'],
                            'short_exchange': opportunity['short_exchange'],
                            'rate_difference': opportunity['rate_difference'],
                            'potential_profit': opportunity['potential_profit']
                        }
                    )
                    signals.append(signal)
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Error generating arbitrage signals: {e}")
            return []
    
    def _calculate_pattern_strength(self, confidence: float) -> SignalStrength:
        """Calculate signal strength based on pattern confidence"""
        if confidence >= 0.9:
            return SignalStrength.EXTREME
        elif confidence >= 0.8:
            return SignalStrength.STRONG
        elif confidence >= 0.7:
            return SignalStrength.MODERATE
        else:
            return SignalStrength.WEAK
    
    def _calculate_pattern_risk(self, pattern: Dict[str, Any]) -> float:
        """Calculate risk score for pattern-based signals"""
        try:
            base_risk = 0.5
            
            if pattern['pattern_type'] == 'trending_up' or pattern['pattern_type'] == 'trending_down':
                # Trending patterns have lower risk
                base_risk -= 0.2
            elif pattern['pattern_type'] == 'mean_reverting':
                # Mean reversion patterns have higher risk
                base_risk += 0.2
            
            # Adjust based on confidence
            if pattern['confidence'] >= 0.9:
                base_risk -= 0.1
            elif pattern['confidence'] < 0.7:
                base_risk += 0.1
            
            return max(0.1, min(1.0, base_risk))
            
        except Exception as e:
            self.logger.error(f"Error calculating pattern risk: {e}")
            return 0.5
    
    def _filter_signals(self, signals: List[FundingRateSignal]) -> List[FundingRateSignal]:
        """Filter signals based on criteria"""
        try:
            filtered = []
            
            for signal in signals:
                # Check confidence threshold
                if signal.confidence < self.min_confidence:
                    continue
                
                # Check risk threshold
                if signal.risk_score > self.max_risk_score:
                    continue
                
                # Check if signal is too recent (avoid duplicates)
                if self._is_duplicate_signal(signal):
                    continue
                
                filtered.append(signal)
            
            return filtered
            
        except Exception as e:
            self.logger.error(f"Error filtering signals: {e}")
            return []
    
    def _is_duplicate_signal(self, signal: FundingRateSignal) -> bool:
        """Check if signal is duplicate of recent signal"""
        try:
            recent_signals = self.signals[signal.symbol]
            
            for recent_signal in recent_signals[-5:]:  # Check last 5 signals
                if (recent_signal.signal_type == signal.signal_type and
                    recent_signal.direction == signal.direction and
                    abs((recent_signal.timestamp - signal.timestamp).total_seconds()) < 3600):  # Within 1 hour
                    return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking duplicate signal: {e}")
            return False
    
    def _rank_signals(self, signals: List[FundingRateSignal]) -> List[FundingRateSignal]:
        """Rank signals by priority"""
        try:
            # Sort by confidence * (1 - risk_score) for optimal ranking
            ranked = sorted(signals, 
                          key=lambda x: x.confidence * (1 - x.risk_score), 
                          reverse=True)
            
            return ranked
            
        except Exception as e:
            self.logger.error(f"Error ranking signals: {e}")
            return signals
    
    async def calculate_position_sizing(self, signal: FundingRateSignal, 
                                      account_balance: float, 
                                      current_price: float) -> Dict[str, Any]:
        """Calculate position sizing for a signal"""
        try:
            # Base position size based on confidence and risk
            base_size = signal.confidence * (1 - signal.risk_score) * self.position_sizing_factor
            
            # Adjust for signal strength
            if signal.strength == SignalStrength.EXTREME:
                base_size *= 1.5
            elif signal.strength == SignalStrength.STRONG:
                base_size *= 1.2
            elif signal.strength == SignalStrength.WEAK:
                base_size *= 0.7
            
            # Calculate position value
            position_value = account_balance * base_size
            
            # Ensure position size doesn't exceed maximum
            if base_size > self.max_position_size:
                base_size = self.max_position_size
                position_value = account_balance * base_size
            
            # Calculate position size in units
            position_size = position_value / current_price
            
            # Calculate stop loss and target based on risk-reward ratio
            if signal.direction == SignalDirection.LONG:
                stop_loss_pct = 0.02  # 2% stop loss
                target_pct = stop_loss_pct * self.risk_reward_ratio
            elif signal.direction == SignalDirection.SHORT:
                stop_loss_pct = 0.02
                target_pct = stop_loss_pct * self.risk_reward_ratio
            else:
                stop_loss_pct = 0.015
                target_pct = stop_loss_pct * self.risk_reward_ratio
            
            stop_loss = current_price * (1 - stop_loss_pct) if signal.direction == SignalDirection.LONG else current_price * (1 + stop_loss_pct)
            target_price = current_price * (1 + target_pct) if signal.direction == SignalDirection.LONG else current_price * (1 - target_pct)
            
            return {
                'position_size': position_size,
                'position_value': position_value,
                'stop_loss': stop_loss,
                'target_price': target_price,
                'risk_amount': position_value * stop_loss_pct,
                'potential_profit': position_value * target_pct
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating position sizing: {e}")
            return {}
    
    async def execute_signal(self, signal: FundingRateSignal, 
                           execution_data: Dict[str, Any]) -> bool:
        """Execute a trading signal"""
        try:
            # Update signal with execution data
            signal.entry_price = execution_data.get('entry_price', 0.0)
            signal.target_price = execution_data.get('target_price', 0.0)
            signal.stop_loss = execution_data.get('stop_loss', 0.0)
            signal.position_size = execution_data.get('position_size', 0.0)
            
            # Store active position
            self.active_positions[signal.symbol] = {
                'signal': signal,
                'entry_time': datetime.now(timezone.utc),
                'entry_price': signal.entry_price,
                'position_size': signal.position_size,
                'status': 'active'
            }
            
            # Update statistics
            self.stats['signals_executed'] += 1
            self.stats['last_update'] = datetime.now(timezone.utc)
            
            # Trigger callback
            await self._trigger_callbacks('signal_executed', {
                'signal': signal,
                'execution_data': execution_data
            })
            
            self.logger.info(f"Signal executed for {signal.symbol}: {signal.signal_type}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error executing signal: {e}")
            return False
    
    async def update_position(self, symbol: str, current_price: float, 
                            pnl: float = None) -> Dict[str, Any]:
        """Update position status and performance"""
        try:
            if symbol not in self.active_positions:
                return {}
            
            position = self.active_positions[symbol]
            signal = position['signal']
            
            # Check if position should be closed
            should_close = False
            close_reason = ""
            
            if signal.direction == SignalDirection.LONG:
                if current_price <= signal.stop_loss:
                    should_close = True
                    close_reason = "stop_loss"
                elif current_price >= signal.target_price:
                    should_close = True
                    close_reason = "target_reached"
            elif signal.direction == SignalDirection.SHORT:
                if current_price >= signal.stop_loss:
                    should_close = True
                    close_reason = "stop_loss"
                elif current_price <= signal.target_price:
                    should_close = True
                    close_reason = "target_reached"
            
            if should_close:
                # Close position
                await self._close_position(symbol, current_price, close_reason, pnl)
            
            return {
                'symbol': symbol,
                'current_price': current_price,
                'entry_price': position['entry_price'],
                'pnl': pnl,
                'should_close': should_close,
                'close_reason': close_reason
            }
            
        except Exception as e:
            self.logger.error(f"Error updating position: {e}")
            return {}
    
    async def _close_position(self, symbol: str, exit_price: float, 
                            reason: str, pnl: float = None):
        """Close a trading position"""
        try:
            if symbol not in self.active_positions:
                return
            
            position = self.active_positions[symbol]
            signal = position['signal']
            
            # Calculate PnL if not provided
            if pnl is None:
                if signal.direction == SignalDirection.LONG:
                    pnl = (exit_price - position['entry_price']) * position['position_size']
                elif signal.direction == SignalDirection.SHORT:
                    pnl = (position['entry_price'] - exit_price) * position['position_size']
                else:
                    pnl = 0.0
            
            # Update performance metrics
            await self._update_performance(symbol, pnl, reason)
            
            # Remove from active positions
            del self.active_positions[symbol]
            
            # Trigger callback
            await self._trigger_callbacks('position_closed', {
                'symbol': symbol,
                'exit_price': exit_price,
                'pnl': pnl,
                'reason': reason,
                'signal': signal
            })
            
            self.logger.info(f"Position closed for {symbol}: {reason}, PnL: {pnl:.4f}")
            
        except Exception as e:
            self.logger.error(f"Error closing position: {e}")
    
    async def _update_performance(self, symbol: str, pnl: float, reason: str):
        """Update performance metrics"""
        try:
            if symbol not in self.performance:
                return
            
            perf = self.performance[symbol]
            perf.total_signals += 1
            
            if pnl > 0:
                perf.successful_signals += 1
                perf.average_profit = (perf.average_profit * (perf.successful_signals - 1) + pnl) / perf.successful_signals
            else:
                perf.failed_signals += 1
                perf.average_loss = (perf.average_loss * (perf.failed_signals - 1) + abs(pnl)) / perf.failed_signals
            
            # Calculate win rate
            if perf.total_signals > 0:
                perf.win_rate = perf.successful_signals / perf.total_signals
            
            # Calculate profit factor
            if perf.average_loss > 0:
                perf.profit_factor = perf.average_profit / perf.average_loss
            
            perf.last_update = datetime.now(timezone.utc)
            
            # Update global statistics
            self.stats['total_pnl'] += pnl
            if pnl > 0:
                self.stats['successful_trades'] += 1
            else:
                self.stats['failed_trades'] += 1
            
        except Exception as e:
            self.logger.error(f"Error updating performance: {e}")
    
    def add_callback(self, event_type: str, callback):
        """Add callback for strategy events"""
        self.strategy_callbacks[event_type].append(callback)
        self.logger.info(f"Added callback for {event_type} events")
    
    async def _trigger_callbacks(self, event_type: str, data: Any):
        """Trigger callbacks for strategy events"""
        callbacks = self.strategy_callbacks.get(event_type, [])
        
        for callback in callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(data)
                else:
                    callback(data)
            except Exception as e:
                self.logger.error(f"Error in {event_type} callback: {e}")
    
    def get_strategy_summary(self, symbol: str = None) -> Dict[str, Any]:
        """Get strategy summary"""
        try:
            summary = {
                'stats': self.stats,
                'active_positions': len(self.active_positions),
                'total_signals': sum(len(signals) for signals in self.signals.values()),
                'performance': {sym: perf.__dict__ for sym, perf in self.performance.items()}
            }
            
            if symbol:
                symbol_summary = {
                    'signals': len(self.signals.get(symbol, [])),
                    'active_position': symbol in self.active_positions,
                    'performance': self.performance.get(symbol, {}).__dict__
                }
                summary['symbol_details'] = {symbol: symbol_summary}
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error getting strategy summary: {e}")
            return {}
    
    async def close(self):
        """Close the strategy service"""
        try:
            self.logger.info("Funding Rate Strategy service closed")
        except Exception as e:
            self.logger.error(f"Error closing strategy service: {e}")
    
    async def __aenter__(self):
        """Async context manager entry"""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()
