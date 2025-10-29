import asyncio
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import pandas as pd
from dataclasses import dataclass
from enum import Enum

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from ..src.services.mtf_pattern_integrator import MTFPatternIntegrator, MTFPatternResult
from ..src.services.mtf_orchestrator import MTFOrchestrator
from ..src.services.mtf_signal_merger import MTFSignalMerger, MergedSignal, SignalType
from ..src.strategies.enhanced_pattern_detector import EnhancedPatternDetector

logger = logging.getLogger(__name__)

class SignalPriority(Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

@dataclass
class MTFSignal:
    symbol: str
    timeframe: str
    signal_type: SignalType
    confidence: float
    priority: SignalPriority
    patterns: List[str]
    technical_indicators: Dict[str, Any]
    market_context: Dict[str, Any]
    timestamp: datetime
    entry_price: float
    stop_loss: Optional[float]
    take_profit: Optional[float]
    risk_reward_ratio: Optional[float]

@dataclass
class RealTimeSignal:
    symbol: str
    signal_type: SignalType
    final_confidence: float
    priority: SignalPriority
    entry_price: float
    stop_loss: Optional[float]
    take_profit: Optional[float]
    risk_reward_ratio: Optional[float]
    patterns: List[str]
    timeframes: List[str]
    mtf_alignment: Dict[str, Any]
    technical_indicators: Dict[str, Any]
    market_context: Dict[str, Any]
    timestamp: datetime
    signal_strength: str

class MTFSignalGenerator:
    """
    Real-time signal generator with MTF integration
    Generates enhanced trading signals using multi-timeframe analysis
    """
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.mtf_pattern_integrator = MTFPatternIntegrator(redis_url)
        self.mtf_orchestrator = MTFOrchestrator(redis_url)
        self.mtf_signal_merger = MTFSignalMerger()
        self.enhanced_detector = EnhancedPatternDetector()
        
        # Signal configuration
        self.min_confidence_threshold = 0.6
        self.high_priority_threshold = 0.8
        self.medium_priority_threshold = 0.7
        
        # Risk management
        self.default_risk_reward_ratio = 2.0
        self.max_risk_per_trade = 0.02  # 2% max risk per trade
        
        # Performance tracking
        self.stats = {
            'total_signals_generated': 0,
            'high_priority_signals': 0,
            'medium_priority_signals': 0,
            'low_priority_signals': 0,
            'mtf_aligned_signals': 0,
            'processing_times': []
        }
        
        logger.info("🚀 MTF Signal Generator initialized")
    
    async def generate_real_time_signals(
        self, 
        symbol: str, 
        timeframe: str, 
        data: pd.DataFrame
    ) -> List[RealTimeSignal]:
        """
        Generate real-time trading signals with MTF integration
        """
        start_time = datetime.now()
        
        try:
            # Detect patterns with MTF context
            mtf_patterns = await self.mtf_pattern_integrator.detect_patterns_with_mtf_context(
                symbol, timeframe, data
            )
            
            if not mtf_patterns:
                return []
            
            # Generate individual timeframe signals
            timeframe_signals = []
            
            for pattern in mtf_patterns:
                signal = await self._create_timeframe_signal(pattern, data)
                if signal:
                    timeframe_signals.append(signal)
            
            # Merge signals across timeframes
            merged_signals = await self._merge_mtf_signals(symbol, timeframe_signals)
            
            # Convert to real-time signals
            real_time_signals = []
            
            for merged_signal in merged_signals:
                real_time_signal = await self._create_real_time_signal(merged_signal, data)
                if real_time_signal:
                    real_time_signals.append(real_time_signal)
            
            # Update statistics
            processing_time = (datetime.now() - start_time).total_seconds()
            self._update_stats(len(real_time_signals), processing_time)
            
            logger.info(f"📡 Generated {len(real_time_signals)} real-time signals for {symbol} {timeframe} in {processing_time:.3f}s")
            
            return real_time_signals
            
        except Exception as e:
            logger.error(f"❌ Error generating real-time signals: {e}")
            return []
    
    async def _create_timeframe_signal(
        self, 
        pattern: MTFPatternResult, 
        data: pd.DataFrame
    ) -> Optional[MTFSignal]:
        """
        Create a signal for a specific timeframe
        """
        try:
            # Determine signal type
            signal_type = self._determine_signal_type(pattern.pattern_name)
            
            # Calculate entry price
            entry_price = self._calculate_entry_price(pattern, data)
            
            # Calculate stop loss and take profit
            stop_loss, take_profit = self._calculate_risk_levels(pattern, data, entry_price)
            
            # Calculate risk/reward ratio
            risk_reward_ratio = self._calculate_risk_reward_ratio(entry_price, stop_loss, take_profit)
            
            # Determine priority
            priority = self._determine_priority(pattern.confidence, pattern.mtf_boost)
            
            # Create signal
            signal = MTFSignal(
                symbol=pattern.symbol,
                timeframe=pattern.timeframe,
                signal_type=signal_type,
                confidence=pattern.confidence,
                priority=priority,
                patterns=[pattern.pattern_name],
                technical_indicators=pattern.technical_indicators,
                market_context=pattern.market_context,
                timestamp=pattern.timestamp,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                risk_reward_ratio=risk_reward_ratio
            )
            
            return signal
            
        except Exception as e:
            logger.error(f"❌ Error creating timeframe signal: {e}")
            return None
    
    def _determine_signal_type(self, pattern_name: str) -> SignalType:
        """
        Determine signal type based on pattern
        """
        pattern_name_lower = pattern_name.lower()
        
        bullish_patterns = ['bullish_engulfing', 'hammer', 'morning_star', 'piercing_line', 'doji_bullish']
        bearish_patterns = ['bearish_engulfing', 'shooting_star', 'evening_star', 'dark_cloud_cover', 'doji_bearish']
        
        if any(bull in pattern_name_lower for bull in bullish_patterns):
            return SignalType.BULLISH
        elif any(bear in pattern_name_lower for bear in bearish_patterns):
            return SignalType.BEARISH
        else:
            return SignalType.NEUTRAL
    
    def _calculate_entry_price(self, pattern: MTFPatternResult, data: pd.DataFrame) -> float:
        """
        Calculate optimal entry price
        """
        try:
            current_price = data['close'].iloc[-1]
            
            # For bullish patterns, entry slightly above current price
            if self._determine_signal_type(pattern.pattern_name) == SignalType.BULLISH:
                return current_price * 1.001  # 0.1% above current price
            
            # For bearish patterns, entry slightly below current price
            elif self._determine_signal_type(pattern.pattern_name) == SignalType.BEARISH:
                return current_price * 0.999  # 0.1% below current price
            
            else:
                return current_price
                
        except Exception as e:
            logger.warning(f"⚠️ Error calculating entry price: {e}")
            return data['close'].iloc[-1] if not data.empty else 0.0
    
    def _calculate_risk_levels(
        self, 
        pattern: MTFPatternResult, 
        data: pd.DataFrame, 
        entry_price: float
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        Calculate stop loss and take profit levels
        """
        try:
            signal_type = self._determine_signal_type(pattern.pattern_name)
            
            # Calculate ATR for volatility-based stops
            atr = self._calculate_atr(data)
            
            if signal_type == SignalType.BULLISH:
                # Bullish signal: stop below entry, target above
                stop_loss = entry_price - (atr * 2)  # 2 ATR below entry
                take_profit = entry_price + (atr * 2 * self.default_risk_reward_ratio)
                
            elif signal_type == SignalType.BEARISH:
                # Bearish signal: stop above entry, target below
                stop_loss = entry_price + (atr * 2)  # 2 ATR above entry
                take_profit = entry_price - (atr * 2 * self.default_risk_reward_ratio)
                
            else:
                # Neutral signal: no clear direction
                stop_loss = None
                take_profit = None
            
            return stop_loss, take_profit
            
        except Exception as e:
            logger.warning(f"⚠️ Error calculating risk levels: {e}")
            return None, None
    
    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> float:
        """
        Calculate Average True Range
        """
        try:
            if len(data) < period + 1:
                return data['close'].std() if not data.empty else 0.0
            
            high = data['high']
            low = data['low']
            close = data['close']
            
            # Calculate True Range
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = true_range.rolling(window=period).mean().iloc[-1]
            
            return atr if pd.notna(atr) else data['close'].std()
            
        except Exception as e:
            logger.warning(f"⚠️ Error calculating ATR: {e}")
            return data['close'].std() if not data.empty else 0.0
    
    def _calculate_risk_reward_ratio(
        self, 
        entry_price: float, 
        stop_loss: Optional[float], 
        take_profit: Optional[float]
    ) -> Optional[float]:
        """
        Calculate risk/reward ratio
        """
        try:
            if stop_loss is None or take_profit is None:
                return None
            
            risk = abs(entry_price - stop_loss)
            reward = abs(take_profit - entry_price)
            
            if risk > 0:
                return reward / risk
            else:
                return None
                
        except Exception as e:
            logger.warning(f"⚠️ Error calculating risk/reward ratio: {e}")
            return None
    
    def _determine_priority(self, confidence: float, mtf_boost: float) -> SignalPriority:
        """
        Determine signal priority based on confidence and MTF boost
        """
        total_score = confidence + mtf_boost
        
        if total_score >= self.high_priority_threshold:
            return SignalPriority.HIGH
        elif total_score >= self.medium_priority_threshold:
            return SignalPriority.MEDIUM
        else:
            return SignalPriority.LOW
    
    async def _merge_mtf_signals(
        self, 
        symbol: str, 
        timeframe_signals: List[MTFSignal]
    ) -> List[MergedSignal]:
        """
        Merge signals across multiple timeframes
        """
        try:
            if not timeframe_signals:
                return []
            
            # Group signals by type
            bullish_signals = [s for s in timeframe_signals if s.signal_type == SignalType.BULLISH]
            bearish_signals = [s for s in timeframe_signals if s.signal_type == SignalType.BEARISH]
            neutral_signals = [s for s in timeframe_signals if s.signal_type == SignalType.NEUTRAL]
            
            merged_signals = []
            
            # Merge bullish signals
            if bullish_signals:
                merged_bullish = self.mtf_signal_merger.merge_signals(symbol, bullish_signals)
                if merged_bullish:
                    merged_signals.append(merged_bullish)
            
            # Merge bearish signals
            if bearish_signals:
                merged_bearish = self.mtf_signal_merger.merge_signals(symbol, bearish_signals)
                if merged_bearish:
                    merged_signals.append(merged_bearish)
            
            # Merge neutral signals
            if neutral_signals:
                merged_neutral = self.mtf_signal_merger.merge_signals(symbol, neutral_signals)
                if merged_neutral:
                    merged_signals.append(merged_neutral)
            
            return merged_signals
            
        except Exception as e:
            logger.error(f"❌ Error merging MTF signals: {e}")
            return []
    
    async def _create_real_time_signal(
        self, 
        merged_signal: MergedSignal, 
        data: pd.DataFrame
    ) -> Optional[RealTimeSignal]:
        """
        Create a real-time signal from merged MTF signal
        """
        try:
            # Determine signal strength
            signal_strength = self._determine_signal_strength(merged_signal.final_confidence)
            
            # Get contributing timeframes
            timeframes = [tf for tf in merged_signal.contributing_timeframes]
            
            # Create MTF alignment summary
            mtf_alignment = {
                'timeframes': timeframes,
                'confidence_breakdown': merged_signal.confidence_breakdown,
                'mtf_boost': merged_signal.mtf_boost,
                'alignment_score': sum(merged_signal.confidence_breakdown.values()) / len(merged_signal.confidence_breakdown) if merged_signal.confidence_breakdown else 0.0
            }
            
            # Create real-time signal
            real_time_signal = RealTimeSignal(
                symbol=merged_signal.symbol,
                signal_type=merged_signal.signal_type,
                final_confidence=merged_signal.final_confidence,
                priority=self._determine_priority(merged_signal.final_confidence, merged_signal.mtf_boost),
                entry_price=data['close'].iloc[-1] if not data.empty else 0.0,
                stop_loss=None,  # Will be calculated based on risk management
                take_profit=None,  # Will be calculated based on risk management
                risk_reward_ratio=self.default_risk_reward_ratio,
                patterns=merged_signal.patterns,
                timeframes=timeframes,
                mtf_alignment=mtf_alignment,
                technical_indicators=merged_signal.technical_indicators,
                market_context=merged_signal.market_context,
                timestamp=merged_signal.timestamp,
                signal_strength=signal_strength
            )
            
            return real_time_signal
            
        except Exception as e:
            logger.error(f"❌ Error creating real-time signal: {e}")
            return None
    
    def _determine_signal_strength(self, confidence: float) -> str:
        """
        Determine signal strength based on confidence
        """
        if confidence >= 0.9:
            return "very_strong"
        elif confidence >= 0.8:
            return "strong"
        elif confidence >= 0.7:
            return "moderate"
        elif confidence >= 0.6:
            return "weak"
        else:
            return "very_weak"
    
    def _update_stats(self, signals_count: int, processing_time: float):
        """
        Update performance statistics
        """
        self.stats['total_signals_generated'] += signals_count
        self.stats['processing_times'].append(processing_time)
        
        # Keep only last 100 processing times
        if len(self.stats['processing_times']) > 100:
            self.stats['processing_times'] = self.stats['processing_times'][-100:]
    
    async def get_signal_stats(self) -> Dict[str, Any]:
        """
        Get signal generation statistics
        """
        avg_processing_time = (
            sum(self.stats['processing_times']) / len(self.stats['processing_times'])
            if self.stats['processing_times'] else 0.0
        )
        
        return {
            'total_signals_generated': self.stats['total_signals_generated'],
            'high_priority_signals': self.stats['high_priority_signals'],
            'medium_priority_signals': self.stats['medium_priority_signals'],
            'low_priority_signals': self.stats['low_priority_signals'],
            'mtf_aligned_signals': self.stats['mtf_aligned_signals'],
            'average_processing_time': avg_processing_time,
            'signal_generation_rate': self.stats['total_signals_generated'] / max(1, len(self.stats['processing_times']))
        }
    
    async def filter_signals_by_priority(
        self, 
        signals: List[RealTimeSignal], 
        min_priority: SignalPriority = SignalPriority.MEDIUM
    ) -> List[RealTimeSignal]:
        """
        Filter signals by priority level
        """
        priority_order = {
            SignalPriority.HIGH: 3,
            SignalPriority.MEDIUM: 2,
            SignalPriority.LOW: 1
        }
        
        min_priority_level = priority_order.get(min_priority, 1)
        
        filtered_signals = []
        for signal in signals:
            signal_priority_level = priority_order.get(signal.priority, 1)
            if signal_priority_level >= min_priority_level:
                filtered_signals.append(signal)
        
        return filtered_signals
    
    async def get_signal_summary(self, signals: List[RealTimeSignal]) -> Dict[str, Any]:
        """
        Generate a summary of signals
        """
        if not signals:
            return {'message': 'No signals generated'}
        
        # Count by type
        bullish_count = len([s for s in signals if s.signal_type == SignalType.BULLISH])
        bearish_count = len([s for s in signals if s.signal_type == SignalType.BEARISH])
        neutral_count = len([s for s in signals if s.signal_type == SignalType.NEUTRAL])
        
        # Count by priority
        high_priority_count = len([s for s in signals if s.priority == SignalPriority.HIGH])
        medium_priority_count = len([s for s in signals if s.priority == SignalPriority.MEDIUM])
        low_priority_count = len([s for s in signals if s.priority == SignalPriority.LOW])
        
        # Average confidence
        avg_confidence = sum(s.final_confidence for s in signals) / len(signals)
        
        # Most common patterns
        pattern_counts = {}
        for signal in signals:
            for pattern in signal.patterns:
                pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
        
        most_common_patterns = sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            'total_signals': len(signals),
            'signal_types': {
                'bullish': bullish_count,
                'bearish': bearish_count,
                'neutral': neutral_count
            },
            'priorities': {
                'high': high_priority_count,
                'medium': medium_priority_count,
                'low': low_priority_count
            },
            'average_confidence': avg_confidence,
            'most_common_patterns': most_common_patterns,
            'timeframe_coverage': list(set(tf for signal in signals for tf in signal.timeframes))
        }
