import asyncio
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import pandas as pd
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class SignalType(Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"

@dataclass
class MTFSignal:
    """Multi-timeframe signal with confidence scores"""
    symbol: str
    timeframe: str
    signal_type: SignalType
    confidence: float
    timestamp: datetime
    patterns: List[str]
    technical_indicators: Dict[str, Any]
    market_context: Dict[str, Any]

@dataclass
class MergedSignal:
    """Final merged signal across multiple timeframes"""
    symbol: str
    signal_type: SignalType
    final_confidence: float
    base_confidence: float
    mtf_boost: float
    contributing_timeframes: List[str]
    confidence_breakdown: Dict[str, float]
    timestamp: datetime
    patterns: List[str]
    technical_indicators: Dict[str, Any]
    market_context: Dict[str, Any]

class MTFSignalMerger:
    """
    Multi-Timeframe Signal Merging System
    Implements mathematical merging of confidence scores across timeframes
    """
    
    def __init__(self):
        # Timeframe hierarchy (higher to lower)
        self.timeframe_hierarchy = ["1d", "4h", "1h", "15m", "5m", "1m"]
        
        # Higher timeframe weights for signal merging
        self.higher_timeframe_weights = {
            "1d": 0.4,    # Daily has highest weight
            "4h": 0.3,    # 4h has high weight
            "1h": 0.2,    # 1h has medium weight
            "15m": 0.1,   # 15m has lower weight
            "5m": 0.05,   # 5m has very low weight
            "1m": 0.02    # 1m has minimal weight
        }
        
        # Signal alignment bonuses
        self.alignment_bonuses = {
            'perfect_alignment': 0.2,    # All timeframes agree
            'strong_alignment': 0.15,    # Most timeframes agree
            'weak_alignment': 0.05,      # Some timeframes agree
            'no_alignment': 0.0          # No alignment
        }
        
        # Minimum confidence thresholds
        self.min_confidence_threshold = 0.3
        self.min_mtf_confidence_threshold = 0.5
        
        logger.info("🚀 MTF Signal Merger initialized")
    
    def calculate_higher_timeframe_weight(self, timeframe: str) -> float:
        """Calculate weight for a specific timeframe"""
        return self.higher_timeframe_weights.get(timeframe, 0.1)
    
    def merge_signals_across_timeframes(
        self, 
        signals: List[MTFSignal], 
        base_timeframe: str = "15m"
    ) -> Optional[MergedSignal]:
        """
        Merge signals across multiple timeframes using mathematical formula
        
        Formula: final_confidence = (lower_tf_conf * (1 + higher_tf_conf * higher_tf_weight))
        
        Args:
            signals: List of MTF signals from different timeframes
            base_timeframe: Base timeframe for the analysis
            
        Returns:
            MergedSignal object with final confidence and breakdown
        """
        if not signals:
            return None
        
        # Group signals by symbol
        signals_by_symbol = {}
        for signal in signals:
            if signal.symbol not in signals_by_symbol:
                signals_by_symbol[signal.symbol] = []
            signals_by_symbol[signal.symbol].append(signal)
        
        merged_signals = []
        
        for symbol, symbol_signals in signals_by_symbol.items():
            merged = self._merge_symbol_signals(symbol_signals, base_timeframe)
            if merged:
                merged_signals.append(merged)
        
        # Return the highest confidence merged signal
        if merged_signals:
            return max(merged_signals, key=lambda x: x.final_confidence)
        
        return None
    
    def _merge_symbol_signals(self, signals: List[MTFSignal], base_timeframe: str) -> Optional[MergedSignal]:
        """Merge signals for a single symbol"""
        if not signals:
            return None
        
        # Find base signal (from base timeframe)
        base_signal = None
        higher_timeframe_signals = []
        
        for signal in signals:
            if signal.timeframe == base_timeframe:
                base_signal = signal
            elif self._is_higher_timeframe(signal.timeframe, base_timeframe):
                higher_timeframe_signals.append(signal)
        
        if not base_signal:
            logger.warning(f"No base signal found for timeframe {base_timeframe}")
            return None
        
        # Calculate MTF boost using the formula
        mtf_boost = self._calculate_mtf_boost(base_signal, higher_timeframe_signals)
        
        # Calculate final confidence
        final_confidence = base_signal.confidence * (1 + mtf_boost)
        final_confidence = min(final_confidence, 1.0)  # Clamp to maximum 1.0
        
        # Create confidence breakdown
        confidence_breakdown = {
            'base_confidence': base_signal.confidence,
            'mtf_boost': mtf_boost,
            'final_confidence': final_confidence
        }
        
        # Add higher timeframe contributions
        for signal in higher_timeframe_signals:
            weight = self.calculate_higher_timeframe_weight(signal.timeframe)
            contribution = signal.confidence * weight
            confidence_breakdown[f'{signal.timeframe}_contribution'] = contribution
        
        # Determine signal type (majority vote with confidence weighting)
        signal_type = self._determine_signal_type(signals)
        
        # Collect contributing timeframes
        contributing_timeframes = [signal.timeframe for signal in signals]
        
        # Merge patterns and indicators
        all_patterns = []
        all_indicators = {}
        market_context = {}
        
        for signal in signals:
            all_patterns.extend(signal.patterns)
            all_indicators.update(signal.technical_indicators)
            market_context.update(signal.market_context)
        
        # Remove duplicates
        all_patterns = list(set(all_patterns))
        
        return MergedSignal(
            symbol=base_signal.symbol,
            signal_type=signal_type,
            final_confidence=final_confidence,
            base_confidence=base_signal.confidence,
            mtf_boost=mtf_boost,
            contributing_timeframes=contributing_timeframes,
            confidence_breakdown=confidence_breakdown,
            timestamp=datetime.utcnow(),
            patterns=all_patterns,
            technical_indicators=all_indicators,
            market_context=market_context
        )
    
    def _calculate_mtf_boost(self, base_signal: MTFSignal, higher_signals: List[MTFSignal]) -> float:
        """
        Calculate MTF boost using the formula: sum(higher_tf_conf * higher_tf_weight)
        
        Args:
            base_signal: Base timeframe signal
            higher_signals: Signals from higher timeframes
            
        Returns:
            MTF boost factor
        """
        mtf_boost = 0.0
        
        for signal in higher_signals:
            weight = self.calculate_higher_timeframe_weight(signal.timeframe)
            contribution = signal.confidence * weight
            mtf_boost += contribution
            
            logger.debug(f"MTF Boost: {signal.timeframe} contributes {contribution:.3f} "
                        f"(confidence: {signal.confidence:.3f}, weight: {weight:.3f})")
        
        # Apply alignment bonus
        alignment_bonus = self._calculate_alignment_bonus(base_signal, higher_signals)
        mtf_boost += alignment_bonus
        
        logger.debug(f"Total MTF Boost: {mtf_boost:.3f} (including alignment bonus: {alignment_bonus:.3f})")
        
        return mtf_boost
    
    def _calculate_alignment_bonus(self, base_signal: MTFSignal, higher_signals: List[MTFSignal]) -> float:
        """Calculate bonus for signal alignment across timeframes"""
        if not higher_signals:
            return 0.0
        
        # Count aligned signals
        aligned_count = 0
        total_count = len(higher_signals)
        
        for signal in higher_signals:
            if signal.signal_type == base_signal.signal_type:
                aligned_count += 1
        
        alignment_ratio = aligned_count / total_count
        
        # Determine alignment level
        if alignment_ratio >= 0.8:
            return self.alignment_bonuses['perfect_alignment']
        elif alignment_ratio >= 0.6:
            return self.alignment_bonuses['strong_alignment']
        elif alignment_ratio >= 0.4:
            return self.alignment_bonuses['weak_alignment']
        else:
            return self.alignment_bonuses['no_alignment']
    
    def _determine_signal_type(self, signals: List[MTFSignal]) -> SignalType:
        """Determine final signal type using weighted voting"""
        bullish_score = 0.0
        bearish_score = 0.0
        
        for signal in signals:
            weight = self.calculate_higher_timeframe_weight(signal.timeframe)
            score = signal.confidence * weight
            
            if signal.signal_type == SignalType.BULLISH:
                bullish_score += score
            elif signal.signal_type == SignalType.BEARISH:
                bearish_score += score
        
        # Determine winner
        if bullish_score > bearish_score * 1.1:  # 10% threshold
            return SignalType.BULLISH
        elif bearish_score > bullish_score * 1.1:
            return SignalType.BEARISH
        else:
            return SignalType.NEUTRAL
    
    def _is_higher_timeframe(self, timeframe: str, base_timeframe: str) -> bool:
        """Check if timeframe is higher than base timeframe"""
        try:
            base_index = self.timeframe_hierarchy.index(base_timeframe)
            current_index = self.timeframe_hierarchy.index(timeframe)
            return current_index < base_index
        except ValueError:
            return False
    
    def filter_high_confidence_signals(self, signals: List[MergedSignal], 
                                     min_confidence: float = None) -> List[MergedSignal]:
        """Filter signals by minimum confidence threshold"""
        if min_confidence is None:
            min_confidence = self.min_mtf_confidence_threshold
        
        filtered_signals = [
            signal for signal in signals 
            if signal.final_confidence >= min_confidence
        ]
        
        logger.info(f"Filtered {len(signals)} signals to {len(filtered_signals)} "
                   f"high-confidence signals (min: {min_confidence})")
        
        return filtered_signals
    
    def get_signal_quality_score(self, merged_signal: MergedSignal) -> float:
        """Calculate signal quality score based on various factors"""
        quality_score = 0.0
        
        # Base confidence contribution (40%)
        quality_score += merged_signal.base_confidence * 0.4
        
        # MTF boost contribution (30%)
        quality_score += min(merged_signal.mtf_boost, 0.5) * 0.6  # Cap at 0.5
        
        # Timeframe diversity contribution (20%)
        timeframe_diversity = len(merged_signal.contributing_timeframes) / 6.0  # Max 6 timeframes
        quality_score += timeframe_diversity * 0.2
        
        # Pattern diversity contribution (10%)
        pattern_diversity = min(len(merged_signal.patterns) / 5.0, 1.0)  # Cap at 5 patterns
        quality_score += pattern_diversity * 0.1
        
        return min(quality_score, 1.0)
    
    def generate_merging_report(self, merged_signal: MergedSignal) -> str:
        """Generate a detailed report of the signal merging process"""
        report = f"""
🎯 MTF Signal Merging Report
============================
Symbol: {merged_signal.symbol}
Signal Type: {merged_signal.signal_type.value}
Final Confidence: {merged_signal.final_confidence:.3f}

📊 Confidence Breakdown:
   • Base Confidence: {merged_signal.base_confidence:.3f}
   • MTF Boost: {merged_signal.mtf_boost:.3f}
   • Quality Score: {self.get_signal_quality_score(merged_signal):.3f}

⏰ Contributing Timeframes: {', '.join(merged_signal.contributing_timeframes)}

🔍 Detailed Contributions:
"""
        
        for key, value in merged_signal.confidence_breakdown.items():
            if key.endswith('_contribution'):
                report += f"   • {key}: {value:.3f}\n"
        
        report += f"""
📈 Patterns Detected: {', '.join(merged_signal.patterns)}
🎛️ Technical Indicators: {len(merged_signal.technical_indicators)} indicators
"""
        
        return report.strip()
    
    def validate_signal_quality(self, merged_signal: MergedSignal) -> Dict[str, Any]:
        """Validate the quality of a merged signal"""
        validation = {
            'is_valid': True,
            'warnings': [],
            'quality_score': self.get_signal_quality_score(merged_signal),
            'recommendations': []
        }
        
        # Check minimum confidence
        if merged_signal.final_confidence < self.min_mtf_confidence_threshold:
            validation['is_valid'] = False
            validation['warnings'].append(f"Confidence too low: {merged_signal.final_confidence:.3f}")
        
        # Check timeframe diversity
        if len(merged_signal.contributing_timeframes) < 2:
            validation['warnings'].append("Limited timeframe diversity")
            validation['recommendations'].append("Consider waiting for more timeframe confirmations")
        
        # Check MTF boost
        if merged_signal.mtf_boost < 0.1:
            validation['warnings'].append("Low MTF boost")
            validation['recommendations'].append("Higher timeframe confirmation would strengthen signal")
        
        # Check signal alignment
        if merged_signal.mtf_boost > 0.3:
            validation['recommendations'].append("Strong MTF confirmation - high confidence signal")
        
        return validation
