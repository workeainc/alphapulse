import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass

from .pattern_detector import CandlestickPatternDetector
from .confidence_factors import ConfidenceFactors
from .indicators import TechnicalIndicators

logger = logging.getLogger(__name__)

@dataclass
class EnhancedPatternSignal:
    """Enhanced pattern signal with confidence scoring"""
    pattern_name: str
    timestamp: datetime
    price: float
    signal_type: str  # "bullish", "bearish", "neutral"
    base_confidence: float  # Raw pattern confidence (0-1)
    final_confidence: float  # Multi-factor confidence (0-1)
    confidence_level: str  # "Very High", "High", "Medium", "Low", "Very Low"
    historical_success_factor: float
    volume_confirmation: Dict
    trend_confirmation: Dict
    multi_timeframe_confirmation: Dict
    breakdown: Dict
    metadata: Dict

class EnhancedPatternDetector:
    """
    Enhanced pattern detector with advanced confidence scoring
    """
    
    def __init__(self, use_historical_data: bool = True):
        self.pattern_detector = CandlestickPatternDetector()
        self.confidence_factors = ConfidenceFactors()
        self.indicators = TechnicalIndicators()
        self.use_historical_data = use_historical_data
        
        # Historical pattern statistics (simulated for now)
        self.pattern_stats = {
            'bullish_engulfing': {'win_rate': 0.65, 'avg_rr': 1.8, 'historical_success_factor': 1.17},
            'hammer': {'win_rate': 0.58, 'avg_rr': 1.6, 'historical_success_factor': 0.93},
            'doji': {'win_rate': 0.52, 'avg_rr': 1.4, 'historical_success_factor': 0.73},
            'shooting_star': {'win_rate': 0.55, 'avg_rr': 1.5, 'historical_success_factor': 0.83},
            'morning_star': {'win_rate': 0.68, 'avg_rr': 1.9, 'historical_success_factor': 1.29},
            'evening_star': {'win_rate': 0.66, 'avg_rr': 1.7, 'historical_success_factor': 1.12}
        }
    
    def detect_patterns_with_confidence(
        self, 
        df: pd.DataFrame, 
        symbol: str = "UNKNOWN",
        timeframe: str = "1h",
        higher_timeframe_data: Optional[pd.DataFrame] = None,
        lower_timeframe_data: Optional[pd.DataFrame] = None
    ) -> List[EnhancedPatternSignal]:
        """
        Detect patterns with comprehensive confidence scoring
        
        Args:
            df: DataFrame with OHLCV data
            symbol: Symbol for historical data lookup
            timeframe: Current timeframe
            higher_timeframe_data: Data from higher timeframe
            lower_timeframe_data: Data from lower timeframe
            
        Returns:
            List of EnhancedPatternSignal objects
        """
        if len(df) < 50:
            logger.warning("Insufficient data for pattern detection (minimum 50 candles required)")
            return []
        
        # Add technical indicators
        df = self.indicators.add_all_indicators(df)
        
        # Detect basic patterns
        pattern_signals = self.pattern_detector.detect_patterns_from_dataframe(df)
        
        if not pattern_signals:
            return []
        
        # Enhance signals with confidence scoring
        enhanced_signals = []
        
        for signal in pattern_signals:
            try:
                enhanced_signal = self._enhance_signal_with_confidence(
                    signal, df, symbol, timeframe, higher_timeframe_data, lower_timeframe_data
                )
                
                if enhanced_signal:
                    enhanced_signals.append(enhanced_signal)
                    
            except Exception as e:
                logger.error(f"Error enhancing signal {signal.pattern}: {e}")
                continue
        
        # Sort by final confidence (highest first)
        enhanced_signals.sort(key=lambda x: x.final_confidence, reverse=True)
        
        logger.info(f"✅ Enhanced pattern detection: {len(enhanced_signals)} signals with confidence scoring")
        
        return enhanced_signals
    
    def _enhance_signal_with_confidence(
        self,
        signal,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str,
        higher_timeframe_data: Optional[pd.DataFrame],
        lower_timeframe_data: Optional[pd.DataFrame]
    ) -> Optional[EnhancedPatternSignal]:
        """Enhance a basic pattern signal with confidence scoring"""
        
        # Get historical success factor
        historical_success_factor = self._get_historical_success_factor(signal.pattern, symbol, timeframe)
        
        # Calculate volume confirmation
        volume_confirmation = self.confidence_factors.calculate_volume_confirmation(
            df, signal.type
        )
        
        # Calculate trend alignment
        trend_confirmation = self.confidence_factors.calculate_trend_alignment(
            df, signal.type
        )
        
        # Calculate multi-timeframe confirmation
        multi_timeframe_factor, multi_timeframe_desc = self.confidence_factors.calculate_multi_timeframe_confirmation(
            current_timeframe=timeframe,
            higher_timeframe_data=higher_timeframe_data,
            lower_timeframe_data=lower_timeframe_data,
            pattern_type=signal.type
        )
        
        # Calculate final confidence
        confidence_result = self.confidence_factors.calculate_final_confidence(
            base_score=signal.confidence,
            historical_success_factor=historical_success_factor,
            volume_confirmation=volume_confirmation,
            trend_confirmation=trend_confirmation,
            multi_timeframe_factor=multi_timeframe_factor
        )
        
        # Create enhanced signal
        enhanced_signal = EnhancedPatternSignal(
            pattern_name=signal.pattern,
            timestamp=signal.timestamp or datetime.now(),
            price=signal.strength,  # Using strength as price proxy
            signal_type=signal.type,
            base_confidence=signal.confidence,
            final_confidence=confidence_result['final_confidence'],
            confidence_level=confidence_result['confidence_level'],
            historical_success_factor=historical_success_factor,
            volume_confirmation={
                'pattern_type': volume_confirmation.pattern_type.value,
                'strength': volume_confirmation.strength.value,
                'factor': volume_confirmation.factor,
                'description': volume_confirmation.description
            },
            trend_confirmation={
                'alignment': trend_confirmation.alignment.value,
                'factor': trend_confirmation.factor,
                'description': trend_confirmation.description
            },
            multi_timeframe_confirmation={
                'factor': multi_timeframe_factor,
                'description': multi_timeframe_desc
            },
            breakdown=confidence_result['breakdown'],
            metadata=signal.additional_info or {}
        )
        
        return enhanced_signal
    
    def _get_historical_success_factor(self, pattern_name: str, symbol: str, timeframe: str) -> float:
        """Get historical success factor for a pattern"""
        if not self.use_historical_data:
            return 1.0  # Neutral factor if no historical data
        
        # Look up in pattern stats
        pattern_key = pattern_name.lower().replace(' ', '_')
        
        if pattern_key in self.pattern_stats:
            stats = self.pattern_stats[pattern_key]
            return stats['historical_success_factor']
        
        # Default factor for unknown patterns
        return 1.0
    
    def get_high_confidence_signals(
        self, 
        signals: List[EnhancedPatternSignal], 
        min_confidence: float = 0.6
    ) -> List[EnhancedPatternSignal]:
        """Filter signals by minimum confidence threshold"""
        return [s for s in signals if s.final_confidence >= min_confidence]
    
    def get_signal_summary(self, signals: List[EnhancedPatternSignal]) -> Dict:
        """Get summary statistics for enhanced signals"""
        if not signals:
            return {}
        
        summary = {
            'total_signals': len(signals),
            'confidence_distribution': {
                'very_high': len([s for s in signals if s.confidence_level == 'Very High']),
                'high': len([s for s in signals if s.confidence_level == 'High']),
                'medium': len([s for s in signals if s.confidence_level == 'Medium']),
                'low': len([s for s in signals if s.confidence_level == 'Low']),
                'very_low': len([s for s in signals if s.confidence_level == 'Very Low'])
            },
            'pattern_distribution': {},
            'average_confidence': np.mean([s.final_confidence for s in signals]),
            'highest_confidence_signal': None,
            'confidence_breakdown': {
                'volume_factors': {},
                'trend_factors': {},
                'timeframe_factors': {}
            }
        }
        
        # Pattern distribution
        for signal in signals:
            if signal.pattern_name not in summary['pattern_distribution']:
                summary['pattern_distribution'][signal.pattern_name] = 0
            summary['pattern_distribution'][signal.pattern_name] += 1
        
        # Highest confidence signal
        if signals:
            highest_signal = max(signals, key=lambda x: x.final_confidence)
            summary['highest_confidence_signal'] = {
                'pattern': highest_signal.pattern_name,
                'confidence': highest_signal.final_confidence,
                'level': highest_signal.confidence_level,
                'type': highest_signal.signal_type
            }
        
        # Factor breakdown
        volume_factors = [s.volume_confirmation['factor'] for s in signals]
        trend_factors = [s.trend_confirmation['factor'] for s in signals]
        timeframe_factors = [s.multi_timeframe_confirmation['factor'] for s in signals]
        
        summary['confidence_breakdown']['volume_factors'] = {
            'average': np.mean(volume_factors),
            'min': np.min(volume_factors),
            'max': np.max(volume_factors)
        }
        summary['confidence_breakdown']['trend_factors'] = {
            'average': np.mean(trend_factors),
            'min': np.min(trend_factors),
            'max': np.max(trend_factors)
        }
        summary['confidence_breakdown']['timeframe_factors'] = {
            'average': np.mean(timeframe_factors),
            'min': np.min(timeframe_factors),
            'max': np.max(timeframe_factors)
        }
        
        return summary
    
    def generate_confidence_report(self, signals: List[EnhancedPatternSignal]) -> str:
        """Generate a human-readable confidence report"""
        if not signals:
            return "No signals detected"
        
        summary = self.get_signal_summary(signals)
        
        report = f"""
🎯 Enhanced Pattern Detection Report
{'='*50}

📊 Signal Summary:
   Total Signals: {summary['total_signals']}
   Average Confidence: {summary['average_confidence']:.3f}

📈 Confidence Distribution:
   Very High: {summary['confidence_distribution']['very_high']}
   High: {summary['confidence_distribution']['high']}
   Medium: {summary['confidence_distribution']['medium']}
   Low: {summary['confidence_distribution']['low']}
   Very Low: {summary['confidence_distribution']['very_low']}

🔍 Pattern Distribution:
"""
        
        for pattern, count in summary['pattern_distribution'].items():
            report += f"   {pattern}: {count}\n"
        
        if summary['highest_confidence_signal']:
            highest = summary['highest_confidence_signal']
            report += f"""
🏆 Highest Confidence Signal:
   Pattern: {highest['pattern']}
   Confidence: {highest['confidence']:.3f} ({highest['level']})
   Type: {highest['type']}
"""
        
        # Factor breakdown
        breakdown = summary['confidence_breakdown']
        report += f"""
📊 Factor Breakdown:
   Volume Factors: {breakdown['volume_factors']['average']:.3f} avg ({breakdown['volume_factors']['min']:.3f}-{breakdown['volume_factors']['max']:.3f})
   Trend Factors: {breakdown['trend_factors']['average']:.3f} avg ({breakdown['trend_factors']['min']:.3f}-{breakdown['trend_factors']['max']:.3f})
   Timeframe Factors: {breakdown['timeframe_factors']['average']:.3f} avg ({breakdown['timeframe_factors']['min']:.3f}-{breakdown['timeframe_factors']['max']:.3f})
"""
        
        return report.strip()
    
    def update_historical_stats(self, pattern_name: str, symbol: str, timeframe: str, stats: Dict):
        """Update historical statistics for a pattern"""
        pattern_key = pattern_name.lower().replace(' ', '_')
        
        if pattern_key not in self.pattern_stats:
            self.pattern_stats[pattern_key] = {}
        
        self.pattern_stats[pattern_key].update(stats)
        
        # Recalculate historical success factor
        if 'win_rate' in stats and 'avg_rr' in stats:
            self.pattern_stats[pattern_key]['historical_success_factor'] = min(
                stats['win_rate'] * stats['avg_rr'], 2.0
            )
        
        logger.info(f"✅ Updated historical stats for {pattern_name}: {stats}")
