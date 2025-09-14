#!/usr/bin/env python3
"""
Trend Context Storage for AlphaPulse
Phase 3: Store Complete Trend Context in Results
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)

class TrendAlignmentLevel(Enum):
    """Trend alignment levels"""
    STRONG_ALIGNMENT = "strong_alignment"
    WEAK_ALIGNMENT = "weak_alignment"
    COUNTER_TREND = "counter_trend"
    NEUTRAL = "neutral"

@dataclass
class TrendContextData:
    """Complete trend context data for storage"""
    # Basic trend information
    trend_direction: str  # "bullish", "bearish", "neutral"
    trend_strength: str   # "weak", "moderate", "strong", "extreme"
    adx_value: float
    ema_alignment: bool
    
    # EMA values
    ema_20: float
    ema_50: float
    ema_200: float
    
    # Additional indicators
    rsi: float
    macd_signal: str
    hull_ma_fast: float
    hull_ma_medium: float
    hull_ma_slow: float
    
    # Trend confidence
    trend_confidence: float
    trend_description: str
    
    # Multi-timeframe context
    higher_tf_trend: Optional[str] = None
    higher_tf_strength: Optional[str] = None
    higher_tf_adx: Optional[float] = None
    lower_tf_trend: Optional[str] = None
    lower_tf_strength: Optional[str] = None
    lower_tf_adx: Optional[float] = None
    
    # Filter results
    ema_filter_passed: bool = False
    adx_filter_passed: bool = False
    hull_ma_filter_passed: bool = False
    trend_consistency_passed: bool = False
    volatility_acceptable: bool = False
    
    # Overall assessment
    trend_alignment: bool = False
    confidence_multiplier: float = 1.0
    alignment_description: str = ""

@dataclass
class EnhancedPatternResult:
    """Enhanced pattern result with comprehensive trend context"""
    # Pattern information
    pattern: str
    timeframe: str
    confidence: float
    
    # Trend alignment information
    trend_alignment: bool
    higher_tf_trend: str
    adx: float
    ema_alignment: bool
    
    # Additional context
    trend_context: TrendContextData
    volume_confirmed: bool = False
    volume_factor: float = 0.0
    
    # Metadata
    timestamp: datetime = None
    symbol: str = ""
    additional_info: Dict[str, Any] = None

class TrendContextStorage:
    """
    Comprehensive trend context storage system
    """
    
    def __init__(self):
        self.trend_cache: Dict[str, Dict[str, Any]] = {}
        self.cache_ttl = 300  # 5 minutes cache TTL
        self.last_cache_update: Dict[str, datetime] = {}
        
        logger.info("🚀 Trend Context Storage initialized")
    
    def create_enhanced_pattern_result(
        self,
        pattern_name: str,
        pattern_direction: str,
        timeframe: str,
        base_confidence: float,
        current_tf_data: pd.DataFrame,
        higher_tf_data: Optional[pd.DataFrame] = None,
        lower_tf_data: Optional[pd.DataFrame] = None,
        volume_confirmed: bool = False,
        volume_factor: float = 0.0,
        symbol: str = "UNKNOWN"
    ) -> EnhancedPatternResult:
        """
        Create enhanced pattern result with comprehensive trend context
        
        Args:
            pattern_name: Name of the detected pattern
            pattern_direction: Direction of the pattern
            timeframe: Current timeframe
            base_confidence: Base pattern confidence
            current_tf_data: Current timeframe data
            higher_tf_data: Higher timeframe data (optional)
            lower_tf_data: Lower timeframe data (optional)
            volume_confirmed: Whether volume confirms the pattern
            volume_factor: Volume confirmation factor
            symbol: Symbol being analyzed
            
        Returns:
            EnhancedPatternResult with complete trend context
        """
        try:
            # Analyze trend context for current timeframe
            current_trend_context = self._analyze_trend_context(current_tf_data, timeframe)
            
            # Analyze higher timeframe if available
            higher_tf_context = None
            if higher_tf_data is not None:
                higher_tf_context = self._analyze_trend_context(higher_tf_data, self._get_higher_timeframe(timeframe))
            
            # Analyze lower timeframe if available
            lower_tf_context = None
            if lower_tf_data is not None:
                lower_tf_context = self._analyze_trend_context(lower_tf_data, self._get_lower_timeframe(timeframe))
            
            # Create comprehensive trend context data
            trend_context_data = self._create_trend_context_data(
                current_trend_context,
                higher_tf_context,
                lower_tf_context,
                pattern_direction
            )
            
            # Calculate trend alignment
            trend_alignment = self._calculate_trend_alignment(
                pattern_name, pattern_direction, trend_context_data
            )
            
            # Apply trend filters
            filter_results = self._apply_trend_filters(
                current_tf_data, pattern_direction, timeframe
            )
            
            # Update trend context with filter results
            trend_context_data.ema_filter_passed = filter_results.ema_alignment
            trend_context_data.adx_filter_passed = filter_results.adx_strength
            trend_context_data.hull_ma_filter_passed = filter_results.hull_ma_alignment
            trend_context_data.trend_consistency_passed = filter_results.trend_consistency
            trend_context_data.volatility_acceptable = filter_results.volatility_acceptable
            trend_context_data.trend_alignment = trend_alignment
            trend_context_data.confidence_multiplier = filter_results.overall_score
            trend_context_data.alignment_description = self._generate_alignment_description(
                pattern_name, pattern_direction, trend_context_data, filter_results
            )
            
            # Calculate final confidence
            final_confidence = base_confidence * trend_context_data.confidence_multiplier
            
            # Create enhanced result
            result = EnhancedPatternResult(
                pattern=pattern_name,
                timeframe=timeframe,
                confidence=final_confidence,
                trend_alignment=trend_alignment,
                higher_tf_trend=trend_context_data.higher_tf_trend or "unknown",
                adx=trend_context_data.adx_value,
                ema_alignment=trend_context_data.ema_alignment,
                trend_context=trend_context_data,
                volume_confirmed=volume_confirmed,
                volume_factor=volume_factor,
                timestamp=datetime.now(),
                symbol=symbol,
                additional_info={
                    'base_confidence': base_confidence,
                    'trend_multiplier': trend_context_data.confidence_multiplier,
                    'filter_results': {
                        'passed_filters': filter_results.passed_filters,
                        'failed_filters': filter_results.failed_filters,
                        'warnings': filter_results.warnings
                    }
                }
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error creating enhanced pattern result: {e}")
            return self._get_default_enhanced_result(pattern_name, timeframe, base_confidence)
    
    def _analyze_trend_context(self, df: pd.DataFrame, timeframe: str) -> Dict[str, Any]:
        """Analyze trend context for a timeframe"""
        try:
            if len(df) < 50:
                return self._get_default_trend_context()
            
            # Calculate technical indicators
            ema_20 = df['close'].ewm(span=20).mean().iloc[-1]
            ema_50 = df['close'].ewm(span=50).mean().iloc[-1]
            ema_200 = df['close'].ewm(span=200).mean().iloc[-1] if len(df) >= 200 else ema_50
            current_price = df['close'].iloc[-1]
            
            # Calculate ADX
            adx_value = self._calculate_adx(df)
            
            # Calculate RSI
            rsi = self._calculate_rsi(df)
            
            # Calculate MACD signal
            macd_signal = self._calculate_macd_signal(df)
            
            # Calculate Hull MAs
            hull_fast = self._calculate_hull_ma(df, 9)
            hull_medium = self._calculate_hull_ma(df, 18)
            hull_slow = self._calculate_hull_ma(df, 36)
            
            # Determine trend direction and strength
            trend_direction = self._determine_trend_direction(current_price, ema_20, ema_50, ema_200, adx_value)
            trend_strength = self._determine_trend_strength(adx_value)
            
            # Check EMA alignment
            ema_alignment = self._check_ema_alignment(ema_20, ema_50, ema_200, trend_direction)
            
            # Calculate trend confidence
            trend_confidence = self._calculate_trend_confidence(
                trend_direction, trend_strength, ema_alignment, adx_value, rsi
            )
            
            # Generate description
            trend_description = self._generate_trend_description(
                trend_direction, trend_strength, ema_alignment, adx_value
            )
            
            return {
                'trend_direction': trend_direction,
                'trend_strength': trend_strength,
                'adx_value': adx_value,
                'ema_alignment': ema_alignment,
                'ema_20': ema_20,
                'ema_50': ema_50,
                'ema_200': ema_200,
                'rsi': rsi,
                'macd_signal': macd_signal,
                'hull_fast': hull_fast,
                'hull_medium': hull_medium,
                'hull_slow': hull_slow,
                'trend_confidence': trend_confidence,
                'trend_description': trend_description
            }
            
        except Exception as e:
            logger.error(f"Error analyzing trend context: {e}")
            return self._get_default_trend_context()
    
    def _create_trend_context_data(
        self,
        current_context: Dict[str, Any],
        higher_context: Optional[Dict[str, Any]],
        lower_context: Optional[Dict[str, Any]],
        pattern_direction: str
    ) -> TrendContextData:
        """Create comprehensive trend context data"""
        return TrendContextData(
            trend_direction=current_context['trend_direction'],
            trend_strength=current_context['trend_strength'],
            adx_value=current_context['adx_value'],
            ema_alignment=current_context['ema_alignment'],
            ema_20=current_context['ema_20'],
            ema_50=current_context['ema_50'],
            ema_200=current_context['ema_200'],
            rsi=current_context['rsi'],
            macd_signal=current_context['macd_signal'],
            hull_ma_fast=current_context['hull_fast'],
            hull_ma_medium=current_context['hull_medium'],
            hull_ma_slow=current_context['hull_slow'],
            trend_confidence=current_context['trend_confidence'],
            trend_description=current_context['trend_description'],
            higher_tf_trend=higher_context['trend_direction'] if higher_context else None,
            higher_tf_strength=higher_context['trend_strength'] if higher_context else None,
            higher_tf_adx=higher_context['adx_value'] if higher_context else None,
            lower_tf_trend=lower_context['trend_direction'] if lower_context else None,
            lower_tf_strength=lower_context['trend_strength'] if lower_context else None,
            lower_tf_adx=lower_context['adx_value'] if lower_context else None
        )
    
    def _calculate_trend_alignment(
        self,
        pattern_name: str,
        pattern_direction: str,
        trend_context: TrendContextData
    ) -> bool:
        """Calculate overall trend alignment"""
        try:
            # Basic alignment check
            if pattern_direction == trend_context.trend_direction:
                base_alignment = True
            elif trend_context.trend_direction == "neutral":
                base_alignment = True
            else:
                base_alignment = False
            
            # Higher timeframe confirmation
            higher_tf_confirmed = False
            if trend_context.higher_tf_trend:
                if pattern_direction == trend_context.higher_tf_trend:
                    higher_tf_confirmed = True
                elif trend_context.higher_tf_trend == "neutral":
                    higher_tf_confirmed = True
            
            # ADX strength check
            adx_strong = trend_context.adx_value >= 25
            
            # EMA alignment check
            ema_aligned = trend_context.ema_alignment
            
            # Overall alignment
            alignment_score = 0
            if base_alignment:
                alignment_score += 1
            if higher_tf_confirmed:
                alignment_score += 1
            if adx_strong:
                alignment_score += 1
            if ema_aligned:
                alignment_score += 1
            
            return alignment_score >= 2  # At least 2 out of 4 criteria met
            
        except Exception as e:
            logger.error(f"Error calculating trend alignment: {e}")
            return False
    
    def _apply_trend_filters(
        self,
        df: pd.DataFrame,
        pattern_direction: str,
        timeframe: str
    ) -> Any:
        """Apply trend filters (import and use AdvancedTrendFilters)"""
        try:
            # Import here to avoid circular imports
            from .advanced_trend_filters import AdvancedTrendFilters
            
            filters = AdvancedTrendFilters()
            return filters.apply_comprehensive_trend_filters(df, pattern_direction, timeframe)
            
        except Exception as e:
            logger.error(f"Error applying trend filters: {e}")
            # Return default filter result
            from .advanced_trend_filters import ComprehensiveTrendFilter
            return ComprehensiveTrendFilter(
                ema_alignment=False,
                adx_strength=False,
                hull_ma_alignment=False,
                trend_consistency=False,
                volatility_acceptable=False,
                overall_score=0.5,
                passed_filters=[],
                failed_filters=["Filter application failed"],
                warnings=[]
            )
    
    def _generate_alignment_description(
        self,
        pattern_name: str,
        pattern_direction: str,
        trend_context: TrendContextData,
        filter_results: Any
    ) -> str:
        """Generate comprehensive alignment description"""
        description_parts = []
        
        # Basic trend alignment
        if pattern_direction == trend_context.trend_direction:
            description_parts.append(f"{pattern_direction} pattern aligns with {trend_context.trend_direction} trend")
        elif trend_context.trend_direction == "neutral":
            description_parts.append(f"{pattern_direction} pattern in neutral trend")
        else:
            description_parts.append(f"{pattern_direction} pattern against {trend_context.trend_direction} trend")
        
        # Higher timeframe confirmation
        if trend_context.higher_tf_trend:
            if pattern_direction == trend_context.higher_tf_trend:
                description_parts.append("Higher timeframe confirms")
            else:
                description_parts.append("Higher timeframe conflicts")
        
        # ADX strength
        if trend_context.adx_value >= 25:
            description_parts.append(f"Strong trend (ADX: {trend_context.adx_value:.1f})")
        else:
            description_parts.append(f"Weak trend (ADX: {trend_context.adx_value:.1f})")
        
        # Filter results
        passed_count = len(filter_results.passed_filters)
        failed_count = len(filter_results.failed_filters)
        
        if passed_count > failed_count:
            description_parts.append(f"Most filters passed ({passed_count}/{passed_count + failed_count})")
        else:
            description_parts.append(f"Multiple filters failed ({failed_count}/{passed_count + failed_count})")
        
        return "; ".join(description_parts)
    
    def get_trend_context_summary(self, results: List[EnhancedPatternResult]) -> Dict[str, Any]:
        """Get summary of trend context across all results"""
        if not results:
            return {'message': 'No results to analyze'}
        
        summary = {
            'total_patterns': len(results),
            'trend_aligned_patterns': len([r for r in results if r.trend_alignment]),
            'alignment_rate': len([r for r in results if r.trend_alignment]) / len(results),
            'average_confidence': np.mean([r.confidence for r in results]),
            'trend_directions': {},
            'trend_strengths': {},
            'adx_distribution': {},
            'timeframe_distribution': {},
            'filter_performance': {
                'passed_filters': [],
                'failed_filters': [],
                'warnings': []
            }
        }
        
        # Count trend directions
        for result in results:
            direction = result.trend_context.trend_direction
            summary['trend_directions'][direction] = summary['trend_directions'].get(direction, 0) + 1
            
            strength = result.trend_context.trend_strength
            summary['trend_strengths'][strength] = summary['trend_strengths'].get(strength, 0) + 1
            
            timeframe = result.timeframe
            summary['timeframe_distribution'][timeframe] = summary['timeframe_distribution'].get(timeframe, 0) + 1
        
        # ADX distribution
        adx_values = [r.trend_context.adx_value for r in results]
        summary['adx_distribution'] = {
            'mean': np.mean(adx_values),
            'median': np.median(adx_values),
            'min': np.min(adx_values),
            'max': np.max(adx_values)
        }
        
        # Filter performance
        for result in results:
            if result.additional_info and 'filter_results' in result.additional_info:
                filter_results = result.additional_info['filter_results']
                summary['filter_performance']['passed_filters'].extend(filter_results.get('passed_filters', []))
                summary['filter_performance']['failed_filters'].extend(filter_results.get('failed_filters', []))
                summary['filter_performance']['warnings'].extend(filter_results.get('warnings', []))
        
        return summary
    
    # Helper methods for technical analysis
    def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate ADX"""
        try:
            high = df['high'].values
            low = df['low'].values
            close = df['close'].values
            
            tr1 = high - low
            tr2 = np.abs(high - np.roll(close, 1))
            tr3 = np.abs(low - np.roll(close, 1))
            tr = np.maximum(tr1, np.maximum(tr2, tr3))
            
            dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                              np.maximum(high - np.roll(high, 1), 0), 0)
            dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                               np.maximum(np.roll(low, 1) - low, 0), 0)
            
            tr_smooth = pd.Series(tr).rolling(period).mean().iloc[-1]
            dm_plus_smooth = pd.Series(dm_plus).rolling(period).mean().iloc[-1]
            dm_minus_smooth = pd.Series(dm_minus).rolling(period).mean().iloc[-1]
            
            di_plus = (dm_plus_smooth / tr_smooth) * 100 if tr_smooth > 0 else 0
            di_minus = (dm_minus_smooth / tr_smooth) * 100 if tr_smooth > 0 else 0
            
            dx = (abs(di_plus - di_minus) / (di_plus + di_minus)) * 100 if (di_plus + di_minus) > 0 else 0
            adx = pd.Series([dx]).rolling(period).mean().iloc[-1]
            
            return adx if not np.isnan(adx) else 0.0
        except Exception as e:
            logger.error(f"Error calculating ADX: {e}")
            return 0.0
    
    def _calculate_rsi(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate RSI"""
        try:
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi.iloc[-1] if not np.isnan(rsi.iloc[-1]) else 50.0
        except Exception as e:
            logger.error(f"Error calculating RSI: {e}")
            return 50.0
    
    def _calculate_macd_signal(self, df: pd.DataFrame) -> str:
        """Calculate MACD signal"""
        try:
            ema_12 = df['close'].ewm(span=12).mean()
            ema_26 = df['close'].ewm(span=26).mean()
            macd = ema_12 - ema_26
            signal = macd.ewm(span=9).mean()
            
            current_macd = macd.iloc[-1]
            current_signal = signal.iloc[-1]
            
            if current_macd > current_signal:
                return "bullish"
            elif current_macd < current_signal:
                return "bearish"
            else:
                return "neutral"
        except Exception as e:
            logger.error(f"Error calculating MACD: {e}")
            return "neutral"
    
    def _calculate_hull_ma(self, df: pd.DataFrame, period: int) -> float:
        """Calculate Hull MA"""
        try:
            wma_half = df['close'].rolling(period // 2).apply(
                lambda x: np.sum(x * np.arange(1, len(x) + 1)) / np.sum(np.arange(1, len(x) + 1))
            )
            wma_full = df['close'].rolling(period).apply(
                lambda x: np.sum(x * np.arange(1, len(x) + 1)) / np.sum(np.arange(1, len(x) + 1))
            )
            
            hull_raw = 2 * wma_half - wma_full
            hull_ma = hull_raw.rolling(int(np.sqrt(period))).apply(
                lambda x: np.sum(x * np.arange(1, len(x) + 1)) / np.sum(np.arange(1, len(x) + 1))
            )
            
            return hull_ma.iloc[-1] if not np.isnan(hull_ma.iloc[-1]) else df['close'].iloc[-1]
        except Exception as e:
            logger.error(f"Error calculating Hull MA: {e}")
            return df['close'].iloc[-1]
    
    def _determine_trend_direction(self, price: float, ema_20: float, ema_50: float, ema_200: float, adx: float) -> str:
        """Determine trend direction"""
        if price > ema_20 > ema_50 > ema_200 and adx > 25:
            return "bullish"
        elif price < ema_20 < ema_50 < ema_200 and adx > 25:
            return "bearish"
        else:
            return "neutral"
    
    def _determine_trend_strength(self, adx: float) -> str:
        """Determine trend strength"""
        if adx >= 40:
            return "extreme"
        elif adx >= 30:
            return "strong"
        elif adx >= 25:
            return "moderate"
        else:
            return "weak"
    
    def _check_ema_alignment(self, ema_20: float, ema_50: float, ema_200: float, direction: str) -> bool:
        """Check EMA alignment"""
        if direction == "bullish":
            return ema_20 > ema_50 > ema_200
        elif direction == "bearish":
            return ema_20 < ema_50 < ema_200
        else:
            return True
    
    def _calculate_trend_confidence(self, direction: str, strength: str, ema_aligned: bool, adx: float, rsi: float) -> float:
        """Calculate trend confidence"""
        confidence = 0.5
        
        if direction in ["bullish", "bearish"]:
            confidence += 0.2
        
        if strength in ["strong", "extreme"]:
            confidence += 0.15
        
        if ema_aligned:
            confidence += 0.1
        
        if adx > 30:
            confidence += 0.05
        
        if 30 <= rsi <= 70:
            confidence += 0.05
        
        return min(1.0, confidence)
    
    def _generate_trend_description(self, direction: str, strength: str, ema_aligned: bool, adx: float) -> str:
        """Generate trend description"""
        desc = f"{strength.title()} {direction} trend (ADX: {adx:.1f})"
        if ema_aligned:
            desc += " with aligned EMAs"
        return desc
    
    def _get_higher_timeframe(self, timeframe: str) -> str:
        """Get higher timeframe"""
        hierarchy = {
            '1m': '5m', '5m': '15m', '15m': '30m', '30m': '1h',
            '1h': '4h', '4h': '1d', '1d': '1w'
        }
        return hierarchy.get(timeframe, timeframe)
    
    def _get_lower_timeframe(self, timeframe: str) -> str:
        """Get lower timeframe"""
        hierarchy = {
            '1w': '1d', '1d': '4h', '4h': '1h', '1h': '30m',
            '30m': '15m', '15m': '5m', '5m': '1m'
        }
        return hierarchy.get(timeframe, timeframe)
    
    def _get_default_trend_context(self) -> Dict[str, Any]:
        """Get default trend context"""
        return {
            'trend_direction': 'neutral',
            'trend_strength': 'weak',
            'adx_value': 0.0,
            'ema_alignment': False,
            'ema_20': 0.0,
            'ema_50': 0.0,
            'ema_200': 0.0,
            'rsi': 50.0,
            'macd_signal': 'neutral',
            'hull_fast': 0.0,
            'hull_medium': 0.0,
            'hull_slow': 0.0,
            'trend_confidence': 0.5,
            'trend_description': 'Insufficient data for trend analysis'
        }
    
    def _get_default_enhanced_result(self, pattern_name: str, timeframe: str, base_confidence: float) -> EnhancedPatternResult:
        """Get default enhanced result"""
        return EnhancedPatternResult(
            pattern=pattern_name,
            timeframe=timeframe,
            confidence=base_confidence,
            trend_alignment=False,
            higher_tf_trend="unknown",
            adx=0.0,
            ema_alignment=False,
            trend_context=TrendContextData(
                trend_direction="neutral",
                trend_strength="weak",
                adx_value=0.0,
                ema_alignment=False,
                ema_20=0.0,
                ema_50=0.0,
                ema_200=0.0,
                rsi=50.0,
                macd_signal="neutral",
                hull_ma_fast=0.0,
                hull_ma_medium=0.0,
                hull_ma_slow=0.0,
                trend_confidence=0.5,
                trend_description="Error in trend analysis"
            ),
            timestamp=datetime.now(),
            additional_info={'error': 'Failed to create enhanced result'}
        )
