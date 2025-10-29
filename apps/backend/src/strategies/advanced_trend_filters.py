#!/usr/bin/env python3
"""
Advanced Trend Filters for AlphaPulse
Phase 2: EMA Alignment, ADX Thresholds, Hull MA, and Comprehensive Trend Filtering
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
from datetime import datetime

logger = logging.getLogger(__name__)

class TrendFilterType(Enum):
    """Types of trend filters"""
    EMA_ALIGNMENT = "ema_alignment"
    ADX_STRENGTH = "adx_strength"
    HULL_MA = "hull_ma"
    TREND_CONSISTENCY = "trend_consistency"
    VOLATILITY_FILTER = "volatility_filter"

class FilterResult(Enum):
    """Filter result enumeration"""
    PASS = "pass"
    FAIL = "fail"
    WARNING = "warning"

@dataclass
class TrendFilterResult:
    """Result of trend filter application"""
    filter_type: TrendFilterType
    result: FilterResult
    score: float  # 0.0 to 1.0
    description: str
    details: Dict[str, Any]

@dataclass
class ComprehensiveTrendFilter:
    """Comprehensive trend filtering system"""
    ema_alignment: bool
    adx_strength: bool
    hull_ma_alignment: bool
    trend_consistency: bool
    volatility_acceptable: bool
    overall_score: float
    passed_filters: List[str]
    failed_filters: List[str]
    warnings: List[str]

class AdvancedTrendFilters:
    """
    Advanced trend filtering system with EMA, ADX, Hull MA, and comprehensive checks
    """
    
    def __init__(self):
        # EMA alignment thresholds
        self.ema_thresholds = {
            'bullish': {
                'ema_20_vs_50': 0.001,  # EMA20 > EMA50 by 0.1%
                'ema_50_vs_200': 0.001,  # EMA50 > EMA200 by 0.1%
                'price_vs_ema20': 0.005   # Price > EMA20 by 0.5%
            },
            'bearish': {
                'ema_20_vs_50': -0.001,  # EMA20 < EMA50 by 0.1%
                'ema_50_vs_200': -0.001,  # EMA50 < EMA200 by 0.1%
                'price_vs_ema20': -0.005  # Price < EMA20 by 0.5%
            }
        }
        
        # ADX strength thresholds
        self.adx_thresholds = {
            'weak': 20,
            'moderate': 25,
            'strong': 30,
            'extreme': 40
        }
        
        # Hull MA parameters
        self.hull_periods = {
            'fast': 9,
            'medium': 18,
            'slow': 36
        }
        
        # Volatility thresholds
        self.volatility_thresholds = {
            'low': 0.01,    # 1% daily volatility
            'medium': 0.025, # 2.5% daily volatility
            'high': 0.05    # 5% daily volatility
        }
        
        logger.info("🚀 Advanced Trend Filters initialized")
    
    def apply_comprehensive_trend_filters(
        self, 
        df: pd.DataFrame, 
        pattern_direction: str,
        timeframe: str = "1h"
    ) -> ComprehensiveTrendFilter:
        """
        Apply all trend filters comprehensively
        
        Args:
            df: DataFrame with OHLCV data
            pattern_direction: Direction of pattern ("bullish", "bearish", "neutral")
            timeframe: Timeframe being analyzed
            
        Returns:
            ComprehensiveTrendFilter with all filter results
        """
        if len(df) < 50:
            return self._get_default_filter_result()
        
        try:
            # Apply individual filters
            ema_result = self._check_ema_alignment(df, pattern_direction)
            adx_result = self._check_adx_strength(df)
            hull_result = self._check_hull_ma_alignment(df, pattern_direction)
            consistency_result = self._check_trend_consistency(df, pattern_direction)
            volatility_result = self._check_volatility_filter(df, timeframe)
            
            # Collect results
            passed_filters = []
            failed_filters = []
            warnings = []
            
            if ema_result.result == FilterResult.PASS:
                passed_filters.append("EMA Alignment")
            elif ema_result.result == FilterResult.FAIL:
                failed_filters.append("EMA Alignment")
            else:
                warnings.append("EMA Alignment")
            
            if adx_result.result == FilterResult.PASS:
                passed_filters.append("ADX Strength")
            elif adx_result.result == FilterResult.FAIL:
                failed_filters.append("ADX Strength")
            else:
                warnings.append("ADX Strength")
            
            if hull_result.result == FilterResult.PASS:
                passed_filters.append("Hull MA")
            elif hull_result.result == FilterResult.FAIL:
                failed_filters.append("Hull MA")
            else:
                warnings.append("Hull MA")
            
            if consistency_result.result == FilterResult.PASS:
                passed_filters.append("Trend Consistency")
            elif consistency_result.result == FilterResult.FAIL:
                failed_filters.append("Trend Consistency")
            else:
                warnings.append("Trend Consistency")
            
            if volatility_result.result == FilterResult.PASS:
                passed_filters.append("Volatility")
            elif volatility_result.result == FilterResult.FAIL:
                failed_filters.append("Volatility")
            else:
                warnings.append("Volatility")
            
            # Calculate overall score
            scores = [
                ema_result.score,
                adx_result.score,
                hull_result.score,
                consistency_result.score,
                volatility_result.score
            ]
            overall_score = np.mean(scores)
            
            return ComprehensiveTrendFilter(
                ema_alignment=ema_result.result == FilterResult.PASS,
                adx_strength=adx_result.result == FilterResult.PASS,
                hull_ma_alignment=hull_result.result == FilterResult.PASS,
                trend_consistency=consistency_result.result == FilterResult.PASS,
                volatility_acceptable=volatility_result.result == FilterResult.PASS,
                overall_score=overall_score,
                passed_filters=passed_filters,
                failed_filters=failed_filters,
                warnings=warnings
            )
            
        except Exception as e:
            logger.error(f"Error applying comprehensive trend filters: {e}")
            return self._get_default_filter_result()
    
    def _check_ema_alignment(self, df: pd.DataFrame, pattern_direction: str) -> TrendFilterResult:
        """
        Check EMA alignment requirements
        
        Args:
            df: DataFrame with OHLCV data
            pattern_direction: Pattern direction ("bullish", "bearish", "neutral")
            
        Returns:
            TrendFilterResult with EMA alignment check
        """
        try:
            # Calculate EMAs
            ema_20 = df['close'].ewm(span=20).mean().iloc[-1]
            ema_50 = df['close'].ewm(span=50).mean().iloc[-1]
            ema_200 = df['close'].ewm(span=200).mean().iloc[-1] if len(df) >= 200 else ema_50
            current_price = df['close'].iloc[-1]
            
            # Calculate ratios
            ema_20_vs_50 = (ema_20 - ema_50) / ema_50
            ema_50_vs_200 = (ema_50 - ema_200) / ema_200
            price_vs_ema20 = (current_price - ema_20) / ema_20
            
            # Get thresholds
            thresholds = self.ema_thresholds.get(pattern_direction, self.ema_thresholds['bullish'])
            
            # Check alignment
            ema_20_aligned = ema_20_vs_50 >= thresholds['ema_20_vs_50'] if pattern_direction == "bullish" else ema_20_vs_50 <= thresholds['ema_20_vs_50']
            ema_50_aligned = ema_50_vs_200 >= thresholds['ema_50_vs_200'] if pattern_direction == "bullish" else ema_50_vs_200 <= thresholds['ema_50_vs_200']
            price_aligned = price_vs_ema20 >= thresholds['price_vs_ema20'] if pattern_direction == "bullish" else price_vs_ema20 <= thresholds['price_vs_ema20']
            
            # Calculate score
            alignment_count = sum([ema_20_aligned, ema_50_aligned, price_aligned])
            score = alignment_count / 3.0
            
            # Determine result
            if score >= 0.8:
                result = FilterResult.PASS
                description = f"Strong EMA alignment for {pattern_direction} pattern"
            elif score >= 0.5:
                result = FilterResult.WARNING
                description = f"Moderate EMA alignment for {pattern_direction} pattern"
            else:
                result = FilterResult.FAIL
                description = f"Weak EMA alignment for {pattern_direction} pattern"
            
            return TrendFilterResult(
                filter_type=TrendFilterType.EMA_ALIGNMENT,
                result=result,
                score=score,
                description=description,
                details={
                    'ema_20_vs_50': ema_20_vs_50,
                    'ema_50_vs_200': ema_50_vs_200,
                    'price_vs_ema20': price_vs_ema20,
                    'ema_20_aligned': ema_20_aligned,
                    'ema_50_aligned': ema_50_aligned,
                    'price_aligned': price_aligned
                }
            )
            
        except Exception as e:
            logger.error(f"Error checking EMA alignment: {e}")
            return TrendFilterResult(
                filter_type=TrendFilterType.EMA_ALIGNMENT,
                result=FilterResult.FAIL,
                score=0.0,
                description="Error in EMA alignment check",
                details={}
            )
    
    def _check_adx_strength(self, df: pd.DataFrame) -> TrendFilterResult:
        """
        Check ADX strength requirements
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            TrendFilterResult with ADX strength check
        """
        try:
            # Calculate ADX
            adx_value = self._calculate_adx(df)
            
            # Determine strength level
            if adx_value >= self.adx_thresholds['extreme']:
                strength_level = "extreme"
                score = 1.0
                result = FilterResult.PASS
                description = f"Extreme trend strength (ADX: {adx_value:.1f})"
            elif adx_value >= self.adx_thresholds['strong']:
                strength_level = "strong"
                score = 0.9
                result = FilterResult.PASS
                description = f"Strong trend strength (ADX: {adx_value:.1f})"
            elif adx_value >= self.adx_thresholds['moderate']:
                strength_level = "moderate"
                score = 0.7
                result = FilterResult.WARNING
                description = f"Moderate trend strength (ADX: {adx_value:.1f})"
            elif adx_value >= self.adx_thresholds['weak']:
                strength_level = "weak"
                score = 0.5
                result = FilterResult.WARNING
                description = f"Weak trend strength (ADX: {adx_value:.1f})"
            else:
                strength_level = "very_weak"
                score = 0.2
                result = FilterResult.FAIL
                description = f"Very weak trend strength (ADX: {adx_value:.1f})"
            
            return TrendFilterResult(
                filter_type=TrendFilterType.ADX_STRENGTH,
                result=result,
                score=score,
                description=description,
                details={
                    'adx_value': adx_value,
                    'strength_level': strength_level,
                    'thresholds': self.adx_thresholds
                }
            )
            
        except Exception as e:
            logger.error(f"Error checking ADX strength: {e}")
            return TrendFilterResult(
                filter_type=TrendFilterType.ADX_STRENGTH,
                result=FilterResult.FAIL,
                score=0.0,
                description="Error in ADX strength check",
                details={}
            )
    
    def _check_hull_ma_alignment(self, df: pd.DataFrame, pattern_direction: str) -> TrendFilterResult:
        """
        Check Hull MA alignment for faster reaction
        
        Args:
            df: DataFrame with OHLCV data
            pattern_direction: Pattern direction
            
        Returns:
            TrendFilterResult with Hull MA alignment check
        """
        try:
            # Calculate Hull MAs
            hull_fast = self._calculate_hull_ma(df, self.hull_periods['fast'])
            hull_medium = self._calculate_hull_ma(df, self.hull_periods['medium'])
            hull_slow = self._calculate_hull_ma(df, self.hull_periods['slow'])
            current_price = df['close'].iloc[-1]
            
            # Check alignment
            if pattern_direction == "bullish":
                fast_above_medium = hull_fast > hull_medium
                medium_above_slow = hull_medium > hull_slow
                price_above_fast = current_price > hull_fast
                alignment_count = sum([fast_above_medium, medium_above_slow, price_above_fast])
            elif pattern_direction == "bearish":
                fast_below_medium = hull_fast < hull_medium
                medium_below_slow = hull_medium < hull_slow
                price_below_fast = current_price < hull_fast
                alignment_count = sum([fast_below_medium, medium_below_slow, price_below_fast])
            else:
                # Neutral - check for any alignment
                bullish_alignment = sum([hull_fast > hull_medium, hull_medium > hull_slow, current_price > hull_fast])
                bearish_alignment = sum([hull_fast < hull_medium, hull_medium < hull_slow, current_price < hull_fast])
                alignment_count = max(bullish_alignment, bearish_alignment)
            
            score = alignment_count / 3.0
            
            # Determine result
            if score >= 0.8:
                result = FilterResult.PASS
                description = f"Strong Hull MA alignment for {pattern_direction} pattern"
            elif score >= 0.5:
                result = FilterResult.WARNING
                description = f"Moderate Hull MA alignment for {pattern_direction} pattern"
            else:
                result = FilterResult.FAIL
                description = f"Weak Hull MA alignment for {pattern_direction} pattern"
            
            return TrendFilterResult(
                filter_type=TrendFilterType.HULL_MA,
                result=result,
                score=score,
                description=description,
                details={
                    'hull_fast': hull_fast,
                    'hull_medium': hull_medium,
                    'hull_slow': hull_slow,
                    'current_price': current_price,
                    'alignment_count': alignment_count
                }
            )
            
        except Exception as e:
            logger.error(f"Error checking Hull MA alignment: {e}")
            return TrendFilterResult(
                filter_type=TrendFilterType.HULL_MA,
                result=FilterResult.FAIL,
                score=0.0,
                description="Error in Hull MA alignment check",
                details={}
            )
    
    def _check_trend_consistency(self, df: pd.DataFrame, pattern_direction: str) -> TrendFilterResult:
        """
        Check trend consistency across multiple indicators
        
        Args:
            df: DataFrame with OHLCV data
            pattern_direction: Pattern direction
            
        Returns:
            TrendFilterResult with trend consistency check
        """
        try:
            # Calculate multiple trend indicators
            ema_20 = df['close'].ewm(span=20).mean().iloc[-1]
            ema_50 = df['close'].ewm(span=50).mean().iloc[-1]
            sma_20 = df['close'].rolling(20).mean().iloc[-1]
            current_price = df['close'].iloc[-1]
            
            # Calculate momentum indicators
            rsi = self._calculate_rsi(df)
            macd_signal = self._calculate_macd_signal(df)
            
            # Check consistency
            bullish_indicators = 0
            bearish_indicators = 0
            
            # Price vs EMAs
            if current_price > ema_20:
                bullish_indicators += 1
            else:
                bearish_indicators += 1
            
            if current_price > ema_50:
                bullish_indicators += 1
            else:
                bearish_indicators += 1
            
            if ema_20 > ema_50:
                bullish_indicators += 1
            else:
                bearish_indicators += 1
            
            # Price vs SMA
            if current_price > sma_20:
                bullish_indicators += 1
            else:
                bearish_indicators += 1
            
            # RSI
            if rsi > 50:
                bullish_indicators += 1
            else:
                bearish_indicators += 1
            
            # MACD
            if macd_signal == "bullish":
                bullish_indicators += 1
            elif macd_signal == "bearish":
                bearish_indicators += 1
            
            # Calculate consistency score
            total_indicators = 6
            if pattern_direction == "bullish":
                consistency_score = bullish_indicators / total_indicators
            elif pattern_direction == "bearish":
                consistency_score = bearish_indicators / total_indicators
            else:
                # Neutral - use the higher score
                consistency_score = max(bullish_indicators, bearish_indicators) / total_indicators
            
            # Determine result
            if consistency_score >= 0.8:
                result = FilterResult.PASS
                description = f"Strong trend consistency for {pattern_direction} pattern"
            elif consistency_score >= 0.6:
                result = FilterResult.WARNING
                description = f"Moderate trend consistency for {pattern_direction} pattern"
            else:
                result = FilterResult.FAIL
                description = f"Weak trend consistency for {pattern_direction} pattern"
            
            return TrendFilterResult(
                filter_type=TrendFilterType.TREND_CONSISTENCY,
                result=result,
                score=consistency_score,
                description=description,
                details={
                    'bullish_indicators': bullish_indicators,
                    'bearish_indicators': bearish_indicators,
                    'total_indicators': total_indicators,
                    'rsi': rsi,
                    'macd_signal': macd_signal
                }
            )
            
        except Exception as e:
            logger.error(f"Error checking trend consistency: {e}")
            return TrendFilterResult(
                filter_type=TrendFilterType.TREND_CONSISTENCY,
                result=FilterResult.FAIL,
                score=0.0,
                description="Error in trend consistency check",
                details={}
            )
    
    def _check_volatility_filter(self, df: pd.DataFrame, timeframe: str) -> TrendFilterResult:
        """
        Check if volatility is acceptable for the timeframe
        
        Args:
            df: DataFrame with OHLCV data
            timeframe: Timeframe being analyzed
            
        Returns:
            TrendFilterResult with volatility check
        """
        try:
            # Calculate volatility (standard deviation of returns)
            returns = df['close'].pct_change().dropna()
            volatility = returns.std()
            
            # Adjust thresholds based on timeframe
            timeframe_multiplier = self._get_timeframe_multiplier(timeframe)
            adjusted_thresholds = {
                'low': self.volatility_thresholds['low'] * timeframe_multiplier,
                'medium': self.volatility_thresholds['medium'] * timeframe_multiplier,
                'high': self.volatility_thresholds['high'] * timeframe_multiplier
            }
            
            # Determine volatility level
            if volatility <= adjusted_thresholds['low']:
                volatility_level = "low"
                score = 0.8
                result = FilterResult.PASS
                description = f"Low volatility ({volatility:.3f}) suitable for {timeframe}"
            elif volatility <= adjusted_thresholds['medium']:
                volatility_level = "medium"
                score = 1.0
                result = FilterResult.PASS
                description = f"Medium volatility ({volatility:.3f}) ideal for {timeframe}"
            elif volatility <= adjusted_thresholds['high']:
                volatility_level = "high"
                score = 0.6
                result = FilterResult.WARNING
                description = f"High volatility ({volatility:.3f}) may affect {timeframe} signals"
            else:
                volatility_level = "extreme"
                score = 0.3
                result = FilterResult.FAIL
                description = f"Extreme volatility ({volatility:.3f}) unsuitable for {timeframe}"
            
            return TrendFilterResult(
                filter_type=TrendFilterType.VOLATILITY_FILTER,
                result=result,
                score=score,
                description=description,
                details={
                    'volatility': volatility,
                    'volatility_level': volatility_level,
                    'adjusted_thresholds': adjusted_thresholds,
                    'timeframe_multiplier': timeframe_multiplier
                }
            )
            
        except Exception as e:
            logger.error(f"Error checking volatility filter: {e}")
            return TrendFilterResult(
                filter_type=TrendFilterType.VOLATILITY_FILTER,
                result=FilterResult.FAIL,
                score=0.0,
                description="Error in volatility filter check",
                details={}
            )
    
    def _calculate_hull_ma(self, df: pd.DataFrame, period: int) -> float:
        """Calculate Hull Moving Average"""
        try:
            # Calculate WMA
            wma_half = df['close'].rolling(period // 2).apply(
                lambda x: np.sum(x * np.arange(1, len(x) + 1)) / np.sum(np.arange(1, len(x) + 1))
            )
            wma_full = df['close'].rolling(period).apply(
                lambda x: np.sum(x * np.arange(1, len(x) + 1)) / np.sum(np.arange(1, len(x) + 1))
            )
            
            # Calculate Hull MA
            hull_raw = 2 * wma_half - wma_full
            hull_ma = hull_raw.rolling(int(np.sqrt(period))).apply(
                lambda x: np.sum(x * np.arange(1, len(x) + 1)) / np.sum(np.arange(1, len(x) + 1))
            )
            
            return hull_ma.iloc[-1] if not np.isnan(hull_ma.iloc[-1]) else df['close'].iloc[-1]
            
        except Exception as e:
            logger.error(f"Error calculating Hull MA: {e}")
            return df['close'].iloc[-1]
    
    def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate ADX (Average Directional Index)"""
        try:
            high = df['high'].values
            low = df['low'].values
            close = df['close'].values
            
            # Calculate True Range
            tr1 = high - low
            tr2 = np.abs(high - np.roll(close, 1))
            tr3 = np.abs(low - np.roll(close, 1))
            tr = np.maximum(tr1, np.maximum(tr2, tr3))
            
            # Calculate Directional Movement
            dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                              np.maximum(high - np.roll(high, 1), 0), 0)
            dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                               np.maximum(np.roll(low, 1) - low, 0), 0)
            
            # Smooth the values
            tr_smooth = pd.Series(tr).rolling(period).mean().iloc[-1]
            dm_plus_smooth = pd.Series(dm_plus).rolling(period).mean().iloc[-1]
            dm_minus_smooth = pd.Series(dm_minus).rolling(period).mean().iloc[-1]
            
            # Calculate DI+ and DI-
            di_plus = (dm_plus_smooth / tr_smooth) * 100 if tr_smooth > 0 else 0
            di_minus = (dm_minus_smooth / tr_smooth) * 100 if tr_smooth > 0 else 0
            
            # Calculate DX and ADX
            dx = (abs(di_plus - di_minus) / (di_plus + di_minus)) * 100 if (di_plus + di_minus) > 0 else 0
            adx = pd.Series([dx]).rolling(period).mean().iloc[-1]
            
            return adx if not np.isnan(adx) else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating ADX: {e}")
            return 0.0
    
    def _calculate_rsi(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate RSI (Relative Strength Index)"""
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
    
    def _get_timeframe_multiplier(self, timeframe: str) -> float:
        """Get volatility multiplier based on timeframe"""
        multipliers = {
            '1m': 0.1,
            '5m': 0.2,
            '15m': 0.5,
            '30m': 0.7,
            '1h': 1.0,
            '4h': 1.5,
            '1d': 2.0
        }
        return multipliers.get(timeframe, 1.0)
    
    def _get_default_filter_result(self) -> ComprehensiveTrendFilter:
        """Return default filter result when analysis fails"""
        return ComprehensiveTrendFilter(
            ema_alignment=False,
            adx_strength=False,
            hull_ma_alignment=False,
            trend_consistency=False,
            volatility_acceptable=False,
            overall_score=0.0,
            passed_filters=[],
            failed_filters=["All filters failed due to insufficient data"],
            warnings=[]
        )
