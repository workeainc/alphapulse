#!/usr/bin/env python3
"""
Pattern Analysis Module for AlphaPulse
Detects candlestick patterns in real-time and stores them for AI analysis
"""

import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import pandas as pd
import numpy as np
from dataclasses import dataclass
import talib

from src.data.storage import DataStorage
from ..src.strategies.pattern_detector import CandlestickPatternDetector

logger = logging.getLogger(__name__)

@dataclass
class DetectedPattern:
    """Detected candlestick pattern with metadata"""
    pattern_name: str
    symbol: str
    timeframe: str
    timestamp: datetime
    confidence: float
    strength: str  # 'weak', 'medium', 'strong'
    price_level: float
    volume_confirmation: bool
    trend_alignment: str  # 'bullish', 'bearish', 'neutral'
    additional_data: Dict

class PatternAnalyzer:
    """
    Real-time candlestick pattern detection and analysis
    Detects common formations and stores them for AI analysis
    """
    
    def __init__(self, config: Dict = None):
        """Initialize pattern analyzer"""
        self.config = config or {}
        
        # Initialize components
        self.pattern_detector = CandlestickPatternDetector()
        self.storage = DataStorage(self.config.get('storage_path', 'data'))
        
        # Pattern detection settings
        self.patterns_to_detect = self.config.get('patterns_to_detect', [
            'doji', 'engulfing', 'hammer', 'shooting_star', 'morning_star', 
            'evening_star', 'three_white_soldiers', 'three_black_crows',
            'tweezers_top', 'tweezers_bottom', 'harami', 'piercing_line',
            'dark_cloud_cover', 'hanging_man', 'inverted_hammer'
        ])
        
        # Confidence thresholds
        self.confidence_thresholds = {
            'weak': 0.6,
            'medium': 0.75,
            'strong': 0.9
        }
        
        # Volume confirmation settings
        self.volume_confirmation_threshold = self.config.get('volume_confirmation_threshold', 1.5)
        self.min_volume_ratio = self.config.get('min_volume_ratio', 0.8)
        
        # Performance tracking
        self.stats = {
            'patterns_detected': 0,
            'patterns_stored': 0,
            'analysis_time_avg': 0.0,
            'last_update': None
        }
        
        # Callbacks
        self.pattern_callbacks = []
        
        logger.info("🔍 Pattern Analyzer initialized")
    
    async def analyze_candlestick(self, candlestick_data: Dict) -> Optional[DetectedPattern]:
        """
        Analyze a single candlestick for pattern detection
        
        Args:
            candlestick_data: Candlestick data with OHLCV and indicators
            
        Returns:
            DetectedPattern if found, None otherwise
        """
        try:
            start_time = datetime.now()
            
            # Extract required data
            symbol = candlestick_data['symbol']
            timeframe = candlestick_data['timeframe']
            timestamp = candlestick_data['timestamp']
            
            # Get historical data for pattern analysis
            historical_data = await self._get_historical_data(symbol, timeframe, limit=50)
            if len(historical_data) < 3:  # Need at least 3 candles for patterns
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(historical_data)
            
            # Detect patterns
            detected_patterns = self._detect_patterns(df)
            
            if not detected_patterns:
                return None
            
            # Get the strongest pattern
            strongest_pattern = max(detected_patterns, key=lambda x: x['confidence'])
            
            # Create pattern object
            pattern = DetectedPattern(
                pattern_name=strongest_pattern['name'],
                symbol=symbol,
                timeframe=timeframe,
                timestamp=timestamp,
                confidence=strongest_pattern['confidence'],
                strength=self._get_strength_level(strongest_pattern['confidence']),
                price_level=candlestick_data['close'],
                volume_confirmation=self._check_volume_confirmation(df, strongest_pattern),
                trend_alignment=self._determine_trend_alignment(df, strongest_pattern),
                additional_data=strongest_pattern.get('metadata', {})
            )
            
            # Store pattern in database
            await self._store_pattern(pattern)
            
            # Notify callbacks
            await self._notify_pattern_callbacks(pattern)
            
            # Update stats
            self._update_stats(start_time)
            
            return pattern
            
        except Exception as e:
            logger.error(f"❌ Error analyzing candlestick: {e}")
            return None
    
    async def analyze_multi_timeframe(self, symbol: str, base_timeframe: str = '1m') -> List[DetectedPattern]:
        """
        Analyze patterns across multiple timeframes for confirmation
        
        Args:
            symbol: Trading symbol
            base_timeframe: Base timeframe to start analysis from
            
        Returns:
            List of confirmed patterns across timeframes
        """
        try:
            # Define timeframe hierarchy
            timeframes = ['1m', '5m', '15m', '1h', '4h', '1d', '1W']
            
            # Find base timeframe index
            try:
                base_index = timeframes.index(base_timeframe)
            except ValueError:
                base_index = 0  # Default to 1m
            
            # Get patterns for each timeframe
            all_patterns = []
            
            for i, timeframe in enumerate(timeframes[base_index:], base_index):
                try:
                    # Get historical data for this timeframe
                    historical_data = await self._get_historical_data(symbol, timeframe, limit=100)
                    if len(historical_data) < 20:  # Need more data for TA-Lib
                        continue
                    
                    # Convert to DataFrame
                    df = pd.DataFrame(historical_data)
                    
                    # Detect patterns
                    detected_patterns = self._detect_patterns(df)
                    
                    # Add timeframe information
                    for pattern in detected_patterns:
                        pattern['timeframe'] = timeframe
                        pattern['timeframe_weight'] = self._get_timeframe_weight(i)
                    
                    all_patterns.extend(detected_patterns)
                    
                except Exception as e:
                    logger.warning(f"⚠️ Error analyzing {timeframe}: {e}")
                    continue
            
            # Group patterns by type and calculate multi-timeframe confidence
            confirmed_patterns = self._calculate_multi_timeframe_confidence(all_patterns)
            
            # Convert to DetectedPattern objects
            result_patterns = []
            for pattern_data in confirmed_patterns:
                if pattern_data['final_confidence'] > 0.6:  # Only high-confidence patterns
                    pattern = DetectedPattern(
                        pattern_name=pattern_data['name'],
                        symbol=symbol,
                        timeframe=pattern_data['primary_timeframe'],
                        timestamp=datetime.now(),
                        confidence=pattern_data['final_confidence'],
                        strength=self._get_strength_level(pattern_data['final_confidence']),
                        price_level=pattern_data.get('price_level', 0.0),
                        volume_confirmation=pattern_data.get('volume_confirmed', False),
                        trend_alignment=pattern_data.get('trend_alignment', 'neutral'),
                        additional_data={
                            'multi_timeframe_confirmed': True,
                            'confirming_timeframes': pattern_data['confirming_timeframes'],
                            'timeframe_weights': pattern_data['timeframe_weights']
                        }
                    )
                    result_patterns.append(pattern)
            
            return result_patterns
            
        except Exception as e:
            logger.error(f"❌ Error in multi-timeframe analysis: {e}")
            return []
    
    def _detect_patterns(self, df: pd.DataFrame) -> List[Dict]:
        """Detect candlestick patterns in the DataFrame"""
        try:
            patterns = []
            
            # Ensure we have enough data
            if len(df) < 3:
                return patterns
            
            # Get latest candles
            current = df.iloc[-1]
            previous = df.iloc[-2] if len(df) > 1 else None
            two_ago = df.iloc[-3] if len(df) > 2 else None
            
            # Single candlestick patterns
            if self._is_doji(current):
                patterns.append({
                    'name': 'doji',
                    'confidence': self._calculate_doji_confidence(current),
                    'metadata': {'body_size': self._get_body_size(current)}
                })
            
            if self._is_hammer(current):
                patterns.append({
                    'name': 'hammer',
                    'confidence': self._calculate_hammer_confidence(current),
                    'metadata': {'shadow_ratio': self._get_shadow_ratio(current)}
                })
            
            if self._is_shooting_star(current):
                patterns.append({
                    'name': 'shooting_star',
                    'confidence': self._calculate_shooting_star_confidence(current),
                    'metadata': {'shadow_ratio': self._get_shadow_ratio(current)}
                })
            
            # Two-candlestick patterns
            if previous is not None:
                if self._is_bullish_engulfing(previous, current):
                    patterns.append({
                        'name': 'bullish_engulfing',
                        'confidence': self._calculate_engulfing_confidence(previous, current),
                        'metadata': {'engulfing_ratio': self._get_engulfing_ratio(previous, current)}
                    })
                
                if self._is_bearish_engulfing(previous, current):
                    patterns.append({
                        'name': 'bearish_engulfing',
                        'confidence': self._calculate_engulfing_confidence(previous, current),
                        'metadata': {'engulfing_ratio': self._get_engulfing_ratio(previous, current)}
                    })
                
                if self._is_harami(previous, current):
                    patterns.append({
                        'name': 'harami',
                        'confidence': self._calculate_harami_confidence(previous, current),
                        'metadata': {'harami_ratio': self._get_harami_ratio(previous, current)}
                    })
            
            # Three-candlestick patterns
            if two_ago is not None and previous is not None:
                if self._is_morning_star(two_ago, previous, current):
                    patterns.append({
                        'name': 'morning_star',
                        'confidence': self._calculate_morning_star_confidence(two_ago, previous, current),
                        'metadata': {'star_formation': 'morning'}
                    })
                
                if self._is_evening_star(two_ago, previous, current):
                    patterns.append({
                        'name': 'evening_star',
                        'confidence': self._calculate_evening_star_confidence(two_ago, previous, current),
                        'metadata': {'star_formation': 'evening'}
                    })
            
            # Add TA-Lib pattern detection
            talib_patterns = self._detect_talib_patterns(df)
            patterns.extend(talib_patterns)
            
            return patterns
            
        except Exception as e:
            logger.error(f"❌ Error detecting patterns: {e}")
            return []
    
    def _is_doji(self, candle: pd.Series) -> bool:
        """Check if candle is a doji pattern"""
        try:
            body_size = abs(candle['close'] - candle['open'])
            total_range = candle['high'] - candle['low']
            
            if total_range == 0:
                return False
            
            body_ratio = body_size / total_range
            return body_ratio < 0.1  # Body is less than 10% of total range
            
        except Exception:
            return False
    
    def _is_hammer(self, candle: pd.Series) -> bool:
        """Check if candle is a hammer pattern"""
        try:
            body_size = abs(candle['close'] - candle['open'])
            upper_shadow = candle['high'] - max(candle['open'], candle['close'])
            lower_shadow = min(candle['open'], candle['close']) - candle['low']
            
            # Lower shadow should be at least 2x body size
            # Upper shadow should be small
            return (lower_shadow >= 2 * body_size and 
                   upper_shadow <= 0.1 * body_size and
                   body_size > 0)
            
        except Exception:
            return False
    
    def _is_shooting_star(self, candle: pd.Series) -> bool:
        """Check if candle is a shooting star pattern"""
        try:
            body_size = abs(candle['close'] - candle['open'])
            upper_shadow = candle['high'] - max(candle['open'], candle['close'])
            lower_shadow = min(candle['open'], candle['close']) - candle['low']
            
            # Upper shadow should be at least 2x body size
            # Lower shadow should be small
            return (upper_shadow >= 2 * body_size and 
                   lower_shadow <= 0.1 * body_size and
                   body_size > 0)
            
        except Exception:
            return False
    
    def _is_bullish_engulfing(self, previous: pd.Series, current: pd.Series) -> bool:
        """Check if current candle engulfs previous candle (bullish)"""
        try:
            prev_open = previous['open']
            prev_close = previous['close']
            curr_open = current['open']
            curr_close = current['close']
            
            # Previous candle should be bearish (red)
            # Current candle should be bullish (green)
            # Current candle should engulf previous candle
            return (prev_close < prev_open and  # Previous bearish
                   curr_close > curr_open and   # Current bullish
                   curr_open < prev_close and   # Current open below previous close
                   curr_close > prev_open)      # Current close above previous open
            
        except Exception:
            return False
    
    def _is_bearish_engulfing(self, previous: pd.Series, current: pd.Series) -> bool:
        """Check if current candle engulfs previous candle (bearish)"""
        try:
            prev_open = previous['open']
            prev_close = previous['close']
            curr_open = current['open']
            curr_close = current['close']
            
            # Previous candle should be bullish (green)
            # Current candle should be bearish (red)
            # Current candle should engulf previous candle
            return (prev_close > prev_open and  # Previous bullish
                   curr_close < curr_open and   # Current bearish
                   curr_open > prev_close and   # Current open above previous close
                   curr_close < prev_open)      # Current close below previous open
            
        except Exception:
            return False
    
    def _is_harami(self, previous: pd.Series, current: pd.Series) -> bool:
        """Check if current candle is contained within previous candle"""
        try:
            prev_open = previous['open']
            prev_close = previous['close']
            curr_open = current['open']
            curr_close = current['close']
            
            # Current candle should be contained within previous candle
            return (curr_open >= min(prev_open, prev_close) and
                   curr_close <= max(prev_open, prev_close))
            
        except Exception:
            return False
    
    def _is_morning_star(self, two_ago: pd.Series, previous: pd.Series, current: pd.Series) -> bool:
        """Check if three candles form a morning star pattern"""
        try:
            # First candle: bearish
            # Second candle: small body (doji-like)
            # Third candle: bullish
            first_bearish = two_ago['close'] < two_ago['open']
            second_small = abs(previous['close'] - previous['open']) < 0.1 * (previous['high'] - previous['low'])
            third_bullish = current['close'] > current['open']
            
            return first_bearish and second_small and third_bullish
            
        except Exception:
            return False
    
    def _is_evening_star(self, two_ago: pd.Series, previous: pd.Series, current: pd.Series) -> bool:
        """Check if three candles form an evening star pattern"""
        try:
            # First candle: bullish
            # Second candle: small body (doji-like)
            # Third candle: bearish
            first_bullish = two_ago['close'] > two_ago['open']
            second_small = abs(previous['close'] - previous['open']) < 0.1 * (previous['high'] - previous['low'])
            third_bearish = current['close'] < current['open']
            
            return first_bullish and second_small and third_bearish
            
        except Exception:
            return False
    
    def _calculate_doji_confidence(self, candle: pd.Series) -> float:
        """Calculate confidence level for doji pattern"""
        try:
            body_size = abs(candle['close'] - candle['open'])
            total_range = candle['high'] - candle['low']
            
            if total_range == 0:
                return 0.0
            
            body_ratio = body_size / total_range
            # Lower body ratio = higher confidence
            confidence = 1.0 - (body_ratio / 0.1)  # Normalize to 0.1 threshold
            return max(0.0, min(1.0, confidence))
            
        except Exception:
            return 0.0
    
    def _calculate_hammer_confidence(self, candle: pd.Series) -> float:
        """Calculate confidence level for hammer pattern"""
        try:
            body_size = abs(candle['close'] - candle['open'])
            upper_shadow = candle['high'] - max(candle['open'], candle['close'])
            lower_shadow = min(candle['open'], candle['close']) - candle['low']
            
            if body_size == 0:
                return 0.0
            
            shadow_ratio = lower_shadow / body_size
            upper_ratio = upper_shadow / body_size
            
            # Higher lower shadow ratio = higher confidence
            # Lower upper shadow ratio = higher confidence
            confidence = min(shadow_ratio / 2.0, 1.0) * (1.0 - upper_ratio)
            return max(0.0, min(1.0, confidence))
            
        except Exception:
            return 0.0
    
    def _calculate_engulfing_confidence(self, previous: pd.Series, current: pd.Series) -> float:
        """Calculate confidence level for engulfing pattern"""
        try:
            prev_body = abs(previous['close'] - previous['open'])
            curr_body = abs(current['close'] - current['open'])
            
            if prev_body == 0:
                return 0.0
            
            engulfing_ratio = curr_body / prev_body
            # Higher engulfing ratio = higher confidence
            confidence = min(engulfing_ratio / 1.5, 1.0)
            return max(0.0, min(1.0, confidence))
            
        except Exception:
            return 0.0
    
    def _calculate_harami_confidence(self, previous: pd.Series, current: pd.Series) -> float:
        """Calculate confidence level for harami pattern"""
        try:
            prev_body = abs(previous['close'] - previous['open'])
            curr_body = abs(current['close'] - current['open'])
            
            if prev_body == 0:
                return 0.0
            
            harami_ratio = curr_body / prev_body
            # Lower harami ratio = higher confidence
            confidence = 1.0 - harami_ratio
            return max(0.0, min(1.0, confidence))
            
        except Exception:
            return 0.0
    
    def _calculate_morning_star_confidence(self, two_ago: pd.Series, previous: pd.Series, current: pd.Series) -> float:
        """Calculate confidence level for morning star pattern"""
        try:
            # Simple confidence based on pattern clarity
            confidence = 0.7  # Base confidence for morning star
            
            # Adjust based on second candle size
            second_body = abs(previous['close'] - previous['open'])
            second_range = previous['high'] - previous['low']
            
            if second_range > 0:
                second_ratio = second_body / second_range
                if second_ratio < 0.05:  # Very small body
                    confidence += 0.2
                elif second_ratio < 0.1:  # Small body
                    confidence += 0.1
            
            return min(1.0, confidence)
            
        except Exception:
            return 0.0
    
    def _calculate_evening_star_confidence(self, two_ago: pd.Series, previous: pd.Series, current: pd.Series) -> float:
        """Calculate confidence level for evening star pattern"""
        try:
            # Similar to morning star but for bearish pattern
            confidence = 0.7  # Base confidence for evening star
            
            # Adjust based on second candle size
            second_body = abs(previous['close'] - previous['open'])
            second_range = previous['high'] - previous['low']
            
            if second_range > 0:
                second_ratio = second_body / second_range
                if second_ratio < 0.05:  # Very small body
                    confidence += 0.2
                elif second_ratio < 0.1:  # Small body
                    confidence += 0.1
            
            return min(1.0, confidence)
            
        except Exception:
            return 0.0
    
    def _get_strength_level(self, confidence: float) -> str:
        """Get strength level based on confidence"""
        if confidence >= self.confidence_thresholds['strong']:
            return 'strong'
        elif confidence >= self.confidence_thresholds['medium']:
            return 'medium'
        else:
            return 'weak'
    
    def _check_volume_confirmation(self, df: pd.DataFrame, pattern: Dict) -> bool:
        """Check if volume confirms the pattern"""
        try:
            if len(df) < 20:  # Need enough data for volume analysis
                return False
            
            current_volume = df.iloc[-1]['volume']
            avg_volume = df['volume'].rolling(20).mean().iloc[-1]
            
            if avg_volume == 0:
                return False
            
            volume_ratio = current_volume / avg_volume
            return volume_ratio >= self.volume_confirmation_threshold
            
        except Exception:
            return False
    
    def _determine_trend_alignment(self, df: pd.DataFrame, pattern: Dict) -> str:
        """Determine if pattern aligns with current trend"""
        try:
            if len(df) < 20:
                return 'neutral'
            
            # Calculate simple trend using SMA
            sma_20 = df['close'].rolling(20).mean().iloc[-1]
            current_price = df.iloc[-1]['close']
            
            if current_price > sma_20 * 1.02:  # 2% above SMA
                return 'bullish'
            elif current_price < sma_20 * 0.98:  # 2% below SMA
                return 'bearish'
            else:
                return 'neutral'
                
        except Exception:
            return 'neutral'
    
    def _get_body_size(self, candle: pd.Series) -> float:
        """Get body size as percentage of total range"""
        try:
            body_size = abs(candle['close'] - candle['open'])
            total_range = candle['high'] - candle['low']
            
            if total_range == 0:
                return 0.0
            
            return (body_size / total_range) * 100
            
        except Exception:
            return 0.0
    
    def _get_shadow_ratio(self, candle: pd.Series) -> float:
        """Get shadow ratio for hammer/shooting star"""
        try:
            body_size = abs(candle['close'] - candle['open'])
            upper_shadow = candle['high'] - max(candle['open'], candle['close'])
            lower_shadow = min(candle['open'], candle['close']) - candle['low']
            
            if body_size == 0:
                return 0.0
            
            return max(upper_shadow, lower_shadow) / body_size
            
        except Exception:
            return 0.0
    
    def _get_engulfing_ratio(self, previous: pd.Series, current: pd.Series) -> float:
        """Get engulfing ratio"""
        try:
            prev_body = abs(previous['close'] - previous['open'])
            curr_body = abs(current['close'] - current['open'])
            
            if prev_body == 0:
                return 0.0
            
            return curr_body / prev_body
            
        except Exception:
            return 0.0
    
    def _get_harami_ratio(self, previous: pd.Series, current: pd.Series) -> float:
        """Get harami ratio"""
        try:
            prev_body = abs(previous['close'] - previous['open'])
            curr_body = abs(current['close'] - current['open'])
            
            if prev_body == 0:
                return 0.0
            
            return curr_body / prev_body
            
        except Exception:
            return 0.0
    
    def _detect_talib_patterns(self, df: pd.DataFrame) -> List[Dict]:
        """Detect patterns using TA-Lib library"""
        try:
            patterns = []
            
            # Ensure we have enough data for TA-Lib
            if len(df) < 20:  # TA-Lib needs more data
                return patterns
            
            # Extract OHLCV data for TA-Lib
            open_prices = df['open'].values
            high_prices = df['high'].values
            low_prices = df['low'].values
            close_prices = df['close'].values
            volumes = df['volume'].values
            
            # TA-Lib Pattern Functions
            pattern_functions = {
                'CDL2CROWS': talib.CDL2CROWS,
                'CDL3BLACKCROWS': talib.CDL3BLACKCROWS,
                'CDL3INSIDE': talib.CDL3INSIDE,
                'CDL3LINESTRIKE': talib.CDL3LINESTRIKE,
                'CDL3OUTSIDE': talib.CDL3OUTSIDE,
                'CDL3STARSINSOUTH': talib.CDL3STARSINSOUTH,
                'CDL3WHITESOLDIERS': talib.CDL3WHITESOLDIERS,
                'CDLABANDONEDBABY': talib.CDLABANDONEDBABY,
                'CDLADVANCEBLOCK': talib.CDLADVANCEBLOCK,
                'CDLBELTHOLD': talib.CDLBELTHOLD,
                'CDLBREAKAWAY': talib.CDLBREAKAWAY,
                'CDLDARKCLOUDCOVER': talib.CDLDARKCLOUDCOVER,
                'CDLDOJI': talib.CDLDOJI,
                'CDLENGULFING': talib.CDLENGULFING,
                'CDLEVENINGDOJISTAR': talib.CDLEVENINGDOJISTAR,
                'CDLEVENINGSTAR': talib.CDLEVENINGSTAR,
                'CDLHAMMER': talib.CDLHAMMER,
                'CDLHANGINGMAN': talib.CDLHANGINGMAN,
                'CDLHARAMI': talib.CDLHARAMI,
                'CDLMARUBOZU': talib.CDLMARUBOZU,
                'CDLMORNINGDOJISTAR': talib.CDLMORNINGDOJISTAR,
                'CDLMORNINGSTAR': talib.CDLMORNINGSTAR,
                'CDLPIERCING': talib.CDLPIERCING,
                'CDLSHOOTINGSTAR': talib.CDLSHOOTINGSTAR,
                'CDLSPINNINGTOP': talib.CDLSPINNINGTOP,
                'CDLTAKURI': talib.CDLTAKURI,
                'CDLTRISTAR': talib.CDLTRISTAR,
                'CDLUNIQUE3RIVER': talib.CDLUNIQUE3RIVER,
                'CDLXSIDEGAP3METHODS': talib.CDLXSIDEGAP3METHODS
            }
            
            # Detect patterns using TA-Lib
            for pattern_name, pattern_func in pattern_functions.items():
                try:
                    # Get pattern values
                    pattern_values = pattern_func(open_prices, high_prices, low_prices, close_prices)
                    
                    # Check if pattern is detected in the latest candle
                    latest_value = pattern_values[-1]
                    
                    if latest_value != 0:  # Pattern detected
                        # Calculate confidence based on pattern strength
                        confidence = self._calculate_talib_confidence(latest_value, pattern_name, df)
                        
                        if confidence > 0.5:  # Only include if confidence > 50%
                            patterns.append({
                                'name': pattern_name.lower(),
                                'confidence': confidence,
                                'metadata': {
                                    'talib_value': int(latest_value),
                                    'pattern_type': 'talib',
                                    'strength': 'strong' if abs(latest_value) >= 100 else 'medium'
                                }
                            })
                            
                except Exception as e:
                    logger.debug(f"Pattern {pattern_name} detection failed: {e}")
                    continue
            
            return patterns
            
        except Exception as e:
            logger.error(f"❌ Error in TA-Lib pattern detection: {e}")
            return []
    
    def _calculate_talib_confidence(self, talib_value: int, pattern_name: str, df: pd.DataFrame) -> float:
        """Calculate confidence for TA-Lib patterns"""
        try:
            # Base confidence from TA-Lib value
            base_confidence = min(abs(talib_value) / 100.0, 1.0)
            
            # Volume confirmation
            volume_ratio = df['volume'].iloc[-1] / df['volume'].rolling(20).mean().iloc[-1]
            volume_boost = min(volume_ratio / 2.0, 0.3)  # Max 30% boost
            
            # Price movement confirmation
            price_change = abs(df['close'].iloc[-1] - df['open'].iloc[-1]) / df['open'].iloc[-1]
            price_boost = min(price_change * 10, 0.2)  # Max 20% boost
            
            # Final confidence
            final_confidence = base_confidence + volume_boost + price_boost
            return min(final_confidence, 1.0)
            
        except Exception:
            return 0.5  # Default confidence
    
    def _get_timeframe_weight(self, timeframe_index: int) -> float:
        """Get weight for timeframe based on its position in hierarchy"""
        # Higher timeframes get more weight
        weights = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]  # 1m to 1W
        return weights[min(timeframe_index, len(weights) - 1)]
    
    def _calculate_multi_timeframe_confidence(self, all_patterns: List[Dict]) -> List[Dict]:
        """Calculate final confidence for patterns across multiple timeframes"""
        try:
            # Group patterns by name
            pattern_groups = {}
            
            for pattern in all_patterns:
                pattern_name = pattern['name']
                if pattern_name not in pattern_groups:
                    pattern_groups[pattern_name] = []
                pattern_groups[pattern_name].append(pattern)
            
            # Calculate multi-timeframe confidence for each pattern type
            confirmed_patterns = []
            
            for pattern_name, patterns in pattern_groups.items():
                if len(patterns) < 2:  # Need at least 2 timeframes for confirmation
                    continue
                
                # Sort by timeframe weight
                patterns.sort(key=lambda x: x['timeframe_weight'], reverse=True)
                
                # Calculate weighted confidence
                total_weight = sum(p['timeframe_weight'] for p in patterns)
                weighted_confidence = sum(
                    p['confidence'] * p['timeframe_weight'] for p in patterns
                ) / total_weight
                
                # Boost confidence based on number of confirming timeframes
                timeframe_boost = min(len(patterns) * 0.1, 0.3)  # Max 30% boost
                final_confidence = min(weighted_confidence + timeframe_boost, 1.0)
                
                # Get primary timeframe (highest weight)
                primary_timeframe = patterns[0]['timeframe']
                
                # Get confirming timeframes
                confirming_timeframes = [p['timeframe'] for p in patterns]
                
                # Get timeframe weights
                timeframe_weights = {p['timeframe']: p['timeframe_weight'] for p in patterns}
                
                confirmed_patterns.append({
                    'name': pattern_name,
                    'primary_timeframe': primary_timeframe,
                    'confirming_timeframes': confirming_timeframes,
                    'timeframe_weights': timeframe_weights,
                    'final_confidence': final_confidence,
                    'volume_confirmed': any(p.get('volume_confirmation', False) for p in patterns),
                    'trend_alignment': self._get_consensus_trend_alignment(patterns)
                })
            
            # Sort by final confidence
            confirmed_patterns.sort(key=lambda x: x['final_confidence'], reverse=True)
            
            return confirmed_patterns
            
        except Exception as e:
            logger.error(f"❌ Error calculating multi-timeframe confidence: {e}")
            return []
    
    def _get_consensus_trend_alignment(self, patterns: List[Dict]) -> str:
        """Get consensus trend alignment across timeframes"""
        try:
            trend_counts = {'bullish': 0, 'bearish': 0, 'neutral': 0}
            
            for pattern in patterns:
                trend = pattern.get('trend_alignment', 'neutral')
                if trend in trend_counts:
                    trend_counts[trend] += 1
            
            # Return most common trend
            return max(trend_counts.items(), key=lambda x: x[1])[0]
            
        except Exception:
            return 'neutral'
    
    async def _get_historical_data(self, symbol: str, timeframe: str, limit: int = 50) -> List[Dict]:
        """Get historical data for pattern analysis"""
        try:
            # This would typically fetch from storage
            # For now, return empty list
            return []
        except Exception as e:
            logger.error(f"❌ Error getting historical data: {e}")
            return []
    
    async def _store_pattern(self, pattern: DetectedPattern):
        """Store detected pattern in database"""
        try:
            # Store pattern data
            await self.storage.store_pattern(
                pattern_name=pattern.pattern_name,
                symbol=pattern.symbol,
                timeframe=pattern.timeframe,
                timestamp=pattern.timestamp,
                confidence=pattern.confidence,
                strength=pattern.strength,
                price_level=pattern.price_level,
                volume_confirmation=pattern.volume_confirmation,
                trend_alignment=pattern.trend_alignment,
                metadata=pattern.additional_data
            )
            
            self.stats['patterns_stored'] += 1
            
        except Exception as e:
            logger.error(f"❌ Error storing pattern: {e}")
    
    async def _notify_pattern_callbacks(self, pattern: DetectedPattern):
        """Notify pattern callbacks"""
        for callback in self.pattern_callbacks:
            try:
                await callback(pattern)
            except Exception as e:
                logger.error(f"❌ Error in pattern callback: {e}")
    
    def _update_stats(self, start_time: datetime):
        """Update analysis statistics"""
        analysis_time = (datetime.now() - start_time).total_seconds() * 1000  # Convert to ms
        
        # Update average analysis time
        if self.stats['patterns_detected'] == 0:
            self.stats['analysis_time_avg'] = analysis_time
        else:
            self.stats['analysis_time_avg'] = (
                (self.stats['analysis_time_avg'] * self.stats['patterns_detected'] + analysis_time) /
                (self.stats['patterns_detected'] + 1)
            )
        
        self.stats['patterns_detected'] += 1
        self.stats['last_update'] = datetime.now()
    
    def add_pattern_callback(self, callback):
        """Add pattern callback"""
        self.pattern_callbacks.append(callback)
    
    def get_analysis_stats(self) -> Dict:
        """Get analysis statistics"""
        return self.stats.copy()

def test_pattern_analyzer():
    """Test the pattern analyzer"""
    config = {
        'patterns_to_detect': ['doji', 'hammer', 'engulfing'],
        'storage_path': 'test_data'
    }
    
    analyzer = PatternAnalyzer(config)
    
    # Test pattern detection
    test_candle = {
        'symbol': 'BTCUSDT',
        'timeframe': '1m',
        'timestamp': datetime.now(),
        'open': 50000.0,
        'high': 50010.0,
        'low': 49990.0,
        'close': 50005.0,
        'volume': 100.0
    }
    
    print("🔍 Pattern Analyzer test completed")
    print(f"Patterns to detect: {analyzer.patterns_to_detect}")
    print(f"Confidence thresholds: {analyzer.confidence_thresholds}")

if __name__ == "__main__":
    test_pattern_analyzer()
