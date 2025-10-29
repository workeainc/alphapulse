"""
Advanced Divergence Analyzer for AlphaPlus
Comprehensive RSI, Volume, and MACD divergence detection with correlation analysis
Integrated with SDE framework for signal quality enhancement
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import asyncio
import asyncpg

logger = logging.getLogger(__name__)

class DivergenceType(Enum):
    """Types of divergence patterns"""
    BULLISH_DIVERGENCE = "bullish_divergence"
    BEARISH_DIVERGENCE = "bearish_divergence"
    HIDDEN_BULLISH = "hidden_bullish"
    HIDDEN_BEARISH = "hidden_bearish"
    NO_DIVERGENCE = "no_divergence"

class DivergenceStrength(Enum):
    """Strength levels of divergence"""
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    EXTREME = "extreme"

class IndicatorType(Enum):
    """Types of indicators for divergence analysis"""
    RSI = "rsi"
    MACD = "macd"
    VOLUME = "volume"
    COMBINED = "combined"

@dataclass
class DivergencePoint:
    """Represents a divergence point"""
    index: int
    price: float
    indicator_value: float
    timestamp: datetime
    strength: float
    divergence_score: float = 0.0

@dataclass
class DivergenceSignal:
    """Complete divergence signal"""
    divergence_type: DivergenceType
    indicator_type: IndicatorType
    strength: DivergenceStrength
    confidence: float
    price_points: List[DivergencePoint]
    indicator_points: List[DivergencePoint]
    correlation_score: float
    confirmation_signals: List[str]
    signal_timestamp: datetime
    analysis_window: int
    divergence_score: float

@dataclass
class DivergenceAnalysis:
    """Complete divergence analysis result"""
    rsi_divergence: Optional[DivergenceSignal]
    macd_divergence: Optional[DivergenceSignal]
    volume_divergence: Optional[DivergenceSignal]
    combined_divergence: Optional[DivergenceSignal]
    overall_confidence: float
    divergence_score: float
    confirmation_count: int
    analysis_timestamp: datetime
    signals: List[DivergenceSignal] = None

class AdvancedDivergenceAnalyzer:
    """Advanced divergence analyzer with SDE integration"""
    
    def __init__(self, db_pool: asyncpg.Pool = None):
        self.db_pool = db_pool
        
        # Divergence detection parameters
        self.divergence_config = {
            'rsi': {
                'period': 14,
                'overbought': 70,
                'oversold': 30,
                'min_swing_distance': 5,
                'confirmation_threshold': 0.6
            },
            'macd': {
                'fast_period': 12,
                'slow_period': 26,
                'signal_period': 9,
                'min_swing_distance': 5,
                'confirmation_threshold': 0.6
            },
            'volume': {
                'period': 20,
                'min_swing_distance': 3,
                'confirmation_threshold': 0.5,
                'volume_threshold': 1.5
            }
        }
        
        # Analysis windows
        self.analysis_windows = {
            'short_term': 10,
            'medium_term': 20,
            'long_term': 50
        }
        
        # Strength thresholds
        self.strength_thresholds = {
            DivergenceStrength.WEAK: 0.3,
            DivergenceStrength.MODERATE: 0.5,
            DivergenceStrength.STRONG: 0.7,
            DivergenceStrength.EXTREME: 0.9
        }
        
        logger.info("🚀 Advanced Divergence Analyzer initialized")
    
    async def analyze_divergences(self, df: pd.DataFrame, symbol: str, timeframe: str) -> DivergenceAnalysis:
        """Comprehensive divergence analysis"""
        try:
            if len(df) < 50:
                logger.warning(f"Insufficient data for divergence analysis: {len(df)} candles")
                return self._create_empty_analysis()
            
            # Calculate indicators
            df_with_indicators = self._calculate_indicators(df)
            
            # Detect divergences for each indicator
            rsi_divergence = await self._detect_rsi_divergence(df_with_indicators, symbol, timeframe)
            macd_divergence = await self._detect_macd_divergence(df_with_indicators, symbol, timeframe)
            volume_divergence = await self._detect_volume_divergence(df_with_indicators, symbol, timeframe)
            
            # Analyze combined divergence
            combined_divergence = await self._analyze_combined_divergence(
                rsi_divergence, macd_divergence, volume_divergence, df_with_indicators
            )
            
            # Calculate overall confidence and score
            overall_confidence = self._calculate_overall_confidence(
                rsi_divergence, macd_divergence, volume_divergence, combined_divergence
            )
            
            divergence_score = self._calculate_divergence_score(
                rsi_divergence, macd_divergence, volume_divergence, combined_divergence
            )
            
            confirmation_count = self._count_confirmations(
                rsi_divergence, macd_divergence, volume_divergence, combined_divergence
            )
            
            # Store analysis results
            await self._store_divergence_analysis(
                symbol, timeframe, rsi_divergence, macd_divergence, 
                volume_divergence, combined_divergence, overall_confidence, divergence_score
            )
            
            # Collect all signals
            signals = []
            if rsi_divergence:
                signals.append(rsi_divergence)
            if macd_divergence:
                signals.append(macd_divergence)
            if volume_divergence:
                signals.append(volume_divergence)
            if combined_divergence:
                signals.append(combined_divergence)
            
            return DivergenceAnalysis(
                rsi_divergence=rsi_divergence,
                macd_divergence=macd_divergence,
                volume_divergence=volume_divergence,
                combined_divergence=combined_divergence,
                overall_confidence=overall_confidence,
                divergence_score=divergence_score,
                confirmation_count=confirmation_count,
                analysis_timestamp=datetime.now(),
                signals=signals
            )
            
        except Exception as e:
            logger.error(f"❌ Divergence analysis failed: {e}")
            return self._create_empty_analysis()
    
    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate RSI, MACD, and Volume indicators"""
        try:
            df = df.copy()
            
            # Calculate RSI
            df['rsi'] = self._calculate_rsi(df['close'], self.divergence_config['rsi']['period'])
            
            # Calculate MACD
            macd_data = self._calculate_macd(
                df['close'], 
                self.divergence_config['macd']['fast_period'],
                self.divergence_config['macd']['slow_period'],
                self.divergence_config['macd']['signal_period']
            )
            df['macd_line'] = macd_data['macd_line']
            df['macd_signal'] = macd_data['macd_signal']
            df['macd_histogram'] = macd_data['macd_histogram']
            
            # Calculate Volume indicators
            df['volume_sma'] = df['volume'].rolling(self.divergence_config['volume']['period']).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            df['volume_ema'] = df['volume'].ewm(span=self.divergence_config['volume']['period']).mean()
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Indicator calculation failed: {e}")
            return df
    
    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate RSI indicator"""
        try:
            delta = prices.diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            
            avg_gain = gain.rolling(window=period).mean()
            avg_loss = loss.rolling(window=period).mean()
            
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            return rsi
            
        except Exception as e:
            logger.error(f"❌ RSI calculation failed: {e}")
            return pd.Series([50] * len(prices), index=prices.index)
    
    def _calculate_macd(self, prices: pd.Series, fast_period: int, slow_period: int, signal_period: int) -> Dict[str, pd.Series]:
        """Calculate MACD indicator"""
        try:
            ema_fast = prices.ewm(span=fast_period).mean()
            ema_slow = prices.ewm(span=slow_period).mean()
            
            macd_line = ema_fast - ema_slow
            macd_signal = macd_line.ewm(span=signal_period).mean()
            macd_histogram = macd_line - macd_signal
            
            return {
                'macd_line': macd_line,
                'macd_signal': macd_signal,
                'macd_histogram': macd_histogram
            }
            
        except Exception as e:
            logger.error(f"❌ MACD calculation failed: {e}")
            return {
                'macd_line': pd.Series([0] * len(prices), index=prices.index),
                'macd_signal': pd.Series([0] * len(prices), index=prices.index),
                'macd_histogram': pd.Series([0] * len(prices), index=prices.index)
            }
    
    async def _detect_rsi_divergence(self, df: pd.DataFrame, symbol: str, timeframe: str) -> Optional[DivergenceSignal]:
        """Detect RSI divergence patterns"""
        try:
            if 'rsi' not in df.columns:
                return None
            
            # Find swing points
            price_swings = self._find_swing_points(df['close'], 'price')
            rsi_swings = self._find_swing_points(df['rsi'], 'rsi')
            
            if len(price_swings) < 2 or len(rsi_swings) < 2:
                return None
            
            # Detect divergence patterns
            divergence_patterns = self._detect_divergence_patterns(
                price_swings, rsi_swings, df, 'rsi'
            )
            
            if not divergence_patterns:
                return None
            
            # Get the strongest divergence
            strongest_divergence = max(divergence_patterns, key=lambda x: x['strength'])
            
            # Calculate correlation and confidence
            correlation_score = self._calculate_correlation(df['close'], df['rsi'])
            confidence = self._calculate_divergence_confidence(strongest_divergence, correlation_score)
            
            # Create divergence signal
            return DivergenceSignal(
                divergence_type=strongest_divergence['type'],
                indicator_type=IndicatorType.RSI,
                strength=strongest_divergence['strength'],
                confidence=confidence,
                price_points=strongest_divergence['price_points'],
                indicator_points=strongest_divergence['indicator_points'],
                correlation_score=correlation_score,
                confirmation_signals=strongest_divergence['confirmations'],
                signal_timestamp=datetime.now(),
                analysis_window=self.analysis_windows['medium_term'],
                divergence_score=strongest_divergence['score']
            )
            
        except Exception as e:
            logger.error(f"❌ RSI divergence detection failed: {e}")
            return None
    
    async def _detect_macd_divergence(self, df: pd.DataFrame, symbol: str, timeframe: str) -> Optional[DivergenceSignal]:
        """Detect MACD divergence patterns"""
        try:
            if 'macd_line' not in df.columns:
                return None
            
            # Find swing points
            price_swings = self._find_swing_points(df['close'], 'price')
            macd_swings = self._find_swing_points(df['macd_line'], 'macd')
            
            if len(price_swings) < 2 or len(macd_swings) < 2:
                return None
            
            # Detect divergence patterns
            divergence_patterns = self._detect_divergence_patterns(
                price_swings, macd_swings, df, 'macd'
            )
            
            if not divergence_patterns:
                return None
            
            # Get the strongest divergence
            strongest_divergence = max(divergence_patterns, key=lambda x: x['strength'])
            
            # Calculate correlation and confidence
            correlation_score = self._calculate_correlation(df['close'], df['macd_line'])
            confidence = self._calculate_divergence_confidence(strongest_divergence, correlation_score)
            
            # Create divergence signal
            return DivergenceSignal(
                divergence_type=strongest_divergence['type'],
                indicator_type=IndicatorType.MACD,
                strength=strongest_divergence['strength'],
                confidence=confidence,
                price_points=strongest_divergence['price_points'],
                indicator_points=strongest_divergence['indicator_points'],
                correlation_score=correlation_score,
                confirmation_signals=strongest_divergence['confirmations'],
                signal_timestamp=datetime.now(),
                analysis_window=self.analysis_windows['medium_term'],
                divergence_score=strongest_divergence['score']
            )
            
        except Exception as e:
            logger.error(f"❌ MACD divergence detection failed: {e}")
            return None
    
    async def _detect_volume_divergence(self, df: pd.DataFrame, symbol: str, timeframe: str) -> Optional[DivergenceSignal]:
        """Detect Volume divergence patterns"""
        try:
            if 'volume_ratio' not in df.columns:
                return None
            
            # Find swing points
            price_swings = self._find_swing_points(df['close'], 'price')
            volume_swings = self._find_swing_points(df['volume_ratio'], 'volume')
            
            if len(price_swings) < 2 or len(volume_swings) < 2:
                return None
            
            # Detect divergence patterns
            divergence_patterns = self._detect_divergence_patterns(
                price_swings, volume_swings, df, 'volume'
            )
            
            if not divergence_patterns:
                return None
            
            # Get the strongest divergence
            strongest_divergence = max(divergence_patterns, key=lambda x: x['strength'])
            
            # Calculate correlation and confidence
            correlation_score = self._calculate_correlation(df['close'], df['volume_ratio'])
            confidence = self._calculate_divergence_confidence(strongest_divergence, correlation_score)
            
            # Create divergence signal
            return DivergenceSignal(
                divergence_type=strongest_divergence['type'],
                indicator_type=IndicatorType.VOLUME,
                strength=strongest_divergence['strength'],
                confidence=confidence,
                price_points=strongest_divergence['price_points'],
                indicator_points=strongest_divergence['indicator_points'],
                correlation_score=correlation_score,
                confirmation_signals=strongest_divergence['confirmations'],
                signal_timestamp=datetime.now(),
                analysis_window=self.analysis_windows['medium_term'],
                divergence_score=strongest_divergence['score']
            )
            
        except Exception as e:
            logger.error(f"❌ Volume divergence detection failed: {e}")
            return None
    
    def _find_swing_points(self, series: pd.Series, series_type: str) -> List[DivergencePoint]:
        """Find swing highs and lows in a series"""
        try:
            swings = []
            min_distance = self.divergence_config.get(series_type, {}).get('min_swing_distance', 5)
            
            for i in range(min_distance, len(series) - min_distance):
                # Check for swing high
                if all(series.iloc[i] >= series.iloc[j] for j in range(i - min_distance, i + min_distance + 1) if j != i):
                    swings.append(DivergencePoint(
                        index=i,
                        price=series.iloc[i],
                        indicator_value=series.iloc[i],
                        timestamp=series.index[i] if hasattr(series.index[i], 'to_pydatetime') else datetime.now(),
                        strength=self._calculate_swing_strength(series, i, 'high')
                    ))
                
                # Check for swing low
                elif all(series.iloc[i] <= series.iloc[j] for j in range(i - min_distance, i + min_distance + 1) if j != i):
                    swings.append(DivergencePoint(
                        index=i,
                        price=series.iloc[i],
                        indicator_value=series.iloc[i],
                        timestamp=series.index[i] if hasattr(series.index[i], 'to_pydatetime') else datetime.now(),
                        strength=self._calculate_swing_strength(series, i, 'low')
                    ))
            
            return swings
            
        except Exception as e:
            logger.error(f"❌ Swing point detection failed: {e}")
            return []
    
    def _calculate_swing_strength(self, series: pd.Series, index: int, swing_type: str) -> float:
        """Calculate the strength of a swing point"""
        try:
            if index < 2 or index >= len(series) - 2:
                return 0.5
            
            current_value = series.iloc[index]
            
            if swing_type == 'high':
                left_values = series.iloc[index-2:index]
                right_values = series.iloc[index+1:index+3]
                strength = (current_value - left_values.min()) / (right_values.max() - left_values.min() + 1e-8)
            else:  # low
                left_values = series.iloc[index-2:index]
                right_values = series.iloc[index+1:index+3]
                strength = (left_values.max() - current_value) / (left_values.max() - right_values.min() + 1e-8)
            
            return min(max(strength, 0.0), 1.0)
            
        except Exception as e:
            logger.error(f"❌ Swing strength calculation failed: {e}")
            return 0.5
    
    def _detect_divergence_patterns(self, price_swings: List[DivergencePoint], 
                                  indicator_swings: List[DivergencePoint], 
                                  df: pd.DataFrame, indicator_type: str) -> List[Dict]:
        """Detect divergence patterns between price and indicator"""
        try:
            patterns = []
            
            # Get recent swings (last 20 periods)
            recent_price_swings = [s for s in price_swings if s.index >= len(df) - 20]
            recent_indicator_swings = [s for s in indicator_swings if s.index >= len(df) - 20]
            
            if len(recent_price_swings) < 2 or len(recent_indicator_swings) < 2:
                return patterns
            
            # Check for regular divergence
            for i in range(len(recent_price_swings) - 1):
                for j in range(len(recent_indicator_swings) - 1):
                    pattern = self._check_divergence_pattern(
                        recent_price_swings[i:i+2], 
                        recent_indicator_swings[j:j+2], 
                        df, indicator_type
                    )
                    if pattern:
                        patterns.append(pattern)
            
            return patterns
            
        except Exception as e:
            logger.error(f"❌ Divergence pattern detection failed: {e}")
            return []
    
    def _check_divergence_pattern(self, price_swings: List[DivergencePoint], 
                                indicator_swings: List[DivergencePoint], 
                                df: pd.DataFrame, indicator_type: str) -> Optional[Dict]:
        """Check for specific divergence pattern"""
        try:
            if len(price_swings) < 2 or len(indicator_swings) < 2:
                return None
            
            price1, price2 = price_swings[0], price_swings[1]
            ind1, ind2 = indicator_swings[0], indicator_swings[1]
            
            # Check for bullish divergence (price lower low, indicator higher low)
            if (price2.price < price1.price and ind2.indicator_value > ind1.indicator_value):
                divergence_type = DivergenceType.BULLISH_DIVERGENCE
                strength = self._calculate_divergence_strength(price1, price2, ind1, ind2)
                confirmations = self._get_divergence_confirmations(df, indicator_type, 'bullish')
                
                return {
                    'type': divergence_type,
                    'strength': strength,
                    'price_points': price_swings,
                    'indicator_points': indicator_swings,
                    'confirmations': confirmations,
                    'score': self._calculate_divergence_score(price1, price2, ind1, ind2)
                }
            
            # Check for bearish divergence (price higher high, indicator lower high)
            elif (price2.price > price1.price and ind2.indicator_value < ind1.indicator_value):
                divergence_type = DivergenceType.BEARISH_DIVERGENCE
                strength = self._calculate_divergence_strength(price1, price2, ind1, ind2)
                confirmations = self._get_divergence_confirmations(df, indicator_type, 'bearish')
                
                return {
                    'type': divergence_type,
                    'strength': strength,
                    'price_points': price_swings,
                    'indicator_points': indicator_swings,
                    'confirmations': confirmations,
                    'score': self._calculate_divergence_score(price1, price2, ind1, ind2)
                }
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Divergence pattern check failed: {e}")
            return None
    
    def _calculate_divergence_strength(self, price1: DivergencePoint, price2: DivergencePoint,
                                     ind1: DivergencePoint, ind2: DivergencePoint) -> DivergenceStrength:
        """Calculate the strength of a divergence pattern"""
        try:
            # Calculate price change magnitude
            price_change = abs(price2.price - price1.price) / price1.price
            
            # Calculate indicator change magnitude
            ind_change = abs(ind2.indicator_value - ind1.indicator_value) / (abs(ind1.indicator_value) + 1e-8)
            
            # Calculate overall strength
            strength = (price_change + ind_change) / 2
            
            # Determine strength level
            if strength >= self.strength_thresholds[DivergenceStrength.EXTREME]:
                return DivergenceStrength.EXTREME
            elif strength >= self.strength_thresholds[DivergenceStrength.STRONG]:
                return DivergenceStrength.STRONG
            elif strength >= self.strength_thresholds[DivergenceStrength.MODERATE]:
                return DivergenceStrength.MODERATE
            else:
                return DivergenceStrength.WEAK
                
        except Exception as e:
            logger.error(f"❌ Divergence strength calculation failed: {e}")
            return DivergenceStrength.WEAK
    
    def _calculate_divergence_score(self, price1: DivergencePoint, price2: DivergencePoint,
                                  ind1: DivergencePoint, ind2: DivergencePoint) -> float:
        """Calculate numerical divergence score (0-1)"""
        try:
            # Base score from strength
            strength_score = {
                DivergenceStrength.WEAK: 0.3,
                DivergenceStrength.MODERATE: 0.5,
                DivergenceStrength.STRONG: 0.7,
                DivergenceStrength.EXTREME: 0.9
            }[self._calculate_divergence_strength(price1, price2, ind1, ind2)]
            
            # Additional factors
            time_factor = 1.0 - (abs(price2.index - price1.index) / 50)  # Closer swings = higher score
            swing_strength_factor = (price1.strength + price2.strength + ind1.strength + ind2.strength) / 4
            
            final_score = (strength_score + time_factor + swing_strength_factor) / 3
            return min(max(final_score, 0.0), 1.0)
            
        except Exception as e:
            logger.error(f"❌ Divergence score calculation failed: {e}")
            return 0.5
    
    def _get_divergence_confirmations(self, df: pd.DataFrame, indicator_type: str, divergence_type: str) -> List[str]:
        """Get confirmation signals for divergence"""
        try:
            confirmations = []
            
            if indicator_type == 'rsi':
                if divergence_type == 'bullish':
                    if df['rsi'].iloc[-1] < self.divergence_config['rsi']['oversold']:
                        confirmations.append('rsi_oversold')
                    if df['close'].iloc[-1] < df['close'].rolling(20).mean().iloc[-1]:
                        confirmations.append('price_below_ma')
                elif divergence_type == 'bearish':
                    if df['rsi'].iloc[-1] > self.divergence_config['rsi']['overbought']:
                        confirmations.append('rsi_overbought')
                    if df['close'].iloc[-1] > df['close'].rolling(20).mean().iloc[-1]:
                        confirmations.append('price_above_ma')
            
            elif indicator_type == 'macd':
                if df['macd_histogram'].iloc[-1] > 0:
                    confirmations.append('macd_positive_histogram')
                if df['macd_line'].iloc[-1] > df['macd_signal'].iloc[-1]:
                    confirmations.append('macd_line_above_signal')
            
            elif indicator_type == 'volume':
                if df['volume_ratio'].iloc[-1] > self.divergence_config['volume']['volume_threshold']:
                    confirmations.append('high_volume')
                if df['volume'].iloc[-1] > df['volume'].rolling(10).mean().iloc[-1]:
                    confirmations.append('volume_above_average')
            
            return confirmations
            
        except Exception as e:
            logger.error(f"❌ Divergence confirmation check failed: {e}")
            return []
    
    def _calculate_correlation(self, series1: pd.Series, series2: pd.Series) -> float:
        """Calculate correlation between two series"""
        try:
            # Remove NaN values
            valid_data = pd.concat([series1, series2], axis=1).dropna()
            if len(valid_data) < 10:
                return 0.0
            
            correlation = valid_data.iloc[:, 0].corr(valid_data.iloc[:, 1])
            return abs(correlation) if not pd.isna(correlation) else 0.0
            
        except Exception as e:
            logger.error(f"❌ Correlation calculation failed: {e}")
            return 0.0
    
    def _calculate_divergence_confidence(self, divergence: Dict, correlation_score: float) -> float:
        """Calculate confidence in divergence signal"""
        try:
            base_confidence = divergence['score']
            correlation_factor = 1.0 - correlation_score  # Lower correlation = higher confidence in divergence
            confirmation_factor = len(divergence['confirmations']) / 5  # More confirmations = higher confidence
            
            confidence = (base_confidence + correlation_factor + confirmation_factor) / 3
            return min(max(confidence, 0.0), 1.0)
            
        except Exception as e:
            logger.error(f"❌ Divergence confidence calculation failed: {e}")
            return 0.5
    
    async def _analyze_combined_divergence(self, rsi_div: Optional[DivergenceSignal], 
                                         macd_div: Optional[DivergenceSignal],
                                         volume_div: Optional[DivergenceSignal],
                                         df: pd.DataFrame) -> Optional[DivergenceSignal]:
        """Analyze combined divergence from multiple indicators"""
        try:
            divergences = [d for d in [rsi_div, macd_div, volume_div] if d is not None]
            
            if not divergences:
                return None
            
            # Count bullish vs bearish divergences
            bullish_count = sum(1 for d in divergences if d.divergence_type in [DivergenceType.BULLISH_DIVERGENCE, DivergenceType.HIDDEN_BULLISH])
            bearish_count = sum(1 for d in divergences if d.divergence_type in [DivergenceType.BEARISH_DIVERGENCE, DivergenceType.HIDDEN_BEARISH])
            
            if bullish_count == 0 and bearish_count == 0:
                return None
            
            # Determine combined divergence type
            if bullish_count > bearish_count:
                combined_type = DivergenceType.BULLISH_DIVERGENCE
            else:
                combined_type = DivergenceType.BEARISH_DIVERGENCE
            
            # Calculate combined metrics
            avg_confidence = np.mean([d.confidence for d in divergences])
            avg_score = np.mean([d.divergence_score for d in divergences])
            avg_correlation = np.mean([d.correlation_score for d in divergences])
            
            # Determine strength
            if avg_score >= 0.8:
                strength = DivergenceStrength.EXTREME
            elif avg_score >= 0.6:
                strength = DivergenceStrength.STRONG
            elif avg_score >= 0.4:
                strength = DivergenceStrength.MODERATE
            else:
                strength = DivergenceStrength.WEAK
            
            # Combine confirmation signals
            all_confirmations = []
            for d in divergences:
                all_confirmations.extend(d.confirmation_signals)
            
            return DivergenceSignal(
                divergence_type=combined_type,
                indicator_type=IndicatorType.COMBINED,
                strength=strength,
                confidence=avg_confidence,
                price_points=divergences[0].price_points if divergences else [],
                indicator_points=divergences[0].indicator_points if divergences else [],
                correlation_score=avg_correlation,
                confirmation_signals=list(set(all_confirmations)),
                signal_timestamp=datetime.now(),
                analysis_window=self.analysis_windows['medium_term'],
                divergence_score=avg_score
            )
            
        except Exception as e:
            logger.error(f"❌ Combined divergence analysis failed: {e}")
            return None
    
    def _calculate_overall_confidence(self, rsi_div: Optional[DivergenceSignal],
                                    macd_div: Optional[DivergenceSignal],
                                    volume_div: Optional[DivergenceSignal],
                                    combined_div: Optional[DivergenceSignal]) -> float:
        """Calculate overall confidence from all divergence signals"""
        try:
            confidences = []
            
            if rsi_div:
                confidences.append(rsi_div.confidence)
            if macd_div:
                confidences.append(macd_div.confidence)
            if volume_div:
                confidences.append(volume_div.confidence)
            if combined_div:
                confidences.append(combined_div.confidence * 1.2)  # Boost for combined signal
            
            return np.mean(confidences) if confidences else 0.0
            
        except Exception as e:
            logger.error(f"❌ Overall confidence calculation failed: {e}")
            return 0.0
    
    def _calculate_divergence_score(self, rsi_div: Optional[DivergenceSignal],
                                  macd_div: Optional[DivergenceSignal],
                                  volume_div: Optional[DivergenceSignal],
                                  combined_div: Optional[DivergenceSignal]) -> float:
        """Calculate overall divergence score"""
        try:
            scores = []
            
            if rsi_div:
                scores.append(rsi_div.divergence_score)
            if macd_div:
                scores.append(macd_div.divergence_score)
            if volume_div:
                scores.append(volume_div.divergence_score)
            if combined_div:
                scores.append(combined_div.divergence_score * 1.3)  # Boost for combined signal
            
            return np.mean(scores) if scores else 0.0
            
        except Exception as e:
            logger.error(f"❌ Overall divergence score calculation failed: {e}")
            return 0.0
    
    def _count_confirmations(self, rsi_div: Optional[DivergenceSignal],
                           macd_div: Optional[DivergenceSignal],
                           volume_div: Optional[DivergenceSignal],
                           combined_div: Optional[DivergenceSignal]) -> int:
        """Count total confirmation signals"""
        try:
            count = 0
            
            if rsi_div:
                count += len(rsi_div.confirmation_signals)
            if macd_div:
                count += len(macd_div.confirmation_signals)
            if volume_div:
                count += len(volume_div.confirmation_signals)
            if combined_div:
                count += len(combined_div.confirmation_signals)
            
            return count
            
        except Exception as e:
            logger.error(f"❌ Confirmation count failed: {e}")
            return 0
    
    async def _store_divergence_analysis(self, symbol: str, timeframe: str,
                                       rsi_div: Optional[DivergenceSignal],
                                       macd_div: Optional[DivergenceSignal],
                                       volume_div: Optional[DivergenceSignal],
                                       combined_div: Optional[DivergenceSignal],
                                       overall_confidence: float,
                                       divergence_score: float) -> None:
        """Store divergence analysis results in database"""
        try:
            if not self.db_pool:
                return
            
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO sde_divergence_analysis (
                        symbol, timeframe, rsi_divergence_type, rsi_confidence, rsi_score,
                        macd_divergence_type, macd_confidence, macd_score,
                        volume_divergence_type, volume_confidence, volume_score,
                        combined_divergence_type, combined_confidence, combined_score,
                        overall_confidence, divergence_score, analysis_timestamp
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17)
                """, symbol, timeframe,
                    rsi_div.divergence_type.value if rsi_div else None,
                    rsi_div.confidence if rsi_div else 0.0,
                    rsi_div.divergence_score if rsi_div else 0.0,
                    macd_div.divergence_type.value if macd_div else None,
                    macd_div.confidence if macd_div else 0.0,
                    macd_div.divergence_score if macd_div else 0.0,
                    volume_div.divergence_type.value if volume_div else None,
                    volume_div.confidence if volume_div else 0.0,
                    volume_div.divergence_score if volume_div else 0.0,
                    combined_div.divergence_type.value if combined_div else None,
                    combined_div.confidence if combined_div else 0.0,
                    combined_div.divergence_score if combined_div else 0.0,
                    overall_confidence, divergence_score, datetime.now())
                
        except Exception as e:
            logger.error(f"❌ Failed to store divergence analysis: {e}")
    
    def _create_empty_analysis(self) -> DivergenceAnalysis:
        """Create empty divergence analysis result"""
        return DivergenceAnalysis(
            rsi_divergence=None,
            macd_divergence=None,
            volume_divergence=None,
            combined_divergence=None,
            overall_confidence=0.0,
            divergence_score=0.0,
            confirmation_count=0,
            analysis_timestamp=datetime.now()
        )
