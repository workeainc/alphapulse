"""
CVD (Cumulative Volume Delta) Analyzer for AlphaPulse
Tracks institutional buying/selling pressure through cumulative volume delta
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import asyncio

logger = logging.getLogger(__name__)

class CVDDivergence(Enum):
    """CVD divergence types"""
    BULLISH = "bullish"  # Price lower low, CVD higher low
    BEARISH = "bearish"  # Price higher high, CVD lower high
    HIDDEN_BULLISH = "hidden_bullish"  # Price higher low, CVD lower low
    HIDDEN_BEARISH = "hidden_bearish"  # Price lower high, CVD higher high
    NONE = "none"

@dataclass
class Trade:
    """Individual trade data"""
    timestamp: datetime
    price: float
    volume: float
    is_buy: bool  # True if buyer was aggressor

@dataclass
class CVDLevel:
    """CVD support/resistance level"""
    level: float
    strength: float
    touch_count: int
    last_touch: datetime
    level_type: str  # 'support' or 'resistance'

@dataclass
class CVDDivergenceSignal:
    """CVD divergence signal"""
    divergence_type: CVDDivergence
    price_points: List[Tuple[int, float]]  # (index, price)
    cvd_points: List[Tuple[int, float]]  # (index, cvd_value)
    strength: float
    confidence: float
    timestamp: datetime
    metadata: Dict[str, Any]

@dataclass
class CVDAnalysis:
    """Complete CVD analysis"""
    symbol: str
    timeframe: str
    timestamp: datetime
    current_cvd: float
    cvd_trend: str  # 'bullish', 'bearish', 'neutral'
    cvd_momentum: float
    divergences: List[CVDDivergenceSignal]
    support_levels: List[CVDLevel]
    resistance_levels: List[CVDLevel]
    breakout_detected: bool
    overall_confidence: float
    cvd_signals: List[Dict[str, Any]]
    metadata: Dict[str, Any]

class CVDAnalyzer:
    """
    Cumulative Volume Delta (CVD) Analysis Engine
    
    CVD tracks the running total of buy volume minus sell volume,
    revealing institutional accumulation/distribution patterns.
    
    Key Features:
    - Real-time CVD calculation from trade data
    - CVD divergence detection (bullish/bearish)
    - CVD support/resistance levels
    - CVD breakout identification
    - Multi-timeframe CVD analysis
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logger
        
        # Configuration
        self.lookback_periods = self.config.get('lookback_periods', 100)
        self.min_confidence = self.config.get('min_confidence', 0.6)
        self.divergence_window = self.config.get('divergence_window', 20)
        self.level_threshold = self.config.get('level_threshold', 0.02)  # 2%
        
        # CVD cache for different timeframes
        self.cvd_cache: Dict[str, pd.Series] = {}
        
        # Performance tracking
        self.stats = {
            'cvd_calculations': 0,
            'divergences_detected': 0,
            'breakouts_detected': 0,
            'analyses_performed': 0,
            'last_update': datetime.now()
        }
        
        logger.info("🚀 CVD Analyzer initialized")
    
    async def analyze_cvd(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str,
        trades_data: Optional[List[Trade]] = None
    ) -> CVDAnalysis:
        """
        Complete CVD analysis
        
        Args:
            df: DataFrame with OHLCV data
            symbol: Trading symbol
            timeframe: Timeframe
            trades_data: Optional list of trades with buy/sell classification
            
        Returns:
            CVDAnalysis with complete CVD metrics
        """
        try:
            if len(df) < self.lookback_periods:
                self.logger.warning(
                    f"Insufficient data for CVD analysis: {len(df)} < {self.lookback_periods}"
                )
                return self._get_default_analysis(symbol, timeframe)
            
            # Calculate CVD
            if trades_data:
                cvd_series = self._calculate_cvd_from_trades(trades_data, df)
            else:
                # Approximate CVD from OHLCV data
                cvd_series = self._approximate_cvd_from_ohlcv(df)
            
            # Add CVD to dataframe
            df['cvd'] = cvd_series
            
            # Current CVD metrics
            current_cvd = cvd_series.iloc[-1]
            cvd_trend = self._determine_cvd_trend(cvd_series)
            cvd_momentum = self._calculate_cvd_momentum(cvd_series)
            
            # Detect divergences
            divergences = await self._detect_divergences(df)
            
            # Identify CVD support/resistance levels
            support_levels, resistance_levels = self._identify_cvd_levels(cvd_series)
            
            # Detect CVD breakouts
            breakout_detected = self._detect_cvd_breakout(
                cvd_series, support_levels, resistance_levels
            )
            
            # Generate CVD signals
            cvd_signals = await self._generate_cvd_signals(
                df, divergences, breakout_detected, cvd_trend
            )
            
            # Calculate overall confidence
            overall_confidence = self._calculate_overall_confidence(
                divergences, cvd_trend, breakout_detected
            )
            
            # Create analysis result
            analysis = CVDAnalysis(
                symbol=symbol,
                timeframe=timeframe,
                timestamp=df['timestamp'].iloc[-1] if 'timestamp' in df.columns else datetime.now(),
                current_cvd=current_cvd,
                cvd_trend=cvd_trend,
                cvd_momentum=cvd_momentum,
                divergences=divergences,
                support_levels=support_levels,
                resistance_levels=resistance_levels,
                breakout_detected=breakout_detected,
                overall_confidence=overall_confidence,
                cvd_signals=cvd_signals,
                metadata={
                    'analysis_version': '1.0',
                    'config': self.config,
                    'stats': self.stats,
                    'data_source': 'trades' if trades_data else 'ohlcv_approximation'
                }
            )
            
            # Update statistics
            self.stats['cvd_calculations'] += 1
            self.stats['divergences_detected'] += len(divergences)
            if breakout_detected:
                self.stats['breakouts_detected'] += 1
            self.stats['analyses_performed'] += 1
            self.stats['last_update'] = datetime.now()
            
            # Cache CVD series
            cache_key = f"{symbol}_{timeframe}"
            self.cvd_cache[cache_key] = cvd_series
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"❌ Error in CVD analysis for {symbol}: {e}")
            return self._get_default_analysis(symbol, timeframe)
    
    def _calculate_cvd_from_trades(
        self,
        trades: List[Trade],
        df: pd.DataFrame
    ) -> pd.Series:
        """Calculate CVD from actual trade data"""
        try:
            # Create trades dataframe
            trade_data = []
            for trade in trades:
                delta = trade.volume if trade.is_buy else -trade.volume
                trade_data.append({
                    'timestamp': trade.timestamp,
                    'delta': delta
                })
            
            trades_df = pd.DataFrame(trade_data)
            
            # Align with candle data
            if 'timestamp' in df.columns:
                # Resample trades to match candle timeframe
                trades_df = trades_df.set_index('timestamp')
                aggregated = trades_df.resample(self._infer_frequency(df)).sum()
                
                # Calculate cumulative delta
                cvd = aggregated['delta'].cumsum()
                
                # Align with original dataframe
                cvd = cvd.reindex(df.index, method='ffill').fillna(0)
            else:
                # Simple cumulative sum
                cvd = pd.Series([t['delta'] for t in trade_data]).cumsum()
            
            return cvd
            
        except Exception as e:
            self.logger.error(f"Error calculating CVD from trades: {e}")
            return pd.Series([0] * len(df))
    
    def _approximate_cvd_from_ohlcv(self, df: pd.DataFrame) -> pd.Series:
        """
        Approximate CVD from OHLCV data when trade data not available
        
        Uses price action and volume to estimate buy/sell pressure:
        - Green candle (close > open) = buy volume
        - Red candle (close < open) = sell volume
        - Weight by candle range and close position
        """
        try:
            # Calculate delta for each candle
            delta = np.zeros(len(df))
            
            for i in range(len(df)):
                close = df['close'].iloc[i]
                open_price = df['open'].iloc[i]
                high = df['high'].iloc[i]
                low = df['low'].iloc[i]
                volume = df['volume'].iloc[i]
                
                # Calculate where close is in the range
                range_size = high - low
                if range_size > 0:
                    close_position = (close - low) / range_size
                else:
                    close_position = 0.5
                
                # Estimate buy/sell split
                # close_position near 1 = more buying
                # close_position near 0 = more selling
                buy_volume = volume * close_position
                sell_volume = volume * (1 - close_position)
                
                delta[i] = buy_volume - sell_volume
            
            # Cumulative sum = CVD
            cvd = pd.Series(delta).cumsum()
            
            return cvd
            
        except Exception as e:
            self.logger.error(f"Error approximating CVD: {e}")
            return pd.Series([0] * len(df))
    
    def _infer_frequency(self, df: pd.DataFrame) -> str:
        """Infer timeframe frequency from dataframe"""
        if 'timestamp' in df.columns and len(df) > 1:
            time_diff = df['timestamp'].iloc[1] - df['timestamp'].iloc[0]
            if time_diff.total_seconds() <= 60:
                return '1min'
            elif time_diff.total_seconds() <= 300:
                return '5min'
            elif time_diff.total_seconds() <= 900:
                return '15min'
            elif time_diff.total_seconds() <= 3600:
                return '1H'
            else:
                return '1D'
        return '1H'  # Default
    
    def _determine_cvd_trend(self, cvd_series: pd.Series) -> str:
        """Determine CVD trend direction"""
        try:
            # Use last 20 periods
            recent_cvd = cvd_series.tail(20)
            
            # Calculate linear regression slope
            x = np.arange(len(recent_cvd))
            y = recent_cvd.values
            
            if len(x) > 1:
                slope = np.polyfit(x, y, 1)[0]
                
                # Normalize slope by CVD magnitude
                avg_cvd = abs(recent_cvd.mean())
                if avg_cvd > 0:
                    normalized_slope = slope / avg_cvd
                else:
                    normalized_slope = 0
                
                if normalized_slope > 0.02:
                    return 'bullish'
                elif normalized_slope < -0.02:
                    return 'bearish'
            
            return 'neutral'
            
        except Exception:
            return 'neutral'
    
    def _calculate_cvd_momentum(self, cvd_series: pd.Series) -> float:
        """Calculate CVD momentum (rate of change)"""
        try:
            if len(cvd_series) < 10:
                return 0.0
            
            # CVD change over last 10 periods
            current = cvd_series.iloc[-1]
            past = cvd_series.iloc[-10]
            
            if abs(past) > 0:
                momentum = (current - past) / abs(past)
            else:
                momentum = 0.0
            
            return momentum
            
        except Exception:
            return 0.0
    
    async def _detect_divergences(
        self,
        df: pd.DataFrame
    ) -> List[CVDDivergenceSignal]:
        """Detect price-CVD divergences"""
        divergences = []
        
        try:
            if len(df) < self.divergence_window:
                return divergences
            
            # Find price peaks and troughs
            price_peaks = self._find_peaks(df['close'].values)
            price_troughs = self._find_troughs(df['close'].values)
            
            # Find CVD peaks and troughs
            cvd_peaks = self._find_peaks(df['cvd'].values)
            cvd_troughs = self._find_troughs(df['cvd'].values)
            
            # Check for regular bullish divergence
            # Price: lower lows, CVD: higher lows
            if len(price_troughs) >= 2 and len(cvd_troughs) >= 2:
                for i in range(len(price_troughs) - 1):
                    p1, p2 = price_troughs[i], price_troughs[i + 1]
                    
                    # Find corresponding CVD troughs
                    cvd_t1 = self._find_nearest_index(cvd_troughs, p1)
                    cvd_t2 = self._find_nearest_index(cvd_troughs, p2)
                    
                    if cvd_t1 is not None and cvd_t2 is not None:
                        price_lower = df['close'].iloc[p2] < df['close'].iloc[p1]
                        cvd_higher = df['cvd'].iloc[cvd_t2] > df['cvd'].iloc[cvd_t1]
                        
                        if price_lower and cvd_higher:
                            strength = self._calculate_divergence_strength(
                                df['close'].iloc[p1], df['close'].iloc[p2],
                                df['cvd'].iloc[cvd_t1], df['cvd'].iloc[cvd_t2]
                            )
                            
                            divergence = CVDDivergenceSignal(
                                divergence_type=CVDDivergence.BULLISH,
                                price_points=[(p1, df['close'].iloc[p1]), (p2, df['close'].iloc[p2])],
                                cvd_points=[(cvd_t1, df['cvd'].iloc[cvd_t1]), (cvd_t2, df['cvd'].iloc[cvd_t2])],
                                strength=strength,
                                confidence=min(0.9, 0.6 + strength * 0.3),
                                timestamp=df['timestamp'].iloc[p2] if 'timestamp' in df.columns else datetime.now(),
                                metadata={'type': 'regular', 'direction': 'bullish'}
                            )
                            divergences.append(divergence)
            
            # Check for regular bearish divergence
            # Price: higher highs, CVD: lower highs
            if len(price_peaks) >= 2 and len(cvd_peaks) >= 2:
                for i in range(len(price_peaks) - 1):
                    p1, p2 = price_peaks[i], price_peaks[i + 1]
                    
                    cvd_p1 = self._find_nearest_index(cvd_peaks, p1)
                    cvd_p2 = self._find_nearest_index(cvd_peaks, p2)
                    
                    if cvd_p1 is not None and cvd_p2 is not None:
                        price_higher = df['close'].iloc[p2] > df['close'].iloc[p1]
                        cvd_lower = df['cvd'].iloc[cvd_p2] < df['cvd'].iloc[cvd_p1]
                        
                        if price_higher and cvd_lower:
                            strength = self._calculate_divergence_strength(
                                df['close'].iloc[p1], df['close'].iloc[p2],
                                df['cvd'].iloc[cvd_p1], df['cvd'].iloc[cvd_p2]
                            )
                            
                            divergence = CVDDivergenceSignal(
                                divergence_type=CVDDivergence.BEARISH,
                                price_points=[(p1, df['close'].iloc[p1]), (p2, df['close'].iloc[p2])],
                                cvd_points=[(cvd_p1, df['cvd'].iloc[cvd_p1]), (cvd_p2, df['cvd'].iloc[cvd_p2])],
                                strength=strength,
                                confidence=min(0.9, 0.6 + strength * 0.3),
                                timestamp=df['timestamp'].iloc[p2] if 'timestamp' in df.columns else datetime.now(),
                                metadata={'type': 'regular', 'direction': 'bearish'}
                            )
                            divergences.append(divergence)
            
            return divergences
            
        except Exception as e:
            self.logger.error(f"Error detecting divergences: {e}")
            return divergences
    
    def _find_peaks(self, data: np.ndarray, window: int = 5) -> List[int]:
        """Find local peaks in data"""
        peaks = []
        for i in range(window, len(data) - window):
            if all(data[i] >= data[i-window:i]) and all(data[i] >= data[i+1:i+window+1]):
                peaks.append(i)
        return peaks
    
    def _find_troughs(self, data: np.ndarray, window: int = 5) -> List[int]:
        """Find local troughs in data"""
        troughs = []
        for i in range(window, len(data) - window):
            if all(data[i] <= data[i-window:i]) and all(data[i] <= data[i+1:i+window+1]):
                troughs.append(i)
        return troughs
    
    def _find_nearest_index(self, indices: List[int], target: int, max_distance: int = 10) -> Optional[int]:
        """Find nearest index in list to target"""
        nearest = None
        min_distance = max_distance
        
        for idx in indices:
            distance = abs(idx - target)
            if distance < min_distance:
                min_distance = distance
                nearest = idx
        
        return nearest
    
    def _calculate_divergence_strength(
        self,
        price1: float,
        price2: float,
        cvd1: float,
        cvd2: float
    ) -> float:
        """Calculate divergence strength (0-1)"""
        try:
            price_change = abs((price2 - price1) / price1) if price1 != 0 else 0
            cvd_change = abs((cvd2 - cvd1) / cvd1) if cvd1 != 0 else 0
            
            # Strength is combination of both changes
            strength = min(1.0, (price_change + cvd_change) / 2)
            
            return strength
        except Exception:
            return 0.0
    
    def _identify_cvd_levels(
        self,
        cvd_series: pd.Series
    ) -> Tuple[List[CVDLevel], List[CVDLevel]]:
        """Identify CVD support and resistance levels"""
        support_levels = []
        resistance_levels = []
        
        try:
            # Find turning points in CVD
            peaks = self._find_peaks(cvd_series.values)
            troughs = self._find_troughs(cvd_series.values)
            
            # Cluster peaks into resistance levels
            if peaks:
                peak_values = [cvd_series.iloc[p] for p in peaks]
                resistance_clusters = self._cluster_levels(peak_values)
                
                for level_value, indices in resistance_clusters.items():
                    level = CVDLevel(
                        level=level_value,
                        strength=len(indices) / len(peaks),
                        touch_count=len(indices),
                        last_touch=datetime.now(),
                        level_type='resistance'
                    )
                    resistance_levels.append(level)
            
            # Cluster troughs into support levels
            if troughs:
                trough_values = [cvd_series.iloc[t] for t in troughs]
                support_clusters = self._cluster_levels(trough_values)
                
                for level_value, indices in support_clusters.items():
                    level = CVDLevel(
                        level=level_value,
                        strength=len(indices) / len(troughs),
                        touch_count=len(indices),
                        last_touch=datetime.now(),
                        level_type='support'
                    )
                    support_levels.append(level)
            
            return support_levels, resistance_levels
            
        except Exception as e:
            self.logger.error(f"Error identifying CVD levels: {e}")
            return support_levels, resistance_levels
    
    def _cluster_levels(self, values: List[float]) -> Dict[float, List[int]]:
        """Cluster similar values together"""
        if not values:
            return {}
        
        clusters = {}
        threshold = np.std(values) * 0.5 if len(values) > 1 else 0
        
        for i, value in enumerate(values):
            found_cluster = False
            
            for cluster_key in clusters:
                if abs(value - cluster_key) <= threshold:
                    clusters[cluster_key].append(i)
                    found_cluster = True
                    break
            
            if not found_cluster:
                clusters[value] = [i]
        
        return clusters
    
    def _detect_cvd_breakout(
        self,
        cvd_series: pd.Series,
        support_levels: List[CVDLevel],
        resistance_levels: List[CVDLevel]
    ) -> bool:
        """Detect CVD breakout from support/resistance"""
        try:
            if len(cvd_series) < 2:
                return False
            
            current_cvd = cvd_series.iloc[-1]
            previous_cvd = cvd_series.iloc[-2]
            
            # Check resistance breakout
            for level in resistance_levels:
                if previous_cvd < level.level <= current_cvd:
                    return True
            
            # Check support breakdown
            for level in support_levels:
                if previous_cvd > level.level >= current_cvd:
                    return True
            
            return False
            
        except Exception:
            return False
    
    async def _generate_cvd_signals(
        self,
        df: pd.DataFrame,
        divergences: List[CVDDivergenceSignal],
        breakout_detected: bool,
        cvd_trend: str
    ) -> List[Dict[str, Any]]:
        """Generate trading signals from CVD analysis"""
        signals = []
        
        try:
            # Divergence signals
            for div in divergences:
                if div.confidence >= self.min_confidence:
                    signal = {
                        'type': 'cvd_divergence',
                        'direction': 'bullish' if div.divergence_type == CVDDivergence.BULLISH else 'bearish',
                        'confidence': div.confidence,
                        'strength': div.strength,
                        'divergence_type': div.divergence_type.value,
                        'timestamp': div.timestamp,
                        'reasoning': f"CVD {div.divergence_type.value} divergence detected",
                        'priority': 'high' if div.confidence > 0.8 else 'medium'
                    }
                    signals.append(signal)
            
            # CVD trend signal
            if cvd_trend != 'neutral':
                signals.append({
                    'type': 'cvd_trend',
                    'direction': cvd_trend,
                    'confidence': 0.7,
                    'reasoning': f"CVD trend is {cvd_trend}",
                    'priority': 'medium'
                })
            
            # Breakout signal
            if breakout_detected:
                signals.append({
                    'type': 'cvd_breakout',
                    'direction': cvd_trend if cvd_trend != 'neutral' else 'unknown',
                    'confidence': 0.75,
                    'reasoning': "CVD breakout detected",
                    'priority': 'high'
                })
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Error generating CVD signals: {e}")
            return signals
    
    def _calculate_overall_confidence(
        self,
        divergences: List[CVDDivergenceSignal],
        cvd_trend: str,
        breakout_detected: bool
    ) -> float:
        """Calculate overall CVD analysis confidence"""
        try:
            confidence = 0.5
            
            # High-confidence divergences
            strong_divergences = [d for d in divergences if d.confidence > 0.8]
            if strong_divergences:
                confidence += 0.3
            elif divergences:
                confidence += 0.15
            
            # CVD trend confirmation
            if cvd_trend != 'neutral':
                confidence += 0.1
            
            # Breakout confirmation
            if breakout_detected:
                confidence += 0.1
            
            return min(0.95, confidence)
            
        except Exception:
            return 0.5
    
    def _get_default_analysis(self, symbol: str, timeframe: str) -> CVDAnalysis:
        """Get default analysis when insufficient data"""
        return CVDAnalysis(
            symbol=symbol,
            timeframe=timeframe,
            timestamp=datetime.now(),
            current_cvd=0.0,
            cvd_trend='neutral',
            cvd_momentum=0.0,
            divergences=[],
            support_levels=[],
            resistance_levels=[],
            breakout_detected=False,
            overall_confidence=0.0,
            cvd_signals=[],
            metadata={'error': 'Insufficient data for analysis'}
        )
    
    async def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        return {
            'stats': self.stats,
            'config': self.config,
            'cached_symbols': list(self.cvd_cache.keys()),
            'last_update': datetime.now().isoformat()
        }

