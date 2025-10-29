#!/usr/bin/env python3
"""
Trend Performance Optimizer for AlphaPulse
Phase 4: Precomputed Trend Values, Caching, and Optimized Calculations
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import threading
from collections import defaultdict

logger = logging.getLogger(__name__)

@dataclass
class PrecomputedTrendData:
    """Precomputed trend data for a timeframe"""
    timeframe: str
    symbol: str
    timestamp: datetime
    ema_20: float
    ema_50: float
    ema_200: float
    adx_value: float
    rsi_value: float
    macd_signal: str
    hull_fast: float
    hull_medium: float
    hull_slow: float
    trend_direction: str
    trend_strength: str
    ema_alignment: bool
    trend_confidence: float
    cache_key: str

@dataclass
class OptimizedTrendResult:
    """Optimized trend analysis result"""
    current_tf_data: PrecomputedTrendData
    higher_tf_data: Optional[PrecomputedTrendData] = None
    lower_tf_data: Optional[PrecomputedTrendData] = None
    analysis_time: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0

class TrendPerformanceOptimizer:
    """
    Performance optimizer for trend analysis with precomputation and caching
    """
    
    def __init__(self):
        # Cache for precomputed trend data
        self.trend_cache: Dict[str, PrecomputedTrendData] = {}
        self.cache_ttl = 300  # 5 minutes TTL
        self.last_cache_cleanup = datetime.now()
        
        # Thread lock for cache operations
        self.cache_lock = threading.Lock()
        
        # Performance metrics
        self.performance_metrics = {
            'total_analyses': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'average_analysis_time': 0.0,
            'total_analysis_time': 0.0
        }
        
        # Precomputed indicators cache
        self.indicators_cache: Dict[str, Dict[str, Any]] = {}
        
        # Batch processing queue
        self.batch_queue: List[Tuple[str, str, pd.DataFrame]] = []
        self.batch_size = 10
        self.last_batch_process = datetime.now()
        
        logger.info("🚀 Trend Performance Optimizer initialized")
    
    def precompute_trend_values(
        self, 
        data_dict: Dict[str, pd.DataFrame],
        symbols: List[str],
        timeframes: List[str]
    ) -> Dict[str, Dict[str, PrecomputedTrendData]]:
        """
        Precompute trend values for all symbols and timeframes
        
        Args:
            data_dict: Dictionary of DataFrames by symbol and timeframe
            symbols: List of symbols to process
            timeframes: List of timeframes to process
            
        Returns:
            Dictionary of precomputed trend data
        """
        start_time = datetime.now()
        precomputed_data = {}
        
        try:
            for symbol in symbols:
                precomputed_data[symbol] = {}
                
                for timeframe in timeframes:
                    cache_key = f"{symbol}_{timeframe}"
                    
                    # Check if data exists
                    if cache_key not in data_dict:
                        logger.warning(f"No data found for {cache_key}")
                        continue
                    
                    df = data_dict[cache_key]
                    
                    if len(df) < 50:
                        logger.warning(f"Insufficient data for {cache_key}")
                        continue
                    
                    # Precompute all indicators
                    trend_data = self._precompute_single_timeframe(df, symbol, timeframe)
                    
                    if trend_data:
                        precomputed_data[symbol][timeframe] = trend_data
                        
                        # Store in cache
                        with self.cache_lock:
                            self.trend_cache[cache_key] = trend_data
            
            # Clean up old cache entries
            self._cleanup_cache()
            
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"✅ Precomputed trend values for {len(symbols)} symbols, {len(timeframes)} timeframes in {processing_time:.3f}s")
            
            return precomputed_data
            
        except Exception as e:
            logger.error(f"Error precomputing trend values: {e}")
            return {}
    
    def get_optimized_trend_analysis(
        self,
        symbol: str,
        current_timeframe: str,
        current_data: pd.DataFrame,
        higher_timeframe: Optional[str] = None,
        higher_data: Optional[pd.DataFrame] = None,
        lower_timeframe: Optional[str] = None,
        lower_data: Optional[pd.DataFrame] = None
    ) -> OptimizedTrendResult:
        """
        Get optimized trend analysis with caching
        
        Args:
            symbol: Symbol being analyzed
            current_timeframe: Current timeframe
            current_data: Current timeframe data
            higher_timeframe: Higher timeframe (optional)
            higher_data: Higher timeframe data (optional)
            lower_timeframe: Lower timeframe (optional)
            lower_data: Lower timeframe data (optional)
            
        Returns:
            OptimizedTrendResult with cached data
        """
        start_time = datetime.now()
        cache_hits = 0
        cache_misses = 0
        
        try:
            # Get current timeframe data (from cache or compute)
            current_tf_data = self._get_cached_or_compute_trend_data(
                symbol, current_timeframe, current_data
            )
            if current_tf_data:
                cache_hits += 1
            else:
                cache_misses += 1
            
            # Get higher timeframe data
            higher_tf_data = None
            if higher_timeframe and higher_data is not None:
                higher_tf_data = self._get_cached_or_compute_trend_data(
                    symbol, higher_timeframe, higher_data
                )
                if higher_tf_data:
                    cache_hits += 1
                else:
                    cache_misses += 1
            
            # Get lower timeframe data
            lower_tf_data = None
            if lower_timeframe and lower_data is not None:
                lower_tf_data = self._get_cached_or_compute_trend_data(
                    symbol, lower_timeframe, lower_data
                )
                if lower_tf_data:
                    cache_hits += 1
                else:
                    cache_misses += 1
            
            analysis_time = (datetime.now() - start_time).total_seconds()
            
            # Update performance metrics
            self._update_performance_metrics(analysis_time, cache_hits, cache_misses)
            
            return OptimizedTrendResult(
                current_tf_data=current_tf_data,
                higher_tf_data=higher_tf_data,
                lower_tf_data=lower_tf_data,
                analysis_time=analysis_time,
                cache_hits=cache_hits,
                cache_misses=cache_misses
            )
            
        except Exception as e:
            logger.error(f"Error in optimized trend analysis: {e}")
            return OptimizedTrendResult(
                current_tf_data=self._get_default_trend_data(symbol, current_timeframe),
                analysis_time=0.0,
                cache_misses=1
            )
    
    def batch_process_trend_data(
        self,
        data_batch: List[Tuple[str, str, pd.DataFrame]]
    ) -> Dict[str, PrecomputedTrendData]:
        """
        Process a batch of trend data for efficiency
        
        Args:
            data_batch: List of (symbol, timeframe, data) tuples
            
        Returns:
            Dictionary of precomputed trend data
        """
        start_time = datetime.now()
        results = {}
        
        try:
            for symbol, timeframe, df in data_batch:
                if len(df) < 50:
                    continue
                
                trend_data = self._precompute_single_timeframe(df, symbol, timeframe)
                if trend_data:
                    results[f"{symbol}_{timeframe}"] = trend_data
                    
                    # Store in cache
                    with self.cache_lock:
                        self.trend_cache[f"{symbol}_{timeframe}"] = trend_data
            
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"✅ Batch processed {len(data_batch)} trend datasets in {processing_time:.3f}s")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in batch processing: {e}")
            return {}
    
    def get_cached_trend_data(self, symbol: str, timeframe: str) -> Optional[PrecomputedTrendData]:
        """
        Get cached trend data if available and valid
        
        Args:
            symbol: Symbol to get data for
            timeframe: Timeframe to get data for
            
        Returns:
            Cached trend data or None if not available/expired
        """
        cache_key = f"{symbol}_{timeframe}"
        
        with self.cache_lock:
            if cache_key in self.trend_cache:
                cached_data = self.trend_cache[cache_key]
                
                # Check if cache is still valid
                if datetime.now() - cached_data.timestamp < timedelta(seconds=self.cache_ttl):
                    return cached_data
                else:
                    # Remove expired cache entry
                    del self.trend_cache[cache_key]
            
            return None
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        with self.cache_lock:
            metrics = self.performance_metrics.copy()
            metrics['cache_size'] = len(self.trend_cache)
            metrics['cache_hit_rate'] = (
                metrics['cache_hits'] / (metrics['cache_hits'] + metrics['cache_misses'])
                if (metrics['cache_hits'] + metrics['cache_misses']) > 0 else 0.0
            )
            metrics['average_analysis_time'] = (
                metrics['total_analysis_time'] / metrics['total_analyses']
                if metrics['total_analyses'] > 0 else 0.0
            )
            return metrics
    
    def clear_cache(self):
        """Clear all cached data"""
        with self.cache_lock:
            self.trend_cache.clear()
            self.indicators_cache.clear()
            logger.info("🗑️ Trend cache cleared")
    
    def _precompute_single_timeframe(
        self, 
        df: pd.DataFrame, 
        symbol: str, 
        timeframe: str
    ) -> Optional[PrecomputedTrendData]:
        """Precompute trend data for a single timeframe"""
        try:
            # Calculate all indicators efficiently
            indicators = self._calculate_all_indicators(df)
            
            if not indicators:
                return None
            
            # Create precomputed trend data
            trend_data = PrecomputedTrendData(
                timeframe=timeframe,
                symbol=symbol,
                timestamp=datetime.now(),
                ema_20=indicators['ema_20'],
                ema_50=indicators['ema_50'],
                ema_200=indicators['ema_200'],
                adx_value=indicators['adx'],
                rsi_value=indicators['rsi'],
                macd_signal=indicators['macd_signal'],
                hull_fast=indicators['hull_fast'],
                hull_medium=indicators['hull_medium'],
                hull_slow=indicators['hull_slow'],
                trend_direction=indicators['trend_direction'],
                trend_strength=indicators['trend_strength'],
                ema_alignment=indicators['ema_alignment'],
                trend_confidence=indicators['trend_confidence'],
                cache_key=f"{symbol}_{timeframe}"
            )
            
            return trend_data
            
        except Exception as e:
            logger.error(f"Error precomputing trend data for {symbol}_{timeframe}: {e}")
            return None
    
    def _calculate_all_indicators(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Calculate all technical indicators efficiently"""
        try:
            if len(df) < 50:
                return None
            
            # Vectorized calculations for efficiency
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            
            # EMAs
            ema_20 = self._calculate_ema_vectorized(close, 20)
            ema_50 = self._calculate_ema_vectorized(close, 50)
            ema_200 = self._calculate_ema_vectorized(close, 200) if len(close) >= 200 else ema_50
            
            # ADX
            adx = self._calculate_adx_vectorized(high, low, close)
            
            # RSI
            rsi = self._calculate_rsi_vectorized(close)
            
            # MACD
            macd_signal = self._calculate_macd_signal_vectorized(close)
            
            # Hull MAs
            hull_fast = self._calculate_hull_ma_vectorized(close, 9)
            hull_medium = self._calculate_hull_ma_vectorized(close, 18)
            hull_slow = self._calculate_hull_ma_vectorized(close, 36)
            
            # Trend analysis
            current_price = close[-1]
            trend_direction = self._determine_trend_direction(current_price, ema_20, ema_50, ema_200, adx)
            trend_strength = self._determine_trend_strength(adx)
            ema_alignment = self._check_ema_alignment(ema_20, ema_50, ema_200, trend_direction)
            trend_confidence = self._calculate_trend_confidence(trend_direction, trend_strength, ema_alignment, adx, rsi)
            
            return {
                'ema_20': ema_20,
                'ema_50': ema_50,
                'ema_200': ema_200,
                'adx': adx,
                'rsi': rsi,
                'macd_signal': macd_signal,
                'hull_fast': hull_fast,
                'hull_medium': hull_medium,
                'hull_slow': hull_slow,
                'trend_direction': trend_direction,
                'trend_strength': trend_strength,
                'ema_alignment': ema_alignment,
                'trend_confidence': trend_confidence
            }
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            return None
    
    def _get_cached_or_compute_trend_data(
        self, 
        symbol: str, 
        timeframe: str, 
        df: pd.DataFrame
    ) -> Optional[PrecomputedTrendData]:
        """Get cached trend data or compute if not available"""
        # Try to get from cache first
        cached_data = self.get_cached_trend_data(symbol, timeframe)
        if cached_data:
            return cached_data
        
        # Compute if not in cache
        return self._precompute_single_timeframe(df, symbol, timeframe)
    
    def _update_performance_metrics(self, analysis_time: float, cache_hits: int, cache_misses: int):
        """Update performance metrics"""
        with self.cache_lock:
            self.performance_metrics['total_analyses'] += 1
            self.performance_metrics['cache_hits'] += cache_hits
            self.performance_metrics['cache_misses'] += cache_misses
            self.performance_metrics['total_analysis_time'] += analysis_time
    
    def _cleanup_cache(self):
        """Clean up expired cache entries"""
        current_time = datetime.now()
        
        # Only cleanup every 5 minutes
        if (current_time - self.last_cache_cleanup).total_seconds() < 300:
            return
        
        with self.cache_lock:
            expired_keys = []
            for cache_key, cached_data in self.trend_cache.items():
                if current_time - cached_data.timestamp > timedelta(seconds=self.cache_ttl):
                    expired_keys.append(cache_key)
            
            for key in expired_keys:
                del self.trend_cache[key]
            
            if expired_keys:
                logger.info(f"🧹 Cleaned up {len(expired_keys)} expired cache entries")
            
            self.last_cache_cleanup = current_time
    
    # Vectorized calculation methods for performance
    def _calculate_ema_vectorized(self, data: np.ndarray, period: int) -> float:
        """Calculate EMA using vectorized operations"""
        try:
            alpha = 2.0 / (period + 1)
            ema = data[0]
            for i in range(1, len(data)):
                ema = alpha * data[i] + (1 - alpha) * ema
            return ema
        except Exception as e:
            logger.error(f"Error calculating EMA: {e}")
            return data[-1] if len(data) > 0 else 0.0
    
    def _calculate_adx_vectorized(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> float:
        """Calculate ADX using vectorized operations"""
        try:
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
            
            # Smooth using EMA
            tr_smooth = self._calculate_ema_vectorized(tr, period)
            dm_plus_smooth = self._calculate_ema_vectorized(dm_plus, period)
            dm_minus_smooth = self._calculate_ema_vectorized(dm_minus, period)
            
            # Calculate DI+ and DI-
            di_plus = (dm_plus_smooth / tr_smooth) * 100 if tr_smooth > 0 else 0
            di_minus = (dm_minus_smooth / tr_smooth) * 100 if tr_smooth > 0 else 0
            
            # Calculate DX and ADX
            dx = (abs(di_plus - di_minus) / (di_plus + di_minus)) * 100 if (di_plus + di_minus) > 0 else 0
            adx = self._calculate_ema_vectorized(np.array([dx]), period)
            
            return adx
        except Exception as e:
            logger.error(f"Error calculating ADX: {e}")
            return 0.0
    
    def _calculate_rsi_vectorized(self, data: np.ndarray, period: int = 14) -> float:
        """Calculate RSI using vectorized operations"""
        try:
            delta = np.diff(data)
            gain = np.where(delta > 0, delta, 0)
            loss = np.where(delta < 0, -delta, 0)
            
            avg_gain = self._calculate_ema_vectorized(gain, period)
            avg_loss = self._calculate_ema_vectorized(loss, period)
            
            rs = avg_gain / avg_loss if avg_loss > 0 else 0
            rsi = 100 - (100 / (1 + rs))
            
            return rsi
        except Exception as e:
            logger.error(f"Error calculating RSI: {e}")
            return 50.0
    
    def _calculate_macd_signal_vectorized(self, data: np.ndarray) -> str:
        """Calculate MACD signal using vectorized operations"""
        try:
            ema_12 = self._calculate_ema_vectorized(data, 12)
            ema_26 = self._calculate_ema_vectorized(data, 26)
            macd = ema_12 - ema_26
            signal = self._calculate_ema_vectorized(np.array([macd]), 9)
            
            if macd > signal:
                return "bullish"
            elif macd < signal:
                return "bearish"
            else:
                return "neutral"
        except Exception as e:
            logger.error(f"Error calculating MACD: {e}")
            return "neutral"
    
    def _calculate_hull_ma_vectorized(self, data: np.ndarray, period: int) -> float:
        """Calculate Hull MA using vectorized operations"""
        try:
            # Calculate WMA for half period
            half_period = period // 2
            wma_half = self._calculate_wma_vectorized(data, half_period)
            wma_full = self._calculate_wma_vectorized(data, period)
            
            # Calculate Hull MA
            hull_raw = 2 * wma_half - wma_full
            hull_ma = self._calculate_wma_vectorized(np.array([hull_raw]), int(np.sqrt(period)))
            
            return hull_ma
        except Exception as e:
            logger.error(f"Error calculating Hull MA: {e}")
            return data[-1] if len(data) > 0 else 0.0
    
    def _calculate_wma_vectorized(self, data: np.ndarray, period: int) -> float:
        """Calculate Weighted Moving Average using vectorized operations"""
        try:
            if len(data) < period:
                return data[-1] if len(data) > 0 else 0.0
            
            weights = np.arange(1, period + 1)
            recent_data = data[-period:]
            
            wma = np.sum(recent_data * weights) / np.sum(weights)
            return wma
        except Exception as e:
            logger.error(f"Error calculating WMA: {e}")
            return data[-1] if len(data) > 0 else 0.0
    
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
    
    def _get_default_trend_data(self, symbol: str, timeframe: str) -> PrecomputedTrendData:
        """Get default trend data"""
        return PrecomputedTrendData(
            timeframe=timeframe,
            symbol=symbol,
            timestamp=datetime.now(),
            ema_20=0.0,
            ema_50=0.0,
            ema_200=0.0,
            adx_value=0.0,
            rsi_value=50.0,
            macd_signal="neutral",
            hull_fast=0.0,
            hull_medium=0.0,
            hull_slow=0.0,
            trend_direction="neutral",
            trend_strength="weak",
            ema_alignment=False,
            trend_confidence=0.5,
            cache_key=f"{symbol}_{timeframe}"
        )
