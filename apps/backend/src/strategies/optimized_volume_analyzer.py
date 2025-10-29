#!/usr/bin/env python3
"""
Optimized Volume Analyzer for AlphaPulse
Real-time vs Historical efficiency with vectorized operations
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'data'))

from src.data.volume_analyzer import VolumeAnalyzer, VolumePattern, VolumePatternType, VolumeStrength

logger = logging.getLogger(__name__)

class AnalysisMode(Enum):
    """Analysis modes for volume analysis"""
    REAL_TIME = "real_time"
    HISTORICAL = "historical"
    BACKTESTING = "backtesting"

@dataclass
class VolumeMetrics:
    """Optimized volume metrics"""
    volume_ratio: float
    volume_spike_ratio: float
    volume_trend_alignment: float
    volume_consistency: float
    volume_momentum: float
    volume_volatility: float
    avg_volume: float
    volume_peaks: List[int]
    volume_troughs: List[int]
    volume_pattern: Optional[str] = None
    confidence_score: float = 0.0

class OptimizedVolumeAnalyzer:
    """
    Optimized volume analyzer with real-time and historical efficiency
    """
    
    def __init__(self, mode: AnalysisMode = AnalysisMode.REAL_TIME):
        self.mode = mode
        self.volume_analyzer = VolumeAnalyzer()
        
        # Rolling window configurations
        self.rolling_windows = {
            'short': 5,
            'medium': 20,
            'long': 50
        }
        
        # Precomputed metrics cache
        self.metrics_cache: Dict[str, Dict[str, Any]] = {}
        self.cache_ttl = timedelta(minutes=5)  # 5 minutes TTL
        self.last_cache_update: Dict[str, datetime] = {}
        
        # Vectorized calculation settings
        self.use_vectorized = True
        self.parallel_processing = True
        
        logger.info(f"🚀 Optimized Volume Analyzer initialized in {mode.value} mode")
    
    def analyze_volume_optimized(
        self, 
        df: pd.DataFrame, 
        symbol: str = "UNKNOWN",
        timeframe: str = "1h"
    ) -> VolumeMetrics:
        """
        Optimized volume analysis with mode-specific optimizations
        
        Args:
            df: DataFrame with OHLCV data
            symbol: Symbol for analysis
            timeframe: Timeframe for analysis
            
        Returns:
            VolumeMetrics object with optimized calculations
        """
        try:
            if len(df) < 20:
                logger.warning("Insufficient data for volume analysis (minimum 20 candles required)")
                return self._get_default_metrics()
            
            # Check cache for historical mode
            cache_key = f"{symbol}_{timeframe}_{len(df)}"
            if self.mode == AnalysisMode.HISTORICAL and self._is_cache_valid(cache_key):
                cached_metrics = self.metrics_cache.get(cache_key)
                if cached_metrics:
                    logger.info(f"📋 Using cached volume metrics for {symbol}")
                    return VolumeMetrics(**cached_metrics)
            
            # Perform optimized analysis based on mode
            if self.mode == AnalysisMode.REAL_TIME:
                metrics = self._analyze_volume_real_time(df)
            elif self.mode == AnalysisMode.HISTORICAL:
                metrics = self._analyze_volume_historical(df)
            elif self.mode == AnalysisMode.BACKTESTING:
                metrics = self._analyze_volume_backtesting(df)
            else:
                metrics = self._analyze_volume_real_time(df)
            
            # Cache results for historical mode
            if self.mode == AnalysisMode.HISTORICAL:
                self._cache_metrics(cache_key, metrics)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error in optimized volume analysis: {e}")
            return self._get_default_metrics()
    
    def _analyze_volume_real_time(self, df: pd.DataFrame) -> VolumeMetrics:
        """
        Real-time volume analysis using rolling averages
        """
        try:
            # Use rolling averages for real-time efficiency
            volume_series = df['volume']
            
            # Calculate rolling averages
            rolling_short = volume_series.rolling(window=self.rolling_windows['short']).mean()
            rolling_medium = volume_series.rolling(window=self.rolling_windows['medium']).mean()
            rolling_long = volume_series.rolling(window=self.rolling_windows['long']).mean()
            
            # Current volume metrics
            current_volume = volume_series.iloc[-1]
            current_short_avg = rolling_short.iloc[-1]
            current_medium_avg = rolling_medium.iloc[-1]
            current_long_avg = rolling_long.iloc[-1]
            
            # Volume ratios
            volume_ratio = current_volume / current_medium_avg if current_medium_avg > 0 else 1.0
            volume_spike_ratio = current_volume / current_short_avg if current_short_avg > 0 else 1.0
            
            # Volume trend alignment (short vs long term)
            volume_trend_alignment = (
                (current_short_avg - current_long_avg) / current_long_avg 
                if current_long_avg > 0 else 0.0
            )
            
            # Volume consistency (coefficient of variation)
            recent_volume = volume_series.tail(self.rolling_windows['medium'])
            volume_consistency = 1.0 - (recent_volume.std() / recent_volume.mean()) if recent_volume.mean() > 0 else 0.0
            volume_consistency = max(0.0, min(1.0, volume_consistency))
            
            # Volume momentum
            volume_momentum = (
                (current_volume - volume_series.iloc[-5]) / volume_series.iloc[-5]
                if volume_series.iloc[-5] > 0 else 0.0
            )
            
            # Volume volatility
            volume_volatility = recent_volume.std() / recent_volume.mean() if recent_volume.mean() > 0 else 0.0
            
            # Find volume peaks and troughs
            volume_peaks = self._find_volume_extremes(volume_series, 'peaks')
            volume_troughs = self._find_volume_extremes(volume_series, 'troughs')
            
            # Determine volume pattern
            volume_pattern = self._classify_volume_pattern(
                volume_ratio, volume_spike_ratio, volume_trend_alignment, volume_consistency
            )
            
            # Calculate confidence score
            confidence_score = self._calculate_volume_confidence(
                volume_ratio, volume_spike_ratio, volume_trend_alignment, volume_consistency
            )
            
            return VolumeMetrics(
                volume_ratio=volume_ratio,
                volume_spike_ratio=volume_spike_ratio,
                volume_trend_alignment=volume_trend_alignment,
                volume_consistency=volume_consistency,
                volume_momentum=volume_momentum,
                volume_volatility=volume_volatility,
                avg_volume=current_medium_avg,
                volume_peaks=volume_peaks,
                volume_troughs=volume_troughs,
                volume_pattern=volume_pattern,
                confidence_score=confidence_score
            )
            
        except Exception as e:
            logger.error(f"Error in real-time volume analysis: {e}")
            return self._get_default_metrics()
    
    def _analyze_volume_historical(self, df: pd.DataFrame) -> VolumeMetrics:
        """
        Historical volume analysis with precomputed metrics
        """
        try:
            # Precompute all volume metrics using vectorized operations
            volume_series = df['volume']
            
            # Vectorized rolling calculations
            rolling_metrics = self._compute_rolling_metrics_vectorized(volume_series)
            
            # Calculate volume ratios for entire series
            volume_ratios = volume_series / rolling_metrics['medium_avg']
            volume_spike_ratios = volume_series / rolling_metrics['short_avg']
            
            # Get current values
            current_volume = volume_series.iloc[-1]
            current_volume_ratio = volume_ratios.iloc[-1]
            current_spike_ratio = volume_spike_ratios.iloc[-1]
            
            # Volume trend alignment
            volume_trend_alignment = (
                (rolling_metrics['short_avg'].iloc[-1] - rolling_metrics['long_avg'].iloc[-1]) / 
                rolling_metrics['long_avg'].iloc[-1]
                if rolling_metrics['long_avg'].iloc[-1] > 0 else 0.0
            )
            
            # Volume consistency across entire series
            volume_consistency = 1.0 - (volume_series.std() / volume_series.mean()) if volume_series.mean() > 0 else 0.0
            volume_consistency = max(0.0, min(1.0, volume_consistency))
            
            # Volume momentum
            volume_momentum = (
                (current_volume - volume_series.iloc[-5]) / volume_series.iloc[-5]
                if volume_series.iloc[-5] > 0 else 0.0
            )
            
            # Volume volatility
            volume_volatility = volume_series.std() / volume_series.mean() if volume_series.mean() > 0 else 0.0
            
            # Find all volume extremes
            volume_peaks = self._find_volume_extremes(volume_series, 'peaks')
            volume_troughs = self._find_volume_extremes(volume_series, 'troughs')
            
            # Classify volume pattern
            volume_pattern = self._classify_volume_pattern(
                current_volume_ratio, current_spike_ratio, volume_trend_alignment, volume_consistency
            )
            
            # Calculate confidence score
            confidence_score = self._calculate_volume_confidence(
                current_volume_ratio, current_spike_ratio, volume_trend_alignment, volume_consistency
            )
            
            return VolumeMetrics(
                volume_ratio=current_volume_ratio,
                volume_spike_ratio=current_spike_ratio,
                volume_trend_alignment=volume_trend_alignment,
                volume_consistency=volume_consistency,
                volume_momentum=volume_momentum,
                volume_volatility=volume_volatility,
                avg_volume=rolling_metrics['medium_avg'].iloc[-1],
                volume_peaks=volume_peaks,
                volume_troughs=volume_troughs,
                volume_pattern=volume_pattern,
                confidence_score=confidence_score
            )
            
        except Exception as e:
            logger.error(f"Error in historical volume analysis: {e}")
            return self._get_default_metrics()
    
    def _analyze_volume_backtesting(self, df: pd.DataFrame) -> VolumeMetrics:
        """
        Backtesting volume analysis with precomputed averages
        """
        try:
            # For backtesting, precompute avg_volume for efficiency
            volume_series = df['volume']
            
            # Precompute average volume for the entire dataset
            avg_volume = volume_series.mean()
            
            # Calculate volume ratios using precomputed average
            volume_ratios = volume_series / avg_volume
            current_volume_ratio = volume_ratios.iloc[-1]
            
            # Calculate spike ratio using short-term average
            short_avg = volume_series.tail(self.rolling_windows['short']).mean()
            current_spike_ratio = volume_series.iloc[-1] / short_avg if short_avg > 0 else 1.0
            
            # Volume trend alignment
            recent_avg = volume_series.tail(self.rolling_windows['medium']).mean()
            volume_trend_alignment = (recent_avg - avg_volume) / avg_volume if avg_volume > 0 else 0.0
            
            # Volume consistency
            volume_consistency = 1.0 - (volume_series.std() / avg_volume) if avg_volume > 0 else 0.0
            volume_consistency = max(0.0, min(1.0, volume_consistency))
            
            # Volume momentum
            volume_momentum = (
                (volume_series.iloc[-1] - volume_series.iloc[-5]) / volume_series.iloc[-5]
                if volume_series.iloc[-5] > 0 else 0.0
            )
            
            # Volume volatility
            volume_volatility = volume_series.std() / avg_volume if avg_volume > 0 else 0.0
            
            # Find volume extremes
            volume_peaks = self._find_volume_extremes(volume_series, 'peaks')
            volume_troughs = self._find_volume_extremes(volume_series, 'troughs')
            
            # Classify volume pattern
            volume_pattern = self._classify_volume_pattern(
                current_volume_ratio, current_spike_ratio, volume_trend_alignment, volume_consistency
            )
            
            # Calculate confidence score
            confidence_score = self._calculate_volume_confidence(
                current_volume_ratio, current_spike_ratio, volume_trend_alignment, volume_consistency
            )
            
            return VolumeMetrics(
                volume_ratio=current_volume_ratio,
                volume_spike_ratio=current_spike_ratio,
                volume_trend_alignment=volume_trend_alignment,
                volume_consistency=volume_consistency,
                volume_momentum=volume_momentum,
                volume_volatility=volume_volatility,
                avg_volume=avg_volume,
                volume_peaks=volume_peaks,
                volume_troughs=volume_troughs,
                volume_pattern=volume_pattern,
                confidence_score=confidence_score
            )
            
        except Exception as e:
            logger.error(f"Error in backtesting volume analysis: {e}")
            return self._get_default_metrics()
    
    def _compute_rolling_metrics_vectorized(self, volume_series: pd.Series) -> Dict[str, pd.Series]:
        """
        Compute rolling metrics using vectorized operations
        """
        try:
            return {
                'short_avg': volume_series.rolling(window=self.rolling_windows['short']).mean(),
                'medium_avg': volume_series.rolling(window=self.rolling_windows['medium']).mean(),
                'long_avg': volume_series.rolling(window=self.rolling_windows['long']).mean(),
                'short_std': volume_series.rolling(window=self.rolling_windows['short']).std(),
                'medium_std': volume_series.rolling(window=self.rolling_windows['medium']).std(),
                'long_std': volume_series.rolling(window=self.rolling_windows['long']).std()
            }
        except Exception as e:
            logger.error(f"Error computing rolling metrics: {e}")
            return {}
    
    def _find_volume_extremes(self, volume_series: pd.Series, extreme_type: str) -> List[int]:
        """
        Find volume peaks or troughs using vectorized operations
        """
        try:
            if len(volume_series) < 3:
                return []
            
            extremes = []
            
            if extreme_type == 'peaks':
                # Find peaks (local maxima)
                for i in range(1, len(volume_series) - 1):
                    if volume_series.iloc[i] > volume_series.iloc[i-1] and volume_series.iloc[i] > volume_series.iloc[i+1]:
                        extremes.append(i)
            elif extreme_type == 'troughs':
                # Find troughs (local minima)
                for i in range(1, len(volume_series) - 1):
                    if volume_series.iloc[i] < volume_series.iloc[i-1] and volume_series.iloc[i] < volume_series.iloc[i+1]:
                        extremes.append(i)
            
            return extremes
            
        except Exception as e:
            logger.error(f"Error finding volume extremes: {e}")
            return []
    
    def _classify_volume_pattern(
        self, 
        volume_ratio: float, 
        spike_ratio: float, 
        trend_alignment: float, 
        consistency: float
    ) -> str:
        """
        Classify volume pattern based on metrics
        """
        try:
            if volume_ratio > 2.0 and spike_ratio > 2.0:
                return "extreme_spike"
            elif volume_ratio > 1.5 and spike_ratio > 1.5:
                return "volume_spike"
            elif volume_ratio > 1.2 and trend_alignment > 0.1:
                return "trend_confirmation"
            elif volume_ratio < 0.8 and consistency < 0.5:
                return "volume_dry_up"
            elif consistency > 0.8 and abs(trend_alignment) < 0.05:
                return "consistent_volume"
            elif trend_alignment > 0.1:
                return "increasing_volume"
            elif trend_alignment < -0.1:
                return "decreasing_volume"
            else:
                return "normal_volume"
                
        except Exception as e:
            logger.error(f"Error classifying volume pattern: {e}")
            return "unknown"
    
    def _calculate_volume_confidence(
        self, 
        volume_ratio: float, 
        spike_ratio: float, 
        trend_alignment: float, 
        consistency: float
    ) -> float:
        """
        Calculate confidence score for volume analysis
        """
        try:
            # Base confidence from volume ratio
            base_confidence = min(1.0, volume_ratio / 2.0)
            
            # Adjust for spike ratio
            spike_factor = min(1.0, spike_ratio / 1.5)
            
            # Adjust for trend alignment
            trend_factor = min(1.0, abs(trend_alignment) * 2.0)
            
            # Adjust for consistency
            consistency_factor = consistency
            
            # Combine factors
            confidence = (base_confidence * 0.4 + 
                         spike_factor * 0.3 + 
                         trend_factor * 0.2 + 
                         consistency_factor * 0.1)
            
            return max(0.0, min(1.0, confidence))
            
        except Exception as e:
            logger.error(f"Error calculating volume confidence: {e}")
            return 0.0
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """
        Check if cached metrics are still valid
        """
        try:
            if cache_key not in self.last_cache_update:
                return False
            
            last_update = self.last_cache_update[cache_key]
            return datetime.now() - last_update < self.cache_ttl
            
        except Exception as e:
            logger.error(f"Error checking cache validity: {e}")
            return False
    
    def _cache_metrics(self, cache_key: str, metrics: VolumeMetrics):
        """
        Cache volume metrics
        """
        try:
            self.metrics_cache[cache_key] = {
                'volume_ratio': metrics.volume_ratio,
                'volume_spike_ratio': metrics.volume_spike_ratio,
                'volume_trend_alignment': metrics.volume_trend_alignment,
                'volume_consistency': metrics.volume_consistency,
                'volume_momentum': metrics.volume_momentum,
                'volume_volatility': metrics.volume_volatility,
                'avg_volume': metrics.avg_volume,
                'volume_peaks': metrics.volume_peaks,
                'volume_troughs': metrics.volume_troughs,
                'volume_pattern': metrics.volume_pattern,
                'confidence_score': metrics.confidence_score
            }
            self.last_cache_update[cache_key] = datetime.now()
            
        except Exception as e:
            logger.error(f"Error caching metrics: {e}")
    
    def _get_default_metrics(self) -> VolumeMetrics:
        """
        Return default volume metrics
        """
        return VolumeMetrics(
            volume_ratio=1.0,
            volume_spike_ratio=1.0,
            volume_trend_alignment=0.0,
            volume_consistency=0.5,
            volume_momentum=0.0,
            volume_volatility=0.0,
            avg_volume=1000.0,
            volume_peaks=[],
            volume_troughs=[],
            volume_pattern="unknown",
            confidence_score=0.0
        )
    
    def clear_cache(self):
        """
        Clear all cached metrics
        """
        try:
            self.metrics_cache.clear()
            self.last_cache_update.clear()
            logger.info("🧹 Volume analyzer cache cleared")
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics
        """
        try:
            return {
                'cache_size': len(self.metrics_cache),
                'cache_keys': list(self.metrics_cache.keys()),
                'last_updates': {
                    key: update_time.isoformat() 
                    for key, update_time in self.last_cache_update.items()
                }
            }
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {}
    
    def switch_mode(self, new_mode: AnalysisMode):
        """
        Switch analysis mode
        """
        try:
            self.mode = new_mode
            logger.info(f"🔄 Switched to {new_mode.value} mode")
        except Exception as e:
            logger.error(f"Error switching mode: {e}")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics for the analyzer
        """
        try:
            return {
                'mode': self.mode.value,
                'use_vectorized': self.use_vectorized,
                'parallel_processing': self.parallel_processing,
                'rolling_windows': self.rolling_windows,
                'cache_size': len(self.metrics_cache),
                'cache_ttl_minutes': self.cache_ttl.total_seconds() / 60
            }
        except Exception as e:
            logger.error(f"Error getting performance metrics: {e}")
            return {}
