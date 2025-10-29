"""
Priority 2: Advanced Feature Engineering
Enhanced sliding window extraction, dimensionality reduction, and caching system
"""

import asyncio
import logging
from typing import Dict, List, Optional, Union, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from collections import defaultdict, deque
import json
import time
import pickle
from pathlib import Path
import redis
from functools import lru_cache
import hashlib

# Import PCA components with error handling
try:
    from sklearn.decomposition import PCA, IncrementalPCA, KernelPCA
    PCA_AVAILABLE = True
except ImportError:
    PCA_AVAILABLE = False
    logger.warning("scikit-learn not available, PCA features will be disabled")

logger = logging.getLogger(__name__)

@dataclass
class CacheEntry:
    """Cache entry with TTL support"""
    data: Any
    timestamp: datetime
    ttl: int = 3600  # 1 hour default TTL
    
    def is_expired(self) -> bool:
        """Check if cache entry is expired"""
        return (datetime.now() - self.timestamp).total_seconds() > self.ttl

class Priority2FeatureEngineering:
    """
    Priority 2 Advanced Feature Engineering System
    Implements optimized feature extraction with caching and performance monitoring
    """
    
    def __init__(self, cache_dir: str = "cache/priority2_features", redis_url: str = None):
        """Initialize Priority 2 feature engineering system"""
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize Redis client for distributed caching
        self.redis_client = None
        if redis_url:
            try:
                self.redis_client = redis.from_url(redis_url)
                self.redis_client.ping()
                logger.info("✅ Redis cache connected")
            except Exception as e:
                logger.warning(f"⚠️ Redis cache not available: {e}")
        
        # Performance tracking
        self.stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'extraction_times': [],
            'total_requests': 0
        }
        
        # Local cache with TTL
        self._local_cache = {}
        self._cache_cleanup_interval = 300  # 5 minutes
        self._last_cleanup = time.time()
        
        logger.info("Priority 2 Advanced Feature Engineering System initialized")
    
    def _calculate_skewness(self, data: np.ndarray) -> np.ndarray:
        """Calculate skewness with local caching for repeated calculations"""
        try:
            if data.size == 0:
                return np.array([])
            
            # Create hash for caching
            data_hash = hash(str(data.shape) + str(data.mean()) + str(data.std()))
            cache_key = f"skewness_{data_hash}"
            
            # Check local cache
            if cache_key in self._local_cache:
                entry = self._local_cache[cache_key]
                if not entry.is_expired():
                    return entry.data
            
            # Vectorized skewness calculation
            mean = np.mean(data, axis=1, keepdims=True)
            std = np.std(data, axis=1, keepdims=True)
            z_scores = (data - mean) / (std + 1e-8)  # Add small epsilon to avoid division by zero
            skewness = np.mean(z_scores ** 3, axis=1)
            
            # Cache result
            self._local_cache[cache_key] = CacheEntry(skewness, datetime.now())
            return skewness
        except Exception as e:
            logger.error(f"Skewness calculation error: {e}", exc_info=True)
            return np.zeros(data.shape[0])
    
    def _calculate_kurtosis(self, data: np.ndarray) -> np.ndarray:
        """Calculate kurtosis with local caching for repeated calculations"""
        try:
            if data.size == 0:
                return np.array([])
            
            # Create hash for caching
            data_hash = hash(str(data.shape) + str(data.mean()) + str(data.std()))
            cache_key = f"kurtosis_{data_hash}"
            
            # Check local cache
            if cache_key in self._local_cache:
                entry = self._local_cache[cache_key]
                if not entry.is_expired():
                    return entry.data
            
            # Vectorized kurtosis calculation
            mean = np.mean(data, axis=1, keepdims=True)
            std = np.std(data, axis=1, keepdims=True)
            z_scores = (data - mean) / (std + 1e-8)
            kurtosis = np.mean(z_scores ** 4, axis=1) - 3  # Excess kurtosis
            
            # Cache result
            self._local_cache[cache_key] = CacheEntry(kurtosis, datetime.now())
            return kurtosis
        except Exception as e:
            logger.error(f"Kurtosis calculation error: {e}", exc_info=True)
            return np.zeros(data.shape[0])
    
    def _cleanup_expired_cache(self):
        """Clean up expired cache entries"""
        current_time = time.time()
        if current_time - self._last_cleanup < self._cache_cleanup_interval:
            return
        
        expired_keys = []
        for key, entry in self._local_cache.items():
            if entry.is_expired():
                expired_keys.append(key)
        
        for key in expired_keys:
            del self._local_cache[key]
        
        if expired_keys:
            logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
        
        self._last_cleanup = current_time
    
    async def extract_priority2_features(self, df: pd.DataFrame, symbol: str) -> Tuple[pd.DataFrame, Dict]:
        """
        Extract Priority 2 features with optimized sliding windows and enhanced PCA.
        
        Args:
            df: OHLCV DataFrame
            symbol: Trading symbol
            
        Returns:
            Tuple of (features_df, metadata_dict)
        """
        start_time = datetime.now()
        
        # Check cache first
        cache_key = self._generate_cache_key(symbol, df.shape)
        cached_features = await self._get_cached_features(cache_key)
        
        if cached_features is not None:
            self.stats['cache_hits'] += 1
            logger.info(f"✅ Using cached Priority 2 features for {symbol}")
            return cached_features, {'source': 'cache', 'symbol': symbol}
        
        self.stats['cache_misses'] += 1
        
        try:
            # 1. Clean data
            df_clean = self._clean_data(df)
            
            # 2. Extract optimized sliding window features
            df_windows = self._extract_optimized_sliding_windows(df_clean)
            
            # 3. Create advanced indicators
            df_indicators = self._create_advanced_indicators(df_windows)
            
            # 4. Apply enhanced dimensionality reduction
            df_reduced = self._apply_enhanced_pca(df_indicators, symbol)
            
            # 5. Cache results
            await self._cache_features(cache_key, df_reduced)
            
            # Update stats
            extraction_time = (datetime.now() - start_time).total_seconds()
            self.stats['extraction_times'].append(extraction_time)
            
            metadata = {
                'symbol': symbol,
                'original_shape': df.shape,
                'final_shape': df_reduced.shape,
                'extraction_time': extraction_time,
                'features_removed': df.shape[1] - df_reduced.shape[1],
                'timestamp': datetime.now(),
                'source': 'computed'
            }
            
            logger.info(f"✅ Priority 2 feature extraction completed for {symbol} in {extraction_time:.3f}s")
            return df_reduced, metadata
            
        except Exception as e:
            logger.error(f"❌ Priority 2 feature extraction failed for {symbol}: {e}")
            raise
    
    def _extract_optimized_sliding_windows(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract optimized sliding window features with overlapping windows.
        
        Features:
        - Adaptive window sizing
        - Overlapping windows for better coverage
        - Memory-efficient implementation
        """
        df_windows = df.copy()
        
        # Price-based sliding windows with optimization
        window_sizes = [5, 10, 20, 50, 100]
        
        for window_size in window_sizes:
            if len(df) < window_size:
                continue
            
            # Create optimized sliding windows
            price_windows = self._create_optimized_windows(df['close'].values, window_size)
            volume_windows = self._create_optimized_windows(df['volume'].values, window_size)
            
            # Statistical features
            df_windows[f'price_mean_{window_size}'] = np.mean(price_windows, axis=1)
            df_windows[f'price_std_{window_size}'] = np.std(price_windows, axis=1)
            df_windows[f'price_min_{window_size}'] = np.min(price_windows, axis=1)
            df_windows[f'price_max_{window_size}'] = np.max(price_windows, axis=1)
            df_windows[f'price_median_{window_size}'] = np.median(price_windows, axis=1)
            
            # Advanced statistical features
            df_windows[f'price_skew_{window_size}'] = self._calculate_skewness(price_windows)
            df_windows[f'price_kurtosis_{window_size}'] = self._calculate_kurtosis(price_windows)
            
            # Range and momentum features
            df_windows[f'price_range_{window_size}'] = (
                df_windows[f'price_max_{window_size}'] - df_windows[f'price_min_{window_size}']
            )
            df_windows[f'price_range_ratio_{window_size}'] = (
                df_windows[f'price_range_{window_size}'] / df_windows[f'price_mean_{window_size}']
            )
            df_windows[f'momentum_{window_size}'] = df['close'] / price_windows[:, 0] - 1
            df_windows[f'volatility_{window_size}'] = (
                df_windows[f'price_std_{window_size}'] / df_windows[f'price_mean_{window_size}']
            )
            
            # Volume features
            df_windows[f'volume_mean_{window_size}'] = np.mean(volume_windows, axis=1)
            df_windows[f'volume_std_{window_size}'] = np.std(volume_windows, axis=1)
            df_windows[f'volume_ratio_{window_size}'] = df['volume'] / df_windows[f'volume_mean_{window_size}']
            
            # Percentile features
            df_windows[f'price_p25_{window_size}'] = np.percentile(price_windows, 25, axis=1)
            df_windows[f'price_p75_{window_size}'] = np.percentile(price_windows, 75, axis=1)
            df_windows[f'price_iqr_{window_size}'] = (
                df_windows[f'price_p75_{window_size}'] - df_windows[f'price_p25_{window_size}']
            )
        
        # Create overlapping window features
        df_windows = self._create_overlapping_windows(df_windows)
        
        return df_windows
    
    def _create_optimized_windows(self, arr: np.ndarray, window_size: int) -> np.ndarray:
        """Create optimized sliding windows using numpy stride_tricks."""
        if len(arr) < window_size:
            return np.array([])
        
        # Pad the array to handle edge cases
        padded = np.pad(arr, (window_size - 1, 0), mode='edge')
        
        # Create sliding windows with optimized strides
        shape = (len(arr), window_size)
        strides = (padded.strides[0], padded.strides[0])
        windows = np.lib.stride_tricks.as_strided(padded, shape=shape, strides=strides)
        
        return windows
    
    def _create_overlapping_windows(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create overlapping window features for better coverage."""
        df_overlap = df.copy()
        
        # Create overlapping price momentum features
        for period in [3, 5, 8, 13, 21]:  # Fibonacci-like periods
            if len(df) >= period:
                df_overlap[f'overlap_momentum_{period}'] = df['close'].pct_change(periods=period)
                df_overlap[f'overlap_volatility_{period}'] = (
                    df['close'].pct_change().rolling(period).std()
                )
        
        # Create adaptive windows based on volatility
        volatility = df['close'].pct_change().rolling(20).std()
        high_vol_mask = volatility > volatility.quantile(0.75)
        low_vol_mask = volatility < volatility.quantile(0.25)
        
        if high_vol_mask.any():
            high_vol_windows = self._create_optimized_windows(df['close'].values, 5)
            df_overlap['adaptive_high_vol_mean'] = np.where(
                high_vol_mask, np.mean(high_vol_windows, axis=1), np.nan
            )
        
        if low_vol_mask.any():
            low_vol_windows = self._create_optimized_windows(df['close'].values, 50)
            df_overlap['adaptive_low_vol_mean'] = np.where(
                low_vol_mask, np.mean(low_vol_windows, axis=1), np.nan
            )
        
        return df_overlap
    
    def _create_advanced_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create advanced technical indicators."""
        df_indicators = df.copy()
        
        # Advanced momentum indicators
        df_indicators['rsi_divergence'] = self._calculate_rsi_divergence(df)
        df_indicators['macd_divergence'] = self._calculate_macd_divergence(df)
        df_indicators['price_momentum'] = self._calculate_price_momentum(df)
        
        # Advanced volatility indicators
        df_indicators['volatility_regime'] = self._calculate_volatility_regime(df)
        df_indicators['volatility_breakout'] = self._calculate_volatility_breakout(df)
        
        # Advanced trend indicators
        df_indicators['trend_strength'] = self._calculate_trend_strength(df)
        df_indicators['trend_reversal'] = self._calculate_trend_reversal(df)
        
        # Advanced volume indicators
        df_indicators['volume_price_trend'] = self._calculate_volume_price_trend(df)
        df_indicators['volume_divergence'] = self._calculate_volume_divergence(df)
        
        return df_indicators
    
    def _apply_enhanced_pca(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Apply enhanced PCA with multiple variants.
        
        Features:
        - Standard PCA
        - Incremental PCA for large datasets
        - Kernel PCA for non-linear relationships
        """
        # Get numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            return df
        
        df_numeric = df[numeric_cols].fillna(0)
        
        # Determine optimal number of components
        n_components = min(50, len(df_numeric.columns), len(df_numeric) - 1)
        
        # Check if PCA is available
        if not PCA_AVAILABLE:
            logger.warning("PCA not available, returning original data")
            return df
        
        # Try different PCA variants
        pca_variants = [
            ('standard', PCA(n_components=n_components, random_state=42)),
            ('incremental', IncrementalPCA(n_components=n_components)),
            ('kernel_rbf', KernelPCA(n_components=n_components, kernel='rbf', random_state=42))
        ]
        
        best_pca = None
        best_score = -np.inf
        
        for variant_name, pca in pca_variants:
            try:
                # Fit PCA
                pca_values = pca.fit_transform(df_numeric)
                
                # Calculate explained variance ratio
                if hasattr(pca, 'explained_variance_ratio_'):
                    score = np.sum(pca.explained_variance_ratio_)
                else:
                    # For kernel PCA, use reconstruction error
                    reconstructed = pca.inverse_transform(pca_values)
                    score = -np.mean((df_numeric - reconstructed) ** 2)
                
                if score > best_score:
                    best_score = score
                    best_pca = (variant_name, pca, pca_values)
                
            except Exception as e:
                logger.warning(f"PCA variant {variant_name} failed: {e}")
                continue
        
        if best_pca is None:
            logger.warning("All PCA variants failed, returning original data")
            return df
        
        variant_name, pca, pca_values = best_pca
        
        # Create column names
        col_names = [f'{variant_name}_pca_{i}' for i in range(pca_values.shape[1])]
        
        # Create result DataFrame
        df_pca = pd.DataFrame(pca_values, columns=col_names, index=df.index)
        
        logger.info(f"✅ Applied {variant_name} PCA to {symbol}: {len(df_numeric.columns)} -> {len(col_names)} features")
        
        return df_pca
    
    def _calculate_rsi_divergence(self, df: pd.DataFrame) -> pd.Series:
        """Calculate RSI divergence."""
        # Simplified RSI calculation
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        # Calculate divergence
        price_highs = df['high'].rolling(20).max()
        rsi_highs = rsi.rolling(20).max()
        
        divergence = np.where(
            (df['close'] > price_highs.shift(1)) & (rsi < rsi_highs.shift(1)),
            -1,  # Bearish divergence
            np.where(
                (df['close'] < df['low'].rolling(20).min().shift(1)) & (rsi > rsi.rolling(20).min().shift(1)),
                1,   # Bullish divergence
                0    # No divergence
            )
        )
        
        return pd.Series(divergence, index=df.index)
    
    def _calculate_macd_divergence(self, df: pd.DataFrame) -> pd.Series:
        """Calculate MACD divergence."""
        # Simplified MACD calculation
        ema12 = df['close'].ewm(span=12).mean()
        ema26 = df['close'].ewm(span=26).mean()
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9).mean()
        
        price_highs = df['high'].rolling(20).max()
        macd_highs = macd_line.rolling(20).max()
        
        divergence = np.where(
            (df['close'] > price_highs.shift(1)) & (macd_line < macd_highs.shift(1)),
            -1,  # Bearish divergence
            np.where(
                (df['close'] < df['low'].rolling(20).min().shift(1)) & (macd_line > macd_line.rolling(20).min().shift(1)),
                1,   # Bullish divergence
                0    # No divergence
            )
        )
        
        return pd.Series(divergence, index=df.index)
    
    def _calculate_price_momentum(self, df: pd.DataFrame) -> pd.Series:
        """Calculate price momentum."""
        return df['close'].pct_change(periods=5)
    
    def _calculate_volatility_regime(self, df: pd.DataFrame) -> pd.Series:
        """Calculate volatility regime."""
        volatility = df['close'].pct_change().rolling(20).std()
        volatility_mean = volatility.rolling(100).mean()
        
        regime = np.where(volatility > volatility_mean * 1.5, 2,  # High volatility
                         np.where(volatility < volatility_mean * 0.5, 0,  # Low volatility
                                 1))  # Normal volatility
        
        return pd.Series(regime, index=df.index)
    
    def _calculate_volatility_breakout(self, df: pd.DataFrame) -> pd.Series:
        """Calculate volatility breakout."""
        volatility = df['close'].pct_change().rolling(20).std()
        volatility_upper = volatility.rolling(100).quantile(0.95)
        
        breakout = (volatility > volatility_upper).astype(int)
        return pd.Series(breakout, index=df.index)
    
    def _calculate_trend_strength(self, df: pd.DataFrame) -> pd.Series:
        """Calculate trend strength."""
        ema_short = df['close'].ewm(span=9).mean()
        ema_long = df['close'].ewm(span=21).mean()
        
        trend_strength = (ema_short - ema_long) / ema_long
        return pd.Series(trend_strength, index=df.index)
    
    def _calculate_trend_reversal(self, df: pd.DataFrame) -> pd.Series:
        """Calculate trend reversal signals."""
        ema_short = df['close'].ewm(span=9).mean()
        ema_long = df['close'].ewm(span=21).mean()
        
        reversal = np.where(
            (ema_short > ema_long) & (ema_short.shift(1) <= ema_long.shift(1)),
            1,   # Bullish reversal
            np.where(
                (ema_short < ema_long) & (ema_short.shift(1) >= ema_long.shift(1)),
                -1,  # Bearish reversal
                0    # No reversal
            )
        )
        
        return pd.Series(reversal, index=df.index)
    
    def _calculate_volume_price_trend(self, df: pd.DataFrame) -> pd.Series:
        """Calculate volume-price trend."""
        return (df['close'] - df['close'].shift(1)) * df['volume']
    
    def _calculate_volume_divergence(self, df: pd.DataFrame) -> pd.Series:
        """Calculate volume divergence."""
        volume_sma = df['volume'].rolling(20).mean()
        price_change = df['close'].pct_change()
        
        divergence = np.where(
            (price_change > 0) & (df['volume'] < volume_sma),
            -1,  # Bearish volume divergence
            np.where(
                (price_change < 0) & (df['volume'] > volume_sma),
                1,   # Bullish volume divergence
                0    # No divergence
            )
        )
        
        return pd.Series(divergence, index=df.index)
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate input data."""
        df_clean = df.copy()
        
        # Ensure required columns exist
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df_clean.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Convert to numeric with error handling
        for col in required_cols:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        
        # Remove rows with NaN values
        df_clean = df_clean.dropna(subset=required_cols)
        
        # Remove infinite values
        df_clean = df_clean.replace([np.inf, -np.inf], np.nan)
        df_clean = df_clean.dropna(subset=required_cols)
        
        return df_clean
    
    async def _get_cached_features(self, cache_key: str) -> Optional[pd.DataFrame]:
        """Get cached features from Redis or local cache."""
        if self.redis_client:
            try:
                cached_data = self.redis_client.get(cache_key)
                if cached_data:
                    data = pickle.loads(cached_data)
                    return data['features']
            except Exception as e:
                logger.warning(f"Redis cache retrieval failed: {e}")
        
        # Fallback to local cache
        cache_path = self.cache_dir / f"{cache_key}.pkl"
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    data = pickle.load(f)
                return data['features']
            except Exception as e:
                logger.warning(f"Local cache retrieval failed: {e}")
        
        return None
    
    async def _cache_features(self, cache_key: str, features: pd.DataFrame):
        """Cache features in Redis and local storage."""
        cache_data = {
            'features': features,
            'timestamp': datetime.now(),
            'cache_key': cache_key
        }
        
        # Cache in Redis
        if self.redis_client:
            try:
                self.redis_client.setex(
                    cache_key, 
                    3600,  # 1 hour expiration
                    pickle.dumps(cache_data)
                )
            except Exception as e:
                logger.warning(f"Redis caching failed: {e}")
        
        # Cache locally
        cache_path = self.cache_dir / f"{cache_key}.pkl"
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(cache_data, f)
        except Exception as e:
            logger.warning(f"Local caching failed: {e}")
    
    def _generate_cache_key(self, symbol: str, shape: Tuple) -> str:
        """Generate unique cache key."""
        return f"priority2_{symbol}_{shape[0]}_{shape[1]}"
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics."""
        total_requests = self.stats['cache_hits'] + self.stats['cache_misses']
        cache_hit_rate = self.stats['cache_hits'] / max(1, total_requests)
        
        return {
            'cache_hits': self.stats['cache_hits'],
            'cache_misses': self.stats['cache_misses'],
            'cache_hit_rate': cache_hit_rate,
            'total_requests': total_requests,
            'avg_extraction_time': np.mean(self.stats['extraction_times']) if self.stats['extraction_times'] else 0,
            'min_extraction_time': min(self.stats['extraction_times']) if self.stats['extraction_times'] else 0,
            'max_extraction_time': max(self.stats['extraction_times']) if self.stats['extraction_times'] else 0
        }


# Global instance
priority2_feature_engineering = Priority2FeatureEngineering()