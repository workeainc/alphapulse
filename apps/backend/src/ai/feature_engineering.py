import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.ensemble import RandomForestRegressor
import ta
import logging
from datetime import datetime, timedelta
import pickle
import os
from pathlib import Path
import psutil
import gc
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import asyncio
from dataclasses import dataclass

logger = logging.getLogger(__name__)

class FeatureExtractor:
    """
    High-performance feature extraction pipeline for AlphaPulse.
    
    Implements:
    1. Vectorized computations for speed
    2. Precomputed indicator caching
    3. Sliding window feature extraction
    4. Feature scaling/normalization
    5. Dimensionality reduction
    """
    
    def __init__(self, cache_dir: str = "cache/features", 
                 scaler_type: str = "standard",
                 n_components: int = 50,
                 feature_selection_method: str = "mutual_info"):
        """
        Initialize the feature extractor.
        
        Args:
            cache_dir: Directory to store precomputed features
            scaler_type: "standard" or "minmax"
            n_components: Number of PCA components to keep
            feature_selection_method: "mutual_info", "f_regression", or "random_forest"
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.scaler_type = scaler_type
        self.n_components = n_components
        self.feature_selection_method = feature_selection_method
        
        # Initialize scalers and reducers
        self.scalers = {}  # Per-symbol scalers
        self.pca = None
        self.feature_selector = None
        
        # Cache for precomputed indicators
        self.indicator_cache = {}
        self.cache_timeout = 300  # 5 minutes
        
        # Performance tracking
        self.extraction_times = []
        
        # Phase 4.2: Memory & CPU Optimization
        self.memory_optimization_enabled = True
        self.cpu_optimization_enabled = True
        self.cache_memory_limit_mb = 256  # 256MB cache limit for features
        self.feature_cache = {}
        self.cache_eviction_policy = "lru"
        self.last_cache_cleanup = time.time()
        self.cache_cleanup_interval = 300  # 5 minutes
        
        # Phase 4.2: CPU Optimization
        self.thread_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="FeatureEng")
        self.process_pool = ProcessPoolExecutor(max_workers=2)
        self.cpu_throttling_threshold = 0.9
        self.parallel_processing_enabled = True
        
        # Phase 4.2: Memory Monitoring
        self.memory_threshold = 0.8
        self.gc_threshold = 500
        self.operation_count = 0
        self.memory_usage_history = []
        
        # Phase 4.2: Performance Metrics
        self.feature_extraction_stats = {
            'total_extractions': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'memory_cleanups': 0,
            'gc_collections': 0
        }
        
    def extract_features(self, df: pd.DataFrame, symbol: str, 
                        target_col: str = None) -> Tuple[pd.DataFrame, Dict]:
        """
        Main feature extraction pipeline.
        
        Args:
            df: OHLCV DataFrame with datetime index
            symbol: Trading symbol for per-symbol scaling
            target_col: Target column for supervised feature selection
            
        Returns:
            Tuple of (features_df, metadata_dict)
        """
        start_time = datetime.now()
        
        try:
            # 1. Clean and validate data
            df_clean = self._clean_data(df)
            
            # 2. Precompute technical indicators (vectorized)
            df_indicators = self._compute_technical_indicators(df_clean)
            
            # 3. Create sliding window features
            df_windows = self._create_sliding_window_features(df_indicators)
            
            # 4. Create lag features
            df_lags = self._create_lag_features(df_windows)
            
            # 5. Create volume features
            df_volume = self._create_volume_features(df_lags)
            
            # 6. Create market regime features
            df_regime = self._create_market_regime_features(df_volume)
            
            # 7. Scale features
            df_scaled = self._scale_features(df_regime, symbol)
            
            # 8. Apply dimensionality reduction
            df_reduced = self._reduce_dimensions(df_scaled, target_col)
            
            # 9. Cache results
            self._cache_features(symbol, df_reduced)
            
            extraction_time = (datetime.now() - start_time).total_seconds()
            self.extraction_times.append(extraction_time)
            
            metadata = {
                "original_shape": df.shape,
                "final_shape": df_reduced.shape,
                "extraction_time": extraction_time,
                "features_removed": df.shape[1] - df_reduced.shape[1],
                "symbol": symbol,
                "timestamp": datetime.now()
            }
            
            logger.info(f"Feature extraction completed for {symbol} in {extraction_time:.3f}s")
            return df_reduced, metadata
            
        except Exception as e:
            logger.error(f"Feature extraction failed for {symbol}: {e}")
            raise
    
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
        
        # Remove rows with NaN values after conversion
        df_clean = df_clean.dropna(subset=required_cols)
        
        # Remove infinite values
        df_clean = df_clean.replace([np.inf, -np.inf], np.nan)
        df_clean = df_clean.dropna(subset=required_cols)
        
        return df_clean
    
    def _compute_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute technical indicators using vectorized operations."""
        df_indicators = df.copy()
        
        # Trend indicators (vectorized)
        df_indicators['ema_9'] = ta.trend.ema_indicator(df['close'], window=9)
        df_indicators['ema_21'] = ta.trend.ema_indicator(df['close'], window=21)
        df_indicators['ema_50'] = ta.trend.ema_indicator(df['close'], window=50)
        df_indicators['sma_20'] = df['close'].rolling(window=20).mean()
        df_indicators['sma_50'] = df['close'].rolling(window=50).mean()
        
        # MACD (vectorized)
        macd = ta.trend.MACD(df['close'])
        df_indicators['macd'] = macd.macd()
        df_indicators['macd_signal'] = macd.macd_signal()
        df_indicators['macd_histogram'] = macd.macd_diff()
        
        # RSI (vectorized)
        df_indicators['rsi'] = ta.momentum.rsi(df['close'], window=14)
        
        # Bollinger Bands (vectorized)
        bb = ta.volatility.BollingerBands(df['close'])
        df_indicators['bb_upper'] = bb.bollinger_hband()
        df_indicators['bb_middle'] = bb.bollinger_mavg()
        df_indicators['bb_lower'] = bb.bollinger_lband()
        df_indicators['bb_width'] = (df_indicators['bb_upper'] - df_indicators['bb_lower']) / df_indicators['bb_middle']
        df_indicators['bb_position'] = (df['close'] - df_indicators['bb_lower']) / (df_indicators['bb_upper'] - df_indicators['bb_lower'])
        
        # ATR (vectorized)
        df_indicators['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'])
        
        # Stochastic (vectorized)
        stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'])
        df_indicators['stoch_k'] = stoch.stoch()
        df_indicators['stoch_d'] = stoch.stoch_signal()
        
        # Williams %R (vectorized)
        df_indicators['williams_r'] = ta.momentum.williams_r(df['high'], df['low'], df['close'])
        
        # ADX (vectorized)
        adx = ta.trend.ADXIndicator(df['high'], df['low'], df['close'])
        df_indicators['adx'] = adx.adx()
        df_indicators['di_plus'] = adx.adx_pos()
        df_indicators['di_minus'] = adx.adx_neg()
        
        # Price-based features
        df_indicators['price_change'] = df['close'].pct_change()
        df_indicators['price_change_abs'] = df_indicators['price_change'].abs()
        df_indicators['high_low_ratio'] = df['high'] / df['low']
        df_indicators['close_open_ratio'] = df['close'] / df['open']
        
        return df_indicators
    
    def _create_sliding_window_features(self, df: pd.DataFrame, 
                                      window_sizes: List[int] = [5, 10, 20]) -> pd.DataFrame:
        """Create sliding window features using numpy stride_tricks for efficiency."""
        df_windows = df.copy()
        
        for window_size in window_sizes:
            if len(df) < window_size:
                continue
                
            # Use numpy stride_tricks for efficient sliding windows
            price_windows = self._create_sliding_windows(df['close'].values, window_size)
            volume_windows = self._create_sliding_windows(df['volume'].values, window_size)
            
            # Price statistics over windows
            df_windows[f'price_mean_{window_size}'] = np.mean(price_windows, axis=1)
            df_windows[f'price_std_{window_size}'] = np.std(price_windows, axis=1)
            df_windows[f'price_min_{window_size}'] = np.min(price_windows, axis=1)
            df_windows[f'price_max_{window_size}'] = np.max(price_windows, axis=1)
            df_windows[f'price_range_{window_size}'] = df_windows[f'price_max_{window_size}'] - df_windows[f'price_min_{window_size}']
            
            # Volume statistics over windows
            df_windows[f'volume_mean_{window_size}'] = np.mean(volume_windows, axis=1)
            df_windows[f'volume_std_{window_size}'] = np.std(volume_windows, axis=1)
            df_windows[f'volume_ratio_{window_size}'] = df['volume'] / df_windows[f'volume_mean_{window_size}']
            
            # Momentum features
            df_windows[f'momentum_{window_size}'] = df['close'] / price_windows[:, 0] - 1
            df_windows[f'volatility_{window_size}'] = df_windows[f'price_std_{window_size}'] / df_windows[f'price_mean_{window_size}']
        
        return df_windows
    
    def _create_sliding_windows(self, arr: np.ndarray, window_size: int) -> np.ndarray:
        """Create sliding windows using numpy stride_tricks."""
        if len(arr) < window_size:
            return np.array([])
        
        # Pad the array to handle edge cases
        padded = np.pad(arr, (window_size - 1, 0), mode='edge')
        
        # Create sliding windows
        shape = (len(arr), window_size)
        strides = (padded.strides[0], padded.strides[0])
        windows = np.lib.stride_tricks.as_strided(padded, shape=shape, strides=strides)
        
        return windows
    
    def _create_lag_features(self, df: pd.DataFrame, 
                           lag_periods: List[int] = [1, 2, 3, 5, 10]) -> pd.DataFrame:
        """Create lag features for time series analysis."""
        df_lags = df.copy()
        
        # Price lags
        for lag in lag_periods:
            df_lags[f'close_lag_{lag}'] = df['close'].shift(lag)
            df_lags[f'volume_lag_{lag}'] = df['volume'].shift(lag)
            df_lags[f'price_change_lag_{lag}'] = df['close'].pct_change().shift(lag)
        
        # Rolling statistics lags
        for lag in [1, 2, 3]:
            df_lags[f'rsi_lag_{lag}'] = df_lags['rsi'].shift(lag)
            df_lags[f'macd_lag_{lag}'] = df_lags['macd'].shift(lag)
            df_lags[f'atr_lag_{lag}'] = df_lags['atr'].shift(lag)
        
        return df_lags
    
    def _create_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create volume-based features."""
        df_volume = df.copy()
        
        # Volume indicators
        df_volume['volume_sma'] = df['volume'].rolling(window=20).mean()
        df_volume['volume_ratio'] = df['volume'] / df_volume['volume_sma']
        df_volume['volume_surge'] = df['volume'] > (df_volume['volume_sma'] * 1.5)
        
        # Volume-price relationship
        df_volume['volume_price_trend'] = (df['close'] - df['close'].shift(1)) * df['volume']
        df_volume['volume_price_ratio'] = df['volume'] / df['close']
        
        # On-balance volume (OBV)
        df_volume['obv'] = ta.volume.on_balance_volume(df['close'], df['volume'])
        
        # Volume-weighted average price (VWAP)
        df_volume['vwap'] = (df['close'] * df['volume']).rolling(window=20).sum() / df['volume'].rolling(window=20).sum()
        
        return df_volume
    
    def _create_market_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create market regime detection features."""
        df_regime = df.copy()
        
        # Volatility regime (numeric)
        df_regime['volatility_regime_high'] = (df_regime['atr'] > df_regime['atr'].rolling(50).mean()).astype(int)
        
        # Trend regime (numeric encoding)
        trend_conditions = [
            (df_regime['ema_9'] > df_regime['ema_21']) & (df_regime['ema_21'] > df_regime['ema_50']),  # strong_uptrend
            (df_regime['ema_9'] > df_regime['ema_21']) & (df_regime['ema_21'] <= df_regime['ema_50']),  # weak_uptrend
            (df_regime['ema_9'] <= df_regime['ema_21']) & (df_regime['ema_21'] < df_regime['ema_50']),  # strong_downtrend
            (df_regime['ema_9'] <= df_regime['ema_21']) & (df_regime['ema_21'] >= df_regime['ema_50'])   # weak_downtrend
        ]
        trend_values = [3, 2, 1, 0]  # strong_uptrend=3, weak_uptrend=2, strong_downtrend=1, weak_downtrend=0
        df_regime['trend_regime'] = np.select(trend_conditions, trend_values, default=0)
        
        # Market structure (boolean to int)
        df_regime['higher_highs'] = (df['high'] > df['high'].shift(1)).astype(int)
        df_regime['higher_lows'] = (df['low'] > df['low'].shift(1)).astype(int)
        df_regime['lower_highs'] = (df['high'] < df['high'].shift(1)).astype(int)
        df_regime['lower_lows'] = (df['low'] < df['low'].shift(1)).astype(int)
        
        # Support/resistance levels
        df_regime['support'] = df['low'].rolling(window=20).min()
        df_regime['resistance'] = df['high'].rolling(window=20).max()
        df_regime['price_to_support'] = (df['close'] - df_regime['support']) / df['close']
        df_regime['price_to_resistance'] = (df_regime['resistance'] - df['close']) / df['close']
        
        return df_regime
    
    def _scale_features(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Scale features using per-symbol scalers."""
        df_scaled = df.copy()
        
        # Get numeric columns only
        numeric_cols = df_scaled.select_dtypes(include=[np.number]).columns
        df_numeric = df_scaled[numeric_cols].fillna(0)
        
        # Initialize scaler for this symbol if not exists
        if symbol not in self.scalers:
            if self.scaler_type == "standard":
                self.scalers[symbol] = StandardScaler()
            else:
                self.scalers[symbol] = MinMaxScaler()
            
            # Fit scaler on training data (first 80% of data)
            train_size = int(len(df_numeric) * 0.8)
            self.scalers[symbol].fit(df_numeric.iloc[:train_size])
        
        # Transform features
        scaled_values = self.scalers[symbol].transform(df_numeric)
        df_scaled[numeric_cols] = scaled_values
        
        return df_scaled
    
    def _reduce_dimensions(self, df: pd.DataFrame, target_col: str = None) -> pd.DataFrame:
        """Apply dimensionality reduction."""
        df_reduced = df.copy()
        
        # Get numeric columns only
        numeric_cols = df_reduced.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            logger.warning("No numeric columns found for dimensionality reduction")
            return df_reduced
        
        df_numeric = df_reduced[numeric_cols].fillna(0)
        
        # Feature selection
        if target_col and target_col in df_numeric.columns:
            X = df_numeric.drop(columns=[target_col])
            y = df_numeric[target_col]
            
            if len(X.columns) == 0:
                logger.warning("No features available for selection")
                return df_reduced
            
            if self.feature_selection_method == "mutual_info":
                selector = SelectKBest(score_func=mutual_info_regression, k=min(50, len(X.columns)))
            elif self.feature_selection_method == "f_regression":
                selector = SelectKBest(score_func=f_regression, k=min(50, len(X.columns)))
            elif self.feature_selection_method == "random_forest":
                rf = RandomForestRegressor(n_estimators=100, random_state=42)
                rf.fit(X, y)
                feature_importance = rf.feature_importances_
                top_features = np.argsort(feature_importance)[-min(50, len(X.columns)):]
                selected_cols = X.columns[top_features]
                df_reduced = df_reduced[selected_cols]
                return df_reduced
            else:
                selector = SelectKBest(score_func=mutual_info_regression, k=min(50, len(X.columns)))
            
            X_selected = selector.fit_transform(X, y)
            selected_cols = X.columns[selector.get_support()]
            df_reduced = df_reduced[selected_cols]
        
        # PCA for further reduction
        if len(df_reduced.columns) > self.n_components:
            if self.pca is None:
                self.pca = PCA(n_components=min(self.n_components, len(df_reduced.columns)))
                self.pca.fit(df_reduced)
            
            pca_values = self.pca.transform(df_reduced)
            pca_cols = [f'pca_{i}' for i in range(pca_values.shape[1])]
            df_reduced = pd.DataFrame(pca_values, columns=pca_cols, index=df_reduced.index)
        
        return df_reduced
    
    def _cache_features(self, symbol: str, df: pd.DataFrame):
        """Cache extracted features for reuse."""
        cache_key = f"{symbol}_{datetime.now().strftime('%Y%m%d_%H')}"
        cache_path = self.cache_dir / f"{cache_key}.pkl"
        
        cache_data = {
            'features': df,
            'timestamp': datetime.now(),
            'symbol': symbol
        }
        
        with open(cache_path, 'wb') as f:
            pickle.dump(cache_data, f)
        
        # Clean old cache files
        self._cleanup_cache()
    
    def _cleanup_cache(self, max_age_hours: int = 24):
        """Clean up old cache files."""
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        
        for cache_file in self.cache_dir.glob("*.pkl"):
            if cache_file.stat().st_mtime < cutoff_time.timestamp():
                cache_file.unlink()
    
    def get_cached_features(self, symbol: str, max_age_minutes: int = 30) -> Optional[pd.DataFrame]:
        """Retrieve cached features if available and recent."""
        for cache_file in self.cache_dir.glob(f"{symbol}_*.pkl"):
            file_time = datetime.fromtimestamp(cache_file.stat().st_mtime)
            if datetime.now() - file_time < timedelta(minutes=max_age_minutes):
                with open(cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                return cache_data['features']
        return None
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics."""
        if not self.extraction_times:
            return {}
        
        return {
            "total_extractions": len(self.extraction_times),
            "avg_extraction_time": np.mean(self.extraction_times),
            "min_extraction_time": np.min(self.extraction_times),
            "max_extraction_time": np.max(self.extraction_times),
            "std_extraction_time": np.std(self.extraction_times)
        }
    
    def save_model(self, model_path: str):
        """Save the trained scalers and reducers."""
        model_data = {
            'scalers': self.scalers,
            'pca': self.pca,
            'feature_selector': self.feature_selector,
            'scaler_type': self.scaler_type,
            'n_components': self.n_components,
            'feature_selection_method': self.feature_selection_method
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, model_path: str):
        """Load trained scalers and reducers."""
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.scalers = model_data['scalers']
        self.pca = model_data['pca']
        self.feature_selector = model_data['feature_selector']
        self.scaler_type = model_data['scaler_type']
        self.n_components = model_data['n_components']
        self.feature_selection_method = model_data['feature_selection_method']
    
    # Phase 4.2: Memory & CPU Optimization Methods
    
    def _check_memory_usage(self):
        """Check current memory usage and optimize if needed"""
        try:
            if not self.memory_optimization_enabled:
                return
            
            memory = psutil.virtual_memory()
            memory_percent = memory.percent / 100.0
            
            # Store memory usage history
            self.memory_usage_history.append({
                'timestamp': datetime.now(),
                'memory_percent': memory_percent,
                'available_mb': memory.available / (1024 * 1024)
            })
            
            # Keep only last 100 entries
            if len(self.memory_usage_history) > 100:
                self.memory_usage_history = self.memory_usage_history[-100:]
            
            # Trigger optimization if memory usage is high
            if memory_percent > self.memory_threshold:
                self._optimize_memory_usage()
            
            # Increment operation count for garbage collection
            self.operation_count += 1
            if self.operation_count >= self.gc_threshold:
                self._force_garbage_collection()
                self.operation_count = 0
                
        except Exception as e:
            logger.error(f"❌ Error checking memory usage: {e}")
    
    def _optimize_memory_usage(self):
        """Optimize memory usage when threshold is exceeded"""
        try:
            logger.info("🧹 Optimizing feature engineering memory usage...")
            
            # Clear feature cache if it's too large
            cache_size_mb = self._estimate_cache_size_mb()
            if cache_size_mb > self.cache_memory_limit_mb:
                self._cleanup_feature_cache()
            
            # Clear old cache files
            self._cleanup_cache()
            
            # Force garbage collection
            self._force_garbage_collection()
            
            # Clear old extraction times if too many
            if len(self.extraction_times) > 1000:
                self.extraction_times = self.extraction_times[-500:]
            
            self.feature_extraction_stats['memory_cleanups'] += 1
            logger.info("✅ Feature engineering memory optimization completed")
            
        except Exception as e:
            logger.error(f"❌ Error optimizing memory usage: {e}")
    
    def _cleanup_feature_cache(self):
        """Clean up feature cache to reduce memory usage"""
        try:
            if not self.feature_cache:
                return
            
            # Remove oldest entries (LRU policy)
            cache_size = len(self.feature_cache)
            if cache_size > 100:  # Keep only 100 most recent entries
                # Sort by timestamp and keep only recent ones
                sorted_items = sorted(self.feature_cache.items(), 
                                    key=lambda x: x[1].get('timestamp', 0), 
                                    reverse=True)
                self.feature_cache = dict(sorted_items[:100])
                
                logger.info(f"🗜️ Feature cache cleaned: {cache_size} -> {len(self.feature_cache)} items")
                
        except Exception as e:
            logger.error(f"❌ Error cleaning feature cache: {e}")
    
    def _estimate_cache_size_mb(self) -> float:
        """Estimate feature cache size in MB"""
        try:
            total_size = 0
            for key, value in self.feature_cache.items():
                # Rough estimation: assume each cache entry is ~1KB
                total_size += 1024
            
            return total_size / (1024 * 1024)  # Convert to MB
            
        except Exception as e:
            logger.error(f"❌ Error estimating cache size: {e}")
            return 0.0
    
    def _force_garbage_collection(self):
        """Force garbage collection and track statistics"""
        try:
            # Get objects before collection
            objects_before = len(gc.get_objects())
            
            # Force collection
            collected = gc.collect()
            
            # Get objects after collection
            objects_after = len(gc.get_objects())
            objects_freed = objects_before - objects_after
            
            # Update statistics
            self.feature_extraction_stats['gc_collections'] += 1
            
            logger.info(f"🧹 Feature engineering GC: {objects_freed} objects freed")
            
        except Exception as e:
            logger.error(f"❌ Error in garbage collection: {e}")
    
    def _parallel_feature_extraction(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Extract features using parallel processing for CPU optimization"""
        try:
            if not self.parallel_processing_enabled:
                return self._extract_features_sequential(df, symbol)
            
            # Check CPU usage before parallel processing
            cpu_percent = psutil.cpu_percent(interval=0.1)
            if cpu_percent > self.cpu_throttling_threshold * 100:
                logger.warning(f"⚠️ High CPU usage ({cpu_percent:.1f}%), using sequential processing")
                return self._extract_features_sequential(df, symbol)
            
            # Split data for parallel processing
            chunk_size = len(df) // 4  # Split into 4 chunks
            chunks = [df.iloc[i:i+chunk_size] for i in range(0, len(df), chunk_size)]
            
            # Process chunks in parallel
            with self.thread_pool as executor:
                futures = [executor.submit(self._extract_features_sequential, chunk, symbol) 
                          for chunk in chunks]
                results = [future.result() for future in futures]
            
            # Combine results
            combined_df = pd.concat(results, ignore_index=True)
            return combined_df
            
        except Exception as e:
            logger.error(f"❌ Error in parallel feature extraction: {e}")
            return self._extract_features_sequential(df, symbol)
    
    def _extract_features_sequential(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Sequential feature extraction (fallback method)"""
        try:
            # This would contain the original feature extraction logic
            # For now, return the input DataFrame
            return df
            
        except Exception as e:
            logger.error(f"❌ Error in sequential feature extraction: {e}")
            return df
    
    def get_phase4_2_stats(self) -> Dict[str, Any]:
        """Get Phase 4.2 optimization statistics"""
        try:
            memory_usage = psutil.virtual_memory()
            cpu_usage = psutil.cpu_percent(interval=0.1)
            
            return {
                'memory_usage_mb': psutil.Process().memory_info().rss / (1024 * 1024),
                'memory_percent': memory_usage.percent,
                'cpu_usage_percent': cpu_usage,
                'cache_size': len(self.feature_cache),
                'cache_size_mb': self._estimate_cache_size_mb(),
                'gc_collections': self.feature_extraction_stats['gc_collections'],
                'memory_cleanups': self.feature_extraction_stats['memory_cleanups'],
                'optimization_enabled': {
                    'memory': self.memory_optimization_enabled,
                    'cpu': self.cpu_optimization_enabled,
                    'parallel': self.parallel_processing_enabled
                },
                'performance_stats': self.get_performance_stats()
            }
        except Exception as e:
            logger.error(f"❌ Error getting Phase 4.2 stats: {e}")
            return {}
