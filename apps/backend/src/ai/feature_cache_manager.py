#!/usr/bin/env python3
"""
Feature Cache Manager for AlphaPulse
Pre-computes and caches technical indicators in Redis for ultra-low latency inference
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import json
import time
import hashlib
from collections import defaultdict, deque

# Redis imports
try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

# Technical analysis imports
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False

# Local imports
from ..src.data.technical_indicators import TechnicalIndicators
from ..advanced_logging_system import redis_logger, EventType, LogLevel

logger = logging.getLogger(__name__)

@dataclass
class FeatureCacheConfig:
    """Configuration for feature caching"""
    redis_url: str = "redis://localhost:6379"
    cache_ttl: int = 3600  # 1 hour cache TTL
    max_cache_size: int = 10000  # Maximum cached features per symbol
    enable_compression: bool = True
    batch_size: int = 100  # Batch size for feature computation
    update_interval: int = 60  # Update interval in seconds
    enable_async: bool = True
    feature_groups: List[str] = None  # Feature groups to cache

@dataclass
class CachedFeature:
    """Cached feature data"""
    symbol: str
    timeframe: str
    timestamp: datetime
    features: Dict[str, float]
    feature_hash: str
    cache_time: datetime
    ttl: int

class FeatureCacheManager:
    """
    Feature cache manager for pre-computing and storing technical indicators
    """
    
    def __init__(self, config: FeatureCacheConfig = None):
        """
        Initialize feature cache manager
        
        Args:
            config: Cache configuration
        """
        self.config = config or FeatureCacheConfig()
        
        # Initialize Redis connection
        self.redis_client = None
        if REDIS_AVAILABLE:
            try:
                self.redis_client = redis.from_url(self.config.redis_url)
                logger.info(f"✅ Redis connection established: {self.config.redis_url}")
            except Exception as e:
                logger.warning(f"⚠️ Redis connection failed: {e}")
                self.redis_client = None
        
        # Initialize technical indicators
        self.technical_indicators = TechnicalIndicators()
        
        # Feature computation cache
        self.feature_cache: Dict[str, CachedFeature] = {}
        self.computation_times: Dict[str, List[float]] = defaultdict(list)
        self.cache_hits: int = 0
        self.cache_misses: int = 0
        
        # Feature groups to cache
        self.feature_groups = self.config.feature_groups or [
            'momentum', 'trend', 'volatility', 'volume', 'oscillators'
        ]
        
        logger.info(f"✅ Feature cache manager initialized")
        logger.info(f"   - Redis available: {self.redis_client is not None}")
        logger.info(f"   - Feature groups: {self.feature_groups}")
        logger.info(f"   - Cache TTL: {self.config.cache_ttl}s")
    
    async def precompute_features(self, 
                                symbol: str, 
                                timeframe: str,
                                candlestick_data: pd.DataFrame) -> Dict[str, float]:
        """
        Pre-compute features for a symbol/timeframe and cache them
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            candlestick_data: OHLCV data
            
        Returns:
            Dictionary of computed features
        """
        start_time = time.time()
        
        try:
            # Generate cache key
            cache_key = self._generate_cache_key(symbol, timeframe, candlestick_data)
            
            # Check if features are already cached
            cached_features = await self._get_cached_features(cache_key)
            if cached_features is not None:
                self.cache_hits += 1
                return cached_features.features
            
            self.cache_misses += 1
            
            # Compute features
            features = await self._compute_features(candlestick_data)
            
            # Cache the features
            await self._cache_features(cache_key, symbol, timeframe, features)
            
            # Track computation time
            computation_time = time.time() - start_time
            self.computation_times[f"{symbol}_{timeframe}"].append(computation_time)
            
            logger.debug(f"✅ Pre-computed features for {symbol}_{timeframe}: {len(features)} features in {computation_time:.3f}s")
            
            return features
            
        except Exception as e:
            logger.error(f"❌ Error pre-computing features for {symbol}_{timeframe}: {e}")
            return {}
    
    async def get_features(self, 
                          symbol: str, 
                          timeframe: str,
                          candlestick_data: pd.DataFrame) -> Dict[str, float]:
        """
        Get features for a symbol/timeframe (cached or computed)
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            candlestick_data: OHLCV data
            
        Returns:
            Dictionary of features
        """
        try:
            # Generate cache key
            cache_key = self._generate_cache_key(symbol, timeframe, candlestick_data)
            
            # Try to get from cache first
            if self.redis_client:
                cached_features = await self._get_cached_features(cache_key)
                if cached_features is not None:
                    self.cache_hits += 1
                    return cached_features.features
            
            # Compute features if not cached
            self.cache_misses += 1
            features = await self._compute_features(candlestick_data)
            
            # Cache for future use
            if self.redis_client:
                await self._cache_features(cache_key, symbol, timeframe, features)
            
            return features
            
        except Exception as e:
            logger.error(f"❌ Error getting features for {symbol}_{timeframe}: {e}")
            return {}
    
    async def _compute_features(self, candlestick_data: pd.DataFrame) -> Dict[str, float]:
        """Compute technical indicators from candlestick data"""
        try:
            if candlestick_data.empty:
                return {}
            
            # Get latest candle
            latest_candle = candlestick_data.iloc[-1]
            
            features = {}
            
            # Momentum indicators
            if 'momentum' in self.feature_groups:
                features.update(await self._compute_momentum_features(candlestick_data))
            
            # Trend indicators
            if 'trend' in self.feature_groups:
                features.update(await self._compute_trend_features(candlestick_data))
            
            # Volatility indicators
            if 'volatility' in self.feature_groups:
                features.update(await self._compute_volatility_features(candlestick_data))
            
            # Volume indicators
            if 'volume' in self.feature_groups:
                features.update(await self._compute_volume_features(candlestick_data))
            
            # Oscillators
            if 'oscillators' in self.feature_groups:
                features.update(await self._compute_oscillator_features(candlestick_data))
            
            # Add price-based features
            features.update({
                'close': float(latest_candle['close']),
                'open': float(latest_candle['open']),
                'high': float(latest_candle['high']),
                'low': float(latest_candle['low']),
                'volume': float(latest_candle['volume']),
                'price_change': float(latest_candle['close'] - latest_candle['open']),
                'price_change_pct': float((latest_candle['close'] - latest_candle['open']) / latest_candle['open'] * 100),
                'high_low_ratio': float(latest_candle['high'] / latest_candle['low']) if latest_candle['low'] > 0 else 1.0,
                'body_size': float(abs(latest_candle['close'] - latest_candle['open'])),
                'upper_shadow': float(latest_candle['high'] - max(latest_candle['open'], latest_candle['close'])),
                'lower_shadow': float(min(latest_candle['open'], latest_candle['close']) - latest_candle['low'])
            })
            
            return features
            
        except Exception as e:
            logger.error(f"Error computing features: {e}")
            return {}
    
    async def _compute_momentum_features(self, candlestick_data: pd.DataFrame) -> Dict[str, float]:
        """Compute momentum indicators"""
        features = {}
        
        try:
            if len(candlestick_data) < 14:
                return features
            
            close_prices = candlestick_data['close'].values
            high_prices = candlestick_data['high'].values
            low_prices = candlestick_data['low'].values
            volume = candlestick_data['volume'].values
            
            # RSI
            if TALIB_AVAILABLE:
                rsi = talib.RSI(close_prices, timeperiod=14)
                if not np.isnan(rsi[-1]):
                    features['rsi'] = float(rsi[-1])
                
                # MACD
                macd, macd_signal, macd_hist = talib.MACD(close_prices)
                if not np.isnan(macd[-1]):
                    features['macd'] = float(macd[-1])
                    features['macd_signal'] = float(macd_signal[-1])
                    features['macd_histogram'] = float(macd_hist[-1])
                
                # Stochastic
                slowk, slowd = talib.STOCH(high_prices, low_prices, close_prices)
                if not np.isnan(slowk[-1]):
                    features['stoch_k'] = float(slowk[-1])
                    features['stoch_d'] = float(slowd[-1])
            else:
                # Fallback calculations
                features.update(self._compute_rsi_fallback(close_prices))
                features.update(self._compute_macd_fallback(close_prices))
            
            # Price momentum
            if len(close_prices) >= 10:
                features['momentum_5'] = float(close_prices[-1] / close_prices[-5] - 1) * 100
                features['momentum_10'] = float(close_prices[-1] / close_prices[-10] - 1) * 100
            
            return features
            
        except Exception as e:
            logger.error(f"Error computing momentum features: {e}")
            return features
    
    async def _compute_trend_features(self, candlestick_data: pd.DataFrame) -> Dict[str, float]:
        """Compute trend indicators"""
        features = {}
        
        try:
            if len(candlestick_data) < 20:
                return features
            
            close_prices = candlestick_data['close'].values
            high_prices = candlestick_data['high'].values
            low_prices = candlestick_data['low'].values
            
            if TALIB_AVAILABLE:
                # Moving averages
                sma_20 = talib.SMA(close_prices, timeperiod=20)
                sma_50 = talib.SMA(close_prices, timeperiod=50)
                ema_12 = talib.EMA(close_prices, timeperiod=12)
                ema_26 = talib.EMA(close_prices, timeperiod=26)
                
                if not np.isnan(sma_20[-1]):
                    features['sma_20'] = float(sma_20[-1])
                    features['price_vs_sma_20'] = float(close_prices[-1] / sma_20[-1] - 1) * 100
                
                if not np.isnan(sma_50[-1]):
                    features['sma_50'] = float(sma_50[-1])
                    features['price_vs_sma_50'] = float(close_prices[-1] / sma_50[-1] - 1) * 100
                
                if not np.isnan(ema_12[-1]) and not np.isnan(ema_26[-1]):
                    features['ema_12'] = float(ema_12[-1])
                    features['ema_26'] = float(ema_26[-1])
                    features['ema_cross'] = float(ema_12[-1] - ema_26[-1])
                
                # ADX (Average Directional Index)
                adx = talib.ADX(high_prices, low_prices, close_prices, timeperiod=14)
                if not np.isnan(adx[-1]):
                    features['adx'] = float(adx[-1])
                
                # Parabolic SAR
                sar = talib.SAR(high_prices, low_prices)
                if not np.isnan(sar[-1]):
                    features['sar'] = float(sar[-1])
                    features['price_vs_sar'] = float(close_prices[-1] / sar[-1] - 1) * 100
            else:
                # Fallback calculations
                features.update(self._compute_ma_fallback(close_prices))
            
            return features
            
        except Exception as e:
            logger.error(f"Error computing trend features: {e}")
            return features
    
    async def _compute_volatility_features(self, candlestick_data: pd.DataFrame) -> Dict[str, float]:
        """Compute volatility indicators"""
        features = {}
        
        try:
            if len(candlestick_data) < 20:
                return features
            
            close_prices = candlestick_data['close'].values
            high_prices = candlestick_data['high'].values
            low_prices = candlestick_data['low'].values
            
            if TALIB_AVAILABLE:
                # Bollinger Bands
                bb_upper, bb_middle, bb_lower = talib.BBANDS(close_prices, timeperiod=20)
                if not np.isnan(bb_upper[-1]):
                    features['bb_upper'] = float(bb_upper[-1])
                    features['bb_middle'] = float(bb_middle[-1])
                    features['bb_lower'] = float(bb_lower[-1])
                    features['bb_position'] = float((close_prices[-1] - bb_lower[-1]) / (bb_upper[-1] - bb_lower[-1]))
                    features['bb_width'] = float((bb_upper[-1] - bb_lower[-1]) / bb_middle[-1])
                
                # ATR (Average True Range)
                atr = talib.ATR(high_prices, low_prices, close_prices, timeperiod=14)
                if not np.isnan(atr[-1]):
                    features['atr'] = float(atr[-1])
                    features['atr_pct'] = float(atr[-1] / close_prices[-1] * 100)
            else:
                # Fallback calculations
                features.update(self._compute_bb_fallback(close_prices))
                features.update(self._compute_atr_fallback(high_prices, low_prices, close_prices))
            
            # Historical volatility
            if len(close_prices) >= 20:
                returns = np.diff(np.log(close_prices))
                features['volatility_20'] = float(np.std(returns[-20:]) * np.sqrt(252) * 100)
            
            return features
            
        except Exception as e:
            logger.error(f"Error computing volatility features: {e}")
            return features
    
    async def _compute_volume_features(self, candlestick_data: pd.DataFrame) -> Dict[str, float]:
        """Compute volume indicators"""
        features = {}
        
        try:
            if len(candlestick_data) < 20:
                return features
            
            close_prices = candlestick_data['close'].values
            volume = candlestick_data['volume'].values
            
            if TALIB_AVAILABLE:
                # Volume SMA
                volume_sma = talib.SMA(volume, timeperiod=20)
                if not np.isnan(volume_sma[-1]):
                    features['volume_sma_20'] = float(volume_sma[-1])
                    features['volume_ratio'] = float(volume[-1] / volume_sma[-1])
                
                # OBV (On Balance Volume)
                obv = talib.OBV(close_prices, volume)
                if len(obv) > 0:
                    features['obv'] = float(obv[-1])
                    if len(obv) >= 20:
                        obv_sma = talib.SMA(obv, timeperiod=20)
                        if not np.isnan(obv_sma[-1]):
                            features['obv_sma_20'] = float(obv_sma[-1])
                            features['obv_ratio'] = float(obv[-1] / obv_sma[-1])
            else:
                # Fallback calculations
                features.update(self._compute_volume_fallback(close_prices, volume))
            
            return features
            
        except Exception as e:
            logger.error(f"Error computing volume features: {e}")
            return features
    
    async def _compute_oscillator_features(self, candlestick_data: pd.DataFrame) -> Dict[str, float]:
        """Compute oscillator indicators"""
        features = {}
        
        try:
            if len(candlestick_data) < 14:
                return features
            
            close_prices = candlestick_data['close'].values
            high_prices = candlestick_data['high'].values
            low_prices = candlestick_data['low'].values
            
            if TALIB_AVAILABLE:
                # Williams %R
                willr = talib.WILLR(high_prices, low_prices, close_prices, timeperiod=14)
                if not np.isnan(willr[-1]):
                    features['williams_r'] = float(willr[-1])
                
                # CCI (Commodity Channel Index)
                cci = talib.CCI(high_prices, low_prices, close_prices, timeperiod=14)
                if not np.isnan(cci[-1]):
                    features['cci'] = float(cci[-1])
                
                # MFI (Money Flow Index)
                mfi = talib.MFI(high_prices, low_prices, close_prices, candlestick_data['volume'].values, timeperiod=14)
                if not np.isnan(mfi[-1]):
                    features['mfi'] = float(mfi[-1])
            
            return features
            
        except Exception as e:
            logger.error(f"Error computing oscillator features: {e}")
            return features
    
    def _compute_rsi_fallback(self, close_prices: np.ndarray) -> Dict[str, float]:
        """Fallback RSI calculation"""
        features = {}
        try:
            if len(close_prices) >= 14:
                deltas = np.diff(close_prices)
                gains = np.where(deltas > 0, deltas, 0)
                losses = np.where(deltas < 0, -deltas, 0)
                
                avg_gain = np.mean(gains[-14:])
                avg_loss = np.mean(losses[-14:])
                
                if avg_loss != 0:
                    rs = avg_gain / avg_loss
                    rsi = 100 - (100 / (1 + rs))
                    features['rsi'] = float(rsi)
        except Exception as e:
            logger.error(f"Error in RSI fallback: {e}")
        return features
    
    def _compute_macd_fallback(self, close_prices: np.ndarray) -> Dict[str, float]:
        """Fallback MACD calculation"""
        features = {}
        try:
            if len(close_prices) >= 26:
                ema_12 = self._compute_ema(close_prices, 12)
                ema_26 = self._compute_ema(close_prices, 26)
                
                if len(ema_12) > 0 and len(ema_26) > 0:
                    macd = ema_12[-1] - ema_26[-1]
                    features['macd'] = float(macd)
        except Exception as e:
            logger.error(f"Error in MACD fallback: {e}")
        return features
    
    def _compute_ema(self, prices: np.ndarray, period: int) -> np.ndarray:
        """Compute Exponential Moving Average"""
        try:
            alpha = 2 / (period + 1)
            ema = [prices[0]]
            for price in prices[1:]:
                ema.append(alpha * price + (1 - alpha) * ema[-1])
            return np.array(ema)
        except Exception:
            return np.array([])
    
    def _compute_ma_fallback(self, close_prices: np.ndarray) -> Dict[str, float]:
        """Fallback moving average calculations"""
        features = {}
        try:
            if len(close_prices) >= 20:
                sma_20 = np.mean(close_prices[-20:])
                features['sma_20'] = float(sma_20)
                features['price_vs_sma_20'] = float(close_prices[-1] / sma_20 - 1) * 100
            
            if len(close_prices) >= 50:
                sma_50 = np.mean(close_prices[-50:])
                features['sma_50'] = float(sma_50)
                features['price_vs_sma_50'] = float(close_prices[-1] / sma_50 - 1) * 100
        except Exception as e:
            logger.error(f"Error in MA fallback: {e}")
        return features
    
    def _compute_bb_fallback(self, close_prices: np.ndarray) -> Dict[str, float]:
        """Fallback Bollinger Bands calculation"""
        features = {}
        try:
            if len(close_prices) >= 20:
                sma = np.mean(close_prices[-20:])
                std = np.std(close_prices[-20:])
                
                bb_upper = sma + (2 * std)
                bb_lower = sma - (2 * std)
                
                features['bb_upper'] = float(bb_upper)
                features['bb_middle'] = float(sma)
                features['bb_lower'] = float(bb_lower)
                features['bb_position'] = float((close_prices[-1] - bb_lower) / (bb_upper - bb_lower))
        except Exception as e:
            logger.error(f"Error in BB fallback: {e}")
        return features
    
    def _compute_atr_fallback(self, high_prices: np.ndarray, low_prices: np.ndarray, close_prices: np.ndarray) -> Dict[str, float]:
        """Fallback ATR calculation"""
        features = {}
        try:
            if len(close_prices) >= 14:
                true_ranges = []
                for i in range(1, len(close_prices)):
                    tr1 = high_prices[i] - low_prices[i]
                    tr2 = abs(high_prices[i] - close_prices[i-1])
                    tr3 = abs(low_prices[i] - close_prices[i-1])
                    true_ranges.append(max(tr1, tr2, tr3))
                
                atr = np.mean(true_ranges[-14:])
                features['atr'] = float(atr)
                features['atr_pct'] = float(atr / close_prices[-1] * 100)
        except Exception as e:
            logger.error(f"Error in ATR fallback: {e}")
        return features
    
    def _compute_volume_fallback(self, close_prices: np.ndarray, volume: np.ndarray) -> Dict[str, float]:
        """Fallback volume calculations"""
        features = {}
        try:
            if len(volume) >= 20:
                volume_sma = np.mean(volume[-20:])
                features['volume_sma_20'] = float(volume_sma)
                features['volume_ratio'] = float(volume[-1] / volume_sma)
        except Exception as e:
            logger.error(f"Error in volume fallback: {e}")
        return features
    
    def _generate_cache_key(self, symbol: str, timeframe: str, candlestick_data: pd.DataFrame) -> str:
        """Generate cache key for features"""
        try:
            # Create hash from latest candle data
            latest_candle = candlestick_data.iloc[-1]
            data_hash = hashlib.md5(
                f"{symbol}_{timeframe}_{latest_candle['timestamp']}_{latest_candle['close']}".encode()
            ).hexdigest()
            
            return f"features:{symbol}:{timeframe}:{data_hash}"
            
        except Exception as e:
            logger.error(f"Error generating cache key: {e}")
            return f"features:{symbol}:{timeframe}:fallback"
    
    async def _get_cached_features(self, cache_key: str) -> Optional[CachedFeature]:
        """Get cached features from Redis"""
        try:
            if not self.redis_client:
                return None
            
            cached_data = await self.redis_client.get(cache_key)
            if cached_data:
                data = json.loads(cached_data)
                return CachedFeature(
                    symbol=data['symbol'],
                    timeframe=data['timeframe'],
                    timestamp=datetime.fromisoformat(data['timestamp']),
                    features=data['features'],
                    feature_hash=data['feature_hash'],
                    cache_time=datetime.fromisoformat(data['cache_time']),
                    ttl=data['ttl']
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting cached features: {e}")
            return None
    
    async def _cache_features(self, cache_key: str, symbol: str, timeframe: str, features: Dict[str, float]):
        """Cache features in Redis"""
        try:
            if not self.redis_client:
                return
            
            # Create feature hash
            feature_hash = hashlib.md5(json.dumps(features, sort_keys=True).encode()).hexdigest()
            
            # Prepare cache data
            cache_data = {
                'symbol': symbol,
                'timeframe': timeframe,
                'timestamp': datetime.now().isoformat(),
                'features': features,
                'feature_hash': feature_hash,
                'cache_time': datetime.now().isoformat(),
                'ttl': self.config.cache_ttl
            }
            
            # Store in Redis
            await self.redis_client.setex(
                cache_key,
                self.config.cache_ttl,
                json.dumps(cache_data)
            )
            
            logger.debug(f"✅ Cached features for {cache_key}")
            
        except Exception as e:
            logger.error(f"Error caching features: {e}")
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': self.cache_hits / (self.cache_hits + self.cache_misses) if (self.cache_hits + self.cache_misses) > 0 else 0,
            'avg_computation_time': {
                key: np.mean(times) if times else 0 
                for key, times in self.computation_times.items()
            },
            'redis_available': self.redis_client is not None,
            'feature_groups': self.feature_groups
        }
    
    async def clear_cache(self, symbol: str = None, timeframe: str = None):
        """Clear feature cache"""
        try:
            if not self.redis_client:
                return
            
            if symbol and timeframe:
                # Clear specific symbol/timeframe
                pattern = f"features:{symbol}:{timeframe}:*"
            elif symbol:
                # Clear specific symbol
                pattern = f"features:{symbol}:*"
            else:
                # Clear all features
                pattern = "features:*"
            
            keys = await self.redis_client.keys(pattern)
            if keys:
                await self.redis_client.delete(*keys)
                logger.info(f"✅ Cleared {len(keys)} cached features for pattern: {pattern}")
            
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
    
    async def close(self):
        """Close Redis connection"""
        try:
            if self.redis_client:
                await self.redis_client.close()
                logger.info("✅ Redis connection closed")
        except Exception as e:
            logger.error(f"Error closing Redis connection: {e}")

# Global instance
feature_cache_manager = FeatureCacheManager()
