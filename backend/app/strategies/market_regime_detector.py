"""
Market Regime Detector for AlphaPlus
Detects different market regimes (trending, ranging, volatile, etc.)
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class MarketRegimeDetector:
    """Market regime detector for identifying market conditions"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the Market Regime Detector"""
        self.config = config or {}
        self.is_running = False
        self.regime_cache = {}
        self.last_update = {}
        
        # Default configuration
        self.lookback_period = self.config.get('lookback_period', 50)
        self.volatility_threshold = self.config.get('volatility_threshold', 0.02)
        self.trend_threshold = self.config.get('trend_threshold', 0.01)
        self.volume_threshold = self.config.get('volume_threshold', 1.5)
        
        # Regime types
        self.regime_types = {
            'trending_up': 'Strong upward trend',
            'trending_down': 'Strong downward trend',
            'ranging': 'Sideways movement',
            'volatile': 'High volatility',
            'low_volatility': 'Low volatility',
            'breakout': 'Price breakout',
            'consolidation': 'Price consolidation'
        }
        
        logger.info("🚀 Market Regime Detector initialized")
    
    async def start(self):
        """Start the Market Regime Detector"""
        if self.is_running:
            logger.warning("Market Regime Detector is already running")
            return
            
        logger.info("🚀 Starting Market Regime Detector...")
        self.is_running = True
        
        # Start background tasks
        asyncio.create_task(self._detect_regimes())
        
        logger.info("✅ Market Regime Detector started successfully")
    
    async def stop(self):
        """Stop the Market Regime Detector"""
        if not self.is_running:
            logger.warning("Market Regime Detector is not running")
            return
            
        logger.info("🛑 Stopping Market Regime Detector...")
        self.is_running = False
        logger.info("✅ Market Regime Detector stopped successfully")
    
    async def _detect_regimes(self):
        """Background task to detect market regimes"""
        while self.is_running:
            try:
                # Detect regimes every 30 seconds
                await asyncio.sleep(30)
                
            except Exception as e:
                logger.error(f"❌ Error detecting market regimes: {e}")
                await asyncio.sleep(60)
    
    async def detect_regime(self, data: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """Detect market regime for given data"""
        try:
            if data.empty or len(data) < self.lookback_period:
                return {
                    'regime': 'unknown',
                    'confidence': 0.0,
                    'description': 'Insufficient data',
                    'timestamp': datetime.now()
                }
            
            # Calculate basic indicators
            if 'close' in data.columns:
                close_prices = data['close'].values
                
                # Calculate returns
                returns = np.diff(close_prices) / close_prices[:-1]
                
                # Calculate volatility
                volatility = np.std(returns)
                
                # Calculate trend
                trend = (close_prices[-1] - close_prices[0]) / close_prices[0]
                
                # Calculate moving averages
                sma_short = np.mean(close_prices[-10:])  # 10-period SMA
                sma_long = np.mean(close_prices[-20:])   # 20-period SMA
                
                # Determine regime based on indicators
                regime = await self._classify_regime(volatility, trend, sma_short, sma_long)
                
                result = {
                    'symbol': symbol,
                    'regime': regime,
                    'confidence': await self._calculate_confidence(volatility, trend),
                    'description': self.regime_types.get(regime, 'Unknown regime'),
                    'timestamp': datetime.now(),
                    'metrics': {
                        'volatility': volatility,
                        'trend': trend,
                        'sma_short': sma_short,
                        'sma_long': sma_long,
                        'price_change': (close_prices[-1] - close_prices[0]) / close_prices[0]
                    }
                }
                
                # Cache the result
                self.regime_cache[symbol] = result
                
                logger.info(f"✅ Detected {regime} regime for {symbol}")
                return result
                
        except Exception as e:
            logger.error(f"❌ Failed to detect regime for {symbol}: {e}")
            return {
                'regime': 'error',
                'confidence': 0.0,
                'description': f'Error: {str(e)}',
                'timestamp': datetime.now()
            }
    
    async def _classify_regime(self, volatility: float, trend: float, sma_short: float, sma_long: float) -> str:
        """Classify market regime based on indicators"""
        try:
            # High volatility regime
            if volatility > self.volatility_threshold:
                if abs(trend) > self.trend_threshold:
                    return 'volatile'
                else:
                    return 'volatile'
            
            # Low volatility regime
            if volatility < self.volatility_threshold * 0.5:
                if abs(trend) < self.trend_threshold * 0.5:
                    return 'low_volatility'
                else:
                    return 'consolidation'
            
            # Trending regimes
            if trend > self.trend_threshold:
                if sma_short > sma_long:
                    return 'trending_up'
                else:
                    return 'breakout'
            elif trend < -self.trend_threshold:
                if sma_short < sma_long:
                    return 'trending_down'
                else:
                    return 'breakout'
            
            # Ranging regime
            if abs(trend) < self.trend_threshold:
                return 'ranging'
            
            # Default to ranging
            return 'ranging'
            
        except Exception as e:
            logger.error(f"❌ Failed to classify regime: {e}")
            return 'unknown'
    
    async def _calculate_confidence(self, volatility: float, trend: float) -> float:
        """Calculate confidence in regime detection"""
        try:
            # Base confidence
            confidence = 0.5
            
            # Adjust based on volatility
            if volatility > self.volatility_threshold:
                confidence += 0.2
            elif volatility < self.volatility_threshold * 0.5:
                confidence += 0.1
            
            # Adjust based on trend strength
            if abs(trend) > self.trend_threshold:
                confidence += 0.2
            elif abs(trend) < self.trend_threshold * 0.5:
                confidence += 0.1
            
            return min(confidence, 1.0)
            
        except Exception as e:
            logger.error(f"❌ Failed to calculate confidence: {e}")
            return 0.0
    
    async def get_regime(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get current regime for a symbol"""
        try:
            return self.regime_cache.get(symbol)
        except Exception as e:
            logger.error(f"❌ Failed to get regime for {symbol}: {e}")
            return None
    
    async def get_all_regimes(self) -> Dict[str, Dict[str, Any]]:
        """Get all detected regimes"""
        try:
            return self.regime_cache.copy()
        except Exception as e:
            logger.error(f"❌ Failed to get all regimes: {e}")
            return {}
    
    async def get_regime_statistics(self) -> Dict[str, Any]:
        """Get statistics about detected regimes"""
        try:
            if not self.regime_cache:
                return {
                    'total_symbols': 0,
                    'regime_distribution': {},
                    'average_confidence': 0.0
                }
            
            # Count regimes
            regime_counts = {}
            total_confidence = 0.0
            
            for symbol, regime_data in self.regime_cache.items():
                regime = regime_data.get('regime', 'unknown')
                regime_counts[regime] = regime_counts.get(regime, 0) + 1
                total_confidence += regime_data.get('confidence', 0.0)
            
            avg_confidence = total_confidence / len(self.regime_cache)
            
            return {
                'total_symbols': len(self.regime_cache),
                'regime_distribution': regime_counts,
                'average_confidence': round(avg_confidence, 3)
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get regime statistics: {e}")
            return {}
    
    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the Market Regime Detector"""
        return {
            'status': 'running' if self.is_running else 'stopped',
            'symbols_monitored': len(self.regime_cache),
            'lookback_period': self.lookback_period,
            'volatility_threshold': self.volatility_threshold,
            'trend_threshold': self.trend_threshold,
            'last_update': self.last_update.get('regimes', None)
        }
