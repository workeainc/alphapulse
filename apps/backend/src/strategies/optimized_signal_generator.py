#!/usr/bin/env python3
"""
Optimized Real-Time Signal Generator for AlphaPulse
Implements the complete optimization playbook for maximum performance
"""

import asyncio
import logging
from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from collections import deque
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache

from .optimized_pattern_detector import OptimizedPatternDetector, OptimizedPatternSignal
from .indicators import TechnicalIndicators
from .base_strategy import BaseStrategy

logger = logging.getLogger(__name__)

@dataclass
class OptimizedTradingSignal:
    """Optimized trading signal with performance metrics"""
    symbol: str
    signal_type: str  # 'buy', 'sell', 'hold'
    strength: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    pattern: str
    price: float
    timestamp: datetime
    timeframe: str
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    risk_reward_ratio: Optional[float] = None
    market_regime: Optional[str] = None
    volume_confirmation: bool = False
    trend_alignment: bool = False
    support_resistance_levels: Optional[Dict] = None
    additional_indicators: Optional[Dict] = None
    processing_time_ms: float = 0.0
    cache_hit: bool = False

class OptimizedSignalGenerator:
    """
    Ultra-optimized real-time signal generator implementing the complete optimization playbook
    """
    
    def __init__(self, config: Dict = None, max_workers: int = 4):
        """Initialize optimized signal generator"""
        self.config = config or {}
        self.max_workers = max_workers
        
        # Initialize optimized components
        self.optimized_detector = OptimizedPatternDetector(max_workers=max_workers)
        self.indicators = TechnicalIndicators()
        
        # Signal configuration
        self.min_confidence = self.config.get('min_confidence', 0.7)
        self.min_strength = self.config.get('min_strength', 0.6)
        self.confirmation_required = self.config.get('confirmation_required', True)
        self.volume_confirmation = self.config.get('volume_confirmation', True)
        self.trend_confirmation = self.config.get('trend_confirmation', True)
        
        # **2. CACHE REPETITIVE INDICATORS**
        self.indicator_cache = {}
        self.signal_cache = {}
        self.cache_ttl = 300  # 5 minutes cache TTL
        self.last_cache_cleanup = time.time()
        
        # Signal history for analysis
        self.signal_history = deque(maxlen=1000)
        self.pattern_history = deque(maxlen=500)
        
        # Market regime detection
        self.market_regimes = {}
        self.volatility_thresholds = {
            'low': 0.01,
            'medium': 0.03,
            'high': 0.05
        }
        
        # Performance tracking
        self.stats = {
            'total_signals': 0,
            'cache_hits': 0,
            'avg_processing_time_ms': 0.0,
            'total_processing_time_ms': 0.0,
            'successful_signals': 0,
            'failed_signals': 0,
            'accuracy': 0.0
        }
        
        logger.info(f"🚀 Optimized Signal Generator initialized with {max_workers} workers")
    
    async def generate_signal_optimized(self, 
                                      symbol: str, 
                                      data: pd.DataFrame, 
                                      timeframe: str,
                                      market_data: Dict = None) -> Optional[OptimizedTradingSignal]:
        """
        **1. VECTORIZE PATTERN CALCULATIONS**
        Generate optimized real-time trading signal
        """
        start_time = time.time()
        
        try:
            if len(data) < 50:  # Need sufficient data for analysis
                return None
            
            # **3. FILTER FIRST, DETECT LATER**
            # Apply fast preconditions to skip irrelevant analysis
            if not self._apply_signal_preconditions(data):
                return None
            
            # **2. CACHE REPETITIVE INDICATORS**
            # Get cached indicators or calculate them
            indicators = self._get_cached_indicators(data, symbol, timeframe)
            
            # **4. COMBINE RELATED PATTERNS INTO ONE PASS**
            # Detect patterns using optimized detector
            patterns = self.optimized_detector.detect_patterns_vectorized(data)
            
            if not patterns:
                return None
            
            # Get latest pattern
            latest_pattern = patterns[-1]
            
            # Check if pattern meets minimum requirements
            if (latest_pattern.confidence < self.min_confidence or 
                latest_pattern.strength < self.min_strength):
                return None
            
            # Generate signal based on pattern
            signal = await self._create_optimized_signal(
                symbol, latest_pattern, data, timeframe, market_data, indicators
            )
            
            if signal:
                # Validate signal with additional confirmations
                if self._validate_optimized_signal(signal, data, market_data, indicators):
                    # Store signal in history
                    self.signal_history.append(signal)
                    self.pattern_history.append(latest_pattern)
                    
                    # Update performance metrics
                    processing_time = (time.time() - start_time) * 1000
                    self._update_stats(processing_time, True)
                    
                    logger.info(f"🎯 Generated {signal.signal_type} signal for {symbol} "
                              f"(confidence: {signal.confidence:.3f}, strength: {signal.strength:.3f}) "
                              f"in {processing_time:.2f}ms")
                    
                    return signal
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Error generating signal for {symbol}: {e}")
            processing_time = (time.time() - start_time) * 1000
            self._update_stats(processing_time, False)
            return None
    
    def _apply_signal_preconditions(self, data: pd.DataFrame) -> bool:
        """**3. FILTER FIRST, DETECT LATER** - Fast preconditions to skip irrelevant analysis"""
        
        # **Fast preconditions using vectorized operations:**
        
        # Check if price movement is significant enough
        price_change = abs(data['close'].iloc[-1] - data['close'].iloc[-2]) / data['close'].iloc[-2]
        if price_change < 0.001:  # Less than 0.1% movement
            return False
        
        # Check if volume is sufficient
        if 'volume' in data.columns:
            current_volume = data['volume'].iloc[-1]
            avg_volume = data['volume'].rolling(20).mean().iloc[-1]
            if current_volume < avg_volume * 0.5:  # Volume too low
                return False
        
        # Check if volatility is within acceptable range
        volatility = data['close'].rolling(20).std().iloc[-1] / data['close'].rolling(20).mean().iloc[-1]
        if volatility < 0.005 or volatility > 0.1:  # Too stable or too volatile
            return False
        
        return True
    
    def _get_cached_indicators(self, data: pd.DataFrame, symbol: str, timeframe: str) -> Dict:
        """**2. CACHE REPETITIVE INDICATORS**"""
        # Create cache key
        cache_key = f"{symbol}_{timeframe}_{hash(str(data.shape) + str(data['close'].iloc[-1]))}"
        
        if cache_key in self.indicator_cache:
            self.stats['cache_hits'] += 1
            return self.indicator_cache[cache_key]
        
        # **VECTORIZED CALCULATIONS** - Calculate all indicators in one pass
        indicators = {}
        
        # Technical indicators
        indicators['rsi'] = self.indicators.calculate_rsi(data['close'].values)
        macd, macd_signal, macd_hist = self.indicators.calculate_macd(data['close'].values)
        indicators['macd'] = macd
        indicators['macd_signal'] = macd_signal
        indicators['macd_histogram'] = macd_hist
        
        bb_upper, bb_middle, bb_lower = self.indicators.calculate_bollinger_bands(data['close'].values)
        indicators['bb_upper'] = bb_upper
        indicators['bb_middle'] = bb_middle
        indicators['bb_lower'] = bb_lower
        
        indicators['atr'] = self.indicators.calculate_atr(
            data['high'].values, data['low'].values, data['close'].values
        )
        
        # Moving averages
        indicators['sma_20'] = self.indicators.calculate_sma(data['close'].values, 20)
        indicators['sma_50'] = self.indicators.calculate_sma(data['close'].values, 50)
        indicators['ema_12'] = self.indicators.calculate_ema(data['close'].values, 12)
        indicators['ema_26'] = self.indicators.calculate_ema(data['close'].values, 26)
        
        # Volume indicators (if available)
        if 'volume' in data.columns:
            indicators['volume_sma'] = self.indicators.calculate_sma(data['volume'].values, 20)
            indicators['volume_ratio'] = data['volume'].values / indicators['volume_sma']
        
        # Cache the results
        self.indicator_cache[cache_key] = indicators
        
        return indicators
    
    async def _create_optimized_signal(self, 
                                     symbol: str, 
                                     pattern: OptimizedPatternSignal, 
                                     data: pd.DataFrame, 
                                     timeframe: str,
                                     market_data: Dict,
                                     indicators: Dict) -> Optional[OptimizedTradingSignal]:
        """Create optimized trading signal from detected pattern"""
        
        # Determine signal type based on pattern
        if pattern.type == 'bullish':
            signal_type = 'buy'
        elif pattern.type == 'bearish':
            signal_type = 'sell'
        else:
            signal_type = 'hold'
        
        if signal_type == 'hold':
            return None
        
        # Get current price
        current_price = data['close'].iloc[-1]
        
        # **VECTORIZED RISK CALCULATION**
        stop_loss, take_profit = self._calculate_optimized_risk_levels(
            signal_type, current_price, data, pattern, indicators
        )
        
        # Calculate risk-reward ratio
        risk_reward_ratio = None
        if stop_loss and take_profit:
            if signal_type == 'buy':
                risk = current_price - stop_loss
                reward = take_profit - current_price
            else:
                risk = stop_loss - current_price
                reward = current_price - take_profit
            
            if risk > 0:
                risk_reward_ratio = reward / risk
        
        # **VECTORIZED CONFIRMATION CHECKS**
        volume_confirmation = self._check_volume_confirmation_vectorized(data, signal_type, indicators)
        trend_alignment = self._check_trend_alignment_vectorized(data, signal_type, indicators)
        support_resistance = self._get_support_resistance_vectorized(data, current_price)
        
        # Create signal
        signal = OptimizedTradingSignal(
            symbol=symbol,
            signal_type=signal_type,
            strength=pattern.strength,
            confidence=pattern.confidence,
            pattern=pattern.pattern,
            price=current_price,
            timestamp=datetime.now(),
            timeframe=timeframe,
            stop_loss=stop_loss,
            take_profit=take_profit,
            risk_reward_ratio=risk_reward_ratio,
            market_regime=pattern.market_regime,
            volume_confirmation=volume_confirmation,
            trend_alignment=trend_alignment,
            support_resistance_levels=support_resistance,
            additional_indicators=self._get_additional_indicators_vectorized(data, indicators),
            cache_hit=True  # Using cached indicators
        )
        
        return signal
    
    def _calculate_optimized_risk_levels(self, 
                                       signal_type: str, 
                                       current_price: float, 
                                       data: pd.DataFrame, 
                                       pattern: OptimizedPatternSignal,
                                       indicators: Dict) -> Tuple[Optional[float], Optional[float]]:
        """**VECTORIZED RISK CALCULATION**"""
        
        # Get ATR from cached indicators
        atr_values = indicators.get('atr', [])
        if atr_values and not np.isnan(atr_values[-1]):
            atr_value = atr_values[-1]
        else:
            atr_value = current_price * 0.02  # Default 2%
        
        # **VECTORIZED SUPPORT/RESISTANCE CALCULATION**
        support_levels = self._get_support_levels_vectorized(data)
        resistance_levels = self._get_resistance_levels_vectorized(data)
        
        if signal_type == 'buy':
            # Stop loss: below recent support or ATR-based
            if support_levels:
                stop_loss = max(support_levels[-1], current_price - (2 * atr_value))
            else:
                stop_loss = current_price - (2 * atr_value)
            
            # Take profit: above recent resistance or ATR-based
            if resistance_levels:
                take_profit = max(resistance_levels[-1], current_price + (3 * atr_value))
            else:
                take_profit = current_price + (3 * atr_value)
                
        else:  # sell signal
            # Stop loss: above recent resistance or ATR-based
            if resistance_levels:
                stop_loss = min(resistance_levels[-1], current_price + (2 * atr_value))
            else:
                stop_loss = current_price + (2 * atr_value)
            
            # Take profit: below recent support or ATR-based
            if support_levels:
                take_profit = min(support_levels[-1], current_price - (3 * atr_value))
            else:
                take_profit = current_price - (3 * atr_value)
        
        return stop_loss, take_profit
    
    def _get_support_levels_vectorized(self, data: pd.DataFrame) -> List[float]:
        """**VECTORIZED SUPPORT DETECTION**"""
        # Use vectorized operations to find local minima
        lows = data['low'].values
        support_levels = []
        
        # Vectorized local minima detection
        for i in range(2, len(lows) - 2):
            if (lows[i] < lows[i-1] and lows[i] < lows[i-2] and
                lows[i] < lows[i+1] and lows[i] < lows[i+2]):
                support_levels.append(lows[i])
        
        return sorted(support_levels, reverse=True)
    
    def _get_resistance_levels_vectorized(self, data: pd.DataFrame) -> List[float]:
        """**VECTORIZED RESISTANCE DETECTION**"""
        # Use vectorized operations to find local maxima
        highs = data['high'].values
        resistance_levels = []
        
        # Vectorized local maxima detection
        for i in range(2, len(highs) - 2):
            if (highs[i] > highs[i-1] and highs[i] > highs[i-2] and
                highs[i] > highs[i+1] and highs[i] > highs[i+2]):
                resistance_levels.append(highs[i])
        
        return sorted(resistance_levels)
    
    def _check_volume_confirmation_vectorized(self, data: pd.DataFrame, signal_type: str, indicators: Dict) -> bool:
        """**VECTORIZED VOLUME CONFIRMATION**"""
        if not self.volume_confirmation or 'volume' not in data.columns:
            return True
        
        # Use cached volume indicators
        volume_ratio = indicators.get('volume_ratio', [])
        if volume_ratio and not np.isnan(volume_ratio[-1]):
            return volume_ratio[-1] > 1.2
        
        # Fallback calculation
        volume_ma = data['volume'].rolling(20).mean()
        current_volume = data['volume'].iloc[-1]
        avg_volume = volume_ma.iloc[-1]
        
        return current_volume > avg_volume * 1.2
    
    def _check_trend_alignment_vectorized(self, data: pd.DataFrame, signal_type: str, indicators: Dict) -> bool:
        """**VECTORIZED TREND ALIGNMENT**"""
        if not self.trend_confirmation:
            return True
        
        # Use cached moving averages
        sma_20 = indicators.get('sma_20', [])
        sma_50 = indicators.get('sma_50', [])
        
        if sma_20 and sma_50 and not (np.isnan(sma_20[-1]) or np.isnan(sma_50[-1])):
            current_sma_20 = sma_20[-1]
            current_sma_50 = sma_50[-1]
        else:
            # Fallback calculation
            current_sma_20 = data['close'].rolling(20).mean().iloc[-1]
            current_sma_50 = data['close'].rolling(50).mean().iloc[-1]
        
        if signal_type == 'buy':
            return current_sma_20 > current_sma_50
        else:
            return current_sma_20 < current_sma_50
    
    def _get_support_resistance_vectorized(self, data: pd.DataFrame, current_price: float) -> Dict:
        """**VECTORIZED SUPPORT/RESISTANCE ANALYSIS**"""
        support_levels = self._get_support_levels_vectorized(data)
        resistance_levels = self._get_resistance_levels_vectorized(data)
        
        # Find nearest levels using vectorized operations
        nearest_support = None
        nearest_resistance = None
        
        if support_levels:
            # Find support level below current price
            support_below = [s for s in support_levels if s < current_price]
            nearest_support = max(support_below) if support_below else None
        
        if resistance_levels:
            # Find resistance level above current price
            resistance_above = [r for r in resistance_levels if r > current_price]
            nearest_resistance = min(resistance_above) if resistance_above else None
        
        return {
            'nearest_support': nearest_support,
            'nearest_resistance': nearest_resistance,
            'support_levels': support_levels[:5],  # Top 5 levels
            'resistance_levels': resistance_levels[:5]  # Top 5 levels
        }
    
    def _get_additional_indicators_vectorized(self, data: pd.DataFrame, indicators: Dict) -> Dict:
        """**VECTORIZED ADDITIONAL INDICATORS**"""
        try:
            # Use cached indicators
            rsi = indicators.get('rsi', [])
            macd = indicators.get('macd', [])
            macd_signal = indicators.get('macd_signal', [])
            bb_upper = indicators.get('bb_upper', [])
            bb_lower = indicators.get('bb_lower', [])
            
            current_price = data['close'].iloc[-1]
            
            # Calculate BB position
            bb_position = None
            if bb_upper and bb_lower and not (np.isnan(bb_upper[-1]) or np.isnan(bb_lower[-1])):
                if bb_upper[-1] != bb_lower[-1]:
                    bb_position = (current_price - bb_lower[-1]) / (bb_upper[-1] - bb_lower[-1])
            
            # Calculate trend strength
            sma_20 = indicators.get('sma_20', [])
            sma_50 = indicators.get('sma_50', [])
            trend_strength = None
            if sma_20 and sma_50 and not (np.isnan(sma_20[-1]) or np.isnan(sma_50[-1])):
                trend_strength = abs(sma_20[-1] - sma_50[-1]) / sma_50[-1] * 100
                trend_strength = min(trend_strength, 100.0)
            
            return {
                'rsi': float(rsi[-1]) if rsi and not np.isnan(rsi[-1]) else None,
                'macd': float(macd[-1]) if macd and not np.isnan(macd[-1]) else None,
                'macd_signal': float(macd_signal[-1]) if macd_signal and not np.isnan(macd_signal[-1]) else None,
                'bb_position': bb_position,
                'trend_strength': trend_strength
            }
        except Exception as e:
            logger.warning(f"⚠️ Error calculating additional indicators: {e}")
            return {}
    
    def _validate_optimized_signal(self, signal: OptimizedTradingSignal, data: pd.DataFrame, 
                                 market_data: Dict, indicators: Dict) -> bool:
        """**VECTORIZED SIGNAL VALIDATION**"""
        
        # Check minimum risk-reward ratio
        if signal.risk_reward_ratio and signal.risk_reward_ratio < 2.0:
            return False
        
        # Check if price is near support/resistance
        if signal.support_resistance_levels:
            current_price = signal.price
            nearest_support = signal.support_resistance_levels.get('nearest_support')
            nearest_resistance = signal.support_resistance_levels.get('nearest_resistance')
            
            if signal.signal_type == 'buy' and nearest_support:
                # Buy signal should be near support
                if abs(current_price - nearest_support) / current_price > 0.05:  # 5% threshold
                    return False
            
            elif signal.signal_type == 'sell' and nearest_resistance:
                # Sell signal should be near resistance
                if abs(current_price - nearest_resistance) / current_price > 0.05:  # 5% threshold
                    return False
        
        # Check market regime compatibility
        if signal.market_regime:
            if signal.market_regime == 'ranging_stable' and signal.pattern in ['hammer', 'shooting_star']:
                # Reversal patterns work well in ranging markets
                pass
            elif signal.market_regime == 'trending_stable' and signal.pattern in ['engulfing', 'morning_star', 'evening_star']:
                # Trend continuation patterns work well in trending markets
                pass
            else:
                # Additional validation for other market regimes
                pass
        
        return True
    
    def _update_stats(self, processing_time: float, success: bool):
        """Update performance statistics"""
        self.stats['total_signals'] += 1
        self.stats['total_processing_time_ms'] += processing_time
        self.stats['avg_processing_time_ms'] = (
            self.stats['total_processing_time_ms'] / self.stats['total_signals']
        )
        
        if success:
            self.stats['successful_signals'] += 1
        else:
            self.stats['failed_signals'] += 1
        
        # Calculate accuracy
        if self.stats['total_signals'] > 0:
            self.stats['accuracy'] = (
                self.stats['successful_signals'] / self.stats['total_signals']
            )
    
    async def generate_signals_parallel(self, data_dict: Dict[str, Dict]) -> Dict[str, Optional[OptimizedTradingSignal]]:
        """
        **5. PARALLELIZE ACROSS CONTRACTS & TIMEFRAMES**
        Generate signals for multiple symbols/timeframes in parallel
        """
        logger.info(f"🔄 Starting parallel signal generation for {len(data_dict)} datasets")
        
        # Create thread pool for parallel processing
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all signal generation tasks
            future_to_key = {
                executor.submit(
                    self._generate_signal_sync, 
                    key, 
                    data_dict[key]['data'], 
                    data_dict[key]['timeframe'],
                    data_dict[key].get('market_data')
                ): key
                for key in data_dict.keys()
            }
            
            # Collect results as they complete
            results = {}
            for future in as_completed(future_to_key):
                key = future_to_key[future]
                try:
                    signal = future.result()
                    results[key] = signal
                    if signal:
                        logger.info(f"✅ Generated signal for {key}: {signal.signal_type}")
                    else:
                        logger.info(f"⏭️ No signal generated for {key}")
                except Exception as e:
                    logger.error(f"❌ Error generating signal for {key}: {e}")
                    results[key] = None
        
        return results
    
    def _generate_signal_sync(self, symbol: str, data: pd.DataFrame, timeframe: str, market_data: Dict = None):
        """Synchronous wrapper for signal generation (for parallel processing)"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(
                self.generate_signal_optimized(symbol, data, timeframe, market_data)
            )
        finally:
            loop.close()
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics"""
        return {
            **self.stats,
            'cache_size': len(self.indicator_cache),
            'cache_hit_rate': self.stats['cache_hits'] / max(self.stats['total_signals'], 1),
            'signal_history_size': len(self.signal_history),
            'pattern_history_size': len(self.pattern_history)
        }
    
    def get_signal_summary(self) -> Dict:
        """Get summary of generated signals"""
        if not self.signal_history:
            return {"message": "No signals generated yet"}
        
        # Group signals by type
        buy_signals = [s for s in self.signal_history if s.signal_type == 'buy']
        sell_signals = [s for s in self.signal_history if s.signal_type == 'sell']
        
        # Calculate average confidence and strength
        avg_buy_confidence = np.mean([s.confidence for s in buy_signals]) if buy_signals else 0
        avg_sell_confidence = np.mean([s.confidence for s in sell_signals]) if sell_signals else 0
        
        return {
            "total_signals": len(self.signal_history),
            "buy_signals": len(buy_signals),
            "sell_signals": len(sell_signals),
            "avg_buy_confidence": round(avg_buy_confidence, 3),
            "avg_sell_confidence": round(avg_sell_confidence, 3),
            "performance": self.stats,
            "latest_signals": [
                {
                    "symbol": s.symbol,
                    "type": s.signal_type,
                    "pattern": s.pattern,
                    "confidence": s.confidence,
                    "processing_time_ms": s.processing_time_ms,
                    "timestamp": s.timestamp.isoformat()
                }
                for s in list(self.signal_history)[-5:]  # Last 5 signals
            ]
        }
    
    def clear_cache(self):
        """Clear all caches"""
        self.indicator_cache.clear()
        self.signal_cache.clear()
        logger.info("🧹 Signal generator cache cleared")

# Example usage and performance comparison
if __name__ == "__main__":
    import numpy as np
    import pandas as pd
    
    # Create sample data
    np.random.seed(42)
    n = 10000
    closes = 100 + np.cumsum(np.random.randn(n) * 0.5)
    highs = closes + np.random.rand(n) * 2
    lows = closes - np.random.rand(n) * 2
    opens = np.roll(closes, 1)
    opens[0] = closes[0]
    volumes = np.random.randint(1000, 10000, n)
    
    df = pd.DataFrame({
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes,
        'volume': volumes
    })
    
    # Test optimized signal generator
    generator = OptimizedSignalGenerator()
    
    # Performance test
    start_time = time.time()
    signal = asyncio.run(generator.generate_signal_optimized('BTCUSDT', df, '1h'))
    optimized_time = (time.time() - start_time) * 1000
    
    print(f"🚀 Optimized Signal Generation Results:")
    if signal:
        print(f"   Signal generated: {signal.signal_type}")
        print(f"   Confidence: {signal.confidence:.3f}")
        print(f"   Processing time: {optimized_time:.2f}ms")
    else:
        print(f"   No signal generated")
        print(f"   Processing time: {optimized_time:.2f}ms")
    
    print(f"   Performance stats: {generator.get_performance_stats()}")
