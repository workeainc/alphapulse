#!/usr/bin/env python3
"""
Technical Indicators Engine
Phase 2C: Enhanced Feature Engineering
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import warnings

# Suppress pandas warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

logger = logging.getLogger(__name__)

@dataclass
class IndicatorConfig:
    """Configuration for technical indicators"""
    name: str
    description: str
    parameters: Dict[str, Any]
    min_periods: int
    max_periods: int
    output_type: str  # 'value', 'signal', 'position'
    tags: List[str]

class TechnicalIndicatorsEngine:
    """Advanced technical indicators engine for trading features"""
    
    def __init__(self):
        self.indicators = {}
        self._initialize_indicators()
        logger.info("🚀 Technical Indicators Engine initialized")
    
    def _initialize_indicators(self):
        """Initialize all available technical indicators"""
        self.indicators = {
            # Momentum Indicators
            "rsi": IndicatorConfig(
                name="rsi",
                description="Relative Strength Index",
                parameters={"period": 14, "overbought": 70, "oversold": 30},
                min_periods=14,
                max_periods=50,
                output_type="value",
                tags=["momentum", "oscillator", "trend_reversal"]
            ),
            "stoch_rsi": IndicatorConfig(
                name="stoch_rsi",
                description="Stochastic RSI",
                parameters={"rsi_period": 14, "stoch_period": 14, "k_period": 3, "d_period": 3},
                min_periods=14,
                max_periods=50,
                output_type="value",
                tags=["momentum", "oscillator", "divergence"]
            ),
            "williams_r": IndicatorConfig(
                name="williams_r",
                description="Williams %R",
                parameters={"period": 14},
                min_periods=14,
                max_periods=50,
                output_type="value",
                tags=["momentum", "oscillator", "overbought_oversold"]
            ),
            
            # Trend Indicators
            "ema": IndicatorConfig(
                name="ema",
                description="Exponential Moving Average",
                parameters={"period": 20, "alpha": None},
                min_periods=5,
                max_periods=200,
                output_type="value",
                tags=["trend", "smoothing", "support_resistance"]
            ),
            "sma": IndicatorConfig(
                name="sma",
                description="Simple Moving Average",
                parameters={"period": 20},
                min_periods=5,
                max_periods=200,
                output_type="value",
                tags=["trend", "smoothing", "support_resistance"]
            ),
            "macd": IndicatorConfig(
                name="macd",
                description="Moving Average Convergence Divergence",
                parameters={"fast_period": 12, "slow_period": 26, "signal_period": 9},
                min_periods=26,
                max_periods=100,
                output_type="signal",
                tags=["trend", "momentum", "divergence"]
            ),
            "adx": IndicatorConfig(
                name="adx",
                description="Average Directional Index",
                parameters={"period": 14},
                min_periods=14,
                max_periods=50,
                output_type="value",
                tags=["trend", "strength", "direction"]
            ),
            
            # Volatility Indicators
            "bollinger_bands": IndicatorConfig(
                name="bollinger_bands",
                description="Bollinger Bands",
                parameters={"period": 20, "std_dev": 2.0},
                min_periods=20,
                max_periods=100,
                output_type="position",
                tags=["volatility", "mean_reversion", "breakout"]
            ),
            "atr": IndicatorConfig(
                name="atr",
                description="Average True Range",
                parameters={"period": 14},
                min_periods=14,
                max_periods=50,
                output_type="value",
                tags=["volatility", "risk", "position_sizing"]
            ),
            "keltner_channels": IndicatorConfig(
                name="keltner_channels",
                description="Keltner Channels",
                parameters={"ema_period": 20, "atr_period": 10, "multiplier": 2.0},
                min_periods=20,
                max_periods=100,
                output_type="position",
                tags=["volatility", "trend", "breakout"]
            ),
            
            # Volume Indicators
            "volume_sma_ratio": IndicatorConfig(
                name="volume_sma_ratio",
                description="Volume to Simple Moving Average Ratio",
                parameters={"volume_period": 20, "price_period": 20},
                min_periods=20,
                max_periods=100,
                output_type="value",
                tags=["volume", "confirmation", "divergence"]
            ),
            "obv": IndicatorConfig(
                name="obv",
                description="On-Balance Volume",
                parameters={"smooth_period": 20},
                min_periods=20,
                max_periods=100,
                output_type="value",
                tags=["volume", "trend", "confirmation"]
            ),
            "vwap": IndicatorConfig(
                name="vwap",
                description="Volume Weighted Average Price",
                parameters={"period": 20},
                min_periods=20,
                max_periods=100,
                output_type="value",
                tags=["volume", "price", "institutional"]
            ),
            
            # Advanced Indicators
            "ichimoku": IndicatorConfig(
                name="ichimoku",
                description="Ichimoku Cloud",
                parameters={"tenkan_period": 9, "kijun_period": 26, "senkou_span_b_period": 52},
                min_periods=52,
                max_periods=200,
                output_type="position",
                tags=["trend", "support_resistance", "cloud"]
            ),
            "fibonacci_retracement": IndicatorConfig(
                name="fibonacci_retracement",
                description="Fibonacci Retracement Levels",
                parameters={"levels": [0.236, 0.382, 0.5, 0.618, 0.786]},
                min_periods=20,
                max_periods=100,
                output_type="position",
                tags=["support_resistance", "retracement", "fibonacci"]
            ),
            "pivot_points": IndicatorConfig(
                name="pivot_points",
                description="Pivot Point Support/Resistance",
                parameters={"method": "standard"},  # standard, fibonacci, camarilla
                min_periods=1,
                max_periods=1,
                output_type="position",
                tags=["support_resistance", "intraday", "pivot"]
            )
        }
    
    def get_available_indicators(self) -> List[str]:
        """Get list of available indicators"""
        return list(self.indicators.keys())
    
    def get_indicator_config(self, indicator_name: str) -> Optional[IndicatorConfig]:
        """Get configuration for a specific indicator"""
        return self.indicators.get(indicator_name.lower())
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index (RSI)"""
        try:
            if len(prices) < period + 1:
                return pd.Series(index=prices.index, dtype=float)
            
            # Calculate price changes
            delta = prices.diff()
            
            # Separate gains and losses
            gains = delta.where(delta > 0, 0)
            losses = -delta.where(delta < 0, 0)
            
            # Calculate average gains and losses
            avg_gains = gains.rolling(window=period).mean()
            avg_losses = losses.rolling(window=period).mean()
            
            # Calculate RS and RSI
            rs = avg_gains / avg_losses
            rsi = 100 - (100 / (1 + rs))
            
            return rsi
            
        except Exception as e:
            logger.error(f"❌ Error calculating RSI: {e}")
            return pd.Series(index=prices.index, dtype=float)
    
    def calculate_macd(self, prices: pd.Series, fast_period: int = 12, 
                       slow_period: int = 26, signal_period: int = 9) -> Dict[str, pd.Series]:
        """Calculate MACD (Moving Average Convergence Divergence)"""
        try:
            if len(prices) < slow_period + signal_period:
                return {
                    "macd": pd.Series(index=prices.index, dtype=float),
                    "signal": pd.Series(index=prices.index, dtype=float),
                    "histogram": pd.Series(index=prices.index, dtype=float)
                }
            
            # Calculate EMAs
            ema_fast = prices.ewm(span=fast_period).mean()
            ema_slow = prices.ewm(span=slow_period).mean()
            
            # Calculate MACD line
            macd_line = ema_fast - ema_slow
            
            # Calculate signal line
            signal_line = macd_line.ewm(span=signal_period).mean()
            
            # Calculate histogram
            histogram = macd_line - signal_line
            
            return {
                "macd": macd_line,
                "signal": signal_line,
                "histogram": histogram
            }
            
        except Exception as e:
            logger.error(f"❌ Error calculating MACD: {e}")
            return {
                "macd": pd.Series(index=prices.index, dtype=float),
                "signal": pd.Series(index=prices.index, dtype=float),
                "histogram": pd.Series(index=prices.index, dtype=float)
            }
    
    def calculate_ema(self, prices: pd.Series, period: int = 20) -> pd.Series:
        """Calculate Exponential Moving Average (EMA)"""
        try:
            if len(prices) < period:
                return pd.Series(index=prices.index, dtype=float)
            
            # Calculate alpha (smoothing factor)
            alpha = 2 / (period + 1)
            
            # Calculate EMA
            ema = prices.ewm(span=period).mean()
            
            return ema
            
        except Exception as e:
            logger.error(f"❌ Error calculating EMA: {e}")
            return pd.Series(index=prices.index, dtype=float)
    
    def calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, 
                                 std_dev: float = 2.0) -> Dict[str, pd.Series]:
        """Calculate Bollinger Bands"""
        try:
            if len(prices) < period:
                return {
                    "upper": pd.Series(index=prices.index, dtype=float),
                    "middle": pd.Series(index=prices.index, dtype=float),
                    "lower": pd.Series(index=prices.index, dtype=float),
                    "position": pd.Series(index=prices.index, dtype=float)
                }
            
            # Calculate middle band (SMA)
            middle = prices.rolling(window=period).mean()
            
            # Calculate standard deviation
            std = prices.rolling(window=period).std()
            
            # Calculate upper and lower bands
            upper = middle + (std * std_dev)
            lower = middle - (std * std_dev)
            
            # Calculate position within bands (0 = at lower, 1 = at upper)
            position = (prices - lower) / (upper - lower)
            position = position.clip(0, 1)  # Clamp between 0 and 1
            
            return {
                "upper": upper,
                "middle": middle,
                "lower": lower,
                "position": position
            }
            
        except Exception as e:
            logger.error(f"❌ Error calculating Bollinger Bands: {e}")
            return {
                "upper": pd.Series(index=prices.index, dtype=float),
                "middle": pd.Series(index=prices.index, dtype=float),
                "lower": pd.Series(index=prices.index, dtype=float),
                "position": pd.Series(index=prices.index, dtype=float)
            }
    
    def calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, 
                      period: int = 14) -> pd.Series:
        """Calculate Average True Range (ATR)"""
        try:
            if len(high) < period + 1:
                return pd.Series(index=high.index, dtype=float)
            
            # Calculate True Range
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            
            # True Range is the maximum of the three
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            # Calculate ATR (exponential moving average of True Range)
            atr = true_range.ewm(span=period).mean()
            
            return atr
            
        except Exception as e:
            logger.error(f"❌ Error calculating ATR: {e}")
            return pd.Series(index=high.index, dtype=float)
    
    def calculate_volume_sma_ratio(self, volume: pd.Series, prices: pd.Series, 
                                  volume_period: int = 20, price_period: int = 20) -> pd.Series:
        """Calculate Volume to Simple Moving Average Ratio"""
        try:
            if len(volume) < max(volume_period, price_period):
                return pd.Series(index=volume.index, dtype=float)
            
            # Calculate volume SMA
            volume_sma = volume.rolling(window=volume_period).mean()
            
            # Calculate price SMA
            price_sma = prices.rolling(window=price_period).mean()
            
            # Calculate ratio
            ratio = volume / volume_sma
            
            return ratio
            
        except Exception as e:
            logger.error(f"❌ Error calculating Volume SMA Ratio: {e}")
            return pd.Series(index=volume.index, dtype=float)
    
    def calculate_vwap(self, high: pd.Series, low: pd.Series, close: pd.Series, 
                       volume: pd.Series, period: int = 20) -> pd.Series:
        """Calculate Volume Weighted Average Price (VWAP)"""
        try:
            if len(high) < period:
                return pd.Series(index=high.index, dtype=float)
            
            # Calculate typical price
            typical_price = (high + low + close) / 3
            
            # Calculate VWAP
            vwap = (typical_price * volume).rolling(window=period).sum() / volume.rolling(window=period).sum()
            
            return vwap
            
        except Exception as e:
            logger.error(f"❌ Error calculating VWAP: {e}")
            return pd.Series(index=high.index, dtype=float)
    
    def calculate_ichimoku(self, high: pd.Series, low: pd.Series, close: pd.Series,
                           tenkan_period: int = 9, kijun_period: int = 26,
                           senkou_span_b_period: int = 52) -> Dict[str, pd.Series]:
        """Calculate Ichimoku Cloud components"""
        try:
            if len(high) < senkou_span_b_period + 26:
                return {
                    "tenkan_sen": pd.Series(index=high.index, dtype=float),
                    "kijun_sen": pd.Series(index=high.index, dtype=float),
                    "senkou_span_a": pd.Series(index=high.index, dtype=float),
                    "senkou_span_b": pd.Series(index=high.index, dtype=float),
                    "chikou_span": pd.Series(index=high.index, dtype=float)
                }
            
            # Tenkan-sen (Conversion Line)
            tenkan_sen = (high.rolling(window=tenkan_period).max() + 
                         low.rolling(window=tenkan_period).min()) / 2
            
            # Kijun-sen (Base Line)
            kijun_sen = (high.rolling(window=kijun_period).max() + 
                        low.rolling(window=kijun_period).min()) / 2
            
            # Senkou Span A (Leading Span A)
            senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)
            
            # Senkou Span B (Leading Span B)
            senkou_span_b = ((high.rolling(window=senkou_span_b_period).max() + 
                             low.rolling(window=senkou_span_b_period).min()) / 2).shift(26)
            
            # Chikou Span (Lagging Span)
            chikou_span = close.shift(-26)
            
            return {
                "tenkan_sen": tenkan_sen,
                "kijun_sen": kijun_sen,
                "senkou_span_a": senkou_span_a,
                "senkou_span_b": senkou_span_b,
                "chikou_span": chikou_span
            }
            
        except Exception as e:
            logger.error(f"❌ Error calculating Ichimoku: {e}")
            return {
                "tenkan_sen": pd.Series(index=high.index, dtype=float),
                "kijun_sen": pd.Series(index=high.index, dtype=float),
                "senkou_span_a": pd.Series(index=high.index, dtype=float),
                "senkou_span_b": pd.Series(index=high.index, dtype=float),
                "chikou_span": pd.Series(index=high.index, dtype=float)
            }
    
    def calculate_all_indicators(self, ohlcv_data: pd.DataFrame, 
                                indicators: List[str] = None) -> Dict[str, pd.Series]:
        """Calculate all requested indicators for OHLCV data"""
        try:
            if indicators is None:
                indicators = list(self.indicators.keys())
            
            results = {}
            
            for indicator in indicators:
                if indicator not in self.indicators:
                    logger.warning(f"⚠️ Unknown indicator: {indicator}")
                    continue
                
                try:
                    if indicator == "rsi":
                        results[indicator] = self.calculate_rsi(ohlcv_data['close'])
                    elif indicator == "macd":
                        macd_data = self.calculate_macd(ohlcv_data['close'])
                        results[f"{indicator}_line"] = macd_data['macd']
                        results[f"{indicator}_signal"] = macd_data['signal']
                        results[f"{indicator}_histogram"] = macd_data['histogram']
                    elif indicator == "ema":
                        results[indicator] = self.calculate_ema(ohlcv_data['close'])
                    elif indicator == "bollinger_bands":
                        bb_data = self.calculate_bollinger_bands(ohlcv_data['close'])
                        results[f"{indicator}_upper"] = bb_data['upper']
                        results[f"{indicator}_middle"] = bb_data['middle']
                        results[f"{indicator}_lower"] = bb_data['lower']
                        results[f"{indicator}_position"] = bb_data['position']
                    elif indicator == "atr":
                        results[indicator] = self.calculate_atr(
                            ohlcv_data['high'], ohlcv_data['low'], ohlcv_data['close']
                        )
                    elif indicator == "volume_sma_ratio":
                        results[indicator] = self.calculate_volume_sma_ratio(
                            ohlcv_data['volume'], ohlcv_data['close']
                        )
                    elif indicator == "vwap":
                        results[indicator] = self.calculate_vwap(
                            ohlcv_data['high'], ohlcv_data['low'], 
                            ohlcv_data['close'], ohlcv_data['volume']
                        )
                    elif indicator == "ichimoku":
                        ichimoku_data = self.calculate_ichimoku(
                            ohlcv_data['high'], ohlcv_data['low'], ohlcv_data['close']
                        )
                        for key, value in ichimoku_data.items():
                            results[f"{indicator}_{key}"] = value
                    
                    logger.debug(f"✅ Calculated {indicator}")
                    
                except Exception as e:
                    logger.error(f"❌ Failed to calculate {indicator}: {e}")
                    continue
            
            logger.info(f"✅ Successfully calculated {len(results)} indicator values")
            return results
            
        except Exception as e:
            logger.error(f"❌ Error calculating indicators: {e}")
            return {}
    
    def get_indicator_metadata(self) -> Dict[str, Dict[str, Any]]:
        """Get metadata for all indicators"""
        metadata = {}
        for name, config in self.indicators.items():
            metadata[name] = {
                "description": config.description,
                "parameters": config.parameters,
                "min_periods": config.min_periods,
                "max_periods": config.max_periods,
                "output_type": config.output_type,
                "tags": config.tags
            }
        return metadata

# Convenience functions
def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI for a price series"""
    engine = TechnicalIndicatorsEngine()
    return engine.calculate_rsi(prices, period)

def calculate_macd(prices: pd.Series, fast_period: int = 12, 
                   slow_period: int = 26, signal_period: int = 9) -> Dict[str, pd.Series]:
    """Calculate MACD for a price series"""
    engine = TechnicalIndicatorsEngine()
    return engine.calculate_macd(prices, fast_period, slow_period, signal_period)

def calculate_ema(prices: pd.Series, period: int = 20) -> pd.Series:
    """Calculate EMA for a price series"""
    engine = TechnicalIndicatorsEngine()
    return engine.calculate_ema(prices, period)

def calculate_bollinger_bands(prices: pd.Series, period: int = 20, 
                             std_dev: float = 2.0) -> Dict[str, pd.Series]:
    """Calculate Bollinger Bands for a price series"""
    engine = TechnicalIndicatorsEngine()
    return engine.calculate_bollinger_bands(prices, period, std_dev)

def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, 
                  period: int = 14) -> pd.Series:
    """Calculate ATR for OHLC data"""
    engine = TechnicalIndicatorsEngine()
    return engine.calculate_atr(high, low, close, period)
