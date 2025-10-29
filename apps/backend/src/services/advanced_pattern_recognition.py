"""
Advanced Pattern Recognition Service for AlphaPlus
Integrates technical analysis, ML models, and multi-timeframe analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import logging
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
import ta
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class AdvancedPatternRecognition:
    """Advanced pattern recognition using technical analysis and ML"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.pattern_models = {}
        self.technical_indicators = {}
        
        # Initialize ML models
        self.initialize_ml_models()
        
        # Pattern definitions
        self.pattern_definitions = {
            'bullish_engulfing': self.detect_bullish_engulfing,
            'bearish_engulfing': self.detect_bearish_engulfing,
            'hammer': self.detect_hammer,
            'shooting_star': self.detect_shooting_star,
            'doji': self.detect_doji,
            'morning_star': self.detect_morning_star,
            'evening_star': self.detect_evening_star,
            'double_bottom': self.detect_double_bottom,
            'double_top': self.detect_double_top,
            'head_shoulders': self.detect_head_shoulders,
            'inverse_head_shoulders': self.detect_inverse_head_shoulders,
            'triangle_ascending': self.detect_triangle_ascending,
            'triangle_descending': self.detect_triangle_descending,
            'flag_bullish': self.detect_bullish_flag,
            'flag_bearish': self.detect_bearish_flag,
            'wedge_rising': self.detect_rising_wedge,
            'wedge_falling': self.detect_falling_wedge
        }
    
    def initialize_ml_models(self):
        """Initialize ML models for pattern classification"""
        try:
            # Random Forest for pattern classification
            self.pattern_models['random_forest'] = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
            
            # Gradient Boosting for trend prediction
            self.pattern_models['gradient_boosting'] = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                random_state=42
            )
            
            logger.info("✅ ML models initialized")
            
        except Exception as e:
            logger.error(f"❌ Error initializing ML models: {e}")
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive technical indicators"""
        try:
            # Trend indicators
            df['sma_20'] = ta.trend.sma_indicator(df['close'], window=20)
            df['sma_50'] = ta.trend.sma_indicator(df['close'], window=50)
            df['ema_12'] = ta.trend.ema_indicator(df['close'], window=12)
            df['ema_26'] = ta.trend.ema_indicator(df['close'], window=26)
            df['macd'] = ta.trend.macd_diff(df['close'])
            df['macd_signal'] = ta.trend.macd_signal(df['close'])
            df['adx'] = ta.trend.adx(df['high'], df['low'], df['close'], window=14)
            
            # Momentum indicators
            df['rsi'] = ta.momentum.rsi(df['close'], window=14)
            df['stoch'] = ta.momentum.stoch(df['high'], df['low'], df['close'], window=14)
            df['stoch_signal'] = ta.momentum.stoch_signal(df['high'], df['low'], df['close'], window=14)
            df['williams_r'] = ta.momentum.williams_r(df['high'], df['low'], df['close'], window=14)
            df['cci'] = ta.trend.cci(df['high'], df['low'], df['close'], window=20)
            
            # Volatility indicators
            df['bb_upper'] = ta.volatility.bollinger_hband(df['close'], window=20)
            df['bb_lower'] = ta.volatility.bollinger_lband(df['close'], window=20)
            df['bb_middle'] = ta.volatility.bollinger_mavg(df['close'], window=20)
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
            df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)
            
            # Volume indicators
            df['volume_sma'] = ta.volume.volume_sma(df['close'], df['volume'], window=20)
            df['obv'] = ta.volume.on_balance_volume(df['close'], df['volume'])
            df['vwap'] = ta.volume.volume_weighted_average_price(df['high'], df['low'], df['close'], df['volume'])
            
            # Price action
            df['body_size'] = abs(df['close'] - df['open'])
            df['upper_shadow'] = df['high'] - np.maximum(df['open'], df['close'])
            df['lower_shadow'] = np.minimum(df['open'], df['close']) - df['low']
            df['body_ratio'] = df['body_size'] / (df['high'] - df['low'])
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Error calculating technical indicators: {e}")
            return df
    
    def detect_candlestick_patterns(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect candlestick patterns"""
        patterns = []
        
        try:
            for i in range(2, len(df)):
                # Get recent candles
                current = df.iloc[i]
                prev = df.iloc[i-1]
                prev2 = df.iloc[i-2]
                
                # Check each pattern
                for pattern_name, pattern_func in self.pattern_definitions.items():
                    if pattern_func(df, i):
                        confidence = self.calculate_pattern_confidence(df, i, pattern_name)
                        patterns.append({
                            'pattern_type': pattern_name,
                            'confidence': confidence,
                            'timestamp': current.name,
                            'price': float(current['close']),
                            'volume': float(current['volume'])
                        })
            
            return patterns
            
        except Exception as e:
            logger.error(f"❌ Error detecting candlestick patterns: {e}")
            return patterns
    
    def detect_bullish_engulfing(self, df: pd.DataFrame, i: int) -> bool:
        """Detect bullish engulfing pattern"""
        try:
            current = df.iloc[i]
            prev = df.iloc[i-1]
            
            return (prev['close'] < prev['open'] and  # Previous bearish
                    current['close'] > current['open'] and  # Current bullish
                    current['open'] < prev['close'] and  # Current opens below prev close
                    current['close'] > prev['open'])  # Current closes above prev open
        except:
            return False
    
    def detect_bearish_engulfing(self, df: pd.DataFrame, i: int) -> bool:
        """Detect bearish engulfing pattern"""
        try:
            current = df.iloc[i]
            prev = df.iloc[i-1]
            
            return (prev['close'] > prev['open'] and  # Previous bullish
                    current['close'] < current['open'] and  # Current bearish
                    current['open'] > prev['close'] and  # Current opens above prev close
                    current['close'] < prev['open'])  # Current closes below prev open
        except:
            return False
    
    def detect_hammer(self, df: pd.DataFrame, i: int) -> bool:
        """Detect hammer pattern"""
        try:
            current = df.iloc[i]
            body_size = abs(current['close'] - current['open'])
            lower_shadow = min(current['open'], current['close']) - current['low']
            upper_shadow = current['high'] - max(current['open'], current['close'])
            
            return (lower_shadow > 2 * body_size and  # Long lower shadow
                    upper_shadow < body_size)  # Short upper shadow
        except:
            return False
    
    def detect_shooting_star(self, df: pd.DataFrame, i: int) -> bool:
        """Detect shooting star pattern"""
        try:
            current = df.iloc[i]
            body_size = abs(current['close'] - current['open'])
            lower_shadow = min(current['open'], current['close']) - current['low']
            upper_shadow = current['high'] - max(current['open'], current['close'])
            
            return (upper_shadow > 2 * body_size and  # Long upper shadow
                    lower_shadow < body_size)  # Short lower shadow
        except:
            return False
    
    def detect_doji(self, df: pd.DataFrame, i: int) -> bool:
        """Detect doji pattern"""
        try:
            current = df.iloc[i]
            body_size = abs(current['close'] - current['open'])
            total_range = current['high'] - current['low']
            
            return body_size < 0.1 * total_range  # Very small body
        except:
            return False
    
    def detect_morning_star(self, df: pd.DataFrame, i: int) -> bool:
        """Detect morning star pattern"""
        try:
            if i < 2:
                return False
                
            prev2 = df.iloc[i-2]  # First bearish candle
            prev = df.iloc[i-1]   # Small body candle
            current = df.iloc[i]  # Bullish candle
            
            return (prev2['close'] < prev2['open'] and  # First bearish
                    abs(prev['close'] - prev['open']) < 0.5 * abs(prev2['close'] - prev2['open']) and  # Small body
                    current['close'] > current['open'] and  # Current bullish
                    current['close'] > (prev2['open'] + prev2['close']) / 2)  # Closes above midpoint
        except:
            return False
    
    def detect_evening_star(self, df: pd.DataFrame, i: int) -> bool:
        """Detect evening star pattern"""
        try:
            if i < 2:
                return False
                
            prev2 = df.iloc[i-2]  # First bullish candle
            prev = df.iloc[i-1]   # Small body candle
            current = df.iloc[i]  # Bearish candle
            
            return (prev2['close'] > prev2['open'] and  # First bullish
                    abs(prev['close'] - prev['open']) < 0.5 * abs(prev2['close'] - prev2['open']) and  # Small body
                    current['close'] < current['open'] and  # Current bearish
                    current['close'] < (prev2['open'] + prev2['close']) / 2)  # Closes below midpoint
        except:
            return False
    
    def detect_double_bottom(self, df: pd.DataFrame, i: int) -> bool:
        """Detect double bottom pattern"""
        try:
            if i < 20:
                return False
                
            # Look for two similar lows with a peak in between
            window = df.iloc[i-20:i+1]
            lows = window.nsmallest(3, 'low')
            
            if len(lows) < 3:
                return False
            
            # Check if two lows are similar and there's a peak between them
            low1, low2 = lows.iloc[0], lows.iloc[1]
            if abs(low1['low'] - low2['low']) / low1['low'] < 0.02:  # Within 2%
                # Check if there's a peak between them
                between_idx = window[(window.index > low1.name) & (window.index < low2.name)]
                if len(between_idx) > 0:
                    peak = between_idx.nlargest(1, 'high').iloc[0]
                    if peak['high'] > low1['low'] * 1.05:  # Peak is 5% higher than lows
                        return True
            
            return False
        except:
            return False
    
    def detect_double_top(self, df: pd.DataFrame, i: int) -> bool:
        """Detect double top pattern"""
        try:
            if i < 20:
                return False
                
            # Look for two similar highs with a trough in between
            window = df.iloc[i-20:i+1]
            highs = window.nlargest(3, 'high')
            
            if len(highs) < 3:
                return False
            
            # Check if two highs are similar and there's a trough between them
            high1, high2 = highs.iloc[0], highs.iloc[1]
            if abs(high1['high'] - high2['high']) / high1['high'] < 0.02:  # Within 2%
                # Check if there's a trough between them
                between_idx = window[(window.index > high1.name) & (window.index < high2.name)]
                if len(between_idx) > 0:
                    trough = between_idx.nsmallest(1, 'low').iloc[0]
                    if trough['low'] < high1['high'] * 0.95:  # Trough is 5% lower than highs
                        return True
            
            return False
        except:
            return False
    
    def detect_head_shoulders(self, df: pd.DataFrame, i: int) -> bool:
        """Detect head and shoulders pattern using standard formula"""
        try:
            if i < 30:
                return False
                
            # Standard Head & Shoulders detection: 3 peaks, middle highest
            # Look for three peaks with middle peak highest
            window = df.iloc[i-30:i+1]
            
            # Find local peaks
            peaks = []
            for j in range(2, len(window) - 2):
                if (window.iloc[j]['high'] > window.iloc[j-1]['high'] and 
                    window.iloc[j]['high'] > window.iloc[j+1]['high'] and
                    window.iloc[j]['high'] > window.iloc[j-2]['high'] and 
                    window.iloc[j]['high'] > window.iloc[j+2]['high']):
                    peaks.append({
                        'index': j,
                        'high': window.iloc[j]['high'],
                        'volume': window.iloc[j]['volume']
                    })
            
            if len(peaks) < 3:
                return False
            
            # Sort peaks by index to get chronological order
            peaks_sorted = sorted(peaks, key=lambda x: x['index'])
            
            # Check Head & Shoulders pattern: Left shoulder, Head (higher), Right shoulder
            left_shoulder = peaks_sorted[0]['high']
            head = peaks_sorted[1]['high']
            right_shoulder = peaks_sorted[2]['high']
            
            # Standard validation: Head > Left Shoulder and Head > Right Shoulder
            # Volume decreases from left to right (distribution)
            volume_decreasing = (peaks_sorted[0]['volume'] > peaks_sorted[1]['volume'] > peaks_sorted[2]['volume'])
            
            return (head > left_shoulder and head > right_shoulder and volume_decreasing)
            
        except Exception as e:
            logger.error(f"Error detecting head and shoulders: {e}")
            return False
    
    def detect_triangle_ascending(self, df: pd.DataFrame, i: int) -> bool:
        """Detect ascending triangle pattern using standard formula"""
        try:
            if i < 20:
                return False
            
            # Standard Triangle detection: Two converging trend lines
            # Support ascending, resistance horizontal/descending
            window = df.iloc[i-20:i+1]
            
            # Find support and resistance levels
            support_levels = []
            resistance_levels = []
            
            for j in range(2, len(window) - 2):
                # Find support (local lows)
                if (window.iloc[j]['low'] < window.iloc[j-1]['low'] and 
                    window.iloc[j]['low'] < window.iloc[j+1]['low']):
                    support_levels.append(window.iloc[j]['low'])
                
                # Find resistance (local highs)
                if (window.iloc[j]['high'] > window.iloc[j-1]['high'] and 
                    window.iloc[j]['high'] > window.iloc[j+1]['high']):
                    resistance_levels.append(window.iloc[j]['high'])
            
            if len(support_levels) < 2 or len(resistance_levels) < 2:
                return False
            
            # Check for converging trend lines
            # Support should be ascending (higher lows)
            support_trend = np.polyfit(range(len(support_levels)), support_levels, 1)[0]
            
            # Resistance should be horizontal or slightly descending
            resistance_trend = np.polyfit(range(len(resistance_levels)), resistance_levels, 1)[0]
            
            # Standard validation: Support ascending, resistance horizontal/descending
            return support_trend > 0 and resistance_trend <= 0
            
        except Exception as e:
            logger.error(f"Error detecting ascending triangle: {e}")
            return False
    
    def detect_triangle_descending(self, df: pd.DataFrame, i: int) -> bool:
        """Detect descending triangle pattern using standard formula"""
        try:
            if i < 20:
                return False
            
            # Standard Triangle detection: Two converging trend lines
            # Support horizontal/ascending, resistance descending
            window = df.iloc[i-20:i+1]
            
            # Find support and resistance levels
            support_levels = []
            resistance_levels = []
            
            for j in range(2, len(window) - 2):
                # Find support (local lows)
                if (window.iloc[j]['low'] < window.iloc[j-1]['low'] and 
                    window.iloc[j]['low'] < window.iloc[j+1]['low']):
                    support_levels.append(window.iloc[j]['low'])
                
                # Find resistance (local highs)
                if (window.iloc[j]['high'] > window.iloc[j-1]['high'] and 
                    window.iloc[j]['high'] > window.iloc[j+1]['high']):
                    resistance_levels.append(window.iloc[j]['high'])
            
            if len(support_levels) < 2 or len(resistance_levels) < 2:
                return False
            
            # Check for converging trend lines
            # Support should be horizontal or slightly ascending
            support_trend = np.polyfit(range(len(support_levels)), support_levels, 1)[0]
            
            # Resistance should be descending (lower highs)
            resistance_trend = np.polyfit(range(len(resistance_levels)), resistance_levels, 1)[0]
            
            # Standard validation: Support horizontal/ascending, resistance descending
            return support_trend >= 0 and resistance_trend < 0
            
        except Exception as e:
            logger.error(f"Error detecting descending triangle: {e}")
            return False
    
    def detect_bullish_flag(self, df: pd.DataFrame, i: int) -> bool:
        """Detect bullish flag pattern using standard formula"""
        try:
            if i < 15:
                return False
            
            # Standard Flag detection: Parallel channels post-impulse move
            # Bullish flag: Impulse up, then parallel channel (flag)
            window = df.iloc[i-15:i+1]
            
            # Check for impulse move (first 5-7 bars)
            impulse_start = window.iloc[0]['low']
            impulse_end = window.iloc[5]['high']
            impulse_move = impulse_end - impulse_start
            
            # Impulse should be significant (>2% of price)
            if impulse_move < (impulse_start * 0.02):
                return False
            
            # Check for parallel channel (flag) in remaining bars
            flag_bars = window.iloc[6:]
            
            if len(flag_bars) < 5:
                return False
            
            # Find flag support and resistance
            flag_support = flag_bars['low'].min()
            flag_resistance = flag_bars['high'].max()
            
            # Flag should be parallel (similar highs and lows)
            flag_range = flag_resistance - flag_support
            flag_height_ratio = flag_range / impulse_move
            
            # Standard validation: Flag height < 50% of impulse move
            return flag_height_ratio < 0.5
            
        except Exception as e:
            logger.error(f"Error detecting bullish flag: {e}")
            return False
    
    def detect_bearish_flag(self, df: pd.DataFrame, i: int) -> bool:
        """Detect bearish flag pattern using standard formula"""
        try:
            if i < 15:
                return False
            
            # Standard Flag detection: Parallel channels post-impulse move
            # Bearish flag: Impulse down, then parallel channel (flag)
            window = df.iloc[i-15:i+1]
            
            # Check for impulse move (first 5-7 bars)
            impulse_start = window.iloc[0]['high']
            impulse_end = window.iloc[5]['low']
            impulse_move = impulse_start - impulse_end
            
            # Impulse should be significant (>2% of price)
            if impulse_move < (impulse_start * 0.02):
                return False
            
            # Check for parallel channel (flag) in remaining bars
            flag_bars = window.iloc[6:]
            
            if len(flag_bars) < 5:
                return False
            
            # Find flag support and resistance
            flag_support = flag_bars['low'].min()
            flag_resistance = flag_bars['high'].max()
            
            # Flag should be parallel (similar highs and lows)
            flag_range = flag_resistance - flag_support
            flag_height_ratio = flag_range / impulse_move
            
            # Standard validation: Flag height < 50% of impulse move
            return flag_height_ratio < 0.5
            
        except Exception as e:
            logger.error(f"Error detecting bearish flag: {e}")
            return False
    
    def validate_pattern_completion(self, df: pd.DataFrame, pattern_type: str, pattern_index: int) -> bool:
        """Validate pattern completion using standard formula: Breakout beyond line with volume > 1.5× average"""
        try:
            if pattern_index >= len(df) - 5:
                return False
            
            # Get pattern completion window
            completion_window = df.iloc[pattern_index:pattern_index+5]
            
            # Calculate volume confirmation: Volume > 1.5× average
            avg_volume = df['volume'].rolling(window=20).mean().iloc[pattern_index]
            completion_volume = completion_window['volume'].mean()
            volume_confirmation = completion_volume > (avg_volume * 1.5)
            
            if not volume_confirmation:
                return False
            
            # Pattern-specific completion validation
            if pattern_type in ['head_shoulders', 'inverse_head_shoulders']:
                # Breakout beyond neckline
                pattern_window = df.iloc[pattern_index-30:pattern_index+1]
                neckline = pattern_window['low'].min() if 'head_shoulders' in pattern_type else pattern_window['high'].max()
                
                # Check for breakout beyond neckline
                breakout_threshold = neckline * 0.02  # 2% breakout
                if 'head_shoulders' in pattern_type:
                    return completion_window['close'].min() < (neckline - breakout_threshold)
                else:
                    return completion_window['close'].max() > (neckline + breakout_threshold)
            
            elif pattern_type in ['triangle_ascending', 'triangle_descending']:
                # Breakout beyond triangle lines
                pattern_window = df.iloc[pattern_index-20:pattern_index+1]
                
                if 'ascending' in pattern_type:
                    # Breakout above resistance
                    resistance = pattern_window['high'].max()
                    return completion_window['close'].max() > (resistance * 1.02)
                else:
                    # Breakout below support
                    support = pattern_window['low'].min()
                    return completion_window['close'].min() < (support * 0.98)
            
            elif pattern_type in ['flag_bullish', 'flag_bearish']:
                # Breakout in direction of flag
                pattern_window = df.iloc[pattern_index-15:pattern_index+1]
                
                if 'bullish' in pattern_type:
                    # Breakout above flag resistance
                    flag_resistance = pattern_window['high'].max()
                    return completion_window['close'].max() > (flag_resistance * 1.02)
                else:
                    # Breakout below flag support
                    flag_support = pattern_window['low'].min()
                    return completion_window['close'].min() < (flag_support * 0.98)
            
            return False
            
        except Exception as e:
            logger.error(f"Error validating pattern completion: {e}")
            return False
            
            return False
        except:
            return False
    
    def detect_inverse_head_shoulders(self, df: pd.DataFrame, i: int) -> bool:
        """Detect inverse head and shoulders pattern (simplified)"""
        try:
            if i < 30:
                return False
                
            # Simplified detection - look for three troughs with middle trough lowest
            window = df.iloc[i-30:i+1]
            troughs = window.nsmallest(5, 'low')
            
            if len(troughs) < 3:
                return False
            
            # Check if middle trough is lowest
            troughs_sorted = troughs.sort_index()
            if len(troughs_sorted) >= 3:
                left_shoulder = troughs_sorted.iloc[0]['low']
                head = troughs_sorted.iloc[1]['low']
                right_shoulder = troughs_sorted.iloc[2]['low']
                
                return head < left_shoulder and head < right_shoulder
            
            return False
        except:
            return False
    
    def detect_triangle_ascending(self, df: pd.DataFrame, i: int) -> bool:
        """Detect ascending triangle pattern"""
        try:
            if i < 20:
                return False
                
            window = df.iloc[i-20:i+1]
            
            # Calculate trend lines
            highs = window.nlargest(5, 'high')
            lows = window.nsmallest(5, 'low')
            
            if len(highs) < 3 or len(lows) < 3:
                return False
            
            # Check if highs are flat and lows are rising
            high_slope = np.polyfit(range(len(highs)), highs['high'], 1)[0]
            low_slope = np.polyfit(range(len(lows)), lows['low'], 1)[0]
            
            return abs(high_slope) < 0.001 and low_slope > 0.001  # Flat highs, rising lows
            
        except:
            return False
    
    def detect_triangle_descending(self, df: pd.DataFrame, i: int) -> bool:
        """Detect descending triangle pattern"""
        try:
            if i < 20:
                return False
                
            window = df.iloc[i-20:i+1]
            
            # Calculate trend lines
            highs = window.nlargest(5, 'high')
            lows = window.nsmallest(5, 'low')
            
            if len(highs) < 3 or len(lows) < 3:
                return False
            
            # Check if highs are falling and lows are flat
            high_slope = np.polyfit(range(len(highs)), highs['high'], 1)[0]
            low_slope = np.polyfit(range(len(lows)), lows['low'], 1)[0]
            
            return high_slope < -0.001 and abs(low_slope) < 0.001  # Falling highs, flat lows
            
        except:
            return False
    
    def detect_bullish_flag(self, df: pd.DataFrame, i: int) -> bool:
        """Detect bullish flag pattern"""
        try:
            if i < 15:
                return False
                
            # Look for strong upward move followed by consolidation
            window = df.iloc[i-15:i+1]
            
            # Check for strong upward move in first half
            first_half = window.iloc[:8]
            second_half = window.iloc[8:]
            
            if len(first_half) < 5 or len(second_half) < 5:
                return False
            
            first_trend = (first_half.iloc[-1]['close'] - first_half.iloc[0]['close']) / first_half.iloc[0]['close']
            second_trend = (second_half.iloc[-1]['close'] - second_half.iloc[0]['close']) / second_half.iloc[0]['close']
            
            return first_trend > 0.05 and abs(second_trend) < 0.02  # Strong up, then consolidation
            
        except:
            return False
    
    def detect_bearish_flag(self, df: pd.DataFrame, i: int) -> bool:
        """Detect bearish flag pattern"""
        try:
            if i < 15:
                return False
                
            # Look for strong downward move followed by consolidation
            window = df.iloc[i-15:i+1]
            
            # Check for strong downward move in first half
            first_half = window.iloc[:8]
            second_half = window.iloc[8:]
            
            if len(first_half) < 5 or len(second_half) < 5:
                return False
            
            first_trend = (first_half.iloc[-1]['close'] - first_half.iloc[0]['close']) / first_half.iloc[0]['close']
            second_trend = (second_half.iloc[-1]['close'] - second_half.iloc[0]['close']) / second_half.iloc[0]['close']
            
            return first_trend < -0.05 and abs(second_trend) < 0.02  # Strong down, then consolidation
            
        except:
            return False
    
    def detect_rising_wedge(self, df: pd.DataFrame, i: int) -> bool:
        """Detect rising wedge pattern"""
        try:
            if i < 20:
                return False
                
            window = df.iloc[i-20:i+1]
            
            # Calculate trend lines
            highs = window.nlargest(5, 'high')
            lows = window.nsmallest(5, 'low')
            
            if len(highs) < 3 or len(lows) < 3:
                return False
            
            # Check if both highs and lows are rising, but highs are rising faster
            high_slope = np.polyfit(range(len(highs)), highs['high'], 1)[0]
            low_slope = np.polyfit(range(len(lows)), lows['low'], 1)[0]
            
            return high_slope > 0 and low_slope > 0 and high_slope > low_slope
            
        except:
            return False
    
    def detect_falling_wedge(self, df: pd.DataFrame, i: int) -> bool:
        """Detect falling wedge pattern"""
        try:
            if i < 20:
                return False
                
            window = df.iloc[i-20:i+1]
            
            # Calculate trend lines
            highs = window.nlargest(5, 'high')
            lows = window.nsmallest(5, 'low')
            
            if len(highs) < 3 or len(lows) < 3:
                return False
            
            # Check if both highs and lows are falling, but lows are falling faster
            high_slope = np.polyfit(range(len(highs)), highs['high'], 1)[0]
            low_slope = np.polyfit(range(len(lows)), lows['low'], 1)[0]
            
            return high_slope < 0 and low_slope < 0 and low_slope < high_slope
            
        except:
            return False
    
    def calculate_pattern_confidence(self, df: pd.DataFrame, i: int, pattern_type: str) -> float:
        """Calculate confidence score for a detected pattern"""
        try:
            current = df.iloc[i]
            
            # Base confidence
            base_confidence = 0.7
            
            # Volume confirmation
            volume_ma = df['volume'].rolling(window=20).mean().iloc[i]
            volume_ratio = current['volume'] / volume_ma if volume_ma > 0 else 1
            volume_boost = min(0.2, (volume_ratio - 1) * 0.1) if volume_ratio > 1 else 0
            
            # RSI confirmation
            rsi = current.get('rsi', 50)
            rsi_boost = 0
            if 'bullish' in pattern_type and rsi < 70:
                rsi_boost = 0.1
            elif 'bearish' in pattern_type and rsi > 30:
                rsi_boost = 0.1
            
            # MACD confirmation
            macd = current.get('macd', 0)
            macd_signal = current.get('macd_signal', 0)
            macd_boost = 0
            if 'bullish' in pattern_type and macd > macd_signal:
                macd_boost = 0.1
            elif 'bearish' in pattern_type and macd < macd_signal:
                macd_boost = 0.1
            
            # Calculate final confidence
            confidence = base_confidence + volume_boost + rsi_boost + macd_boost
            return min(0.95, max(0.5, confidence))
            
        except Exception as e:
            logger.error(f"❌ Error calculating pattern confidence: {e}")
            return 0.7
    
    def analyze_market_structure(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze overall market structure"""
        try:
            if len(df) < 50:
                return {}
            
            # Calculate key levels
            recent_high = df['high'].tail(20).max()
            recent_low = df['low'].tail(20).min()
            current_price = df['close'].iloc[-1]
            
            # Support and resistance levels
            support_levels = self.find_support_levels(df)
            resistance_levels = self.find_resistance_levels(df)
            
            # Trend analysis
            sma_20 = df['sma_20'].iloc[-1]
            sma_50 = df['sma_50'].iloc[-1]
            
            trend = "bullish" if sma_20 > sma_50 else "bearish"
            trend_strength = abs(sma_20 - sma_50) / sma_50
            
            # Volatility analysis
            atr = df['atr'].iloc[-1]
            avg_atr = df['atr'].tail(20).mean()
            volatility_ratio = atr / avg_atr if avg_atr > 0 else 1
            
            return {
                'trend': trend,
                'trend_strength': trend_strength,
                'current_price': float(current_price),
                'recent_high': float(recent_high),
                'recent_low': float(recent_low),
                'support_levels': support_levels,
                'resistance_levels': resistance_levels,
                'volatility_ratio': volatility_ratio,
                'sma_20': float(sma_20),
                'sma_50': float(sma_50)
            }
            
        except Exception as e:
            logger.error(f"❌ Error analyzing market structure: {e}")
            return {}
    
    def find_support_levels(self, df: pd.DataFrame) -> List[float]:
        """Find support levels using pivot points"""
        try:
            support_levels = []
            
            for i in range(1, len(df) - 1):
                if (df['low'].iloc[i] < df['low'].iloc[i-1] and 
                    df['low'].iloc[i] < df['low'].iloc[i+1]):
                    support_levels.append(float(df['low'].iloc[i]))
            
            # Return unique levels within 1% of each other
            unique_levels = []
            for level in sorted(support_levels):
                if not any(abs(level - existing) / existing < 0.01 for existing in unique_levels):
                    unique_levels.append(level)
            
            return unique_levels[-3:]  # Return last 3 support levels
            
        except Exception as e:
            logger.error(f"❌ Error finding support levels: {e}")
            return []
    
    def find_resistance_levels(self, df: pd.DataFrame) -> List[float]:
        """Find resistance levels using pivot points"""
        try:
            resistance_levels = []
            
            for i in range(1, len(df) - 1):
                if (df['high'].iloc[i] > df['high'].iloc[i-1] and 
                    df['high'].iloc[i] > df['high'].iloc[i+1]):
                    resistance_levels.append(float(df['high'].iloc[i]))
            
            # Return unique levels within 1% of each other
            unique_levels = []
            for level in sorted(resistance_levels):
                if not any(abs(level - existing) / existing < 0.01 for existing in unique_levels):
                    unique_levels.append(level)
            
            return unique_levels[-3:]  # Return last 3 resistance levels
            
        except Exception as e:
            logger.error(f"❌ Error finding resistance levels: {e}")
            return []
    
    def analyze_patterns(self, market_data: List[Dict]) -> Dict[str, Any]:
        """Main method to analyze patterns in market data"""
        try:
            if len(market_data) < 50:
                return {}
            
            # Convert to DataFrame
            df = pd.DataFrame(market_data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            
            # Calculate technical indicators
            df = self.calculate_technical_indicators(df)
            
            # Detect patterns
            patterns = self.detect_candlestick_patterns(df)
            
            # Analyze market structure
            market_structure = self.analyze_market_structure(df)
            
            return {
                'patterns': patterns,
                'market_structure': market_structure,
                'technical_indicators': {
                    'rsi': float(df['rsi'].iloc[-1]) if 'rsi' in df.columns else 50,
                    'macd': float(df['macd'].iloc[-1]) if 'macd' in df.columns else 0,
                    'bb_position': float((df['close'].iloc[-1] - df['bb_lower'].iloc[-1]) / 
                                       (df['bb_upper'].iloc[-1] - df['bb_lower'].iloc[-1])) if 'bb_upper' in df.columns else 0.5,
                    'volume_ratio': float(df['volume'].iloc[-1] / df['volume_sma'].iloc[-1]) if 'volume_sma' in df.columns else 1
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Error analyzing patterns: {e}")
            return {}

# Global instance
pattern_recognition = AdvancedPatternRecognition()
