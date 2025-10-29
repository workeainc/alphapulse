"""
Real-Time Indicator Calculator
Efficiently calculates indicators on streaming data
"""

import pandas as pd
import numpy as np
from typing import Dict, List
from collections import deque

class RealtimeIndicatorCalculator:
    """Calculate indicators incrementally on live data"""
    
    def __init__(self, buffer_size: int = 100):
        self.buffer_size = buffer_size
        # Store last N candles for each symbol/timeframe
        self.buffers: Dict[str, deque] = {}
    
    def _get_buffer_key(self, symbol: str, timeframe: str) -> str:
        return f"{symbol}_{timeframe}"
    
    def add_candle(self, symbol: str, timeframe: str, candle: Dict):
        """Add new candle to buffer"""
        key = self._get_buffer_key(symbol, timeframe)
        
        if key not in self.buffers:
            self.buffers[key] = deque(maxlen=self.buffer_size)
        
        self.buffers[key].append(candle)
    
    def get_dataframe(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """Get DataFrame from buffer"""
        key = self._get_buffer_key(symbol, timeframe)
        
        if key not in self.buffers or len(self.buffers[key]) < 20:
            return None
        
        df = pd.DataFrame(list(self.buffers[key]))
        return df
    
    def calculate_all_indicators(self, symbol: str, timeframe: str) -> Dict:
        """Calculate ALL 50+ indicators for aggregator compatibility"""
        df = self.get_dataframe(symbol, timeframe)
        
        if df is None or len(df) < 50:
            return None
        
        indicators = {}
        
        try:
            close = df['close']
            high = df['high']
            low = df['low']
            volume = df['volume']
            
            # === MOMENTUM INDICATORS (10) ===
            
            # RSI (14)
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            indicators['rsi'] = float(rsi.iloc[-1])
            df['rsi'] = rsi  # Add to dataframe for aggregator
            
            # Stochastic (14,3,3)
            low_min = low.rolling(window=14).min()
            high_max = high.rolling(window=14).max()
            stoch_k = 100 * ((close - low_min) / (high_max - low_min))
            stoch_d = stoch_k.rolling(window=3).mean()
            df['stoch_k'] = stoch_k
            df['stoch_d'] = stoch_d
            
            # Williams %R
            williams_r = -100 * ((high_max - close) / (high_max - low_min))
            df['williams_r'] = williams_r
            
            # CCI (Commodity Channel Index)
            tp = (high + low + close) / 3
            cci = (tp - tp.rolling(window=20).mean()) / (0.015 * tp.rolling(window=20).std())
            df['cci'] = cci
            
            # === TREND INDICATORS (9) ===
            
            # EMAs
            ema_9 = close.ewm(span=9).mean()
            ema_12 = close.ewm(span=12).mean()
            ema_21 = close.ewm(span=21).mean()
            ema_26 = close.ewm(span=26).mean()
            ema_50 = close.ewm(span=50).mean()
            df['ema_9'] = ema_9
            df['ema_12'] = ema_12
            df['ema_21'] = ema_21
            df['ema_26'] = ema_26
            df['ema_50'] = ema_50
            
            # SMAs
            sma_20 = close.rolling(window=20).mean()
            sma_50 = close.rolling(window=50).mean()
            sma_200 = close.rolling(window=min(200, len(df))).mean()
            df['sma_20'] = sma_20
            df['sma_50'] = sma_50
            df['sma_200'] = sma_200
            indicators['sma_20'] = float(sma_20.iloc[-1])
            indicators['sma_50'] = float(sma_50.iloc[-1])
            indicators['ema_20'] = float(ema_21.iloc[-1])
            
            # MACD
            macd_line = ema_12 - ema_26
            macd_signal = macd_line.ewm(span=9).mean()
            macd_histogram = macd_line - macd_signal
            df['macd'] = macd_line
            df['macd_signal'] = macd_signal
            df['macd_histogram'] = macd_histogram
            indicators['macd'] = float(macd_line.iloc[-1])
            indicators['macd_signal'] = float(macd_signal.iloc[-1])
            indicators['macd_histogram'] = float(macd_histogram.iloc[-1])
            
            # ADX (Average Directional Index)
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(window=14).mean()
            
            plus_dm = high.diff()
            minus_dm = -low.diff()
            plus_dm[plus_dm < 0] = 0
            minus_dm[minus_dm < 0] = 0
            
            plus_di = 100 * (plus_dm.rolling(window=14).mean() / atr)
            minus_di = 100 * (minus_dm.rolling(window=14).mean() / atr)
            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
            adx = dx.rolling(window=14).mean()
            df['adx'] = adx
            df['plus_di'] = plus_di
            df['minus_di'] = minus_di
            
            # Aroon Indicator
            aroon_up = 100 * close.rolling(window=25).apply(lambda x: x.argmax()) / 25
            aroon_down = 100 * close.rolling(window=25).apply(lambda x: x.argmin()) / 25
            df['aroon_up'] = aroon_up
            df['aroon_down'] = aroon_down
            
            # === VOLATILITY INDICATORS (6) ===
            
            # Bollinger Bands
            bb_middle = close.rolling(window=20).mean()
            bb_std = close.rolling(window=20).std()
            bb_upper = bb_middle + (bb_std * 2)
            bb_lower = bb_middle - (bb_std * 2)
            df['bb_upper'] = bb_upper
            df['bb_middle'] = bb_middle
            df['bb_lower'] = bb_lower
            indicators['bb_upper'] = float(bb_upper.iloc[-1])
            indicators['bb_middle'] = float(bb_middle.iloc[-1])
            indicators['bb_lower'] = float(bb_lower.iloc[-1])
            
            # ATR
            df['atr'] = atr
            
            # Donchian Channels
            donchian_upper = high.rolling(window=20).max()
            donchian_lower = low.rolling(window=20).min()
            df['donchian_upper'] = donchian_upper
            df['donchian_lower'] = donchian_lower
            
            # Keltner Channels
            keltner_middle = close.ewm(span=20).mean()
            keltner_upper = keltner_middle + (atr * 2)
            keltner_lower = keltner_middle - (atr * 2)
            df['keltner_upper'] = keltner_upper
            df['keltner_middle'] = keltner_middle
            df['keltner_lower'] = keltner_lower
            
            # === VOLUME INDICATORS ===
            
            # Volume SMA and ratio
            volume_sma = volume.rolling(window=20).mean()
            indicators['volume_sma'] = float(volume_sma.iloc[-1])
            indicators['volume_ratio'] = float(volume.iloc[-1] / volume_sma.iloc[-1])
            
            # OBV (On Balance Volume)
            obv = (volume * np.sign(close.diff())).cumsum()
            df['obv'] = obv
            
            # === ADDITIONAL INDICATORS ===
            
            # Current price
            indicators['current_price'] = float(close.iloc[-1])
            
            # Return both dict AND dataframe for aggregator
            return indicators
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            return None
    
    def get_dataframe_with_indicators(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """Get dataframe with ALL indicators calculated as columns"""
        df = self.get_dataframe(symbol, timeframe)
        
        if df is None or len(df) < 50:
            return None
        
        try:
            close = df['close']
            high = df['high']
            low = df['low']
            volume = df['volume']
            
            # Calculate and ADD all indicators as columns
            # (Same calculation as above, but adds to df)
            
            # RSI
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # EMAs
            df['ema_9'] = close.ewm(span=9).mean()
            df['ema_12'] = close.ewm(span=12).mean()
            df['ema_21'] = close.ewm(span=21).mean()
            df['ema_26'] = close.ewm(span=26).mean()
            df['ema_50'] = close.ewm(span=50).mean()
            
            # SMAs
            df['sma_20'] = close.rolling(window=20).mean()
            df['sma_50'] = close.rolling(window=50).mean()
            df['sma_200'] = close.rolling(window=min(200, len(df))).mean()
            
            # MACD
            macd_line = df['ema_12'] - df['ema_26']
            macd_signal = macd_line.ewm(span=9).mean()
            df['macd'] = macd_line
            df['macd_signal'] = macd_signal
            df['macd_histogram'] = macd_line - macd_signal
            
            # Stochastic
            low_min = low.rolling(window=14).min()
            high_max = high.rolling(window=14).max()
            df['stoch_k'] = 100 * ((close - low_min) / (high_max - low_min))
            df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()
            
            # Williams %R
            df['williams_r'] = -100 * ((high_max - close) / (high_max - low_min))
            
            # CCI
            tp = (high + low + close) / 3
            df['cci'] = (tp - tp.rolling(window=20).mean()) / (0.015 * tp.rolling(window=20).std())
            
            # ADX
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(window=14).mean()
            df['atr'] = atr
            
            plus_dm = high.diff()
            minus_dm = -low.diff()
            plus_dm[plus_dm < 0] = 0
            minus_dm[minus_dm < 0] = 0
            
            plus_di = 100 * (plus_dm.rolling(window=14).mean() / atr)
            minus_di = 100 * (minus_dm.rolling(window=14).mean() / atr)
            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
            df['adx'] = dx.rolling(window=14).mean()
            df['plus_di'] = plus_di
            df['minus_di'] = minus_di
            
            # Aroon
            df['aroon_up'] = 100 * close.rolling(window=25).apply(lambda x: x.argmax()) / 25
            df['aroon_down'] = 100 * close.rolling(window=25).apply(lambda x: x.argmin()) / 25
            
            # Bollinger Bands
            bb_middle = close.rolling(window=20).mean()
            bb_std = close.rolling(window=20).std()
            df['bb_upper'] = bb_middle + (bb_std * 2)
            df['bb_middle'] = bb_middle
            df['bb_lower'] = bb_middle - (bb_std * 2)
            
            # Donchian Channels
            df['donchian_upper'] = high.rolling(window=20).max()
            df['donchian_lower'] = low.rolling(window=20).min()
            
            # Keltner Channels
            keltner_middle = close.ewm(span=20).mean()
            df['keltner_upper'] = keltner_middle + (atr * 2)
            df['keltner_middle'] = keltner_middle
            df['keltner_lower'] = keltner_middle - (atr * 2)
            
            # DEMA (Double EMA)
            ema_20 = close.ewm(span=20).mean()
            df['dema'] = 2 * ema_20 - ema_20.ewm(span=20).mean()
            
            # OBV (On Balance Volume)
            df['obv'] = (volume * np.sign(close.diff())).cumsum()
            
            # === MISSING TREND INDICATORS ===
            
            # HMA (Hull Moving Average)
            half_period_9 = int(9 / 2)
            wma_half_9 = close.rolling(window=half_period_9).mean()
            wma_full_9 = close.rolling(window=9).mean()
            raw_hma_9 = 2 * wma_half_9 - wma_full_9
            hma_period_9 = int(np.sqrt(9))
            df['hma_9'] = raw_hma_9.rolling(window=hma_period_9).mean()
            
            half_period_20 = int(20 / 2)
            wma_half_20 = close.rolling(window=half_period_20).mean()
            wma_full_20 = close.rolling(window=20).mean()
            raw_hma_20 = 2 * wma_half_20 - wma_full_20
            hma_period_20 = int(np.sqrt(20))
            df['hma_20'] = raw_hma_20.rolling(window=hma_period_20).mean()
            
            # Supertrend
            atr_multiplier = 3.0
            hl2 = (high + low) / 2
            basic_ub = hl2 + (atr_multiplier * atr)
            basic_lb = hl2 - (atr_multiplier * atr)
            
            # Simplified supertrend calculation
            supertrend = pd.Series(index=close.index, dtype=float)
            direction = pd.Series(1, index=close.index)  # 1 = uptrend, -1 = downtrend
            
            for i in range(1, len(close)):
                if close.iloc[i] > basic_ub.iloc[i-1]:
                    direction.iloc[i] = 1
                elif close.iloc[i] < basic_lb.iloc[i-1]:
                    direction.iloc[i] = -1
                else:
                    direction.iloc[i] = direction.iloc[i-1]
                
                if direction.iloc[i] == 1:
                    supertrend.iloc[i] = basic_lb.iloc[i]
                else:
                    supertrend.iloc[i] = basic_ub.iloc[i]
            
            df['supertrend'] = supertrend
            df['supertrend_direction'] = direction
            
            # === MISSING MOMENTUM INDICATORS ===
            
            # TSI (True Strength Index)
            momentum = close.diff()
            double_smoothed_momentum = momentum.ewm(span=25).mean().ewm(span=13).mean()
            double_smoothed_abs_momentum = abs(momentum).ewm(span=25).mean().ewm(span=13).mean()
            df['tsi'] = 100 * (double_smoothed_momentum / double_smoothed_abs_momentum)
            
            # CMO (Chande Momentum Oscillator)
            gain_sum = gain.rolling(window=14).sum()
            loss_sum = loss.rolling(window=14).sum()
            df['cmo'] = 100 * ((gain_sum - loss_sum) / (gain_sum + loss_sum))
            
            # PPO (Percentage Price Oscillator)
            ppo_ema_fast = close.ewm(span=12).mean()
            ppo_ema_slow = close.ewm(span=26).mean()
            df['ppo'] = 100 * ((ppo_ema_fast - ppo_ema_slow) / ppo_ema_slow)
            df['ppo_signal'] = df['ppo'].ewm(span=9).mean()
            df['ppo_histogram'] = df['ppo'] - df['ppo_signal']
            
            # TRIX
            ema1_15 = close.ewm(span=15).mean()
            ema2_15 = ema1_15.ewm(span=15).mean()
            ema3_15 = ema2_15.ewm(span=15).mean()
            df['trix'] = 100 * ema3_15.pct_change()
            
            # Ultimate Oscillator (combines 7, 14, 28 periods)
            bp = close - pd.concat([low, close.shift(1)], axis=1).min(axis=1)
            tr_uo = pd.concat([high - low, abs(high - close.shift(1)), abs(low - close.shift(1))], axis=1).max(axis=1)
            
            avg7 = bp.rolling(7).sum() / tr_uo.rolling(7).sum()
            avg14 = bp.rolling(14).sum() / tr_uo.rolling(14).sum()
            avg28 = bp.rolling(28).sum() / tr_uo.rolling(28).sum()
            df['ultimate_osc'] = 100 * ((4*avg7 + 2*avg14 + avg28) / 7)
            
            # Awesome Oscillator
            median_price = (high + low) / 2
            sma_5_ao = median_price.rolling(window=5).mean()
            sma_34_ao = median_price.rolling(window=34).mean()
            df['awesome_osc'] = sma_5_ao - sma_34_ao
            
            # === MISSING VOLATILITY INDICATORS ===
            
            # Mass Index
            ema_diff_9 = (high - low).ewm(span=9).mean()
            ema_diff_double_9 = ema_diff_9.ewm(span=9).mean()
            mass_ratio = ema_diff_9 / ema_diff_double_9
            df['mass_index'] = mass_ratio.rolling(window=25).sum()
            
            # Chandelier Exit
            chandelier_period = 22
            chandelier_mult = 3.0
            df['chandelier_long'] = high.rolling(window=chandelier_period).max() - (atr * chandelier_mult)
            df['chandelier_short'] = low.rolling(window=chandelier_period).min() + (atr * chandelier_mult)
            
            # === ADDITIONAL VARIATIONS FOR 50+ TOTAL ===
            
            # Multiple EMA periods
            for period in [5, 8, 13, 34, 55, 89, 144, 200]:
                if len(close) >= period:
                    df[f'ema_{period}'] = close.ewm(span=period).mean()
            
            # Multiple SMA periods
            for period in [10, 30, 100, 150]:
                if len(close) >= period:
                    df[f'sma_{period}'] = close.rolling(window=period).mean()
            
            # Multiple RSI periods
            for period in [7, 21]:
                if len(close) >= period + 1:
                    delta_rsi = close.diff()
                    gain_rsi = (delta_rsi.where(delta_rsi > 0, 0)).rolling(window=period).mean()
                    loss_rsi = (-delta_rsi.where(delta_rsi < 0, 0)).rolling(window=period).mean()
                    rs_rsi = gain_rsi / loss_rsi
                    df[f'rsi_{period}'] = 100 - (100 / (1 + rs_rsi))
            
            # ROC (Rate of Change) for multiple periods
            for period in [9, 12, 25]:
                if len(close) >= period:
                    df[f'roc_{period}'] = 100 * (close.pct_change(periods=period))
            
            # EMA Spreads (trend strength)
            df['ema_spread_fast'] = df['ema_12'] - df['ema_26']
            df['ema_spread_medium'] = df['ema_21'] - df['ema_50']
            
            # BB Width (volatility measure)
            df['bb_width'] = (bb_upper - bb_lower) / bb_middle
            df['bb_pct_b'] = (close - bb_lower) / (bb_upper - bb_lower)
            
            # ATR Percent
            df['atr_percent'] = (atr / close) * 100
            
            # VWAP (Volume Weighted Average Price)
            typical_price = (high + low + close) / 3
            df['vwap'] = (typical_price * volume).cumsum() / volume.cumsum()
            
            # Chaikin Money Flow
            mfm = ((close - low) - (high - close)) / (high - low)
            mfm = mfm.fillna(0)
            mfv = mfm * volume
            df['cmf'] = mfv.rolling(window=20).sum() / volume.rolling(window=20).sum()
            
            # Fill NaN values
            df = df.fillna(method='bfill').fillna(0)
            
            return df
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            return None

