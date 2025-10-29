"""
Calculate agreement from ALL 69 indicators
While using only 25 core for weighted scoring (avoid double-counting)
"""

import pandas as pd
import numpy as np

def calculate_full_indicator_agreement(df: pd.DataFrame, technical_score: float) -> dict:
    """
    Calculate how many of ALL 69 indicators agree with the direction
    
    Args:
        df: DataFrame with all 69 indicator columns
        technical_score: Final weighted score from 25 core indicators
        
    Returns:
        dict with agreement stats
    """
    
    # Determine direction from technical score
    if technical_score >= 0.55:
        target_direction = 'bullish'
        check_value = 0.55  # Indicators should be > 0.55 for bullish
    elif technical_score <= 0.45:
        target_direction = 'bearish'
        check_value = 0.45  # Indicators should be < 0.45 for bearish
    else:
        target_direction = 'neutral'
        return {
            'total_indicators': 0,
            'agreeing_count': 0,
            'agreement_rate': 0.0,
            'direction': 'neutral'
        }
    
    # Get all indicator columns (exclude OHLCV)
    indicator_cols = [col for col in df.columns if col not in ['open', 'high', 'low', 'close', 'volume', 'timestamp']]
    
    total_count = len(indicator_cols)
    agreeing_count = 0
    
    # Check each indicator
    for col in indicator_cols:
        try:
            value = df[col].iloc[-1]
            
            if pd.isna(value) or np.isinf(value):
                continue
            
            # Normalize value to 0-1 for comparison
            normalized = normalize_indicator_value(col, value, df)
            
            # Check if agrees with direction
            if target_direction == 'bullish' and normalized > 0.55:
                agreeing_count += 1
            elif target_direction == 'bearish' and normalized < 0.45:
                agreeing_count += 1
                
        except Exception as e:
            continue
    
    agreement_rate = agreeing_count / total_count if total_count > 0 else 0.0
    
    return {
        'total_indicators': total_count,
        'agreeing_count': agreeing_count,
        'disagreeing_count': total_count - agreeing_count,
        'agreement_rate': agreement_rate,
        'direction': target_direction
    }

def normalize_indicator_value(indicator_name: str, value: float, df: pd.DataFrame) -> float:
    """
    Normalize different indicator types to 0-1 scale
    
    0.0 = Strong bearish
    0.5 = Neutral
    1.0 = Strong bullish
    """
    
    # RSI-type (0-100)
    if 'rsi' in indicator_name.lower():
        return value / 100.0
    
    # Oscillators (-100 to 100)
    elif indicator_name in ['cci', 'cmo', 'tsi']:
        return (value + 100) / 200.0
    
    # Williams %R (-100 to 0)
    elif 'williams' in indicator_name:
        return (value + 100) / 100.0
    
    # Stochastic (0-100)
    elif 'stoch' in indicator_name:
        return value / 100.0
    
    # Ultimate Oscillator (0-100)
    elif 'ultimate' in indicator_name:
        return value / 100.0
    
    # Moving averages (compare to price)
    elif 'ema_' in indicator_name or 'sma_' in indicator_name:
        close = df['close'].iloc[-1]
        # If price > MA = bullish (>0.5), price < MA = bearish (<0.5)
        return 1.0 if close > value else 0.0
    
    # HMA
    elif 'hma' in indicator_name:
        close = df['close'].iloc[-1]
        return 1.0 if close > value else 0.0
    
    # Supertrend
    elif 'supertrend' in indicator_name and 'direction' not in indicator_name:
        close = df['close'].iloc[-1]
        return 1.0 if close > value else 0.0
    
    # MACD (positive = bullish, negative = bearish)
    elif indicator_name in ['macd', 'macd_histogram', 'ppo', 'ppo_histogram']:
        # Normalize around 0
        if value > 0:
            return 0.5 + min(0.5, abs(value) / 100)
        else:
            return 0.5 - min(0.5, abs(value) / 100)
    
    # ADX (strength indicator, use with trend)
    elif indicator_name == 'adx':
        # High ADX = strong trend
        # Check if trending up or down
        close_current = df['close'].iloc[-1]
        close_prev = df['close'].iloc[-10] if len(df) > 10 else close_current
        if close_current > close_prev:
            return min(1.0, value / 50)  # Strong uptrend
        else:
            return max(0.0, 1.0 - value / 50)  # Strong downtrend
    
    # Aroon
    elif 'aroon_up' in indicator_name:
        return value / 100.0
    elif 'aroon_down' in indicator_name:
        return 1.0 - (value / 100.0)  # Inverse
    
    # Bollinger Bands (position)
    elif 'bb_' in indicator_name and 'upper' not in indicator_name and 'lower' not in indicator_name:
        close = df['close'].iloc[-1]
        if 'bb_upper' in df.columns and 'bb_lower' in df.columns:
            upper = df['bb_upper'].iloc[-1]
            lower = df['bb_lower'].iloc[-1]
            position = (close - lower) / (upper - lower) if upper != lower else 0.5
            return np.clip(position, 0, 1)
        return 0.5
    
    # Donchian
    elif 'donchian' in indicator_name:
        close = df['close'].iloc[-1]
        if 'donchian_upper' in df.columns and 'donchian_lower' in df.columns:
            upper = df['donchian_upper'].iloc[-1]
            lower = df['donchian_lower'].iloc[-1]
            position = (close - lower) / (upper - lower) if upper != lower else 0.5
            return np.clip(position, 0, 1)
        return 0.5
    
    # Keltner
    elif 'keltner' in indicator_name and 'upper' not in indicator_name and 'lower' not in indicator_name:
        close = df['close'].iloc[-1]
        if 'keltner_upper' in df.columns and 'keltner_lower' in df.columns:
            upper = df['keltner_upper'].iloc[-1]
            lower = df['keltner_lower'].iloc[-1]
            position = (close - lower) / (upper - lower) if upper != lower else 0.5
            return np.clip(position, 0, 1)
        return 0.5
    
    # ROC (Rate of Change - percentage)
    elif 'roc_' in indicator_name:
        # ROC > 0 = bullish, < 0 = bearish
        return 0.5 + np.clip(value / 20, -0.5, 0.5)
    
    # TRIX
    elif indicator_name == 'trix':
        # TRIX > 0 = bullish
        return 0.5 + np.clip(value / 2, -0.5, 0.5)
    
    # Awesome Oscillator
    elif 'awesome' in indicator_name:
        # AO > 0 = bullish
        return 1.0 if value > 0 else 0.0
    
    # Mass Index (> 27 = reversal warning)
    elif 'mass_index' in indicator_name:
        return 0.5  # Neutral for direction
    
    # Chandelier
    elif 'chandelier' in indicator_name:
        close = df['close'].iloc[-1]
        return 1.0 if 'long' in indicator_name and close > value else 0.0 if 'short' in indicator_name and close < value else 0.5
    
    # Default: assume already normalized
    else:
        return np.clip(value, 0, 1)

