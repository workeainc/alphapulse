"""
Verify how many indicators are actually being calculated
"""

import pandas as pd
import numpy as np

# Generate sample data
prices = [42000 + np.random.randn() * 200 for _ in range(100)]

df = pd.DataFrame({
    'close': prices,
    'high': [p * 1.002 for p in prices],
    'low': [p * 0.998 for p in prices],
    'volume': [1000000 + np.random.randint(-200000, 200000) for _ in range(100)],
})

close = df['close']
high = df['high']
low = df['low']
volume = df['volume']

# Calculate indicators (copy from script)
delta = close.diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss
df['rsi'] = 100 - (100 / (1 + rs))

# Stochastic
low_min = low.rolling(window=14).min()
high_max = high.rolling(window=14).max()
df['stoch_k'] = 100 * ((close - low_min) / (high_max - low_min))
df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()
df['williams_r'] = -100 * ((high_max - close) / (high_max - low_min))

# CCI
tp = (high + low + close) / 3
df['cci'] = (tp - tp.rolling(window=20).mean()) / (0.015 * tp.rolling(window=20).std())

# EMAs
for period in [5, 8, 9, 12, 13, 21, 26, 34, 50, 55, 89, 144, 200]:
    if len(close) >= period:
        df[f'ema_{period}'] = close.ewm(span=period).mean()

# SMAs
for period in [10, 20, 30, 50, 100, 150, 200]:
    if len(close) >= period:
        df[f'sma_{period}'] = close.rolling(window=period).mean()

# MACD
df['macd'] = df['ema_12'] - df['ema_26']
df['macd_signal'] = df['macd'].ewm(span=9).mean()
df['macd_histogram'] = df['macd'] - df['macd_signal']

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

# Aroon
df['aroon_up'] = 100 * close.rolling(window=25).apply(lambda x: x.argmax()) / 25
df['aroon_down'] = 100 * close.rolling(window=25).apply(lambda x: x.argmin()) / 25

# BB
bb_middle = close.rolling(window=20).mean()
bb_std = close.rolling(window=20).std()
df['bb_upper'] = bb_middle + (bb_std * 2)
df['bb_middle'] = bb_middle
df['bb_lower'] = bb_middle - (bb_std * 2)

# Donchian
df['donchian_upper'] = high.rolling(window=20).max()
df['donchian_lower'] = low.rolling(window=20).min()

# Keltner
keltner_middle = close.ewm(span=20).mean()
df['keltner_upper'] = keltner_middle + (atr * 2)
df['keltner_middle'] = keltner_middle
df['keltner_lower'] = keltner_middle - (atr * 2)

# DEMA
ema_20 = close.ewm(span=20).mean()
df['dema'] = 2 * ema_20 - ema_20.ewm(span=20).mean()

# OBV
df['obv'] = (volume * np.sign(close.diff())).cumsum()

# Ichimoku
tenkan = (high.rolling(window=9).max() + low.rolling(window=9).min()) / 2
kijun = (high.rolling(window=26).max() + low.rolling(window=26).min()) / 2
df['ichimoku_base'] = kijun

# HMA
half_period = int(20 / 2)
wma_half = close.rolling(window=half_period).mean()
wma_full = close.rolling(window=20).mean()
raw_hma = 2 * wma_half - wma_full
hma_period = int(np.sqrt(20))
df['hma'] = raw_hma.rolling(window=hma_period).mean()

# TSI
momentum = close.diff()
double_smoothed = momentum.ewm(span=25).mean().ewm(span=13).mean()
double_smoothed_abs = abs(momentum).ewm(span=25).mean().ewm(span=13).mean()
df['tsi'] = 100 * (double_smoothed / double_smoothed_abs)

# CMO
gain_sum = gain.rolling(window=14).sum()
loss_sum = loss.rolling(window=14).sum()
df['cmo'] = 100 * ((gain_sum - loss_sum) / (gain_sum + loss_sum))

# PPO
ppo_fast = close.ewm(span=12).mean()
ppo_slow = close.ewm(span=26).mean()
df['ppo'] = 100 * ((ppo_fast - ppo_slow) / ppo_slow)

# TRIX
ema1 = close.ewm(span=15).mean()
ema2 = ema1.ewm(span=15).mean()
ema3 = ema2.ewm(span=15).mean()
df['trix'] = 100 * ema3.pct_change()

# Ultimate Oscillator
bp = close - pd.concat([low, close.shift(1)], axis=1).min(axis=1)
tr_calc = pd.concat([high - low, abs(high - close.shift(1)), abs(low - close.shift(1))], axis=1).max(axis=1)
avg7 = bp.rolling(7).sum() / tr_calc.rolling(7).sum()
avg14 = bp.rolling(14).sum() / tr_calc.rolling(14).sum()
avg28 = bp.rolling(28).sum() / tr_calc.rolling(28).sum()
df['ultimate_osc'] = 100 * ((4*avg7 + 2*avg14 + avg28) / 7)

# Awesome Oscillator
median_price = (high + low) / 2
sma_5 = median_price.rolling(window=5).mean()
sma_34 = median_price.rolling(window=34).mean()
df['awesome_osc'] = sma_5 - sma_34

# Mass Index
ema_diff = (high - low).ewm(span=9).mean()
ema_diff_double = ema_diff.ewm(span=9).mean()
mass_ratio = ema_diff / ema_diff_double
df['mass_index'] = mass_ratio.rolling(window=25).sum()

# Chandelier
df['chandelier_long'] = high.rolling(window=22).max() - (atr * 3.0)
df['chandelier_short'] = low.rolling(window=22).min() + (atr * 3.0)

# Multiple RSI periods
for period in [7, 21, 28]:
    if len(close) >= period + 1:
        d = close.diff()
        g = (d.where(d > 0, 0)).rolling(window=period).mean()
        l = (-d.where(d < 0, 0)).rolling(window=period).mean()
        r = g / l
        df[f'rsi_{period}'] = 100 - (100 / (1 + r))

# ROC variations
for period in [9, 12, 25]:
    if len(close) >= period:
        df[f'roc_{period}'] = 100 * close.pct_change(periods=period)

# Count total indicator columns
indicator_cols = [col for col in df.columns if col not in ['open', 'high', 'low', 'close', 'volume']]

print(f"\n{'=' * 60}")
print(f"TOTAL INDICATOR COLUMNS: {len(indicator_cols)}")
print(f"{'=' * 60}\n")

print("Indicator Columns:")
for i, col in enumerate(sorted(indicator_cols), 1):
    print(f"  {i:2d}. {col}")

print(f"\n{'=' * 60}")
print(f"✓ {len(indicator_cols)} technical indicators available!")
print(f"{'=' * 60}\n")

