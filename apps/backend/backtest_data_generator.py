"""
AlphaPulse Historical Data Generator & Backtester
Fetches real historical data and generates signals for database population
"""

import asyncio
import aiohttp
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict
import json
import time

class HistoricalDataGenerator:
    """Fetch and process historical crypto data"""
    
    def __init__(self):
        self.base_url = "https://api.binance.com/api/v3"
        self.symbols = [
            "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "ADAUSDT",
            "XRPUSDT", "DOTUSDT", "AVAXUSDT", "MATICUSDT", "LINKUSDT"
        ]
        self.timeframes = ["1h", "4h", "1d"]
        self.signals = []
        
    async def fetch_klines(self, symbol: str, interval: str, limit: int = 1000):
        """Fetch historical candlestick data from Binance"""
        url = f"{self.base_url}/klines"
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return self._process_klines(data, symbol, interval)
                    else:
                        print(f"Error fetching {symbol}: {response.status}")
                        return None
        except Exception as e:
            print(f"Exception fetching {symbol}: {e}")
            return None
    
    def _process_klines(self, klines: List, symbol: str, interval: str) -> pd.DataFrame:
        """Convert klines to pandas DataFrame"""
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])
        
        # Convert to numeric
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['symbol'] = symbol
        df['timeframe'] = interval
        
        return df[['timestamp', 'symbol', 'timeframe', 'open', 'high', 'low', 'close', 'volume']]
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators"""
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Moving Averages
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        df['ema_20'] = df['close'].ewm(span=20).mean()
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        
        # MACD
        exp1 = df['close'].ewm(span=12).mean()
        exp2 = df['close'].ewm(span=26).mean()
        df['macd'] = exp1 - exp2
        df['signal_line'] = df['macd'].ewm(span=9).mean()
        
        # Volume indicators
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        return df
    
    def detect_signals(self, df: pd.DataFrame) -> List[Dict]:
        """Detect trading signals from indicators"""
        signals = []
        
        for i in range(50, len(df)):
            row = df.iloc[i]
            prev_row = df.iloc[i-1]
            
            # Skip if not enough data
            if pd.isna(row['rsi']) or pd.isna(row['macd']):
                continue
            
            signal = None
            pattern_type = None
            confidence = 0.5
            
            # RSI Oversold/Overbought
            if row['rsi'] < 30 and prev_row['rsi'] >= 30:
                signal = 'long'
                pattern_type = 'rsi_oversold'
                confidence = 0.65 + (30 - row['rsi']) / 100
                
            elif row['rsi'] > 70 and prev_row['rsi'] <= 70:
                signal = 'short'
                pattern_type = 'rsi_overbought'
                confidence = 0.65 + (row['rsi'] - 70) / 100
            
            # MACD Crossover
            elif row['macd'] > row['signal_line'] and prev_row['macd'] <= prev_row['signal_line']:
                signal = 'long'
                pattern_type = 'macd_bullish_cross'
                confidence = 0.70
                
            elif row['macd'] < row['signal_line'] and prev_row['macd'] >= prev_row['signal_line']:
                signal = 'short'
                pattern_type = 'macd_bearish_cross'
                confidence = 0.70
            
            # Bollinger Band Bounce
            elif row['close'] < row['bb_lower'] and prev_row['close'] >= prev_row['bb_lower']:
                signal = 'long'
                pattern_type = 'bb_lower_bounce'
                confidence = 0.75
                
            elif row['close'] > row['bb_upper'] and prev_row['close'] <= prev_row['bb_upper']:
                signal = 'short'
                pattern_type = 'bb_upper_bounce'
                confidence = 0.75
            
            # Moving Average Crossover
            elif row['sma_20'] > row['sma_50'] and prev_row['sma_20'] <= prev_row['sma_50']:
                signal = 'long'
                pattern_type = 'golden_cross'
                confidence = 0.80
                
            elif row['sma_20'] < row['sma_50'] and prev_row['sma_20'] >= prev_row['sma_50']:
                signal = 'short'
                pattern_type = 'death_cross'
                confidence = 0.80
            
            # Volume Confirmation
            if signal and row['volume_ratio'] > 1.5:
                confidence = min(confidence + 0.10, 0.95)
                pattern_type = f"{pattern_type}_volume_confirmed"
            
            if signal:
                entry_price = float(row['close'])
                
                # Calculate stop loss and take profit
                if signal == 'long':
                    stop_loss = entry_price * 0.97  # 3% stop loss
                    take_profit = entry_price * 1.06  # 6% take profit
                else:
                    stop_loss = entry_price * 1.03
                    take_profit = entry_price * 0.94
                
                signals.append({
                    'timestamp': row['timestamp'].isoformat(),
                    'symbol': row['symbol'],
                    'timeframe': row['timeframe'],
                    'direction': signal,
                    'pattern_type': pattern_type,
                    'confidence': round(confidence, 2),
                    'entry_price': round(entry_price, 2),
                    'stop_loss': round(stop_loss, 2),
                    'take_profit': round(take_profit, 2),
                    'rsi': round(float(row['rsi']), 2),
                    'macd': round(float(row['macd']), 4),
                    'volume_ratio': round(float(row['volume_ratio']), 2)
                })
        
        return signals
    
    async def generate_historical_signals(self, days: int = 30):
        """Generate signals from historical data for all symbols"""
        print(f"Starting historical signal generation for {days} days...")
        print(f"Symbols: {', '.join(self.symbols)}")
        print(f"Timeframes: {', '.join(self.timeframes)}\n")
        
        total_signals = 0
        
        for symbol in self.symbols:
            print(f"Processing {symbol}...")
            
            for timeframe in self.timeframes:
                # Calculate how many candles we need
                if timeframe == "1h":
                    limit = min(days * 24, 1000)
                elif timeframe == "4h":
                    limit = min(days * 6, 1000)
                else:  # 1d
                    limit = min(days, 1000)
                
                # Fetch data
                df = await self.fetch_klines(symbol, timeframe, limit)
                
                if df is not None and len(df) > 50:
                    # Calculate indicators
                    df = self.calculate_indicators(df)
                    
                    # Detect signals
                    signals = self.detect_signals(df)
                    
                    print(f"  {timeframe}: Found {len(signals)} signals")
                    total_signals += len(signals)
                    self.signals.extend(signals)
                
                # Rate limiting
                await asyncio.sleep(0.2)
            
            print(f"  Total for {symbol}: {len([s for s in self.signals if s['symbol'] == symbol])} signals\n")
        
        print(f"\nTotal signals generated: {total_signals}")
        return self.signals
    
    def save_to_json(self, filename: str = "historical_signals.json"):
        """Save signals to JSON file"""
        with open(filename, 'w') as f:
            json.dump(self.signals, f, indent=2)
        print(f"Saved {len(self.signals)} signals to {filename}")
    
    def get_statistics(self):
        """Get statistics about generated signals"""
        if not self.signals:
            return None
        
        df = pd.DataFrame(self.signals)
        
        stats = {
            'total_signals': len(self.signals),
            'by_symbol': df.groupby('symbol').size().to_dict(),
            'by_timeframe': df.groupby('timeframe').size().to_dict(),
            'by_direction': df.groupby('direction').size().to_dict(),
            'avg_confidence': round(df['confidence'].mean(), 2),
            'high_confidence_signals': len(df[df['confidence'] >= 0.80]),
            'date_range': {
                'start': df['timestamp'].min(),
                'end': df['timestamp'].max()
            }
        }
        
        return stats

async def main():
    """Main execution"""
    print("=" * 60)
    print("AlphaPulse Historical Signal Generator")
    print("=" * 60)
    print()
    
    generator = HistoricalDataGenerator()
    
    # Generate signals for last 30 days
    signals = await generator.generate_historical_signals(days=30)
    
    # Save to file
    generator.save_to_json("historical_signals.json")
    
    # Show statistics
    print("\n" + "=" * 60)
    print("STATISTICS")
    print("=" * 60)
    stats = generator.get_statistics()
    if stats:
        print(f"\nTotal Signals: {stats['total_signals']}")
        print(f"Average Confidence: {stats['avg_confidence'] * 100}%")
        print(f"High Confidence (80%+): {stats['high_confidence_signals']}")
        
        print("\nBy Symbol:")
        for symbol, count in sorted(stats['by_symbol'].items(), key=lambda x: x[1], reverse=True):
            print(f"  {symbol}: {count}")
        
        print("\nBy Timeframe:")
        for tf, count in stats['by_timeframe'].items():
            print(f"  {tf}: {count}")
        
        print("\nBy Direction:")
        for direction, count in stats['by_direction'].items():
            print(f"  {direction.upper()}: {count}")
        
        print(f"\nDate Range:")
        print(f"  From: {stats['date_range']['start']}")
        print(f"  To: {stats['date_range']['end']}")
    
    print("\n" + "=" * 60)
    print("COMPLETE!")
    print("=" * 60)
    print("\nNext step: Update backend to use historical_signals.json")

if __name__ == "__main__":
    asyncio.run(main())

