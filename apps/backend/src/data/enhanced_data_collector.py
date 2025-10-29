"""
Enhanced Data Collection System for AlphaPlus
Collects comprehensive market data for advanced pattern detection
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import ccxt
import asyncpg
from dataclasses import dataclass
import json

logger = logging.getLogger(__name__)

@dataclass
class MarketDataPoint:
    """Enhanced market data point with comprehensive information"""
    symbol: str
    timeframe: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    price_change: float
    volume_change: float
    volatility: float
    rsi: float
    macd: float
    macd_signal: float
    bollinger_upper: float
    bollinger_lower: float
    bollinger_middle: float
    atr: float
    support_level: float
    resistance_level: float
    market_sentiment: float
    data_quality_score: float

class EnhancedDataCollector:
    """Enhanced data collection system with comprehensive market analysis"""
    
    def __init__(self, db_pool: asyncpg.Pool, exchange: ccxt.Exchange):
        self.db_pool = db_pool
        self.exchange = exchange
        self.data_buffer: Dict[str, Dict[str, List[MarketDataPoint]]] = {}
        self.volatility_window = 20
        self.rsi_period = 14
        self.macd_fast = 12
        self.macd_slow = 26
        self.macd_signal = 9
        self.bollinger_period = 20
        self.bollinger_std = 2
        self.atr_period = 14
        
    async def collect_enhanced_data(self, symbol: str, timeframe: str, limit: int = 500) -> List[MarketDataPoint]:
        """Collect enhanced market data with technical indicators"""
        try:
            # Fetch OHLCV data
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            
            if not ohlcv or len(ohlcv) < 50:
                logger.warning(f"Insufficient data for {symbol} {timeframe}: {len(ohlcv) if ohlcv else 0} points")
                return []
            
            # Convert to DataFrame for technical analysis
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Calculate technical indicators
            df = self._calculate_technical_indicators(df)
            
            # Calculate support and resistance levels
            df = self._calculate_support_resistance(df)
            
            # Calculate market sentiment
            df = self._calculate_market_sentiment(df)
            
            # Calculate data quality score
            df = self._calculate_data_quality(df)
            
            # Convert to MarketDataPoint objects
            data_points = []
            for _, row in df.iterrows():
                data_point = MarketDataPoint(
                    symbol=symbol,
                    timeframe=timeframe,
                    timestamp=row['timestamp'],
                    open=float(row['open']),
                    high=float(row['high']),
                    low=float(row['low']),
                    close=float(row['close']),
                    volume=float(row['volume']),
                    price_change=float(row['price_change']),
                    volume_change=float(row['volume_change']),
                    volatility=float(row['volatility']),
                    rsi=float(row['rsi']),
                    macd=float(row['macd']),
                    macd_signal=float(row['macd_signal']),
                    bollinger_upper=float(row['bollinger_upper']),
                    bollinger_lower=float(row['bollinger_lower']),
                    bollinger_middle=float(row['bollinger_middle']),
                    atr=float(row['atr']),
                    support_level=float(row['support_level']),
                    resistance_level=float(row['resistance_level']),
                    market_sentiment=float(row['market_sentiment']),
                    data_quality_score=float(row['data_quality_score'])
                )
                data_points.append(data_point)
            
            logger.info(f"✅ Enhanced data collected for {symbol} {timeframe}: {len(data_points)} points")
            return data_points
            
        except Exception as e:
            logger.error(f"❌ Error collecting enhanced data for {symbol} {timeframe}: {e}")
            return []
    
    def _calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive technical indicators"""
        try:
            # Price change
            df['price_change'] = df['close'].pct_change()
            
            # Volume change
            df['volume_change'] = df['volume'].pct_change()
            
            # Volatility (rolling standard deviation of returns)
            df['volatility'] = df['price_change'].rolling(window=self.volatility_window).std()
            
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # MACD
            exp1 = df['close'].ewm(span=self.macd_fast).mean()
            exp2 = df['close'].ewm(span=self.macd_slow).mean()
            df['macd'] = exp1 - exp2
            df['macd_signal'] = df['macd'].ewm(span=self.macd_signal).mean()
            
            # Bollinger Bands
            df['bollinger_middle'] = df['close'].rolling(window=self.bollinger_period).mean()
            bb_std = df['close'].rolling(window=self.bollinger_period).std()
            df['bollinger_upper'] = df['bollinger_middle'] + (bb_std * self.bollinger_std)
            df['bollinger_lower'] = df['bollinger_middle'] - (bb_std * self.bollinger_std)
            
            # ATR (Average True Range)
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift())
            low_close = np.abs(df['low'] - df['close'].shift())
            true_range = np.maximum(high_low, np.maximum(high_close, low_close))
            df['atr'] = true_range.rolling(window=self.atr_period).mean()
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Error calculating technical indicators: {e}")
            return df
    
    def _calculate_support_resistance(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate dynamic support and resistance levels"""
        try:
            # Simple support and resistance using recent highs and lows
            window = 20
            
            # Resistance level (recent highs)
            df['resistance_level'] = df['high'].rolling(window=window).max()
            
            # Support level (recent lows)
            df['support_level'] = df['low'].rolling(window=window).min()
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Error calculating support/resistance: {e}")
            df['resistance_level'] = df['high']
            df['support_level'] = df['low']
            return df
    
    def _calculate_market_sentiment(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate market sentiment based on price action and volume"""
        try:
            # Simple sentiment based on price momentum and volume
            price_momentum = df['close'].pct_change(periods=5)
            volume_momentum = df['volume'].pct_change(periods=5)
            
            # Combine price and volume momentum for sentiment
            df['market_sentiment'] = (price_momentum * 0.7 + volume_momentum * 0.3)
            
            # Normalize to -1 to 1 range
            df['market_sentiment'] = np.clip(df['market_sentiment'], -1, 1)
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Error calculating market sentiment: {e}")
            df['market_sentiment'] = 0
            return df
    
    def _calculate_data_quality(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate data quality score based on various factors"""
        try:
            # Check for missing values
            missing_ratio = df.isnull().sum().sum() / (len(df) * len(df.columns))
            
            # Check for price anomalies (extreme price changes)
            price_changes = df['close'].pct_change().abs()
            anomaly_ratio = (price_changes > 0.1).sum() / len(df)
            
            # Check for volume anomalies
            volume_changes = df['volume'].pct_change().abs()
            volume_anomaly_ratio = (volume_changes > 2.0).sum() / len(df)
            
            # Calculate quality score (0-1, higher is better)
            quality_score = 1.0 - (missing_ratio + anomaly_ratio + volume_anomaly_ratio) / 3
            quality_score = max(0, min(1, quality_score))
            
            df['data_quality_score'] = quality_score
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Error calculating data quality: {e}")
            df['data_quality_score'] = 0.5
            return df
    
    async def store_enhanced_data(self, data_points: List[MarketDataPoint]) -> bool:
        """Store enhanced data in TimescaleDB"""
        try:
            if not data_points:
                return False
            
            async with self.db_pool.acquire() as conn:
                # Insert enhanced market data
                await conn.executemany("""
                    INSERT INTO enhanced_market_data (
                        symbol, timeframe, timestamp, open, high, low, close, volume,
                        price_change, volume_change, volatility, rsi, macd, macd_signal,
                        bollinger_upper, bollinger_lower, bollinger_middle, atr,
                        support_level, resistance_level, market_sentiment, data_quality_score
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14,
                             $15, $16, $17, $18, $19, $20, $21, $22)
                    ON CONFLICT (symbol, timeframe, timestamp) DO UPDATE SET
                        open = EXCLUDED.open, high = EXCLUDED.high, low = EXCLUDED.low,
                        close = EXCLUDED.close, volume = EXCLUDED.volume,
                        price_change = EXCLUDED.price_change, volume_change = EXCLUDED.volume_change,
                        volatility = EXCLUDED.volatility, rsi = EXCLUDED.rsi,
                        macd = EXCLUDED.macd, macd_signal = EXCLUDED.macd_signal,
                        bollinger_upper = EXCLUDED.bollinger_upper, bollinger_lower = EXCLUDED.bollinger_lower,
                        bollinger_middle = EXCLUDED.bollinger_middle, atr = EXCLUDED.atr,
                        support_level = EXCLUDED.support_level, resistance_level = EXCLUDED.resistance_level,
                        market_sentiment = EXCLUDED.market_sentiment, data_quality_score = EXCLUDED.data_quality_score
                """, [
                    (dp.symbol, dp.timeframe, dp.timestamp, dp.open, dp.high, dp.low, dp.close, dp.volume,
                     dp.price_change, dp.volume_change, dp.volatility, dp.rsi, dp.macd, dp.macd_signal,
                     dp.bollinger_upper, dp.bollinger_lower, dp.bollinger_middle, dp.atr,
                     dp.support_level, dp.resistance_level, dp.market_sentiment, dp.data_quality_score)
                    for dp in data_points
                ])
            
            logger.info(f"✅ Stored {len(data_points)} enhanced data points in database")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error storing enhanced data: {e}")
            return False
    
    async def get_enhanced_data(self, symbol: str, timeframe: str, limit: int = 200) -> List[MarketDataPoint]:
        """Retrieve enhanced data from database"""
        try:
            async with self.db_pool.acquire() as conn:
                rows = await conn.fetch("""
                    SELECT symbol, timeframe, timestamp, open, high, low, close, volume,
                           price_change, volume_change, volatility, rsi, macd, macd_signal,
                           bollinger_upper, bollinger_lower, bollinger_middle, atr,
                           support_level, resistance_level, market_sentiment, data_quality_score
                    FROM enhanced_market_data
                    WHERE symbol = $1 AND timeframe = $2
                    ORDER BY timestamp DESC
                    LIMIT $3
                """, symbol, timeframe, limit)
                
                data_points = []
                for row in rows:
                    data_point = MarketDataPoint(
                        symbol=row['symbol'],
                        timeframe=row['timeframe'],
                        timestamp=row['timestamp'],
                        open=float(row['open']),
                        high=float(row['high']),
                        low=float(row['low']),
                        close=float(row['close']),
                        volume=float(row['volume']),
                        price_change=float(row['price_change']),
                        volume_change=float(row['volume_change']),
                        volatility=float(row['volatility']),
                        rsi=float(row['rsi']),
                        macd=float(row['macd']),
                        macd_signal=float(row['macd_signal']),
                        bollinger_upper=float(row['bollinger_upper']),
                        bollinger_lower=float(row['bollinger_lower']),
                        bollinger_middle=float(row['bollinger_middle']),
                        atr=float(row['atr']),
                        support_level=float(row['support_level']),
                        resistance_level=float(row['resistance_level']),
                        market_sentiment=float(row['market_sentiment']),
                        data_quality_score=float(row['data_quality_score'])
                    )
                    data_points.append(data_point)
                
                return data_points
                
        except Exception as e:
            logger.error(f"❌ Error retrieving enhanced data: {e}")
            return []
