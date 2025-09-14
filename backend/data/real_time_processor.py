#!/usr/bin/env python3
"""
Real-Time Candlestick Processor for AlphaPulse
Processes incoming WebSocket candlestick data and generates signals
"""

import asyncio
import logging
from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from collections import defaultdict, deque
import json
import sys
import os

# Add the backend directory to Python path for imports
current_dir = os.path.dirname(__file__)
backend_dir = os.path.dirname(current_dir)
sys.path.insert(0, backend_dir)

# Import with proper paths
try:
    from ..strategies.real_time_signal_generator import RealTimeSignalGenerator, TradingSignal
    from ..strategies.ml_pattern_detector import MLPatternDetector
    from ..database.models import MarketData
    from market_regime_detector import MarketRegimeDetector
    # Create a placeholder for Signal if it doesn't exist
    try:
        from ..database.models import Signal
    except ImportError:
        class Signal:
            pass
except ImportError as e:
    logger = logging.getLogger(__name__)
    logger.warning(f"Could not import some modules: {e}")
    # Create placeholder classes for testing
    class RealTimeSignalGenerator:
        def __init__(self, config=None):
            pass
    class TradingSignal:
        pass
    class MLPatternDetector:
        def __init__(self):
            pass
    class MarketData:
        pass
    class Signal:
        pass

# Create placeholder for BinanceWebSocketClient
class BinanceWebSocketClient:
    def __init__(self):
        pass

logger = logging.getLogger(__name__)

@dataclass
class ProcessedCandlestick:
    """Processed candlestick with technical indicators"""
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    timeframe: str
    indicators: Dict
    patterns: List
    signals: List

class RealTimeCandlestickProcessor:
    """
    Real-time processor for WebSocket candlestick data
    Handles data processing, pattern detection, and signal generation
    """
    
    def __init__(self, config: Dict = None):
        """Initialize real-time processor"""
        self.config = config or {}
        
        # Initialize components
        self.signal_generator = RealTimeSignalGenerator(config)
        self.ml_detector = MLPatternDetector()
        
        # Market regime detector
        self.regime_detector = MarketRegimeDetector(
            symbol=self.config.get('symbol', 'BTC/USDT'),
            timeframe=self.config.get('timeframe', '15m'),
            enable_ml=True
        )
        
        # Data storage
        self.candlestick_data = defaultdict(lambda: defaultdict(deque))  # symbol -> timeframe -> data
        self.processed_data = defaultdict(lambda: defaultdict(deque))    # symbol -> timeframe -> processed
        self.signal_history = defaultdict(deque)  # symbol -> signals
        
        # Configuration
        self.min_data_points = self.config.get('min_data_points', 50)
        self.max_data_points = self.config.get('max_data_points', 1000)
        self.signal_cooldown = self.config.get('signal_cooldown', 300)  # 5 minutes
        self.last_signal_time = defaultdict(dict)  # symbol -> timeframe -> last signal time
        
        # Performance tracking
        self.processing_stats = {
            'total_candlesticks': 0,
            'total_signals': 0,
            'processing_time_avg': 0.0,
            'last_update': None
        }
        
        # Callbacks
        self.signal_callbacks = []
        self.data_callbacks = []
        
        logger.info("Real-Time Candlestick Processor initialized")
    
    async def process_candlestick(self, 
                                symbol: str, 
                                timeframe: str, 
                                candlestick_data: Dict) -> Optional[ProcessedCandlestick]:
        """
        Process incoming candlestick data
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe of the data
            candlestick_data: Raw candlestick data from WebSocket
            
        Returns:
            ProcessedCandlestick with indicators and patterns
        """
        start_time = datetime.now()
        
        try:
            # Parse candlestick data
            candlestick = self._parse_candlestick(candlestick_data)
            if not candlestick:
                return None
            
            # Store data
            self._store_candlestick(symbol, timeframe, candlestick)
            
            # Check if we have enough data for analysis
            if len(self.candlestick_data[symbol][timeframe]) < self.min_data_points:
                return None
            
            # Convert to DataFrame for analysis
            df = self._to_dataframe(symbol, timeframe)
            
            # Calculate technical indicators
            indicators = self._calculate_indicators(df)
            
            # Update market regime detection
            regime_state = None
            if hasattr(self, 'regime_detector'):
                regime_state = self.regime_detector.update_regime(indicators, candlestick_data)
            
            # Detect patterns
            patterns = self.ml_detector.detect_patterns_ml(df)
            
            # Generate signals
            signals = []
            if self._should_generate_signal(symbol, timeframe):
                signal = await self.signal_generator.generate_signal(
                    symbol, df, timeframe, {'volume': candlestick['volume']}
                )
                if signal:
                    signals.append(signal)
                    self._store_signal(symbol, signal)
                    self.last_signal_time[symbol][timeframe] = datetime.now()
                    
                    # Notify signal callbacks
                    await self._notify_signal_callbacks(signal)
            
            # Create processed candlestick
            processed = ProcessedCandlestick(
                symbol=symbol,
                timestamp=candlestick['timestamp'],
                open=candlestick['open'],
                high=candlestick['high'],
                low=candlestick['low'],
                close=candlestick['close'],
                volume=candlestick['volume'],
                timeframe=timeframe,
                indicators=indicators,
                patterns=[p.pattern for p in patterns],
                signals=[s.signal_type for s in signals]
            )
            
            # Store processed data
            self.processed_data[symbol][timeframe].append(processed)
            
            # Update statistics
            self._update_stats(start_time)
            
            # Notify data callbacks
            await self._notify_data_callbacks(processed)
            
            return processed
            
        except Exception as e:
            logger.error(f"❌ Error processing candlestick for {symbol} {timeframe}: {e}")
            return None
    
    def _parse_candlestick(self, candlestick_data: Dict) -> Optional[Dict]:
        """Parse raw candlestick data from WebSocket"""
        try:
            # Handle different WebSocket data formats
            if 'k' in candlestick_data:  # Binance format
                k = candlestick_data['k']
                return {
                    'timestamp': datetime.fromtimestamp(k['t'] / 1000),
                    'open': float(k['o']),
                    'high': float(k['h']),
                    'low': float(k['l']),
                    'close': float(k['c']),
                    'volume': float(k['v']),
                    'is_closed': k['x']
                }
            elif 'data' in candlestick_data:  # Generic format
                data = candlestick_data['data']
                return {
                    'timestamp': datetime.fromtimestamp(data['timestamp'] / 1000),
                    'open': float(data['open']),
                    'high': float(data['high']),
                    'low': float(data['low']),
                    'close': float(data['close']),
                    'volume': float(data.get('volume', 0)),
                    'is_closed': data.get('is_closed', True)
                }
            else:
                logger.warning(f"⚠️ Unknown candlestick format: {candlestick_data}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Error parsing candlestick data: {e}")
            return None
    
    def _store_candlestick(self, symbol: str, timeframe: str, candlestick: Dict):
        """Store candlestick data with size limits"""
        data_queue = self.candlestick_data[symbol][timeframe]
        data_queue.append(candlestick)
        
        # Maintain size limits
        if len(data_queue) > self.max_data_points:
            data_queue.popleft()
    
    def _to_dataframe(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """Convert stored candlestick data to DataFrame"""
        data = list(self.candlestick_data[symbol][timeframe])
        
        if not data:
            return pd.DataFrame()
        
        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)
        
        return df
    
    def _calculate_indicators(self, df: pd.DataFrame) -> Dict:
        """Calculate technical indicators for the candlestick"""
        try:
            indicators = {}
            
            if len(df) < 20:
                return indicators
            
            # Price-based indicators
            indicators['sma_20'] = float(df['close'].rolling(20).mean().iloc[-1])
            indicators['sma_50'] = float(df['close'].rolling(50).mean().iloc[-1])
            indicators['ema_12'] = float(df['close'].ewm(span=12).mean().iloc[-1])
            indicators['ema_26'] = float(df['close'].ewm(span=26).mean().iloc[-1])
            
            # Volatility indicators
            if len(df) >= 14:
                atr = self._calculate_atr(df, 14)
                indicators['atr'] = float(atr.iloc[-1]) if not pd.isna(atr.iloc[-1]) else None
            
            # Momentum indicators
            if len(df) >= 14:
                rsi = self._calculate_rsi(df['close'], 14)
                indicators['rsi'] = float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else None
            
            # Volume indicators
            if 'volume' in df.columns and len(df) >= 20:
                volume_ma = df['volume'].rolling(20).mean()
                indicators['volume_ma'] = float(volume_ma.iloc[-1]) if not pd.isna(volume_ma.iloc[-1]) else None
                indicators['volume_ratio'] = float(df['volume'].iloc[-1] / volume_ma.iloc[-1]) if not pd.isna(volume_ma.iloc[-1]) else None
            
            # Support and resistance
            if len(df) >= 20:
                support_resistance = self._calculate_support_resistance(df)
                indicators.update(support_resistance)
            
            return indicators
            
        except Exception as e:
            logger.warning(f"⚠️ Error calculating indicators: {e}")
            return {}
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        atr = true_range.rolling(period).mean()
        return atr
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_support_resistance(self, df: pd.DataFrame) -> Dict:
        """Calculate support and resistance levels"""
        try:
            # Simple support/resistance detection
            highs = df['high'].rolling(5).max()
            lows = df['low'].rolling(5).min()
            
            current_high = highs.iloc[-1]
            current_low = lows.iloc[-1]
            current_price = df['close'].iloc[-1]
            
            return {
                'nearest_resistance': float(current_high) if not pd.isna(current_high) else None,
                'nearest_support': float(current_low) if not pd.isna(current_low) else None,
                'price_position': float((current_price - current_low) / (current_high - current_low)) if current_high != current_low else 0.5
            }
        except Exception as e:
            logger.warning(f"⚠️ Error calculating support/resistance: {e}")
            return {}
    
    def _should_generate_signal(self, symbol: str, timeframe: str) -> bool:
        """Check if we should generate a signal (cooldown check)"""
        last_signal = self.last_signal_time[symbol].get(timeframe)
        if not last_signal:
            return True
        
        time_since_last = (datetime.now() - last_signal).total_seconds()
        return time_since_last >= self.signal_cooldown
    
    def _store_signal(self, symbol: str, signal: TradingSignal):
        """Store generated signal"""
        self.signal_history[symbol].append(signal)
        
        # Maintain signal history size
        if len(self.signal_history[symbol]) > 100:
            self.signal_history[symbol].popleft()
    
    async def _notify_signal_callbacks(self, signal: TradingSignal):
        """Notify registered signal callbacks"""
        for callback in self.signal_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(signal)
                else:
                    callback(signal)
            except Exception as e:
                logger.error(f"❌ Error in signal callback: {e}")
    
    async def _notify_data_callbacks(self, processed_data: ProcessedCandlestick):
        """Notify registered data callbacks"""
        for callback in self.data_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(processed_data)
                else:
                    callback(processed_data)
            except Exception as e:
                logger.error(f"❌ Error in data callback: {e}")
    
    def _update_stats(self, start_time: datetime):
        """Update processing statistics"""
        processing_time = (datetime.now() - start_time).total_seconds()
        
        self.processing_stats['total_candlesticks'] += 1
        self.processing_stats['last_update'] = datetime.now()
        
        # Update average processing time
        current_avg = self.processing_stats['processing_time_avg']
        total = self.processing_stats['total_candlesticks']
        self.processing_stats['processing_time_avg'] = (
            (current_avg * (total - 1) + processing_time) / total
        )
    
    def add_signal_callback(self, callback):
        """Add callback for signal notifications"""
        self.signal_callbacks.append(callback)
        logger.info(f"📡 Added signal callback: {callback.__name__ if hasattr(callback, '__name__') else 'anonymous'}")
    
    def add_data_callback(self, callback):
        """Add callback for data notifications"""
        self.data_callbacks.append(callback)
        logger.info(f"📡 Added data callback: {callback.__name__ if hasattr(callback, '__name__') else 'anonymous'}")
    
    def get_processing_stats(self) -> Dict:
        """Get current processing statistics"""
        return self.processing_stats.copy()
    
    def get_symbol_data(self, symbol: str, timeframe: str = None) -> Dict:
        """Get stored data for a symbol"""
        if timeframe:
            return {
                'candlesticks': list(self.candlestick_data[symbol][timeframe]),
                'processed': list(self.processed_data[symbol][timeframe]),
                'signals': list(self.signal_history[symbol])
            }
        else:
            return {
                'timeframes': list(self.candlestick_data[symbol].keys()),
                'total_signals': len(self.signal_history[symbol])
            }
    
    def get_signal_summary(self, symbol: str = None) -> Dict:
        """Get summary of generated signals"""
        if symbol:
            signals = self.signal_history[symbol]
        else:
            # Combine all symbols
            all_signals = []
            for symbol_signals in self.signal_history.values():
                all_signals.extend(symbol_signals)
            signals = all_signals
        
        if not signals:
            return {"message": "No signals generated yet"}
        
        # Group by signal type
        buy_signals = [s for s in signals if s.signal_type == 'buy']
        sell_signals = [s for s in signals if s.signal_type == 'sell']
        
        # Calculate statistics
        total_signals = len(signals)
        avg_confidence = np.mean([s.confidence for s in signals]) if signals else 0
        avg_strength = np.mean([s.strength for s in signals]) if signals else 0
        
        return {
            "total_signals": total_signals,
            "buy_signals": len(buy_signals),
            "sell_signals": len(sell_signals),
            "avg_confidence": round(avg_confidence, 3),
            "avg_strength": round(avg_strength, 3),
            "latest_signals": [
                {
                    "symbol": s.symbol,
                    "type": s.signal_type,
                    "pattern": s.pattern,
                    "confidence": s.confidence,
                    "timestamp": s.timestamp.isoformat()
                }
                for s in sorted(signals, key=lambda x: x.timestamp, reverse=True)[:10]
            ]
        }
    
    def clear_data(self, symbol: str = None, timeframe: str = None):
        """Clear stored data"""
        if symbol and timeframe:
            self.candlestick_data[symbol][timeframe].clear()
            self.processed_data[symbol][timeframe].clear()
            logger.info(f"🗑️ Cleared data for {symbol} {timeframe}")
        elif symbol:
            self.candlestick_data[symbol].clear()
            self.processed_data[symbol].clear()
            self.signal_history[symbol].clear()
            logger.info(f"🗑️ Cleared all data for {symbol}")
        else:
            self.candlestick_data.clear()
            self.processed_data.clear()
            self.signal_history.clear()
            logger.info("🗑️ Cleared all data")
    
    async def start_processing(self, websocket_client: BinanceWebSocketClient, symbols: List[str], timeframes: List[str]):
        """Start real-time processing for specified symbols and timeframes"""
        logger.info(f"🚀 Starting real-time processing for {len(symbols)} symbols and {len(timeframes)} timeframes")
        
        # Set up WebSocket subscriptions
        for symbol in symbols:
            for timeframe in timeframes:
                try:
                    # Subscribe to candlestick streams
                    await websocket_client.subscribe_candlesticks(
                        symbol, timeframe, 
                        lambda data, s=symbol, tf=timeframe: asyncio.create_task(
                            self.process_candlestick(s, tf, data)
                        )
                    )
                    logger.info(f"📡 Subscribed to {symbol} {timeframe}")
                except Exception as e:
                    logger.error(f"❌ Failed to subscribe to {symbol} {timeframe}: {e}")
        
        logger.info("✅ Real-time processing started successfully")
    
    async def stop_processing(self):
        """Stop real-time processing"""
        logger.info("🛑 Stopping real-time processing")
        
        # Clear callbacks
        self.signal_callbacks.clear()
        self.data_callbacks.clear()
        
        logger.info("✅ Real-time processing stopped")

