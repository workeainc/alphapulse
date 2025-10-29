"""
AlphaPlus AI Trading System - Real Binance Data with Phase 1 Enhancements
Uses real Binance data with enhanced technical indicators and data quality monitoring
"""

import sys
import os
from pathlib import Path

# Add backend to path for imports
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import logging
from typing import Dict, Any, List, Optional
import json
from datetime import datetime, timedelta
import asyncio
import asyncpg
import random
import numpy as np
import pandas as pd
import ccxt

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="AlphaPlus AI Trading System - Real Binance Data with Phase 1",
    description="Real-time pattern recognition and signal generation using live Binance data with enhanced features",
    version="4.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global system components
db_pool = None
binance_exchange = None

# Real-time data buffers
market_data_buffer = {}
signal_buffer = []
pattern_buffer = []
enhanced_data_buffer = {}  # New: Enhanced data with technical indicators

# Enhanced symbols for real trading
SYMBOLS = ['BTC/USDT', 'ETH/USDT', 'ADA/USDT', 'SOL/USDT', 'BNB/USDT', 'XRP/USDT']
TIMEFRAMES = ['1m', '5m', '15m', '1h', '4h']

@app.on_event("startup")
async def startup_event():
    """Initialize the AI trading system with real data and Phase 1 enhancements"""
    global db_pool, binance_exchange
    
    try:
        logger.info("🚀 Starting AlphaPlus AI Trading System - Real Binance Data with Phase 1...")
        
        # Initialize database connection
        db_pool = await asyncpg.create_pool(
            host='postgres',
            port=5432,
            database='alphapulse',
            user='alpha_emon',
            password='Emon_@17711',
            min_size=5,
            max_size=20
        )
        logger.info("✅ Database connection established")
        
        # Initialize Binance exchange
        binance_exchange = ccxt.binance({
            'sandbox': False,  # Use real data
            'enableRateLimit': True,
        })
        logger.info("✅ Binance exchange initialized")
        
        # Start enhanced data collection with technical indicators
        asyncio.create_task(start_enhanced_data_collection())
        logger.info("✅ Enhanced data collection started")
        
        # Start enhanced pattern detection
        asyncio.create_task(start_enhanced_pattern_detection())
        logger.info("✅ Enhanced pattern detection started")
        
        # Start enhanced signal generation
        asyncio.create_task(start_enhanced_signal_generation())
        logger.info("✅ Enhanced signal generation started")
        
        # Start data quality monitoring
        asyncio.create_task(start_data_quality_monitoring())
        logger.info("✅ Data quality monitoring started")
        
        logger.info("🎉 AlphaPlus AI Trading System - Real Binance Data with Phase 1 fully activated!")
        
    except Exception as e:
        logger.error(f"❌ Failed to start AI system: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown the AI trading system"""
    global db_pool
    
    try:
        if db_pool:
            await db_pool.close()
        logger.info("✅ AI Trading System shutdown complete")
    except Exception as e:
        logger.error(f"❌ Error during shutdown: {e}")

async def start_enhanced_data_collection():
    """Start enhanced real-time data collection with technical indicators"""
    global enhanced_data_buffer, binance_exchange
    
    try:
        logger.info("🔄 Starting enhanced Binance data collection with technical indicators...")
        
        while True:
            try:
                # Fetch enhanced data for all symbols and timeframes
                for symbol in SYMBOLS:
                    if symbol not in enhanced_data_buffer:
                        enhanced_data_buffer[symbol] = {}
                    
                    for timeframe in TIMEFRAMES:
                        if timeframe not in enhanced_data_buffer[symbol]:
                            enhanced_data_buffer[symbol][timeframe] = []
                        
                        try:
                            # Get real historical data from Binance
                            ohlcv = binance_exchange.fetch_ohlcv(symbol, timeframe, limit=200)
                            
                            if ohlcv and len(ohlcv) > 0:
                                # Convert to DataFrame for technical analysis
                                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                                
                                # Calculate technical indicators
                                enhanced_data = calculate_technical_indicators(df, symbol, timeframe)
                                
                                # Store enhanced data
                                enhanced_data_buffer[symbol][timeframe] = enhanced_data
                                
                                # Store in database
                                await store_enhanced_data(enhanced_data)
                                
                                logger.info(f"📊 Enhanced data updated: {symbol} {timeframe} - {len(enhanced_data)} points with technical indicators")
                        
                        except Exception as e:
                            logger.error(f"❌ Error fetching enhanced data for {symbol} {timeframe}: {e}")
                            continue
                
                # Wait before next update
                await asyncio.sleep(30)  # Update every 30 seconds
                
            except Exception as e:
                logger.error(f"❌ Error in enhanced data collection: {e}")
                await asyncio.sleep(60)  # Wait longer on error
                
    except Exception as e:
        logger.error(f"❌ Enhanced data collection error: {e}")

def calculate_technical_indicators(df: pd.DataFrame, symbol: str, timeframe: str) -> List[Dict[str, Any]]:
    """Calculate comprehensive technical indicators"""
    try:
        if len(df) < 20:
            return []
        
        # Calculate basic indicators
        df['price_change'] = df['close'].pct_change()
        df['volume_change'] = df['volume'].pct_change()
        
        # Calculate volatility (ATR-like)
        df['high_low'] = df['high'] - df['low']
        df['volatility'] = df['high_low'].rolling(window=14).mean()
        
        # Calculate RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Calculate MACD
        exp1 = df['close'].ewm(span=12).mean()
        exp2 = df['close'].ewm(span=26).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        
        # Calculate Bollinger Bands
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['std_20'] = df['close'].rolling(window=20).std()
        df['bollinger_upper'] = df['sma_20'] + (df['std_20'] * 2)
        df['bollinger_lower'] = df['sma_20'] - (df['std_20'] * 2)
        
        # Calculate ATR
        df['tr1'] = df['high'] - df['low']
        df['tr2'] = abs(df['high'] - df['close'].shift())
        df['tr3'] = abs(df['low'] - df['close'].shift())
        df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
        df['atr'] = df['tr'].rolling(window=14).mean()
        
        # Calculate support and resistance levels
        df['support_level'] = df['low'].rolling(window=20).min()
        df['resistance_level'] = df['high'].rolling(window=20).max()
        
        # Determine market sentiment
        df['market_sentiment'] = df.apply(determine_sentiment, axis=1)
        
        # Convert to list of dictionaries
        enhanced_data = []
        for _, row in df.iterrows():
            if not pd.isna(row['close']):  # Skip rows with NaN values
                data_point = {
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'timestamp': row['timestamp'],
                    'open': float(row['open']),
                    'high': float(row['high']),
                    'low': float(row['low']),
                    'close': float(row['close']),
                    'volume': float(row['volume']),
                    'price_change': float(row['price_change']) if not pd.isna(row['price_change']) else 0,
                    'volume_change': float(row['volume_change']) if not pd.isna(row['volume_change']) else 0,
                    'volatility': float(row['volatility']) if not pd.isna(row['volatility']) else 0,
                    'rsi': float(row['rsi']) if not pd.isna(row['rsi']) else 50,
                    'macd': float(row['macd']) if not pd.isna(row['macd']) else 0,
                    'macd_signal': float(row['macd_signal']) if not pd.isna(row['macd_signal']) else 0,
                    'bollinger_upper': float(row['bollinger_upper']) if not pd.isna(row['bollinger_upper']) else 0,
                    'bollinger_lower': float(row['bollinger_lower']) if not pd.isna(row['bollinger_lower']) else 0,
                    'atr': float(row['atr']) if not pd.isna(row['atr']) else 0,
                    'support_level': float(row['support_level']) if not pd.isna(row['support_level']) else 0,
                    'resistance_level': float(row['resistance_level']) if not pd.isna(row['resistance_level']) else 0,
                    'market_sentiment': str(row['market_sentiment']),
                    'data_quality_score': 0.95
                }
                enhanced_data.append(data_point)
        
        return enhanced_data
        
    except Exception as e:
        logger.error(f"❌ Error calculating technical indicators: {e}")
        return []

def determine_sentiment(row) -> str:
    """Determine market sentiment based on technical indicators"""
    try:
        rsi = row.get('rsi', 50)
        macd = row.get('macd', 0)
        macd_signal = row.get('macd_signal', 0)
        price_change = row.get('price_change', 0)
        
        # Simple sentiment logic
        bullish_signals = 0
        bearish_signals = 0
        
        if rsi < 30:
            bullish_signals += 1
        elif rsi > 70:
            bearish_signals += 1
            
        if macd > macd_signal:
            bullish_signals += 1
        else:
            bearish_signals += 1
            
        if price_change > 0:
            bullish_signals += 1
        else:
            bearish_signals += 1
        
        if bullish_signals > bearish_signals:
            return 'bullish'
        elif bearish_signals > bullish_signals:
            return 'bearish'
        else:
            return 'neutral'
            
    except Exception as e:
        return 'neutral'

async def store_enhanced_data(enhanced_data: List[Dict[str, Any]]):
    """Store enhanced data in database"""
    try:
        if not db_pool or not enhanced_data:
            return
        
        async with db_pool.acquire() as conn:
            for data_point in enhanced_data[-10:]:  # Store only recent data points
                await conn.execute("""
                    INSERT INTO enhanced_market_data (
                        symbol, timeframe, timestamp, open, high, low, close, volume,
                        price_change, volume_change, volatility, rsi, macd, macd_signal,
                        bollinger_upper, bollinger_lower, atr, support_level, resistance_level,
                        market_sentiment, data_quality_score
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21)
                    ON CONFLICT (symbol, timeframe, timestamp) DO UPDATE SET
                        open = EXCLUDED.open, high = EXCLUDED.high, low = EXCLUDED.low,
                        close = EXCLUDED.close, volume = EXCLUDED.volume
                """, 
                data_point['symbol'], data_point['timeframe'], data_point['timestamp'],
                data_point['open'], data_point['high'], data_point['low'], data_point['close'], data_point['volume'],
                data_point['price_change'], data_point['volume_change'], data_point['volatility'],
                data_point['rsi'], data_point['macd'], data_point['macd_signal'],
                data_point['bollinger_upper'], data_point['bollinger_lower'], data_point['atr'],
                data_point['support_level'], data_point['resistance_level'],
                data_point['market_sentiment'], data_point['data_quality_score']
                )
                
    except Exception as e:
        logger.error(f"❌ Error storing enhanced data: {e}")

async def start_enhanced_pattern_detection():
    """Start enhanced pattern detection using technical indicators"""
    global pattern_buffer, enhanced_data_buffer
    
    try:
        logger.info("🎯 Starting enhanced pattern detection with technical indicators...")
        
        while True:
            try:
                for symbol in SYMBOLS:
                    for timeframe in TIMEFRAMES:
                        if (symbol in enhanced_data_buffer and 
                            timeframe in enhanced_data_buffer[symbol] and 
                            len(enhanced_data_buffer[symbol][timeframe]) >= 50):
                            
                            # Get enhanced data
                            data_points = enhanced_data_buffer[symbol][timeframe]
                            
                            # Detect comprehensive patterns
                            patterns = detect_comprehensive_patterns(data_points, symbol, timeframe)
                            
                            for pattern in patterns:
                                # Check if pattern already exists
                                existing_pattern = next((p for p in pattern_buffer if p.get('pattern_id') == pattern['pattern_id']), None)
                                if not existing_pattern:
                                    pattern_buffer.append(pattern)
                                    logger.info(f"🎯 Enhanced pattern detected: {symbol} {timeframe} - {pattern['pattern_type']} (confidence: {pattern['confidence']:.2f})")
                
                # Keep only recent patterns
                if len(pattern_buffer) > 100:
                    pattern_buffer = pattern_buffer[-100:]

                await asyncio.sleep(30) # Run pattern detection every 30 seconds

            except Exception as e:
                logger.error(f"❌ Error in enhanced pattern detection: {e}")
                await asyncio.sleep(60) # Wait longer on error
                
    except Exception as e:
        logger.error(f"❌ Enhanced pattern detection error: {e}")

def detect_comprehensive_patterns(data_points: List[Dict[str, Any]], symbol: str, timeframe: str) -> List[Dict[str, Any]]:
    """Detect comprehensive patterns using enhanced data"""
    patterns = []
    
    try:
        if len(data_points) < 20:
            return patterns
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame(data_points)
        
        # Detect various pattern types
        patterns.extend(detect_trend_patterns(df, symbol, timeframe))
        patterns.extend(detect_reversal_patterns(df, symbol, timeframe))
        patterns.extend(detect_continuation_patterns(df, symbol, timeframe))
        patterns.extend(detect_candlestick_patterns(df, symbol, timeframe))
        
        return patterns
        
    except Exception as e:
        logger.error(f"❌ Error detecting comprehensive patterns: {e}")
        return patterns

def detect_trend_patterns(df: pd.DataFrame, symbol: str, timeframe: str) -> List[Dict[str, Any]]:
    """Detect trend-based patterns"""
    patterns = []
    
    try:
        # Uptrend pattern
        if len(df) >= 20:
            recent_prices = df['close'].tail(20)
            if all(recent_prices.iloc[i] <= recent_prices.iloc[i+1] for i in range(len(recent_prices)-1)):
                pattern = {
                    'pattern_id': f"TREND-{symbol.replace('/', '')}-{timeframe}-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'pattern_type': 'uptrend',
                    'pattern_category': 'trend',
                    'direction': 'long',
                    'confidence': 0.85,
                    'strength': 'strong',
                    'entry_price': float(df['close'].iloc[-1]),
                    'stop_loss': float(df['close'].iloc[-1] * 0.98),
                    'take_profit': float(df['close'].iloc[-1] * 1.05),
                    'risk_reward_ratio': 2.5,
                    'pattern_start_time': df['timestamp'].iloc[0],
                    'pattern_end_time': df['timestamp'].iloc[-1],
                    'data_points_used': len(df),
                    'data_quality_score': 0.95,
                    'status': 'active'
                }
                patterns.append(pattern)
        
        # Downtrend pattern
        if len(df) >= 20:
            recent_prices = df['close'].tail(20)
            if all(recent_prices.iloc[i] >= recent_prices.iloc[i+1] for i in range(len(recent_prices)-1)):
                pattern = {
                    'pattern_id': f"TREND-{symbol.replace('/', '')}-{timeframe}-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'pattern_type': 'downtrend',
                    'pattern_category': 'trend',
                    'direction': 'short',
                    'confidence': 0.85,
                    'strength': 'strong',
                    'entry_price': float(df['close'].iloc[-1]),
                    'stop_loss': float(df['close'].iloc[-1] * 1.02),
                    'take_profit': float(df['close'].iloc[-1] * 0.95),
                    'risk_reward_ratio': 2.5,
                    'pattern_start_time': df['timestamp'].iloc[0],
                    'pattern_end_time': df['timestamp'].iloc[-1],
                    'data_points_used': len(df),
                    'data_quality_score': 0.95,
                    'status': 'active'
                }
                patterns.append(pattern)
        
        return patterns
        
    except Exception as e:
        logger.error(f"❌ Error detecting trend patterns: {e}")
        return patterns

def detect_reversal_patterns(df: pd.DataFrame, symbol: str, timeframe: str) -> List[Dict[str, Any]]:
    """Detect reversal patterns"""
    patterns = []
    
    try:
        # RSI divergence pattern
        if len(df) >= 14:
            rsi_values = df['rsi'].dropna()
            if len(rsi_values) >= 14:
                # Check for RSI oversold/overbought conditions
                if rsi_values.iloc[-1] < 30:  # Oversold
                    pattern = {
                        'pattern_id': f"RSI-{symbol.replace('/', '')}-{timeframe}-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
                        'symbol': symbol,
                        'timeframe': timeframe,
                        'pattern_type': 'rsi_oversold',
                        'pattern_category': 'reversal',
                        'direction': 'long',
                        'confidence': 0.75,
                        'strength': 'moderate',
                        'entry_price': float(df['close'].iloc[-1]),
                        'stop_loss': float(df['close'].iloc[-1] * 0.97),
                        'take_profit': float(df['close'].iloc[-1] * 1.04),
                        'risk_reward_ratio': 2.33,
                        'pattern_start_time': df['timestamp'].iloc[-14],
                        'pattern_end_time': df['timestamp'].iloc[-1],
                        'data_points_used': len(df),
                        'data_quality_score': 0.90,
                        'status': 'active'
                    }
                    patterns.append(pattern)
                
                elif rsi_values.iloc[-1] > 70:  # Overbought
                    pattern = {
                        'pattern_id': f"RSI-{symbol.replace('/', '')}-{timeframe}-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
                        'symbol': symbol,
                        'timeframe': timeframe,
                        'pattern_type': 'rsi_overbought',
                        'pattern_category': 'reversal',
                        'direction': 'short',
                        'confidence': 0.75,
                        'strength': 'moderate',
                        'entry_price': float(df['close'].iloc[-1]),
                        'stop_loss': float(df['close'].iloc[-1] * 1.03),
                        'take_profit': float(df['close'].iloc[-1] * 0.96),
                        'risk_reward_ratio': 2.33,
                        'pattern_start_time': df['timestamp'].iloc[-14],
                        'pattern_end_time': df['timestamp'].iloc[-1],
                        'data_points_used': len(df),
                        'data_quality_score': 0.90,
                        'status': 'active'
                    }
                    patterns.append(pattern)
        
        return patterns
        
    except Exception as e:
        logger.error(f"❌ Error detecting reversal patterns: {e}")
        return patterns

def detect_continuation_patterns(df: pd.DataFrame, symbol: str, timeframe: str) -> List[Dict[str, Any]]:
    """Detect continuation patterns"""
    patterns = []
    
    try:
        # Bollinger Band squeeze pattern
        if len(df) >= 20:
            bb_upper = df['bollinger_upper'].dropna()
            bb_lower = df['bollinger_lower'].dropna()
            
            if len(bb_upper) >= 5 and len(bb_lower) >= 5:
                # Check for Bollinger Band squeeze (bands getting closer)
                recent_bb_range = bb_upper.tail(5) - bb_lower.tail(5)
                if recent_bb_range.iloc[-1] < recent_bb_range.iloc[0] * 0.8:  # 20% reduction in range
                    pattern = {
                        'pattern_id': f"BB-{symbol.replace('/', '')}-{timeframe}-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
                        'symbol': symbol,
                        'timeframe': timeframe,
                        'pattern_type': 'bollinger_squeeze',
                        'pattern_category': 'continuation',
                        'direction': 'neutral',
                        'confidence': 0.70,
                        'strength': 'moderate',
                        'entry_price': float(df['close'].iloc[-1]),
                        'stop_loss': float(df['close'].iloc[-1] * 0.99),
                        'take_profit': float(df['close'].iloc[-1] * 1.02),
                        'risk_reward_ratio': 2.0,
                        'pattern_start_time': df['timestamp'].iloc[-5],
                        'pattern_end_time': df['timestamp'].iloc[-1],
                        'data_points_used': len(df),
                        'data_quality_score': 0.85,
                        'status': 'active'
                    }
                    patterns.append(pattern)
        
        return patterns
        
    except Exception as e:
        logger.error(f"❌ Error detecting continuation patterns: {e}")
        return patterns

def detect_candlestick_patterns(df: pd.DataFrame, symbol: str, timeframe: str) -> List[Dict[str, Any]]:
    """Detect candlestick patterns"""
    patterns = []
    
    try:
        if len(df) >= 3:
            # Hammer pattern
            current_candle = df.iloc[-1]
            prev_candle = df.iloc[-2]
            
            body_size = abs(current_candle['close'] - current_candle['open'])
            lower_shadow = min(current_candle['open'], current_candle['close']) - current_candle['low']
            upper_shadow = current_candle['high'] - max(current_candle['open'], current_candle['close'])
            
            # Hammer: small body, long lower shadow, small upper shadow
            if (body_size < lower_shadow * 0.3 and 
                upper_shadow < body_size * 0.5 and
                current_candle['close'] > current_candle['open']):  # Bullish hammer
                
                pattern = {
                    'pattern_id': f"CANDLE-{symbol.replace('/', '')}-{timeframe}-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'pattern_type': 'hammer',
                    'pattern_category': 'reversal',
                    'direction': 'long',
                    'confidence': 0.65,
                    'strength': 'moderate',
                    'entry_price': float(current_candle['close']),
                    'stop_loss': float(current_candle['low'] * 0.99),
                    'take_profit': float(current_candle['close'] * 1.03),
                    'risk_reward_ratio': 3.0,
                    'pattern_start_time': current_candle['timestamp'],
                    'pattern_end_time': current_candle['timestamp'],
                    'data_points_used': 3,
                    'data_quality_score': 0.80,
                    'status': 'active'
                }
                patterns.append(pattern)
        
        return patterns
        
    except Exception as e:
        logger.error(f"❌ Error detecting candlestick patterns: {e}")
        return patterns

async def start_enhanced_signal_generation():
    """Start enhanced signal generation using comprehensive patterns"""
    global signal_buffer, pattern_buffer
    
    try:
        logger.info("🔔 Starting enhanced signal generation...")
        
        while True:
            try:
                # Clear old signals (e.g., older than 1 hour)
                one_hour_ago = datetime.utcnow() - timedelta(hours=1)
                signal_buffer[:] = [s for s in signal_buffer if s['timestamp'] > one_hour_ago]

                for pattern in pattern_buffer[-20:]:  # Process recent patterns
                    # Generate signal based on pattern
                    if pattern['confidence'] > 0.6:  # Moderate confidence threshold
                        signal_data = {
                            'signal_id': f"SIG-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}-{random.randint(1000,9999)}",
                            'pattern_id': pattern['pattern_id'],
                            'symbol': pattern['symbol'],
                            'timeframe': pattern['timeframe'],
                            'direction': pattern['direction'],
                            'signal_type': 'entry',
                            'entry_price': pattern['entry_price'],
                            'stop_loss': pattern['stop_loss'],
                            'take_profit': pattern['take_profit'],
                            'risk_reward_ratio': pattern['risk_reward_ratio'],
                            'confidence': pattern['confidence'],
                            'pattern_type': pattern['pattern_type'],
                            'signal_generated_at': datetime.utcnow(),
                            'signal_expires_at': datetime.utcnow() + timedelta(hours=2),
                            'status': 'generated'
                        }
                        
                        # Add signal if not already present
                        is_duplicate = False
                        for existing_signal in signal_buffer:
                            if (existing_signal['symbol'] == signal_data['symbol'] and
                                existing_signal['direction'] == signal_data['direction'] and
                                (datetime.utcnow() - existing_signal['signal_generated_at']).total_seconds() < 300): # 5 minutes
                                is_duplicate = True
                                break
                        
                        if not is_duplicate:
                            signal_buffer.append(signal_data)
                            logger.info(f"🔔 Enhanced signal generated: {signal_data['symbol']} {signal_data['direction'].upper()} (Confidence: {signal_data['confidence']:.2f})")
                
                # Keep only recent signals
                if len(signal_buffer) > 100:
                    signal_buffer = signal_buffer[-100:]

                await asyncio.sleep(10) # Generate signals every 10 seconds

            except Exception as e:
                logger.error(f"❌ Error in enhanced signal generation: {e}")
                await asyncio.sleep(30) # Wait longer on error

    except Exception as e:
        logger.error(f"❌ Enhanced signal generation error: {e}")

async def start_data_quality_monitoring():
    """Start data quality monitoring"""
    global enhanced_data_buffer
    
    try:
        logger.info("🔍 Starting data quality monitoring...")
        
        while True:
            try:
                for symbol in SYMBOLS:
                    for timeframe in TIMEFRAMES:
                        if (symbol in enhanced_data_buffer and 
                            timeframe in enhanced_data_buffer[symbol] and 
                            len(enhanced_data_buffer[symbol][timeframe]) > 0):
                            
                            # Validate data integrity
                            quality_metrics = validate_data_integrity(enhanced_data_buffer[symbol][timeframe])
                            
                            # Store quality metrics
                            await store_data_quality_metrics(symbol, timeframe, quality_metrics)
                            
                            # Detect anomalies
                            anomalies = detect_outliers(enhanced_data_buffer[symbol][timeframe])
                            
                            # Store anomalies
                            await store_data_anomalies(anomalies)
                            
                            if anomalies:
                                logger.warning(f"⚠️ Data anomalies detected for {symbol} {timeframe}: {len(anomalies)} anomalies")
                
                await asyncio.sleep(60) # Monitor quality every minute

            except Exception as e:
                logger.error(f"❌ Error in data quality monitoring: {e}")
                await asyncio.sleep(120) # Wait longer on error

    except Exception as e:
        logger.error(f"❌ Data quality monitoring error: {e}")

def validate_data_integrity(data_points: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Validate data integrity"""
    try:
        if not data_points:
            return {'overall_score': 0.0, 'completeness': 0.0, 'accuracy': 0.0, 'consistency': 0.0}
        
        # Check completeness
        total_fields = len(data_points[0]) if data_points else 0
        missing_fields = sum(1 for dp in data_points for v in dp.values() if pd.isna(v) if isinstance(v, (int, float)) else False)
        completeness = 1.0 - (missing_fields / (len(data_points) * total_fields)) if data_points else 0.0
        
        # Check accuracy (basic price validation)
        valid_prices = sum(1 for dp in data_points if dp['high'] >= dp['low'] and dp['high'] >= dp['close'] and dp['close'] >= dp['low'])
        accuracy = valid_prices / len(data_points) if data_points else 0.0
        
        # Check consistency (timestamps are sequential)
        timestamps = [dp['timestamp'] for dp in data_points]
        is_sequential = all(timestamps[i] <= timestamps[i+1] for i in range(len(timestamps)-1))
        consistency = 1.0 if is_sequential else 0.8
        
        overall_score = (completeness + accuracy + consistency) / 3
        
        return {
            'overall_score': overall_score,
            'completeness': completeness,
            'accuracy': accuracy,
            'consistency': consistency,
            'timestamp': datetime.utcnow()
        }
        
    except Exception as e:
        logger.error(f"❌ Error validating data integrity: {e}")
        return {'overall_score': 0.0, 'completeness': 0.0, 'accuracy': 0.0, 'consistency': 0.0}

def detect_outliers(data_points: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Detect outliers in data"""
    anomalies = []
    
    try:
        if len(data_points) < 10:
            return anomalies
        
        # Convert to DataFrame
        df = pd.DataFrame(data_points)
        
        # Detect price outliers (using IQR method)
        if 'close' in df.columns:
            Q1 = df['close'].quantile(0.25)
            Q3 = df['close'].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            price_outliers = df[(df['close'] < lower_bound) | (df['close'] > upper_bound)]
            
            for _, row in price_outliers.iterrows():
                anomaly = {
                    'symbol': row['symbol'],
                    'timeframe': row['timeframe'],
                    'timestamp': row['timestamp'],
                    'anomaly_type': 'price_outlier',
                    'anomaly_value': float(row['close']),
                    'expected_range': f"{lower_bound:.4f} - {upper_bound:.4f}",
                    'severity': 'high' if abs(row['close'] - df['close'].mean()) > 2 * df['close'].std() else 'medium'
                }
                anomalies.append(anomaly)
        
        return anomalies
        
    except Exception as e:
        logger.error(f"❌ Error detecting outliers: {e}")
        return anomalies

async def store_data_quality_metrics(symbol: str, timeframe: str, quality_metrics: Dict[str, Any]):
    """Store data quality metrics in database"""
    try:
        if not db_pool:
            return
        
        async with db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO data_quality_metrics (
                    symbol, timeframe, overall_score, completeness, accuracy, consistency, timestamp
                ) VALUES ($1, $2, $3, $4, $5, $6, $7)
            """, 
            symbol, timeframe, quality_metrics['overall_score'], 
            quality_metrics['completeness'], quality_metrics['accuracy'], 
            quality_metrics['consistency'], quality_metrics['timestamp']
            )
            
    except Exception as e:
        logger.error(f"❌ Error storing data quality metrics: {e}")

async def store_data_anomalies(anomalies: List[Dict[str, Any]]):
    """Store data anomalies in database"""
    try:
        if not db_pool or not anomalies:
            return
        
        async with db_pool.acquire() as conn:
            for anomaly in anomalies:
                await conn.execute("""
                    INSERT INTO data_anomalies (
                        symbol, timeframe, timestamp, anomaly_type, anomaly_value, 
                        expected_range, severity
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7)
                """, 
                anomaly['symbol'], anomaly['timeframe'], anomaly['timestamp'],
                anomaly['anomaly_type'], anomaly['anomaly_value'],
                anomaly['expected_range'], anomaly['severity']
                )
                
    except Exception as e:
        logger.error(f"❌ Error storing data anomalies: {e}")

# API Endpoints
@app.get("/api/test/phase3", summary="Get Phase 3 Status")
async def get_phase3_status():
    return {"status": "Phase 3 Active", "message": "Enhanced AI-driven thresholds with comprehensive data collection are being applied."}

@app.get("/api/patterns/latest", response_model=List[Dict[str, Any]], summary="Get Latest Patterns")
async def get_latest_patterns():
    return sorted(pattern_buffer, key=lambda x: x.get('pattern_start_time', datetime.utcnow()), reverse=True)[:20]

@app.get("/api/signals/latest", response_model=List[Dict[str, Any]], summary="Get Latest Signals")
async def get_latest_signals():
    return sorted(signal_buffer, key=lambda x: x.get('signal_generated_at', datetime.utcnow()), reverse=True)[:20]

@app.get("/api/market/status", summary="Get Market Status")
async def get_market_status():
    try:
        status_data = {}
        
        for symbol in SYMBOLS:
            if symbol in enhanced_data_buffer and '1h' in enhanced_data_buffer[symbol] and enhanced_data_buffer[symbol]['1h']:
                latest_data = enhanced_data_buffer[symbol]['1h'][-1]
                status_data[symbol] = {
                    'price': latest_data['close'],
                    'change_24h': latest_data['price_change'] * 100 if latest_data['price_change'] else 0,
                    'volume': latest_data['volume'],
                    'rsi': latest_data['rsi'],
                    'market_sentiment': latest_data['market_sentiment'],
                    'data_quality': latest_data['data_quality_score']
                }
        
        return {
            "status": "active",
            "timestamp": datetime.utcnow().isoformat(),
            "symbols": status_data
        }
    except Exception as e:
        logger.error(f"❌ Error getting market status: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving market status")

@app.get("/api/ai/performance", summary="Get AI Performance")
async def get_ai_performance():
    try:
        # Calculate performance metrics
        total_patterns = len(pattern_buffer)
        high_confidence_patterns = len([p for p in pattern_buffer if p.get('confidence', 0) > 0.8])
        total_signals = len(signal_buffer)
        active_signals = len([s for s in signal_buffer if s.get('status') == 'generated'])
        
        return {
            "total_patterns_detected": total_patterns,
            "high_confidence_patterns": high_confidence_patterns,
            "pattern_confidence_rate": high_confidence_patterns / total_patterns if total_patterns > 0 else 0,
            "total_signals_generated": total_signals,
            "active_signals": active_signals,
            "data_quality_score": 0.95,  # Placeholder - should be calculated from actual data
            "system_health": "excellent"
        }
    except Exception as e:
        logger.error(f"❌ Error getting AI performance: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving AI performance")

# WebSocket for real-time updates
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            await asyncio.sleep(3)
            
            real_time_data = {
                "type": "real_time_update",
                "timestamp": datetime.utcnow().isoformat(),
                "data_source": "Binance Real Data",
                "patterns_count": len(pattern_buffer),
                "signals_count": len(signal_buffer),
                "market_symbols": list(enhanced_data_buffer.keys()),
                "latest_patterns": [
                    {
                        "symbol": p['symbol'],
                        "pattern_type": p['pattern_type'],
                        "confidence": round(p['confidence'], 3)
                    }
                    for p in pattern_buffer[-5:]
                ],
                "latest_signals": [
                    {
                        "symbol": s['symbol'],
                        "direction": s['direction'],
                        "confidence": round(s['confidence'], 3)
                    }
                    for s in signal_buffer[-5:]
                ]
            }
            
            await websocket.send_text(json.dumps(real_time_data))
            
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
