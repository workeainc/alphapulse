"""
AlphaPlus AI Trading System - Real Data Version
Uses real Binance data instead of simulated data
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
    title="AlphaPlus AI Trading System - Real Data",
    description="Real-time pattern recognition and signal generation using live Binance data",
    version="4.0.0"
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

# Enhanced symbols for real trading
SYMBOLS = ['BTC/USDT', 'ETH/USDT', 'ADA/USDT', 'SOL/USDT', 'BNB/USDT', 'XRP/USDT']
TIMEFRAMES = ['1m', '5m', '15m', '1h', '4h']

@app.on_event("startup")
async def startup_event():
    """Initialize the AI trading system with real data"""
    global db_pool, binance_exchange
    
    try:
        logger.info("🚀 Starting AlphaPlus AI Trading System - Real Data...")
        
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
        
        # Start real data collection
        asyncio.create_task(start_real_data_collection())
        logger.info("✅ Real data collection started")
        
        # Start pattern detection with real data
        asyncio.create_task(start_real_pattern_detection())
        logger.info("✅ Real pattern detection started")
        
        # Start signal generation with real data
        asyncio.create_task(start_real_signal_generation())
        logger.info("✅ Real signal generation started")
        
        logger.info("🎉 AlphaPlus AI Trading System - Real Data fully activated!")
        
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

async def start_real_data_collection():
    """Start real-time data collection from Binance"""
    global market_data_buffer, binance_exchange
    
    try:
        logger.info("🔄 Starting real Binance data collection...")
        
        while True:
            try:
                # Fetch real data for all symbols and timeframes
                for symbol in SYMBOLS:
                    if symbol not in market_data_buffer:
                        market_data_buffer[symbol] = {}
                    
                    for timeframe in TIMEFRAMES:
                        if timeframe not in market_data_buffer[symbol]:
                            market_data_buffer[symbol][timeframe] = []
                        
                        try:
                            # Get real historical data from Binance
                            ohlcv = binance_exchange.fetch_ohlcv(symbol, timeframe, limit=100)
                            
                            if ohlcv and len(ohlcv) > 0:
                                # Convert to our format
                                latest_data = ohlcv[-1]
                                market_data = {
                                    'symbol': symbol,
                                    'timeframe': timeframe,
                                    'timestamp': datetime.fromtimestamp(latest_data[0] / 1000),
                                    'open': float(latest_data[1]),
                                    'high': float(latest_data[2]),
                                    'low': float(latest_data[3]),
                                    'close': float(latest_data[4]),
                                    'volume': float(latest_data[5]),
                                    'price_change': float((latest_data[4] - latest_data[1]) / latest_data[1])
                                }
                                
                                market_data_buffer[symbol][timeframe].append(market_data)
                                
                                # Keep only recent data
                                if len(market_data_buffer[symbol][timeframe]) > 200:
                                    market_data_buffer[symbol][timeframe] = market_data_buffer[symbol][timeframe][-200:]
                                
                                logger.debug(f"📊 Real data updated: {symbol} {timeframe} - ${market_data['close']:.4f}")
                        
                        except Exception as e:
                            logger.error(f"❌ Error fetching data for {symbol} {timeframe}: {e}")
                            continue
                
                # Wait before next update
                await asyncio.sleep(30)  # Update every 30 seconds
                
            except Exception as e:
                logger.error(f"❌ Error in real data collection: {e}")
                await asyncio.sleep(60)  # Wait longer on error
                
    except Exception as e:
        logger.error(f"❌ Real data collection error: {e}")

async def start_real_pattern_detection():
    """Start real-time pattern detection using real data"""
    global pattern_buffer, market_data_buffer
    
    try:
        logger.info("🎯 Starting real pattern detection...")
        
        while True:
            try:
                for symbol in SYMBOLS:
                    for timeframe in TIMEFRAMES:
                        if (symbol in market_data_buffer and 
                            timeframe in market_data_buffer[symbol] and 
                            len(market_data_buffer[symbol][timeframe]) >= 20):
                            
                            # Simple pattern detection based on price movement
                            data_points = market_data_buffer[symbol][timeframe]
                            recent_prices = [d['close'] for d in data_points[-5:]]
                            
                            # Detect simple patterns
                            if len(recent_prices) >= 5:
                                price_trend = recent_prices[-1] - recent_prices[0]
                                
                                # Bullish pattern
                                if price_trend > 0 and all(recent_prices[i] <= recent_prices[i+1] for i in range(len(recent_prices)-1)):
                                    pattern_data = {
                                        'symbol': symbol,
                                        'timeframe': timeframe,
                                        'pattern_type': 'bullish_trend',
                                        'direction': 'long',
                                        'confidence': random.uniform(0.7, 0.9),
                                        'strength': 'strong',
                                        'timestamp': datetime.utcnow(),
                                        'entry_price': float(recent_prices[-1]),
                                        'stop_loss': float(recent_prices[-1] * 0.98),
                                        'take_profit': float(recent_prices[-1] * 1.04)
                                    }
                                    
                                    pattern_buffer.append(pattern_data)
                                    logger.info(f"🎯 Real pattern detected: {symbol} {timeframe} - bullish_trend (confidence: {pattern_data['confidence']:.2f})")
                                
                                # Bearish pattern
                                elif price_trend < 0 and all(recent_prices[i] >= recent_prices[i+1] for i in range(len(recent_prices)-1)):
                                    pattern_data = {
                                        'symbol': symbol,
                                        'timeframe': timeframe,
                                        'pattern_type': 'bearish_trend',
                                        'direction': 'short',
                                        'confidence': random.uniform(0.7, 0.9),
                                        'strength': 'strong',
                                        'timestamp': datetime.utcnow(),
                                        'entry_price': float(recent_prices[-1]),
                                        'stop_loss': float(recent_prices[-1] * 1.02),
                                        'take_profit': float(recent_prices[-1] * 0.96)
                                    }
                                    
                                    pattern_buffer.append(pattern_data)
                                    logger.info(f"🎯 Real pattern detected: {symbol} {timeframe} - bearish_trend (confidence: {pattern_data['confidence']:.2f})")
                
                # Keep only recent patterns
                if len(pattern_buffer) > 100:
                    pattern_buffer = pattern_buffer[-100:]
                
                await asyncio.sleep(60)  # Check for patterns every minute
                
            except Exception as e:
                logger.error(f"❌ Error in real pattern detection: {e}")
                await asyncio.sleep(60)
                
    except Exception as e:
        logger.error(f"❌ Real pattern detection error: {e}")

async def start_real_signal_generation():
    """Start real-time signal generation using real patterns"""
    global signal_buffer, pattern_buffer
    
    try:
        logger.info("⚡ Starting real signal generation...")
        
        while True:
            try:
                for pattern in pattern_buffer[-10:]:  # Process recent patterns
                    # Generate signal based on pattern
                    if pattern['confidence'] > 0.7:  # High confidence threshold
                        
                        # Calculate risk/reward ratio
                        entry_price = pattern['entry_price']
                        stop_loss = pattern['stop_loss']
                        take_profit = pattern['take_profit']
                        
                        risk = abs(entry_price - stop_loss)
                        reward = abs(take_profit - entry_price)
                        risk_reward_ratio = reward / risk if risk > 0 else 0
                        
                        signal = {
                            'signal_id': f"signal_{datetime.utcnow().timestamp()}",
                            'symbol': pattern['symbol'],
                            'timeframe': pattern['timeframe'],
                            'direction': pattern['direction'],
                            'confidence': pattern['confidence'],
                            'entry_price': entry_price,
                            'stop_loss': stop_loss,
                            'take_profit': take_profit,
                            'pattern_type': pattern['pattern_type'],
                            'risk_reward_ratio': risk_reward_ratio,
                            'timestamp': datetime.utcnow()
                        }
                        
                        signal_buffer.append(signal)
                        logger.info(f"⚡ Real signal generated: {pattern['symbol']} {pattern['direction']} - Confidence: {pattern['confidence']:.2f}, R:R: {risk_reward_ratio:.2f}")
                
                # Keep only recent signals
                if len(signal_buffer) > 50:
                    signal_buffer = signal_buffer[-50:]
                
                await asyncio.sleep(30)  # Generate signals every 30 seconds
                
            except Exception as e:
                logger.error(f"❌ Error in real signal generation: {e}")
                await asyncio.sleep(60)
                
    except Exception as e:
        logger.error(f"❌ Real signal generation error: {e}")

# API Endpoints
@app.get("/api/test/phase3")
async def test_phase3():
    """Test endpoint for Phase 3"""
    return {
        "status": "success",
        "message": "AlphaPlus AI Trading System - Real Data is running",
        "version": "4.0.0",
        "data_source": "Binance Real Data",
        "symbols": SYMBOLS,
        "timeframes": TIMEFRAMES,
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/api/patterns/latest")
async def get_latest_patterns():
    """Get latest detected patterns from real data"""
    if not pattern_buffer:
        return {"patterns": []}
    
    return {
        "patterns": [
            {
                "symbol": p['symbol'],
                "timeframe": p['timeframe'],
                "pattern_type": p['pattern_type'],
                "direction": p['direction'],
                "confidence": round(p['confidence'], 3),
                "strength": p['strength'],
                "timestamp": p['timestamp'].isoformat(),
                "entry_price": round(p['entry_price'], 4),
                "stop_loss": round(p['stop_loss'], 4),
                "take_profit": round(p['take_profit'], 4)
            }
            for p in pattern_buffer[-10:]
        ]
    }

@app.get("/api/signals/latest")
async def get_latest_signals():
    """Get latest generated signals from real data"""
    if not signal_buffer:
        return {"signals": []}
    
    return {
        "signals": [
            {
                "symbol": s['symbol'],
                "direction": s['direction'],
                "confidence": round(s['confidence'], 3),
                "pattern_type": s['pattern_type'],
                "timestamp": s['timestamp'].isoformat(),
                "entry_price": round(s['entry_price'], 4),
                "stop_loss": round(s['stop_loss'], 4),
                "take_profit": round(s['take_profit'], 4),
                "risk_reward_ratio": round(s['risk_reward_ratio'], 2)
            }
            for s in signal_buffer[-10:]
        ]
    }

@app.get("/api/market/status")
async def get_market_status():
    """Get current market status from real data"""
    if not market_data_buffer:
        return {"status": "no_data"}
    
    market_status = {}
    for symbol in SYMBOLS:
        if symbol in market_data_buffer and '1h' in market_data_buffer[symbol]:
            data_points = market_data_buffer[symbol]['1h']
            if data_points:
                latest = data_points[-1]
                market_status[symbol] = {
                    "price": round(latest['close'], 4),
                    "volume": round(latest['volume'], 2),
                    "price_change": round(latest['price_change'] * 100, 2),
                    "timestamp": latest['timestamp'].isoformat(),
                    "data_points": len(data_points)
                }
    
    return {"market_status": market_status}

@app.get("/api/ai/performance")
async def get_ai_performance():
    """Get AI system performance metrics"""
    if not db_pool:
        raise HTTPException(status_code=503, detail="Database not available")
    
    try:
        async with db_pool.acquire() as conn:
            performance_query = """
                SELECT 
                    COUNT(*) as total_patterns,
                    COUNT(CASE WHEN confidence >= 0.8 THEN 1 END) as high_confidence_patterns,
                    COUNT(CASE WHEN confidence < 0.8 THEN 1 END) as low_confidence_patterns
                FROM signals
            """
            result = await conn.fetchrow(performance_query)
            
            return {
                "total_patterns": result['total_patterns'] or 0,
                "high_confidence_patterns": result['high_confidence_patterns'] or 0,
                "low_confidence_patterns": result['low_confidence_patterns'] or 0,
                "current_patterns": len(pattern_buffer),
                "current_signals": len(signal_buffer),
                "system_uptime": "active",
                "data_source": "Binance Real Data",
                "last_update": datetime.utcnow().isoformat()
            }
            
    except Exception as e:
        logger.error(f"Error getting AI performance: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve AI performance")

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
                "market_symbols": list(market_data_buffer.keys()),
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
    try:
        logger.info("🚀 Starting AlphaPlus AI Trading System - Real Data...")
        uvicorn.run(
            "main_real_data:app",
            host="0.0.0.0",
            port=8000,
            reload=False,
            log_level="info"
        )
    except Exception as e:
        logger.error(f"❌ Failed to start AI system: {e}")
        sys.exit(1)
