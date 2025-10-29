#!/usr/bin/env python3
"""
Test Chart Server - Minimal server to test chart endpoints
"""
import asyncio
import logging
import os
from datetime import datetime
from typing import Dict, Any, List
import asyncpg
import ccxt
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="AlphaPulse Chart Test Server",
    description="Test server for chart endpoints",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
exchange = None
db_pool = None

@app.on_event("startup")
async def startup_event():
    """Initialize minimal components"""
    global exchange, db_pool
    
    try:
        logger.info("🚀 Starting Chart Test Server...")
        
        # Initialize exchange
        exchange = ccxt.binance({
            'sandbox': False,
            'enableRateLimit': True,
        })
        logger.info("✅ Exchange connection established")
        
        # Initialize database connection
        db_pool = await asyncpg.create_pool(
            host=os.getenv('DB_HOST', 'localhost'),
            port=int(os.getenv('DB_PORT', 5432)),
            database=os.getenv('DB_NAME', 'alphapulse'),
            user=os.getenv('DB_USER', 'alpha_emon'),
            password=os.getenv('DB_PASSWORD', 'Emon_@17711'),
            min_size=1,
            max_size=5
        )
        logger.info("✅ Database connection established")
        
        logger.info("🎉 Chart Test Server started successfully!")
        
    except Exception as e:
        logger.error(f"❌ Error during startup: {e}")
        # Don't raise - let the server start even if some components fail

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Chart Test Server",
        "database": "connected" if db_pool else "disconnected",
        "exchange": "connected" if exchange else "disconnected",
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/api/charts/candlestick/{symbol}")
async def get_candlestick_data(
    symbol: str,
    timeframe: str = Query("1h", description="Timeframe: 1m, 5m, 15m, 1h, 4h, 1d"),
    limit: int = Query(100, description="Number of candlesticks to return", ge=1, le=1000)
):
    """Get candlestick data for charting"""
    try:
        if not exchange:
            raise HTTPException(status_code=503, detail="Exchange connection not available")
        
        # Validate timeframe
        valid_timeframes = ["1m", "5m", "15m", "1h", "4h", "1d"]
        if timeframe not in valid_timeframes:
            raise HTTPException(status_code=400, detail=f"Invalid timeframe. Must be one of: {valid_timeframes}")
        
        # Fetch OHLCV data from exchange
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        
        if not ohlcv:
            raise HTTPException(status_code=404, detail=f"No candlestick data available for {symbol}")
        
        # Transform data for frontend
        candlestick_data = []
        for candle in ohlcv:
            candlestick_data.append({
                "timestamp": datetime.fromtimestamp(candle[0] / 1000).isoformat(),
                "open": float(candle[1]),
                "high": float(candle[2]),
                "low": float(candle[3]),
                "close": float(candle[4]),
                "volume": float(candle[5])
            })
        
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "data": candlestick_data,
            "count": len(candlestick_data),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting candlestick data for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting candlestick data: {e}")

@app.get("/api/charts/technical-indicators/{symbol}")
async def get_technical_indicators(
    symbol: str,
    timeframe: str = Query("1h", description="Timeframe for analysis"),
    indicators: str = Query("rsi,macd,sma,ema,bollinger", description="Comma-separated list of indicators")
):
    """Get technical indicators for charting"""
    try:
        if not exchange:
            raise HTTPException(status_code=503, detail="Exchange connection not available")
        
        # Parse requested indicators
        requested_indicators = [ind.strip() for ind in indicators.split(",")]
        
        # Get candlestick data first
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=100)
        if not ohlcv:
            raise HTTPException(status_code=404, detail=f"No data available for {symbol}")
        
        # Calculate indicators
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        indicators_data = {}
        
        # Simple RSI calculation
        if 'rsi' in requested_indicators:
            prices = df['close']
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            indicators_data['rsi'] = rsi.fillna(50.0).tolist()
        
        # Simple MACD calculation
        if 'macd' in requested_indicators:
            prices = df['close']
            ema_fast = prices.ewm(span=12).mean()
            ema_slow = prices.ewm(span=26).mean()
            macd_line = ema_fast - ema_slow
            signal_line = macd_line.ewm(span=9).mean()
            histogram = macd_line - signal_line
            
            indicators_data['macd'] = [
                {
                    'macd': float(macd_line.iloc[i]),
                    'signal': float(signal_line.iloc[i]),
                    'histogram': float(histogram.iloc[i])
                }
                for i in range(len(df))
            ]
        
        # Simple SMA calculation
        if 'sma' in requested_indicators:
            prices = df['close']
            sma = prices.rolling(window=20).mean()
            indicators_data['sma'] = sma.fillna(prices.iloc[0]).tolist()
        
        # Simple EMA calculation
        if 'ema' in requested_indicators:
            prices = df['close']
            ema = prices.ewm(span=12).mean()
            indicators_data['ema'] = ema.fillna(prices.iloc[0]).tolist()
        
        # Simple Bollinger Bands calculation
        if 'bollinger' in requested_indicators:
            prices = df['close']
            sma = prices.rolling(window=20).mean()
            std = prices.rolling(window=20).std()
            upper_band = sma + (std * 2)
            lower_band = sma - (std * 2)
            
            indicators_data['bollinger'] = [
                {
                    'upper': float(upper_band.iloc[i]),
                    'middle': float(sma.iloc[i]),
                    'lower': float(lower_band.iloc[i])
                }
                for i in range(len(df))
            ]
        
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "indicators": indicators_data,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting technical indicators for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting technical indicators: {e}")

@app.get("/api/charts/real-time/{symbol}")
async def get_real_time_chart_data(
    symbol: str,
    timeframe: str = Query("1h", description="Timeframe for real-time data")
):
    """Get real-time chart data including candlesticks and indicators"""
    try:
        if not exchange:
            raise HTTPException(status_code=503, detail="Exchange connection not available")
        
        # Get latest candlestick data
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=100)
        if not ohlcv:
            raise HTTPException(status_code=404, detail=f"No data available for {symbol}")
        
        # Transform candlestick data
        candlestick_data = []
        for candle in ohlcv:
            candlestick_data.append({
                "timestamp": datetime.fromtimestamp(candle[0] / 1000).isoformat(),
                "open": float(candle[1]),
                "high": float(candle[2]),
                "low": float(candle[3]),
                "close": float(candle[4]),
                "volume": float(candle[5])
            })
        
        # Calculate technical indicators
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # Get latest values for indicators
        latest_indicators = {}
        
        # RSI
        prices = df['close']
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        if len(rsi.dropna()) > 0:
            latest_indicators['rsi'] = float(rsi.iloc[-1])
        
        # MACD
        ema_fast = prices.ewm(span=12).mean()
        ema_slow = prices.ewm(span=26).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=9).mean()
        histogram = macd_line - signal_line
        latest_indicators['macd'] = {
            'macd': float(macd_line.iloc[-1]),
            'signal': float(signal_line.iloc[-1]),
            'histogram': float(histogram.iloc[-1])
        }
        
        # SMA
        sma = prices.rolling(window=20).mean()
        if len(sma.dropna()) > 0:
            latest_indicators['sma_20'] = float(sma.iloc[-1])
        
        # EMA
        ema = prices.ewm(span=12).mean()
        latest_indicators['ema_12'] = float(ema.iloc[-1])
        
        # Bollinger Bands
        sma = prices.rolling(window=20).mean()
        std = prices.rolling(window=20).std()
        upper_band = sma + (std * 2)
        lower_band = sma - (std * 2)
        latest_indicators['bollinger'] = {
            'upper': float(upper_band.iloc[-1]),
            'middle': float(sma.iloc[-1]),
            'lower': float(lower_band.iloc[-1])
        }
        
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "candlestick_data": candlestick_data,
            "indicators": latest_indicators,
            "latest_price": float(ohlcv[-1][4]) if ohlcv else None,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting real-time chart data for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting real-time chart data: {e}")

if __name__ == "__main__":
    uvicorn.run(
        "test_chart_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
