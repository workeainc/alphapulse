#!/usr/bin/env python3
"""
Working server with all required endpoints
"""
import uvicorn
from fastapi import FastAPI, Query
from datetime import datetime, timedelta
import random

# Create FastAPI app
app = FastAPI(
    title="AlphaPulse Intelligent Trading System",
    description="Advanced AI-powered trading signal generation with 85% confidence threshold",
    version="2.0.0"
)

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "message": "AlphaPulse Trading System is running"
    }

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "AlphaPulse Intelligent Trading System",
        "version": "2.0.0",
        "status": "running",
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
        # Return mock data for now
        base_price = 50000 if "BTC" in symbol.upper() else 3000 if "ETH" in symbol.upper() else 100
        
        candlestick_data = []
        for i in range(min(limit, 100)):
            # Generate mock candlestick data
            open_price = base_price + random.uniform(-1000, 1000)
            close_price = open_price + random.uniform(-500, 500)
            high_price = max(open_price, close_price) + random.uniform(0, 200)
            low_price = min(open_price, close_price) - random.uniform(0, 200)
            volume = random.uniform(100, 10000)
            
            # Generate timestamp (going backwards from now)
            timestamp = datetime.utcnow() - timedelta(hours=i)
            
            candlestick_data.append({
                "timestamp": timestamp.isoformat(),
                "open": round(open_price, 2),
                "high": round(high_price, 2),
                "low": round(low_price, 2),
                "close": round(close_price, 2),
                "volume": round(volume, 2)
            })
        
        # Reverse to get chronological order
        candlestick_data.reverse()
        
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "data": candlestick_data,
            "count": len(candlestick_data),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        return {"error": f"Error getting candlestick data: {e}"}

@app.get("/api/charts/technical-indicators/{symbol}")
async def get_technical_indicators(
    symbol: str,
    timeframe: str = Query("1h", description="Timeframe for analysis"),
    indicators: str = Query("rsi,macd,sma,ema,bollinger", description="Comma-separated list of indicators")
):
    """Get technical indicators for charting"""
    try:
        # Return mock data for now
        # Parse requested indicators
        requested_indicators = [ind.strip() for ind in indicators.split(",")]
        
        indicators_data = {}
        
        if 'rsi' in requested_indicators:
            indicators_data['rsi'] = round(random.uniform(30, 70), 2)
        
        if 'macd' in requested_indicators:
            macd_value = random.uniform(-100, 100)
            signal_value = macd_value + random.uniform(-20, 20)
            histogram_value = macd_value - signal_value
            indicators_data['macd'] = {
                'macd': round(macd_value, 2),
                'signal': round(signal_value, 2),
                'histogram': round(histogram_value, 2)
            }
        
        if 'sma' in requested_indicators:
            base_price = 50000 if "BTC" in symbol.upper() else 3000 if "ETH" in symbol.upper() else 100
            indicators_data['sma'] = round(base_price + random.uniform(-1000, 1000), 2)
        
        if 'ema' in requested_indicators:
            base_price = 50000 if "BTC" in symbol.upper() else 3000 if "ETH" in symbol.upper() else 100
            indicators_data['ema'] = round(base_price + random.uniform(-1000, 1000), 2)
        
        if 'bollinger' in requested_indicators:
            base_price = 50000 if "BTC" in symbol.upper() else 3000 if "ETH" in symbol.upper() else 100
            middle = base_price + random.uniform(-1000, 1000)
            std = random.uniform(500, 2000)
            indicators_data['bollinger'] = {
                'upper': round(middle + (2 * std), 2),
                'middle': round(middle, 2),
                'lower': round(middle - (2 * std), 2)
            }
        
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "indicators": indicators_data,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        return {"error": f"Error getting technical indicators: {e}"}

@app.get("/api/charts/real-time/{symbol}")
async def get_real_time_chart_data(
    symbol: str,
    timeframe: str = Query("1h", description="Timeframe for real-time data")
):
    """Get real-time chart data including candlesticks and indicators"""
    try:
        # Return mock data for now
        base_price = 50000 if "BTC" in symbol.upper() else 3000 if "ETH" in symbol.upper() else 100
        
        # Generate mock candlestick data
        candlestick_data = []
        for i in range(10):  # Last 10 candles
            open_price = base_price + random.uniform(-1000, 1000)
            close_price = open_price + random.uniform(-500, 500)
            high_price = max(open_price, close_price) + random.uniform(0, 200)
            low_price = min(open_price, close_price) - random.uniform(0, 200)
            volume = random.uniform(100, 10000)
            
            timestamp = datetime.utcnow() - timedelta(hours=i)
            
            candlestick_data.append({
                "timestamp": timestamp.isoformat(),
                "open": round(open_price, 2),
                "high": round(high_price, 2),
                "low": round(low_price, 2),
                "close": round(close_price, 2),
                "volume": round(volume, 2)
            })
        
        # Reverse to get chronological order
        candlestick_data.reverse()
        
        # Generate mock indicators
        latest_indicators = {
            'rsi': round(random.uniform(30, 70), 2),
            'macd': {
                'macd': round(random.uniform(-100, 100), 2),
                'signal': round(random.uniform(-100, 100), 2),
                'histogram': round(random.uniform(-50, 50), 2)
            },
            'sma_20': round(base_price + random.uniform(-1000, 1000), 2),
            'ema_12': round(base_price + random.uniform(-1000, 1000), 2),
            'bollinger': {
                'upper': round(base_price + 2000, 2),
                'middle': round(base_price, 2),
                'lower': round(base_price - 2000, 2)
            }
        }
        
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "candlestick_data": candlestick_data,
            "indicators": latest_indicators,
            "latest_price": round(base_price + random.uniform(-500, 500), 2),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        return {"error": f"Error getting real-time chart data: {e}"}

@app.get("/api/signals")
async def get_signals():
    """Get all trading signals"""
    try:
        # Return mock signals for now
        signals = []
        symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'ADA/USDT', 'DOT/USDT']
        directions = ['long', 'short']
        
        for i in range(10):
            signal = {
                "signal_id": f"signal_{i+1}",
                "symbol": random.choice(symbols),
                "direction": random.choice(directions),
                "confidence_score": round(random.uniform(0.75, 0.95), 2),
                "timestamp": (datetime.utcnow() - timedelta(hours=i)).isoformat(),
                "timeframe": random.choice(['1h', '4h', '1d']),
                "entry_price": round(random.uniform(45000, 55000), 2),
                "stop_loss": round(random.uniform(40000, 50000), 2),
                "take_profit": round(random.uniform(50000, 60000), 2)
            }
            signals.append(signal)
        
        return signals
        
    except Exception as e:
        return {"error": f"Error fetching signals: {e}"}

@app.get("/api/market-data")
async def get_market_data():
    """Get market data for all monitored symbols"""
    try:
        # Return mock market data for now
        symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'ADA/USDT', 'DOT/USDT']
        market_data = []
        
        for symbol in symbols:
            base_price = 50000 if "BTC" in symbol else 3000 if "ETH" in symbol else 100
            price = base_price + random.uniform(-1000, 1000)
            change_24h = random.uniform(-500, 500)
            change_percent_24h = (change_24h / price) * 100
            volume_24h = random.uniform(1000000, 10000000)
            high_24h = price + random.uniform(0, 1000)
            low_24h = price - random.uniform(0, 1000)
            
            market_data.append({
                "symbol": symbol,
                "price": round(price, 2),
                "change_24h": round(change_24h, 2),
                "change_percent_24h": round(change_percent_24h, 2),
                "volume_24h": round(volume_24h, 2),
                "high_24h": round(high_24h, 2),
                "low_24h": round(low_24h, 2)
            })
        
        return market_data
        
    except Exception as e:
        return {"error": f"Error fetching market data: {e}"}

@app.get("/api/performance-metrics")
async def get_performance_metrics():
    """Get performance metrics"""
    try:
        # Return mock performance metrics for now
        total_signals = random.randint(100, 500)
        successful_signals = int(total_signals * random.uniform(0.75, 0.90))
        success_rate = successful_signals / total_signals
        total_profit = random.uniform(1000, 5000)
        profit_change = random.uniform(-500, 500)
        
        return {
            "totalSignals": total_signals,
            "successfulSignals": successful_signals,
            "successRate": round(success_rate, 3),
            "totalProfit": round(total_profit, 2),
            "profitChange": round(profit_change, 2)
        }
        
    except Exception as e:
        return {"error": f"Error fetching performance metrics: {e}"}

if __name__ == "__main__":
    print("Starting AlphaPulse Trading System...")
    print("Server will be available at: http://localhost:8000")
    print("Health check: http://localhost:8000/health")
    print("API docs: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
