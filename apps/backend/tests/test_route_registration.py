#!/usr/bin/env python3
"""
Test route registration during server startup
"""
import asyncio
import uvicorn
from fastapi import FastAPI
from datetime import datetime

app = FastAPI()

@app.get("/health")
async def health():
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}

@app.get("/api/charts/candlestick/{symbol}")
async def get_candlestick_data(symbol: str):
    """Mock candlestick data"""
    return {
        "symbol": symbol,
        "data": [
            {
                "timestamp": datetime.utcnow().isoformat(),
                "open": 50000,
                "high": 51000,
                "low": 49000,
                "close": 50500,
                "volume": 1000
            }
        ]
    }

@app.get("/api/charts/technical-indicators/{symbol}")
async def get_technical_indicators(symbol: str):
    """Mock technical indicators"""
    return {
        "symbol": symbol,
        "indicators": {
            "rsi": 65.5,
            "macd": {"macd": 100, "signal": 95, "histogram": 5},
            "sma": 50200,
            "ema": 50300,
            "bollinger": {"upper": 52000, "middle": 50000, "lower": 48000}
        }
    }

@app.get("/api/charts/real-time/{symbol}")
async def get_real_time_data(symbol: str):
    """Mock real-time data"""
    return {
        "symbol": symbol,
        "latest_price": 50500,
        "timestamp": datetime.utcnow().isoformat()
    }

def print_routes():
    """Print all routes"""
    print("=== Routes ===")
    for route in app.routes:
        if hasattr(route, 'path'):
            print(f"  {route.path}")
    
    print("\n=== Chart Routes ===")
    chart_routes = [route for route in app.routes if hasattr(route, 'path') and 'charts' in route.path]
    for route in chart_routes:
        print(f"  {route.path}")

if __name__ == "__main__":
    print("Before uvicorn.run:")
    print_routes()
    
    print("\nStarting server...")
    uvicorn.run(app, host="0.0.0.0", port=8003)
