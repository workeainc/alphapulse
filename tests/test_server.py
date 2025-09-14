#!/usr/bin/env python3
"""
Simple test server to verify endpoints
"""
from fastapi import FastAPI
from datetime import datetime

app = FastAPI()

@app.get("/test")
async def test():
    return {"message": "Test endpoint working", "timestamp": datetime.utcnow().isoformat()}

@app.get("/api/charts/candlestick/{symbol}")
async def get_candlestick_data(symbol: str):
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
