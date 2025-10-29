"""
Simple FastAPI backend for AlphaPulse Frontend Demo
Provides mock data for frontend development
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, timezone
import asyncio
import random
from typing import List, Dict

app = FastAPI(title="AlphaPulse API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mock data store
signals_store: List[Dict] = []
active_connections: List[WebSocket] = []

def generate_mock_signal():
    """Generate a mock trading signal"""
    symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "ADAUSDT"]
    directions = ["long", "short"]
    patterns = ["wyckoff_spring", "bullish_engulfing", "rsi_divergence", "harmonic_gartley", "ict_ote"]
    
    symbol = random.choice(symbols)
    direction = random.choice(directions)
    confidence = round(random.uniform(0.65, 0.95), 2)
    entry_price = random.uniform(100, 50000)
    
    return {
        "symbol": symbol,
        "direction": direction,
        "confidence": confidence,
        "pattern_type": random.choice(patterns),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "entry_price": round(entry_price, 2),
        "stop_loss": round(entry_price * 0.95 if direction == "long" else entry_price * 1.05, 2),
        "take_profit": round(entry_price * 1.05 if direction == "long" else entry_price * 0.95, 2),
        "timeframe": "1h",
        "market_regime": "trending"
    }

# Initialize with some signals
for _ in range(5):
    signals_store.append(generate_mock_signal())

@app.get("/")
async def root():
    return {
        "message": "AlphaPulse API",
        "version": "1.0.0",
        "status": "running",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "database": {"status": "healthy"},
        "websocket": {"status": "active"}
    }

@app.get("/api/signals/latest")
async def get_latest_signals():
    return {"signals": signals_store[-10:]}

@app.get("/api/signals/performance")
async def get_signal_performance():
    return {
        "performance": {
            "total_signals": len(signals_store),
            "high_confidence_signals": len([s for s in signals_store if s["confidence"] > 0.8]),
            "avg_confidence": sum(s["confidence"] for s in signals_store) / len(signals_store) if signals_store else 0
        },
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

@app.get("/api/patterns/latest")
async def get_latest_patterns():
    patterns = [
        {
            "symbol": "BTCUSDT",
            "pattern_type": "bullish_engulfing",
            "confidence": 0.85,
            "strength": "strong",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "timeframe": "1h",
            "price_level": 45000
        }
    ]
    return {"patterns": patterns}

@app.get("/api/market/status")
async def get_market_status():
    conditions = ["bullish", "bearish", "neutral"]
    return {
        "market_condition": random.choice(conditions),
        "volatility": round(random.uniform(0.05, 0.25), 3),
        "trend_direction": "upward",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

@app.get("/api/ai/performance")
async def get_ai_performance():
    return {
        "accuracy": 0.78,
        "total_signals": len(signals_store),
        "profitable_signals": int(len(signals_store) * 0.78),
        "average_return": 0.042,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_connections.append(websocket)
    
    try:
        # Send initial connection message
        await websocket.send_json({
            "type": "system_alert",
            "data": {"message": "Connected to AlphaPulse"},
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        
        # Keep connection alive and send periodic updates
        while True:
            await asyncio.sleep(15)
            
            # Generate and broadcast new signal every 15 seconds
            new_signal = generate_mock_signal()
            signals_store.append(new_signal)
            
            # Keep only last 50 signals
            if len(signals_store) > 50:
                signals_store.pop(0)
            
            # Send to all connected clients
            message = {
                "type": "signal",
                "data": new_signal,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            for connection in active_connections:
                try:
                    await connection.send_json(message)
                except:
                    pass
                    
    except WebSocketDisconnect:
        active_connections.remove(websocket)

@app.websocket("/ws/signals")
async def signals_websocket(websocket: WebSocket):
    await websocket.accept()
    active_connections.append(websocket)
    
    try:
        while True:
            await asyncio.sleep(10)
            
            # Send signal updates
            message = {
                "type": "signal",
                "data": generate_mock_signal(),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            await websocket.send_json(message)
            
    except WebSocketDisconnect:
        active_connections.remove(websocket)

if __name__ == "__main__":
    import uvicorn
    print("Starting AlphaPulse Backend...")
    print("API available at: http://localhost:8000")
    print("WebSocket available at: ws://localhost:8000/ws")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

