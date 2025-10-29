"""
AlphaPulse Production Backend
Real-time signal generation with SDE consensus and entry proximity validation
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import logging
import json
from datetime import datetime
from typing import List, Dict, Set
from src.streaming.live_market_connector import LiveMarketConnector
from src.indicators.realtime_calculator import RealtimeIndicatorCalculator
from src.services.live_signal_generator import LiveSignalGenerator
from src.services.signal_lifecycle_manager import SignalLifecycleManager

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(title="AlphaPulse Live Signal API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database config
DB_CONFIG = {
    'host': 'localhost',
    'port': 55433,
    'database': 'alphapulse',
    'user': 'alpha_emon',
    'password': 'Emon_@17711'
}

# Trading config
SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "ADAUSDT",
           "XRPUSDT", "DOTUSDT", "AVAXUSDT", "MATICUSDT", "LINKUSDT"]
TIMEFRAMES = ["1h", "4h"]

# Global components
market_connector: LiveMarketConnector = None
indicator_calculator: RealtimeIndicatorCalculator = None
signal_generator: LiveSignalGenerator = None
lifecycle_manager: SignalLifecycleManager = None
active_connections: List[WebSocket] = []
signal_cache: Dict[str, Dict] = {}  # Cache active signals

# Track which symbols have signals
symbols_with_signals: Set[str] = set()

async def on_new_candle(candle_data: Dict):
    """Callback when new candle closes"""
    global signal_cache, symbols_with_signals
    
    symbol = candle_data['symbol']
    timeframe = candle_data['timeframe']
    
    logger.info(f"New candle closed: {symbol} {timeframe} @ {candle_data['close']}")
    
    # Add to indicator buffer
    indicator_calculator.add_candle(symbol, timeframe, candle_data)
    
    # Calculate indicators
    indicators = indicator_calculator.calculate_all_indicators(symbol, timeframe)
    
    if not indicators:
        return  # Not enough data yet
    
    # Check if we already have a signal for this symbol
    if symbol in symbols_with_signals:
        logger.info(f"Signal already exists for {symbol}, skipping (deduplication)")
        return
    
    # Generate signal
    signal = await signal_generator.generate_signal_from_candle(candle_data, indicators)
    
    if signal:
        # Store in database
        await signal_generator.store_signal(signal)
        
        # Add to cache
        signal_cache[symbol] = signal
        symbols_with_signals.add(symbol)
        
        # Broadcast to all connected clients
        message = {
            "type": "new_signal",
            "data": signal,
            "timestamp": datetime.now().isoformat()
        }
        
        for ws in active_connections:
            try:
                await ws.send_json(message)
            except:
                pass
        
        logger.info(f"NEW SIGNAL: {symbol} {signal['direction'].upper()} @ {signal['confidence']} (SDE: {signal['agreeing_heads']}/9)")

@app.on_event("startup")
async def startup():
    """Initialize all components"""
    global market_connector, indicator_calculator, signal_generator, lifecycle_manager
    
    logger.info("=" * 70)
    logger.info("AlphaPulse Production Backend Starting...")
    logger.info("=" * 70)
    
    try:
        # Initialize components
        indicator_calculator = RealtimeIndicatorCalculator(buffer_size=200)
        signal_generator = LiveSignalGenerator(DB_CONFIG)
        lifecycle_manager = SignalLifecycleManager(DB_CONFIG)
        
        await signal_generator.initialize()
        await lifecycle_manager.initialize()
        
        logger.info("Components initialized")
        
        # Start market data connector
        market_connector = LiveMarketConnector(SYMBOLS, TIMEFRAMES)
        market_connector.add_callback(on_new_candle)
        
        # Start lifecycle monitoring
        asyncio.create_task(lifecycle_manager.start_monitoring())
        asyncio.create_task(price_monitor())
        
        # Start market connector
        asyncio.create_task(market_connector.start())
        
        logger.info("=" * 70)
        logger.info(f"Monitoring {len(SYMBOLS)} symbols on {len(TIMEFRAMES)} timeframes")
        logger.info("Features: Real-Time SDE | Entry Proximity | Signal Persistence")
        logger.info("=" * 70)
        
    except Exception as e:
        logger.error(f"Startup error: {e}")

async def price_monitor():
    """Monitor current prices and update lifecycle manager"""
    while True:
        try:
            if market_connector:
                for symbol in SYMBOLS:
                    price = market_connector.get_current_price(symbol)
                    if price:
                        lifecycle_manager.update_current_price(symbol, price)
            
            await asyncio.sleep(1)  # Update every second
        except Exception as e:
            logger.error(f"Price monitor error: {e}")
            await asyncio.sleep(5)

@app.get("/")
async def root():
    return {
        "message": "AlphaPulse Live Signal API",
        "version": "1.0.0",
        "mode": "production",
        "features": [
            "Real-time signal generation",
            "9-head SDE consensus",
            "Entry proximity validation",
            "Signal persistence",
            "One signal per symbol"
        ],
        "symbols": SYMBOLS,
        "timeframes": TIMEFRAMES,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health():
    active_signals = await lifecycle_manager.get_active_signals() if lifecycle_manager else []
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "database": {
            "status": "connected",
            "active_signals": len(active_signals)
        },
        "websocket": {
            "status": "active",
            "connections": len(active_connections),
            "market_connector": "running" if market_connector and market_connector.is_running else "stopped"
        },
        "current_prices": {symbol: price for symbol, price in market_connector.current_prices.items()} if market_connector else {}
    }

@app.get("/api/signals/active")
async def get_active_signals():
    """Get only active signals (entry proximity validated)"""
    
    if not lifecycle_manager:
        return {"signals": []}
    
    raw_signals = await lifecycle_manager.get_active_signals()
    
    # Format signals for frontend
    formatted_signals = []
    for sig in raw_signals:
        formatted_signals.append({
            'symbol': sig['symbol'],
            'direction': sig['direction'],
            'confidence': float(sig['confidence']),
            'pattern_type': sig['pattern_type'],
            'timestamp': sig['created_at'].isoformat(),
            'entry_price': float(sig['entry_price']),
            'stop_loss': float(sig['stop_loss']),
            'take_profit': float(sig['take_profit']),
            'timeframe': sig['timeframe'],
            'quality_score': float(sig['quality_score']),
            'entry_proximity_status': sig['entry_proximity_status'],
            'sde_consensus': json.loads(sig['sde_consensus']) if isinstance(sig['sde_consensus'], str) else sig['sde_consensus'],
            'mtf_analysis': json.loads(sig['mtf_analysis']) if isinstance(sig['mtf_analysis'], str) else sig['mtf_analysis'],
            'agreeing_heads': sig['agreeing_heads']
        })
    
    return {"signals": formatted_signals}

@app.get("/api/signals/latest")
async def get_latest_signals():
    """Alias for active signals (for frontend compatibility)"""
    return await get_active_signals()

@app.get("/api/market/status")
async def get_market_status():
    """Get current market status"""
    
    signals = await lifecycle_manager.get_active_signals() if lifecycle_manager else []
    
    long_count = len([s for s in signals if s['direction'] == 'long'])
    short_count = len([s for s in signals if s['direction'] == 'short'])
    
    if long_count > short_count:
        condition = "bullish"
    elif short_count > long_count:
        condition = "bearish"
    else:
        condition = "neutral"
    
    return {
        "market_condition": condition,
        "active_signals": len(signals),
        "long_signals": long_count,
        "short_signals": short_count,
        "timestamp": datetime.now().isoformat()
    }

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_connections.append(websocket)
    
    try:
        # Send initial active signals
        signals = await lifecycle_manager.get_active_signals() if lifecycle_manager else []
        
        await websocket.send_json({
            "type": "initial_signals",
            "data": signals,
            "timestamp": datetime.now().isoformat()
        })
        
        # Keep connection alive
        while True:
            # Wait for messages (just to keep connection alive)
            try:
                message = await asyncio.wait_for(websocket.receive_text(), timeout=30)
            except asyncio.TimeoutError:
                # Send ping
                await websocket.send_json({
                    "type": "ping",
                    "timestamp": datetime.now().isoformat()
                })
                
    except WebSocketDisconnect:
        if websocket in active_connections:
            active_connections.remove(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        if websocket in active_connections:
            active_connections.remove(websocket)

if __name__ == "__main__":
    import uvicorn
    print("\n" + "=" * 70)
    print("AlphaPulse Production Backend")
    print("=" * 70)
    print("Mode: Real-Time Signal Generation")
    print("Database: TimescaleDB")
    print("Symbols: 10 major cryptocurrencies")
    print("Signal Logic: Live SDE + Entry Proximity + Lifecycle Management")
    print("=" * 70 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

