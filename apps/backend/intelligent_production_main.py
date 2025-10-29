"""
Intelligent Production Backend
Integrates ALL existing components with strict quality gates
Adaptive, intelligent, professional signal generation
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import logging
import json
import uuid
from datetime import datetime
from typing import List, Dict
import asyncpg

# Import existing components
from src.core.adaptive_intelligence_coordinator import AdaptiveIntelligenceCoordinator
from src.core.adaptive_timeframe_selector import AdaptiveTimeframeSelector
from src.core.regime_based_signal_limiter import RegimeBasedSignalLimiter
from src.core.signal_aggregation_window import SignalAggregationWindow
from src.strategies.confluence_entry_finder import ConfluenceEntryFinder
from src.validators.historical_performance_validator import HistoricalPerformanceValidator
from src.streaming.mtf_data_manager import MTFDataManager
from src.streaming.live_market_connector import LiveMarketConnector
from src.indicators.realtime_calculator import RealtimeIndicatorCalculator

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# FastAPI
app = FastAPI(title="AlphaPulse Intelligent Production API")

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

# Symbols to monitor
SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "ADAUSDT"]
BASE_TIMEFRAME = '1m'  # Base data stream

# Global components
db_pool = None
adaptive_coordinator: AdaptiveIntelligenceCoordinator = None
mtf_data_manager: MTFDataManager = None
market_connector: LiveMarketConnector = None
indicator_calculator: RealtimeIndicatorCalculator = None

# Quality control components
timeframe_selector: AdaptiveTimeframeSelector = None
confluence_finder: ConfluenceEntryFinder = None
performance_validator: HistoricalPerformanceValidator = None
regime_limiter: RegimeBasedSignalLimiter = None
aggregation_window: SignalAggregationWindow = None

# WebSocket connections
active_connections: List[WebSocket] = []

# Statistics
stats = {
    'scans_performed': 0,
    'signals_generated': 0,
    'rejection_reasons': {},
    'quality_gates_passed': 0
}

async def on_candle_complete(symbol: str, timeframe: str, candle: Dict):
    """
    Callback when any timeframe candle completes
    Main entry point for signal generation pipeline
    """
    
    global stats
    stats['scans_performed'] += 1
    
    try:
        logger.info(f"Candle closed: {symbol} {timeframe} @ {candle['close']}")
        
        # Calculate ALL 69 indicators (returns dict for compatibility)
        indicators_dict = indicator_calculator.calculate_all_indicators(symbol, timeframe)
        
        if not indicators_dict:
            logger.debug(f"{symbol}: Not enough data for indicators yet")
            return
        
        # Get dataframe with ALL 69 indicator columns for aggregator
        df_with_indicators = indicator_calculator.get_dataframe_with_indicators(symbol, timeframe)
        
        if df_with_indicators is None:
            logger.debug(f"{symbol}: Not enough data for full indicator calculation")
            # Fall back to dict only
            df_with_indicators = None
        
        # === PROCESS THROUGH ADAPTIVE INTELLIGENT SYSTEM ===
        signal_candidate = await adaptive_coordinator.process_candle(
            symbol, timeframe, candle, indicators_dict, df_with_indicators
        )
        
        if not signal_candidate:
            # Normal - 98% of scans produce no signal
            return
        
        # Signal passed ALL internal quality gates in coordinator!
        # Now do final validations
        
        # === EXTERNAL VALIDATION 1: Historical Performance ===
        valid, reason = await performance_validator.validate_signal(signal_candidate)
        if not valid:
            stats['rejection_reasons']['historical_performance'] = stats['rejection_reasons'].get('historical_performance', 0) + 1
            logger.debug(f"{symbol}: {reason}")
            return
        
        # === EXTERNAL VALIDATION 2: Regime Limits ===
        active_signals = await get_active_signals_from_db()
        valid, min_conf = regime_limiter.should_generate_signal(signal_candidate['regime'], active_signals)
        
        if not valid:
            stats['rejection_reasons']['regime_limit'] = stats['rejection_reasons'].get('regime_limit', 0) + 1
            logger.debug(f"{symbol}: Regime limit reached")
            return
        
        if signal_candidate['confidence'] < min_conf:
            stats['rejection_reasons']['regime_confidence'] = stats['rejection_reasons'].get('regime_confidence', 0) + 1
            logger.debug(f"{symbol}: Below regime min confidence")
            return
        
        # === EXTERNAL VALIDATION 3: Cooldown Window ===
        recent_signals = await get_recent_signals_from_db(120)
        valid, reason = aggregation_window.can_generate_signal(signal_candidate, recent_signals)
        
        if not valid:
            stats['rejection_reasons']['cooldown'] = stats['rejection_reasons'].get('cooldown', 0) + 1
            logger.debug(f"{symbol}: {reason}")
            return
        
        # === ALL QUALITY GATES PASSED! ===
        stats['quality_gates_passed'] += 1
        stats['signals_generated'] += 1
        
        # Store signal in database
        await store_signal(signal_candidate)
        
        # Broadcast to frontend
        await broadcast_new_signal(signal_candidate)
        
        logger.info(f"🎯 HIGH-QUALITY SIGNAL GENERATED: {symbol} {signal_candidate['direction'].upper()} "
                   f"@ {signal_candidate['confidence']:.0%} | Quality: {signal_candidate['quality_score']:.0%} | "
                   f"Confluence: {signal_candidate['confluence_analysis']['score']:.0%}")
        
    except Exception as e:
        logger.error(f"Error processing candle: {e}")

async def store_signal(signal: Dict):
    """Store signal in database"""
    
    global db_pool
    
    signal_id = f"INTEL_{uuid.uuid4().hex[:12].upper()}"
    
    async with db_pool.acquire() as conn:
        await conn.execute("""
            INSERT INTO live_signals (
                signal_id, symbol, timeframe, direction,
                entry_price, current_price, stop_loss, take_profit,
                confidence, quality_score, pattern_type,
                entry_proximity_pct, entry_proximity_status,
                sde_consensus, mtf_analysis, agreeing_heads,
                status, created_at, last_validated_at
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11,
                $12, $13, $14, $15, $16, $17, $18, $19
            )
        """,
        signal_id,
        signal['symbol'],
        signal.get('timeframe', '1h'),
        signal['direction'],
        signal['entry_price'],
        signal['entry_price'],  # current_price = entry for new signals
        signal['stop_loss'],
        signal['take_profit'],
        signal['confidence'],
        signal['quality_score'],
        signal['pattern_type'],
        0.0,  # entry_proximity_pct (0 for new signals)
        'imminent',  # New signals are always imminent
        json.dumps(signal['sde_consensus']),
        json.dumps(signal.get('confluence_analysis', {})),
        signal['sde_consensus']['agreeing_heads'],
        'active',
        datetime.now(),
        datetime.now()
        )
        
        logger.info(f"Signal stored in database: {signal_id}")

async def broadcast_new_signal(signal: Dict):
    """Broadcast new signal to all WebSocket connections"""
    
    message = {
        'type': 'new_signal',
        'data': signal,
        'timestamp': datetime.now().isoformat()
    }
    
    for ws in active_connections:
        try:
            await ws.send_json(message)
        except:
            pass

async def get_active_signals_from_db() -> List[Dict]:
    """Get active signals from database"""
    
    global db_pool
    
    async with db_pool.acquire() as conn:
        signals = await conn.fetch("""
            SELECT symbol, direction, confidence, quality_score
            FROM live_signals
            WHERE status = 'active'
            ORDER BY quality_score DESC
        """)
        return [dict(s) for s in signals]

async def get_recent_signals_from_db(minutes: int) -> List[Dict]:
    """Get recent signals from database"""
    
    global db_pool
    
    async with db_pool.acquire() as conn:
        signals = await conn.fetch("""
            SELECT signal_id, symbol, direction, created_at
            FROM live_signals
            WHERE created_at >= NOW() - ($1 || ' minutes')::INTERVAL
            ORDER BY created_at DESC
        """, minutes)
        return [dict(s) for s in signals]

@app.on_event("startup")
async def startup():
    """Initialize intelligent production system"""
    
    global db_pool, adaptive_coordinator, mtf_data_manager, market_connector
    global indicator_calculator, timeframe_selector, confluence_finder
    global performance_validator, regime_limiter, aggregation_window
    
    logger.info("=" * 80)
    logger.info("AlphaPulse Intelligent Production Backend Starting...")
    logger.info("=" * 80)
    
    try:
        # Initialize database
        db_pool = await asyncpg.create_pool(**DB_CONFIG, min_size=2, max_size=10)
        logger.info("✓ Database connection pool created")
        
        # Initialize quality control components
        timeframe_selector = AdaptiveTimeframeSelector()
        confluence_finder = ConfluenceEntryFinder()
        performance_validator = HistoricalPerformanceValidator(db_pool)
        regime_limiter = RegimeBasedSignalLimiter()
        aggregation_window = SignalAggregationWindow()
        logger.info("✓ Quality control components initialized")
        
        # Initialize adaptive coordinator
        adaptive_coordinator = AdaptiveIntelligenceCoordinator(db_pool)
        adaptive_coordinator.set_components(
            timeframe_selector=timeframe_selector,
            confluence_finder=confluence_finder,
            performance_validator=performance_validator,
            regime_limiter=regime_limiter,
            aggregation_window=aggregation_window
        )
        logger.info("✓ Adaptive Intelligence Coordinator initialized")
        
        # Initialize MTF data manager
        mtf_data_manager = MTFDataManager()
        indicator_calculator = RealtimeIndicatorCalculator()
        logger.info("✓ MTF Data Manager and Indicator Calculator initialized")
        
        # Register callback for candle completion
        for tf in ['1m', '5m', '15m', '1h', '4h']:
            mtf_data_manager.register_callback(tf, on_candle_complete)
        
        # Initialize market connector (1m candles)
        market_connector = LiveMarketConnector(SYMBOLS, [BASE_TIMEFRAME])
        
        # Add callback to MTF manager
        async def on_1m_candle(candle_data):
            symbol = candle_data['symbol']
            candle = {
                'timestamp': candle_data['timestamp'],
                'open': candle_data['open'],
                'high': candle_data['high'],
                'low': candle_data['low'],
                'close': candle_data['close'],
                'volume': candle_data['volume']
            }
            
            # Add to indicator calculator
            indicator_calculator.add_candle(symbol, '1m', candle)
            
            # Process through MTF manager (aggregates and triggers callbacks)
            await mtf_data_manager.on_1m_candle(symbol, candle)
        
        market_connector.add_callback(on_1m_candle)
        
        # Start market connector
        asyncio.create_task(market_connector.start())
        logger.info("✓ Live market connector started (Binance WebSocket)")
        
        logger.info("=" * 80)
        logger.info("System Features:")
        logger.info("  - Adaptive timeframe selection (regime-based)")
        logger.info("  - Multi-stage quality filtering (98-99% rejection)")
        logger.info("  - Confluence-based entry finding (70%+ required)")
        logger.info("  - Historical performance validation (60%+ win rate)")
        logger.info("  - Regime-based signal limits (1-3 per regime)")
        logger.info("  - Cooldown management (15-60 min between signals)")
        logger.info("=" * 80)
        logger.info(f"Monitoring: {len(SYMBOLS)} symbols on 1m base stream")
        logger.info("Target: 1-3 HIGH-QUALITY signals per day per symbol")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"Startup error: {e}")

@app.get("/")
async def root():
    return {
        "message": "AlphaPulse Intelligent Production API",
        "version": "2.0.0",
        "mode": "intelligent_adaptive",
        "features": [
            "Adaptive timeframe selection",
            "Multi-stage quality filtering",
            "Confluence-based entries",
            "Historical performance validation",
            "Regime-based limits",
            "98-99% rejection rate"
        ],
        "symbols": SYMBOLS,
        "base_timeframe": BASE_TIMEFRAME,
        "target_signals_per_day": "1-3 per symbol",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health():
    active_signals = await get_active_signals_from_db()
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "database": {
            "status": "connected",
            "active_signals": len(active_signals)
        },
        "websocket": {
            "status": "active",
            "connections": len(active_connections)
        },
        "statistics": stats
    }

@app.get("/api/signals/active")
async def get_active_signals():
    """Get active signals with full SDE+MTF data"""
    
    if not db_pool:
        return {"signals": []}
    
    async with db_pool.acquire() as conn:
        signals = await conn.fetch("""
            SELECT 
                signal_id, symbol, timeframe, direction,
                entry_price, current_price, stop_loss, take_profit,
                confidence, quality_score, pattern_type,
                entry_proximity_status, sde_consensus, mtf_analysis,
                agreeing_heads, created_at
            FROM live_signals
            WHERE status = 'active'
            ORDER BY quality_score DESC, confidence DESC
            LIMIT 5
        """)
        
        formatted = []
        for sig in signals:
            formatted.append({
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
        
        return {"signals": formatted}

@app.get("/api/signals/latest")
async def get_latest_signals():
    """Alias for frontend compatibility"""
    return await get_active_signals()

@app.get("/api/market/status")
async def get_market_status():
    """Get market status"""
    
    signals = await get_active_signals_from_db()
    
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
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/system/stats")
async def get_system_stats():
    """Get system statistics"""
    
    global stats
    
    rejection_rate = 0
    if stats['scans_performed'] > 0:
        rejection_rate = (1 - stats['signals_generated'] / stats['scans_performed']) * 100
    
    return {
        "scans_performed": stats['scans_performed'],
        "signals_generated": stats['signals_generated'],
        "rejection_rate": f"{rejection_rate:.1f}%",
        "quality_gates_passed": stats['quality_gates_passed'],
        "rejection_breakdown": stats['rejection_reasons'],
        "timestamp": datetime.now().isoformat()
    }

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_connections.append(websocket)
    
    try:
        # Send initial active signals
        signals_response = await get_active_signals()
        
        await websocket.send_json({
            "type": "initial_signals",
            "data": signals_response['signals'],
            "timestamp": datetime.now().isoformat()
        })
        
        # Keep connection alive
        while True:
            try:
                await asyncio.wait_for(websocket.receive_text(), timeout=30)
            except asyncio.TimeoutError:
                await websocket.send_json({
                    "type": "ping",
                    "timestamp": datetime.now().isoformat()
                })
                
    except WebSocketDisconnect:
        if websocket in active_connections:
            active_connections.remove(websocket)

if __name__ == "__main__":
    import uvicorn
    print("\n" + "=" * 80)
    print("AlphaPulse Intelligent Production Backend")
    print("=" * 80)
    print("Mode: Adaptive Intelligent Signal Generation")
    print("Quality Control: Multi-stage filtering (98-99% rejection)")
    print("Base Stream: 1m candles from Binance")
    print("Timeframe Selection: Adaptive (regime-based)")
    print("Signal Target: 1-3 per day per symbol (HIGH QUALITY)")
    print("=" * 80 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

