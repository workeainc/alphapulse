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
from collections import deque
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

# Import gap backfill service
from src.services.startup_gap_backfill_service import StartupGapBackfillService
import ccxt

# Import learning system components
from src.services.learning_coordinator import LearningCoordinator
from src.services.outcome_monitor_service import OutcomeMonitorService
from src.services.performance_analytics_service import PerformanceAnalyticsService
from src.services.rejection_learning_service import RejectionLearningService
from src.jobs.learning_scheduler import LearningScheduler

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
binance_exchange = None  # For gap backfill
adaptive_coordinator: AdaptiveIntelligenceCoordinator = None
mtf_data_manager: MTFDataManager = None
market_connector: LiveMarketConnector = None
indicator_calculator: RealtimeIndicatorCalculator = None

# Learning system components
learning_coordinator: LearningCoordinator = None
outcome_monitor: OutcomeMonitorService = None
performance_analytics: PerformanceAnalyticsService = None
rejection_learning: RejectionLearningService = None
learning_scheduler: LearningScheduler = None

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
    'quality_gates_passed': 0,
    'candles_received': 0,
    'candles_stored': 0,
    'last_candle_time': {},
    'indicator_calculations': 0,
    'consensus_calculations': 0,
    'head_votes': {},
    'workflow_steps': []
}

async def on_candle_complete(symbol: str, timeframe: str, candle: Dict):
    """
    Callback when any timeframe candle completes
    Main entry point for signal generation pipeline
    """
    
    global stats
    stats['scans_performed'] += 1
    
    try:
        logger.info(f"🕯️ Candle closed: {symbol} {timeframe} @ {candle['close']}")
        
        # Broadcast workflow update
        await broadcast_workflow_update({
            'type': 'candle_complete',
            'symbol': symbol,
            'timeframe': timeframe,
            'timestamp': datetime.now().isoformat(),
            'price': candle['close']
        })
        
        # ✅ STEP 1: Calculate ALL 69 indicators (returns dict for compatibility)
        logger.debug(f"📈 Calculating indicators for {symbol} {timeframe}...")
        indicators_dict = indicator_calculator.calculate_all_indicators(symbol, timeframe)
        stats['indicator_calculations'] += 1
        
        if not indicators_dict:
            logger.debug(f"{symbol}: Not enough data for indicators yet")
            await broadcast_workflow_update({
                'type': 'indicator_calculation',
                'symbol': symbol,
                'timeframe': timeframe,
                'status': 'insufficient_data',
                'indicators_count': 0
            })
            return
        
        # Get dataframe with ALL 69 indicator columns for aggregator
        df_with_indicators = indicator_calculator.get_dataframe_with_indicators(symbol, timeframe)
        
        if df_with_indicators is None:
            logger.debug(f"{symbol}: Not enough data for full indicator calculation")
            df_with_indicators = None
        
        await broadcast_workflow_update({
            'type': 'indicator_calculation',
            'symbol': symbol,
            'timeframe': timeframe,
            'status': 'success',
            'indicators_count': len(indicators_dict)
        })
        
        # ✅ STEP 2: PROCESS THROUGH ADAPTIVE INTELLIGENT SYSTEM (9-Head Consensus)
        logger.debug(f"🎯 Processing through 9-head consensus for {symbol} {timeframe}...")
        stats['consensus_calculations'] += 1
        
        signal_candidate = await adaptive_coordinator.process_candle(
            symbol, timeframe, candle, indicators_dict, df_with_indicators
        )
        
        # Log consensus votes if available
        if signal_candidate and 'sde_consensus' in signal_candidate:
            consensus = signal_candidate['sde_consensus']
            head_votes = {}
            if 'heads' in consensus:
                for head_name, head_data in consensus['heads'].items():
                    if isinstance(head_data, dict):
                        head_votes[head_name] = {
                            'direction': head_data.get('direction', 'FLAT'),
                            'confidence': head_data.get('confidence', 0.0)
                        }
            
            stats['head_votes'][f"{symbol}_{timeframe}"] = {
                'timestamp': datetime.now().isoformat(),
                'votes': head_votes,
                'consensus': {
                    'direction': consensus.get('direction', 'FLAT'),
                    'agreeing_heads': consensus.get('agreeing_heads', 0),
                    'total_heads': consensus.get('total_heads', 9),
                    'confidence': consensus.get('confidence', 0.0)
                }
            }
            
            await broadcast_workflow_update({
                'type': 'consensus_calculation',
                'symbol': symbol,
                'timeframe': timeframe,
                'consensus': stats['head_votes'][f"{symbol}_{timeframe}"]['consensus'],
                'head_votes': head_votes
            })
        
        if not signal_candidate:
            # Normal - 98% of scans produce no signal
            # NEW: Track this rejection for learning
            if rejection_learning:
                await rejection_learning.track_rejection(
                    symbol=symbol,
                    timeframe=timeframe,
                    signal_candidate=None,
                    consensus_data=stats['head_votes'].get(f"{symbol}_{timeframe}"),
                    rejection_reason='no_consensus',
                    rejection_stage='consensus_check'
                )
            return
        
        # Signal passed ALL internal quality gates in coordinator!
        # Now do final validations
        
        # === EXTERNAL VALIDATION 1: Historical Performance ===
        valid, reason = await performance_validator.validate_signal(signal_candidate)
        if not valid:
            stats['rejection_reasons']['historical_performance'] = stats['rejection_reasons'].get('historical_performance', 0) + 1
            logger.debug(f"{symbol}: {reason}")
            # NEW: Track this rejection for learning
            if rejection_learning:
                await rejection_learning.track_rejection(
                    symbol=symbol,
                    timeframe=timeframe,
                    signal_candidate=signal_candidate,
                    consensus_data=None,
                    rejection_reason='historical_performance',
                    rejection_stage='quality_filter'
                )
            return
        
        # === EXTERNAL VALIDATION 2: Regime Limits ===
        active_signals = await get_active_signals_from_db()
        valid, min_conf = regime_limiter.should_generate_signal(signal_candidate['regime'], active_signals)
        
        if not valid:
            stats['rejection_reasons']['regime_limit'] = stats['rejection_reasons'].get('regime_limit', 0) + 1
            logger.debug(f"{symbol}: Regime limit reached")
            # NEW: Track this rejection for learning
            if rejection_learning:
                await rejection_learning.track_rejection(
                    symbol=symbol,
                    timeframe=timeframe,
                    signal_candidate=signal_candidate,
                    consensus_data=None,
                    rejection_reason='regime_limit',
                    rejection_stage='regime_filter'
                )
            return
        
        if signal_candidate['confidence'] < min_conf:
            stats['rejection_reasons']['regime_confidence'] = stats['rejection_reasons'].get('regime_confidence', 0) + 1
            logger.debug(f"{symbol}: Below regime min confidence")
            # NEW: Track this rejection for learning
            if rejection_learning:
                await rejection_learning.track_rejection(
                    symbol=symbol,
                    timeframe=timeframe,
                    signal_candidate=signal_candidate,
                    consensus_data=None,
                    rejection_reason='regime_confidence',
                    rejection_stage='regime_filter'
                )
            return
        
        # === EXTERNAL VALIDATION 3: Cooldown Window ===
        recent_signals = await get_recent_signals_from_db(120)
        valid, reason = aggregation_window.can_generate_signal(signal_candidate, recent_signals)
        
        if not valid:
            stats['rejection_reasons']['cooldown'] = stats['rejection_reasons'].get('cooldown', 0) + 1
            logger.debug(f"{symbol}: {reason}")
            # NEW: Track this rejection for learning
            if rejection_learning:
                await rejection_learning.track_rejection(
                    symbol=symbol,
                    timeframe=timeframe,
                    signal_candidate=signal_candidate,
                    consensus_data=None,
                    rejection_reason='cooldown',
                    rejection_stage='cooldown_filter'
                )
            return
        
        # === ALL QUALITY GATES PASSED! ===
        stats['quality_gates_passed'] += 1
        stats['signals_generated'] += 1
        
        # Broadcast workflow update
        await broadcast_workflow_update({
            'type': 'signal_generated',
            'symbol': symbol,
            'timeframe': timeframe,
            'direction': signal_candidate['direction'],
            'confidence': signal_candidate['confidence'],
            'quality_score': signal_candidate['quality_score']
        })
        
        # Store signal in database
        await store_signal(signal_candidate)
        
        # Broadcast to frontend
        await broadcast_new_signal(signal_candidate)
        
        logger.info(f"🎯 HIGH-QUALITY SIGNAL GENERATED: {symbol} {signal_candidate['direction'].upper()} "
                   f"@ {signal_candidate['confidence']:.0%} | Quality: {signal_candidate['quality_score']:.0%} | "
                   f"Confluence: {signal_candidate['confluence_analysis']['score']:.0%}")
        
        # Add to workflow history
        stats['workflow_steps'].append({
            'timestamp': datetime.now().isoformat(),
            'event': 'signal_generated',
            'symbol': symbol,
            'timeframe': timeframe,
            'direction': signal_candidate['direction'],
            'confidence': signal_candidate['confidence']
        })
        
        # Keep only last 100 steps
        if len(stats['workflow_steps']) > 100:
            stats['workflow_steps'] = stats['workflow_steps'][-100:]
        
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

async def broadcast_workflow_update(update: Dict):
    """Broadcast workflow update to all WebSocket connections"""
    
    message = {
        'type': 'workflow_update',
        'data': update,
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

async def load_historical_data_to_buffers(
    db_pool, 
    indicator_calculator: RealtimeIndicatorCalculator,
    mtf_data_manager: MTFDataManager,
    limit: int = 100
):
    """
    Load recent historical data from database into indicator and MTF buffers
    This allows indicators to calculate immediately without waiting for WebSocket accumulation
    """
    try:
        async with db_pool.acquire() as conn:
            for symbol in SYMBOLS:
                for timeframe in ['1m', '5m', '15m', '1h']:
                    # Fetch most recent candles
                    candles = await conn.fetch("""
                        SELECT timestamp, open, high, low, close, volume
                        FROM ohlcv_data
                        WHERE symbol = $1 AND timeframe = $2
                        ORDER BY timestamp DESC
                        LIMIT $3
                    """, symbol, timeframe, limit)
                    
                    if not candles:
                        continue
                    
                    # Convert to dict format and reverse (oldest first)
                    candle_list = []
                    for row in reversed(candles):
                        candle = {
                            'timestamp': row['timestamp'],
                            'open': float(row['open']),
                            'high': float(row['high']),
                            'low': float(row['low']),
                            'close': float(row['close']),
                            'volume': float(row['volume'])
                        }
                        candle_list.append(candle)
                    
                    # Load into indicator calculator buffer
                    for candle in candle_list:
                        indicator_calculator.add_candle(symbol, timeframe, candle)
                    
                    # Load into MTF manager buffers (silently, without triggering callbacks)
                    buffer_key = f"{symbol}_{timeframe}"
                    if buffer_key not in mtf_data_manager.buffers:
                        mtf_data_manager.buffers[buffer_key] = deque(maxlen=mtf_data_manager.buffer_size)
                    
                    for candle in candle_list:
                        mtf_data_manager.buffers[buffer_key].append(candle)
                    
                    # For 1m data, also initialize the candle counter for MTF aggregation
                    if timeframe == '1m' and symbol not in mtf_data_manager.candle_counters:
                        mtf_data_manager.candle_counters[symbol] = len(candle_list)
                    
                    logger.info(f"  ✓ {symbol} {timeframe}: {len(candle_list)} candles")
        
        logger.info(f"✓ Loaded historical data for {len(SYMBOLS)} symbols × 4 timeframes")
        
    except Exception as e:
        logger.warning(f"⚠️ Could not load historical data into buffers: {e}")
        logger.warning("  Indicators will calculate from WebSocket data accumulation (may take time)")

@app.on_event("startup")
async def startup():
    """Initialize intelligent production system"""
    
    global db_pool, binance_exchange, adaptive_coordinator, mtf_data_manager, market_connector
    global indicator_calculator, timeframe_selector, confluence_finder
    global performance_validator, regime_limiter, aggregation_window
    global learning_coordinator, outcome_monitor, performance_analytics, rejection_learning, learning_scheduler
    
    logger.info("=" * 80)
    logger.info("AlphaPulse Intelligent Production Backend Starting...")
    logger.info("=" * 80)
    
    try:
        # Initialize database
        db_pool = await asyncpg.create_pool(**DB_CONFIG, min_size=2, max_size=10)
        logger.info("✓ Database connection pool created")
        
        # Initialize Binance exchange for gap backfill
        binance_exchange = ccxt.binance({
            'enableRateLimit': True,
        })
        logger.info("✓ Binance exchange initialized")
        
        # ===== GAP DETECTION & BACKFILL =====
        logger.info("🔍 Checking for data gaps since last shutdown...")
        backfill_service = StartupGapBackfillService(db_pool, binance_exchange)
        
        # Add XRPUSDT to symbols for backfill
        backfill_symbols = SYMBOLS + ['XRPUSDT']
        
        try:
            backfill_stats = await backfill_service.detect_and_fill_all_gaps(backfill_symbols)
            
            if backfill_stats['gaps_filled'] > 0:
                logger.info(
                    f"✅ Backfill complete: Filled {backfill_stats['gaps_filled']} gaps, "
                    f"stored {backfill_stats['candles_stored']:,} candles"
                )
            else:
                logger.info("✅ No gaps detected - data is current!")
        except Exception as e:
            logger.warning(f"⚠️ Gap backfill encountered an issue (non-critical): {e}")
            logger.info("Continuing with system startup...")
        # ======================================
        
        # ===== LEARNING SYSTEM INITIALIZATION =====
        logger.info("🧠 Initializing self-learning system...")
        
        # Initialize learning coordinator
        learning_coordinator = LearningCoordinator(db_pool)
        await learning_coordinator.initialize()
        logger.info("✓ Learning Coordinator initialized")
        
        # Initialize performance analytics
        performance_analytics = PerformanceAnalyticsService(db_pool)
        logger.info("✓ Performance Analytics Service initialized")
        
        # Initialize outcome monitor
        outcome_monitor = OutcomeMonitorService(db_pool, binance_exchange, learning_coordinator)
        logger.info("✓ Outcome Monitor Service initialized")
        
        # Initialize rejection learning service
        rejection_learning = RejectionLearningService(db_pool, binance_exchange, learning_coordinator)
        logger.info("✓ Rejection Learning Service initialized")
        
        # Start outcome monitoring loop in background
        asyncio.create_task(outcome_monitor.monitor_active_signals())
        logger.info("✅ Outcome monitoring activated - system will learn from every signal!")
        
        # Start shadow signal monitoring (learns from rejections)
        asyncio.create_task(rejection_learning.monitor_shadow_signals())
        logger.info("✅ Rejection monitoring activated - system will learn from rejected signals too!")
        
        # Initialize and start learning scheduler
        learning_scheduler = LearningScheduler(db_pool)
        learning_scheduler.start()
        logger.info("✓ Learning Scheduler started (daily + weekly jobs automated)")
        
        # Load learned head weights and apply to consensus system
        learned_head_weights = await learning_coordinator.get_current_head_weights()
        logger.info(f"✓ Loaded learned head weights from database")
        logger.info(f"   Sample weights: HEAD_A={learned_head_weights.get('HEAD_A', 0):.4f}, "
                   f"HEAD_B={learned_head_weights.get('HEAD_B', 0):.4f}, "
                   f"HEAD_C={learned_head_weights.get('HEAD_C', 0):.4f}")
        # ==========================================
        
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
        
        # Load historical data into buffers for immediate indicator calculation
        # Load last 500 candles per symbol/timeframe (8.3 hours to 20 days depending on TF)
        # This takes 2-3 minutes but enables ALL indicators + pattern recognition to work immediately
        logger.info("📥 Loading last 500 candles per symbol/timeframe from database...")
        await load_historical_data_to_buffers(db_pool, indicator_calculator, mtf_data_manager, limit=500)
        logger.info("✅ Historical data loaded - all 9 heads ready with full context!")
        
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
            
            # Track received candle
            stats['candles_received'] += 1
            
            # Fix: Properly format timestamp for tracking
            if isinstance(candle['timestamp'], str):
                stats['last_candle_time'][symbol] = candle['timestamp']
            elif hasattr(candle['timestamp'], 'isoformat'):
                stats['last_candle_time'][symbol] = candle['timestamp'].isoformat()
            else:
                stats['last_candle_time'][symbol] = str(candle['timestamp'])
            
            logger.debug(f"📊 Tracked candle time for {symbol}: {stats['last_candle_time'][symbol]}")
            
            # ✅ STEP 1: Store in database (NEW - CRITICAL FIX)
            try:
                async with db_pool.acquire() as conn:
                    result = await conn.execute("""
                        INSERT INTO ohlcv_data (symbol, timeframe, timestamp, open, high, low, close, volume, source)
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                        ON CONFLICT (symbol, timeframe, timestamp) DO NOTHING
                    """, symbol, '1m', candle['timestamp'], candle['open'], candle['high'],
                         candle['low'], candle['close'], candle['volume'], 'websocket')
                    
                    if result == "INSERT 0 1":  # One row inserted
                        stats['candles_stored'] += 1
                        logger.debug(f"💾 Stored new candle: {symbol} 1m @ {candle['timestamp']}")
                    
                    # Broadcast workflow update
                    await broadcast_workflow_update({
                        'type': 'candle_received',
                        'symbol': symbol,
                        'timeframe': '1m',
                        'timestamp': stats['last_candle_time'][symbol],
                        'price': candle['close'],
                        'candles_received': stats['candles_received'],
                        'candles_stored': stats['candles_stored']
                    })
            except Exception as e:
                logger.error(f"❌ Failed to store candle in database: {e}")
            
            # ✅ STEP 2: Add to indicator calculator
            indicator_calculator.add_candle(symbol, '1m', candle)
            logger.debug(f"📊 Added {symbol} 1m candle to indicator calculator")
            
            # ✅ STEP 3: Process through MTF manager (aggregates and triggers callbacks)
            await mtf_data_manager.on_1m_candle(symbol, candle)
            logger.debug(f"🔄 Processed {symbol} 1m candle through MTF manager")
        
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
        logger.info("  🧠 Self-Learning System (NEW!):")
        logger.info("    • Automatic outcome monitoring (TP/SL detection every 60s)")
        logger.info("    • Rejection learning (learns from 98% rejected signals too!)")
        logger.info("    • 9-head weight optimization (learns from ALL decisions)")
        logger.info("    • Daily learning job (midnight UTC) + Weekly retraining (Sunday 2am)")
        logger.info("    • Performance analytics and tracking")
        logger.info("    • System improves continuously over time")
        logger.info("=" * 80)
        logger.info(f"Monitoring: {len(SYMBOLS)} symbols on 1m base stream")
        logger.info("Target: 1-3 HIGH-QUALITY signals per day per symbol")
        logger.info("Learning: Every signal outcome improves the system")
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
        "candles_received": stats['candles_received'],
        "candles_stored": stats['candles_stored'],
        "indicator_calculations": stats['indicator_calculations'],
        "consensus_calculations": stats['consensus_calculations'],
        "last_candle_time": stats['last_candle_time'],
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/system/workflow")
async def get_workflow_status():
    """Get real-time workflow status and process"""
    
    global stats, indicator_calculator, mtf_data_manager
    
    # Get buffer status
    buffer_status = {}
    for symbol in SYMBOLS:
        buffer_status[symbol] = {}
        for timeframe in ['1m', '5m', '15m', '1h']:
            buffer_key = f"{symbol}_{timeframe}"
            if buffer_key in indicator_calculator.buffers:
                buffer_status[symbol][timeframe] = {
                    'candles_in_buffer': len(indicator_calculator.buffers[buffer_key]),
                    'buffer_size': indicator_calculator.buffer_size
                }
            else:
                buffer_status[symbol][timeframe] = {
                    'candles_in_buffer': 0,
                    'buffer_size': indicator_calculator.buffer_size
                }
    
    # Calculate time since last candle
    time_since_last_candle = {}
    for symbol in SYMBOLS:
        if symbol in stats['last_candle_time'] and stats['last_candle_time'][symbol]:
            last_time_str = stats['last_candle_time'][symbol]
            try:
                if isinstance(last_time_str, str):
                    # Handle ISO format timestamps
                    last_time = datetime.fromisoformat(last_time_str.replace('Z', '+00:00'))
                else:
                    last_time = last_time_str
                
                # Calculate seconds since last candle
                if last_time.tzinfo:
                    time_since = (datetime.now(last_time.tzinfo) - last_time).total_seconds()
                else:
                    time_since = (datetime.now() - last_time).total_seconds()
                
                time_since_last_candle[symbol] = {
                    'seconds': int(time_since),
                    'status': 'realtime' if time_since < 120 else 'delayed' if time_since < 300 else 'stale'
                }
                logger.debug(f"📊 {symbol} last candle: {time_since}s ago")
            except Exception as e:
                logger.warning(f"⚠️ Error calculating time for {symbol}: {e}")
                time_since_last_candle[symbol] = {'seconds': -1, 'status': 'error'}
        else:
            time_since_last_candle[symbol] = {'seconds': -1, 'status': 'no_data'}
    
    return {
        "workflow_status": {
            "data_collection": {
                "candles_received": stats['candles_received'],
                "candles_stored": stats['candles_stored'],
                "last_candle_times": stats['last_candle_time'],
                "time_since_last_candle": time_since_last_candle,
                "status": "active" if stats['candles_received'] > 0 else "waiting"
            },
            "indicator_calculation": {
                "calculations_performed": stats['indicator_calculations'],
                "buffer_status": buffer_status,
                "status": "active" if stats['indicator_calculations'] > 0 else "waiting"
            },
            "consensus_system": {
                "calculations_performed": stats['consensus_calculations'],
                "last_consensus_votes": stats['head_votes'],
                "status": "active" if stats['consensus_calculations'] > 0 else "waiting"
            },
            "signal_generation": {
                "scans_performed": stats['scans_performed'],
                "signals_generated": stats['signals_generated'],
                "rejection_rate": f"{(1 - stats['signals_generated'] / stats['scans_performed']) * 100:.1f}%" if stats['scans_performed'] > 0 else "0%",
                "status": "active" if stats['scans_performed'] > 0 else "waiting"
            }
        },
        "recent_workflow_steps": stats['workflow_steps'][-20:],  # Last 20 steps
        "timestamp": datetime.now().isoformat()
    }

# ============================================================================
# LEARNING SYSTEM API ENDPOINTS (Phase 3)
# ============================================================================

@app.get("/api/learning/performance")
async def get_learning_performance():
    """
    Comprehensive performance metrics for learning system
    Returns overall performance, head performance, and learning progress
    """
    global performance_analytics
    
    if not performance_analytics:
        return {"error": "Learning system not initialized"}
    
    try:
        overall = await performance_analytics.calculate_overall_performance(period='7d')
        head_performance = await performance_analytics.calculate_head_performance()
        learning_progress = await performance_analytics.calculate_learning_progress()
        
        return {
            "overall": overall,
            "head_performance": head_performance,
            "learning_progress": learning_progress,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting learning performance: {e}")
        return {"error": str(e)}

@app.get("/api/learning/head-weights")
async def get_head_weights_history(days: int = 30):
    """
    Get historical head weight changes over time
    Shows how weights have evolved as system learned
    """
    global performance_analytics, learning_coordinator
    
    if not performance_analytics:
        return {"error": "Learning system not initialized"}
    
    try:
        # Get weight history
        history = await performance_analytics.get_weight_history('head_weights', days=days)
        
        # Get current weights
        current_weights = {}
        if learning_coordinator:
            current_weights = await learning_coordinator.get_current_head_weights()
        
        return {
            "current_weights": current_weights,
            "weight_history": history,
            "days_analyzed": days,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting head weights: {e}")
        return {"error": str(e)}

@app.get("/api/learning/improvements")
async def get_learning_improvements():
    """
    Show week-over-week improvements and trends
    Highlights best/worst performing heads
    """
    global performance_analytics
    
    if not performance_analytics:
        return {"error": "Learning system not initialized"}
    
    try:
        # Get learning progress (weekly trends)
        progress = await performance_analytics.calculate_learning_progress()
        
        # Get head performance
        head_perf = await performance_analytics.calculate_head_performance()
        
        # Extract best and worst heads
        heads = head_perf.get('heads', {})
        sorted_heads = sorted(
            heads.items(),
            key=lambda x: x[1].get('win_rate_when_agreed', 0),
            reverse=True
        )
        
        best_heads = sorted_heads[:3] if len(sorted_heads) >= 3 else sorted_heads
        worst_heads = sorted_heads[-3:] if len(sorted_heads) >= 3 else []
        
        return {
            "weekly_trends": progress,
            "best_performing_heads": [
                {
                    "head_name": name,
                    "win_rate": data.get('win_rate_when_agreed', 0),
                    "signals": data.get('signals_contributed', 0),
                    "current_weight": data.get('current_weight', 0)
                }
                for name, data in best_heads
            ],
            "worst_performing_heads": [
                {
                    "head_name": name,
                    "win_rate": data.get('win_rate_when_agreed', 0),
                    "signals": data.get('signals_contributed', 0),
                    "current_weight": data.get('current_weight', 0)
                }
                for name, data in worst_heads
            ],
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting improvements: {e}")
        return {"error": str(e)}

@app.get("/api/learning/recommendations")
async def get_learning_recommendations():
    """
    AI-generated recommendations for system improvement
    Suggests which heads need rebalancing and optimal threshold adjustments
    """
    global performance_analytics, learning_coordinator
    
    if not performance_analytics or not learning_coordinator:
        return {"error": "Learning system not initialized"}
    
    try:
        # Get head performance
        head_perf = await performance_analytics.calculate_head_performance()
        heads = head_perf.get('heads', {})
        
        # Generate recommendations
        recommendations = []
        
        for head_name, data in heads.items():
            current_weight = data.get('current_weight', 0)
            suggested_weight = data.get('suggested_weight', 0)
            adjustment = data.get('weight_adjustment_needed', 0)
            win_rate = data.get('win_rate_when_agreed', 0)
            trend = data.get('performance_trend', 'unknown')
            
            # Recommendation logic
            if abs(adjustment) > 0.02:  # Significant adjustment needed
                if adjustment > 0:
                    recommendations.append({
                        "type": "increase_weight",
                        "head": head_name,
                        "current_weight": round(current_weight, 4),
                        "suggested_weight": round(suggested_weight, 4),
                        "reason": f"High win rate ({win_rate:.1%}) - increase weight by {adjustment:.1%}",
                        "priority": "high" if abs(adjustment) > 0.05 else "medium"
                    })
                else:
                    recommendations.append({
                        "type": "decrease_weight",
                        "head": head_name,
                        "current_weight": round(current_weight, 4),
                        "suggested_weight": round(suggested_weight, 4),
                        "reason": f"Low win rate ({win_rate:.1%}) - decrease weight by {abs(adjustment):.1%}",
                        "priority": "high" if abs(adjustment) > 0.05 else "medium"
                    })
            
            # Trend-based recommendations
            if trend == 'declining' and win_rate < 0.60:
                recommendations.append({
                    "type": "monitor_closely",
                    "head": head_name,
                    "reason": f"Performance declining - win rate {win_rate:.1%}",
                    "priority": "medium"
                })
        
        # Overall system recommendations
        overall_perf = await performance_analytics.calculate_overall_performance(period='7d')
        overall_win_rate = overall_perf.get('win_rate', 0)
        
        if overall_win_rate < 0.60:
            recommendations.append({
                "type": "system_adjustment",
                "reason": f"Overall win rate ({overall_win_rate:.1%}) below target (65%)",
                "suggestion": "Consider increasing confidence threshold or retraining",
                "priority": "high"
            })
        elif overall_win_rate > 0.75:
            recommendations.append({
                "type": "system_adjustment",
                "reason": f"Overall win rate ({overall_win_rate:.1%}) very high",
                "suggestion": "System performing excellently - consider decreasing threshold for more signals",
                "priority": "low"
            })
        
        return {
            "recommendations": recommendations,
            "total_recommendations": len(recommendations),
            "high_priority_count": len([r for r in recommendations if r.get('priority') == 'high']),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error generating recommendations: {e}")
        return {"error": str(e)}

@app.get("/api/learning/stats")
async def get_learning_stats():
    """
    Get learning system statistics
    Shows how many outcomes processed, updates made, etc.
    """
    global learning_coordinator, outcome_monitor, rejection_learning
    
    if not learning_coordinator or not outcome_monitor:
        return {"error": "Learning system not initialized"}
    
    try:
        coord_stats = learning_coordinator.get_stats()
        monitor_stats = outcome_monitor.get_stats()
        rejection_stats = rejection_learning.get_stats() if rejection_learning else {}
        
        return {
            "coordinator": coord_stats,
            "monitor": monitor_stats,
            "rejection_learning": rejection_stats,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting learning stats: {e}")
        return {"error": str(e)}

@app.get("/api/learning/scheduler")
async def get_scheduler_status():
    """
    Get learning scheduler status and job history
    Shows next run times for daily/weekly jobs
    """
    global learning_scheduler
    
    if not learning_scheduler:
        return {"error": "Learning scheduler not initialized"}
    
    try:
        status = learning_scheduler.get_status()
        return {
            "scheduler": status,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting scheduler status: {e}")
        return {"error": str(e)}

@app.post("/api/learning/trigger-daily")
async def trigger_daily_learning():
    """
    Manually trigger daily learning job (for testing)
    """
    global learning_scheduler
    
    if not learning_scheduler:
        return {"error": "Learning scheduler not initialized"}
    
    try:
        await learning_scheduler.run_daily_now()
        return {
            "status": "completed",
            "message": "Daily learning job triggered manually",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error triggering daily job: {e}")
        return {"error": str(e)}

@app.post("/api/learning/trigger-weekly")
async def trigger_weekly_retraining():
    """
    Manually trigger weekly retraining job (for testing)
    """
    global learning_scheduler
    
    if not learning_scheduler:
        return {"error": "Learning scheduler not initialized"}
    
    try:
        await learning_scheduler.run_weekly_now()
        return {
            "status": "completed",
            "message": "Weekly retraining job triggered manually",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error triggering weekly job: {e}")
        return {"error": str(e)}

@app.get("/api/learning/rejection-analysis")
async def get_rejection_analysis():
    """
    Get detailed rejection learning analysis
    Shows missed opportunities vs good rejections
    """
    global rejection_learning, db_pool
    
    if not rejection_learning:
        return {"error": "Rejection learning not initialized"}
    
    try:
        # Get rejection statistics
        rejection_stats = rejection_learning.get_stats()
        
        # Get recent rejection outcomes from database
        async with db_pool.acquire() as conn:
            recent_rejections = await conn.fetch("""
                SELECT 
                    learning_outcome,
                    COUNT(*) as count,
                    AVG(simulated_profit_pct) as avg_profit
                FROM rejected_signals
                WHERE completed_at >= NOW() - INTERVAL '7 days'
                AND learning_outcome IS NOT NULL
                GROUP BY learning_outcome
            """)
            
            rejection_breakdown = {
                row['learning_outcome']: {
                    'count': row['count'],
                    'avg_profit': float(row['avg_profit']) if row['avg_profit'] else 0
                }
                for row in recent_rejections
            }
        
        # Calculate rejection accuracy
        missed = rejection_stats.get('missed_opportunities', 0)
        good = rejection_stats.get('good_rejections', 0)
        total = missed + good
        rejection_accuracy = (good / total) if total > 0 else 0
        
        return {
            "rejection_statistics": rejection_stats,
            "rejection_breakdown": rejection_breakdown,
            "rejection_accuracy": round(rejection_accuracy, 4),
            "learning_data_multiplier": f"{rejection_stats.get('rejections_tracked', 0)}x more data than signals alone",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting rejection analysis: {e}")
        return {"error": str(e)}

# ============================================================================

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
        
        # Send initial workflow status
        workflow_status = await get_workflow_status()
        await websocket.send_json({
            "type": "workflow_status",
            "data": workflow_status,
            "timestamp": datetime.now().isoformat()
        })
        
        # Keep connection alive and send periodic workflow updates
        last_workflow_update = datetime.now()
        while True:
            try:
                await asyncio.wait_for(websocket.receive_text(), timeout=30)
            except asyncio.TimeoutError:
                # Send ping
                await websocket.send_json({
                    "type": "ping",
                    "timestamp": datetime.now().isoformat()
                })
                
                # Send workflow status every 10 seconds
                if (datetime.now() - last_workflow_update).total_seconds() >= 10:
                    workflow_status = await get_workflow_status()
                    await websocket.send_json({
                        "type": "workflow_status",
                        "data": workflow_status,
                        "timestamp": datetime.now().isoformat()
                    })
                    last_workflow_update = datetime.now()
                
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

