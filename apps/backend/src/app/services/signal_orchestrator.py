"""
Signal Orchestrator for AlphaPulse
Coordinates signal generation, analysis, and recommendation delivery
This is NOT a trading execution engine - it provides analysis and recommendations only
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from src.app.core.config import settings
from src.app.services.market_data_service import MarketDataService
from src.app.services.sentiment_service import SentimentService
from src.app.services.risk_manager import RiskManager
from src.app.strategies.strategy_manager import StrategyManager
from src.app.database.models import Signal, SignalRecommendation, MarketData
from src.app.database.connection import get_db

# Import real-time components
from src.app.strategies.real_time_signal_generator import RealTimeSignalGenerator
from src.app.data.real_time_processor import RealTimeCandlestickProcessor
from src.app.core.unified_websocket_client import UnifiedWebSocketClient, UnifiedWebSocketManager

# Import Enhanced Algorithm Integration
try:
    from src.services.algorithm_integration_service import AlgorithmIntegrationService
    from src.strategies.standalone_psychological_levels_analyzer import StandalonePsychologicalLevelsAnalyzer
    from src.strategies.enhanced_volume_weighted_levels_analyzer import EnhancedVolumeWeightedLevelsAnalyzer
    from src.services.enhanced_orderbook_integration import EnhancedOrderBookIntegration
    ENHANCED_ALGORITHMS_AVAILABLE = True
except ImportError:
    ENHANCED_ALGORITHMS_AVAILABLE = False
    logging.warning("Enhanced Algorithm Integration not available")

# Import latency tracking
from src.app.core.latency_tracker import track_trading_pipeline, latency_tracker

logger = logging.getLogger(__name__)

class SignalOrchestrator:
    """
    Signal Orchestrator that coordinates signal generation and analysis
    
    This orchestrator:
    - Analyzes market data in real-time
    - Generates trading signals with confidence scores
    - Provides risk recommendations (SL/TP/position sizing)
    - Delivers alerts and notifications
    - Does NOT execute trades - provides recommendations only
    """
    
    def __init__(self):
        self.is_running = False
        self.market_data_service = MarketDataService()
        self.sentiment_service = SentimentService()
        self.risk_manager = RiskManager()
        self.strategy_manager = StrategyManager()
        
        # Real-time components
        self.signal_generator = RealTimeSignalGenerator()
        self.candlestick_processor = RealTimeCandlestickProcessor()
        self.websocket_client = UnifiedWebSocketClient()
        
        # Enhanced Algorithm Integration
        self.algorithm_integration_service = None
        self.psychological_levels_analyzer = None
        self.volume_weighted_levels_analyzer = None
        self.enhanced_orderbook_integration = None
        
        if ENHANCED_ALGORITHMS_AVAILABLE:
            try:
                self.algorithm_integration_service = AlgorithmIntegrationService()
                self.psychological_levels_analyzer = StandalonePsychologicalLevelsAnalyzer()
                self.volume_weighted_levels_analyzer = EnhancedVolumeWeightedLevelsAnalyzer()
                self.enhanced_orderbook_integration = EnhancedOrderBookIntegration()
                logger.info("✅ Enhanced Algorithm Integration initialized in Signal Orchestrator")
            except Exception as e:
                logger.error(f"❌ Failed to initialize Enhanced Algorithm Integration: {e}")
        
        # Signal state
        self.active_signals = {}
        self.signal_history = []
        self.trading_pairs = ["BTCUSDT", "ETHUSDT", "ADAUSDT"]
        
        # Performance tracking
        self.signals_generated = 0
        self.high_confidence_signals = 0
        self.current_model_id = "signal_orchestrator_v1"
        
        logger.info("📊 Signal Orchestrator initialized (Analysis Mode Only)")

    async def start(self):
        """Start the signal orchestration system"""
        if self.is_running:
            logger.warning("Signal Orchestrator is already running")
            return
        
        try:
            logger.info("🚀 Starting Signal Orchestrator...")
            
            # Start services
            await self.market_data_service.start()
            await self.sentiment_service.start()
            
            # Start real-time data collection
            asyncio.create_task(self.collect_market_data())
            asyncio.create_task(self.analyze_and_generate_signals())
            
            self.is_running = True
            logger.info("✅ Signal Orchestrator started successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to start Signal Orchestrator: {e}")
            raise

    async def stop(self):
        """Stop the signal orchestration system"""
        if not self.is_running:
            return
        
        try:
            logger.info("🛑 Stopping Signal Orchestrator...")
            
            # Stop services
            await self.market_data_service.stop()
            await self.sentiment_service.stop()
            
            self.is_running = False
            logger.info("✅ Signal Orchestrator stopped")
            
        except Exception as e:
            logger.error(f"❌ Error stopping Signal Orchestrator: {e}")

    async def collect_market_data(self):
        """Collect real-time market data"""
        while self.is_running:
            try:
                for symbol in self.trading_pairs:
                    # Fetch latest market data
                    market_data = await self.market_data_service.get_latest_data(symbol)
                    
                    if market_data:
                        # Process candlestick data
                        await self.candlestick_processor.process(market_data)
                
                await asyncio.sleep(1)  # 1-second update frequency
                
            except Exception as e:
                logger.error(f"Error collecting market data: {e}")
                await asyncio.sleep(5)

    async def analyze_and_generate_signals(self):
        """Analyze market and generate signals"""
        while self.is_running:
            try:
                for symbol in self.trading_pairs:
                    # Get market data
                    market_data = await self.market_data_service.get_data(symbol, timeframe="1h", limit=100)
                    
                    if market_data is None or len(market_data) < 20:
                        continue
                    
                    # Get sentiment analysis
                    sentiment = await self.sentiment_service.analyze(symbol)
                    
                    # Generate signals using strategy manager
                    signals = await self.strategy_manager.generate_signals(
                        symbol=symbol,
                        market_data=market_data,
                        sentiment=sentiment
                    )
                    
                    # Process and validate signals
                    for signal in signals:
                        await self.process_signal(signal, symbol)
                
                await asyncio.sleep(60)  # Analyze every minute
                
            except Exception as e:
                logger.error(f"Error analyzing and generating signals: {e}")
                await asyncio.sleep(10)

    async def process_signal(self, signal: Dict[str, Any], symbol: str):
        """Process and create signal recommendation"""
        try:
            confidence = signal.get('confidence', 0.0)
            
            # Only process high-confidence signals
            if confidence < 0.70:
                return
            
            direction = signal.get('direction', 'long')
            entry_price = signal.get('entry_price', 0.0)
            
            # Calculate risk recommendations
            risk_params = await self.risk_manager.calculate_risk_parameters(
                symbol=symbol,
                direction=direction,
                entry_price=entry_price,
                confidence=confidence
            )
            
            # Create signal recommendation
            recommendation = SignalRecommendation(
                signal_id=f"sig_{symbol}_{int(datetime.utcnow().timestamp())}",
                symbol=symbol,
                side=direction,
                suggested_entry_price=entry_price,
                suggested_quantity=risk_params.get('suggested_quantity', 0.01),
                suggested_leverage=risk_params.get('suggested_leverage', 1),
                suggested_stop_loss=risk_params.get('stop_loss'),
                suggested_take_profit=risk_params.get('take_profit'),
                strategy_name=signal.get('strategy_name', 'unknown'),
                market_regime=signal.get('market_regime'),
                sentiment_score=signal.get('sentiment_score'),
                volatility=signal.get('volatility'),
                status='pending'
            )
            
            # Store recommendation
            await self.store_recommendation(recommendation)
            
            # Send alert/notification
            await self.send_alert(recommendation, confidence)
            
            self.signals_generated += 1
            if confidence >= 0.80:
                self.high_confidence_signals += 1
            
            logger.info(f"🎯 Generated signal recommendation: {symbol} {direction} @ {entry_price} (confidence: {confidence:.2%})")
            
        except Exception as e:
            logger.error(f"Error processing signal: {e}")

    async def store_recommendation(self, recommendation: SignalRecommendation):
        """Store signal recommendation in database"""
        try:
            async with get_db() as db:
                db.add(recommendation)
                await db.commit()
                logger.debug(f"Stored recommendation: {recommendation.signal_id}")
        except Exception as e:
            logger.error(f"Error storing recommendation: {e}")

    async def send_alert(self, recommendation: SignalRecommendation, confidence: float):
        """Send alert for high-confidence signals"""
        try:
            alert_message = (
                f"🚨 New Signal: {recommendation.symbol}\n"
                f"Direction: {recommendation.side.upper()}\n"
                f"Entry: ${recommendation.suggested_entry_price:.2f}\n"
                f"Stop Loss: ${recommendation.suggested_stop_loss:.2f}\n"
                f"Take Profit: ${recommendation.suggested_take_profit:.2f}\n"
                f"Confidence: {confidence:.2%}\n"
                f"Strategy: {recommendation.strategy_name}"
            )
            
            # TODO: Integrate with notification service (Telegram/Discord/Email)
            logger.info(f"📢 Alert: {alert_message}")
            
        except Exception as e:
            logger.error(f"Error sending alert: {e}")

    async def get_active_recommendations(self, symbol: Optional[str] = None) -> List[SignalRecommendation]:
        """Get active signal recommendations"""
        try:
            async with get_db() as db:
                query = db.query(SignalRecommendation).filter(
                    SignalRecommendation.status == 'pending'
                )
                
                if symbol:
                    query = query.filter(SignalRecommendation.symbol == symbol)
                
                recommendations = await query.all()
                return recommendations
        except Exception as e:
            logger.error(f"Error getting active recommendations: {e}")
            return []

    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for signal generation"""
        try:
            return {
                'signals_generated': self.signals_generated,
                'high_confidence_signals': self.high_confidence_signals,
                'active_recommendations': len(self.active_signals),
                'model_id': self.current_model_id,
                'is_running': self.is_running,
                'timestamp': datetime.utcnow().isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting performance metrics: {e}")
            return {}

    async def health_check(self) -> Dict[str, Any]:
        """Health check for signal orchestrator"""
        return {
            'status': 'healthy' if self.is_running else 'stopped',
            'service': 'SignalOrchestrator',
            'signals_generated': self.signals_generated,
            'high_confidence_signals': self.high_confidence_signals,
            'timestamp': datetime.utcnow().isoformat()
        }


# Backward compatibility alias (DEPRECATED)
import warnings
TradingEngine = SignalOrchestrator
warnings.warn(
    "TradingEngine has been renamed to SignalOrchestrator. Please update your imports.",
    DeprecationWarning,
    stacklevel=2
)

