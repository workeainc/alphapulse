"""
AI/ML Model Integration Service
Connects sophisticated single-pair interface to existing AI/ML models
Phase 6: Real Data Integration
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from dataclasses import dataclass

from src.ai.model_heads import ModelHeadsManager, ModelHeadResult, SignalDirection, ModelHead
from src.ai.consensus_manager import ConsensusManager, ConsensusResult
from src.ai.signal_risk_enhancement import SignalRiskEnhancement, EnhancedSignalResult
from src.services.real_data_integration_service import real_data_service, RealMarketData, RealSentimentData, RealTechnicalIndicators
from src.services.mtf_entry_system import MTFEntrySystem

logger = logging.getLogger(__name__)

@dataclass
class AIModelSignal:
    """AI model signal result with MTF entry support and Step 8 enhancements"""
    symbol: str
    timeframe: str
    signal_direction: str
    confidence_score: float
    probability: float
    consensus_achieved: bool
    consensus_score: float
    agreeing_heads: List[str]
    model_reasoning: Dict[str, str]
    timestamp: datetime
    data_quality: float
    # MTF Entry fields
    entry_price: float = None
    stop_loss: float = None
    take_profit_levels: List[float] = None
    entry_timeframe: str = None
    entry_strategy: str = None
    entry_pattern: str = None
    entry_confidence: float = None
    fibonacci_level: float = None
    atr_entry_tf: float = None
    risk_reward_ratio: float = None
    # NEW Step 8 Fields
    consensus_probability: float = None
    consensus_confidence: float = None
    position_size_pct: float = None
    position_size_usd: float = None
    confidence_band: str = None
    expected_win_rate: float = None
    liquidation_risk_detected: bool = False
    extreme_leverage_detected: bool = False
    entry_zone_status: str = None
    signal_quality: str = None
    metadata: Dict[str, Any] = None

class AIModelIntegrationService:
    """Service for integrating AI/ML models with sophisticated interface"""
    
    def __init__(self):
        self.model_heads_manager = ModelHeadsManager()
        self.consensus_manager = ConsensusManager()
        self.real_data_service = real_data_service
        self.mtf_entry_system = MTFEntrySystem()
        self.logger = logger
        
        # NEW: Initialize risk enhancement service
        self.risk_enhancement = None  # Lazy initialization
        
        # Cache for performance
        self._model_cache: Dict[str, AIModelSignal] = {}
    
    async def _initialize_risk_enhancement(self):
        """Lazy initialization of risk enhancement service"""
        if self.risk_enhancement is None:
            try:
                # Try to import risk management components
                from src.app.services.risk_manager import RiskManager
                from src.strategies.derivatives_analyzer import DerivativesAnalyzer
                from src.strategies.enhanced_market_structure_engine import EnhancedMarketStructureEngine
                
                risk_manager = RiskManager()
                derivatives_analyzer = DerivativesAnalyzer()
                market_structure_engine = EnhancedMarketStructureEngine()
                
                self.risk_enhancement = SignalRiskEnhancement(
                    risk_manager=risk_manager,
                    derivatives_analyzer=derivatives_analyzer,
                    market_structure_engine=market_structure_engine,
                    config={'default_capital': 10000.0}
                )
                
                logger.info("✅ Risk Enhancement Service initialized in AI Model Integration")
            except Exception as e:
                logger.warning(f"⚠️ Could not initialize risk enhancement: {e}")
                # Create without external services
                self.risk_enhancement = SignalRiskEnhancement(config={'default_capital': 10000.0})
        self._cache_ttl = 60  # seconds
        
    async def generate_ai_signal(self, symbol: str, timeframe: str = "1h") -> Optional[AIModelSignal]:
        """Generate AI signal using existing model heads"""
        try:
            # Check cache first
            cache_key = f"{symbol}_{timeframe}"
            if cache_key in self._model_cache:
                cached_signal = self._model_cache[cache_key]
                if (datetime.now(timezone.utc) - cached_signal.timestamp).seconds < self._cache_ttl:
                    return cached_signal
            
            # Get real data
            market_data = await self.real_data_service.get_real_market_data(symbol, timeframe)
            sentiment_data = await self.real_data_service.get_real_sentiment_data(symbol, 24)
            technical_data = await self.real_data_service.get_real_technical_indicators(symbol, timeframe)
            
            if not market_data:
                self.logger.warning(f"No market data available for {symbol}")
                return None
            
            # Prepare data for AI models
            analysis_data = await self._prepare_analysis_data(symbol, timeframe, market_data, sentiment_data, technical_data)
            market_data_dict = await self._prepare_market_data_dict(symbol, timeframe, market_data, technical_data)
            
            # Run all model heads
            model_results = await self.model_heads_manager.analyze_all_heads(
                market_data_dict, 
                analysis_data
            )
            
            # Check consensus
            consensus_result = await self.consensus_manager.check_consensus(model_results)
            
            # Generate signal if consensus achieved
            if consensus_result.consensus_achieved:
                signal = await self._create_ai_signal(
                    symbol, timeframe, model_results, consensus_result, market_data
                )
                
                # Cache the signal
                self._model_cache[cache_key] = signal
                return signal
            else:
                self.logger.debug(f"No consensus achieved for {symbol}: {consensus_result}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error generating AI signal for {symbol}: {e}")
            return None
    
    async def _prepare_ohlcv_dataframe(self, symbol: str, timeframe: str, limit: int = 200) -> Optional[pd.DataFrame]:
        """Fetch OHLCV data and convert to DataFrame"""
        try:
            # Try to get from TimescaleDB first
            ohlcv_data = await self.real_data_service.get_ohlcv_data(symbol, timeframe, limit)
            
            if ohlcv_data and len(ohlcv_data) > 0:
                # Create DataFrame from database data
                df = pd.DataFrame(ohlcv_data)
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.sort_values('timestamp').reset_index(drop=True)
                self.logger.debug(f"✅ Created DataFrame with {len(df)} candles from database for {symbol}")
                return df
            
            # Fallback: try to fetch from exchange using CCXT
            try:
                import ccxt.async_support as ccxt
                exchange = ccxt.binance()
                ohlcv = await exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
                await exchange.close()
                
                if ohlcv and len(ohlcv) > 0:
                    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    df = df.sort_values('timestamp').reset_index(drop=True)
                    self.logger.debug(f"✅ Created DataFrame with {len(df)} candles from exchange for {symbol}")
                    return df
            except Exception as exchange_error:
                self.logger.warning(f"Failed to fetch from exchange: {exchange_error}")
            
            self.logger.warning(f"No OHLCV data available for {symbol} {timeframe}")
            return None
            
        except Exception as e:
            self.logger.error(f"Error creating OHLCV DataFrame: {e}")
            return None
    
    async def _prepare_analysis_data(self, symbol: str, timeframe: str, market_data: RealMarketData, sentiment_data: List[RealSentimentData], technical_data: Optional[RealTechnicalIndicators]) -> Dict[str, Any]:
        """Prepare analysis data for AI models with all required keys"""
        try:
            # Get OHLCV DataFrame - CRITICAL for all model heads
            ohlcv_df = await self._prepare_ohlcv_dataframe(symbol, timeframe, limit=200)
            
            # Fundamental analysis
            fundamental = {
                "market_regime": "Bullish" if market_data.price_change_24h > 0 else "Bearish" if market_data.price_change_24h < 0 else "Neutral",
                "price_change_24h": market_data.price_change_24h,
                "volume_change_24h": market_data.volume_change_24h,
                "fear_greed_index": market_data.fear_greed_index,
                "market_cap": market_data.market_cap,
                "data_quality": market_data.data_quality_score
            }
            
            # Sentiment analysis - FIXED KEY NAMES for Head B
            sentiment_analysis = {
                "overall_sentiment": np.mean([s.sentiment_score for s in sentiment_data]) if sentiment_data else 0.0,  # FIXED: was "avg_sentiment"
                "confidence": np.mean([s.confidence for s in sentiment_data]) if sentiment_data else 0.0,  # FIXED: was "sentiment_confidence"
                "sentiment_volume": len(sentiment_data),
                "sentiment_sources": list(set([s.source for s in sentiment_data])) if sentiment_data else [],
                "data_quality": np.mean([s.data_quality_score for s in sentiment_data]) if sentiment_data else 0.0
            }
            
            # Technical analysis - FIXED KEY NAMES and added trend/strength for Head D
            technical_analysis = {}
            if technical_data:
                # Calculate trend based on moving averages
                trend = "bullish" if technical_data.sma_20 > technical_data.sma_50 else "bearish"
                
                # Calculate strength based on RSI
                rsi = technical_data.rsi
                if rsi > 70 or rsi < 30:
                    strength = "strong"
                elif rsi > 60 or rsi < 40:
                    strength = "moderate"
                else:
                    strength = "normal"
                
                technical_analysis = {
                    "trend": trend,  # ADDED for Head D
                    "strength": strength,  # ADDED for Head D
                    "rsi": technical_data.rsi,
                    "macd": technical_data.macd,
                    "macd_signal": technical_data.macd_signal,
                    "macd_histogram": technical_data.macd_histogram,
                    "sma_20": technical_data.sma_20,
                    "sma_50": technical_data.sma_50,
                    "ema_12": technical_data.ema_12,
                    "ema_26": technical_data.ema_26,
                    "bollinger_upper": technical_data.bollinger_upper,
                    "bollinger_lower": technical_data.bollinger_lower,
                    "bollinger_middle": technical_data.bollinger_middle,
                    "volume_sma": technical_data.volume_sma,
                    "data_quality": technical_data.data_quality_score
                }
            else:
                # Provide defaults if no technical data
                technical_analysis = {
                    "trend": "neutral",
                    "strength": "normal"
                }
            
            # Volume analysis - ADDED for Head C fallback
            volume_analysis = {
                "volume_trend": "increasing" if market_data.volume_change_24h > 0 else "decreasing" if market_data.volume_change_24h < 0 else "stable",
                "volume_strength": "strong" if abs(market_data.volume_change_24h) > 0.2 else "weak" if abs(market_data.volume_change_24h) < 0.05 else "normal"
            }
            
            return {
                "dataframe": ohlcv_df,  # CRITICAL: Added DataFrame for all heads
                "fundamental": fundamental,
                "sentiment_analysis": sentiment_analysis,  # FIXED: was "sentiment"
                "technical_analysis": technical_analysis,  # FIXED: was "technical"
                "volume_analysis": volume_analysis,  # ADDED for Head C
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error preparing analysis data: {e}")
            return {
                "dataframe": None,
                "fundamental": {},
                "sentiment_analysis": {"overall_sentiment": 0.0, "confidence": 0.5},
                "technical_analysis": {"trend": "neutral", "strength": "normal"},
                "volume_analysis": {"volume_trend": "stable", "volume_strength": "normal"}
            }
    
    async def _prepare_market_data_dict(self, symbol: str, timeframe: str, market_data: RealMarketData, technical_data: Optional[RealTechnicalIndicators]) -> Dict[str, Any]:
        """Prepare market data dictionary for AI models"""
        try:
            market_dict = {
                "symbol": symbol,  # ADDED for engine lookup
                "timeframe": timeframe,  # ADDED for engine configuration
                "current_price": market_data.price,
                "volume_24h": market_data.volume_24h,
                "price_change_24h": market_data.price_change_24h,
                "volume_change_24h": market_data.volume_change_24h,
                "market_cap": market_data.market_cap,
                "fear_greed_index": market_data.fear_greed_index,
                "timestamp": market_data.timestamp.isoformat(),
                "data_quality_score": market_data.data_quality_score
            }
            
            # Add technical indicators if available
            if technical_data:
                market_dict["indicators"] = {
                    "sma_20": technical_data.sma_20,
                    "sma_50": technical_data.sma_50,
                    "rsi_14": technical_data.rsi,
                    "macd": technical_data.macd,
                    "macd_signal": technical_data.macd_signal,
                    "macd_histogram": technical_data.macd_histogram,
                    "ema_12": technical_data.ema_12,
                    "ema_26": technical_data.ema_26,
                    "bollinger_upper": technical_data.bollinger_upper,
                    "bollinger_lower": technical_data.bollinger_lower,
                    "bollinger_middle": technical_data.bollinger_middle,
                    "volume_sma": technical_data.volume_sma
                }
            else:
                market_dict["indicators"] = {}
            
            return market_dict
            
        except Exception as e:
            self.logger.error(f"Error preparing market data dict: {e}")
            return {}
    
    async def _create_ai_signal(self, symbol: str, timeframe: str, model_results: List[ModelHeadResult], consensus_result: ConsensusResult, market_data: RealMarketData) -> AIModelSignal:
        """Create AI signal from model results and consensus with Step 8 enhancements"""
        try:
            # Calculate weighted confidence and probability
            weighted_confidence = 0.0
            weighted_probability = 0.0
            total_weight = 0.0
            
            model_reasoning = {}
            
            for result in model_results:
                if result.head_type in consensus_result.agreeing_heads:
                    weight = self.consensus_manager.head_weights.get(result.head_type, 0.25)
                    weighted_confidence += result.confidence * weight
                    weighted_probability += result.probability * weight
                    total_weight += weight
                    
                    model_reasoning[result.head_type.value] = result.reasoning
            
            if total_weight > 0:
                weighted_confidence /= total_weight
                weighted_probability /= total_weight
            
            # Determine signal direction
            signal_direction = consensus_result.consensus_direction.value if consensus_result.consensus_direction else "flat"
            
            # Calculate overall confidence (consensus + model confidence)
            overall_confidence = (consensus_result.consensus_score + weighted_confidence) / 2
            
            # Base signal object
            base_signal = AIModelSignal(
                symbol=symbol,
                timeframe=timeframe,
                signal_direction=signal_direction,
                confidence_score=overall_confidence,
                probability=weighted_probability,
                consensus_achieved=consensus_result.consensus_achieved,
                consensus_score=consensus_result.consensus_score,
                agreeing_heads=[head.value for head in consensus_result.agreeing_heads],
                model_reasoning=model_reasoning,
                timestamp=datetime.now(timezone.utc),
                data_quality=market_data.data_quality_score,
                # NEW: Add consensus fields from Step 5
                consensus_probability=consensus_result.consensus_probability,
                consensus_confidence=consensus_result.consensus_confidence
            )
            
            # NEW: Apply Step 8 risk enhancement if available
            await self._initialize_risk_enhancement()
            
            if self.risk_enhancement and signal_direction in ['long', 'short']:
                try:
                    # Prepare market data dict for risk checks
                    market_data_dict = {
                        'current_price': market_data.current_price,
                        'spot_price': market_data.current_price,
                        'perpetual_price': market_data.current_price * 1.001,  # Estimate if not available
                        'leverage': 1,  # Conservative default
                        'timeframe': timeframe
                    }
                    
                    # Estimate entry and stop loss if not set
                    entry_price = market_data.current_price
                    atr_estimate = market_data.current_price * 0.02  # 2% ATR estimate
                    
                    if signal_direction == 'long':
                        stop_loss = entry_price - (atr_estimate * 1.5)
                    else:
                        stop_loss = entry_price + (atr_estimate * 1.5)
                    
                    # Apply risk enhancement
                    enhanced_signal = await self.risk_enhancement.enhance_signal(
                        symbol=symbol,
                        direction=signal_direction.upper(),
                        entry_price=entry_price,
                        stop_loss=stop_loss,
                        consensus_result=consensus_result,
                        market_data=market_data_dict,
                        available_capital=10000.0  # Default capital
                    )
                    
                    if enhanced_signal:
                        # Update signal with enhancement fields
                        base_signal.entry_price = enhanced_signal.entry_price
                        base_signal.stop_loss = enhanced_signal.stop_loss
                        base_signal.take_profit_levels = [enhanced_signal.take_profit]
                        base_signal.risk_reward_ratio = enhanced_signal.risk_reward_ratio
                        base_signal.position_size_pct = enhanced_signal.position_size_pct
                        base_signal.position_size_usd = enhanced_signal.position_size_usd
                        base_signal.confidence_band = enhanced_signal.confidence_band
                        base_signal.expected_win_rate = enhanced_signal.expected_win_rate
                        base_signal.liquidation_risk_detected = enhanced_signal.liquidation_risk_detected
                        base_signal.extreme_leverage_detected = enhanced_signal.extreme_leverage_detected
                        base_signal.entry_zone_status = enhanced_signal.entry_zone_status
                        base_signal.signal_quality = enhanced_signal.signal_quality
                        base_signal.entry_strategy = enhanced_signal.entry_strategy
                        base_signal.metadata = enhanced_signal.metadata
                        
                        logger.info(f"✅ Signal enhanced: {enhanced_signal.signal_quality} quality | "
                                   f"Size: ${enhanced_signal.position_size_usd:.2f} | "
                                   f"R:R: {enhanced_signal.risk_reward_ratio:.2f}:1")
                    else:
                        logger.warning(f"⚠️ Risk enhancement returned None (signal may have been filtered)")
                        return None
                        
                except Exception as e:
                    logger.warning(f"⚠️ Risk enhancement failed, using base signal: {e}")
            
            return base_signal
            
        except Exception as e:
            self.logger.error(f"Error creating AI signal: {e}")
            return None
    
    async def get_ai_confidence(self, symbol: str, timeframe: str = "1h") -> Dict[str, Any]:
        """Get AI model confidence for single pair"""
        try:
            # Get AI signal
            ai_signal = await self.generate_ai_signal(symbol, timeframe)
            
            if ai_signal:
                return {
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "ai_confidence": ai_signal.confidence_score,
                    "ai_probability": ai_signal.probability,
                    "consensus_achieved": ai_signal.consensus_achieved,
                    "consensus_score": ai_signal.consensus_score,
                    "agreeing_heads": ai_signal.agreeing_heads,
                    "model_reasoning": ai_signal.model_reasoning,
                    "data_quality": ai_signal.data_quality,
                    "timestamp": ai_signal.timestamp.isoformat()
                }
            else:
                return {
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "ai_confidence": 0.0,
                    "ai_probability": 0.0,
                    "consensus_achieved": False,
                    "consensus_score": 0.0,
                    "agreeing_heads": [],
                    "model_reasoning": {},
                    "data_quality": 0.0,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
                
        except Exception as e:
            self.logger.error(f"Error getting AI confidence for {symbol}: {e}")
            return {
                "symbol": symbol,
                "timeframe": timeframe,
                "ai_confidence": 0.0,
                "ai_probability": 0.0,
                "consensus_achieved": False,
                "consensus_score": 0.0,
                "agreeing_heads": [],
                "model_reasoning": {},
                "data_quality": 0.0,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
    
    async def get_ai_analysis(self, symbol: str, timeframe: str = "1h") -> Dict[str, Any]:
        """Get comprehensive AI analysis for single pair"""
        try:
            # Get real data
            market_data = await self.real_data_service.get_real_market_data(symbol, timeframe)
            sentiment_data = await self.real_data_service.get_real_sentiment_data(symbol, 24)
            technical_data = await self.real_data_service.get_real_technical_indicators(symbol, timeframe)
            
            if not market_data:
                return self._get_fallback_analysis(symbol, timeframe)
            
            # Prepare analysis data
            analysis_data = await self._prepare_analysis_data(symbol, timeframe, market_data, sentiment_data, technical_data)
            market_data_dict = await self._prepare_market_data_dict(symbol, timeframe, market_data, technical_data)
            
            # Run model heads for detailed analysis
            model_results = await self.model_heads_manager.analyze_all_heads(
                market_data_dict, 
                analysis_data
            )
            
            # Organize results by head type
            head_analysis = {}
            for result in model_results:
                head_analysis[result.head_type.value] = {
                    "direction": result.direction.value,
                    "probability": result.probability,
                    "confidence": result.confidence,
                    "features_used": result.features_used,
                    "reasoning": result.reasoning
                }
            
            return {
                "pair": symbol,
                "timeframe": timeframe,
                "ai_analysis": {
                    "fundamental": analysis_data.get("fundamental", {}),
                    "sentiment": analysis_data.get("sentiment_analysis", {}),  # FIXED: use correct key
                    "technical": analysis_data.get("technical_analysis", {}),  # FIXED: use correct key
                    "model_heads": head_analysis
                },
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting AI analysis for {symbol}: {e}")
            return self._get_fallback_analysis(symbol, timeframe)
    
    async def generate_ai_signal_with_mtf_entry(
        self, 
        symbol: str, 
        signal_timeframe: str = "1h",
        entry_timeframe: Optional[str] = None
    ) -> Optional[AIModelSignal]:
        """
        Generate AI signal with Multi-Timeframe Entry Refinement
        
        Industry Standard Approach:
        1. Higher TF (signal_timeframe): Determine trend/bias using 9-head consensus
        2. Lower TF (entry_timeframe): Find precise entry using Fibonacci/EMA/Order Blocks
        
        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            signal_timeframe: Higher TF for trend ('1h', '4h', '1d')
            entry_timeframe: Lower TF for entry (auto if None)
        
        Returns:
            AIModelSignal with refined entry price, stop loss, and targets
        """
        try:
            # Auto-determine entry timeframe
            if entry_timeframe is None:
                entry_timeframe = self.mtf_entry_system.get_entry_timeframe(signal_timeframe)
            
            self.logger.info(
                f"🎯 MTF Analysis: {symbol} | "
                f"Signal TF: {signal_timeframe} | Entry TF: {entry_timeframe}"
            )
            
            # STEP 1: Generate signal on higher timeframe
            higher_tf_signal = await self.generate_ai_signal(symbol, timeframe=signal_timeframe)
            
            if not higher_tf_signal:
                self.logger.debug(f"No consensus on {signal_timeframe} for {symbol}")
                return None
            
            if higher_tf_signal.signal_direction == 'FLAT':
                self.logger.debug(f"Signal direction is FLAT for {symbol}")
                return None
            
            # STEP 2: Get current market price
            market_data = await self.real_data_service.get_real_market_data(symbol, timeframe=signal_timeframe)
            if not market_data:
                self.logger.warning(f"No market data for {symbol}")
                return higher_tf_signal
            
            current_price = market_data.price
            
            # STEP 3: Get lower timeframe data for entry refinement
            entry_df = await self._prepare_ohlcv_dataframe(symbol, entry_timeframe, limit=200)
            
            if entry_df is None or len(entry_df) < 50:
                self.logger.warning(f"Insufficient entry timeframe data for {symbol}")
                # Return signal with market price as entry
                higher_tf_signal.entry_price = current_price
                higher_tf_signal.entry_timeframe = entry_timeframe
                return higher_tf_signal
            
            # STEP 4: Refine entry using MTF Entry System
            entry_analysis = await self.mtf_entry_system.refine_entry(
                symbol=symbol,
                entry_df=entry_df,
                signal_direction=higher_tf_signal.signal_direction,
                current_price=current_price,
                signal_timeframe=signal_timeframe
            )
            
            # STEP 5: Apply entry refinement to signal
            if entry_analysis:
                higher_tf_signal.entry_price = entry_analysis['refined_entry']
                higher_tf_signal.stop_loss = entry_analysis['refined_stop']
                higher_tf_signal.take_profit_levels = entry_analysis['refined_targets']
                higher_tf_signal.entry_timeframe = entry_timeframe
                higher_tf_signal.entry_strategy = entry_analysis['entry_strategy']
                higher_tf_signal.entry_pattern = entry_analysis['entry_pattern']
                higher_tf_signal.entry_confidence = entry_analysis['entry_confidence']
                higher_tf_signal.fibonacci_level = entry_analysis.get('fibonacci_level')
                higher_tf_signal.atr_entry_tf = entry_analysis['atr']
                higher_tf_signal.risk_reward_ratio = entry_analysis['risk_reward_ratio']
                
                # Add metadata
                higher_tf_signal.metadata = {
                    'mtf_analysis': {
                        'signal_timeframe': signal_timeframe,
                        'entry_timeframe': entry_timeframe,
                        'entry_confidence': entry_analysis['entry_confidence'],
                        'entry_pattern': entry_analysis['entry_pattern'],
                        'entry_strategy': entry_analysis['entry_strategy'],
                        'fibonacci_level': entry_analysis.get('fibonacci_level'),
                        'ema_levels': entry_analysis.get('ema_levels', {}),
                        'volume_confirmed': entry_analysis.get('volume_confirmed', False)
                    }
                }
                
                self.logger.info(
                    f"✅ MTF Entry refined: {symbol} | "
                    f"Strategy: {entry_analysis['entry_strategy']} | "
                    f"Entry: ${entry_analysis['refined_entry']:.2f} | "
                    f"R:R: {entry_analysis['risk_reward_ratio']:.2f}"
                )
            else:
                # No entry refinement possible, use market price
                higher_tf_signal.entry_price = current_price
                higher_tf_signal.entry_timeframe = entry_timeframe
                self.logger.warning(f"Could not refine entry for {symbol}, using market price")
            
            return higher_tf_signal
            
        except Exception as e:
            self.logger.error(f"Error in MTF signal generation for {symbol}: {e}")
            return None
    
    def _get_fallback_analysis(self, symbol: str, timeframe: str) -> Dict[str, Any]:
        """Fallback analysis when AI models fail"""
        return {
            "pair": symbol,
            "timeframe": timeframe,
            "ai_analysis": {
                "fundamental": {
                    "market_regime": "Neutral",
                    "price_change_24h": 0.0,
                    "volume_change_24h": 0.0,
                    "fear_greed_index": 50,
                    "market_cap": 0.0,
                    "data_quality": 0.0
                },
                "sentiment": {
                    "avg_sentiment": 0.0,
                    "sentiment_confidence": 0.0,
                    "sentiment_volume": 0,
                    "sentiment_sources": [],
                    "data_quality": 0.0
                },
                "technical": {
                    "rsi": 50.0,
                    "macd": 0.0,
                    "macd_signal": 0.0,
                    "macd_histogram": 0.0,
                    "sma_20": 0.0,
                    "sma_50": 0.0,
                    "ema_12": 0.0,
                    "ema_26": 0.0,
                    "bollinger_upper": 0.0,
                    "bollinger_lower": 0.0,
                    "bollinger_middle": 0.0,
                    "volume_sma": 0.0,
                    "data_quality": 0.0
                },
                "model_heads": {}
            },
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

# Global instance
ai_model_service = AIModelIntegrationService()
