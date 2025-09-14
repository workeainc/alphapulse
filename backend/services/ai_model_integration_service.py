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
from dataclasses import dataclass

from ai.model_heads import ModelHeadsManager, ModelHeadResult, SignalDirection, ModelHead
from ai.consensus_manager import ConsensusManager, ConsensusResult
from services.real_data_integration_service import real_data_service, RealMarketData, RealSentimentData, RealTechnicalIndicators

logger = logging.getLogger(__name__)

@dataclass
class AIModelSignal:
    """AI model signal result"""
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

class AIModelIntegrationService:
    """Service for integrating AI/ML models with sophisticated interface"""
    
    def __init__(self):
        self.model_heads_manager = ModelHeadsManager()
        self.consensus_manager = ConsensusManager()
        self.real_data_service = real_data_service
        self.logger = logger
        
        # Cache for performance
        self._model_cache: Dict[str, AIModelSignal] = {}
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
            analysis_data = await self._prepare_analysis_data(market_data, sentiment_data, technical_data)
            market_data_dict = await self._prepare_market_data_dict(market_data, technical_data)
            
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
    
    async def _prepare_analysis_data(self, market_data: RealMarketData, sentiment_data: List[RealSentimentData], technical_data: Optional[RealTechnicalIndicators]) -> Dict[str, Any]:
        """Prepare analysis data for AI models"""
        try:
            # Fundamental analysis
            fundamental = {
                "market_regime": "Bullish" if market_data.price_change_24h > 0 else "Bearish" if market_data.price_change_24h < 0 else "Neutral",
                "price_change_24h": market_data.price_change_24h,
                "volume_change_24h": market_data.volume_change_24h,
                "fear_greed_index": market_data.fear_greed_index,
                "market_cap": market_data.market_cap,
                "data_quality": market_data.data_quality_score
            }
            
            # Sentiment analysis
            sentiment = {
                "avg_sentiment": np.mean([s.sentiment_score for s in sentiment_data]) if sentiment_data else 0.0,
                "sentiment_confidence": np.mean([s.confidence for s in sentiment_data]) if sentiment_data else 0.0,
                "sentiment_volume": len(sentiment_data),
                "sentiment_sources": list(set([s.source for s in sentiment_data])) if sentiment_data else [],
                "data_quality": np.mean([s.data_quality_score for s in sentiment_data]) if sentiment_data else 0.0
            }
            
            # Technical analysis
            technical = {}
            if technical_data:
                technical = {
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
            
            return {
                "fundamental": fundamental,
                "sentiment": sentiment,
                "technical": technical,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error preparing analysis data: {e}")
            return {}
    
    async def _prepare_market_data_dict(self, market_data: RealMarketData, technical_data: Optional[RealTechnicalIndicators]) -> Dict[str, Any]:
        """Prepare market data dictionary for AI models"""
        try:
            market_dict = {
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
        """Create AI signal from model results and consensus"""
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
            
            return AIModelSignal(
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
                data_quality=market_data.data_quality_score
            )
            
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
            analysis_data = await self._prepare_analysis_data(market_data, sentiment_data, technical_data)
            market_data_dict = await self._prepare_market_data_dict(market_data, technical_data)
            
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
                    "sentiment": analysis_data.get("sentiment", {}),
                    "technical": analysis_data.get("technical", {}),
                    "model_heads": head_analysis
                },
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting AI analysis for {symbol}: {e}")
            return self._get_fallback_analysis(symbol, timeframe)
    
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
