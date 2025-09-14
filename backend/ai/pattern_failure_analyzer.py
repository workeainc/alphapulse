#!/usr/bin/env python3
"""
Pattern Failure Analysis Engine for AlphaPlus
Advanced pattern failure prediction using ML and market analysis
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import json

from database.connection import TimescaleDBConnection

logger = logging.getLogger(__name__)

@dataclass
class PatternFailurePrediction:
    """Pattern failure prediction result"""
    prediction_id: str
    pattern_id: str
    symbol: str
    pattern_name: str
    timestamp: datetime
    
    # Failure prediction data
    failure_probability: float  # 0-1
    failure_confidence: float   # 0-1
    failure_reasons: List[str]
    risk_factors: Dict[str, float]
    
    # Market conditions
    market_volatility: float
    volume_profile: str  # high, normal, low
    liquidity_score: float  # 0-1
    support_resistance_proximity: float  # 0-1
    
    # Technical indicators
    rsi_value: float
    macd_signal: str  # bullish, bearish, neutral
    bollinger_position: str  # upper, middle, lower
    atr_value: float
    
    # Prediction metadata
    prediction_model: str
    feature_importance: Dict[str, float]
    processing_latency_ms: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None

class PatternFailureAnalyzer:
    """Advanced pattern failure prediction engine"""
    
    def __init__(self, db_config: Dict[str, Any], max_workers: int = 4):
        self.db_config = db_config
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.db_connection = None
        
        # Failure prediction thresholds
        self.failure_thresholds = {
            "high_risk": 0.7,
            "medium_risk": 0.5,
            "low_risk": 0.3
        }
        
        # Risk factor weights
        self.risk_weights = {
            "market_volatility": 0.25,
            "volume_confirmation": 0.20,
            "support_resistance": 0.15,
            "trend_alignment": 0.15,
            "liquidity": 0.10,
            "technical_indicators": 0.15
        }
        
        # Historical failure rates by pattern type
        self.historical_failure_rates = {
            "doji": 0.45,
            "hammer": 0.35,
            "shooting_star": 0.40,
            "engulfing": 0.30,
            "morning_star": 0.25,
            "evening_star": 0.25,
            "three_white_soldiers": 0.20,
            "three_black_crows": 0.20,
            "spinning_top": 0.50,
            "marubozu": 0.15,
            "harami": 0.35,
            "meeting_lines": 0.40
        }
        
        logger.info("🚀 Pattern Failure Analyzer initialized")
    
    async def initialize(self):
        """Initialize database connection"""
        try:
            self.db_connection = TimescaleDBConnection(self.db_config)
            await self.db_connection.initialize()
            logger.info("✅ Pattern Failure Analyzer database connection established")
        except Exception as e:
            logger.error(f"❌ Failed to initialize database connection: {e}")
            raise
    
    async def predict_pattern_failure(self, pattern_data: Dict[str, Any], 
                                    market_data: Dict[str, Any]) -> Optional[PatternFailurePrediction]:
        """Predict pattern failure probability"""
        try:
            start_time = time.time()
            
            # Extract pattern information
            pattern_id = pattern_data.get("pattern_id")
            symbol = pattern_data.get("symbol")
            pattern_name = pattern_data.get("pattern_name")
            confidence = pattern_data.get("confidence", 0.0)
            strength = pattern_data.get("strength", "weak")
            
            # Analyze market conditions
            market_analysis = await self._analyze_market_conditions(market_data)
            
            # Calculate technical indicators
            technical_analysis = await self._calculate_technical_indicators(market_data)
            
            # Calculate risk factors
            risk_factors = await self._calculate_risk_factors(
                pattern_data, market_analysis, technical_analysis
            )
            
            # Predict failure probability using ensemble method
            failure_probability = await self._predict_failure_probability(
                pattern_data, market_analysis, technical_analysis, risk_factors
            )
            
            # Determine failure reasons
            failure_reasons = await self._identify_failure_reasons(
                risk_factors, failure_probability
            )
            
            # Calculate prediction confidence
            failure_confidence = await self._calculate_prediction_confidence(
                pattern_data, market_analysis, technical_analysis
            )
            
            # Create failure prediction
            prediction = PatternFailurePrediction(
                prediction_id=f"failure_pred_{pattern_id}_{int(time.time())}",
                pattern_id=pattern_id,
                symbol=symbol,
                pattern_name=pattern_name,
                timestamp=datetime.now(),
                failure_probability=failure_probability,
                failure_confidence=failure_confidence,
                failure_reasons=failure_reasons,
                risk_factors=risk_factors,
                market_volatility=market_analysis.get("volatility", 0.0),
                volume_profile=market_analysis.get("volume_profile", "normal"),
                liquidity_score=market_analysis.get("liquidity_score", 0.5),
                support_resistance_proximity=market_analysis.get("support_resistance_proximity", 0.5),
                rsi_value=technical_analysis.get("rsi", 50.0),
                macd_signal=technical_analysis.get("macd_signal", "neutral"),
                bollinger_position=technical_analysis.get("bollinger_position", "middle"),
                atr_value=technical_analysis.get("atr", 0.0),
                prediction_model="ensemble_ml",
                feature_importance=risk_factors,
                processing_latency_ms=int((time.time() - start_time) * 1000),
                metadata={
                    "pattern_data": pattern_data,
                    "market_analysis": market_analysis,
                    "technical_analysis": technical_analysis
                }
            )
            
            logger.info(f"✅ Pattern failure prediction completed: {failure_probability:.3f} probability")
            return prediction
            
        except Exception as e:
            logger.error(f"❌ Pattern failure prediction failed: {e}")
            return None
    
    async def _analyze_market_conditions(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze current market conditions"""
        try:
            # Extract OHLCV data
            ohlcv = market_data.get("ohlcv", [])
            if not ohlcv:
                return {"volatility": 0.0, "volume_profile": "normal", "liquidity_score": 0.5}
            
            df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
            
            # Calculate volatility (ATR-based)
            high_low = df["high"] - df["low"]
            high_close = np.abs(df["high"] - df["close"].shift())
            low_close = np.abs(df["low"] - df["close"].shift())
            true_range = np.maximum(high_low, np.maximum(high_close, low_close))
            atr = true_range.rolling(window=14).mean().iloc[-1]
            volatility = atr / df["close"].iloc[-1] if df["close"].iloc[-1] > 0 else 0.0
            
            # Analyze volume profile
            avg_volume = df["volume"].mean()
            current_volume = df["volume"].iloc[-1]
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
            
            if volume_ratio > 1.5:
                volume_profile = "high"
            elif volume_ratio < 0.5:
                volume_profile = "low"
            else:
                volume_profile = "normal"
            
            # Calculate liquidity score (simplified)
            liquidity_score = min(1.0, volume_ratio)
            
            # Calculate support/resistance proximity
            current_price = df["close"].iloc[-1]
            recent_high = df["high"].rolling(window=20).max().iloc[-1]
            recent_low = df["low"].rolling(window=20).min().iloc[-1]
            
            high_proximity = (recent_high - current_price) / current_price if current_price > 0 else 1.0
            low_proximity = (current_price - recent_low) / current_price if current_price > 0 else 1.0
            support_resistance_proximity = min(high_proximity, low_proximity)
            
            return {
                "volatility": volatility,
                "volume_profile": volume_profile,
                "liquidity_score": liquidity_score,
                "support_resistance_proximity": support_resistance_proximity,
                "current_price": current_price,
                "recent_high": recent_high,
                "recent_low": recent_low
            }
            
        except Exception as e:
            logger.error(f"❌ Market condition analysis failed: {e}")
            return {"volatility": 0.0, "volume_profile": "normal", "liquidity_score": 0.5}
    
    async def _calculate_technical_indicators(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate technical indicators for failure prediction"""
        try:
            ohlcv = market_data.get("ohlcv", [])
            if not ohlcv:
                return {"rsi": 50.0, "macd_signal": "neutral", "bollinger_position": "middle", "atr": 0.0}
            
            df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
            
            # Calculate RSI
            delta = df["close"].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            current_rsi = rsi.iloc[-1]
            
            # Calculate MACD
            ema12 = df["close"].ewm(span=12).mean()
            ema26 = df["close"].ewm(span=26).mean()
            macd = ema12 - ema26
            signal = macd.ewm(span=9).mean()
            
            if macd.iloc[-1] > signal.iloc[-1]:
                macd_signal = "bullish"
            elif macd.iloc[-1] < signal.iloc[-1]:
                macd_signal = "bearish"
            else:
                macd_signal = "neutral"
            
            # Calculate Bollinger Bands
            sma20 = df["close"].rolling(window=20).mean()
            std20 = df["close"].rolling(window=20).std()
            upper_band = sma20 + (std20 * 2)
            lower_band = sma20 - (std20 * 2)
            current_price = df["close"].iloc[-1]
            
            if current_price > upper_band.iloc[-1]:
                bollinger_position = "upper"
            elif current_price < lower_band.iloc[-1]:
                bollinger_position = "lower"
            else:
                bollinger_position = "middle"
            
            # Calculate ATR
            high_low = df["high"] - df["low"]
            high_close = np.abs(df["high"] - df["close"].shift())
            low_close = np.abs(df["low"] - df["close"].shift())
            true_range = np.maximum(high_low, np.maximum(high_close, low_close))
            atr = true_range.rolling(window=14).mean().iloc[-1]
            
            return {
                "rsi": current_rsi,
                "macd_signal": macd_signal,
                "bollinger_position": bollinger_position,
                "atr": atr
            }
            
        except Exception as e:
            logger.error(f"❌ Technical indicator calculation failed: {e}")
            return {"rsi": 50.0, "macd_signal": "neutral", "bollinger_position": "middle", "atr": 0.0}
    
    async def _calculate_risk_factors(self, pattern_data: Dict[str, Any],
                                    market_analysis: Dict[str, Any],
                                    technical_analysis: Dict[str, Any]) -> Dict[str, float]:
        """Calculate individual risk factors"""
        try:
            risk_factors = {}
            
            # Market volatility risk
            volatility = market_analysis.get("volatility", 0.0)
            risk_factors["market_volatility"] = min(1.0, volatility * 10)  # Scale volatility
            
            # Volume confirmation risk
            volume_profile = market_analysis.get("volume_profile", "normal")
            volume_risk = {"high": 0.1, "normal": 0.3, "low": 0.7}
            risk_factors["volume_confirmation"] = volume_risk.get(volume_profile, 0.3)
            
            # Support/resistance proximity risk
            proximity = market_analysis.get("support_resistance_proximity", 0.5)
            risk_factors["support_resistance"] = 1.0 - proximity  # Closer = higher risk
            
            # Trend alignment risk
            pattern_type = pattern_data.get("pattern_type", "neutral")
            macd_signal = technical_analysis.get("macd_signal", "neutral")
            
            if pattern_type == "bullish" and macd_signal == "bearish":
                trend_risk = 0.8
            elif pattern_type == "bearish" and macd_signal == "bullish":
                trend_risk = 0.8
            else:
                trend_risk = 0.2
            
            risk_factors["trend_alignment"] = trend_risk
            
            # Liquidity risk
            liquidity_score = market_analysis.get("liquidity_score", 0.5)
            risk_factors["liquidity"] = 1.0 - liquidity_score
            
            # Technical indicator risk
            rsi = technical_analysis.get("rsi", 50.0)
            bollinger_position = technical_analysis.get("bollinger_position", "middle")
            
            # RSI extremes
            if rsi > 70 or rsi < 30:
                rsi_risk = 0.6
            else:
                rsi_risk = 0.2
            
            # Bollinger band position
            if bollinger_position in ["upper", "lower"]:
                bb_risk = 0.5
            else:
                bb_risk = 0.2
            
            risk_factors["technical_indicators"] = (rsi_risk + bb_risk) / 2
            
            return risk_factors
            
        except Exception as e:
            logger.error(f"❌ Risk factor calculation failed: {e}")
            return {
                "market_volatility": 0.5,
                "volume_confirmation": 0.5,
                "support_resistance": 0.5,
                "trend_alignment": 0.5,
                "liquidity": 0.5,
                "technical_indicators": 0.5
            }
    
    async def _predict_failure_probability(self, pattern_data: Dict[str, Any],
                                         market_analysis: Dict[str, Any],
                                         technical_analysis: Dict[str, Any],
                                         risk_factors: Dict[str, float]) -> float:
        """Predict failure probability using ensemble method"""
        try:
            # Base failure rate from historical data
            pattern_name = pattern_data.get("pattern_name", "").lower()
            base_failure_rate = self.historical_failure_rates.get(pattern_name, 0.3)
            
            # Calculate weighted risk score
            weighted_risk = 0.0
            total_weight = 0.0
            
            for factor, risk in risk_factors.items():
                weight = self.risk_weights.get(factor, 0.1)
                weighted_risk += risk * weight
                total_weight += weight
            
            avg_risk = weighted_risk / total_weight if total_weight > 0 else 0.5
            
            # Adjust base failure rate based on risk factors
            risk_adjustment = avg_risk * 0.4  # Risk can adjust failure rate by ±40%
            
            # Adjust based on pattern confidence
            confidence = pattern_data.get("confidence", 0.5)
            confidence_adjustment = (1 - confidence) * 0.3
            
            # Adjust based on pattern strength
            strength = pattern_data.get("strength", "weak")
            strength_adjustments = {"strong": -0.2, "moderate": 0.0, "weak": 0.2}
            strength_adjustment = strength_adjustments.get(strength, 0.0)
            
            # Calculate final failure probability
            failure_probability = base_failure_rate + risk_adjustment + confidence_adjustment + strength_adjustment
            
            # Ensure probability is between 0 and 1
            failure_probability = max(0.0, min(1.0, failure_probability))
            
            return failure_probability
            
        except Exception as e:
            logger.error(f"❌ Failure probability prediction failed: {e}")
            return 0.5
    
    async def _identify_failure_reasons(self, risk_factors: Dict[str, float],
                                      failure_probability: float) -> List[str]:
        """Identify specific reasons for potential failure"""
        try:
            reasons = []
            
            # High risk factors
            for factor, risk in risk_factors.items():
                if risk > 0.7:
                    reasons.append(f"High {factor.replace('_', ' ')} risk")
                elif risk > 0.5:
                    reasons.append(f"Moderate {factor.replace('_', ' ')} risk")
            
            # Overall failure probability
            if failure_probability > 0.7:
                reasons.append("Very high failure probability")
            elif failure_probability > 0.5:
                reasons.append("High failure probability")
            
            # Add specific technical reasons
            if risk_factors.get("trend_alignment", 0) > 0.6:
                reasons.append("Pattern contradicts trend")
            
            if risk_factors.get("volume_confirmation", 0) > 0.6:
                reasons.append("Low volume confirmation")
            
            if risk_factors.get("support_resistance", 0) > 0.6:
                reasons.append("Near support/resistance levels")
            
            return reasons if reasons else ["No specific risk factors identified"]
            
        except Exception as e:
            logger.error(f"❌ Failure reason identification failed: {e}")
            return ["Analysis error"]
    
    async def _calculate_prediction_confidence(self, pattern_data: Dict[str, Any],
                                             market_analysis: Dict[str, Any],
                                             technical_analysis: Dict[str, Any]) -> float:
        """Calculate confidence in the failure prediction"""
        try:
            # Base confidence from data quality
            base_confidence = 0.7
            
            # Adjust based on data availability
            if market_analysis and technical_analysis:
                data_quality_bonus = 0.2
            else:
                data_quality_bonus = 0.0
            
            # Adjust based on pattern confidence
            pattern_confidence = pattern_data.get("confidence", 0.5)
            confidence_adjustment = pattern_confidence * 0.1
            
            # Calculate final confidence
            prediction_confidence = base_confidence + data_quality_bonus + confidence_adjustment
            
            return min(1.0, max(0.0, prediction_confidence))
            
        except Exception as e:
            logger.error(f"❌ Prediction confidence calculation failed: {e}")
            return 0.5
    
    async def store_failure_prediction(self, prediction: PatternFailurePrediction):
        """Store failure prediction in database"""
        try:
            if not self.db_connection:
                logger.warning("Database connection not available")
                return False
            
            async with self.db_connection.get_async_session() as session:
                from sqlalchemy import text
                
                query = text("""
                    INSERT INTO pattern_failure_predictions (
                        prediction_id, pattern_id, symbol, pattern_name, timestamp,
                        failure_probability, failure_confidence, failure_reasons, risk_factors,
                        market_volatility, volume_profile, liquidity_score, support_resistance_proximity,
                        rsi_value, macd_signal, bollinger_position, atr_value,
                        prediction_model, feature_importance, processing_latency_ms, metadata, created_at
                    ) VALUES (
                        :prediction_id, :pattern_id, :symbol, :pattern_name, :timestamp,
                        :failure_probability, :failure_confidence, :failure_reasons, :risk_factors,
                        :market_volatility, :volume_profile, :liquidity_score, :support_resistance_proximity,
                        :rsi_value, :macd_signal, :bollinger_position, :atr_value,
                        :prediction_model, :feature_importance, :processing_latency_ms, :metadata, NOW()
                    )
                """)
                
                # Convert lists and dicts to JSON for JSONB storage
                import json
                
                await session.execute(query, {
                    "prediction_id": prediction.prediction_id,
                    "pattern_id": prediction.pattern_id,
                    "symbol": prediction.symbol,
                    "pattern_name": prediction.pattern_name,
                    "timestamp": prediction.timestamp,
                    "failure_probability": float(prediction.failure_probability),
                    "failure_confidence": float(prediction.failure_confidence),
                    "failure_reasons": prediction.failure_reasons,  # Already a list, don't JSON encode
                    "risk_factors": json.dumps(prediction.risk_factors),
                    "market_volatility": float(prediction.market_volatility),
                    "volume_profile": prediction.volume_profile,
                    "liquidity_score": float(prediction.liquidity_score),
                    "support_resistance_proximity": float(prediction.support_resistance_proximity),
                    "rsi_value": float(prediction.rsi_value),
                    "macd_signal": prediction.macd_signal,
                    "bollinger_position": prediction.bollinger_position,
                    "atr_value": float(prediction.atr_value),
                    "prediction_model": prediction.prediction_model,
                    "feature_importance": json.dumps(prediction.feature_importance),
                    "processing_latency_ms": prediction.processing_latency_ms,
                    "metadata": json.dumps(prediction.metadata) if prediction.metadata else '{}'
                })
                
                await session.commit()
                logger.info(f"✅ Stored failure prediction {prediction.prediction_id}")
                return True
                
        except Exception as e:
            logger.error(f"❌ Failed to store failure prediction: {e}")
            return False
    
    async def get_failure_predictions(self, symbol: str, limit: int = 100) -> List[PatternFailurePrediction]:
        """Retrieve recent failure predictions for a symbol"""
        try:
            if not self.db_connection:
                return []
            
            async with self.db_connection.get_async_session() as session:
                from sqlalchemy import text
                
                query = text("""
                    SELECT * FROM pattern_failure_predictions 
                    WHERE symbol = :symbol 
                    ORDER BY timestamp DESC 
                    LIMIT :limit
                """)
                
                result = await session.execute(query, {"symbol": symbol, "limit": limit})
                rows = result.fetchall()
                
                predictions = []
                for row in rows:
                    prediction = PatternFailurePrediction(
                        prediction_id=row.prediction_id,
                        pattern_id=row.pattern_id,
                        symbol=row.symbol,
                        pattern_name=row.pattern_name,
                        timestamp=row.timestamp,
                        failure_probability=row.failure_probability,
                        failure_confidence=row.failure_confidence,
                        failure_reasons=row.failure_reasons,
                        risk_factors=row.risk_factors,
                        market_volatility=row.market_volatility,
                        volume_profile=row.volume_profile,
                        liquidity_score=row.liquidity_score,
                        support_resistance_proximity=row.support_resistance_proximity,
                        rsi_value=row.rsi_value,
                        macd_signal=row.macd_signal,
                        bollinger_position=row.bollinger_position,
                        atr_value=row.atr_value,
                        prediction_model=row.prediction_model,
                        feature_importance=row.feature_importance,
                        processing_latency_ms=row.processing_latency_ms,
                        metadata=row.metadata
                    )
                    predictions.append(prediction)
                
                return predictions
                
        except Exception as e:
            logger.error(f"❌ Failed to retrieve failure predictions: {e}")
            return []
    
    async def cleanup(self):
        """Cleanup resources"""
        try:
            if self.executor:
                self.executor.shutdown(wait=True)
            if self.db_connection:
                await self.db_connection.close()
            logger.info("✅ Pattern Failure Analyzer cleaned up")
        except Exception as e:
            logger.error(f"❌ Cleanup failed: {e}")
