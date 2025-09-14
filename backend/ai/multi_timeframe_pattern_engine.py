#!/usr/bin/env python3
"""
Multi-Timeframe Pattern Confirmation Engine for AlphaPlus
Integrates with existing ultra-low latency system for advanced pattern recognition
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
import talib

from strategies.vectorized_pattern_detector import VectorizedPatternDetector, VectorizedPattern
from database.connection import TimescaleDBConnection
from data.volume_analyzer import VolumeAnalyzer, VolumePattern

logger = logging.getLogger(__name__)

@dataclass
class MultiTimeframePattern:
    """Multi-timeframe pattern confirmation result"""
    pattern_id: str
    symbol: str
    primary_timeframe: str
    pattern_name: str
    pattern_type: str
    primary_confidence: float
    primary_strength: str
    timestamp: datetime
    price_level: float
    
    # Multi-timeframe confirmation data
    confirmation_timeframes: List[str]
    timeframe_confidences: Dict[str, float]
    timeframe_alignments: Dict[str, str]  # bullish, bearish, neutral
    overall_confidence: float
    confirmation_score: float  # 0-100
    trend_alignment: str
    failure_probability: float
    
    # Metadata
    detection_method: str = "multi_timeframe"
    processing_latency_ms: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None

class MultiTimeframePatternEngine:
    """Advanced multi-timeframe pattern confirmation engine"""
    
    def __init__(self, db_config: Dict[str, Any], max_workers: int = 4):
        self.db_config = db_config
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.pattern_detector = VectorizedPatternDetector(max_workers=max_workers)
        self.volume_analyzer = VolumeAnalyzer()
        self.db_connection = None
        
        # Timeframe hierarchy and weights
        self.timeframes = ["1m", "5m", "15m", "1h", "4h", "1d", "1w"]
        self.timeframe_weights = {
            "1m": 0.05,   # 5% weight
            "5m": 0.10,   # 10% weight
            "15m": 0.15,  # 15% weight
            "1h": 0.20,   # 20% weight
            "4h": 0.25,   # 25% weight
            "1d": 0.20,   # 20% weight
            "1w": 0.05    # 5% weight
        }
        
        # Pattern confirmation thresholds
        self.confirmation_thresholds = {
            "strong": 0.8,
            "moderate": 0.6,
            "weak": 0.4
        }
        
        logger.info(f"🚀 Multi-Timeframe Pattern Engine initialized with {len(self.timeframes)} timeframes")
    
    async def initialize(self):
        """Initialize database connection"""
        try:
            self.db_connection = TimescaleDBConnection(self.db_config)
            await self.db_connection.initialize()
            logger.info("✅ Multi-Timeframe Pattern Engine database connection established")
        except Exception as e:
            logger.error(f"❌ Failed to initialize database connection: {e}")
            raise
    
    async def detect_multi_timeframe_patterns(self, symbol: str, primary_timeframe: str, 
                                            candlestick_data: Dict[str, pd.DataFrame]) -> List[MultiTimeframePattern]:
        """Detect patterns across multiple timeframes and confirm alignment"""
        try:
            start_time = time.time()
            
            # Detect patterns on primary timeframe
            primary_df = candlestick_data.get(primary_timeframe)
            if primary_df is None or primary_df.empty:
                logger.warning(f"No data available for {symbol} {primary_timeframe}")
                return []
            
            # Detect traditional patterns
            primary_patterns = await self.pattern_detector.detect_patterns_vectorized(
                primary_df, use_talib=True, use_incremental=True
            )
            
            # Detect Wyckoff patterns
            wyckoff_patterns = self.volume_analyzer.detect_wyckoff_patterns(
                primary_df, symbol, primary_timeframe
            )
            
            # Convert Wyckoff patterns to VectorizedPattern format for consistency
            wyckoff_vectorized_patterns = []
            for wyckoff_pattern in wyckoff_patterns:
                vectorized_pattern = VectorizedPattern(
                    pattern_name=wyckoff_pattern.pattern_type.value,
                    pattern_type='bullish' if 'spring' in wyckoff_pattern.pattern_type.value or 'strength' in wyckoff_pattern.pattern_type.value else 'bearish',
                    confidence=wyckoff_pattern.confidence,
                    strength=wyckoff_pattern.strength.value,
                    timestamp=wyckoff_pattern.timestamp,
                    price_level=primary_df['close'].iloc[-1],
                    volume_confirmation=wyckoff_pattern.volume_ratio > 1.0,
                    volume_confidence=wyckoff_pattern.volume_ratio,
                    trend_alignment='bullish' if 'spring' in wyckoff_pattern.pattern_type.value or 'strength' in wyckoff_pattern.pattern_type.value else 'bearish',
                    metadata=wyckoff_pattern.pattern_data
                )
                wyckoff_vectorized_patterns.append(vectorized_pattern)
            
            # Combine all patterns
            all_patterns = primary_patterns + wyckoff_vectorized_patterns
            
            if not all_patterns:
                return []
            
            # Process each pattern for multi-timeframe confirmation
            multi_timeframe_patterns = []
            
            for primary_pattern in all_patterns:
                # Confirm pattern across other timeframes
                confirmation_result = await self._confirm_pattern_across_timeframes(
                    symbol, primary_pattern, candlestick_data, primary_timeframe
                )
                
                if confirmation_result:
                    multi_timeframe_patterns.append(confirmation_result)
            
            processing_time = (time.time() - start_time) * 1000
            logger.info(f"✅ Multi-timeframe pattern detection completed in {processing_time:.2f}ms")
            logger.info(f"📊 Detected {len(multi_timeframe_patterns)} multi-timeframe patterns")
            
            return multi_timeframe_patterns
            
        except Exception as e:
            logger.error(f"❌ Multi-timeframe pattern detection failed: {e}")
            return []
    
    async def _confirm_pattern_across_timeframes(self, symbol: str, primary_pattern: VectorizedPattern,
                                                candlestick_data: Dict[str, pd.DataFrame], primary_timeframe: str = "1h") -> Optional[MultiTimeframePattern]:
        """Confirm a pattern across multiple timeframes"""
        try:
            timeframe_confidences = {}
            timeframe_alignments = {}
            confirmation_timeframes = []
            
            # Optimized: Run pattern detection only once and check consistency
            # Use the primary timeframe pattern as reference
            primary_pattern_name = primary_pattern.pattern_name
            
            # Check pattern consistency across timeframes without re-running detection
            for timeframe in self.timeframes:
                if timeframe not in candlestick_data or candlestick_data[timeframe].empty:
                    timeframe_confidences[timeframe] = 0.0
                    timeframe_alignments[timeframe] = "neutral"
                    continue
                
                # Use a simplified check based on price action consistency
                df = candlestick_data[timeframe]
                if len(df) < 5:
                    timeframe_confidences[timeframe] = 0.0
                    timeframe_alignments[timeframe] = "neutral"
                    continue
                
                # Calculate basic pattern consistency score
                consistency_score = self._calculate_pattern_consistency(primary_pattern, df)
                
                if consistency_score > 0.5:  # Threshold for confirmation
                    timeframe_confidences[timeframe] = consistency_score
                    timeframe_alignments[timeframe] = primary_pattern.pattern_type
                    confirmation_timeframes.append(timeframe)
                else:
                    timeframe_confidences[timeframe] = 0.0
                    timeframe_alignments[timeframe] = "neutral"
            
            # Calculate overall confidence and confirmation score
            overall_confidence = self._calculate_overall_confidence(
                primary_pattern.confidence, timeframe_confidences
            )
            
            confirmation_score = self._calculate_confirmation_score(
                primary_pattern.confidence, timeframe_confidences, timeframe_alignments
            )
            
            # Determine trend alignment
            trend_alignment = self._determine_trend_alignment(timeframe_alignments)
            
            # Calculate failure probability
            failure_probability = self._calculate_failure_probability(
                primary_pattern, timeframe_confidences, trend_alignment
            )
            
            # Create multi-timeframe pattern
            multi_timeframe_pattern = MultiTimeframePattern(
                pattern_id=f"{primary_pattern.pattern_name}_{symbol}_{int(time.time())}",
                symbol=symbol,
                primary_timeframe=primary_timeframe,
                pattern_name=primary_pattern.pattern_name,
                pattern_type=primary_pattern.pattern_type,
                primary_confidence=primary_pattern.confidence,
                primary_strength=primary_pattern.strength,
                timestamp=primary_pattern.timestamp,
                price_level=primary_pattern.price_level,
                confirmation_timeframes=confirmation_timeframes,
                timeframe_confidences=timeframe_confidences,
                timeframe_alignments=timeframe_alignments,
                overall_confidence=overall_confidence,
                confirmation_score=confirmation_score,
                trend_alignment=trend_alignment,
                failure_probability=failure_probability,
                processing_latency_ms=int((time.time() - time.time()) * 1000),
                metadata={
                    "primary_timeframe": primary_pattern.timestamp.strftime("%H") if hasattr(primary_pattern, 'timestamp') else "1h",
                    "timeframe_weights": self.timeframe_weights,
                    "confirmation_thresholds": self.confirmation_thresholds
                }
            )
            
            return multi_timeframe_pattern
            
        except Exception as e:
            logger.error(f"❌ Pattern confirmation failed: {e}")
            return None
    
    def _find_matching_pattern(self, primary_pattern: VectorizedPattern, 
                              patterns: List[VectorizedPattern]) -> Optional[VectorizedPattern]:
        """Find a matching pattern in the list"""
        for pattern in patterns:
            if pattern.pattern_name == primary_pattern.pattern_name:
                return pattern
        return None
    
    def _calculate_overall_confidence(self, primary_confidence: float, 
                                    timeframe_confidences: Dict[str, float]) -> float:
        """Calculate weighted overall confidence across timeframes"""
        weighted_sum = primary_confidence * self.timeframe_weights.get("1h", 0.2)
        total_weight = self.timeframe_weights.get("1h", 0.2)
        
        for timeframe, confidence in timeframe_confidences.items():
            weight = self.timeframe_weights.get(timeframe, 0.0)
            weighted_sum += confidence * weight
            total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else primary_confidence
    
    def _calculate_confirmation_score(self, primary_confidence: float,
                                    timeframe_confidences: Dict[str, float],
                                    timeframe_alignments: Dict[str, str]) -> float:
        """Calculate confirmation score (0-100)"""
        # Count confirmations
        confirmations = 0
        total_timeframes = len(timeframe_confidences)
        
        for timeframe, confidence in timeframe_confidences.items():
            if confidence >= self.confirmation_thresholds["moderate"]:
                confirmations += 1
        
        # Calculate alignment bonus
        alignment_bonus = 0
        bullish_count = sum(1 for alignment in timeframe_alignments.values() if alignment == "bullish")
        bearish_count = sum(1 for alignment in timeframe_alignments.values() if alignment == "bearish")
        
        if bullish_count > bearish_count:
            alignment_bonus = (bullish_count / total_timeframes) * 20
        elif bearish_count > bullish_count:
            alignment_bonus = (bearish_count / total_timeframes) * 20
        
        # Base score from confirmations
        base_score = (confirmations / total_timeframes) * 60
        
        # Confidence bonus
        confidence_bonus = primary_confidence * 20
        
        return min(100, base_score + alignment_bonus + confidence_bonus)
    
    def _determine_trend_alignment(self, timeframe_alignments: Dict[str, str]) -> str:
        """Determine overall trend alignment"""
        bullish_count = sum(1 for alignment in timeframe_alignments.values() if alignment == "bullish")
        bearish_count = sum(1 for alignment in timeframe_alignments.values() if alignment == "bearish")
        neutral_count = sum(1 for alignment in timeframe_alignments.values() if alignment == "neutral")
        
        total = len(timeframe_alignments)
        
        if bullish_count > bearish_count and bullish_count > neutral_count:
            return "bullish"
        elif bearish_count > bullish_count and bearish_count > neutral_count:
            return "bearish"
        else:
            return "neutral"
    
    def _calculate_pattern_consistency(self, primary_pattern: VectorizedPattern, df: pd.DataFrame) -> float:
        """Calculate pattern consistency score across timeframes"""
        try:
            if len(df) < 5:
                return 0.0
            
            # Get recent price action
            recent_close = df['close'].iloc[-1]
            recent_open = df['open'].iloc[-1]
            recent_high = df['high'].iloc[-1]
            recent_low = df['low'].iloc[-1]
            
            # Handle Wyckoff patterns specifically
            if 'wyckoff' in primary_pattern.pattern_name.lower():
                return self._calculate_wyckoff_consistency(primary_pattern, df)
            
            # Calculate basic pattern consistency for traditional patterns
            if primary_pattern.pattern_type == 'bullish':
                # Check for bullish characteristics
                body_size = abs(recent_close - recent_open)
                total_range = recent_high - recent_low
                body_ratio = body_size / total_range if total_range > 0 else 0
                
                # Bullish pattern should have close > open
                if recent_close > recent_open:
                    return min(1.0, body_ratio + 0.3)
                else:
                    return max(0.0, body_ratio - 0.2)
                    
            elif primary_pattern.pattern_type == 'bearish':
                # Check for bearish characteristics
                body_size = abs(recent_close - recent_open)
                total_range = recent_high - recent_low
                body_ratio = body_size / total_range if total_range > 0 else 0
                
                # Bearish pattern should have close < open
                if recent_close < recent_open:
                    return min(1.0, body_ratio + 0.3)
                else:
                    return max(0.0, body_ratio - 0.2)
                    
            else:  # neutral
                # For neutral patterns, check for small body
                body_size = abs(recent_close - recent_open)
                total_range = recent_high - recent_low
                body_ratio = body_size / total_range if total_range > 0 else 0
                
                # Neutral patterns should have small body
                if body_ratio < 0.3:
                    return 0.7
                else:
                    return max(0.0, 0.5 - body_ratio)
                    
        except Exception as e:
            logger.error(f"❌ Pattern consistency calculation error: {e}")
            return 0.0
    
    def _calculate_wyckoff_consistency(self, primary_pattern: VectorizedPattern, df: pd.DataFrame) -> float:
        """Calculate Wyckoff pattern consistency across timeframes"""
        try:
            if len(df) < 10:
                return 0.0
            
            pattern_name = primary_pattern.pattern_name.lower()
            
            # Get recent price action
            recent_lows = df['low'].tail(10).values
            recent_highs = df['high'].tail(10).values
            recent_closes = df['close'].tail(10).values
            recent_volumes = df['volume'].tail(10).values if 'volume' in df.columns else None
            
            if 'spring' in pattern_name:
                # Spring pattern: check for false breakdown and recovery
                support_level = np.min(recent_lows[:-3])
                breakdown_candles = recent_lows[-3:] < support_level * 0.995
                recovery_candles = recent_closes[-2:] > support_level
                
                if np.any(breakdown_candles) and np.any(recovery_candles):
                    return 0.8
                elif np.any(breakdown_candles):
                    return 0.5
                else:
                    return 0.2
                    
            elif 'upthrust' in pattern_name:
                # Upthrust pattern: check for false breakout and rejection
                resistance_level = np.max(recent_highs[:-3])
                breakout_candles = recent_highs[-3:] > resistance_level * 1.005
                rejection_candles = recent_closes[-2:] < resistance_level
                
                if np.any(breakout_candles) and np.any(rejection_candles):
                    return 0.8
                elif np.any(breakout_candles):
                    return 0.5
                else:
                    return 0.2
                    
            elif 'accumulation' in pattern_name:
                # Accumulation: check for price stability near support
                support_level = np.min(recent_lows[:-5])
                price_stability = np.std(recent_closes[-5:]) / np.mean(recent_closes[-5:])
                
                if price_stability < 0.02:  # Low volatility
                    return 0.7
                else:
                    return 0.3
                    
            elif 'distribution' in pattern_name:
                # Distribution: check for price stability near resistance
                resistance_level = np.max(recent_highs[:-5])
                price_stability = np.std(recent_closes[-5:]) / np.mean(recent_closes[-5:])
                
                if price_stability < 0.02:  # Low volatility
                    return 0.7
                else:
                    return 0.3
                    
            elif 'test' in pattern_name:
                # Test: check for price returning to support/resistance
                support_level = np.min(recent_lows[:-5])
                resistance_level = np.max(recent_highs[:-5])
                
                recent_low = recent_lows[-1]
                recent_high = recent_highs[-1]
                
                if abs(recent_low - support_level) / support_level < 0.01:
                    return 0.8  # Support test
                elif abs(recent_high - resistance_level) / resistance_level < 0.01:
                    return 0.8  # Resistance test
                else:
                    return 0.2
                    
            elif 'strength' in pattern_name:
                # Sign of Strength: check for strong upward move
                price_change = (recent_closes[-1] - recent_closes[-5]) / recent_closes[-5]
                
                if price_change > 0.02:  # At least 2% move
                    return 0.8
                elif price_change > 0.01:  # At least 1% move
                    return 0.5
                else:
                    return 0.2
                    
            elif 'weakness' in pattern_name:
                # Sign of Weakness: check for strong downward move
                price_change = (recent_closes[-1] - recent_closes[-5]) / recent_closes[-5]
                
                if price_change < -0.02:  # At least 2% move down
                    return 0.8
                elif price_change < -0.01:  # At least 1% move down
                    return 0.5
                else:
                    return 0.2
                    
            else:
                # Default Wyckoff consistency
                return 0.5
                
        except Exception as e:
            logger.error(f"❌ Wyckoff consistency calculation error: {e}")
            return 0.0
    
    def _calculate_failure_probability(self, primary_pattern: VectorizedPattern,
                                     timeframe_confidences: Dict[str, float],
                                     trend_alignment: str) -> float:
        """Calculate pattern failure probability"""
        # Base failure probability from pattern type
        base_failure_rates = {
            "doji": 0.4,
            "hammer": 0.3,
            "shooting_star": 0.35,
            "engulfing": 0.25,
            "morning_star": 0.2,
            "evening_star": 0.2,
            "three_white_soldiers": 0.15,
            "three_black_crows": 0.15,
            # Wyckoff patterns (generally more reliable)
            "wyckoff_spring": 0.15,
            "wyckoff_upthrust": 0.15,
            "wyckoff_accumulation": 0.2,
            "wyckoff_distribution": 0.2,
            "wyckoff_test": 0.25,
            "wyckoff_sign_of_strength": 0.1,
            "wyckoff_sign_of_weakness": 0.1
        }
        
        base_failure = base_failure_rates.get(primary_pattern.pattern_name.lower(), 0.3)
        
        # Adjust based on confirmation score
        avg_confidence = np.mean(list(timeframe_confidences.values()))
        confidence_adjustment = (1 - avg_confidence) * 0.3
        
        # Adjust based on trend alignment
        alignment_adjustment = 0.1 if trend_alignment == "neutral" else 0.0
        
        # Adjust based on pattern strength
        strength_adjustment = 0.1 if primary_pattern.strength == "weak" else 0.0
        
        failure_probability = base_failure + confidence_adjustment + alignment_adjustment + strength_adjustment
        
        return min(1.0, max(0.0, failure_probability))
    
    async def store_multi_timeframe_pattern(self, pattern: MultiTimeframePattern):
        """Store multi-timeframe pattern in database"""
        try:
            if not self.db_connection:
                logger.warning("Database connection not available")
                return False
            
            async with self.db_connection.get_async_session() as session:
                from sqlalchemy import text
                
                query = text("""
                    INSERT INTO multi_timeframe_patterns (
                        pattern_id, symbol, primary_timeframe, pattern_name, pattern_type,
                        primary_confidence, primary_strength, timestamp, price_level,
                        confirmation_timeframes, timeframe_confidences, timeframe_alignments,
                        overall_confidence, confirmation_score, trend_alignment, failure_probability,
                        detection_method, processing_latency_ms, metadata, created_at
                    ) VALUES (
                        :pattern_id, :symbol, :primary_timeframe, :pattern_name, :pattern_type,
                        :primary_confidence, :primary_strength, :timestamp, :price_level,
                        :confirmation_timeframes, :timeframe_confidences, :timeframe_alignments,
                        :overall_confidence, :confirmation_score, :trend_alignment, :failure_probability,
                        :detection_method, :processing_latency_ms, :metadata, NOW()
                    )
                """)
                
                # Convert lists and dicts to JSON for JSONB storage
                import json
                
                await session.execute(query, {
                    "pattern_id": pattern.pattern_id,
                    "symbol": pattern.symbol,
                    "primary_timeframe": pattern.primary_timeframe,
                    "pattern_name": pattern.pattern_name,
                    "pattern_type": pattern.pattern_type,
                    "primary_confidence": float(pattern.primary_confidence),
                    "primary_strength": pattern.primary_strength,
                    "timestamp": pattern.timestamp.tz_localize('UTC') if hasattr(pattern.timestamp, 'tz') and pattern.timestamp.tz is None else pattern.timestamp,
                    "price_level": float(pattern.price_level),
                    "confirmation_timeframes": json.dumps(pattern.confirmation_timeframes),
                    "timeframe_confidences": json.dumps(pattern.timeframe_confidences),
                    "timeframe_alignments": json.dumps(pattern.timeframe_alignments),
                    "overall_confidence": float(pattern.overall_confidence),
                    "confirmation_score": float(pattern.confirmation_score),
                    "trend_alignment": pattern.trend_alignment,
                    "failure_probability": float(pattern.failure_probability),
                    "detection_method": pattern.detection_method,
                    "processing_latency_ms": pattern.processing_latency_ms,
                    "metadata": json.dumps(pattern.metadata) if pattern.metadata else '{}'
                })
                
                await session.commit()
                logger.info(f"✅ Stored multi-timeframe pattern {pattern.pattern_id}")
                return True
                
        except Exception as e:
            logger.error(f"❌ Failed to store multi-timeframe pattern: {e}")
            return False
    
    async def get_multi_timeframe_patterns(self, symbol: str, limit: int = 100) -> List[MultiTimeframePattern]:
        """Retrieve recent multi-timeframe patterns for a symbol"""
        try:
            if not self.db_connection:
                return []
            
            async with self.db_connection.get_async_session() as session:
                from sqlalchemy import text
                
                query = text("""
                    SELECT * FROM multi_timeframe_patterns 
                    WHERE symbol = :symbol 
                    ORDER BY timestamp DESC 
                    LIMIT :limit
                """)
                
                result = await session.execute(query, {"symbol": symbol, "limit": limit})
                rows = result.fetchall()
                
                patterns = []
                for row in rows:
                    pattern = MultiTimeframePattern(
                        pattern_id=row.pattern_id,
                        symbol=row.symbol,
                        primary_timeframe=row.primary_timeframe,
                        pattern_name=row.pattern_name,
                        pattern_type=row.pattern_type,
                        primary_confidence=row.primary_confidence,
                        primary_strength=row.primary_strength,
                        timestamp=row.timestamp,
                        price_level=row.price_level,
                        confirmation_timeframes=row.confirmation_timeframes,
                        timeframe_confidences=row.timeframe_confidences,
                        timeframe_alignments=row.timeframe_alignments,
                        overall_confidence=row.overall_confidence,
                        confirmation_score=row.confirmation_score,
                        trend_alignment=row.trend_alignment,
                        failure_probability=row.failure_probability,
                        detection_method=row.detection_method,
                        processing_latency_ms=row.processing_latency_ms,
                        metadata=row.metadata
                    )
                    patterns.append(pattern)
                
                return patterns
                
        except Exception as e:
            logger.error(f"❌ Failed to retrieve multi-timeframe patterns: {e}")
            return []
    
    async def cleanup(self):
        """Cleanup resources"""
        try:
            if self.executor:
                self.executor.shutdown(wait=True)
            if self.pattern_detector:
                await self.pattern_detector.cleanup()
            if self.db_connection:
                await self.db_connection.close()
            logger.info("✅ Multi-Timeframe Pattern Engine cleaned up")
        except Exception as e:
            logger.error(f"❌ Cleanup failed: {e}")
