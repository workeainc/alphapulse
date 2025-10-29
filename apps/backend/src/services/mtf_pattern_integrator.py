import asyncio
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import pandas as pd
from dataclasses import dataclass

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from ..src.services.mtf_orchestrator import MTFOrchestrator
from ..src.services.mtf_cache_manager import MTFCacheManager
from ..src.strategies.pattern_detector import CandlestickPatternDetector
from ..src.strategies.enhanced_pattern_detector import EnhancedPatternDetector
from ..src.database.models import MTFContext
from ..src.database.database import get_db

logger = logging.getLogger(__name__)

@dataclass
class MTFPatternResult:
    symbol: str
    timeframe: str
    pattern_name: str
    confidence: float
    mtf_boost: float
    trend_alignment: str
    volume_confirmation: str
    timestamp: datetime
    market_context: Dict[str, Any]
    technical_indicators: Dict[str, Any]

class MTFPatternIntegrator:
    """
    Integrates MTF infrastructure with existing pattern detection system
    Provides MTF-enhanced pattern recognition with context inheritance
    """
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.mtf_orchestrator = MTFOrchestrator(redis_url)
        self.cache_manager = MTFCacheManager(redis_url)
        self.pattern_detector = CandlestickPatternDetector()
        self.enhanced_detector = EnhancedPatternDetector()
        
        # MTF configuration
        self.timeframe_hierarchy = ["1d", "4h", "1h", "15m", "5m", "1m"]
        self.min_confidence_threshold = 0.3
        self.mtf_boost_threshold = 0.1
        
        # Performance tracking
        self.stats = {
            'total_patterns_detected': 0,
            'mtf_enhanced_patterns': 0,
            'trend_aligned_patterns': 0,
            'volume_confirmed_patterns': 0,
            'processing_times': []
        }
        
        logger.info("🚀 MTF Pattern Integrator initialized")
    
    async def detect_patterns_with_mtf_context(
        self, 
        symbol: str, 
        timeframe: str, 
        data: pd.DataFrame
    ) -> List[MTFPatternResult]:
        """
        Detect patterns with MTF context inheritance and enhancement
        """
        start_time = datetime.now()
        
        try:
            # Get MTF context from higher timeframes
            mtf_context = await self._get_mtf_context(symbol, timeframe)
            
            # Detect base patterns
            base_patterns = self.pattern_detector.detect_patterns(data)
            
            # Enhance patterns with MTF context
            enhanced_patterns = []
            
            for pattern in base_patterns:
                enhanced_pattern = await self._enhance_pattern_with_mtf(
                    pattern, symbol, timeframe, mtf_context, data
                )
                
                if enhanced_pattern and enhanced_pattern.confidence >= self.min_confidence_threshold:
                    enhanced_patterns.append(enhanced_pattern)
            
            # Update statistics
            processing_time = (datetime.now() - start_time).total_seconds()
            self._update_stats(len(enhanced_patterns), processing_time)
            
            logger.info(f"🔍 MTF Pattern Detection: {len(enhanced_patterns)} patterns for {symbol} {timeframe} in {processing_time:.3f}s")
            
            return enhanced_patterns
            
        except Exception as e:
            logger.error(f"❌ Error in MTF pattern detection: {e}")
            return []
    
    async def _get_mtf_context(self, symbol: str, timeframe: str) -> Dict[str, Any]:
        """
        Get MTF context from higher timeframes
        """
        context = {}
        
        # Get higher timeframe contexts
        higher_timeframes = self._get_higher_timeframes(timeframe)
        
        for higher_tf in higher_timeframes:
            try:
                # Try to get from cache first
                cached_context = self.cache_manager.get_mtf_context(symbol, higher_tf)
                
                if cached_context is not None:
                    context[higher_tf] = cached_context
                else:
                    # Get from database
                    db_context = await self._get_context_from_db(symbol, higher_tf)
                    if db_context:
                        context[higher_tf] = db_context
                        # Cache it
                        self.cache_manager.cache_mtf_context(symbol, higher_tf, db_context)
                        
            except Exception as e:
                logger.warning(f"⚠️ Could not get MTF context for {symbol} {higher_tf}: {e}")
        
        return context
    
    def _get_higher_timeframes(self, timeframe: str) -> List[str]:
        """
        Get higher timeframes in hierarchy
        """
        try:
            current_index = self.timeframe_hierarchy.index(timeframe)
            return self.timeframe_hierarchy[:current_index]
        except ValueError:
            return []
    
    async def _get_context_from_db(self, symbol: str, timeframe: str) -> Optional[Dict[str, Any]]:
        """
        Get MTF context from database
        """
        try:
            db = next(get_db())
            context = db.query(MTFContext).filter(
                MTFContext.symbol == symbol,
                MTFContext.timeframe == timeframe,
                MTFContext.is_active == True
            ).first()
            
            if context:
                return {
                    'trend_direction': context.trend_direction,
                    'trend_strength': context.trend_strength,
                    'pattern_confirmed': context.pattern_confirmed,
                    'market_regime': context.market_regime,
                    'confidence_score': context.confidence_score,
                    'technical_indicators': context.technical_indicators or {},
                    'market_conditions': context.market_conditions or {}
                }
            
        except Exception as e:
            logger.error(f"❌ Database error getting MTF context: {e}")
        
        return None
    
    async def _enhance_pattern_with_mtf(
        self, 
        pattern: Any, 
        symbol: str, 
        timeframe: str, 
        mtf_context: Dict[str, Any], 
        data: pd.DataFrame
    ) -> Optional[MTFPatternResult]:
        """
        Enhance a pattern with MTF context and confidence scoring
        """
        try:
            # Base confidence from pattern detection
            base_confidence = getattr(pattern, 'confidence', 0.5)
            
            # Calculate MTF boost
            mtf_boost = self._calculate_mtf_boost(pattern, mtf_context)
            
            # Calculate trend alignment
            trend_alignment = self._calculate_trend_alignment(pattern, mtf_context)
            
            # Calculate volume confirmation
            volume_confirmation = self._calculate_volume_confirmation(pattern, data)
            
            # Final confidence calculation
            final_confidence = self._calculate_final_confidence(
                base_confidence, mtf_boost, trend_alignment, volume_confirmation
            )
            
            # Create enhanced pattern result
            enhanced_pattern = MTFPatternResult(
                symbol=symbol,
                timeframe=timeframe,
                pattern_name=pattern.pattern,
                confidence=final_confidence,
                mtf_boost=mtf_boost,
                trend_alignment=trend_alignment['alignment'],
                volume_confirmation=volume_confirmation['type'],
                timestamp=datetime.now(),
                market_context=mtf_context,
                technical_indicators=self._extract_technical_indicators(data)
            )
            
            return enhanced_pattern
            
        except Exception as e:
            logger.error(f"❌ Error enhancing pattern: {e}")
            return None
    
    def _calculate_mtf_boost(self, pattern: Any, mtf_context: Dict[str, Any]) -> float:
        """
        Calculate MTF confidence boost based on higher timeframe alignment
        """
        boost = 0.0
        
        for tf, context in mtf_context.items():
            if not context:
                continue
                
            # Check if pattern aligns with higher timeframe trend
            pattern_direction = self._get_pattern_direction(pattern)
            trend_direction = context.get('trend_direction', 'neutral')
            
            if pattern_direction == trend_direction:
                # Strong alignment bonus
                trend_strength = context.get('trend_strength', 0.5)
                boost += trend_strength * 0.2
            elif trend_direction == 'neutral':
                # Neutral trend - small boost
                boost += 0.05
            else:
                # Counter-trend - penalty
                boost -= 0.1
            
            # Add confidence score bonus
            confidence_score = context.get('confidence_score', 0.0)
            boost += confidence_score * 0.1
        
        return max(0.0, min(1.0, boost))
    
    def _get_pattern_direction(self, pattern: Any) -> str:
        """
        Determine if pattern is bullish, bearish, or neutral
        """
        pattern_name = pattern.pattern.lower()
        
        bullish_patterns = ['bullish_engulfing', 'hammer', 'morning_star', 'piercing_line']
        bearish_patterns = ['bearish_engulfing', 'shooting_star', 'evening_star', 'dark_cloud_cover']
        
        if any(bull in pattern_name for bull in bullish_patterns):
            return 'bullish'
        elif any(bear in pattern_name for bear in bearish_patterns):
            return 'bearish'
        else:
            return 'neutral'
    
    def _calculate_trend_alignment(self, pattern: Any, mtf_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate trend alignment score and type
        """
        pattern_direction = self._get_pattern_direction(pattern)
        alignment_score = 0.0
        alignment_type = 'neutral'
        
        for tf, context in mtf_context.items():
            if not context:
                continue
                
            trend_direction = context.get('trend_direction', 'neutral')
            trend_strength = context.get('trend_strength', 0.0)
            
            if pattern_direction == trend_direction:
                alignment_score += trend_strength
            elif pattern_direction == 'neutral' or trend_direction == 'neutral':
                alignment_score += 0.1
            else:
                alignment_score -= trend_strength * 0.5
        
        # Determine alignment type
        if alignment_score > 0.5:
            alignment_type = 'strong_alignment'
        elif alignment_score > 0.1:
            alignment_type = 'weak_alignment'
        elif alignment_score < -0.3:
            alignment_type = 'counter_trend'
        else:
            alignment_type = 'neutral'
        
        return {
            'alignment': alignment_type,
            'score': alignment_score
        }
    
    def _calculate_volume_confirmation(self, pattern: Any, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate volume confirmation for the pattern
        """
        try:
            if 'volume' not in data.columns:
                return {'type': 'no_volume_data', 'strength': 0.0}
            
            # Get recent volume data
            recent_volume = data['volume'].tail(5)
            avg_volume = recent_volume.mean()
            current_volume = recent_volume.iloc[-1]
            
            # Calculate volume ratio
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
            
            # Determine volume pattern
            if volume_ratio > 1.5:
                volume_type = 'spike'
                strength = 'strong'
            elif volume_ratio > 1.2:
                volume_type = 'spike'
                strength = 'moderate'
            elif volume_ratio > 0.8:
                volume_type = 'normal'
                strength = 'weak'
            else:
                volume_type = 'low'
                strength = 'weak'
            
            return {
                'type': volume_type,
                'strength': strength,
                'ratio': volume_ratio
            }
            
        except Exception as e:
            logger.warning(f"⚠️ Error calculating volume confirmation: {e}")
            return {'type': 'error', 'strength': 'weak'}
    
    def _calculate_final_confidence(
        self, 
        base_confidence: float, 
        mtf_boost: float, 
        trend_alignment: Dict[str, Any], 
        volume_confirmation: Dict[str, Any]
    ) -> float:
        """
        Calculate final confidence score using multi-factor formula
        """
        # Base confidence
        confidence = base_confidence
        
        # Apply MTF boost
        confidence *= (1 + mtf_boost)
        
        # Apply trend alignment factor
        alignment_factors = {
            'strong_alignment': 1.25,
            'weak_alignment': 1.10,
            'neutral': 1.0,
            'counter_trend': 0.80
        }
        alignment_factor = alignment_factors.get(trend_alignment['alignment'], 1.0)
        confidence *= alignment_factor
        
        # Apply volume confirmation factor
        volume_factors = {
            'spike': {'strong': 1.20, 'moderate': 1.10, 'weak': 1.05},
            'normal': {'strong': 1.0, 'moderate': 1.0, 'weak': 0.95},
            'low': {'strong': 0.90, 'moderate': 0.85, 'weak': 0.80}
        }
        
        volume_type = volume_confirmation['type']
        volume_strength = volume_confirmation['strength']
        
        if volume_type in volume_factors and volume_strength in volume_factors[volume_type]:
            volume_factor = volume_factors[volume_type][volume_strength]
            confidence *= volume_factor
        
        # Clamp to valid range
        return max(0.0, min(1.0, confidence))
    
    def _extract_technical_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Extract technical indicators from the data
        """
        indicators = {}
        
        # Common indicator columns to extract
        indicator_columns = ['rsi', 'macd', 'macd_signal', 'ema_20', 'ema_50', 'sma_20', 'bb_upper', 'bb_lower']
        
        for col in indicator_columns:
            if col in data.columns:
                indicators[col] = data[col].iloc[-1] if not data.empty else None
        
        return indicators
    
    def _update_stats(self, patterns_count: int, processing_time: float):
        """
        Update performance statistics
        """
        self.stats['total_patterns_detected'] += patterns_count
        self.stats['processing_times'].append(processing_time)
        
        # Keep only last 100 processing times
        if len(self.stats['processing_times']) > 100:
            self.stats['processing_times'] = self.stats['processing_times'][-100:]
    
    async def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics
        """
        avg_processing_time = (
            sum(self.stats['processing_times']) / len(self.stats['processing_times'])
            if self.stats['processing_times'] else 0.0
        )
        
        return {
            'total_patterns_detected': self.stats['total_patterns_detected'],
            'mtf_enhanced_patterns': self.stats['mtf_enhanced_patterns'],
            'trend_aligned_patterns': self.stats['trend_aligned_patterns'],
            'volume_confirmed_patterns': self.stats['volume_confirmed_patterns'],
            'average_processing_time': avg_processing_time,
            'cache_stats': self.cache_manager.get_stats()
        }
    
    async def clear_cache(self):
        """
        Clear all caches
        """
        await self.cache_manager.clear_all_caches()
        logger.info("🧹 MTF Pattern Integrator cache cleared")
