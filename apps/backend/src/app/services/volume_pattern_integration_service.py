#!/usr/bin/env python3
"""
Volume Pattern Integration Service for AlphaPlus
Integrates enhanced volume analysis with existing pattern detection system
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone
import asyncpg

from .enhanced_volume_analyzer_service import EnhancedVolumeAnalyzerService, VolumeAnalysisResult
from ..src.data_collection.volume_positioning_analyzer import VolumePositioningAnalyzer

logger = logging.getLogger(__name__)

class VolumePatternIntegrationService:
    """
    Service that integrates volume analysis with pattern detection
    Provides unified volume-enhanced pattern analysis
    """
    
    def __init__(self, db_pool: asyncpg.Pool):
        self.db_pool = db_pool
        self.logger = logging.getLogger(__name__)
        
        # Initialize volume analyzers
        self.enhanced_volume_analyzer = EnhancedVolumeAnalyzerService(db_pool)
        self.volume_positioning_analyzer = VolumePositioningAnalyzer(db_pool, None)  # Exchange will be set later
        
        # Integration configuration
        self.volume_confirmation_threshold = 0.7
        self.pattern_volume_bonus = 0.15
        self.min_volume_ratio_for_confirmation = 1.2
        
        self.logger.info("🔗 Volume Pattern Integration Service initialized")
    
    async def analyze_pattern_with_volume(self, pattern_data: Dict, ohlcv_data: List[Dict]) -> Dict:
        """
        Analyze a detected pattern with volume confirmation
        
        Args:
            pattern_data: Detected pattern data
            ohlcv_data: OHLCV data for volume analysis
            
        Returns:
            Enhanced pattern data with volume analysis
        """
        try:
            symbol = pattern_data.get('symbol', '')
            timeframe = pattern_data.get('timeframe', '')
            
            if not symbol or not timeframe:
                self.logger.warning("Missing symbol or timeframe in pattern data")
                return pattern_data
            
            # Perform volume analysis
            volume_analysis = await self.enhanced_volume_analyzer.analyze_volume(
                symbol, timeframe, ohlcv_data
            )
            
            # Enhance pattern with volume analysis
            enhanced_pattern = await self._enhance_pattern_with_volume(
                pattern_data, volume_analysis
            )
            
            # Store enhanced pattern
            await self._store_enhanced_pattern(enhanced_pattern)
            
            return enhanced_pattern
            
        except Exception as e:
            self.logger.error(f"Error in pattern volume analysis: {e}")
            return pattern_data
    
    async def _enhance_pattern_with_volume(self, pattern_data: Dict, volume_analysis: VolumeAnalysisResult) -> Dict:
        """Enhance pattern data with volume analysis results"""
        try:
            enhanced_pattern = pattern_data.copy()
            
            # Add volume metrics
            enhanced_pattern['volume_metrics'] = {
                'volume_ratio': volume_analysis.volume_ratio,
                'volume_trend': volume_analysis.volume_trend,
                'volume_positioning_score': volume_analysis.volume_positioning_score,
                'order_book_imbalance': volume_analysis.order_book_imbalance,
                'buy_volume_ratio': volume_analysis.buy_volume_ratio,
                'sell_volume_ratio': volume_analysis.sell_volume_ratio,
                'volume_breakout': volume_analysis.volume_breakout
            }
            
            # Add advanced volume metrics
            enhanced_pattern['advanced_volume_metrics'] = {
                'vwap': volume_analysis.vwap,
                'cumulative_volume_delta': volume_analysis.cumulative_volume_delta,
                'relative_volume': volume_analysis.relative_volume,
                'volume_weighted_price': volume_analysis.volume_weighted_price,
                'volume_flow_imbalance': volume_analysis.volume_flow_imbalance
            }
            
            # Add volume pattern information
            if volume_analysis.volume_pattern_type:
                enhanced_pattern['volume_pattern'] = {
                    'type': volume_analysis.volume_pattern_type,
                    'strength': volume_analysis.volume_pattern_strength,
                    'confidence': volume_analysis.volume_pattern_confidence
                }
            
            # Calculate volume-enhanced confidence
            original_confidence = pattern_data.get('confidence', 0.5)
            volume_enhanced_confidence = await self._calculate_volume_enhanced_confidence(
                original_confidence, volume_analysis
            )
            
            enhanced_pattern['volume_enhanced_confidence'] = volume_enhanced_confidence
            enhanced_pattern['volume_confirmation'] = volume_enhanced_confidence > self.volume_confirmation_threshold
            
            # Add volume analysis text
            enhanced_pattern['volume_analysis'] = volume_analysis.volume_analysis
            enhanced_pattern['volume_context'] = volume_analysis.volume_context
            
            # Update pattern strength based on volume
            enhanced_pattern['pattern_strength'] = await self._determine_enhanced_pattern_strength(
                pattern_data.get('pattern_strength', 'medium'),
                volume_analysis
            )
            
            # Add volume-based recommendations
            enhanced_pattern['volume_recommendations'] = await self._generate_volume_recommendations(
                pattern_data, volume_analysis
            )
            
            return enhanced_pattern
            
        except Exception as e:
            self.logger.error(f"Error enhancing pattern with volume: {e}")
            return pattern_data
    
    async def _calculate_volume_enhanced_confidence(self, original_confidence: float, 
                                                  volume_analysis: VolumeAnalysisResult) -> float:
        """Calculate volume-enhanced confidence score"""
        try:
            # Base confidence from pattern
            base_confidence = original_confidence
            
            # Volume positioning bonus
            volume_bonus = volume_analysis.volume_positioning_score * self.pattern_volume_bonus
            
            # Volume ratio bonus
            volume_ratio_bonus = 0
            if volume_analysis.volume_ratio >= self.min_volume_ratio_for_confirmation:
                volume_ratio_bonus = min((volume_analysis.volume_ratio - 1.0) * 0.1, 0.2)
            
            # Volume breakout bonus
            breakout_bonus = 0.1 if volume_analysis.volume_breakout else 0
            
            # Pattern confirmation bonus
            pattern_bonus = 0
            if volume_analysis.volume_pattern_confidence:
                pattern_bonus = volume_analysis.volume_pattern_confidence * 0.15
            
            # Advanced metrics bonuses
            vwap_bonus = 0
            if volume_analysis.vwap and volume_analysis.vwap > 0:
                # VWAP confirmation bonus
                current_price = volume_analysis.volume_weighted_price or 0
                if current_price > 0:
                    vwap_distance = abs(current_price - volume_analysis.vwap) / volume_analysis.vwap
                    if vwap_distance < 0.01:  # Price close to VWAP
                        vwap_bonus = 0.05
            
            # CVD (Cumulative Volume Delta) bonus
            cvd_bonus = 0
            if volume_analysis.cumulative_volume_delta:
                cvd_strength = min(abs(volume_analysis.cumulative_volume_delta) / 1000, 1.0)
                cvd_bonus = cvd_strength * 0.1
            
            # Volume flow imbalance bonus
            flow_bonus = 0
            if volume_analysis.volume_flow_imbalance:
                flow_strength = min(abs(volume_analysis.volume_flow_imbalance), 1.0)
                flow_bonus = flow_strength * 0.08
            
            # Calculate enhanced confidence
            enhanced_confidence = base_confidence + volume_bonus + volume_ratio_bonus + breakout_bonus + pattern_bonus + vwap_bonus + cvd_bonus + flow_bonus
            
            # Cap at 1.0
            return min(1.0, enhanced_confidence)
            
        except Exception as e:
            self.logger.error(f"Error calculating volume-enhanced confidence: {e}")
            return original_confidence
    
    async def _determine_enhanced_pattern_strength(self, original_strength: str, 
                                                 volume_analysis: VolumeAnalysisResult) -> str:
        """Determine enhanced pattern strength based on volume analysis"""
        try:
            strength_levels = ['weak', 'medium', 'strong', 'very_strong']
            current_index = strength_levels.index(original_strength) if original_strength in strength_levels else 1
            
            # Volume-based strength adjustments
            if volume_analysis.volume_ratio >= 3.0:
                current_index = min(current_index + 1, len(strength_levels) - 1)
            elif volume_analysis.volume_ratio >= 2.0:
                current_index = min(current_index + 1, len(strength_levels) - 1)
            elif volume_analysis.volume_ratio < 0.8:
                current_index = max(current_index - 1, 0)
            
            if volume_analysis.volume_breakout:
                current_index = min(current_index + 1, len(strength_levels) - 1)
            
            return strength_levels[current_index]
            
        except Exception as e:
            self.logger.error(f"Error determining enhanced pattern strength: {e}")
            return original_strength
    
    async def _generate_volume_recommendations(self, pattern_data: Dict, 
                                             volume_analysis: VolumeAnalysisResult) -> List[str]:
        """Generate volume-based trading recommendations"""
        try:
            recommendations = []
            
            # Volume confirmation recommendations
            if volume_analysis.volume_ratio >= 2.0:
                recommendations.append("Strong volume confirmation - high confidence signal")
            elif volume_analysis.volume_ratio >= 1.5:
                recommendations.append("Good volume confirmation - moderate confidence")
            elif volume_analysis.volume_ratio < 0.8:
                recommendations.append("Low volume - consider waiting for better confirmation")
            
            # Volume trend recommendations
            if volume_analysis.volume_trend == "increasing":
                recommendations.append("Volume trend increasing - bullish momentum")
            elif volume_analysis.volume_trend == "decreasing":
                recommendations.append("Volume trend decreasing - momentum weakening")
            
            # Order book recommendations
            if abs(volume_analysis.order_book_imbalance) > 0.3:
                side = "buying" if volume_analysis.order_book_imbalance > 0 else "selling"
                recommendations.append(f"Strong {side} pressure detected")
            
            # Volume breakout recommendations
            if volume_analysis.volume_breakout:
                recommendations.append("Volume breakout confirmed - strong signal")
            
            # Pattern-specific recommendations
            if volume_analysis.volume_pattern_type:
                recommendations.append(f"Volume pattern detected: {volume_analysis.volume_pattern_type}")
            
            # Advanced metrics recommendations
            if volume_analysis.vwap and volume_analysis.vwap > 0:
                current_price = volume_analysis.volume_weighted_price or 0
                if current_price > 0:
                    vwap_distance = abs(current_price - volume_analysis.vwap) / volume_analysis.vwap
                    if vwap_distance < 0.005:
                        recommendations.append("Price at VWAP - strong support/resistance level")
                    elif vwap_distance < 0.01:
                        recommendations.append("Price near VWAP - moderate support/resistance")
            
            # CVD recommendations
            if volume_analysis.cumulative_volume_delta:
                if volume_analysis.cumulative_volume_delta > 1000:
                    recommendations.append("Strong buying pressure (CVD positive)")
                elif volume_analysis.cumulative_volume_delta < -1000:
                    recommendations.append("Strong selling pressure (CVD negative)")
            
            # Volume flow recommendations
            if volume_analysis.volume_flow_imbalance:
                if volume_analysis.volume_flow_imbalance > 0.5:
                    recommendations.append("Strong volume flow imbalance - bullish")
                elif volume_analysis.volume_flow_imbalance < -0.5:
                    recommendations.append("Strong volume flow imbalance - bearish")
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error generating volume recommendations: {e}")
            return ["Volume analysis unavailable"]
    
    async def _store_enhanced_pattern(self, enhanced_pattern: Dict):
        """Store enhanced pattern in database"""
        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO comprehensive_analysis (
                        symbol, timeframe, timestamp, pattern_confidence, pattern_type,
                        pattern_strength, volume_confidence, volume_ratio, volume_positioning,
                        order_book_imbalance, overall_confidence, analysis_reasoning,
                        signal_direction, signal_strength, created_at
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, NOW())
                """, 
                enhanced_pattern.get('symbol'),
                enhanced_pattern.get('timeframe'),
                enhanced_pattern.get('timestamp', datetime.now(timezone.utc)),
                enhanced_pattern.get('confidence', 0.5),
                enhanced_pattern.get('pattern_type'),
                enhanced_pattern.get('pattern_strength'),
                enhanced_pattern.get('volume_enhanced_confidence', 0.5),
                enhanced_pattern.get('volume_metrics', {}).get('volume_ratio', 1.0),
                enhanced_pattern.get('volume_metrics', {}).get('volume_positioning_score', 0.5),
                enhanced_pattern.get('volume_metrics', {}).get('order_book_imbalance', 0.0),
                enhanced_pattern.get('volume_enhanced_confidence', 0.5),
                enhanced_pattern.get('volume_analysis', ''),
                enhanced_pattern.get('direction', 'neutral'),
                enhanced_pattern.get('pattern_strength', 'medium')
                )
                
        except Exception as e:
            self.logger.error(f"Error storing enhanced pattern: {e}")
    
    async def get_volume_enhanced_patterns(self, symbol: str, timeframe: str, 
                                         limit: int = 50) -> List[Dict]:
        """Get volume-enhanced patterns for a symbol"""
        try:
            async with self.db_pool.acquire() as conn:
                rows = await conn.fetch("""
                    SELECT * FROM comprehensive_analysis 
                    WHERE symbol = $1 AND timeframe = $2 
                    ORDER BY timestamp DESC 
                    LIMIT $3
                """, symbol, timeframe, limit)
                
                patterns = []
                for row in rows:
                    pattern = {
                        'symbol': row['symbol'],
                        'timeframe': row['timeframe'],
                        'timestamp': row['timestamp'],
                        'pattern_confidence': float(row['pattern_confidence']),
                        'pattern_type': row['pattern_type'],
                        'pattern_strength': row['pattern_strength'],
                        'volume_confidence': float(row['volume_confidence']),
                        'volume_ratio': float(row['volume_ratio']),
                        'volume_positioning': row['volume_positioning'],
                        'order_book_imbalance': float(row['order_book_imbalance']),
                        'overall_confidence': float(row['overall_confidence']),
                        'analysis_reasoning': row['analysis_reasoning'],
                        'signal_direction': row['signal_direction'],
                        'signal_strength': row['signal_strength']
                    }
                    patterns.append(pattern)
                
                return patterns
                
        except Exception as e:
            self.logger.error(f"Error getting volume-enhanced patterns: {e}")
            return []
    
    async def get_volume_statistics(self, symbol: str, timeframe: str) -> Dict:
        """Get volume statistics for a symbol"""
        try:
            async with self.db_pool.acquire() as conn:
                # Get recent volume analysis
                volume_rows = await conn.fetch("""
                    SELECT 
                        AVG(volume_ratio) as avg_volume_ratio,
                        AVG(volume_positioning_score) as avg_volume_positioning,
                        AVG(order_book_imbalance) as avg_order_book_imbalance,
                        COUNT(*) as total_analyses,
                        COUNT(CASE WHEN volume_breakout THEN 1 END) as breakout_count
                    FROM volume_analysis 
                    WHERE symbol = $1 AND timeframe = $2 
                    AND timestamp >= NOW() - INTERVAL '24 hours'
                """, symbol, timeframe)
                
                if volume_rows:
                    row = volume_rows[0]
                    return {
                        'avg_volume_ratio': float(row['avg_volume_ratio']) if row['avg_volume_ratio'] else 1.0,
                        'avg_volume_positioning': float(row['avg_volume_positioning']) if row['avg_volume_positioning'] else 0.5,
                        'avg_order_book_imbalance': float(row['avg_order_book_imbalance']) if row['avg_order_book_imbalance'] else 0.0,
                        'total_analyses': row['total_analyses'],
                        'breakout_count': row['breakout_count'],
                        'breakout_percentage': (row['breakout_count'] / row['total_analyses'] * 100) if row['total_analyses'] > 0 else 0
                    }
                
                return {}
                
        except Exception as e:
            self.logger.error(f"Error getting volume statistics: {e}")
            return {}
