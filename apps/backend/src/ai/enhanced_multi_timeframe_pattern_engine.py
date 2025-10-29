"""
Enhanced Multi-Timeframe Pattern Engine for AlphaPlus
Integrates Volume Profile, Elliott Wave, Wyckoff, and SMC analysis
"""

import pandas as pd
import numpy as np
import asyncio
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from datetime import datetime
import json

from src.strategies.vectorized_pattern_detector import VectorizedPatternDetector, VectorizedPattern
from src.database.connection import TimescaleDBConnection
from src.data.volume_analyzer import VolumeAnalyzer, VolumePattern
from src.data.volume_profile_analyzer import VolumeProfileAnalyzer, VolumeProfileAnalysis
from src.data.elliott_wave_analyzer import ElliottWaveAnalyzer, ElliottWaveAnalysis

logger = logging.getLogger(__name__)

@dataclass
class EnhancedMultiTimeframePattern:
    """Enhanced multi-timeframe pattern with Volume Profile and Elliott Wave analysis"""
    symbol: str
    timestamp: datetime
    primary_timeframe: str
    pattern_name: str
    pattern_type: str  # 'bullish', 'bearish', 'neutral'
    confidence: float
    strength: str  # 'weak', 'medium', 'strong', 'extreme'
    price_level: float
    volume_confirmation: bool
    volume_confidence: float
    trend_alignment: str
    confirmation_timeframes: List[str]
    timeframe_confidences: Dict[str, float]
    timeframe_alignments: Dict[str, str]
    failure_probability: float
    processing_latency_ms: float
    
    # Volume Profile Analysis
    poc_level: Optional[float] = None
    value_area_high: Optional[float] = None
    value_area_low: Optional[float] = None
    volume_profile_confidence: Optional[float] = None
    volume_nodes_count: Optional[int] = None
    volume_gaps_count: Optional[int] = None
    
    # Elliott Wave Analysis
    current_wave: Optional[str] = None
    wave_count: Optional[int] = None
    pattern_type_elliott: Optional[str] = None
    trend_direction_elliott: Optional[str] = None
    next_target_elliott: Optional[float] = None
    elliott_confidence: Optional[float] = None
    fibonacci_levels: Optional[Dict[str, float]] = None
    
    # Wyckoff Analysis
    wyckoff_pattern: Optional[str] = None
    wyckoff_confidence: Optional[float] = None
    wyckoff_phase: Optional[str] = None
    
    # SMC Analysis
    smc_patterns: Optional[List[str]] = None
    smc_confidence: Optional[float] = None
    order_blocks_count: Optional[int] = None
    fair_value_gaps_count: Optional[int] = None
    
    # Metadata
    metadata: Optional[Dict[str, Any]] = None

class EnhancedMultiTimeframePatternEngine:
    """
    Enhanced multi-timeframe pattern engine with comprehensive analysis
    Integrates Volume Profile, Elliott Wave, Wyckoff, and SMC analysis
    """
    
    def __init__(self, db_config: Dict[str, Any], max_workers: int = 4):
        """
        Initialize enhanced multi-timeframe pattern engine
        
        Args:
            db_config: Database configuration
            max_workers: Maximum number of worker threads
        """
        self.db_config = db_config
        self.max_workers = max_workers
        
        # Initialize analyzers
        self.pattern_detector = VectorizedPatternDetector(max_workers=max_workers)
        self.volume_analyzer = VolumeAnalyzer()
        self.volume_profile_analyzer = VolumeProfileAnalyzer()
        self.elliott_wave_analyzer = ElliottWaveAnalyzer()
        
        # Database connection
        self.db_connection = None
        
        # Performance tracking
        self.performance_metrics = {
            'total_patterns_detected': 0,
            'volume_profile_patterns': 0,
            'elliott_wave_patterns': 0,
            'wyckoff_patterns': 0,
            'smc_patterns': 0,
            'average_processing_time': 0.0
        }
        
        logger.info("Enhanced Multi-Timeframe Pattern Engine initialized")
    
    async def initialize(self):
        """Initialize database connection"""
        try:
            self.db_connection = TimescaleDBConnection(self.db_config)
            await self.db_connection.initialize()
            logger.info("Database connection initialized")
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            # Don't raise, just log the error
            self.db_connection = None
    
    async def detect_enhanced_multi_timeframe_patterns(self, 
                                                     symbol: str, 
                                                     primary_timeframe: str,
                                                     candlestick_data: Dict[str, pd.DataFrame]) -> List[EnhancedMultiTimeframePattern]:
        """
        Detect enhanced multi-timeframe patterns with comprehensive analysis
        
        Args:
            symbol: Trading symbol
            primary_timeframe: Primary timeframe for analysis
            candlestick_data: Dictionary of timeframe -> DataFrame
            
        Returns:
            List of enhanced multi-timeframe patterns
        """
        start_time = datetime.now()
        
        try:
            if not candlestick_data or primary_timeframe not in candlestick_data:
                logger.warning(f"No data available for {symbol} on {primary_timeframe}")
                return []
            
            primary_df = candlestick_data[primary_timeframe]
            if len(primary_df) < 50:  # Minimum data requirement
                logger.warning(f"Insufficient data for {symbol}: {len(primary_df)} candles")
                return []
            
            logger.info(f"Starting enhanced pattern detection for {symbol} on {primary_timeframe}")
            
            # 1. Traditional Pattern Detection
            primary_patterns = await self.pattern_detector.detect_patterns_vectorized(
                primary_df, use_talib=True, use_incremental=True
            )
            
            # 2. Wyckoff Pattern Detection
            wyckoff_patterns = self.volume_analyzer.detect_wyckoff_patterns(
                primary_df, symbol, primary_timeframe
            )
            
            # 3. Volume Profile Analysis
            volume_profile_analysis = self.volume_profile_analyzer.analyze_volume_profile(
                primary_df, symbol, primary_timeframe
            )
            
            # 4. Elliott Wave Analysis
            elliott_wave_analysis = self.elliott_wave_analyzer.analyze_elliott_waves(
                primary_df, symbol, primary_timeframe
            )
            
            # 5. SMC Pattern Detection (if available)
            smc_patterns = []
            try:
                from src.data.enhanced_real_time_pipeline import EnhancedRealTimePipeline
                smc_analyzer = EnhancedRealTimePipeline()
                smc_analysis = smc_analyzer.analyze_smc_patterns(primary_df, symbol, primary_timeframe)
                if smc_analysis:
                    smc_patterns = smc_analysis
            except ImportError:
                logger.info("SMC analyzer not available, skipping SMC analysis")
            
            # 6. Combine and enhance patterns
            enhanced_patterns = []
            
            # Convert Wyckoff patterns to VectorizedPattern format
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
                logger.info(f"No patterns detected for {symbol}")
                return []
            
            # Process each pattern for enhanced multi-timeframe confirmation
            for primary_pattern in all_patterns:
                enhanced_pattern = await self._create_enhanced_pattern(
                    symbol, primary_pattern, candlestick_data, primary_timeframe,
                    volume_profile_analysis, elliott_wave_analysis, smc_patterns
                )
                
                if enhanced_pattern:
                    enhanced_patterns.append(enhanced_pattern)
            
            # Update performance metrics
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            self._update_performance_metrics(enhanced_patterns, processing_time)
            
            # Log performance for debugging
            logger.info(f"Enhanced pattern detection performance: {processing_time:.2f}ms for {len(enhanced_patterns)} patterns")
            
            logger.info(f"Enhanced pattern detection completed for {symbol}: {len(enhanced_patterns)} patterns in {processing_time:.2f}ms")
            
            return enhanced_patterns
            
        except Exception as e:
            logger.error(f"Enhanced pattern detection failed for {symbol}: {e}")
            return []
    
    async def _create_enhanced_pattern(self, 
                                     symbol: str, 
                                     primary_pattern: VectorizedPattern,
                                     candlestick_data: Dict[str, pd.DataFrame],
                                     primary_timeframe: str,
                                     volume_profile_analysis: VolumeProfileAnalysis,
                                     elliott_wave_analysis: ElliottWaveAnalysis,
                                     smc_patterns: List[Any]) -> Optional[EnhancedMultiTimeframePattern]:
        """Create enhanced pattern with comprehensive analysis"""
        try:
            # Confirm pattern across other timeframes
            confirmation_result = await self._confirm_pattern_across_timeframes(
                symbol, primary_pattern, candlestick_data, primary_timeframe
            )
            
            if not confirmation_result:
                return None
            
            # Create enhanced pattern
            enhanced_pattern = EnhancedMultiTimeframePattern(
                symbol=symbol,
                timestamp=primary_pattern.timestamp,
                primary_timeframe=primary_timeframe,
                pattern_name=primary_pattern.pattern_name,
                pattern_type=primary_pattern.pattern_type,
                confidence=primary_pattern.confidence,
                strength=primary_pattern.strength,
                price_level=primary_pattern.price_level,
                volume_confirmation=primary_pattern.volume_confirmation,
                volume_confidence=primary_pattern.volume_confidence,
                trend_alignment=primary_pattern.trend_alignment,
                confirmation_timeframes=confirmation_result['confirmation_timeframes'],
                timeframe_confidences=confirmation_result['timeframe_confidences'],
                timeframe_alignments=confirmation_result['timeframe_alignments'],
                failure_probability=confirmation_result['failure_probability'],
                processing_latency_ms=confirmation_result['processing_latency_ms'],
                
                # Volume Profile Analysis
                poc_level=volume_profile_analysis.poc_level,
                value_area_high=volume_profile_analysis.value_area_high,
                value_area_low=volume_profile_analysis.value_area_low,
                volume_profile_confidence=volume_profile_analysis.confidence_score,
                volume_nodes_count=len(volume_profile_analysis.volume_nodes),
                volume_gaps_count=len(volume_profile_analysis.volume_gaps),
                
                # Elliott Wave Analysis
                current_wave=elliott_wave_analysis.current_wave.value if elliott_wave_analysis.current_wave else None,
                wave_count=elliott_wave_analysis.wave_count,
                pattern_type_elliott=elliott_wave_analysis.pattern_type.value,
                trend_direction_elliott=elliott_wave_analysis.trend_direction,
                next_target_elliott=elliott_wave_analysis.next_target,
                elliott_confidence=elliott_wave_analysis.confidence_score,
                fibonacci_levels=elliott_wave_analysis.fibonacci_levels,
                
                # Wyckoff Analysis (extract from primary pattern if it's a Wyckoff pattern)
                wyckoff_pattern=primary_pattern.pattern_name if 'wyckoff' in primary_pattern.pattern_name.lower() else None,
                wyckoff_confidence=primary_pattern.confidence if 'wyckoff' in primary_pattern.pattern_name.lower() else None,
                wyckoff_phase=self._extract_wyckoff_phase(primary_pattern),
                
                # SMC Analysis
                smc_patterns=[pattern.pattern_type if hasattr(pattern, 'pattern_type') else str(pattern) for pattern in smc_patterns] if smc_patterns and isinstance(smc_patterns, list) else None,
                smc_confidence=np.mean([pattern.confidence if hasattr(pattern, 'confidence') else 0.0 for pattern in smc_patterns]) if smc_patterns and isinstance(smc_patterns, list) else None,
                order_blocks_count=len([p for p in smc_patterns if hasattr(p, 'pattern_type') and p.pattern_type == 'order_block']) if smc_patterns and isinstance(smc_patterns, list) else None,
                fair_value_gaps_count=len([p for p in smc_patterns if hasattr(p, 'pattern_type') and p.pattern_type == 'fair_value_gap']) if smc_patterns and isinstance(smc_patterns, list) else None,
                
                # Metadata
                metadata={
                    'volume_profile_analysis': {
                        'total_volume': volume_profile_analysis.total_volume,
                        'price_range': volume_profile_analysis.price_range,
                        'single_prints_count': len(volume_profile_analysis.single_prints),
                        'volume_climax_count': len(volume_profile_analysis.volume_climax_levels)
                    },
                    'elliott_wave_analysis': {
                        'support_levels_count': len(elliott_wave_analysis.support_levels),
                        'resistance_levels_count': len(elliott_wave_analysis.resistance_levels),
                        'waves_analyzed': len(elliott_wave_analysis.waves)
                    },
                    'enhanced_analysis': True
                }
            )
            
            return enhanced_pattern
            
        except Exception as e:
            logger.error(f"Failed to create enhanced pattern: {e}")
            return None
    
    async def _confirm_pattern_across_timeframes(self, 
                                               symbol: str, 
                                               primary_pattern: VectorizedPattern,
                                               candlestick_data: Dict[str, pd.DataFrame],
                                               primary_timeframe: str) -> Optional[Dict[str, Any]]:
        """Confirm pattern across multiple timeframes"""
        try:
            confirmation_timeframes = []
            timeframe_confidences = {}
            timeframe_alignments = {}
            
            # Check each timeframe for pattern confirmation
            for timeframe, df in candlestick_data.items():
                if timeframe == primary_timeframe or len(df) < 20:
                    continue
                
                # Detect patterns on this timeframe
                patterns = await self.pattern_detector.detect_patterns_vectorized(
                    df, use_talib=True, use_incremental=True
                )
                
                # Check for similar patterns
                for pattern in patterns:
                    if self._is_similar_pattern(primary_pattern, pattern):
                        confirmation_timeframes.append(timeframe)
                        timeframe_confidences[timeframe] = pattern.confidence
                        timeframe_alignments[timeframe] = pattern.trend_alignment
                        break
            
            # Calculate failure probability
            failure_probability = self._calculate_enhanced_failure_probability(
                primary_pattern, timeframe_confidences, primary_pattern.trend_alignment
            )
            
            # Calculate processing latency
            processing_latency_ms = 0.0  # Will be updated by caller
            
            return {
                'confirmation_timeframes': confirmation_timeframes,
                'timeframe_confidences': timeframe_confidences,
                'timeframe_alignments': timeframe_alignments,
                'failure_probability': failure_probability,
                'processing_latency_ms': processing_latency_ms
            }
            
        except Exception as e:
            logger.error(f"Pattern confirmation failed: {e}")
            return None
    
    def _is_similar_pattern(self, pattern1: VectorizedPattern, pattern2: VectorizedPattern) -> bool:
        """Check if two patterns are similar"""
        try:
            # Check pattern name similarity
            name_similarity = pattern1.pattern_name.lower() == pattern2.pattern_name.lower()
            
            # Check pattern type alignment
            type_alignment = pattern1.pattern_type == pattern2.pattern_type
            
            # Check trend alignment
            trend_alignment = pattern1.trend_alignment == pattern2.trend_alignment
            
            # Check confidence threshold
            confidence_threshold = pattern2.confidence > 0.5
            
            return name_similarity and type_alignment and trend_alignment and confidence_threshold
            
        except Exception as e:
            logger.error(f"Pattern similarity check failed: {e}")
            return False
    
    def _calculate_enhanced_failure_probability(self, 
                                              primary_pattern: VectorizedPattern,
                                              timeframe_confidences: Dict[str, float],
                                              trend_alignment: str) -> float:
        """Calculate enhanced failure probability with multiple factors"""
        try:
            # Base failure rates for different pattern types
            base_failure_rates = {
                # Traditional patterns
                "doji": 0.4,
                "hammer": 0.3,
                "engulfing": 0.25,
                "morning_star": 0.2,
                "evening_star": 0.2,
                
                # Wyckoff patterns (generally more reliable)
                "wyckoff_spring": 0.15,
                "wyckoff_upthrust": 0.15,
                "wyckoff_accumulation": 0.2,
                "wyckoff_distribution": 0.2,
                "wyckoff_test": 0.25,
                "wyckoff_sign_of_strength": 0.1,
                "wyckoff_sign_of_weakness": 0.1
            }
            
            # Get base failure rate
            base_rate = base_failure_rates.get(primary_pattern.pattern_name.lower(), 0.3)
            
            # Adjust based on confidence
            confidence_adjustment = 1.0 - primary_pattern.confidence
            
            # Adjust based on timeframe confirmations
            confirmation_adjustment = 0.0
            if timeframe_confidences:
                avg_confidence = np.mean(list(timeframe_confidences.values()))
                confirmation_adjustment = (1.0 - avg_confidence) * 0.2
            
            # Adjust based on volume confirmation
            volume_adjustment = 0.0
            if not primary_pattern.volume_confirmation:
                volume_adjustment = 0.1
            
            # Calculate final failure probability
            failure_probability = base_rate + confidence_adjustment + confirmation_adjustment + volume_adjustment
            
            # Ensure within bounds
            return max(0.0, min(1.0, failure_probability))
            
        except Exception as e:
            logger.error(f"Failure probability calculation failed: {e}")
            return 0.5
    
    def _extract_wyckoff_phase(self, pattern: VectorizedPattern) -> Optional[str]:
        """Extract Wyckoff phase from pattern"""
        try:
            pattern_name = pattern.pattern_name.lower()
            
            if 'spring' in pattern_name:
                return 'accumulation'
            elif 'upthrust' in pattern_name:
                return 'distribution'
            elif 'test' in pattern_name:
                return 'test'
            elif 'strength' in pattern_name:
                return 'markup'
            elif 'weakness' in pattern_name:
                return 'markdown'
            else:
                return None
                
        except Exception as e:
            logger.error(f"Wyckoff phase extraction failed: {e}")
            return None
    
    def _update_performance_metrics(self, patterns: List[EnhancedMultiTimeframePattern], processing_time: float):
        """Update performance metrics"""
        try:
            self.performance_metrics['total_patterns_detected'] += len(patterns)
            
            for pattern in patterns:
                if pattern.volume_profile_confidence:
                    self.performance_metrics['volume_profile_patterns'] += 1
                if pattern.elliott_confidence:
                    self.performance_metrics['elliott_wave_patterns'] += 1
                if pattern.wyckoff_pattern:
                    self.performance_metrics['wyckoff_patterns'] += 1
                if pattern.smc_patterns:
                    self.performance_metrics['smc_patterns'] += 1
            
            # Update average processing time
            current_avg = self.performance_metrics['average_processing_time']
            total_patterns = self.performance_metrics['total_patterns_detected']
            if total_patterns > 0:
                self.performance_metrics['average_processing_time'] = (
                    (current_avg * (total_patterns - len(patterns)) + processing_time) / total_patterns
                )
            else:
                self.performance_metrics['average_processing_time'] = processing_time
            
        except Exception as e:
            logger.error(f"Performance metrics update failed: {e}")
    
    async def store_enhanced_patterns(self, patterns: List[EnhancedMultiTimeframePattern]):
        """Store enhanced patterns in database"""
        try:
            if not self.db_connection:
                logger.warning("Database connection not available")
                return
            
            async with self.db_connection.get_async_session() as session:
                for pattern in patterns:
                    # Store in enhanced_patterns table
                    query = """
                        INSERT INTO enhanced_patterns (
                            symbol, timestamp, primary_timeframe, pattern_name, pattern_type,
                            confidence, strength, price_level, volume_confirmation, volume_confidence,
                            trend_alignment, confirmation_timeframes, timeframe_confidences,
                            timeframe_alignments, failure_probability, processing_latency_ms,
                            poc_level, value_area_high, value_area_low, volume_profile_confidence,
                            volume_nodes_count, volume_gaps_count, current_wave, wave_count,
                            pattern_type_elliott, trend_direction_elliott, next_target_elliott,
                            elliott_confidence, fibonacci_levels, wyckoff_pattern, wyckoff_confidence,
                            wyckoff_phase, smc_patterns, smc_confidence, order_blocks_count,
                            fair_value_gaps_count, metadata
                        ) VALUES (
                            :symbol, :timestamp, :primary_timeframe, :pattern_name, :pattern_type,
                            :confidence, :strength, :price_level, :volume_confirmation, :volume_confidence,
                            :trend_alignment, :confirmation_timeframes, :timeframe_confidences,
                            :timeframe_alignments, :failure_probability, :processing_latency_ms,
                            :poc_level, :value_area_high, :value_area_low, :volume_profile_confidence,
                            :volume_nodes_count, :volume_gaps_count, :current_wave, :wave_count,
                            :pattern_type_elliott, :trend_direction_elliott, :next_target_elliott,
                            :elliott_confidence, :fibonacci_levels, :wyckoff_pattern, :wyckoff_confidence,
                            :wyckoff_phase, :smc_patterns, :smc_confidence, :order_blocks_count,
                            :fair_value_gaps_count, :metadata
                        )
                    """
                    
                    await session.execute(query, {
                        "symbol": pattern.symbol,
                        "timestamp": pattern.timestamp,
                        "primary_timeframe": pattern.primary_timeframe,
                        "pattern_name": pattern.pattern_name,
                        "pattern_type": pattern.pattern_type,
                        "confidence": float(pattern.confidence),
                        "strength": pattern.strength,
                        "price_level": pattern.price_level,
                        "volume_confirmation": pattern.volume_confirmation,
                        "volume_confidence": float(pattern.volume_confidence),
                        "trend_alignment": pattern.trend_alignment,
                        "confirmation_timeframes": json.dumps(pattern.confirmation_timeframes),
                        "timeframe_confidences": json.dumps(pattern.timeframe_confidences),
                        "timeframe_alignments": json.dumps(pattern.timeframe_alignments),
                        "failure_probability": float(pattern.failure_probability),
                        "processing_latency_ms": float(pattern.processing_latency_ms),
                        "poc_level": pattern.poc_level,
                        "value_area_high": pattern.value_area_high,
                        "value_area_low": pattern.value_area_low,
                        "volume_profile_confidence": pattern.volume_profile_confidence,
                        "volume_nodes_count": pattern.volume_nodes_count,
                        "volume_gaps_count": pattern.volume_gaps_count,
                        "current_wave": pattern.current_wave,
                        "wave_count": pattern.wave_count,
                        "pattern_type_elliott": pattern.pattern_type_elliott,
                        "trend_direction_elliott": pattern.trend_direction_elliott,
                        "next_target_elliott": pattern.next_target_elliott,
                        "elliott_confidence": pattern.elliott_confidence,
                        "fibonacci_levels": json.dumps(pattern.fibonacci_levels) if pattern.fibonacci_levels else None,
                        "wyckoff_pattern": pattern.wyckoff_pattern,
                        "wyckoff_confidence": pattern.wyckoff_confidence,
                        "wyckoff_phase": pattern.wyckoff_phase,
                        "smc_patterns": json.dumps(pattern.smc_patterns) if pattern.smc_patterns else None,
                        "smc_confidence": pattern.smc_confidence,
                        "order_blocks_count": pattern.order_blocks_count,
                        "fair_value_gaps_count": pattern.fair_value_gaps_count,
                        "metadata": json.dumps(pattern.metadata) if pattern.metadata else '{}'
                    })
                
                await session.commit()
                logger.info(f"Stored {len(patterns)} enhanced patterns in database")
                
        except Exception as e:
            logger.error(f"Failed to store enhanced patterns: {e}")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        return self.performance_metrics.copy()
    
    async def close(self):
        """Close database connection"""
        try:
            if self.db_connection:
                await self.db_connection.close()
                logger.info("Database connection closed")
        except Exception as e:
            logger.error(f"Failed to close database connection: {e}")
