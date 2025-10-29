"""
Async Pattern Detector for AlphaPlus
Runs pattern checks across multiple timeframes in parallel using async coroutines
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone
import numpy as np
from dataclasses import dataclass

from backend.strategies.ultra_fast_pattern_detector import UltraFastPatternDetector, PatternDetectionResult
from backend.strategies.sliding_window_buffer import AsyncSlidingWindowBuffer

logger = logging.getLogger(__name__)

@dataclass
class MultiTimeframeResult:
    """Result of multi-timeframe pattern detection"""
    symbol: str
    timeframe: str
    patterns: List[PatternDetectionResult]
    detection_latency_ms: float
    confidence_score: float
    metadata: Dict[str, Any]
    
    # Phase 4A: Enhanced multi-timeframe fields
    timeframe_hierarchy: Optional[Dict[str, Any]] = None
    multi_timeframe_alignment: Optional[float] = None
    timeframe_confirmation_count: Optional[int] = None
    market_regime: Optional[str] = None
    regime_adjusted_confidence: Optional[float] = None

class AsyncPatternDetector:
    """Async pattern detector with parallel multi-timeframe processing"""
    
    def __init__(self, max_concurrent_detections: int = 10):
        self.max_concurrent_detections = max_concurrent_detections
        self.ultra_fast_detector = UltraFastPatternDetector()
        self.async_buffer = AsyncSlidingWindowBuffer(max_size=1000)
        self.semaphore = asyncio.Semaphore(max_concurrent_detections)
        
        # Phase 4A: Initialize quality scorer for multi-timeframe analysis
        from .signal_quality_scorer import SignalQualityScorer, QualityScoringConfig
        quality_config = QualityScoringConfig(enable_multi_timeframe_scoring=True)
        self.quality_scorer = SignalQualityScorer(quality_config)
        
        # Performance tracking
        self.performance_stats = {
            'total_detections': 0,
            'avg_latency_ms': 0.0,
            'concurrent_detections': 0,
            'timeframe_combinations': 0
        }
        
        # Supported timeframes for parallel processing
        self.supported_timeframes = ['1m', '5m', '15m', '30m', '1h', '4h', '1d']
        
        # Phase 4A: Multi-timeframe hierarchy configuration
        self.timeframe_hierarchy = {
            '1m': ['1m'],
            '5m': ['1m', '5m'],
            '15m': ['5m', '15m'],
            '30m': ['15m', '30m'],
            '1h': ['30m', '1h'],
            '4h': ['1h', '4h'],
            '1d': ['4h', '1d']
        }
        
        logger.info(f"⚡ Async pattern detector initialized with Phase 4A multi-timeframe enhancements")
    
    async def detect_patterns_multi_timeframe(self, symbol: str, 
                                            timeframes: List[str],
                                            candles_data: Dict[str, List[Dict]]) -> List[MultiTimeframeResult]:
        """Detect patterns across multiple timeframes in parallel"""
        start_time = datetime.now()
        
        try:
            # Validate timeframes
            valid_timeframes = [tf for tf in timeframes if tf in self.supported_timeframes]
            if not valid_timeframes:
                logger.warning(f"No valid timeframes found for {symbol}")
                return []
            
            # Create detection tasks for each timeframe
            detection_tasks = []
            for timeframe in valid_timeframes:
                if timeframe in candles_data and candles_data[timeframe]:
                    task = asyncio.create_task(
                        self._detect_single_timeframe(symbol, timeframe, candles_data[timeframe])
                    )
                    detection_tasks.append(task)
            
            # Wait for all detections to complete
            results = await asyncio.gather(*detection_tasks, return_exceptions=True)
            
            # Process results
            multi_timeframe_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Error in timeframe {valid_timeframes[i]}: {result}")
                    continue
                if result:
                    multi_timeframe_results.append(result)
            
            # Calculate overall performance
            end_time = datetime.now()
            total_latency = (end_time - start_time).total_seconds() * 1000
            
            # Update performance stats
            self._update_performance_stats(total_latency, len(multi_timeframe_results))
            
            logger.info(f"⚡ Multi-timeframe detection completed: {len(multi_timeframe_results)} timeframes in {total_latency:.2f}ms")
            
            return multi_timeframe_results
            
        except Exception as e:
            logger.error(f"Error in multi-timeframe detection: {e}")
            return []
    
    async def analyze_multi_timeframe_alignment(self, symbol: str, 
                                              multi_timeframe_results: List[MultiTimeframeResult]) -> Dict[str, Any]:
        """Analyze multi-timeframe alignment and hierarchy for a symbol"""
        try:
            alignment_analysis = {
                'symbol': symbol,
                'timeframe_hierarchy': {},
                'overall_alignment_score': 0.0,
                'confirmation_count': 0,
                'conflicting_patterns': [],
                'strongest_timeframe': None,
                'alignment_details': {}
            }
            
            # Group patterns by timeframe
            timeframe_patterns = {}
            for result in multi_timeframe_results:
                timeframe_patterns[result.timeframe] = result.patterns
            
            # Analyze hierarchy alignment
            for primary_tf, hierarchy_tfs in self.timeframe_hierarchy.items():
                if primary_tf not in timeframe_patterns:
                    continue
                
                primary_patterns = timeframe_patterns[primary_tf]
                hierarchy_analysis = {
                    'primary_patterns': len(primary_patterns),
                    'confirming_patterns': 0,
                    'conflicting_patterns': 0,
                    'alignment_score': 0.0,
                    'hierarchy_details': {}
                }
                
                # Check alignment with hierarchy timeframes
                for hierarchy_tf in hierarchy_tfs:
                    if hierarchy_tf == primary_tf:
                        continue
                    
                    if hierarchy_tf not in timeframe_patterns:
                        continue
                    
                    hierarchy_patterns = timeframe_patterns[hierarchy_tf]
                    tf_alignment = self._analyze_timeframe_alignment(
                        primary_patterns, hierarchy_patterns, primary_tf, hierarchy_tf
                    )
                    
                    hierarchy_analysis['hierarchy_details'][hierarchy_tf] = tf_alignment
                    hierarchy_analysis['confirming_patterns'] += tf_alignment['confirming_count']
                    hierarchy_analysis['conflicting_patterns'] += tf_alignment['conflicting_count']
                    hierarchy_analysis['alignment_score'] += tf_alignment['alignment_score']
                
                # Normalize alignment score
                if hierarchy_tfs:
                    hierarchy_analysis['alignment_score'] /= len(hierarchy_tfs)
                
                alignment_analysis['timeframe_hierarchy'][primary_tf] = hierarchy_analysis
                alignment_analysis['overall_alignment_score'] += hierarchy_analysis['alignment_score']
                alignment_analysis['confirmation_count'] += hierarchy_analysis['confirming_patterns']
            
            # Find strongest timeframe
            if alignment_analysis['timeframe_hierarchy']:
                strongest_tf = max(alignment_analysis['timeframe_hierarchy'].items(), 
                                 key=lambda x: x[1]['alignment_score'])
                alignment_analysis['strongest_timeframe'] = strongest_tf[0]
            
            return alignment_analysis
            
        except Exception as e:
            logger.error(f"Error in multi-timeframe alignment analysis: {e}")
            return {}
    
    def _analyze_timeframe_alignment(self, primary_patterns: List[PatternDetectionResult], 
                                   hierarchy_patterns: List[PatternDetectionResult],
                                   primary_tf: str, hierarchy_tf: str) -> Dict[str, Any]:
        """Analyze alignment between two timeframes"""
        alignment_result = {
            'confirming_count': 0,
            'conflicting_count': 0,
            'alignment_score': 0.0,
            'pattern_matches': []
        }
        
        # Get timeframe durations in seconds
        tf_durations = {
            '1m': 60, '5m': 300, '15m': 900, '30m': 1800,
            '1h': 3600, '4h': 14400, '1d': 86400
        }
        
        primary_duration = tf_durations.get(primary_tf, 3600)
        hierarchy_duration = tf_durations.get(hierarchy_tf, 3600)
        
        # Allow ±2 candles overlap
        max_time_diff = max(primary_duration, hierarchy_duration) * 2
        
        for primary_pattern in primary_patterns:
            primary_timestamp = primary_pattern.timestamp
            primary_direction = primary_pattern.direction
            
            for hierarchy_pattern in hierarchy_patterns:
                hierarchy_timestamp = hierarchy_pattern.timestamp
                hierarchy_direction = hierarchy_pattern.direction
                
                # Check if patterns are in the same time window
                time_diff = abs((primary_timestamp - hierarchy_timestamp).total_seconds())
                
                if time_diff <= max_time_diff:
                    if primary_direction == hierarchy_direction:
                        alignment_result['confirming_count'] += 1
                        alignment_result['alignment_score'] += 0.3
                    elif hierarchy_direction != 'neutral':
                        alignment_result['conflicting_count'] += 1
                        alignment_result['alignment_score'] -= 0.1
                    
                    alignment_result['pattern_matches'].append({
                        'primary_pattern': primary_pattern.pattern_name,
                        'hierarchy_pattern': hierarchy_pattern.pattern_name,
                        'primary_direction': primary_direction,
                        'hierarchy_direction': hierarchy_direction,
                        'time_diff_seconds': time_diff
                    })
        
        return alignment_result
    
    async def _detect_single_timeframe(self, symbol: str, timeframe: str, 
                                     candles: List[Dict]) -> Optional[MultiTimeframeResult]:
        """Detect patterns for a single timeframe with concurrency control"""
        async with self.semaphore:
            try:
                start_time = datetime.now()
                
                # Update sliding buffer
                for candle in candles:
                    await self.async_buffer.add_candle_async(symbol, timeframe, candle)
                
                # Detect patterns using ultra-fast detector
                patterns = await self.ultra_fast_detector.detect_patterns_async(
                    symbol, timeframe, candles
                )
                
                # Calculate confidence score
                confidence_score = self._calculate_timeframe_confidence(patterns)
                
                # Calculate detection latency
                end_time = datetime.now()
                latency_ms = (end_time - start_time).total_seconds() * 1000
                
                return MultiTimeframeResult(
                    symbol=symbol,
                    timeframe=timeframe,
                    patterns=patterns,
                    detection_latency_ms=latency_ms,
                    confidence_score=confidence_score,
                    metadata={
                        'candles_processed': len(candles),
                        'patterns_detected': len(patterns),
                        'detection_method': 'async_parallel'
                    }
                )
                
            except Exception as e:
                logger.error(f"Error detecting patterns for {symbol} {timeframe}: {e}")
                return None
    
    async def detect_patterns_with_correlation(self, symbol: str, 
                                             timeframes: List[str],
                                             candles_data: Dict[str, List[Dict]],
                                             correlation_symbols: List[str] = None) -> List[MultiTimeframeResult]:
        """Detect patterns with correlation analysis across timeframes"""
        try:
            # Primary pattern detection
            primary_results = await self.detect_patterns_multi_timeframe(symbol, timeframes, candles_data)
            
            # Correlation analysis if correlation symbols provided
            if correlation_symbols:
                correlation_results = await self._analyze_correlations(
                    primary_results, correlation_symbols, timeframes
                )
                
                # Merge correlation data into primary results
                for result in primary_results:
                    if result.symbol in correlation_results:
                        result.metadata['correlation_data'] = correlation_results[result.symbol]
                        result.confidence_score *= correlation_results[result.symbol].get('correlation_multiplier', 1.0)
            
            return primary_results
            
        except Exception as e:
            logger.error(f"Error in correlation-based detection: {e}")
            return []
    
    async def _analyze_correlations(self, primary_results: List[MultiTimeframeResult],
                                  correlation_symbols: List[str],
                                  timeframes: List[str]) -> Dict[str, Dict[str, Any]]:
        """Analyze correlations between primary symbol and correlation symbols"""
        try:
            correlation_data = {}
            
            for result in primary_results:
                symbol_correlations = {}
                
                for corr_symbol in correlation_symbols:
                    if corr_symbol != result.symbol:
                        # Get recent patterns for correlation symbol
                        corr_patterns = await self._get_correlation_patterns(
                            corr_symbol, result.timeframe
                        )
                        
                        # Calculate correlation strength
                        correlation_strength = self._calculate_correlation_strength(
                            result.patterns, corr_patterns
                        )
                        
                        symbol_correlations[corr_symbol] = {
                            'correlation_strength': correlation_strength,
                            'recent_patterns': len(corr_patterns),
                            'timestamp': datetime.now(timezone.utc).isoformat()
                        }
                
                # Calculate overall correlation multiplier
                if symbol_correlations:
                    avg_correlation = sum(
                        corr['correlation_strength'] for corr in symbol_correlations.values()
                    ) / len(symbol_correlations)
                    
                    symbol_correlations['correlation_multiplier'] = 1.0 + (avg_correlation * 0.2)
                
                correlation_data[result.symbol] = symbol_correlations
            
            return correlation_data
            
        except Exception as e:
            logger.error(f"Error analyzing correlations: {e}")
            return {}
    
    async def _get_correlation_patterns(self, symbol: str, timeframe: str) -> List[Dict]:
        """Get recent patterns for correlation analysis"""
        try:
            # Get recent candles from buffer
            recent_candles = await self.async_buffer.get_recent_candles_async(symbol, timeframe, 100)
            
            if len(recent_candles) < 5:
                return []
            
            # Detect patterns for correlation symbol
            patterns = await self.ultra_fast_detector.detect_patterns_async(
                symbol, timeframe, recent_candles
            )
            
            # Convert to simple format for correlation analysis
            return [
                {
                    'pattern_name': p.pattern_name,
                    'direction': p.direction,
                    'confidence': p.confidence,
                    'timestamp': p.timestamp
                }
                for p in patterns
            ]
            
        except Exception as e:
            logger.error(f"Error getting correlation patterns: {e}")
            return []
    
    def _calculate_correlation_strength(self, primary_patterns: List[PatternDetectionResult],
                                      correlation_patterns: List[Dict]) -> float:
        """Calculate correlation strength between primary and correlation patterns"""
        try:
            if not primary_patterns or not correlation_patterns:
                return 0.5  # Neutral correlation
            
            # Count matching patterns
            matching_patterns = 0
            total_patterns = len(primary_patterns)
            
            for primary in primary_patterns:
                for corr in correlation_patterns:
                    if (primary.pattern_name == corr['pattern_name'] and 
                        primary.direction == corr['direction']):
                        matching_patterns += 1
                        break
            
            # Calculate correlation strength
            if total_patterns > 0:
                correlation_strength = matching_patterns / total_patterns
                return min(correlation_strength, 1.0)
            
            return 0.5
            
        except Exception as e:
            logger.error(f"Error calculating correlation strength: {e}")
            return 0.5
    
    def _calculate_timeframe_confidence(self, patterns: List[PatternDetectionResult]) -> float:
        """Calculate overall confidence score for a timeframe"""
        try:
            if not patterns:
                return 0.0
            
            # Calculate weighted average confidence
            total_confidence = 0.0
            total_weight = 0.0
            
            for pattern in patterns:
                # Weight by pattern strength
                weight = 1.0
                if pattern.strength == 'strong':
                    weight = 3.0
                elif pattern.strength == 'moderate':
                    weight = 2.0
                
                total_confidence += pattern.confidence * weight
                total_weight += weight
            
            return total_confidence / total_weight if total_weight > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating timeframe confidence: {e}")
            return 0.0
    
    def _update_performance_stats(self, latency_ms: float, num_results: int):
        """Update performance statistics"""
        try:
            self.performance_stats['total_detections'] += num_results
            
            # Update average latency
            if self.performance_stats['total_detections'] > 0:
                current_avg = self.performance_stats['avg_latency_ms']
                total_detections = self.performance_stats['total_detections']
                new_avg = ((current_avg * (total_detections - num_results)) + 
                           (latency_ms * num_results)) / total_detections
                self.performance_stats['avg_latency_ms'] = new_avg
            
            # Update concurrent detections
            self.performance_stats['concurrent_detections'] = self.max_concurrent_detections - self.semaphore._value
            
        except Exception as e:
            logger.error(f"Error updating performance stats: {e}")
    
    async def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        try:
            # Get buffer stats
            buffer_stats = await self.async_buffer.get_global_stats_async()
            
            # Get ultra-fast detector stats
            detector_stats = self.ultra_fast_detector.get_performance_stats()
            
            return {
                'async_detector_stats': self.performance_stats,
                'buffer_stats': buffer_stats,
                'ultra_fast_detector_stats': detector_stats,
                'supported_timeframes': self.supported_timeframes,
                'max_concurrent_detections': self.max_concurrent_detections,
                'current_concurrent_detections': self.performance_stats['concurrent_detections'],
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting performance summary: {e}")
            return {
                'async_detector_stats': self.performance_stats,
                'error': str(e),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
    
    async def clear_all_caches(self):
        """Clear all caches and buffers"""
        try:
            await self.async_buffer.clear_all_buffers_async()
            self.ultra_fast_detector.clear_cache()
            logger.info("🧹 All caches cleared")
            
        except Exception as e:
            logger.error(f"Error clearing caches: {e}")
    
    async def update_concurrency_limit(self, new_limit: int):
        """Update the maximum concurrent detection limit"""
        try:
            self.max_concurrent_detections = new_limit
            self.semaphore = asyncio.Semaphore(new_limit)
            logger.info(f"Updated concurrency limit to {new_limit}")
            
        except Exception as e:
            logger.error(f"Error updating concurrency limit: {e}")

class PatternDetectionOrchestrator:
    """Orchestrates pattern detection across multiple symbols and timeframes"""
    
    def __init__(self, max_concurrent_symbols: int = 5):
        self.async_detector = AsyncPatternDetector()
        self.max_concurrent_symbols = max_concurrent_symbols
        self.symbol_semaphore = asyncio.Semaphore(max_concurrent_symbols)
        
        logger.info(f"🎼 Pattern detection orchestrator initialized with max_concurrent_symbols={max_concurrent_symbols}")
    
    async def detect_patterns_bulk(self, symbols_data: Dict[str, Dict[str, List[Dict]]],
                                 timeframes: List[str] = None) -> Dict[str, List[MultiTimeframeResult]]:
        """Detect patterns for multiple symbols in parallel"""
        try:
            if timeframes is None:
                timeframes = ['1m', '5m', '15m', '1h']
            
            # Create detection tasks for each symbol
            detection_tasks = []
            for symbol, candles_data in symbols_data.items():
                task = asyncio.create_task(
                    self._detect_symbol_patterns(symbol, candles_data, timeframes)
                )
                detection_tasks.append(task)
            
            # Wait for all detections to complete
            results = await asyncio.gather(*detection_tasks, return_exceptions=True)
            
            # Process results
            bulk_results = {}
            for i, result in enumerate(results):
                symbol = list(symbols_data.keys())[i]
                if isinstance(result, Exception):
                    logger.error(f"Error detecting patterns for {symbol}: {result}")
                    bulk_results[symbol] = []
                else:
                    bulk_results[symbol] = result
            
            logger.info(f"🎼 Bulk pattern detection completed for {len(bulk_results)} symbols")
            return bulk_results
            
        except Exception as e:
            logger.error(f"Error in bulk pattern detection: {e}")
            return {}
    
    async def _detect_symbol_patterns(self, symbol: str, candles_data: Dict[str, List[Dict]],
                                    timeframes: List[str]) -> List[MultiTimeframeResult]:
        """Detect patterns for a single symbol with concurrency control"""
        async with self.symbol_semaphore:
            try:
                return await self.async_detector.detect_patterns_multi_timeframe(
                    symbol, timeframes, candles_data
                )
            except Exception as e:
                logger.error(f"Error detecting patterns for {symbol}: {e}")
                return []

