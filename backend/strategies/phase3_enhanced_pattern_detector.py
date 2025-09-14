#!/usr/bin/env python3
"""
Phase 3: Enhanced Pattern Detector with Quality & Filtering
Integrates noise filtering, post-detection validation, and signal quality scoring
to provide comprehensive pattern detection with quality filtering.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime, timezone
import logging
import asyncio
import time

# Import Phase 3 components
from .noise_filtering_layer import NoiseFilteringLayer, NoiseFilterConfig, NoiseFilterResult
from .post_detection_validator import PostDetectionValidator, ValidationConfig, ValidationResult
from .signal_quality_scorer import SignalQualityScorer, QualityScoringConfig, SignalQualityResult, SignalQuality

# Import Phase 2 components for integration
from .phase2_enhanced_pattern_detector import Phase2EnhancedPatternDetector

logger = logging.getLogger(__name__)

@dataclass
class Phase3PatternResult:
    """Comprehensive result from Phase 3 enhanced pattern detection with Phase 4A enhancements"""
    pattern_id: str
    symbol: str
    timeframe: str
    timestamp: datetime
    pattern_type: str
    direction: str
    original_confidence: float
    final_confidence: float
    quality_level: SignalQuality
    quality_score: float
    priority_rank: int
    passed_filters: bool
    validation_passed: bool
    noise_score: float
    follow_through_score: float
    volume_confirmation_score: float
    momentum_score: float
    risk_score: float
    reward_potential: float
    signal_strength: float
    filter_reasons: List[str]
    validation_reasons: List[str]
    quality_reasons: List[str]
    processing_time_ms: float
    
    # Phase 4A: Enhanced fields
    calibrated_confidence: Optional[float] = None
    calibration_confidence_interval: Optional[Dict[str, float]] = None
    multi_timeframe_alignment: Optional[float] = None
    timeframe_confirmation_count: Optional[int] = None
    market_regime: Optional[str] = None
    regime_adjusted_confidence: Optional[float] = None
    timeframe_hierarchy: Optional[Dict[str, Any]] = None
    explanation_factors: Optional[Dict[str, Any]] = None

@dataclass
class Phase3Config:
    """Configuration for Phase 3 enhanced pattern detection with Phase 4A enhancements"""
    # Component enable/disable flags
    enable_noise_filtering: bool = True
    enable_post_detection_validation: bool = True
    enable_signal_quality_scoring: bool = True
    
    # Phase 4A: Enhanced component flags
    enable_confidence_calibration: bool = True
    enable_multi_timeframe_analysis: bool = True
    enable_market_regime_detection: bool = True
    
    # Quality thresholds
    min_quality_level: SignalQuality = SignalQuality.FAIR
    min_final_confidence: float = 0.6
    min_calibrated_confidence: float = 0.5
    min_multi_timeframe_alignment: float = 0.4
    
    # Performance settings
    max_concurrent_symbols: int = 5
    enable_caching: bool = True
    cache_ttl_seconds: int = 300
    
    # Component configurations
    noise_filter_config: Optional[NoiseFilterConfig] = None
    validation_config: Optional[ValidationConfig] = None
    quality_scoring_config: Optional[QualityScoringConfig] = None

class Phase3EnhancedPatternDetector:
    """
    Phase 3 Enhanced Pattern Detector with Quality & Filtering
    Integrates all Phase 3 components for comprehensive pattern detection.
    """
    
    def __init__(self, config: Optional[Phase3Config] = None):
        self.config = config or Phase3Config()
        
        # Initialize Phase 2 detector (base functionality)
        self.phase2_detector = Phase2EnhancedPatternDetector()
        
        # Initialize Phase 3 components
        if self.config.enable_noise_filtering:
            self.noise_filter = NoiseFilteringLayer(self.config.noise_filter_config)
        else:
            self.noise_filter = None
        
        if self.config.enable_post_detection_validation:
            self.validator = PostDetectionValidator(self.config.validation_config)
        else:
            self.validator = None
        
        if self.config.enable_signal_quality_scoring:
            self.quality_scorer = SignalQualityScorer(self.config.quality_scoring_config)
        else:
            self.quality_scorer = None
        
        # Performance tracking
        self.stats = {
            'total_patterns_detected': 0,
            'patterns_after_noise_filtering': 0,
            'patterns_after_validation': 0,
            'patterns_after_quality_scoring': 0,
            'total_processing_time_ms': 0.0,
            'avg_processing_time_ms': 0.0,
            'quality_distribution': {},
            'filter_rejection_rate': 0.0,
            'validation_rejection_rate': 0.0
        }
        
        logger.info("Phase 3 Enhanced Pattern Detector initialized with config: %s", 
                   {k: v for k, v in self.config.__dict__.items() 
                    if not k.startswith('_') and not callable(v)})
    
    async def detect_enhanced_patterns(self, symbol: str, timeframe: str, 
                                     ohlcv_data: Dict[str, np.ndarray]) -> List[Phase3PatternResult]:
        """
        Detect patterns with Phase 3 quality filtering and validation
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe for analysis
            ohlcv_data: OHLCV data arrays
            
        Returns:
            List of enhanced pattern results with quality filtering
        """
        start_time = time.time()
        
        try:
            # Step 1: Phase 2 pattern detection (base functionality)
            phase2_results = await self.phase2_detector.detect_enhanced_patterns(
                symbol, timeframe, ohlcv_data
            )
            
            if not phase2_results:
                return []
            
            # Phase 4A: Multi-timeframe analysis (if enabled)
            multi_timeframe_patterns = {}
            if self.config.enable_multi_timeframe_analysis:
                multi_timeframe_patterns = await self._analyze_multi_timeframe_context(symbol, timeframe, ohlcv_data)
            
            # Phase 4A: Market regime detection (if enabled)
            market_regime = 'neutral'
            if self.config.enable_market_regime_detection:
                market_regime = self._detect_market_regime(ohlcv_data)
            
            # Convert Phase 2 results to pattern dictionaries
            patterns = []
            for result in phase2_results:
                pattern = {
                    'pattern_id': f"{symbol}_{timeframe}_{result.pattern_type}_{result.timestamp.strftime('%Y%m%d_%H%M%S')}",
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'timestamp': result.timestamp,
                    'pattern_type': result.pattern_type,
                    'direction': result.direction,
                    'confidence': result.final_confidence,
                    'index': result.pattern_index
                }
                patterns.append(pattern)
            
            # Step 2: Noise filtering
            noise_filter_results = []
            if self.noise_filter and self.config.enable_noise_filtering:
                noise_filter_results = self.noise_filter.filter_patterns(patterns, ohlcv_data)
                
                # Filter out patterns that didn't pass noise filtering
                passed_noise_filter = [r for r in noise_filter_results if r.passed_filters]
                patterns = [p for p in patterns if any(r.pattern_id == p['pattern_id'] and r.passed_filters 
                                                     for r in noise_filter_results)]
                
                logger.info(f"Noise filtering: {len(patterns)}/{len(phase2_results)} patterns passed")
            else:
                # Create dummy noise filter results
                for pattern in patterns:
                    noise_filter_results.append(NoiseFilterResult(
                        pattern_id=pattern['pattern_id'],
                        symbol=pattern['symbol'],
                        timeframe=pattern['timeframe'],
                        timestamp=pattern['timestamp'],
                        original_confidence=pattern['confidence'],
                        filtered_confidence=pattern['confidence'],
                        passed_filters=True,
                        filter_reasons=[],
                        atr_percentage=0.0,
                        volume_ratio=1.0,
                        price_movement=0.0,
                        noise_score=1.0
                    ))
            
            # Step 3: Post-detection validation
            validation_results = []
            if self.validator and self.config.enable_post_detection_validation:
                validation_results = self.validator.validate_patterns(patterns, ohlcv_data)
                
                # Filter out patterns that didn't pass validation
                passed_validation = [r for r in validation_results if r.validation_passed]
                patterns = [p for p in patterns if any(r.pattern_id == p['pattern_id'] and r.validation_passed 
                                                     for r in validation_results)]
                
                logger.info(f"Post-detection validation: {len(patterns)}/{len(noise_filter_results)} patterns passed")
            else:
                # Create dummy validation results
                for pattern in patterns:
                    validation_results.append(ValidationResult(
                        pattern_id=pattern['pattern_id'],
                        symbol=pattern['symbol'],
                        timeframe=pattern['timeframe'],
                        timestamp=pattern['timestamp'],
                        original_confidence=pattern['confidence'],
                        validated_confidence=pattern['confidence'],
                        validation_passed=True,
                        validation_reasons=[],
                        follow_through_score=0.5,
                        volume_confirmation_score=0.5,
                        momentum_score=0.5,
                        overall_validation_score=0.5,
                        validation_details={}
                    ))
            
            # Step 4: Signal quality scoring
            quality_results = []
            if self.quality_scorer and self.config.enable_signal_quality_scoring:
                # Convert noise filter and validation results to dictionaries
                noise_dicts = [self._noise_result_to_dict(r) for r in noise_filter_results]
                validation_dicts = [self._validation_result_to_dict(r) for r in validation_results]
                
                quality_results = self.quality_scorer.score_signals(
                    patterns, noise_dicts, validation_dicts, ohlcv_data
                )
                
                # Filter by minimum quality level
                quality_results = self.quality_scorer.filter_by_quality(
                    quality_results, self.config.min_quality_level
                )
                
                logger.info(f"Quality scoring: {len(quality_results)}/{len(validation_results)} patterns meet quality threshold")
            else:
                # Create dummy quality results
                for pattern in patterns:
                    quality_results.append(SignalQualityResult(
                        pattern_id=pattern['pattern_id'],
                        symbol=pattern['symbol'],
                        timeframe=pattern['timeframe'],
                        timestamp=pattern['timestamp'],
                        original_confidence=pattern['confidence'],
                        quality_score=0.7,
                        quality_level=SignalQuality.GOOD,
                        quality_factors={},
                        quality_reasons=[],
                        risk_score=1.0,
                        reward_potential=2.0,
                        signal_strength=0.7,
                        priority_rank=1
                    ))
            
            # Step 5: Combine results into final Phase 3 results
            final_results = self._combine_results(
                patterns, noise_filter_results, validation_results, quality_results
            )
            
            # Step 6: Apply final confidence filter
            final_results = [r for r in final_results if r.final_confidence >= self.config.min_final_confidence]
            
            # Update statistics
            processing_time_ms = (time.time() - start_time) * 1000
            self._update_stats(len(phase2_results), len(patterns), len(validation_results), 
                             len(quality_results), processing_time_ms)
            
            logger.info(f"Phase 3 detection complete: {len(final_results)}/{len(phase2_results)} patterns "
                       f"passed all filters in {processing_time_ms:.2f}ms")
            
            return final_results
            
        except Exception as e:
            logger.error(f"Error in Phase 3 pattern detection: {e}")
            return []
    
    async def detect_patterns_bulk(self, symbols_data: Dict[str, Dict[str, np.ndarray]], 
                                 timeframes: List[str]) -> Dict[str, List[Phase3PatternResult]]:
        """
        Detect patterns for multiple symbols and timeframes with Phase 3 filtering
        
        Args:
            symbols_data: Dictionary of symbol -> OHLCV data
            timeframes: List of timeframes to analyze
            
        Returns:
            Dictionary of symbol -> list of pattern results
        """
        all_results = {}
        
        # Process symbols concurrently
        semaphore = asyncio.Semaphore(self.config.max_concurrent_symbols)
        
        async def process_symbol(symbol: str, data: Dict[str, np.ndarray]):
            async with semaphore:
                symbol_results = {}
                for timeframe in timeframes:
                    if timeframe in data:
                        results = await self.detect_enhanced_patterns(symbol, timeframe, data[timeframe])
                        symbol_results[timeframe] = results
                return symbol, symbol_results
        
        # Create tasks for all symbols
        tasks = [
            process_symbol(symbol, data) 
            for symbol, data in symbols_data.items()
        ]
        
        # Execute all tasks
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Combine results
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Error processing symbol: {result}")
                continue
            
            symbol, symbol_results = result
            all_results[symbol] = symbol_results
        
        return all_results
    
    def _noise_result_to_dict(self, result: NoiseFilterResult) -> Dict:
        """Convert NoiseFilterResult to dictionary"""
        return {
            'pattern_id': result.pattern_id,
            'noise_score': result.noise_score,
            'atr_percentage': result.atr_percentage,
            'volume_ratio': result.volume_ratio,
            'price_movement': result.price_movement,
            'passed_filters': result.passed_filters,
            'filter_reasons': result.filter_reasons
        }
    
    def _validation_result_to_dict(self, result: ValidationResult) -> Dict:
        """Convert ValidationResult to dictionary"""
        return {
            'pattern_id': result.pattern_id,
            'follow_through_score': result.follow_through_score,
            'volume_confirmation_score': result.volume_confirmation_score,
            'momentum_score': result.momentum_score,
            'overall_validation_score': result.overall_validation_score,
            'validation_passed': result.validation_passed,
            'validation_reasons': result.validation_reasons
        }
    
    def _combine_results(self, patterns: List[Dict], 
                        noise_results: List[NoiseFilterResult],
                        validation_results: List[ValidationResult],
                        quality_results: List[SignalQualityResult]) -> List[Phase3PatternResult]:
        """Combine all results into final Phase 3 results"""
        
        # Create lookup dictionaries
        noise_lookup = {r.pattern_id: r for r in noise_results}
        validation_lookup = {r.pattern_id: r for r in validation_results}
        quality_lookup = {r.pattern_id: r for r in quality_results}
        
        final_results = []
        
        for pattern in patterns:
            pattern_id = pattern['pattern_id']
            
            noise_result = noise_lookup.get(pattern_id)
            validation_result = validation_lookup.get(pattern_id)
            quality_result = quality_lookup.get(pattern_id)
            
            if not all([noise_result, validation_result, quality_result]):
                continue
            
            # Calculate final confidence (average of all confidence scores)
            final_confidence = (
                noise_result.filtered_confidence +
                validation_result.validated_confidence +
                quality_result.original_confidence
            ) / 3
            
            result = Phase3PatternResult(
                pattern_id=pattern_id,
                symbol=pattern['symbol'],
                timeframe=pattern['timeframe'],
                timestamp=pattern['timestamp'],
                pattern_type=pattern['pattern_type'],
                direction=pattern['direction'],
                original_confidence=pattern['confidence'],
                final_confidence=final_confidence,
                quality_level=quality_result.quality_level,
                quality_score=quality_result.quality_score,
                priority_rank=quality_result.priority_rank,
                passed_filters=noise_result.passed_filters,
                validation_passed=validation_result.validation_passed,
                noise_score=noise_result.noise_score,
                follow_through_score=validation_result.follow_through_score,
                volume_confirmation_score=validation_result.volume_confirmation_score,
                momentum_score=validation_result.momentum_score,
                risk_score=quality_result.risk_score,
                reward_potential=quality_result.reward_potential,
                signal_strength=quality_result.signal_strength,
                filter_reasons=noise_result.filter_reasons,
                validation_reasons=validation_result.validation_reasons,
                quality_reasons=quality_result.quality_reasons,
                processing_time_ms=0.0  # Will be set by caller
            )
            
            final_results.append(result)
        
        return final_results
    
    def _update_stats(self, total_detected: int, after_noise_filter: int, 
                     after_validation: int, after_quality: int, processing_time_ms: float):
        """Update performance statistics"""
        self.stats['total_patterns_detected'] += total_detected
        self.stats['patterns_after_noise_filtering'] += after_noise_filter
        self.stats['patterns_after_validation'] += after_validation
        self.stats['patterns_after_quality_scoring'] += after_quality
        self.stats['total_processing_time_ms'] += processing_time_ms
        
        # Calculate averages
        total_runs = self.stats['total_patterns_detected'] // max(1, total_detected)
        self.stats['avg_processing_time_ms'] = self.stats['total_processing_time_ms'] / max(1, total_runs)
        
        # Calculate rejection rates
        if total_detected > 0:
            self.stats['filter_rejection_rate'] = (total_detected - after_noise_filter) / total_detected
            self.stats['validation_rejection_rate'] = (after_noise_filter - after_validation) / max(1, after_noise_filter)
    
    def get_detection_stats(self) -> Dict:
        """Get comprehensive detection statistics"""
        stats = self.stats.copy()
        
        # Add component-specific stats
        if self.noise_filter:
            noise_stats = self.noise_filter.get_filtering_stats([])
            stats['noise_filtering_stats'] = noise_stats
        
        if self.validator:
            validation_stats = self.validator.get_validation_stats([])
            stats['validation_stats'] = validation_stats
        
        if self.quality_scorer:
            quality_stats = self.quality_scorer.get_quality_stats([])
            stats['quality_scoring_stats'] = quality_stats
        
        return stats
    
    def get_performance_summary(self) -> Dict:
        """Get performance summary"""
        return {
            'total_patterns_processed': self.stats['total_patterns_detected'],
            'final_patterns_delivered': self.stats['patterns_after_quality_scoring'],
            'filter_rejection_rate': f"{self.stats['filter_rejection_rate']:.1%}",
            'validation_rejection_rate': f"{self.stats['validation_rejection_rate']:.1%}",
            'avg_processing_time_ms': f"{self.stats['avg_processing_time_ms']:.2f}",
            'components_enabled': {
                'noise_filtering': self.config.enable_noise_filtering,
                'post_detection_validation': self.config.enable_post_detection_validation,
                'signal_quality_scoring': self.config.enable_signal_quality_scoring,
                'confidence_calibration': self.config.enable_confidence_calibration,
                'multi_timeframe_analysis': self.config.enable_multi_timeframe_analysis,
                'market_regime_detection': self.config.enable_market_regime_detection
            }
        }
    
    # Phase 4A: Enhanced Methods
    
    async def _analyze_multi_timeframe_context(self, symbol: str, timeframe: str, 
                                             ohlcv_data: Dict[str, np.ndarray]) -> Dict[str, List[Dict]]:
        """Analyze multi-timeframe context for pattern confirmation"""
        try:
            # For now, return empty dict - this would be enhanced with actual multi-timeframe data
            # In a full implementation, this would fetch data from other timeframes
            return {}
        except Exception as e:
            logger.error(f"Error in multi-timeframe context analysis: {e}")
            return {}
    
    def _detect_market_regime(self, ohlcv_data: Dict[str, np.ndarray]) -> str:
        """Detect current market regime based on price action"""
        try:
            closes = ohlcv_data.get('close', [])
            if len(closes) < 50:
                return 'neutral'
            
            # Calculate trend indicators
            sma_20 = np.mean(closes[-20:])
            sma_50 = np.mean(closes[-50:])
            current_price = closes[-1]
            
            # Calculate volatility
            returns = np.diff(closes[-20:]) / closes[-21:-1]
            volatility = np.std(returns) * np.sqrt(252)  # Annualized
            
            # Calculate momentum
            momentum_5 = (closes[-1] - closes[-5]) / closes[-5]
            momentum_20 = (closes[-1] - closes[-20]) / closes[-20]
            
            # Determine regime
            if volatility > 0.8:  # High volatility
                if momentum_5 < -0.1 and momentum_20 < -0.2:
                    return 'crash'
                elif momentum_5 > 0.1 and momentum_20 > 0.2:
                    return 'bull'
                else:
                    return 'sideways'
            else:  # Normal volatility
                if current_price > sma_20 > sma_50 and momentum_20 > 0.05:
                    return 'bull'
                elif current_price < sma_20 < sma_50 and momentum_20 < -0.05:
                    return 'bear'
                else:
                    return 'sideways'
                    
        except Exception as e:
            logger.error(f"Error in market regime detection: {e}")
            return 'neutral'
    
    def _apply_phase4a_enhancements(self, result: Phase3PatternResult, 
                                  multi_timeframe_patterns: Dict[str, List[Dict]],
                                  market_regime: str) -> Phase3PatternResult:
        """Apply Phase 4A enhancements to a pattern result"""
        try:
            # Multi-timeframe alignment analysis
            if self.config.enable_multi_timeframe_analysis and multi_timeframe_patterns:
                alignment_score, confirmation_count, alignment_details = self.quality_scorer.analyze_multi_timeframe_alignment(
                    {
                        'pattern_type': result.pattern_type,
                        'timeframe': result.timeframe,
                        'direction': result.direction,
                        'timestamp': result.timestamp
                    },
                    multi_timeframe_patterns
                )
                
                result.multi_timeframe_alignment = alignment_score
                result.timeframe_confirmation_count = confirmation_count
                result.timeframe_hierarchy = alignment_details
            
            # Market regime adjustment
            if self.config.enable_market_regime_detection:
                result.market_regime = market_regime
                result.regime_adjusted_confidence = self.quality_scorer.apply_regime_adjustment(
                    result.final_confidence, market_regime
                )
            
            # Confidence calibration (placeholder - would need historical data)
            if self.config.enable_confidence_calibration:
                # In a real implementation, this would use historical calibration data
                result.calibrated_confidence = result.final_confidence
                result.calibration_confidence_interval = {
                    'lower': result.final_confidence * 0.9,
                    'upper': result.final_confidence * 1.1,
                    'confidence_level': 0.95
                }
            
            # Explanation factors
            result.explanation_factors = {
                'pattern_confidence': result.original_confidence,
                'noise_filter_score': 1.0 - result.noise_score,
                'validation_score': result.follow_through_score,
                'quality_score': result.quality_score,
                'multi_timeframe_alignment': result.multi_timeframe_alignment,
                'market_regime': result.market_regime,
                'regime_adjustment': result.regime_adjusted_confidence / result.final_confidence if result.final_confidence > 0 else 1.0
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error applying Phase 4A enhancements: {e}")
            return result
