#!/usr/bin/env python3
"""
Phase 2 Enhanced Pattern Detector
Integrated system combining hybrid ML, multi-symbol correlation, and dynamic thresholds
"""

import numpy as np
import pandas as pd
import logging
import asyncio
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
import talib
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Import Phase 2 components
from .hybrid_ml_pattern_detector import HybridMLPatternDetector, MLPatternResult
from .multi_symbol_correlation_detector import MultiSymbolCorrelationDetector, MultiSymbolPatternResult, CorrelationResult
from .dynamic_confidence_threshold_detector import DynamicConfidenceThresholdDetector, DynamicThresholdResult, VolatilityMetrics

logger = logging.getLogger(__name__)

@dataclass
class Phase2PatternResult:
    """Final result from Phase 2 enhanced pattern detection"""
    pattern_name: str
    symbol: str
    timeframe: str
    confidence: float
    direction: str
    timestamp: datetime
    price_level: float
    volume_confirmation: bool
    volume_confidence: float
    trend_alignment: str
    detection_method: str
    
    # Phase 2 enhancements
    ml_confidence: float
    talib_confidence: float
    correlation_boost: float
    dynamic_threshold_adjustment: float
    market_regime: str
    btc_confirmation: bool
    alt_confirmation: bool
    is_fuzzy_pattern: bool
    volatility_metrics: Dict[str, Any]
    metadata: Dict[str, Any]

class Phase2EnhancedPatternDetector:
    """Phase 2 Enhanced Pattern Detector - Integrated system"""
    
    def __init__(self, model_dir: str = "models/ml_patterns", btc_symbol: str = "BTC/USDT"):
        # Initialize Phase 2 components
        self.hybrid_ml_detector = HybridMLPatternDetector(model_dir)
        self.correlation_detector = MultiSymbolCorrelationDetector(btc_symbol)
        self.dynamic_threshold_detector = DynamicConfidenceThresholdDetector()
        
        # Performance tracking
        self.detection_stats = {
            'total_detections': 0,
            'ml_detections': 0,
            'talib_detections': 0,
            'fuzzy_detections': 0,
            'correlation_boosts': 0,
            'threshold_adjustments': 0,
            'avg_confidence': 0.0,
            'avg_processing_time_ms': 0.0
        }
        
        # Configuration
        self.min_confidence_threshold = 0.5
        self.enable_ml_detection = True
        self.enable_correlation_analysis = True
        self.enable_dynamic_thresholds = True
        
        logger.info("🚀 Phase 2 Enhanced Pattern Detector initialized")
        logger.info(f"   - Hybrid ML Detection: {self.enable_ml_detection}")
        logger.info(f"   - Correlation Analysis: {self.enable_correlation_analysis}")
        logger.info(f"   - Dynamic Thresholds: {self.enable_dynamic_thresholds}")
    
    async def detect_enhanced_patterns(self, symbol: str, timeframe: str, 
                                     candles: List[Dict], 
                                     all_symbols_data: Dict[str, Dict] = None,
                                     market_data: Dict = None) -> List[Phase2PatternResult]:
        """Detect patterns using Phase 2 enhanced system"""
        start_time = datetime.now()
        
        try:
            if len(candles) < 20:
                logger.warning(f"Insufficient candles for {symbol}: {len(candles)}")
                return []
            
            # Extract OHLCV arrays
            opens = np.array([c['open'] for c in candles])
            highs = np.array([c['high'] for c in candles])
            lows = np.array([c['low'] for c in candles])
            closes = np.array([c['close'] for c in candles])
            volumes = np.array([c.get('volume', 1000) for c in candles])
            
            results = []
            
            # Step 1: Hybrid ML Detection
            ml_results = []
            if self.enable_ml_detection:
                ml_results = self.hybrid_ml_detector.detect_patterns(
                    opens, highs, lows, closes, volumes
                )
                self.detection_stats['ml_detections'] += len(ml_results)
            
            # Step 2: Multi-Symbol Correlation Analysis
            correlation_results = []
            if self.enable_correlation_analysis and all_symbols_data and market_data:
                # Convert ML results to format expected by correlation detector
                ml_patterns = [
                    {
                        'pattern_name': r.pattern_name,
                        'confidence': r.confidence,
                        'direction': 'bullish' if r.probability > 0.5 else 'bearish',
                        'price_level': r.price_level
                    }
                    for r in ml_results
                ]
                
                correlation_results = await self.correlation_detector.detect_multi_symbol_patterns(
                    symbol, ml_patterns, all_symbols_data, market_data
                )
                self.detection_stats['correlation_boosts'] += len(correlation_results)
            
            # Step 3: Dynamic Threshold Adjustment
            final_results = []
            for i, ml_result in enumerate(ml_results):
                # Get correlation result if available
                correlation_result = correlation_results[i] if i < len(correlation_results) else None
                
                # Calculate base confidence
                base_confidence = ml_result.confidence
                
                # Apply correlation boost
                correlation_boost = 0.0
                btc_confirmation = False
                alt_confirmation = False
                market_regime = "neutral"
                
                if correlation_result:
                    correlation_boost = correlation_result.correlation_boost
                    btc_confirmation = correlation_result.btc_confirmation
                    alt_confirmation = correlation_result.alt_confirmation
                    market_regime = correlation_result.market_regime
                    base_confidence += correlation_boost
                
                # Apply dynamic threshold adjustment
                dynamic_adjustment = 0.0
                volatility_metrics = {}
                
                if self.enable_dynamic_thresholds:
                    threshold_result = self.dynamic_threshold_detector.get_dynamic_threshold(
                        ml_result.pattern_name, symbol, timeframe, highs, lows, closes, volumes
                    )
                    
                    dynamic_adjustment = threshold_result.confidence_boost
                    base_confidence += dynamic_adjustment
                    
                    # Get volatility metrics
                    vol_metrics = self.dynamic_threshold_detector.get_volatility_metrics(
                        highs, lows, closes, volumes
                    )
                    volatility_metrics = {
                        'atr': vol_metrics.atr,
                        'bollinger_width': vol_metrics.bollinger_width,
                        'price_volatility': vol_metrics.price_volatility,
                        'volume_volatility': vol_metrics.volume_volatility,
                        'market_regime': vol_metrics.market_regime,
                        'volatility_score': vol_metrics.volatility_score
                    }
                    
                    self.detection_stats['threshold_adjustments'] += 1
                
                # Ensure confidence is within bounds
                final_confidence = max(min(base_confidence, 1.0), 0.0)
                
                # Only include if confidence meets minimum threshold
                if final_confidence >= self.min_confidence_threshold:
                    # Determine direction
                    direction = 'bullish' if ml_result.probability > 0.5 else 'bearish'
                    
                    # Create final result
                    result = Phase2PatternResult(
                        pattern_name=ml_result.pattern_name,
                        symbol=symbol,
                        timeframe=timeframe,
                        confidence=final_confidence,
                        direction=direction,
                        timestamp=datetime.now(timezone.utc),
                        price_level=ml_result.price_level or closes[-1],
                        volume_confirmation=ml_result.features.get('volume_ratio', 0) > 1.2,
                        volume_confidence=ml_result.features.get('volume_ratio', 1.0),
                        trend_alignment=self._determine_trend_alignment(closes),
                        detection_method='hybrid_ml' if ml_result.is_fuzzy else 'talib_ml',
                        ml_confidence=ml_result.probability,
                        talib_confidence=ml_result.talib_confidence or 0.0,
                        correlation_boost=correlation_boost,
                        dynamic_threshold_adjustment=dynamic_adjustment,
                        market_regime=market_regime,
                        btc_confirmation=btc_confirmation,
                        alt_confirmation=alt_confirmation,
                        is_fuzzy_pattern=ml_result.is_fuzzy,
                        volatility_metrics=volatility_metrics,
                        metadata={
                            'original_ml_confidence': ml_result.confidence,
                            'features': ml_result.features,
                            'correlation_score': correlation_result.correlation_score if correlation_result else 0.0,
                            'btc_dominance': correlation_result.btc_dominance if correlation_result else 50.0,
                            'market_sentiment': correlation_result.market_sentiment if correlation_result else 'neutral'
                        }
                    )
                    
                    final_results.append(result)
            
            # Update statistics
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            self.detection_stats['total_detections'] += len(final_results)
            self.detection_stats['avg_processing_time_ms'] = (
                (self.detection_stats['avg_processing_time_ms'] * 
                 (self.detection_stats['total_detections'] - len(final_results)) + 
                 processing_time * len(final_results)) / self.detection_stats['total_detections']
            )
            
            if final_results:
                avg_confidence = np.mean([r.confidence for r in final_results])
                self.detection_stats['avg_confidence'] = (
                    (self.detection_stats['avg_confidence'] * 
                     (self.detection_stats['total_detections'] - len(final_results)) + 
                     avg_confidence * len(final_results)) / self.detection_stats['total_detections']
                )
            
            logger.info(f"⚡ Phase 2 detection completed for {symbol}: {len(final_results)} patterns in {processing_time:.2f}ms")
            
            return final_results
            
        except Exception as e:
            logger.error(f"Error in Phase 2 pattern detection for {symbol}: {e}")
            return []
    
    def _determine_trend_alignment(self, closes: np.ndarray) -> str:
        """Determine trend alignment based on recent price action"""
        if len(closes) < 10:
            return "neutral"
        
        # Calculate short-term trend (last 5 candles)
        short_trend = closes[-1] - closes[-5]
        
        # Calculate medium-term trend (last 10 candles)
        medium_trend = closes[-1] - closes[-10]
        
        # Determine alignment
        if short_trend > 0 and medium_trend > 0:
            return "bullish"
        elif short_trend < 0 and medium_trend < 0:
            return "bearish"
        elif short_trend > 0 and medium_trend < 0:
            return "bullish_reversal"
        elif short_trend < 0 and medium_trend > 0:
            return "bearish_reversal"
        else:
            return "neutral"
    
    async def detect_patterns_bulk(self, symbols_data: Dict[str, Dict[str, List[Dict]]],
                                 timeframes: List[str] = None,
                                 market_data: Dict = None) -> Dict[str, List[Phase2PatternResult]]:
        """Detect patterns for multiple symbols and timeframes"""
        if timeframes is None:
            timeframes = ['1m', '5m', '15m', '1h']
        
        results = {}
        
        for symbol, timeframe_data in symbols_data.items():
            symbol_results = []
            
            for timeframe in timeframes:
                if timeframe in timeframe_data:
                    candles = timeframe_data[timeframe]
                    
                    # Get all symbols data for correlation analysis
                    all_symbols_data = symbols_data if self.enable_correlation_analysis else None
                    
                    timeframe_results = await self.detect_enhanced_patterns(
                        symbol, timeframe, candles, all_symbols_data, market_data
                    )
                    
                    symbol_results.extend(timeframe_results)
            
            if symbol_results:
                results[symbol] = symbol_results
        
        return results
    
    def train_ml_models(self, training_data: Dict[str, List[Dict]], 
                       labels: Dict[str, List[int]]) -> Dict[str, float]:
        """Train ML models for all patterns"""
        accuracies = {}
        
        for pattern_name in self.hybrid_ml_detector.pattern_configs.keys():
            if pattern_name in training_data and pattern_name in labels:
                try:
                    accuracy = self.hybrid_ml_detector.train_model(
                        pattern_name, training_data[pattern_name], labels[pattern_name]
                    )
                    accuracies[pattern_name] = accuracy
                    logger.info(f"✅ Trained {pattern_name} model with accuracy: {accuracy:.3f}")
                except Exception as e:
                    logger.error(f"❌ Failed to train {pattern_name} model: {e}")
                    accuracies[pattern_name] = 0.0
        
        return accuracies
    
    def get_detection_stats(self) -> Dict[str, Any]:
        """Get comprehensive detection statistics"""
        return {
            'phase2_stats': self.detection_stats,
            'ml_stats': self.hybrid_ml_detector.get_model_performance('doji'),  # Example
            'correlation_stats': self.correlation_detector.get_correlation_stats(),
            'threshold_stats': self.dynamic_threshold_detector.get_detector_stats()
        }
    
    def save_models(self):
        """Save all ML models"""
        self.hybrid_ml_detector.save_models()
        logger.info("💾 Saved all Phase 2 ML models")
    
    def load_models(self):
        """Load all ML models"""
        self.hybrid_ml_detector.load_models()
        logger.info("📂 Loaded all Phase 2 ML models")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for Phase 2 system"""
        stats = self.detection_stats
        
        return {
            'total_detections': stats['total_detections'],
            'ml_detections': stats['ml_detections'],
            'talib_detections': stats['talib_detections'],
            'fuzzy_detections': stats['fuzzy_detections'],
            'correlation_boosts': stats['correlation_boosts'],
            'threshold_adjustments': stats['threshold_adjustments'],
            'avg_confidence': stats['avg_confidence'],
            'avg_processing_time_ms': stats['avg_processing_time_ms'],
            'ml_detection_rate': stats['ml_detections'] / max(stats['total_detections'], 1),
            'correlation_boost_rate': stats['correlation_boosts'] / max(stats['total_detections'], 1),
            'threshold_adjustment_rate': stats['threshold_adjustments'] / max(stats['total_detections'], 1)
        }
