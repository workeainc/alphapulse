#!/usr/bin/env python3
"""
Pattern Integration Service for AlphaPulse Trading Bot
Integrates pattern detection algorithms with pattern storage and provides
unified pattern management capabilities.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass
import json
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import psutil

from app.services.pattern_storage_service import PatternStorageService, PatternData
from ..strategies.pattern_detector import CandlestickPatternDetector
from ..strategies.advanced_pattern_detector import AdvancedPatternDetector
from ..strategies.ml_pattern_detector import MLPatternDetector
from data.pattern_analyzer import PatternAnalyzer
from data.volume_analyzer import VolumeAnalyzer, VolumePattern, VolumePatternType, VolumeStrength
from data.fetcher import MarketDataService
from ..strategies.real_time_signal_generator import RealTimeSignalGenerator

logger = logging.getLogger(__name__)

@dataclass
class PatternDetectionResult:
    """Result of pattern detection operation"""
    pattern: PatternData
    detection_method: str
    processing_time_ms: float
    additional_metadata: Dict[str, Any]

class PatternIntegrationService:
    """Service that integrates pattern detection with storage and management"""
    
    def __init__(self):
        self.pattern_storage = PatternStorageService()
        self.pattern_detector = CandlestickPatternDetector()
        self.advanced_detector = AdvancedPatternDetector()
        self.ml_detector = MLPatternDetector()
        self.pattern_analyzer = PatternAnalyzer()
        self.volume_analyzer = VolumeAnalyzer()
        self.market_data_service = MarketDataService()
        self.real_time_generator = RealTimeSignalGenerator()
        
        self._initialized = False
        self.logger = logging.getLogger(__name__)
        
        # Pattern detection callbacks
        self.detection_callbacks: List[Callable[[PatternData], None]] = []
        
        # Performance tracking
        self.detection_stats = {
            "total_patterns_detected": 0,
            "patterns_stored": 0,
            "storage_errors": 0,
            "avg_processing_time_ms": 0.0,
            "batch_processing_stats": {
                "total_batches": 0,
                "parallel_batches": 0,
                "sequential_batches": 0,
                "avg_batch_size": 0.0,
                "total_processing_time": 0.0,
                "peak_patterns_per_second": 0.0
            },
            "storage_performance": {
                "copy_operations": 0,
                "parallel_copy_operations": 0,
                "regular_batches": 0,
                "avg_storage_time_ms": 0.0,
                "storage_throughput": 0.0
            },
            "real_time_metrics": {
                "current_patterns_per_second": 0.0,
                "current_batch_size": 0,
                "active_workers": 0,
                "memory_usage_mb": 0.0,
                "last_optimization_check": None
            }
        }
        
        # Volume confirmation filtering configuration
        self.volume_filtering_config = {
            "enable_filtering": True,
            "filtering_mode": "both",            # "filter", "rank", or "both"
            "min_volume_confirmation": 0.3,
            "confidence_boost": 0.2,
            "confidence_penalty": -0.1,
            "filter_weak_patterns": True,
            "enable_ranking": True,
            "keep_unconfirmed": False,           # Whether to keep patterns without volume confirmation
            "volume_pattern_weights": {
                "spike": 1.2,
                "divergence": 0.8,
                "climax": 1.5,
                "dry_up": 0.9,
                "accumulation": 1.1,
                "distribution": 0.7
            },
            "lazy_volume_analysis": True,        # Only run volume analysis when needed
            "min_confidence_for_volume": 0.5,    # Only analyze volume for patterns above this confidence
            "volume_analysis_batch_size": 10,    # Process volume analysis in batches
            "skip_volume_for_low_priority": True, # Skip volume analysis for low-priority patterns
            "parallel_processing": True,         # Enable parallel pattern detection
            "max_workers": 4,                    # Maximum concurrent workers
            "detection_timeout": 30,             # Timeout for detection methods (seconds)
            "enable_shared_preprocessing": True  # Enable shared data preprocessing
        }
        
        # Production optimization configuration
        self.production_config = {
            "enable_optimized_storage": True,     # Use optimized batch storage methods
            "auto_parallel_processing": True,    # Automatically enable parallel processing for large batches
            "parallel_threshold": 100,           # Minimum patterns to trigger parallel processing
            "max_parallel_workers": 8,           # Maximum parallel workers
            "batch_size_optimization": True,     # Enable automatic batch size optimization
            "memory_monitoring": True,           # Monitor memory usage during processing
            "performance_tracking": True,        # Track detailed performance metrics
            "auto_optimization": True,           # Automatically optimize based on performance
            "optimization_check_interval": 300,  # Check for optimization every 5 minutes
            "peak_performance_tracking": True,   # Track peak performance metrics
            "real_time_monitoring": True,        # Enable real-time performance monitoring
            "alert_thresholds": {
                "low_performance": 100,          # Alert if patterns/second drops below this
                "high_memory": 80,               # Alert if memory usage exceeds 80%
                "storage_errors": 5              # Alert if storage errors exceed this count
            }
        }
    
    async def initialize(self):
        """Initialize the pattern integration service"""
        if self._initialized:
            return
            
        try:
            # Initialize storage service
            await self.pattern_storage.initialize()
            
            # Initialize pattern detectors
            await self._initialize_detectors()
            
            self._initialized = True
            self.logger.info("✅ Pattern Integration Service initialized successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize Pattern Integration Service: {e}")
            raise
    
    async def _initialize_detectors(self):
        """Initialize all pattern detection components"""
        try:
            # Initialize basic pattern detector
            if hasattr(self.pattern_detector, 'initialize'):
                await self.pattern_detector.initialize()
            
            # Initialize advanced pattern detector
            if hasattr(self.advanced_detector, 'initialize'):
                await self.advanced_detector.initialize()
            
            # Initialize ML pattern detector
            if hasattr(self.ml_detector, 'initialize'):
                await self.ml_detector.initialize()
            
            # Initialize pattern analyzer
            if hasattr(self.pattern_analyzer, 'initialize'):
                await self.pattern_analyzer.initialize()
                
            self.logger.info("✅ All pattern detectors initialized")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Some pattern detectors failed to initialize: {e}")
    
    def add_detection_callback(self, callback: Callable[[PatternData], None]):
        """Add a callback function to be called when patterns are detected"""
        self.detection_callbacks.append(callback)
        self.logger.info(f"✅ Added pattern detection callback (total: {len(self.detection_callbacks)})")
    
    def remove_detection_callback(self, callback: Callable[[PatternData], None]):
        """Remove a pattern detection callback"""
        if callback in self.detection_callbacks:
            self.detection_callbacks.remove(callback)
            self.logger.info(f"✅ Removed pattern detection callback (total: {len(self.detection_callbacks)})")
    
    async def _notify_detection_callbacks(self, pattern: PatternData):
        """Notify all registered callbacks about a detected pattern"""
        if not self.detection_callbacks:
            return
            
        for callback in self.detection_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(pattern)
                else:
                    callback(pattern)
            except Exception as e:
                self.logger.error(f"❌ Error in detection callback: {e}")
    
    async def detect_and_store_patterns(
        self,
        candlestick_data: List[Dict],
        symbol: str,
        timeframe: str,
        detection_methods: List[str] = None
    ) -> List[PatternDetectionResult]:
        """Detect patterns in candlestick data and store them"""
        if not self._initialized:
            await self.initialize()
        
        if detection_methods is None:
            detection_methods = ["basic", "advanced", "ml"]
        
        results = []
        start_time = datetime.utcnow()
        
        try:
            # Convert candlestick data to the format expected by detectors
            processed_data = self._prepare_candlestick_data(candlestick_data)
            
            # PARALLEL PROCESSING: Execute different detection methods concurrently
            # Apply shared preprocessing to avoid redundant calculations
            df = await self._preprocess_data_shared(
                self._prepare_dataframe_for_volume_analysis(candlestick_data), 
                symbol, 
                timeframe
            )
            
            # Use parallel detection for better performance
            detection_results = await self._detect_patterns_parallel(df, symbol, timeframe)
            
            # Combine all detected patterns with metadata
            all_detected_patterns = []
            for method_name, patterns in detection_results.items():
                if patterns:
                    for pattern in patterns:
                        pattern.metadata = pattern.metadata or {}
                        pattern.metadata['detection_method'] = method_name
                        pattern.metadata['detection_time_ms'] = 0  # Will be updated if sequential fallback is used
                    all_detected_patterns.extend(patterns)
                    self.logger.info(f"✅ {method_name.title()} detection: {len(patterns)} patterns found")
            
            self.logger.info(f"🎯 Total patterns detected: {len(all_detected_patterns)}")
            
            # LAZY VOLUME ANALYSIS: Only analyze volume for high-confidence patterns
            volume_patterns = []
            
            # LAZY VOLUME ANALYSIS: Only analyze volume for high-confidence patterns
            if all_detected_patterns and self.volume_filtering_config["lazy_volume_analysis"]:
                volume_patterns = await self._perform_lazy_volume_analysis(
                    all_detected_patterns, df, symbol, timeframe
                )
                self.logger.info(f"✅ Lazy volume analysis completed: {len(volume_patterns)} volume patterns detected")
            elif all_detected_patterns:
                # Fallback to traditional volume analysis if lazy mode is disabled
                volume_patterns = await self.volume_analyzer.analyze_volume_patterns(df, symbol, timeframe)
                self.logger.info(f"✅ Traditional volume analysis completed: {len(volume_patterns)} volume patterns detected")
            
            # STEP 3: Apply volume confirmation filtering and ranking
            if all_detected_patterns:
                self.logger.info(f"🔍 Starting volume confirmation filtering for {len(all_detected_patterns)} total patterns")
                filtered_and_ranked_patterns = self._apply_volume_confirmation_filtering(all_detected_patterns, volume_patterns)
                self.logger.info(f"✅ Volume confirmation filtering completed: {len(filtered_and_ranked_patterns)} patterns ready for storage")
            else:
                filtered_and_ranked_patterns = []
                self.logger.warning("⚠️ No patterns detected from any method")
            
            # Store filtered and ranked patterns
            for pattern in filtered_and_ranked_patterns:
                try:
                    # Store pattern
                    success = await self.pattern_storage.store_pattern(pattern)
                    
                    if success:
                        # Create result
                        result = PatternDetectionResult(
                            pattern=pattern,
                            detection_method=pattern.metadata.get('detection_method', 'unknown'),
                            processing_time_ms=pattern.metadata.get('detection_time_ms', 0.0),
                            additional_metadata={
                                "candlestick_count": len(candlestick_data),
                                "detection_timestamp": datetime.utcnow().isoformat(),
                                "volume_confirmation": pattern.volume_confirmation,
                                "volume_confidence": pattern.volume_confidence,
                                "final_confidence": pattern.confidence
                            }
                        )
                        results.append(result)
                        
                        # Update stats
                        self.detection_stats["total_patterns_detected"] += 1
                        self.detection_stats["patterns_stored"] += 1
                        
                        # Notify callbacks
                        await self._notify_detection_callbacks(pattern)
                        
                        self.logger.info(f"✅ Pattern stored: {pattern.pattern_name} (confidence: {pattern.confidence:.2f}, volume: {pattern.volume_confidence:.2f})")
                    else:
                        self.detection_stats["storage_errors"] += 1
                        self.logger.error(f"❌ Failed to store pattern: {pattern.pattern_name}")
                        
                except Exception as e:
                    self.detection_stats["storage_errors"] += 1
                    self.logger.error(f"❌ Error processing pattern {pattern.pattern_name}: {e}")
            
            # Update average processing time
            total_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            if results:
                self.detection_stats["avg_processing_time_ms"] = (
                    (self.detection_stats["avg_processing_time_ms"] * (len(results) - 1) + total_time) / len(results)
                )
            
            self.logger.info(f"✅ Pattern detection and volume filtering completed: {len(results)} patterns stored")
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Error in pattern detection and storage: {e}")
            return []

    async def detect_patterns_batch(
        self,
        symbols: List[str],
        timeframe: str,
        detection_methods: List[str] = None,
        limit: int = 200
    ) -> Dict[str, List[PatternDetectionResult]]:
        """
        CRITICAL EFFICIENCY OPTIMIZATION: Detect patterns across multiple symbols efficiently.
        This method reduces API calls from N to 1 per timeframe by using batch OHLCV fetch.
        
        Args:
            symbols: List of trading symbols to analyze
            timeframe: Timeframe for analysis
            detection_methods: List of detection methods to use
            limit: Number of candles to fetch per symbol
            
        Returns:
            Dictionary mapping symbols to their pattern detection results
        """
        if not self._initialized:
            await self.initialize()
        
        if detection_methods is None:
            detection_methods = ["basic", "advanced", "ml"]
        
        self.logger.info(f"🚀 Starting BATCH pattern detection for {len(symbols)} symbols: {timeframe}")
        batch_start_time = datetime.utcnow()
        
        try:
            # STEP 1: Batch OHLCV fetch - CRITICAL EFFICIENCY GAIN
            market_data_service = MarketDataService()
            
            # OPTIMIZATION: Warm cache for common timeframes to improve subsequent requests
            if timeframe in ['1m', '5m', '15m', '1h', '4h', '1d']:
                self.logger.info(f"🔥 Warming cache for {len(symbols)} symbols across common timeframes")
                await market_data_service.warm_cache_for_symbols(symbols)
            
            # Fetch all symbols' data in a single batch operation
            batch_ohlcv_data = await market_data_service.get_batch_historical_data(
                symbols, timeframe, limit
            )
            
            self.logger.info(f"✅ Batch OHLCV fetch completed: {len(batch_ohlcv_data)} symbols")
            
            # STEP 2: Process each symbol with its data
            all_results = {}
            
            for symbol, ohlcv_data in batch_ohlcv_data.items():
                if ohlcv_data is None:
                    self.logger.warning(f"⚠️ No data available for {symbol}, skipping")
                    all_results[symbol] = []
                    continue
                
                try:
                    # Convert DataFrame to list of dicts for compatibility
                    candlestick_data = ohlcv_data.to_dict('records')
                    
                    # Detect patterns for this symbol
                    symbol_results = await self.detect_and_store_patterns(
                        candlestick_data, symbol, timeframe, detection_methods
                    )
                    
                    all_results[symbol] = symbol_results
                    self.logger.info(f"✅ {symbol}: {len(symbol_results)} patterns detected")
                    
                except Exception as e:
                    self.logger.error(f"❌ Error processing {symbol}: {e}")
                    all_results[symbol] = []
            
            # Log batch performance metrics
            total_batch_time = (datetime.utcnow() - batch_start_time).total_seconds()
            total_patterns = sum(len(results) for results in all_results.values())
            
            self.logger.info(f"🎯 BATCH PATTERN DETECTION COMPLETED:")
            self.logger.info(f"   • Symbols processed: {len(symbols)}")
            self.logger.info(f"   • Total patterns detected: {total_patterns}")
            self.logger.info(f"   • Total time: {total_batch_time:.2f}s")
            self.logger.info(f"   • Efficiency gain: ~{len(symbols)}x faster than individual processing")
            
            return all_results
            
        except Exception as e:
            self.logger.error(f"❌ Error in batch pattern detection: {e}")
            return {symbol: [] for symbol in symbols}
    
    def _prepare_candlestick_data(self, candlestick_data: List[Dict]) -> List[Dict]:
        """Prepare candlestick data for pattern detection"""
        # This method should convert your candlestick data format to what the detectors expect
        # You may need to adjust this based on your actual data structure
        
        prepared_data = []
        for candle in candlestick_data:
            prepared_candle = {
                "timestamp": candle.get("timestamp", candle.get("time", datetime.utcnow())),
                "open": float(candle.get("open", candle.get("open_price", 0.0))),
                "high": float(candle.get("high", candle.get("high_price", 0.0))),
                "low": float(candle.get("low", candle.get("low_price", 0.0))),
                "close": float(candle.get("close", candle.get("close_price", 0.0))),
                "volume": float(candle.get("volume", 0.0))
            }
            prepared_data.append(prepared_candle)
        
        return prepared_data
    
    async def _detect_basic_patterns(
        self, 
        candlestick_data: List[Dict], 
        symbol: str, 
        timeframe: str
    ) -> List[PatternData]:
        """Detect basic patterns using the basic pattern detector"""
        try:
            if not hasattr(self.pattern_detector, 'detect_patterns'):
                self.logger.warning("⚠️ Basic pattern detector not available")
                return []
            
            # Detect patterns
            detected_patterns = await self.pattern_detector.detect_patterns(candlestick_data)
            
            # Convert to PatternData objects with advanced volume confirmation
            patterns = []
            for pattern_info in detected_patterns:
                # Check if volume confirms this pattern using advanced VolumePattern analysis
                volume_confirmed, volume_confidence, volume_pattern_type, volume_strength, volume_context = self._check_volume_confirmation(
                    pattern_info, candlestick_data, symbol, timeframe
                )
                
                pattern = PatternData(
                    symbol=symbol,
                    timeframe=timeframe,
                    pattern_name=pattern_info.get("pattern_name", "unknown"),
                    timestamp=pattern_info.get("timestamp", datetime.utcnow()),
                    confidence=pattern_info.get("confidence", 0.5),
                    strength=pattern_info.get("strength", "medium"),
                    price_level=pattern_info.get("price_level", 0.0),
                    volume_confirmation=volume_confirmed,
                    volume_confidence=volume_confidence,
                    volume_pattern_type=volume_pattern_type,
                    volume_strength=volume_strength,
                    volume_context=volume_context,
                    trend_alignment=pattern_info.get("trend_alignment", "neutral"),
                    metadata=pattern_info.get("metadata", {})
                )
                patterns.append(pattern)
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"❌ Error in basic pattern detection: {e}")
            return []
    
    async def _detect_advanced_patterns(
        self, 
        candlestick_data: List[Dict], 
        symbol: str, 
        timeframe: str
    ) -> List[PatternData]:
        """Detect advanced patterns using the advanced pattern detector"""
        try:
            if not hasattr(self.advanced_detector, 'detect_patterns'):
                self.logger.warning("⚠️ Advanced pattern detector not available")
                return []
            
            # Detect patterns
            detected_patterns = await self.advanced_detector.detect_patterns(candlestick_data)
            
            # Use advanced volume confirmation for advanced patterns
            patterns = []
            for pattern_info in detected_patterns:
                # Check if volume confirms this pattern using advanced VolumePattern analysis
                volume_confirmed, volume_confidence, volume_pattern_type, volume_strength, volume_context = self._check_volume_confirmation(
                    pattern_info, candlestick_data, symbol, timeframe
                )
                
                pattern = PatternData(
                    symbol=symbol,
                    timeframe=timeframe,
                    pattern_name=pattern_info.get("pattern_name", "unknown"),
                    timestamp=pattern_info.get("timestamp", datetime.utcnow()),
                    confidence=pattern_info.get("confidence", 0.5),
                    strength=pattern_info.get("strength", "medium"),
                    price_level=pattern_info.get("price_level", 0.0),
                    volume_confirmation=volume_confirmed,
                    volume_confidence=volume_confidence,
                    volume_pattern_type=volume_pattern_type,
                    volume_strength=volume_strength,
                    volume_context=volume_context,
                    trend_alignment=pattern_info.get("trend_alignment", "neutral"),
                    metadata=pattern_info.get("metadata", {})
                )
                patterns.append(pattern)
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"❌ Error in advanced pattern detection: {e}")
            return []
    
    async def _detect_ml_patterns(
        self, 
        candlestick_data: List[Dict], 
        symbol: str, 
        timeframe: str
    ) -> List[PatternData]:
        """Detect ML-based patterns using the ML pattern detector"""
        try:
            if not hasattr(self.ml_detector, 'detect_patterns'):
                self.logger.warning("⚠️ ML pattern detector not available")
                return []
            
            # Detect patterns
            detected_patterns = await self.ml_detector.detect_patterns(candlestick_data)
            
            # Use advanced volume confirmation for ML patterns
            patterns = []
            for pattern_info in detected_patterns:
                # Check if volume confirms this pattern using advanced VolumePattern analysis
                volume_confirmed, volume_confidence, volume_pattern_type, volume_strength, volume_context = self._check_volume_confirmation(
                    pattern_info, candlestick_data, symbol, timeframe
                )
                
                pattern = PatternData(
                    symbol=symbol,
                    timeframe=timeframe,
                    pattern_name=pattern_info.get("pattern_name", "unknown"),
                    timestamp=pattern_info.get("timestamp", datetime.utcnow()),
                    confidence=pattern_info.get("confidence", 0.5),
                    strength=pattern_info.get("strength", "medium"),
                    price_level=pattern_info.get("price_level", 0.0),
                    volume_confirmation=volume_confirmed,
                    volume_confidence=volume_confidence,
                    volume_pattern_type=volume_pattern_type,
                    volume_strength=volume_strength,
                    volume_context=volume_context,
                    trend_alignment=pattern_info.get("trend_alignment", "neutral"),
                    metadata=pattern_info.get("metadata", {})
                )
                patterns.append(pattern)
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"❌ Error in ML pattern detection: {e}")
            return []
    
    async def get_patterns_with_analysis(
        self,
        symbol: Optional[str] = None,
        timeframe: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        min_confidence: Optional[float] = None,
        pattern_name: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get patterns with additional analysis and insights"""
        if not self._initialized:
            await self.initialize()
        
        try:
            # Get patterns from storage
            patterns = await self.pattern_storage.get_patterns(
                symbol=symbol,
                timeframe=timeframe,
                start_time=start_time,
                end_time=end_time,
                min_confidence=min_confidence,
                pattern_name=pattern_name
            )
            
            # Add analysis for each pattern
            enriched_patterns = []
            for pattern in patterns:
                try:
                    # Get pattern analysis
                    analysis = await self._analyze_pattern(pattern)
                    
                    # Create enriched pattern data
                    enriched_pattern = {
                        "pattern": pattern,
                        "analysis": analysis,
                        "insights": self._generate_pattern_insights(pattern, analysis),
                        "recommendations": self._generate_recommendations(pattern, analysis)
                    }
                    enriched_patterns.append(enriched_pattern)
                    
                except Exception as e:
                    self.logger.error(f"❌ Error analyzing pattern {pattern.pattern_name}: {e}")
                    # Add pattern without analysis
                    enriched_patterns.append({
                        "pattern": pattern,
                        "analysis": {},
                        "insights": [],
                        "recommendations": []
                    })
            
            return enriched_patterns
            
        except Exception as e:
            self.logger.error(f"❌ Error getting patterns with analysis: {e}")
            return []
    
    async def _analyze_pattern(self, pattern: PatternData) -> Dict[str, Any]:
        """Analyze a single pattern for additional insights"""
        try:
            if not hasattr(self.pattern_analyzer, 'analyze_pattern'):
                return {}
            
            # Get historical context for analysis
            historical_patterns = await self.pattern_storage.get_patterns(
                symbol=pattern.symbol,
                pattern_name=pattern.pattern_name,
                start_time=pattern.timestamp - timedelta(days=30),
                end_time=pattern.timestamp,
                limit=100
            )
            
            # Analyze pattern
            analysis = await self.pattern_analyzer.analyze_pattern(
                pattern, 
                historical_patterns
            )
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"❌ Error analyzing pattern: {e}")
            return {}
    
    def _generate_pattern_insights(self, pattern: PatternData, analysis: Dict[str, Any]) -> List[str]:
        """Generate human-readable insights from pattern analysis"""
        insights = []
        
        # Confidence-based insights
        if pattern.confidence >= 0.9:
            insights.append("Very high confidence pattern - strong signal")
        elif pattern.confidence >= 0.8:
            insights.append("High confidence pattern - reliable signal")
        elif pattern.confidence >= 0.6:
            insights.append("Medium confidence pattern - moderate signal")
        else:
            insights.append("Low confidence pattern - weak signal")
        
        # Volume confirmation insights
        if pattern.volume_confirmation:
            insights.append("Volume confirms pattern validity")
        else:
            insights.append("Volume does not confirm pattern - exercise caution")
        
        # Trend alignment insights
        if pattern.trend_alignment == "bullish":
            insights.append("Pattern aligns with bullish trend")
        elif pattern.trend_alignment == "bearish":
            insights.append("Pattern aligns with bearish trend")
        else:
            insights.append("Pattern shows neutral trend alignment")
        
        # Add analysis-based insights
        if analysis:
            if "success_rate" in analysis:
                success_rate = analysis["success_rate"]
                insights.append(f"Historical success rate: {success_rate:.1%}")
            
            if "avg_profit" in analysis:
                avg_profit = analysis["avg_profit"]
                insights.append(f"Average profit potential: {avg_profit:.2f}%")
        
        return insights
    
    def _generate_recommendations(self, pattern: PatternData, analysis: Dict[str, Any]) -> List[str]:
        """Generate trading recommendations based on pattern analysis"""
        recommendations = []
        
        # Basic recommendations based on pattern type
        if "hammer" in pattern.pattern_name.lower():
            recommendations.append("Consider long position with stop loss below low")
        elif "shooting_star" in pattern.pattern_name.lower():
            recommendations.append("Consider short position with stop loss above high")
        elif "engulfing" in pattern.pattern_name.lower():
            if pattern.trend_alignment == "bullish":
                recommendations.append("Strong bullish reversal signal - consider long entry")
            else:
                recommendations.append("Strong bearish reversal signal - consider short entry")
        
        # Confidence-based recommendations
        if pattern.confidence >= 0.8:
            recommendations.append("High confidence allows for larger position sizing")
        elif pattern.confidence < 0.6:
            recommendations.append("Low confidence - use smaller position size or wait for confirmation")
        
        # Volume-based recommendations
        if not pattern.volume_confirmation:
            recommendations.append("Wait for volume confirmation before entering position")
        
        # Analysis-based recommendations
        if analysis:
            if "success_rate" in analysis and analysis["success_rate"] < 0.5:
                recommendations.append("Low historical success rate - consider alternative strategies")
            
            if "avg_profit" in analysis and analysis["avg_profit"] < 0:
                recommendations.append("Historical pattern shows losses - avoid this setup")
        
        return recommendations
    
    async def get_pattern_performance_metrics(
        self,
        symbol: Optional[str] = None,
        timeframe: Optional[str] = None,
        days: int = 30
    ) -> Dict[str, Any]:
        """Get performance metrics for patterns"""
        if not self._initialized:
            await self.initialize()
        
        try:
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(days=days)
            
            # Get patterns in time range
            patterns = await self.pattern_storage.get_patterns(
                symbol=symbol,
                timeframe=timeframe,
                start_time=start_time,
                end_time=end_time
            )
            
            if not patterns:
                return {"message": "No patterns found in specified time range"}
            
            # Calculate metrics
            total_patterns = len(patterns)
            high_confidence_patterns = len([p for p in patterns if p.confidence >= 0.8])
            volume_confirmed_patterns = len([p for p in patterns if p.volume_confirmation])
            
            # Pattern type distribution
            pattern_types = {}
            for pattern in patterns:
                pattern_types[pattern.pattern_name] = pattern_types.get(pattern.pattern_name, 0) + 1
            
            # Trend alignment distribution
            trend_distribution = {}
            for pattern in patterns:
                trend_distribution[pattern.trend_alignment] = trend_distribution.get(pattern.trend_alignment, 0) + 1
            
            # Confidence statistics
            confidences = [p.confidence for p in patterns]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            metrics = {
                "time_period": f"Last {days} days",
                "total_patterns": total_patterns,
                "high_confidence_patterns": high_confidence_patterns,
                "high_confidence_rate": high_confidence_patterns / total_patterns if total_patterns > 0 else 0,
                "volume_confirmed_patterns": volume_confirmed_patterns,
                "volume_confirmation_rate": volume_confirmed_patterns / total_patterns if total_patterns > 0 else 0,
                "average_confidence": avg_confidence,
                "pattern_type_distribution": pattern_types,
                "trend_alignment_distribution": trend_distribution,
                "detection_stats": self.detection_stats.copy()
            }
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"❌ Error getting pattern performance metrics: {e}")
            return {"error": str(e)}
    
    async def cleanup_old_patterns(self, days_to_keep: int = 30) -> int:
        """Clean up old patterns to maintain storage efficiency"""
        if not self._initialized:
            await self.initialize()
        
        try:
            deleted_count = await self.pattern_storage.cleanup_old_patterns(days_to_keep)
            self.logger.info(f"✅ Cleaned up {deleted_count} old patterns")
            return deleted_count
            
        except Exception as e:
            self.logger.error(f"❌ Error cleaning up old patterns: {e}")
            return 0
    
    async def get_service_health(self) -> Dict[str, Any]:
        """Get comprehensive health status of the pattern integration service"""
        if not self._initialized:
            await self.initialize()
        
        try:
            # Get storage health
            storage_health = await self.pattern_storage.get_storage_health()
            
            # Get detection stats
            detection_health = {
                "total_patterns_detected": self.detection_stats["total_patterns_detected"],
                "patterns_stored": self.detection_stats["patterns_stored"],
                "storage_errors": self.detection_stats["storage_errors"],
                "avg_processing_time_ms": self.detection_stats["avg_processing_time_ms"],
                "callback_count": len(self.detection_callbacks)
            }
            
            # Overall health
            overall_health = {
                "service_initialized": self._initialized,
                "storage_health": storage_health,
                "detection_health": detection_health,
                "detectors_available": {
                    "basic": hasattr(self.pattern_detector, 'detect_patterns'),
                    "advanced": hasattr(self.advanced_detector, 'detect_patterns'),
                    "ml": hasattr(self.ml_detector, 'detect_patterns'),
                    "analyzer": hasattr(self.pattern_analyzer, 'analyze_pattern')
                }
            }
            
            return overall_health
            
        except Exception as e:
            self.logger.error(f"❌ Error getting service health: {e}")
            return {"error": str(e)}
    
    def _prepare_dataframe_for_volume_analysis(self, candlestick_data: List[Dict]) -> pd.DataFrame:
        """Prepare candlestick data as DataFrame for volume analysis"""
        try:
            df = pd.DataFrame(candlestick_data)
            required_columns = ['open', 'high', 'low', 'close', 'volume', 'timestamp']
            
            # Ensure all required columns exist
            for col in required_columns:
                if col not in df.columns:
                    self.logger.warning(f"⚠️ Missing column '{col}' in candlestick data")
                    return pd.DataFrame()
            
            # Convert timestamp to datetime if it's not already
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Sort by timestamp
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            return df
            
        except Exception as e:
            self.logger.error(f"❌ Error preparing DataFrame for volume analysis: {e}")
            return pd.DataFrame()
    
    async def _perform_lazy_volume_analysis(
        self, 
        patterns: List[PatternData], 
        df: pd.DataFrame, 
        symbol: str, 
        timeframe: str
    ) -> List[VolumePattern]:
        """
        Perform volume analysis only for patterns that meet confidence criteria
        This implements lazy evaluation to avoid unnecessary volume analysis
        """
        if not patterns:
            return []
        
        # Filter patterns by confidence threshold
        min_confidence = self.volume_filtering_config.get("min_confidence_for_volume", 0.5)
        high_confidence_patterns = [p for p in patterns if p.confidence >= min_confidence]
        
        if not high_confidence_patterns:
            self.logger.info(f"⏭️ Skipping volume analysis: no patterns meet confidence threshold {min_confidence}")
            return []
        
        # Calculate efficiency gain
        skipped_count = len(patterns) - len(high_confidence_patterns)
        efficiency_gain = (skipped_count / len(patterns)) * 100 if patterns else 0
        
        self.logger.info(f"🎯 Lazy volume analysis: {len(high_confidence_patterns)}/{len(patterns)} patterns qualify (skipping {skipped_count}, {efficiency_gain:.1f}% efficiency gain)")
        
        # Process in batches for better performance
        batch_size = self.volume_filtering_config.get("volume_analysis_batch_size", 10)
        volume_patterns = []
        
        for i in range(0, len(high_confidence_patterns), batch_size):
            batch = high_confidence_patterns[i:i + batch_size]
            batch_patterns = await self.volume_analyzer.analyze_volume_patterns(df, symbol, timeframe)
            volume_patterns.extend(batch_patterns)
            
            if len(high_confidence_patterns) > batch_size:
                self.logger.debug(f"📦 Processed batch {i//batch_size + 1}/{(len(high_confidence_patterns) + batch_size - 1)//batch_size}")
        
        return volume_patterns

    async def _detect_patterns_parallel(
        self, 
        df: pd.DataFrame, 
        symbol: str, 
        timeframe: str
    ) -> Dict[str, List[PatternData]]:
        """
        Execute different pattern detection methods in parallel for improved performance
        """
        if not self.volume_filtering_config.get("parallel_processing", True):
            # Fallback to sequential processing
            return await self._detect_patterns_sequential(df, symbol, timeframe)
        
        max_workers = self.volume_filtering_config.get("max_workers", 4)
        timeout = self.volume_filtering_config.get("detection_timeout", 30)
        
        self.logger.info(f"🚀 Starting parallel pattern detection with {max_workers} workers (timeout: {timeout}s)")
        start_time = time.time()
        
        # Define detection tasks
        detection_tasks = {
            "basic": lambda: self.pattern_analyzer.detect_patterns(df, symbol, timeframe),
            "advanced": lambda: self.advanced_detector.detect_patterns(df, symbol, timeframe),
            "ml": lambda: self.ml_detector.detect_patterns(df, symbol, timeframe),
            "real_time": lambda: self.real_time_generator.detect_patterns(df, symbol, timeframe)
        }
        
        results = {}
        completed_tasks = 0
        
        # Execute tasks in parallel using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_task = {
                executor.submit(task): task_name 
                for task_name, task in detection_tasks.items()
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_task, timeout=timeout):
                task_name = future_to_task[future]
                try:
                    task_result = future.result()
                    results[task_name] = task_result if task_result else []
                    completed_tasks += 1
                    self.logger.debug(f"✅ {task_name} detection completed: {len(results[task_name])} patterns")
                except Exception as e:
                    self.logger.error(f"❌ {task_name} detection failed: {str(e)}")
                    results[task_name] = []
                    completed_tasks += 1
        
        execution_time = time.time() - start_time
        total_patterns = sum(len(patterns) for patterns in results.values())
        
        self.logger.info(f"🎯 Parallel detection completed in {execution_time:.2f}s: {completed_tasks}/{len(detection_tasks)} methods, {total_patterns} total patterns")
        
        return results

    async def _detect_patterns_sequential(
        self, 
        df: pd.DataFrame, 
        symbol: str, 
        timeframe: str
    ) -> Dict[str, List[PatternData]]:
        """
        Fallback sequential pattern detection method
        """
        self.logger.info("🐌 Using sequential pattern detection (parallel processing disabled)")
        
        results = {}
        
        # Basic pattern detection
        try:
            results["basic"] = self.pattern_analyzer.detect_patterns(df, symbol, timeframe) or []
            self.logger.debug(f"✅ Basic detection: {len(results['basic'])} patterns")
        except Exception as e:
            self.logger.error(f"❌ Basic detection failed: {str(e)}")
            results["basic"] = []
        
        # Advanced pattern detection
        try:
            results["advanced"] = self.advanced_detector.detect_patterns(df, symbol, timeframe) or []
            self.logger.debug(f"✅ Advanced detection: {len(results['advanced'])} patterns")
        except Exception as e:
            self.logger.error(f"❌ Advanced detection failed: {str(e)}")
            results["advanced"] = []
        
        # ML pattern detection
        try:
            results["ml"] = self.ml_detector.detect_patterns(df, symbol, timeframe) or []
            self.logger.debug(f"✅ ML detection: {len(results['ml'])} patterns")
        except Exception as e:
            self.logger.error(f"❌ ML detection failed: {str(e)}")
            results["ml"] = []
        
        # Real-time signal detection
        try:
            results["real_time"] = self.real_time_generator.detect_patterns(df, symbol, timeframe) or []
            self.logger.debug(f"✅ Real-time detection: {len(results['real_time'])} patterns")
        except Exception as e:
            self.logger.error(f"❌ Real-time detection failed: {str(e)}")
            results["real_time"] = []
        
        return results

    async def _preprocess_data_shared(
        self, 
        df: pd.DataFrame, 
        symbol: str, 
        timeframe: str
    ) -> pd.DataFrame:
        """
        Perform shared data preprocessing to avoid redundant calculations across detection methods
        """
        if not self.volume_filtering_config.get("enable_shared_preprocessing", True):
            return df
        
        self.logger.debug("🔧 Performing shared data preprocessing")
        start_time = time.time()
        
        # Create a copy to avoid modifying original data
        processed_df = df.copy()
        
        # Calculate common technical indicators that multiple detection methods use
        try:
            # Moving averages (used by basic, advanced, and ML detectors)
            processed_df['sma_20'] = processed_df['close'].rolling(window=20).mean()
            processed_df['sma_50'] = processed_df['close'].rolling(window=50).mean()
            processed_df['ema_12'] = processed_df['close'].ewm(span=12).mean()
            processed_df['ema_26'] = processed_df['close'].ewm(span=26).mean()
            
            # RSI (used by multiple detectors)
            delta = processed_df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            processed_df['rsi'] = 100 - (100 / (1 + rs))
            
            # Bollinger Bands (used by advanced and ML detectors)
            processed_df['bb_middle'] = processed_df['close'].rolling(window=20).mean()
            bb_std = processed_df['close'].rolling(window=20).std()
            processed_df['bb_upper'] = processed_df['bb_middle'] + (bb_std * 2)
            processed_df['bb_lower'] = processed_df['bb_middle'] - (bb_std * 2)
            
            # MACD (used by advanced and real-time detectors)
            processed_df['macd'] = processed_df['ema_12'] - processed_df['ema_26']
            processed_df['macd_signal'] = processed_df['macd'].ewm(span=9).mean()
            processed_df['macd_histogram'] = processed_df['macd'] - processed_df['macd_signal']
            
            # Volume indicators (used by volume analysis and some detectors)
            processed_df['volume_sma'] = processed_df['volume'].rolling(window=20).mean()
            processed_df['volume_ratio'] = processed_df['volume'] / processed_df['volume_sma']
            
            preprocessing_time = time.time() - start_time
            self.logger.debug(f"✅ Shared preprocessing completed in {preprocessing_time:.3f}s")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Shared preprocessing failed: {str(e)}, continuing with original data")
            return df
        
        return processed_df

    async def get_parallel_processing_metrics(self) -> Dict[str, Any]:
        """
        Get parallel processing performance metrics and configuration
        """
        return {
            "parallel_processing_enabled": self.volume_filtering_config.get("parallel_processing", True),
            "max_workers": self.volume_filtering_config.get("max_workers", 4),
            "detection_timeout": self.volume_filtering_config.get("detection_timeout", 30),
            "shared_preprocessing_enabled": self.volume_filtering_config.get("enable_shared_preprocessing", True),
            "configuration": {
                "parallel_processing": self.volume_filtering_config.get("parallel_processing", True),
                "max_workers": self.volume_filtering_config.get("max_workers", 4),
                "detection_timeout": self.volume_filtering_config.get("detection_timeout", 30),
                "enable_shared_preprocessing": self.volume_filtering_config.get("enable_shared_preprocessing", True)
            },
            "recommendations": [
                "Increase max_workers for systems with more CPU cores",
                "Adjust detection_timeout based on your data size and complexity",
                "Enable shared_preprocessing to reduce redundant calculations",
                "Monitor execution times to optimize worker allocation"
            ]
        }

    async def update_parallel_processing_config(self, config_updates: Dict[str, Any]) -> bool:
        """
        Update parallel processing configuration parameters
        """
        try:
            for key, value in config_updates.items():
                if key in self.volume_filtering_config:
                    self.volume_filtering_config[key] = value
                    self.logger.info(f"✅ Updated {key}: {value}")
                else:
                    self.logger.warning(f"⚠️ Unknown config key: {key}")
            
            return True
        except Exception as e:
            self.logger.error(f"❌ Failed to update parallel processing config: {str(e)}")
            return False
    
    def _check_volume_confirmation(self, pattern_info: Dict, candlestick_data: List[Dict], symbol: str = "", timeframe: str = "") -> Tuple[bool, float, str, str, Dict]:
        """
        Advanced volume confirmation using VolumePattern.check_confirmation()
        
        Args:
            pattern_info: Detected pattern information
            candlestick_data: OHLCV data for the pattern's timeframe
            
        Returns:
            Tuple of (volume_confirmed: bool, volume_confidence: float, volume_pattern_type: str, volume_strength: str, volume_context: Dict)
        """
        try:
            if not candlestick_data or len(candlestick_data) < 10:
                self.logger.warning("⚠️ Insufficient candlestick data for volume confirmation")
                return False, 0.0
            
            # Get pattern details
            pattern_name = pattern_info.get("pattern_name", "unknown")
            pattern_timestamp = pattern_info.get("timestamp", datetime.utcnow())
            
            # Prepare data for volume analysis
            df = self._prepare_dataframe_for_volume_analysis(candlestick_data)
            if df.empty:
                self.logger.warning("⚠️ Failed to prepare DataFrame for volume analysis")
                return False, 0.0
            
            # Create VolumePattern instance with meaningful data for this pattern
            # Calculate volume ratio from recent data
            if len(normalized_candlestick_data) >= 2:
                current_volume = normalized_candlestick_data[-1].get('volume', 0)
                avg_volume = sum(c.get('volume', 0) for c in normalized_candlestick_data[:-1]) / (len(normalized_candlestick_data) - 1)
                volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
                
                # Calculate price change
                current_close = normalized_candlestick_data[-1].get('close', 0)
                prev_close = normalized_candlestick_data[-2].get('close', 0)
                price_change = ((current_close - prev_close) / prev_close * 100) if prev_close > 0 else 0.0
            else:
                volume_ratio = 1.0
                price_change = 0.0
            
            # Determine pattern type based on pattern name
            pattern_type = VolumePatternType.VOLUME_BREAKOUT  # Default
            if any(term in pattern_name.lower() for term in ['breakout', 'gap', 'engulfing']):
                pattern_type = VolumePatternType.VOLUME_BREAKOUT
            elif any(term in pattern_name.lower() for term in ['divergence', 'hidden']):
                pattern_type = VolumePatternType.VOLUME_DIVERGENCE
            elif any(term in pattern_name.lower() for term in ['climax', 'exhaustion']):
                pattern_type = VolumePatternType.VOLUME_CLIMAX
            elif any(term in pattern_name.lower() for term in ['dry', 'low_volume']):
                pattern_type = VolumePatternType.VOLUME_DRY_UP
            
            # Determine strength based on volume ratio
            if volume_ratio > 2.0:
                strength = VolumeStrength.EXTREME
            elif volume_ratio > 1.5:
                strength = VolumeStrength.STRONG
            elif volume_ratio > 1.2:
                strength = VolumeStrength.MEDIUM
            else:
                strength = VolumeStrength.WEAK
            
            volume_pattern = VolumePattern(
                pattern_type=pattern_type,
                symbol=symbol,
                timeframe=timeframe,
                timestamp=pattern_timestamp,
                strength=strength,
                confidence=min(1.0, volume_ratio / 2.0),  # Scale confidence with volume ratio
                volume_ratio=volume_ratio,
                price_change=price_change,
                pattern_data={
                    'pattern_name': pattern_name,
                    'volume_ratio': volume_ratio,
                    'price_change': price_change,
                    'candlestick_count': len(normalized_candlestick_data)
                },
                description=f"Volume confirmation for {pattern_name} (ratio: {volume_ratio:.2f}, change: {price_change:.2f}%)"
            )
            
            # Ensure candlestick data has the expected column names for VolumePattern
            normalized_candlestick_data = []
            for candle in candlestick_data:
                normalized_candle = {
                    'open': candle.get('open', candle.get('open_price', 0)),
                    'high': candle.get('high', candle.get('high_price', 0)),
                    'low': candle.get('low', candle.get('low_price', 0)),
                    'close': candle.get('close', candle.get('close_price', 0)),
                    'volume': candle.get('volume', 0),
                    'timestamp': candle.get('timestamp', datetime.utcnow())
                }
                normalized_candlestick_data.append(normalized_candle)
            
            # Get volume data aligned with price data
            volume_data = [{"volume": candle.get('volume', 0)} for candle in normalized_candlestick_data]
            
            # Use advanced volume confirmation with confidence scoring
            volume_confirmed = volume_pattern.check_confirmation(
                pattern_name, 
                normalized_candlestick_data, 
                volume_data
            )
            
            # Get confidence score for additional insight
            volume_confidence = volume_pattern.check_confirmation_with_confidence(
                pattern_name, 
                normalized_candlestick_data, 
                volume_data
            )
            
            self.logger.debug(f"Volume confirmation for {pattern_name}: confirmed={volume_confirmed}, confidence={volume_confidence:.2f}")
            
            return (
                volume_confirmed, 
                volume_confidence, 
                volume_pattern.pattern_type.value,
                volume_pattern.strength.value,
                volume_pattern.pattern_data
            )
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error in advanced volume confirmation: {e}")
            return False, 0.0

    async def close(self):
        """Close the pattern integration service"""
        try:
            if hasattr(self.pattern_storage, 'close'):
                await self.pattern_storage.close()
            self.logger.info("✅ Pattern Integration Service closed")
        except Exception as e:
            self.logger.error(f"❌ Error closing Pattern Integration Service: {e}")
    
    def _apply_volume_confirmation_filtering(self, patterns: List[PatternData], volume_patterns: List[VolumePattern]) -> List[PatternData]:
        """
        Step 3: Apply volume confirmation filtering and ranking to patterns
        
        This method implements the quality control layer that:
        1. Filters out weak patterns based on volume confirmation
        2. Boosts/demotes patterns in ranking based on volume confidence
        3. Prioritizes patterns for trading decisions
        
        Args:
            patterns: List of detected patterns
            volume_patterns: List of detected volume patterns
            
        Returns:
            Filtered and ranked list of patterns
        """
        try:
            if not patterns:
                return []
            
            self.logger.info(f"🔍 Applying volume confirmation filtering to {len(patterns)} patterns")
            
            # Apply filtering based on configuration mode
            if self.volume_filtering_config["filtering_mode"] in ["filter", "both"]:
                # Step 3A: Filter out weak patterns (Option A from your breakdown)
                filtered_patterns = self._filter_patterns_by_volume_confirmation(patterns, volume_patterns)
                self.logger.info(f"✅ Volume filtering completed: {len(filtered_patterns)} patterns passed filter")
            else:
                # Skip filtering, keep all patterns
                filtered_patterns = patterns
                self.logger.info("ℹ️ Volume filtering skipped (mode: rank only)")
            
            if self.volume_filtering_config["filtering_mode"] in ["rank", "both"]:
                # Step 3B: Boost/demote patterns based on volume confidence (Option B from your breakdown)
                ranked_patterns = self._rank_patterns_by_volume_confidence(filtered_patterns, volume_patterns)
                self.logger.info(f"✅ Volume ranking completed: {len(ranked_patterns)} patterns ranked")
            else:
                # Skip ranking, keep original order
                ranked_patterns = filtered_patterns
                self.logger.info("ℹ️ Volume ranking skipped (mode: filter only)")
            
            self.logger.info(f"✅ Volume confirmation filtering completed: {len(filtered_patterns)} patterns passed filter, {len(ranked_patterns)} patterns ranked")
            
            return ranked_patterns
            
        except Exception as e:
            self.logger.error(f"❌ Error in volume confirmation filtering: {e}")
            return patterns  # Return original patterns if filtering fails
    
    def _filter_patterns_by_volume_confirmation(self, patterns: List[PatternData], volume_patterns: List[VolumePattern]) -> List[PatternData]:
        """
        Filter patterns based on volume confirmation (Option A)
        """
        try:
            min_volume_confidence = self.volume_filtering_config.get("min_volume_confidence", 0.3)
            filtered_patterns = []
            
            for pattern in patterns:
                # Check if this pattern has volume confirmation
                volume_confirmed, pattern_type, strength = self._check_pattern_volume_confirmation(pattern, volume_patterns)
                
                if volume_confirmed:
                    # Set the volume pattern details
                    pattern.volume_confirmation = True
                    pattern.volume_pattern_type = pattern_type
                    pattern.volume_strength = strength
                    
                    # Get volume confidence score
                    pattern.volume_confidence = self._get_pattern_volume_confidence(pattern, volume_patterns)
                    
                    # Store multi-timeframe volume context
                    self._store_multi_timeframe_volume_context(pattern, volume_patterns)
                    
                    # Only keep patterns that meet the minimum volume confidence threshold
                    if pattern.volume_confidence >= min_volume_confidence:
                        filtered_patterns.append(pattern)
                        self.logger.debug(f"✅ Pattern '{pattern.pattern_name}' passed volume filtering (confidence: {pattern.volume_confidence:.2f})")
                    else:
                        self.logger.debug(f"❌ Pattern '{pattern.pattern_name}' failed threshold ({pattern.volume_confidence:.2f} < {min_volume_confidence})")
                else:
                    # No volume confirmation
                    pattern.volume_confirmation = False
                    pattern.volume_pattern_type = ""
                    pattern.volume_strength = ""
                    pattern.volume_confidence = 0.0
                    pattern.volume_context = {}
                    
                    # Optionally keep unconfirmed patterns with lower confidence
                    if self.volume_filtering_config.get("keep_unconfirmed", False):
                        filtered_patterns.append(pattern)
                        self.logger.debug(f"⚠️ Pattern '{pattern.pattern_name}' kept without volume confirmation")
                    else:
                        self.logger.debug(f"❌ Pattern '{pattern.pattern_name}' filtered out (no volume confirmation)")
            
            self.logger.info(f"✅ Volume filtering completed: {len(filtered_patterns)}/{len(patterns)} patterns passed")
            return filtered_patterns
            
        except Exception as e:
            self.logger.error(f"❌ Error in volume confirmation filtering: {e}")
            return patterns  # Return all patterns on error
    
    def _rank_patterns_by_volume_confidence(self, patterns: List[PatternData], volume_patterns: List[VolumePattern]) -> List[PatternData]:
        """
        Rank patterns by adjusting confidence based on volume confirmation (Option B)
        """
        try:
            strong_volume_threshold = self.volume_filtering_config.get("strong_volume_threshold", 0.7)
            moderate_volume_threshold = self.volume_filtering_config.get("moderate_volume_threshold", 0.5)
            weak_volume_threshold = self.volume_filtering_config.get("weak_volume_threshold", 0.3)
            
            strong_volume_boost = self.volume_filtering_config.get("strong_volume_boost", 0.2)
            moderate_volume_boost = self.volume_filtering_config.get("moderate_volume_boost", 0.1)
            weak_volume_reduction = self.volume_filtering_config.get("weak_volume_reduction", 0.2)
            
            ranked_patterns = []
            
            for pattern in patterns:
                # Check if this pattern has volume confirmation
                volume_confirmed, pattern_type, strength = self._check_pattern_volume_confirmation(pattern, volume_patterns)
                
                if volume_confirmed:
                    # Set the volume pattern details
                    pattern.volume_confirmation = True
                    pattern.volume_pattern_type = pattern_type
                    pattern.volume_strength = strength
                    
                    # Get volume confidence score
                    pattern.volume_confidence = self._get_pattern_volume_confidence(pattern, volume_patterns)
                    
                    # Store multi-timeframe volume context
                    self._store_multi_timeframe_volume_context(pattern, volume_patterns)
                    
                    # Apply pattern type weight
                    pattern_type_weight = self.volume_filtering_config.get("volume_pattern_weights", {}).get(
                        self._get_pattern_category(pattern.pattern_name), 1.0
                    )
                    
                    # Adjust confidence based on volume strength
                    if pattern.volume_confidence >= strong_volume_threshold:
                        confidence_adjustment = strong_volume_boost * pattern_type_weight
                        self.logger.debug(f"🚀 Pattern '{pattern.pattern_name}' gets strong volume boost: +{confidence_adjustment:.2f}")
                    elif pattern.volume_confidence >= moderate_volume_threshold:
                        confidence_adjustment = moderate_volume_boost * pattern_type_weight
                        self.logger.debug(f"📈 Pattern '{pattern.pattern_name}' gets moderate volume boost: +{confidence_adjustment:.2f}")
                    elif pattern.volume_confidence >= weak_volume_threshold:
                        confidence_adjustment = 0.0  # No change for weak volume
                        self.logger.debug(f"➡️ Pattern '{pattern.pattern_name}' gets no confidence change (weak volume)")
                    else:
                        confidence_adjustment = -weak_volume_reduction * pattern_type_weight
                        self.logger.debug(f"📉 Pattern '{pattern.pattern_name}' gets volume reduction: {confidence_adjustment:.2f}")
                    
                    # Apply confidence adjustment
                    pattern.confidence = min(1.0, max(0.0, pattern.confidence + confidence_adjustment))
                    
                else:
                    # No volume confirmation
                    pattern.volume_confirmation = False
                    pattern.volume_pattern_type = ""
                    pattern.volume_strength = ""
                    pattern.volume_confidence = 0.0
                    pattern.volume_context = {}
                    
                    # Apply penalty for no volume confirmation
                    penalty = -weak_volume_reduction * 0.5  # Smaller penalty
                    pattern.confidence = min(1.0, max(0.0, pattern.confidence + penalty))
                    self.logger.debug(f"⚠️ Pattern '{pattern.pattern_name}' gets no-confirmation penalty: {penalty:.2f}")
                
                ranked_patterns.append(pattern)
            
            # Sort by final confidence
            ranked_patterns.sort(key=lambda p: p.confidence, reverse=True)
            
            self.logger.info(f"✅ Volume ranking completed: {len(ranked_patterns)} patterns ranked by confidence")
            return ranked_patterns
            
        except Exception as e:
            self.logger.error(f"❌ Error in volume confidence ranking: {e}")
            return patterns  # Return original patterns on error
    
    def _get_pattern_category(self, pattern_name: str) -> str:
        """Get the category of a pattern for weight calculation"""
        pattern_name_lower = pattern_name.lower()
        
        if any(term in pattern_name_lower for term in ['breakout', 'gap', 'engulfing']):
            return "breakout"
        elif any(term in pattern_name_lower for term in ['reversal', 'divergence', 'hidden']):
            return "reversal"
        elif any(term in pattern_name_lower for term in ['triangle', 'wedge', 'channel', 'consolidation']):
            return "consolidation"
        else:
            return "default"
    
    def _store_multi_timeframe_volume_context(self, pattern: PatternData, volume_patterns: List[VolumePattern], 
                                            timeframes: List[str] = None) -> None:
        """
        Store multi-timeframe volume context for a pattern
        
        Args:
            pattern: The pattern to store volume context for
            volume_patterns: Available volume patterns
            timeframes: List of timeframes to analyze (default: current + common timeframes)
        """
        try:
            if timeframes is None:
                timeframes = [pattern.timeframe, "15m", "1h", "4h", "1d"]
            
            volume_context = {}
            
            for timeframe in timeframes:
                # Filter volume patterns for this timeframe
                timeframe_patterns = [vp for vp in volume_patterns if vp.timeframe == timeframe]
                
                if timeframe_patterns:
                    # Get the strongest volume pattern for this timeframe
                    strongest_pattern = max(timeframe_patterns, key=lambda vp: vp.confidence)
                    
                    volume_context[timeframe] = {
                        "pattern_type": strongest_pattern.pattern_type.value if strongest_pattern.pattern_type else "",
                        "strength": strongest_pattern.strength.value if strongest_pattern.strength else "",
                        "confidence": strongest_pattern.confidence,
                        "volume_ratio": strongest_pattern.volume_ratio,
                        "price_change": strongest_pattern.price_change,
                        "description": strongest_pattern.description
                    }
                else:
                    # No volume patterns for this timeframe
                    volume_context[timeframe] = {
                        "pattern_type": "",
                        "strength": "",
                        "confidence": 0.0,
                        "volume_ratio": 1.0,
                        "price_change": 0.0,
                        "description": "No volume pattern detected"
                    }
            
            # Store the multi-timeframe context
            pattern.volume_context = volume_context
            
            self.logger.debug(f"✅ Stored multi-timeframe volume context for {pattern.pattern_name}: {list(volume_context.keys())}")
            
        except Exception as e:
            self.logger.error(f"❌ Error storing multi-timeframe volume context: {e}")
            pattern.volume_context = {}
    
    def get_adaptive_risk_parameters(self, pattern: PatternData) -> Dict[str, Any]:
        """
        Get adaptive risk management parameters based on volume strength
        
        Args:
            pattern: The pattern to calculate risk parameters for
            
        Returns:
            Dictionary with risk management parameters
        """
        try:
            if not pattern.volume_confirmation:
                # No volume confirmation - use conservative parameters
                return {
                    "position_size_multiplier": 0.5,
                    "max_leverage": 1.0,
                    "stop_loss_tightness": "tight",
                    "take_profit_aggressiveness": "conservative",
                    "risk_level": "high",
                    "recommendation": "Avoid trading - no volume confirmation"
                }
            
            # Base risk parameters
            base_params = {
                "position_size_multiplier": 1.0,
                "max_leverage": 2.0,
                "stop_loss_tightness": "normal",
                "take_profit_aggressiveness": "moderate",
                "risk_level": "medium",
                "recommendation": "Standard trading parameters"
            }
            
            # Adjust based on volume strength
            volume_strength = pattern.volume_strength.lower()
            
            if volume_strength == "extreme":
                # Extreme volume strength - aggressive parameters
                base_params.update({
                    "position_size_multiplier": 1.5,
                    "max_leverage": 3.0,
                    "stop_loss_tightness": "loose",
                    "take_profit_aggressiveness": "aggressive",
                    "risk_level": "low",
                    "recommendation": "Strong volume confirmation - can trade aggressively"
                })
            elif volume_strength == "strong":
                # Strong volume strength - moderate-aggressive parameters
                base_params.update({
                    "position_size_multiplier": 1.25,
                    "max_leverage": 2.5,
                    "stop_loss_tightness": "normal",
                    "take_profit_aggressiveness": "moderate-aggressive",
                    "risk_level": "low-medium",
                    "recommendation": "Good volume confirmation - can increase position size"
                })
            elif volume_strength == "medium":
                # Medium volume strength - standard parameters
                base_params.update({
                    "position_size_multiplier": 1.0,
                    "max_leverage": 2.0,
                    "stop_loss_tightness": "normal",
                    "take_profit_aggressiveness": "moderate",
                    "risk_level": "medium",
                    "recommendation": "Moderate volume confirmation - standard trading"
                })
            elif volume_strength == "weak":
                # Weak volume strength - conservative parameters
                base_params.update({
                    "position_size_multiplier": 0.75,
                    "max_leverage": 1.5,
                    "stop_loss_tightness": "tight",
                    "take_profit_aggressiveness": "conservative",
                    "risk_level": "medium-high",
                    "recommendation": "Weak volume confirmation - reduce position size"
                })
            
            # Additional adjustments based on volume confidence
            if pattern.volume_confidence >= 0.8:
                base_params["position_size_multiplier"] *= 1.1
                base_params["max_leverage"] = min(base_params["max_leverage"] * 1.1, 5.0)
            elif pattern.volume_confidence <= 0.4:
                base_params["position_size_multiplier"] *= 0.8
                base_params["max_leverage"] *= 0.8
            
            # Ensure parameters are within reasonable bounds
            base_params["position_size_multiplier"] = max(0.25, min(2.0, base_params["position_size_multiplier"]))
            base_params["max_leverage"] = max(1.0, min(5.0, base_params["max_leverage"]))
            
            return base_params
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating adaptive risk parameters: {e}")
            return {
                "position_size_multiplier": 0.5,
                "max_leverage": 1.0,
                "stop_loss_tightness": "tight",
                "take_profit_aggressiveness": "conservative",
                "risk_level": "high",
                "recommendation": "Error calculating parameters - use conservative settings"
            }
    
    def get_volume_based_trading_recommendations(self, pattern: PatternData) -> List[str]:
        """
        Get trading recommendations based on volume analysis
        
        Args:
            pattern: The pattern to analyze
            
        Returns:
            List of trading recommendations
        """
        try:
            recommendations = []
            
            if not pattern.volume_confirmation:
                recommendations.append("❌ No volume confirmation - consider avoiding this trade")
                return recommendations
            
            # Volume strength recommendations
            volume_strength = pattern.volume_strength.lower()
            if volume_strength == "extreme":
                recommendations.append("🚀 Extreme volume strength - excellent trading opportunity")
                recommendations.append("💡 Can use aggressive position sizing and wider stops")
            elif volume_strength == "strong":
                recommendations.append("✅ Strong volume confirmation - good trading setup")
                recommendations.append("💡 Consider above-average position sizing")
            elif volume_strength == "medium":
                recommendations.append("⚠️ Medium volume confirmation - standard trading setup")
                recommendations.append("💡 Use normal position sizing and risk management")
            elif volume_strength == "weak":
                recommendations.append("⚠️ Weak volume confirmation - exercise caution")
                recommendations.append("💡 Reduce position size and use tighter stops")
            
            # Volume pattern type recommendations
            volume_type = pattern.volume_pattern_type.lower()
            if "spike" in volume_type:
                recommendations.append("📈 Volume spike detected - momentum likely to continue")
            elif "divergence" in volume_type:
                recommendations.append("🔄 Volume divergence - potential reversal signal")
            elif "climax" in volume_type:
                recommendations.append("🔥 Volume climax - exhaustion may be near")
            elif "dry_up" in volume_type:
                recommendations.append("💧 Volume dry-up - consolidation or breakout ahead")
            elif "accumulation" in volume_type:
                recommendations.append("📊 Accumulation pattern - smart money buying")
            elif "distribution" in volume_type:
                recommendations.append("📉 Distribution pattern - smart money selling")
            
            # Multi-timeframe analysis recommendations
            if pattern.volume_context:
                timeframe_agreement = self._analyze_timeframe_agreement(pattern.volume_context)
                if timeframe_agreement["agreement_score"] >= 0.8:
                    recommendations.append("🎯 Strong multi-timeframe volume agreement")
                elif timeframe_agreement["agreement_score"] >= 0.6:
                    recommendations.append("📊 Moderate multi-timeframe volume agreement")
                else:
                    recommendations.append("⚠️ Weak multi-timeframe volume agreement - conflicting signals")
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"❌ Error generating volume-based recommendations: {e}")
            return ["❌ Error generating recommendations"]
    
    def _analyze_timeframe_agreement(self, volume_context: Dict) -> Dict[str, Any]:
        """
        Analyze agreement between different timeframes
        
        Args:
            volume_context: Multi-timeframe volume context
            
        Returns:
            Dictionary with agreement analysis
        """
        try:
            if not volume_context or len(volume_context) < 2:
                return {"agreement_score": 0.0, "conflicting_timeframes": [], "agreement_details": {}}
            
            # Extract pattern types and strengths for each timeframe
            timeframe_data = {}
            for timeframe, context in volume_context.items():
                if context.get("pattern_type") and context.get("strength"):
                    timeframe_data[timeframe] = {
                        "pattern_type": context["pattern_type"],
                        "strength": context["strength"],
                        "confidence": context.get("confidence", 0.0)
                    }
            
            if len(timeframe_data) < 2:
                return {"agreement_score": 0.0, "conflicting_timeframes": [], "agreement_details": {}}
            
            # Calculate agreement score
            agreement_scores = []
            conflicting_pairs = []
            
            timeframes = list(timeframe_data.keys())
            for i in range(len(timeframes)):
                for j in range(i + 1, len(timeframes)):
                    tf1, tf2 = timeframes[i], timeframes[j]
                    data1, data2 = timeframe_data[tf1], timeframe_data[tf2]
                    
                    # Pattern type agreement
                    type_agreement = 1.0 if data1["pattern_type"] == data2["pattern_type"] else 0.0
                    
                    # Strength agreement (convert to numeric for comparison)
                    strength_map = {"weak": 1, "medium": 2, "strong": 3, "extreme": 4}
                    strength1 = strength_map.get(data1["strength"], 2)
                    strength2 = strength_map.get(data2["strength"], 2)
                    strength_diff = abs(strength1 - strength2) / 3.0  # Normalize to 0-1
                    strength_agreement = 1.0 - strength_diff
                    
                    # Confidence agreement
                    conf_diff = abs(data1["confidence"] - data2["confidence"])
                    conf_agreement = 1.0 - conf_diff
                    
                    # Overall agreement for this pair
                    pair_agreement = (type_agreement + strength_agreement + conf_agreement) / 3.0
                    agreement_scores.append(pair_agreement)
                    
                    if pair_agreement < 0.5:
                        conflicting_pairs.append((tf1, tf2))
            
            # Calculate overall agreement score
            overall_agreement = sum(agreement_scores) / len(agreement_scores) if agreement_scores else 0.0
            
            return {
                "agreement_score": overall_agreement,
                "conflicting_timeframes": conflicting_pairs,
                "agreement_details": {
                    "total_comparisons": len(agreement_scores),
                    "average_pair_agreement": overall_agreement,
                    "timeframe_data": timeframe_data
                }
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error analyzing timeframe agreement: {e}")
            return {"agreement_score": 0.0, "conflicting_timeframes": [], "agreement_details": {}}
    
    def update_volume_filtering_config(self, new_config: Dict[str, Any]) -> None:
        """
        Update volume confirmation filtering configuration
        
        Args:
            new_config: Dictionary with new configuration values
        """
        try:
            # Update configuration with new values
            for key, value in new_config.items():
                if key in self.volume_filtering_config:
                    self.volume_filtering_config[key] = value
                    self.logger.info(f"✅ Updated volume filtering config: {key} = {value}")
                else:
                    self.logger.warning(f"⚠️ Unknown volume filtering config key: {key}")
            
            self.logger.info("✅ Volume filtering configuration updated successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Error updating volume filtering config: {e}")
    
    def get_volume_filtering_config(self) -> Dict[str, Any]:
        """
        Get current volume confirmation filtering configuration
        
        Returns:
            Current configuration dictionary
        """
        return self.volume_filtering_config.copy()
    
    async def get_trading_ready_patterns(
        self,
        symbol: Optional[str] = None,
        timeframe: Optional[str] = None,
        min_confidence: float = 0.6,
        min_volume_confidence: float = 0.5,
        limit: int = 50
    ) -> List[PatternData]:
        """
        Get patterns that are ready for trading decisions after volume confirmation filtering
        
        This method provides the final output of Step 3 - patterns that have been:
        1. Filtered by volume confirmation
        2. Ranked by confidence scores
        3. Ready for trading decision logic
        
        Args:
            symbol: Filter by trading symbol
            timeframe: Filter by timeframe
            min_confidence: Minimum final confidence score (after volume adjustments)
            min_volume_confidence: Minimum volume confirmation confidence
            limit: Maximum number of patterns to return
            
        Returns:
            List of trading-ready patterns, sorted by confidence
        """
        try:
            # Get all patterns from storage
            all_patterns = await self.pattern_storage.get_patterns(
                symbol=symbol,
                timeframe=timeframe,
                limit=1000  # Get more patterns for filtering
            )
            
            if not all_patterns:
                self.logger.info("ℹ️ No patterns found in storage")
                return []
            
            self.logger.info(f"🔍 Found {len(all_patterns)} patterns for trading readiness analysis")
            
            # Apply volume confirmation filtering to stored patterns
            # Note: We need to recreate volume patterns for this analysis
            # In a real implementation, you might want to store volume analysis results
            # For now, we'll use the patterns that already have volume confirmation data
            
            # Filter patterns by volume confirmation and confidence thresholds
            trading_ready_patterns = []
            
            for pattern in all_patterns:
                # Check if pattern has volume confirmation
                if not hasattr(pattern, 'volume_confirmation') or not pattern.volume_confirmation:
                    continue
                
                # Check volume confidence threshold
                if hasattr(pattern, 'volume_confidence') and pattern.volume_confidence < min_volume_confidence:
                    continue
                
                # Check final confidence threshold
                if pattern.confidence < min_confidence:
                    continue
                
                # Pattern meets all criteria
                trading_ready_patterns.append(pattern)
            
            # Sort by confidence (highest first)
            trading_ready_patterns.sort(key=lambda p: p.confidence, reverse=True)
            
            # Apply limit
            if limit > 0:
                trading_ready_patterns = trading_ready_patterns[:limit]
            
            self.logger.info(f"✅ Found {len(trading_ready_patterns)} trading-ready patterns (min_confidence: {min_confidence}, min_volume: {min_volume_confidence})")
            
            return trading_ready_patterns
            
        except Exception as e:
            self.logger.error(f"❌ Error getting trading-ready patterns: {e}")
            return []
    
    async def get_pattern_quality_metrics(
        self,
        symbol: Optional[str] = None,
        timeframe: Optional[str] = None,
        days: int = 7
    ) -> Dict[str, Any]:
        """
        Get quality metrics for patterns after volume confirmation filtering
        
        This provides insights into how well the volume confirmation filtering is working
        
        Args:
            symbol: Filter by trading symbol
            timeframe: Filter by timeframe
            days: Number of days to analyze
            
        Returns:
            Dictionary with quality metrics
        """
        try:
            # Get patterns from recent days
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(days=days)
            
            patterns = await self.pattern_storage.get_patterns(
                symbol=symbol,
                timeframe=timeframe,
                start_time=start_time,
                end_time=end_time,
                limit=1000
            )
            
            if not patterns:
                return {
                    "total_patterns": 0,
                    "volume_confirmed_patterns": 0,
                    "volume_confirmation_rate": 0.0,
                    "average_volume_confidence": 0.0,
                    "confidence_distribution": {},
                    "quality_score": 0.0
                }
            
            # Calculate metrics
            total_patterns = len(patterns)
            volume_confirmed_patterns = sum(1 for p in patterns if getattr(p, 'volume_confirmation', False))
            volume_confirmation_rate = volume_confirmed_patterns / total_patterns if total_patterns > 0 else 0.0
            
            # Volume confidence metrics
            volume_confidences = [getattr(p, 'volume_confidence', 0.0) for p in patterns]
            average_volume_confidence = sum(volume_confidences) / len(volume_confidences) if volume_confidences else 0.0
            
            # Confidence distribution
            confidence_ranges = {
                "high": sum(1 for p in patterns if p.confidence >= 0.8),
                "medium": sum(1 for p in patterns if 0.6 <= p.confidence < 0.8),
                "low": sum(1 for p in patterns if p.confidence < 0.6)
            }
            
            # Quality score (weighted combination of metrics)
            quality_score = (
                volume_confirmation_rate * 0.4 +
                average_volume_confidence * 0.3 +
                (confidence_ranges["high"] / total_patterns) * 0.3
            )
            
            return {
                "total_patterns": total_patterns,
                "volume_confirmed_patterns": volume_confirmed_patterns,
                "volume_confirmation_rate": volume_confirmation_rate,
                "average_volume_confidence": average_volume_confidence,
                "confidence_distribution": confidence_ranges,
                "quality_score": quality_score,
                "analysis_period_days": days
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error getting pattern quality metrics: {e}")
            return {}
    
    async def get_cache_performance_metrics(self) -> Dict[str, Any]:
        """Get cache performance metrics and recommendations"""
        try:
            # Get cache stats from market data service
            cache_stats = self.market_data_service.get_cache_stats()
            
            # Analyze cache performance
            hit_rate = cache_stats.get('hit_rate', 0.0)
            cache_size_mb = cache_stats.get('cache_size_mb', 0.0)
            total_requests = cache_stats.get('total_requests', 0)
            
            # Performance analysis
            performance_analysis = {
                "cache_efficiency": "excellent" if hit_rate >= 80 else "good" if hit_rate >= 60 else "poor",
                "memory_usage": "optimal" if cache_size_mb <= 50 else "moderate" if cache_size_mb <= 100 else "high",
                "request_volume": "high" if total_requests >= 1000 else "moderate" if total_requests >= 100 else "low"
            }
            
            # Recommendations
            recommendations = []
            if hit_rate < 60:
                recommendations.append("Consider increasing cache timeout or enabling cache warming")
            if cache_size_mb > 100:
                recommendations.append("Cache size is high - consider reducing max_cache_size or implementing cleanup")
            if total_requests < 100:
                recommendations.append("Low request volume - cache warming may not be necessary")
            
            return {
                "cache_stats": cache_stats,
                "performance_analysis": performance_analysis,
                "recommendations": recommendations,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error getting cache performance metrics: {e}")
            return {"error": str(e)}
    
    async def get_lazy_volume_analysis_metrics(self) -> Dict[str, Any]:
        """Get lazy volume analysis performance metrics and efficiency data"""
        try:
            # Get recent detection stats
            total_patterns = self.detection_stats.get("total_patterns_detected", 0)
            
            # Calculate efficiency metrics
            min_confidence = self.volume_filtering_config.get("min_confidence_for_volume", 0.5)
            batch_size = self.volume_filtering_config.get("volume_analysis_batch_size", 10)
            lazy_enabled = self.volume_filtering_config.get("lazy_volume_analysis", True)
            
            # Estimate efficiency gains (this would be more accurate with actual tracking)
            estimated_efficiency = {
                "patterns_above_threshold": f"~{int(total_patterns * 0.6)}",  # Estimate 60% above threshold
                "patterns_below_threshold": f"~{int(total_patterns * 0.4)}",  # Estimate 40% below threshold
                "efficiency_gain_percentage": "~40%",
                "volume_analysis_batches": f"~{max(1, int(total_patterns * 0.6 / batch_size))}"
            }
            
            # Configuration analysis
            config_analysis = {
                "lazy_analysis_enabled": lazy_enabled,
                "confidence_threshold": min_confidence,
                "batch_size": batch_size,
                "skip_low_priority": self.volume_filtering_config.get("skip_volume_for_low_priority", True)
            }
            
            # Recommendations
            recommendations = []
            if not lazy_enabled:
                recommendations.append("Enable lazy volume analysis for better performance")
            if min_confidence > 0.7:
                recommendations.append("Consider lowering confidence threshold to analyze more patterns")
            if batch_size < 5:
                recommendations.append("Increase batch size for better processing efficiency")
            
            return {
                "efficiency_metrics": estimated_efficiency,
                "configuration": config_analysis,
                "recommendations": recommendations,
                "total_patterns_processed": total_patterns,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error getting lazy volume analysis metrics: {e}")
            return {"error": str(e)}
    
    async def optimize_cache_settings(self, target_hit_rate: float = 80.0, max_cache_size_mb: float = 100.0):
        """
        Automatically optimize cache settings based on performance targets.
        
        Args:
            target_hit_rate: Target cache hit rate percentage
            max_cache_size_mb: Maximum cache size in MB
        """
        try:
            market_data_service = MarketDataService()
            
            current_stats = market_data_service.get_cache_stats()
            optimizations_applied = []
            
            # Optimize cache timeout based on hit rate
            if current_stats['hit_rate_percent'] < target_hit_rate:
                # Increase cache timeout to improve hit rate
                new_timeout = min(300, current_stats.get('cache_timeout', 60) * 1.5)  # Max 5 minutes
                market_data_service.set_cache_timeout(int(new_timeout))
                optimizations_applied.append(f"Increased cache timeout to {new_timeout}s for better hit rate")
            
            # Enable cache warming if hit rate is low
            if current_stats['hit_rate_percent'] < 60:
                market_data_service.enable_cache_warming(True)
                optimizations_applied.append("Enabled cache warming to improve performance")
            
            # Clean up cache if size is too large
            if current_stats['cache_size_mb'] > max_cache_size_mb:
                market_data_service.clear_cache()
                optimizations_applied.append("Cleared cache to reduce memory usage")
            
            return {
                'optimizations_applied': optimizations_applied,
                'previous_stats': current_stats,
                'new_stats': market_data_service.get_cache_stats()
            }
            
        except Exception as e:
            self.logger.error(f"Error optimizing cache settings: {e}")
            return {
                'error': str(e),
                'optimizations_applied': [],
                'previous_stats': {},
                'new_stats': {}
            }
    
    def _check_pattern_volume_confirmation(self, pattern: PatternData, volume_patterns: List[VolumePattern]) -> Tuple[bool, str, str]:
        """
        Check if a specific pattern has volume confirmation and return details
        
        Args:
            pattern: The pattern to check
            volume_patterns: Available volume patterns
            
        Returns:
            Tuple of (has_confirmation, pattern_type, strength)
        """
        try:
            # Look for volume patterns that confirm this price pattern
            for volume_pattern in volume_patterns:
                # Check if volume pattern type aligns with price pattern
                if self._volume_pattern_confirms_price_pattern(pattern, volume_pattern):
                    pattern_type = volume_pattern.pattern_type.value if volume_pattern.pattern_type else ""
                    strength = volume_pattern.strength.value if volume_pattern.strength else ""
                    return True, pattern_type, strength
            
            # If no specific volume pattern found, check general volume conditions
            general_confirmed, pattern_type, strength = self._check_general_volume_confirmation(pattern, volume_patterns)
            return general_confirmed, pattern_type, strength
            
        except Exception as e:
            self.logger.error(f"❌ Error checking pattern volume confirmation: {e}")
            return False, "", ""
    
    def _volume_pattern_confirms_price_pattern(self, price_pattern: PatternData, volume_pattern: VolumePattern) -> bool:
        """
        Check if a volume pattern confirms a specific price pattern
        """
        try:
            pattern_name = price_pattern.pattern_name.lower()
            
            # Head and Shoulders patterns
            if any(term in pattern_name for term in ['head', 'shoulder', 'headandshoulder']):
                # Should have volume climax or distribution pattern
                return volume_pattern.pattern_type in [
                    VolumePatternType.VOLUME_CLIMAX,
                    VolumePatternType.VOLUME_DISTRIBUTION
                ]
            
            # Breakout patterns
            elif any(term in pattern_name for term in ['breakout', 'gap', 'engulfing']):
                # Should have volume breakout or spike
                return volume_pattern.pattern_type in [
                    VolumePatternType.VOLUME_BREAKOUT,
                    VolumePatternType.VOLUME_SPIKE
                ]
            
            # Reversal patterns
            elif any(term in pattern_name for term in ['reversal', 'divergence', 'hidden']):
                # Should have volume divergence or climax
                return volume_pattern.pattern_type in [
                    VolumePatternType.VOLUME_DIVERGENCE,
                    VolumePatternType.VOLUME_CLIMAX
                ]
            
            # Consolidation patterns
            elif any(term in pattern_name for term in ['triangle', 'wedge', 'channel']):
                # Should have volume dry-up or accumulation
                return volume_pattern.pattern_type in [
                    VolumePatternType.VOLUME_DRY_UP,
                    VolumePatternType.VOLUME_ACCUMULATION
                ]
            
            # Default: any volume pattern provides some confirmation
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error in volume pattern confirmation check: {e}")
            return False
    
    def _check_general_volume_confirmation(self, pattern: PatternData, volume_patterns: List[VolumePattern]) -> Tuple[bool, str, str]:
        """
        Check general volume confirmation when no specific pattern match is found
        """
        try:
            if not volume_patterns:
                return False, "", ""
            
            # Check if any volume pattern has high confidence
            high_confidence_patterns = [vp for vp in volume_patterns if vp.confidence > 0.6]
            
            if high_confidence_patterns:
                # At least one high-confidence volume pattern exists
                return True, "", "" # No specific type/strength for general confirmation
            
            # Check if volume patterns show strong market participation
            strong_patterns = [vp for vp in volume_patterns if vp.strength in [VolumeStrength.STRONG, VolumeStrength.EXTREME]]
            
            if strong_patterns:
                return True, "", "" # No specific type/strength for general confirmation
            
            return False, "", ""
            
        except Exception as e:
            self.logger.error(f"❌ Error in general volume confirmation check: {e}")
            return False, "", ""
    
    def _get_pattern_volume_confidence(self, pattern: PatternData, volume_patterns: List[VolumePattern]) -> float:
        """
        Get volume confidence score for a specific pattern
        
        Args:
            pattern: The pattern to get volume confidence for
            volume_patterns: Available volume patterns
            
        Returns:
            Volume confidence score from 0.0 to 1.0
        """
        try:
            if not volume_patterns:
                return 0.0
            
            # Find volume patterns that confirm this price pattern
            confirming_patterns = []
            
            for volume_pattern in volume_patterns:
                if self._volume_pattern_confirms_price_pattern(pattern, volume_pattern):
                    confirming_patterns.append(volume_pattern)
            
            if not confirming_patterns:
                return 0.0
            
            # Calculate weighted average confidence
            total_weight = 0.0
            weighted_sum = 0.0
            
            for vp in confirming_patterns:
                # Weight by pattern strength and confidence
                weight = self._get_volume_pattern_weight(vp)
                weighted_sum += vp.confidence * weight
                total_weight += weight
            
            if total_weight == 0:
                return 0.0
            
            final_confidence = weighted_sum / total_weight
            
            # Ensure confidence is between 0.0 and 1.0
            return min(1.0, max(0.0, final_confidence))
            
        except Exception as e:
            self.logger.error(f"❌ Error getting pattern volume confidence: {e}")
            return 0.0
    
    def _get_volume_pattern_weight(self, volume_pattern: VolumePattern) -> float:
        """
        Get weight for a volume pattern based on its strength and type
        """
        try:
            # Base weight from pattern strength
            strength_weights = {
                VolumeStrength.WEAK: 0.5,
                VolumeStrength.MEDIUM: 1.0,
                VolumeStrength.STRONG: 1.5,
                VolumeStrength.EXTREME: 2.0
            }
            
            base_weight = strength_weights.get(volume_pattern.strength, 1.0)
            
            # Additional weight from pattern type importance
            type_weights = {
                VolumePatternType.VOLUME_BREAKOUT: 1.2,      # Important for trend continuation
                VolumePatternType.VOLUME_DIVERGENCE: 1.3,     # Important for reversals
                VolumePatternType.VOLUME_CLIMAX: 1.1,         # Important for exhaustion
                VolumePatternType.VOLUME_SPIKE: 1.0,          # Standard confirmation
                VolumePatternType.VOLUME_ACCUMULATION: 1.1,   # Important for bottoms
                VolumePatternType.VOLUME_DISTRIBUTION: 1.1,   # Important for tops
                VolumePatternType.VOLUME_DRY_UP: 0.8,         # Less important
            }
            
            type_weight = type_weights.get(volume_pattern.pattern_type, 1.0)
            
            return base_weight * type_weight
            
        except Exception as e:
            self.logger.error(f"❌ Error getting volume pattern weight: {e}")
            return 1.0

    async def store_patterns_batch(self, patterns: List[PatternData]) -> Tuple[int, int]:
        """Store multiple patterns in a batch with production optimization"""
        try:
            if not patterns:
                return 0, 0
            
            start_time = time.time()
            
            # Determine if we should use optimized storage
            if self.production_config["enable_optimized_storage"]:
                # Calculate optimal batch size
                optimal_batch_size = None
                if self.production_config["batch_size_optimization"]:
                    optimal_batch_size = self.pattern_storage._calculate_optimal_batch_size(len(patterns))
                
                # Determine if we should use parallel processing
                use_parallel = False
                max_workers = None
                
                if (self.production_config["auto_parallel_processing"] and 
                    len(patterns) >= self.production_config["parallel_threshold"]):
                    use_parallel = True
                    max_workers = min(
                        self.production_config["max_parallel_workers"],
                        self.pattern_storage._calculate_optimal_worker_count()
                    )
                
                # Use optimized storage
                inserted, skipped = await self.pattern_storage.store_patterns_batch_optimized(
                    patterns, 
                    batch_size=optimal_batch_size,
                    use_parallel=use_parallel,
                    max_workers=max_workers
                )
            else:
                # Use regular batch storage
                inserted, skipped = await self.pattern_storage.store_patterns_batch(patterns)
            
            storage_time = (time.time() - start_time) * 1000  # Convert to milliseconds
            
            # Update detailed statistics
            self._update_batch_processing_stats(len(patterns), storage_time, use_parallel)
            self._update_storage_performance_stats(inserted, storage_time)
            self._update_real_time_metrics(len(patterns), storage_time)
            
            # Update basic statistics
            self.detection_stats["patterns_stored"] += inserted
            if skipped > 0:
                self.detection_stats["storage_errors"] += skipped
            
            # Check for performance alerts
            await self._check_performance_alerts()
            
            self.logger.info(f"✅ Production-optimized batch storage: {inserted} inserted, {skipped} skipped in {storage_time:.2f}ms")
            return inserted, skipped
            
        except Exception as e:
            self.logger.error(f"❌ Error in production-optimized batch storage: {e}")
            self.detection_stats["storage_errors"] += len(patterns)
            return 0, len(patterns)

    def _update_batch_processing_stats(self, batch_size: int, processing_time_ms: float, used_parallel: bool):
        """Update batch processing statistics"""
        stats = self.detection_stats["batch_processing_stats"]
        
        stats["total_batches"] += 1
        if used_parallel:
            stats["parallel_batches"] += 1
        else:
            stats["sequential_batches"] += 1
        
        # Update average batch size
        total_batches = stats["total_batches"]
        current_avg = stats["avg_batch_size"]
        stats["avg_batch_size"] = ((current_avg * (total_batches - 1)) + batch_size) / total_batches
        
        # Update processing time
        stats["total_processing_time"] += processing_time_ms
        
        # Calculate patterns per second
        patterns_per_second = (batch_size / (processing_time_ms / 1000)) if processing_time_ms > 0 else 0
        if patterns_per_second > stats["peak_patterns_per_second"]:
            stats["peak_patterns_per_second"] = patterns_per_second

    def _update_storage_performance_stats(self, patterns_inserted: int, storage_time_ms: float):
        """Update storage performance statistics"""
        stats = self.detection_stats["storage_performance"]
        
        # Get storage service performance stats
        storage_stats = self.pattern_storage.get_performance_stats()
        
        stats["copy_operations"] = storage_stats.get("copy_operations", 0)
        stats["parallel_copy_operations"] = storage_stats.get("copy_operations", 0)  # Will be updated when we add parallel copy tracking
        stats["regular_batches"] = storage_stats.get("regular_batches", 0)
        
        # Update average storage time
        total_patterns = stats.get("total_patterns", 0) + patterns_inserted
        current_avg = stats.get("avg_storage_time_ms", 0.0)
        stats["avg_storage_time_ms"] = ((current_avg * (total_patterns - patterns_inserted)) + storage_time_ms) / total_patterns
        
        # Calculate storage throughput
        stats["storage_throughput"] = (patterns_inserted / (storage_time_ms / 1000)) if storage_time_ms > 0 else 0
        stats["total_patterns"] = total_patterns

    def _update_real_time_metrics(self, batch_size: int, processing_time_ms: float):
        """Update real-time performance metrics"""
        metrics = self.detection_stats["real_time_metrics"]
        
        # Update current patterns per second
        metrics["current_patterns_per_second"] = (batch_size / (processing_time_ms / 1000)) if processing_time_ms > 0 else 0
        metrics["current_batch_size"] = batch_size
        
        # Update active workers (approximate)
        if self.production_config["auto_parallel_processing"]:
            metrics["active_workers"] = min(
                self.production_config["max_parallel_workers"],
                self.pattern_storage._calculate_optimal_worker_count()
            )
        else:
            metrics["active_workers"] = 1
        
        # Update memory usage
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            metrics["memory_usage_mb"] = memory_info.rss / 1024 / 1024  # Convert to MB
        except ImportError:
            metrics["memory_usage_mb"] = 0.0
        
        # Update last optimization check
        metrics["last_optimization_check"] = datetime.now()

    async def _check_performance_alerts(self):
        """Check for performance alerts and log warnings"""
        try:
            alerts = self.production_config["alert_thresholds"]
            current_metrics = self.detection_stats["real_time_metrics"]
            storage_stats = self.detection_stats["storage_performance"]
            
            # Check performance threshold
            if current_metrics["current_patterns_per_second"] < alerts["low_performance"]:
                self.logger.warning(f"⚠️ Performance Alert: Current throughput {current_metrics['current_patterns_per_second']:.1f} patterns/s is below threshold {alerts['low_performance']}")
            
            # Check memory usage
            if current_metrics["memory_usage_mb"] > 0:
                memory_percentage = (current_metrics["memory_usage_mb"] / 1024) * 100  # Assuming 1GB baseline
                if memory_percentage > alerts["high_memory"]:
                    self.logger.warning(f"⚠️ Memory Alert: Memory usage {memory_percentage:.1f}% exceeds threshold {alerts['high_memory']}%")
            
            # Check storage errors
            if self.detection_stats["storage_errors"] > alerts["storage_errors"]:
                self.logger.warning(f"⚠️ Storage Alert: Storage errors {self.detection_stats['storage_errors']} exceed threshold {alerts['storage_errors']}")
                
        except Exception as e:
            self.logger.error(f"❌ Error checking performance alerts: {e}")

    async def get_production_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive production performance report"""
        try:
            # Get storage service performance
            storage_performance = self.pattern_storage.get_performance_stats()
            
            # Get system health
            system_health = await self._get_system_health()
            
            # Calculate performance metrics
            performance_metrics = self._calculate_performance_metrics()
            
            return {
                "timestamp": datetime.now().isoformat(),
                "service_status": "healthy" if self._initialized else "initializing",
                "detection_stats": self.detection_stats,
                "storage_performance": storage_performance,
                "system_health": system_health,
                "performance_metrics": performance_metrics,
                "optimization_recommendations": await self._get_optimization_recommendations(),
                "production_config": self.production_config
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error generating performance report: {e}")
            return {"error": str(e)}

    def _calculate_performance_metrics(self) -> Dict[str, Any]:
        """Calculate key performance metrics"""
        try:
            batch_stats = self.detection_stats["batch_processing_stats"]
            storage_stats = self.detection_stats["storage_performance"]
            
            # Calculate efficiency metrics
            total_patterns = self.detection_stats["total_patterns_detected"]
            stored_patterns = self.detection_stats["patterns_stored"]
            
            storage_efficiency = (stored_patterns / total_patterns * 100) if total_patterns > 0 else 0
            parallel_efficiency = (batch_stats["parallel_batches"] / batch_stats["total_batches"] * 100) if batch_stats["total_batches"] > 0 else 0
            
            # Calculate throughput metrics
            avg_throughput = batch_stats["peak_patterns_per_second"]
            current_throughput = self.detection_stats["real_time_metrics"]["current_patterns_per_second"]
            
            # Calculate resource utilization
            memory_usage_mb = self.detection_stats["real_time_metrics"]["memory_usage_mb"]
            active_workers = self.detection_stats["real_time_metrics"]["active_workers"]
            
            return {
                "storage_efficiency_percentage": storage_efficiency,
                "parallel_efficiency_percentage": parallel_efficiency,
                "peak_throughput": avg_throughput,
                "current_throughput": current_throughput,
                "memory_usage_mb": memory_usage_mb,
                "active_workers": active_workers,
                "throughput_efficiency": (current_throughput / avg_throughput * 100) if avg_throughput > 0 else 0
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating performance metrics: {e}")
            return {}

    async def _get_system_health(self) -> Dict[str, Any]:
        """Get system health information"""
        try:
            health_info = {
                "database_health": await self.pattern_storage.get_storage_health(),
                "memory_usage": self.detection_stats["real_time_metrics"]["memory_usage_mb"],
                "active_workers": self.detection_stats["real_time_metrics"]["active_workers"],
                "last_optimization": self.detection_stats["real_time_metrics"]["last_optimization_check"]
            }
            
            # Determine overall health status
            health_score = 100
            
            # Deduct points for issues
            if self.detection_stats["storage_errors"] > 0:
                health_score -= min(20, self.detection_stats["storage_errors"] * 2)
            
            if health_info["memory_usage"] > 800:  # If memory usage > 800MB
                health_score -= 15
            
            if not health_info["database_health"].get("healthy", False):
                health_score -= 30
            
            health_info["overall_health_score"] = max(0, health_score)
            health_info["health_status"] = "excellent" if health_score >= 90 else "good" if health_score >= 70 else "fair" if health_score >= 50 else "poor"
            
            return health_info
            
        except Exception as e:
            self.logger.error(f"❌ Error getting system health: {e}")
            return {"error": str(e)}

    async def _get_optimization_recommendations(self) -> List[str]:
        """Get optimization recommendations based on current performance"""
        try:
            recommendations = []
            
            # Check batch size optimization
            batch_stats = self.detection_stats["batch_processing_stats"]
            if batch_stats["avg_batch_size"] < 100:
                recommendations.append("Consider increasing batch sizes for better throughput")
            
            # Check parallel processing usage
            if batch_stats["parallel_batches"] / max(batch_stats["total_batches"], 1) < 0.3:
                recommendations.append("Consider enabling parallel processing for more batches")
            
            # Check memory usage
            memory_usage = self.detection_stats["real_time_metrics"]["memory_usage_mb"]
            if memory_usage > 500:
                recommendations.append("High memory usage detected - consider optimizing batch sizes or reducing worker count")
            
            # Check storage performance
            storage_stats = self.detection_stats["storage_performance"]
            if storage_stats.get("copy_operations", 0) == 0:
                recommendations.append("No COPY operations detected - consider using optimized storage for large batches")
            
            # Check throughput efficiency
            performance_metrics = self._calculate_performance_metrics()
            if performance_metrics.get("throughput_efficiency", 100) < 70:
                recommendations.append("Current throughput is below peak - check for bottlenecks in processing pipeline")
            
            if not recommendations:
                recommendations.append("System is performing optimally - no immediate optimizations needed")
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"❌ Error getting optimization recommendations: {e}")
            return ["Error generating optimization recommendations"]

# Global instance for easy access
pattern_integration_service = PatternIntegrationService()
