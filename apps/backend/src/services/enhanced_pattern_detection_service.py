"""
Enhanced Pattern Detection Service for AlphaPlus
Integrates ultra-fast detection, hybrid ML, and multi-symbol correlation
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd
from dataclasses import dataclass

from backend.database.connection import TimescaleDBConnection
from backend.strategies.ultra_fast_pattern_detector import UltraFastPatternDetector, PatternDetectionResult
from backend.database.advanced_indexing import AdvancedIndexingManager

logger = logging.getLogger(__name__)

@dataclass
class EnhancedPatternResult:
    """Enhanced pattern detection result with all metadata"""
    pattern_id: Optional[int] = None
    symbol: str = ""
    timeframe: str = ""
    pattern_name: str = ""
    confidence: float = 0.0
    strength: str = ""
    direction: str = ""
    timestamp: datetime = None
    price_level: float = 0.0
    volume_confirmation: bool = False
    volume_confidence: float = 0.0
    trend_alignment: str = ""
    detection_method: str = ""
    ml_confidence: float = 0.0
    talib_confidence: float = 0.0
    noise_filter_passed: bool = True
    atr_percent: float = 0.0
    body_ratio: float = 0.0
    detection_latency_ms: float = 0.0
    correlation_strength: float = 0.0
    validation_score: float = 0.0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)

class EnhancedPatternDetectionService:
    """Main service for enhanced pattern detection"""
    
    def __init__(self, db_connection: TimescaleDBConnection):
        self.db_connection = db_connection
        self.ultra_fast_detector = UltraFastPatternDetector()
        self.indexing_manager = AdvancedIndexingManager(db_connection.get_async_session)
        
        # Performance tracking
        self.performance_stats = {
            'total_detections': 0,
            'avg_latency_ms': 0.0,
            'cache_hit_rate': 0.0,
            'throughput_per_sec': 0.0,
            'memory_usage_mb': 0.0,
            'cpu_usage_pct': 0.0,
            'accuracy_score': 0.0,
            'false_positive_rate': 0.0,
            'false_negative_rate': 0.0
        }
        
        # Configuration
        self.config = {
            'enable_ml_detection': True,
            'enable_correlation_check': True,
            'enable_validation': True,
            'min_confidence_threshold': 0.6,
            'max_detection_latency_ms': 50.0,
            'correlation_symbols': ['BTC/USDT', 'ETH/USDT', 'BNB/USDT'],
            'validation_period_hours': 24
        }
        
        logger.info("🚀 Enhanced Pattern Detection Service initialized")
    
    async def initialize(self):
        """Initialize the service and ensure database tables exist"""
        try:
            # Ensure database connection
            await self.db_connection.initialize()
            
            # Create advanced indexes if they don't exist
            await self.indexing_manager.create_all_advanced_indexes()
            
            logger.info("✅ Enhanced Pattern Detection Service initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Error initializing Enhanced Pattern Detection Service: {e}")
            raise
    
    async def detect_patterns_enhanced(self, symbol: str, timeframe: str, 
                                     candles: List[Dict]) -> List[EnhancedPatternResult]:
        """Enhanced pattern detection with all optimizations"""
        start_time = datetime.now()
        
        try:
            # Step 1: Ultra-fast pattern detection
            ultra_results = await self.ultra_fast_detector.detect_patterns_async(
                symbol, timeframe, candles
            )
            
            # Step 2: Convert to enhanced results
            enhanced_results = []
            for result in ultra_results:
                enhanced_result = self._convert_to_enhanced_result(result, symbol, timeframe)
                enhanced_results.append(enhanced_result)
            
            # Step 3: Multi-symbol correlation check
            if self.config['enable_correlation_check']:
                enhanced_results = await self._apply_correlation_check(enhanced_results)
            
            # Step 4: Store patterns in database
            stored_results = await self._store_patterns(enhanced_results)
            
            # Step 5: Post-detection validation
            if self.config['enable_validation']:
                await self._schedule_validation(stored_results)
            
            # Step 6: Update performance metrics
            end_time = datetime.now()
            latency_ms = (end_time - start_time).total_seconds() * 1000
            await self._update_performance_metrics(latency_ms, len(stored_results))
            
            logger.info(f"⚡ Enhanced pattern detection completed: {len(stored_results)} patterns in {latency_ms:.2f}ms")
            
            return stored_results
            
        except Exception as e:
            logger.error(f"❌ Error in enhanced pattern detection: {e}")
            return []
    
    def _convert_to_enhanced_result(self, result: PatternDetectionResult, 
                                  symbol: str, timeframe: str) -> EnhancedPatternResult:
        """Convert ultra-fast result to enhanced result"""
        return EnhancedPatternResult(
            symbol=symbol,
            timeframe=timeframe,
            pattern_name=result.pattern_name,
            confidence=result.confidence,
            strength=result.strength,
            direction=result.direction,
            timestamp=result.timestamp,
            price_level=result.price_level,
            volume_confirmation=result.volume_confirmation,
            volume_confidence=result.volume_confidence,
            trend_alignment=result.trend_alignment,
            detection_method=result.metadata.get('detection_method', 'unknown'),
            ml_confidence=result.metadata.get('ml_confidence', 0.0),
            talib_confidence=result.metadata.get('talib_confidence', 0.0),
            noise_filter_passed=result.metadata.get('noise_filter_passed', True),
            atr_percent=result.metadata.get('atr_percent', 0.0),
            body_ratio=result.metadata.get('body_ratio', 0.0),
            detection_latency_ms=result.detection_latency_ms,
            metadata=result.metadata
        )
    
    async def _apply_correlation_check(self, results: List[EnhancedPatternResult]) -> List[EnhancedPatternResult]:
        """Apply multi-symbol correlation check"""
        try:
            for result in results:
                if result.confidence >= self.config['min_confidence_threshold']:
                    correlation_strength = await self._calculate_correlation_strength(result)
                    result.correlation_strength = correlation_strength
                    
                    # Adjust confidence based on correlation
                    if correlation_strength > 0.7:
                        result.confidence = min(result.confidence * 1.1, 1.0)
                    elif correlation_strength < 0.3:
                        result.confidence = result.confidence * 0.9
                    
                    # Store correlation data
                    result.metadata['correlation_data'] = {
                        'strength': correlation_strength,
                        'checked_symbols': self.config['correlation_symbols'],
                        'timestamp': datetime.now(timezone.utc).isoformat()
                    }
            
            return results
            
        except Exception as e:
            logger.error(f"Error applying correlation check: {e}")
            return results
    
    async def _calculate_correlation_strength(self, result: EnhancedPatternResult) -> float:
        """Calculate correlation strength with other symbols"""
        try:
            # Get recent patterns for correlation symbols
            correlation_patterns = []
            
            for corr_symbol in self.config['correlation_symbols']:
                if corr_symbol != result.symbol:
                    recent_patterns = await self._get_recent_patterns(
                        corr_symbol, result.timeframe, hours=6
                    )
                    
                    # Check for similar patterns
                    for pattern in recent_patterns:
                        if (pattern['pattern_name'] == result.pattern_name and 
                            pattern['direction'] == result.direction):
                            correlation_patterns.append(pattern)
            
            if not correlation_patterns:
                return 0.5  # Neutral correlation
            
            # Calculate correlation strength based on pattern frequency and timing
            total_correlation = 0.0
            for pattern in correlation_patterns:
                time_diff = abs((result.timestamp - pattern['timestamp']).total_seconds() / 3600)
                if time_diff <= 2:  # Within 2 hours
                    total_correlation += 1.0
                elif time_diff <= 6:  # Within 6 hours
                    total_correlation += 0.5
            
            return min(total_correlation / len(self.config['correlation_symbols']), 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating correlation strength: {e}")
            return 0.5
    
    async def _store_patterns(self, results: List[EnhancedPatternResult]) -> List[EnhancedPatternResult]:
        """Store patterns in the enhanced database"""
        try:
            async with self.db_connection.get_async_session() as session:
                stored_results = []
                
                for result in results:
                    # Insert into enhanced_candlestick_patterns table
                    query = """
                        INSERT INTO enhanced_candlestick_patterns (
                            symbol, timeframe, pattern_name, timestamp, confidence, strength,
                            direction, price_level, volume_confirmation, volume_confidence,
                            trend_alignment, detection_method, ml_confidence, talib_confidence,
                            noise_filter_passed, atr_percent, body_ratio, detection_latency_ms,
                            metadata, created_at, updated_at
                        ) VALUES (
                            :symbol, :timeframe, :pattern_name, :timestamp, :confidence, :strength,
                            :direction, :price_level, :volume_confirmation, :volume_confidence,
                            :trend_alignment, :detection_method, :ml_confidence, :talib_confidence,
                            :noise_filter_passed, :atr_percent, :body_ratio, :detection_latency_ms,
                            :metadata, NOW(), NOW()
                        ) RETURNING pattern_id
                    """
                    
                    result_dict = {
                        'symbol': result.symbol,
                        'timeframe': result.timeframe,
                        'pattern_name': result.pattern_name,
                        'timestamp': result.timestamp,
                        'confidence': result.confidence,
                        'strength': result.strength,
                        'direction': result.direction,
                        'price_level': result.price_level,
                        'volume_confirmation': result.volume_confirmation,
                        'volume_confidence': result.volume_confidence,
                        'trend_alignment': result.trend_alignment,
                        'detection_method': result.detection_method,
                        'ml_confidence': result.ml_confidence,
                        'talib_confidence': result.talib_confidence,
                        'noise_filter_passed': result.noise_filter_passed,
                        'atr_percent': result.atr_percent,
                        'body_ratio': result.body_ratio,
                        'detection_latency_ms': result.detection_latency_ms,
                        'metadata': result.metadata
                    }
                    
                    db_result = await session.execute(query, result_dict)
                    pattern_id = db_result.scalar()
                    
                    result.pattern_id = pattern_id
                    stored_results.append(result)
                
                await session.commit()
                logger.info(f"✅ Stored {len(stored_results)} patterns in database")
                
                return stored_results
                
        except Exception as e:
            logger.error(f"Error storing patterns: {e}")
            return results
    
    async def _schedule_validation(self, results: List[EnhancedPatternResult]):
        """Schedule post-detection validation"""
        try:
            async with self.db_connection.get_async_session() as session:
                for result in results:
                    if result.pattern_id and result.confidence >= 0.7:
                        # Create validation record
                        validation_query = """
                            INSERT INTO pattern_validations (
                                pattern_id, validation_status, created_at
                            ) VALUES (
                                :pattern_id, 'pending', NOW()
                            )
                        """
                        
                        await session.execute(validation_query, {
                            'pattern_id': result.pattern_id
                        })
                
                await session.commit()
                logger.info(f"✅ Scheduled validation for {len(results)} patterns")
                
        except Exception as e:
            logger.error(f"Error scheduling validation: {e}")
    
    async def _get_recent_patterns(self, symbol: str, timeframe: str, 
                                 hours: int = 6) -> List[Dict]:
        """Get recent patterns for correlation analysis"""
        try:
            async with self.db_connection.get_async_session() as session:
                query = """
                    SELECT pattern_name, direction, timestamp, confidence
                    FROM enhanced_candlestick_patterns
                    WHERE symbol = :symbol 
                    AND timeframe = :timeframe
                    AND timestamp >= NOW() - INTERVAL ':hours hours'
                    ORDER BY timestamp DESC
                """
                
                result = await session.execute(query, {
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'hours': hours
                })
                
                patterns = []
                for row in result:
                    patterns.append({
                        'pattern_name': row[0],
                        'direction': row[1],
                        'timestamp': row[2],
                        'confidence': float(row[3])
                    })
                
                return patterns
                
        except Exception as e:
            logger.error(f"Error getting recent patterns: {e}")
            return []
    
    async def _update_performance_metrics(self, latency_ms: float, num_detections: int):
        """Update performance metrics"""
        try:
            # Update local stats
            self.performance_stats['total_detections'] += num_detections
            
            # Calculate running average latency
            if self.performance_stats['total_detections'] > 0:
                current_avg = self.performance_stats['avg_latency_ms']
                total_detections = self.performance_stats['total_detections']
                new_avg = ((current_avg * (total_detections - num_detections)) + 
                          (latency_ms * num_detections)) / total_detections
                self.performance_stats['avg_latency_ms'] = new_avg
            
            # Calculate throughput
            if latency_ms > 0:
                throughput = (num_detections / (latency_ms / 1000))
                self.performance_stats['throughput_per_sec'] = throughput
            
            # Store metrics in database
            async with self.db_connection.get_async_session() as session:
                metrics_query = """
                    INSERT INTO pattern_performance_metrics (
                        detector_type, total_detections, avg_latency_ms, throughput_per_sec,
                        metrics_timestamp, metadata
                    ) VALUES (
                        'enhanced', :total_detections, :avg_latency_ms, :throughput_per_sec,
                        NOW(), :metadata
                    )
                """
                
                await session.execute(metrics_query, {
                    'total_detections': self.performance_stats['total_detections'],
                    'avg_latency_ms': self.performance_stats['avg_latency_ms'],
                    'throughput_per_sec': self.performance_stats['throughput_per_sec'],
                    'metadata': {
                        'detection_methods': ['ultra_fast', 'ml', 'correlation'],
                        'config': self.config
                    }
                })
                
                await session.commit()
                
        except Exception as e:
            logger.error(f"Error updating performance metrics: {e}")
    
    async def get_high_confidence_patterns(self, symbol: str, timeframe: str, 
                                         min_confidence: float = 0.8,
                                         hours: int = 24) -> List[EnhancedPatternResult]:
        """Get high-confidence patterns from database"""
        try:
            async with self.db_connection.get_async_session() as session:
                query = """
                    SELECT pattern_id, symbol, timeframe, pattern_name, timestamp, confidence,
                           strength, direction, price_level, volume_confirmation, volume_confidence,
                           trend_alignment, detection_method, ml_confidence, talib_confidence,
                           noise_filter_passed, atr_percent, body_ratio, detection_latency_ms, metadata
                    FROM enhanced_candlestick_patterns
                    WHERE symbol = :symbol 
                    AND timeframe = :timeframe
                    AND confidence >= :min_confidence
                    AND timestamp >= NOW() - INTERVAL ':hours hours'
                    ORDER BY confidence DESC, timestamp DESC
                """
                
                result = await session.execute(query, {
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'min_confidence': min_confidence,
                    'hours': hours
                })
                
                patterns = []
                for row in result:
                    pattern = EnhancedPatternResult(
                        pattern_id=row[0],
                        symbol=row[1],
                        timeframe=row[2],
                        pattern_name=row[3],
                        timestamp=row[4],
                        confidence=float(row[5]),
                        strength=row[6],
                        direction=row[7],
                        price_level=float(row[8]),
                        volume_confirmation=row[9],
                        volume_confidence=float(row[10]),
                        trend_alignment=row[11],
                        detection_method=row[12],
                        ml_confidence=float(row[13]) if row[13] else 0.0,
                        talib_confidence=float(row[14]) if row[14] else 0.0,
                        noise_filter_passed=row[15],
                        atr_percent=float(row[16]) if row[16] else 0.0,
                        body_ratio=float(row[17]) if row[17] else 0.0,
                        detection_latency_ms=float(row[18]) if row[18] else 0.0,
                        metadata=row[19] if row[19] else {}
                    )
                    patterns.append(pattern)
                
                return patterns
                
        except Exception as e:
            logger.error(f"Error getting high-confidence patterns: {e}")
            return []
    
    async def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        try:
            # Get database performance metrics
            async with self.db_connection.get_async_session() as session:
                query = """
                    SELECT 
                        AVG(avg_latency_ms) as avg_latency,
                        AVG(throughput_per_sec) as avg_throughput,
                        MAX(total_detections) as total_detections,
                        COUNT(*) as metric_count
                    FROM pattern_performance_metrics
                    WHERE detector_type = 'enhanced'
                    AND metrics_timestamp >= NOW() - INTERVAL '24 hours'
                """
                
                result = await session.execute(query)
                row = result.fetchone()
                
                db_metrics = {
                    'avg_latency_ms': float(row[0]) if row[0] else 0.0,
                    'avg_throughput_per_sec': float(row[1]) if row[1] else 0.0,
                    'total_detections': int(row[2]) if row[2] else 0,
                    'metric_count': int(row[3]) if row[3] else 0
                }
            
            # Combine with local stats
            summary = {
                'local_stats': self.performance_stats,
                'database_metrics': db_metrics,
                'ultra_fast_stats': self.ultra_fast_detector.get_performance_stats(),
                'configuration': self.config,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting performance summary: {e}")
            return {
                'local_stats': self.performance_stats,
                'error': str(e),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
    
    async def clear_cache(self):
        """Clear all caches"""
        try:
            self.ultra_fast_detector.clear_cache()
            logger.info("🧹 All caches cleared")
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
    
    async def update_config(self, new_config: Dict[str, Any]):
        """Update service configuration"""
        try:
            self.config.update(new_config)
            logger.info(f"✅ Configuration updated: {new_config}")
        except Exception as e:
            logger.error(f"Error updating configuration: {e}")
