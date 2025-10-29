"""
Smart Signal Generator for AlphaPulse
Integrates adaptive thresholds, context-aware filtering, and intelligent aggregation
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import asyncio
import pandas as pd

logger = logging.getLogger(__name__)

@dataclass
class SmartSignalResult:
    """Result from smart signal generation"""
    signal_generated: bool
    symbol: str
    timeframe: str
    direction: str
    confidence: float
    quality_score: float
    consensus_score: float
    market_regime: str
    contributing_heads: List[str]
    reasoning: str
    metadata: Dict[str, Any]
    timestamp: datetime

class SmartSignalGenerator:
    """
    Smart Signal Generator with Adaptive Intelligence
    
    Integrates:
    1. Technical Indicator Aggregator (50+ indicators)
    2. Volume Indicator Aggregator (10+ volume indicators)
    3. Adaptive Signal Rate Controller (auto-adjusting thresholds)
    4. Context-Aware Market Filter (regime-specific requirements)
    5. Enhanced Duplicate Detection
    6. Multi-Head Consensus with Dynamic Thresholds
    
    Goal: 3-8 quality signals/day with NO duplicates
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize Smart Signal Generator"""
        self.config = config or {}
        
        # Initialize components
        from .adaptive_signal_controller import AdaptiveSignalController
        from .context_aware_filter import ContextAwareMarketFilter
        from .consensus_manager import ConsensusManager
        from .model_heads import ModelHeadsManager
        
        self.adaptive_controller = AdaptiveSignalController(config.get('adaptive', {}))
        self.context_filter = ContextAwareMarketFilter(config.get('context_filter', {}))
        self.consensus_manager = ConsensusManager()
        self.model_heads_manager = ModelHeadsManager()
        
        # Signal history for duplicate detection
        self.signal_history = []
        self.history_window_hours = 24
        
        # Performance tracking
        self.stats = {
            'total_evaluations': 0,
            'signals_generated': 0,
            'signals_blocked_duplicate': 0,
            'signals_blocked_context': 0,
            'signals_blocked_consensus': 0,
            'signals_blocked_quality': 0,
            'threshold_adjustments': 0,
            'avg_confidence': 0.0,
            'avg_quality_score': 0.0
        }
        
        # Periodically adjust thresholds
        self.last_threshold_check = datetime.now()
        
        logger.info("🚀 Smart Signal Generator initialized with Adaptive Intelligence")
    
    async def generate_signal(
        self,
        symbol: str,
        timeframe: str,
        market_data: Dict[str, Any],
        analysis_results: Dict[str, Any]
    ) -> Optional[SmartSignalResult]:
        """
        Generate a smart signal with all intelligence layers
        
        Args:
            symbol: Trading pair symbol
            timeframe: Signal timeframe
            market_data: Market data dict
            analysis_results: Analysis results from various engines
            
        Returns:
            SmartSignalResult if signal passed all filters, None otherwise
        """
        start_time = datetime.now()
        self.stats['total_evaluations'] += 1
        
        try:
            # Step 1: Check if time to adjust adaptive thresholds
            if await self.adaptive_controller.should_adjust_thresholds():
                await self._periodic_threshold_adjustment()
            
            # Step 2: Run all model heads with aggregated indicators
            logger.debug(f"Running model heads for {symbol} {timeframe}")
            model_head_results = await self.model_heads_manager.analyze_all_heads(
                market_data,
                analysis_results
            )
            
            if not model_head_results:
                logger.warning(f"No model head results for {symbol} {timeframe}")
                return None
            
            # Step 3: Check consensus with current adaptive thresholds
            logger.debug(f"Checking consensus (threshold: {self.consensus_manager.confidence_threshold:.3f}, heads: {self.consensus_manager.min_agreeing_heads})")
            consensus_result = await self.consensus_manager.check_consensus(model_head_results)
            
            if not consensus_result.consensus_achieved:
                logger.debug(f"❌ Consensus not achieved for {symbol} {timeframe}")
                self.stats['signals_blocked_consensus'] += 1
                return None
            
            # Step 4: Context-aware filtering (market regime)
            logger.debug(f"Applying context-aware filtering")
            filter_decision = await self.context_filter.evaluate_signal(
                market_data,
                [self._head_result_to_dict(r) for r in model_head_results],
                self._consensus_result_to_dict(consensus_result),
                analysis_results
            )
            
            if not filter_decision.should_generate_signal:
                logger.debug(f"❌ Context filter blocked signal: {filter_decision.reasoning}")
                self.stats['signals_blocked_context'] += 1
                return None
            
            # Step 5: Calculate quality score
            quality_score = await self._calculate_quality_score(
                consensus_result,
                filter_decision,
                model_head_results
            )
            
            # Get current quality threshold from adaptive controller
            min_quality = self.adaptive_controller.min_quality_score
            
            if quality_score < min_quality:
                logger.debug(f"❌ Quality score too low: {quality_score:.3f} < {min_quality:.3f}")
                self.stats['signals_blocked_quality'] += 1
                return None
            
            # Step 6: Apply confidence adjustment from context filter
            adjusted_confidence = consensus_result.consensus_score * filter_decision.confidence_adjustment
            adjusted_confidence = min(1.0, max(0.0, adjusted_confidence))
            
            # Step 7: Duplicate detection
            direction = consensus_result.consensus_direction.value
            is_duplicate = self.adaptive_controller.is_duplicate_signal(
                symbol,
                timeframe,
                direction,
                self.signal_history
            )
            
            if is_duplicate:
                logger.debug(f"❌ Duplicate signal detected for {symbol} {timeframe} {direction}")
                self.stats['signals_blocked_duplicate'] += 1
                return None
            
            # Step 8: Generate final signal
            signal_result = SmartSignalResult(
                signal_generated=True,
                symbol=symbol,
                timeframe=timeframe,
                direction=direction,
                confidence=adjusted_confidence,
                quality_score=quality_score,
                consensus_score=consensus_result.consensus_score,
                market_regime=filter_decision.regime.value,
                contributing_heads=[h.value for h in consensus_result.agreeing_heads],
                reasoning=self._build_reasoning(
                    consensus_result,
                    filter_decision,
                    quality_score,
                    adjusted_confidence
                ),
                metadata={
                    'consensus_heads_count': len(consensus_result.agreeing_heads),
                    'min_heads_required': consensus_result.min_agreeing_heads,
                    'total_heads': consensus_result.total_heads,
                    'confidence_adjustment': filter_decision.confidence_adjustment,
                    'priority_score': filter_decision.priority_score,
                    'adaptive_thresholds': self.adaptive_controller.get_current_thresholds(),
                    'calculation_time_ms': (datetime.now() - start_time).total_seconds() * 1000
                },
                timestamp=datetime.now()
            )
            
            # Step 9: Record signal in history
            await self._record_signal(signal_result)
            
            # Update stats
            self.stats['signals_generated'] += 1
            self._update_stats(adjusted_confidence, quality_score)
            
            logger.info(
                f"✅ Smart Signal Generated: {symbol} {timeframe} {direction.upper()} | "
                f"Confidence: {adjusted_confidence:.3f}, Quality: {quality_score:.3f}, "
                f"Heads: {len(consensus_result.agreeing_heads)}/{consensus_result.total_heads}, "
                f"Regime: {filter_decision.regime.value}"
            )
            
            return signal_result
            
        except Exception as e:
            logger.error(f"❌ Error generating smart signal for {symbol} {timeframe}: {e}")
            return None
    
    async def _periodic_threshold_adjustment(self):
        """Periodically adjust thresholds based on signal flow"""
        try:
            logger.info("⚙️ Running periodic threshold adjustment...")
            
            # Get recent signals for analysis
            recent_signals = [
                {
                    'timestamp': sig.timestamp,
                    'confidence': sig.confidence,
                    'quality_score': sig.quality_score,
                    'symbol': sig.symbol,
                    'timeframe': sig.timeframe,
                    'direction': sig.direction
                }
                for sig in self.signal_history
            ]
            
            # Run adaptive adjustment
            adjustment = await self.adaptive_controller.adjust_thresholds(
                recent_signals,
                market_regime=None,  # Could pass detected regime
                win_rate=None  # Could pass win rate if available
            )
            
            # Apply adjustments to consensus manager
            if adjustment.action != 'maintain':
                self.consensus_manager.update_adaptive_thresholds(
                    min_agreeing_heads=adjustment.new_min_consensus_heads,
                    confidence_threshold=adjustment.new_min_confidence,
                    min_probability_threshold=None
                )
                self.stats['threshold_adjustments'] += 1
                
                logger.info(
                    f"✅ Thresholds adjusted ({adjustment.action}): "
                    f"Heads: {adjustment.new_min_consensus_heads}, "
                    f"Confidence: {adjustment.new_min_confidence:.3f}, "
                    f"Quality: {adjustment.new_min_quality_score:.3f} | "
                    f"Reason: {adjustment.reason}"
                )
            
        except Exception as e:
            logger.error(f"❌ Error in periodic threshold adjustment: {e}")
    
    async def _calculate_quality_score(
        self,
        consensus_result: Any,
        filter_decision: Any,
        model_head_results: List[Any]
    ) -> float:
        """Calculate overall signal quality score"""
        try:
            # Component scores
            consensus_quality = consensus_result.consensus_score * 0.40  # 40%
            context_quality = filter_decision.priority_score * 0.30  # 30%
            
            # Agreement quality (how many heads agree vs total)
            agreement_ratio = len(consensus_result.agreeing_heads) / max(1, consensus_result.total_heads)
            agreement_quality = agreement_ratio * 0.20  # 20%
            
            # Confidence spread (lower std = higher quality)
            confidences = [r.confidence for r in model_head_results if r.head_type in consensus_result.agreeing_heads]
            if confidences:
                import numpy as np
                confidence_std = np.std(confidences)
                spread_quality = (1.0 - min(confidence_std, 1.0)) * 0.10  # 10%
            else:
                spread_quality = 0.0
            
            # Total quality score
            quality_score = consensus_quality + context_quality + agreement_quality + spread_quality
            
            return min(1.0, max(0.0, quality_score))
            
        except Exception as e:
            logger.error(f"Error calculating quality score: {e}")
            return 0.5
    
    def _build_reasoning(
        self,
        consensus_result: Any,
        filter_decision: Any,
        quality_score: float,
        adjusted_confidence: float
    ) -> str:
        """Build comprehensive reasoning string"""
        parts = []
        
        # Consensus info
        parts.append(
            f"Consensus: {len(consensus_result.agreeing_heads)}/{consensus_result.total_heads} heads agree"
        )
        parts.append(f"Contributing: {', '.join([h.value for h in consensus_result.agreeing_heads])}")
        
        # Market context
        parts.append(f"Market: {filter_decision.regime.value}")
        parts.append(f"Context: {filter_decision.reasoning}")
        
        # Quality metrics
        parts.append(f"Quality: {quality_score:.3f}, Confidence: {adjusted_confidence:.3f}")
        
        return "; ".join(parts)
    
    async def _record_signal(self, signal: SmartSignalResult):
        """Record signal in history for duplicate detection and analysis"""
        self.signal_history.append(signal)
        
        # Clean old signals (keep only last 24 hours)
        cutoff_time = datetime.now() - timedelta(hours=self.history_window_hours)
        self.signal_history = [
            sig for sig in self.signal_history
            if sig.timestamp > cutoff_time
        ]
        
        # Also record in adaptive controller
        await self.adaptive_controller.record_signal({
            'timestamp': signal.timestamp,
            'symbol': signal.symbol,
            'timeframe': signal.timeframe,
            'direction': signal.direction,
            'confidence': signal.confidence,
            'quality_score': signal.quality_score
        })
    
    def _head_result_to_dict(self, result: Any) -> Dict[str, Any]:
        """Convert ModelHeadResult to dict"""
        return {
            'head_type': result.head_type.value if hasattr(result.head_type, 'value') else str(result.head_type),
            'direction': result.direction.value if hasattr(result.direction, 'value') else str(result.direction),
            'probability': result.probability,
            'confidence': result.confidence,
            'features_used': result.features_used,
            'reasoning': result.reasoning
        }
    
    def _consensus_result_to_dict(self, result: Any) -> Dict[str, Any]:
        """Convert ConsensusResult to dict"""
        return {
            'consensus_achieved': result.consensus_achieved,
            'consensus_direction': result.consensus_direction.value if result.consensus_direction else None,
            'consensus_score': result.consensus_score,
            'agreeing_heads': [h.value if hasattr(h, 'value') else str(h) for h in result.agreeing_heads],
            'disagreeing_heads': [h.value if hasattr(h, 'value') else str(h) for h in result.disagreeing_heads],
            'min_agreeing_heads': result.min_agreeing_heads,
            'total_heads': result.total_heads
        }
    
    def _update_stats(self, confidence: float, quality_score: float):
        """Update running statistics"""
        n = self.stats['signals_generated']
        
        self.stats['avg_confidence'] = (
            (self.stats['avg_confidence'] * (n - 1) + confidence) / n
        )
        self.stats['avg_quality_score'] = (
            (self.stats['avg_quality_score'] * (n - 1) + quality_score) / n
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics"""
        total_evals = max(1, self.stats['total_evaluations'])
        
        return {
            **self.stats,
            'signal_rate': self.stats['signals_generated'] / total_evals,
            'duplicate_block_rate': self.stats['signals_blocked_duplicate'] / total_evals,
            'context_block_rate': self.stats['signals_blocked_context'] / total_evals,
            'consensus_block_rate': self.stats['signals_blocked_consensus'] / total_evals,
            'quality_block_rate': self.stats['signals_blocked_quality'] / total_evals,
            'signals_last_24h': len(self.signal_history),
            'adaptive_controller_stats': self.adaptive_controller.get_performance_stats(),
            'context_filter_stats': self.context_filter.get_stats()
        }

