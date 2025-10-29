"""
Adaptive Signal Rate Controller for AlphaPulse
Maintains target signal rate through dynamic threshold adjustment
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from collections import deque
import asyncio

logger = logging.getLogger(__name__)

@dataclass
class SignalRateMetrics:
    """Metrics for signal rate monitoring"""
    signals_last_1h: int
    signals_last_6h: int
    signals_last_24h: int
    avg_confidence: float
    avg_quality_score: float
    win_rate: float
    current_thresholds: Dict[str, Any]
    adjustment_recommendation: str

@dataclass
class ThresholdAdjustment:
    """Threshold adjustment recommendation"""
    action: str  # 'loosen', 'tighten', 'maintain'
    adjustment_magnitude: float  # 0.0-1.0
    reason: str
    new_min_confidence: float
    new_min_consensus_heads: int
    new_min_quality_score: float
    new_duplicate_window_hours: int

class AdaptiveSignalController:
    """
    Maintains target signal rate through dynamic threshold adjustment
    
    Target: Flexible 3-8 signals/day (can vary based on market conditions)
    Strategy: Relaxes thresholds by 15-20% when signal flow is low
    
    Adjusts:
    - min_confidence: 0.70 - 0.90 (base: 0.78)
    - min_consensus_heads: 3 - 6 (base: 4)
    - min_quality_score: 0.65 - 0.85 (base: 0.70)
    - duplicate_window_hours: 2 - 8 (base: 4)
    
    Monitors:
    - Signals generated (1h, 6h, 24h windows)
    - Signal quality scores
    - Win rate (if available)
    - Market volatility
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize Adaptive Signal Controller"""
        self.config = config or {}
        
        # Target signal rates (flexible ranges)
        self.target_min_signals_daily = self.config.get('target_min_signals', 3)
        self.target_max_signals_daily = self.config.get('target_max_signals', 8)
        self.target_optimal_signals_daily = 5  # Sweet spot
        
        # Adjustment parameters
        self.adjustment_interval_hours = 6  # Adjust every 6 hours
        self.last_adjustment_time = datetime.now()
        self.max_adjustment_per_cycle = 0.20  # 20% max change per cycle
        
        # Current adaptive thresholds (calibrated for 2-5 signals/day) - will auto-adjust
        self.min_confidence = 0.78
        self.min_consensus_heads = 5  # CALIBRATED: Start at 5/9 heads
        self.min_quality_score = 0.70
        self.duplicate_window_hours = 6  # CALIBRATED: Extended to 6 hours
        
        # Threshold bounds
        self.confidence_bounds = (0.70, 0.90)
        self.consensus_heads_bounds = (3, 6)  # Can relax to 3, tighten to 6
        self.quality_score_bounds = (0.65, 0.85)
        self.duplicate_window_bounds = (4, 8)  # CALIBRATED: 4-8 hour range
        
        # Signal history tracking (last 24 hours)
        self.signal_history = deque(maxlen=1000)
        
        # Performance tracking
        self.performance_history = {
            'adjustments_made': 0,
            'loosened_count': 0,
            'tightened_count': 0,
            'signals_generated_total': 0,
            'avg_daily_signals': 0.0,
            'threshold_history': []
        }
        
        # Market regime awareness
        self.current_market_regime = 'unknown'
        self.regime_adjustment_factors = {
            'trending': 0.9,  # Slightly easier in trending markets
            'ranging': 1.1,  # Slightly harder in ranging markets
            'volatile': 1.15,  # Harder in volatile markets
            'accumulation': 1.0,  # Normal in accumulation
            'unknown': 1.0
        }
        
        logger.info(f"✅ Adaptive Signal Controller initialized (target: {self.target_min_signals_daily}-{self.target_max_signals_daily} signals/day)")
    
    async def should_adjust_thresholds(self) -> bool:
        """Check if it's time to adjust thresholds"""
        time_since_last = (datetime.now() - self.last_adjustment_time).total_seconds() / 3600
        return time_since_last >= self.adjustment_interval_hours
    
    async def adjust_thresholds(
        self,
        recent_signals: List[Dict[str, Any]],
        market_regime: Optional[str] = None,
        win_rate: Optional[float] = None
    ) -> ThresholdAdjustment:
        """
        Analyze signal flow and adjust thresholds accordingly
        
        Args:
            recent_signals: List of recent signals with timestamps and quality metrics
            market_regime: Current market regime ('trending', 'ranging', 'volatile', etc.)
            win_rate: Optional win rate for performance-based adjustment
            
        Returns:
            ThresholdAdjustment with new threshold values
        """
        try:
            # Update market regime
            if market_regime:
                self.current_market_regime = market_regime
            
            # Calculate current metrics
            metrics = await self._calculate_metrics(recent_signals, win_rate)
            
            # Determine adjustment action
            adjustment = await self._determine_adjustment(metrics)
            
            # Apply adjustments
            if adjustment.action != 'maintain':
                self._apply_adjustment(adjustment)
                self.last_adjustment_time = datetime.now()
                
                # Log adjustment
                logger.info(
                    f"📊 Threshold Adjustment ({adjustment.action.upper()}): "
                    f"confidence: {self.min_confidence:.3f}, "
                    f"consensus: {self.min_consensus_heads}, "
                    f"quality: {self.min_quality_score:.3f}, "
                    f"dup_window: {self.duplicate_window_hours}h | "
                    f"Reason: {adjustment.reason}"
                )
                
                # Update performance history
                self.performance_history['adjustments_made'] += 1
                if adjustment.action == 'loosen':
                    self.performance_history['loosened_count'] += 1
                elif adjustment.action == 'tighten':
                    self.performance_history['tightened_count'] += 1
                
                self.performance_history['threshold_history'].append({
                    'timestamp': datetime.now().isoformat(),
                    'action': adjustment.action,
                    'reason': adjustment.reason,
                    'min_confidence': self.min_confidence,
                    'min_consensus_heads': self.min_consensus_heads,
                    'min_quality_score': self.min_quality_score,
                    'signals_24h': metrics.signals_last_24h
                })
            
            return adjustment
            
        except Exception as e:
            logger.error(f"❌ Error adjusting thresholds: {e}")
            return ThresholdAdjustment(
                action='maintain',
                adjustment_magnitude=0.0,
                reason=f"Error: {str(e)}",
                new_min_confidence=self.min_confidence,
                new_min_consensus_heads=self.min_consensus_heads,
                new_min_quality_score=self.min_quality_score,
                new_duplicate_window_hours=self.duplicate_window_hours
            )
    
    async def _calculate_metrics(
        self,
        recent_signals: List[Dict[str, Any]],
        win_rate: Optional[float]
    ) -> SignalRateMetrics:
        """Calculate current signal rate metrics"""
        now = datetime.now()
        
        # Count signals in different time windows
        signals_1h = 0
        signals_6h = 0
        signals_24h = 0
        confidences = []
        quality_scores = []
        
        for signal in recent_signals:
            # Parse timestamp
            if isinstance(signal.get('timestamp'), str):
                sig_time = datetime.fromisoformat(signal['timestamp'].replace('Z', '+00:00'))
            elif isinstance(signal.get('timestamp'), datetime):
                sig_time = signal['timestamp']
            else:
                continue
            
            age_hours = (now - sig_time).total_seconds() / 3600
            
            if age_hours <= 1:
                signals_1h += 1
            if age_hours <= 6:
                signals_6h += 1
            if age_hours <= 24:
                signals_24h += 1
                confidences.append(signal.get('confidence', 0.5))
                quality_scores.append(signal.get('quality_score', 0.5))
        
        # Calculate averages
        avg_confidence = np.mean(confidences) if confidences else 0.5
        avg_quality_score = np.mean(quality_scores) if quality_scores else 0.5
        
        return SignalRateMetrics(
            signals_last_1h=signals_1h,
            signals_last_6h=signals_6h,
            signals_last_24h=signals_24h,
            avg_confidence=avg_confidence,
            avg_quality_score=avg_quality_score,
            win_rate=win_rate or 0.5,
            current_thresholds={
                'min_confidence': self.min_confidence,
                'min_consensus_heads': self.min_consensus_heads,
                'min_quality_score': self.min_quality_score,
                'duplicate_window_hours': self.duplicate_window_hours
            },
            adjustment_recommendation=''
        )
    
    async def _determine_adjustment(
        self,
        metrics: SignalRateMetrics
    ) -> ThresholdAdjustment:
        """Determine if and how to adjust thresholds"""
        
        # Calculate expected signals (assuming linear projection)
        expected_signals_daily = metrics.signals_last_6h * 4  # Project 6h to 24h
        
        # Get market regime adjustment factor
        regime_factor = self.regime_adjustment_factors.get(
            self.current_market_regime, 1.0
        )
        
        # Adjust targets based on regime
        adjusted_min = self.target_min_signals_daily * regime_factor
        adjusted_max = self.target_max_signals_daily * regime_factor
        
        # Determine action
        reasons = []
        
        # Signal flow analysis
        if metrics.signals_last_24h < adjusted_min:
            # Too few signals - LOOSEN
            deficit = adjusted_min - metrics.signals_last_24h
            urgency = min(deficit / adjusted_min, 1.0)  # 0-1
            
            # Adjust based on how severe the drought is
            if metrics.signals_last_24h == 0:
                magnitude = 0.20  # Max loosening (20%)
                reasons.append("Signal drought - aggressive loosening")
            elif metrics.signals_last_24h < adjusted_min * 0.5:
                magnitude = 0.15  # Significant loosening (15%)
                reasons.append(f"Very low signal rate ({metrics.signals_last_24h}/{adjusted_min:.0f})")
            else:
                magnitude = 0.10  # Moderate loosening (10%)
                reasons.append(f"Below target signal rate ({metrics.signals_last_24h}/{adjusted_min:.0f})")
            
            action = 'loosen'
            
        elif metrics.signals_last_24h > adjusted_max:
            # Too many signals - TIGHTEN
            excess = metrics.signals_last_24h - adjusted_max
            urgency = min(excess / adjusted_max, 1.0)
            
            # Tighten based on how much we're overshooting
            if metrics.signals_last_24h > adjusted_max * 2:
                magnitude = 0.20  # Max tightening (20%)
                reasons.append("Signal flood - aggressive tightening")
            elif metrics.signals_last_24h > adjusted_max * 1.5:
                magnitude = 0.15  # Significant tightening (15%)
                reasons.append(f"Very high signal rate ({metrics.signals_last_24h}/{adjusted_max:.0f})")
            else:
                magnitude = 0.10  # Moderate tightening (10%)
                reasons.append(f"Above target signal rate ({metrics.signals_last_24h}/{adjusted_max:.0f})")
            
            action = 'tighten'
            
        else:
            # Within target range - but check quality
            action = 'maintain'
            magnitude = 0.0
            reasons.append(f"Signal rate optimal ({metrics.signals_last_24h} within {adjusted_min:.0f}-{adjusted_max:.0f})")
            
            # Quality-based fine-tuning
            if metrics.avg_quality_score < 0.65:
                action = 'tighten'
                magnitude = 0.05  # Small tightening for quality
                reasons.append("Low average quality - minor tightening")
            elif metrics.win_rate is not None and metrics.win_rate < 0.45:
                action = 'tighten'
                magnitude = 0.08  # Medium tightening for poor performance
                reasons.append(f"Low win rate ({metrics.win_rate:.2%}) - tightening for quality")
        
        # Calculate new thresholds
        if action == 'loosen':
            new_confidence = max(
                self.confidence_bounds[0],
                self.min_confidence - (self.min_confidence * magnitude * 0.5)
            )
            new_quality = max(
                self.quality_score_bounds[0],
                self.min_quality_score - (self.min_quality_score * magnitude * 0.5)
            )
            new_consensus = max(
                self.consensus_heads_bounds[0],
                self.min_consensus_heads - (1 if magnitude >= 0.15 else 0)
            )
            new_dup_window = max(
                self.duplicate_window_bounds[0],
                self.duplicate_window_hours - (1 if magnitude >= 0.15 else 0)
            )
            
        elif action == 'tighten':
            new_confidence = min(
                self.confidence_bounds[1],
                self.min_confidence + (self.min_confidence * magnitude * 0.5)
            )
            new_quality = min(
                self.quality_score_bounds[1],
                self.min_quality_score + (self.min_quality_score * magnitude * 0.5)
            )
            new_consensus = min(
                self.consensus_heads_bounds[1],
                self.min_consensus_heads + (1 if magnitude >= 0.15 else 0)
            )
            new_dup_window = min(
                self.duplicate_window_bounds[1],
                self.duplicate_window_hours + (1 if magnitude >= 0.15 else 0)
            )
            
        else:
            # Maintain current
            new_confidence = self.min_confidence
            new_quality = self.min_quality_score
            new_consensus = self.min_consensus_heads
            new_dup_window = self.duplicate_window_hours
        
        return ThresholdAdjustment(
            action=action,
            adjustment_magnitude=magnitude,
            reason="; ".join(reasons),
            new_min_confidence=new_confidence,
            new_min_consensus_heads=int(new_consensus),
            new_min_quality_score=new_quality,
            new_duplicate_window_hours=int(new_dup_window)
        )
    
    def _apply_adjustment(self, adjustment: ThresholdAdjustment):
        """Apply threshold adjustments"""
        self.min_confidence = adjustment.new_min_confidence
        self.min_consensus_heads = adjustment.new_min_consensus_heads
        self.min_quality_score = adjustment.new_min_quality_score
        self.duplicate_window_hours = adjustment.new_duplicate_window_hours
    
    def get_current_thresholds(self) -> Dict[str, Any]:
        """Get current adaptive thresholds"""
        return {
            'min_confidence': self.min_confidence,
            'min_consensus_heads': self.min_consensus_heads,
            'min_quality_score': self.min_quality_score,
            'duplicate_window_hours': self.duplicate_window_hours,
            'market_regime': self.current_market_regime,
            'last_adjustment': self.last_adjustment_time.isoformat()
        }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        return {
            **self.performance_history,
            'current_thresholds': self.get_current_thresholds(),
            'target_range': f"{self.target_min_signals_daily}-{self.target_max_signals_daily}/day"
        }
    
    async def record_signal(self, signal: Dict[str, Any]):
        """Record a generated signal for tracking"""
        signal_with_time = {
            **signal,
            'recorded_at': datetime.now().isoformat()
        }
        self.signal_history.append(signal_with_time)
        self.performance_history['signals_generated_total'] += 1
    
    def is_duplicate_signal(
        self,
        symbol: str,
        timeframe: str,
        direction: str,
        recent_signals: List[Dict[str, Any]]
    ) -> bool:
        """
        Check if this would be a duplicate signal
        
        Args:
            symbol: Trading pair symbol
            timeframe: Signal timeframe
            direction: Signal direction
            recent_signals: List of recent signals
            
        Returns:
            True if duplicate, False otherwise
        """
        now = datetime.now()
        duplicate_window = timedelta(hours=self.duplicate_window_hours)
        
        for signal in recent_signals:
            # Check if same symbol and timeframe
            if signal.get('symbol') != symbol or signal.get('timeframe') != timeframe:
                continue
            
            # Parse timestamp
            if isinstance(signal.get('timestamp'), str):
                sig_time = datetime.fromisoformat(signal['timestamp'].replace('Z', '+00:00'))
            elif isinstance(signal.get('timestamp'), datetime):
                sig_time = signal['timestamp']
            else:
                continue
            
            # Check if within duplicate window
            if (now - sig_time) <= duplicate_window:
                # Check if same direction
                if signal.get('direction') == direction:
                    logger.debug(
                        f"Duplicate signal detected: {symbol} {timeframe} {direction} "
                        f"within {self.duplicate_window_hours}h window"
                    )
                    return True
        
        return False

