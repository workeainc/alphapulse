#!/usr/bin/env python3
"""
Simplified Hard Example Buffer Service for AlphaPulse
Phase 5C: Misclassification Capture Implementation

Core logic without database dependencies for testing
"""

import logging
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import json
import time

logger = logging.getLogger(__name__)

class BufferType(Enum):
    """Hard example buffer types"""
    HARD_NEGATIVE = "hard_negative"  # 60% - misclassified or low-quality
    NEAR_POSITIVE = "near_positive"   # 40% - near decision boundary

class OutcomeStatus(Enum):
    """Trade outcome status"""
    WIN = "win"
    LOSS = "loss"
    EXPIRED = "expired"
    UNKNOWN = "unknown"

@dataclass
class TradeOutcome:
    """Computed trade outcome metrics"""
    signal_id: int
    outcome: OutcomeStatus
    realized_rr: float
    max_drawdown: float
    confidence: float
    prediction_correct: bool
    buffer_type: Optional[BufferType] = None
    reason: Optional[str] = None

@dataclass
class BufferStats:
    """Hard example buffer statistics"""
    total_examples: int
    hard_negatives: int
    near_positives: int
    hard_negative_ratio: float
    near_positive_ratio: float
    last_updated: datetime
    buffer_size_mb: float

class HardExampleBufferServiceSimple:
    """
    Simplified service for managing hard examples and misclassification capture
    Core logic without database dependencies
    """
    
    def __init__(self):
        # Buffer configuration
        self.buffer_config = {
            'max_size': 10000,  # Maximum buffer size
            'target_hard_negative_ratio': 0.60,  # 60% hard negatives
            'target_near_positive_ratio': 0.40,  # 40% near positives
            'balance_tolerance': 0.05,  # ±5% tolerance
            'min_realized_rr_threshold': 0.5,  # Minimum R/R for quality
            'max_drawdown_threshold': 0.5,  # Maximum drawdown threshold
            'confidence_boundary_low': 0.4,  # Low confidence boundary
            'confidence_boundary_high': 0.6,  # High confidence boundary
        }
        
        # Performance tracking
        self.performance_metrics = {
            'outcome_computation_time': 0.0,
            'buffer_update_time': 0.0,
            'total_trades_processed': 0,
            'total_hard_examples_captured': 0,
            'last_metrics_reset': datetime.now()
        }
        
        logger.info("🚀 Simplified Hard Example Buffer Service initialized")
    
    def categorize_single_outcome(self, 
                                  outcome: TradeOutcome, 
                                  buffer_stats: BufferStats) -> Optional[BufferType]:
        """Categorize a single trade outcome"""
        try:
            # Check if this is a hard negative (misclassified or low-quality)
            is_hard_negative = (
                not outcome.prediction_correct or  # Wrong prediction
                outcome.realized_rr < self.buffer_config['min_realized_rr_threshold'] or  # Low R/R
                outcome.max_drawdown > self.buffer_config['max_drawdown_threshold']  # High drawdown
            )
            
            # Check if this is a near positive (near decision boundary)
            is_near_positive = (
                outcome.prediction_correct and  # Correct prediction
                (self.buffer_config['confidence_boundary_low'] <= outcome.confidence <= 
                 self.buffer_config['confidence_boundary_high'])  # Low confidence
            )
            
            # Determine buffer type based on current balance
            if is_hard_negative:
                # Check if we can add more hard negatives
                current_hard_ratio = buffer_stats.hard_negative_ratio
                if current_hard_ratio < (self.buffer_config['target_hard_negative_ratio'] + 
                                       self.buffer_config['balance_tolerance']):
                    return BufferType.HARD_NEGATIVE
            
            elif is_near_positive:
                # Check if we can add more near positives
                current_near_ratio = buffer_stats.near_positive_ratio
                if current_near_ratio < (self.buffer_config['target_near_positive_ratio'] + 
                                       self.buffer_config['balance_tolerance']):
                    return BufferType.NEAR_POSITIVE
            
            # Not categorized as hard example
            return None
            
        except Exception as e:
            logger.error(f"❌ Error categorizing outcome: {e}")
            return None
    
    def determine_retrain_reason(self, example: TradeOutcome) -> str:
        """Determine reason for adding to retrain queue"""
        if not example.prediction_correct:
            return "misclassified"
        elif example.realized_rr < self.buffer_config['min_realized_rr_threshold']:
            return "low_rr"
        elif example.max_drawdown > self.buffer_config['max_drawdown_threshold']:
            return "high_drawdown"
        elif example.confidence <= self.buffer_config['confidence_boundary_high']:
            return "low_confidence"
        else:
            return "hard_example"
    
    def maintain_buffer_balance(self, buffer_stats: BufferStats):
        """Maintain buffer balance by adjusting categorization thresholds"""
        try:
            if buffer_stats.total_examples == 0:
                return
            
            # Check if balance is within tolerance
            hard_negative_balanced = (
                abs(buffer_stats.hard_negative_ratio - self.buffer_config['target_hard_negative_ratio']) 
                <= self.buffer_config['balance_tolerance']
            )
            
            near_positive_balanced = (
                abs(buffer_stats.near_positive_ratio - self.buffer_config['target_near_positive_ratio']) 
                <= self.buffer_config['balance_tolerance']
            )
            
            if hard_negative_balanced and near_positive_balanced:
                logger.info("✅ Buffer balance is within tolerance")
                return
            
            # Adjust thresholds to rebalance
            if buffer_stats.hard_negative_ratio > self.buffer_config['target_hard_negative_ratio']:
                # Too many hard negatives, make it harder to qualify
                self.buffer_config['min_realized_rr_threshold'] = min(
                    0.8, self.buffer_config['min_realized_rr_threshold'] + 0.1
                )
                logger.info(f"🔧 Adjusted R/R threshold to {self.buffer_config['min_realized_rr_threshold']}")
            
            elif buffer_stats.near_positive_ratio > self.buffer_config['target_near_positive_ratio']:
                # Too many near positives, make it harder to qualify
                self.buffer_config['confidence_boundary_high'] = max(
                    0.5, self.buffer_config['confidence_boundary_high'] - 0.05
                )
                logger.info(f"🔧 Adjusted confidence boundary to {self.buffer_config['confidence_boundary_high']}")
            
            logger.info("🔧 Buffer balance thresholds adjusted")
            
        except Exception as e:
            logger.error(f"❌ Error maintaining buffer balance: {e}")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for monitoring"""
        return {
            **self.performance_metrics,
            'config': self.buffer_config
        }
    
    def reset_performance_metrics(self):
        """Reset performance metrics"""
        self.performance_metrics = {
            'outcome_computation_time': 0.0,
            'buffer_update_time': 0.0,
            'total_trades_processed': 0,
            'total_hard_examples_captured': 0,
            'last_metrics_reset': datetime.now()
        }
        logger.info("🔄 Performance metrics reset")

# Export for use in other modules
__all__ = [
    'HardExampleBufferServiceSimple',
    'TradeOutcome',
    'BufferType',
    'OutcomeStatus',
    'BufferStats'
]
