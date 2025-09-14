"""
Consensus Mechanism for AlphaPlus SDE Framework
Ensures 3+ model heads must agree before generating signals
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import asyncio

logger = logging.getLogger(__name__)

class SignalDirection(Enum):
    """Signal direction enumeration"""
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"

class ModelHead(Enum):
    """Model head types"""
    HEAD_A = "head_a"  # Technical Analysis
    HEAD_B = "head_b"  # Sentiment Analysis
    HEAD_C = "head_c"  # Volume/Orderbook Analysis
    HEAD_D = "head_d"  # Rule-based Analysis

@dataclass
class ModelHeadResult:
    """Result from a single model head"""
    head_type: ModelHead
    direction: SignalDirection
    probability: float
    confidence: float
    features_used: List[str]
    reasoning: str

@dataclass
class ConsensusResult:
    """Model consensus result"""
    consensus_achieved: bool
    consensus_direction: Optional[SignalDirection]
    consensus_score: float
    agreeing_heads: List[ModelHead]
    disagreeing_heads: List[ModelHead]
    confidence_threshold: float
    min_agreeing_heads: int
    total_heads: int

class ConsensusManager:
    """Manages consensus mechanism for AI model heads"""
    
    def __init__(self):
        self.min_agreeing_heads = 3  # Minimum 3 heads must agree
        self.min_probability_threshold = 0.6  # Minimum probability threshold
        self.confidence_threshold = 0.7  # Minimum confidence threshold
        self.direction_agreement_required = True  # All agreeing heads must have same direction
        
        # Head weights for consensus calculation
        self.head_weights = {
            ModelHead.HEAD_A: 0.3,  # Technical Analysis
            ModelHead.HEAD_B: 0.25, # Sentiment Analysis
            ModelHead.HEAD_C: 0.25, # Volume Analysis
            ModelHead.HEAD_D: 0.2   # Rule-based Analysis
        }
    
    async def check_consensus(self, model_results: List[ModelHeadResult]) -> ConsensusResult:
        """Check if model heads reach consensus"""
        try:
            if len(model_results) < 4:
                logger.warning("Insufficient model heads for consensus check")
                return ConsensusResult(
                    consensus_achieved=False,
                    consensus_direction=None,
                    consensus_score=0.0,
                    agreeing_heads=[],
                    disagreeing_heads=[],
                    confidence_threshold=self.confidence_threshold,
                    min_agreeing_heads=self.min_agreeing_heads,
                    total_heads=len(model_results)
                )
            
            # Filter heads that meet minimum thresholds
            valid_heads = []
            for result in model_results:
                if (result.probability >= self.min_probability_threshold and 
                    result.confidence >= self.confidence_threshold):
                    valid_heads.append(result)
            
            if len(valid_heads) < self.min_agreeing_heads:
                logger.debug(f"Insufficient valid heads: {len(valid_heads)} < {self.min_agreeing_heads}")
                return ConsensusResult(
                    consensus_achieved=False,
                    consensus_direction=None,
                    consensus_score=0.0,
                    agreeing_heads=[],
                    disagreeing_heads=[r.head_type for r in model_results],
                    confidence_threshold=self.confidence_threshold,
                    min_agreeing_heads=self.min_agreeing_heads,
                    total_heads=len(model_results)
                )
            
            # Count agreeing heads by direction
            direction_counts = {}
            for result in valid_heads:
                direction = result.direction
                if direction not in direction_counts:
                    direction_counts[direction] = []
                direction_counts[direction].append(result)
            
            # Find the direction with most agreeing heads
            max_agreeing_count = 0
            consensus_direction = None
            agreeing_heads = []
            
            for direction, heads in direction_counts.items():
                if len(heads) > max_agreeing_count:
                    max_agreeing_count = len(heads)
                    consensus_direction = direction
                    agreeing_heads = heads
            
            # Check if consensus is achieved
            consensus_achieved = (max_agreeing_count >= self.min_agreeing_heads and 
                                 consensus_direction != SignalDirection.FLAT)
            
            if consensus_achieved:
                # Calculate weighted consensus score
                consensus_score = self._calculate_weighted_consensus_score(agreeing_heads)
                
                # Get disagreeing heads
                disagreeing_heads = [r.head_type for r in model_results 
                                   if r.head_type not in [h.head_type for h in agreeing_heads]]
                
                logger.info(f"✅ Consensus achieved: {consensus_direction.value} "
                           f"(score: {consensus_score:.3f}, agreeing: {len(agreeing_heads)})")
                
                return ConsensusResult(
                    consensus_achieved=True,
                    consensus_direction=consensus_direction,
                    consensus_score=consensus_score,
                    agreeing_heads=[h.head_type for h in agreeing_heads],
                    disagreeing_heads=disagreeing_heads,
                    confidence_threshold=self.confidence_threshold,
                    min_agreeing_heads=self.min_agreeing_heads,
                    total_heads=len(model_results)
                )
            else:
                # No consensus achieved
                disagreeing_heads = [r.head_type for r in model_results]
                
                logger.debug(f"❌ No consensus: max agreeing heads = {max_agreeing_count} < {self.min_agreeing_heads}")
                
                return ConsensusResult(
                    consensus_achieved=False,
                    consensus_direction=None,
                    consensus_score=0.0,
                    agreeing_heads=[],
                    disagreeing_heads=disagreeing_heads,
                    confidence_threshold=self.confidence_threshold,
                    min_agreeing_heads=self.min_agreeing_heads,
                    total_heads=len(model_results)
                )
                
        except Exception as e:
            logger.error(f"Consensus check error: {e}")
            return ConsensusResult(
                consensus_achieved=False,
                consensus_direction=None,
                consensus_score=0.0,
                agreeing_heads=[],
                disagreeing_heads=[],
                confidence_threshold=self.confidence_threshold,
                min_agreeing_heads=self.min_agreeing_heads,
                total_heads=0
            )
    
    def _calculate_weighted_consensus_score(self, agreeing_heads: List[ModelHeadResult]) -> float:
        """Calculate weighted consensus score"""
        try:
            total_weighted_score = 0.0
            total_weight = 0.0
            
            for head_result in agreeing_heads:
                weight = self.head_weights.get(head_result.head_type, 0.25)
                weighted_score = head_result.probability * head_result.confidence * weight
                
                total_weighted_score += weighted_score
                total_weight += weight
            
            if total_weight > 0:
                consensus_score = total_weighted_score / total_weight
                return min(1.0, consensus_score)
            else:
                return 0.0
                
        except Exception as e:
            logger.error(f"Weighted consensus score calculation error: {e}")
            return 0.0
    
    async def get_consensus_summary(self, model_results: List[ModelHeadResult]) -> Dict[str, Any]:
        """Get detailed consensus summary"""
        try:
            consensus = await self.check_consensus(model_results)
            
            # Create detailed summary
            summary = {
                'consensus_achieved': consensus.consensus_achieved,
                'consensus_direction': consensus.consensus_direction.value if consensus.consensus_direction else None,
                'consensus_score': consensus.consensus_score,
                'agreeing_heads_count': len(consensus.agreeing_heads),
                'disagreeing_heads_count': len(consensus.disagreeing_heads),
                'min_agreeing_heads': consensus.min_agreeing_heads,
                'total_heads': consensus.total_heads,
                'confidence_threshold': consensus.confidence_threshold,
                'head_results': []
            }
            
            # Add individual head results
            for result in model_results:
                head_summary = {
                    'head_type': result.head_type.value,
                    'direction': result.direction.value,
                    'probability': result.probability,
                    'confidence': result.confidence,
                    'meets_thresholds': (result.probability >= self.min_probability_threshold and 
                                       result.confidence >= self.confidence_threshold),
                    'reasoning': result.reasoning
                }
                summary['head_results'].append(head_summary)
            
            return summary
            
        except Exception as e:
            logger.error(f"Consensus summary error: {e}")
            return {
                'consensus_achieved': False,
                'error': str(e)
            }
    
    def update_consensus_config(self, 
                              min_agreeing_heads: Optional[int] = None,
                              min_probability_threshold: Optional[float] = None,
                              confidence_threshold: Optional[float] = None,
                              head_weights: Optional[Dict[ModelHead, float]] = None):
        """Update consensus configuration"""
        try:
            if min_agreeing_heads is not None:
                self.min_agreeing_heads = min_agreeing_heads
                logger.info(f"Updated min_agreeing_heads to {min_agreeing_heads}")
            
            if min_probability_threshold is not None:
                self.min_probability_threshold = min_probability_threshold
                logger.info(f"Updated min_probability_threshold to {min_probability_threshold}")
            
            if confidence_threshold is not None:
                self.confidence_threshold = confidence_threshold
                logger.info(f"Updated confidence_threshold to {confidence_threshold}")
            
            if head_weights is not None:
                self.head_weights.update(head_weights)
                logger.info(f"Updated head_weights: {head_weights}")
                
        except Exception as e:
            logger.error(f"Consensus config update error: {e}")
    
    def get_consensus_config(self) -> Dict[str, Any]:
        """Get current consensus configuration"""
        return {
            'min_agreeing_heads': self.min_agreeing_heads,
            'min_probability_threshold': self.min_probability_threshold,
            'confidence_threshold': self.confidence_threshold,
            'head_weights': {k.value: v for k, v in self.head_weights.items()},
            'direction_agreement_required': self.direction_agreement_required
        }
