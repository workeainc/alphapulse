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
    ICT_CONCEPTS = "ict_concepts"  # ICT Concepts Analysis
    WYCKOFF = "wyckoff"  # Wyckoff Methodology
    HARMONIC = "harmonic"  # Harmonic Patterns
    MARKET_STRUCTURE = "market_structure"  # Enhanced Market Structure
    CRYPTO_METRICS = "crypto_metrics"  # Crypto-Specific Metrics

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
    
    # NEW: Separate probability and confidence (per specification)
    consensus_probability: float = 0.0  # Weighted avg of probabilities
    consensus_confidence: float = 0.0   # Base + agreement bonus + strength bonus
    
    # For backwards compatibility
    consensus_score: float = 0.0  # Combined metric (deprecated but kept for compatibility)
    
    agreeing_heads: List[ModelHead] = None
    disagreeing_heads: List[ModelHead] = None
    confidence_threshold: float = 0.0
    min_agreeing_heads: int = 0
    total_heads: int = 0
    
    def __post_init__(self):
        if self.agreeing_heads is None:
            self.agreeing_heads = []
        if self.disagreeing_heads is None:
            self.disagreeing_heads = []

class ConsensusManager:
    """Manages consensus mechanism for AI model heads with adaptive thresholds"""
    
    def __init__(self):
        # Base thresholds (calibrated for 2-5 signals/day from 1000 pairs)
        self.base_min_agreeing_heads = 4  # Minimum 4 heads must agree (44% of 9 heads) - PER SPECIFICATION
        self.base_min_probability_threshold = 0.6  # Minimum probability threshold
        self.base_confidence_threshold = 0.70  # Minimum confidence threshold - PER SPECIFICATION
        
        # Current thresholds (adaptive - will change at runtime)
        self.min_agreeing_heads = self.base_min_agreeing_heads
        self.min_probability_threshold = self.base_min_probability_threshold
        self.confidence_threshold = self.base_confidence_threshold
        
        self.direction_agreement_required = True  # All agreeing heads must have same direction
        
        # Head weights for consensus calculation (9 heads total, balanced distribution)
        self.head_weights = {
            ModelHead.HEAD_A: 0.13,  # Technical Analysis
            ModelHead.HEAD_B: 0.09,  # Sentiment Analysis
            ModelHead.HEAD_C: 0.13,  # Volume Analysis
            ModelHead.HEAD_D: 0.09,  # Rule-based Analysis
            ModelHead.ICT_CONCEPTS: 0.13,  # ICT Concepts
            ModelHead.WYCKOFF: 0.13,  # Wyckoff Methodology
            ModelHead.HARMONIC: 0.09,  # Harmonic Patterns
            ModelHead.MARKET_STRUCTURE: 0.09,  # Enhanced Market Structure
            ModelHead.CRYPTO_METRICS: 0.12  # Crypto-Specific Metrics (NEW)
        }
        
        logger.info(f"✅ Consensus Manager initialized (base: {self.base_min_agreeing_heads}/{len(self.head_weights)} heads @ {self.base_confidence_threshold:.2f} confidence)")
    
    def update_adaptive_thresholds(
        self,
        min_agreeing_heads: Optional[int] = None,
        confidence_threshold: Optional[float] = None,
        min_probability_threshold: Optional[float] = None
    ):
        """
        Update thresholds dynamically (called by Adaptive Signal Controller)
        
        Args:
            min_agreeing_heads: New minimum agreeing heads
            confidence_threshold: New confidence threshold
            min_probability_threshold: New probability threshold
        """
        if min_agreeing_heads is not None:
            self.min_agreeing_heads = min_agreeing_heads
            logger.debug(f"Updated min_agreeing_heads to {min_agreeing_heads}")
        
        if confidence_threshold is not None:
            self.confidence_threshold = confidence_threshold
            logger.debug(f"Updated confidence_threshold to {confidence_threshold:.3f}")
        
        if min_probability_threshold is not None:
            self.min_probability_threshold = min_probability_threshold
            logger.debug(f"Updated min_probability_threshold to {min_probability_threshold:.3f}")
    
    async def check_consensus(self, model_results: List[ModelHeadResult]) -> ConsensusResult:
        """Check if model heads reach consensus"""
        try:
            if len(model_results) < 4:
                logger.warning("Insufficient model heads for consensus check")
                return ConsensusResult(
                    consensus_achieved=False,
                    consensus_direction=None,
                    consensus_probability=0.0,
                    consensus_confidence=0.0,
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
                    consensus_probability=0.0,
                    consensus_confidence=0.0,
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
                # Calculate NEW consensus metrics (per specification)
                consensus_probability = self._calculate_consensus_probability(agreeing_heads)
                consensus_confidence = self._calculate_consensus_confidence(agreeing_heads)
                
                # Calculate OLD consensus score (for backwards compatibility)
                consensus_score = self._calculate_weighted_consensus_score(agreeing_heads)
                
                # CRITICAL: Check minimum consensus confidence threshold (0.65)
                if consensus_confidence < 0.65:
                    logger.info(f"⚠️ Consensus achieved but confidence too low: {consensus_confidence:.3f} < 0.65 "
                               f"({consensus_direction.value}, {len(agreeing_heads)} heads)")
                    consensus_achieved = False
                    
                    # Get disagreeing heads
                    disagreeing_heads = [r.head_type for r in model_results]
                    
                    return ConsensusResult(
                        consensus_achieved=False,
                        consensus_direction=None,
                        consensus_probability=0.0,
                        consensus_confidence=consensus_confidence,
                        consensus_score=0.0,
                        agreeing_heads=[],
                        disagreeing_heads=disagreeing_heads,
                        confidence_threshold=self.confidence_threshold,
                        min_agreeing_heads=self.min_agreeing_heads,
                        total_heads=len(model_results)
                    )
                
                # Get disagreeing heads
                disagreeing_heads = [r.head_type for r in model_results 
                                   if r.head_type not in [h.head_type for h in agreeing_heads]]
                
                logger.info(f"✅ Consensus achieved: {consensus_direction.value} "
                           f"(probability: {consensus_probability:.3f}, confidence: {consensus_confidence:.3f}, "
                           f"agreeing: {len(agreeing_heads)}/{len(model_results)} heads)")
                
                return ConsensusResult(
                    consensus_achieved=True,
                    consensus_direction=consensus_direction,
                    consensus_probability=consensus_probability,
                    consensus_confidence=consensus_confidence,
                    consensus_score=consensus_score,  # Backwards compatibility
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
                    consensus_probability=0.0,
                    consensus_confidence=0.0,
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
                consensus_probability=0.0,
                consensus_confidence=0.0,
                consensus_score=0.0,
                agreeing_heads=[],
                disagreeing_heads=[],
                confidence_threshold=self.confidence_threshold,
                min_agreeing_heads=self.min_agreeing_heads,
                total_heads=0
            )
    
    def _calculate_weighted_consensus_score(self, agreeing_heads: List[ModelHeadResult]) -> float:
        """Calculate weighted consensus score (DEPRECATED - kept for backwards compatibility)"""
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
    
    def _calculate_consensus_probability(self, agreeing_heads: List[ModelHeadResult]) -> float:
        """
        Calculate consensus probability as weighted average of probabilities
        
        Per specification: Weighted average of probabilities (weighted by head importance)
        Example: (0.75×0.13 + 0.78×0.13 + 0.88×0.13 + 0.90×0.13 + 0.83×0.12) / (0.13+0.13+0.13+0.13+0.12)
        """
        try:
            total_weighted_probability = 0.0
            total_weight = 0.0
            
            for head_result in agreeing_heads:
                weight = self.head_weights.get(head_result.head_type, 0.25)
                # Only weight the probability, NOT probability * confidence
                total_weighted_probability += head_result.probability * weight
                total_weight += weight
            
            if total_weight > 0:
                consensus_probability = total_weighted_probability / total_weight
                return min(1.0, consensus_probability)
            else:
                return 0.0
                
        except Exception as e:
            logger.error(f"Consensus probability calculation error: {e}")
            return 0.0
    
    def _calculate_consensus_confidence(self, agreeing_heads: List[ModelHeadResult]) -> float:
        """
        Calculate consensus confidence with agreement and strength bonuses
        
        Per specification:
        Consensus Confidence = Base Confidence + Agreement Bonus + Strength Bonus
        
        Base Confidence: Average confidence of agreeing heads
        
        Agreement Bonus:
        - 4 heads: +0.00
        - 5 heads: +0.03
        - 6 heads: +0.06
        - 7 heads: +0.09
        - 8 heads: +0.12
        - 9 heads: +0.15
        
        Strength Bonus:
        - Avg probability 0.85+: +0.08
        - Avg probability 0.75-0.85: +0.05
        - Avg probability 0.60-0.75: +0.02
        """
        try:
            if not agreeing_heads:
                return 0.0
            
            # 1. Base Confidence: Average confidence of agreeing heads
            base_confidence = sum(h.confidence for h in agreeing_heads) / len(agreeing_heads)
            
            # 2. Agreement Bonus: More heads agreeing = more confidence
            num_agreeing = len(agreeing_heads)
            agreement_bonus_map = {
                4: 0.00,
                5: 0.03,
                6: 0.06,
                7: 0.09,
                8: 0.12,
                9: 0.15
            }
            agreement_bonus = agreement_bonus_map.get(num_agreeing, 0.00)
            
            # 3. Strength Bonus: Higher average probability = more confidence
            avg_probability = sum(h.probability for h in agreeing_heads) / len(agreeing_heads)
            if avg_probability >= 0.85:
                strength_bonus = 0.08
            elif avg_probability >= 0.75:
                strength_bonus = 0.05
            elif avg_probability >= 0.60:
                strength_bonus = 0.02
            else:
                strength_bonus = 0.00
            
            # Final Consensus Confidence
            consensus_confidence = base_confidence + agreement_bonus + strength_bonus
            
            logger.debug(f"Consensus confidence breakdown: base={base_confidence:.3f}, "
                        f"agreement_bonus={agreement_bonus:.3f} ({num_agreeing} heads), "
                        f"strength_bonus={strength_bonus:.3f} (avg_prob={avg_probability:.3f}), "
                        f"final={consensus_confidence:.3f}")
            
            return min(1.0, consensus_confidence)
            
        except Exception as e:
            logger.error(f"Consensus confidence calculation error: {e}")
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
