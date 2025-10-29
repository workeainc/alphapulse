"""
Confidence-Based Position Sizing Module
Implements the specification for position sizing based on consensus confidence

Position Sizing Bands (per specification):
- Very High Confidence (0.85-0.95): 2.0-3.0% of capital
- High Confidence (0.75-0.85): 1.5-2.5% of capital
- Medium Confidence (0.65-0.75): 1.0-1.5% of capital
- Below 0.65: NO TRADE (filtered by consensus gate)

Expected Win Rates:
- Very High Confidence: 75-85%
- High Confidence: 65-75%
- Medium Confidence: 55-65%
"""

import logging
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ConfidenceBand(Enum):
    """Confidence band classifications"""
    VERY_HIGH = "very_high"  # 0.85-0.95
    HIGH = "high"  # 0.75-0.85
    MEDIUM = "medium"  # 0.65-0.75
    INSUFFICIENT = "insufficient"  # < 0.65


@dataclass
class PositionSizingResult:
    """Result of position sizing calculation"""
    position_size_pct: float  # Position size as % of capital
    position_size_usd: float  # Position size in USD
    confidence_band: ConfidenceBand
    expected_win_rate: float  # Expected win rate (0-1)
    reasoning: str
    risk_amount: float  # Risk amount in USD
    metadata: Dict


class ConfidenceBasedPositionSizing:
    """
    Implements confidence-based position sizing per specification
    
    Position size scales with consensus confidence in specific bands
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize confidence-based position sizing
        
        Args:
            config: Configuration dictionary with optional overrides
        """
        self.config = config or {}
        
        # Position sizing bands (per specification)
        self.sizing_bands = {
            ConfidenceBand.VERY_HIGH: {
                'min_pct': 0.020,  # 2.0%
                'max_pct': 0.030,  # 3.0%
                'min_confidence': 0.85,
                'max_confidence': 0.95,
                'expected_win_rate': 0.80,  # 80% (75-85% range)
                'description': 'Very High Confidence - Perfect storm setups'
            },
            ConfidenceBand.HIGH: {
                'min_pct': 0.015,  # 1.5%
                'max_pct': 0.025,  # 2.5%
                'min_confidence': 0.75,
                'max_confidence': 0.85,
                'expected_win_rate': 0.70,  # 70% (65-75% range)
                'description': 'High Confidence - Strong setups with 5+ heads'
            },
            ConfidenceBand.MEDIUM: {
                'min_pct': 0.010,  # 1.0%
                'max_pct': 0.015,  # 1.5%
                'min_confidence': 0.65,
                'max_confidence': 0.75,
                'expected_win_rate': 0.60,  # 60% (55-65% range)
                'description': 'Medium Confidence - Minimum viable setups'
            }
        }
        
        # Safety limits
        self.max_position_size_pct = self.config.get('max_position_size_pct', 0.10)  # 10% max
        self.min_position_size_usd = self.config.get('min_position_size_usd', 10.0)  # $10 min
        
        logger.info("✅ Confidence-Based Position Sizing initialized")
    
    def calculate_position_size(
        self,
        consensus_confidence: float,
        available_capital: float,
        entry_price: float,
        stop_loss: float,
        additional_adjustments: Dict = None
    ) -> PositionSizingResult:
        """
        Calculate position size based on consensus confidence
        
        Args:
            consensus_confidence: Consensus confidence from ConsensusManager (0-1)
            available_capital: Available capital in USD
            entry_price: Entry price for the trade
            stop_loss: Stop loss price
            additional_adjustments: Optional dict with risk adjustments
        
        Returns:
            PositionSizingResult with position size and metadata
        """
        try:
            # Determine confidence band
            confidence_band = self._determine_confidence_band(consensus_confidence)
            
            # Get sizing parameters for this band
            if confidence_band == ConfidenceBand.INSUFFICIENT:
                logger.warning(f"Insufficient confidence: {consensus_confidence:.3f} < 0.65")
                return PositionSizingResult(
                    position_size_pct=0.0,
                    position_size_usd=0.0,
                    confidence_band=confidence_band,
                    expected_win_rate=0.0,
                    reasoning="Confidence below minimum threshold (0.65)",
                    risk_amount=0.0,
                    metadata={'confidence': consensus_confidence}
                )
            
            band_params = self.sizing_bands[confidence_band]
            
            # Calculate position size percentage within band
            # Scale linearly within the confidence range
            confidence_range = band_params['max_confidence'] - band_params['min_confidence']
            confidence_position = (consensus_confidence - band_params['min_confidence']) / confidence_range
            confidence_position = max(0.0, min(1.0, confidence_position))  # Clamp to [0, 1]
            
            pct_range = band_params['max_pct'] - band_params['min_pct']
            position_size_pct = band_params['min_pct'] + (pct_range * confidence_position)
            
            # Apply additional adjustments if provided
            adjustment_factor = 1.0
            adjustment_reasons = []
            
            if additional_adjustments:
                # Liquidation risk adjustment
                if additional_adjustments.get('liquidation_risk_high'):
                    adjustment_factor *= 0.5  # Reduce by 50%
                    adjustment_reasons.append("Liquidation risk detected (-50%)")
                
                # Extreme leverage adjustment
                if additional_adjustments.get('extreme_leverage_with_position'):
                    adjustment_factor *= 0.5  # Reduce by 50%
                    adjustment_reasons.append("Overleveraged market, going WITH leverage (-50%)")
                
                # Market regime adjustment
                regime_multiplier = additional_adjustments.get('regime_multiplier', 1.0)
                adjustment_factor *= regime_multiplier
                if regime_multiplier != 1.0:
                    adjustment_reasons.append(f"Market regime adjustment ({regime_multiplier:.2f}x)")
                
                # Volatility adjustment
                volatility_multiplier = additional_adjustments.get('volatility_multiplier', 1.0)
                adjustment_factor *= volatility_multiplier
                if volatility_multiplier != 1.0:
                    adjustment_reasons.append(f"Volatility adjustment ({volatility_multiplier:.2f}x)")
            
            # Apply adjustments
            position_size_pct *= adjustment_factor
            
            # Apply safety limits
            position_size_pct = min(position_size_pct, self.max_position_size_pct)
            
            # Calculate position size in USD
            position_size_usd = available_capital * position_size_pct
            
            # Ensure minimum position size
            if position_size_usd < self.min_position_size_usd:
                position_size_usd = self.min_position_size_usd
                position_size_pct = position_size_usd / available_capital
            
            # Calculate risk amount
            risk_per_share = abs(entry_price - stop_loss)
            position_quantity = position_size_usd / entry_price
            risk_amount = position_quantity * risk_per_share
            risk_pct = (risk_amount / available_capital) * 100
            
            # Build reasoning
            reasoning_parts = [
                f"Confidence: {consensus_confidence:.3f}",
                f"Band: {confidence_band.value.upper()}",
                f"Base size: {position_size_pct*100:.2f}% of capital",
                f"Expected win rate: {band_params['expected_win_rate']*100:.0f}%"
            ]
            
            if adjustment_reasons:
                reasoning_parts.extend(adjustment_reasons)
            
            reasoning_parts.append(f"Final size: ${position_size_usd:.2f} ({position_size_pct*100:.2f}%)")
            reasoning_parts.append(f"Risk: ${risk_amount:.2f} ({risk_pct:.2f}%)")
            
            reasoning = "; ".join(reasoning_parts)
            
            # Create metadata
            metadata = {
                'confidence': consensus_confidence,
                'confidence_band': confidence_band.value,
                'base_pct': band_params['min_pct'] + (pct_range * confidence_position),
                'adjustment_factor': adjustment_factor,
                'adjustments_applied': adjustment_reasons,
                'position_quantity': position_quantity,
                'risk_per_share': risk_per_share,
                'risk_pct': risk_pct,
                'band_description': band_params['description']
            }
            
            logger.info(f"✅ Position sizing: {confidence_band.value} | "
                       f"Size: ${position_size_usd:.2f} ({position_size_pct*100:.2f}%) | "
                       f"Risk: {risk_pct:.2f}%")
            
            return PositionSizingResult(
                position_size_pct=position_size_pct,
                position_size_usd=position_size_usd,
                confidence_band=confidence_band,
                expected_win_rate=band_params['expected_win_rate'],
                reasoning=reasoning,
                risk_amount=risk_amount,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"❌ Position sizing calculation failed: {e}")
            return PositionSizingResult(
                position_size_pct=0.0,
                position_size_usd=0.0,
                confidence_band=ConfidenceBand.INSUFFICIENT,
                expected_win_rate=0.0,
                reasoning=f"Error: {str(e)}",
                risk_amount=0.0,
                metadata={'error': str(e)}
            )
    
    def _determine_confidence_band(self, consensus_confidence: float) -> ConfidenceBand:
        """Determine which confidence band a value falls into"""
        if consensus_confidence >= 0.85:
            return ConfidenceBand.VERY_HIGH
        elif consensus_confidence >= 0.75:
            return ConfidenceBand.HIGH
        elif consensus_confidence >= 0.65:
            return ConfidenceBand.MEDIUM
        else:
            return ConfidenceBand.INSUFFICIENT
    
    def get_expected_win_rate(self, consensus_confidence: float) -> float:
        """Get expected win rate for a given confidence level"""
        confidence_band = self._determine_confidence_band(consensus_confidence)
        if confidence_band == ConfidenceBand.INSUFFICIENT:
            return 0.0
        return self.sizing_bands[confidence_band]['expected_win_rate']
    
    def get_band_info(self, consensus_confidence: float) -> Dict:
        """Get detailed information about the confidence band"""
        confidence_band = self._determine_confidence_band(consensus_confidence)
        if confidence_band == ConfidenceBand.INSUFFICIENT:
            return {
                'band': 'insufficient',
                'description': 'Below minimum confidence threshold',
                'position_size_range': '0%',
                'expected_win_rate': '0%'
            }
        
        band_params = self.sizing_bands[confidence_band]
        return {
            'band': confidence_band.value,
            'description': band_params['description'],
            'position_size_range': f"{band_params['min_pct']*100:.1f}%-{band_params['max_pct']*100:.1f}%",
            'expected_win_rate': f"{band_params['expected_win_rate']*100:.0f}%",
            'confidence_range': f"{band_params['min_confidence']:.2f}-{band_params['max_confidence']:.2f}"
        }

