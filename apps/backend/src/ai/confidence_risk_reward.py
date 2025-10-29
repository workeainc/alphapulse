"""
Confidence-Based Risk-Reward Calculator
Implements dynamic risk-reward ratio scaling based on consensus confidence

Risk-Reward Ratios (per specification):
- Very High Confidence (0.85-0.95): 3:1 R:R
- High Confidence (0.75-0.85): 2.5:1 R:R
- Medium Confidence (0.65-0.75): 2:1 R:R

Rationale:
Higher confidence signals have higher win rates, so we can afford to aim for bigger targets.
Lower confidence needs smaller targets to maintain profitability.
"""

import logging
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


@dataclass
class RiskRewardResult:
    """Result of risk-reward calculation"""
    take_profit: float
    risk_reward_ratio: float
    confidence_band: str
    risk_amount: float
    reward_amount: float
    reasoning: str
    metadata: Dict


class ConfidenceBasedRiskReward:
    """
    Calculates take profit levels based on consensus confidence
    
    Uses dynamic risk-reward ratios that scale with confidence:
    - Very High (0.85+): 3:1 R:R
    - High (0.75+): 2.5:1 R:R
    - Medium (0.65+): 2:1 R:R
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize confidence-based risk-reward calculator
        
        Args:
            config: Configuration dictionary with optional overrides
        """
        self.config = config or {}
        
        # Risk-reward ratios by confidence band (per specification)
        self.rr_ratios = {
            'very_high': {
                'ratio': 3.0,
                'min_confidence': 0.85,
                'max_confidence': 0.95,
                'description': 'Very High Confidence - Aim for maximum targets'
            },
            'high': {
                'ratio': 2.5,
                'min_confidence': 0.75,
                'max_confidence': 0.85,
                'description': 'High Confidence - Balanced risk-reward'
            },
            'medium': {
                'ratio': 2.0,
                'min_confidence': 0.65,
                'max_confidence': 0.75,
                'description': 'Medium Confidence - Conservative targets'
            }
        }
        
        # Safety limits
        self.min_rr_ratio = self.config.get('min_rr_ratio', 1.5)  # Minimum 1.5:1
        self.max_rr_ratio = self.config.get('max_rr_ratio', 5.0)  # Maximum 5:1
        
        logger.info("✅ Confidence-Based Risk-Reward Calculator initialized")
    
    def calculate_take_profit(
        self,
        entry_price: float,
        stop_loss: float,
        consensus_confidence: float,
        direction: str,
        swing_high: Optional[float] = None,
        swing_low: Optional[float] = None
    ) -> RiskRewardResult:
        """
        Calculate take profit based on consensus confidence
        
        Args:
            entry_price: Entry price for the trade
            stop_loss: Stop loss price
            consensus_confidence: Consensus confidence from ConsensusManager (0-1)
            direction: Trade direction ('LONG' or 'SHORT')
            swing_high: Optional recent swing high (for LONG exits)
            swing_low: Optional recent swing low (for SHORT exits)
        
        Returns:
            RiskRewardResult with take profit and metadata
        """
        try:
            # Determine confidence band and R:R ratio
            confidence_band, rr_ratio = self._determine_rr_ratio(consensus_confidence)
            
            if confidence_band == 'insufficient':
                logger.warning(f"Insufficient confidence: {consensus_confidence:.3f} < 0.65")
                return RiskRewardResult(
                    take_profit=0.0,
                    risk_reward_ratio=0.0,
                    confidence_band=confidence_band,
                    risk_amount=0.0,
                    reward_amount=0.0,
                    reasoning="Confidence below minimum threshold (0.65)",
                    metadata={'confidence': consensus_confidence}
                )
            
            # Calculate risk amount
            risk_amount = abs(entry_price - stop_loss)
            
            # Calculate reward amount based on R:R ratio
            reward_amount = risk_amount * rr_ratio
            
            # Calculate take profit price
            if direction.upper() == 'LONG':
                take_profit = entry_price + reward_amount
                
                # Check if swing high is closer (don't target beyond resistance)
                if swing_high and swing_high < take_profit:
                    # Adjust take profit to swing high
                    take_profit = swing_high * 0.995  # 0.5% before resistance
                    reward_amount = take_profit - entry_price
                    actual_rr = reward_amount / risk_amount if risk_amount > 0 else rr_ratio
                    logger.debug(f"Adjusted TP to swing high: ${take_profit:.2f} (R:R {actual_rr:.2f}:1)")
                
            elif direction.upper() == 'SHORT':
                take_profit = entry_price - reward_amount
                
                # Check if swing low is closer (don't target beyond support)
                if swing_low and swing_low > take_profit:
                    # Adjust take profit to swing low
                    take_profit = swing_low * 1.005  # 0.5% after support
                    reward_amount = entry_price - take_profit
                    actual_rr = reward_amount / risk_amount if risk_amount > 0 else rr_ratio
                    logger.debug(f"Adjusted TP to swing low: ${take_profit:.2f} (R:R {actual_rr:.2f}:1)")
            else:
                raise ValueError(f"Invalid direction: {direction}")
            
            # Calculate actual R:R ratio
            actual_rr_ratio = reward_amount / risk_amount if risk_amount > 0 else 0.0
            
            # Apply safety limits
            if actual_rr_ratio < self.min_rr_ratio:
                logger.warning(f"R:R ratio {actual_rr_ratio:.2f} below minimum {self.min_rr_ratio}")
                # Recalculate with minimum R:R
                reward_amount = risk_amount * self.min_rr_ratio
                if direction.upper() == 'LONG':
                    take_profit = entry_price + reward_amount
                else:
                    take_profit = entry_price - reward_amount
                actual_rr_ratio = self.min_rr_ratio
            
            if actual_rr_ratio > self.max_rr_ratio:
                logger.warning(f"R:R ratio {actual_rr_ratio:.2f} above maximum {self.max_rr_ratio}")
                # Recalculate with maximum R:R
                reward_amount = risk_amount * self.max_rr_ratio
                if direction.upper() == 'LONG':
                    take_profit = entry_price + reward_amount
                else:
                    take_profit = entry_price - reward_amount
                actual_rr_ratio = self.max_rr_ratio
            
            # Build reasoning
            reasoning_parts = [
                f"Confidence: {consensus_confidence:.3f}",
                f"Band: {confidence_band.upper()}",
                f"Target R:R: {rr_ratio:.1f}:1",
                f"Actual R:R: {actual_rr_ratio:.2f}:1",
                f"Risk: ${risk_amount:.2f}",
                f"Reward: ${reward_amount:.2f}",
                f"TP: ${take_profit:.2f}"
            ]
            
            if swing_high and direction.upper() == 'LONG' and take_profit >= swing_high * 0.995:
                reasoning_parts.append("Adjusted to swing high")
            if swing_low and direction.upper() == 'SHORT' and take_profit <= swing_low * 1.005:
                reasoning_parts.append("Adjusted to swing low")
            
            reasoning = "; ".join(reasoning_parts)
            
            # Create metadata
            band_info = self.rr_ratios.get(confidence_band, {})
            metadata = {
                'confidence': consensus_confidence,
                'confidence_band': confidence_band,
                'target_rr_ratio': rr_ratio,
                'actual_rr_ratio': actual_rr_ratio,
                'band_description': band_info.get('description', ''),
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'direction': direction,
                'swing_high': swing_high,
                'swing_low': swing_low,
                'adjusted_to_structure': bool(
                    (swing_high and direction.upper() == 'LONG' and take_profit >= swing_high * 0.995) or
                    (swing_low and direction.upper() == 'SHORT' and take_profit <= swing_low * 1.005)
                )
            }
            
            logger.info(f"✅ Risk-Reward: {confidence_band} | "
                       f"R:R {actual_rr_ratio:.2f}:1 | "
                       f"TP: ${take_profit:.2f}")
            
            return RiskRewardResult(
                take_profit=take_profit,
                risk_reward_ratio=actual_rr_ratio,
                confidence_band=confidence_band,
                risk_amount=risk_amount,
                reward_amount=reward_amount,
                reasoning=reasoning,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"❌ Risk-reward calculation failed: {e}")
            return RiskRewardResult(
                take_profit=0.0,
                risk_reward_ratio=0.0,
                confidence_band='error',
                risk_amount=0.0,
                reward_amount=0.0,
                reasoning=f"Error: {str(e)}",
                metadata={'error': str(e)}
            )
    
    def _determine_rr_ratio(self, consensus_confidence: float) -> Tuple[str, float]:
        """Determine confidence band and corresponding R:R ratio"""
        if consensus_confidence >= 0.85:
            return 'very_high', 3.0
        elif consensus_confidence >= 0.75:
            return 'high', 2.5
        elif consensus_confidence >= 0.65:
            return 'medium', 2.0
        else:
            return 'insufficient', 0.0
    
    def calculate_multiple_targets(
        self,
        entry_price: float,
        stop_loss: float,
        consensus_confidence: float,
        direction: str,
        num_targets: int = 3
    ) -> Dict:
        """
        Calculate multiple take profit targets
        
        Args:
            entry_price: Entry price
            stop_loss: Stop loss price
            consensus_confidence: Consensus confidence
            direction: Trade direction
            num_targets: Number of targets to calculate
        
        Returns:
            Dict with multiple targets
        """
        try:
            # Get base R:R ratio
            confidence_band, base_rr_ratio = self._determine_rr_ratio(consensus_confidence)
            
            if confidence_band == 'insufficient':
                return {'targets': [], 'error': 'Insufficient confidence'}
            
            risk_amount = abs(entry_price - stop_loss)
            
            # Calculate progressive targets
            targets = []
            for i in range(1, num_targets + 1):
                # Scale R:R progressively
                target_rr = base_rr_ratio * i
                reward = risk_amount * target_rr
                
                if direction.upper() == 'LONG':
                    tp = entry_price + reward
                else:
                    tp = entry_price - reward
                
                targets.append({
                    'target_number': i,
                    'take_profit': tp,
                    'risk_reward_ratio': target_rr,
                    'distance_pct': (reward / entry_price) * 100
                })
            
            return {
                'confidence_band': confidence_band,
                'base_rr_ratio': base_rr_ratio,
                'targets': targets,
                'risk_amount': risk_amount
            }
            
        except Exception as e:
            logger.error(f"❌ Multiple targets calculation failed: {e}")
            return {'targets': [], 'error': str(e)}
    
    def get_rr_ratio_info(self, consensus_confidence: float) -> Dict:
        """Get detailed information about the R:R ratio for a confidence level"""
        confidence_band, rr_ratio = self._determine_rr_ratio(consensus_confidence)
        
        if confidence_band == 'insufficient':
            return {
                'band': 'insufficient',
                'rr_ratio': 0.0,
                'description': 'Below minimum confidence threshold'
            }
        
        band_info = self.rr_ratios[confidence_band]
        return {
            'band': confidence_band,
            'rr_ratio': rr_ratio,
            'description': band_info['description'],
            'confidence_range': f"{band_info['min_confidence']:.2f}-{band_info['max_confidence']:.2f}"
        }

