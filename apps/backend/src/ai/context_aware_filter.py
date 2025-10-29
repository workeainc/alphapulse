"""
Context-Aware Market Filter for AlphaPulse
Adjusts signal requirements based on market regime and conditions
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import asyncio

logger = logging.getLogger(__name__)

class MarketRegime(Enum):
    """Market regime types"""
    TRENDING = "trending"
    RANGING = "ranging"
    VOLATILE = "volatile"
    ACCUMULATION = "accumulation"
    DISTRIBUTION = "distribution"
    BREAKOUT = "breakout"
    UNKNOWN = "unknown"

@dataclass
class RegimeRequirements:
    """Requirements for specific market regime"""
    regime: MarketRegime
    min_consensus_heads: int
    min_confidence: float
    priority_heads: List[str]  # Heads to prioritize in this regime
    reasoning: str

@dataclass
class FilterDecision:
    """Result from context-aware filtering"""
    should_generate_signal: bool
    regime: MarketRegime
    requirements_met: bool
    confidence_adjustment: float  # Multiplier: 0.8-1.2
    priority_score: float  # 0-1
    reasoning: str
    contributing_factors: Dict[str, Any]

class ContextAwareMarketFilter:
    """
    Context-aware filtering system that adapts requirements to market conditions
    
    Different markets need different confirmation:
    - TRENDING: Easier (need 3/9 heads, prioritize trend indicators)
    - RANGING: Normal (need 4/9 heads, prioritize mean reversion)
    - VOLATILE: Harder (need 5/9 heads, require more confirmation)
    - ACCUMULATION: Focus on Wyckoff + SMC (need specific heads)
    - BREAKOUT: Require volume confirmation
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize Context-Aware Market Filter"""
        self.config = config or {}
        
        # Regime-specific requirements
        self.regime_requirements = {
            MarketRegime.TRENDING: RegimeRequirements(
                regime=MarketRegime.TRENDING,
                min_consensus_heads=3,
                min_confidence=0.75,
                priority_heads=['technical', 'ict', 'market_structure'],
                reasoning="Trending markets: prioritize trend following"
            ),
            MarketRegime.RANGING: RegimeRequirements(
                regime=MarketRegime.RANGING,
                min_consensus_heads=4,
                min_confidence=0.78,
                priority_heads=['harmonic', 'technical', 'market_structure'],
                reasoning="Ranging markets: prioritize mean reversion patterns"
            ),
            MarketRegime.VOLATILE: RegimeRequirements(
                regime=MarketRegime.VOLATILE,
                min_consensus_heads=5,
                min_confidence=0.82,
                priority_heads=['technical', 'volume', 'wyckoff', 'market_structure'],
                reasoning="Volatile markets: require strong confirmation"
            ),
            MarketRegime.ACCUMULATION: RegimeRequirements(
                regime=MarketRegime.ACCUMULATION,
                min_consensus_heads=4,
                min_confidence=0.76,
                priority_heads=['wyckoff', 'volume', 'crypto_metrics'],
                reasoning="Accumulation: focus on Wyckoff + smart money flow"
            ),
            MarketRegime.DISTRIBUTION: RegimeRequirements(
                regime=MarketRegime.DISTRIBUTION,
                min_consensus_heads=4,
                min_confidence=0.76,
                priority_heads=['wyckoff', 'volume', 'crypto_metrics'],
                reasoning="Distribution: focus on Wyckoff + smart money flow"
            ),
            MarketRegime.BREAKOUT: RegimeRequirements(
                regime=MarketRegime.BREAKOUT,
                min_consensus_heads=4,
                min_confidence=0.80,
                priority_heads=['volume', 'technical', 'market_structure'],
                reasoning="Breakout: require volume confirmation"
            ),
            MarketRegime.UNKNOWN: RegimeRequirements(
                regime=MarketRegime.UNKNOWN,
                min_consensus_heads=4,
                min_confidence=0.78,
                priority_heads=['technical', 'volume'],
                reasoning="Unknown regime: use standard requirements"
            )
        }
        
        # Confidence adjustment factors by regime
        self.confidence_multipliers = {
            MarketRegime.TRENDING: 1.05,  # Slight boost
            MarketRegime.RANGING: 1.0,  # Normal
            MarketRegime.VOLATILE: 0.90,  # Penalty for uncertainty
            MarketRegime.ACCUMULATION: 1.02,  # Slight boost
            MarketRegime.DISTRIBUTION: 0.98,  # Slight penalty
            MarketRegime.BREAKOUT: 1.08,  # Good boost for confirmed breakouts
            MarketRegime.UNKNOWN: 0.95  # Slight penalty for uncertainty
        }
        
        # Statistics
        self.stats = {
            'total_evaluations': 0,
            'signals_passed': 0,
            'signals_blocked': 0,
            'regime_distribution': {regime: 0 for regime in MarketRegime},
            'avg_priority_score': 0.0
        }
        
        logger.info("✅ Context-Aware Market Filter initialized")
    
    async def evaluate_signal(
        self,
        market_data: Dict[str, Any],
        model_head_results: List[Dict[str, Any]],
        consensus_result: Dict[str, Any],
        analysis_results: Dict[str, Any]
    ) -> FilterDecision:
        """
        Evaluate if signal should be generated based on context
        
        Args:
            market_data: Current market data
            model_head_results: Results from all model heads
            consensus_result: Consensus mechanism result
            analysis_results: Additional analysis results
            
        Returns:
            FilterDecision with evaluation result
        """
        start_time = datetime.now()
        
        try:
            # Detect market regime
            regime = await self._detect_market_regime(market_data, analysis_results)
            
            # Get requirements for this regime
            requirements = self.regime_requirements[regime]
            
            # Check if consensus meets regime-specific requirements
            requirements_met = await self._check_regime_requirements(
                requirements,
                model_head_results,
                consensus_result
            )
            
            # Calculate priority score based on participating heads
            priority_score = await self._calculate_priority_score(
                requirements,
                model_head_results
            )
            
            # Calculate confidence adjustment
            confidence_adj = self._get_confidence_adjustment(
                regime,
                priority_score,
                model_head_results
            )
            
            # Build reasoning
            reasoning = self._build_reasoning(
                regime,
                requirements,
                requirements_met,
                priority_score,
                confidence_adj
            )
            
            # Final decision
            should_generate = requirements_met and priority_score >= 0.5
            
            # Update stats
            self._update_stats(regime, should_generate, priority_score)
            
            calc_time = (datetime.now() - start_time).total_seconds() * 1000
            
            return FilterDecision(
                should_generate_signal=should_generate,
                regime=regime,
                requirements_met=requirements_met,
                confidence_adjustment=confidence_adj,
                priority_score=priority_score,
                reasoning=reasoning,
                contributing_factors={
                    'regime': regime.value,
                    'min_consensus_heads': requirements.min_consensus_heads,
                    'min_confidence': requirements.min_confidence,
                    'priority_heads': requirements.priority_heads,
                    'calculation_time_ms': calc_time
                }
            )
            
        except Exception as e:
            logger.error(f"❌ Error in context-aware filtering: {e}")
            return FilterDecision(
                should_generate_signal=False,
                regime=MarketRegime.UNKNOWN,
                requirements_met=False,
                confidence_adjustment=1.0,
                priority_score=0.0,
                reasoning=f"Error: {str(e)}",
                contributing_factors={}
            )
    
    async def _detect_market_regime(
        self,
        market_data: Dict[str, Any],
        analysis_results: Dict[str, Any]
    ) -> MarketRegime:
        """Detect current market regime from market data"""
        try:
            # Check if regime already detected
            if 'market_regime' in analysis_results:
                regime_str = analysis_results['market_regime'].get('regime', 'unknown')
                try:
                    return MarketRegime(regime_str.lower())
                except ValueError:
                    pass
            
            # Fallback: detect from indicators
            df = analysis_results.get('dataframe')
            if df is None or len(df) < 50:
                return MarketRegime.UNKNOWN
            
            # Calculate key metrics
            close = df['close'].iloc[-20:]
            high = df['high'].iloc[-20:]
            low = df['low'].iloc[-20:]
            volume = df['volume'].iloc[-20:]
            
            # Volatility
            volatility = close.std() / close.mean() if close.mean() > 0 else 0
            
            # Trend strength (using simple linear regression)
            x = np.arange(len(close))
            slope, _ = np.polyfit(x, close.values, 1)
            trend_strength = abs(slope) / close.mean() if close.mean() > 0 else 0
            
            # Range detection (price staying within bounds)
            price_range = (high.max() - low.min())
            range_pct = (price_range / close.mean() * 100) if close.mean() > 0 else 0
            
            # Volume trend
            volume_increasing = volume.iloc[-5:].mean() > volume.iloc[-15:-5].mean()
            
            # Decision logic
            if volatility > 0.05:  # 5% volatility
                regime = MarketRegime.VOLATILE
            elif trend_strength > 0.003 and volume_increasing:  # Strong trend with volume
                regime = MarketRegime.TRENDING
            elif range_pct < 3.0:  # Price in tight range
                # Check volume behavior to determine accumulation vs ranging
                if volume_increasing:
                    regime = MarketRegime.ACCUMULATION
                else:
                    regime = MarketRegime.RANGING
            elif slope > 0 and volume_increasing:
                regime = MarketRegime.BREAKOUT
            else:
                regime = MarketRegime.RANGING
            
            logger.debug(f"Detected market regime: {regime.value}")
            return regime
            
        except Exception as e:
            logger.error(f"Error detecting market regime: {e}")
            return MarketRegime.UNKNOWN
    
    async def _check_regime_requirements(
        self,
        requirements: RegimeRequirements,
        model_head_results: List[Dict[str, Any]],
        consensus_result: Dict[str, Any]
    ) -> bool:
        """Check if consensus meets regime-specific requirements"""
        try:
            # Check consensus achievement
            if not consensus_result.get('consensus_achieved', False):
                return False
            
            # Check minimum number of agreeing heads
            agreeing_heads = consensus_result.get('agreeing_heads', [])
            if len(agreeing_heads) < requirements.min_consensus_heads:
                return False
            
            # Check if priority heads are participating
            priority_heads_participating = 0
            for head_result in model_head_results:
                head_type = head_result.get('head_type', '')
                if head_type in requirements.priority_heads:
                    if head_result.get('confidence', 0) >= requirements.min_confidence:
                        priority_heads_participating += 1
            
            # At least 1 priority head must participate
            if priority_heads_participating < 1:
                logger.debug(f"No priority heads participating for {requirements.regime.value}")
                return False
            
            # Check consensus confidence
            consensus_confidence = consensus_result.get('consensus_score', 0)
            if consensus_confidence < requirements.min_confidence:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking regime requirements: {e}")
            return False
    
    async def _calculate_priority_score(
        self,
        requirements: RegimeRequirements,
        model_head_results: List[Dict[str, Any]]
    ) -> float:
        """Calculate priority score based on participating heads"""
        try:
            priority_head_scores = []
            
            for head_result in model_head_results:
                head_type = head_result.get('head_type', '')
                if head_type in requirements.priority_heads:
                    confidence = head_result.get('confidence', 0)
                    probability = head_result.get('probability', 0.5)
                    
                    # Combined score
                    head_score = (confidence * 0.6 + (abs(probability - 0.5) * 2) * 0.4)
                    priority_head_scores.append(head_score)
            
            if not priority_head_scores:
                return 0.0
            
            # Average of priority head scores
            priority_score = np.mean(priority_head_scores)
            
            # Bonus for multiple priority heads
            if len(priority_head_scores) >= 2:
                priority_score *= 1.1
            if len(priority_head_scores) >= 3:
                priority_score *= 1.15
            
            return min(priority_score, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating priority score: {e}")
            return 0.0
    
    def _get_confidence_adjustment(
        self,
        regime: MarketRegime,
        priority_score: float,
        model_head_results: List[Dict[str, Any]]
    ) -> float:
        """Calculate confidence adjustment multiplier"""
        base_multiplier = self.confidence_multipliers.get(regime, 1.0)
        
        # Adjust based on priority score
        priority_adjustment = 1.0 + ((priority_score - 0.5) * 0.2)  # ±10%
        
        # Combine
        final_multiplier = base_multiplier * priority_adjustment
        
        # Clamp to reasonable bounds
        return np.clip(final_multiplier, 0.8, 1.2)
    
    def _build_reasoning(
        self,
        regime: MarketRegime,
        requirements: RegimeRequirements,
        requirements_met: bool,
        priority_score: float,
        confidence_adj: float
    ) -> str:
        """Build human-readable reasoning"""
        parts = []
        
        parts.append(f"Market Regime: {regime.value.upper()}")
        parts.append(requirements.reasoning)
        parts.append(f"Requirements {'MET' if requirements_met else 'NOT MET'}")
        parts.append(f"Priority Score: {priority_score:.3f}")
        parts.append(f"Confidence Adjustment: {confidence_adj:.2f}x")
        
        return "; ".join(parts)
    
    def _update_stats(self, regime: MarketRegime, passed: bool, priority_score: float):
        """Update filter statistics"""
        self.stats['total_evaluations'] += 1
        self.stats['regime_distribution'][regime] += 1
        
        if passed:
            self.stats['signals_passed'] += 1
        else:
            self.stats['signals_blocked'] += 1
        
        # Running average for priority score
        n = self.stats['total_evaluations']
        self.stats['avg_priority_score'] = (
            (self.stats['avg_priority_score'] * (n - 1) + priority_score) / n
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get filter statistics"""
        total = self.stats['total_evaluations']
        return {
            **self.stats,
            'pass_rate': self.stats['signals_passed'] / max(1, total),
            'block_rate': self.stats['signals_blocked'] / max(1, total),
            'regime_percentages': {
                regime.value: count / max(1, total)
                for regime, count in self.stats['regime_distribution'].items()
            }
        }

