"""
Signal Risk Enhancement Service
Integrates all Step 8 risk management components into signal generation:
- Liquidation cascade risk checks
- Extreme leverage checks
- Premium/discount zone entry logic
- Confidence-based position sizing
- Risk-reward scaling with confidence
- Complete signal object with all required fields
"""

import logging
from typing import Dict, Optional, Any
from datetime import datetime
from dataclasses import dataclass, asdict

from .confidence_position_sizing import ConfidenceBasedPositionSizing, PositionSizingResult
from .confidence_risk_reward import ConfidenceBasedRiskReward, RiskRewardResult

logger = logging.getLogger(__name__)


@dataclass
class EnhancedSignalResult:
    """Enhanced signal with all risk management and Step 8 components"""
    # Core signal fields
    symbol: str
    direction: str  # 'LONG' or 'SHORT'
    entry_price: float
    stop_loss: float
    take_profit: float
    
    # Consensus fields (NEW from Step 5)
    consensus_probability: float  # Weighted avg of probabilities
    consensus_confidence: float  # Base + agreement + strength bonuses
    agreeing_heads: int  # Number of heads that agreed
    total_heads: int  # Total heads analyzed
    
    # Position sizing fields (NEW from Step 8)
    position_size_pct: float  # Position size as % of capital
    position_size_usd: float  # Position size in USD
    confidence_band: str  # 'very_high', 'high', or 'medium'
    
    # Risk-reward fields (NEW from Step 8)
    risk_reward_ratio: float  # Actual R:R ratio
    risk_amount: float  # Risk in USD
    reward_amount: float  # Reward in USD
    expected_win_rate: float  # Expected win rate (0-1)
    
    # Risk management flags (NEW from Step 8)
    liquidation_risk_detected: bool
    extreme_leverage_detected: bool
    entry_zone_status: str  # 'discount', 'premium', or 'equilibrium'
    entry_strategy: str  # 'immediate', 'wait_pullback', or 'limit_order'
    
    # Additional metadata
    timestamp: datetime
    signal_quality: str  # 'excellent', 'good', or 'acceptable'
    reasoning: str
    metadata: Dict
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return asdict(self)


class SignalRiskEnhancement:
    """
    Comprehensive signal enhancement with all Step 8 risk management
    
    Integrates:
    1. Liquidation cascade risk checks
    2. Extreme leverage checks  
    3. Market regime validation
    4. Confidence-based position sizing
    5. Risk-reward scaling
    6. Premium/discount zone entry logic
    7. Complete signal object generation
    """
    
    def __init__(self, 
                 risk_manager = None,
                 derivatives_analyzer = None,
                 market_structure_engine = None,
                 config: Dict = None):
        """
        Initialize signal risk enhancement
        
        Args:
            risk_manager: RiskManager instance for liquidation checks
            derivatives_analyzer: DerivativesAnalyzer for leverage checks
            market_structure_engine: EnhancedMarketStructureEngine for zones
            config: Configuration dictionary
        """
        self.risk_manager = risk_manager
        self.derivatives_analyzer = derivatives_analyzer
        self.market_structure_engine = market_structure_engine
        self.config = config or {}
        
        # Initialize core modules
        self.position_sizer = ConfidenceBasedPositionSizing(config)
        self.rr_calculator = ConfidenceBasedRiskReward(config)
        
        # Default available capital
        self.default_capital = self.config.get('default_capital', 10000.0)  # $10k default
        
        logger.info("✅ Signal Risk Enhancement Service initialized")
    
    async def enhance_signal(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        stop_loss: float,
        consensus_result: Any,  # ConsensusResult from ConsensusManager
        market_data: Dict = None,
        available_capital: Optional[float] = None
    ) -> Optional[EnhancedSignalResult]:
        """
        Enhance signal with all Step 8 risk management components
        
        Args:
            symbol: Trading symbol
            direction: 'LONG' or 'SHORT'
            entry_price: Entry price
            stop_loss: Stop loss price
            consensus_result: ConsensusResult from ConsensusManager
            market_data: Optional market data for risk checks
            available_capital: Available capital (defaults to config value)
        
        Returns:
            EnhancedSignalResult or None if signal should be skipped
        """
        try:
            # Get consensus fields
            consensus_probability = consensus_result.consensus_probability
            consensus_confidence = consensus_result.consensus_confidence
            agreeing_heads = len(consensus_result.agreeing_heads)
            total_heads = consensus_result.total_heads
            
            logger.info(f"🔍 Enhancing signal for {symbol} {direction} | "
                       f"Confidence: {consensus_confidence:.3f} | "
                       f"Heads: {agreeing_heads}/{total_heads}")
            
            # Use provided capital or default
            capital = available_capital if available_capital else self.default_capital
            
            # STEP 1: Check Liquidation Cascade Risk
            liquidation_risk_detected = False
            liquidation_adjustment = {}
            
            if self.risk_manager and market_data:
                try:
                    liquidation_check = await self._check_liquidation_risk(
                        symbol, entry_price, market_data
                    )
                    liquidation_risk_detected = liquidation_check['risk_detected']
                    liquidation_adjustment = liquidation_check['adjustment']
                    
                    # Skip trade if too close to liquidation
                    if liquidation_check['skip_trade']:
                        logger.warning(f"⚠️ Signal SKIPPED: Too close to liquidation ({liquidation_check['reason']})")
                        return None
                        
                except Exception as e:
                    logger.warning(f"Liquidation check failed: {e}")
            
            # STEP 2: Check Extreme Leverage
            extreme_leverage_detected = False
            leverage_adjustment = {}
            
            if self.derivatives_analyzer and market_data:
                try:
                    leverage_check = await self._check_extreme_leverage(
                        symbol, direction, market_data
                    )
                    extreme_leverage_detected = leverage_check['leverage_detected']
                    leverage_adjustment = leverage_check['adjustment']
                    
                except Exception as e:
                    logger.warning(f"Leverage check failed: {e}")
            
            # STEP 3: Determine Entry Zone (Premium/Discount)
            entry_zone_info = {'zone': 'unknown', 'strategy': 'immediate'}
            
            if self.market_structure_engine and market_data:
                try:
                    entry_zone_info = await self._determine_entry_zone(
                        symbol, direction, entry_price, market_data
                    )
                    
                    # Wait for pullback if in bad zone
                    if entry_zone_info['strategy'] == 'wait_pullback':
                        logger.info(f"📍 Entry zone suggestion: Wait for pullback (currently in {entry_zone_info['zone']} zone)")
                        # Note: In production, this might trigger a pending order
                        # For now, we'll adjust entry price slightly
                        if direction == 'LONG':
                            entry_price *= 0.997  # 0.3% better entry
                        else:
                            entry_price *= 1.003
                        
                except Exception as e:
                    logger.warning(f"Entry zone check failed: {e}")
            
            # STEP 4: Calculate Position Size (with all adjustments)
            position_adjustments = {
                **liquidation_adjustment,
                **leverage_adjustment
            }
            
            position_result = self.position_sizer.calculate_position_size(
                consensus_confidence=consensus_confidence,
                available_capital=capital,
                entry_price=entry_price,
                stop_loss=stop_loss,
                additional_adjustments=position_adjustments
            )
            
            # Check if position size is zero (signal rejected)
            if position_result.position_size_usd == 0.0:
                logger.warning(f"⚠️ Signal REJECTED: {position_result.reasoning}")
                return None
            
            # STEP 5: Calculate Take Profit (with confidence-based R:R)
            # Get swing highs/lows if available
            swing_high = market_data.get('swing_high') if market_data else None
            swing_low = market_data.get('swing_low') if market_data else None
            
            rr_result = self.rr_calculator.calculate_take_profit(
                entry_price=entry_price,
                stop_loss=stop_loss,
                consensus_confidence=consensus_confidence,
                direction=direction,
                swing_high=swing_high,
                swing_low=swing_low
            )
            
            # STEP 6: Determine Signal Quality
            signal_quality = self._determine_signal_quality(
                consensus_confidence,
                agreeing_heads,
                total_heads,
                liquidation_risk_detected,
                extreme_leverage_detected
            )
            
            # STEP 7: Build Comprehensive Reasoning
            reasoning_parts = [
                f"Consensus: {agreeing_heads}/{total_heads} heads agree",
                f"Confidence: {consensus_confidence:.3f} ({position_result.confidence_band.value})",
                f"Position: ${position_result.position_size_usd:.2f} ({position_result.position_size_pct*100:.2f}%)",
                f"R:R: {rr_result.risk_reward_ratio:.2f}:1",
                f"Expected Win Rate: {position_result.expected_win_rate*100:.0f}%",
                f"Entry Zone: {entry_zone_info['zone']}",
            ]
            
            if liquidation_risk_detected:
                reasoning_parts.append("⚠️ Liquidation risk detected (size reduced)")
            if extreme_leverage_detected:
                reasoning_parts.append("⚠️ Extreme leverage detected (adjusted)")
            if entry_zone_info['strategy'] != 'immediate':
                reasoning_parts.append(f"Entry Strategy: {entry_zone_info['strategy']}")
            
            reasoning = "; ".join(reasoning_parts)
            
            # STEP 8: Create Enhanced Signal Result
            metadata = {
                'consensus_score': consensus_result.consensus_score,  # For backwards compatibility
                'position_sizing': position_result.metadata,
                'risk_reward': rr_result.metadata,
                'entry_zone': entry_zone_info,
                'risk_checks': {
                    'liquidation_risk': liquidation_risk_detected,
                    'extreme_leverage': extreme_leverage_detected
                },
                'adjustments_applied': {
                    **liquidation_adjustment,
                    **leverage_adjustment
                }
            }
            
            enhanced_signal = EnhancedSignalResult(
                symbol=symbol,
                direction=direction,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=rr_result.take_profit,
                consensus_probability=consensus_probability,
                consensus_confidence=consensus_confidence,
                agreeing_heads=agreeing_heads,
                total_heads=total_heads,
                position_size_pct=position_result.position_size_pct,
                position_size_usd=position_result.position_size_usd,
                confidence_band=position_result.confidence_band.value,
                risk_reward_ratio=rr_result.risk_reward_ratio,
                risk_amount=rr_result.risk_amount,
                reward_amount=rr_result.reward_amount,
                expected_win_rate=position_result.expected_win_rate,
                liquidation_risk_detected=liquidation_risk_detected,
                extreme_leverage_detected=extreme_leverage_detected,
                entry_zone_status=entry_zone_info['zone'],
                entry_strategy=entry_zone_info['strategy'],
                timestamp=datetime.now(),
                signal_quality=signal_quality,
                reasoning=reasoning,
                metadata=metadata
            )
            
            logger.info(f"✅ Enhanced signal generated: {signal_quality.upper()} quality | "
                       f"Size: ${position_result.position_size_usd:.2f} | "
                       f"R:R: {rr_result.risk_reward_ratio:.2f}:1")
            
            return enhanced_signal
            
        except Exception as e:
            logger.error(f"❌ Signal enhancement failed: {e}")
            return None
    
    async def _check_liquidation_risk(self, symbol: str, entry_price: float, market_data: Dict) -> Dict:
        """Check liquidation cascade risk"""
        try:
            # Get position size for simulation
            test_position_size = market_data.get('test_position_size', 1000.0)
            leverage = market_data.get('leverage', 1)
            
            # Simulate liquidation impact
            liquidation_result = await self.risk_manager.simulate_liquidation_impact(
                symbol, test_position_size, leverage
            )
            
            if 'error' in liquidation_result:
                return {'risk_detected': False, 'adjustment': {}, 'skip_trade': False, 'reason': ''}
            
            distance_to_liq = liquidation_result.get('distance_to_liquidation', 1.0)
            liq_probability = liquidation_result.get('liquidation_probability', 0.0)
            
            # Check thresholds
            skip_trade = distance_to_liq < 0.02  # < 2% from liquidation
            risk_detected = liq_probability > 0.3 or distance_to_liq < 0.05  # High risk
            
            adjustment = {}
            reason = ''
            
            if risk_detected:
                adjustment['liquidation_risk_high'] = True
                reason = f"Distance to liquidation: {distance_to_liq*100:.1f}%"
                logger.warning(f"⚠️ Liquidation risk: {reason}")
            
            return {
                'risk_detected': risk_detected,
                'adjustment': adjustment,
                'skip_trade': skip_trade,
                'reason': reason
            }
            
        except Exception as e:
            logger.error(f"Liquidation risk check error: {e}")
            return {'risk_detected': False, 'adjustment': {}, 'skip_trade': False, 'reason': ''}
    
    async def _check_extreme_leverage(self, symbol: str, direction: str, market_data: Dict) -> Dict:
        """Check extreme leverage in perpetual markets"""
        try:
            spot_price = market_data.get('spot_price', market_data.get('current_price'))
            perpetual_price = market_data.get('perpetual_price')
            
            if not spot_price or not perpetual_price:
                return {'leverage_detected': False, 'adjustment': {}}
            
            # Analyze derivatives
            derivatives_result = await self.derivatives_analyzer.analyze_derivatives(
                symbol=symbol,
                spot_price=spot_price,
                perpetual_price=perpetual_price
            )
            
            perpetual_premium = derivatives_result.perpetual_premium
            
            # Check if overleveraged (premium > 0.5%)
            is_overleveraged = abs(perpetual_premium.premium_pct) > 0.5
            
            adjustment = {}
            
            if is_overleveraged:
                # Check if going WITH or AGAINST leverage
                if perpetual_premium.premium_pct > 0.5:
                    # Positive premium = long-heavy
                    if direction.upper() == 'LONG':
                        # Going WITH leverage (crowded trade)
                        adjustment['extreme_leverage_with_position'] = True
                        logger.warning(f"⚠️ Overleveraged market (premium: {perpetual_premium.premium_pct:.2f}%), going WITH crowd")
                    else:
                        # Going AGAINST leverage (contrarian)
                        adjustment['extreme_leverage_against_position'] = True
                        logger.info(f"✅ Contrarian trade: Going SHORT in overleveraged long market")
                elif perpetual_premium.premium_pct < -0.3:
                    # Negative premium = short-heavy
                    if direction.upper() == 'SHORT':
                        adjustment['extreme_leverage_with_position'] = True
                        logger.warning(f"⚠️ Overleveraged market (discount: {perpetual_premium.premium_pct:.2f}%), going WITH crowd")
                    else:
                        adjustment['extreme_leverage_against_position'] = True
                        logger.info(f"✅ Contrarian trade: Going LONG in overleveraged short market")
            
            return {
                'leverage_detected': is_overleveraged,
                'adjustment': adjustment
            }
            
        except Exception as e:
            logger.error(f"Leverage check error: {e}")
            return {'leverage_detected': False, 'adjustment': {}}
    
    async def _determine_entry_zone(self, symbol: str, direction: str, entry_price: float, market_data: Dict) -> Dict:
        """Determine if price is in premium or discount zone"""
        try:
            # Get dataframe if available
            df = market_data.get('dataframe')
            
            if df is None or len(df) < 50:
                return {'zone': 'unknown', 'strategy': 'immediate'}
            
            # Analyze premium/discount zones
            structure_analysis = await self.market_structure_engine.analyze_enhanced_structure(
                df, symbol, market_data.get('timeframe', '1h')
            )
            
            pd_zone = structure_analysis.premium_discount
            current_zone = pd_zone.current_zone.value
            
            # Determine entry strategy
            strategy = 'immediate'
            
            if direction.upper() == 'LONG':
                if current_zone == 'discount':
                    strategy = 'immediate'  # Good entry zone
                elif current_zone == 'premium':
                    strategy = 'wait_pullback'  # Wait for better price
                else:
                    strategy = 'immediate'  # Equilibrium
            
            elif direction.upper() == 'SHORT':
                if current_zone == 'premium':
                    strategy = 'immediate'  # Good entry zone
                elif current_zone == 'discount':
                    strategy = 'wait_pullback'  # Wait for better price
                else:
                    strategy = 'immediate'  # Equilibrium
            
            logger.info(f"📍 Entry zone: {current_zone} | Strategy: {strategy}")
            
            return {
                'zone': current_zone,
                'strategy': strategy,
                'distance_to_equilibrium': pd_zone.distance_to_equilibrium,
                'confidence': pd_zone.confidence
            }
            
        except Exception as e:
            logger.error(f"Entry zone determination error: {e}")
            return {'zone': 'unknown', 'strategy': 'immediate'}
    
    def _determine_signal_quality(
        self,
        consensus_confidence: float,
        agreeing_heads: int,
        total_heads: int,
        liquidation_risk: bool,
        extreme_leverage: bool
    ) -> str:
        """Determine overall signal quality"""
        # Start with confidence-based quality
        if consensus_confidence >= 0.85 and agreeing_heads >= 7:
            quality = 'excellent'
        elif consensus_confidence >= 0.75 and agreeing_heads >= 5:
            quality = 'good'
        else:
            quality = 'acceptable'
        
        # Downgrade if risks detected
        if liquidation_risk and quality == 'excellent':
            quality = 'good'
        if extreme_leverage and quality == 'excellent':
            quality = 'good'
        
        return quality

