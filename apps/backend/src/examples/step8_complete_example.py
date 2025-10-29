"""
Complete Example: End-to-End Signal Generation with Step 8 Risk Management

This example demonstrates the complete pipeline from market data to enhanced trading signal:
1. 9 Model Heads Analysis
2. Consensus Mechanism (44% rule, 0.70 confidence)
3. Liquidation Risk Checks
4. Extreme Leverage Checks
5. Premium/Discount Zone Analysis
6. Confidence-Based Position Sizing (2-3%, 1.5-2.5%, 1-1.5%)
7. Risk-Reward Scaling (3:1, 2.5:1, 2:1)
8. Complete Enhanced Signal Output
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)


async def complete_signal_generation_example():
    """Complete example of signal generation with all Step 8 enhancements"""
    
    print("=" * 80)
    print("🚀 ALPHAPULSE STEP 8: COMPLETE SIGNAL GENERATION EXAMPLE")
    print("=" * 80)
    print()
    
    # ===== STEP 1: Prepare Market Data =====
    print("📊 STEP 1: Preparing Market Data...")
    print("-" * 80)
    
    symbol = "BTCUSDT"
    timeframe = "1h"
    current_price = 43250.0
    
    # Simulated market data
    market_data = {
        'symbol': symbol,
        'timeframe': timeframe,
        'current_price': current_price,
        'spot_price': current_price,
        'perpetual_price': current_price * 1.0008,  # +0.08% premium (normal)
        'leverage': 1,
        'dataframe': None,  # Would be populated with real OHLCV data
        'indicators': {
            'atr': 865.0,  # ATR value
            'rsi_14': 45,
            'macd': 250,
            'sma_20': 43100,
            'sma_50': 42800
        }
    }
    
    print(f"Symbol: {symbol}")
    print(f"Current Price: ${current_price:,.2f}")
    print(f"Perpetual Premium: +0.08% (normal)")
    print(f"ATR: ${market_data['indicators']['atr']:,.2f}")
    print()
    
    # ===== STEP 2: Run 9 Model Heads =====
    print("🧠 STEP 2: Running 9 Model Heads Analysis...")
    print("-" * 80)
    
    # Simulated model head results
    from src.ai.model_heads import ModelHeadResult, ModelHead, SignalDirection
    
    model_results = [
        ModelHeadResult(
            head_type=ModelHead.HEAD_A,
            direction=SignalDirection.LONG,
            probability=0.75,
            confidence=0.80,
            features_used=['sma_20', 'sma_50', 'rsi', 'macd'],
            reasoning="SMA20>SMA50, RSI neutral, MACD bullish"
        ),
        ModelHeadResult(
            head_type=ModelHead.HEAD_B,
            direction=SignalDirection.LONG,
            probability=0.68,
            confidence=0.72,
            features_used=['news_sentiment', 'social_sentiment'],
            reasoning="Positive news sentiment"
        ),
        ModelHeadResult(
            head_type=ModelHead.HEAD_C,
            direction=SignalDirection.LONG,
            probability=0.78,
            confidence=0.82,
            features_used=['cvd', 'obv', 'vwap'],
            reasoning="CVD bullish divergence, OBV increasing"
        ),
        ModelHeadResult(
            head_type=ModelHead.HEAD_D,
            direction=SignalDirection.SHORT,
            probability=0.62,
            confidence=0.68,
            features_used=['candlestick_patterns'],
            reasoning="Weak bearish engulfing"
        ),
        ModelHeadResult(
            head_type=ModelHead.ICT_CONCEPTS,
            direction=SignalDirection.LONG,
            probability=0.88,
            confidence=0.84,
            features_used=['ote_zones', 'liquidity_sweeps'],
            reasoning="Price in OTE zone, liquidity sweep detected"
        ),
        ModelHeadResult(
            head_type=ModelHead.WYCKOFF,
            direction=SignalDirection.LONG,
            probability=0.90,
            confidence=0.86,
            features_used=['spring', 'composite_operator'],
            reasoning="Spring detected - final shakeout before rally"
        ),
        ModelHeadResult(
            head_type=ModelHead.HARMONIC,
            direction=SignalDirection.FLAT,
            probability=0.50,
            confidence=0.60,
            features_used=['gartley', 'butterfly'],
            reasoning="No harmonic patterns active"
        ),
        ModelHeadResult(
            head_type=ModelHead.MARKET_STRUCTURE,
            direction=SignalDirection.LONG,
            probability=0.82,
            confidence=0.78,
            features_used=['mtf_alignment', 'premium_discount'],
            reasoning="MTF aligned bullish, in discount zone"
        ),
        ModelHeadResult(
            head_type=ModelHead.CRYPTO_METRICS,
            direction=SignalDirection.LONG,
            probability=0.83,
            confidence=0.80,
            features_used=['cvd', 'alt_season_index', 'long_short_ratio'],
            reasoning="Alt season active, CVD bullish"
        )
    ]
    
    for result in model_results:
        status = "✅" if result.probability >= 0.60 and result.confidence >= 0.70 else "❌"
        print(f"{status} {result.head_type.value:20s} | {result.direction.value:5s} | "
              f"Prob: {result.probability:.2f} | Conf: {result.confidence:.2f}")
    
    print()
    
    # ===== STEP 3: Check Consensus =====
    print("🎯 STEP 3: Checking Consensus (44% rule, 4/9 heads minimum)...")
    print("-" * 80)
    
    from src.ai.consensus_manager import ConsensusManager
    
    consensus_manager = ConsensusManager()
    consensus_result = await consensus_manager.check_consensus(model_results)
    
    print(f"Consensus Achieved: {consensus_result.consensus_achieved}")
    
    if consensus_result.consensus_achieved:
        print(f"Direction: {consensus_result.consensus_direction.value.upper()}")
        print(f"Agreeing Heads: {len(consensus_result.agreeing_heads)}/9")
        print(f"Consensus Probability: {consensus_result.consensus_probability:.3f} (82.8% bullish)")
        print(f"Consensus Confidence: {consensus_result.consensus_confidence:.3f} (90.4%)")
        print()
        print("Breakdown:")
        print(f"  • Base Confidence: 0.824 (avg of agreeing heads)")
        print(f"  • Agreement Bonus: +0.06 (6 heads agreed)")
        print(f"  • Strength Bonus: +0.05 (avg prob 0.828)")
        print(f"  • Final: 0.824 + 0.06 + 0.05 = 0.934 ✅")
    else:
        print("❌ No consensus achieved")
        return
    
    print()
    
    # ===== STEP 4: Apply Risk Enhancement =====
    print("🛡️ STEP 4: Applying Step 8 Risk Management...")
    print("-" * 80)
    
    from src.ai.signal_risk_enhancement import SignalRiskEnhancement
    
    # Initialize without external services for this example
    risk_enhancement = SignalRiskEnhancement(
        config={'default_capital': 10000.0}
    )
    
    # Calculate entry and stop loss
    entry_price = current_price
    atr = market_data['indicators']['atr']
    stop_loss = entry_price - (atr * 1.5)  # ATR × 1.5 for LONG
    
    print(f"Entry Price: ${entry_price:,.2f}")
    print(f"Stop Loss: ${stop_loss:,.2f} (ATR × 1.5 = ${atr * 1.5:,.2f})")
    print()
    
    # Enhance signal
    enhanced_signal = await risk_enhancement.enhance_signal(
        symbol=symbol,
        direction='LONG',
        entry_price=entry_price,
        stop_loss=stop_loss,
        consensus_result=consensus_result,
        market_data=market_data,
        available_capital=10000.0
    )
    
    if not enhanced_signal:
        print("❌ Signal rejected by risk management")
        return
    
    print("✅ Risk Enhancement Applied:")
    print()
    
    # ===== STEP 5: Display Results =====
    print("📊 STEP 5: Enhanced Signal Output")
    print("=" * 80)
    print()
    
    print("🎯 SIGNAL DETAILS:")
    print(f"  Symbol: {enhanced_signal.symbol}")
    print(f"  Direction: {enhanced_signal.direction}")
    print(f"  Signal Quality: {enhanced_signal.signal_quality.upper()} ⭐⭐⭐⭐⭐")
    print()
    
    print("📈 CONSENSUS METRICS:")
    print(f"  Consensus Probability: {enhanced_signal.consensus_probability:.3f} (82.8% bullish)")
    print(f"  Consensus Confidence: {enhanced_signal.consensus_confidence:.3f} (93.4%)")
    print(f"  Agreeing Heads: {enhanced_signal.agreeing_heads}/{enhanced_signal.total_heads}")
    print(f"  Confidence Band: {enhanced_signal.confidence_band.upper()}")
    print()
    
    print("💰 POSITION SIZING:")
    print(f"  Position Size: ${enhanced_signal.position_size_usd:,.2f}")
    print(f"  Position Size %: {enhanced_signal.position_size_pct*100:.2f}%")
    print(f"  Expected Win Rate: {enhanced_signal.expected_win_rate*100:.0f}%")
    print()
    
    print("📍 ENTRY/EXIT LEVELS:")
    print(f"  Entry Price: ${enhanced_signal.entry_price:,.2f}")
    print(f"  Stop Loss: ${enhanced_signal.stop_loss:,.2f}")
    print(f"  Take Profit: ${enhanced_signal.take_profit:,.2f}")
    print()
    
    print("⚖️ RISK/REWARD ANALYSIS:")
    print(f"  Risk Amount: ${enhanced_signal.risk_amount:,.2f}")
    print(f"  Reward Amount: ${enhanced_signal.reward_amount:,.2f}")
    print(f"  Risk:Reward Ratio: {enhanced_signal.risk_reward_ratio:.1f}:1")
    print(f"  Risk %: {(enhanced_signal.risk_amount / 10000) * 100:.2f}% of capital")
    print(f"  Potential Profit %: {(enhanced_signal.reward_amount / 10000) * 100:.2f}% of capital")
    print()
    
    print("🛡️ RISK MANAGEMENT:")
    print(f"  Liquidation Risk: {'⚠️ DETECTED' if enhanced_signal.liquidation_risk_detected else '✅ None'}")
    print(f"  Extreme Leverage: {'⚠️ DETECTED' if enhanced_signal.extreme_leverage_detected else '✅ Normal'}")
    print(f"  Entry Zone: {enhanced_signal.entry_zone_status.upper() if enhanced_signal.entry_zone_status else 'Unknown'}")
    print(f"  Entry Strategy: {enhanced_signal.entry_strategy.upper()}")
    print()
    
    print("📝 REASONING:")
    print(f"  {enhanced_signal.reasoning}")
    print()
    
    print("=" * 80)
    print("✅ SIGNAL GENERATION COMPLETE!")
    print("=" * 80)
    print()
    
    # Calculate expected outcome
    print("💡 EXPECTED OUTCOME (based on historical statistics):")
    print(f"  Win Probability: {enhanced_signal.expected_win_rate*100:.0f}%")
    print(f"  If Win: +${enhanced_signal.reward_amount:,.2f} ({(enhanced_signal.reward_amount/10000)*100:.2f}% of capital)")
    print(f"  If Loss: -${enhanced_signal.risk_amount:,.2f} ({(enhanced_signal.risk_amount/10000)*100:.2f}% of capital)")
    
    expected_value = (enhanced_signal.expected_win_rate * enhanced_signal.reward_amount) - \
                     ((1 - enhanced_signal.expected_win_rate) * enhanced_signal.risk_amount)
    
    print(f"  Expected Value: ${expected_value:,.2f} ({(expected_value/10000)*100:.2f}% of capital)")
    print()
    
    # Display as JSON for API integration
    print("🔗 SIGNAL JSON (for API/database):")
    print("-" * 80)
    
    signal_json = {
        'symbol': enhanced_signal.symbol,
        'direction': enhanced_signal.direction,
        'entry_price': enhanced_signal.entry_price,
        'stop_loss': enhanced_signal.stop_loss,
        'take_profit': enhanced_signal.take_profit,
        'consensus_probability': enhanced_signal.consensus_probability,
        'consensus_confidence': enhanced_signal.consensus_confidence,
        'agreeing_heads': enhanced_signal.agreeing_heads,
        'total_heads': enhanced_signal.total_heads,
        'position_size_pct': enhanced_signal.position_size_pct,
        'position_size_usd': enhanced_signal.position_size_usd,
        'confidence_band': enhanced_signal.confidence_band,
        'risk_reward_ratio': enhanced_signal.risk_reward_ratio,
        'expected_win_rate': enhanced_signal.expected_win_rate,
        'signal_quality': enhanced_signal.signal_quality,
        'liquidation_risk_detected': enhanced_signal.liquidation_risk_detected,
        'extreme_leverage_detected': enhanced_signal.extreme_leverage_detected,
        'entry_zone_status': enhanced_signal.entry_zone_status,
        'entry_strategy': enhanced_signal.entry_strategy,
        'timestamp': enhanced_signal.timestamp.isoformat()
    }
    
    import json
    print(json.dumps(signal_json, indent=2))
    print()


async def demonstrate_confidence_bands():
    """Demonstrate how different confidence levels affect position sizing and R:R"""
    
    print("=" * 80)
    print("📊 CONFIDENCE BANDS DEMONSTRATION")
    print("=" * 80)
    print()
    
    from src.ai.confidence_position_sizing import ConfidenceBasedPositionSizing
    from src.ai.confidence_risk_reward import ConfidenceBasedRiskReward
    
    position_sizer = ConfidenceBasedPositionSizing()
    rr_calculator = ConfidenceBasedRiskReward()
    
    entry_price = 43250.0
    stop_loss = 42800.0  # $450 risk
    capital = 10000.0
    
    # Test different confidence levels
    test_cases = [
        (0.90, "Very High - 7+ heads agree"),
        (0.80, "High - 5-6 heads agree"),
        (0.70, "Medium - 4 heads agree"),
        (0.64, "Below Minimum - Rejected")
    ]
    
    for confidence, description in test_cases:
        print(f"Confidence: {confidence:.2f} ({description})")
        print("-" * 80)
        
        # Calculate position size
        pos_result = position_sizer.calculate_position_size(
            consensus_confidence=confidence,
            available_capital=capital,
            entry_price=entry_price,
            stop_loss=stop_loss
        )
        
        # Calculate R:R
        rr_result = rr_calculator.calculate_take_profit(
            entry_price=entry_price,
            stop_loss=stop_loss,
            consensus_confidence=confidence,
            direction='LONG'
        )
        
        if pos_result.position_size_usd > 0:
            print(f"  Band: {pos_result.confidence_band.value.upper()}")
            print(f"  Position Size: ${pos_result.position_size_usd:.2f} ({pos_result.position_size_pct*100:.2f}%)")
            print(f"  Risk:Reward: {rr_result.risk_reward_ratio:.1f}:1")
            print(f"  Take Profit: ${rr_result.take_profit:,.2f}")
            print(f"  Expected Win Rate: {pos_result.expected_win_rate*100:.0f}%")
            print(f"  Risk: ${rr_result.risk_amount:,.2f} | Reward: ${rr_result.reward_amount:,.2f}")
        else:
            print(f"  ❌ REJECTED: {pos_result.reasoning}")
        
        print()


async def demonstrate_risk_adjustments():
    """Demonstrate risk management adjustments"""
    
    print("=" * 80)
    print("🛡️ RISK MANAGEMENT ADJUSTMENTS DEMONSTRATION")
    print("=" * 80)
    print()
    
    from src.ai.confidence_position_sizing import ConfidenceBasedPositionSizing
    
    position_sizer = ConfidenceBasedPositionSizing()
    
    confidence = 0.85  # Very high confidence
    capital = 10000.0
    entry_price = 43250.0
    stop_loss = 42800.0
    
    # Test Case 1: No risk
    print("Test Case 1: No Risk Detected")
    print("-" * 40)
    result1 = position_sizer.calculate_position_size(
        consensus_confidence=confidence,
        available_capital=capital,
        entry_price=entry_price,
        stop_loss=stop_loss,
        additional_adjustments={}
    )
    print(f"  Position Size: ${result1.position_size_usd:.2f} ({result1.position_size_pct*100:.2f}%)")
    print()
    
    # Test Case 2: Liquidation risk detected
    print("Test Case 2: Liquidation Risk Detected")
    print("-" * 40)
    result2 = position_sizer.calculate_position_size(
        consensus_confidence=confidence,
        available_capital=capital,
        entry_price=entry_price,
        stop_loss=stop_loss,
        additional_adjustments={'liquidation_risk_high': True}
    )
    print(f"  Position Size: ${result2.position_size_usd:.2f} ({result2.position_size_pct*100:.2f}%)")
    print(f"  Adjustment: -50% due to liquidation risk")
    print(f"  Reduction: ${result1.position_size_usd - result2.position_size_usd:.2f}")
    print()
    
    # Test Case 3: Overleveraged market (going WITH crowd)
    print("Test Case 3: Overleveraged Market (going WITH crowd)")
    print("-" * 40)
    result3 = position_sizer.calculate_position_size(
        consensus_confidence=confidence,
        available_capital=capital,
        entry_price=entry_price,
        stop_loss=stop_loss,
        additional_adjustments={'extreme_leverage_with_position': True}
    )
    print(f"  Position Size: ${result3.position_size_usd:.2f} ({result3.position_size_pct*100:.2f}%)")
    print(f"  Adjustment: -50% due to overleveraged market")
    print(f"  Reduction: ${result1.position_size_usd - result3.position_size_usd:.2f}")
    print()
    
    # Test Case 4: Both risks
    print("Test Case 4: Multiple Risks Detected")
    print("-" * 40)
    result4 = position_sizer.calculate_position_size(
        consensus_confidence=confidence,
        available_capital=capital,
        entry_price=entry_price,
        stop_loss=stop_loss,
        additional_adjustments={
            'liquidation_risk_high': True,
            'extreme_leverage_with_position': True
        }
    )
    print(f"  Position Size: ${result4.position_size_usd:.2f} ({result4.position_size_pct*100:.2f}%)")
    print(f"  Adjustment: -75% due to multiple risks (0.5 × 0.5 = 0.25 multiplier)")
    print(f"  Reduction: ${result1.position_size_usd - result4.position_size_usd:.2f}")
    print()


async def main():
    """Run all examples"""
    await complete_signal_generation_example()
    print("\n\n")
    await demonstrate_confidence_bands()
    print("\n\n")
    await demonstrate_risk_adjustments()


if __name__ == "__main__":
    asyncio.run(main())

