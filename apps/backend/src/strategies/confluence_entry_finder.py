"""
Confluence-Based Entry Finder
Finds entries ONLY when multiple factors align (70%+ confluence required)
"""

import logging
from typing import Dict, Optional, List

logger = logging.getLogger(__name__)

class ConfluenceEntryFinder:
    """
    Multi-factor entry validation
    Uses price action, structure, patterns, volume for confluence
    """
    
    def __init__(self):
        self.min_confluence_score = 0.70  # Need 70%+ to pass
        logger.info("Confluence Entry Finder initialized")
    
    async def find_entry(
        self,
        symbol: str,
        entry_tf: str,
        bias: Dict,
        regime: Dict,
        indicators: Dict
    ) -> Optional[Dict]:
        """
        Find high-confluence entry point
        Returns entry only if 70%+ confluence score
        """
        
        current_price = indicators['current_price']
        rsi = indicators.get('rsi', 50)
        volume_ratio = indicators.get('volume_ratio', 1.0)
        macd = indicators.get('macd', 0)
        macd_signal = indicators.get('macd_signal', 0)
        bb_upper = indicators.get('bb_upper', 0)
        bb_lower = indicators.get('bb_lower', 0)
        sma_20 = indicators.get('sma_20', 0)
        
        confluence_score = 0.0
        reasons = []
        
        # Check if bias direction allows entry
        direction = bias['direction']
        
        if direction == 'LONG':
            entry_price = current_price
            
            # === CONFLUENCE FACTOR 1: Price Action (30%) ===
            if rsi < 35:  # Strong oversold
                confluence_score += 0.30
                reasons.append(f"RSI oversold: {rsi:.1f}")
            elif rsi < 45:  # Moderate oversold
                confluence_score += 0.15
                reasons.append(f"RSI favorable: {rsi:.1f}")
            
            # === CONFLUENCE FACTOR 2: Bollinger Bands (20%) ===
            if current_price <= bb_lower * 1.01:  # At or below lower band
                confluence_score += 0.20
                reasons.append("At Bollinger lower band")
            elif current_price <= sma_20:  # Below middle
                confluence_score += 0.10
                reasons.append("Below BB middle")
            
            # === CONFLUENCE FACTOR 3: Volume (20%) ===
            if volume_ratio > 2.0:  # Strong volume
                confluence_score += 0.20
                reasons.append(f"Strong volume: {volume_ratio:.2f}x")
            elif volume_ratio > 1.5:  # Good volume
                confluence_score += 0.10
                reasons.append(f"Good volume: {volume_ratio:.2f}x")
            
            # === CONFLUENCE FACTOR 4: MACD (15%) ===
            if macd > macd_signal:
                confluence_score += 0.15
                reasons.append("MACD bullish")
            elif macd > macd_signal * 0.9:  # Almost crossed
                confluence_score += 0.07
                reasons.append("MACD turning bullish")
            
            # === CONFLUENCE FACTOR 5: Moving Average Support (15%) ===
            ma_distance = abs(current_price - sma_20) / current_price
            if ma_distance < 0.005:  # Within 0.5% of SMA20
                confluence_score += 0.15
                reasons.append("At SMA20 support")
            elif ma_distance < 0.01:  # Within 1%
                confluence_score += 0.07
                reasons.append("Near SMA20")
            
        elif direction == 'SHORT':
            entry_price = current_price
            
            # === CONFLUENCE FACTOR 1: Price Action (30%) ===
            if rsi > 65:  # Strong overbought
                confluence_score += 0.30
                reasons.append(f"RSI overbought: {rsi:.1f}")
            elif rsi > 55:
                confluence_score += 0.15
                reasons.append(f"RSI favorable: {rsi:.1f}")
            
            # === CONFLUENCE FACTOR 2: Bollinger Bands (20%) ===
            if current_price >= bb_upper * 0.99:
                confluence_score += 0.20
                reasons.append("At Bollinger upper band")
            elif current_price >= sma_20:
                confluence_score += 0.10
                reasons.append("Above BB middle")
            
            # === CONFLUENCE FACTOR 3: Volume (20%) ===
            if volume_ratio > 2.0:
                confluence_score += 0.20
                reasons.append(f"Strong volume: {volume_ratio:.2f}x")
            elif volume_ratio > 1.5:
                confluence_score += 0.10
                reasons.append(f"Good volume: {volume_ratio:.2f}x")
            
            # === CONFLUENCE FACTOR 4: MACD (15%) ===
            if macd < macd_signal:
                confluence_score += 0.15
                reasons.append("MACD bearish")
            elif macd < macd_signal * 1.1:
                confluence_score += 0.07
                reasons.append("MACD turning bearish")
            
            # === CONFLUENCE FACTOR 5: Moving Average Resistance (15%) ===
            ma_distance = abs(current_price - sma_20) / current_price
            if ma_distance < 0.005:
                confluence_score += 0.15
                reasons.append("At SMA20 resistance")
            elif ma_distance < 0.01:
                confluence_score += 0.07
                reasons.append("Near SMA20")
        
        else:
            return {'found': False, 'confluence_score': 0}
        
        # === STRICT THRESHOLD: Need 70%+ confluence ===
        if confluence_score < self.min_confluence_score:
            logger.debug(f"{symbol}: Confluence too low: {confluence_score:.2f} (need {self.min_confluence_score})")
            return {'found': False, 'confluence_score': confluence_score}
        
        # Calculate risk/reward
        if direction == 'LONG':
            stop_loss = entry_price * 0.97  # 3% stop
            take_profit = entry_price * 1.075  # 7.5% target (2.5:1 R:R)
        else:
            stop_loss = entry_price * 1.03
            take_profit = entry_price * 0.925
        
        risk = abs(entry_price - stop_loss) / entry_price
        reward = abs(take_profit - entry_price) / entry_price
        rr_ratio = reward / risk if risk > 0 else 0
        
        logger.info(f"{symbol}: HIGH CONFLUENCE ENTRY FOUND! Score: {confluence_score:.2f}, Reasons: {', '.join(reasons)}")
        
        return {
            'found': True,
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'confluence_score': confluence_score,
            'confluence_reasons': reasons,
            'risk_reward_ratio': rr_ratio,
            'risk_pct': risk,
            'reward_pct': reward
        }

