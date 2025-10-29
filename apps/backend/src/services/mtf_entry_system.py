"""
Multi-Timeframe Entry Refinement System
Industry-standard implementation for precise trade entries

Entry Strategies (Priority Order):
1. Fibonacci Retracement (38.2%, 50%, 61.8%)
2. EMA Pullback (9, 21, 50 EMAs)
3. Order Block Retracement
4. Break & Retest
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class MTFEntrySystem:
    """
    Multi-Timeframe Entry Refinement System
    
    Professional trading approach:
    - Higher TF: Determine trend/bias
    - Lower TF: Find precise entry using Fibonacci, EMAs, Order Blocks
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.logger = logger
        self.config = config or {}
        
        # Load MTF configuration
        mtf_config = self.config.get('mtf_strategies', {})
        
        # Timeframe hierarchy for MTF analysis (from config or default)
        self.timeframe_hierarchy = mtf_config.get('timeframe_mappings', {
            '1d': '4h',
            '4h': '1h',
            '1h': '15m',
            '15m': '5m',
            '5m': '1m'
        })
        
        # Risk management parameters (from config or default)
        risk_config = self.config.get('risk_management', {})
        self.stop_loss_atr_multiplier = risk_config.get('stop_loss', {}).get('atr_multiplier', 1.5)
        self.tp_multipliers = risk_config.get('take_profit', {
            'tp1_atr_multiplier': 2.0,
            'tp2_atr_multiplier': 3.5,
            'tp3_atr_multiplier': 5.0
        })
        
        # Statistics
        self.stats = {
            'total_refinements': 0,
            'successful_refinements': 0,
            'fallback_entries': 0,
            'fibonacci_entries': 0,
            'ema_entries': 0,
            'order_block_entries': 0,
            'market_entries': 0
        }
    
    def get_entry_timeframe(self, signal_timeframe: str) -> str:
        """Get appropriate entry timeframe for signal timeframe"""
        return self.timeframe_hierarchy.get(signal_timeframe, '15m')
    
    async def refine_entry(
        self,
        symbol: str,
        entry_df: pd.DataFrame,
        signal_direction: str,
        current_price: float,
        signal_timeframe: str
    ) -> Optional[Dict[str, Any]]:
        """
        Refine entry using lower timeframe data
        
        Args:
            symbol: Trading symbol
            entry_df: Lower timeframe OHLCV data
            signal_direction: 'LONG' or 'SHORT' from higher TF
            current_price: Current market price
            signal_timeframe: Higher timeframe used for signal
        
        Returns:
            Dict with refined entry, stop, targets, strategy, confidence
        """
        try:
            self.stats['total_refinements'] += 1
            
            # Data sufficiency check (need 200 candles for proper analysis)
            if entry_df is None or len(entry_df) < 200:
                self.logger.warning(
                    f"⚠️ Insufficient entry data for {symbol}: "
                    f"{len(entry_df) if entry_df is not None else 0} candles (need 200). "
                    f"Using market entry fallback."
                )
                self.stats['fallback_entries'] += 1
                return self._create_fallback_entry(
                    current_price=current_price,
                    signal_direction=signal_direction,
                    signal_timeframe=signal_timeframe
                )
            
            # Add indicators for entry analysis
            entry_df = self._add_entry_indicators(entry_df)
            
            # Get latest values
            latest = entry_df.iloc[-1]
            atr = latest.get('atr', 0)
            
            if atr == 0 or pd.isna(atr):
                self.logger.warning(f"Invalid ATR for {symbol}")
                return None
            
            # Find swing points for Fibonacci
            swing_high, swing_low = self._find_recent_swing_points(entry_df, lookback=50)
            
            # Try different entry strategies
            fib_entry = self._calculate_fibonacci_entry(
                signal_direction, swing_high, swing_low, current_price, atr
            )
            
            ema_entry = self._calculate_ema_entry(
                signal_direction, latest, current_price, atr
            )
            
            ob_entry = self._calculate_order_block_entry(
                signal_direction, entry_df, current_price, atr
            )
            
            # Select best strategy
            selected_entry = self._select_best_entry_strategy(
                signal_direction, current_price, fib_entry, ema_entry, ob_entry
            )
            
            if not selected_entry:
                # Fallback to market entry
                selected_entry = {
                    'entry': current_price,
                    'strategy': 'MARKET_ENTRY',
                    'confidence': 0.5,
                    'fibonacci_level': None
                }
            
            # Calculate stop loss and take profits using config parameters
            if signal_direction == 'LONG':
                stop_loss = selected_entry['entry'] - (atr * self.stop_loss_atr_multiplier)
                tp1 = selected_entry['entry'] + (atr * self.tp_multipliers.get('tp1_atr_multiplier', 2.0))
                tp2 = selected_entry['entry'] + (atr * self.tp_multipliers.get('tp2_atr_multiplier', 3.5))
                tp3 = selected_entry['entry'] + (atr * self.tp_multipliers.get('tp3_atr_multiplier', 5.0))
            else:  # SHORT
                stop_loss = selected_entry['entry'] + (atr * self.stop_loss_atr_multiplier)
                tp1 = selected_entry['entry'] - (atr * self.tp_multipliers.get('tp1_atr_multiplier', 2.0))
                tp2 = selected_entry['entry'] - (atr * self.tp_multipliers.get('tp2_atr_multiplier', 3.5))
                tp3 = selected_entry['entry'] - (atr * self.tp_multipliers.get('tp3_atr_multiplier', 5.0))
            
            # Risk:Reward ratio
            risk = abs(selected_entry['entry'] - stop_loss)
            reward = abs(tp1 - selected_entry['entry'])
            rr_ratio = reward / risk if risk > 0 else 0
            
            # Entry confidence
            entry_confidence = self._calculate_entry_confidence(
                entry_df, signal_direction, selected_entry['entry'], current_price
            )
            
            # Entry pattern
            entry_pattern = self._detect_entry_pattern(entry_df, signal_direction)
            
            # Volume confirmation
            volume_confirmed = self._check_volume_confirmation(entry_df)
            
            result = {
                'refined_entry': selected_entry['entry'],
                'refined_stop': stop_loss,
                'refined_targets': [tp1, tp2, tp3],
                'entry_strategy': selected_entry['strategy'],
                'entry_pattern': entry_pattern,
                'entry_confidence': entry_confidence,
                'fibonacci_level': selected_entry.get('fibonacci_level'),
                'atr': atr,
                'risk_reward_ratio': rr_ratio,
                'volume_confirmed': volume_confirmed,
                'ema_levels': {
                    'ema_9': float(latest.get('ema_9', 0)),
                    'ema_21': float(latest.get('ema_21', 0)),
                    'ema_50': float(latest.get('ema_50', 0))
                }
            }
            
            self.logger.info(
                f"✅ Entry refined for {symbol}: {selected_entry['strategy']} @ ${selected_entry['entry']:.2f} | R:R {rr_ratio:.2f}"
            )
            
            # Track strategy usage
            strategy_name = selected_entry['strategy']
            if 'FIBONACCI' in strategy_name:
                self.stats['fibonacci_entries'] += 1
            elif 'EMA' in strategy_name:
                self.stats['ema_entries'] += 1
            elif 'ORDER_BLOCK' in strategy_name:
                self.stats['order_block_entries'] += 1
            elif strategy_name == 'MARKET_ENTRY':
                self.stats['market_entries'] += 1
            
            self.stats['successful_refinements'] += 1
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error refining entry for {symbol}: {e}")
            self.stats['fallback_entries'] += 1
            
            # Return fallback entry on error
            return self._create_fallback_entry(
                current_price=current_price,
                signal_direction=signal_direction,
                signal_timeframe=signal_timeframe
            )
    
    def _add_entry_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add essential indicators for entry analysis"""
        try:
            # EMAs for pullback detection
            df['ema_9'] = df['close'].ewm(span=9, adjust=False).mean()
            df['ema_21'] = df['close'].ewm(span=21, adjust=False).mean()
            df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
            
            # ATR for stop loss/take profit
            df['high_low'] = df['high'] - df['low']
            df['high_close'] = abs(df['high'] - df['close'].shift(1))
            df['low_close'] = abs(df['low'] - df['close'].shift(1))
            df['tr'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)
            df['atr'] = df['tr'].rolling(window=14).mean()
            
            # Volume MA
            df['volume_ma'] = df['volume'].rolling(window=20).mean()
            
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error adding indicators: {e}")
            return df
    
    def _find_recent_swing_points(self, df: pd.DataFrame, lookback: int = 50) -> Tuple[float, float]:
        """Find recent swing high and low for Fibonacci"""
        try:
            recent_data = df.tail(lookback)
            swing_high = float(recent_data['high'].max())
            swing_low = float(recent_data['low'].min())
            return swing_high, swing_low
        except:
            return 0.0, 0.0
    
    def _calculate_fibonacci_entry(
        self,
        direction: str,
        swing_high: float,
        swing_low: float,
        current_price: float,
        atr: float
    ) -> Optional[Dict[str, Any]]:
        """Calculate Fibonacci retracement entry (38.2%, 50%, 61.8%)"""
        try:
            if swing_high == 0 or swing_low == 0 or swing_high <= swing_low:
                return None
            
            range_size = swing_high - swing_low
            
            if direction == 'LONG':
                # Retracement from swing high
                fib_618 = swing_high - (range_size * 0.618)
                fib_50 = swing_high - (range_size * 0.5)
                fib_382 = swing_high - (range_size * 0.382)
                
                # Check proximity (within 1 ATR)
                if abs(current_price - fib_618) < atr:
                    return {'entry': fib_618, 'strategy': 'FIBONACCI_618', 'confidence': 0.85, 'fibonacci_level': 0.618}
                elif abs(current_price - fib_50) < atr:
                    return {'entry': fib_50, 'strategy': 'FIBONACCI_50', 'confidence': 0.75, 'fibonacci_level': 0.5}
                elif abs(current_price - fib_382) < atr:
                    return {'entry': fib_382, 'strategy': 'FIBONACCI_382', 'confidence': 0.70, 'fibonacci_level': 0.382}
            else:  # SHORT
                # Retracement from swing low
                fib_618 = swing_low + (range_size * 0.618)
                fib_50 = swing_low + (range_size * 0.5)
                fib_382 = swing_low + (range_size * 0.382)
                
                if abs(current_price - fib_618) < atr:
                    return {'entry': fib_618, 'strategy': 'FIBONACCI_618', 'confidence': 0.85, 'fibonacci_level': 0.618}
                elif abs(current_price - fib_50) < atr:
                    return {'entry': fib_50, 'strategy': 'FIBONACCI_50', 'confidence': 0.75, 'fibonacci_level': 0.5}
                elif abs(current_price - fib_382) < atr:
                    return {'entry': fib_382, 'strategy': 'FIBONACCI_382', 'confidence': 0.70, 'fibonacci_level': 0.382}
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error calculating Fibonacci entry: {e}")
            return None
    
    def _calculate_ema_entry(
        self,
        direction: str,
        latest: pd.Series,
        current_price: float,
        atr: float
    ) -> Optional[Dict[str, Any]]:
        """Calculate EMA pullback entry"""
        try:
            ema_9 = latest.get('ema_9', 0)
            ema_21 = latest.get('ema_21', 0)
            ema_50 = latest.get('ema_50', 0)
            
            if pd.isna(ema_9) or pd.isna(ema_21) or pd.isna(ema_50):
                return None
            
            if direction == 'LONG':
                # Check proximity to EMAs (within 0.5 ATR)
                if ema_9 > 0 and abs(current_price - ema_9) < (atr * 0.5):
                    return {'entry': ema_9, 'strategy': 'EMA_9_PULLBACK', 'confidence': 0.80}
                elif ema_21 > 0 and abs(current_price - ema_21) < (atr * 0.5):
                    return {'entry': ema_21, 'strategy': 'EMA_21_PULLBACK', 'confidence': 0.75}
                elif ema_50 > 0 and abs(current_price - ema_50) < (atr * 0.5):
                    return {'entry': ema_50, 'strategy': 'EMA_50_PULLBACK', 'confidence': 0.70}
            else:  # SHORT
                if ema_9 > 0 and abs(current_price - ema_9) < (atr * 0.5):
                    return {'entry': ema_9, 'strategy': 'EMA_9_REJECTION', 'confidence': 0.80}
                elif ema_21 > 0 and abs(current_price - ema_21) < (atr * 0.5):
                    return {'entry': ema_21, 'strategy': 'EMA_21_REJECTION', 'confidence': 0.75}
                elif ema_50 > 0 and abs(current_price - ema_50) < (atr * 0.5):
                    return {'entry': ema_50, 'strategy': 'EMA_50_REJECTION', 'confidence': 0.70}
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error calculating EMA entry: {e}")
            return None
    
    def _calculate_order_block_entry(
        self,
        direction: str,
        df: pd.DataFrame,
        current_price: float,
        atr: float
    ) -> Optional[Dict[str, Any]]:
        """Calculate order block entry (simplified ICT concept)"""
        try:
            # Look for order blocks in last 20 candles
            recent = df.tail(20)
            
            if direction == 'LONG':
                # Bullish OB: Down candle + strong up move
                for i in range(len(recent) - 1):
                    candle = recent.iloc[i]
                    next_candle = recent.iloc[i + 1]
                    
                    if candle['close'] < candle['open']:  # Bearish candle
                        if next_candle['close'] > next_candle['open']:  # Bullish move
                            move_size = next_candle['close'] - next_candle['open']
                            prev_body = abs(candle['open'] - candle['close'])
                            
                            if move_size > prev_body * 2:  # Strong move
                                ob_mid = (candle['low'] + candle['high']) / 2
                                
                                if abs(current_price - ob_mid) < atr:
                                    return {'entry': ob_mid, 'strategy': 'ORDER_BLOCK_LONG', 'confidence': 0.75}
            else:  # SHORT
                # Bearish OB: Up candle + strong down move
                for i in range(len(recent) - 1):
                    candle = recent.iloc[i]
                    next_candle = recent.iloc[i + 1]
                    
                    if candle['close'] > candle['open']:  # Bullish candle
                        if next_candle['close'] < next_candle['open']:  # Bearish move
                            move_size = next_candle['open'] - next_candle['close']
                            prev_body = abs(candle['open'] - candle['close'])
                            
                            if move_size > prev_body * 2:
                                ob_mid = (candle['low'] + candle['high']) / 2
                                
                                if abs(current_price - ob_mid) < atr:
                                    return {'entry': ob_mid, 'strategy': 'ORDER_BLOCK_SHORT', 'confidence': 0.75}
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error calculating order block entry: {e}")
            return None
    
    def _select_best_entry_strategy(
        self,
        direction: str,
        current_price: float,
        fib_entry: Optional[Dict],
        ema_entry: Optional[Dict],
        ob_entry: Optional[Dict]
    ) -> Optional[Dict]:
        """Select best entry based on confidence"""
        entries = []
        
        if fib_entry:
            entries.append(fib_entry)
        if ema_entry:
            entries.append(ema_entry)
        if ob_entry:
            entries.append(ob_entry)
        
        if not entries:
            return None
        
        # Sort by confidence
        entries.sort(key=lambda x: x['confidence'], reverse=True)
        
        return entries[0]
    
    def _calculate_entry_confidence(
        self,
        df: pd.DataFrame,
        direction: str,
        entry_price: float,
        current_price: float
    ) -> float:
        """Calculate entry timing confidence (0-1)"""
        try:
            confidence = 0.5
            
            latest = df.iloc[-1]
            
            # Volume confirmation
            volume_ma = latest.get('volume_ma', 0)
            if volume_ma > 0 and latest['volume'] > volume_ma * 1.2:
                confidence += 0.15
            
            # EMA alignment
            ema_9 = latest.get('ema_9', 0)
            ema_21 = latest.get('ema_21', 0)
            
            if not pd.isna(ema_9) and not pd.isna(ema_21):
                if direction == 'LONG' and ema_9 > ema_21:
                    confidence += 0.15
                elif direction == 'SHORT' and ema_9 < ema_21:
                    confidence += 0.15
            
            # Proximity to entry
            if current_price > 0 and entry_price > 0:
                distance_percent = abs(current_price - entry_price) / current_price
                if distance_percent < 0.005:  # Within 0.5%
                    confidence += 0.20
                elif distance_percent < 0.01:  # Within 1%
                    confidence += 0.10
            
            return min(confidence, 1.0)
            
        except:
            return 0.5
    
    def _detect_entry_pattern(self, df: pd.DataFrame, direction: str) -> str:
        """Detect candlestick pattern"""
        try:
            if len(df) < 2:
                return 'UNKNOWN'
            
            latest = df.iloc[-1]
            prev = df.iloc[-2]
            volume_ma = df['volume'].rolling(20).mean().iloc[-1] if len(df) >= 20 else 0
            
            if direction == 'LONG':
                if (latest['close'] > latest['open'] and 
                    latest['close'] > prev['open'] and
                    volume_ma > 0 and latest['volume'] > volume_ma):
                    return 'BULLISH_ENGULFING'
                elif (latest['close'] > latest['open'] and
                      (latest['close'] - latest['open']) > (latest['high'] - latest['low']) * 0.7):
                    return 'STRONG_BULLISH_CANDLE'
                else:
                    return 'BULLISH_SETUP'
            else:  # SHORT
                if (latest['close'] < latest['open'] and 
                    latest['close'] < prev['open'] and
                    volume_ma > 0 and latest['volume'] > volume_ma):
                    return 'BEARISH_ENGULFING'
                elif (latest['close'] < latest['open'] and
                      (latest['open'] - latest['close']) > (latest['high'] - latest['low']) * 0.7):
                    return 'STRONG_BEARISH_CANDLE'
                else:
                    return 'BEARISH_SETUP'
                    
        except:
            return 'UNKNOWN'
    
    def _check_volume_confirmation(self, df: pd.DataFrame) -> bool:
        """Check volume confirmation"""
        try:
            if len(df) < 20:
                return False
            
            latest = df.iloc[-1]
            volume_ma = df['volume'].rolling(20).mean().iloc[-1]
            
            return latest['volume'] > volume_ma * 1.1
            
        except:
            return False
    
    def _create_fallback_entry(
        self,
        current_price: float,
        signal_direction: str,
        signal_timeframe: str
    ) -> Dict[str, Any]:
        """
        Create market entry when insufficient data for refinement
        
        Args:
            current_price: Current market price
            signal_direction: Signal direction (LONG/SHORT)
            signal_timeframe: Higher timeframe being analyzed
        
        Returns:
            Dict with fallback entry data using estimated ATR
        """
        try:
            # Estimate ATR as 1% of price (conservative)
            estimated_atr = current_price * 0.01
            
            # Calculate levels
            if signal_direction == 'LONG':
                stop_loss = current_price - (estimated_atr * self.stop_loss_atr_multiplier)
                tp1 = current_price + (estimated_atr * self.tp_multipliers.get('tp1_atr_multiplier', 2.0))
                tp2 = current_price + (estimated_atr * self.tp_multipliers.get('tp2_atr_multiplier', 3.5))
                tp3 = current_price + (estimated_atr * self.tp_multipliers.get('tp3_atr_multiplier', 5.0))
            else:  # SHORT
                stop_loss = current_price + (estimated_atr * self.stop_loss_atr_multiplier)
                tp1 = current_price - (estimated_atr * self.tp_multipliers.get('tp1_atr_multiplier', 2.0))
                tp2 = current_price - (estimated_atr * self.tp_multipliers.get('tp2_atr_multiplier', 3.5))
                tp3 = current_price - (estimated_atr * self.tp_multipliers.get('tp3_atr_multiplier', 5.0))
            
            # Calculate R:R
            risk = abs(current_price - stop_loss)
            reward = abs(tp1 - current_price)
            rr_ratio = reward / risk if risk > 0 else 1.33
            
            self.logger.warning(
                f"⚠️ Using fallback entry for {signal_direction} signal "
                f"(estimated ATR: ${estimated_atr:.2f})"
            )
            
            return {
                'refined_entry': current_price,
                'refined_stop': stop_loss,
                'refined_targets': [tp1, tp2, tp3],
                'entry_strategy': 'MARKET_ENTRY',
                'entry_pattern': 'INSUFFICIENT_DATA',
                'entry_confidence': 0.50,
                'fibonacci_level': None,
                'atr': estimated_atr,
                'risk_reward_ratio': rr_ratio,
                'volume_confirmed': False,
                'ema_levels': {}
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error creating fallback entry: {e}")
            # Return minimal fallback
            return {
                'refined_entry': current_price,
                'refined_stop': current_price * 0.98 if signal_direction == 'LONG' else current_price * 1.02,
                'refined_targets': [
                    current_price * 1.02 if signal_direction == 'LONG' else current_price * 0.98,
                    current_price * 1.035 if signal_direction == 'LONG' else current_price * 0.965,
                    current_price * 1.05 if signal_direction == 'LONG' else current_price * 0.95
                ],
                'entry_strategy': 'MARKET_ENTRY',
                'entry_pattern': 'ERROR_FALLBACK',
                'entry_confidence': 0.50,
                'fibonacci_level': None,
                'atr': current_price * 0.01,
                'risk_reward_ratio': 1.0,
                'volume_confirmed': False,
                'ema_levels': {}
            }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get MTF entry system statistics"""
        return {
            'total_refinements': self.stats['total_refinements'],
            'successful_refinements': self.stats['successful_refinements'],
            'fallback_entries': self.stats['fallback_entries'],
            'success_rate': (
                self.stats['successful_refinements'] / self.stats['total_refinements']
                if self.stats['total_refinements'] > 0 else 0
            ),
            'strategy_distribution': {
                'fibonacci': self.stats['fibonacci_entries'],
                'ema': self.stats['ema_entries'],
                'order_block': self.stats['order_block_entries'],
                'market': self.stats['market_entries']
            }
        }

