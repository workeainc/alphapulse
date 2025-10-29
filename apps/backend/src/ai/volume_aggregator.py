"""
Volume Indicator Aggregator for AlphaPulse
Aggregates volume-based indicators into weighted scores for institutional flow detection
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from dataclasses import dataclass
import asyncio

logger = logging.getLogger(__name__)

@dataclass
class AggregatedVolumeResult:
    """Result from volume indicator aggregation"""
    volume_score: float  # 0-1
    accumulation_score: float  # 0-1
    distribution_score: float  # 0-1
    smart_money_flow: str  # 'accumulating', 'distributing', 'neutral'
    confidence: float  # 0-1
    contributing_indicators: List[str]
    indicator_signals: Dict[str, float]
    reasoning: str
    calculation_time_ms: float

class VolumeIndicatorAggregator:
    """
    Aggregates volume indicators for institutional flow detection
    
    Indicators aggregated:
    - CVD (Cumulative Volume Delta) - 20% weight
    - OBV (On Balance Volume) - 15% weight
    - VWAP position - 15% weight
    - Volume Profile (HVN/LVN) - 12% weight
    - Chaikin Money Flow - 10% weight
    - A/D Line (Accumulation/Distribution) - 10% weight
    - Force Index - 8% weight
    - Ease of Movement - 5% weight
    - Taker Flow (Buy/Sell ratio) - 5% weight
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize Volume Indicator Aggregator"""
        self.config = config or {}
        
        # Individual indicator weights (total = 1.0)
        self.indicator_weights = {
            'cvd': 0.20,  # Cumulative Volume Delta
            'obv': 0.15,  # On Balance Volume
            'vwap': 0.15,  # VWAP position
            'volume_profile': 0.12,  # Volume Profile
            'chaikin_mf': 0.10,  # Chaikin Money Flow
            'ad_line': 0.10,  # Accumulation/Distribution
            'force_index': 0.08,  # Force Index
            'ease_of_movement': 0.05,  # Ease of Movement
            'taker_flow': 0.05  # Taker Buy/Sell Ratio
        }
        
        # Thresholds
        self.accumulation_threshold = 0.55
        self.distribution_threshold = 0.45
        
        # Statistics
        self.stats = {
            'total_aggregations': 0,
            'accumulation_signals': 0,
            'distribution_signals': 0,
            'neutral_signals': 0,
            'avg_confidence': 0.0,
            'avg_calculation_time_ms': 0.0
        }
        
        logger.info("✅ Volume Indicator Aggregator initialized")
    
    async def aggregate_volume_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        orderbook_data: Optional[Dict[str, Any]] = None
    ) -> AggregatedVolumeResult:
        """
        Aggregate all volume indicators into single score
        
        Args:
            df: DataFrame with OHLCV data
            indicators: Dict of pre-calculated indicators
            orderbook_data: Optional orderbook data for additional context
            
        Returns:
            AggregatedVolumeResult with aggregated scores
        """
        start_time = datetime.now()
        
        try:
            if len(df) < 20:
                return self._get_default_result(0.0, "Insufficient data")
            
            # Calculate individual indicator signals
            signals = {}
            
            # CVD (Cumulative Volume Delta)
            if 'cvd' in indicators:
                cvd_data = indicators['cvd']
                cvd_trend = cvd_data.get('trend', 'neutral')
                if cvd_trend == 'accumulation':
                    signals['cvd'] = 0.75
                elif cvd_trend == 'distribution':
                    signals['cvd'] = 0.25
                else:
                    signals['cvd'] = 0.5
            
            # OBV (On Balance Volume)
            if 'obv' in df.columns:
                obv_signal = self._calculate_obv_signal(df)
                signals['obv'] = obv_signal
            
            # VWAP position
            if 'vwap' in df.columns:
                vwap_signal = self._calculate_vwap_signal(df)
                signals['vwap'] = vwap_signal
            
            # Volume Profile
            if 'volume_profile' in indicators:
                vp_signal = self._calculate_volume_profile_signal(indicators['volume_profile'])
                signals['volume_profile'] = vp_signal
            
            # Chaikin Money Flow
            if 'chaikin_mf' in indicators:
                cmf_value = indicators['chaikin_mf'].get('value', 0)
                # CMF ranges from -1 to 1
                signals['chaikin_mf'] = (cmf_value + 1) / 2  # Normalize to 0-1
            
            # A/D Line (Accumulation/Distribution)
            if 'ad_line' in indicators:
                ad_signal = self._calculate_ad_line_signal(indicators['ad_line'])
                signals['ad_line'] = ad_signal
            
            # Force Index
            if 'force_index' in indicators:
                force_signal = self._calculate_force_index_signal(indicators['force_index'])
                signals['force_index'] = force_signal
            
            # Ease of Movement
            if 'ease_of_movement' in indicators:
                emv_signal = self._calculate_emv_signal(indicators['ease_of_movement'])
                signals['ease_of_movement'] = emv_signal
            
            # Taker Flow (Buy/Sell Ratio)
            if 'taker_flow' in indicators:
                taker_data = indicators['taker_flow']
                buy_ratio = taker_data.get('buy_ratio', 0.5)
                signals['taker_flow'] = buy_ratio
            
            # Calculate weighted volume score
            volume_score = 0.0
            total_weight = 0.0
            
            for indicator, weight in self.indicator_weights.items():
                if indicator in signals:
                    volume_score += signals[indicator] * weight
                    total_weight += weight
            
            # Normalize by actual weights used
            if total_weight > 0:
                volume_score = volume_score / total_weight
            else:
                volume_score = 0.5  # Neutral
            
            # Calculate accumulation vs distribution scores
            accumulation_score = volume_score
            distribution_score = 1.0 - volume_score
            
            # Determine smart money flow
            if volume_score >= self.accumulation_threshold:
                smart_money_flow = "accumulating"
            elif volume_score <= self.distribution_threshold:
                smart_money_flow = "distributing"
            else:
                smart_money_flow = "neutral"
            
            # Calculate confidence
            confidence = self._calculate_confidence(volume_score, signals)
            
            # Build reasoning
            reasoning = self._build_reasoning(
                smart_money_flow,
                volume_score,
                accumulation_score,
                distribution_score,
                signals
            )
            
            # Get contributing indicators
            contributing_indicators = [
                name for name, signal in signals.items()
                if abs(signal - 0.5) > 0.15  # Contributing if not neutral
            ]
            
            calc_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # Update stats
            self._update_stats(smart_money_flow, confidence, calc_time)
            
            return AggregatedVolumeResult(
                volume_score=volume_score,
                accumulation_score=accumulation_score,
                distribution_score=distribution_score,
                smart_money_flow=smart_money_flow,
                confidence=confidence,
                contributing_indicators=contributing_indicators,
                indicator_signals=signals,
                reasoning=reasoning,
                calculation_time_ms=calc_time
            )
            
        except Exception as e:
            logger.error(f"❌ Error aggregating volume signals: {e}")
            calc_time = (datetime.now() - start_time).total_seconds() * 1000
            return self._get_default_result(calc_time, f"Error: {str(e)}")
    
    def _calculate_obv_signal(self, df: pd.DataFrame) -> float:
        """Calculate OBV trend signal"""
        try:
            obv = df['obv'].iloc[-20:]  # Last 20 periods
            
            # Calculate OBV trend
            obv_sma = obv.rolling(window=5).mean()
            current_obv = obv.iloc[-1]
            current_sma = obv_sma.iloc[-1]
            
            # OBV above its SMA = accumulation
            if current_obv > current_sma:
                # Check strength
                ratio = current_obv / current_sma if current_sma != 0 else 1.0
                signal = 0.5 + min((ratio - 1.0) * 5, 0.5)  # 0.5 to 1.0
            else:
                # OBV below its SMA = distribution
                ratio = current_sma / current_obv if current_obv != 0 else 1.0
                signal = 0.5 - min((ratio - 1.0) * 5, 0.5)  # 0.0 to 0.5
            
            return np.clip(signal, 0, 1)
            
        except Exception as e:
            logger.error(f"Error calculating OBV signal: {e}")
            return 0.5
    
    def _calculate_vwap_signal(self, df: pd.DataFrame) -> float:
        """Calculate VWAP position signal"""
        try:
            close = df['close'].iloc[-1]
            vwap = df['vwap'].iloc[-1]
            
            # Price above VWAP = accumulation zone
            # Price below VWAP = distribution zone
            diff_pct = ((close - vwap) / vwap * 100) if vwap > 0 else 0
            
            # Normalize: +2% = 1.0 (strong accumulation), -2% = 0.0 (strong distribution)
            signal = 0.5 + (diff_pct / 4.0)  # ±2% maps to 0-1
            
            return np.clip(signal, 0, 1)
            
        except Exception as e:
            logger.error(f"Error calculating VWAP signal: {e}")
            return 0.5
    
    def _calculate_volume_profile_signal(self, vp_data: Dict[str, Any]) -> float:
        """Calculate Volume Profile signal"""
        try:
            # Check if price is near high volume nodes (HVN) or low volume nodes (LVN)
            price_at_hvn = vp_data.get('price_at_hvn', False)
            price_at_lvn = vp_data.get('price_at_lvn', False)
            
            if price_at_hvn:
                # Price at HVN = support/resistance, likely consolidation
                return 0.5  # Neutral
            elif price_at_lvn:
                # Price at LVN = breakout potential, check volume
                volume_trend = vp_data.get('volume_trend', 'neutral')
                if volume_trend == 'increasing':
                    return 0.7  # Accumulation
                else:
                    return 0.3  # Distribution
            else:
                return 0.5  # Neutral
                
        except Exception as e:
            logger.error(f"Error calculating Volume Profile signal: {e}")
            return 0.5
    
    def _calculate_ad_line_signal(self, ad_data: Dict[str, Any]) -> float:
        """Calculate A/D Line signal"""
        try:
            ad_trend = ad_data.get('trend', 'neutral')
            ad_divergence = ad_data.get('divergence', False)
            
            if ad_trend == 'accumulation':
                signal = 0.7
            elif ad_trend == 'distribution':
                signal = 0.3
            else:
                signal = 0.5
            
            # Adjust for divergence
            if ad_divergence:
                # Divergence = potential reversal, moderate signal
                signal = 0.5 + (signal - 0.5) * 0.5
            
            return signal
            
        except Exception as e:
            logger.error(f"Error calculating A/D Line signal: {e}")
            return 0.5
    
    def _calculate_force_index_signal(self, force_data: Dict[str, Any]) -> float:
        """Calculate Force Index signal"""
        try:
            force_value = force_data.get('value', 0)
            force_ma = force_data.get('moving_average', 0)
            
            # Positive force = buying pressure
            # Negative force = selling pressure
            if force_value > force_ma:
                signal = 0.6
            elif force_value < force_ma:
                signal = 0.4
            else:
                signal = 0.5
            
            return signal
            
        except Exception as e:
            logger.error(f"Error calculating Force Index signal: {e}")
            return 0.5
    
    def _calculate_emv_signal(self, emv_data: Dict[str, Any]) -> float:
        """Calculate Ease of Movement signal"""
        try:
            emv_value = emv_data.get('value', 0)
            
            # Positive EMV = easy to move price up (accumulation)
            # Negative EMV = easy to move price down (distribution)
            # Normalize around typical EMV range
            signal = 0.5 + np.tanh(emv_value) * 0.5  # Maps to 0-1
            
            return np.clip(signal, 0, 1)
            
        except Exception as e:
            logger.error(f"Error calculating EMV signal: {e}")
            return 0.5
    
    def _calculate_confidence(
        self,
        volume_score: float,
        signals: Dict[str, float]
    ) -> float:
        """Calculate confidence based on signal agreement"""
        if not signals:
            return 0.0
        
        # Calculate how much signals agree
        signal_values = list(signals.values())
        std = np.std(signal_values)
        
        # Calculate distance from neutral
        distance_from_neutral = abs(volume_score - 0.5)
        
        # Lower std = higher agreement, higher distance = stronger signal
        agreement_confidence = 1.0 - (std / 0.5)  # 0-1
        strength_confidence = distance_from_neutral * 2  # 0-1
        
        confidence = (agreement_confidence * 0.6 + strength_confidence * 0.4)
        
        return np.clip(confidence, 0, 1)
    
    def _build_reasoning(
        self,
        smart_money_flow: str,
        volume_score: float,
        accumulation_score: float,
        distribution_score: float,
        signals: Dict[str, float]
    ) -> str:
        """Build human-readable reasoning string"""
        reasons = []
        
        # Overall flow
        reasons.append(f"Smart Money: {smart_money_flow.upper()} (score: {volume_score:.3f})")
        
        # Accumulation vs Distribution
        reasons.append(f"Accumulation: {accumulation_score:.3f}, Distribution: {distribution_score:.3f}")
        
        # Top contributing signals
        sorted_signals = sorted(signals.items(), key=lambda x: abs(x[1] - 0.5), reverse=True)
        
        top_signals = sorted_signals[:5]  # Top 5 most significant
        if top_signals:
            signal_str = ", ".join([f"{name}:{val:.2f}" for name, val in top_signals])
            reasons.append(f"Key volume indicators: {signal_str}")
        
        return "; ".join(reasons)
    
    def _update_stats(self, smart_money_flow: str, confidence: float, calc_time: float):
        """Update aggregation statistics"""
        self.stats['total_aggregations'] += 1
        
        if smart_money_flow == 'accumulating':
            self.stats['accumulation_signals'] += 1
        elif smart_money_flow == 'distributing':
            self.stats['distribution_signals'] += 1
        else:
            self.stats['neutral_signals'] += 1
        
        # Running average for confidence
        n = self.stats['total_aggregations']
        self.stats['avg_confidence'] = (
            (self.stats['avg_confidence'] * (n - 1) + confidence) / n
        )
        
        # Running average for calc time
        self.stats['avg_calculation_time_ms'] = (
            (self.stats['avg_calculation_time_ms'] * (n - 1) + calc_time) / n
        )
    
    def _get_default_result(self, calc_time: float, reason: str) -> AggregatedVolumeResult:
        """Return default neutral result"""
        return AggregatedVolumeResult(
            volume_score=0.5,
            accumulation_score=0.5,
            distribution_score=0.5,
            smart_money_flow="neutral",
            confidence=0.0,
            contributing_indicators=[],
            indicator_signals={},
            reasoning=reason,
            calculation_time_ms=calc_time
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get aggregation statistics"""
        return {
            **self.stats,
            'accumulation_rate': (
                self.stats['accumulation_signals'] / max(1, self.stats['total_aggregations'])
            ),
            'distribution_rate': (
                self.stats['distribution_signals'] / max(1, self.stats['total_aggregations'])
            ),
            'neutral_rate': (
                self.stats['neutral_signals'] / max(1, self.stats['total_aggregations'])
            )
        }

