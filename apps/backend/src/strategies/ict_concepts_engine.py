"""
ICT (Inner Circle Trader) Concepts Engine for AlphaPulse
Implements OTE zones, Judas Swings, Balanced Price Range, and ICT-specific analysis
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import asyncio

logger = logging.getLogger(__name__)

class ICTConceptType(Enum):
    """ICT concept types"""
    OTE_ZONE = "ote_zone"
    BALANCED_PRICE_RANGE = "balanced_price_range"
    JUDAS_SWING = "judas_swing"
    FAIR_VALUE_GAP = "fair_value_gap"
    LIQUIDITY_VOID = "liquidity_void"

@dataclass
class OTEZone:
    """Optimal Trade Entry Zone (0.62-0.79 Fibonacci retracement)"""
    symbol: str
    timestamp: datetime
    timeframe: str
    zone_type: str  # 'bullish' or 'bearish'
    swing_high: float
    swing_low: float
    ote_low: float   # 0.62 retracement
    ote_high: float  # 0.79 retracement
    current_price: float
    is_price_in_zone: bool
    distance_to_zone: float
    confidence: float
    metadata: Dict[str, Any]

@dataclass
class BalancedPriceRange:
    """Balanced Price Range (50% equilibrium)"""
    symbol: str
    timestamp: datetime
    timeframe: str
    range_high: float
    range_low: float
    equilibrium: float  # 50% level
    current_price: float
    distance_to_equilibrium: float
    is_near_equilibrium: bool
    confidence: float
    metadata: Dict[str, Any]

@dataclass
class JudasSwing:
    """Judas Swing (false move before reversal)"""
    symbol: str
    timestamp: datetime
    timeframe: str
    swing_type: str  # 'bullish' or 'bearish'
    fake_breakout_price: float
    reversal_price: float
    asian_session_range_high: float
    asian_session_range_low: float
    reversal_strength: float
    confidence: float
    metadata: Dict[str, Any]

@dataclass
class LiquiditySweep:
    """Liquidity Sweep (stop hunt before reversal)"""
    symbol: str
    timestamp: datetime
    timeframe: str
    sweep_type: str  # 'bullish' or 'bearish'
    swept_level: float  # The key level that was swept
    sweep_high: float  # Highest point of sweep
    sweep_low: float  # Lowest point of sweep
    reversal_price: float
    volume_spike: bool
    reversal_strength: float
    confidence: float
    metadata: Dict[str, Any]

@dataclass
class ICTAnalysis:
    """Complete ICT concepts analysis"""
    symbol: str
    timeframe: str
    timestamp: datetime
    ote_zones: List[OTEZone]
    balanced_price_ranges: List[BalancedPriceRange]
    judas_swings: List[JudasSwing]
    liquidity_sweeps: List[LiquiditySweep]
    overall_confidence: float
    ict_signals: List[Dict[str, Any]]
    metadata: Dict[str, Any]

class ICTConceptsEngine:
    """
    ICT (Inner Circle Trader) Concepts Analysis Engine
    
    Implements professional ICT concepts:
    - Optimal Trade Entry (OTE) zones
    - Balanced Price Range (BPR)
    - Judas Swings
    - Enhanced Fair Value Gaps (ICT-style)
    - Liquidity Voids
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logger
        
        # Configuration
        self.ote_fib_low = self.config.get('ote_fib_low', 0.62)
        self.ote_fib_high = self.config.get('ote_fib_high', 0.79)
        self.lookback_periods = self.config.get('lookback_periods', 100)
        self.min_confidence = self.config.get('min_confidence', 0.6)
        self.equilibrium_threshold = self.config.get('equilibrium_threshold', 0.02)  # 2% threshold
        
        # Performance tracking
        self.stats = {
            'ote_zones_detected': 0,
            'bpr_detected': 0,
            'judas_swings_detected': 0,
            'analyses_performed': 0,
            'last_update': datetime.now()
        }
        
        logger.info("🚀 ICT Concepts Engine initialized")
    
    async def analyze_ict_concepts(
        self, 
        df: pd.DataFrame, 
        symbol: str, 
        timeframe: str
    ) -> ICTAnalysis:
        """
        Analyze ICT concepts for given data
        
        Args:
            df: DataFrame with OHLCV data
            symbol: Trading symbol
            timeframe: Timeframe
            
        Returns:
            ICTAnalysis with all ICT concepts
        """
        try:
            if len(df) < self.lookback_periods:
                self.logger.warning(
                    f"Insufficient data for ICT analysis: {len(df)} < {self.lookback_periods}"
                )
                return self._get_default_analysis(symbol, timeframe)
            
            # Detect OTE zones
            ote_zones = await self._detect_ote_zones(df, symbol, timeframe)
            
            # Detect Balanced Price Ranges
            bpr_list = await self._detect_balanced_price_ranges(df, symbol, timeframe)
            
            # Detect Judas Swings
            judas_swings = await self._detect_judas_swings(df, symbol, timeframe)
            
            # Detect Liquidity Sweeps (CRITICAL - stop hunts)
            liquidity_sweeps = await self._detect_liquidity_sweeps(df, symbol, timeframe)
            
            # Generate ICT signals
            ict_signals = await self._generate_ict_signals(
                ote_zones, bpr_list, judas_swings, liquidity_sweeps, df
            )
            
            # Calculate overall confidence
            overall_confidence = await self._calculate_overall_confidence(
                ote_zones, bpr_list, judas_swings, liquidity_sweeps
            )
            
            # Create analysis result
            analysis = ICTAnalysis(
                symbol=symbol,
                timeframe=timeframe,
                timestamp=df['timestamp'].iloc[-1] if 'timestamp' in df.columns else datetime.now(),
                ote_zones=ote_zones,
                balanced_price_ranges=bpr_list,
                judas_swings=judas_swings,
                liquidity_sweeps=liquidity_sweeps,
                overall_confidence=overall_confidence,
                ict_signals=ict_signals,
                metadata={
                    'analysis_version': '1.0',
                    'config': self.config,
                    'stats': self.stats
                }
            )
            
            # Update statistics
            self.stats['ote_zones_detected'] += len(ote_zones)
            self.stats['bpr_detected'] += len(bpr_list)
            self.stats['judas_swings_detected'] += len(judas_swings)
            self.stats['analyses_performed'] += 1
            self.stats['last_update'] = datetime.now()
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"❌ Error analyzing ICT concepts for {symbol}: {e}")
            return self._get_default_analysis(symbol, timeframe)
    
    async def _detect_ote_zones(
        self, 
        df: pd.DataFrame, 
        symbol: str, 
        timeframe: str
    ) -> List[OTEZone]:
        """
        Detect Optimal Trade Entry (OTE) zones
        
        OTE is the 0.62-0.79 Fibonacci retracement zone of a significant move
        """
        ote_zones = []
        
        try:
            if len(df) < 20:
                return ote_zones
            
            # Find significant swing highs and lows
            swing_high_indices = self._find_swing_highs(df)
            swing_low_indices = self._find_swing_lows(df)
            
            current_price = df['close'].iloc[-1]
            current_timestamp = df['timestamp'].iloc[-1] if 'timestamp' in df.columns else datetime.now()
            
            # Check recent swings for OTE zones
            for i in range(max(0, len(df) - 50), len(df)):
                if i in swing_high_indices and i - 20 >= 0:
                    # Find corresponding swing low before this high
                    swing_low_idx = self._find_nearest_swing_low_before(
                        swing_low_indices, i
                    )
                    
                    if swing_low_idx is not None:
                        swing_high = df['high'].iloc[i]
                        swing_low = df['low'].iloc[swing_low_idx]
                        
                        # Calculate OTE zone (bullish - expecting retracement then up)
                        ote_zone = self._calculate_ote_zone(
                            swing_high, swing_low, 'bullish'
                        )
                        
                        is_in_zone = ote_zone['low'] <= current_price <= ote_zone['high']
                        distance = self._calculate_distance_to_zone(
                            current_price, ote_zone['low'], ote_zone['high']
                        )
                        
                        confidence = self._calculate_ote_confidence(
                            is_in_zone, distance, swing_high - swing_low, swing_high
                        )
                        
                        if confidence >= self.min_confidence:
                            ote = OTEZone(
                                symbol=symbol,
                                timestamp=current_timestamp,
                                timeframe=timeframe,
                                zone_type='bullish',
                                swing_high=swing_high,
                                swing_low=swing_low,
                                ote_low=ote_zone['low'],
                                ote_high=ote_zone['high'],
                                current_price=current_price,
                                is_price_in_zone=is_in_zone,
                                distance_to_zone=distance,
                                confidence=confidence,
                                metadata={
                                    'swing_high_idx': i,
                                    'swing_low_idx': swing_low_idx,
                                    'range_size': swing_high - swing_low,
                                    'fib_62': ote_zone['low'],
                                    'fib_79': ote_zone['high']
                                }
                            )
                            ote_zones.append(ote)
                
                if i in swing_low_indices and i - 20 >= 0:
                    # Find corresponding swing high before this low
                    swing_high_idx = self._find_nearest_swing_high_before(
                        swing_high_indices, i
                    )
                    
                    if swing_high_idx is not None:
                        swing_low = df['low'].iloc[i]
                        swing_high = df['high'].iloc[swing_high_idx]
                        
                        # Calculate OTE zone (bearish - expecting retracement then down)
                        ote_zone = self._calculate_ote_zone(
                            swing_high, swing_low, 'bearish'
                        )
                        
                        is_in_zone = ote_zone['low'] <= current_price <= ote_zone['high']
                        distance = self._calculate_distance_to_zone(
                            current_price, ote_zone['low'], ote_zone['high']
                        )
                        
                        confidence = self._calculate_ote_confidence(
                            is_in_zone, distance, swing_high - swing_low, swing_high
                        )
                        
                        if confidence >= self.min_confidence:
                            ote = OTEZone(
                                symbol=symbol,
                                timestamp=current_timestamp,
                                timeframe=timeframe,
                                zone_type='bearish',
                                swing_high=swing_high,
                                swing_low=swing_low,
                                ote_low=ote_zone['low'],
                                ote_high=ote_zone['high'],
                                current_price=current_price,
                                is_price_in_zone=is_in_zone,
                                distance_to_zone=distance,
                                confidence=confidence,
                                metadata={
                                    'swing_high_idx': swing_high_idx,
                                    'swing_low_idx': i,
                                    'range_size': swing_high - swing_low,
                                    'fib_62': ote_zone['low'],
                                    'fib_79': ote_zone['high']
                                }
                            )
                            ote_zones.append(ote)
            
            # Keep only the most relevant OTE zones (max 3)
            ote_zones.sort(key=lambda x: x.confidence, reverse=True)
            ote_zones = ote_zones[:3]
            
            self.logger.info(f"📊 Detected {len(ote_zones)} OTE zones for {symbol}")
            return ote_zones
            
        except Exception as e:
            self.logger.error(f"❌ Error detecting OTE zones: {e}")
            return ote_zones
    
    def _calculate_ote_zone(
        self, 
        swing_high: float, 
        swing_low: float, 
        zone_type: str
    ) -> Dict[str, float]:
        """Calculate OTE zone levels (0.62-0.79 Fibonacci)"""
        range_size = swing_high - swing_low
        
        if zone_type == 'bullish':
            # For bullish OTE: retracement from high to low, then expect move up
            fib_62_level = swing_high - (range_size * self.ote_fib_low)
            fib_79_level = swing_high - (range_size * self.ote_fib_high)
        else:  # bearish
            # For bearish OTE: retracement from low to high, then expect move down
            fib_62_level = swing_low + (range_size * self.ote_fib_low)
            fib_79_level = swing_low + (range_size * self.ote_fib_high)
        
        return {
            'low': min(fib_62_level, fib_79_level),
            'high': max(fib_62_level, fib_79_level)
        }
    
    def _calculate_distance_to_zone(
        self, 
        current_price: float, 
        zone_low: float, 
        zone_high: float
    ) -> float:
        """Calculate normalized distance to OTE zone"""
        if zone_low <= current_price <= zone_high:
            return 0.0  # Price is in zone
        elif current_price < zone_low:
            return (zone_low - current_price) / current_price
        else:  # current_price > zone_high
            return (current_price - zone_high) / current_price
    
    def _calculate_ote_confidence(
        self, 
        is_in_zone: bool, 
        distance: float, 
        range_size: float, 
        swing_high: float
    ) -> float:
        """Calculate OTE zone confidence"""
        # Base confidence
        confidence = 0.5
        
        # Bonus if price is in zone
        if is_in_zone:
            confidence += 0.3
        else:
            # Reduce confidence based on distance
            confidence -= min(0.3, distance * 10)
        
        # Bonus for significant range
        range_percent = (range_size / swing_high) * 100
        if range_percent > 5:  # >5% move
            confidence += 0.2
        elif range_percent > 3:  # >3% move
            confidence += 0.1
        
        return min(1.0, max(0.0, confidence))
    
    async def _detect_balanced_price_ranges(
        self, 
        df: pd.DataFrame, 
        symbol: str, 
        timeframe: str
    ) -> List[BalancedPriceRange]:
        """
        Detect Balanced Price Ranges (50% equilibrium levels)
        
        BPR is the midpoint of a significant range - equilibrium pricing
        """
        bpr_list = []
        
        try:
            if len(df) < 20:
                return bpr_list
            
            current_price = df['close'].iloc[-1]
            current_timestamp = df['timestamp'].iloc[-1] if 'timestamp' in df.columns else datetime.now()
            
            # Find significant ranges in recent data
            for lookback in [20, 50, 100]:
                if len(df) < lookback:
                    continue
                
                recent_data = df.tail(lookback)
                range_high = recent_data['high'].max()
                range_low = recent_data['low'].min()
                equilibrium = (range_high + range_low) / 2
                
                # Calculate distance to equilibrium
                distance = abs(current_price - equilibrium) / current_price
                is_near_equilibrium = distance <= self.equilibrium_threshold
                
                # Calculate confidence
                range_size = range_high - range_low
                range_percent = (range_size / range_high) * 100
                
                confidence = 0.5
                if is_near_equilibrium:
                    confidence += 0.3
                else:
                    confidence -= min(0.3, distance * 5)
                
                if range_percent > 5:
                    confidence += 0.2
                
                if confidence >= self.min_confidence:
                    bpr = BalancedPriceRange(
                        symbol=symbol,
                        timestamp=current_timestamp,
                        timeframe=timeframe,
                        range_high=range_high,
                        range_low=range_low,
                        equilibrium=equilibrium,
                        current_price=current_price,
                        distance_to_equilibrium=distance,
                        is_near_equilibrium=is_near_equilibrium,
                        confidence=confidence,
                        metadata={
                            'lookback_periods': lookback,
                            'range_size': range_size,
                            'range_percent': range_percent
                        }
                    )
                    bpr_list.append(bpr)
            
            # Keep most confident BPR
            bpr_list.sort(key=lambda x: x.confidence, reverse=True)
            bpr_list = bpr_list[:2]
            
            self.logger.info(f"📊 Detected {len(bpr_list)} BPR zones for {symbol}")
            return bpr_list
            
        except Exception as e:
            self.logger.error(f"❌ Error detecting BPR: {e}")
            return bpr_list
    
    async def _detect_judas_swings(
        self, 
        df: pd.DataFrame, 
        symbol: str, 
        timeframe: str
    ) -> List[JudasSwing]:
        """
        Detect Judas Swings (false moves before London/NY open reversal)
        
        Pattern: Asian session creates range, then fake breakout, then reversal
        """
        judas_swings = []
        
        try:
            if len(df) < 50:
                return judas_swings
            
            # Look for Judas patterns in recent data
            # This is a simplified version - in production, you'd use session detection
            
            for i in range(max(0, len(df) - 30), len(df) - 5):
                # Define "Asian range" as previous 10-20 candles
                asian_start = max(0, i - 20)
                asian_end = i
                
                if asian_end <= asian_start:
                    continue
                
                asian_range = df.iloc[asian_start:asian_end]
                asian_high = asian_range['high'].max()
                asian_low = asian_range['low'].min()
                asian_range_size = asian_high - asian_low
                
                # Check for breakout and reversal
                breakout_candles = df.iloc[i:i+5]
                
                if len(breakout_candles) < 5:
                    continue
                
                # Bullish Judas: Fake breakdown below asian low, then reversal up
                fake_low = breakout_candles['low'].min()
                if fake_low < asian_low:
                    # Check for reversal
                    reversal_high = breakout_candles['high'].iloc[-1]
                    if reversal_high > asian_low:  # Reversed back into range
                        reversal_strength = (reversal_high - fake_low) / asian_range_size
                        
                        if reversal_strength > 0.3:  # Significant reversal
                            confidence = min(0.9, 0.5 + reversal_strength)
                            
                            judas = JudasSwing(
                                symbol=symbol,
                                timestamp=df['timestamp'].iloc[i+4] if 'timestamp' in df.columns else datetime.now(),
                                timeframe=timeframe,
                                swing_type='bullish',
                                fake_breakout_price=fake_low,
                                reversal_price=reversal_high,
                                asian_session_range_high=asian_high,
                                asian_session_range_low=asian_low,
                                reversal_strength=reversal_strength,
                                confidence=confidence,
                                metadata={
                                    'asian_range_size': asian_range_size,
                                    'fake_breakout_distance': asian_low - fake_low,
                                    'reversal_distance': reversal_high - fake_low
                                }
                            )
                            judas_swings.append(judas)
                
                # Bearish Judas: Fake breakout above asian high, then reversal down
                fake_high = breakout_candles['high'].max()
                if fake_high > asian_high:
                    # Check for reversal
                    reversal_low = breakout_candles['low'].iloc[-1]
                    if reversal_low < asian_high:  # Reversed back into range
                        reversal_strength = (fake_high - reversal_low) / asian_range_size
                        
                        if reversal_strength > 0.3:  # Significant reversal
                            confidence = min(0.9, 0.5 + reversal_strength)
                            
                            judas = JudasSwing(
                                symbol=symbol,
                                timestamp=df['timestamp'].iloc[i+4] if 'timestamp' in df.columns else datetime.now(),
                                timeframe=timeframe,
                                swing_type='bearish',
                                fake_breakout_price=fake_high,
                                reversal_price=reversal_low,
                                asian_session_range_high=asian_high,
                                asian_session_range_low=asian_low,
                                reversal_strength=reversal_strength,
                                confidence=confidence,
                                metadata={
                                    'asian_range_size': asian_range_size,
                                    'fake_breakout_distance': fake_high - asian_high,
                                    'reversal_distance': fake_high - reversal_low
                                }
                            )
                            judas_swings.append(judas)
            
            # Keep most recent and confident
            judas_swings.sort(key=lambda x: x.confidence, reverse=True)
            judas_swings = judas_swings[:2]
            
            self.logger.info(f"📊 Detected {len(judas_swings)} Judas swings for {symbol}")
            return judas_swings
            
        except Exception as e:
            self.logger.error(f"❌ Error detecting Judas swings: {e}")
            return judas_swings
    
    async def _detect_liquidity_sweeps(
        self, 
        df: pd.DataFrame, 
        symbol: str, 
        timeframe: str
    ) -> List[LiquiditySweep]:
        """
        Detect liquidity sweeps (stop hunts before real moves)
        
        Pattern: Price spikes beyond key level (sweeping stops), 
        then quickly reverses with volume spike
        """
        sweeps = []
        
        try:
            if len(df) < 20:
                return sweeps
            
            # Find recent swing highs/lows (potential liquidity pools)
            swing_highs = self._find_swing_highs(df, window=5)
            swing_lows = self._find_swing_lows(df, window=5)
            
            current_price = df['close'].iloc[-1]
            current_timestamp = df['timestamp'].iloc[-1] if 'timestamp' in df.columns else datetime.now()
            
            # Check last 10 candles for sweep patterns
            for i in range(max(0, len(df) - 10), len(df)):
                candle_high = df['high'].iloc[i]
                candle_low = df['low'].iloc[i]
                candle_close = df['close'].iloc[i]
                candle_open = df['open'].iloc[i]
                
                # Bullish sweep: Price dips below swing low, then reverses up
                for swing_low_idx in swing_lows:
                    if swing_low_idx >= i - 5 and swing_low_idx < i:
                        swing_low_price = df['low'].iloc[swing_low_idx]
                        
                        # Check if current candle swept below then reversed
                        if (candle_low < swing_low_price and 
                            candle_close > swing_low_price):
                            
                            # Calculate reversal strength
                            sweep_distance = swing_low_price - candle_low
                            reversal_distance = candle_close - candle_low
                            reversal_strength = reversal_distance / sweep_distance if sweep_distance > 0 else 0
                            
                            # Check for volume spike
                            avg_volume = df['volume'].iloc[max(0, i-10):i].mean() if 'volume' in df.columns and i >= 10 else 0
                            current_volume = df['volume'].iloc[i] if 'volume' in df.columns else 0
                            volume_spike = current_volume > avg_volume * 1.5 if avg_volume > 0 else False
                            
                            if reversal_strength > 0.5:  # Strong reversal
                                confidence = min(0.9, 0.6 + reversal_strength * 0.3)
                                if volume_spike:
                                    confidence += 0.1
                                    confidence = min(0.95, confidence)
                                
                                sweep = LiquiditySweep(
                                    symbol=symbol,
                                    timestamp=current_timestamp,
                                    timeframe=timeframe,
                                    sweep_type='bullish',
                                    swept_level=swing_low_price,
                                    sweep_high=candle_high,
                                    sweep_low=candle_low,
                                    reversal_price=candle_close,
                                    volume_spike=volume_spike,
                                    reversal_strength=reversal_strength,
                                    confidence=confidence,
                                    metadata={
                                        'sweep_distance': sweep_distance,
                                        'reversal_distance': reversal_distance,
                                        'volume_ratio': current_volume / avg_volume if avg_volume > 0 else 1.0,
                                        'candle_index': i
                                    }
                                )
                                sweeps.append(sweep)
                
                # Bearish sweep: Price spikes above swing high, then reverses down
                for swing_high_idx in swing_highs:
                    if swing_high_idx >= i - 5 and swing_high_idx < i:
                        swing_high_price = df['high'].iloc[swing_high_idx]
                        
                        # Check if current candle swept above then reversed
                        if (candle_high > swing_high_price and 
                            candle_close < swing_high_price):
                            
                            # Calculate reversal strength
                            sweep_distance = candle_high - swing_high_price
                            reversal_distance = candle_high - candle_close
                            reversal_strength = reversal_distance / sweep_distance if sweep_distance > 0 else 0
                            
                            # Check for volume spike
                            avg_volume = df['volume'].iloc[max(0, i-10):i].mean() if 'volume' in df.columns and i >= 10 else 0
                            current_volume = df['volume'].iloc[i] if 'volume' in df.columns else 0
                            volume_spike = current_volume > avg_volume * 1.5 if avg_volume > 0 else False
                            
                            if reversal_strength > 0.5:  # Strong reversal
                                confidence = min(0.9, 0.6 + reversal_strength * 0.3)
                                if volume_spike:
                                    confidence += 0.1
                                    confidence = min(0.95, confidence)
                                
                                sweep = LiquiditySweep(
                                    symbol=symbol,
                                    timestamp=current_timestamp,
                                    timeframe=timeframe,
                                    sweep_type='bearish',
                                    swept_level=swing_high_price,
                                    sweep_high=candle_high,
                                    sweep_low=candle_low,
                                    reversal_price=candle_close,
                                    volume_spike=volume_spike,
                                    reversal_strength=reversal_strength,
                                    confidence=confidence,
                                    metadata={
                                        'sweep_distance': sweep_distance,
                                        'reversal_distance': reversal_distance,
                                        'volume_ratio': current_volume / avg_volume if avg_volume > 0 else 1.0,
                                        'candle_index': i
                                    }
                                )
                                sweeps.append(sweep)
            
            # Keep most recent and confident sweeps
            sweeps.sort(key=lambda x: x.confidence, reverse=True)
            sweeps = sweeps[:2]
            
            self.logger.info(f"📊 Detected {len(sweeps)} liquidity sweeps for {symbol}")
            return sweeps
            
        except Exception as e:
            self.logger.error(f"❌ Error detecting liquidity sweeps: {e}")
            return sweeps
    
    def _find_swing_highs(self, df: pd.DataFrame, window: int = 5) -> List[int]:
        """Find swing high indices"""
        try:
            swing_highs = []
            for i in range(window, len(df) - window):
                is_swing_high = True
                current_high = df['high'].iloc[i]
                
                # Check if this is highest in window
                for j in range(i - window, i + window + 1):
                    if j != i and df['high'].iloc[j] >= current_high:
                        is_swing_high = False
                        break
                
                if is_swing_high:
                    swing_highs.append(i)
            
            return swing_highs
        except Exception:
            return []
    
    def _find_swing_lows(self, df: pd.DataFrame, window: int = 5) -> List[int]:
        """Find swing low indices"""
        try:
            swing_lows = []
            for i in range(window, len(df) - window):
                is_swing_low = True
                current_low = df['low'].iloc[i]
                
                # Check if this is lowest in window
                for j in range(i - window, i + window + 1):
                    if j != i and df['low'].iloc[j] <= current_low:
                        is_swing_low = False
                        break
                
                if is_swing_low:
                    swing_lows.append(i)
            
            return swing_lows
        except Exception:
            return []
    
    def _find_nearest_swing_low_before(
        self, 
        swing_low_indices: List[int], 
        current_idx: int
    ) -> Optional[int]:
        """Find nearest swing low before current index"""
        valid_lows = [idx for idx in swing_low_indices if idx < current_idx]
        return valid_lows[-1] if valid_lows else None
    
    def _find_nearest_swing_high_before(
        self, 
        swing_high_indices: List[int], 
        current_idx: int
    ) -> Optional[int]:
        """Find nearest swing high before current index"""
        valid_highs = [idx for idx in swing_high_indices if idx < current_idx]
        return valid_highs[-1] if valid_highs else None
    
    async def _generate_ict_signals(
        self,
        ote_zones: List[OTEZone],
        bpr_list: List[BalancedPriceRange],
        judas_swings: List[JudasSwing],
        liquidity_sweeps: List[LiquiditySweep],
        df: pd.DataFrame
    ) -> List[Dict[str, Any]]:
        """Generate trading signals based on ICT analysis"""
        signals = []
        
        try:
            # Generate signals from OTE zones
            for ote in ote_zones:
                if ote.is_price_in_zone and ote.confidence >= self.min_confidence:
                    signal = {
                        'type': 'ote_zone',
                        'direction': ote.zone_type,
                        'confidence': ote.confidence,
                        'price_level': (ote.ote_low + ote.ote_high) / 2,
                        'zone_low': ote.ote_low,
                        'zone_high': ote.ote_high,
                        'timestamp': ote.timestamp,
                        'reasoning': f"Price in OTE zone ({ote.ote_low:.2f}-{ote.ote_high:.2f})",
                        'metadata': ote.metadata
                    }
                    signals.append(signal)
            
            # Generate signals from BPR
            for bpr in bpr_list:
                if bpr.is_near_equilibrium and bpr.confidence >= self.min_confidence:
                    signal = {
                        'type': 'balanced_price_range',
                        'direction': 'neutral',  # BPR is a decision point
                        'confidence': bpr.confidence,
                        'price_level': bpr.equilibrium,
                        'range_high': bpr.range_high,
                        'range_low': bpr.range_low,
                        'timestamp': bpr.timestamp,
                        'reasoning': f"Price near equilibrium ({bpr.equilibrium:.2f})",
                        'metadata': bpr.metadata
                    }
                    signals.append(signal)
            
            # Generate signals from Judas Swings
            for judas in judas_swings:
                if judas.confidence >= self.min_confidence:
                    signal = {
                        'type': 'judas_swing',
                        'direction': judas.swing_type,
                        'confidence': judas.confidence,
                        'price_level': judas.reversal_price,
                        'fake_breakout': judas.fake_breakout_price,
                        'reversal_strength': judas.reversal_strength,
                        'timestamp': judas.timestamp,
                        'reasoning': f"Judas swing reversal detected (strength: {judas.reversal_strength:.2f})",
                        'metadata': judas.metadata
                    }
                    signals.append(signal)
            
            # Generate signals from Liquidity Sweeps (CRITICAL for ICT)
            for sweep in liquidity_sweeps:
                if sweep.confidence >= self.min_confidence:
                    vol_status = "with volume spike" if sweep.volume_spike else "without volume"
                    signal = {
                        'type': 'liquidity_sweep',
                        'direction': sweep.sweep_type,
                        'confidence': sweep.confidence,
                        'price_level': sweep.reversal_price,
                        'swept_level': sweep.swept_level,
                        'reversal_strength': sweep.reversal_strength,
                        'volume_spike': sweep.volume_spike,
                        'timestamp': sweep.timestamp,
                        'reasoning': f"Liquidity sweep detected ({vol_status}, strength: {sweep.reversal_strength:.2f})",
                        'metadata': sweep.metadata
                    }
                    signals.append(signal)
            
            # Sort signals by confidence
            signals.sort(key=lambda x: x['confidence'], reverse=True)
            
            self.logger.info(f"📊 Generated {len(signals)} ICT signals")
            return signals
            
        except Exception as e:
            self.logger.error(f"❌ Error generating ICT signals: {e}")
            return signals
    
    async def _calculate_overall_confidence(
        self,
        ote_zones: List[OTEZone],
        bpr_list: List[BalancedPriceRange],
        judas_swings: List[JudasSwing],
        liquidity_sweeps: List[LiquiditySweep]
    ) -> float:
        """Calculate overall confidence for ICT analysis"""
        try:
            total_confidence = 0.0
            total_weight = 0.0
            
            # Weight different ICT components (liquidity sweeps are highly valuable)
            weights = {
                'ote_zones': 0.3,
                'bpr': 0.2,
                'judas_swings': 0.25,
                'liquidity_sweeps': 0.25  # Sweeps are critical indicators
            }
            
            # Calculate weighted confidence
            if ote_zones:
                avg_ote_confidence = sum(ote.confidence for ote in ote_zones) / len(ote_zones)
                total_confidence += avg_ote_confidence * weights['ote_zones']
                total_weight += weights['ote_zones']
            
            if bpr_list:
                avg_bpr_confidence = sum(bpr.confidence for bpr in bpr_list) / len(bpr_list)
                total_confidence += avg_bpr_confidence * weights['bpr']
                total_weight += weights['bpr']
            
            if judas_swings:
                avg_judas_confidence = sum(js.confidence for js in judas_swings) / len(judas_swings)
                total_confidence += avg_judas_confidence * weights['judas_swings']
                total_weight += weights['judas_swings']
            
            if liquidity_sweeps:
                avg_sweep_confidence = sum(sweep.confidence for sweep in liquidity_sweeps) / len(liquidity_sweeps)
                total_confidence += avg_sweep_confidence * weights['liquidity_sweeps']
                total_weight += weights['liquidity_sweeps']
            
            # Return weighted average confidence
            return total_confidence / total_weight if total_weight > 0 else 0.0
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating overall confidence: {e}")
            return 0.0
    
    def _get_default_analysis(self, symbol: str, timeframe: str) -> ICTAnalysis:
        """Get default analysis when insufficient data"""
        return ICTAnalysis(
            symbol=symbol,
            timeframe=timeframe,
            timestamp=datetime.now(),
            ote_zones=[],
            balanced_price_ranges=[],
            judas_swings=[],
            liquidity_sweeps=[],
            overall_confidence=0.0,
            ict_signals=[],
            metadata={'error': 'Insufficient data for analysis'}
        )
    
    async def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        return {
            'stats': self.stats,
            'config': self.config,
            'last_update': datetime.now().isoformat()
        }

