"""
Enhanced Market Structure Engine for AlphaPulse
Implements multi-timeframe alignment, premium/discount zones, mitigation blocks, and breaker blocks
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

class PriceZone(Enum):
    """Price zone relative to range"""
    PREMIUM = "premium"  # 50-100%
    EQUILIBRIUM = "equilibrium"  # 45-55%
    DISCOUNT = "discount"  # 0-50%
    UNKNOWN = "unknown"

class StructureBreak(Enum):
    """Structure break types"""
    BOS = "break_of_structure"  # Continuation
    CHOCH = "change_of_character"  # Reversal
    NONE = "none"

@dataclass
class TimeframeStructure:
    """Market structure for a single timeframe"""
    timeframe: str
    trend: str  # 'bullish', 'bearish', 'neutral'
    last_swing_high: float
    last_swing_low: float
    structure_break: StructureBreak
    break_price: Optional[float]
    confidence: float
    metadata: Dict[str, Any]

@dataclass
class PremiumDiscountZone:
    """Premium/Discount zone identification"""
    symbol: str
    timeframe: str
    timestamp: datetime
    range_high: float
    range_low: float
    equilibrium: float  # 50% level
    premium_25: float  # 75% level
    discount_25: float  # 25% level
    current_price: float
    current_zone: PriceZone
    distance_to_equilibrium: float
    confidence: float
    metadata: Dict[str, Any]

@dataclass
class MitigationBlock:
    """Unmitigated order block"""
    symbol: str
    timeframe: str
    timestamp: datetime
    block_type: str  # 'bullish' or 'bearish'
    high: float
    low: float
    is_mitigated: bool
    mitigation_time: Optional[datetime]
    mitigation_count: int  # How many times price returned
    strength: float
    confidence: float
    metadata: Dict[str, Any]

@dataclass
class BreakerBlock:
    """Failed order block that flipped polarity"""
    symbol: str
    timeframe: str
    timestamp: datetime
    original_type: str  # Original polarity
    breaker_type: str  # New polarity (opposite)
    high: float
    low: float
    break_timestamp: datetime
    retest_count: int
    strength: float
    confidence: float
    metadata: Dict[str, Any]

@dataclass
class MultiTimeframeAlignment:
    """Multi-timeframe structure alignment"""
    symbol: str
    timestamp: datetime
    timeframes: List[str]
    structures: Dict[str, TimeframeStructure]
    aligned: bool
    alignment_direction: str  # 'bullish', 'bearish', 'neutral'
    alignment_score: float  # 0-1
    confidence: float
    metadata: Dict[str, Any]

@dataclass
class EnhancedStructureAnalysis:
    """Complete enhanced market structure analysis"""
    symbol: str
    timeframe: str
    timestamp: datetime
    mtf_alignment: MultiTimeframeAlignment
    premium_discount: PremiumDiscountZone
    mitigation_blocks: List[MitigationBlock]
    breaker_blocks: List[BreakerBlock]
    overall_confidence: float
    structure_signals: List[Dict[str, Any]]
    metadata: Dict[str, Any]

class EnhancedMarketStructureEngine:
    """
    Enhanced Market Structure Analysis Engine
    
    Implements:
    - Multi-timeframe structure alignment
    - Premium/Discount zones
    - Inducement patterns
    - Mitigation blocks
    - Breaker blocks
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logger
        
        # Configuration
        self.timeframes = self.config.get('timeframes', ['5m', '15m', '1h', '4h'])
        self.lookback_periods = self.config.get('lookback_periods', 100)
        self.min_confidence = self.config.get('min_confidence', 0.6)
        self.equilibrium_threshold = self.config.get('equilibrium_threshold', 0.05)  # ±5%
        
        # Performance tracking
        self.stats = {
            'analyses_performed': 0,
            'mtf_alignments_detected': 0,
            'mitigation_blocks_tracked': 0,
            'breaker_blocks_detected': 0,
            'last_update': datetime.now()
        }
        
        logger.info("🚀 Enhanced Market Structure Engine initialized")
    
    async def analyze_enhanced_structure(
        self, 
        df: pd.DataFrame, 
        symbol: str, 
        timeframe: str,
        multi_tf_data: Optional[Dict[str, pd.DataFrame]] = None
    ) -> EnhancedStructureAnalysis:
        """
        Complete enhanced market structure analysis
        
        Args:
            df: Primary timeframe DataFrame with OHLCV data
            symbol: Trading symbol
            timeframe: Primary timeframe
            multi_tf_data: Optional dict of DataFrames for other timeframes
            
        Returns:
            EnhancedStructureAnalysis with complete structure analysis
        """
        try:
            if len(df) < self.lookback_periods:
                self.logger.warning(
                    f"Insufficient data for structure analysis: {len(df)} < {self.lookback_periods}"
                )
                return self._get_default_analysis(symbol, timeframe)
            
            # Multi-timeframe alignment analysis
            mtf_alignment = await self._analyze_mtf_alignment(
                df, symbol, timeframe, multi_tf_data
            )
            
            # Premium/Discount zone analysis
            premium_discount = await self._analyze_premium_discount(
                df, symbol, timeframe
            )
            
            # Mitigation blocks detection
            mitigation_blocks = await self._detect_mitigation_blocks(
                df, symbol, timeframe
            )
            
            # Breaker blocks detection
            breaker_blocks = await self._detect_breaker_blocks(
                df, symbol, timeframe, mitigation_blocks
            )
            
            # Generate structure signals
            structure_signals = await self._generate_structure_signals(
                mtf_alignment, premium_discount, mitigation_blocks, breaker_blocks
            )
            
            # Calculate overall confidence
            overall_confidence = await self._calculate_overall_confidence(
                mtf_alignment, premium_discount, mitigation_blocks, breaker_blocks
            )
            
            # Create analysis result
            analysis = EnhancedStructureAnalysis(
                symbol=symbol,
                timeframe=timeframe,
                timestamp=df['timestamp'].iloc[-1] if 'timestamp' in df.columns else datetime.now(),
                mtf_alignment=mtf_alignment,
                premium_discount=premium_discount,
                mitigation_blocks=mitigation_blocks,
                breaker_blocks=breaker_blocks,
                overall_confidence=overall_confidence,
                structure_signals=structure_signals,
                metadata={
                    'analysis_version': '1.0',
                    'config': self.config,
                    'stats': self.stats
                }
            )
            
            # Update statistics
            self.stats['analyses_performed'] += 1
            if mtf_alignment.aligned:
                self.stats['mtf_alignments_detected'] += 1
            self.stats['mitigation_blocks_tracked'] += len(mitigation_blocks)
            self.stats['breaker_blocks_detected'] += len(breaker_blocks)
            self.stats['last_update'] = datetime.now()
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"❌ Error in enhanced structure analysis for {symbol}: {e}")
            return self._get_default_analysis(symbol, timeframe)
    
    async def _analyze_mtf_alignment(
        self,
        df: pd.DataFrame,
        symbol: str,
        primary_timeframe: str,
        multi_tf_data: Optional[Dict[str, pd.DataFrame]]
    ) -> MultiTimeframeAlignment:
        """Analyze multi-timeframe structure alignment"""
        try:
            structures = {}
            
            # Analyze primary timeframe
            primary_structure = self._analyze_single_timeframe_structure(
                df, primary_timeframe
            )
            structures[primary_timeframe] = primary_structure
            
            # Analyze other timeframes if provided
            if multi_tf_data:
                for tf, tf_df in multi_tf_data.items():
                    if len(tf_df) >= 20:
                        tf_structure = self._analyze_single_timeframe_structure(tf_df, tf)
                        structures[tf] = tf_structure
            
            # Check alignment
            aligned = False
            alignment_direction = 'neutral'
            alignment_score = 0.0
            
            if len(structures) >= 2:
                # Count bullish/bearish trends
                bullish_count = sum(1 for s in structures.values() if s.trend == 'bullish')
                bearish_count = sum(1 for s in structures.values() if s.trend == 'bearish')
                total_count = len(structures)
                
                # Check if majority align
                if bullish_count >= total_count * 0.75:
                    aligned = True
                    alignment_direction = 'bullish'
                    alignment_score = bullish_count / total_count
                elif bearish_count >= total_count * 0.75:
                    aligned = True
                    alignment_direction = 'bearish'
                    alignment_score = bearish_count / total_count
                else:
                    alignment_score = max(bullish_count, bearish_count) / total_count
            
            # Calculate confidence
            confidence = alignment_score if aligned else 0.5
            
            return MultiTimeframeAlignment(
                symbol=symbol,
                timestamp=df['timestamp'].iloc[-1] if 'timestamp' in df.columns else datetime.now(),
                timeframes=list(structures.keys()),
                structures=structures,
                aligned=aligned,
                alignment_direction=alignment_direction,
                alignment_score=alignment_score,
                confidence=confidence,
                metadata={
                    'timeframe_count': len(structures),
                    'bullish_count': sum(1 for s in structures.values() if s.trend == 'bullish'),
                    'bearish_count': sum(1 for s in structures.values() if s.trend == 'bearish')
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error analyzing MTF alignment: {e}")
            return MultiTimeframeAlignment(
                symbol=symbol,
                timestamp=datetime.now(),
                timeframes=[],
                structures={},
                aligned=False,
                alignment_direction='neutral',
                alignment_score=0.0,
                confidence=0.0,
                metadata={}
            )
    
    def _analyze_single_timeframe_structure(
        self,
        df: pd.DataFrame,
        timeframe: str
    ) -> TimeframeStructure:
        """Analyze structure for a single timeframe"""
        try:
            # Find swing highs and lows
            swing_highs = self._find_swing_highs(df)
            swing_lows = self._find_swing_lows(df)
            
            if not swing_highs or not swing_lows:
                return TimeframeStructure(
                    timeframe=timeframe,
                    trend='neutral',
                    last_swing_high=df['high'].iloc[-1],
                    last_swing_low=df['low'].iloc[-1],
                    structure_break=StructureBreak.NONE,
                    break_price=None,
                    confidence=0.0,
                    metadata={}
                )
            
            # Get recent swings
            last_swing_high_idx = swing_highs[-1] if swing_highs else len(df) - 1
            last_swing_low_idx = swing_lows[-1] if swing_lows else len(df) - 1
            
            last_swing_high = df['high'].iloc[last_swing_high_idx]
            last_swing_low = df['low'].iloc[last_swing_low_idx]
            
            current_price = df['close'].iloc[-1]
            
            # Determine trend
            if len(swing_highs) >= 2 and len(swing_lows) >= 2:
                prev_swing_high = df['high'].iloc[swing_highs[-2]]
                prev_swing_low = df['low'].iloc[swing_lows[-2]]
                
                # Bullish: Higher highs and higher lows
                if last_swing_high > prev_swing_high and last_swing_low > prev_swing_low:
                    trend = 'bullish'
                    structure_break = StructureBreak.BOS if current_price > last_swing_high else StructureBreak.NONE
                    break_price = last_swing_high if structure_break == StructureBreak.BOS else None
                # Bearish: Lower lows and lower highs
                elif last_swing_high < prev_swing_high and last_swing_low < prev_swing_low:
                    trend = 'bearish'
                    structure_break = StructureBreak.BOS if current_price < last_swing_low else StructureBreak.NONE
                    break_price = last_swing_low if structure_break == StructureBreak.BOS else None
                # Change of character
                else:
                    trend = 'neutral'
                    structure_break = StructureBreak.CHOCH
                    break_price = current_price
            else:
                trend = 'neutral'
                structure_break = StructureBreak.NONE
                break_price = None
            
            confidence = 0.7 if structure_break != StructureBreak.NONE else 0.5
            
            return TimeframeStructure(
                timeframe=timeframe,
                trend=trend,
                last_swing_high=last_swing_high,
                last_swing_low=last_swing_low,
                structure_break=structure_break,
                break_price=break_price,
                confidence=confidence,
                metadata={
                    'swing_high_count': len(swing_highs),
                    'swing_low_count': len(swing_lows),
                    'current_price': current_price
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error analyzing single timeframe structure: {e}")
            return TimeframeStructure(
                timeframe=timeframe,
                trend='neutral',
                last_swing_high=0.0,
                last_swing_low=0.0,
                structure_break=StructureBreak.NONE,
                break_price=None,
                confidence=0.0,
                metadata={}
            )
    
    async def _analyze_premium_discount(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str
    ) -> PremiumDiscountZone:
        """Analyze premium/discount zones"""
        try:
            # Calculate range
            lookback_data = df.tail(50)
            range_high = lookback_data['high'].max()
            range_low = lookback_data['low'].min()
            range_size = range_high - range_low
            
            # Calculate key levels
            equilibrium = range_low + (range_size * 0.5)
            premium_25 = range_low + (range_size * 0.75)
            discount_25 = range_low + (range_size * 0.25)
            
            current_price = df['close'].iloc[-1]
            
            # Determine current zone
            price_pct = (current_price - range_low) / range_size if range_size > 0 else 0.5
            
            if 0.45 <= price_pct <= 0.55:
                current_zone = PriceZone.EQUILIBRIUM
            elif price_pct > 0.55:
                current_zone = PriceZone.PREMIUM
            else:
                current_zone = PriceZone.DISCOUNT
            
            # Distance to equilibrium
            distance = abs(current_price - equilibrium) / current_price if current_price > 0 else 0
            
            # Confidence
            if current_zone == PriceZone.EQUILIBRIUM:
                confidence = 0.7
            elif current_zone in [PriceZone.PREMIUM, PriceZone.DISCOUNT]:
                confidence = 0.8  # Clear zones
            else:
                confidence = 0.5
            
            return PremiumDiscountZone(
                symbol=symbol,
                timeframe=timeframe,
                timestamp=df['timestamp'].iloc[-1] if 'timestamp' in df.columns else datetime.now(),
                range_high=range_high,
                range_low=range_low,
                equilibrium=equilibrium,
                premium_25=premium_25,
                discount_25=discount_25,
                current_price=current_price,
                current_zone=current_zone,
                distance_to_equilibrium=distance,
                confidence=confidence,
                metadata={
                    'range_size': range_size,
                    'price_percentage': price_pct * 100
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error analyzing premium/discount: {e}")
            return PremiumDiscountZone(
                symbol=symbol,
                timeframe=timeframe,
                timestamp=datetime.now(),
                range_high=0.0,
                range_low=0.0,
                equilibrium=0.0,
                premium_25=0.0,
                discount_25=0.0,
                current_price=0.0,
                current_zone=PriceZone.UNKNOWN,
                distance_to_equilibrium=0.0,
                confidence=0.0,
                metadata={}
            )
    
    async def _detect_mitigation_blocks(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str
    ) -> List[MitigationBlock]:
        """Detect unmitigated order blocks"""
        mitigation_blocks = []
        
        try:
            # Look for order blocks (strong moves with consolidation)
            for i in range(10, len(df) - 5):
                # Calculate body ratio
                body = abs(df['close'].iloc[i] - df['open'].iloc[i])
                total_range = df['high'].iloc[i] - df['low'].iloc[i]
                body_ratio = body / total_range if total_range > 0 else 0
                
                # Bullish order block: Strong up move
                if (df['close'].iloc[i] > df['open'].iloc[i] and
                    body_ratio > 0.6 and
                    df['volume'].iloc[i] > df['volume'].iloc[i-5:i].mean() * 1.2):
                    
                    block_high = df['high'].iloc[i]
                    block_low = df['low'].iloc[i]
                    
                    # Check if mitigated (price returned to block)
                    future_data = df.iloc[i+1:]
                    is_mitigated = any(future_data['low'] <= block_high)
                    mitigation_time = None
                    mitigation_count = 0
                    
                    if is_mitigated:
                        mitigated_indices = future_data[future_data['low'] <= block_high].index
                        if len(mitigated_indices) > 0:
                            mitigation_time = future_data.loc[mitigated_indices[0], 'timestamp'] if 'timestamp' in future_data.columns else None
                            mitigation_count = len(mitigated_indices)
                    
                    strength = min(1.0, body_ratio * 1.5)
                    confidence = 0.7 if not is_mitigated else 0.6
                    
                    block = MitigationBlock(
                        symbol=symbol,
                        timeframe=timeframe,
                        timestamp=df['timestamp'].iloc[i] if 'timestamp' in df.columns else datetime.now(),
                        block_type='bullish',
                        high=block_high,
                        low=block_low,
                        is_mitigated=is_mitigated,
                        mitigation_time=mitigation_time,
                        mitigation_count=mitigation_count,
                        strength=strength,
                        confidence=confidence,
                        metadata={
                            'body_ratio': body_ratio,
                            'volume_ratio': df['volume'].iloc[i] / df['volume'].iloc[i-5:i].mean()
                        }
                    )
                    mitigation_blocks.append(block)
            
            # Keep recent blocks
            mitigation_blocks = mitigation_blocks[-10:]
            
            self.logger.info(f"📊 Detected {len(mitigation_blocks)} mitigation blocks for {symbol}")
            return mitigation_blocks
            
        except Exception as e:
            self.logger.error(f"Error detecting mitigation blocks: {e}")
            return mitigation_blocks
    
    async def _detect_breaker_blocks(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str,
        mitigation_blocks: List[MitigationBlock]
    ) -> List[BreakerBlock]:
        """Detect breaker blocks (failed order blocks with polarity flip)"""
        breaker_blocks = []
        
        try:
            # A breaker block is an order block that failed and flipped polarity
            for block in mitigation_blocks:
                if not block.is_mitigated:
                    continue
                
                # Check if price decisively broke through the block
                # For bullish block: if price closed below block low = breaker
                # For bearish block: if price closed above block high = breaker
                
                future_data = df[df.index > df[df['timestamp'] == block.timestamp].index[0] if 'timestamp' in df.columns else 0]
                
                if block.block_type == 'bullish':
                    # Check for close below block low (failed support)
                    breaks = future_data[future_data['close'] < block.low]
                    if len(breaks) > 0:
                        break_timestamp = breaks['timestamp'].iloc[0] if 'timestamp' in breaks.columns else datetime.now()
                        
                        # Count retests
                        retests = future_data[
                            (future_data['close'] > block.low) &
                            (future_data['close'] < block.high)
                        ]
                        retest_count = len(retests)
                        
                        breaker = BreakerBlock(
                            symbol=symbol,
                            timeframe=timeframe,
                            timestamp=block.timestamp,
                            original_type='bullish',
                            breaker_type='bearish',  # Flipped to resistance
                            high=block.high,
                            low=block.low,
                            break_timestamp=break_timestamp,
                            retest_count=retest_count,
                            strength=block.strength,
                            confidence=0.75,
                            metadata={
                                'original_block': block.metadata,
                                'flip_confirmed': retest_count > 0
                            }
                        )
                        breaker_blocks.append(breaker)
            
            self.logger.info(f"📊 Detected {len(breaker_blocks)} breaker blocks for {symbol}")
            return breaker_blocks
            
        except Exception as e:
            self.logger.error(f"Error detecting breaker blocks: {e}")
            return breaker_blocks
    
    def _find_swing_highs(self, df: pd.DataFrame, window: int = 5) -> List[int]:
        """Find swing high indices"""
        try:
            swing_highs = []
            for i in range(window, len(df) - window):
                is_swing_high = True
                current_high = df['high'].iloc[i]
                
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
                
                for j in range(i - window, i + window + 1):
                    if j != i and df['low'].iloc[j] <= current_low:
                        is_swing_low = False
                        break
                
                if is_swing_low:
                    swing_lows.append(i)
            
            return swing_lows
        except Exception:
            return []
    
    async def _generate_structure_signals(
        self,
        mtf_alignment: MultiTimeframeAlignment,
        premium_discount: PremiumDiscountZone,
        mitigation_blocks: List[MitigationBlock],
        breaker_blocks: List[BreakerBlock]
    ) -> List[Dict[str, Any]]:
        """Generate trading signals from structure analysis"""
        signals = []
        
        try:
            # MTF alignment signal
            if mtf_alignment.aligned and mtf_alignment.alignment_score > 0.75:
                signals.append({
                    'type': 'mtf_alignment',
                    'direction': mtf_alignment.alignment_direction,
                    'confidence': mtf_alignment.confidence,
                    'alignment_score': mtf_alignment.alignment_score,
                    'timeframes': len(mtf_alignment.timeframes),
                    'reasoning': f"Multi-timeframe {mtf_alignment.alignment_direction} alignment",
                    'priority': 'high'
                })
            
            # Premium/Discount zone signal
            if premium_discount.current_zone == PriceZone.DISCOUNT and mtf_alignment.alignment_direction == 'bullish':
                signals.append({
                    'type': 'premium_discount',
                    'direction': 'bullish',
                    'confidence': premium_discount.confidence,
                    'zone': 'discount',
                    'reasoning': "Price in discount zone with bullish structure",
                    'priority': 'high'
                })
            elif premium_discount.current_zone == PriceZone.PREMIUM and mtf_alignment.alignment_direction == 'bearish':
                signals.append({
                    'type': 'premium_discount',
                    'direction': 'bearish',
                    'confidence': premium_discount.confidence,
                    'zone': 'premium',
                    'reasoning': "Price in premium zone with bearish structure",
                    'priority': 'high'
                })
            
            # Unmitigated blocks signal
            unmitigated = [b for b in mitigation_blocks if not b.is_mitigated]
            if unmitigated:
                for block in unmitigated[:2]:  # Top 2
                    signals.append({
                        'type': 'mitigation_block',
                        'direction': block.block_type,
                        'confidence': block.confidence,
                        'price_high': block.high,
                        'price_low': block.low,
                        'reasoning': f"Unmitigated {block.block_type} order block",
                        'priority': 'medium'
                    })
            
            # Breaker blocks signal
            if breaker_blocks:
                for breaker in breaker_blocks[:1]:  # Most recent
                    signals.append({
                        'type': 'breaker_block',
                        'direction': breaker.breaker_type,
                        'confidence': breaker.confidence,
                        'reasoning': f"Breaker block: {breaker.original_type} flipped to {breaker.breaker_type}",
                        'priority': 'high' if breaker.retest_count > 0 else 'medium'
                    })
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Error generating structure signals: {e}")
            return signals
    
    async def _calculate_overall_confidence(
        self,
        mtf_alignment: MultiTimeframeAlignment,
        premium_discount: PremiumDiscountZone,
        mitigation_blocks: List[MitigationBlock],
        breaker_blocks: List[BreakerBlock]
    ) -> float:
        """Calculate overall structure analysis confidence"""
        try:
            confidence = 0.5
            
            # MTF alignment boost
            if mtf_alignment.aligned:
                confidence += mtf_alignment.alignment_score * 0.3
            
            # Premium/discount context
            if premium_discount.current_zone in [PriceZone.PREMIUM, PriceZone.DISCOUNT]:
                confidence += 0.1
            
            # Unmitigated blocks
            unmitigated_count = sum(1 for b in mitigation_blocks if not b.is_mitigated)
            if unmitigated_count > 0:
                confidence += min(0.2, unmitigated_count * 0.05)
            
            # Breaker blocks
            if breaker_blocks:
                confidence += min(0.2, len(breaker_blocks) * 0.1)
            
            return min(0.95, confidence)
            
        except Exception:
            return 0.5
    
    def _get_default_analysis(self, symbol: str, timeframe: str) -> EnhancedStructureAnalysis:
        """Get default analysis when insufficient data"""
        return EnhancedStructureAnalysis(
            symbol=symbol,
            timeframe=timeframe,
            timestamp=datetime.now(),
            mtf_alignment=MultiTimeframeAlignment(
                symbol=symbol,
                timestamp=datetime.now(),
                timeframes=[],
                structures={},
                aligned=False,
                alignment_direction='neutral',
                alignment_score=0.0,
                confidence=0.0,
                metadata={}
            ),
            premium_discount=PremiumDiscountZone(
                symbol=symbol,
                timeframe=timeframe,
                timestamp=datetime.now(),
                range_high=0.0,
                range_low=0.0,
                equilibrium=0.0,
                premium_25=0.0,
                discount_25=0.0,
                current_price=0.0,
                current_zone=PriceZone.UNKNOWN,
                distance_to_equilibrium=0.0,
                confidence=0.0,
                metadata={}
            ),
            mitigation_blocks=[],
            breaker_blocks=[],
            overall_confidence=0.0,
            structure_signals=[],
            metadata={'error': 'Insufficient data for analysis'}
        )
    
    async def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        return {
            'stats': self.stats,
            'config': self.config,
            'last_update': datetime.now().isoformat()
        }

