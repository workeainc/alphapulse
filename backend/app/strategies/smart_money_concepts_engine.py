"""
Smart Money Concepts Engine for AlphaPulse
Comprehensive implementation of Order Blocks, Fair Value Gaps, Market Structure, and Liquidity Analysis
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

class SMCType(Enum):
    """Smart Money Concepts types"""
    ORDER_BLOCK = "order_block"
    FAIR_VALUE_GAP = "fair_value_gap"
    LIQUIDITY_SWEEP = "liquidity_sweep"
    MARKET_STRUCTURE = "market_structure"

@dataclass
class OrderBlock:
    """Order Block structure"""
    symbol: str
    timestamp: datetime
    timeframe: str
    block_type: str  # 'bullish' or 'bearish'
    high: float
    low: float
    open: float
    close: float
    volume: float
    strength: float  # 0.0 to 1.0
    confidence: float
    is_mitigated: bool = False
    mitigation_time: Optional[datetime] = None
    metadata: Dict[str, Any] = None

@dataclass
class FairValueGap:
    """Fair Value Gap structure"""
    symbol: str
    timestamp: datetime
    timeframe: str
    gap_type: str  # 'bullish' or 'bearish'
    high: float
    low: float
    gap_size: float
    fill_probability: float
    strength: float
    confidence: float
    is_filled: bool = False
    fill_time: Optional[datetime] = None
    metadata: Dict[str, Any] = None

@dataclass
class LiquiditySweep:
    """Liquidity Sweep structure"""
    symbol: str
    timestamp: datetime
    timeframe: str
    sweep_type: str  # 'bullish' or 'bearish'
    price_level: float
    volume: float
    sweep_strength: float
    reversal_probability: float
    confidence: float
    is_reversed: bool = False
    reversal_time: Optional[datetime] = None
    metadata: Dict[str, Any] = None

@dataclass
class MarketStructure:
    """Market Structure structure"""
    symbol: str
    timestamp: datetime
    timeframe: str
    structure_type: str  # 'BOS', 'CHoCH', 'Liquidity', 'OrderBlock'
    price_level: float
    direction: str  # 'bullish' or 'bearish'
    strength: float
    confidence: float
    is_breakout: bool = False
    breakout_time: Optional[datetime] = None
    metadata: Dict[str, Any] = None

@dataclass
class SMCAnalysis:
    """Complete Smart Money Concepts analysis"""
    symbol: str
    timeframe: str
    timestamp: datetime
    order_blocks: List[OrderBlock]
    fair_value_gaps: List[FairValueGap]
    liquidity_sweeps: List[LiquiditySweep]
    market_structures: List[MarketStructure]
    overall_confidence: float
    smc_signals: List[Dict[str, Any]]
    metadata: Dict[str, Any]

class SmartMoneyConceptsEngine:
    """Smart Money Concepts analysis engine"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logger
        
        # Configuration
        self.min_volume_threshold = self.config.get('min_volume_threshold', 1.5)
        self.min_gap_size = self.config.get('min_gap_size', 0.001)  # 0.1%
        self.lookback_periods = self.config.get('lookback_periods', 100)
        self.min_confidence = self.config.get('min_confidence', 0.6)
        
        # Performance tracking
        self.stats = {
            'order_blocks_detected': 0,
            'fair_value_gaps_detected': 0,
            'liquidity_sweeps_detected': 0,
            'market_structures_detected': 0,
            'analyses_performed': 0,
            'last_update': datetime.now()
        }
        
        logger.info("🚀 Smart Money Concepts Engine initialized")
    
    async def analyze_smart_money_concepts(self, df: pd.DataFrame, symbol: str, timeframe: str) -> SMCAnalysis:
        """Analyze Smart Money Concepts for given data"""
        try:
            if len(df) < self.lookback_periods:
                self.logger.warning(f"Insufficient data for SMC analysis: {len(df)} < {self.lookback_periods}")
                return self._get_default_analysis(symbol, timeframe)
            
            # Detect Order Blocks
            order_blocks = await self._detect_order_blocks(df, symbol, timeframe)
            
            # Detect Fair Value Gaps
            fair_value_gaps = await self._detect_fair_value_gaps(df, symbol, timeframe)
            
            # Detect Liquidity Sweeps
            liquidity_sweeps = await self._detect_liquidity_sweeps(df, symbol, timeframe)
            
            # Analyze Market Structure
            market_structures = await self._analyze_market_structure(df, symbol, timeframe)
            
            # Generate SMC signals
            smc_signals = await self._generate_smc_signals(
                order_blocks, fair_value_gaps, liquidity_sweeps, market_structures
            )
            
            # Calculate overall confidence
            overall_confidence = await self._calculate_overall_confidence(
                order_blocks, fair_value_gaps, liquidity_sweeps, market_structures
            )
            
            # Create analysis result
            analysis = SMCAnalysis(
                symbol=symbol,
                timeframe=timeframe,
                timestamp=df['timestamp'].iloc[-1] if 'timestamp' in df.columns else datetime.now(),
                order_blocks=order_blocks,
                fair_value_gaps=fair_value_gaps,
                liquidity_sweeps=liquidity_sweeps,
                market_structures=market_structures,
                overall_confidence=overall_confidence,
                smc_signals=smc_signals,
                metadata={
                    'analysis_version': '1.0',
                    'config': self.config,
                    'stats': self.stats
                }
            )
            
            # Update statistics
            self.stats['order_blocks_detected'] += len(order_blocks)
            self.stats['fair_value_gaps_detected'] += len(fair_value_gaps)
            self.stats['liquidity_sweeps_detected'] += len(liquidity_sweeps)
            self.stats['market_structures_detected'] += len(market_structures)
            self.stats['analyses_performed'] += 1
            self.stats['last_update'] = datetime.now()
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"❌ Error analyzing Smart Money Concepts for {symbol}: {e}")
            return self._get_default_analysis(symbol, timeframe)
    
    async def _detect_order_blocks(self, df: pd.DataFrame, symbol: str, timeframe: str) -> List[OrderBlock]:
        """Detect Order Blocks in the data"""
        order_blocks = []
        
        try:
            # Need at least 10 bars for order block detection
            if len(df) < 10:
                return order_blocks
            
            # Calculate volume moving average
            df['volume_sma'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            
            # Detect bullish order blocks (strong move up after consolidation)
            for i in range(5, len(df) - 5):
                # Check for bullish order block pattern
                if (df.iloc[i]['close'] > df.iloc[i]['open'] and  # Bullish candle
                    df.iloc[i]['volume_ratio'] > self.min_volume_threshold and  # High volume
                    df.iloc[i]['close'] > df.iloc[i-1:i+1]['high'].max() and  # Break above previous highs
                    df.iloc[i-5:i]['close'].std() < df.iloc[i-5:i]['close'].mean() * 0.02):  # Low volatility before
                    
                    # Calculate strength and confidence
                    volume_strength = min(1.0, df.iloc[i]['volume_ratio'] / 3.0)
                    price_strength = min(1.0, (df.iloc[i]['close'] - df.iloc[i]['open']) / df.iloc[i]['open'])
                    confidence = (volume_strength + price_strength) / 2
                    
                    if confidence >= self.min_confidence:
                        order_block = OrderBlock(
                            symbol=symbol,
                            timestamp=df.iloc[i]['timestamp'] if 'timestamp' in df.columns else datetime.now(),
                            timeframe=timeframe,
                            block_type='bullish',
                            high=df.iloc[i]['high'],
                            low=df.iloc[i]['low'],
                            open=df.iloc[i]['open'],
                            close=df.iloc[i]['close'],
                            volume=df.iloc[i]['volume'],
                            strength=confidence,
                            confidence=confidence,
                            metadata={
                                'volume_ratio': df.iloc[i]['volume_ratio'],
                                'price_strength': price_strength,
                                'pattern_index': i
                            }
                        )
                        order_blocks.append(order_block)
            
            # Detect bearish order blocks (strong move down after consolidation)
            for i in range(5, len(df) - 5):
                # Check for bearish order block pattern
                if (df.iloc[i]['close'] < df.iloc[i]['open'] and  # Bearish candle
                    df.iloc[i]['volume_ratio'] > self.min_volume_threshold and  # High volume
                    df.iloc[i]['close'] < df.iloc[i-1:i+1]['low'].min() and  # Break below previous lows
                    df.iloc[i-5:i]['close'].std() < df.iloc[i-5:i]['close'].mean() * 0.02):  # Low volatility before
                    
                    # Calculate strength and confidence
                    volume_strength = min(1.0, df.iloc[i]['volume_ratio'] / 3.0)
                    price_strength = min(1.0, (df.iloc[i]['open'] - df.iloc[i]['close']) / df.iloc[i]['open'])
                    confidence = (volume_strength + price_strength) / 2
                    
                    if confidence >= self.min_confidence:
                        order_block = OrderBlock(
                            symbol=symbol,
                            timestamp=df.iloc[i]['timestamp'] if 'timestamp' in df.columns else datetime.now(),
                            timeframe=timeframe,
                            block_type='bearish',
                            high=df.iloc[i]['high'],
                            low=df.iloc[i]['low'],
                            open=df.iloc[i]['open'],
                            close=df.iloc[i]['close'],
                            volume=df.iloc[i]['volume'],
                            strength=confidence,
                            confidence=confidence,
                            metadata={
                                'volume_ratio': df.iloc[i]['volume_ratio'],
                                'price_strength': price_strength,
                                'pattern_index': i
                            }
                        )
                        order_blocks.append(order_block)
            
            self.logger.info(f"📊 Detected {len(order_blocks)} order blocks for {symbol}")
            return order_blocks
            
        except Exception as e:
            self.logger.error(f"❌ Error detecting order blocks: {e}")
            return order_blocks
    
    async def _detect_fair_value_gaps(self, df: pd.DataFrame, symbol: str, timeframe: str) -> List[FairValueGap]:
        """Detect Fair Value Gaps in the data"""
        fair_value_gaps = []
        
        try:
            # Need at least 3 bars for gap detection
            if len(df) < 3:
                return fair_value_gaps
            
            for i in range(1, len(df) - 1):
                current_high = df.iloc[i]['high']
                current_low = df.iloc[i]['low']
                prev_low = df.iloc[i-1]['low']
                next_high = df.iloc[i+1]['high']
                
                # Detect bullish fair value gap
                if current_low > prev_low:
                    gap_size = current_low - prev_low
                    if gap_size > self.min_gap_size * prev_low:  # Gap is significant
                        # Calculate fill probability based on gap size and volume
                        fill_probability = max(0.1, min(0.9, 1.0 - (gap_size / prev_low) * 10))
                        strength = min(1.0, gap_size / (prev_low * 0.05))  # Normalize to 5% move
                        confidence = (fill_probability + strength) / 2
                        
                        if confidence >= self.min_confidence:
                            gap = FairValueGap(
                                symbol=symbol,
                                timestamp=df.iloc[i]['timestamp'] if 'timestamp' in df.columns else datetime.now(),
                                timeframe=timeframe,
                                gap_type='bullish',
                                high=current_high,
                                low=current_low,
                                gap_size=gap_size,
                                fill_probability=fill_probability,
                                strength=strength,
                                confidence=confidence,
                                metadata={
                                    'gap_percentage': (gap_size / prev_low) * 100,
                                    'pattern_index': i
                                }
                            )
                            fair_value_gaps.append(gap)
                
                # Detect bearish fair value gap
                if current_high < next_high:
                    gap_size = next_high - current_high
                    if gap_size > self.min_gap_size * current_high:  # Gap is significant
                        # Calculate fill probability based on gap size and volume
                        fill_probability = max(0.1, min(0.9, 1.0 - (gap_size / current_high) * 10))
                        strength = min(1.0, gap_size / (current_high * 0.05))  # Normalize to 5% move
                        confidence = (fill_probability + strength) / 2
                        
                        if confidence >= self.min_confidence:
                            gap = FairValueGap(
                                symbol=symbol,
                                timestamp=df.iloc[i]['timestamp'] if 'timestamp' in df.columns else datetime.now(),
                                timeframe=timeframe,
                                gap_type='bearish',
                                high=current_high,
                                low=current_low,
                                gap_size=gap_size,
                                fill_probability=fill_probability,
                                strength=strength,
                                confidence=confidence,
                                metadata={
                                    'gap_percentage': (gap_size / current_high) * 100,
                                    'pattern_index': i
                                }
                            )
                            fair_value_gaps.append(gap)
            
            self.logger.info(f"📊 Detected {len(fair_value_gaps)} fair value gaps for {symbol}")
            return fair_value_gaps
            
        except Exception as e:
            self.logger.error(f"❌ Error detecting fair value gaps: {e}")
            return fair_value_gaps
    
    async def _detect_liquidity_sweeps(self, df: pd.DataFrame, symbol: str, timeframe: str) -> List[LiquiditySweep]:
        """Detect Liquidity Sweeps in the data"""
        liquidity_sweeps = []
        
        try:
            # Need at least 10 bars for sweep detection
            if len(df) < 10:
                return liquidity_sweeps
            
            # Calculate support and resistance levels
            df['sma_20'] = df['close'].rolling(window=20).mean()
            df['sma_50'] = df['close'].rolling(window=50).mean()
            
            for i in range(5, len(df) - 5):
                current_high = df.iloc[i]['high']
                current_low = df.iloc[i]['low']
                current_volume = df.iloc[i]['volume']
                
                # Check for bullish liquidity sweep (sweep below support then reversal)
                if (current_low < df.iloc[i-5:i]['low'].min() and  # Sweep below recent lows
                    current_high > df.iloc[i]['open'] and  # Close above open (bullish)
                    current_volume > df.iloc[i-5:i]['volume'].mean() * 1.5):  # High volume
                    
                    # Calculate sweep strength and reversal probability
                    sweep_strength = min(1.0, (df.iloc[i-5:i]['low'].min() - current_low) / current_low)
                    reversal_probability = min(0.9, max(0.1, 0.5 + sweep_strength * 0.4))
                    confidence = (sweep_strength + reversal_probability) / 2
                    
                    if confidence >= self.min_confidence:
                        sweep = LiquiditySweep(
                            symbol=symbol,
                            timestamp=df.iloc[i]['timestamp'] if 'timestamp' in df.columns else datetime.now(),
                            timeframe=timeframe,
                            sweep_type='bullish',
                            price_level=current_low,
                            volume=current_volume,
                            sweep_strength=sweep_strength,
                            reversal_probability=reversal_probability,
                            confidence=confidence,
                            metadata={
                                'sweep_depth': (df.iloc[i-5:i]['low'].min() - current_low) / current_low * 100,
                                'volume_ratio': current_volume / df.iloc[i-5:i]['volume'].mean(),
                                'pattern_index': i
                            }
                        )
                        liquidity_sweeps.append(sweep)
                
                # Check for bearish liquidity sweep (sweep above resistance then reversal)
                if (current_high > df.iloc[i-5:i]['high'].max() and  # Sweep above recent highs
                    current_low < df.iloc[i]['open'] and  # Close below open (bearish)
                    current_volume > df.iloc[i-5:i]['volume'].mean() * 1.5):  # High volume
                    
                    # Calculate sweep strength and reversal probability
                    sweep_strength = min(1.0, (current_high - df.iloc[i-5:i]['high'].max()) / current_high)
                    reversal_probability = min(0.9, max(0.1, 0.5 + sweep_strength * 0.4))
                    confidence = (sweep_strength + reversal_probability) / 2
                    
                    if confidence >= self.min_confidence:
                        sweep = LiquiditySweep(
                            symbol=symbol,
                            timestamp=df.iloc[i]['timestamp'] if 'timestamp' in df.columns else datetime.now(),
                            timeframe=timeframe,
                            sweep_type='bearish',
                            price_level=current_high,
                            volume=current_volume,
                            sweep_strength=sweep_strength,
                            reversal_probability=reversal_probability,
                            confidence=confidence,
                            metadata={
                                'sweep_depth': (current_high - df.iloc[i-5:i]['high'].max()) / current_high * 100,
                                'volume_ratio': current_volume / df.iloc[i-5:i]['volume'].mean(),
                                'pattern_index': i
                            }
                        )
                        liquidity_sweeps.append(sweep)
            
            self.logger.info(f"📊 Detected {len(liquidity_sweeps)} liquidity sweeps for {symbol}")
            return liquidity_sweeps
            
        except Exception as e:
            self.logger.error(f"❌ Error detecting liquidity sweeps: {e}")
            return liquidity_sweeps
    
    async def _analyze_market_structure(self, df: pd.DataFrame, symbol: str, timeframe: str) -> List[MarketStructure]:
        """Analyze Market Structure (BOS, CHoCH, etc.)"""
        market_structures = []
        
        try:
            # Need at least 20 bars for market structure analysis
            if len(df) < 20:
                return market_structures
            
            # Calculate swing highs and lows
            df['swing_high'] = df['high'].rolling(window=5, center=True).max()
            df['swing_low'] = df['low'].rolling(window=5, center=True).min()
            
            # Detect Break of Structure (BOS)
            for i in range(10, len(df) - 5):
                # Bullish BOS
                if (df.iloc[i]['close'] > df.iloc[i-10:i]['high'].max() and
                    df.iloc[i]['volume'] > df.iloc[i-5:i]['volume'].mean() * 1.2):
                    
                    structure = MarketStructure(
                        symbol=symbol,
                        timestamp=df.iloc[i]['timestamp'] if 'timestamp' in df.columns else datetime.now(),
                        timeframe=timeframe,
                        structure_type='BOS',
                        price_level=df.iloc[i]['close'],
                        direction='bullish',
                        strength=0.8,
                        confidence=0.7,
                        metadata={
                            'breakout_level': df.iloc[i-10:i]['high'].max(),
                            'volume_confirmation': True,
                            'pattern_index': i
                        }
                    )
                    market_structures.append(structure)
                
                # Bearish BOS
                if (df.iloc[i]['close'] < df.iloc[i-10:i]['low'].min() and
                    df.iloc[i]['volume'] > df.iloc[i-5:i]['volume'].mean() * 1.2):
                    
                    structure = MarketStructure(
                        symbol=symbol,
                        timestamp=df.iloc[i]['timestamp'] if 'timestamp' in df.columns else datetime.now(),
                        timeframe=timeframe,
                        structure_type='BOS',
                        price_level=df.iloc[i]['close'],
                        direction='bearish',
                        strength=0.8,
                        confidence=0.7,
                        metadata={
                            'breakout_level': df.iloc[i-10:i]['low'].min(),
                            'volume_confirmation': True,
                            'pattern_index': i
                        }
                    )
                    market_structures.append(structure)
            
            self.logger.info(f"📊 Detected {len(market_structures)} market structures for {symbol}")
            return market_structures
            
        except Exception as e:
            self.logger.error(f"❌ Error analyzing market structure: {e}")
            return market_structures
    
    async def _generate_smc_signals(self, order_blocks: List[OrderBlock], 
                                  fair_value_gaps: List[FairValueGap],
                                  liquidity_sweeps: List[LiquiditySweep],
                                  market_structures: List[MarketStructure]) -> List[Dict[str, Any]]:
        """Generate trading signals based on SMC analysis"""
        signals = []
        
        try:
            # Generate signals from order blocks
            for ob in order_blocks:
                if ob.confidence >= self.min_confidence:
                    signal = {
                        'type': 'order_block',
                        'direction': ob.block_type,
                        'confidence': ob.confidence,
                        'strength': ob.strength,
                        'price_level': ob.low if ob.block_type == 'bullish' else ob.high,
                        'timestamp': ob.timestamp,
                        'metadata': ob.metadata
                    }
                    signals.append(signal)
            
            # Generate signals from fair value gaps
            for fvg in fair_value_gaps:
                if fvg.confidence >= self.min_confidence:
                    signal = {
                        'type': 'fair_value_gap',
                        'direction': fvg.gap_type,
                        'confidence': fvg.confidence,
                        'strength': fvg.strength,
                        'price_level': fvg.low if fvg.gap_type == 'bullish' else fvg.high,
                        'timestamp': fvg.timestamp,
                        'metadata': fvg.metadata
                    }
                    signals.append(signal)
            
            # Generate signals from liquidity sweeps
            for ls in liquidity_sweeps:
                if ls.confidence >= self.min_confidence:
                    signal = {
                        'type': 'liquidity_sweep',
                        'direction': ls.sweep_type,
                        'confidence': ls.confidence,
                        'strength': ls.sweep_strength,
                        'price_level': ls.price_level,
                        'timestamp': ls.timestamp,
                        'metadata': ls.metadata
                    }
                    signals.append(signal)
            
            # Generate signals from market structures
            for ms in market_structures:
                if ms.confidence >= self.min_confidence:
                    signal = {
                        'type': 'market_structure',
                        'direction': ms.direction,
                        'confidence': ms.confidence,
                        'strength': ms.strength,
                        'price_level': ms.price_level,
                        'timestamp': ms.timestamp,
                        'metadata': ms.metadata
                    }
                    signals.append(signal)
            
            # Sort signals by confidence
            signals.sort(key=lambda x: x['confidence'], reverse=True)
            
            self.logger.info(f"📊 Generated {len(signals)} SMC signals")
            return signals
            
        except Exception as e:
            self.logger.error(f"❌ Error generating SMC signals: {e}")
            return signals
    
    async def _calculate_overall_confidence(self, order_blocks: List[OrderBlock],
                                          fair_value_gaps: List[FairValueGap],
                                          liquidity_sweeps: List[LiquiditySweep],
                                          market_structures: List[MarketStructure]) -> float:
        """Calculate overall confidence for SMC analysis"""
        try:
            total_confidence = 0.0
            total_weight = 0.0
            
            # Weight different SMC components
            weights = {
                'order_blocks': 0.4,
                'fair_value_gaps': 0.3,
                'liquidity_sweeps': 0.2,
                'market_structures': 0.1
            }
            
            # Calculate weighted confidence
            if order_blocks:
                avg_ob_confidence = sum(ob.confidence for ob in order_blocks) / len(order_blocks)
                total_confidence += avg_ob_confidence * weights['order_blocks']
                total_weight += weights['order_blocks']
            
            if fair_value_gaps:
                avg_fvg_confidence = sum(fvg.confidence for fvg in fair_value_gaps) / len(fair_value_gaps)
                total_confidence += avg_fvg_confidence * weights['fair_value_gaps']
                total_weight += weights['fair_value_gaps']
            
            if liquidity_sweeps:
                avg_ls_confidence = sum(ls.confidence for ls in liquidity_sweeps) / len(liquidity_sweeps)
                total_confidence += avg_ls_confidence * weights['liquidity_sweeps']
                total_weight += weights['liquidity_sweeps']
            
            if market_structures:
                avg_ms_confidence = sum(ms.confidence for ms in market_structures) / len(market_structures)
                total_confidence += avg_ms_confidence * weights['market_structures']
                total_weight += weights['market_structures']
            
            # Return weighted average confidence
            return total_confidence / total_weight if total_weight > 0 else 0.0
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating overall confidence: {e}")
            return 0.0
    
    def _get_default_analysis(self, symbol: str, timeframe: str) -> SMCAnalysis:
        """Get default analysis when insufficient data"""
        return SMCAnalysis(
            symbol=symbol,
            timeframe=timeframe,
            timestamp=datetime.now(),
            order_blocks=[],
            fair_value_gaps=[],
            liquidity_sweeps=[],
            market_structures=[],
            overall_confidence=0.0,
            smc_signals=[],
            metadata={'error': 'Insufficient data for analysis'}
        )
    
    async def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        return {
            'stats': self.stats,
            'config': self.config,
            'last_update': datetime.now().isoformat()
        }
