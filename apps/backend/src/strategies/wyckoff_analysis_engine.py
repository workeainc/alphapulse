"""
Wyckoff Methodology Analysis Engine for AlphaPulse
Implements complete Wyckoff phases, composite operator analysis, and volume analysis
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

class WyckoffPhase(Enum):
    """Wyckoff accumulation/distribution phases"""
    # Accumulation phases
    PS = "preliminary_support"  # Phase A
    SC = "selling_climax"  # Phase A
    AR = "automatic_rally"  # Phase A
    ST = "secondary_test"  # Phase B
    SPRING = "spring"  # Phase C
    TEST = "test"  # Phase C
    SOS = "sign_of_strength"  # Phase D
    LPS = "last_point_of_support"  # Phase D
    MARKUP = "markup"  # Phase E
    
    # Distribution phases
    PSY = "preliminary_supply"  # Phase A
    BC = "buying_climax"  # Phase A
    AR_DIST = "automatic_reaction"  # Phase A
    ST_DIST = "secondary_test_dist"  # Phase B
    UTAD = "upthrust_after_distribution"  # Phase C
    SOW = "sign_of_weakness"  # Phase D
    LPSY = "last_point_of_supply"  # Phase D
    MARKDOWN = "markdown"  # Phase E
    
    UNKNOWN = "unknown"

class WyckoffSchematic(Enum):
    """Wyckoff schematics"""
    ACCUMULATION = "accumulation"
    DISTRIBUTION = "distribution"
    REACCUMULATION = "reaccumulation"
    REDISTRIBUTION = "redistribution"
    UNKNOWN = "unknown"

@dataclass
class WyckoffPhaseInfo:
    """Information about a Wyckoff phase"""
    phase: WyckoffPhase
    schematic: WyckoffSchematic
    start_index: int
    end_index: Optional[int]
    confidence: float
    volume_confirmation: bool
    key_levels: Dict[str, float]
    metadata: Dict[str, Any]

@dataclass
class WyckoffEvent:
    """Specific Wyckoff event (SC, Spring, SOS, etc.)"""
    event_type: WyckoffPhase
    timestamp: datetime
    price: float
    volume: float
    spread: float
    significance: float
    confidence: float
    metadata: Dict[str, Any]

@dataclass
class CompositeOperatorAnalysis:
    """Composite operator (smart money) analysis"""
    is_accumulating: bool
    is_distributing: bool
    absorption_detected: bool
    effort_vs_result_score: float
    institutional_footprint: float
    confidence: float
    metadata: Dict[str, Any]

@dataclass
class CauseEffect:
    """Cause and Effect measurement"""
    cause_size: float  # Point & figure count
    potential_effect: float  # Projected price move
    target_price: float
    confidence: float
    metadata: Dict[str, Any]

@dataclass
class WyckoffAnalysis:
    """Complete Wyckoff analysis"""
    symbol: str
    timeframe: str
    timestamp: datetime
    current_schematic: WyckoffSchematic
    current_phase: WyckoffPhase
    phase_history: List[WyckoffPhaseInfo]
    wyckoff_events: List[WyckoffEvent]
    composite_operator: CompositeOperatorAnalysis
    cause_effect: Optional[CauseEffect]
    overall_confidence: float
    wyckoff_signals: List[Dict[str, Any]]
    metadata: Dict[str, Any]

class WyckoffAnalysisEngine:
    """
    Complete Wyckoff Methodology Analysis Engine
    
    Implements:
    - Phase identification (PS, SC, AR, ST, Spring, Test, SOS, LPS, Markup)
    - Distribution phases (PSY, BC, AR, ST, UTAD, SOW, LPSY, Markdown)
    - Composite operator analysis
    - Effort vs Result analysis
    - Cause and Effect measurement
    - Volume analysis (Wyckoff-specific)
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logger
        
        # Configuration
        self.lookback_periods = self.config.get('lookback_periods', 100)
        self.min_confidence = self.config.get('min_confidence', 0.6)
        self.volume_threshold = self.config.get('volume_threshold', 1.5)
        self.spread_threshold = self.config.get('spread_threshold', 0.02)  # 2%
        
        # Performance tracking
        self.stats = {
            'phases_detected': 0,
            'events_detected': 0,
            'springs_detected': 0,
            'utads_detected': 0,
            'analyses_performed': 0,
            'last_update': datetime.now()
        }
        
        logger.info("🚀 Wyckoff Analysis Engine initialized")
    
    async def analyze_wyckoff(
        self, 
        df: pd.DataFrame, 
        symbol: str, 
        timeframe: str
    ) -> WyckoffAnalysis:
        """
        Complete Wyckoff analysis
        
        Args:
            df: DataFrame with OHLCV data
            symbol: Trading symbol
            timeframe: Timeframe
            
        Returns:
            WyckoffAnalysis with complete methodology analysis
        """
        try:
            if len(df) < self.lookback_periods:
                self.logger.warning(
                    f"Insufficient data for Wyckoff analysis: {len(df)} < {self.lookback_periods}"
                )
                return self._get_default_analysis(symbol, timeframe)
            
            # Prepare volume and spread data
            df = self._prepare_data(df)
            
            # Detect Wyckoff events
            wyckoff_events = await self._detect_wyckoff_events(df, symbol, timeframe)
            
            # Identify current schematic and phase
            current_schematic, current_phase, phase_history = await self._identify_phase(
                df, wyckoff_events
            )
            
            # Composite operator analysis
            composite_operator = await self._analyze_composite_operator(df, wyckoff_events)
            
            # Cause and Effect measurement
            cause_effect = await self._measure_cause_effect(
                df, current_schematic, phase_history
            )
            
            # Generate Wyckoff signals
            wyckoff_signals = await self._generate_wyckoff_signals(
                current_schematic, current_phase, wyckoff_events, composite_operator
            )
            
            # Calculate overall confidence
            overall_confidence = await self._calculate_overall_confidence(
                current_phase, wyckoff_events, composite_operator
            )
            
            # Create analysis result
            analysis = WyckoffAnalysis(
                symbol=symbol,
                timeframe=timeframe,
                timestamp=df['timestamp'].iloc[-1] if 'timestamp' in df.columns else datetime.now(),
                current_schematic=current_schematic,
                current_phase=current_phase,
                phase_history=phase_history,
                wyckoff_events=wyckoff_events,
                composite_operator=composite_operator,
                cause_effect=cause_effect,
                overall_confidence=overall_confidence,
                wyckoff_signals=wyckoff_signals,
                metadata={
                    'analysis_version': '1.0',
                    'config': self.config,
                    'stats': self.stats
                }
            )
            
            # Update statistics
            self.stats['phases_detected'] += len(phase_history)
            self.stats['events_detected'] += len(wyckoff_events)
            self.stats['springs_detected'] += sum(
                1 for e in wyckoff_events if e.event_type == WyckoffPhase.SPRING
            )
            self.stats['utads_detected'] += sum(
                1 for e in wyckoff_events if e.event_type == WyckoffPhase.UTAD
            )
            self.stats['analyses_performed'] += 1
            self.stats['last_update'] = datetime.now()
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"❌ Error in Wyckoff analysis for {symbol}: {e}")
            return self._get_default_analysis(symbol, timeframe)
    
    def _prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare Wyckoff-specific metrics"""
        try:
            # Calculate spread (high-low range)
            df['spread'] = df['high'] - df['low']
            df['spread_pct'] = (df['spread'] / df['close']) * 100
            
            # Calculate body and wicks
            df['body'] = abs(df['close'] - df['open'])
            df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
            df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
            
            # Volume metrics
            df['volume_ma'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_ma']
            
            # Up/Down volume (approximation)
            df['is_up'] = df['close'] > df['open']
            df['up_volume'] = df['volume'] * df['is_up']
            df['down_volume'] = df['volume'] * (~df['is_up'])
            
            # Cumulative volume delta
            df['volume_delta'] = df['up_volume'] - df['down_volume']
            df['cumulative_volume_delta'] = df['volume_delta'].cumsum()
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error preparing data: {e}")
            return df
    
    async def _detect_wyckoff_events(
        self, 
        df: pd.DataFrame, 
        symbol: str, 
        timeframe: str
    ) -> List[WyckoffEvent]:
        """Detect specific Wyckoff events"""
        events = []
        
        try:
            # Detect Selling Climax (SC) - High volume, wide spread, down close
            sc_events = self._detect_selling_climax(df, symbol)
            events.extend(sc_events)
            
            # Detect Buying Climax (BC) - High volume, wide spread, up close
            bc_events = self._detect_buying_climax(df, symbol)
            events.extend(bc_events)
            
            # Detect Spring - False breakdown with low volume
            spring_events = self._detect_spring(df, symbol)
            events.extend(spring_events)
            
            # Detect UTAD - False breakout with low volume
            utad_events = self._detect_utad(df, symbol)
            events.extend(utad_events)
            
            # Detect Sign of Strength (SOS) - High volume breakout upward
            sos_events = self._detect_sign_of_strength(df, symbol)
            events.extend(sos_events)
            
            # Detect Sign of Weakness (SOW) - High volume breakdown
            sow_events = self._detect_sign_of_weakness(df, symbol)
            events.extend(sow_events)
            
            # Sort by timestamp
            events.sort(key=lambda x: x.timestamp)
            
            self.logger.info(f"📊 Detected {len(events)} Wyckoff events for {symbol}")
            return events
            
        except Exception as e:
            self.logger.error(f"❌ Error detecting Wyckoff events: {e}")
            return events
    
    def _detect_selling_climax(self, df: pd.DataFrame, symbol: str) -> List[WyckoffEvent]:
        """Detect Selling Climax (SC) - High volume capitulation"""
        events = []
        
        try:
            for i in range(20, len(df) - 5):
                # SC criteria:
                # 1. Very high volume (>2x average)
                # 2. Wide spread (>2% of price)
                # 3. Close near low (bearish)
                # 4. Often followed by rally
                
                if (df['volume_ratio'].iloc[i] > 2.0 and
                    df['spread_pct'].iloc[i] > 2.0 and
                    df['close'].iloc[i] < df['open'].iloc[i]):  # Bearish candle
                    
                    # Check if followed by rally (automatic rally)
                    next_5_candles = df.iloc[i+1:i+6]
                    if len(next_5_candles) > 0 and next_5_candles['close'].iloc[-1] > df['close'].iloc[i]:
                        # This looks like SC
                        significance = min(1.0, df['volume_ratio'].iloc[i] / 3.0)
                        confidence = 0.7 + (significance * 0.2)
                        
                        event = WyckoffEvent(
                            event_type=WyckoffPhase.SC,
                            timestamp=df['timestamp'].iloc[i] if 'timestamp' in df.columns else datetime.now(),
                            price=df['close'].iloc[i],
                            volume=df['volume'].iloc[i],
                            spread=df['spread'].iloc[i],
                            significance=significance,
                            confidence=confidence,
                            metadata={
                                'volume_ratio': df['volume_ratio'].iloc[i],
                                'spread_pct': df['spread_pct'].iloc[i],
                                'followed_by_rally': True,
                                'index': i
                            }
                        )
                        events.append(event)
            
            return events
            
        except Exception as e:
            self.logger.error(f"Error detecting SC: {e}")
            return events
    
    def _detect_buying_climax(self, df: pd.DataFrame, symbol: str) -> List[WyckoffEvent]:
        """Detect Buying Climax (BC) - High volume top"""
        events = []
        
        try:
            for i in range(20, len(df) - 5):
                # BC criteria:
                # 1. Very high volume (>2x average)
                # 2. Wide spread (>2% of price)
                # 3. Close near high (bullish)
                # 4. Often followed by decline
                
                if (df['volume_ratio'].iloc[i] > 2.0 and
                    df['spread_pct'].iloc[i] > 2.0 and
                    df['close'].iloc[i] > df['open'].iloc[i]):  # Bullish candle
                    
                    # Check if followed by decline (automatic reaction)
                    next_5_candles = df.iloc[i+1:i+6]
                    if len(next_5_candles) > 0 and next_5_candles['close'].iloc[-1] < df['close'].iloc[i]:
                        # This looks like BC
                        significance = min(1.0, df['volume_ratio'].iloc[i] / 3.0)
                        confidence = 0.7 + (significance * 0.2)
                        
                        event = WyckoffEvent(
                            event_type=WyckoffPhase.BC,
                            timestamp=df['timestamp'].iloc[i] if 'timestamp' in df.columns else datetime.now(),
                            price=df['close'].iloc[i],
                            volume=df['volume'].iloc[i],
                            spread=df['spread'].iloc[i],
                            significance=significance,
                            confidence=confidence,
                            metadata={
                                'volume_ratio': df['volume_ratio'].iloc[i],
                                'spread_pct': df['spread_pct'].iloc[i],
                                'followed_by_decline': True,
                                'index': i
                            }
                        )
                        events.append(event)
            
            return events
            
        except Exception as e:
            self.logger.error(f"Error detecting BC: {e}")
            return events
    
    def _detect_spring(self, df: pd.DataFrame, symbol: str) -> List[WyckoffEvent]:
        """Detect Spring - Final shakeout before markup"""
        events = []
        
        try:
            for i in range(30, len(df) - 5):
                # Spring criteria:
                # 1. Breaks below recent support/range low
                # 2. LOW volume (no supply)
                # 3. Quick reversal back into range
                # 4. Classic "stop hunt" pattern
                
                # Find recent range
                recent_data = df.iloc[i-20:i]
                range_low = recent_data['low'].min()
                range_high = recent_data['high'].max()
                
                # Check for spring
                if (df['low'].iloc[i] < range_low and  # Breaks below range
                    df['volume_ratio'].iloc[i] < 0.8 and  # Low volume
                    df['close'].iloc[i] > df['open'].iloc[i]):  # Closes up (reversal)
                    
                    # Check if next candles stay above range_low
                    next_candles = df.iloc[i+1:i+4]
                    if len(next_candles) > 0 and next_candles['low'].min() > range_low:
                        # This is a spring!
                        penetration = (range_low - df['low'].iloc[i]) / range_low
                        significance = min(1.0, penetration * 50)  # Normalize
                        confidence = 0.75 + min(0.2, significance * 2)
                        
                        event = WyckoffEvent(
                            event_type=WyckoffPhase.SPRING,
                            timestamp=df['timestamp'].iloc[i] if 'timestamp' in df.columns else datetime.now(),
                            price=df['low'].iloc[i],
                            volume=df['volume'].iloc[i],
                            spread=df['spread'].iloc[i],
                            significance=significance,
                            confidence=confidence,
                            metadata={
                                'range_low': range_low,
                                'range_high': range_high,
                                'penetration': penetration,
                                'volume_ratio': df['volume_ratio'].iloc[i],
                                'index': i
                            }
                        )
                        events.append(event)
            
            return events
            
        except Exception as e:
            self.logger.error(f"Error detecting Spring: {e}")
            return events
    
    def _detect_utad(self, df: pd.DataFrame, symbol: str) -> List[WyckoffEvent]:
        """Detect UTAD (Upthrust After Distribution) - Final pump before markdown
        
        Two-stage detection:
        1. Volume climax on breakout above resistance (high volume thrust)
        2. Followed by reversal with declining volume (weakness)
        """
        events = []
        
        try:
            for i in range(30, len(df) - 5):
                # Find recent range
                recent_data = df.iloc[i-20:i]
                range_low = recent_data['low'].min()
                range_high = recent_data['high'].max()
                
                # UTAD Two-Stage Detection:
                # Stage 1: High volume breakout above resistance (climax)
                # Stage 2: Reversal with lower volume (weakness)
                
                if df['high'].iloc[i] > range_high:  # Breaks above range
                    # Check for volume climax OR subsequent weakness
                    volume_climax = df['volume_ratio'].iloc[i] > 1.5  # High volume on breakout
                    
                    # Check reversal
                    closes_down = df['close'].iloc[i] < df['open'].iloc[i]  # Reversal candle
                    
                    # Check if next candles show weakness (stay below range_high)
                    next_candles = df.iloc[i+1:i+4]
                    if len(next_candles) > 0:
                        stays_below = next_candles['high'].max() < range_high
                        declining_volume = next_candles['volume_ratio'].mean() < 1.0
                        
                        # UTAD confirmed if:
                        # A) Volume climax + reversal, OR
                        # B) Breakout + reversal + declining volume
                        is_utad = (volume_climax and closes_down) or (closes_down and stays_below and declining_volume)
                        
                        if is_utad and stays_below:
                            # This is UTAD!
                            penetration = (df['high'].iloc[i] - range_high) / range_high
                            significance = min(1.0, penetration * 50)
                            
                            # Higher confidence if volume climax present
                            base_confidence = 0.75 + min(0.2, significance * 2)
                            if volume_climax:
                                base_confidence += 0.05  # Bonus for volume climax
                            
                            confidence = min(0.95, base_confidence)
                            
                            event = WyckoffEvent(
                                event_type=WyckoffPhase.UTAD,
                                timestamp=df['timestamp'].iloc[i] if 'timestamp' in df.columns else datetime.now(),
                                price=df['high'].iloc[i],
                                volume=df['volume'].iloc[i],
                                spread=df['spread'].iloc[i],
                                significance=significance,
                                confidence=confidence,
                                metadata={
                                    'range_low': range_low,
                                    'range_high': range_high,
                                    'penetration': penetration,
                                    'volume_ratio': df['volume_ratio'].iloc[i],
                                    'volume_climax': volume_climax,
                                    'declining_volume_after': declining_volume if 'declining_volume' in locals() else False,
                                    'index': i
                                }
                            )
                            events.append(event)
            
            return events
            
        except Exception as e:
            self.logger.error(f"Error detecting UTAD: {e}")
            return events
    
    def _detect_sign_of_strength(self, df: pd.DataFrame, symbol: str) -> List[WyckoffEvent]:
        """Detect Sign of Strength (SOS) - Bullish breakout with volume"""
        events = []
        
        try:
            for i in range(20, len(df) - 2):
                # SOS criteria:
                # 1. Strong bullish candle
                # 2. HIGH volume (demand)
                # 3. Breaks above resistance
                # 4. Wide spread
                
                if (df['close'].iloc[i] > df['open'].iloc[i] and  # Bullish
                    df['volume_ratio'].iloc[i] > 1.5 and  # High volume
                    df['spread_pct'].iloc[i] > 1.5):  # Wide spread
                    
                    # Check if breaks above recent highs
                    recent_high = df['high'].iloc[i-10:i].max()
                    if df['close'].iloc[i] > recent_high:
                        significance = min(1.0, df['volume_ratio'].iloc[i] / 2.5)
                        confidence = 0.7 + (significance * 0.2)
                        
                        event = WyckoffEvent(
                            event_type=WyckoffPhase.SOS,
                            timestamp=df['timestamp'].iloc[i] if 'timestamp' in df.columns else datetime.now(),
                            price=df['close'].iloc[i],
                            volume=df['volume'].iloc[i],
                            spread=df['spread'].iloc[i],
                            significance=significance,
                            confidence=confidence,
                            metadata={
                                'volume_ratio': df['volume_ratio'].iloc[i],
                                'spread_pct': df['spread_pct'].iloc[i],
                                'recent_high': recent_high,
                                'index': i
                            }
                        )
                        events.append(event)
            
            return events
            
        except Exception as e:
            self.logger.error(f"Error detecting SOS: {e}")
            return events
    
    def _detect_sign_of_weakness(self, df: pd.DataFrame, symbol: str) -> List[WyckoffEvent]:
        """Detect Sign of Weakness (SOW) - Bearish breakdown with volume"""
        events = []
        
        try:
            for i in range(20, len(df) - 2):
                # SOW criteria:
                # 1. Strong bearish candle
                # 2. HIGH volume (supply)
                # 3. Breaks below support
                # 4. Wide spread
                
                if (df['close'].iloc[i] < df['open'].iloc[i] and  # Bearish
                    df['volume_ratio'].iloc[i] > 1.5 and  # High volume
                    df['spread_pct'].iloc[i] > 1.5):  # Wide spread
                    
                    # Check if breaks below recent lows
                    recent_low = df['low'].iloc[i-10:i].min()
                    if df['close'].iloc[i] < recent_low:
                        significance = min(1.0, df['volume_ratio'].iloc[i] / 2.5)
                        confidence = 0.7 + (significance * 0.2)
                        
                        event = WyckoffEvent(
                            event_type=WyckoffPhase.SOW,
                            timestamp=df['timestamp'].iloc[i] if 'timestamp' in df.columns else datetime.now(),
                            price=df['close'].iloc[i],
                            volume=df['volume'].iloc[i],
                            spread=df['spread'].iloc[i],
                            significance=significance,
                            confidence=confidence,
                            metadata={
                                'volume_ratio': df['volume_ratio'].iloc[i],
                                'spread_pct': df['spread_pct'].iloc[i],
                                'recent_low': recent_low,
                                'index': i
                            }
                        )
                        events.append(event)
            
            return events
            
        except Exception as e:
            self.logger.error(f"Error detecting SOW: {e}")
            return events
    
    async def _identify_phase(
        self, 
        df: pd.DataFrame, 
        events: List[WyckoffEvent]
    ) -> Tuple[WyckoffSchematic, WyckoffPhase, List[WyckoffPhaseInfo]]:
        """Identify current Wyckoff schematic and phase"""
        try:
            # Analyze event sequence to determine schematic
            has_sc = any(e.event_type == WyckoffPhase.SC for e in events)
            has_bc = any(e.event_type == WyckoffPhase.BC for e in events)
            has_spring = any(e.event_type == WyckoffPhase.SPRING for e in events)
            has_utad = any(e.event_type == WyckoffPhase.UTAD for e in events)
            has_sos = any(e.event_type == WyckoffPhase.SOS for e in events)
            has_sow = any(e.event_type == WyckoffPhase.SOW for e in events)
            
            # Determine schematic
            if has_sc and has_spring:
                current_schematic = WyckoffSchematic.ACCUMULATION
                if has_sos:
                    current_phase = WyckoffPhase.SOS
                elif has_spring:
                    current_phase = WyckoffPhase.SPRING
                else:
                    current_phase = WyckoffPhase.ST
            elif has_bc and has_utad:
                current_schematic = WyckoffSchematic.DISTRIBUTION
                if has_sow:
                    current_phase = WyckoffPhase.SOW
                elif has_utad:
                    current_phase = WyckoffPhase.UTAD
                else:
                    current_phase = WyckoffPhase.ST_DIST
            elif has_sos and not has_sc:
                current_schematic = WyckoffSchematic.REACCUMULATION
                current_phase = WyckoffPhase.SOS
            elif has_sow and not has_bc:
                current_schematic = WyckoffSchematic.REDISTRIBUTION
                current_phase = WyckoffPhase.SOW
            else:
                current_schematic = WyckoffSchematic.UNKNOWN
                current_phase = WyckoffPhase.UNKNOWN
            
            # Build phase history (simplified - would be more detailed in production)
            phase_history = []
            if events:
                for event in events[-5:]:  # Last 5 events
                    phase_info = WyckoffPhaseInfo(
                        phase=event.event_type,
                        schematic=current_schematic,
                        start_index=event.metadata.get('index', 0),
                        end_index=None,
                        confidence=event.confidence,
                        volume_confirmation=event.volume > 0,
                        key_levels={'price': event.price},
                        metadata=event.metadata
                    )
                    phase_history.append(phase_info)
            
            return current_schematic, current_phase, phase_history
            
        except Exception as e:
            self.logger.error(f"Error identifying phase: {e}")
            return WyckoffSchematic.UNKNOWN, WyckoffPhase.UNKNOWN, []
    
    async def _analyze_composite_operator(
        self, 
        df: pd.DataFrame, 
        events: List[WyckoffEvent]
    ) -> CompositeOperatorAnalysis:
        """Analyze composite operator (smart money) activity"""
        try:
            # Check for accumulation signals
            is_accumulating = any(
                e.event_type in [WyckoffPhase.SC, WyckoffPhase.SPRING, WyckoffPhase.SOS]
                for e in events[-5:]
            )
            
            # Check for distribution signals
            is_distributing = any(
                e.event_type in [WyckoffPhase.BC, WyckoffPhase.UTAD, WyckoffPhase.SOW]
                for e in events[-5:]
            )
            
            # Detect absorption (high volume, narrow spread = absorption)
            recent_data = df.tail(20)
            absorption_detected = any(
                (recent_data['volume_ratio'] > 1.5) & 
                (recent_data['spread_pct'] < 1.0)
            )
            
            # Effort vs Result analysis
            # Effort = volume, Result = price change
            recent_volume = recent_data['volume'].sum()
            recent_price_change = abs(recent_data['close'].iloc[-1] - recent_data['close'].iloc[0])
            avg_price = recent_data['close'].mean()
            price_change_pct = (recent_price_change / avg_price) * 100
            
            # High effort (volume) with low result (price change) = absorption
            effort_vs_result_score = price_change_pct / max(0.1, recent_data['volume_ratio'].mean())
            
            # Institutional footprint (composite score)
            institutional_footprint = 0.0
            if is_accumulating:
                institutional_footprint += 0.3
            if is_distributing:
                institutional_footprint += 0.3
            if absorption_detected:
                institutional_footprint += 0.2
            if effort_vs_result_score < 0.5:  # High effort, low result
                institutional_footprint += 0.2
            
            confidence = min(0.9, institutional_footprint + 0.3)
            
            return CompositeOperatorAnalysis(
                is_accumulating=is_accumulating,
                is_distributing=is_distributing,
                absorption_detected=absorption_detected,
                effort_vs_result_score=effort_vs_result_score,
                institutional_footprint=institutional_footprint,
                confidence=confidence,
                metadata={
                    'recent_volume': float(recent_volume),
                    'price_change_pct': price_change_pct,
                    'effort_result_ratio': effort_vs_result_score
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error analyzing composite operator: {e}")
            return CompositeOperatorAnalysis(
                is_accumulating=False,
                is_distributing=False,
                absorption_detected=False,
                effort_vs_result_score=0.0,
                institutional_footprint=0.0,
                confidence=0.0,
                metadata={}
            )
    
    async def _measure_cause_effect(
        self, 
        df: pd.DataFrame,
        schematic: WyckoffSchematic,
        phase_history: List[WyckoffPhaseInfo]
    ) -> Optional[CauseEffect]:
        """Measure Cause and Effect for price targets"""
        try:
            if not phase_history or schematic == WyckoffSchematic.UNKNOWN:
                return None
            
            # Simplified Point & Figure count (in production would use actual P&F chart)
            # Cause = horizontal accumulation/distribution width
            # Effect = vertical move potential
            
            # Estimate cause size from phase duration and range
            recent_data = df.tail(50)
            range_size = recent_data['high'].max() - recent_data['low'].min()
            time_in_range = len(recent_data)
            
            # Cause score (more time + wider range = larger cause)
            cause_size = (time_in_range / 50) * (range_size / recent_data['close'].mean())
            
            # Effect projection (cause * multiplier = potential move)
            # Classic Wyckoff: 1 unit of cause = 1 unit of effect
            current_price = df['close'].iloc[-1]
            potential_effect = range_size * cause_size
            
            if schematic in [WyckoffSchematic.ACCUMULATION, WyckoffSchematic.REACCUMULATION]:
                target_price = current_price + potential_effect
            else:  # Distribution
                target_price = current_price - potential_effect
            
            confidence = min(0.8, cause_size)
            
            return CauseEffect(
                cause_size=cause_size,
                potential_effect=potential_effect,
                target_price=target_price,
                confidence=confidence,
                metadata={
                    'range_size': range_size,
                    'time_in_range': time_in_range,
                    'current_price': current_price
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error measuring cause-effect: {e}")
            return None
    
    async def _generate_wyckoff_signals(
        self,
        schematic: WyckoffSchematic,
        phase: WyckoffPhase,
        events: List[WyckoffEvent],
        composite_operator: CompositeOperatorAnalysis
    ) -> List[Dict[str, Any]]:
        """Generate trading signals based on Wyckoff analysis"""
        signals = []
        
        try:
            # Signal from current phase
            if phase == WyckoffPhase.SPRING:
                signals.append({
                    'type': 'wyckoff_spring',
                    'direction': 'bullish',
                    'confidence': 0.85,
                    'reasoning': 'Spring detected - final shakeout before markup',
                    'priority': 'high'
                })
            elif phase == WyckoffPhase.UTAD:
                signals.append({
                    'type': 'wyckoff_utad',
                    'direction': 'bearish',
                    'confidence': 0.85,
                    'reasoning': 'UTAD detected - final pump before markdown',
                    'priority': 'high'
                })
            elif phase == WyckoffPhase.SOS:
                signals.append({
                    'type': 'wyckoff_sos',
                    'direction': 'bullish',
                    'confidence': 0.75,
                    'reasoning': 'Sign of Strength - bullish breakout',
                    'priority': 'medium'
                })
            elif phase == WyckoffPhase.SOW:
                signals.append({
                    'type': 'wyckoff_sow',
                    'direction': 'bearish',
                    'confidence': 0.75,
                    'reasoning': 'Sign of Weakness - bearish breakdown',
                    'priority': 'medium'
                })
            
            # Signal from composite operator
            if composite_operator.is_accumulating and composite_operator.confidence > 0.7:
                signals.append({
                    'type': 'composite_operator',
                    'direction': 'bullish',
                    'confidence': composite_operator.confidence,
                    'reasoning': 'Smart money accumulation detected',
                    'priority': 'high'
                })
            elif composite_operator.is_distributing and composite_operator.confidence > 0.7:
                signals.append({
                    'type': 'composite_operator',
                    'direction': 'bearish',
                    'confidence': composite_operator.confidence,
                    'reasoning': 'Smart money distribution detected',
                    'priority': 'high'
                })
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Error generating Wyckoff signals: {e}")
            return signals
    
    async def _calculate_overall_confidence(
        self,
        phase: WyckoffPhase,
        events: List[WyckoffEvent],
        composite_operator: CompositeOperatorAnalysis
    ) -> float:
        """Calculate overall Wyckoff analysis confidence"""
        try:
            confidence = 0.5
            
            # High-confidence phases
            if phase in [WyckoffPhase.SPRING, WyckoffPhase.UTAD]:
                confidence += 0.3
            elif phase in [WyckoffPhase.SOS, WyckoffPhase.SOW]:
                confidence += 0.2
            
            # Event confirmation
            if events:
                avg_event_confidence = sum(e.confidence for e in events[-5:]) / min(5, len(events))
                confidence += avg_event_confidence * 0.2
            
            # Composite operator confirmation
            confidence += composite_operator.confidence * 0.2
            
            return min(0.95, confidence)
            
        except Exception as e:
            self.logger.error(f"Error calculating overall confidence: {e}")
            return 0.5
    
    def _get_default_analysis(self, symbol: str, timeframe: str) -> WyckoffAnalysis:
        """Get default analysis when insufficient data"""
        return WyckoffAnalysis(
            symbol=symbol,
            timeframe=timeframe,
            timestamp=datetime.now(),
            current_schematic=WyckoffSchematic.UNKNOWN,
            current_phase=WyckoffPhase.UNKNOWN,
            phase_history=[],
            wyckoff_events=[],
            composite_operator=CompositeOperatorAnalysis(
                is_accumulating=False,
                is_distributing=False,
                absorption_detected=False,
                effort_vs_result_score=0.0,
                institutional_footprint=0.0,
                confidence=0.0,
                metadata={}
            ),
            cause_effect=None,
            overall_confidence=0.0,
            wyckoff_signals=[],
            metadata={'error': 'Insufficient data for analysis'}
        )
    
    async def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        return {
            'stats': self.stats,
            'config': self.config,
            'last_update': datetime.now().isoformat()
        }

