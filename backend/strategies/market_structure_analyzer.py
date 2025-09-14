"""
Market Structure Analyzer for AlphaPulse
Comprehensive market structure analysis including HH/LH/HL/LL detection and trend lines
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass
from enum import Enum
import asyncio
import math

# Import support/resistance analyzer
try:
    from .dynamic_support_resistance_analyzer import DynamicSupportResistanceAnalyzer, SupportResistanceAnalysis
except ImportError:
    # Fallback for testing
    DynamicSupportResistanceAnalyzer = None
    SupportResistanceAnalysis = None

logger = logging.getLogger(__name__)

class MarketStructureType(Enum):
    """Market structure types"""
    UPTREND = "uptrend"
    DOWNTREND = "downtrend"
    CONSOLIDATION = "consolidation"
    SIDEWAYS = "sideways"
    UNKNOWN = "unknown"

class SwingPointType(Enum):
    """Swing point types"""
    HIGH = "high"
    LOW = "low"

class TrendLineType(Enum):
    """Trend line types"""
    SUPPORT = "support"
    RESISTANCE = "resistance"
    DYNAMIC = "dynamic"

@dataclass
class SwingPoint:
    """Swing point data structure"""
    timestamp: datetime
    price: float
    volume: float
    swing_type: SwingPointType
    is_higher_high: bool = False
    is_lower_high: bool = False
    is_higher_low: bool = False
    is_lower_low: bool = False
    swing_strength: float = 0.0
    volume_confirmation: bool = False
    momentum_confirmation: bool = False

@dataclass
class TrendLine:
    """Trend line data structure"""
    start_time: datetime
    end_time: datetime
    start_price: float
    end_price: float
    trend_line_type: TrendLineType
    direction: str  # 'up' or 'down'
    touch_count: int = 0
    touch_points: List[Dict] = None
    validation_score: float = 0.0
    is_active: bool = True
    is_broken: bool = False
    break_time: Optional[datetime] = None
    break_price: Optional[float] = None
    slope_angle: float = 0.0
    duration_bars: int = 0

@dataclass
class MarketStructureAnalysis:
    """Market structure analysis result"""
    symbol: str
    timeframe: str
    timestamp: datetime
    market_structure_type: MarketStructureType
    structure_strength: float
    structure_breakout: bool = False
    breakout_direction: Optional[str] = None
    higher_highs: List[SwingPoint] = None
    lower_highs: List[SwingPoint] = None
    higher_lows: List[SwingPoint] = None
    lower_lows: List[SwingPoint] = None
    trend_lines: List[TrendLine] = None
    trend_line_breaks: List[Dict] = None
    analysis_confidence: float = 0.0
    last_swing_high: Optional[float] = None
    last_swing_low: Optional[float] = None
    current_structure_phase: str = "unknown"
    structure_duration_bars: int = 0
    structure_quality_score: float = 0.0
    # Enhanced with support/resistance analysis
    support_resistance_analysis: Optional['SupportResistanceAnalysis'] = None
    dynamic_levels_count: int = 0
    volume_confirmed_levels: int = 0
    psychological_levels_count: int = 0

class MarketStructureAnalyzer:
    """Advanced market structure analyzer"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logger
        
        # Analysis settings
        self.min_swing_distance = self.config.get('min_swing_distance', 0.005)  # 0.5% minimum swing distance
        self.min_touch_count = self.config.get('min_touch_count', 2)  # Minimum touches for trend line validation
        self.lookback_periods = self.config.get('lookback_periods', 50)  # Periods to look back for structure analysis
        self.trend_line_tolerance = self.config.get('trend_line_tolerance', 0.002)  # 0.2% tolerance for trend line touches
        
        # Support/Resistance analyzer
        if DynamicSupportResistanceAnalyzer:
            self.sr_analyzer = DynamicSupportResistanceAnalyzer(self.config)
        else:
            self.sr_analyzer = None
        
        # Performance tracking
        self.stats = {
            'analyses_performed': 0,
            'swing_points_detected': 0,
            'trend_lines_detected': 0,
            'structure_breakouts_detected': 0,
            'support_resistance_analyses': 0,
            'last_update': None
        }
        
    async def analyze_market_structure(self, symbol: str, timeframe: str, 
                                     candlestick_data: List[Dict]) -> MarketStructureAnalysis:
        """Analyze market structure for a symbol"""
        try:
            if len(candlestick_data) < self.lookback_periods:
                self.logger.warning(f"Insufficient data for {symbol}: {len(candlestick_data)} < {self.lookback_periods}")
                return self._get_default_analysis(symbol, timeframe)
            
            # Convert to DataFrame
            df = pd.DataFrame(candlestick_data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            # Detect swing points
            swing_points = await self._detect_swing_points(df)
            
            # Classify swing points (HH/LH/HL/LL)
            classified_swings = await self._classify_swing_points(swing_points)
            
            # Detect trend lines
            trend_lines = await self._detect_trend_lines(df, classified_swings)
            
            # Analyze market structure
            structure_type, structure_strength = await self._analyze_structure_type(classified_swings)
            
            # Check for structure breakouts
            structure_breakout, breakout_direction = await self._check_structure_breakout(
                df, classified_swings, structure_type
            )
            
            # Calculate analysis confidence
            analysis_confidence = await self._calculate_analysis_confidence(
                classified_swings, trend_lines, structure_type
            )
            
            # Get current structure phase
            current_phase = await self._determine_structure_phase(classified_swings, structure_type)
            
            # Perform support/resistance analysis
            sr_analysis = None
            dynamic_levels_count = 0
            volume_confirmed_levels = 0
            psychological_levels_count = 0
            
            if self.sr_analyzer:
                sr_analysis = await self.sr_analyzer.analyze_support_resistance(
                    symbol, timeframe, candlestick_data
                )
                dynamic_levels_count = len(sr_analysis.support_levels) + len(sr_analysis.resistance_levels)
                volume_confirmed_levels = len([l for l in sr_analysis.support_levels + sr_analysis.resistance_levels if l.volume_confirmation])
                psychological_levels_count = len(sr_analysis.psychological_levels)
                self.stats['support_resistance_analyses'] += 1
            
            # Create analysis result
            analysis = MarketStructureAnalysis(
                symbol=symbol,
                timeframe=timeframe,
                timestamp=df['timestamp'].iloc[-1],
                market_structure_type=structure_type,
                structure_strength=structure_strength,
                structure_breakout=structure_breakout,
                breakout_direction=breakout_direction,
                higher_highs=[s for s in classified_swings if s.is_higher_high],
                lower_highs=[s for s in classified_swings if s.is_lower_high],
                higher_lows=[s for s in classified_swings if s.is_higher_low],
                lower_lows=[s for s in classified_swings if s.is_lower_low],
                trend_lines=trend_lines,
                analysis_confidence=analysis_confidence,
                last_swing_high=classified_swings[-1].price if classified_swings and classified_swings[-1].swing_type == SwingPointType.HIGH else None,
                last_swing_low=classified_swings[-1].price if classified_swings and classified_swings[-1].swing_type == SwingPointType.LOW else None,
                current_structure_phase=current_phase,
                structure_duration_bars=len(df),
                structure_quality_score=structure_strength,
                # Enhanced with support/resistance analysis
                support_resistance_analysis=sr_analysis,
                dynamic_levels_count=dynamic_levels_count,
                volume_confirmed_levels=volume_confirmed_levels,
                psychological_levels_count=psychological_levels_count
            )
            
            # Update statistics
            self.stats['analyses_performed'] += 1
            self.stats['swing_points_detected'] += len(classified_swings)
            self.stats['trend_lines_detected'] += len(trend_lines)
            if structure_breakout:
                self.stats['structure_breakouts_detected'] += 1
            self.stats['last_update'] = datetime.now(timezone.utc)
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing market structure for {symbol}: {e}")
            return self._get_default_analysis(symbol, timeframe)
    
    async def _detect_swing_points(self, df: pd.DataFrame) -> List[SwingPoint]:
        """Detect swing highs and lows"""
        swing_points = []
        
        try:
            # Need at least 5 bars to detect swings (2 before + current + 2 after)
            if len(df) < 5:
                self.logger.warning(f"Insufficient data for swing detection: {len(df)} < 5")
                return swing_points
            
            self.logger.info(f"Detecting swing points in {len(df)} bars")
            
            # Use a more robust swing detection algorithm
            for i in range(2, len(df) - 2):
                current_high = df.iloc[i]['high']
                current_low = df.iloc[i]['low']
                
                # Check previous 2 bars
                prev_high_1 = df.iloc[i-1]['high']
                prev_high_2 = df.iloc[i-2]['high']
                prev_low_1 = df.iloc[i-1]['low']
                prev_low_2 = df.iloc[i-2]['low']
                
                # Check next 2 bars
                next_high_1 = df.iloc[i+1]['high']
                next_high_2 = df.iloc[i+2]['high']
                next_low_1 = df.iloc[i+1]['low']
                next_low_2 = df.iloc[i+2]['low']
                
                # Detect swing high (current high is higher than surrounding highs)
                if (current_high > prev_high_1 and current_high > prev_high_2 and 
                    current_high > next_high_1 and current_high > next_high_2):
                    swing_points.append(SwingPoint(
                        timestamp=df.iloc[i]['timestamp'],
                        price=current_high,
                        volume=df.iloc[i]['volume'],
                        swing_type=SwingPointType.HIGH
                    ))
                    self.logger.debug(f"Swing high detected at index {i}, price: {current_high}")
                
                # Detect swing low (current low is lower than surrounding lows)
                if (current_low < prev_low_1 and current_low < prev_low_2 and 
                    current_low < next_low_1 and current_low < next_low_2):
                    swing_points.append(SwingPoint(
                        timestamp=df.iloc[i]['timestamp'],
                        price=current_low,
                        volume=df.iloc[i]['volume'],
                        swing_type=SwingPointType.LOW
                    ))
                    self.logger.debug(f"Swing low detected at index {i}, price: {current_low}")
            
            self.logger.info(f"Detected {len(swing_points)} swing points")
            return swing_points
            
        except Exception as e:
            self.logger.error(f"Error detecting swing points: {e}")
            return []
    
    async def _classify_swing_points(self, swing_points: List[SwingPoint]) -> List[SwingPoint]:
        """Classify swing points as HH/LH/HL/LL"""
        if len(swing_points) < 2:
            return swing_points
        
        try:
            for i in range(1, len(swing_points)):
                current = swing_points[i]
                previous = swing_points[i-1]
                
                # Calculate price difference
                price_diff = (current.price - previous.price) / previous.price
                
                if current.swing_type == SwingPointType.HIGH:
                    if price_diff > self.min_swing_distance:
                        current.is_higher_high = True
                    elif price_diff < -self.min_swing_distance:
                        current.is_lower_high = True
                
                elif current.swing_type == SwingPointType.LOW:
                    if price_diff > self.min_swing_distance:
                        current.is_higher_low = True
                    elif price_diff < -self.min_swing_distance:
                        current.is_lower_low = True
                
                # Calculate swing strength
                current.swing_strength = abs(price_diff)
                
                # Check volume confirmation
                current.volume_confirmation = current.volume > previous.volume * 1.2
                
                # Check momentum confirmation (simplified)
                current.momentum_confirmation = current.swing_strength > 0.01
            
            return swing_points
            
        except Exception as e:
            self.logger.error(f"Error classifying swing points: {e}")
            return swing_points
    
    async def _detect_trend_lines(self, df: pd.DataFrame, swing_points: List[SwingPoint]) -> List[TrendLine]:
        """Detect trend lines from swing points"""
        trend_lines = []
        
        try:
            if len(swing_points) < 3:
                return trend_lines
            
            # Group swing points by type
            highs = [s for s in swing_points if s.swing_type == SwingPointType.HIGH]
            lows = [s for s in swing_points if s.swing_type == SwingPointType.LOW]
            
            # Detect resistance trend lines (from highs)
            resistance_lines = await self._detect_trend_lines_from_points(highs, TrendLineType.RESISTANCE)
            trend_lines.extend(resistance_lines)
            
            # Detect support trend lines (from lows)
            support_lines = await self._detect_trend_lines_from_points(lows, TrendLineType.SUPPORT)
            trend_lines.extend(support_lines)
            
            return trend_lines
            
        except Exception as e:
            self.logger.error(f"Error detecting trend lines: {e}")
            return []
    
    async def _detect_trend_lines_from_points(self, points: List[SwingPoint], 
                                            line_type: TrendLineType) -> List[TrendLine]:
        """Detect trend lines from a list of swing points"""
        trend_lines = []
        
        try:
            if len(points) < 2:
                return trend_lines
            
            # Try to connect points to form trend lines
            for i in range(len(points) - 1):
                for j in range(i + 1, len(points)):
                    start_point = points[i]
                    end_point = points[j]
                    
                    # Calculate line parameters
                    time_diff = (end_point.timestamp - start_point.timestamp).total_seconds()
                    price_diff = end_point.price - start_point.price
                    
                    if time_diff == 0:
                        continue
                    
                    slope = price_diff / time_diff
                    slope_angle = math.degrees(math.atan(slope))
                    
                    # Check if other points touch this line
                    touch_points = []
                    for k, point in enumerate(points):
                        if k == i or k == j:
                            continue
                        
                        # Calculate expected price on the line
                        time_from_start = (point.timestamp - start_point.timestamp).total_seconds()
                        expected_price = start_point.price + (slope * time_from_start)
                        
                        # Check if point touches the line (within tolerance)
                        price_diff_ratio = abs(point.price - expected_price) / expected_price
                        if price_diff_ratio <= self.trend_line_tolerance:
                            touch_points.append({
                                'timestamp': point.timestamp,
                                'price': point.price,
                                'index': k
                            })
                    
                    # Create trend line if enough touches
                    if len(touch_points) >= self.min_touch_count - 2:  # -2 for start and end points
                        trend_line = TrendLine(
                            start_time=start_point.timestamp,
                            end_time=end_point.timestamp,
                            start_price=start_point.price,
                            end_price=end_point.price,
                            trend_line_type=line_type,
                            direction='up' if slope > 0 else 'down',
                            touch_count=len(touch_points) + 2,
                            touch_points=touch_points,
                            validation_score=min(1.0, (len(touch_points) + 2) / 5.0),  # Normalize to 0-1
                            slope_angle=slope_angle,
                            duration_bars=len(points)
                        )
                        trend_lines.append(trend_line)
            
            return trend_lines
            
        except Exception as e:
            self.logger.error(f"Error detecting trend lines from points: {e}")
            return []
    
    async def _analyze_structure_type(self, swing_points: List[SwingPoint]) -> Tuple[MarketStructureType, float]:
        """Analyze market structure type and strength"""
        try:
            if len(swing_points) < 4:
                return MarketStructureType.UNKNOWN, 0.0
            
            # Count different types of swings
            hh_count = len([s for s in swing_points if s.is_higher_high])
            lh_count = len([s for s in swing_points if s.is_lower_high])
            hl_count = len([s for s in swing_points if s.is_higher_low])
            ll_count = len([s for s in swing_points if s.is_lower_low])
            
            self.logger.debug(f"Structure analysis - HH: {hh_count}, LH: {lh_count}, HL: {hl_count}, LL: {ll_count}")
            
            # Determine structure type with more lenient conditions
            if hh_count >= lh_count and hl_count >= ll_count and (hh_count > 0 or hl_count > 0):
                structure_type = MarketStructureType.UPTREND
                strength = min(1.0, (hh_count + hl_count) / max(len(swing_points), 1))
            elif lh_count >= hh_count and ll_count >= hl_count and (lh_count > 0 or ll_count > 0):
                structure_type = MarketStructureType.DOWNTREND
                strength = min(1.0, (lh_count + ll_count) / max(len(swing_points), 1))
            elif abs(hh_count - lh_count) <= 1 and abs(hl_count - ll_count) <= 1:
                structure_type = MarketStructureType.CONSOLIDATION
                strength = 0.5
            else:
                structure_type = MarketStructureType.SIDEWAYS
                strength = 0.3
            
            self.logger.debug(f"Determined structure type: {structure_type.value}, strength: {strength}")
            return structure_type, strength
            
        except Exception as e:
            self.logger.error(f"Error analyzing structure type: {e}")
            return MarketStructureType.UNKNOWN, 0.0
    
    async def _check_structure_breakout(self, df: pd.DataFrame, swing_points: List[SwingPoint], 
                                      structure_type: MarketStructureType) -> Tuple[bool, Optional[str]]:
        """Check for market structure breakouts"""
        try:
            if len(swing_points) < 2:
                return False, None
            
            current_price = df['close'].iloc[-1]
            last_swing = swing_points[-1]
            
            # Check for structure breakout based on current structure type
            if structure_type == MarketStructureType.UPTREND:
                # Check if price breaks below the last higher low
                higher_lows = [s for s in swing_points if s.is_higher_low]
                if higher_lows and current_price < higher_lows[-1].price:
                    return True, 'down'
            
            elif structure_type == MarketStructureType.DOWNTREND:
                # Check if price breaks above the last lower high
                lower_highs = [s for s in swing_points if s.is_lower_high]
                if lower_highs and current_price > lower_highs[-1].price:
                    return True, 'up'
            
            return False, None
            
        except Exception as e:
            self.logger.error(f"Error checking structure breakout: {e}")
            return False, None
    
    async def _calculate_analysis_confidence(self, swing_points: List[SwingPoint], 
                                           trend_lines: List[TrendLine], 
                                           structure_type: MarketStructureType) -> float:
        """Calculate overall analysis confidence"""
        try:
            confidence = 0.5  # Base confidence
            
            # Swing point confidence
            if swing_points:
                avg_swing_strength = np.mean([s.swing_strength for s in swing_points])
                volume_confirmed = np.mean([s.volume_confirmation for s in swing_points])
                momentum_confirmed = np.mean([s.momentum_confirmation for s in swing_points])
                
                confidence += avg_swing_strength * 0.2
                confidence += volume_confirmed * 0.15
                confidence += momentum_confirmed * 0.15
            
            # Trend line confidence
            if trend_lines:
                avg_validation = np.mean([t.validation_score for t in trend_lines])
                confidence += avg_validation * 0.1
            
            # Structure type confidence
            if structure_type != MarketStructureType.UNKNOWN:
                confidence += 0.1
            
            return min(1.0, confidence)
            
        except Exception as e:
            self.logger.error(f"Error calculating analysis confidence: {e}")
            return 0.5
    
    async def _determine_structure_phase(self, swing_points: List[SwingPoint], 
                                       structure_type: MarketStructureType) -> str:
        """Determine current market structure phase"""
        try:
            if len(swing_points) < 4:
                return "unknown"
            
            # Analyze recent swing patterns
            recent_swings = swing_points[-4:]
            hh_count = len([s for s in recent_swings if s.is_higher_high])
            lh_count = len([s for s in recent_swings if s.is_lower_high])
            hl_count = len([s for s in recent_swings if s.is_higher_low])
            ll_count = len([s for s in recent_swings if s.is_lower_low])
            
            if structure_type == MarketStructureType.UPTREND:
                if hh_count > lh_count and hl_count > ll_count:
                    return "markup"
                else:
                    return "accumulation"
            
            elif structure_type == MarketStructureType.DOWNTREND:
                if lh_count > hh_count and ll_count > hl_count:
                    return "markdown"
                else:
                    return "distribution"
            
            else:
                return "consolidation"
            
        except Exception as e:
            self.logger.error(f"Error determining structure phase: {e}")
            return "unknown"
    
    def _get_default_analysis(self, symbol: str, timeframe: str) -> MarketStructureAnalysis:
        """Get default analysis when insufficient data"""
        return MarketStructureAnalysis(
            symbol=symbol,
            timeframe=timeframe,
            timestamp=datetime.now(timezone.utc),
            market_structure_type=MarketStructureType.UNKNOWN,
            structure_strength=0.0,
            analysis_confidence=0.0,
            current_structure_phase="unknown"
        )

# Global instance
market_structure_analyzer = MarketStructureAnalyzer()
