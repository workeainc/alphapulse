"""
Advanced Pattern Detection Strategy for AlphaPlus
Detects complex candlestick patterns with machine learning enhancement
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass
from enum import Enum
import asyncio

# Import our enhanced components
try:
    from ..src.data.storage import DataStorage
    from ..src.database.connection import TimescaleDBConnection
except ImportError:
    # Fallback for testing
    DataStorage = None
    TimescaleDBConnection = None

try:
    from .market_structure_analyzer import MarketStructureAnalyzer, MarketStructureAnalysis
    from .dynamic_support_resistance_analyzer import DynamicSupportResistanceAnalyzer, SupportResistanceAnalysis
    from .demand_supply_zone_analyzer import DemandSupplyZoneAnalyzer, DemandSupplyAnalysis
except ImportError:
    # Handle import errors for market structure and SR components
    MarketStructureAnalyzer = None
    MarketStructureAnalysis = None
    DynamicSupportResistanceAnalyzer = None
    SupportResistanceAnalysis = None
    DemandSupplyZoneAnalyzer = None
    DemandSupplyAnalysis = None

logger = logging.getLogger(__name__)

class PatternType(Enum):
    """Candlestick pattern types"""
    BULLISH_ENGULFING = "bullish_engulfing"
    BEARISH_ENGULFING = "bearish_engulfing"
    HAMMER = "hammer"
    SHOOTING_STAR = "shooting_star"
    DOJI = "doji"
    MORNING_STAR = "morning_star"
    EVENING_STAR = "evening_star"
    THREE_WHITE_SOLDIERS = "three_white_soldiers"
    THREE_BLACK_CROWS = "three_black_crows"
    TRIANGLE_ASCENDING = "triangle_ascending"
    TRIANGLE_DESCENDING = "triangle_descending"
    TRIANGLE_SYMMETRICAL = "triangle_symmetrical"
    WEDGE_RISING = "wedge_rising"
    WEDGE_FALLING = "wedge_falling"
    DOUBLE_TOP = "double_top"
    DOUBLE_BOTTOM = "double_bottom"
    HEAD_AND_SHOULDERS = "head_and_shoulders"
    INVERSE_HEAD_AND_SHOULDERS = "inverse_head_and_shoulders"

class PatternStrength(Enum):
    """Pattern strength levels"""
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    VERY_STRONG = "very_strong"

@dataclass
class PatternResult:
    """Result of pattern detection"""
    pattern_type: PatternType
    strength: PatternStrength
    confidence: float  # 0.0 to 1.0
    entry_price: float
    stop_loss: Optional[float]
    take_profit: Optional[float]
    timestamp: datetime
    symbol: str
    timeframe: str
    volume_confirmation: bool
    trend_alignment: str  # "bullish", "bearish", "neutral"
    metadata: Dict[str, Any]

class AdvancedPatternDetector:
    """Advanced candlestick pattern detector with ML enhancement"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logger
        
        # Pattern detection settings
        self.min_pattern_bars = self.config.get('min_pattern_bars', 3)
        self.max_pattern_bars = self.config.get('max_pattern_bars', 20)
        self.min_confidence = self.config.get('min_confidence', 0.6)
        self.volume_threshold = self.config.get('volume_threshold', 1.5)
        
        # Data storage
        self.storage = None
        self.db_connection = None
        
        # Pattern recognition models (placeholder for ML models)
        self.pattern_models = {}
        
        # Market structure analyzer
        self.market_structure_analyzer = None
        
        # Dynamic support/resistance analyzer
        self.support_resistance_analyzer = None
        
        # Demand/supply zone analyzer
        self.demand_supply_analyzer = None
        
        # Performance tracking
        self.stats = {
            'patterns_detected': 0,
            'high_confidence_patterns': 0,
            'false_positives': 0,
            'last_update': None
        }
        
    async def initialize(self):
        """Initialize the pattern detector"""
        try:
            self.logger.info("Initializing Advanced Pattern Detector...")
            
            # Initialize data storage
            if DataStorage:
                self.storage = DataStorage(
                    storage_path=self.config.get('storage_path', 'data'),
                    db_config=self.config.get('db_config', {})
                )
                await self.storage.initialize()
            
            # Initialize database connection
            if TimescaleDBConnection:
                self.db_connection = TimescaleDBConnection(
                    self.config.get('db_config', {})
                )
                await self.db_connection.initialize()
            
            # Load pattern recognition models
            await self._load_pattern_models()
            
            # Initialize market structure analyzer
            if MarketStructureAnalyzer:
                self.market_structure_analyzer = MarketStructureAnalyzer(self.config)
            
            # Initialize dynamic support/resistance analyzer
            if DynamicSupportResistanceAnalyzer:
                self.support_resistance_analyzer = DynamicSupportResistanceAnalyzer(self.config)
            
            # Initialize demand/supply zone analyzer
            if DemandSupplyZoneAnalyzer:
                self.demand_supply_analyzer = DemandSupplyZoneAnalyzer(self.config)
            
            self.logger.info("Advanced Pattern Detector initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Pattern Detector: {e}")
            raise
    
    async def _load_pattern_models(self):
        """Load pattern recognition models"""
        try:
            # Placeholder for ML model loading
            # In production, this would load trained models for pattern recognition
            self.pattern_models = {
                'engulfing': None,
                'reversal': None,
                'continuation': None,
                'consolidation': None
            }
            
            self.logger.info("Pattern models loaded (placeholder)")
            
        except Exception as e:
            self.logger.error(f"Error loading pattern models: {e}")
    
    async def detect_patterns(self, symbol: str, timeframe: str, 
                            limit: int = 100) -> List[PatternResult]:
        """Detect patterns in candlestick data"""
        try:
            # Get candlestick data
            candlestick_data = await self._get_candlestick_data(symbol, timeframe, limit)
            if not candlestick_data or len(candlestick_data) < self.min_pattern_bars:
                return []
            
            # Convert to DataFrame
            df = pd.DataFrame(candlestick_data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            # Detect patterns
            patterns = []
            
            # Basic pattern detection
            patterns.extend(await self._detect_basic_patterns(df, symbol, timeframe))
            
            # Advanced pattern detection
            patterns.extend(await self._detect_advanced_patterns(df, symbol, timeframe))
            
            # Market structure analysis
            market_structure = None
            if self.market_structure_analyzer:
                market_structure = await self.market_structure_analyzer.analyze_market_structure(
                    symbol, timeframe, candlestick_data
                )
            
            # Volume and trend analysis
            patterns = await self._enhance_patterns_with_volume_trend(patterns, df)
            
            # Enhance patterns with market structure
            patterns = await self._enhance_patterns_with_market_structure(patterns, market_structure)
            
            # Enhance patterns with support/resistance analysis
            if market_structure and market_structure.support_resistance_analysis:
                patterns = await self._enhance_patterns_with_support_resistance(
                    patterns, market_structure.support_resistance_analysis
                )
            
            # Enhance patterns with demand/supply zone analysis
            if self.demand_supply_analyzer:
                demand_supply_analysis = await self.demand_supply_analyzer.analyze_demand_supply_zones(
                    symbol, timeframe, df
                )
                patterns = await self._enhance_patterns_with_demand_supply_zones(
                    patterns, demand_supply_analysis
                )
            
            # Filter by confidence
            patterns = [p for p in patterns if p.confidence >= self.min_confidence]
            
            # Update statistics
            self.stats['patterns_detected'] += len(patterns)
            self.stats['high_confidence_patterns'] += len([p for p in patterns if p.confidence >= 0.8])
            self.stats['last_update'] = datetime.now(timezone.utc)
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Error detecting patterns: {e}")
            return []
    
    async def _get_candlestick_data(self, symbol: str, timeframe: str, 
                                   limit: int) -> List[Dict[str, Any]]:
        """Get candlestick data from storage"""
        try:
            if self.db_connection:
                # Try database first
                data = await self.db_connection.get_candlestick_data(
                            symbol=symbol,
                            timeframe=timeframe,
                    limit=limit
                )
                if data:
                    return data
            
            if self.storage:
                # Fallback to storage
                key = f"candlestick_{symbol}_{timeframe}"
                data = await self.storage.load_data(key)
                if data:
                    return data if isinstance(data, list) else [data]
            
            return []
                        
        except Exception as e:
            self.logger.error(f"Error getting candlestick data: {e}")
            return []
    
    async def _detect_basic_patterns(self, df: pd.DataFrame, symbol: str, 
                                   timeframe: str) -> List[PatternResult]:
        """Detect basic candlestick patterns"""
        patterns = []
        
        try:
            # Need at least 3 bars for basic patterns
            if len(df) < 3:
                return patterns
            
            for i in range(2, len(df)):
                # Engulfing patterns
                if await self._is_bullish_engulfing(df, i):
                    patterns.append(await self._create_pattern_result(
                        PatternType.BULLISH_ENGULFING, df, i, symbol, timeframe
                    ))
                
                if await self._is_bearish_engulfing(df, i):
                    patterns.append(await self._create_pattern_result(
                        PatternType.BEARISH_ENGULFING, df, i, symbol, timeframe
                    ))
                
                # Hammer and Shooting Star
                if await self._is_hammer(df, i):
                    patterns.append(await self._create_pattern_result(
                        PatternType.HAMMER, df, i, symbol, timeframe
                    ))
                
                if await self._is_shooting_star(df, i):
                    patterns.append(await self._create_pattern_result(
                        PatternType.SHOOTING_STAR, df, i, symbol, timeframe
                    ))
                
                # Doji
                if await self._is_doji(df, i):
                    patterns.append(await self._create_pattern_result(
                        PatternType.DOJI, df, i, symbol, timeframe
                    ))
            
            return patterns
                        
        except Exception as e:
            self.logger.error(f"Error detecting basic patterns: {e}")
            return []
    
    async def _detect_advanced_patterns(self, df: pd.DataFrame, symbol: str, 
                                      timeframe: str) -> List[PatternResult]:
        """Detect advanced patterns (multi-bar patterns)"""
        patterns = []
        
        try:
            # Need at least 5 bars for advanced patterns
            if len(df) < 5:
                return patterns
            
            for i in range(4, len(df)):
                # Morning Star and Evening Star
                if await self._is_morning_star(df, i):
                    patterns.append(await self._create_pattern_result(
                        PatternType.MORNING_STAR, df, i, symbol, timeframe
                    ))
                
                if await self._is_evening_star(df, i):
                    patterns.append(await self._create_pattern_result(
                        PatternType.EVENING_STAR, df, i, symbol, timeframe
                    ))
                
                # Three White Soldiers and Three Black Crows
                if await self._is_three_white_soldiers(df, i):
                    patterns.append(await self._create_pattern_result(
                        PatternType.THREE_WHITE_SOLDIERS, df, i, symbol, timeframe
                    ))
                
                if await self._is_three_black_crows(df, i):
                    patterns.append(await self._create_pattern_result(
                        PatternType.THREE_BLACK_CROWS, df, i, symbol, timeframe
                    ))
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Error detecting advanced patterns: {e}")
            return []
    
    async def _is_bullish_engulfing(self, df: pd.DataFrame, i: int) -> bool:
        """Check if bars i-1 and i form a bullish engulfing pattern using standard formula"""
        try:
            prev_open = df.iloc[i-1]['open']
            prev_close = df.iloc[i-1]['close']
            curr_open = df.iloc[i]['open']
            curr_close = df.iloc[i]['close']
            
            # Previous bar should be bearish (red)
            if prev_close >= prev_open:
                return False
            
            # Current bar should be bullish (green)
            if curr_close <= curr_open:
                return False
            
            # Standard Engulfing Pattern: Current body engulfs previous body's range
            # Current High > Prev High and Current Low < Prev Low
            return (curr_close > prev_open and curr_open < prev_close)
            
        except Exception as e:
            self.logger.error(f"Error checking bullish engulfing: {e}")
            return False
    
    async def _is_bearish_engulfing(self, df: pd.DataFrame, i: int) -> bool:
        """Check if bars i-1 and i form a bearish engulfing pattern using standard formula"""
        try:
            prev_open = df.iloc[i-1]['open']
            prev_close = df.iloc[i-1]['close']
            curr_open = df.iloc[i]['open']
            curr_close = df.iloc[i]['close']
            
            # Previous bar should be bullish (green)
            if prev_close <= prev_open:
                return False
            
            # Current bar should be bearish (red)
            if curr_close >= curr_open:
                return False
            
            # Standard Engulfing Pattern: Current body engulfs previous body's range
            # Current High > Prev High and Current Low < Prev Low
            return (curr_open > prev_close and curr_close < prev_open)
            
        except Exception as e:
            self.logger.error(f"Error checking bearish engulfing: {e}")
            return False
    
    async def _is_hammer(self, df: pd.DataFrame, i: int) -> bool:
        """Check if bar i is a hammer pattern using standard formula"""
        try:
            open_price = df.iloc[i]['open']
            high = df.iloc[i]['high']
            low = df.iloc[i]['low']
            close = df.iloc[i]['close']
            
            # Calculate body and wick sizes
            body_size = abs(close - open_price)
            lower_wick = min(open_price, close) - low
            upper_wick = high - max(open_price, close)
            total_range = high - low
            
            if total_range == 0:
                return False
            
            # Standard Hammer formula: Lower wick ≥ 2× body, upper wick ≤ 0.5× body
            # Also occurs after downtrend (simplified check)
            return (lower_wick >= 2 * body_size and 
                    upper_wick <= 0.5 * body_size and
                    body_size > 0)
            
        except Exception as e:
            self.logger.error(f"Error checking hammer: {e}")
            return False
    
    async def _is_shooting_star(self, df: pd.DataFrame, i: int) -> bool:
        """Check if bar i is a shooting star pattern using standard formula"""
        try:
            open_price = df.iloc[i]['open']
            high = df.iloc[i]['high']
            low = df.iloc[i]['low']
            close = df.iloc[i]['close']
            
            # Calculate body and wick sizes
            body_size = abs(close - open_price)
            lower_wick = min(open_price, close) - low
            upper_wick = high - max(open_price, close)
            total_range = high - low
            
            if total_range == 0:
                return False
            
            # Standard Shooting Star formula: Upper wick ≥ 2× body, lower wick ≤ 0.5× body
            # Also occurs after uptrend (simplified check)
            return (upper_wick >= 2 * body_size and 
                    lower_wick <= 0.5 * body_size and
                    body_size > 0)
            
        except Exception as e:
            self.logger.error(f"Error checking shooting star: {e}")
            return False
    
    async def _is_doji(self, df: pd.DataFrame, i: int) -> bool:
        """Check if bar i is a doji pattern using standard formula"""
        try:
            open_price = df.iloc[i]['open']
            high = df.iloc[i]['high']
            low = df.iloc[i]['low']
            close = df.iloc[i]['close']
            
            # Calculate body size and total range
            body_size = abs(close - open_price)
            total_range = high - low
            
            if total_range == 0:
                return False
            
            # Standard Doji formula: |Open - Close| ≤ 0.1% × Range
            return body_size <= (0.001 * total_range)
            
        except Exception as e:
            self.logger.error(f"Error checking doji: {e}")
            return False
    
    async def _is_hammer(self, df: pd.DataFrame, i: int) -> bool:
        """Check if bar i is a hammer pattern"""
        try:
            open_price = df.iloc[i]['open']
            high = df.iloc[i]['high']
            low = df.iloc[i]['low']
            close = df.iloc[i]['close']
            
            body = abs(close - open_price)
            lower_shadow = min(open_price, close) - low
            upper_shadow = high - max(open_price, close)
            
            # Hammer criteria
            return (lower_shadow > 2 * body and upper_shadow < body)
                        
        except Exception as e:
            self.logger.error(f"Error checking hammer: {e}")
            return False
    
    async def _is_shooting_star(self, df: pd.DataFrame, i: int) -> bool:
        """Check if bar i is a shooting star pattern"""
        try:
            open_price = df.iloc[i]['open']
            high = df.iloc[i]['high']
            low = df.iloc[i]['low']
            close = df.iloc[i]['close']
            
            body = abs(close - open_price)
            lower_shadow = min(open_price, close) - low
            upper_shadow = high - max(open_price, close)
            
            # Shooting star criteria
            return (upper_shadow > 2 * body and lower_shadow < body)
            
        except Exception as e:
            self.logger.error(f"Error checking shooting star: {e}")
            return False
    
    async def _is_doji(self, df: pd.DataFrame, i: int) -> bool:
        """Check if bar i is a doji pattern"""
        try:
            open_price = df.iloc[i]['open']
            close = df.iloc[i]['close']
            
            body = abs(close - open_price)
            total_range = df.iloc[i]['high'] - df.iloc[i]['low']
            
            # Doji criteria: body is very small compared to total range
            return (body <= total_range * 0.1)
            
        except Exception as e:
            self.logger.error(f"Error checking doji: {e}")
            return False
    
    async def _is_morning_star(self, df: pd.DataFrame, i: int) -> bool:
        """Check if bars i-2, i-1, i form a morning star pattern"""
        try:
            if i < 2:
                return False
            
            # First bar: long bearish
            first_bearish = (df.iloc[i-2]['close'] < df.iloc[i-2]['open'])
            first_long = abs(df.iloc[i-2]['close'] - df.iloc[i-2]['open']) > df.iloc[i-2]['high'] * 0.01
            
            # Second bar: small body (doji-like)
            second_small = abs(df.iloc[i-1]['close'] - df.iloc[i-1]['open']) < df.iloc[i-1]['high'] * 0.005
            
            # Third bar: bullish
            third_bullish = (df.iloc[i]['close'] > df.iloc[i]['open'])
            
            return first_bearish and first_long and second_small and third_bullish
            
        except Exception as e:
            self.logger.error(f"Error checking morning star: {e}")
            return False
    
    async def _is_evening_star(self, df: pd.DataFrame, i: int) -> bool:
        """Check if bars i-2, i-1, i form an evening star pattern"""
        try:
            if i < 2:
                return False
            
            # First bar: long bullish
            first_bullish = (df.iloc[i-2]['close'] > df.iloc[i-2]['open'])
            first_long = abs(df.iloc[i-2]['close'] - df.iloc[i-2]['open']) > df.iloc[i-2]['high'] * 0.01
            
            # Second bar: small body (doji-like)
            second_small = abs(df.iloc[i-1]['close'] - df.iloc[i-1]['open']) < df.iloc[i-1]['high'] * 0.005
            
            # Third bar: bearish
            third_bearish = (df.iloc[i]['close'] < df.iloc[i]['open'])
            
            return first_bullish and first_long and second_small and third_bearish
            
        except Exception as e:
            self.logger.error(f"Error checking evening star: {e}")
            return False
    
    async def _is_three_white_soldiers(self, df: pd.DataFrame, i: int) -> bool:
        """Check if bars i-2, i-1, i form three white soldiers pattern"""
        try:
            if i < 2:
                return False
            
            # All three bars should be bullish
            for j in range(i-2, i+1):
                if df.iloc[j]['close'] <= df.iloc[j]['open']:
                    return False
            
            # Each bar should open within the previous bar's body
            for j in range(i-1, i+1):
                prev_open = df.iloc[j-1]['open']
                prev_close = df.iloc[j-1]['close']
                curr_open = df.iloc[j]['open']
                
                if not (prev_open <= curr_open <= prev_close):
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking three white soldiers: {e}")
            return False
    
    async def _is_three_black_crows(self, df: pd.DataFrame, i: int) -> bool:
        """Check if bars i-2, i-1, i form three black crows pattern"""
        try:
            if i < 2:
                return False
            
            # All three bars should be bearish
            for j in range(i-2, i+1):
                if df.iloc[j]['close'] >= df.iloc[j]['open']:
                    return False
            
            # Each bar should open within the previous bar's body
            for j in range(i-1, i+1):
                prev_open = df.iloc[j-1]['open']
                prev_close = df.iloc[j-1]['close']
                curr_open = df.iloc[j]['open']
                
                if not (prev_close <= curr_open <= prev_open):
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking three black crows: {e}")
            return False
    
    async def _create_pattern_result(self, pattern_type: PatternType, df: pd.DataFrame, 
                                   i: int, symbol: str, timeframe: str) -> PatternResult:
        """Create a pattern result object"""
        try:
            current_bar = df.iloc[i]
            
            # Calculate confidence based on pattern strength
            confidence = await self._calculate_pattern_confidence(pattern_type, df, i)
            
            # Determine pattern strength
            strength = await self._determine_pattern_strength(confidence)
            
            # Calculate entry price, stop loss, and take profit
            entry_price = current_bar['close']
            stop_loss, take_profit = await self._calculate_risk_levels(pattern_type, df, i)
            
            # Basic metadata
            metadata = {
                'pattern_bars': 1,  # Will be updated for multi-bar patterns
                'volume_ratio': await self._calculate_volume_ratio(df, i),
                'trend_strength': await self._calculate_trend_strength(df, i),
                'support_resistance': await self._identify_support_resistance(df, i)
            }
            
            return PatternResult(
                pattern_type=pattern_type,
                strength=strength,
                confidence=confidence,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                timestamp=current_bar['timestamp'],
                symbol=symbol,
                timeframe=timeframe,
                volume_confirmation=False,  # Will be updated later
                trend_alignment='neutral',  # Will be updated later
                metadata=metadata
            )
            
        except Exception as e:
            self.logger.error(f"Error creating pattern result: {e}")
            # Return a basic pattern result
            return PatternResult(
                pattern_type=pattern_type,
                strength=PatternStrength.WEAK,
                confidence=0.5,
                entry_price=100.0,
                stop_loss=None,
                take_profit=None,
                timestamp=datetime.now(timezone.utc),
                            symbol=symbol,
                            timeframe=timeframe,
                volume_confirmation=False,
                trend_alignment='neutral',
                metadata={}
            )
    
    async def _calculate_pattern_confidence(self, pattern_type: PatternType, 
                                          df: pd.DataFrame, i: int) -> float:
        """Calculate confidence level for a pattern"""
        try:
            base_confidence = 0.7  # Base confidence for basic patterns
            
            # Adjust based on pattern type
            if pattern_type in [PatternType.MORNING_STAR, PatternType.EVENING_STAR]:
                base_confidence = 0.8
            elif pattern_type in [PatternType.THREE_WHITE_SOLDIERS, PatternType.THREE_BLACK_CROWS]:
                base_confidence = 0.75
            
            # Adjust based on volume
            volume_ratio = await self._calculate_volume_ratio(df, i)
            if volume_ratio > self.volume_threshold:
                base_confidence += 0.1
            
            # Adjust based on trend alignment
            trend_strength = await self._calculate_trend_strength(df, i)
            if trend_strength > 0.6:
                base_confidence += 0.1
            
            # Cap confidence at 1.0
            return min(base_confidence, 1.0)
            
        except Exception as e:
            self.logger.error(f"Error calculating pattern confidence: {e}")
            return 0.5
    
    async def _determine_pattern_strength(self, confidence: float) -> PatternStrength:
        """Determine pattern strength based on confidence"""
        if confidence >= 0.9:
            return PatternStrength.VERY_STRONG
        elif confidence >= 0.8:
            return PatternStrength.STRONG
        elif confidence >= 0.7:
            return PatternStrength.MODERATE
        else:
            return PatternStrength.WEAK
    
    async def _calculate_risk_levels(self, pattern_type: PatternType, 
                                   df: pd.DataFrame, i: int) -> Tuple[Optional[float], Optional[float]]:
        """Calculate stop loss and take profit levels"""
        try:
            current_bar = df.iloc[i]
            atr = await self._calculate_atr(df, i)
            
            if pattern_type in [PatternType.BULLISH_ENGULFING, PatternType.HAMMER, PatternType.MORNING_STAR]:
                # Bullish patterns
                stop_loss = current_bar['low'] - (atr * 1.5)
                take_profit = current_bar['close'] + (atr * 2.5)
            elif pattern_type in [PatternType.BEARISH_ENGULFING, PatternType.SHOOTING_STAR, PatternType.EVENING_STAR]:
                # Bearish patterns
                stop_loss = current_bar['high'] + (atr * 1.5)
                take_profit = current_bar['close'] - (atr * 2.5)
            else:
                # Neutral patterns
                stop_loss = None
                take_profit = None
            
            return stop_loss, take_profit
            
        except Exception as e:
            self.logger.error(f"Error calculating risk levels: {e}")
            return None, None
    
    async def _calculate_atr(self, df: pd.DataFrame, i: int, period: int = 14) -> float:
        """Calculate Average True Range"""
        try:
            if i < period:
                return df.iloc[i]['high'] - df.iloc[i]['low']
            
            true_ranges = []
            for j in range(i-period+1, i+1):
                high = df.iloc[j]['high']
                low = df.iloc[j]['low']
                prev_close = df.iloc[j-1]['close'] if j > 0 else high
                
                tr1 = high - low
                tr2 = abs(high - prev_close)
                tr3 = abs(low - prev_close)
                
                true_ranges.append(max(tr1, tr2, tr3))
            
            return np.mean(true_ranges)
            
        except Exception as e:
            self.logger.error(f"Error calculating ATR: {e}")
            return 0.0
    
    async def _calculate_volume_ratio(self, df: pd.DataFrame, i: int, period: int = 10) -> float:
        """Calculate volume ratio compared to recent average"""
        try:
            if i < period:
                return 1.0
            
            current_volume = df.iloc[i]['volume']
            avg_volume = df.iloc[i-period:i]['volume'].mean()
            
            return current_volume / avg_volume if avg_volume > 0 else 1.0
            
        except Exception as e:
            self.logger.error(f"Error calculating volume ratio: {e}")
            return 1.0
    
    async def _calculate_trend_strength(self, df: pd.DataFrame, i: int, period: int = 20) -> float:
        """Calculate trend strength using linear regression"""
        try:
            if i < period:
                return 0.5
            
            # Use close prices for trend calculation
            prices = df.iloc[i-period+1:i+1]['close'].values
            x = np.arange(len(prices))
            
            # Linear regression
            slope, intercept = np.polyfit(x, prices, 1)
            
            # Calculate R-squared (trend strength)
            y_pred = slope * x + intercept
            ss_res = np.sum((prices - y_pred) ** 2)
            ss_tot = np.sum((prices - np.mean(prices)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            return max(0.0, min(1.0, r_squared))
            
        except Exception as e:
            self.logger.error(f"Error calculating trend strength: {e}")
            return 0.5
    
    async def _identify_support_resistance(self, df: pd.DataFrame, i: int, 
                                         period: int = 20) -> Dict[str, float]:
        """Identify nearby support and resistance levels"""
        try:
            if i < period:
                return {'support': None, 'resistance': None}
            
            recent_data = df.iloc[i-period+1:i+1]
            
            # Find local minima and maxima
            highs = recent_data['high'].values
            lows = recent_data['low'].values
            
            # Simple support/resistance identification
            resistance = np.max(highs)
            support = np.min(lows)
            
            return {
                'support': float(support),
                'resistance': float(resistance)
            }
            
        except Exception as e:
            self.logger.error(f"Error identifying support/resistance: {e}")
            return {'support': None, 'resistance': None}
    
    async def _enhance_patterns_with_volume_trend(self, patterns: List[PatternResult], 
                                                 df: pd.DataFrame) -> List[PatternResult]:
        """Enhance patterns with volume and trend analysis"""
        try:
            for pattern in patterns:
            # Volume confirmation
                pattern.volume_confirmation = await self._check_volume_confirmation(pattern, df)
                
                # Trend alignment
                pattern.trend_alignment = await self._determine_trend_alignment(pattern, df)
                
                # Update confidence based on volume and trend
                pattern.confidence = await self._adjust_confidence_with_volume_trend(pattern)
                
                # Update pattern strength
                pattern.strength = await self._determine_pattern_strength(pattern.confidence)
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Error enhancing patterns: {e}")
            return patterns
    
    async def _enhance_patterns_with_market_structure(self, patterns: List[PatternResult], 
                                                     market_structure: MarketStructureAnalysis) -> List[PatternResult]:
        """Enhance patterns with market structure analysis"""
        try:
            if not patterns or not market_structure:
                return patterns
            
            for pattern in patterns:
                # Check market structure alignment
                structure_alignment = await self._check_market_structure_alignment(pattern, market_structure)
                
                # Check for structure breakouts
                breakout_confirmation = await self._check_structure_breakout_confirmation(pattern, market_structure)
                
                # Check trend line validation
                trend_line_validation = await self._check_trend_line_validation(pattern, market_structure)
                
                # Adjust confidence based on market structure
                confidence_adjustment = 0.0
                
                if structure_alignment:
                    confidence_adjustment += 0.15
                
                if breakout_confirmation:
                    confidence_adjustment += 0.2
                
                if trend_line_validation:
                    confidence_adjustment += 0.1
                
                # Add market structure metadata
                pattern.metadata.update({
                    'market_structure_type': market_structure.market_structure_type.value,
                    'structure_strength': market_structure.structure_strength,
                    'structure_breakout': market_structure.structure_breakout,
                    'breakout_direction': market_structure.breakout_direction,
                    'structure_phase': market_structure.current_structure_phase,
                    'structure_confidence': market_structure.analysis_confidence
                })
                
                pattern.confidence = min(1.0, pattern.confidence + confidence_adjustment)
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Error enhancing patterns with market structure: {e}")
            return patterns
     
    async def _enhance_patterns_with_support_resistance(self, patterns: List[PatternResult], 
                                                       sr_analysis: SupportResistanceAnalysis) -> List[PatternResult]:
         """Enhance patterns with support/resistance analysis"""
         try:
             if not patterns or not sr_analysis:
                 return patterns
             
             for pattern in patterns:
                 # Check support/resistance level proximity
                 level_proximity = await self._check_sr_level_proximity(pattern, sr_analysis)
                 
                 # Check volume-weighted level validation
                 volume_level_validation = await self._check_volume_level_validation(pattern, sr_analysis)
                 
                 # Check psychological level alignment
                 psychological_alignment = await self._check_psychological_level_alignment(pattern, sr_analysis)
                 
                 # Check level interaction quality
                 interaction_quality = await self._check_level_interaction_quality(pattern, sr_analysis)
                 
                 # Adjust confidence based on support/resistance factors
                 confidence_adjustment = 0.0
                 
                 if level_proximity:
                     confidence_adjustment += 0.2  # Strong bonus for proximity to levels
                 
                 if volume_level_validation:
                     confidence_adjustment += 0.15  # Volume-weighted level confirmation
                 
                 if psychological_alignment:
                     confidence_adjustment += 0.1  # Psychological level alignment
                 
                 if interaction_quality > 0.7:
                     confidence_adjustment += 0.1  # High-quality level interaction
                 
                 # Add support/resistance metadata
                 pattern.metadata.update({
                     'sr_level_proximity': level_proximity,
                     'volume_level_validation': volume_level_validation,
                     'psychological_alignment': psychological_alignment,
                     'level_interaction_quality': interaction_quality,
                     'nearest_support': await self._get_nearest_support_level(pattern, sr_analysis),
                     'nearest_resistance': await self._get_nearest_resistance_level(pattern, sr_analysis),
                     'dynamic_levels_count': sr_analysis.dynamic_levels_count,
                     'volume_confirmed_levels': sr_analysis.volume_confirmed_levels,
                     'sr_analysis_confidence': sr_analysis.analysis_confidence
                 })
                 
                 pattern.confidence = min(1.0, pattern.confidence + confidence_adjustment)
             
             return patterns
             
         except Exception as e:
             self.logger.error(f"Error enhancing patterns with support/resistance: {e}")
             return patterns
     
    async def _enhance_patterns_with_demand_supply_zones(self, patterns: List[PatternResult], 
                                                        ds_analysis: DemandSupplyAnalysis) -> List[PatternResult]:
        """Enhance patterns with demand/supply zone analysis"""
        try:
            if not patterns or not ds_analysis:
                return patterns
            
            for pattern in patterns:
                # Check demand zone proximity
                demand_zone_proximity = await self._check_demand_zone_proximity(pattern, ds_analysis)
                
                # Check supply zone proximity
                supply_zone_proximity = await self._check_supply_zone_proximity(pattern, ds_analysis)
                
                # Check volume profile alignment
                volume_profile_alignment = await self._check_volume_profile_alignment(pattern, ds_analysis)
                
                # Check zone breakout potential
                zone_breakout_potential = await self._check_zone_breakout_potential(pattern, ds_analysis)
                
                # Adjust confidence based on demand/supply zone factors
                confidence_adjustment = 0.0
                
                if demand_zone_proximity:
                    confidence_adjustment += 0.25  # Strong bonus for demand zone proximity
                
                if supply_zone_proximity:
                    confidence_adjustment += 0.25  # Strong bonus for supply zone proximity
                
                if volume_profile_alignment > 0.7:
                    confidence_adjustment += 0.15  # Volume profile alignment
                
                if zone_breakout_potential > 0.6:
                    confidence_adjustment += 0.1  # Zone breakout potential
                
                # Add demand/supply zone metadata
                pattern.metadata.update({
                    'demand_zone_proximity': demand_zone_proximity,
                    'supply_zone_proximity': supply_zone_proximity,
                    'volume_profile_alignment': volume_profile_alignment,
                    'zone_breakout_potential': zone_breakout_potential,
                    'nearest_demand_zone': await self._get_nearest_demand_zone(pattern, ds_analysis),
                    'nearest_supply_zone': await self._get_nearest_supply_zone(pattern, ds_analysis),
                    'strongest_demand_zone': ds_analysis.strongest_demand_zone.zone_start_price if ds_analysis.strongest_demand_zone else None,
                    'strongest_supply_zone': ds_analysis.strongest_supply_zone.zone_start_price if ds_analysis.strongest_supply_zone else None,
                    'demand_zones_count': len(ds_analysis.demand_zones),
                    'supply_zones_count': len(ds_analysis.supply_zones),
                    'volume_profile_nodes_count': len(ds_analysis.volume_profile_nodes),
                    'ds_analysis_confidence': ds_analysis.analysis_confidence
                })
                
                pattern.confidence = min(1.0, pattern.confidence + confidence_adjustment)
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Error enhancing patterns with demand/supply zones: {e}")
            return patterns
     
    async def _check_volume_confirmation(self, pattern: PatternResult, df: pd.DataFrame) -> bool:
        """Check if volume confirms the pattern"""
        try:
            # Find the pattern's timestamp in the dataframe
            pattern_idx = df[df['timestamp'] == pattern.timestamp].index
            if len(pattern_idx) == 0:
                return False
            
            i = pattern_idx[0]
            volume_ratio = await self._calculate_volume_ratio(df, i)
            
            return volume_ratio > self.volume_threshold
            
        except Exception as e:
            self.logger.error(f"Error checking volume confirmation: {e}")
            return False
    
    async def _determine_trend_alignment(self, pattern: PatternResult, df: pd.DataFrame) -> str:
        """Determine if pattern aligns with the trend"""
        try:
            # Find the pattern's timestamp in the dataframe
            pattern_idx = df[df['timestamp'] == pattern.timestamp].index
            if len(pattern_idx) == 0:
                return 'neutral'
            
            i = pattern_idx[0]
            trend_strength = await self._calculate_trend_strength(df, i)
            
            if trend_strength < 0.3:
                return 'neutral'
            
            # Determine trend direction
            if i < 20:
                return 'neutral'
            
            recent_prices = df.iloc[i-20:i+1]['close'].values
            trend_slope = np.polyfit(range(len(recent_prices)), recent_prices, 1)[0]
            
            if trend_slope > 0:
                return 'bullish'
            else:
                return 'bearish'
            
        except Exception as e:
            self.logger.error(f"Error determining trend alignment: {e}")
            return 'neutral'
    
    async def _adjust_confidence_with_volume_trend(self, pattern: PatternResult) -> float:
        """Adjust confidence based on volume and trend alignment"""
        try:
            adjusted_confidence = pattern.confidence
            
            # Volume confirmation bonus
            if pattern.volume_confirmation:
                adjusted_confidence += 0.1
            
            # Trend alignment bonus
            if pattern.trend_alignment == 'bullish' and pattern.pattern_type in [
                PatternType.BULLISH_ENGULFING, PatternType.HAMMER, PatternType.MORNING_STAR
            ]:
                adjusted_confidence += 0.1
            elif pattern.trend_alignment == 'bearish' and pattern.pattern_type in [
                PatternType.BEARISH_ENGULFING, PatternType.SHOOTING_STAR, PatternType.EVENING_STAR
            ]:
                adjusted_confidence += 0.1
            
            # Cap confidence at 1.0
            return min(adjusted_confidence, 1.0)
            
        except Exception as e:
            self.logger.error(f"Error adjusting confidence: {e}")
            return pattern.confidence
    
    async def save_patterns(self, patterns: List[PatternResult]) -> bool:
        """Save detected patterns to storage"""
        try:
            if not patterns:
                return True
            
            for pattern in patterns:
                # Save to database if available
                if self.db_connection:
                    pattern_data = {
                        'symbol': pattern.symbol,
                        'timeframe': pattern.timeframe,
                        'pattern_name': pattern.pattern_type.value,
                        'timestamp': pattern.timestamp,
                        'confidence': pattern.confidence,
                        'strength': pattern.strength.value,
                        'price_level': pattern.entry_price,
                        'volume_confirmation': pattern.volume_confirmation,
                        'trend_alignment': pattern.trend_alignment,
                        'metadata': pattern.metadata
                    }
                    
                    # Note: This would require a patterns table in the database
                    # For now, just log the pattern
                    self.logger.info(f"Pattern detected: {pattern.pattern_type.value} on {pattern.symbol}")
                
                # Save to file storage as fallback
                if self.storage:
                    key = f"pattern_{pattern.symbol}_{pattern.timeframe}_{pattern.timestamp.strftime('%Y%m%d_%H%M%S')}"
                    await self.storage.save_data(key, {
                        'pattern_type': pattern.pattern_type.value,
                        'strength': pattern.strength.value,
                        'confidence': pattern.confidence,
                        'entry_price': pattern.entry_price,
                        'stop_loss': pattern.stop_loss,
                        'take_profit': pattern.take_profit,
                        'timestamp': pattern.timestamp.isoformat(),
                        'symbol': pattern.symbol,
                        'timeframe': pattern.timeframe,
                        'volume_confirmation': pattern.volume_confirmation,
                        'trend_alignment': pattern.trend_alignment,
                        'metadata': pattern.metadata
                    })
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving patterns: {e}")
            return False
    
    async def get_pattern_statistics(self) -> Dict[str, Any]:
        """Get pattern detection statistics"""
        try:
            stats = self.stats.copy()
            
            # Add storage statistics if available
            if self.storage:
                storage_stats = await self.storage.get_storage_stats()
                stats['storage'] = storage_stats
            
            # Add database health if available
            if self.db_connection:
                db_health = await self.db_connection.health_check()
                stats['database_health'] = db_health
            
            return stats
                    
        except Exception as e:
            self.logger.error(f"Error getting pattern statistics: {e}")
            return {'error': str(e)}
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check for pattern detector"""
        try:
            health_status = {
                'status': 'healthy',
                'patterns_detected': self.stats['patterns_detected'],
                'high_confidence_patterns': self.stats['high_confidence_patterns'],
                'last_update': self.stats['last_update'].isoformat() if self.stats['last_update'] else None
            }
            
            # Check storage health
            if self.storage:
                storage_health = await self.storage.health_check()
                health_status['storage_health'] = storage_health
                
                if storage_health.get('status') != 'healthy':
                    health_status['status'] = 'degraded'
                    health_status['warnings'] = ['Storage issues detected']
            
            # Check database health
            if self.db_connection:
                db_health = await self.db_connection.health_check()
                health_status['database_health'] = db_health
                
                if db_health.get('status') != 'healthy':
                    health_status['status'] = 'degraded'
                    if 'warnings' not in health_status:
                        health_status['warnings'] = []
                    health_status['warnings'].append('Database connection issues')
            
            return health_status
            
        except Exception as e:
            self.logger.error(f"Error in health check: {e}")
            return {'status': 'error', 'error': str(e)}
    
    async def _check_market_structure_alignment(self, pattern: PatternResult, 
                                               market_structure: MarketStructureAnalysis) -> bool:
        """Check if pattern aligns with market structure"""
        try:
            if not market_structure:
                return False
            
            # Check if pattern direction aligns with market structure
            if market_structure.market_structure_type.value == 'uptrend':
                return pattern.pattern_type in [
                    PatternType.BULLISH_ENGULFING, PatternType.HAMMER, 
                    PatternType.MORNING_STAR, PatternType.THREE_WHITE_SOLDIERS
                ]
            elif market_structure.market_structure_type.value == 'downtrend':
                return pattern.pattern_type in [
                    PatternType.BEARISH_ENGULFING, PatternType.SHOOTING_STAR,
                    PatternType.EVENING_STAR, PatternType.THREE_BLACK_CROWS
                ]
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking market structure alignment: {e}")
            return False
    
    async def _check_structure_breakout_confirmation(self, pattern: PatternResult, 
                                                    market_structure: MarketStructureAnalysis) -> bool:
        """Check if pattern confirms a structure breakout"""
        try:
            if not market_structure or not market_structure.structure_breakout:
                return False
            
            # Check if pattern confirms the breakout direction
            if market_structure.breakout_direction == 'up':
                return pattern.pattern_type in [
                    PatternType.BULLISH_ENGULFING, PatternType.HAMMER,
                    PatternType.MORNING_STAR, PatternType.THREE_WHITE_SOLDIERS
                ]
            elif market_structure.breakout_direction == 'down':
                return pattern.pattern_type in [
                    PatternType.BEARISH_ENGULFING, PatternType.SHOOTING_STAR,
                    PatternType.EVENING_STAR, PatternType.THREE_BLACK_CROWS
                ]
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking structure breakout confirmation: {e}")
            return False
    
    async def _check_trend_line_validation(self, pattern: PatternResult, 
                                          market_structure: MarketStructureAnalysis) -> bool:
        """Check if pattern validates trend lines"""
        try:
            if not market_structure or not market_structure.trend_lines:
                return False
            
            # Check if pattern occurs near a trend line
            pattern_price = pattern.entry_price
            
            for trend_line in market_structure.trend_lines:
                # Calculate distance from trend line
                if trend_line.trend_line_type.value == 'support':
                    if abs(pattern_price - trend_line.start_price) / trend_line.start_price < 0.01:
                        return True
                elif trend_line.trend_line_type.value == 'resistance':
                    if abs(pattern_price - trend_line.start_price) / trend_line.start_price < 0.01:
                        return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking trend line validation: {e}")
            return False
     
    async def _check_sr_level_proximity(self, pattern: PatternResult, 
                                        sr_analysis: SupportResistanceAnalysis) -> bool:
         """Check if pattern occurs near support/resistance levels"""
         try:
             pattern_price = pattern.entry_price
             proximity_threshold = 0.01  # 1% proximity threshold
             
             # Check support levels
             for level in sr_analysis.support_levels:
                 if level.is_active and abs(pattern_price - level.price_level) / level.price_level < proximity_threshold:
                     return True
             
             # Check resistance levels
             for level in sr_analysis.resistance_levels:
                 if level.is_active and abs(pattern_price - level.price_level) / level.price_level < proximity_threshold:
                     return True
             
             return False
             
         except Exception as e:
             self.logger.error(f"Error checking SR level proximity: {e}")
             return False
     
    async def _check_volume_level_validation(self, pattern: PatternResult, 
                                            sr_analysis: SupportResistanceAnalysis) -> bool:
         """Check if pattern aligns with volume-weighted levels"""
         try:
             pattern_price = pattern.entry_price
             proximity_threshold = 0.015  # 1.5% proximity threshold for volume levels
             
             for level in sr_analysis.volume_weighted_levels:
                 if level.validation_score > 0.6:  # High validation score
                     if abs(pattern_price - level.price_level) / level.price_level < proximity_threshold:
                         return True
             
             return False
             
         except Exception as e:
             self.logger.error(f"Error checking volume level validation: {e}")
             return False
     
    async def _check_psychological_level_alignment(self, pattern: PatternResult, 
                                                  sr_analysis: SupportResistanceAnalysis) -> bool:
         """Check if pattern aligns with psychological levels"""
         try:
             pattern_price = pattern.entry_price
             proximity_threshold = 0.005  # 0.5% proximity threshold for psychological levels
             
             for level in sr_analysis.psychological_levels:
                 if level.reliability_score > 0.7:  # High reliability
                     if abs(pattern_price - level.price_level) / level.price_level < proximity_threshold:
                         return True
             
             return False
             
         except Exception as e:
             self.logger.error(f"Error checking psychological level alignment: {e}")
             return False
     
    async def _check_level_interaction_quality(self, pattern: PatternResult, 
                                               sr_analysis: SupportResistanceAnalysis) -> float:
         """Check the quality of recent level interactions"""
         try:
             if not sr_analysis.recent_interactions:
                 return 0.0
             
             # Calculate average success rate of recent interactions
             total_quality = 0.0
             relevant_interactions = 0
             
             for interaction in sr_analysis.recent_interactions:
                 if interaction.success_probability > 0:
                     total_quality += interaction.success_probability
                     relevant_interactions += 1
             
             if relevant_interactions > 0:
                 return total_quality / relevant_interactions
             
             return 0.0
             
         except Exception as e:
             self.logger.error(f"Error checking level interaction quality: {e}")
             return 0.0
     
    async def _get_nearest_support_level(self, pattern: PatternResult, 
                                        sr_analysis: SupportResistanceAnalysis) -> Optional[float]:
         """Get the nearest support level to the pattern"""
         try:
             pattern_price = pattern.entry_price
             nearest_level = None
             min_distance = float('inf')
             
             for level in sr_analysis.support_levels:
                 if level.is_active and level.price_level < pattern_price:
                     distance = pattern_price - level.price_level
                     if distance < min_distance:
                         min_distance = distance
                         nearest_level = level.price_level
             
             return nearest_level
             
         except Exception as e:
             self.logger.error(f"Error getting nearest support level: {e}")
             return None
     
    async def _get_nearest_resistance_level(self, pattern: PatternResult, 
                                           sr_analysis: SupportResistanceAnalysis) -> Optional[float]:
         """Get the nearest resistance level to the pattern"""
         try:
             pattern_price = pattern.entry_price
             nearest_level = None
             min_distance = float('inf')
             
             for level in sr_analysis.resistance_levels:
                 if level.is_active and level.price_level > pattern_price:
                     distance = level.price_level - pattern_price
                     if distance < min_distance:
                         min_distance = distance
                         nearest_level = level.price_level
             
             return nearest_level
             
         except Exception as e:
             self.logger.error(f"Error getting nearest resistance level: {e}")
             return None
     
    # Demand/Supply Zone Helper Methods
    async def _check_demand_zone_proximity(self, pattern: PatternResult, 
                                          ds_analysis: DemandSupplyAnalysis) -> bool:
         """Check if pattern is near a demand zone"""
         try:
             if not ds_analysis.demand_zones:
                 return False
             
             pattern_price = pattern.entry_price
             proximity_threshold = 0.02  # 2% proximity
             
             for zone in ds_analysis.demand_zones:
                 if zone.zone_strength > 0.5:  # Strong zone
                     zone_price_range = (zone.zone_start_price, zone.zone_end_price)
                     if (zone_price_range[0] <= pattern_price <= zone_price_range[1] or
                         abs(pattern_price - zone_price_range[1]) / zone_price_range[1] < proximity_threshold):
                         return True
             
             return False
             
         except Exception as e:
             self.logger.error(f"Error checking demand zone proximity: {e}")
             return False
     
    async def _check_supply_zone_proximity(self, pattern: PatternResult, 
                                          ds_analysis: DemandSupplyAnalysis) -> bool:
         """Check if pattern is near a supply zone"""
         try:
             if not ds_analysis.supply_zones:
                 return False
             
             pattern_price = pattern.entry_price
             proximity_threshold = 0.02  # 2% proximity
             
             for zone in ds_analysis.supply_zones:
                 if zone.zone_strength > 0.5:  # Strong zone
                     zone_price_range = (zone.zone_start_price, zone.zone_end_price)
                     if (zone_price_range[0] <= pattern_price <= zone_price_range[1] or
                         abs(pattern_price - zone_price_range[0]) / zone_price_range[0] < proximity_threshold):
                         return True
             
             return False
             
         except Exception as e:
             self.logger.error(f"Error checking supply zone proximity: {e}")
             return False
     
    async def _check_volume_profile_alignment(self, pattern: PatternResult, 
                                             ds_analysis: DemandSupplyAnalysis) -> float:
         """Check alignment with volume profile"""
         try:
             if not ds_analysis.volume_profile_nodes:
                 return 0.0
             
             pattern_price = pattern.entry_price
             alignment_score = 0.0
             
             for node in ds_analysis.volume_profile_nodes:
                 if node.volume_node_type.value == 'high':  # High volume node
                     price_distance = abs(pattern_price - node.price_level) / node.price_level
                     if price_distance < 0.01:  # Very close to high volume node
                         alignment_score += 0.8
                     elif price_distance < 0.02:  # Close to high volume node
                         alignment_score += 0.5
                     elif price_distance < 0.05:  # Near high volume node
                         alignment_score += 0.2
             
             return min(1.0, alignment_score)
             
         except Exception as e:
             self.logger.error(f"Error checking volume profile alignment: {e}")
             return 0.0
     
    async def _check_zone_breakout_potential(self, pattern: PatternResult, 
                                            ds_analysis: DemandSupplyAnalysis) -> float:
         """Check potential for zone breakout"""
         try:
             if not ds_analysis.zone_breakouts:
                 return 0.0
             
             # Calculate average breakout strength
             total_strength = 0.0
             relevant_breakouts = 0
             
             for breakout in ds_analysis.zone_breakouts:
                 if breakout.breakout_strength > 0:
                     total_strength += breakout.breakout_strength
                     relevant_breakouts += 1
             
             if relevant_breakouts > 0:
                 return total_strength / relevant_breakouts
             
             return 0.0
             
         except Exception as e:
             self.logger.error(f"Error checking zone breakout potential: {e}")
             return 0.0
     
    async def _get_nearest_demand_zone(self, pattern: PatternResult, 
                                      ds_analysis: DemandSupplyAnalysis) -> Optional[float]:
         """Get the nearest demand zone to the pattern"""
         try:
             if not ds_analysis.demand_zones:
                 return None
             
             pattern_price = pattern.entry_price
             nearest_zone = None
             min_distance = float('inf')
             
             for zone in ds_analysis.demand_zones:
                 if zone.zone_strength > 0.5:  # Strong zone
                     zone_center = (zone.zone_start_price + zone.zone_end_price) / 2
                     distance = abs(pattern_price - zone_center)
                     if distance < min_distance:
                         min_distance = distance
                         nearest_zone = zone_center
             
             return nearest_zone
             
         except Exception as e:
             self.logger.error(f"Error getting nearest demand zone: {e}")
             return None
     
    async def _get_nearest_supply_zone(self, pattern: PatternResult, 
                                      ds_analysis: DemandSupplyAnalysis) -> Optional[float]:
         """Get the nearest supply zone to the pattern"""
         try:
             if not ds_analysis.supply_zones:
                 return None
             
             pattern_price = pattern.entry_price
             nearest_zone = None
             min_distance = float('inf')
             
             for zone in ds_analysis.supply_zones:
                 if zone.zone_strength > 0.5:  # Strong zone
                     zone_center = (zone.zone_start_price + zone.zone_end_price) / 2
                     distance = abs(pattern_price - zone_center)
                     if distance < min_distance:
                         min_distance = distance
                         nearest_zone = zone_center
             
             return nearest_zone
             
         except Exception as e:
             self.logger.error(f"Error getting nearest supply zone: {e}")
             return None
     
    async def close(self):
        """Close pattern detector and cleanup"""
        try:
            if self.storage:
                await self.storage.close()
            
            if self.db_connection:
                await self.db_connection.close()
            
            self.logger.info("Pattern detector closed")
            
        except Exception as e:
            self.logger.error(f"Error closing pattern detector: {e}")
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()
    
    async def _calculate_pattern_strength(self, df: pd.DataFrame, i: int, pattern_type: PatternType) -> float:
        """Calculate pattern strength using standard formula: (Body Size × Volume) / Range"""
        try:
            open_price = df.iloc[i]['open']
            high = df.iloc[i]['high']
            low = df.iloc[i]['low']
            close = df.iloc[i]['close']
            volume = df.iloc[i]['volume']
            
            # Calculate body size and total range
            body_size = abs(close - open_price)
            total_range = high - low
            
            if total_range == 0:
                return 0.0
            
            # Standard Pattern Strength formula: (Body Size × Volume) / Range
            pattern_strength = (body_size * volume) / total_range
            
            # Normalize to 0-1 range
            return min(pattern_strength / 1000000, 1.0)  # Adjust divisor based on typical values
            
        except Exception as e:
            self.logger.error(f"Error calculating pattern strength: {e}")
            return 0.0
