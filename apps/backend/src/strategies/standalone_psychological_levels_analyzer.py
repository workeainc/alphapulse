#!/usr/bin/env python3
"""
Standalone Psychological Levels Analyzer for AlphaPlus
Detects and analyzes psychological price levels for trading signals
"""

import asyncio
import logging
import asyncpg
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json
import time
import math

logger = logging.getLogger(__name__)

class PsychologicalLevelType(Enum):
    """Types of psychological levels"""
    ROUND_NUMBER = "round_number"
    FIBONACCI_RETRACEMENT = "fibonacci_retracement"
    FIBONACCI_EXTENSION = "fibonacci_extension"
    GOLDEN_RATIO = "golden_ratio"
    MAJOR_SUPPORT_RESISTANCE = "major_support_resistance"
    PRICE_MILESTONE = "price_milestone"
    PERCENTAGE_LEVEL = "percentage_level"

@dataclass
class PsychologicalLevel:
    """Psychological price level"""
    level_type: PsychologicalLevelType
    price_level: float
    strength: float
    confidence: float
    touch_count: int = 0
    first_touch_time: Optional[datetime] = None
    last_touch_time: Optional[datetime] = None
    touch_points: List[Dict] = field(default_factory=list)
    is_active: bool = True
    is_broken: bool = False
    break_time: Optional[datetime] = None
    rejection_count: int = 0
    penetration_count: int = 0
    volume_at_level: float = 0.0
    market_context: Optional[str] = None

@dataclass
class PsychologicalAnalysis:
    """Comprehensive psychological levels analysis"""
    symbol: str
    timeframe: str
    timestamp: datetime
    current_price: float
    psychological_levels: List[PsychologicalLevel]
    nearest_support: Optional[PsychologicalLevel]
    nearest_resistance: Optional[PsychologicalLevel]
    level_interactions: List[Dict]
    market_regime: str
    analysis_confidence: float
    algorithm_inputs: Dict[str, Any]
    nearest_support_price: Optional[float] = None
    nearest_resistance_price: Optional[float] = None

class StandalonePsychologicalLevelsAnalyzer:
    """Standalone psychological levels analyzer"""
    
    def __init__(self, db_url: str = None):
        self.db_url = db_url or "postgresql://alpha_emon:Emon_%4017711@localhost:5432/alphapulse"
        self.logger = logger
        self.db_pool = None
        
        # Configuration
        self.config = {
            'lookback_periods': 200,
            'min_touch_count': 2,
            'level_tolerance': 0.002,  # 0.2%
            'strength_threshold': 0.3,
            'confidence_threshold': 0.5,
            'round_number_levels': [10, 50, 100, 500, 1000, 5000, 10000, 50000, 100000],
            'fibonacci_ratios': [0.236, 0.382, 0.5, 0.618, 0.786, 1.0, 1.272, 1.618, 2.0, 2.618],
            'golden_ratio': 1.618,
            'price_milestones': [1000, 5000, 10000, 25000, 50000, 100000, 250000, 500000, 1000000]
        }
        
        # Statistics
        self.stats = {
            'total_analyses': 0,
            'successful_analyses': 0,
            'failed_analyses': 0,
            'levels_detected': 0,
            'round_numbers_detected': 0,
            'fibonacci_levels_detected': 0,
            'milestones_detected': 0,
            'last_analysis': None
        }
        
        logger.info("🔧 Standalone Psychological Levels Analyzer initialized")
    
    async def initialize(self):
        """Initialize database connection pool"""
        if not self.db_pool:
            self.db_pool = await asyncpg.create_pool(self.db_url)
            self.logger.info("✅ Database connection pool created for Psychological Levels Analyzer")
    
    async def close(self):
        """Close database connection pool"""
        if self.db_pool:
            await self.db_pool.close()
            self.logger.info("🔌 Database connection pool closed for Psychological Levels Analyzer")
    
    async def analyze_psychological_levels(self, symbol: str, timeframe: str) -> PsychologicalAnalysis:
        """Analyze psychological levels for a symbol"""
        try:
            self.logger.info(f"🧠 Analyzing psychological levels for {symbol} {timeframe}")
            
            # Get recent OHLCV data
            ohlcv_data = await self._get_recent_ohlcv_data(symbol, timeframe)
            if not ohlcv_data or len(ohlcv_data) < 50:
                self.logger.warning(f"Insufficient data for psychological analysis: {symbol}")
                return self._get_default_analysis(symbol, timeframe)
            
            # Convert to DataFrame
            df = pd.DataFrame(ohlcv_data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            # Convert decimal columns to float to avoid type conflicts
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = df[col].astype(float)
            
            current_price = float(df['close'].iloc[-1])
            
            # Detect different types of psychological levels
            round_number_levels = await self._detect_round_number_levels(df, symbol)
            fibonacci_levels = await self._detect_fibonacci_levels(df, symbol)
            milestone_levels = await self._detect_milestone_levels(df, symbol)
            percentage_levels = await self._detect_percentage_levels(df, symbol)
            
            # Combine all levels
            all_levels = round_number_levels + fibonacci_levels + milestone_levels + percentage_levels
            
            # Validate and score levels
            validated_levels = await self._validate_and_score_levels(df, all_levels)
            
            # Find nearest support and resistance
            nearest_support, nearest_resistance = await self._find_nearest_levels(
                current_price, validated_levels
            )
            
            # Analyze level interactions
            level_interactions = await self._analyze_level_interactions(df, validated_levels)
            
            # Determine market regime
            market_regime = await self._determine_market_regime(df, validated_levels)
            
            # Calculate analysis confidence
            analysis_confidence = await self._calculate_analysis_confidence(df, validated_levels)
            
            # Prepare algorithm inputs
            algorithm_inputs = await self._prepare_algorithm_inputs(
                validated_levels, nearest_support, nearest_resistance, market_regime
            )
            
            # Create analysis
            analysis = PsychologicalAnalysis(
                symbol=symbol,
                timeframe=timeframe,
                timestamp=datetime.now(timezone.utc),
                current_price=current_price,
                psychological_levels=validated_levels,
                nearest_support=nearest_support,
                nearest_resistance=nearest_resistance,
                nearest_support_price=nearest_support.price_level if nearest_support else None,
                nearest_resistance_price=nearest_resistance.price_level if nearest_resistance else None,
                level_interactions=level_interactions,
                market_regime=market_regime,
                analysis_confidence=analysis_confidence,
                algorithm_inputs=algorithm_inputs
            )
            
            # Store analysis in database
            await self._store_psychological_analysis(analysis)
            
            # Update statistics
            self.stats['total_analyses'] += 1
            self.stats['successful_analyses'] += 1
            self.stats['levels_detected'] += len(validated_levels)
            self.stats['round_numbers_detected'] += len(round_number_levels)
            self.stats['fibonacci_levels_detected'] += len(fibonacci_levels)
            self.stats['milestones_detected'] += len(milestone_levels)
            self.stats['last_analysis'] = datetime.now()
            
            self.logger.info(f"✅ Psychological analysis completed for {symbol} {timeframe}")
            self.logger.info(f"📊 Detected {len(validated_levels)} psychological levels")
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"❌ Error analyzing psychological levels for {symbol} {timeframe}: {e}")
            self.stats['total_analyses'] += 1
            self.stats['failed_analyses'] += 1
            return self._get_default_analysis(symbol, timeframe)
    
    async def _get_recent_ohlcv_data(self, symbol: str, timeframe: str) -> List[Dict]:
        """Get recent OHLCV data"""
        try:
            async with self.db_pool.acquire() as conn:
                query = """
                    SELECT timestamp, open, high, low, close, volume
                    FROM ohlcv_data
                    WHERE symbol = $1 AND timeframe = $2
                    AND timestamp >= NOW() - INTERVAL '30 days'
                    ORDER BY timestamp DESC
                    LIMIT $3
                """
                
                rows = await conn.fetch(query, symbol, timeframe, self.config['lookback_periods'])
                
                return [dict(row) for row in rows]
                
        except Exception as e:
            self.logger.error(f"❌ Error getting OHLCV data: {e}")
            return []
    
    async def _detect_round_number_levels(self, df: pd.DataFrame, symbol: str) -> List[PsychologicalLevel]:
        """Detect round number psychological levels"""
        try:
            levels = []
            price_range = df['high'].max() - df['low'].min()
            current_price = df['close'].iloc[-1]
            
            # Determine appropriate round number levels based on price range
            if current_price < 100:
                round_levels = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
            elif current_price < 1000:
                round_levels = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
            elif current_price < 10000:
                round_levels = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
            else:
                round_levels = [10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000]
            
            for level_price in round_levels:
                # Check if level is within reasonable range
                if df['low'].min() <= level_price <= df['high'].max():
                    touch_count = await self._count_level_touches(df, level_price)
                    
                    if touch_count >= self.config['min_touch_count']:
                        strength = await self._calculate_level_strength(df, level_price, touch_count)
                        confidence = await self._calculate_level_confidence(df, level_price, touch_count)
                        
                        level = PsychologicalLevel(
                            level_type=PsychologicalLevelType.ROUND_NUMBER,
                            price_level=level_price,
                            strength=strength,
                            confidence=confidence,
                            touch_count=touch_count,
                            first_touch_time=await self._get_first_touch_time(df, level_price),
                            last_touch_time=await self._get_last_touch_time(df, level_price),
                            market_context=self._get_market_context(df, level_price)
                        )
                        levels.append(level)
            
            return levels
            
        except Exception as e:
            self.logger.error(f"❌ Error detecting round number levels: {e}")
            return []
    
    async def _detect_fibonacci_levels(self, df: pd.DataFrame, symbol: str) -> List[PsychologicalLevel]:
        """Detect Fibonacci retracement and extension levels"""
        try:
            levels = []
            
            # Find significant high and low points
            high_price = df['high'].max()
            low_price = df['low'].min()
            price_range = high_price - low_price
            
            # Calculate Fibonacci retracement levels
            for ratio in self.config['fibonacci_ratios']:
                if ratio <= 1.0:  # Retracement levels
                    fib_price = high_price - (price_range * ratio)
                    
                    if df['low'].min() <= fib_price <= df['high'].max():
                        touch_count = await self._count_level_touches(df, fib_price)
                        
                        if touch_count >= self.config['min_touch_count']:
                            strength = await self._calculate_level_strength(df, fib_price, touch_count)
                            confidence = await self._calculate_level_confidence(df, fib_price, touch_count)
                            
                            level = PsychologicalLevel(
                                level_type=PsychologicalLevelType.FIBONACCI_RETRACEMENT,
                                price_level=fib_price,
                                strength=strength,
                                confidence=confidence,
                                touch_count=touch_count,
                                first_touch_time=await self._get_first_touch_time(df, fib_price),
                                last_touch_time=await self._get_last_touch_time(df, fib_price),
                                market_context=self._get_market_context(df, fib_price)
                            )
                            levels.append(level)
                
                else:  # Extension levels
                    fib_price = high_price + (price_range * (ratio - 1.0))
                    
                    if df['low'].min() <= fib_price <= df['high'].max():
                        touch_count = await self._count_level_touches(df, fib_price)
                        
                        if touch_count >= self.config['min_touch_count']:
                            strength = await self._calculate_level_strength(df, fib_price, touch_count)
                            confidence = await self._calculate_level_confidence(df, fib_price, touch_count)
                            
                            level = PsychologicalLevel(
                                level_type=PsychologicalLevelType.FIBONACCI_EXTENSION,
                                price_level=fib_price,
                                strength=strength,
                                confidence=confidence,
                                touch_count=touch_count,
                                first_touch_time=await self._get_first_touch_time(df, fib_price),
                                last_touch_time=await self._get_last_touch_time(df, fib_price),
                                market_context=self._get_market_context(df, fib_price)
                            )
                            levels.append(level)
            
            return levels
            
        except Exception as e:
            self.logger.error(f"❌ Error detecting Fibonacci levels: {e}")
            return []
    
    async def _detect_milestone_levels(self, df: pd.DataFrame, symbol: str) -> List[PsychologicalLevel]:
        """Detect price milestone levels"""
        try:
            levels = []
            current_price = df['close'].iloc[-1]
            
            # Determine appropriate milestones based on current price
            if current_price < 1000:
                milestones = [100, 500, 1000]
            elif current_price < 10000:
                milestones = [1000, 5000, 10000]
            elif current_price < 100000:
                milestones = [10000, 50000, 100000]
            else:
                milestones = [100000, 250000, 500000, 1000000]
            
            for milestone in milestones:
                if df['low'].min() <= milestone <= df['high'].max():
                    touch_count = await self._count_level_touches(df, milestone)
                    
                    if touch_count >= self.config['min_touch_count']:
                        strength = await self._calculate_level_strength(df, milestone, touch_count)
                        confidence = await self._calculate_level_confidence(df, milestone, touch_count)
                        
                        level = PsychologicalLevel(
                            level_type=PsychologicalLevelType.PRICE_MILESTONE,
                            price_level=milestone,
                            strength=strength,
                            confidence=confidence,
                            touch_count=touch_count,
                            first_touch_time=await self._get_first_touch_time(df, milestone),
                            last_touch_time=await self._get_last_touch_time(df, milestone),
                            market_context=self._get_market_context(df, milestone)
                        )
                        levels.append(level)
            
            return levels
            
        except Exception as e:
            self.logger.error(f"❌ Error detecting milestone levels: {e}")
            return []
    
    async def _detect_percentage_levels(self, df: pd.DataFrame, symbol: str) -> List[PsychologicalLevel]:
        """Detect percentage-based psychological levels"""
        try:
            levels = []
            current_price = df['close'].iloc[-1]
            
            # Calculate percentage levels from current price
            percentage_levels = [0.5, 0.75, 0.9, 1.1, 1.25, 1.5, 2.0]  # 50%, 75%, 90%, 110%, 125%, 150%, 200%
            
            for percentage in percentage_levels:
                level_price = current_price * percentage
                
                if df['low'].min() <= level_price <= df['high'].max():
                    touch_count = await self._count_level_touches(df, level_price)
                    
                    if touch_count >= self.config['min_touch_count']:
                        strength = await self._calculate_level_strength(df, level_price, touch_count)
                        confidence = await self._calculate_level_confidence(df, level_price, touch_count)
                        
                        level = PsychologicalLevel(
                            level_type=PsychologicalLevelType.PERCENTAGE_LEVEL,
                            price_level=level_price,
                            strength=strength,
                            confidence=confidence,
                            touch_count=touch_count,
                            first_touch_time=await self._get_first_touch_time(df, level_price),
                            last_touch_time=await self._get_last_touch_time(df, level_price),
                            market_context=self._get_market_context(df, level_price)
                        )
                        levels.append(level)
            
            return levels
            
        except Exception as e:
            self.logger.error(f"❌ Error detecting percentage levels: {e}")
            return []
    
    async def _count_level_touches(self, df: pd.DataFrame, level_price: float) -> int:
        """Count how many times price touched a level"""
        try:
            tolerance = level_price * self.config['level_tolerance']
            touch_count = 0
            
            for _, row in df.iterrows():
                # Check if high or low touched the level
                if (abs(row['high'] - level_price) <= tolerance or 
                    abs(row['low'] - level_price) <= tolerance):
                    touch_count += 1
            
            return touch_count
            
        except Exception as e:
            self.logger.error(f"❌ Error counting level touches: {e}")
            return 0
    
    async def _calculate_level_strength(self, df: pd.DataFrame, level_price: float, touch_count: int) -> float:
        """Calculate strength of a psychological level"""
        try:
            # Base strength from touch count
            base_strength = min(touch_count / 10.0, 1.0)
            
            # Volume confirmation
            volume_strength = await self._calculate_volume_strength(df, level_price)
            
            # Time persistence
            time_strength = await self._calculate_time_strength(df, level_price)
            
            # Combine factors
            total_strength = (base_strength * 0.4 + volume_strength * 0.3 + time_strength * 0.3)
            
            return min(total_strength, 1.0)
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating level strength: {e}")
            return 0.0
    
    async def _calculate_level_confidence(self, df: pd.DataFrame, level_price: float, touch_count: int) -> float:
        """Calculate confidence in a psychological level"""
        try:
            # Base confidence from touch count and data quality
            base_confidence = min(touch_count / 5.0, 1.0)
            
            # Data quality factor
            data_quality = min(len(df) / self.config['lookback_periods'], 1.0)
            
            # Level consistency
            consistency = await self._calculate_level_consistency(df, level_price)
            
            # Combine factors
            total_confidence = (base_confidence * 0.5 + data_quality * 0.3 + consistency * 0.2)
            
            return min(total_confidence, 1.0)
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating level confidence: {e}")
            return 0.0
    
    async def _calculate_volume_strength(self, df: pd.DataFrame, level_price: float) -> float:
        """Calculate volume strength at a level"""
        try:
            tolerance = level_price * self.config['level_tolerance']
            volumes_at_level = []
            
            for _, row in df.iterrows():
                if (abs(row['high'] - level_price) <= tolerance or 
                    abs(row['low'] - level_price) <= tolerance):
                    volumes_at_level.append(row['volume'])
            
            if not volumes_at_level:
                return 0.0
            
            avg_volume = df['volume'].mean()
            level_avg_volume = np.mean(volumes_at_level)
            
            return min(level_avg_volume / avg_volume, 2.0) / 2.0
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating volume strength: {e}")
            return 0.0
    
    async def _calculate_time_strength(self, df: pd.DataFrame, level_price: float) -> float:
        """Calculate time persistence strength"""
        try:
            tolerance = level_price * self.config['level_tolerance']
            touch_times = []
            
            for _, row in df.iterrows():
                if (abs(row['high'] - level_price) <= tolerance or 
                    abs(row['low'] - level_price) <= tolerance):
                    touch_times.append(row['timestamp'])
            
            if len(touch_times) < 2:
                return 0.0
            
            # Calculate time span
            time_span = (max(touch_times) - min(touch_times)).total_seconds() / 3600  # hours
            
            # Normalize by total analysis period
            total_period = (df['timestamp'].max() - df['timestamp'].min()).total_seconds() / 3600
            
            return min(time_span / total_period, 1.0)
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating time strength: {e}")
            return 0.0
    
    async def _calculate_level_consistency(self, df: pd.DataFrame, level_price: float) -> float:
        """Calculate level consistency"""
        try:
            tolerance = level_price * self.config['level_tolerance']
            touches = []
            
            for _, row in df.iterrows():
                if abs(row['high'] - level_price) <= tolerance:
                    touches.append(row['high'])
                elif abs(row['low'] - level_price) <= tolerance:
                    touches.append(row['low'])
            
            if len(touches) < 2:
                return 0.0
            
            # Calculate standard deviation of touches
            std_dev = np.std(touches)
            
            # Lower std dev = higher consistency
            consistency = max(0, 1 - (std_dev / level_price))
            
            return consistency
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating level consistency: {e}")
            return 0.0
    
    async def _get_first_touch_time(self, df: pd.DataFrame, level_price: float) -> Optional[datetime]:
        """Get first touch time for a level"""
        try:
            tolerance = level_price * self.config['level_tolerance']
            
            for _, row in df.iterrows():
                if (abs(row['high'] - level_price) <= tolerance or 
                    abs(row['low'] - level_price) <= tolerance):
                    return row['timestamp']
            
            return None
            
        except Exception as e:
            self.logger.error(f"❌ Error getting first touch time: {e}")
            return None
    
    async def _get_last_touch_time(self, df: pd.DataFrame, level_price: float) -> Optional[datetime]:
        """Get last touch time for a level"""
        try:
            tolerance = level_price * self.config['level_tolerance']
            last_touch = None
            
            for _, row in df.iterrows():
                if (abs(row['high'] - level_price) <= tolerance or 
                    abs(row['low'] - level_price) <= tolerance):
                    last_touch = row['timestamp']
            
            return last_touch
            
        except Exception as e:
            self.logger.error(f"❌ Error getting last touch time: {e}")
            return None
    
    def _get_market_context(self, df: pd.DataFrame, level_price: float) -> str:
        """Get market context for a level"""
        try:
            current_price = df['close'].iloc[-1]
            
            if level_price > current_price:
                return "resistance"
            elif level_price < current_price:
                return "support"
            else:
                return "current"
                
        except Exception as e:
            self.logger.error(f"❌ Error getting market context: {e}")
            return "unknown"
    
    async def _validate_and_score_levels(self, df: pd.DataFrame, levels: List[PsychologicalLevel]) -> List[PsychologicalLevel]:
        """Validate and score psychological levels"""
        try:
            validated_levels = []
            
            for level in levels:
                # Filter by minimum strength and confidence
                if (level.strength >= self.config['strength_threshold'] and 
                    level.confidence >= self.config['confidence_threshold']):
                    validated_levels.append(level)
            
            # Sort by strength and confidence
            validated_levels.sort(key=lambda x: (x.strength, x.confidence), reverse=True)
            
            return validated_levels
            
        except Exception as e:
            self.logger.error(f"❌ Error validating levels: {e}")
            return []
    
    async def _find_nearest_levels(self, current_price: float, levels: List[PsychologicalLevel]) -> Tuple[Optional[PsychologicalLevel], Optional[PsychologicalLevel]]:
        """Find nearest support and resistance levels"""
        try:
            support_levels = [level for level in levels if level.price_level < current_price]
            resistance_levels = [level for level in levels if level.price_level > current_price]
            
            nearest_support = max(support_levels, key=lambda x: x.price_level) if support_levels else None
            nearest_resistance = min(resistance_levels, key=lambda x: x.price_level) if resistance_levels else None
            
            return nearest_support, nearest_resistance
            
        except Exception as e:
            self.logger.error(f"❌ Error finding nearest levels: {e}")
            return None, None
    
    async def _analyze_level_interactions(self, df: pd.DataFrame, levels: List[PsychologicalLevel]) -> List[Dict]:
        """Analyze interactions between psychological levels"""
        try:
            interactions = []
            
            # Analyze recent level interactions
            recent_df = df.tail(50)  # Last 50 candles
            
            for level in levels:
                tolerance = level.price_level * self.config['level_tolerance']
                
                for _, row in recent_df.iterrows():
                    if (abs(row['high'] - level.price_level) <= tolerance or 
                        abs(row['low'] - level.price_level) <= tolerance):
                        
                        interaction = {
                            'level_price': level.price_level,
                            'level_type': level.level_type.value,
                            'timestamp': row['timestamp'],
                            'price': row['close'],
                            'volume': row['volume'],
                            'reaction_type': 'rejection' if row['close'] != row['high'] and row['close'] != row['low'] else 'penetration'
                        }
                        interactions.append(interaction)
            
            return interactions
            
        except Exception as e:
            self.logger.error(f"❌ Error analyzing level interactions: {e}")
            return []
    
    async def _determine_market_regime(self, df: pd.DataFrame, levels: List[PsychologicalLevel]) -> str:
        """Determine market regime based on psychological levels"""
        try:
            # Simple regime detection based on price action and levels
            recent_df = df.tail(20)
            
            # Calculate price trend
            price_change = (recent_df['close'].iloc[-1] - recent_df['close'].iloc[0]) / recent_df['close'].iloc[0]
            
            # Count active levels
            current_price = df['close'].iloc[-1]
            active_levels = [level for level in levels if abs(level.price_level - current_price) / current_price < 0.05]
            
            if abs(price_change) < 0.02:  # Less than 2% change
                return "ranging"
            elif price_change > 0.02 and len(active_levels) > 2:
                return "trending_up"
            elif price_change < -0.02 and len(active_levels) > 2:
                return "trending_down"
            else:
                return "volatile"
                
        except Exception as e:
            self.logger.error(f"❌ Error determining market regime: {e}")
            return "unknown"
    
    async def _calculate_analysis_confidence(self, df: pd.DataFrame, levels: List[PsychologicalLevel]) -> float:
        """Calculate overall analysis confidence"""
        try:
            if not levels:
                return 0.0
            
            # Average confidence of all levels
            avg_level_confidence = np.mean([level.confidence for level in levels])
            
            # Data quality factor
            data_quality = min(len(df) / self.config['lookback_periods'], 1.0)
            
            # Level diversity factor
            level_types = set(level.level_type for level in levels)
            diversity_factor = len(level_types) / 7.0  # 7 different level types
            
            # Combine factors
            total_confidence = (avg_level_confidence * 0.5 + data_quality * 0.3 + diversity_factor * 0.2)
            
            return min(total_confidence, 1.0)
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating analysis confidence: {e}")
            return 0.0
    
    async def _prepare_algorithm_inputs(self, levels: List[PsychologicalLevel], 
                                       nearest_support: Optional[PsychologicalLevel],
                                       nearest_resistance: Optional[PsychologicalLevel],
                                       market_regime: str) -> Dict[str, Any]:
        """Prepare inputs for other algorithms"""
        try:
            # Group levels by type
            level_groups = {}
            for level in levels:
                level_type = level.level_type.value
                if level_type not in level_groups:
                    level_groups[level_type] = []
                
                level_groups[level_type].append({
                    'price': level.price_level,
                    'strength': level.strength,
                    'confidence': level.confidence,
                    'touch_count': level.touch_count,
                    'market_context': level.market_context
                })
            
            return {
                'psychological_levels': level_groups,
                'nearest_support': {
                    'price': nearest_support.price_level,
                    'strength': nearest_support.strength,
                    'confidence': nearest_support.confidence
                } if nearest_support else None,
                'nearest_resistance': {
                    'price': nearest_resistance.price_level,
                    'strength': nearest_resistance.strength,
                    'confidence': nearest_resistance.confidence
                } if nearest_resistance else None,
                'market_regime': market_regime,
                'total_levels': len(levels),
                'active_levels': len([level for level in levels if level.is_active])
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error preparing algorithm inputs: {e}")
            return {}
    
    async def _store_psychological_analysis(self, analysis: PsychologicalAnalysis):
        """Store psychological analysis in database"""
        try:
            async with self.db_pool.acquire() as conn:
                # Store main analysis
                query = """
                    INSERT INTO psychological_levels_analysis (
                        symbol, timeframe, timestamp, current_price,
                        nearest_support_price, nearest_resistance_price,
                        market_regime, analysis_confidence, algorithm_inputs
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                    ON CONFLICT (symbol, timeframe, timestamp) DO UPDATE SET
                        current_price = EXCLUDED.current_price,
                        nearest_support_price = EXCLUDED.nearest_support_price,
                        nearest_resistance_price = EXCLUDED.nearest_resistance_price,
                        market_regime = EXCLUDED.market_regime,
                        analysis_confidence = EXCLUDED.analysis_confidence,
                        algorithm_inputs = EXCLUDED.algorithm_inputs
                """
                
                await conn.execute(
                    query,
                    analysis.symbol,
                    analysis.timeframe,
                    analysis.timestamp,
                    analysis.current_price,
                    analysis.nearest_support.price_level if analysis.nearest_support else None,
                    analysis.nearest_resistance.price_level if analysis.nearest_resistance else None,
                    analysis.market_regime,
                    analysis.analysis_confidence,
                    json.dumps(analysis.algorithm_inputs)
                )
                
                # Store individual levels
                for level in analysis.psychological_levels:
                    level_query = """
                        INSERT INTO psychological_levels (
                            symbol, timestamp, level_type, price_level,
                            strength, confidence, touch_count, market_context,
                            first_touch_time, last_touch_time, is_active
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                        ON CONFLICT (symbol, level_type, price_level, timestamp) DO UPDATE SET
                            strength = EXCLUDED.strength,
                            confidence = EXCLUDED.confidence,
                            touch_count = EXCLUDED.touch_count,
                            market_context = EXCLUDED.market_context,
                            last_touch_time = EXCLUDED.last_touch_time,
                            is_active = EXCLUDED.is_active
                    """
                    
                    await conn.execute(
                        level_query,
                        analysis.symbol,
                        analysis.timestamp,
                        level.level_type.value,
                        level.price_level,
                        level.strength,
                        level.confidence,
                        level.touch_count,
                        level.market_context,
                        level.first_touch_time,
                        level.last_touch_time,
                        level.is_active
                    )
                
        except Exception as e:
            self.logger.error(f"❌ Error storing psychological analysis: {e}")
    
    def _get_default_analysis(self, symbol: str, timeframe: str) -> PsychologicalAnalysis:
        """Get default analysis when data is insufficient"""
        return PsychologicalAnalysis(
            symbol=symbol,
            timeframe=timeframe,
            timestamp=datetime.now(timezone.utc),
            current_price=0.0,
            psychological_levels=[],
            nearest_support=None,
            nearest_resistance=None,
            level_interactions=[],
            market_regime="unknown",
            analysis_confidence=0.0,
            algorithm_inputs={}
        )

# Example usage
async def main():
    """Test the standalone psychological levels analyzer"""
    analyzer = StandalonePsychologicalLevelsAnalyzer()
    
    try:
        await analyzer.initialize()
        
        # Analyze psychological levels
        analysis = await analyzer.analyze_psychological_levels('BTCUSDT', '1h')
        
        print(f"\n🧠 Psychological Analysis for {analysis.symbol}:")
        print(f"  Current Price: {analysis.current_price}")
        print(f"  Market Regime: {analysis.market_regime}")
        print(f"  Analysis Confidence: {analysis.analysis_confidence:.3f}")
        print(f"  Total Levels: {len(analysis.psychological_levels)}")
        
        if analysis.nearest_support:
            print(f"  Nearest Support: {analysis.nearest_support.price_level} (strength: {analysis.nearest_support.strength:.3f})")
        
        if analysis.nearest_resistance:
            print(f"  Nearest Resistance: {analysis.nearest_resistance.price_level} (strength: {analysis.nearest_resistance.strength:.3f})")
        
        print(f"\n📊 Level Types Detected:")
        level_types = {}
        for level in analysis.psychological_levels:
            level_type = level.level_type.value
            if level_type not in level_types:
                level_types[level_type] = 0
            level_types[level_type] += 1
        
        for level_type, count in level_types.items():
            print(f"  {level_type}: {count} levels")
        
    finally:
        await analyzer.close()

if __name__ == "__main__":
    asyncio.run(main())
