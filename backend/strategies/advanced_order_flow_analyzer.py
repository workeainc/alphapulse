"""
Advanced Order Flow Analyzer for AlphaPlus
Comprehensive order flow analysis including toxicity, market maker vs taker, large orders, and patterns
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass
from enum import Enum
import json

logger = logging.getLogger(__name__)

class ToxicityTrend(Enum):
    """Order flow toxicity trends"""
    INCREASING = 'increasing'
    DECREASING = 'decreasing'
    STABLE = 'stable'

class MarketMakerActivity(Enum):
    """Market maker activity levels"""
    HIGH = 'high'
    MEDIUM = 'medium'
    LOW = 'low'

class TakerAggression(Enum):
    """Taker aggression levels"""
    HIGH = 'high'
    MEDIUM = 'medium'
    LOW = 'low'

class OrderSizeCategory(Enum):
    """Order size categories"""
    LARGE = 'large'
    VERY_LARGE = 'very_large'
    WHALE = 'whale'

class OrderFlowPatternType(Enum):
    """Order flow pattern types"""
    ABSORPTION = 'absorption'
    DISTRIBUTION = 'distribution'
    ACCUMULATION = 'accumulation'
    MANIPULATION = 'manipulation'
    LIQUIDATION = 'liquidation'
    STOP_HUNTING = 'stop_hunting'
    GAMMA_SQUEEZE = 'gamma_squeeze'
    SHORT_SQUEEZE = 'short_squeeze'

class AlertLevel(Enum):
    """Alert levels for monitoring"""
    LOW = 'low'
    MEDIUM = 'medium'
    HIGH = 'high'
    CRITICAL = 'critical'

@dataclass
class OrderFlowToxicityAnalysis:
    """Order flow toxicity analysis result"""
    timestamp: datetime
    symbol: str
    timeframe: str
    toxicity_score: float  # -1 to +1
    bid_toxicity: float  # -1 to +1
    ask_toxicity: float  # -1 to +1
    large_order_ratio: float  # 0 to 1
    order_size_distribution: Dict[str, Any]
    toxicity_trend: ToxicityTrend
    toxicity_confidence: float  # 0 to 1
    market_impact_score: float  # 0 to 1
    analysis_metadata: Dict[str, Any]

@dataclass
class MarketMakerTakerAnalysis:
    """Market maker vs taker analysis result"""
    timestamp: datetime
    symbol: str
    timeframe: str
    maker_volume_ratio: float  # 0 to 1
    taker_volume_ratio: float  # 0 to 1
    maker_buy_volume: float
    maker_sell_volume: float
    taker_buy_volume: float
    taker_sell_volume: float
    maker_taker_imbalance: float  # -1 to +1
    market_maker_activity: MarketMakerActivity
    taker_aggression: TakerAggression
    spread_impact: float  # Impact on spread
    liquidity_provision_score: float  # 0 to 1
    analysis_metadata: Dict[str, Any]

@dataclass
class LargeOrder:
    """Large order tracking result"""
    timestamp: datetime
    symbol: str
    order_id: Optional[str]
    side: str  # 'buy' or 'sell'
    price: float
    quantity: float
    quote_quantity: float
    order_type: Optional[str]
    size_category: OrderSizeCategory
    size_percentile: float  # 0 to 1
    market_impact: float  # 0 to 1
    execution_time: Optional[float]  # Seconds
    fill_ratio: Optional[float]  # 0 to 1
    slippage: Optional[float]  # Percentage
    order_flow_pattern: Optional[str]
    institutional_indicator: bool
    analysis_metadata: Dict[str, Any]

@dataclass
class OrderFlowPattern:
    """Order flow pattern result"""
    timestamp: datetime
    symbol: str
    timeframe: str
    pattern_type: OrderFlowPatternType
    pattern_confidence: float  # 0 to 1
    pattern_strength: float  # 0 to 1
    volume_profile: Dict[str, Any]
    price_action: Dict[str, Any]
    order_flow_signature: Dict[str, Any]
    duration_minutes: Optional[int]
    breakout_direction: Optional[str]  # 'up', 'down', 'none'
    breakout_strength: Optional[float]  # 0 to 1
    pattern_completion: Optional[bool]
    analysis_metadata: Dict[str, Any]

@dataclass
class OrderFlowAlert:
    """Order flow monitoring alert"""
    timestamp: datetime
    symbol: str
    monitoring_type: str
    alert_level: AlertLevel
    alert_message: str
    metric_value: float
    threshold_value: float
    alert_triggered: bool
    alert_metadata: Dict[str, Any]

@dataclass
class AdvancedOrderFlowAnalysis:
    """Comprehensive order flow analysis result"""
    timestamp: datetime
    symbol: str
    timeframe: str
    toxicity_analysis: Optional[OrderFlowToxicityAnalysis]
    maker_taker_analysis: Optional[MarketMakerTakerAnalysis]
    large_orders: List[LargeOrder]
    order_flow_patterns: List[OrderFlowPattern]
    alerts: List[OrderFlowAlert]
    overall_toxicity_score: float  # -1 to +1
    market_maker_dominance: float  # 0 to 1
    large_order_activity: float  # 0 to 1
    pattern_activity: float  # 0 to 1
    analysis_confidence: float  # 0 to 1
    market_context: Dict[str, Any]
    analysis_metadata: Dict[str, Any]

class AdvancedOrderFlowAnalyzer:
    """
    Advanced Order Flow Analyzer
    Provides comprehensive order flow analysis including toxicity, market maker vs taker, large orders, and patterns
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Configuration parameters
        self.toxicity_threshold = self.config.get('toxicity_threshold', 0.3)
        self.large_order_threshold = self.config.get('large_order_threshold', 0.1)  # 10% of avg volume
        self.whale_order_threshold = self.config.get('whale_order_threshold', 0.5)  # 50% of avg volume
        self.pattern_confidence_threshold = self.config.get('pattern_confidence_threshold', 0.7)
        self.min_data_points = self.config.get('min_data_points', 100)
        self.volume_threshold = self.config.get('volume_threshold', 0.05)
        
        # Statistics tracking
        self.stats = {
            'toxicity_analyses': 0,
            'maker_taker_analyses': 0,
            'large_orders_detected': 0,
            'patterns_detected': 0,
            'alerts_generated': 0,
            'total_analyses': 0,
            'errors': 0
        }
        
        logger.info("Advanced Order Flow Analyzer initialized")
    
    async def analyze_order_flow(self, symbol: str, timeframe: str, 
                                order_book_data: pd.DataFrame,
                                trade_data: pd.DataFrame,
                                volume_data: pd.DataFrame) -> AdvancedOrderFlowAnalysis:
        """
        Perform comprehensive order flow analysis
        
        Args:
            symbol: Trading symbol
            timeframe: Analysis timeframe
            order_book_data: Order book snapshots
            trade_data: Trade data
            volume_data: Volume data
            
        Returns:
            AdvancedOrderFlowAnalysis: Comprehensive order flow analysis
        """
        try:
            logger.info(f"🔄 Analyzing order flow for {symbol} ({timeframe})")
            
            # Validate input data
            if len(order_book_data) < self.min_data_points:
                logger.warning(f"Insufficient data for {symbol}: {len(order_book_data)} < {self.min_data_points}")
                return await self._create_default_analysis(symbol, timeframe)
            
            # Perform individual analyses
            toxicity_analysis = await self._analyze_order_flow_toxicity(symbol, timeframe, order_book_data, trade_data)
            maker_taker_analysis = await self._analyze_market_maker_taker(symbol, timeframe, trade_data, volume_data)
            large_orders = await self._detect_large_orders(symbol, trade_data, volume_data)
            order_flow_patterns = await self._detect_order_flow_patterns(symbol, timeframe, order_book_data, trade_data)
            alerts = await self._generate_alerts(symbol, toxicity_analysis, maker_taker_analysis, large_orders, order_flow_patterns)
            
            # Calculate overall metrics
            overall_toxicity = toxicity_analysis.toxicity_score if toxicity_analysis else 0.0
            market_maker_dominance = maker_taker_analysis.maker_volume_ratio if maker_taker_analysis else 0.5
            large_order_activity = len(large_orders) / max(1, len(trade_data)) * 100
            pattern_activity = len(order_flow_patterns) / max(1, len(trade_data)) * 100
            
            # Calculate analysis confidence
            analysis_confidence = await self._calculate_analysis_confidence(
                toxicity_analysis, maker_taker_analysis, large_orders, order_flow_patterns
            )
            
            # Generate market context
            market_context = await self._calculate_market_context(
                symbol, order_book_data, trade_data, volume_data
            )
            
            # Create comprehensive analysis
            analysis = AdvancedOrderFlowAnalysis(
                timestamp=datetime.now(timezone.utc),
                symbol=symbol,
                timeframe=timeframe,
                toxicity_analysis=toxicity_analysis,
                maker_taker_analysis=maker_taker_analysis,
                large_orders=large_orders,
                order_flow_patterns=order_flow_patterns,
                alerts=alerts,
                overall_toxicity_score=overall_toxicity,
                market_maker_dominance=market_maker_dominance,
                large_order_activity=large_order_activity,
                pattern_activity=pattern_activity,
                analysis_confidence=analysis_confidence,
                market_context=market_context,
                analysis_metadata={
                    'data_points_analyzed': len(order_book_data),
                    'analysis_duration_ms': 0,  # TODO: Add timing
                    'config_used': self.config
                }
            )
            
            # Update statistics
            self.stats['total_analyses'] += 1
            if toxicity_analysis:
                self.stats['toxicity_analyses'] += 1
            if maker_taker_analysis:
                self.stats['maker_taker_analyses'] += 1
            self.stats['large_orders_detected'] += len(large_orders)
            self.stats['patterns_detected'] += len(order_flow_patterns)
            self.stats['alerts_generated'] += len(alerts)
            
            logger.info(f"✅ Order flow analysis completed for {symbol}: {len(large_orders)} large orders, {len(order_flow_patterns)} patterns")
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing order flow for {symbol}: {e}")
            self.stats['errors'] += 1
            return await self._create_default_analysis(symbol, timeframe)
    
    async def _analyze_order_flow_toxicity(self, symbol: str, timeframe: str,
                                         order_book_data: pd.DataFrame,
                                         trade_data: pd.DataFrame) -> Optional[OrderFlowToxicityAnalysis]:
        """Analyze order flow toxicity"""
        try:
            if len(order_book_data) < 10:
                return None
            
            # Calculate toxicity from order book data
            toxicity_scores = []
            bid_toxicities = []
            ask_toxicities = []
            large_order_ratios = []
            
            for _, row in order_book_data.iterrows():
                # Extract order book data
                bids = row.get('bids', [])
                asks = row.get('asks', [])
                
                if not bids or not asks:
                    continue
                
                # Calculate bid and ask toxicities
                bid_sizes = [bid[1] for bid in bids if len(bid) >= 2]
                ask_sizes = [ask[1] for ask in asks if len(ask) >= 2]
                
                if not bid_sizes or not ask_sizes:
                    continue
                
                avg_bid_size = np.mean(bid_sizes)
                avg_ask_size = np.mean(ask_sizes)
                
                # Calculate toxicity based on size distribution
                bid_toxicity = sum(1 for size in bid_sizes if size > avg_bid_size * 2) / len(bid_sizes)
                ask_toxicity = sum(1 for size in ask_sizes if size > avg_ask_size * 2) / len(ask_sizes)
                
                # Overall toxicity
                toxicity = (bid_toxicity + ask_toxicity) / 2
                toxicity_scores.append((toxicity - 0.5) * 2)  # Convert to -1 to 1
                bid_toxicities.append((bid_toxicity - 0.5) * 2)
                ask_toxicities.append((ask_toxicity - 0.5) * 2)
                
                # Large order ratio
                large_bid_orders = sum(1 for size in bid_sizes if size > avg_bid_size * 3)
                large_ask_orders = sum(1 for size in ask_sizes if size > avg_ask_size * 3)
                large_order_ratio = (large_bid_orders + large_ask_orders) / (len(bid_sizes) + len(ask_sizes))
                large_order_ratios.append(large_order_ratio)
            
            if not toxicity_scores:
                return None
            
            # Calculate final metrics
            avg_toxicity = np.mean(toxicity_scores)
            avg_bid_toxicity = np.mean(bid_toxicities)
            avg_ask_toxicity = np.mean(ask_toxicities)
            avg_large_order_ratio = np.mean(large_order_ratios)
            
            # Determine toxicity trend
            if len(toxicity_scores) >= 3:
                recent_toxicity = np.mean(toxicity_scores[-3:])
                earlier_toxicity = np.mean(toxicity_scores[:-3])
                if recent_toxicity > earlier_toxicity + 0.1:
                    toxicity_trend = ToxicityTrend.INCREASING
                elif recent_toxicity < earlier_toxicity - 0.1:
                    toxicity_trend = ToxicityTrend.DECREASING
                else:
                    toxicity_trend = ToxicityTrend.STABLE
            else:
                toxicity_trend = ToxicityTrend.STABLE
            
            # Calculate confidence and market impact
            toxicity_confidence = min(1.0, len(toxicity_scores) / 50)  # More data = higher confidence
            market_impact_score = abs(avg_toxicity) * toxicity_confidence
            
            # Order size distribution statistics
            order_size_distribution = {
                'mean_size': float(np.mean([size for row in order_book_data.itertuples() 
                                          for bid in getattr(row, 'bids', []) 
                                          for size in [bid[1]] if len(bid) >= 2])),
                'std_size': float(np.std([size for row in order_book_data.itertuples() 
                                        for bid in getattr(row, 'bids', []) 
                                        for size in [bid[1]] if len(bid) >= 2])),
                'size_percentiles': {
                    '25': float(np.percentile([size for row in order_book_data.itertuples() 
                                             for bid in getattr(row, 'bids', []) 
                                             for size in [bid[1]] if len(bid) >= 2], 25)),
                    '50': float(np.percentile([size for row in order_book_data.itertuples() 
                                             for bid in getattr(row, 'bids', []) 
                                             for size in [bid[1]] if len(bid) >= 2], 50)),
                    '75': float(np.percentile([size for row in order_book_data.itertuples() 
                                             for bid in getattr(row, 'bids', []) 
                                             for size in [bid[1]] if len(bid) >= 2], 75)),
                    '95': float(np.percentile([size for row in order_book_data.itertuples() 
                                             for bid in getattr(row, 'bids', []) 
                                             for size in [bid[1]] if len(bid) >= 2], 95))
                }
            }
            
            return OrderFlowToxicityAnalysis(
                timestamp=datetime.now(timezone.utc),
                symbol=symbol,
                timeframe=timeframe,
                toxicity_score=avg_toxicity,
                bid_toxicity=avg_bid_toxicity,
                ask_toxicity=avg_ask_toxicity,
                large_order_ratio=avg_large_order_ratio,
                order_size_distribution=order_size_distribution,
                toxicity_trend=toxicity_trend,
                toxicity_confidence=toxicity_confidence,
                market_impact_score=market_impact_score,
                analysis_metadata={
                    'data_points_analyzed': len(toxicity_scores),
                    'analysis_method': 'size_distribution_analysis'
                }
            )
            
        except Exception as e:
            logger.error(f"Error analyzing order flow toxicity: {e}")
            return None
    
    async def _analyze_market_maker_taker(self, symbol: str, timeframe: str,
                                        trade_data: pd.DataFrame,
                                        volume_data: pd.DataFrame) -> Optional[MarketMakerTakerAnalysis]:
        """Analyze market maker vs taker activity"""
        try:
            if len(trade_data) < 10:
                return None
            
            # Simulate maker/taker analysis (in real implementation, this would use actual maker/taker flags)
            # For now, we'll estimate based on trade characteristics
            
            maker_volumes = []
            taker_volumes = []
            maker_buy_volumes = []
            maker_sell_volumes = []
            taker_buy_volumes = []
            taker_sell_volumes = []
            
            for _, row in trade_data.iterrows():
                # Estimate maker vs taker based on trade size and timing
                # Smaller, more frequent trades are likely makers
                # Larger, less frequent trades are likely takers
                
                trade_size = row.get('quantity', 0)
                trade_value = row.get('quote_quantity', 0)
                
                # Simple heuristic: trades below median size are likely makers
                if trade_size < trade_data['quantity'].median():
                    maker_volumes.append(trade_value)
                    if row.get('side', 'buy') == 'buy':
                        maker_buy_volumes.append(trade_value)
                    else:
                        maker_sell_volumes.append(trade_value)
                else:
                    taker_volumes.append(trade_value)
                    if row.get('side', 'buy') == 'buy':
                        taker_buy_volumes.append(trade_value)
                    else:
                        taker_sell_volumes.append(trade_value)
            
            if not maker_volumes and not taker_volumes:
                return None
            
            # Calculate ratios
            total_volume = sum(maker_volumes) + sum(taker_volumes)
            maker_volume_ratio = sum(maker_volumes) / total_volume if total_volume > 0 else 0.5
            taker_volume_ratio = sum(taker_volumes) / total_volume if total_volume > 0 else 0.5
            
            # Calculate individual volumes
            maker_buy_volume = sum(maker_buy_volumes)
            maker_sell_volume = sum(maker_sell_volumes)
            taker_buy_volume = sum(taker_buy_volumes)
            taker_sell_volume = sum(taker_sell_volumes)
            
            # Calculate imbalance
            maker_taker_imbalance = (maker_volume_ratio - taker_volume_ratio)
            
            # Determine activity levels
            if maker_volume_ratio > 0.7:
                market_maker_activity = MarketMakerActivity.HIGH
            elif maker_volume_ratio > 0.4:
                market_maker_activity = MarketMakerActivity.MEDIUM
            else:
                market_maker_activity = MarketMakerActivity.LOW
            
            if taker_volume_ratio > 0.7:
                taker_aggression = TakerAggression.HIGH
            elif taker_volume_ratio > 0.4:
                taker_aggression = TakerAggression.MEDIUM
            else:
                taker_aggression = TakerAggression.LOW
            
            # Calculate spread impact (simplified)
            spread_impact = abs(maker_taker_imbalance) * 0.1  # Simplified calculation
            
            # Liquidity provision score
            liquidity_provision_score = maker_volume_ratio * 0.8 + (1 - abs(maker_taker_imbalance)) * 0.2
            
            return MarketMakerTakerAnalysis(
                timestamp=datetime.now(timezone.utc),
                symbol=symbol,
                timeframe=timeframe,
                maker_volume_ratio=maker_volume_ratio,
                taker_volume_ratio=taker_volume_ratio,
                maker_buy_volume=maker_buy_volume,
                maker_sell_volume=maker_sell_volume,
                taker_buy_volume=taker_buy_volume,
                taker_sell_volume=taker_sell_volume,
                maker_taker_imbalance=maker_taker_imbalance,
                market_maker_activity=market_maker_activity,
                taker_aggression=taker_aggression,
                spread_impact=spread_impact,
                liquidity_provision_score=liquidity_provision_score,
                analysis_metadata={
                    'data_points_analyzed': len(trade_data),
                    'analysis_method': 'trade_characteristic_estimation'
                }
            )
            
        except Exception as e:
            logger.error(f"Error analyzing market maker taker: {e}")
            return None
    
    async def _detect_large_orders(self, symbol: str, trade_data: pd.DataFrame,
                                 volume_data: pd.DataFrame) -> List[LargeOrder]:
        """Detect and analyze large orders"""
        try:
            large_orders = []
            
            if len(trade_data) < 10:
                return large_orders
            
            # Calculate volume thresholds
            avg_volume = volume_data['volume'].mean() if len(volume_data) > 0 else 1000
            large_threshold = avg_volume * self.large_order_threshold
            whale_threshold = avg_volume * self.whale_order_threshold
            
            # Analyze each trade
            for _, row in trade_data.iterrows():
                trade_size = row.get('quantity', 0)
                trade_value = row.get('quote_quantity', 0)
                
                # Determine if this is a large order
                if trade_size >= whale_threshold:
                    size_category = OrderSizeCategory.WHALE
                elif trade_size >= large_threshold:
                    size_category = OrderSizeCategory.LARGE
                else:
                    continue  # Skip non-large orders
                
                # Calculate size percentile
                size_percentile = (trade_size / avg_volume) if avg_volume > 0 else 0
                
                # Estimate market impact (simplified)
                market_impact = min(1.0, size_percentile * 0.1)
                
                # Determine if likely institutional
                institutional_indicator = size_percentile > 2.0 or trade_value > 100000
                
                # Create large order object
                large_order = LargeOrder(
                    timestamp=row.get('timestamp', datetime.now(timezone.utc)),
                    symbol=symbol,
                    order_id=row.get('order_id'),
                    side=row.get('side', 'unknown'),
                    price=row.get('price', 0),
                    quantity=trade_size,
                    quote_quantity=trade_value,
                    order_type=row.get('order_type'),
                    size_category=size_category,
                    size_percentile=size_percentile,
                    market_impact=market_impact,
                    execution_time=None,  # Would need execution data
                    fill_ratio=None,  # Would need order book data
                    slippage=None,  # Would need price impact data
                    order_flow_pattern=None,  # Would need pattern analysis
                    institutional_indicator=institutional_indicator,
                    analysis_metadata={
                        'avg_volume': avg_volume,
                        'volume_threshold': large_threshold,
                        'whale_threshold': whale_threshold
                    }
                )
                
                large_orders.append(large_order)
            
            return large_orders
            
        except Exception as e:
            logger.error(f"Error detecting large orders: {e}")
            return []
    
    async def _detect_order_flow_patterns(self, symbol: str, timeframe: str,
                                        order_book_data: pd.DataFrame,
                                        trade_data: pd.DataFrame) -> List[OrderFlowPattern]:
        """Detect order flow patterns"""
        try:
            patterns = []
            
            if len(trade_data) < 20:
                return patterns
            
            # Analyze for absorption patterns (price stays stable despite large volume)
            absorption_pattern = await self._detect_absorption_pattern(symbol, timeframe, trade_data)
            if absorption_pattern:
                patterns.append(absorption_pattern)
            
            # Analyze for distribution patterns (price declines despite large volume)
            distribution_pattern = await self._detect_distribution_pattern(symbol, timeframe, trade_data)
            if distribution_pattern:
                patterns.append(distribution_pattern)
            
            # Analyze for accumulation patterns (price rises with large volume)
            accumulation_pattern = await self._detect_accumulation_pattern(symbol, timeframe, trade_data)
            if accumulation_pattern:
                patterns.append(accumulation_pattern)
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error detecting order flow patterns: {e}")
            return []
    
    async def _detect_absorption_pattern(self, symbol: str, timeframe: str,
                                       trade_data: pd.DataFrame) -> Optional[OrderFlowPattern]:
        """Detect absorption pattern (price stable, high volume)"""
        try:
            if len(trade_data) < 10:
                return None
            
            # Look for periods where price is stable but volume is high
            window_size = min(10, len(trade_data) // 2)
            
            for i in range(len(trade_data) - window_size):
                window = trade_data.iloc[i:i+window_size]
                
                # Calculate price stability
                price_std = window['price'].std()
                price_mean = window['price'].mean()
                price_stability = 1 - (price_std / price_mean) if price_mean > 0 else 0
                
                # Calculate volume intensity
                avg_volume = trade_data['quantity'].mean()
                window_volume = window['quantity'].mean()
                volume_intensity = window_volume / avg_volume if avg_volume > 0 else 1
                
                # Check for absorption pattern
                if price_stability > 0.95 and volume_intensity > 1.5:
                    # Calculate pattern confidence
                    confidence = min(1.0, price_stability * volume_intensity / 2)
                    
                    if confidence >= self.pattern_confidence_threshold:
                        return OrderFlowPattern(
                            timestamp=window.iloc[-1]['timestamp'],
                            symbol=symbol,
                            timeframe=timeframe,
                            pattern_type=OrderFlowPatternType.ABSORPTION,
                            pattern_confidence=confidence,
                            pattern_strength=volume_intensity,
                            volume_profile={
                                'avg_volume': float(avg_volume),
                                'pattern_volume': float(window_volume),
                                'volume_intensity': float(volume_intensity)
                            },
                            price_action={
                                'price_stability': float(price_stability),
                                'price_std': float(price_std),
                                'price_range': float(window['price'].max() - window['price'].min())
                            },
                            order_flow_signature={
                                'pattern_duration': window_size,
                                'volume_distribution': 'concentrated',
                                'price_behavior': 'stable'
                            },
                            duration_minutes=window_size,
                            breakout_direction=None,
                            breakout_strength=None,
                            pattern_completion=True,
                            analysis_metadata={
                                'detection_method': 'price_stability_volume_intensity',
                                'window_size': window_size
                            }
                        )
            
            return None
            
        except Exception as e:
            logger.error(f"Error detecting absorption pattern: {e}")
            return None
    
    async def _detect_distribution_pattern(self, symbol: str, timeframe: str,
                                         trade_data: pd.DataFrame) -> Optional[OrderFlowPattern]:
        """Detect distribution pattern (price declines, high volume)"""
        try:
            if len(trade_data) < 10:
                return None
            
            window_size = min(10, len(trade_data) // 2)
            
            for i in range(len(trade_data) - window_size):
                window = trade_data.iloc[i:i+window_size]
                
                # Calculate price decline
                start_price = window.iloc[0]['price']
                end_price = window.iloc[-1]['price']
                price_change = (end_price - start_price) / start_price if start_price > 0 else 0
                
                # Calculate volume intensity
                avg_volume = trade_data['quantity'].mean()
                window_volume = window['quantity'].mean()
                volume_intensity = window_volume / avg_volume if avg_volume > 0 else 1
                
                # Check for distribution pattern (declining price with high volume)
                if price_change < -0.02 and volume_intensity > 1.3:  # 2% decline
                    confidence = min(1.0, abs(price_change) * volume_intensity / 2)
                    
                    if confidence >= self.pattern_confidence_threshold:
                        return OrderFlowPattern(
                            timestamp=window.iloc[-1]['timestamp'],
                            symbol=symbol,
                            timeframe=timeframe,
                            pattern_type=OrderFlowPatternType.DISTRIBUTION,
                            pattern_confidence=confidence,
                            pattern_strength=abs(price_change),
                            volume_profile={
                                'avg_volume': float(avg_volume),
                                'pattern_volume': float(window_volume),
                                'volume_intensity': float(volume_intensity)
                            },
                            price_action={
                                'price_change': float(price_change),
                                'start_price': float(start_price),
                                'end_price': float(end_price)
                            },
                            order_flow_signature={
                                'pattern_duration': window_size,
                                'volume_distribution': 'high',
                                'price_behavior': 'declining'
                            },
                            duration_minutes=window_size,
                            breakout_direction='down',
                            breakout_strength=abs(price_change),
                            pattern_completion=True,
                            analysis_metadata={
                                'detection_method': 'price_decline_volume_intensity',
                                'window_size': window_size
                            }
                        )
            
            return None
            
        except Exception as e:
            logger.error(f"Error detecting distribution pattern: {e}")
            return None
    
    async def _detect_accumulation_pattern(self, symbol: str, timeframe: str,
                                         trade_data: pd.DataFrame) -> Optional[OrderFlowPattern]:
        """Detect accumulation pattern (price rises, high volume)"""
        try:
            if len(trade_data) < 10:
                return None
            
            window_size = min(10, len(trade_data) // 2)
            
            for i in range(len(trade_data) - window_size):
                window = trade_data.iloc[i:i+window_size]
                
                # Calculate price rise
                start_price = window.iloc[0]['price']
                end_price = window.iloc[-1]['price']
                price_change = (end_price - start_price) / start_price if start_price > 0 else 0
                
                # Calculate volume intensity
                avg_volume = trade_data['quantity'].mean()
                window_volume = window['quantity'].mean()
                volume_intensity = window_volume / avg_volume if avg_volume > 0 else 1
                
                # Check for accumulation pattern (rising price with high volume)
                if price_change > 0.02 and volume_intensity > 1.3:  # 2% rise
                    confidence = min(1.0, price_change * volume_intensity / 2)
                    
                    if confidence >= self.pattern_confidence_threshold:
                        return OrderFlowPattern(
                            timestamp=window.iloc[-1]['timestamp'],
                            symbol=symbol,
                            timeframe=timeframe,
                            pattern_type=OrderFlowPatternType.ACCUMULATION,
                            pattern_confidence=confidence,
                            pattern_strength=price_change,
                            volume_profile={
                                'avg_volume': float(avg_volume),
                                'pattern_volume': float(window_volume),
                                'volume_intensity': float(volume_intensity)
                            },
                            price_action={
                                'price_change': float(price_change),
                                'start_price': float(start_price),
                                'end_price': float(end_price)
                            },
                            order_flow_signature={
                                'pattern_duration': window_size,
                                'volume_distribution': 'high',
                                'price_behavior': 'rising'
                            },
                            duration_minutes=window_size,
                            breakout_direction='up',
                            breakout_strength=price_change,
                            pattern_completion=True,
                            analysis_metadata={
                                'detection_method': 'price_rise_volume_intensity',
                                'window_size': window_size
                            }
                        )
            
            return None
            
        except Exception as e:
            logger.error(f"Error detecting accumulation pattern: {e}")
            return None
    
    async def _generate_alerts(self, symbol: str,
                             toxicity_analysis: Optional[OrderFlowToxicityAnalysis],
                             maker_taker_analysis: Optional[MarketMakerTakerAnalysis],
                             large_orders: List[LargeOrder],
                             order_flow_patterns: List[OrderFlowPattern]) -> List[OrderFlowAlert]:
        """Generate alerts based on order flow analysis"""
        try:
            alerts = []
            
            # Toxicity alerts
            if toxicity_analysis:
                if abs(toxicity_analysis.toxicity_score) > 0.7:
                    alerts.append(OrderFlowAlert(
                        timestamp=datetime.now(timezone.utc),
                        symbol=symbol,
                        monitoring_type='toxicity',
                        alert_level=AlertLevel.HIGH if abs(toxicity_analysis.toxicity_score) > 0.8 else AlertLevel.MEDIUM,
                        alert_message=f"High order flow toxicity detected: {toxicity_analysis.toxicity_score:.3f}",
                        metric_value=toxicity_analysis.toxicity_score,
                        threshold_value=0.7,
                        alert_triggered=True,
                        alert_metadata={'analysis_id': 'toxicity_high'}
                    ))
            
            # Large order alerts
            whale_orders = [order for order in large_orders if order.size_category == OrderSizeCategory.WHALE]
            if whale_orders:
                alerts.append(OrderFlowAlert(
                    timestamp=datetime.now(timezone.utc),
                    symbol=symbol,
                    monitoring_type='large_orders',
                    alert_level=AlertLevel.HIGH,
                    alert_message=f"Whale order detected: {len(whale_orders)} orders",
                    metric_value=len(whale_orders),
                    threshold_value=1,
                    alert_triggered=True,
                    alert_metadata={'analysis_id': 'whale_orders'}
                ))
            
            # Pattern alerts
            if order_flow_patterns:
                high_confidence_patterns = [p for p in order_flow_patterns if p.pattern_confidence > 0.8]
                if high_confidence_patterns:
                    alerts.append(OrderFlowAlert(
                        timestamp=datetime.now(timezone.utc),
                        symbol=symbol,
                        monitoring_type='patterns',
                        alert_level=AlertLevel.MEDIUM,
                        alert_message=f"High confidence order flow pattern detected: {high_confidence_patterns[0].pattern_type.value}",
                        metric_value=high_confidence_patterns[0].pattern_confidence,
                        threshold_value=0.8,
                        alert_triggered=True,
                        alert_metadata={'analysis_id': 'high_confidence_pattern'}
                    ))
            
            return alerts
            
        except Exception as e:
            logger.error(f"Error generating alerts: {e}")
            return []
    
    async def _calculate_analysis_confidence(self,
                                           toxicity_analysis: Optional[OrderFlowToxicityAnalysis],
                                           maker_taker_analysis: Optional[MarketMakerTakerAnalysis],
                                           large_orders: List[LargeOrder],
                                           order_flow_patterns: List[OrderFlowPattern]) -> float:
        """Calculate overall analysis confidence"""
        try:
            confidence_scores = []
            
            if toxicity_analysis:
                confidence_scores.append(toxicity_analysis.toxicity_confidence)
            
            if maker_taker_analysis:
                # Estimate confidence based on data quality
                confidence_scores.append(0.8)  # Simplified
            
            # Large orders confidence
            if large_orders:
                confidence_scores.append(min(1.0, len(large_orders) / 10))
            
            # Patterns confidence
            if order_flow_patterns:
                avg_pattern_confidence = np.mean([p.pattern_confidence for p in order_flow_patterns])
                confidence_scores.append(avg_pattern_confidence)
            
            return np.mean(confidence_scores) if confidence_scores else 0.5
            
        except Exception as e:
            logger.error(f"Error calculating analysis confidence: {e}")
            return 0.5
    
    async def _calculate_market_context(self, symbol: str,
                                      order_book_data: pd.DataFrame,
                                      trade_data: pd.DataFrame,
                                      volume_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate market context for order flow analysis"""
        try:
            context = {
                'symbol': symbol,
                'data_points': len(order_book_data),
                'time_range': {
                    'start': order_book_data['timestamp'].min() if len(order_book_data) > 0 else None,
                    'end': order_book_data['timestamp'].max() if len(order_book_data) > 0 else None
                },
                'volume_metrics': {
                    'total_volume': float(volume_data['volume'].sum()) if len(volume_data) > 0 else 0,
                    'avg_volume': float(volume_data['volume'].mean()) if len(volume_data) > 0 else 0,
                    'volume_volatility': float(volume_data['volume'].std()) if len(volume_data) > 0 else 0
                },
                'price_metrics': {
                    'price_range': float(trade_data['price'].max() - trade_data['price'].min()) if len(trade_data) > 0 else 0,
                    'price_volatility': float(trade_data['price'].std()) if len(trade_data) > 0 else 0
                },
                'order_book_metrics': {
                    'avg_spread': 0.0,  # Would need spread calculation
                    'depth_levels': 20,  # Default
                    'liquidity_score': 0.5  # Default
                }
            }
            
            return context
            
        except Exception as e:
            logger.error(f"Error calculating market context: {e}")
            return {'symbol': symbol, 'error': str(e)}
    
    async def _create_default_analysis(self, symbol: str, timeframe: str) -> AdvancedOrderFlowAnalysis:
        """Create default analysis when insufficient data"""
        return AdvancedOrderFlowAnalysis(
            timestamp=datetime.now(timezone.utc),
            symbol=symbol,
            timeframe=timeframe,
            toxicity_analysis=None,
            maker_taker_analysis=None,
            large_orders=[],
            order_flow_patterns=[],
            alerts=[],
            overall_toxicity_score=0.0,
            market_maker_dominance=0.5,
            large_order_activity=0.0,
            pattern_activity=0.0,
            analysis_confidence=0.0,
            market_context={'symbol': symbol, 'insufficient_data': True},
            analysis_metadata={'error': 'Insufficient data for analysis'}
        )
    
    async def close(self):
        """Close analyzer and cleanup"""
        try:
            logger.info("Advanced Order Flow Analyzer closed")
        except Exception as e:
            logger.error(f"Error closing analyzer: {e}")
