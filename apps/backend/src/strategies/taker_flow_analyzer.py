"""
Taker Flow Analyzer for AlphaPulse
Analyzes aggressive order flow (market orders vs limit orders)
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import asyncio
import aiohttp

logger = logging.getLogger(__name__)

class TakerSentiment(Enum):
    """Taker flow sentiment"""
    STRONG_BUY_PRESSURE = "strong_buy_pressure"  # >0.60
    MODERATE_BUY_PRESSURE = "moderate_buy_pressure"  # 0.55-0.60
    BALANCED = "balanced"  # 0.45-0.55
    MODERATE_SELL_PRESSURE = "moderate_sell_pressure"  # 0.40-0.45
    STRONG_SELL_PRESSURE = "strong_sell_pressure"  # <0.40

@dataclass
class TakerFlowMetrics:
    """Taker flow metrics for a period"""
    timestamp: datetime
    taker_buy_volume: float
    taker_sell_volume: float
    total_volume: float
    taker_buy_ratio: float  # 0-1
    maker_buy_volume: float
    maker_sell_volume: float
    aggressor_side: str  # 'buyers' or 'sellers'
    sentiment: TakerSentiment

@dataclass
class TakerFlowDivergence:
    """Taker flow divergence from price"""
    divergence_type: str  # 'bullish' or 'bearish'
    price_trend: str
    taker_flow_trend: str
    strength: float
    confidence: float
    timestamp: datetime

@dataclass
class TakerFlowAnalysis:
    """Complete taker flow analysis"""
    symbol: str
    timeframe: str
    timestamp: datetime
    current_taker_buy_ratio: float
    taker_sentiment: TakerSentiment
    taker_flow_trend: str  # 'increasing_buy', 'increasing_sell', 'stable'
    taker_momentum: float
    divergences: List[TakerFlowDivergence]
    flow_imbalance: float  # How imbalanced is the flow
    overall_confidence: float
    taker_signals: List[Dict[str, Any]]
    metadata: Dict[str, Any]

class TakerFlowAnalyzer:
    """
    Taker (Aggressive) Order Flow Analyzer
    
    Analyzes:
    - Taker buy vs sell volume
    - Market order vs limit order ratios
    - Aggressive buyer/seller identification
    - Flow divergences from price
    - Flow momentum and trends
    
    Taker = Aggressor (places market orders, removes liquidity)
    Maker = Passive (places limit orders, provides liquidity)
    
    Taker buy ratio > 0.55 = Buyers are aggressive (bullish)
    Taker buy ratio < 0.45 = Sellers are aggressive (bearish)
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logger
        
        # Configuration
        self.lookback_periods = self.config.get('lookback_periods', 100)
        self.min_confidence = self.config.get('min_confidence', 0.6)
        self.imbalance_threshold = self.config.get('imbalance_threshold', 0.10)  # 10% imbalance
        
        # Performance tracking
        self.stats = {
            'analyses_performed': 0,
            'divergences_detected': 0,
            'strong_imbalances_detected': 0,
            'last_update': datetime.now()
        }
        
        logger.info("🚀 Taker Flow Analyzer initialized")
    
    async def analyze_taker_flow(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str,
        trades_data: Optional[List[Dict[str, Any]]] = None
    ) -> TakerFlowAnalysis:
        """
        Analyze taker flow
        
        Args:
            df: DataFrame with OHLCV data
            symbol: Trading symbol
            timeframe: Timeframe
            trades_data: Optional list of trades with maker/taker info
            
        Returns:
            TakerFlowAnalysis with complete metrics
        """
        try:
            if len(df) < 20:
                self.logger.warning("Insufficient data for taker flow analysis")
                return self._get_default_analysis(symbol, timeframe)
            
            # Calculate taker ratios
            if trades_data:
                taker_ratios = self._calculate_taker_ratios_from_trades(trades_data, df)
            else:
                # Approximate from price action
                taker_ratios = self._approximate_taker_ratios(df)
            
            # Add to dataframe
            df['taker_buy_ratio'] = taker_ratios
            
            # Current metrics
            current_ratio = df['taker_buy_ratio'].iloc[-1]
            taker_sentiment = self._determine_taker_sentiment(current_ratio)
            
            # Flow trend
            flow_trend = self._determine_flow_trend(df['taker_buy_ratio'])
            
            # Flow momentum
            taker_momentum = self._calculate_flow_momentum(df['taker_buy_ratio'])
            
            # Detect divergences
            divergences = await self._detect_flow_divergences(df)
            
            # Calculate flow imbalance
            flow_imbalance = abs(current_ratio - 0.5)
            
            # Generate signals
            taker_signals = await self._generate_taker_signals(
                current_ratio, taker_sentiment, flow_trend, divergences, flow_imbalance
            )
            
            # Calculate overall confidence
            overall_confidence = self._calculate_overall_confidence(
                taker_sentiment, divergences, flow_imbalance
            )
            
            # Create analysis
            analysis = TakerFlowAnalysis(
                symbol=symbol,
                timeframe=timeframe,
                timestamp=df['timestamp'].iloc[-1] if 'timestamp' in df.columns else datetime.now(),
                current_taker_buy_ratio=current_ratio,
                taker_sentiment=taker_sentiment,
                taker_flow_trend=flow_trend,
                taker_momentum=taker_momentum,
                divergences=divergences,
                flow_imbalance=flow_imbalance,
                overall_confidence=overall_confidence,
                taker_signals=taker_signals,
                metadata={
                    'analysis_version': '1.0',
                    'data_source': 'trades' if trades_data else 'price_approximation',
                    'stats': self.stats
                }
            )
            
            # Update statistics
            self.stats['analyses_performed'] += 1
            self.stats['divergences_detected'] += len(divergences)
            if flow_imbalance > self.imbalance_threshold:
                self.stats['strong_imbalances_detected'] += 1
            self.stats['last_update'] = datetime.now()
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"❌ Error analyzing taker flow for {symbol}: {e}")
            return self._get_default_analysis(symbol, timeframe)
    
    def _calculate_taker_ratios_from_trades(
        self,
        trades: List[Dict[str, Any]],
        df: pd.DataFrame
    ) -> pd.Series:
        """Calculate taker buy ratios from actual trade data"""
        try:
            # Aggregate trades by candle
            taker_buy_volumes = []
            total_volumes = []
            
            for i in range(len(df)):
                # Get trades for this candle (simplified - would need timestamp matching)
                candle_taker_buy = 0.0
                candle_total = 0.0
                
                for trade in trades:
                    # Check if trade is in this candle period
                    # Simplified: assume trades are aligned with df
                    is_buyer_maker = trade.get('isBuyerMaker', False)
                    volume = trade.get('volume', 0)
                    
                    if not is_buyer_maker:  # Buyer was taker (aggressive buy)
                        candle_taker_buy += volume
                    
                    candle_total += volume
                
                if candle_total > 0:
                    ratio = candle_taker_buy / candle_total
                else:
                    ratio = 0.5
                
                taker_buy_volumes.append(candle_taker_buy)
                total_volumes.append(candle_total)
            
            ratios = pd.Series([
                taker_buy_volumes[i] / total_volumes[i] if total_volumes[i] > 0 else 0.5
                for i in range(len(df))
            ])
            
            return ratios
            
        except Exception as e:
            self.logger.error(f"Error calculating taker ratios from trades: {e}")
            return pd.Series([0.5] * len(df))
    
    def _approximate_taker_ratios(self, df: pd.DataFrame) -> pd.Series:
        """
        Approximate taker buy ratios from price action
        
        When trade data unavailable, estimate from:
        - Close position in range (high close = more buying)
        - Price momentum
        - Volume
        """
        try:
            ratios = []
            
            for i in range(len(df)):
                close = df['close'].iloc[i]
                open_price = df['open'].iloc[i]
                high = df['high'].iloc[i]
                low = df['low'].iloc[i]
                
                # Calculate close position in range
                range_size = high - low
                if range_size > 0:
                    close_position = (close - low) / range_size
                else:
                    close_position = 0.5
                
                # Adjust for candle direction
                is_green = close > open_price
                
                if is_green:
                    # Green candle: assume more taker buying
                    ratio = 0.5 + (close_position * 0.3)
                else:
                    # Red candle: assume more taker selling
                    ratio = 0.5 - ((1 - close_position) * 0.3)
                
                ratios.append(max(0.0, min(1.0, ratio)))
            
            return pd.Series(ratios)
            
        except Exception as e:
            self.logger.error(f"Error approximating taker ratios: {e}")
            return pd.Series([0.5] * len(df))
    
    def _determine_taker_sentiment(self, ratio: float) -> TakerSentiment:
        """Determine taker sentiment from ratio"""
        if ratio > 0.60:
            return TakerSentiment.STRONG_BUY_PRESSURE
        elif ratio > 0.55:
            return TakerSentiment.MODERATE_BUY_PRESSURE
        elif ratio > 0.45:
            return TakerSentiment.BALANCED
        elif ratio > 0.40:
            return TakerSentiment.MODERATE_SELL_PRESSURE
        else:
            return TakerSentiment.STRONG_SELL_PRESSURE
    
    def _determine_flow_trend(self, taker_ratios: pd.Series) -> str:
        """Determine taker flow trend"""
        try:
            if len(taker_ratios) < 10:
                return 'stable'
            
            recent = taker_ratios.tail(20)
            
            # Linear regression slope
            x = np.arange(len(recent))
            y = recent.values
            slope = np.polyfit(x, y, 1)[0]
            
            if slope > 0.01:
                return 'increasing_buy_pressure'
            elif slope < -0.01:
                return 'increasing_sell_pressure'
            else:
                return 'stable'
                
        except Exception:
            return 'stable'
    
    def _calculate_flow_momentum(self, taker_ratios: pd.Series) -> float:
        """Calculate taker flow momentum"""
        try:
            if len(taker_ratios) < 10:
                return 0.0
            
            current = taker_ratios.iloc[-1]
            past = taker_ratios.iloc[-10]
            
            momentum = current - past
            return momentum
            
        except Exception:
            return 0.0
    
    async def _detect_flow_divergences(
        self,
        df: pd.DataFrame
    ) -> List[TakerFlowDivergence]:
        """Detect taker flow divergences from price"""
        divergences = []
        
        try:
            if len(df) < 20 or 'taker_buy_ratio' not in df.columns:
                return divergences
            
            # Simplify price and flow to trends
            price_trend = 'up' if df['close'].iloc[-1] > df['close'].iloc[-20] else 'down'
            
            flow_current = df['taker_buy_ratio'].iloc[-1]
            flow_past = df['taker_buy_ratio'].iloc[-20]
            flow_trend = 'increasing_buy' if flow_current > flow_past + 0.05 else \
                        'increasing_sell' if flow_current < flow_past - 0.05 else 'stable'
            
            # Bearish divergence: Price up, taker flow decreasing
            if price_trend == 'up' and flow_trend == 'increasing_sell':
                divergence = TakerFlowDivergence(
                    divergence_type='bearish',
                    price_trend='up',
                    taker_flow_trend='weakening_buy_pressure',
                    strength=0.7,
                    confidence=0.75,
                    timestamp=df['timestamp'].iloc[-1] if 'timestamp' in df.columns else datetime.now()
                )
                divergences.append(divergence)
            
            # Bullish divergence: Price down, taker flow increasing
            elif price_trend == 'down' and flow_trend == 'increasing_buy':
                divergence = TakerFlowDivergence(
                    divergence_type='bullish',
                    price_trend='down',
                    taker_flow_trend='increasing_buy_pressure',
                    strength=0.7,
                    confidence=0.75,
                    timestamp=df['timestamp'].iloc[-1] if 'timestamp' in df.columns else datetime.now()
                )
                divergences.append(divergence)
            
            return divergences
            
        except Exception as e:
            self.logger.error(f"Error detecting flow divergences: {e}")
            return divergences
    
    async def _generate_taker_signals(
        self,
        current_ratio: float,
        sentiment: TakerSentiment,
        flow_trend: str,
        divergences: List[TakerFlowDivergence],
        imbalance: float
    ) -> List[Dict[str, Any]]:
        """Generate trading signals from taker flow"""
        signals = []
        
        try:
            # Strong imbalance signals
            if sentiment == TakerSentiment.STRONG_BUY_PRESSURE:
                signals.append({
                    'type': 'taker_flow',
                    'direction': 'bullish',
                    'confidence': 0.75,
                    'taker_buy_ratio': current_ratio,
                    'reasoning': f"Strong taker buy pressure ({current_ratio:.2%})",
                    'priority': 'high'
                })
            elif sentiment == TakerSentiment.STRONG_SELL_PRESSURE:
                signals.append({
                    'type': 'taker_flow',
                    'direction': 'bearish',
                    'confidence': 0.75,
                    'taker_buy_ratio': current_ratio,
                    'reasoning': f"Strong taker sell pressure ({current_ratio:.2%})",
                    'priority': 'high'
                })
            
            # Divergence signals
            for div in divergences:
                signals.append({
                    'type': 'taker_flow_divergence',
                    'direction': div.divergence_type,
                    'confidence': div.confidence,
                    'strength': div.strength,
                    'reasoning': f"Taker flow {div.divergence_type} divergence",
                    'priority': 'high'
                })
            
            # Flow trend signals
            if flow_trend == 'increasing_buy_pressure':
                signals.append({
                    'type': 'taker_flow_trend',
                    'direction': 'bullish',
                    'confidence': 0.65,
                    'reasoning': "Taker buy pressure increasing",
                    'priority': 'medium'
                })
            elif flow_trend == 'increasing_sell_pressure':
                signals.append({
                    'type': 'taker_flow_trend',
                    'direction': 'bearish',
                    'confidence': 0.65,
                    'reasoning': "Taker sell pressure increasing",
                    'priority': 'medium'
                })
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Error generating taker signals: {e}")
            return signals
    
    def _calculate_overall_confidence(
        self,
        sentiment: TakerSentiment,
        divergences: List[TakerFlowDivergence],
        imbalance: float
    ) -> float:
        """Calculate overall taker flow confidence"""
        try:
            confidence = 0.5
            
            # Strong sentiment
            if sentiment in [TakerSentiment.STRONG_BUY_PRESSURE, TakerSentiment.STRONG_SELL_PRESSURE]:
                confidence += 0.25
            elif sentiment in [TakerSentiment.MODERATE_BUY_PRESSURE, TakerSentiment.MODERATE_SELL_PRESSURE]:
                confidence += 0.10
            
            # Divergences
            if divergences:
                confidence += min(0.2, len(divergences) * 0.1)
            
            # Imbalance magnitude
            if imbalance > self.imbalance_threshold:
                confidence += 0.1
            
            return min(0.90, confidence)
            
        except Exception:
            return 0.5
    
    def _get_default_analysis(self, symbol: str, timeframe: str) -> TakerFlowAnalysis:
        """Get default analysis when data unavailable"""
        return TakerFlowAnalysis(
            symbol=symbol,
            timeframe=timeframe,
            timestamp=datetime.now(),
            current_taker_buy_ratio=0.5,
            taker_sentiment=TakerSentiment.BALANCED,
            taker_flow_trend='stable',
            taker_momentum=0.0,
            divergences=[],
            flow_imbalance=0.0,
            overall_confidence=0.0,
            taker_signals=[],
            metadata={'error': 'Insufficient data'}
        )
    
    async def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        return {
            'stats': self.stats,
            'config': self.config,
            'last_update': datetime.now().isoformat()
        }

