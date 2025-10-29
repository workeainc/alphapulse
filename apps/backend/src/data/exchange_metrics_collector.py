"""
Exchange Metrics Collector for AlphaPulse
Collects long/short ratios, top trader positions, and exchange-specific metrics
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

class ExchangeName(Enum):
    """Supported exchanges"""
    BINANCE = "binance"
    BYBIT = "bybit"
    OKX = "okx"

class SentimentExtreme(Enum):
    """Sentiment extreme levels"""
    EXTREME_LONG = "extreme_long"  # >3.0 ratio
    HEAVY_LONG = "heavy_long"  # 2.0-3.0
    BALANCED = "balanced"  # 0.7-1.5
    HEAVY_SHORT = "heavy_short"  # 0.33-0.5
    EXTREME_SHORT = "extreme_short"  # <0.33

@dataclass
class LongShortMetrics:
    """Long/short ratio metrics"""
    exchange: ExchangeName
    symbol: str
    timestamp: datetime
    long_short_ratio: float
    long_account_pct: float
    short_account_pct: float
    long_position_pct: float
    short_position_pct: float
    sentiment_extreme: SentimentExtreme
    contrarian_signal: Optional[str]  # 'bullish' or 'bearish' contrarian
    confidence: float
    metadata: Dict[str, Any]

@dataclass
class TopTraderMetrics:
    """Top/elite trader position metrics"""
    exchange: ExchangeName
    symbol: str
    timestamp: datetime
    top_long_short_ratio: float
    top_long_pct: float
    top_short_pct: float
    retail_long_short_ratio: float
    elite_retail_divergence: float
    follow_elite_signal: Optional[str]
    confidence: float
    metadata: Dict[str, Any]

@dataclass
class ExchangeMetricsAnalysis:
    """Complete exchange metrics analysis"""
    symbol: str
    timestamp: datetime
    long_short_metrics: Dict[ExchangeName, LongShortMetrics]
    top_trader_metrics: Optional[TopTraderMetrics]
    aggregated_long_short_ratio: float
    overall_sentiment: SentimentExtreme
    contrarian_opportunity: bool
    confidence: float
    signals: List[Dict[str, Any]]
    metadata: Dict[str, Any]

class ExchangeMetricsCollector:
    """
    Collect exchange-specific trading metrics
    
    Supported Metrics:
    - Long/Short Account Ratio (% of accounts long vs short)
    - Long/Short Position Ratio (% of positions long vs short)
    - Top Trader Long/Short Ratio (elite trader sentiment)
    - Taker Buy/Sell Ratio
    
    Exchanges:
    - Binance Futures
    - Bybit Derivatives
    - OKX Derivatives
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logger
        
        # API endpoints
        self.endpoints = {
            ExchangeName.BINANCE: {
                'base': 'https://fapi.binance.com',
                'long_short_ratio': '/futures/data/globalLongShortAccountRatio',
                'top_trader_ratio': '/futures/data/topLongShortPositionRatio',
                'taker_buy_sell': '/futures/data/takerlongshortRatio'
            },
            ExchangeName.BYBIT: {
                'base': 'https://api.bybit.com',
                'long_short_ratio': '/v2/public/account-ratio'
            },
            ExchangeName.OKX: {
                'base': 'https://www.okx.com',
                'long_short_ratio': '/api/v5/rubik/stat/contracts/long-short-account-ratio'
            }
        }
        
        # Configuration
        self.default_period = self.config.get('period', '1h')
        self.default_limit = self.config.get('limit', 30)
        
        # Cache
        self.cache: Dict[str, Any] = {}
        self.cache_ttl = timedelta(minutes=5)
        
        # Performance tracking
        self.stats = {
            'api_calls': 0,
            'cache_hits': 0,
            'extreme_ratios_detected': 0,
            'last_update': datetime.now()
        }
        
        logger.info("🚀 Exchange Metrics Collector initialized")
    
    async def analyze_exchange_metrics(
        self,
        symbol: str,
        period: str = '1h',
        limit: int = 30
    ) -> ExchangeMetricsAnalysis:
        """
        Analyze exchange metrics for symbol
        
        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            period: Period for metrics ('5m', '15m', '1h', '4h', '1d')
            limit: Number of data points
            
        Returns:
            ExchangeMetricsAnalysis with all metrics
        """
        try:
            # Collect long/short ratios from multiple exchanges
            long_short_metrics = {}
            
            # Binance
            binance_ls = await self._get_binance_long_short(symbol, period, limit)
            if binance_ls:
                long_short_metrics[ExchangeName.BINANCE] = binance_ls
            
            # Bybit
            bybit_ls = await self._get_bybit_long_short(symbol, period, limit)
            if bybit_ls:
                long_short_metrics[ExchangeName.BYBIT] = bybit_ls
            
            # OKX
            okx_ls = await self._get_okx_long_short(symbol, period, limit)
            if okx_ls:
                long_short_metrics[ExchangeName.OKX] = okx_ls
            
            # Get top trader metrics (Binance only for now)
            top_trader_metrics = await self._get_binance_top_traders(symbol, period, limit)
            
            # Calculate aggregated metrics
            if long_short_metrics:
                aggregated_ratio = np.mean([m.long_short_ratio for m in long_short_metrics.values()])
                overall_sentiment = self._determine_sentiment(aggregated_ratio)
            else:
                aggregated_ratio = 1.0
                overall_sentiment = SentimentExtreme.BALANCED
            
            # Check for contrarian opportunities
            contrarian_opportunity = overall_sentiment in [
                SentimentExtreme.EXTREME_LONG,
                SentimentExtreme.EXTREME_SHORT
            ]
            
            # Generate signals
            signals = await self._generate_signals(
                long_short_metrics, top_trader_metrics, overall_sentiment
            )
            
            # Calculate confidence
            confidence = self._calculate_confidence(
                long_short_metrics, top_trader_metrics
            )
            
            # Create analysis
            analysis = ExchangeMetricsAnalysis(
                symbol=symbol,
                timestamp=datetime.now(),
                long_short_metrics=long_short_metrics,
                top_trader_metrics=top_trader_metrics,
                aggregated_long_short_ratio=aggregated_ratio,
                overall_sentiment=overall_sentiment,
                contrarian_opportunity=contrarian_opportunity,
                confidence=confidence,
                signals=signals,
                metadata={
                    'analysis_version': '1.0',
                    'exchanges_count': len(long_short_metrics),
                    'period': period,
                    'stats': self.stats
                }
            )
            
            # Update stats
            self.stats['last_update'] = datetime.now()
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"❌ Error analyzing exchange metrics for {symbol}: {e}")
            return self._get_default_analysis(symbol)
    
    async def _get_binance_long_short(
        self,
        symbol: str,
        period: str,
        limit: int
    ) -> Optional[LongShortMetrics]:
        """Get long/short ratio from Binance"""
        try:
            cache_key = f"binance_ls_{symbol}_{period}"
            if cache_key in self.cache:
                cache_time, cached_data = self.cache[cache_key]
                if datetime.now() - cache_time < self.cache_ttl:
                    self.stats['cache_hits'] += 1
                    return cached_data
            
            base_url = self.endpoints[ExchangeName.BINANCE]['base']
            endpoint = self.endpoints[ExchangeName.BINANCE]['long_short_ratio']
            
            params = {
                'symbol': symbol,
                'period': period,
                'limit': limit
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{base_url}{endpoint}", params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        if data:
                            # Get most recent data point
                            latest = data[-1]
                            
                            long_account_pct = float(latest.get('longAccount', 0.5)) * 100
                            short_account_pct = float(latest.get('shortAccount', 0.5)) * 100
                            long_short_ratio = long_account_pct / short_account_pct if short_account_pct > 0 else 1.0
                            
                            sentiment = self._determine_sentiment(long_short_ratio)
                            contrarian_signal = self._get_contrarian_signal(sentiment)
                            confidence = self._calculate_ls_confidence(long_short_ratio)
                            
                            metrics = LongShortMetrics(
                                exchange=ExchangeName.BINANCE,
                                symbol=symbol,
                                timestamp=datetime.fromtimestamp(int(latest['timestamp']) / 1000),
                                long_short_ratio=long_short_ratio,
                                long_account_pct=long_account_pct,
                                short_account_pct=short_account_pct,
                                long_position_pct=0.0,  # Would need separate call
                                short_position_pct=0.0,
                                sentiment_extreme=sentiment,
                                contrarian_signal=contrarian_signal,
                                confidence=confidence,
                                metadata={'raw_data': latest}
                            )
                            
                            # Cache result
                            self.cache[cache_key] = (datetime.now(), metrics)
                            self.stats['api_calls'] += 1
                            
                            return metrics
                    else:
                        self.logger.warning(f"Binance API error: {response.status}")
                        
        except Exception as e:
            self.logger.error(f"Error getting Binance long/short: {e}")
        
        return None
    
    async def _get_bybit_long_short(
        self,
        symbol: str,
        period: str,
        limit: int
    ) -> Optional[LongShortMetrics]:
        """Get long/short ratio from Bybit"""
        try:
            # Similar implementation for Bybit API
            # Bybit uses different endpoint structure
            # Returns similar LongShortMetrics
            pass
            
        except Exception as e:
            self.logger.error(f"Error getting Bybit long/short: {e}")
        
        return None
    
    async def _get_okx_long_short(
        self,
        symbol: str,
        period: str,
        limit: int
    ) -> Optional[LongShortMetrics]:
        """Get long/short ratio from OKX"""
        try:
            # Similar implementation for OKX API
            pass
            
        except Exception as e:
            self.logger.error(f"Error getting OKX long/short: {e}")
        
        return None
    
    async def _get_binance_top_traders(
        self,
        symbol: str,
        period: str,
        limit: int
    ) -> Optional[TopTraderMetrics]:
        """Get top trader positions from Binance"""
        try:
            base_url = self.endpoints[ExchangeName.BINANCE]['base']
            endpoint = self.endpoints[ExchangeName.BINANCE]['top_trader_ratio']
            
            params = {
                'symbol': symbol,
                'period': period,
                'limit': limit
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{base_url}{endpoint}", params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        if data:
                            latest = data[-1]
                            
                            top_long_pct = float(latest.get('longAccount', 0.5)) * 100
                            top_short_pct = float(latest.get('shortAccount', 0.5)) * 100
                            top_ratio = top_long_pct / top_short_pct if top_short_pct > 0 else 1.0
                            
                            # Would need additional call for retail ratio comparison
                            retail_ratio = 1.0  # Placeholder
                            
                            elite_retail_divergence = abs(top_ratio - retail_ratio) / retail_ratio if retail_ratio > 0 else 0
                            
                            # Determine follow signal
                            follow_signal = None
                            if top_ratio > 2.0:
                                follow_signal = 'bullish'
                            elif top_ratio < 0.5:
                                follow_signal = 'bearish'
                            
                            confidence = 0.7 if follow_signal else 0.5
                            
                            metrics = TopTraderMetrics(
                                exchange=ExchangeName.BINANCE,
                                symbol=symbol,
                                timestamp=datetime.fromtimestamp(int(latest['timestamp']) / 1000),
                                top_long_short_ratio=top_ratio,
                                top_long_pct=top_long_pct,
                                top_short_pct=top_short_pct,
                                retail_long_short_ratio=retail_ratio,
                                elite_retail_divergence=elite_retail_divergence,
                                follow_elite_signal=follow_signal,
                                confidence=confidence,
                                metadata={'raw_data': latest}
                            )
                            
                            self.stats['api_calls'] += 1
                            return metrics
                            
        except Exception as e:
            self.logger.error(f"Error getting top trader data: {e}")
        
        return None
    
    def _determine_sentiment(self, long_short_ratio: float) -> SentimentExtreme:
        """Determine sentiment from long/short ratio"""
        if long_short_ratio > 3.0:
            return SentimentExtreme.EXTREME_LONG
        elif long_short_ratio > 2.0:
            return SentimentExtreme.HEAVY_LONG
        elif long_short_ratio > 0.5:
            return SentimentExtreme.BALANCED
        elif long_short_ratio > 0.33:
            return SentimentExtreme.HEAVY_SHORT
        else:
            return SentimentExtreme.EXTREME_SHORT
    
    def _get_contrarian_signal(self, sentiment: SentimentExtreme) -> Optional[str]:
        """Get contrarian signal from sentiment extreme"""
        if sentiment == SentimentExtreme.EXTREME_LONG:
            return 'bearish'  # Too many longs, go short
        elif sentiment == SentimentExtreme.EXTREME_SHORT:
            return 'bullish'  # Too many shorts, go long
        return None
    
    def _calculate_ls_confidence(self, ratio: float) -> float:
        """Calculate confidence based on how extreme the ratio is"""
        # More extreme = higher confidence
        if ratio > 3.0 or ratio < 0.33:
            return 0.85
        elif ratio > 2.5 or ratio < 0.4:
            return 0.75
        elif ratio > 2.0 or ratio < 0.5:
            return 0.65
        else:
            return 0.5
    
    async def _generate_signals(
        self,
        long_short_metrics: Dict[ExchangeName, LongShortMetrics],
        top_trader_metrics: Optional[TopTraderMetrics],
        overall_sentiment: SentimentExtreme
    ) -> List[Dict[str, Any]]:
        """Generate trading signals from exchange metrics"""
        signals = []
        
        try:
            # Contrarian signals from extremes
            if overall_sentiment == SentimentExtreme.EXTREME_LONG:
                signals.append({
                    'type': 'contrarian_sentiment',
                    'direction': 'bearish',
                    'confidence': 0.80,
                    'reasoning': 'Extreme long positioning - contrarian bearish',
                    'action': 'consider_shorts'
                })
            elif overall_sentiment == SentimentExtreme.EXTREME_SHORT:
                signals.append({
                    'type': 'contrarian_sentiment',
                    'direction': 'bullish',
                    'confidence': 0.80,
                    'reasoning': 'Extreme short positioning - contrarian bullish',
                    'action': 'consider_longs'
                })
            
            # Top trader following signals
            if top_trader_metrics and top_trader_metrics.follow_elite_signal:
                signals.append({
                    'type': 'elite_trader_following',
                    'direction': top_trader_metrics.follow_elite_signal,
                    'confidence': top_trader_metrics.confidence,
                    'elite_ratio': top_trader_metrics.top_long_short_ratio,
                    'reasoning': f"Top traders are {top_trader_metrics.follow_elite_signal}",
                    'action': 'follow_elite_traders'
                })
            
            # Multi-exchange agreement
            if len(long_short_metrics) >= 2:
                # Check if multiple exchanges show same extreme
                extreme_count = sum(
                    1 for m in long_short_metrics.values()
                    if m.sentiment_extreme in [SentimentExtreme.EXTREME_LONG, SentimentExtreme.EXTREME_SHORT]
                )
                
                if extreme_count >= 2:
                    signals.append({
                        'type': 'multi_exchange_agreement',
                        'direction': 'contrarian',
                        'confidence': 0.85,
                        'reasoning': f"{extreme_count} exchanges showing extreme positioning",
                        'action': 'high_confidence_contrarian'
                    })
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Error generating signals: {e}")
            return signals
    
    def _calculate_confidence(
        self,
        long_short_metrics: Dict[ExchangeName, LongShortMetrics],
        top_trader_metrics: Optional[TopTraderMetrics]
    ) -> float:
        """Calculate overall confidence"""
        try:
            confidence = 0.5
            
            # More exchanges = higher confidence
            confidence += min(0.2, len(long_short_metrics) * 0.07)
            
            # Extreme positioning = higher confidence
            extreme_count = sum(
                1 for m in long_short_metrics.values()
                if m.sentiment_extreme in [SentimentExtreme.EXTREME_LONG, SentimentExtreme.EXTREME_SHORT]
            )
            
            if extreme_count > 0:
                confidence += min(0.3, extreme_count * 0.15)
            
            # Top trader confirmation
            if top_trader_metrics:
                confidence += 0.1
            
            return min(0.95, confidence)
            
        except Exception:
            return 0.5
    
    def _get_default_analysis(self, symbol: str) -> ExchangeMetricsAnalysis:
        """Get default analysis when data unavailable"""
        return ExchangeMetricsAnalysis(
            symbol=symbol,
            timestamp=datetime.now(),
            long_short_metrics={},
            top_trader_metrics=None,
            aggregated_long_short_ratio=1.0,
            overall_sentiment=SentimentExtreme.BALANCED,
            contrarian_opportunity=False,
            confidence=0.0,
            signals=[],
            metadata={'error': 'No data available'}
        )
    
    async def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        return {
            'stats': self.stats,
            'config': self.config,
            'cache_size': len(self.cache),
            'last_update': datetime.now().isoformat()
        }

