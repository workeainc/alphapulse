"""
Exchange Reserves Tracker for AlphaPulse
Tracks absolute cryptocurrency reserves on exchanges for supply/demand analysis
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

class ReserveLevel(Enum):
    """Exchange reserve levels"""
    MULTI_YEAR_LOW = "multi_year_low"
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    MULTI_YEAR_HIGH = "multi_year_high"

class ReserveTrend(Enum):
    """Reserve trend direction"""
    SHARP_OUTFLOW = "sharp_outflow"  # >5% in 24h
    CONSISTENT_OUTFLOW = "consistent_outflow"
    STABLE = "stable"
    CONSISTENT_INFLOW = "consistent_inflow"
    SHARP_INFLOW = "sharp_inflow"  # >5% in 24h

@dataclass
class ExchangeReserve:
    """Reserve data for a single exchange"""
    exchange_name: str
    asset: str
    timestamp: datetime
    absolute_reserve: float
    reserve_change_24h: float
    reserve_change_7d: float
    reserve_change_30d: float
    reserve_level: ReserveLevel
    trend: ReserveTrend
    historical_percentile: float  # 0-100
    metadata: Dict[str, Any]

@dataclass
class ReserveAnomaly:
    """Detected reserve anomaly"""
    exchange_name: str
    asset: str
    anomaly_type: str
    timestamp: datetime
    magnitude: float
    significance: float
    signal: str  # 'bullish' or 'bearish'
    confidence: float
    metadata: Dict[str, Any]

@dataclass
class ReservesAnalysis:
    """Complete exchange reserves analysis"""
    asset: str
    timestamp: datetime
    exchange_reserves: Dict[str, ExchangeReserve]
    total_exchange_reserves: float
    reserve_dominance: Dict[str, float]  # % of total per exchange
    anomalies: List[ReserveAnomaly]
    overall_trend: ReserveTrend
    supply_shock_risk: float
    overall_confidence: float
    reserve_signals: List[Dict[str, Any]]
    metadata: Dict[str, Any]

class ExchangeReservesTracker:
    """
    Exchange Reserves Absolute Tracking System
    
    Tracks:
    - Absolute BTC/ETH reserves on major exchanges
    - Reserve trends and flows
    - Supply shock detection
    - Reserve anomalies
    - Multi-year comparisons
    
    Data Sources:
    - CryptoQuant API
    - Glassnode API
    - On-chain wallet tracking
    
    Signals:
    - Multi-year low reserves = Supply shock risk (bullish)
    - Sharp outflows = Potential volatility
    - Consistent outflows = Long-term bullish (HODLing)
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logger
        
        # Configuration
        self.tracked_exchanges = self.config.get('tracked_exchanges', [
            'Binance', 'Coinbase', 'Kraken', 'Bitfinex', 'Huobi'
        ])
        self.tracked_assets = self.config.get('tracked_assets', ['BTC', 'ETH'])
        
        # API configuration
        self.cryptoquant_api_key = self.config.get('cryptoquant_api_key')
        self.glassnode_api_key = self.config.get('glassnode_api_key')
        
        # Thresholds
        self.sharp_flow_threshold = self.config.get('sharp_flow_threshold', 0.05)  # 5%
        self.supply_shock_threshold = self.config.get('supply_shock_threshold', 10)  # 10th percentile
        
        # Cache
        self.cache: Dict[str, Tuple[datetime, Any]] = {}
        self.cache_ttl = timedelta(hours=1)  # Reserve data doesn't change that fast
        
        # Performance tracking
        self.stats = {
            'api_calls': 0,
            'cache_hits': 0,
            'anomalies_detected': 0,
            'supply_shocks_detected': 0,
            'last_update': datetime.now()
        }
        
        logger.info("🚀 Exchange Reserves Tracker initialized")
    
    async def analyze_reserves(
        self,
        asset: str = 'BTC'
    ) -> ReservesAnalysis:
        """
        Analyze exchange reserves for an asset
        
        Args:
            asset: Asset to analyze (BTC, ETH, etc.)
            
        Returns:
            ReservesAnalysis with complete reserve metrics
        """
        try:
            # Check cache
            cache_key = f"reserves_{asset}"
            if cache_key in self.cache:
                cache_time, cached_data = self.cache[cache_key]
                if datetime.now() - cache_time < self.cache_ttl:
                    self.stats['cache_hits'] += 1
                    return cached_data
            
            # Fetch reserve data for each exchange
            exchange_reserves = {}
            
            for exchange in self.tracked_exchanges:
                reserve = await self._fetch_exchange_reserve(exchange, asset)
                if reserve:
                    exchange_reserves[exchange] = reserve
            
            if not exchange_reserves:
                self.logger.warning(f"No reserve data available for {asset}")
                return self._get_default_analysis(asset)
            
            # Calculate total reserves
            total_reserves = sum(r.absolute_reserve for r in exchange_reserves.values())
            
            # Calculate reserve dominance
            reserve_dominance = {
                exchange: (reserve.absolute_reserve / total_reserves) * 100
                for exchange, reserve in exchange_reserves.items()
            }
            
            # Detect anomalies
            anomalies = await self._detect_anomalies(exchange_reserves)
            
            # Determine overall trend
            overall_trend = self._determine_overall_trend(exchange_reserves)
            
            # Calculate supply shock risk
            supply_shock_risk = self._calculate_supply_shock_risk(
                exchange_reserves, total_reserves
            )
            
            # Generate signals
            reserve_signals = await self._generate_reserve_signals(
                exchange_reserves, anomalies, overall_trend, supply_shock_risk
            )
            
            # Calculate confidence
            overall_confidence = self._calculate_confidence(
                len(exchange_reserves), anomalies
            )
            
            # Create analysis
            analysis = ReservesAnalysis(
                asset=asset,
                timestamp=datetime.now(),
                exchange_reserves=exchange_reserves,
                total_exchange_reserves=total_reserves,
                reserve_dominance=reserve_dominance,
                anomalies=anomalies,
                overall_trend=overall_trend,
                supply_shock_risk=supply_shock_risk,
                overall_confidence=overall_confidence,
                reserve_signals=reserve_signals,
                metadata={
                    'analysis_version': '1.0',
                    'exchanges_tracked': len(exchange_reserves),
                    'stats': self.stats
                }
            )
            
            # Update cache
            self.cache[cache_key] = (datetime.now(), analysis)
            
            # Update statistics
            self.stats['anomalies_detected'] += len(anomalies)
            if supply_shock_risk > 0.7:
                self.stats['supply_shocks_detected'] += 1
            self.stats['last_update'] = datetime.now()
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"❌ Error analyzing reserves for {asset}: {e}")
            return self._get_default_analysis(asset)
    
    async def _fetch_exchange_reserve(
        self,
        exchange: str,
        asset: str
    ) -> Optional[ExchangeReserve]:
        """Fetch reserve data for a single exchange"""
        try:
            # Try CryptoQuant API first
            if self.cryptoquant_api_key:
                reserve = await self._fetch_from_cryptoquant(exchange, asset)
                if reserve:
                    return reserve
            
            # Fallback to Glassnode
            if self.glassnode_api_key:
                reserve = await self._fetch_from_glassnode(exchange, asset)
                if reserve:
                    return reserve
            
            # Fallback to estimation (placeholder)
            return self._estimate_reserve(exchange, asset)
            
        except Exception as e:
            self.logger.error(f"Error fetching reserve for {exchange}/{asset}: {e}")
            return None
    
    async def _fetch_from_cryptoquant(
        self,
        exchange: str,
        asset: str
    ) -> Optional[ExchangeReserve]:
        """Fetch from CryptoQuant API"""
        try:
            # CryptoQuant API endpoint
            base_url = "https://api.cryptoquant.com/v1"
            endpoint = f"{base_url}/{asset.lower()}/exchange-flows/reserve"
            
            headers = {'Authorization': f'Bearer {self.cryptoquant_api_key}'}
            params = {'exchange': exchange.lower(), 'window': 'day', 'limit': 30}
            
            async with aiohttp.ClientSession() as session:
                async with session.get(endpoint, headers=headers, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        if data and 'result' in data and 'data' in data['result']:
                            reserve_data = data['result']['data']
                            
                            if reserve_data:
                                latest = reserve_data[-1]
                                
                                # Calculate changes
                                reserve_24h_ago = reserve_data[-2]['reserve'] if len(reserve_data) > 1 else latest['reserve']
                                reserve_7d_ago = reserve_data[-7]['reserve'] if len(reserve_data) > 7 else latest['reserve']
                                reserve_30d_ago = reserve_data[0]['reserve']
                                
                                current_reserve = latest['reserve']
                                
                                change_24h = ((current_reserve - reserve_24h_ago) / reserve_24h_ago) * 100 if reserve_24h_ago > 0 else 0
                                change_7d = ((current_reserve - reserve_7d_ago) / reserve_7d_ago) * 100 if reserve_7d_ago > 0 else 0
                                change_30d = ((current_reserve - reserve_30d_ago) / reserve_30d_ago) * 100 if reserve_30d_ago > 0 else 0
                                
                                # Determine level and trend
                                level = self._determine_reserve_level(current_reserve, reserve_data)
                                trend = self._determine_reserve_trend(change_24h, change_7d)
                                
                                # Calculate percentile
                                all_reserves = [d['reserve'] for d in reserve_data]
                                percentile = (sum(1 for r in all_reserves if r <= current_reserve) / len(all_reserves)) * 100
                                
                                reserve = ExchangeReserve(
                                    exchange_name=exchange,
                                    asset=asset,
                                    timestamp=datetime.now(),
                                    absolute_reserve=current_reserve,
                                    reserve_change_24h=change_24h,
                                    reserve_change_7d=change_7d,
                                    reserve_change_30d=change_30d,
                                    reserve_level=level,
                                    trend=trend,
                                    historical_percentile=percentile,
                                    metadata={'source': 'cryptoquant'}
                                )
                                
                                self.stats['api_calls'] += 1
                                return reserve
                                
        except Exception as e:
            self.logger.error(f"Error fetching from CryptoQuant: {e}")
        
        return None
    
    async def _fetch_from_glassnode(
        self,
        exchange: str,
        asset: str
    ) -> Optional[ExchangeReserve]:
        """Fetch from Glassnode API"""
        # Similar implementation to CryptoQuant
        return None
    
    def _estimate_reserve(self, exchange: str, asset: str) -> ExchangeReserve:
        """Estimate reserve when no API data available"""
        # Placeholder estimates
        return ExchangeReserve(
            exchange_name=exchange,
            asset=asset,
            timestamp=datetime.now(),
            absolute_reserve=0.0,
            reserve_change_24h=0.0,
            reserve_change_7d=0.0,
            reserve_change_30d=0.0,
            reserve_level=ReserveLevel.NORMAL,
            trend=ReserveTrend.STABLE,
            historical_percentile=50.0,
            metadata={'source': 'estimated'}
        )
    
    def _determine_reserve_level(
        self,
        current: float,
        historical: List[Dict[str, Any]]
    ) -> ReserveLevel:
        """Determine if reserves are at multi-year lows/highs"""
        try:
            values = [d['reserve'] for d in historical]
            percentile = (sum(1 for v in values if v <= current) / len(values)) * 100
            
            if percentile < 10:
                return ReserveLevel.MULTI_YEAR_LOW
            elif percentile < 30:
                return ReserveLevel.LOW
            elif percentile > 90:
                return ReserveLevel.MULTI_YEAR_HIGH
            elif percentile > 70:
                return ReserveLevel.HIGH
            else:
                return ReserveLevel.NORMAL
                
        except Exception:
            return ReserveLevel.NORMAL
    
    def _determine_reserve_trend(
        self,
        change_24h: float,
        change_7d: float
    ) -> ReserveTrend:
        """Determine reserve trend"""
        if change_24h < -5:
            return ReserveTrend.SHARP_OUTFLOW
        elif change_7d < -3:
            return ReserveTrend.CONSISTENT_OUTFLOW
        elif change_24h > 5:
            return ReserveTrend.SHARP_INFLOW
        elif change_7d > 3:
            return ReserveTrend.CONSISTENT_INFLOW
        else:
            return ReserveTrend.STABLE
    
    async def _detect_anomalies(
        self,
        reserves: Dict[str, ExchangeReserve]
    ) -> List[ReserveAnomaly]:
        """Detect reserve anomalies"""
        anomalies = []
        
        try:
            for exchange, reserve in reserves.items():
                # Multi-year low anomaly
                if reserve.reserve_level == ReserveLevel.MULTI_YEAR_LOW:
                    anomaly = ReserveAnomaly(
                        exchange_name=exchange,
                        asset=reserve.asset,
                        anomaly_type='multi_year_low',
                        timestamp=reserve.timestamp,
                        magnitude=reserve.historical_percentile,
                        significance=1.0 - (reserve.historical_percentile / 100),
                        signal='bullish',
                        confidence=0.85,
                        metadata={'percentile': reserve.historical_percentile}
                    )
                    anomalies.append(anomaly)
                
                # Sharp outflow anomaly
                if reserve.trend == ReserveTrend.SHARP_OUTFLOW:
                    anomaly = ReserveAnomaly(
                        exchange_name=exchange,
                        asset=reserve.asset,
                        anomaly_type='sharp_outflow',
                        timestamp=reserve.timestamp,
                        magnitude=abs(reserve.reserve_change_24h),
                        significance=min(1.0, abs(reserve.reserve_change_24h) / 10),
                        signal='bullish',
                        confidence=0.75,
                        metadata={'change_24h': reserve.reserve_change_24h}
                    )
                    anomalies.append(anomaly)
                
                # Sharp inflow anomaly (bearish - selling pressure coming)
                if reserve.trend == ReserveTrend.SHARP_INFLOW:
                    anomaly = ReserveAnomaly(
                        exchange_name=exchange,
                        asset=reserve.asset,
                        anomaly_type='sharp_inflow',
                        timestamp=reserve.timestamp,
                        magnitude=abs(reserve.reserve_change_24h),
                        significance=min(1.0, abs(reserve.reserve_change_24h) / 10),
                        signal='bearish',
                        confidence=0.70,
                        metadata={'change_24h': reserve.reserve_change_24h}
                    )
                    anomalies.append(anomaly)
            
            return anomalies
            
        except Exception as e:
            self.logger.error(f"Error detecting anomalies: {e}")
            return anomalies
    
    def _determine_overall_trend(
        self,
        reserves: Dict[str, ExchangeReserve]
    ) -> ReserveTrend:
        """Determine overall reserve trend across exchanges"""
        try:
            trends = [r.trend for r in reserves.values()]
            
            # Count trend types
            outflow_count = sum(1 for t in trends if 'outflow' in t.value)
            inflow_count = sum(1 for t in trends if 'inflow' in t.value)
            
            if outflow_count > len(trends) * 0.6:
                # Check if sharp
                sharp_count = sum(1 for t in trends if t == ReserveTrend.SHARP_OUTFLOW)
                if sharp_count > 0:
                    return ReserveTrend.SHARP_OUTFLOW
                else:
                    return ReserveTrend.CONSISTENT_OUTFLOW
            elif inflow_count > len(trends) * 0.6:
                sharp_count = sum(1 for t in trends if t == ReserveTrend.SHARP_INFLOW)
                if sharp_count > 0:
                    return ReserveTrend.SHARP_INFLOW
                else:
                    return ReserveTrend.CONSISTENT_INFLOW
            else:
                return ReserveTrend.STABLE
                
        except Exception:
            return ReserveTrend.STABLE
    
    def _calculate_supply_shock_risk(
        self,
        reserves: Dict[str, ExchangeReserve],
        total_reserves: float
    ) -> float:
        """Calculate supply shock risk (0-1)"""
        try:
            # Count exchanges at low levels
            low_count = sum(
                1 for r in reserves.values()
                if r.reserve_level in [ReserveLevel.MULTI_YEAR_LOW, ReserveLevel.LOW]
            )
            
            # Risk increases with more exchanges at low levels
            risk = low_count / len(reserves) if reserves else 0.0
            
            # Boost if consistent outflows
            outflow_count = sum(
                1 for r in reserves.values()
                if 'outflow' in r.trend.value
            )
            
            if outflow_count > len(reserves) * 0.5:
                risk *= 1.2
            
            return min(1.0, risk)
            
        except Exception:
            return 0.0
    
    async def _generate_reserve_signals(
        self,
        reserves: Dict[str, ExchangeReserve],
        anomalies: List[ReserveAnomaly],
        overall_trend: ReserveTrend,
        supply_shock_risk: float
    ) -> List[Dict[str, Any]]:
        """Generate trading signals from reserve analysis"""
        signals = []
        
        try:
            # Supply shock signal
            if supply_shock_risk > 0.7:
                signals.append({
                    'type': 'supply_shock_risk',
                    'direction': 'bullish',
                    'confidence': 0.85,
                    'shock_risk': supply_shock_risk,
                    'reasoning': "High supply shock risk - reserves at multi-year lows",
                    'action': 'potential_supply_squeeze',
                    'priority': 'high'
                })
            
            # Overall trend signal
            if overall_trend == ReserveTrend.SHARP_OUTFLOW:
                signals.append({
                    'type': 'reserve_trend',
                    'direction': 'bullish',
                    'confidence': 0.80,
                    'trend': 'sharp_outflow',
                    'reasoning': "Sharp reserve outflows - bullish accumulation",
                    'priority': 'high'
                })
            elif overall_trend == ReserveTrend.CONSISTENT_OUTFLOW:
                signals.append({
                    'type': 'reserve_trend',
                    'direction': 'bullish',
                    'confidence': 0.75,
                    'trend': 'consistent_outflow',
                    'reasoning': "Consistent reserve outflows - long-term bullish",
                    'priority': 'medium'
                })
            elif overall_trend == ReserveTrend.SHARP_INFLOW:
                signals.append({
                    'type': 'reserve_trend',
                    'direction': 'bearish',
                    'confidence': 0.70,
                    'trend': 'sharp_inflow',
                    'reasoning': "Sharp reserve inflows - potential selling pressure",
                    'priority': 'medium'
                })
            
            # Anomaly signals
            for anomaly in anomalies:
                if anomaly.confidence >= 0.7:
                    signals.append({
                        'type': 'reserve_anomaly',
                        'direction': anomaly.signal,
                        'confidence': anomaly.confidence,
                        'anomaly_type': anomaly.anomaly_type,
                        'exchange': anomaly.exchange_name,
                        'magnitude': anomaly.magnitude,
                        'reasoning': f"{anomaly.anomaly_type} on {anomaly.exchange_name}",
                        'priority': 'high' if anomaly.significance > 0.7 else 'medium'
                    })
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Error generating reserve signals: {e}")
            return signals
    
    def _calculate_confidence(
        self,
        exchange_count: int,
        anomalies: List[ReserveAnomaly]
    ) -> float:
        """Calculate overall confidence"""
        try:
            confidence = 0.5
            
            # More exchanges = higher confidence
            confidence += min(0.3, exchange_count * 0.06)
            
            # Anomalies detected
            if anomalies:
                confidence += min(0.2, len(anomalies) * 0.05)
            
            return min(0.90, confidence)
            
        except Exception:
            return 0.5
    
    def _get_default_analysis(self, asset: str) -> ReservesAnalysis:
        """Get default analysis when data unavailable"""
        return ReservesAnalysis(
            asset=asset,
            timestamp=datetime.now(),
            exchange_reserves={},
            total_exchange_reserves=0.0,
            reserve_dominance={},
            anomalies=[],
            overall_trend=ReserveTrend.STABLE,
            supply_shock_risk=0.0,
            overall_confidence=0.0,
            reserve_signals=[],
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

