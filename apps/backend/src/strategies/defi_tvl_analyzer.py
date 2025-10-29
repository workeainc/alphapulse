"""
DeFi TVL (Total Value Locked) Analyzer for AlphaPulse
Analyzes DeFi TVL correlations with token prices
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

class TVLTrend(Enum):
    """TVL trend direction"""
    STRONG_GROWTH = "strong_growth"  # >10% growth
    MODERATE_GROWTH = "moderate_growth"  # 3-10%
    STABLE = "stable"  # -3% to 3%
    MODERATE_DECLINE = "moderate_decline"  # -10% to -3%
    STRONG_DECLINE = "strong_decline"  # <-10%

@dataclass
class ProtocolTVL:
    """TVL data for a single protocol"""
    protocol_name: str
    chain: str
    category: str
    current_tvl: float
    tvl_24h_change: float
    tvl_7d_change: float
    tvl_30d_change: float
    trend: TVLTrend
    rank: int
    metadata: Dict[str, Any]

@dataclass
class ChainTVL:
    """TVL data for a blockchain"""
    chain_name: str
    total_tvl: float
    tvl_change_24h: float
    tvl_change_7d: float
    dominant_protocols: List[str]
    trend: TVLTrend
    market_share: float
    metadata: Dict[str, Any]

@dataclass
class TVLPriceCorrelation:
    """TVL-Price correlation analysis"""
    asset: str
    correlation_7d: float
    correlation_30d: float
    correlation_90d: float
    lead_lag_relationship: str  # 'tvl_leads', 'price_leads', 'simultaneous'
    correlation_strength: str  # 'strong', 'moderate', 'weak'
    confidence: float

@dataclass
class DeFiTVLAnalysis:
    """Complete DeFi TVL analysis"""
    timestamp: datetime
    total_defi_tvl: float
    chain_tvl: Dict[str, ChainTVL]
    protocol_tvl: List[ProtocolTVL]
    tvl_correlations: Dict[str, TVLPriceCorrelation]
    overall_defi_health: float  # 0-1
    overall_confidence: float
    tvl_signals: List[Dict[str, Any]]
    metadata: Dict[str, Any]

class DeFiTVLAnalyzer:
    """
    DeFi Total Value Locked (TVL) Analyzer
    
    Analyzes:
    - Protocol-level TVL
    - Chain-level TVL
    - TVL-price correlations
    - DeFi sector health
    - TVL trends and patterns
    
    Data Source: DeFiLlama API
    
    Signals:
    - Rising TVL + Price correlation = Bullish
    - Declining TVL + Price correlation = Bearish
    - TVL growth divergence = Leading indicator
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logger
        
        # Configuration
        self.api_base_url = 'https://api.llama.fi'
        self.top_protocols_count = self.config.get('top_protocols', 20)
        
        # Cache
        self.cache: Dict[str, Tuple[datetime, Any]] = {}
        self.cache_ttl = timedelta(hours=2)  # TVL doesn't change rapidly
        
        # Performance tracking
        self.stats = {
            'api_calls': 0,
            'cache_hits': 0,
            'analyses_performed': 0,
            'last_update': datetime.now()
        }
        
        logger.info("🚀 DeFi TVL Analyzer initialized")
    
    async def analyze_defi_tvl(
        self,
        asset: Optional[str] = None
    ) -> DeFiTVLAnalysis:
        """
        Analyze DeFi TVL metrics
        
        Args:
            asset: Optional specific asset to analyze
            
        Returns:
            DeFiTVLAnalysis with complete TVL metrics
        """
        try:
            # Check cache
            cache_key = f"tvl_analysis_{asset}"
            if cache_key in self.cache:
                cache_time, cached_data = self.cache[cache_key]
                if datetime.now() - cache_time < self.cache_ttl:
                    self.stats['cache_hits'] += 1
                    return cached_data
            
            # Fetch total TVL
            total_tvl = await self._fetch_total_tvl()
            
            # Fetch chain TVL
            chain_tvl = await self._fetch_chain_tvl()
            
            # Fetch protocol TVL
            protocol_tvl = await self._fetch_protocol_tvl()
            
            # Calculate TVL-price correlations
            tvl_correlations = {}
            if asset:
                correlation = await self._calculate_tvl_price_correlation(asset)
                if correlation:
                    tvl_correlations[asset] = correlation
            
            # Calculate DeFi health score
            defi_health = self._calculate_defi_health(total_tvl, chain_tvl, protocol_tvl)
            
            # Generate signals
            tvl_signals = await self._generate_tvl_signals(
                total_tvl, chain_tvl, protocol_tvl, tvl_correlations
            )
            
            # Calculate confidence
            overall_confidence = self._calculate_confidence(
                total_tvl, len(chain_tvl), len(protocol_tvl)
            )
            
            # Create analysis
            analysis = DeFiTVLAnalysis(
                timestamp=datetime.now(),
                total_defi_tvl=total_tvl,
                chain_tvl=chain_tvl,
                protocol_tvl=protocol_tvl,
                tvl_correlations=tvl_correlations,
                overall_defi_health=defi_health,
                overall_confidence=overall_confidence,
                tvl_signals=tvl_signals,
                metadata={
                    'analysis_version': '1.0',
                    'data_source': 'defillama',
                    'stats': self.stats
                }
            )
            
            # Update cache
            self.cache[cache_key] = (datetime.now(), analysis)
            
            # Update statistics
            self.stats['analyses_performed'] += 1
            self.stats['last_update'] = datetime.now()
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"❌ Error analyzing DeFi TVL: {e}")
            return self._get_default_analysis()
    
    async def _fetch_total_tvl(self) -> float:
        """Fetch total DeFi TVL"""
        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self.api_base_url}/charts"
                
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        if data:
                            # Get most recent TVL
                            latest = data[-1]
                            self.stats['api_calls'] += 1
                            return float(latest.get('totalLiquidityUSD', 0))
                            
        except Exception as e:
            self.logger.error(f"Error fetching total TVL: {e}")
        
        return 0.0
    
    async def _fetch_chain_tvl(self) -> Dict[str, ChainTVL]:
        """Fetch chain-level TVL data"""
        chain_data = {}
        
        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self.api_base_url}/chains"
                
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        total_tvl = sum(chain.get('tvl', 0) for chain in data)
                        
                        for chain in data[:15]:  # Top 15 chains
                            chain_name = chain.get('name', 'Unknown')
                            tvl = float(chain.get('tvl', 0))
                            change_24h = float(chain.get('change_1d', 0))
                            change_7d = float(chain.get('change_7d', 0))
                            
                            trend = self._determine_tvl_trend(change_7d)
                            market_share = (tvl / total_tvl * 100) if total_tvl > 0 else 0
                            
                            chain_tvl = ChainTVL(
                                chain_name=chain_name,
                                total_tvl=tvl,
                                tvl_change_24h=change_24h,
                                tvl_change_7d=change_7d,
                                dominant_protocols=[],  # Would need separate call
                                trend=trend,
                                market_share=market_share,
                                metadata={'gecko_id': chain.get('gecko_id')}
                            )
                            
                            chain_data[chain_name] = chain_tvl
                        
                        self.stats['api_calls'] += 1
                        
        except Exception as e:
            self.logger.error(f"Error fetching chain TVL: {e}")
        
        return chain_data
    
    async def _fetch_protocol_tvl(self) -> List[ProtocolTVL]:
        """Fetch protocol-level TVL data"""
        protocols = []
        
        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self.api_base_url}/protocols"
                
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        for i, protocol in enumerate(data[:self.top_protocols_count]):
                            protocol_tvl = ProtocolTVL(
                                protocol_name=protocol.get('name', 'Unknown'),
                                chain=protocol.get('chain', 'Unknown'),
                                category=protocol.get('category', 'Unknown'),
                                current_tvl=float(protocol.get('tvl', 0)),
                                tvl_24h_change=float(protocol.get('change_1d', 0)),
                                tvl_7d_change=float(protocol.get('change_7d', 0)),
                                tvl_30d_change=float(protocol.get('change_1m', 0)),
                                trend=self._determine_tvl_trend(float(protocol.get('change_7d', 0))),
                                rank=i + 1,
                                metadata={'slug': protocol.get('slug')}
                            )
                            protocols.append(protocol_tvl)
                        
                        self.stats['api_calls'] += 1
                        
        except Exception as e:
            self.logger.error(f"Error fetching protocol TVL: {e}")
        
        return protocols
    
    async def _calculate_tvl_price_correlation(
        self,
        asset: str
    ) -> Optional[TVLPriceCorrelation]:
        """Calculate TVL-price correlation for an asset"""
        # This would require historical TVL and price data
        # Placeholder implementation
        return TVLPriceCorrelation(
            asset=asset,
            correlation_7d=0.0,
            correlation_30d=0.0,
            correlation_90d=0.0,
            lead_lag_relationship='simultaneous',
            correlation_strength='weak',
            confidence=0.5
        )
    
    def _determine_tvl_trend(self, change_7d: float) -> TVLTrend:
        """Determine TVL trend from 7-day change"""
        if change_7d > 10:
            return TVLTrend.STRONG_GROWTH
        elif change_7d > 3:
            return TVLTrend.MODERATE_GROWTH
        elif change_7d > -3:
            return TVLTrend.STABLE
        elif change_7d > -10:
            return TVLTrend.MODERATE_DECLINE
        else:
            return TVLTrend.STRONG_DECLINE
    
    def _calculate_defi_health(
        self,
        total_tvl: float,
        chain_tvl: Dict[str, ChainTVL],
        protocol_tvl: List[ProtocolTVL]
    ) -> float:
        """Calculate overall DeFi ecosystem health (0-1)"""
        try:
            health = 0.5
            
            # Growing chains
            if chain_tvl:
                growing_chains = sum(
                    1 for c in chain_tvl.values()
                    if c.trend in [TVLTrend.STRONG_GROWTH, TVLTrend.MODERATE_GROWTH]
                )
                health += (growing_chains / len(chain_tvl)) * 0.3
            
            # Growing protocols
            if protocol_tvl:
                growing_protocols = sum(
                    1 for p in protocol_tvl
                    if p.trend in [TVLTrend.STRONG_GROWTH, TVLTrend.MODERATE_GROWTH]
                )
                health += (growing_protocols / len(protocol_tvl)) * 0.2
            
            return min(1.0, health)
            
        except Exception:
            return 0.5
    
    async def _generate_tvl_signals(
        self,
        total_tvl: float,
        chain_tvl: Dict[str, ChainTVL],
        protocol_tvl: List[ProtocolTVL],
        correlations: Dict[str, TVLPriceCorrelation]
    ) -> List[Dict[str, Any]]:
        """Generate trading signals from TVL analysis"""
        signals = []
        
        try:
            # Chain growth signals
            for chain_name, chain in chain_tvl.items():
                if chain.trend == TVLTrend.STRONG_GROWTH:
                    signals.append({
                        'type': 'chain_tvl_growth',
                        'chain': chain_name,
                        'direction': 'bullish',
                        'confidence': 0.70,
                        'tvl_change': chain.tvl_change_7d,
                        'reasoning': f"{chain_name} TVL strong growth ({chain.tvl_change_7d:.1f}%)",
                        'priority': 'medium'
                    })
            
            # Protocol growth signals
            top_growing = sorted(protocol_tvl, key=lambda x: x.tvl_7d_change, reverse=True)[:3]
            for protocol in top_growing:
                if protocol.trend == TVLTrend.STRONG_GROWTH:
                    signals.append({
                        'type': 'protocol_tvl_growth',
                        'protocol': protocol.protocol_name,
                        'direction': 'bullish',
                        'confidence': 0.65,
                        'tvl_change': protocol.tvl_7d_change,
                        'reasoning': f"{protocol.protocol_name} TVL surging",
                        'priority': 'low'
                    })
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Error generating TVL signals: {e}")
            return signals
    
    def _calculate_confidence(
        self,
        total_tvl: float,
        chain_count: int,
        protocol_count: int
    ) -> float:
        """Calculate overall confidence"""
        try:
            confidence = 0.5
            
            if total_tvl > 0:
                confidence += 0.2
            if chain_count > 5:
                confidence += 0.15
            if protocol_count > 10:
                confidence += 0.15
            
            return min(0.80, confidence)
            
        except Exception:
            return 0.5
    
    def _get_default_analysis(self) -> DeFiTVLAnalysis:
        """Get default analysis when data unavailable"""
        return DeFiTVLAnalysis(
            timestamp=datetime.now(),
            total_defi_tvl=0.0,
            chain_tvl={},
            protocol_tvl=[],
            tvl_correlations={},
            overall_defi_health=0.5,
            overall_confidence=0.0,
            tvl_signals=[],
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

