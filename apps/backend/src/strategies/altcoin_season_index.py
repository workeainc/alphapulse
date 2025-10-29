"""
Altcoin Season Index Calculator for AlphaPulse
Tracks when altcoins outperform Bitcoin (alt season detection)
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

class AltSeasonPhase(Enum):
    """Altcoin season phases"""
    BTC_SEASON = "btc_season"  # 0-25: BTC dominance
    MIXED = "mixed"  # 25-75: Mixed market
    ALT_SEASON = "alt_season"  # 75-100: Alt season
    TRANSITION_TO_ALT = "transition_to_alt"  # 60-75: Entering alt season
    TRANSITION_TO_BTC = "transition_to_btc"  # 25-40: Exiting alt season

class SectorType(Enum):
    """Crypto sector types"""
    DEFI = "defi"
    LAYER1 = "layer1"
    LAYER2 = "layer2"
    MEME = "meme"
    GAMING = "gaming"
    AI = "ai"
    INFRASTRUCTURE = "infrastructure"

@dataclass
class AltcoinPerformance:
    """Individual altcoin performance data"""
    symbol: str
    name: str
    market_cap_rank: int
    return_90d: float
    return_30d: float
    return_7d: float
    outperforms_btc: bool
    sector: Optional[SectorType]

@dataclass
class SectorIndex:
    """Sector-specific alt season index"""
    sector: SectorType
    index_value: float  # 0-100
    phase: AltSeasonPhase
    top_performers: List[str]
    bottom_performers: List[str]
    average_return: float

@dataclass
class AltSeasonAnalysis:
    """Complete altcoin season analysis"""
    timestamp: datetime
    index_value: float  # 0-100
    phase: AltSeasonPhase
    btc_return_90d: float
    alt_outperformers: List[AltcoinPerformance]
    alt_underperformers: List[AltcoinPerformance]
    sector_indexes: Dict[SectorType, SectorIndex]
    phase_duration: int  # Days in current phase
    strength_score: float  # How strong is current phase
    transition_probability: float  # Probability of phase change
    signals: List[Dict[str, Any]]
    metadata: Dict[str, Any]

class AltcoinSeasonIndex:
    """
    Altcoin Season Index Calculator
    
    Calculates 0-100 index showing when altcoins outperform Bitcoin:
    - 0-25: Bitcoin Season (BTC outperforming)
    - 25-75: Mixed/Neutral
    - 75-100: Altcoin Season (alts outperforming)
    
    Uses CoinGecko API to track top 50 altcoins vs BTC performance
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logger
        
        # Configuration
        self.top_n_coins = self.config.get('top_n_coins', 50)
        self.lookback_days = self.config.get('lookback_days', 90)
        self.api_base_url = self.config.get(
            'api_base_url',
            'https://api.coingecko.com/api/v3'
        )
        
        # Sector classifications
        self.sector_mapping = {
            'ethereum': SectorType.LAYER1,
            'solana': SectorType.LAYER1,
            'cardano': SectorType.LAYER1,
            'avalanche': SectorType.LAYER1,
            'polygon': SectorType.LAYER2,
            'arbitrum': SectorType.LAYER2,
            'optimism': SectorType.LAYER2,
            'uniswap': SectorType.DEFI,
            'aave': SectorType.DEFI,
            'curve-dao-token': SectorType.DEFI,
            'dogecoin': SectorType.MEME,
            'shiba-inu': SectorType.MEME,
            'pepe': SectorType.MEME,
        }
        
        # Cache
        self.cache = {
            'last_update': None,
            'index_value': None,
            'coin_data': None
        }
        self.cache_ttl = timedelta(hours=1)
        
        # Performance tracking
        self.stats = {
            'calculations': 0,
            'phase_changes': 0,
            'last_phase': None,
            'last_update': datetime.now()
        }
        
        logger.info("🚀 Altcoin Season Index initialized")
    
    async def calculate_index(self) -> AltSeasonAnalysis:
        """
        Calculate current altcoin season index
        
        Returns:
            AltSeasonAnalysis with complete alt season metrics
        """
        try:
            # Check cache
            if self._is_cache_valid():
                self.logger.info("Using cached alt season index")
                return self._get_cached_analysis()
            
            # Fetch coin data
            coin_data = await self._fetch_coin_data()
            
            if not coin_data:
                self.logger.error("No coin data available")
                return self._get_default_analysis()
            
            # Calculate BTC performance
            btc_return_90d = await self._get_btc_return(days=90)
            
            # Analyze altcoin performance
            alt_performances = []
            outperformers = 0
            
            for coin in coin_data:
                performance = self._analyze_coin_performance(
                    coin, btc_return_90d
                )
                alt_performances.append(performance)
                
                if performance.outperforms_btc:
                    outperformers += 1
            
            # Calculate index (0-100)
            index_value = (outperformers / len(coin_data)) * 100
            
            # Determine phase
            phase = self._determine_phase(index_value)
            
            # Track phase changes
            if self.stats['last_phase'] and self.stats['last_phase'] != phase:
                self.stats['phase_changes'] += 1
            self.stats['last_phase'] = phase
            
            # Separate outperformers and underperformers
            alt_outperformers = [p for p in alt_performances if p.outperforms_btc]
            alt_underperformers = [p for p in alt_performances if not p.outperforms_btc]
            
            # Sort by performance
            alt_outperformers.sort(key=lambda x: x.return_90d, reverse=True)
            alt_underperformers.sort(key=lambda x: x.return_90d)
            
            # Calculate sector-specific indexes
            sector_indexes = await self._calculate_sector_indexes(
                alt_performances, btc_return_90d
            )
            
            # Calculate phase strength and transition probability
            strength_score = self._calculate_strength_score(index_value, phase)
            transition_probability = self._calculate_transition_probability(
                index_value, phase
            )
            
            # Generate signals
            signals = await self._generate_signals(
                index_value, phase, sector_indexes, transition_probability
            )
            
            # Create analysis
            analysis = AltSeasonAnalysis(
                timestamp=datetime.now(),
                index_value=index_value,
                phase=phase,
                btc_return_90d=btc_return_90d,
                alt_outperformers=alt_outperformers,
                alt_underperformers=alt_underperformers,
                sector_indexes=sector_indexes,
                phase_duration=0,  # Would need historical tracking
                strength_score=strength_score,
                transition_probability=transition_probability,
                signals=signals,
                metadata={
                    'analysis_version': '1.0',
                    'total_coins_analyzed': len(coin_data),
                    'outperformers_count': outperformers,
                    'underperformers_count': len(coin_data) - outperformers,
                    'data_source': 'coingecko',
                    'stats': self.stats
                }
            )
            
            # Update cache
            self._update_cache(analysis)
            
            # Update statistics
            self.stats['calculations'] += 1
            self.stats['last_update'] = datetime.now()
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating alt season index: {e}")
            return self._get_default_analysis()
    
    async def _fetch_coin_data(self) -> List[Dict[str, Any]]:
        """Fetch top N coins data from CoinGecko"""
        try:
            async with aiohttp.ClientSession() as session:
                # Get top coins by market cap
                url = f"{self.api_base_url}/coins/markets"
                params = {
                    'vs_currency': 'usd',
                    'order': 'market_cap_desc',
                    'per_page': self.top_n_coins + 1,  # +1 for BTC
                    'page': 1,
                    'sparkline': False,
                    'price_change_percentage': '7d,30d,90d'
                }
                
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        # Filter out BTC and stablecoins
                        filtered_data = []
                        for coin in data:
                            if coin['id'] != 'bitcoin' and not self._is_stablecoin(coin['id']):
                                filtered_data.append(coin)
                                
                                if len(filtered_data) >= self.top_n_coins:
                                    break
                        
                        return filtered_data
                    else:
                        self.logger.error(f"CoinGecko API error: {response.status}")
                        return []
                        
        except Exception as e:
            self.logger.error(f"Error fetching coin data: {e}")
            return []
    
    async def _get_btc_return(self, days: int = 90) -> float:
        """Get Bitcoin return for specified period"""
        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self.api_base_url}/coins/bitcoin/market_chart"
                params = {
                    'vs_currency': 'usd',
                    'days': days
                }
                
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        prices = data.get('prices', [])
                        
                        if len(prices) >= 2:
                            start_price = prices[0][1]
                            end_price = prices[-1][1]
                            
                            return_pct = ((end_price - start_price) / start_price) * 100
                            return return_pct
                        
            return 0.0
            
        except Exception as e:
            self.logger.error(f"Error getting BTC return: {e}")
            return 0.0
    
    def _analyze_coin_performance(
        self,
        coin: Dict[str, Any],
        btc_return_90d: float
    ) -> AltcoinPerformance:
        """Analyze individual coin performance"""
        try:
            return_90d = coin.get('price_change_percentage_90d_in_currency', 0)
            return_30d = coin.get('price_change_percentage_30d_in_currency', 0)
            return_7d = coin.get('price_change_percentage_7d_in_currency', 0)
            
            outperforms = return_90d > btc_return_90d
            
            sector = self.sector_mapping.get(coin['id'], None)
            
            return AltcoinPerformance(
                symbol=coin['symbol'].upper(),
                name=coin['name'],
                market_cap_rank=coin.get('market_cap_rank', 999),
                return_90d=return_90d,
                return_30d=return_30d,
                return_7d=return_7d,
                outperforms_btc=outperforms,
                sector=sector
            )
            
        except Exception as e:
            self.logger.error(f"Error analyzing coin {coin.get('id')}: {e}")
            return AltcoinPerformance(
                symbol=coin.get('symbol', 'UNKNOWN').upper(),
                name=coin.get('name', 'Unknown'),
                market_cap_rank=999,
                return_90d=0.0,
                return_30d=0.0,
                return_7d=0.0,
                outperforms_btc=False,
                sector=None
            )
    
    def _determine_phase(self, index_value: float) -> AltSeasonPhase:
        """Determine current alt season phase"""
        if index_value <= 25:
            return AltSeasonPhase.BTC_SEASON
        elif index_value <= 40:
            return AltSeasonPhase.TRANSITION_TO_BTC
        elif index_value <= 60:
            return AltSeasonPhase.MIXED
        elif index_value <= 75:
            return AltSeasonPhase.TRANSITION_TO_ALT
        else:
            return AltSeasonPhase.ALT_SEASON
    
    async def _calculate_sector_indexes(
        self,
        performances: List[AltcoinPerformance],
        btc_return: float
    ) -> Dict[SectorType, SectorIndex]:
        """Calculate sector-specific alt season indexes"""
        sector_indexes = {}
        
        try:
            # Group by sector
            sector_coins = {}
            for perf in performances:
                if perf.sector:
                    if perf.sector not in sector_coins:
                        sector_coins[perf.sector] = []
                    sector_coins[perf.sector].append(perf)
            
            # Calculate index for each sector
            for sector, coins in sector_coins.items():
                if len(coins) < 3:  # Need minimum coins for reliable index
                    continue
                
                outperformers = [c for c in coins if c.outperforms_btc]
                index_value = (len(outperformers) / len(coins)) * 100
                
                # Sort by performance
                coins_sorted = sorted(coins, key=lambda x: x.return_90d, reverse=True)
                top_performers = [c.symbol for c in coins_sorted[:3]]
                bottom_performers = [c.symbol for c in coins_sorted[-3:]]
                
                average_return = np.mean([c.return_90d for c in coins])
                
                sector_index = SectorIndex(
                    sector=sector,
                    index_value=index_value,
                    phase=self._determine_phase(index_value),
                    top_performers=top_performers,
                    bottom_performers=bottom_performers,
                    average_return=average_return
                )
                
                sector_indexes[sector] = sector_index
            
            return sector_indexes
            
        except Exception as e:
            self.logger.error(f"Error calculating sector indexes: {e}")
            return sector_indexes
    
    def _calculate_strength_score(self, index_value: float, phase: AltSeasonPhase) -> float:
        """Calculate how strong the current phase is (0-1)"""
        try:
            if phase == AltSeasonPhase.BTC_SEASON:
                # Stronger when closer to 0
                strength = 1.0 - (index_value / 25)
            elif phase == AltSeasonPhase.ALT_SEASON:
                # Stronger when closer to 100
                strength = (index_value - 75) / 25
            else:
                # Transition or mixed phases are weaker
                strength = 0.5
            
            return max(0.0, min(1.0, strength))
            
        except Exception:
            return 0.5
    
    def _calculate_transition_probability(
        self,
        index_value: float,
        phase: AltSeasonPhase
    ) -> float:
        """Calculate probability of phase transition"""
        try:
            # Higher probability near phase boundaries
            if phase == AltSeasonPhase.TRANSITION_TO_ALT:
                # Near alt season threshold (75)
                distance_to_threshold = abs(index_value - 75)
                probability = 1.0 - (distance_to_threshold / 15)
            elif phase == AltSeasonPhase.TRANSITION_TO_BTC:
                # Near BTC season threshold (25)
                distance_to_threshold = abs(index_value - 25)
                probability = 1.0 - (distance_to_threshold / 15)
            elif phase == AltSeasonPhase.MIXED:
                # In middle, moderate probability
                probability = 0.5
            else:
                # In established phase, lower probability
                probability = 0.3
            
            return max(0.0, min(1.0, probability))
            
        except Exception:
            return 0.5
    
    async def _generate_signals(
        self,
        index_value: float,
        phase: AltSeasonPhase,
        sector_indexes: Dict[SectorType, SectorIndex],
        transition_probability: float
    ) -> List[Dict[str, Any]]:
        """Generate trading signals based on alt season analysis"""
        signals = []
        
        try:
            # Phase-based signals
            if phase == AltSeasonPhase.ALT_SEASON:
                signals.append({
                    'type': 'alt_season',
                    'direction': 'altcoins',
                    'confidence': 0.85,
                    'index_value': index_value,
                    'reasoning': f"Altcoin season active (index: {index_value:.1f})",
                    'action': 'favor_altcoins_over_btc'
                })
            elif phase == AltSeasonPhase.BTC_SEASON:
                signals.append({
                    'type': 'btc_season',
                    'direction': 'bitcoin',
                    'confidence': 0.85,
                    'index_value': index_value,
                    'reasoning': f"Bitcoin season active (index: {index_value:.1f})",
                    'action': 'favor_btc_over_altcoins'
                })
            
            # Transition signals
            if phase == AltSeasonPhase.TRANSITION_TO_ALT and transition_probability > 0.7:
                signals.append({
                    'type': 'phase_transition',
                    'direction': 'entering_alt_season',
                    'confidence': transition_probability,
                    'reasoning': f"Likely entering alt season (transition prob: {transition_probability:.2f})",
                    'action': 'start_accumulating_altcoins'
                })
            elif phase == AltSeasonPhase.TRANSITION_TO_BTC and transition_probability > 0.7:
                signals.append({
                    'type': 'phase_transition',
                    'direction': 'exiting_alt_season',
                    'confidence': transition_probability,
                    'reasoning': f"Likely exiting alt season (transition prob: {transition_probability:.2f})",
                    'action': 'rotate_to_bitcoin'
                })
            
            # Sector rotation signals
            for sector, sector_index in sector_indexes.items():
                if sector_index.phase == AltSeasonPhase.ALT_SEASON and phase != AltSeasonPhase.ALT_SEASON:
                    # Sector outperforming overall market
                    signals.append({
                        'type': 'sector_rotation',
                        'sector': sector.value,
                        'direction': 'bullish',
                        'confidence': 0.75,
                        'sector_index': sector_index.index_value,
                        'reasoning': f"{sector.value} sector outperforming (sector index: {sector_index.index_value:.1f})",
                        'top_coins': sector_index.top_performers
                    })
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Error generating signals: {e}")
            return signals
    
    def _is_stablecoin(self, coin_id: str) -> bool:
        """Check if coin is a stablecoin"""
        stablecoins = [
            'tether', 'usd-coin', 'binance-usd', 'dai',
            'true-usd', 'paxos-standard', 'terrausd', 'frax'
        ]
        return coin_id in stablecoins
    
    def _is_cache_valid(self) -> bool:
        """Check if cache is still valid"""
        if not self.cache['last_update']:
            return False
        
        time_since_update = datetime.now() - self.cache['last_update']
        return time_since_update < self.cache_ttl
    
    def _update_cache(self, analysis: AltSeasonAnalysis):
        """Update analysis cache"""
        self.cache['last_update'] = datetime.now()
        self.cache['index_value'] = analysis.index_value
        self.cache['analysis'] = analysis
    
    def _get_cached_analysis(self) -> AltSeasonAnalysis:
        """Get cached analysis"""
        return self.cache.get('analysis', self._get_default_analysis())
    
    def _get_default_analysis(self) -> AltSeasonAnalysis:
        """Get default analysis when data unavailable"""
        return AltSeasonAnalysis(
            timestamp=datetime.now(),
            index_value=50.0,  # Neutral
            phase=AltSeasonPhase.MIXED,
            btc_return_90d=0.0,
            alt_outperformers=[],
            alt_underperformers=[],
            sector_indexes={},
            phase_duration=0,
            strength_score=0.0,
            transition_probability=0.0,
            signals=[],
            metadata={'error': 'No data available'}
        )
    
    async def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        return {
            'stats': self.stats,
            'config': self.config,
            'cache_status': {
                'is_valid': self._is_cache_valid(),
                'last_update': self.cache['last_update'].isoformat() if self.cache['last_update'] else None
            },
            'last_update': datetime.now().isoformat()
        }

