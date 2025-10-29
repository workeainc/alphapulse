"""
Layer 1 vs Layer 2 Performance Tracker for AlphaPulse
Tracks and compares L1 and L2 blockchain performance for sector rotation
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class LayerType(Enum):
    """Blockchain layer classification"""
    LAYER1 = "layer1"
    LAYER2 = "layer2"

class SectorPhase(Enum):
    """Sector rotation phase"""
    L1_DOMINANCE = "l1_dominance"
    L2_DOMINANCE = "l2_dominance"
    ROTATION_TO_L2 = "rotation_to_l2"
    ROTATION_TO_L1 = "rotation_to_l1"
    BALANCED = "balanced"

@dataclass
class LayerToken:
    """Token with layer classification"""
    symbol: str
    name: str
    layer: LayerType
    market_cap: float
    return_7d: float
    return_30d: float
    return_90d: float
    volume_24h: float
    tvl: Optional[float]

@dataclass
class LayerPerformance:
    """Performance metrics for a layer"""
    layer: LayerType
    token_count: int
    avg_return_7d: float
    avg_return_30d: float
    avg_return_90d: float
    outperformers: List[str]
    underperformers: List[str]
    total_market_cap: float
    total_volume_24h: float
    total_tvl: float
    strength_score: float

@dataclass
class L1L2Analysis:
    """Complete L1 vs L2 analysis"""
    timestamp: datetime
    l1_performance: LayerPerformance
    l2_performance: LayerPerformance
    l1_vs_l2_ratio: float  # L1 return / L2 return
    sector_phase: SectorPhase
    rotation_strength: float
    rotation_probability: float
    overall_confidence: float
    sector_signals: List[Dict[str, Any]]
    metadata: Dict[str, Any]

class LayerPerformanceTracker:
    """
    Layer 1 vs Layer 2 Performance Tracker
    
    Tracks comparative performance between L1 and L2 blockchains
    to identify sector rotation opportunities.
    
    L1 Tokens: ETH, SOL, AVAX, DOT, ATOM, ADA, ALGO, FTM, NEAR
    L2 Tokens: MATIC, ARB, OP, IMX, METIS, LRC, STRK
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logger
        
        # Token classifications
        self.l1_tokens = {
            'ETH': 'Ethereum',
            'SOL': 'Solana',
            'AVAX': 'Avalanche',
            'DOT': 'Polkadot',
            'ATOM': 'Cosmos',
            'ADA': 'Cardano',
            'ALGO': 'Algorand',
            'FTM': 'Fantom',
            'NEAR': 'Near'
        }
        
        self.l2_tokens = {
            'MATIC': 'Polygon',
            'ARB': 'Arbitrum',
            'OP': 'Optimism',
            'IMX': 'Immutable X',
            'METIS': 'Metis',
            'LRC': 'Loopring'
        }
        
        # Performance tracking
        self.stats = {
            'analyses_performed': 0,
            'rotation_signals_issued': 0,
            'last_phase': None,
            'last_update': datetime.now()
        }
        
        logger.info("🚀 L1/L2 Performance Tracker initialized")
    
    async def analyze_layer_performance(self) -> L1L2Analysis:
        """Analyze L1 vs L2 performance"""
        try:
            # Fetch performance data for L1 tokens
            l1_data = await self._fetch_layer_data(self.l1_tokens, LayerType.LAYER1)
            
            # Fetch performance data for L2 tokens
            l2_data = await self._fetch_layer_data(self.l2_tokens, LayerType.LAYER2)
            
            # Calculate layer performance metrics
            l1_performance = self._calculate_layer_performance(l1_data, LayerType.LAYER1)
            l2_performance = self._calculate_layer_performance(l2_data, LayerType.LAYER2)
            
            # Calculate L1/L2 ratio
            if l2_performance.avg_return_30d != 0:
                l1_vs_l2_ratio = l1_performance.avg_return_30d / l2_performance.avg_return_30d
            else:
                l1_vs_l2_ratio = 1.0
            
            # Determine sector phase
            sector_phase = self._determine_sector_phase(l1_performance, l2_performance)
            
            # Calculate rotation metrics
            rotation_strength = abs(l1_performance.avg_return_30d - l2_performance.avg_return_30d) / 100
            rotation_probability = self._calculate_rotation_probability(
                l1_vs_l2_ratio, sector_phase
            )
            
            # Generate signals
            sector_signals = await self._generate_sector_signals(
                l1_performance, l2_performance, sector_phase, rotation_probability
            )
            
            # Calculate confidence
            overall_confidence = self._calculate_confidence(
                len(l1_data), len(l2_data), rotation_strength
            )
            
            # Create analysis
            analysis = L1L2Analysis(
                timestamp=datetime.now(),
                l1_performance=l1_performance,
                l2_performance=l2_performance,
                l1_vs_l2_ratio=l1_vs_l2_ratio,
                sector_phase=sector_phase,
                rotation_strength=rotation_strength,
                rotation_probability=rotation_probability,
                overall_confidence=overall_confidence,
                sector_signals=sector_signals,
                metadata={
                    'analysis_version': '1.0',
                    'l1_count': len(l1_data),
                    'l2_count': len(l2_data),
                    'stats': self.stats
                }
            )
            
            # Update statistics
            self.stats['analyses_performed'] += 1
            if self.stats['last_phase'] and self.stats['last_phase'] != sector_phase:
                self.stats['rotation_signals_issued'] += 1
            self.stats['last_phase'] = sector_phase
            self.stats['last_update'] = datetime.now()
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"❌ Error analyzing L1/L2 performance: {e}")
            return self._get_default_analysis()
    
    async def _fetch_layer_data(
        self,
        tokens: Dict[str, str],
        layer: LayerType
    ) -> List[LayerToken]:
        """Fetch performance data for layer tokens"""
        # Placeholder - would fetch from CoinGecko or similar
        layer_data = []
        
        for symbol, name in tokens.items():
            token = LayerToken(
                symbol=symbol,
                name=name,
                layer=layer,
                market_cap=0.0,
                return_7d=0.0,
                return_30d=0.0,
                return_90d=0.0,
                volume_24h=0.0,
                tvl=None
            )
            layer_data.append(token)
        
        return layer_data
    
    def _calculate_layer_performance(
        self,
        tokens: List[LayerToken],
        layer: LayerType
    ) -> LayerPerformance:
        """Calculate aggregate performance for a layer"""
        try:
            if not tokens:
                return LayerPerformance(
                    layer=layer,
                    token_count=0,
                    avg_return_7d=0.0,
                    avg_return_30d=0.0,
                    avg_return_90d=0.0,
                    outperformers=[],
                    underperformers=[],
                    total_market_cap=0.0,
                    total_volume_24h=0.0,
                    total_tvl=0.0,
                    strength_score=0.0
                )
            
            avg_return_7d = np.mean([t.return_7d for t in tokens])
            avg_return_30d = np.mean([t.return_30d for t in tokens])
            avg_return_90d = np.mean([t.return_90d for t in tokens])
            
            # Top and bottom performers
            sorted_tokens = sorted(tokens, key=lambda x: x.return_30d, reverse=True)
            outperformers = [t.symbol for t in sorted_tokens[:3]]
            underperformers = [t.symbol for t in sorted_tokens[-3:]]
            
            total_market_cap = sum(t.market_cap for t in tokens)
            total_volume_24h = sum(t.volume_24h for t in tokens)
            total_tvl = sum(t.tvl or 0 for t in tokens)
            
            # Strength score
            strength_score = min(1.0, abs(avg_return_30d) / 50)
            
            return LayerPerformance(
                layer=layer,
                token_count=len(tokens),
                avg_return_7d=avg_return_7d,
                avg_return_30d=avg_return_30d,
                avg_return_90d=avg_return_90d,
                outperformers=outperformers,
                underperformers=underperformers,
                total_market_cap=total_market_cap,
                total_volume_24h=total_volume_24h,
                total_tvl=total_tvl,
                strength_score=strength_score
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating layer performance: {e}")
            return LayerPerformance(
                layer=layer,
                token_count=0,
                avg_return_7d=0.0,
                avg_return_30d=0.0,
                avg_return_90d=0.0,
                outperformers=[],
                underperformers=[],
                total_market_cap=0.0,
                total_volume_24h=0.0,
                total_tvl=0.0,
                strength_score=0.0
            )
    
    def _determine_sector_phase(
        self,
        l1_perf: LayerPerformance,
        l2_perf: LayerPerformance
    ) -> SectorPhase:
        """Determine current sector rotation phase"""
        diff = l1_perf.avg_return_30d - l2_perf.avg_return_30d
        
        if diff > 10:
            return SectorPhase.L1_DOMINANCE
        elif diff > 3:
            return SectorPhase.ROTATION_TO_L1
        elif diff < -10:
            return SectorPhase.L2_DOMINANCE
        elif diff < -3:
            return SectorPhase.ROTATION_TO_L2
        else:
            return SectorPhase.BALANCED
    
    def _calculate_rotation_probability(
        self,
        l1_vs_l2_ratio: float,
        phase: SectorPhase
    ) -> float:
        """Calculate probability of sector rotation"""
        if phase in [SectorPhase.ROTATION_TO_L1, SectorPhase.ROTATION_TO_L2]:
            return 0.7
        elif phase in [SectorPhase.L1_DOMINANCE, SectorPhase.L2_DOMINANCE]:
            return 0.4  # Might rotate back
        else:
            return 0.5
    
    async def _generate_sector_signals(
        self,
        l1_perf: LayerPerformance,
        l2_perf: LayerPerformance,
        phase: SectorPhase,
        rotation_prob: float
    ) -> List[Dict[str, Any]]:
        """Generate sector rotation signals"""
        signals = []
        
        try:
            if phase == SectorPhase.L1_DOMINANCE:
                signals.append({
                    'type': 'sector_dominance',
                    'sector': 'layer1',
                    'confidence': 0.75,
                    'reasoning': f"L1 outperforming L2 ({l1_perf.avg_return_30d:.1f}% vs {l2_perf.avg_return_30d:.1f}%)",
                    'top_performers': l1_perf.outperformers
                })
            elif phase == SectorPhase.L2_DOMINANCE:
                signals.append({
                    'type': 'sector_dominance',
                    'sector': 'layer2',
                    'confidence': 0.75,
                    'reasoning': f"L2 outperforming L1 ({l2_perf.avg_return_30d:.1f}% vs {l1_perf.avg_return_30d:.1f}%)",
                    'top_performers': l2_perf.outperformers
                })
            elif phase == SectorPhase.ROTATION_TO_L2 and rotation_prob > 0.6:
                signals.append({
                    'type': 'sector_rotation',
                    'direction': 'to_layer2',
                    'confidence': rotation_prob,
                    'reasoning': "Sector rotation toward L2 tokens",
                    'action': 'consider_l2_positions'
                })
            elif phase == SectorPhase.ROTATION_TO_L1 and rotation_prob > 0.6:
                signals.append({
                    'type': 'sector_rotation',
                    'direction': 'to_layer1',
                    'confidence': rotation_prob,
                    'reasoning': "Sector rotation toward L1 tokens",
                    'action': 'consider_l1_positions'
                })
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Error generating sector signals: {e}")
            return signals
    
    def _calculate_confidence(
        self,
        l1_count: int,
        l2_count: int,
        rotation_strength: float
    ) -> float:
        """Calculate overall confidence"""
        try:
            confidence = 0.5
            
            if l1_count >= 5 and l2_count >= 3:
                confidence += 0.2
            
            if rotation_strength > 0.1:
                confidence += min(0.3, rotation_strength * 2)
            
            return min(0.85, confidence)
            
        except Exception:
            return 0.5
    
    def _get_default_analysis(self) -> L1L2Analysis:
        """Get default analysis"""
        return L1L2Analysis(
            timestamp=datetime.now(),
            l1_performance=LayerPerformance(
                layer=LayerType.LAYER1,
                token_count=0,
                avg_return_7d=0.0,
                avg_return_30d=0.0,
                avg_return_90d=0.0,
                outperformers=[],
                underperformers=[],
                total_market_cap=0.0,
                total_volume_24h=0.0,
                total_tvl=0.0,
                strength_score=0.0
            ),
            l2_performance=LayerPerformance(
                layer=LayerType.LAYER2,
                token_count=0,
                avg_return_7d=0.0,
                avg_return_30d=0.0,
                avg_return_90d=0.0,
                outperformers=[],
                underperformers=[],
                total_market_cap=0.0,
                total_volume_24h=0.0,
                total_tvl=0.0,
                strength_score=0.0
            ),
            l1_vs_l2_ratio=1.0,
            sector_phase=SectorPhase.BALANCED,
            rotation_strength=0.0,
            rotation_probability=0.0,
            overall_confidence=0.0,
            sector_signals=[],
            metadata={'error': 'No data available'}
        )
    
    async def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        return {
            'stats': self.stats,
            'l1_tokens': list(self.l1_tokens.keys()),
            'l2_tokens': list(self.l2_tokens.keys()),
            'last_update': datetime.now().isoformat()
        }

