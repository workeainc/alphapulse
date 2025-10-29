"""
Liquidation Cascade Predictor for AlphaPulse
Predicts and tracks liquidation cascades and cluster zones
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import asyncio

logger = logging.getLogger(__name__)

class CascadeRisk(Enum):
    """Liquidation cascade risk levels"""
    EXTREME = "extreme"  # >80% probability
    HIGH = "high"  # 60-80%
    MODERATE = "moderate"  # 40-60%
    LOW = "low"  # 20-40%
    MINIMAL = "minimal"  # <20%

class LiquidationSide(Enum):
    """Liquidation side"""
    LONG = "long"  # Long liquidations (price going down)
    SHORT = "short"  # Short liquidations (price going up)
    BOTH = "both"

@dataclass
class LiquidationCluster:
    """Cluster of liquidations at a price level"""
    price_level: float
    liquidation_volume: float
    leverage_estimate: float
    side: LiquidationSide
    cluster_strength: float
    estimated_impact: float  # Expected price impact
    metadata: Dict[str, Any]

@dataclass
class CascadeZone:
    """Zone with high cascade risk"""
    price_start: float
    price_end: float
    total_liquidation_volume: float
    cascade_risk: CascadeRisk
    cascade_probability: float
    estimated_slippage: float
    domino_effect_size: float
    side: LiquidationSide
    metadata: Dict[str, Any]

@dataclass
class LiquidationHeatmap:
    """Liquidation heatmap data"""
    symbol: str
    timestamp: datetime
    current_price: float
    liquidation_clusters: List[LiquidationCluster]
    cascade_zones: List[CascadeZone]
    nearest_cascade_zone: Optional[CascadeZone]
    distance_to_cascade: float
    overall_risk: CascadeRisk
    metadata: Dict[str, Any]

@dataclass
class CascadeImpact:
    """Predicted cascade impact"""
    trigger_price: float
    cascade_side: LiquidationSide
    total_liquidation_volume: float
    estimated_price_impact: float
    cascade_probability: float
    affected_levels: List[float]
    recovery_probability: float
    metadata: Dict[str, Any]

@dataclass
class LiquidationCascadeAnalysis:
    """Complete liquidation cascade analysis"""
    symbol: str
    timestamp: datetime
    liquidation_heatmap: LiquidationHeatmap
    cascade_impacts: List[CascadeImpact]
    current_cascade_risk: CascadeRisk
    overall_confidence: float
    cascade_signals: List[Dict[str, Any]]
    metadata: Dict[str, Any]

class LiquidationCascadePredictor:
    """
    Liquidation Cascade Prediction Engine
    
    Predicts where liquidation cascades will occur by:
    - Building liquidation heatmap (clusters by price)
    - Estimating leverage distribution
    - Calculating domino effect zones
    - Predicting cascade probability
    - Alerting before high-risk zones
    
    Data Sources:
    - Open interest by price level
    - Funding rates (leverage proxy)
    - Long/short ratios
    - Historical liquidation events
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logger
        
        # Configuration
        self.leverage_estimates = self.config.get('leverage_estimates', {
            'low': 5,
            'medium': 10,
            'high': 20,
            'extreme': 50
        })
        self.min_cluster_volume = self.config.get('min_cluster_volume', 1000000)  # $1M
        self.cascade_threshold = self.config.get('cascade_threshold', 0.6)
        
        # Performance tracking
        self.stats = {
            'analyses_performed': 0,
            'cascade_zones_detected': 0,
            'cascade_warnings_issued': 0,
            'last_update': datetime.now()
        }
        
        logger.info("🚀 Liquidation Cascade Predictor initialized")
    
    async def analyze_liquidation_cascade(
        self,
        symbol: str,
        current_price: float,
        open_interest_data: Optional[Dict[float, float]] = None,
        long_short_ratio: Optional[float] = None,
        funding_rate: Optional[float] = None,
        recent_liquidations: Optional[List[Dict[str, Any]]] = None
    ) -> LiquidationCascadeAnalysis:
        """
        Analyze liquidation cascade risk
        
        Args:
            symbol: Trading symbol
            current_price: Current market price
            open_interest_data: Dict of {price_level: open_interest}
            long_short_ratio: Current long/short ratio
            funding_rate: Current funding rate
            recent_liquidations: Recent liquidation events
            
        Returns:
            LiquidationCascadeAnalysis with predictions
        """
        try:
            # Build liquidation heatmap
            heatmap = await self._build_liquidation_heatmap(
                symbol, current_price, open_interest_data, long_short_ratio, funding_rate
            )
            
            # Predict cascade impacts
            cascade_impacts = await self._predict_cascade_impacts(
                current_price, heatmap.liquidation_clusters, heatmap.cascade_zones
            )
            
            # Determine current risk
            current_risk = self._determine_current_risk(heatmap, current_price)
            
            # Generate signals
            cascade_signals = await self._generate_cascade_signals(
                heatmap, cascade_impacts, current_risk
            )
            
            # Calculate confidence
            overall_confidence = self._calculate_confidence(
                heatmap, open_interest_data is not None
            )
            
            # Create analysis
            analysis = LiquidationCascadeAnalysis(
                symbol=symbol,
                timestamp=datetime.now(),
                liquidation_heatmap=heatmap,
                cascade_impacts=cascade_impacts,
                current_cascade_risk=current_risk,
                overall_confidence=overall_confidence,
                cascade_signals=cascade_signals,
                metadata={
                    'analysis_version': '1.0',
                    'has_open_interest_data': open_interest_data is not None,
                    'stats': self.stats
                }
            )
            
            # Update statistics
            self.stats['analyses_performed'] += 1
            self.stats['cascade_zones_detected'] += len(heatmap.cascade_zones)
            if current_risk in [CascadeRisk.EXTREME, CascadeRisk.HIGH]:
                self.stats['cascade_warnings_issued'] += 1
            self.stats['last_update'] = datetime.now()
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"❌ Error analyzing liquidation cascade for {symbol}: {e}")
            return self._get_default_analysis(symbol)
    
    async def _build_liquidation_heatmap(
        self,
        symbol: str,
        current_price: float,
        open_interest_data: Optional[Dict[float, float]],
        long_short_ratio: Optional[float],
        funding_rate: Optional[float]
    ) -> LiquidationHeatmap:
        """Build liquidation heatmap"""
        try:
            # Estimate leverage distribution from funding rate
            if funding_rate:
                avg_leverage = self._estimate_leverage_from_funding(funding_rate)
            else:
                avg_leverage = self.leverage_estimates['medium']
            
            # Create liquidation clusters
            liquidation_clusters = []
            
            if open_interest_data:
                # Use actual OI data
                for price_level, oi in open_interest_data.items():
                    if oi >= self.min_cluster_volume:
                        # Estimate liquidation parameters
                        cluster = self._create_cluster_from_oi(
                            price_level, oi, current_price, avg_leverage, long_short_ratio
                        )
                        liquidation_clusters.append(cluster)
            else:
                # Estimate clusters based on support/resistance and leverage
                estimated_clusters = self._estimate_liquidation_clusters(
                    current_price, avg_leverage, long_short_ratio
                )
                liquidation_clusters.extend(estimated_clusters)
            
            # Identify cascade zones
            cascade_zones = self._identify_cascade_zones(
                liquidation_clusters, current_price
            )
            
            # Find nearest cascade zone
            nearest_zone = self._find_nearest_cascade_zone(cascade_zones, current_price)
            
            # Calculate distance to nearest cascade
            if nearest_zone:
                distance = abs(nearest_zone.price_start - current_price) / current_price
            else:
                distance = 1.0  # Far away
            
            # Determine overall risk
            overall_risk = self._calculate_overall_risk(cascade_zones, distance)
            
            return LiquidationHeatmap(
                symbol=symbol,
                timestamp=datetime.now(),
                current_price=current_price,
                liquidation_clusters=liquidation_clusters,
                cascade_zones=cascade_zones,
                nearest_cascade_zone=nearest_zone,
                distance_to_cascade=distance,
                overall_risk=overall_risk,
                metadata={
                    'avg_leverage': avg_leverage,
                    'cluster_count': len(liquidation_clusters),
                    'cascade_zone_count': len(cascade_zones)
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error building liquidation heatmap: {e}")
            return LiquidationHeatmap(
                symbol=symbol,
                timestamp=datetime.now(),
                current_price=current_price,
                liquidation_clusters=[],
                cascade_zones=[],
                nearest_cascade_zone=None,
                distance_to_cascade=1.0,
                overall_risk=CascadeRisk.MINIMAL,
                metadata={}
            )
    
    def _estimate_leverage_from_funding(self, funding_rate: float) -> float:
        """Estimate average leverage from funding rate"""
        # High funding rate = high leverage
        abs_funding = abs(funding_rate)
        
        if abs_funding > 0.0002:  # Very high
            return self.leverage_estimates['extreme']
        elif abs_funding > 0.0001:
            return self.leverage_estimates['high']
        elif abs_funding > 0.00005:
            return self.leverage_estimates['medium']
        else:
            return self.leverage_estimates['low']
    
    def _create_cluster_from_oi(
        self,
        price_level: float,
        open_interest: float,
        current_price: float,
        avg_leverage: float,
        long_short_ratio: Optional[float]
    ) -> LiquidationCluster:
        """Create liquidation cluster from OI data"""
        # Determine if this is long or short liquidation zone
        if price_level < current_price:
            # Below current price = long liquidation zone
            side = LiquidationSide.LONG
        else:
            # Above current price = short liquidation zone
            side = LiquidationSide.SHORT
        
        # Calculate cluster strength
        distance = abs(price_level - current_price) / current_price
        cluster_strength = open_interest / (1 + distance * 10)
        
        # Estimate price impact
        estimated_impact = (open_interest / 1000000000) * 0.01  # Rough estimate
        
        return LiquidationCluster(
            price_level=price_level,
            liquidation_volume=open_interest,
            leverage_estimate=avg_leverage,
            side=side,
            cluster_strength=cluster_strength,
            estimated_impact=estimated_impact,
            metadata={'oi_source': 'actual'}
        )
    
    def _estimate_liquidation_clusters(
        self,
        current_price: float,
        avg_leverage: float,
        long_short_ratio: Optional[float]
    ) -> List[LiquidationCluster]:
        """Estimate liquidation clusters when no OI data available"""
        clusters = []
        
        try:
            # Estimate liquidation prices based on leverage
            # Long liquidations occur at: entry_price * (1 - 1/leverage)
            # Short liquidations occur at: entry_price * (1 + 1/leverage)
            
            leverage_levels = [5, 10, 20, 50]
            
            for leverage in leverage_levels:
                # Long liquidation estimate (below current price)
                long_liq_price = current_price * (1 - 1/leverage)
                
                # Short liquidation estimate (above current price)
                short_liq_price = current_price * (1 + 1/leverage)
                
                # Estimate volume based on leverage popularity
                volume_estimate = 10000000 / leverage  # Higher leverage = less volume
                
                # Long cluster
                clusters.append(LiquidationCluster(
                    price_level=long_liq_price,
                    liquidation_volume=volume_estimate,
                    leverage_estimate=leverage,
                    side=LiquidationSide.LONG,
                    cluster_strength=0.5,
                    estimated_impact=0.01,
                    metadata={'oi_source': 'estimated'}
                ))
                
                # Short cluster
                clusters.append(LiquidationCluster(
                    price_level=short_liq_price,
                    liquidation_volume=volume_estimate,
                    leverage_estimate=leverage,
                    side=LiquidationSide.SHORT,
                    cluster_strength=0.5,
                    estimated_impact=0.01,
                    metadata={'oi_source': 'estimated'}
                ))
            
            return clusters
            
        except Exception as e:
            self.logger.error(f"Error estimating liquidation clusters: {e}")
            return clusters
    
    def _identify_cascade_zones(
        self,
        clusters: List[LiquidationCluster],
        current_price: float
    ) -> List[CascadeZone]:
        """Identify zones where cascades are likely"""
        cascade_zones = []
        
        try:
            # Group nearby clusters
            sorted_clusters = sorted(clusters, key=lambda x: x.price_level)
            
            # Sliding window to find dense cluster zones
            window_size = 5
            for i in range(len(sorted_clusters) - window_size + 1):
                window = sorted_clusters[i:i+window_size]
                
                # Check if clusters are close together (within 2% price range)
                price_range = window[-1].price_level - window[0].price_level
                avg_price = np.mean([c.price_level for c in window])
                
                if price_range / avg_price < 0.02:  # Tight cluster
                    # This is a cascade zone
                    total_volume = sum(c.liquidation_volume for c in window)
                    
                    # Calculate cascade probability
                    cascade_prob = min(0.95, total_volume / 50000000)  # Based on volume
                    
                    # Determine risk level
                    if cascade_prob > 0.8:
                        risk = CascadeRisk.EXTREME
                    elif cascade_prob > 0.6:
                        risk = CascadeRisk.HIGH
                    elif cascade_prob > 0.4:
                        risk = CascadeRisk.MODERATE
                    elif cascade_prob > 0.2:
                        risk = CascadeRisk.LOW
                    else:
                        risk = CascadeRisk.MINIMAL
                    
                    # Estimate slippage
                    estimated_slippage = (total_volume / 100000000) * 0.02  # 2% per $100M
                    
                    # Domino effect size
                    domino_effect = cascade_prob * estimated_slippage
                    
                    # Determine side
                    long_clusters = sum(1 for c in window if c.side == LiquidationSide.LONG)
                    short_clusters = sum(1 for c in window if c.side == LiquidationSide.SHORT)
                    
                    if long_clusters > short_clusters:
                        side = LiquidationSide.LONG
                    elif short_clusters > long_clusters:
                        side = LiquidationSide.SHORT
                    else:
                        side = LiquidationSide.BOTH
                    
                    zone = CascadeZone(
                        price_start=window[0].price_level,
                        price_end=window[-1].price_level,
                        total_liquidation_volume=total_volume,
                        cascade_risk=risk,
                        cascade_probability=cascade_prob,
                        estimated_slippage=estimated_slippage,
                        domino_effect_size=domino_effect,
                        side=side,
                        metadata={
                            'cluster_count': len(window),
                            'price_range': price_range
                        }
                    )
                    cascade_zones.append(zone)
            
            # Sort by risk (highest first)
            cascade_zones.sort(key=lambda x: x.cascade_probability, reverse=True)
            
            return cascade_zones
            
        except Exception as e:
            self.logger.error(f"Error identifying cascade zones: {e}")
            return cascade_zones
    
    def _find_nearest_cascade_zone(
        self,
        zones: List[CascadeZone],
        current_price: float
    ) -> Optional[CascadeZone]:
        """Find nearest cascade zone to current price"""
        if not zones:
            return None
        
        nearest = None
        min_distance = float('inf')
        
        for zone in zones:
            # Distance to zone
            if current_price < zone.price_start:
                distance = zone.price_start - current_price
            elif current_price > zone.price_end:
                distance = current_price - zone.price_end
            else:
                distance = 0  # Currently in zone
            
            if distance < min_distance:
                min_distance = distance
                nearest = zone
        
        return nearest
    
    def _calculate_overall_risk(
        self,
        zones: List[CascadeZone],
        distance_to_nearest: float
    ) -> CascadeRisk:
        """Calculate overall cascade risk"""
        if not zones:
            return CascadeRisk.MINIMAL
        
        # If very close to high-risk zone
        if distance_to_nearest < 0.01:  # Within 1%
            highest_risk = max((z.cascade_risk for z in zones), default=CascadeRisk.MINIMAL)
            return highest_risk
        elif distance_to_nearest < 0.03:  # Within 3%
            return CascadeRisk.MODERATE
        else:
            return CascadeRisk.LOW
    
    async def _predict_cascade_impacts(
        self,
        current_price: float,
        clusters: List[LiquidationCluster],
        zones: List[CascadeZone]
    ) -> List[CascadeImpact]:
        """Predict cascade impacts if triggered"""
        impacts = []
        
        try:
            # For each high-risk zone, predict impact if triggered
            for zone in zones[:3]:  # Top 3 zones
                if zone.cascade_risk in [CascadeRisk.EXTREME, CascadeRisk.HIGH]:
                    # Affected price levels
                    affected_levels = [
                        c.price_level for c in clusters
                        if zone.price_start <= c.price_level <= zone.price_end
                    ]
                    
                    # Recovery probability (lower for extreme cascades)
                    recovery_prob = 1.0 - zone.cascade_probability
                    
                    impact = CascadeImpact(
                        trigger_price=(zone.price_start + zone.price_end) / 2,
                        cascade_side=zone.side,
                        total_liquidation_volume=zone.total_liquidation_volume,
                        estimated_price_impact=zone.estimated_slippage,
                        cascade_probability=zone.cascade_probability,
                        affected_levels=affected_levels,
                        recovery_probability=recovery_prob,
                        metadata={
                            'zone_risk': zone.cascade_risk.value,
                            'domino_effect': zone.domino_effect_size
                        }
                    )
                    impacts.append(impact)
            
            return impacts
            
        except Exception as e:
            self.logger.error(f"Error predicting cascade impacts: {e}")
            return impacts
    
    def _determine_current_risk(
        self,
        heatmap: LiquidationHeatmap,
        current_price: float
    ) -> CascadeRisk:
        """Determine current cascade risk level"""
        return heatmap.overall_risk
    
    async def _generate_cascade_signals(
        self,
        heatmap: LiquidationHeatmap,
        impacts: List[CascadeImpact],
        current_risk: CascadeRisk
    ) -> List[Dict[str, Any]]:
        """Generate trading signals from cascade analysis"""
        signals = []
        
        try:
            # Risk warning signals
            if current_risk == CascadeRisk.EXTREME:
                signals.append({
                    'type': 'liquidation_cascade_risk',
                    'risk_level': 'extreme',
                    'confidence': 0.90,
                    'distance_to_zone': heatmap.distance_to_cascade,
                    'reasoning': "Extreme liquidation cascade risk nearby",
                    'action': 'reduce_position_size',
                    'priority': 'critical'
                })
            elif current_risk == CascadeRisk.HIGH:
                signals.append({
                    'type': 'liquidation_cascade_risk',
                    'risk_level': 'high',
                    'confidence': 0.80,
                    'distance_to_zone': heatmap.distance_to_cascade,
                    'reasoning': "High liquidation cascade risk nearby",
                    'action': 'use_caution',
                    'priority': 'high'
                })
            
            # Cascade zone approach signals
            if heatmap.nearest_cascade_zone and heatmap.distance_to_cascade < 0.05:  # Within 5%
                zone = heatmap.nearest_cascade_zone
                signals.append({
                    'type': 'approaching_cascade_zone',
                    'direction': 'long' if zone.side == LiquidationSide.SHORT else 'short',
                    'confidence': zone.cascade_probability,
                    'zone_distance': heatmap.distance_to_cascade,
                    'reasoning': f"Approaching {zone.side.value} liquidation cascade zone",
                    'priority': 'high'
                })
            
            # Post-cascade opportunity signals
            for impact in impacts:
                if impact.recovery_probability > 0.7:
                    signals.append({
                        'type': 'post_cascade_opportunity',
                        'direction': 'long' if impact.cascade_side == LiquidationSide.LONG else 'short',
                        'confidence': impact.recovery_probability,
                        'trigger_price': impact.trigger_price,
                        'reasoning': f"Potential counter-trade after {impact.cascade_side.value} cascade",
                        'priority': 'medium'
                    })
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Error generating cascade signals: {e}")
            return signals
    
    def _calculate_confidence(
        self,
        heatmap: LiquidationHeatmap,
        has_real_data: bool
    ) -> float:
        """Calculate analysis confidence"""
        try:
            confidence = 0.5
            
            # Real data bonus
            if has_real_data:
                confidence += 0.2
            
            # Multiple cascade zones detected
            if len(heatmap.cascade_zones) > 0:
                confidence += min(0.2, len(heatmap.cascade_zones) * 0.05)
            
            # High risk nearby
            if heatmap.overall_risk in [CascadeRisk.EXTREME, CascadeRisk.HIGH]:
                confidence += 0.1
            
            return min(0.95, confidence)
            
        except Exception:
            return 0.5
    
    def _get_default_analysis(self, symbol: str) -> LiquidationCascadeAnalysis:
        """Get default analysis when data unavailable"""
        return LiquidationCascadeAnalysis(
            symbol=symbol,
            timestamp=datetime.now(),
            liquidation_heatmap=LiquidationHeatmap(
                symbol=symbol,
                timestamp=datetime.now(),
                current_price=0.0,
                liquidation_clusters=[],
                cascade_zones=[],
                nearest_cascade_zone=None,
                distance_to_cascade=1.0,
                overall_risk=CascadeRisk.MINIMAL,
                metadata={}
            ),
            cascade_impacts=[],
            current_cascade_risk=CascadeRisk.MINIMAL,
            overall_confidence=0.0,
            cascade_signals=[],
            metadata={'error': 'No data available'}
        )
    
    async def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        return {
            'stats': self.stats,
            'config': self.config,
            'last_update': datetime.now().isoformat()
        }

