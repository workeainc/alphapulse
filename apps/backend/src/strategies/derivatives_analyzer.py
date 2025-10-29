"""
Derivatives Analyzer for AlphaPulse
Analyzes perpetual premium, basis spread, and futures metrics
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import asyncio

logger = logging.getLogger(__name__)

class PremiumLevel(Enum):
    """Perpetual premium levels"""
    EXTREME_PREMIUM = "extreme_premium"  # >0.5%
    HIGH_PREMIUM = "high_premium"  # 0.2-0.5%
    NORMAL = "normal"  # -0.1 to 0.2%
    DISCOUNT = "discount"  # -0.3 to -0.1%
    EXTREME_DISCOUNT = "extreme_discount"  # <-0.3%

class MarketStructure(Enum):
    """Futures market structure"""
    CONTANGO = "contango"  # Futures > Spot (normal, bullish)
    BACKWARDATION = "backwardation"  # Futures < Spot (bearish)
    FLAT = "flat"

@dataclass
class PerpetualPremium:
    """Perpetual futures premium/discount"""
    symbol: str
    timestamp: datetime
    spot_price: float
    perpetual_price: float
    premium_pct: float
    premium_level: PremiumLevel
    signal: Optional[str]  # 'bullish' or 'bearish'
    confidence: float
    metadata: Dict[str, Any]

@dataclass
class BasisSpread:
    """Futures basis spread"""
    symbol: str
    timestamp: datetime
    spot_price: float
    futures_price: float
    expiry_date: datetime
    days_to_expiry: int
    basis_pct: float
    annualized_basis: float
    market_structure: MarketStructure
    signal: Optional[str]
    confidence: float
    metadata: Dict[str, Any]

@dataclass
class FundingRateAnalysis:
    """Funding rate analysis"""
    symbol: str
    timestamp: datetime
    current_funding_rate: float
    avg_funding_rate_24h: float
    funding_rate_trend: str  # 'increasing', 'decreasing', 'stable'
    extreme_funding: bool
    signal: Optional[str]
    confidence: float
    metadata: Dict[str, Any]

@dataclass
class DerivativesAnalysis:
    """Complete derivatives analysis"""
    symbol: str
    timestamp: datetime
    perpetual_premium: PerpetualPremium
    basis_spreads: List[BasisSpread]  # Multiple expiries
    funding_rate_analysis: FundingRateAnalysis
    overall_signal: Optional[str]
    overall_confidence: float
    derivative_signals: List[Dict[str, Any]]
    metadata: Dict[str, Any]

class DerivativesAnalyzer:
    """
    Derivatives Analysis Engine
    
    Analyzes:
    - Perpetual futures premium/discount
    - Futures basis spread
    - Funding rates
    - Contango/backwardation
    - Leverage sentiment
    
    Signals:
    - Extreme premium = Overleveraged (bearish)
    - Extreme discount = Fear (bullish)
    - High contango = Bullish sentiment
    - Backwardation = Bearish sentiment
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logger
        
        # Configuration
        self.extreme_premium_threshold = self.config.get('extreme_premium', 0.5)  # 0.5%
        self.extreme_discount_threshold = self.config.get('extreme_discount', -0.3)  # -0.3%
        self.high_basis_threshold = self.config.get('high_basis', 20.0)  # 20% annualized
        
        # Performance tracking
        self.stats = {
            'analyses_performed': 0,
            'extreme_premiums_detected': 0,
            'extreme_discounts_detected': 0,
            'backwardation_detected': 0,
            'last_update': datetime.now()
        }
        
        logger.info("🚀 Derivatives Analyzer initialized")
    
    async def analyze_derivatives(
        self,
        symbol: str,
        spot_price: float,
        perpetual_price: Optional[float] = None,
        futures_prices: Optional[Dict[str, Tuple[float, datetime]]] = None,
        funding_rate: Optional[float] = None
    ) -> DerivativesAnalysis:
        """
        Complete derivatives analysis
        
        Args:
            symbol: Trading symbol
            spot_price: Current spot price
            perpetual_price: Perpetual futures price
            futures_prices: Dict of {contract_name: (price, expiry_date)}
            funding_rate: Current funding rate
            
        Returns:
            DerivativesAnalysis with complete metrics
        """
        try:
            # Analyze perpetual premium
            if perpetual_price:
                perp_premium = self._analyze_perpetual_premium(
                    symbol, spot_price, perpetual_price
                )
            else:
                perp_premium = self._get_default_perpetual(symbol, spot_price)
            
            # Analyze basis spreads
            basis_spreads = []
            if futures_prices:
                for contract_name, (futures_price, expiry) in futures_prices.items():
                    basis = self._calculate_basis_spread(
                        symbol, spot_price, futures_price, expiry, contract_name
                    )
                    basis_spreads.append(basis)
            
            # Analyze funding rate
            if funding_rate is not None:
                funding_analysis = self._analyze_funding_rate(
                    symbol, funding_rate
                )
            else:
                funding_analysis = self._get_default_funding(symbol)
            
            # Determine overall signal
            overall_signal, overall_confidence = self._determine_overall_signal(
                perp_premium, basis_spreads, funding_analysis
            )
            
            # Generate derivative signals
            derivative_signals = await self._generate_signals(
                perp_premium, basis_spreads, funding_analysis
            )
            
            # Create analysis
            analysis = DerivativesAnalysis(
                symbol=symbol,
                timestamp=datetime.now(),
                perpetual_premium=perp_premium,
                basis_spreads=basis_spreads,
                funding_rate_analysis=funding_analysis,
                overall_signal=overall_signal,
                overall_confidence=overall_confidence,
                derivative_signals=derivative_signals,
                metadata={
                    'analysis_version': '1.0',
                    'config': self.config,
                    'stats': self.stats
                }
            )
            
            # Update statistics
            self.stats['analyses_performed'] += 1
            if perp_premium.premium_level == PremiumLevel.EXTREME_PREMIUM:
                self.stats['extreme_premiums_detected'] += 1
            elif perp_premium.premium_level == PremiumLevel.EXTREME_DISCOUNT:
                self.stats['extreme_discounts_detected'] += 1
            
            if any(b.market_structure == MarketStructure.BACKWARDATION for b in basis_spreads):
                self.stats['backwardation_detected'] += 1
            
            self.stats['last_update'] = datetime.now()
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"❌ Error analyzing derivatives for {symbol}: {e}")
            return self._get_default_analysis(symbol)
    
    def _analyze_perpetual_premium(
        self,
        symbol: str,
        spot_price: float,
        perpetual_price: float
    ) -> PerpetualPremium:
        """Analyze perpetual futures premium/discount"""
        try:
            # Calculate premium percentage
            premium_pct = ((perpetual_price - spot_price) / spot_price) * 100
            
            # Determine premium level
            if premium_pct > self.extreme_premium_threshold:
                premium_level = PremiumLevel.EXTREME_PREMIUM
                signal = 'bearish'  # Overleveraged
                confidence = 0.85
            elif premium_pct > 0.2:
                premium_level = PremiumLevel.HIGH_PREMIUM
                signal = 'bearish'
                confidence = 0.70
            elif premium_pct < self.extreme_discount_threshold:
                premium_level = PremiumLevel.EXTREME_DISCOUNT
                signal = 'bullish'  # Extreme fear
                confidence = 0.85
            elif premium_pct < -0.1:
                premium_level = PremiumLevel.DISCOUNT
                signal = 'bullish'
                confidence = 0.70
            else:
                premium_level = PremiumLevel.NORMAL
                signal = None
                confidence = 0.5
            
            return PerpetualPremium(
                symbol=symbol,
                timestamp=datetime.now(),
                spot_price=spot_price,
                perpetual_price=perpetual_price,
                premium_pct=premium_pct,
                premium_level=premium_level,
                signal=signal,
                confidence=confidence,
                metadata={
                    'price_difference': perpetual_price - spot_price,
                    'interpretation': self._interpret_premium(premium_pct)
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error analyzing perpetual premium: {e}")
            return self._get_default_perpetual(symbol, spot_price)
    
    def _calculate_basis_spread(
        self,
        symbol: str,
        spot_price: float,
        futures_price: float,
        expiry_date: datetime,
        contract_name: str
    ) -> BasisSpread:
        """Calculate futures basis spread"""
        try:
            # Calculate basis
            basis_pct = ((futures_price - spot_price) / spot_price) * 100
            
            # Calculate days to expiry
            days_to_expiry = (expiry_date - datetime.now()).days
            
            # Annualize the basis
            if days_to_expiry > 0:
                annualized_basis = basis_pct * (365 / days_to_expiry)
            else:
                annualized_basis = 0.0
            
            # Determine market structure
            if basis_pct > 0.1:
                market_structure = MarketStructure.CONTANGO
                signal = 'bullish' if annualized_basis < self.high_basis_threshold else 'bearish'
                confidence = 0.7
            elif basis_pct < -0.1:
                market_structure = MarketStructure.BACKWARDATION
                signal = 'bearish'
                confidence = 0.8
            else:
                market_structure = MarketStructure.FLAT
                signal = None
                confidence = 0.5
            
            return BasisSpread(
                symbol=symbol,
                timestamp=datetime.now(),
                spot_price=spot_price,
                futures_price=futures_price,
                expiry_date=expiry_date,
                days_to_expiry=days_to_expiry,
                basis_pct=basis_pct,
                annualized_basis=annualized_basis,
                market_structure=market_structure,
                signal=signal,
                confidence=confidence,
                metadata={
                    'contract_name': contract_name,
                    'interpretation': self._interpret_basis(basis_pct, market_structure)
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating basis spread: {e}")
            return BasisSpread(
                symbol=symbol,
                timestamp=datetime.now(),
                spot_price=spot_price,
                futures_price=futures_price,
                expiry_date=expiry_date,
                days_to_expiry=0,
                basis_pct=0.0,
                annualized_basis=0.0,
                market_structure=MarketStructure.FLAT,
                signal=None,
                confidence=0.0,
                metadata={}
            )
    
    def _analyze_funding_rate(
        self,
        symbol: str,
        current_funding_rate: float,
        historical_rates: Optional[List[float]] = None
    ) -> FundingRateAnalysis:
        """Analyze funding rate patterns"""
        try:
            # Calculate average (would use historical data in production)
            avg_funding_24h = current_funding_rate  # Placeholder
            
            # Determine trend
            if current_funding_rate > avg_funding_24h * 1.2:
                trend = 'increasing'
            elif current_funding_rate < avg_funding_24h * 0.8:
                trend = 'decreasing'
            else:
                trend = 'stable'
            
            # Check for extreme funding
            extreme_funding = abs(current_funding_rate) > 0.0001  # 0.01% (very high for 8h rate)
            
            # Determine signal
            signal = None
            confidence = 0.5
            
            if current_funding_rate > 0.0001:  # High positive funding
                signal = 'bearish'  # Longs paying shorts (overleveraged)
                confidence = 0.75
            elif current_funding_rate < -0.0001:  # High negative funding
                signal = 'bullish'  # Shorts paying longs (oversold)
                confidence = 0.75
            
            return FundingRateAnalysis(
                symbol=symbol,
                timestamp=datetime.now(),
                current_funding_rate=current_funding_rate,
                avg_funding_rate_24h=avg_funding_24h,
                funding_rate_trend=trend,
                extreme_funding=extreme_funding,
                signal=signal,
                confidence=confidence,
                metadata={
                    'annualized_funding': current_funding_rate * 365 * 3,  # 3 funding periods per day
                    'interpretation': self._interpret_funding(current_funding_rate)
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error analyzing funding rate: {e}")
            return self._get_default_funding(symbol)
    
    def _interpret_premium(self, premium_pct: float) -> str:
        """Interpret perpetual premium"""
        if premium_pct > 0.5:
            return "Excessive leverage - overleveraged longs (bearish)"
        elif premium_pct > 0.2:
            return "High premium - bullish sentiment but caution"
        elif premium_pct < -0.3:
            return "Extreme discount - fear, potential bottom (bullish)"
        elif premium_pct < -0.1:
            return "Discount - bearish sentiment"
        else:
            return "Normal range - balanced"
    
    def _interpret_basis(self, basis_pct: float, structure: MarketStructure) -> str:
        """Interpret basis spread"""
        if structure == MarketStructure.CONTANGO:
            if basis_pct > 1.0:
                return "Strong contango - bullish expectations"
            else:
                return "Normal contango - neutral to bullish"
        elif structure == MarketStructure.BACKWARDATION:
            return "Backwardation - bearish sentiment or supply shortage"
        else:
            return "Flat curve - neutral"
    
    def _interpret_funding(self, funding_rate: float) -> str:
        """Interpret funding rate"""
        annualized = funding_rate * 365 * 3 * 100  # As percentage
        
        if funding_rate > 0.0001:
            return f"High positive funding ({annualized:.1f}% annualized) - longs overleveraged"
        elif funding_rate < -0.0001:
            return f"High negative funding ({annualized:.1f}% annualized) - shorts overleveraged"
        else:
            return "Normal funding - balanced"
    
    def _determine_overall_signal(
        self,
        perp_premium: PerpetualPremium,
        basis_spreads: List[BasisSpread],
        funding_analysis: FundingRateAnalysis
    ) -> Tuple[Optional[str], float]:
        """Determine overall derivatives signal"""
        try:
            signals_list = []
            confidences = []
            
            # Perpetual premium signal
            if perp_premium.signal:
                signals_list.append(perp_premium.signal)
                confidences.append(perp_premium.confidence)
            
            # Basis spread signal
            for basis in basis_spreads:
                if basis.signal:
                    signals_list.append(basis.signal)
                    confidences.append(basis.confidence)
            
            # Funding rate signal
            if funding_analysis.signal:
                signals_list.append(funding_analysis.signal)
                confidences.append(funding_analysis.confidence)
            
            if not signals_list:
                return None, 0.5
            
            # Count bullish vs bearish
            bullish_count = signals_list.count('bullish')
            bearish_count = signals_list.count('bearish')
            
            if bullish_count > bearish_count:
                overall_signal = 'bullish'
            elif bearish_count > bullish_count:
                overall_signal = 'bearish'
            else:
                overall_signal = None
            
            overall_confidence = np.mean(confidences) if confidences else 0.5
            
            return overall_signal, overall_confidence
            
        except Exception:
            return None, 0.5
    
    async def _generate_signals(
        self,
        perp_premium: PerpetualPremium,
        basis_spreads: List[BasisSpread],
        funding_analysis: FundingRateAnalysis
    ) -> List[Dict[str, Any]]:
        """Generate trading signals from derivatives analysis"""
        signals = []
        
        try:
            # Perpetual premium signals
            if perp_premium.premium_level == PremiumLevel.EXTREME_PREMIUM:
                signals.append({
                    'type': 'perpetual_premium',
                    'direction': 'bearish',
                    'confidence': perp_premium.confidence,
                    'premium_pct': perp_premium.premium_pct,
                    'reasoning': f"Extreme perpetual premium ({perp_premium.premium_pct:.2f}%) - overleveraged",
                    'action': 'reduce_long_exposure',
                    'priority': 'high'
                })
            elif perp_premium.premium_level == PremiumLevel.EXTREME_DISCOUNT:
                signals.append({
                    'type': 'perpetual_premium',
                    'direction': 'bullish',
                    'confidence': perp_premium.confidence,
                    'premium_pct': perp_premium.premium_pct,
                    'reasoning': f"Extreme perpetual discount ({perp_premium.premium_pct:.2f}%) - extreme fear",
                    'action': 'consider_long_entry',
                    'priority': 'high'
                })
            
            # Basis spread signals
            for basis in basis_spreads:
                if basis.market_structure == MarketStructure.BACKWARDATION:
                    signals.append({
                        'type': 'basis_spread',
                        'direction': 'bearish',
                        'confidence': basis.confidence,
                        'basis_pct': basis.basis_pct,
                        'annualized_basis': basis.annualized_basis,
                        'reasoning': f"Backwardation detected - bearish structure",
                        'priority': 'high'
                    })
                elif basis.annualized_basis > self.high_basis_threshold:
                    signals.append({
                        'type': 'basis_spread',
                        'direction': 'bearish',
                        'confidence': 0.75,
                        'annualized_basis': basis.annualized_basis,
                        'reasoning': f"Excessive basis ({basis.annualized_basis:.1f}% ann.) - overheated",
                        'priority': 'medium'
                    })
            
            # Funding rate signals
            if funding_analysis.extreme_funding:
                signals.append({
                    'type': 'funding_rate',
                    'direction': funding_analysis.signal,
                    'confidence': funding_analysis.confidence,
                    'funding_rate': funding_analysis.current_funding_rate,
                    'reasoning': f"Extreme funding rate - {funding_analysis.signal} contrarian",
                    'priority': 'high'
                })
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Error generating derivative signals: {e}")
            return signals
    
    def _get_default_perpetual(self, symbol: str, spot_price: float) -> PerpetualPremium:
        """Get default perpetual premium"""
        return PerpetualPremium(
            symbol=symbol,
            timestamp=datetime.now(),
            spot_price=spot_price,
            perpetual_price=spot_price,
            premium_pct=0.0,
            premium_level=PremiumLevel.NORMAL,
            signal=None,
            confidence=0.0,
            metadata={}
        )
    
    def _get_default_funding(self, symbol: str) -> FundingRateAnalysis:
        """Get default funding rate analysis"""
        return FundingRateAnalysis(
            symbol=symbol,
            timestamp=datetime.now(),
            current_funding_rate=0.0,
            avg_funding_rate_24h=0.0,
            funding_rate_trend='stable',
            extreme_funding=False,
            signal=None,
            confidence=0.0,
            metadata={}
        )
    
    def _get_default_analysis(self, symbol: str) -> DerivativesAnalysis:
        """Get default analysis when data unavailable"""
        return DerivativesAnalysis(
            symbol=symbol,
            timestamp=datetime.now(),
            perpetual_premium=self._get_default_perpetual(symbol, 0.0),
            basis_spreads=[],
            funding_rate_analysis=self._get_default_funding(symbol),
            overall_signal=None,
            overall_confidence=0.0,
            derivative_signals=[],
            metadata={'error': 'No data available'}
        )
    
    async def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        return {
            'stats': self.stats,
            'config': self.config,
            'last_update': datetime.now().isoformat()
        }

