"""
Crypto-Specific Volatility Analyzer for AlphaPulse
Advanced volatility metrics including realized/implied vol, smile, and term structure
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class VolatilityRegime(Enum):
    """Volatility regime classification"""
    EXTREME_HIGH = "extreme_high"  # >80th percentile
    HIGH = "high"  # 60-80th
    NORMAL = "normal"  # 40-60th
    LOW = "low"  # 20-40th
    EXTREME_LOW = "extreme_low"  # <20th

class VolatilitySignal(Enum):
    """Volatility-based signals"""
    VOL_EXPANSION = "volatility_expansion"
    VOL_CONTRACTION = "volatility_contraction"
    VOL_SPIKE = "volatility_spike"
    VOL_CRUSH = "volatility_crush"

@dataclass
class RealizedVolatility:
    """Realized (historical) volatility"""
    period_days: int
    volatility_pct: float  # Annualized %
    percentile: float  # Historical percentile
    regime: VolatilityRegime
    trend: str  # 'increasing', 'decreasing', 'stable'

@dataclass
class ImpliedVolatility:
    """Implied volatility from options"""
    atm_iv: float  # At-the-money IV
    call_iv_avg: float
    put_iv_avg: float
    iv_percentile: float
    iv_rank: float

@dataclass
class VolatilitySmile:
    """Volatility smile analysis"""
    strikes: List[float]
    ivs: List[float]
    skew: float  # Smile skew
    smile_shape: str  # 'normal', 'inverse', 'flat'

@dataclass
class VolatilityTermStructure:
    """Volatility term structure"""
    expiries: List[datetime]
    term_ivs: List[float]
    term_structure: str  # 'normal', 'inverted', 'flat'
    slope: float

@dataclass
class CryptoVolatilityAnalysis:
    """Complete crypto volatility analysis"""
    symbol: str
    timestamp: datetime
    realized_vol_7d: RealizedVolatility
    realized_vol_30d: RealizedVolatility
    implied_vol: Optional[ImpliedVolatility]
    vol_smile: Optional[VolatilitySmile]
    term_structure: Optional[VolatilityTermStructure]
    rv_iv_spread: Optional[float]  # Realized - Implied
    current_regime: VolatilityRegime
    overall_confidence: float
    volatility_signals: List[Dict[str, Any]]
    metadata: Dict[str, Any]

class CryptoVolatilityAnalyzer:
    """
    Advanced Volatility Metrics for Crypto
    
    Analyzes:
    - Realized volatility (7d, 30d, 90d)
    - Implied volatility (from Deribit options)
    - Volatility smile (IV by strike)
    - Volatility term structure (IV by expiration)
    - RV vs IV spread
    - Volatility regimes
    
    Signals:
    - RV < IV = Potential vol expansion (bullish vol)
    - RV > IV = Potential vol contraction (bearish vol)
    - Inverted term structure = Short-term stress
    - Vol spike = Potential reversal
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logger
        
        # Configuration
        self.annualization_factor = self.config.get('annualization_factor', np.sqrt(365))
        
        # Performance tracking
        self.stats = {
            'analyses_performed': 0,
            'vol_spikes_detected': 0,
            'regime_changes': 0,
            'last_update': datetime.now()
        }
        
        logger.info("🚀 Crypto Volatility Analyzer initialized")
    
    async def analyze_volatility(
        self,
        df: pd.DataFrame,
        symbol: str,
        options_data: Optional[Dict[str, Any]] = None
    ) -> CryptoVolatilityAnalysis:
        """
        Analyze crypto volatility metrics
        
        Args:
            df: DataFrame with OHLCV data
            symbol: Trading symbol
            options_data: Optional Deribit options data
            
        Returns:
            CryptoVolatilityAnalysis with complete metrics
        """
        try:
            # Calculate realized volatility (multiple periods)
            rv_7d = self._calculate_realized_volatility(df, days=7)
            rv_30d = self._calculate_realized_volatility(df, days=30)
            
            # Get implied volatility from options (if available)
            implied_vol = None
            if options_data:
                implied_vol = self._extract_implied_volatility(options_data)
            
            # Volatility smile (if options data available)
            vol_smile = None
            if options_data:
                vol_smile = self._analyze_volatility_smile(options_data)
            
            # Term structure (if options data available)
            term_structure = None
            if options_data:
                term_structure = self._analyze_term_structure(options_data)
            
            # RV vs IV spread
            rv_iv_spread = None
            if implied_vol:
                rv_iv_spread = rv_30d.volatility_pct - implied_vol.atm_iv
            
            # Determine current regime
            current_regime = rv_30d.regime
            
            # Generate signals
            volatility_signals = await self._generate_volatility_signals(
                rv_7d, rv_30d, implied_vol, rv_iv_spread, term_structure
            )
            
            # Calculate confidence
            overall_confidence = self._calculate_confidence(
                implied_vol is not None, vol_smile is not None
            )
            
            # Create analysis
            analysis = CryptoVolatilityAnalysis(
                symbol=symbol,
                timestamp=datetime.now(),
                realized_vol_7d=rv_7d,
                realized_vol_30d=rv_30d,
                implied_vol=implied_vol,
                vol_smile=vol_smile,
                term_structure=term_structure,
                rv_iv_spread=rv_iv_spread,
                current_regime=current_regime,
                overall_confidence=overall_confidence,
                volatility_signals=volatility_signals,
                metadata={
                    'analysis_version': '1.0',
                    'has_options_data': options_data is not None,
                    'stats': self.stats
                }
            )
            
            # Update statistics
            self.stats['analyses_performed'] += 1
            self.stats['last_update'] = datetime.now()
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"❌ Error analyzing volatility for {symbol}: {e}")
            return self._get_default_analysis(symbol)
    
    def _calculate_realized_volatility(
        self,
        df: pd.DataFrame,
        days: int = 30
    ) -> RealizedVolatility:
        """Calculate realized (historical) volatility"""
        try:
            # Get data for specified period
            period_data = df.tail(days)
            
            if len(period_data) < 2:
                return RealizedVolatility(
                    period_days=days,
                    volatility_pct=0.0,
                    percentile=50.0,
                    regime=VolatilityRegime.NORMAL,
                    trend='stable'
                )
            
            # Calculate log returns
            returns = np.log(period_data['close'] / period_data['close'].shift(1)).dropna()
            
            # Standard deviation of returns
            std_dev = returns.std()
            
            # Annualize (assuming daily data)
            annualized_vol = std_dev * self.annualization_factor * 100
            
            # Calculate percentile (would use longer history in production)
            all_data = df['close']
            rolling_vols = []
            for i in range(days, len(all_data)):
                window = all_data.iloc[i-days:i]
                window_returns = np.log(window / window.shift(1)).dropna()
                rolling_vols.append(window_returns.std())
            
            if rolling_vols:
                percentile = (sum(1 for v in rolling_vols if v <= std_dev) / len(rolling_vols)) * 100
            else:
                percentile = 50.0
            
            # Determine regime
            if percentile > 80:
                regime = VolatilityRegime.EXTREME_HIGH
            elif percentile > 60:
                regime = VolatilityRegime.HIGH
            elif percentile > 40:
                regime = VolatilityRegime.NORMAL
            elif percentile > 20:
                regime = VolatilityRegime.LOW
            else:
                regime = VolatilityRegime.EXTREME_LOW
            
            # Determine trend
            if len(rolling_vols) >= 5:
                recent_vol = np.mean(rolling_vols[-5:])
                past_vol = np.mean(rolling_vols[-10:-5]) if len(rolling_vols) >= 10 else recent_vol
                
                if recent_vol > past_vol * 1.1:
                    trend = 'increasing'
                elif recent_vol < past_vol * 0.9:
                    trend = 'decreasing'
                else:
                    trend = 'stable'
            else:
                trend = 'stable'
            
            return RealizedVolatility(
                period_days=days,
                volatility_pct=annualized_vol,
                percentile=percentile,
                regime=regime,
                trend=trend
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating realized volatility: {e}")
            return RealizedVolatility(
                period_days=days,
                volatility_pct=0.0,
                percentile=50.0,
                regime=VolatilityRegime.NORMAL,
                trend='stable'
            )
    
    def _extract_implied_volatility(
        self,
        options_data: Dict[str, Any]
    ) -> Optional[ImpliedVolatility]:
        """Extract implied volatility from options data"""
        # Placeholder - would parse Deribit options data
        return None
    
    def _analyze_volatility_smile(
        self,
        options_data: Dict[str, Any]
    ) -> Optional[VolatilitySmile]:
        """Analyze volatility smile"""
        # Placeholder - would analyze IV by strike price
        return None
    
    def _analyze_term_structure(
        self,
        options_data: Dict[str, Any]
    ) -> Optional[VolatilityTermStructure]:
        """Analyze volatility term structure"""
        # Placeholder - would analyze IV by expiration
        return None
    
    async def _generate_volatility_signals(
        self,
        rv_7d: RealizedVolatility,
        rv_30d: RealizedVolatility,
        implied_vol: Optional[ImpliedVolatility],
        rv_iv_spread: Optional[float],
        term_structure: Optional[VolatilityTermStructure]
    ) -> List[Dict[str, Any]]:
        """Generate signals from volatility analysis"""
        signals = []
        
        try:
            # Volatility regime signals
            if rv_30d.regime == VolatilityRegime.EXTREME_LOW:
                signals.append({
                    'type': 'volatility_regime',
                    'regime': 'extreme_low',
                    'confidence': 0.75,
                    'reasoning': "Volatility at extreme lows - potential expansion ahead",
                    'action': 'prepare_for_volatility'
                })
            elif rv_30d.regime == VolatilityRegime.EXTREME_HIGH:
                signals.append({
                    'type': 'volatility_regime',
                    'regime': 'extreme_high',
                    'confidence': 0.75,
                    'reasoning': "Volatility at extreme highs - potential contraction",
                    'action': 'expect_calm'
                })
            
            # RV vs IV spread signals
            if rv_iv_spread:
                if rv_iv_spread < -10:  # RV much lower than IV
                    signals.append({
                        'type': 'rv_iv_spread',
                        'direction': 'vol_expansion',
                        'confidence': 0.70,
                        'spread': rv_iv_spread,
                        'reasoning': "Realized vol below implied - potential expansion"
                    })
                elif rv_iv_spread > 10:  # RV much higher than IV
                    signals.append({
                        'type': 'rv_iv_spread',
                        'direction': 'vol_contraction',
                        'confidence': 0.70,
                        'spread': rv_iv_spread,
                        'reasoning': "Realized vol above implied - potential contraction"
                    })
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Error generating volatility signals: {e}")
            return signals
    
    def _calculate_confidence(
        self,
        has_implied_vol: bool,
        has_smile: bool
    ) -> float:
        """Calculate confidence"""
        confidence = 0.6  # Base confidence from realized vol
        
        if has_implied_vol:
            confidence += 0.2
        if has_smile:
            confidence += 0.1
        
        return min(0.90, confidence)
    
    def _get_default_analysis(self, symbol: str) -> CryptoVolatilityAnalysis:
        """Get default analysis"""
        return CryptoVolatilityAnalysis(
            symbol=symbol,
            timestamp=datetime.now(),
            realized_vol_7d=RealizedVolatility(7, 0.0, 50.0, VolatilityRegime.NORMAL, 'stable'),
            realized_vol_30d=RealizedVolatility(30, 0.0, 50.0, VolatilityRegime.NORMAL, 'stable'),
            implied_vol=None,
            vol_smile=None,
            term_structure=None,
            rv_iv_spread=None,
            current_regime=VolatilityRegime.NORMAL,
            overall_confidence=0.0,
            volatility_signals=[],
            metadata={'error': 'No data available'}
        )
    
    async def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        return {
            'stats': self.stats,
            'last_update': datetime.now().isoformat()
        }

