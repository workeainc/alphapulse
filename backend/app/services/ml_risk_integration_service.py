#!/usr/bin/env python3
"""
ML + Risk Integration Service for AlphaPulse
Combines ML predictions with risk engine outputs for actionable trade signals
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import json
import os
from sqlalchemy import create_engine, text

# Import existing services
from app.services.ensemble_system_service import EnsembleSystemService, EnsemblePrediction
from app.services.risk_manager import RiskManager
from app.services.monitoring_service import MonitoringService

logger = logging.getLogger(__name__)

class SignalStrength(Enum):
    STRONG_BUY = "strong_buy"
    BUY = "buy"
    HOLD = "hold"
    SELL = "sell"
    STRONG_SELL = "strong_sell"

class RiskLevel(Enum):
    MINIMAL = "minimal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class MarketRegime(Enum):
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    VOLATILE = "volatile"
    LIQUIDATION_EVENT = "liquidation_event"

@dataclass
class ActionableTradeSignal:
    """Actionable trade signal with ML + Risk integration"""
    symbol: str
    timestamp: datetime
    signal_type: str  # 'long', 'short', 'close', 'hold'
    signal_strength: SignalStrength
    confidence_score: float
    risk_level: RiskLevel
    market_regime: MarketRegime
    
    # Position sizing and risk management
    recommended_leverage: int
    position_size_usdt: float
    stop_loss_price: float
    take_profit_price: float
    risk_reward_ratio: float
    
    # ML model contributions
    ml_confidence: float
    ml_prediction: str
    model_contributions: Dict[str, float]
    
    # Risk engine outputs
    risk_score: float  # 0-100
    liquidation_risk: float  # 0-100
    portfolio_impact: float  # Expected portfolio impact
    
    # Market context
    volatility_score: float
    liquidity_score: float
    market_depth_analysis: Dict[str, Any]
    
    # Metadata
    metadata: Dict[str, Any] = None

class MLRiskIntegrationService:
    """Service for integrating ML predictions with risk management for actionable signals"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logger
        
        # Service configuration
        self.database_url = self.config.get('database_url', os.getenv("DATABASE_URL", "postgresql://alpha_emon:Emon_@17711@localhost:5432/alphapulse"))
        self.engine = create_engine(self.database_url)
        
        # Initialize component services
        self.ensemble_service = None
        self.risk_manager = None
        self.monitoring_service = None
        
        # Integration parameters
        self.integration_params = {
            'ml_weight': 0.6,  # Weight for ML predictions
            'risk_weight': 0.4,  # Weight for risk considerations
            'min_confidence_threshold': 0.7,
            'max_risk_threshold': 70.0,  # Maximum acceptable risk score
            'leverage_adjustment_factor': 0.8,  # Conservative leverage adjustment
            'position_size_multiplier': 1.2,  # Position size adjustment
            'stop_loss_buffer': 1.1,  # Additional buffer for stop loss
            'take_profit_buffer': 0.9  # Conservative take profit
        }
        
        # Signal generation thresholds
        self.signal_thresholds = {
            'strong_buy': {'confidence': 0.85, 'risk_score': 30},
            'buy': {'confidence': 0.75, 'risk_score': 50},
            'hold': {'confidence': 0.6, 'risk_score': 70},
            'sell': {'confidence': 0.75, 'risk_score': 50},
            'strong_sell': {'confidence': 0.85, 'risk_score': 30}
        }
        
        # Performance tracking
        self.performance_metrics = {
            'signals_generated': 0,
            'signals_executed': 0,
            'average_confidence': 0.0,
            'average_risk_score': 0.0,
            'success_rate': 0.0,
            'total_pnl': 0.0
        }
        
    async def initialize(self):
        """Initialize the ML + Risk integration service"""
        self.logger.info("🚀 Initializing ML + Risk Integration Service...")
        
        try:
            # Initialize ensemble system
            ensemble_config = {
                'database_url': self.database_url,
                'models_dir': 'models/ensemble',
                'update_frequency': 300
            }
            self.ensemble_service = EnsembleSystemService(ensemble_config)
            await self.ensemble_service.initialize()
            
            # Initialize risk manager
            self.risk_manager = RiskManager()
            await self.risk_manager.start()
            
            # Initialize monitoring service
            try:
                monitoring_config = {
                    'database_url': self.database_url,
                    'prometheus_enabled': True
                }
                self.monitoring_service = MonitoringService(monitoring_config)
            except Exception as e:
                logger.warning(f"Monitoring service initialization failed: {e}")
                self.monitoring_service = None
            
            self.logger.info("✅ ML + Risk Integration Service initialized successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Error initializing ML + Risk integration service: {e}")
            raise
    
    async def generate_actionable_signal(self, symbol: str, market_data: Dict[str, Any]) -> ActionableTradeSignal:
        """Generate actionable trade signal combining ML predictions with risk analysis"""
        try:
            start_time = datetime.now()
            
            # Step 1: Get ML ensemble prediction
            ensemble_prediction = await self.ensemble_service.predict_unified_signal(symbol, market_data)
            
            # Step 2: Get risk analysis
            risk_analysis = await self._analyze_risk(symbol, market_data, ensemble_prediction)
            
            # Step 3: Determine market regime
            market_regime = await self._determine_market_regime(symbol, market_data, ensemble_prediction)
            
            # Step 4: Generate actionable signal
            actionable_signal = await self._create_actionable_signal(
                symbol, ensemble_prediction, risk_analysis, market_regime, market_data
            )
            
            # Step 5: Validate and adjust signal
            validated_signal = await self._validate_and_adjust_signal(actionable_signal)
            
            # Step 6: Store signal for monitoring
            await self._store_actionable_signal(validated_signal)
            
            # Step 7: Update performance metrics
            self._update_performance_metrics(validated_signal)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            self.logger.info(f"✅ Generated actionable signal for {symbol} in {processing_time:.3f}s")
            
            return validated_signal
            
        except Exception as e:
            self.logger.error(f"❌ Error generating actionable signal for {symbol}: {e}")
            return self._get_default_signal(symbol)
    
    async def _analyze_risk(self, symbol: str, market_data: Dict[str, Any], ensemble_prediction: EnsemblePrediction) -> Dict[str, Any]:
        """Analyze risk factors for the symbol"""
        try:
            # Get portfolio risk metrics
            portfolio_metrics = await self.risk_manager.get_portfolio_risk_metrics()
            
            # Calculate liquidation risk
            liquidation_risk = await self.risk_manager.calculate_liquidation_risk_score(symbol)
            
            # Calculate dynamic leverage
            base_leverage = 1
            dynamic_leverage = await self.risk_manager.calculate_dynamic_leverage(symbol, base_leverage)
            
            # Simulate liquidation impact
            position_size = 1000.0  # Default position size for simulation
            liquidation_impact = await self.risk_manager.simulate_liquidation_impact(
                symbol, position_size, dynamic_leverage
            )
            
            # Calculate volatility and liquidity scores
            volatility_score = await self._calculate_volatility_score(symbol, market_data)
            liquidity_score = await self._calculate_liquidity_score(symbol, market_data)
            
            # Calculate overall risk score
            overall_risk_score = self._calculate_overall_risk_score(
                liquidation_risk, volatility_score, liquidity_score, portfolio_metrics
            )
            
            risk_analysis = {
                'portfolio_metrics': portfolio_metrics,
                'liquidation_risk': liquidation_risk,
                'dynamic_leverage': dynamic_leverage,
                'liquidation_impact': liquidation_impact,
                'volatility_score': volatility_score,
                'liquidity_score': liquidity_score,
                'overall_risk_score': overall_risk_score
            }
            
            return risk_analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing risk for {symbol}: {e}")
            return self._get_default_risk_analysis()
    
    async def _determine_market_regime(self, symbol: str, market_data: Dict[str, Any], ensemble_prediction: EnsemblePrediction) -> MarketRegime:
        """Determine current market regime"""
        try:
            # Extract price and volume data
            prices = market_data.get('close_prices', [])
            volumes = market_data.get('volumes', [])
            
            if len(prices) < 20:
                return MarketRegime.RANGING
            
            # Calculate trend strength
            price_change = (prices[-1] - prices[-20]) / prices[-20]
            trend_strength = abs(price_change)
            
            # Calculate volatility
            returns = np.diff(prices) / prices[:-1]
            volatility = np.std(returns) * np.sqrt(252)  # Annualized
            
            # Determine regime based on ensemble prediction and market data
            if ensemble_prediction.unified_signal in ['strong_buy', 'strong_sell']:
                if trend_strength > 0.7:
                    return MarketRegime.TRENDING_UP if price_change > 0 else MarketRegime.TRENDING_DOWN
                elif volatility > 0.3:
                    return MarketRegime.VOLATILE
            
            # Check for liquidation event indicators
            if ensemble_prediction.risk_level == 'critical' or ensemble_prediction.unified_signal in ['strong_sell']:
                return MarketRegime.LIQUIDATION_EVENT
            
            return MarketRegime.RANGING
            
        except Exception as e:
            self.logger.error(f"Error determining market regime for {symbol}: {e}")
            return MarketRegime.RANGING
    
    async def _create_actionable_signal(self, symbol: str, ensemble_prediction: EnsemblePrediction, 
                                      risk_analysis: Dict[str, Any], market_regime: MarketRegime, 
                                      market_data: Dict[str, Any]) -> ActionableTradeSignal:
        """Create actionable trade signal from ML prediction and risk analysis"""
        try:
            # Determine signal type and strength
            signal_type, signal_strength = self._determine_signal_type_and_strength(ensemble_prediction)
            
            # Calculate confidence score (ML + Risk weighted)
            ml_confidence = ensemble_prediction.confidence_score
            risk_confidence = 1.0 - (risk_analysis['overall_risk_score'] / 100.0)
            combined_confidence = (
                ml_confidence * self.integration_params['ml_weight'] +
                risk_confidence * self.integration_params['risk_weight']
            )
            
            # Determine risk level
            risk_level = self._determine_risk_level(risk_analysis['overall_risk_score'])
            
            # Calculate position sizing and risk management
            position_calculation = await self._calculate_position_sizing(
                symbol, ensemble_prediction, risk_analysis, market_regime
            )
            
            # Create actionable signal
            actionable_signal = ActionableTradeSignal(
                symbol=symbol,
                timestamp=datetime.now(),
                signal_type=signal_type,
                signal_strength=signal_strength,
                confidence_score=combined_confidence,
                risk_level=risk_level,
                market_regime=market_regime,
                recommended_leverage=position_calculation['leverage'],
                position_size_usdt=position_calculation['position_size'],
                stop_loss_price=position_calculation['stop_loss'],
                take_profit_price=position_calculation['take_profit'],
                risk_reward_ratio=position_calculation['risk_reward_ratio'],
                ml_confidence=ml_confidence,
                ml_prediction=ensemble_prediction.unified_signal,
                model_contributions=ensemble_prediction.model_contributions,
                risk_score=risk_analysis['overall_risk_score'],
                liquidation_risk=risk_analysis['liquidation_risk'],
                portfolio_impact=risk_analysis['liquidation_impact'].get('portfolio_impact', 0.0),
                volatility_score=risk_analysis['volatility_score'],
                liquidity_score=risk_analysis['liquidity_score'],
                market_depth_analysis={},
                metadata={
                    'ensemble_prediction': {
                        'unified_signal': ensemble_prediction.unified_signal,
                        'confidence_score': ensemble_prediction.confidence_score,
                        'model_contributions': ensemble_prediction.model_contributions,
                        'risk_level': getattr(ensemble_prediction, 'risk_level', 'medium')
                    },
                    'risk_analysis': {
                        'overall_risk_score': risk_analysis['overall_risk_score'],
                        'liquidation_risk': risk_analysis['liquidation_risk'],
                        'dynamic_leverage': risk_analysis['dynamic_leverage'],
                        'volatility_score': risk_analysis['volatility_score'],
                        'liquidity_score': risk_analysis['liquidity_score']
                    }
                }
            )
            
            return actionable_signal
            
        except Exception as e:
            self.logger.error(f"Error creating actionable signal for {symbol}: {e}")
            return self._get_default_signal(symbol)
    
    async def _calculate_position_sizing(self, symbol: str, ensemble_prediction: EnsemblePrediction, 
                                       risk_analysis: Dict[str, Any], market_regime: MarketRegime) -> Dict[str, Any]:
        """Calculate position sizing and risk management parameters"""
        try:
            # Base position size (USDT)
            base_position_size = 1000.0  # Default base size
            
            # Adjust based on confidence and risk
            confidence_multiplier = ensemble_prediction.confidence_score
            risk_multiplier = 1.0 - (risk_analysis['overall_risk_score'] / 100.0)
            
            # Adjust based on market regime
            regime_multiplier = self._get_regime_multiplier(market_regime)
            
            # Calculate adjusted position size
            adjusted_position_size = base_position_size * confidence_multiplier * risk_multiplier * regime_multiplier
            
            # Get current price for calculations
            current_price = 50000.0  # Default price
            
            # Calculate leverage (conservative approach)
            base_leverage = risk_analysis['dynamic_leverage']
            leverage_adjustment = self.integration_params['leverage_adjustment_factor']
            recommended_leverage = max(1, int(base_leverage * leverage_adjustment))
            
            # Calculate stop loss and take profit
            stop_loss_pct = 0.02  # 2% default
            take_profit_pct = 0.04  # 4% default
            
            # Adjust based on volatility
            volatility_adjustment = 1.0 + (risk_analysis['volatility_score'] - 0.5) * 0.5
            stop_loss_pct *= volatility_adjustment * self.integration_params['stop_loss_buffer']
            take_profit_pct *= volatility_adjustment * self.integration_params['take_profit_buffer']
            
            # Calculate prices
            if ensemble_prediction.unified_signal in ['strong_buy', 'buy']:
                stop_loss_price = current_price * (1 - stop_loss_pct)
                take_profit_price = current_price * (1 + take_profit_pct)
            else:
                stop_loss_price = current_price * (1 + stop_loss_pct)
                take_profit_price = current_price * (1 - take_profit_pct)
            
            # Calculate risk-reward ratio
            risk_reward_ratio = take_profit_pct / stop_loss_pct
            
            return {
                'leverage': recommended_leverage,
                'position_size': float(adjusted_position_size),
                'stop_loss': float(stop_loss_price),
                'take_profit': float(take_profit_price),
                'risk_reward_ratio': float(risk_reward_ratio)
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating position sizing for {symbol}: {e}")
            return {
                'leverage': 1,
                'position_size': 1000.0,
                'stop_loss': 0.0,
                'take_profit': 0.0,
                'risk_reward_ratio': 2.0
            }
    
    async def _validate_and_adjust_signal(self, signal: ActionableTradeSignal) -> ActionableTradeSignal:
        """Validate and adjust signal based on risk limits and portfolio constraints"""
        try:
            # Check confidence threshold
            if signal.confidence_score < self.integration_params['min_confidence_threshold']:
                signal.signal_type = 'hold'
                signal.signal_strength = SignalStrength.HOLD
                signal.confidence_score = 0.5
            
            # Check risk threshold
            if signal.risk_score > self.integration_params['max_risk_threshold']:
                signal.signal_type = 'hold'
                signal.signal_strength = SignalStrength.HOLD
                signal.recommended_leverage = 1
                signal.position_size_usdt *= 0.5  # Reduce position size
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Error validating signal: {e}")
            return signal
    
    def _determine_signal_type_and_strength(self, ensemble_prediction: EnsemblePrediction) -> Tuple[str, SignalStrength]:
        """Determine signal type and strength from ensemble prediction"""
        try:
            signal_map = {
                'strong_buy': ('long', SignalStrength.STRONG_BUY),
                'buy': ('long', SignalStrength.BUY),
                'hold': ('hold', SignalStrength.HOLD),
                'sell': ('short', SignalStrength.SELL),
                'strong_sell': ('short', SignalStrength.STRONG_SELL)
            }
            
            return signal_map.get(ensemble_prediction.unified_signal, ('hold', SignalStrength.HOLD))
            
        except Exception as e:
            self.logger.error(f"Error determining signal type: {e}")
            return ('hold', SignalStrength.HOLD)
    
    def _determine_risk_level(self, risk_score: float) -> RiskLevel:
        """Determine risk level from risk score"""
        try:
            if risk_score <= 20:
                return RiskLevel.MINIMAL
            elif risk_score <= 40:
                return RiskLevel.LOW
            elif risk_score <= 60:
                return RiskLevel.MEDIUM
            elif risk_score <= 80:
                return RiskLevel.HIGH
            else:
                return RiskLevel.CRITICAL
        except Exception as e:
            self.logger.error(f"Error determining risk level: {e}")
            return RiskLevel.MEDIUM
    
    def _calculate_overall_risk_score(self, liquidation_risk: float, volatility_score: float, 
                                    liquidity_score: float, portfolio_metrics: Dict[str, Any]) -> float:
        """Calculate overall risk score"""
        try:
            # Weighted combination of risk factors
            weights = {
                'liquidation_risk': 0.4,
                'volatility_risk': 0.3,
                'liquidity_risk': 0.2,
                'portfolio_risk': 0.1
            }
            
            volatility_risk = (1.0 - volatility_score) * 100  # Invert so higher volatility = higher risk
            liquidity_risk = (1.0 - liquidity_score) * 100  # Invert so lower liquidity = higher risk
            portfolio_risk = portfolio_metrics.get('current_drawdown', 0.0) * 100
            
            overall_risk = (
                liquidation_risk * weights['liquidation_risk'] +
                volatility_risk * weights['volatility_risk'] +
                liquidity_risk * weights['liquidity_risk'] +
                portfolio_risk * weights['portfolio_risk']
            )
            
            return float(min(max(overall_risk, 0.0), 100.0))
            
        except Exception as e:
            self.logger.error(f"Error calculating overall risk score: {e}")
            return 50.0
    
    def _get_regime_multiplier(self, market_regime: MarketRegime) -> float:
        """Get position size multiplier based on market regime"""
        try:
            regime_multipliers = {
                MarketRegime.TRENDING_UP: 1.2,
                MarketRegime.TRENDING_DOWN: 1.1,
                MarketRegime.RANGING: 0.8,
                MarketRegime.VOLATILE: 0.6,
                MarketRegime.LIQUIDATION_EVENT: 0.3
            }
            
            return regime_multipliers.get(market_regime, 1.0)
            
        except Exception as e:
            self.logger.error(f"Error getting regime multiplier: {e}")
            return 1.0
    
    async def _calculate_volatility_score(self, symbol: str, market_data: Dict[str, Any]) -> float:
        """Calculate volatility score (0-1, lower = less volatile)"""
        try:
            prices = market_data.get('close_prices', [])
            if len(prices) < 20:
                return 0.5
            
            returns = np.diff(prices) / prices[:-1]
            volatility = np.std(returns)
            
            # Normalize to 0-1 scale (lower volatility = higher score)
            volatility_score = max(0, 1 - (volatility * 100))
            return float(min(volatility_score, 1.0))
            
        except Exception as e:
            self.logger.error(f"Error calculating volatility score: {e}")
            return 0.5
    
    async def _calculate_liquidity_score(self, symbol: str, market_data: Dict[str, Any]) -> float:
        """Calculate liquidity score (0-1, higher = more liquid)"""
        try:
            volumes = market_data.get('volumes', [])
            if len(volumes) < 20:
                return 0.5
            
            avg_volume = np.mean(volumes[-20:])
            current_volume = volumes[-1]
            
            # Normalize volume to 0-1 scale
            liquidity_score = min(current_volume / (avg_volume * 2), 1.0)
            return float(max(liquidity_score, 0.0))
            
        except Exception as e:
            self.logger.error(f"Error calculating liquidity score: {e}")
            return 0.5
    
    async def _store_actionable_signal(self, signal: ActionableTradeSignal):
        """Store actionable signal in database"""
        try:
            with self.engine.connect() as conn:
                query = text("""
                    INSERT INTO actionable_trade_signals (
                        symbol, timestamp, signal_type, signal_strength, confidence_score,
                        risk_level, market_regime, recommended_leverage, position_size_usdt,
                        stop_loss_price, take_profit_price, risk_reward_ratio, ml_confidence,
                        ml_prediction, model_contributions, risk_score, liquidation_risk,
                        portfolio_impact, volatility_score, liquidity_score, market_depth_analysis,
                        metadata
                    ) VALUES (
                        :symbol, :timestamp, :signal_type, :signal_strength, :confidence_score,
                        :risk_level, :market_regime, :recommended_leverage, :position_size_usdt,
                        :stop_loss_price, :take_profit_price, :risk_reward_ratio, :ml_confidence,
                        :ml_prediction, :model_contributions, :risk_score, :liquidation_risk,
                        :portfolio_impact, :volatility_score, :liquidity_score, :market_depth_analysis,
                        :metadata
                    )
                """)
                
                conn.execute(query, {
                    'symbol': signal.symbol,
                    'timestamp': signal.timestamp,
                    'signal_type': signal.signal_type,
                    'signal_strength': signal.signal_strength.value,
                    'confidence_score': float(signal.confidence_score),
                    'risk_level': signal.risk_level.value,
                    'market_regime': signal.market_regime.value,
                    'recommended_leverage': signal.recommended_leverage,
                    'position_size_usdt': float(signal.position_size_usdt),
                    'stop_loss_price': float(signal.stop_loss_price),
                    'take_profit_price': float(signal.take_profit_price),
                    'risk_reward_ratio': float(signal.risk_reward_ratio),
                    'ml_confidence': float(signal.ml_confidence),
                    'ml_prediction': signal.ml_prediction,
                    'model_contributions': json.dumps(signal.model_contributions),
                    'risk_score': float(signal.risk_score),
                    'liquidation_risk': float(signal.liquidation_risk),
                    'portfolio_impact': float(signal.portfolio_impact),
                    'volatility_score': float(signal.volatility_score),
                    'liquidity_score': float(signal.liquidity_score),
                    'market_depth_analysis': json.dumps(signal.market_depth_analysis),
                    'metadata': json.dumps(signal.metadata or {})
                })
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Error storing actionable signal: {e}")
    
    def _update_performance_metrics(self, signal: ActionableTradeSignal):
        """Update performance metrics"""
        try:
            self.performance_metrics['signals_generated'] += 1
            self.performance_metrics['average_confidence'] = (
                (self.performance_metrics['average_confidence'] * (self.performance_metrics['signals_generated'] - 1) +
                 signal.confidence_score) / self.performance_metrics['signals_generated']
            )
            self.performance_metrics['average_risk_score'] = (
                (self.performance_metrics['average_risk_score'] * (self.performance_metrics['signals_generated'] - 1) +
                 signal.risk_score) / self.performance_metrics['signals_generated']
            )
            
        except Exception as e:
            self.logger.error(f"Error updating performance metrics: {e}")
    
    def _get_default_signal(self, symbol: str) -> ActionableTradeSignal:
        """Get default signal when generation fails"""
        return ActionableTradeSignal(
            symbol=symbol,
            timestamp=datetime.now(),
            signal_type='hold',
            signal_strength=SignalStrength.HOLD,
            confidence_score=0.5,
            risk_level=RiskLevel.MEDIUM,
            market_regime=MarketRegime.RANGING,
            recommended_leverage=1,
            position_size_usdt=0.0,
            stop_loss_price=0.0,
            take_profit_price=0.0,
            risk_reward_ratio=1.0,
            ml_confidence=0.5,
            ml_prediction='hold',
            model_contributions={},
            risk_score=50.0,
            liquidation_risk=50.0,
            portfolio_impact=0.0,
            volatility_score=0.5,
            liquidity_score=0.5,
            market_depth_analysis={},
            metadata={'error': 'Signal generation failed'}
        )
    
    def _get_default_risk_analysis(self) -> Dict[str, Any]:
        """Get default risk analysis when analysis fails"""
        return {
            'portfolio_metrics': {},
            'liquidation_risk': 50.0,
            'dynamic_leverage': 1,
            'liquidation_impact': {},
            'volatility_score': 0.5,
            'liquidity_score': 0.5,
            'overall_risk_score': 50.0
        }
    
    async def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        return {
            'performance_metrics': self.performance_metrics,
            'integration_params': self.integration_params,
            'signal_thresholds': self.signal_thresholds
        }
    
    async def stop(self):
        """Stop the service"""
        try:
            if self.risk_manager:
                await self.risk_manager.stop()
            self.logger.info("✅ ML + Risk Integration Service stopped")
        except Exception as e:
            self.logger.error(f"Error stopping service: {e}")
