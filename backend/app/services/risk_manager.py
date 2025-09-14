"""
Risk Manager Service for AlphaPulse
Handles risk management and position sizing
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class RiskManager:
    """Service for managing trading risk"""
    
    def __init__(self):
        self.is_running = False
        self.risk_limits = {
            "max_daily_loss": 1000.0,  # USDT
            "max_drawdown": 0.1,  # 10%
            "max_position_size": 1000.0,  # USDT
            "max_open_positions": 10,
            "stop_loss_percentage": 0.02,  # 2%
            "take_profit_percentage": 0.04,  # 4%
            "max_risk_per_trade": 0.02  # 2% of portfolio
        }
        
        # Current risk metrics
        self.daily_pnl = 0.0
        self.current_drawdown = 0.0
        self.open_positions = 0
        self.portfolio_value = 10000.0  # Starting portfolio value
        
        # Enhanced risk metrics for leverage and liquidity
        self.leverage_limits = {
            "max_leverage": 125,
            "default_leverage": 1,
            "dynamic_leverage_enabled": True,
            "liquidity_threshold": 0.3,  # Minimum liquidity score for high leverage
            "volatility_threshold": 0.05  # Maximum volatility for high leverage
        }
        
        # Liquidation risk tracking
        self.liquidation_risk_scores = {}  # symbol -> risk_score (0-100)
        self.liquidation_levels = {}  # symbol -> liquidation_levels
        self.market_depth_cache = {}  # symbol -> market_depth_data
        
        # Dynamic leverage adjustment
        self.dynamic_leverage_factors = {
            "liquidity_weight": 0.4,
            "volatility_weight": 0.3,
            "market_regime_weight": 0.2,
            "portfolio_risk_weight": 0.1
        }
        
    async def start(self):
        """Start the risk manager"""
        if self.is_running:
            logger.warning("Risk manager is already running")
            return
            
        logger.info("🚀 Starting Risk Manager...")
        self.is_running = True
        
        # Start background tasks
        asyncio.create_task(self._monitor_risk())
        
        logger.info("✅ Risk Manager started successfully")
    
    async def stop(self):
        """Stop the risk manager"""
        if not self.is_running:
            logger.warning("Risk manager is not running")
            return
            
        logger.info("🛑 Stopping Risk Manager...")
        self.is_running = False
        logger.info("✅ Risk Manager stopped successfully")
    
    # ==================== ENHANCED LEVERAGE AND LIQUIDITY RISK METHODS ====================
    
    async def calculate_dynamic_leverage(self, symbol: str, base_leverage: int = 1) -> int:
        """Calculate dynamic leverage based on market conditions"""
        try:
            if not self.leverage_limits["dynamic_leverage_enabled"]:
                return base_leverage
            
            # Get market conditions
            liquidity_score = await self._get_liquidity_score(symbol)
            volatility_score = await self._get_volatility_score(symbol)
            market_regime_score = await self._get_market_regime_score(symbol)
            portfolio_risk_score = await self._get_portfolio_risk_score()
            
            # Calculate weighted risk score
            risk_score = (
                liquidity_score * self.dynamic_leverage_factors["liquidity_weight"] +
                volatility_score * self.dynamic_leverage_factors["volatility_weight"] +
                market_regime_score * self.dynamic_leverage_factors["market_regime_weight"] +
                portfolio_risk_score * self.dynamic_leverage_factors["portfolio_risk_weight"]
            )
            
            # Adjust leverage based on risk score
            if risk_score >= 0.8:  # Low risk
                adjusted_leverage = min(base_leverage * 2, self.leverage_limits["max_leverage"])
            elif risk_score >= 0.6:  # Medium risk
                adjusted_leverage = base_leverage
            elif risk_score >= 0.4:  # High risk
                adjusted_leverage = max(base_leverage // 2, 1)
            else:  # Very high risk
                adjusted_leverage = 1
            
            logger.info(f"Dynamic leverage for {symbol}: {base_leverage} -> {adjusted_leverage} (risk_score: {risk_score:.3f})")
            return adjusted_leverage
            
        except Exception as e:
            logger.error(f"Error calculating dynamic leverage: {e}")
            return base_leverage
    
    async def calculate_liquidation_risk_score(self, symbol: str) -> float:
        """Calculate liquidation risk score (0-100) for a symbol"""
        try:
            # Get current market data
            current_price = await self._get_current_price(symbol)
            liquidation_levels = await self._get_liquidation_levels(symbol)
            market_depth = await self._get_market_depth(symbol)
            
            if not current_price or not liquidation_levels:
                return 50.0  # Default medium risk
            
            risk_factors = []
            
            # Factor 1: Distance to nearest liquidation level
            nearest_liquidation = self._find_nearest_liquidation_level(current_price, liquidation_levels)
            if nearest_liquidation:
                distance_to_liquidation = abs(current_price - nearest_liquidation['price_level']) / current_price
                distance_risk = max(0, 100 - (distance_to_liquidation * 1000))  # Closer = higher risk
                risk_factors.append(distance_risk)
            
            # Factor 2: Liquidity at liquidation levels
            if market_depth:
                liquidity_risk = self._calculate_liquidity_risk_at_levels(current_price, liquidation_levels, market_depth)
                risk_factors.append(liquidity_risk)
            
            # Factor 3: Market volatility
            volatility_risk = await self._calculate_volatility_risk(symbol)
            risk_factors.append(volatility_risk)
            
            # Factor 4: Open interest and funding rates
            funding_risk = await self._calculate_funding_risk(symbol)
            risk_factors.append(funding_risk)
            
            # Calculate weighted average risk score
            if risk_factors:
                weights = [0.3, 0.25, 0.25, 0.2]  # Adjust weights based on importance
                weighted_risk = sum(r * w for r, w in zip(risk_factors, weights))
                final_risk_score = min(max(weighted_risk, 0.0), 100.0)
            else:
                final_risk_score = 50.0
            
            # Cache the risk score
            self.liquidation_risk_scores[symbol] = final_risk_score
            
            logger.info(f"Liquidation risk score for {symbol}: {final_risk_score:.2f}")
            return final_risk_score
            
        except Exception as e:
            logger.error(f"Error calculating liquidation risk score: {e}")
            return 50.0
    
    async def get_portfolio_risk_metrics(self) -> Dict[str, Any]:
        """Get comprehensive portfolio risk metrics"""
        try:
            # Calculate VaR (Value at Risk)
            var_95 = await self._calculate_var_95()
            var_99 = await self._calculate_var_99()
            
            # Calculate portfolio leverage
            total_leverage = await self._calculate_total_portfolio_leverage()
            
            # Calculate margin utilization
            margin_utilization = await self._calculate_margin_utilization()
            
            # Calculate correlation risk
            correlation_risk = await self._calculate_correlation_risk()
            
            # Calculate liquidity risk
            portfolio_liquidity_risk = await self._calculate_portfolio_liquidity_risk()
            
            metrics = {
                "var_95": var_95,
                "var_99": var_99,
                "total_leverage": total_leverage,
                "margin_utilization": margin_utilization,
                "correlation_risk": correlation_risk,
                "portfolio_liquidity_risk": portfolio_liquidity_risk,
                "current_drawdown": self.current_drawdown,
                "daily_pnl": self.daily_pnl,
                "open_positions": self.open_positions,
                "portfolio_value": self.portfolio_value,
                "liquidation_risk_scores": self.liquidation_risk_scores.copy()
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating portfolio risk metrics: {e}")
            return {}
    
    async def simulate_liquidation_impact(self, symbol: str, position_size: float, leverage: int) -> Dict[str, Any]:
        """Simulate the impact of a liquidation event on the portfolio"""
        try:
            current_price = await self._get_current_price(symbol)
            liquidation_levels = await self._get_liquidation_levels(symbol)
            
            if not current_price or not liquidation_levels:
                return {"error": "Unable to get market data"}
            
            # Find nearest liquidation level
            nearest_liquidation = self._find_nearest_liquidation_level(current_price, liquidation_levels)
            
            if not nearest_liquidation:
                return {"error": "No liquidation levels found"}
            
            # Calculate liquidation impact
            distance_to_liquidation = abs(current_price - nearest_liquidation['price_level']) / current_price
            liquidation_probability = max(0, 1 - (distance_to_liquidation * 10))  # Closer = higher probability
            
            # Calculate potential loss
            position_value = position_size * current_price
            potential_loss = position_value * (1 - 1/leverage)  # Assuming full liquidation
            
            # Calculate portfolio impact
            portfolio_impact = potential_loss / self.portfolio_value
            
            # Calculate cascading risk
            cascading_risk = await self._calculate_cascading_liquidation_risk(symbol, position_size)
            
            simulation_result = {
                "symbol": symbol,
                "current_price": current_price,
                "nearest_liquidation_price": nearest_liquidation['price_level'],
                "distance_to_liquidation": distance_to_liquidation,
                "liquidation_probability": liquidation_probability,
                "position_value": position_value,
                "potential_loss": potential_loss,
                "portfolio_impact": portfolio_impact,
                "cascading_risk": cascading_risk,
                "risk_level": self._classify_risk_level(portfolio_impact, liquidation_probability)
            }
            
            return simulation_result
            
        except Exception as e:
            logger.error(f"Error simulating liquidation impact: {e}")
            return {"error": str(e)}
    
    # ==================== HELPER METHODS ====================
    
    async def _get_liquidity_score(self, symbol: str) -> float:
        """Get liquidity score for a symbol (0-1)"""
        try:
            # This would integrate with the volume positioning analyzer
            # For now, return a default score
            return 0.7
        except Exception as e:
            logger.error(f"Error getting liquidity score: {e}")
            return 0.5
    
    async def _get_volatility_score(self, symbol: str) -> float:
        """Get volatility score for a symbol (0-1, lower = less volatile)"""
        try:
            # This would calculate recent volatility
            # For now, return a default score
            return 0.6
        except Exception as e:
            logger.error(f"Error getting volatility score: {e}")
            return 0.5
    
    async def _get_market_regime_score(self, symbol: str) -> float:
        """Get market regime score (0-1)"""
        try:
            # This would integrate with market regime detection
            # For now, return a default score
            return 0.7
        except Exception as e:
            logger.error(f"Error getting market regime score: {e}")
            return 0.5
    
    async def _get_portfolio_risk_score(self) -> float:
        """Get portfolio risk score (0-1, lower = less risky)"""
        try:
            # Calculate based on current drawdown and position concentration
            drawdown_risk = min(self.current_drawdown * 2, 1.0)  # Scale drawdown to 0-1
            concentration_risk = min(self.open_positions / 10, 1.0)  # Scale positions to 0-1
            
            portfolio_risk = (drawdown_risk * 0.6 + concentration_risk * 0.4)
            return max(0, 1 - portfolio_risk)  # Invert so lower risk = higher score
            
        except Exception as e:
            logger.error(f"Error getting portfolio risk score: {e}")
            return 0.5
    
    async def _get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price for a symbol"""
        try:
            # This would integrate with market data service
            # For now, return None
            return None
        except Exception as e:
            logger.error(f"Error getting current price: {e}")
            return None
    
    async def _get_liquidation_levels(self, symbol: str) -> List[Dict[str, Any]]:
        """Get liquidation levels for a symbol"""
        try:
            # This would integrate with CCXT service
            # For now, return empty list
            return []
        except Exception as e:
            logger.error(f"Error getting liquidation levels: {e}")
            return []
    
    async def _get_market_depth(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get market depth data for a symbol"""
        try:
            # This would integrate with order book service
            # For now, return None
            return None
        except Exception as e:
            logger.error(f"Error getting market depth: {e}")
            return None
    
    def _find_nearest_liquidation_level(self, current_price: float, liquidation_levels: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Find the nearest liquidation level to current price"""
        try:
            if not liquidation_levels:
                return None
            
            nearest_level = min(liquidation_levels, key=lambda x: abs(x['price_level'] - current_price))
            return nearest_level
            
        except Exception as e:
            logger.error(f"Error finding nearest liquidation level: {e}")
            return None
    
    def _calculate_liquidity_risk_at_levels(self, current_price: float, liquidation_levels: List[Dict], market_depth: Dict[str, Any]) -> float:
        """Calculate liquidity risk at liquidation levels"""
        try:
            # This would analyze market depth at liquidation levels
            # For now, return a default risk score
            return 30.0
        except Exception as e:
            logger.error(f"Error calculating liquidity risk at levels: {e}")
            return 50.0
    
    async def _calculate_volatility_risk(self, symbol: str) -> float:
        """Calculate volatility risk for a symbol"""
        try:
            # This would calculate recent volatility
            # For now, return a default risk score
            return 40.0
        except Exception as e:
            logger.error(f"Error calculating volatility risk: {e}")
            return 50.0
    
    async def _calculate_funding_risk(self, symbol: str) -> float:
        """Calculate funding rate risk for a symbol"""
        try:
            # This would analyze funding rates and open interest
            # For now, return a default risk score
            return 35.0
        except Exception as e:
            logger.error(f"Error calculating funding risk: {e}")
            return 50.0
    
    async def _calculate_var_95(self) -> float:
        """Calculate 95% Value at Risk"""
        try:
            # This would use historical data to calculate VaR
            # For now, return a default value
            return self.portfolio_value * 0.02  # 2% VaR
        except Exception as e:
            logger.error(f"Error calculating VaR 95: {e}")
            return 0.0
    
    async def _calculate_var_99(self) -> float:
        """Calculate 99% Value at Risk"""
        try:
            # This would use historical data to calculate VaR
            # For now, return a default value
            return self.portfolio_value * 0.03  # 3% VaR
        except Exception as e:
            logger.error(f"Error calculating VaR 99: {e}")
            return 0.0
    
    async def _calculate_total_portfolio_leverage(self) -> float:
        """Calculate total portfolio leverage"""
        try:
            # This would sum up all position leverages
            # For now, return a default value
            return 1.5
        except Exception as e:
            logger.error(f"Error calculating total portfolio leverage: {e}")
            return 1.0
    
    async def _calculate_margin_utilization(self) -> float:
        """Calculate margin utilization percentage"""
        try:
            # This would calculate used margin vs available margin
            # For now, return a default value
            return 0.3  # 30% margin utilization
        except Exception as e:
            logger.error(f"Error calculating margin utilization: {e}")
            return 0.0
    
    async def _calculate_correlation_risk(self) -> float:
        """Calculate correlation risk between positions"""
        try:
            # This would analyze position correlations
            # For now, return a default value
            return 0.4  # 40% correlation risk
        except Exception as e:
            logger.error(f"Error calculating correlation risk: {e}")
            return 0.0
    
    async def _calculate_portfolio_liquidity_risk(self) -> float:
        """Calculate portfolio liquidity risk"""
        try:
            # This would analyze overall portfolio liquidity
            # For now, return a default value
            return 0.3  # 30% liquidity risk
        except Exception as e:
            logger.error(f"Error calculating portfolio liquidity risk: {e}")
            return 0.0
    
    async def _calculate_cascading_liquidation_risk(self, symbol: str, position_size: float) -> float:
        """Calculate cascading liquidation risk"""
        try:
            # This would analyze the risk of cascading liquidations
            # For now, return a default value
            return 0.2  # 20% cascading risk
        except Exception as e:
            logger.error(f"Error calculating cascading liquidation risk: {e}")
            return 0.0
    
    def _classify_risk_level(self, portfolio_impact: float, liquidation_probability: float) -> str:
        """Classify risk level based on impact and probability"""
        try:
            risk_score = (portfolio_impact * 0.6 + liquidation_probability * 0.4) * 100
            
            if risk_score >= 80:
                return "CRITICAL"
            elif risk_score >= 60:
                return "HIGH"
            elif risk_score >= 40:
                return "MEDIUM"
            elif risk_score >= 20:
                return "LOW"
            else:
                return "MINIMAL"
                
        except Exception as e:
            logger.error(f"Error classifying risk level: {e}")
            return "UNKNOWN"
    
    async def _monitor_risk(self):
        """Background task to monitor risk metrics"""
        while self.is_running:
            try:
                # Update risk metrics every 10 seconds
                await asyncio.sleep(10)
                
                # Calculate current drawdown
                self._calculate_drawdown()
                
                # Check risk limits
                await self._check_risk_limits()
                
            except Exception as e:
                logger.error(f"❌ Error monitoring risk: {e}")
                await asyncio.sleep(30)
    
    def _calculate_drawdown(self):
        """Calculate current drawdown"""
        try:
            # This would typically use actual portfolio value
            # For now, use a simple calculation
            if self.portfolio_value > 0:
                self.current_drawdown = abs(min(0, self.daily_pnl)) / self.portfolio_value
            else:
                self.current_drawdown = 0.0
                
        except Exception as e:
            logger.error(f"❌ Error calculating drawdown: {e}")
    
    async def _check_risk_limits(self):
        """Check if any risk limits are exceeded"""
        try:
            # Check daily loss limit
            if abs(self.daily_pnl) > self.risk_limits["max_daily_loss"]:
                logger.warning(f"⚠️ Daily loss limit exceeded: {self.daily_pnl:.2f} USDT")
                await self._trigger_risk_alert("daily_loss_limit")
            
            # Check drawdown limit
            if self.current_drawdown > self.risk_limits["max_drawdown"]:
                logger.warning(f"⚠️ Drawdown limit exceeded: {self.current_drawdown:.2%}")
                await self._trigger_risk_alert("drawdown_limit")
            
            # Check position count limit
            if self.open_positions > self.risk_limits["max_open_positions"]:
                logger.warning(f"⚠️ Position count limit exceeded: {self.open_positions}")
                await self._trigger_risk_alert("position_count_limit")
                
        except Exception as e:
            logger.error(f"❌ Error checking risk limits: {e}")
    
    async def _trigger_risk_alert(self, alert_type: str):
        """Trigger a risk alert"""
        try:
            logger.warning(f"🚨 RISK ALERT: {alert_type}")
            
            # This would typically send notifications (email, Discord, etc.)
            # For now, just log the alert
            
        except Exception as e:
            logger.error(f"❌ Error triggering risk alert: {e}")
    
    async def validate_signal(self, signal: Any) -> bool:
        """Validate if a trading signal meets risk requirements"""
        try:
            # Check if we have too many open positions
            if self.open_positions >= self.risk_limits["max_open_positions"]:
                logger.info(f"⚠️ Signal rejected: Too many open positions ({self.open_positions})")
                return False
            
            # Check if daily loss limit would be exceeded
            if abs(self.daily_pnl) >= self.risk_limits["max_daily_loss"]:
                logger.info(f"⚠️ Signal rejected: Daily loss limit reached ({self.daily_pnl:.2f} USDT)")
                return False
            
            # Check if drawdown limit would be exceeded
            if self.current_drawdown >= self.risk_limits["max_drawdown"]:
                logger.info(f"⚠️ Signal rejected: Drawdown limit reached ({self.current_drawdown:.2%})")
                return False
            
            # Check signal confidence
            if hasattr(signal, 'confidence') and signal.confidence < 0.7:
                logger.info(f"⚠️ Signal rejected: Low confidence ({signal.confidence:.2f})")
                return False
            
            # Signal passed all risk checks
            return True
            
        except Exception as e:
            logger.error(f"❌ Error validating signal: {e}")
            return False
    
    def calculate_position_size(self, signal: Any, available_balance: float) -> float:
        """Calculate safe position size based on risk management rules"""
        try:
            # Base position size
            base_size = min(
                self.risk_limits["max_position_size"],
                available_balance * self.risk_limits["max_risk_per_trade"]
            )
            
            # Adjust based on signal confidence
            if hasattr(signal, 'confidence'):
                confidence_multiplier = signal.confidence
                adjusted_size = base_size * confidence_multiplier
            else:
                adjusted_size = base_size
            
            # Ensure position size doesn't exceed available balance
            final_size = min(adjusted_size, available_balance * 0.95)  # Leave 5% buffer
            
            return max(final_size, 0)  # Ensure non-negative
            
        except Exception as e:
            logger.error(f"❌ Error calculating position size: {e}")
            return 0.0
    
    def calculate_stop_loss(self, entry_price: float, signal_type: str) -> float:
        """Calculate stop loss price"""
        try:
            if signal_type.lower() == "buy":
                # Long position: stop loss below entry
                stop_loss = entry_price * (1 - self.risk_limits["stop_loss_percentage"])
            elif signal_type.lower() == "sell":
                # Short position: stop loss above entry
                stop_loss = entry_price * (1 + self.risk_limits["stop_loss_percentage"])
            else:
                return 0.0
            
            return stop_loss
            
        except Exception as e:
            logger.error(f"❌ Error calculating stop loss: {e}")
            return 0.0
    
    def calculate_take_profit(self, entry_price: float, signal_type: str) -> float:
        """Calculate take profit price"""
        try:
            if signal_type.lower() == "buy":
                # Long position: take profit above entry
                take_profit = entry_price * (1 + self.risk_limits["take_profit_percentage"])
            elif signal_type.lower() == "sell":
                # Short position: take profit below entry
                take_profit = entry_price * (1 - self.risk_limits["take_profit_percentage"])
            else:
                return 0.0
            
            return take_profit
            
        except Exception as e:
            logger.error(f"❌ Error calculating take profit: {e}")
            return 0.0
    
    def update_portfolio_metrics(self, pnl: float, position_count: int):
        """Update portfolio metrics"""
        try:
            self.daily_pnl = pnl
            self.open_positions = position_count
            
            # Update portfolio value (simplified)
            self.portfolio_value = 10000.0 + self.daily_pnl
            
        except Exception as e:
            logger.error(f"❌ Error updating portfolio metrics: {e}")
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """Get current risk summary"""
        try:
            return {
                "daily_pnl": self.daily_pnl,
                "current_drawdown": self.current_drawdown,
                "open_positions": self.open_positions,
                "portfolio_value": self.portfolio_value,
                "risk_limits": self.risk_limits,
                "risk_status": "healthy" if self._is_risk_healthy() else "warning"
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting risk summary: {e}")
            return {}
    
    def _is_risk_healthy(self) -> bool:
        """Check if current risk metrics are within healthy limits"""
        try:
            return (
                abs(self.daily_pnl) < self.risk_limits["max_daily_loss"] and
                self.current_drawdown < self.risk_limits["max_drawdown"] and
                self.open_positions < self.risk_limits["max_open_positions"]
            )
            
        except Exception as e:
            logger.error(f"❌ Error checking risk health: {e}")
            return False
