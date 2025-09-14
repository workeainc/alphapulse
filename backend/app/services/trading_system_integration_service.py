import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import asyncpg

class SignalType(Enum):
    ENTRY = "entry"
    EXIT = "exit"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"

class SignalSource(Enum):
    VOLUME_ANALYSIS = "volume_analysis"
    ML_PREDICTION = "ml_prediction"
    RL_AGENT = "rl_agent"
    ANOMALY_DETECTION = "anomaly_detection"

class OptimizationType(Enum):
    POSITION_SIZING = "position_sizing"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"

class AlertPriority(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class TradingSignal:
    signal_type: SignalType
    signal_strength: float
    signal_source: SignalSource
    entry_price: float
    stop_loss_price: float
    take_profit_price: float
    position_size: float
    risk_reward_ratio: float
    confidence_score: float
    metadata: Dict

@dataclass
class PositionOptimization:
    optimization_type: OptimizationType
    current_position_size: float
    recommended_position_size: float
    current_stop_loss: float
    recommended_stop_loss: float
    current_take_profit: float
    recommended_take_profit: float
    optimization_factors: Dict
    confidence_score: float

@dataclass
class Alert:
    alert_type: str
    priority_level: AlertPriority
    alert_score: float
    contributing_factors: Dict
    alert_message: str

class TradingSystemIntegrationService:
    """Service for integrating all analysis components into trading signals and position management"""
    
    def __init__(self, db_pool):
        self.db_pool = db_pool
        self.logger = logging.getLogger(__name__)
        
        # Signal generation parameters
        self.signal_thresholds = {
            'min_signal_strength': 0.6,
            'min_confidence_score': 0.7,
            'min_risk_reward_ratio': 2.0,
            'max_position_size': 0.1,  # 10% of portfolio
            'default_stop_loss_pct': 0.02,  # 2%
            'default_take_profit_pct': 0.04  # 4%
        }
        
        # Position optimization parameters
        self.optimization_params = {
            'volatility_multiplier': 1.5,
            'volume_confidence_multiplier': 1.2,
            'ml_confidence_multiplier': 1.3,
            'rl_confidence_multiplier': 1.1,
            'anomaly_penalty_multiplier': 0.8
        }
        
        # Alert system parameters
        self.alert_thresholds = {
            'critical_anomaly_score': 0.8,
            'high_volume_spike': 5.0,
            'low_confidence_signal': 0.5,
            'poor_risk_reward': 1.5
        }
        
        self.logger.info("🎯 Trading System Integration Service initialized")
    
    async def generate_trading_signals(self, symbol: str, timeframe: str, 
                                     volume_analysis: Dict, ml_prediction: Dict = None,
                                     rl_action: Dict = None, anomalies: List[Dict] = None) -> List[TradingSignal]:
        """Generate trading signals by integrating all analysis components"""
        try:
            signals = []
            
            # Generate volume-based signals
            volume_signals = await self._generate_volume_signals(symbol, timeframe, volume_analysis)
            signals.extend(volume_signals)
            
            # Generate ML-based signals
            if ml_prediction:
                ml_signals = await self._generate_ml_signals(symbol, timeframe, ml_prediction)
                signals.extend(ml_signals)
            
            # Generate RL-based signals
            if rl_action:
                rl_signals = await self._generate_rl_signals(symbol, timeframe, rl_action)
                signals.extend(rl_signals)
            
            # Apply anomaly filters
            if anomalies:
                signals = await self._apply_anomaly_filters(signals, anomalies)
            
            # Store signals in database
            if signals:
                await self._store_trading_signals(symbol, timeframe, signals)
            
            self.logger.info(f"🎯 Generated {len(signals)} trading signals for {symbol}")
            return signals
            
        except Exception as e:
            self.logger.error(f"❌ Error generating trading signals: {e}")
            return []
    
    async def _generate_volume_signals(self, symbol: str, timeframe: str, volume_analysis: Dict) -> List[TradingSignal]:
        """Generate signals based on volume analysis"""
        signals = []
        
        try:
            # Extract key metrics
            volume_ratio = volume_analysis.get('volume_ratio', 1.0)
            volume_positioning_score = volume_analysis.get('volume_positioning_score', 0.5)
            volume_pattern_type = volume_analysis.get('volume_pattern_type', 'NONE')
            volume_breakout = volume_analysis.get('volume_breakout', False)
            current_price = volume_analysis.get('close', 0)
            
            if current_price <= 0:
                return signals
            
            # Volume breakout signals
            if volume_breakout and volume_ratio > 2.0:
                signal_strength = min(volume_ratio / 5.0, 1.0)
                confidence_score = min(volume_positioning_score * 1.2, 1.0)
                
                if signal_strength >= self.signal_thresholds['min_signal_strength'] and confidence_score >= self.signal_thresholds['min_confidence_score']:
                    # Calculate entry, stop loss, and take profit
                    entry_price = current_price
                    stop_loss_pct = self.signal_thresholds['default_stop_loss_pct']
                    take_profit_pct = self.signal_thresholds['default_take_profit_pct']
                    
                    # Adjust based on volume strength
                    if volume_ratio > 5.0:
                        take_profit_pct *= 1.5  # More aggressive target
                    
                    stop_loss_price = entry_price * (1 - stop_loss_pct)
                    take_profit_price = entry_price * (1 + take_profit_pct)
                    risk_reward_ratio = take_profit_pct / stop_loss_pct
                    
                    if risk_reward_ratio >= self.signal_thresholds['min_risk_reward_ratio']:
                        signal = TradingSignal(
                            signal_type=SignalType.ENTRY,
                            signal_strength=signal_strength,
                            signal_source=SignalSource.VOLUME_ANALYSIS,
                            entry_price=entry_price,
                            stop_loss_price=stop_loss_price,
                            take_profit_price=take_profit_price,
                            position_size=self.signal_thresholds['max_position_size'],
                            risk_reward_ratio=risk_reward_ratio,
                            confidence_score=confidence_score,
                            metadata={
                                'volume_ratio': volume_ratio,
                                'volume_pattern_type': volume_pattern_type,
                                'volume_positioning_score': volume_positioning_score
                            }
                        )
                        signals.append(signal)
            
            # Volume pattern signals
            if volume_pattern_type in ['VOLUME_SPIKE', 'VOLUME_BREAKOUT'] and volume_positioning_score > 0.7:
                signal_strength = volume_positioning_score
                confidence_score = min(volume_positioning_score * 1.1, 1.0)
                
                if signal_strength >= self.signal_thresholds['min_signal_strength']:
                    entry_price = current_price
                    stop_loss_price = entry_price * (1 - self.signal_thresholds['default_stop_loss_pct'])
                    take_profit_price = entry_price * (1 + self.signal_thresholds['default_take_profit_pct'])
                    risk_reward_ratio = self.signal_thresholds['default_take_profit_pct'] / self.signal_thresholds['default_stop_loss_pct']
                    
                    signal = TradingSignal(
                        signal_type=SignalType.ENTRY,
                        signal_strength=signal_strength,
                        signal_source=SignalSource.VOLUME_ANALYSIS,
                        entry_price=entry_price,
                        stop_loss_price=stop_loss_price,
                        take_profit_price=take_profit_price,
                        position_size=self.signal_thresholds['max_position_size'] * 0.8,
                        risk_reward_ratio=risk_reward_ratio,
                        confidence_score=confidence_score,
                        metadata={
                            'volume_pattern_type': volume_pattern_type,
                            'volume_positioning_score': volume_positioning_score
                        }
                    )
                    signals.append(signal)
            
        except Exception as e:
            self.logger.error(f"❌ Error generating volume signals: {e}")
        
        return signals
    
    async def _generate_ml_signals(self, symbol: str, timeframe: str, ml_prediction: Dict) -> List[TradingSignal]:
        """Generate signals based on ML predictions"""
        signals = []
        
        try:
            prediction_type = ml_prediction.get('prediction_type', '')
            prediction_value = ml_prediction.get('prediction_value', 0.0)
            confidence_score = ml_prediction.get('confidence_score', 0.5)
            current_price = ml_prediction.get('current_price', 0)
            
            if current_price <= 0 or confidence_score < self.signal_thresholds['min_confidence_score']:
                return signals
            
            # Breakout prediction signals
            if prediction_type == 'breakout' and prediction_value > 0.6:
                signal_strength = prediction_value
                
                if signal_strength >= self.signal_thresholds['min_signal_strength']:
                    entry_price = current_price
                    stop_loss_price = entry_price * (1 - self.signal_thresholds['default_stop_loss_pct'])
                    take_profit_price = entry_price * (1 + self.signal_thresholds['default_take_profit_pct'])
                    risk_reward_ratio = self.signal_thresholds['default_take_profit_pct'] / self.signal_thresholds['default_stop_loss_pct']
                    
                    signal = TradingSignal(
                        signal_type=SignalType.ENTRY,
                        signal_strength=signal_strength,
                        signal_source=SignalSource.ML_PREDICTION,
                        entry_price=entry_price,
                        stop_loss_price=stop_loss_price,
                        take_profit_price=take_profit_price,
                        position_size=self.signal_thresholds['max_position_size'] * 0.9,
                        risk_reward_ratio=risk_reward_ratio,
                        confidence_score=confidence_score,
                        metadata={
                            'prediction_type': prediction_type,
                            'prediction_value': prediction_value,
                            'model_version': ml_prediction.get('model_version', 'unknown')
                        }
                    )
                    signals.append(signal)
            
            # Return prediction signals
            elif prediction_type == 'return' and abs(prediction_value) > 0.02:  # 2% expected return
                signal_strength = min(abs(prediction_value) / 0.05, 1.0)  # Normalize to 0-1
                
                if signal_strength >= self.signal_thresholds['min_signal_strength']:
                    entry_price = current_price
                    
                    if prediction_value > 0:  # Bullish
                        stop_loss_price = entry_price * (1 - self.signal_thresholds['default_stop_loss_pct'])
                        take_profit_price = entry_price * (1 + abs(prediction_value))
                    else:  # Bearish
                        stop_loss_price = entry_price * (1 + self.signal_thresholds['default_stop_loss_pct'])
                        take_profit_price = entry_price * (1 - abs(prediction_value))
                    
                    risk_reward_ratio = abs(prediction_value) / self.signal_thresholds['default_stop_loss_pct']
                    
                    if risk_reward_ratio >= self.signal_thresholds['min_risk_reward_ratio']:
                        signal = TradingSignal(
                            signal_type=SignalType.ENTRY,
                            signal_strength=signal_strength,
                            signal_source=SignalSource.ML_PREDICTION,
                            entry_price=entry_price,
                            stop_loss_price=stop_loss_price,
                            take_profit_price=take_profit_price,
                            position_size=self.signal_thresholds['max_position_size'] * 0.8,
                            risk_reward_ratio=risk_reward_ratio,
                            confidence_score=confidence_score,
                            metadata={
                                'prediction_type': prediction_type,
                                'prediction_value': prediction_value,
                                'model_version': ml_prediction.get('model_version', 'unknown')
                            }
                        )
                        signals.append(signal)
            
        except Exception as e:
            self.logger.error(f"❌ Error generating ML signals: {e}")
        
        return signals
    
    async def _generate_rl_signals(self, symbol: str, timeframe: str, rl_action: Dict) -> List[TradingSignal]:
        """Generate signals based on RL agent actions"""
        signals = []
        
        try:
            action_type = rl_action.get('action_type', 'hold')
            confidence = rl_action.get('confidence', 0.5)
            current_price = rl_action.get('current_price', 0)
            
            if current_price <= 0 or confidence < self.signal_thresholds['min_confidence_score']:
                return signals
            
            if action_type in ['buy', 'sell']:
                signal_strength = confidence
                
                if signal_strength >= self.signal_thresholds['min_signal_strength']:
                    entry_price = current_price
                    stop_loss_price = entry_price * (1 - self.signal_thresholds['default_stop_loss_pct'])
                    take_profit_price = entry_price * (1 + self.signal_thresholds['default_take_profit_pct'])
                    risk_reward_ratio = self.signal_thresholds['default_take_profit_pct'] / self.signal_thresholds['default_stop_loss_pct']
                    
                    signal = TradingSignal(
                        signal_type=SignalType.ENTRY,
                        signal_strength=signal_strength,
                        signal_source=SignalSource.RL_AGENT,
                        entry_price=entry_price,
                        stop_loss_price=stop_loss_price,
                        take_profit_price=take_profit_price,
                        position_size=self.signal_thresholds['max_position_size'] * 0.7,  # Conservative for RL
                        risk_reward_ratio=risk_reward_ratio,
                        confidence_score=confidence,
                        metadata={
                            'action_type': action_type,
                            'agent_id': rl_action.get('agent_id', 'unknown'),
                            'epsilon': rl_action.get('epsilon', 0.1)
                        }
                    )
                    signals.append(signal)
            
        except Exception as e:
            self.logger.error(f"❌ Error generating RL signals: {e}")
        
        return signals
    
    async def _apply_anomaly_filters(self, signals: List[TradingSignal], anomalies: List[Dict]) -> List[TradingSignal]:
        """Filter signals based on detected anomalies"""
        try:
            filtered_signals = []
            
            for signal in signals:
                should_keep = True
                
                for anomaly in anomalies:
                    anomaly_score = anomaly.get('anomaly_score', 0.0)
                    anomaly_type = anomaly.get('anomaly_type', '')
                    severity_level = anomaly.get('severity_level', 'low')
                    
                    # Reduce signal strength for high-severity anomalies
                    if severity_level in ['high', 'critical'] and anomaly_score > 0.7:
                        signal.signal_strength *= 0.7
                        signal.confidence_score *= 0.8
                        signal.position_size *= 0.6
                    
                    # Block signals for manipulation anomalies
                    if anomaly_type == 'manipulation' and anomaly_score > 0.8:
                        should_keep = False
                        break
                
                if should_keep and signal.signal_strength >= self.signal_thresholds['min_signal_strength']:
                    filtered_signals.append(signal)
            
            return filtered_signals
            
        except Exception as e:
            self.logger.error(f"❌ Error applying anomaly filters: {e}")
            return signals
    
    async def optimize_position_parameters(self, symbol: str, timeframe: str, 
                                         current_position: Dict, signals: List[TradingSignal],
                                         volume_analysis: Dict, volatility: float) -> PositionOptimization:
        """Optimize position parameters based on current market conditions and signals"""
        try:
            current_position_size = current_position.get('position_size', 0.0)
            current_stop_loss = current_position.get('stop_loss_price', 0.0)
            current_take_profit = current_position.get('take_profit_price', 0.0)
            current_price = current_position.get('current_price', 0.0)
            
            if current_price <= 0:
                return None
            
            # Calculate optimization factors
            volume_confidence = volume_analysis.get('volume_positioning_score', 0.5)
            signal_strength = max([s.signal_strength for s in signals]) if signals else 0.5
            signal_confidence = max([s.confidence_score for s in signals]) if signals else 0.5
            
            # Position sizing optimization
            base_position_size = self.signal_thresholds['max_position_size']
            
            # Adjust based on volume confidence
            if volume_confidence > 0.8:
                volume_multiplier = self.optimization_params['volume_confidence_multiplier']
            else:
                volume_multiplier = 1.0
            
            # Adjust based on signal strength
            signal_multiplier = 1.0 + (signal_strength - 0.5) * 0.5
            
            # Adjust based on volatility
            volatility_multiplier = 1.0 / (1.0 + volatility * self.optimization_params['volatility_multiplier'])
            
            recommended_position_size = base_position_size * volume_multiplier * signal_multiplier * volatility_multiplier
            recommended_position_size = min(recommended_position_size, self.signal_thresholds['max_position_size'])
            
            # Stop loss optimization
            base_stop_loss_pct = self.signal_thresholds['default_stop_loss_pct']
            
            # Adjust based on volatility
            volatility_adjusted_stop_loss = base_stop_loss_pct * (1.0 + volatility)
            
            # Adjust based on signal confidence
            confidence_adjusted_stop_loss = volatility_adjusted_stop_loss * (1.0 - signal_confidence * 0.3)
            
            recommended_stop_loss = current_price * (1.0 - confidence_adjusted_stop_loss)
            
            # Take profit optimization
            base_take_profit_pct = self.signal_thresholds['default_take_profit_pct']
            
            # Adjust based on signal strength
            strength_adjusted_take_profit = base_take_profit_pct * (1.0 + signal_strength * 0.5)
            
            # Ensure minimum risk/reward ratio
            risk_reward_ratio = strength_adjusted_take_profit / confidence_adjusted_stop_loss
            if risk_reward_ratio < self.signal_thresholds['min_risk_reward_ratio']:
                strength_adjusted_take_profit = confidence_adjusted_stop_loss * self.signal_thresholds['min_risk_reward_ratio']
            
            recommended_take_profit = current_price * (1.0 + strength_adjusted_take_profit)
            
            # Calculate confidence score
            confidence_score = (volume_confidence + signal_confidence) / 2.0
            
            optimization = PositionOptimization(
                optimization_type=OptimizationType.POSITION_SIZING,
                current_position_size=current_position_size,
                recommended_position_size=recommended_position_size,
                current_stop_loss=current_stop_loss,
                recommended_stop_loss=recommended_stop_loss,
                current_take_profit=current_take_profit,
                recommended_take_profit=recommended_take_profit,
                optimization_factors={
                    'volume_confidence': volume_confidence,
                    'signal_strength': signal_strength,
                    'signal_confidence': signal_confidence,
                    'volatility': volatility,
                    'volume_multiplier': volume_multiplier,
                    'signal_multiplier': signal_multiplier,
                    'volatility_multiplier': volatility_multiplier
                },
                confidence_score=confidence_score
            )
            
            # Store optimization in database
            await self._store_position_optimization(symbol, timeframe, optimization)
            
            return optimization
            
        except Exception as e:
            self.logger.error(f"❌ Error optimizing position parameters: {e}")
            return None
    
    async def generate_alerts(self, symbol: str, timeframe: str, 
                            signals: List[TradingSignal], anomalies: List[Dict],
                            volume_analysis: Dict) -> List[Alert]:
        """Generate priority alerts based on analysis results"""
        try:
            alerts = []
            
            # Critical anomaly alerts
            for anomaly in anomalies:
                if anomaly.get('severity_level') == 'critical' and anomaly.get('anomaly_score', 0) > self.alert_thresholds['critical_anomaly_score']:
                    alert = Alert(
                        alert_type="Critical Anomaly Detected",
                        priority_level=AlertPriority.CRITICAL,
                        alert_score=anomaly.get('anomaly_score', 0),
                        contributing_factors={
                            'anomaly_type': anomaly.get('anomaly_type'),
                            'detection_method': anomaly.get('detection_method'),
                            'confidence_score': anomaly.get('confidence_score')
                        },
                        alert_message=f"Critical {anomaly.get('anomaly_type', 'anomaly')} detected for {symbol} with score {anomaly.get('anomaly_score', 0):.2f}"
                    )
                    alerts.append(alert)
            
            # High volume spike alerts
            volume_ratio = volume_analysis.get('volume_ratio', 1.0)
            if volume_ratio > self.alert_thresholds['high_volume_spike']:
                alert = Alert(
                    alert_type="High Volume Spike",
                    priority_level=AlertPriority.HIGH,
                    alert_score=min(volume_ratio / 10.0, 1.0),
                    contributing_factors={
                        'volume_ratio': volume_ratio,
                        'volume_positioning_score': volume_analysis.get('volume_positioning_score', 0)
                    },
                    alert_message=f"High volume spike detected for {symbol}: {volume_ratio:.1f}x average volume"
                )
                alerts.append(alert)
            
            # Low confidence signal alerts
            for signal in signals:
                if signal.confidence_score < self.alert_thresholds['low_confidence_signal']:
                    alert = Alert(
                        alert_type="Low Confidence Signal",
                        priority_level=AlertPriority.MEDIUM,
                        alert_score=1.0 - signal.confidence_score,
                        contributing_factors={
                            'signal_type': signal.signal_type.value,
                            'signal_source': signal.signal_source.value,
                            'signal_strength': signal.signal_strength
                        },
                        alert_message=f"Low confidence {signal.signal_type.value} signal for {symbol}: {signal.confidence_score:.2f}"
                    )
                    alerts.append(alert)
                
                # Poor risk/reward alerts
                if signal.risk_reward_ratio < self.alert_thresholds['poor_risk_reward']:
                    alert = Alert(
                        alert_type="Poor Risk/Reward Ratio",
                        priority_level=AlertPriority.MEDIUM,
                        alert_score=1.0 - (signal.risk_reward_ratio / 3.0),
                        contributing_factors={
                            'risk_reward_ratio': signal.risk_reward_ratio,
                            'signal_type': signal.signal_type.value,
                            'entry_price': signal.entry_price,
                            'stop_loss_price': signal.stop_loss_price,
                            'take_profit_price': signal.take_profit_price
                        },
                        alert_message=f"Poor risk/reward ratio for {symbol}: {signal.risk_reward_ratio:.2f}"
                    )
                    alerts.append(alert)
            
            # Store alerts in database
            if alerts:
                await self._store_alerts(symbol, timeframe, alerts)
            
            return alerts
            
        except Exception as e:
            self.logger.error(f"❌ Error generating alerts: {e}")
            return []
    
    async def _store_trading_signals(self, symbol: str, timeframe: str, signals: List[TradingSignal]):
        """Store trading signals in database"""
        try:
            async with self.db_pool.acquire() as conn:
                for signal in signals:
                    await conn.execute("""
                        INSERT INTO trading_signals (
                            symbol, timeframe, timestamp, signal_type, signal_strength,
                            signal_source, entry_price, stop_loss_price, take_profit_price,
                            position_size, risk_reward_ratio, confidence_score, signal_metadata
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
                    """,
                    symbol, timeframe, datetime.now(), signal.signal_type.value,
                    signal.signal_strength, signal.signal_source.value, signal.entry_price,
                    signal.stop_loss_price, signal.take_profit_price, signal.position_size,
                    signal.risk_reward_ratio, signal.confidence_score, signal.metadata
                    )
            
            self.logger.info(f"💾 Stored {len(signals)} trading signals for {symbol}")
            
        except Exception as e:
            self.logger.error(f"❌ Error storing trading signals: {e}")
    
    async def _store_position_optimization(self, symbol: str, timeframe: str, optimization: PositionOptimization):
        """Store position optimization in database"""
        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO position_optimization (
                        symbol, timeframe, timestamp, optimization_type,
                        current_position_size, recommended_position_size,
                        current_stop_loss, recommended_stop_loss,
                        current_take_profit, recommended_take_profit,
                        optimization_factors, confidence_score
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                """,
                symbol, timeframe, datetime.now(), optimization.optimization_type.value,
                optimization.current_position_size, optimization.recommended_position_size,
                optimization.current_stop_loss, optimization.recommended_stop_loss,
                optimization.current_take_profit, optimization.recommended_take_profit,
                optimization.optimization_factors, optimization.confidence_score
                )
            
            self.logger.info(f"💾 Stored position optimization for {symbol}")
            
        except Exception as e:
            self.logger.error(f"❌ Error storing position optimization: {e}")
    
    async def _store_alerts(self, symbol: str, timeframe: str, alerts: List[Alert]):
        """Store alerts in database"""
        try:
            async with self.db_pool.acquire() as conn:
                for alert in alerts:
                    await conn.execute("""
                        INSERT INTO alert_priority (
                            symbol, timeframe, timestamp, alert_type, priority_level,
                            alert_score, contributing_factors, alert_message
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    """,
                    symbol, timeframe, datetime.now(), alert.alert_type,
                    alert.priority_level.value, alert.alert_score,
                    alert.contributing_factors, alert.alert_message
                    )
            
            self.logger.info(f"💾 Stored {len(alerts)} alerts for {symbol}")
            
        except Exception as e:
            self.logger.error(f"❌ Error storing alerts: {e}")
    
    async def get_recent_signals(self, symbol: str, timeframe: str, hours: int = 24) -> List[Dict]:
        """Get recent trading signals for a symbol"""
        try:
            async with self.db_pool.acquire() as conn:
                rows = await conn.fetch("""
                    SELECT * FROM trading_signals 
                    WHERE symbol = $1 AND timeframe = $2 
                    AND timestamp >= NOW() - INTERVAL '1 hour' * $3
                    ORDER BY timestamp DESC
                """, symbol, timeframe, hours)
                
                return [dict(row) for row in rows]
                
        except Exception as e:
            self.logger.error(f"❌ Error fetching recent signals: {e}")
            return []
    
    async def get_high_priority_alerts(self, hours: int = 24) -> List[Dict]:
        """Get high priority alerts"""
        try:
            async with self.db_pool.acquire() as conn:
                rows = await conn.fetch("""
                    SELECT * FROM alert_priority 
                    WHERE priority_level IN ('high', 'critical')
                    AND timestamp >= NOW() - INTERVAL '1 hour' * $1
                    ORDER BY alert_score DESC, timestamp DESC
                """, hours)
                
                return [dict(row) for row in rows]
                
        except Exception as e:
            self.logger.error(f"❌ Error fetching high priority alerts: {e}")
            return []
