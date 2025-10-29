"""
Advanced Analytics Engine for AlphaPulse
Provides real-time market insights, predictive analytics, and advanced reporting
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Tuple, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass
from collections import defaultdict, deque
import json
import numpy as np
import pandas as pd

# Import our components
try:
    from ..src.data.enhanced_real_time_pipeline import EnhancedRealTimePipeline
    from ..src.database.connection import TimescaleDBConnection
    from ..src.strategies.advanced_ml_strategy import AdvancedMLStrategy
except ImportError as e:
    logging.warning(f"Some imports not available: {e}")
    EnhancedRealTimePipeline = None
    TimescaleDBConnection = None
    AdvancedMLStrategy = None

logger = logging.getLogger(__name__)

@dataclass
class MarketRegime:
    """Market regime classification"""
    symbol: str
    timestamp: datetime
    regime_type: str  # 'trending', 'ranging', 'volatile', 'consolidating'
    confidence: float
    duration: timedelta
    characteristics: Dict[str, Any]
    metadata: Dict[str, Any]

@dataclass
class PredictiveSignal:
    """Predictive trading signal"""
    symbol: str
    timestamp: datetime
    signal_type: str  # 'price_prediction', 'volatility_forecast', 'regime_change'
    confidence: float
    time_horizon: timedelta
    prediction_value: float
    metadata: Dict[str, Any]

@dataclass
class MarketInsight:
    """Market insight and analysis"""
    symbol: str
    timestamp: datetime
    insight_type: str  # 'liquidity_analysis', 'correlation_shift', 'anomaly_detection'
    severity: str  # 'low', 'medium', 'high', 'critical'
    description: str
    actionable: bool
    recommendations: List[str]
    metadata: Dict[str, Any]

class AdvancedAnalyticsEngine:
    """Advanced analytics engine for real-time market insights"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logger
        
        # Analytics configuration
        self.analysis_window = self.config.get('analysis_window', 1000)
        self.update_frequency = self.config.get('update_frequency', 5.0)  # seconds
        self.symbols = self.config.get('symbols', ['BTC/USDT', 'ETH/USDT', 'ADA/USDT'])
        self.enable_predictions = self.config.get('enable_predictions', True)
        self.enable_regime_detection = self.config.get('enable_regime_detection', True)
        self.enable_anomaly_detection = self.config.get('enable_anomaly_detection', True)
        
        # Component references
        self.real_time_pipeline = None
        self.db_connection = None
        self.ml_strategy = None
        
        # Analytics state
        self.market_regimes = defaultdict(dict)  # symbol -> current_regime
        self.prediction_history = defaultdict(deque)  # symbol -> predictions
        self.insight_history = defaultdict(deque)  # symbol -> insights
        self.anomaly_scores = defaultdict(deque)  # symbol -> anomaly_scores
        
        # Performance tracking
        self.stats = {
            'total_analyses': 0,
            'regime_detections': 0,
            'predictions_generated': 0,
            'insights_generated': 0,
            'anomalies_detected': 0,
            'last_analysis': None,
            'processing_times': deque(maxlen=100)
        }
        
        # Callbacks
        self.analytics_callbacks = defaultdict(list)  # event_type -> [callback]
        
        # Initialize components
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize analytics components"""
        try:
            # Initialize real-time pipeline if available
            if EnhancedRealTimePipeline:
                pipeline_config = {
                    'symbols': self.symbols,
                    'exchanges': ['binance', 'okx'],
                    'update_frequency': 1.0,
                    'analysis_enabled': True
                }
                self.real_time_pipeline = EnhancedRealTimePipeline(pipeline_config)
                self.logger.info("Real-time pipeline initialized for analytics")
            
            # Initialize database connection if available
            if TimescaleDBConnection:
                db_config = self.config.get('database', {})
                self.db_connection = TimescaleDBConnection(db_config)
                self.logger.info("Database connection initialized for analytics")
            
            # Initialize ML strategy if available
            if AdvancedMLStrategy:
                strategy_config = {
                    'strategy_type': 'HYBRID',
                    'prediction_threshold': 0.6,
                    'confidence_threshold': 0.7
                }
                self.ml_strategy = AdvancedMLStrategy(strategy_config)
                self.logger.info("ML strategy initialized for analytics")
                
        except Exception as e:
            self.logger.error(f"Failed to initialize analytics components: {e}")
    
    async def initialize(self):
        """Initialize the analytics engine"""
        try:
            self.logger.info("Initializing Advanced Analytics Engine...")
            
            # Initialize real-time pipeline
            if self.real_time_pipeline:
                await self.real_time_pipeline.initialize()
                
                # Add callbacks for analytics
                self.real_time_pipeline.add_callback('order_book', self._on_order_book_update)
                self.real_time_pipeline.add_callback('market_data', self._on_market_data_update)
                self.real_time_pipeline.add_callback('market_depth_analysis', self._on_market_depth_update)
            
            # Initialize database connection
            if self.db_connection:
                await self.db_connection.initialize()
            
            # Initialize ML strategy
            if self.ml_strategy:
                await self.ml_strategy.initialize()
            
            self.logger.info("Advanced Analytics Engine initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize analytics engine: {e}")
            raise
    
    async def start_analytics(self):
        """Start the analytics engine"""
        try:
            self.logger.info("Starting Advanced Analytics Engine...")
            
            # Start real-time pipeline
            if self.real_time_pipeline:
                await self.real_time_pipeline.start()
            
            # Start analytics loop
            self.analytics_task = asyncio.create_task(self._analytics_loop())
            
            self.logger.info("✅ Advanced Analytics Engine started successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to start analytics engine: {e}")
            raise
    
    async def stop_analytics(self):
        """Stop the analytics engine"""
        try:
            self.logger.info("Stopping Advanced Analytics Engine...")
            
            # Stop real-time pipeline
            if self.real_time_pipeline:
                await self.real_time_pipeline.stop()
            
            # Cancel analytics task
            if hasattr(self, 'analytics_task'):
                self.analytics_task.cancel()
                try:
                    await self.analytics_task
                except asyncio.CancelledError:
                    pass
            
            self.logger.info("✅ Advanced Analytics Engine stopped successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to stop analytics engine: {e}")
    
    async def _analytics_loop(self):
        """Main analytics loop"""
        try:
            while True:
                start_time = time.time()
                
                # Perform analytics for all symbols
                for symbol in self.symbols:
                    try:
                        await self._analyze_symbol(symbol)
                    except Exception as e:
                        self.logger.error(f"Error analyzing symbol {symbol}: {e}")
                
                # Update statistics
                self.stats['last_analysis'] = datetime.now()
                processing_time = time.time() - start_time
                self.stats['processing_times'].append(processing_time)
                
                # Wait for next analysis cycle
                await asyncio.sleep(self.update_frequency)
                
        except asyncio.CancelledError:
            self.logger.info("Analytics loop cancelled")
        except Exception as e:
            self.logger.error(f"Analytics loop error: {e}")
    
    async def _analyze_symbol(self, symbol: str):
        """Perform comprehensive analysis for a symbol"""
        try:
            # Market regime detection
            if self.enable_regime_detection:
                regime = await self._detect_market_regime(symbol)
                if regime:
                    self.market_regimes[symbol] = regime
                    self.stats['regime_detections'] += 1
                    await self._trigger_callbacks('market_regime', regime)
            
            # Predictive analytics
            if self.enable_predictions:
                predictions = await self._generate_predictions(symbol)
                for prediction in predictions:
                    self.prediction_history[symbol].append(prediction)
                    self.stats['predictions_generated'] += 1
                    await self._trigger_callbacks('prediction', prediction)
            
            # Anomaly detection
            if self.enable_anomaly_detection:
                anomaly_score = await self._detect_anomalies(symbol)
                if anomaly_score > 0.7:  # High anomaly threshold
                    insight = await self._create_anomaly_insight(symbol, anomaly_score)
                    self.insight_history[symbol].append(insight)
                    self.stats['anomalies_detected'] += 1
                    await self._trigger_callbacks('anomaly', insight)
            
            # Market insights generation
            insights = await self._generate_market_insights(symbol)
            for insight in insights:
                self.insight_history[symbol].append(insight)
                self.stats['insights_generated'] += 1
                await self._trigger_callbacks('insight', insight)
            
            # Update statistics
            self.stats['total_analyses'] += 1
            
        except Exception as e:
            self.logger.error(f"Error analyzing symbol {symbol}: {e}")
    
    async def _detect_market_regime(self, symbol: str) -> Optional[MarketRegime]:
        """Detect current market regime for a symbol"""
        try:
            # Get recent market data
            if not self.real_time_pipeline:
                return None
            
            market_data = self.real_time_pipeline.get_symbol_data(symbol, 'market_data')
            if len(market_data) < 50:
                return None
            
            # Extract price and volume data
            prices = [data.price for data in market_data[-50:]]
            volumes = [data.volume for data in market_data[-50:]]
            
            # Calculate regime characteristics
            price_returns = np.diff(np.log(prices))
            volatility = np.std(price_returns) * np.sqrt(252)
            
            # Calculate trend strength
            sma_short = np.mean(prices[-10:])
            sma_long = np.mean(prices[-50:])
            trend_strength = abs(sma_short - sma_long) / sma_long
            
            # Calculate volume characteristics
            volume_mean = np.mean(volumes)
            volume_std = np.std(volumes)
            volume_trend = np.corrcoef(range(len(volumes)), volumes)[0, 1]
            
            # Determine regime type
            regime_type = 'consolidating'
            confidence = 0.5
            
            if volatility > 0.8:  # High volatility
                regime_type = 'volatile'
                confidence = min(0.9, volatility / 1.2)
            elif trend_strength > 0.05:  # Strong trend
                regime_type = 'trending'
                confidence = min(0.9, trend_strength / 0.1)
            elif abs(volume_trend) > 0.3:  # Volume trend
                regime_type = 'ranging'
                confidence = min(0.8, abs(volume_trend) / 0.5)
            
            # Calculate regime duration
            current_regime = self.market_regimes.get(symbol)
            if current_regime and current_regime.regime_type == regime_type:
                duration = datetime.now() - current_regime.timestamp
            else:
                duration = timedelta(0)
            
            # Create regime object
            regime = MarketRegime(
                symbol=symbol,
                timestamp=datetime.now(),
                regime_type=regime_type,
                confidence=confidence,
                duration=duration,
                characteristics={
                    'volatility': volatility,
                    'trend_strength': trend_strength,
                    'volume_trend': volume_trend,
                    'price_range': (min(prices), max(prices)),
                    'volume_profile': {
                        'mean': volume_mean,
                        'std': volume_std,
                        'trend': volume_trend
                    }
                },
                metadata={
                    'analysis_window': len(prices),
                    'last_update': datetime.now().isoformat()
                }
            )
            
            return regime
            
        except Exception as e:
            self.logger.error(f"Error detecting market regime for {symbol}: {e}")
            return None
    
    async def _generate_predictions(self, symbol: str) -> List[PredictiveSignal]:
        """Generate predictive signals for a symbol"""
        try:
            predictions = []
            
            # Get recent market data
            if not self.real_time_pipeline:
                return predictions
            
            market_data = self.real_time_pipeline.get_symbol_data(symbol, 'market_data')
            if len(market_data) < 100:
                return predictions
            
            # Price prediction using simple linear regression
            price_prediction = await self._predict_price(symbol, market_data)
            if price_prediction:
                predictions.append(price_prediction)
            
            # Volatility forecast
            volatility_forecast = await self._forecast_volatility(symbol, market_data)
            if volatility_forecast:
                predictions.append(volatility_forecast)
            
            # Regime change prediction
            regime_prediction = await self._predict_regime_change(symbol)
            if regime_prediction:
                predictions.append(regime_prediction)
            
            return predictions
            
        except Exception as e:
            self.logger.error(f"Error generating predictions for {symbol}: {e}")
            return []
    
    async def _predict_price(self, symbol: str, market_data: List) -> Optional[PredictiveSignal]:
        """Predict future price using simple linear regression"""
        try:
            if len(market_data) < 50:
                return None
            
            # Extract price data
            prices = [data.price for data in market_data[-50:]]
            timestamps = [i for i in range(len(prices))]
            
            # Simple linear regression
            x = np.array(timestamps)
            y = np.array(prices)
            
            # Calculate regression coefficients
            n = len(x)
            sum_x = np.sum(x)
            sum_y = np.sum(y)
            sum_xy = np.sum(x * y)
            sum_x2 = np.sum(x ** 2)
            
            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
            intercept = (sum_y - slope * sum_x) / n
            
            # Predict next 5 periods
            future_timestamps = np.array([n, n+1, n+2, n+3, n+4])
            future_prices = slope * future_timestamps + intercept
            
            # Calculate confidence based on R-squared
            y_pred = slope * x + intercept
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            confidence = min(0.9, max(0.1, r_squared))
            
            # Create prediction signal
            prediction = PredictiveSignal(
                symbol=symbol,
                timestamp=datetime.now(),
                signal_type='price_prediction',
                confidence=confidence,
                time_horizon=timedelta(minutes=5),
                prediction_value=future_prices[0],
                metadata={
                    'slope': slope,
                    'intercept': intercept,
                    'r_squared': r_squared,
                    'future_prices': future_prices.tolist(),
                    'current_price': prices[-1],
                    'prediction_method': 'linear_regression'
                }
            )
            
            return prediction
            
        except Exception as e:
            self.logger.error(f"Error predicting price for {symbol}: {e}")
            return None
    
    async def _forecast_volatility(self, symbol: str, market_data: List) -> Optional[PredictiveSignal]:
        """Forecast future volatility"""
        try:
            if len(market_data) < 100:
                return None
            
            # Extract price data
            prices = [data.price for data in market_data[-100:]]
            
            # Calculate historical volatility
            returns = np.diff(np.log(prices))
            historical_vol = np.std(returns) * np.sqrt(252)
            
            # Simple volatility forecast using exponential smoothing
            alpha = 0.1  # Smoothing factor
            forecast_vol = historical_vol
            
            # Apply some trend to volatility (simplified)
            vol_trend = np.corrcoef(range(len(returns)), np.abs(returns))[0, 1]
            if not np.isnan(vol_trend):
                forecast_vol *= (1 + vol_trend * 0.1)
            
            # Calculate confidence based on volatility stability
            vol_stability = 1 - np.std(np.abs(returns[-20:])) / np.mean(np.abs(returns[-20:]))
            confidence = min(0.8, max(0.2, vol_stability))
            
            # Create volatility forecast
            forecast = PredictiveSignal(
                symbol=symbol,
                timestamp=datetime.now(),
                signal_type='volatility_forecast',
                confidence=confidence,
                time_horizon=timedelta(hours=1),
                prediction_value=forecast_vol,
                metadata={
                    'historical_volatility': historical_vol,
                    'volatility_trend': vol_trend,
                    'volatility_stability': vol_stability,
                    'forecast_method': 'exponential_smoothing'
                }
            )
            
            return forecast
            
        except Exception as e:
            self.logger.error(f"Error forecasting volatility for {symbol}: {e}")
            return None
    
    async def _predict_regime_change(self, symbol: str) -> Optional[PredictiveSignal]:
        """Predict regime change"""
        try:
            current_regime = self.market_regimes.get(symbol)
            if not current_regime:
                return None
            
            # Simple regime change prediction based on duration and characteristics
            regime_duration = current_regime.duration
            confidence = 0.5
            
            # Predict regime change based on duration
            if regime_duration > timedelta(hours=24):
                # Long regime, higher chance of change
                confidence = min(0.8, confidence + 0.3)
            
            # Predict based on regime characteristics
            if current_regime.regime_type == 'volatile':
                # Volatile regimes tend to change quickly
                confidence = min(0.9, confidence + 0.2)
            elif current_regime.regime_type == 'trending':
                # Trending regimes can be stable
                confidence = max(0.3, confidence - 0.1)
            
            # Create regime change prediction
            prediction = PredictiveSignal(
                symbol=symbol,
                timestamp=datetime.now(),
                signal_type='regime_change',
                confidence=confidence,
                time_horizon=timedelta(hours=6),
                prediction_value=confidence,  # Probability of regime change
                metadata={
                    'current_regime': current_regime.regime_type,
                    'regime_duration': str(regime_duration),
                    'regime_confidence': current_regime.confidence,
                    'prediction_method': 'duration_and_characteristics'
                }
            )
            
            return prediction
            
        except Exception as e:
            self.logger.error(f"Error predicting regime change for {symbol}: {e}")
            return None
    
    async def _detect_anomalies(self, symbol: str) -> float:
        """Detect anomalies in market data"""
        try:
            if not self.real_time_pipeline:
                return 0.0
            
            # Get recent market data
            market_data = self.real_time_pipeline.get_symbol_data(symbol, 'market_data')
            if len(market_data) < 50:
                return 0.0
            
            # Extract price and volume data
            prices = [data.price for data in market_data[-50:]]
            volumes = [data.volume for data in market_data[-50:]]
            
            # Calculate price anomalies
            price_returns = np.diff(np.log(prices))
            price_mean = np.mean(price_returns)
            price_std = np.std(price_returns)
            
            # Calculate volume anomalies
            volume_mean = np.mean(volumes)
            volume_std = np.std(volumes)
            
            # Detect outliers using z-score
            price_anomalies = []
            for i, ret in enumerate(price_returns):
                z_score = abs(ret - price_mean) / price_std if price_std > 0 else 0
                if z_score > 2.5:  # Significant outlier
                    price_anomalies.append(z_score)
            
            volume_anomalies = []
            for vol in volumes:
                z_score = abs(vol - volume_mean) / volume_std if volume_std > 0 else 0
                if z_score > 2.0:  # Volume outlier
                    volume_anomalies.append(z_score)
            
            # Calculate overall anomaly score
            price_anomaly_score = np.mean(price_anomalies) if price_anomalies else 0
            volume_anomaly_score = np.mean(volume_anomalies) if volume_anomalies else 0
            
            # Combine scores (price anomalies are more important)
            overall_score = (price_anomaly_score * 0.7 + volume_anomaly_score * 0.3) / 3.0
            overall_score = min(1.0, max(0.0, overall_score))
            
            # Store anomaly score
            self.anomaly_scores[symbol].append(overall_score)
            if len(self.anomaly_scores[symbol]) > 100:
                self.anomaly_scores[symbol].popleft()
            
            return overall_score
            
        except Exception as e:
            self.logger.error(f"Error detecting anomalies for {symbol}: {e}")
            return 0.0
    
    async def _create_anomaly_insight(self, symbol: str, anomaly_score: float) -> MarketInsight:
        """Create insight from anomaly detection"""
        try:
            # Determine severity
            if anomaly_score > 0.9:
                severity = 'critical'
            elif anomaly_score > 0.7:
                severity = 'high'
            elif anomaly_score > 0.5:
                severity = 'medium'
            else:
                severity = 'low'
            
            # Generate recommendations
            recommendations = []
            if anomaly_score > 0.7:
                recommendations.append("Consider reducing position size due to high market volatility")
                recommendations.append("Monitor for potential trend reversal or breakout")
            elif anomaly_score > 0.5:
                recommendations.append("Exercise caution with new positions")
                recommendations.append("Monitor market conditions closely")
            
            # Create insight
            insight = MarketInsight(
                symbol=symbol,
                timestamp=datetime.now(),
                insight_type='anomaly_detection',
                severity=severity,
                description=f"Unusual market activity detected with anomaly score {anomaly_score:.3f}",
                actionable=len(recommendations) > 0,
                recommendations=recommendations,
                metadata={
                    'anomaly_score': anomaly_score,
                    'detection_method': 'z_score_outlier',
                    'analysis_window': 50
                }
            )
            
            return insight
            
        except Exception as e:
            self.logger.error(f"Error creating anomaly insight for {symbol}: {e}")
            return None
    
    async def _generate_market_insights(self, symbol: str) -> List[MarketInsight]:
        """Generate general market insights"""
        try:
            insights = []
            
            # Get current market regime
            current_regime = self.market_regimes.get(symbol)
            if current_regime:
                # Create regime-based insight
                regime_insight = MarketInsight(
                    symbol=symbol,
                    timestamp=datetime.now(),
                    insight_type='regime_analysis',
                    severity='low',
                    description=f"Market currently in {current_regime.regime_type} regime with {current_regime.confidence:.1%} confidence",
                    actionable=True,
                    recommendations=[
                        f"Adapt strategy for {current_regime.regime_type} market conditions",
                        "Monitor regime duration for potential changes"
                    ],
                    metadata={
                        'regime_type': current_regime.regime_type,
                        'regime_confidence': current_regime.confidence,
                        'regime_duration': str(current_regime.duration)
                    }
                )
                insights.append(regime_insight)
            
            # Get recent predictions
            recent_predictions = list(self.prediction_history.get(symbol, []))[-3:]
            if recent_predictions:
                # Create prediction-based insight
                pred_insight = MarketInsight(
                    symbol=symbol,
                    timestamp=datetime.now(),
                    insight_type='prediction_summary',
                    severity='medium',
                    description=f"Generated {len(recent_predictions)} predictions with average confidence {np.mean([p.confidence for p in recent_predictions]):.1%}",
                    actionable=True,
                    recommendations=[
                        "Review recent predictions for trading opportunities",
                        "Monitor prediction accuracy over time"
                    ],
                    metadata={
                        'prediction_count': len(recent_predictions),
                        'average_confidence': np.mean([p.confidence for p in recent_predictions]),
                        'prediction_types': [p.signal_type for p in recent_predictions]
                    }
                )
                insights.append(pred_insight)
            
            return insights
            
        except Exception as e:
            self.logger.error(f"Error generating market insights for {symbol}: {e}")
            return []
    
    # Callback methods for real-time pipeline
    async def _on_order_book_update(self, order_book):
        """Handle order book updates"""
        # Order book updates are handled in the main analytics loop
        pass
    
    async def _on_market_data_update(self, market_data):
        """Handle market data updates"""
        # Market data updates are handled in the main analytics loop
        pass
    
    async def _on_market_depth_update(self, market_depth):
        """Handle market depth updates"""
        # Market depth updates are handled in the main analytics loop
        pass
    
    # Public methods
    def add_callback(self, event_type: str, callback: Callable):
        """Add callback for analytics events"""
        self.analytics_callbacks[event_type].append(callback)
        self.logger.info(f"Added callback for {event_type} events")
    
    async def _trigger_callbacks(self, event_type: str, data: Any):
        """Trigger callbacks for analytics events"""
        callbacks = self.analytics_callbacks.get(event_type, [])
        
        for callback in callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(data)
                else:
                    callback(data)
            except Exception as e:
                self.logger.error(f"Error in {event_type} callback: {e}")
    
    def get_analytics_statistics(self) -> Dict[str, Any]:
        """Get analytics engine statistics"""
        return {
            'stats': self.stats,
            'market_regimes': {
                symbol: regime.regime_type for symbol, regime in self.market_regimes.items()
            },
            'prediction_counts': {
                symbol: len(predictions) for symbol, predictions in self.prediction_history.items()
            },
            'insight_counts': {
                symbol: len(insights) for symbol, insights in self.insight_history.items()
            },
            'anomaly_scores': {
                symbol: np.mean(scores) if scores else 0.0 
                for symbol, scores in self.anomaly_scores.items()
            }
        }
    
    def get_symbol_analytics(self, symbol: str) -> Dict[str, Any]:
        """Get comprehensive analytics for a symbol"""
        try:
            return {
                'market_regime': self.market_regimes.get(symbol),
                'recent_predictions': list(self.prediction_history.get(symbol, []))[-10:],
                'recent_insights': list(self.insight_history.get(symbol, []))[-10:],
                'anomaly_score': np.mean(self.anomaly_scores.get(symbol, [])) if self.anomaly_scores.get(symbol) else 0.0,
                'analytics_summary': {
                    'total_predictions': len(self.prediction_history.get(symbol, [])),
                    'total_insights': len(self.insight_history.get(symbol, [])),
                    'regime_stability': self._calculate_regime_stability(symbol)
                }
            }
        except Exception as e:
            self.logger.error(f"Error getting analytics for {symbol}: {e}")
            return {}
    
    def _calculate_regime_stability(self, symbol: str) -> float:
        """Calculate regime stability for a symbol"""
        try:
            current_regime = self.market_regimes.get(symbol)
            if not current_regime:
                return 0.0
            
            # Stability based on duration and confidence
            duration_hours = current_regime.duration.total_seconds() / 3600
            stability = min(1.0, duration_hours / 24) * current_regime.confidence
            
            return stability
            
        except Exception as e:
            self.logger.error(f"Error calculating regime stability for {symbol}: {e}")
            return 0.0
    
    async def close(self):
        """Close the analytics engine"""
        try:
            # Stop analytics
            await self.stop_analytics()
            
            # Close components
            if self.real_time_pipeline:
                await self.real_time_pipeline.close()
            
            if self.db_connection:
                await self.db_connection.close()
            
            if self.ml_strategy:
                await self.ml_strategy.close()
            
            self.logger.info("Advanced Analytics Engine closed successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to close analytics engine: {e}")
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()
