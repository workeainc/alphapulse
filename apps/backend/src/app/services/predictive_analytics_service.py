#!/usr/bin/env python3
"""
Predictive Analytics Service for AlphaPulse
Advanced predictive models for liquidation prediction, order book forecasting, and market microstructure analysis
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from collections import deque, defaultdict
import json
import pickle
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, classification_report
import lightgbm as lgb
import joblib
import os
import hashlib
from sqlalchemy import create_engine, text

logger = logging.getLogger(__name__)

@dataclass
class LiquidationPrediction:
    """Liquidation prediction result"""
    symbol: str
    timestamp: datetime
    prediction_horizon: int  # minutes
    liquidation_probability: float
    expected_liquidation_volume: float
    confidence_score: float
    risk_level: str  # 'low', 'medium', 'high', 'critical'
    factors: Dict[str, float]
    metadata: Dict[str, Any]

@dataclass
class OrderBookForecast:
    """Order book forecasting result"""
    symbol: str
    timestamp: datetime
    forecast_horizon: int  # minutes
    predicted_spread: float
    predicted_depth: Dict[str, float]
    predicted_imbalance: float
    confidence_score: float
    volatility_forecast: float
    metadata: Dict[str, Any]

@dataclass
class MarketMicrostructureAnalysis:
    """Market microstructure analysis result"""
    symbol: str
    timestamp: datetime
    order_flow_toxicity: float
    price_impact: float
    market_resilience: float
    information_asymmetry: float
    market_efficiency: float
    microstructure_score: float
    recommendations: List[str]
    metadata: Dict[str, Any]

class PredictiveAnalyticsService:
    """Advanced predictive analytics service for trading"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logger
        
        # Service configuration
        self.models_dir = self.config.get('models_dir', 'models/predictive')
        self.update_frequency = self.config.get('update_frequency', 60)  # seconds
        self.prediction_horizons = self.config.get('prediction_horizons', [5, 15, 30, 60])  # minutes
        self.confidence_threshold = self.config.get('confidence_threshold', 0.7)
        
        # Model storage
        self.liquidation_models = {}
        self.orderbook_models = {}
        self.microstructure_models = {}
        
        # Data buffers for training
        self.liquidation_data = defaultdict(lambda: deque(maxlen=10000))
        self.orderbook_data = defaultdict(lambda: deque(maxlen=10000))
        self.market_data = defaultdict(lambda: deque(maxlen=10000))
        
        # Feature engineering
        self.feature_engineers = {
            'liquidation': self._engineer_liquidation_features,
            'orderbook': self._engineer_orderbook_features,
            'microstructure': self._engineer_microstructure_features
        }
        
        # Database connection
        self.database_url = self.config.get('database_url', os.getenv("DATABASE_URL", "postgresql://alpha_emon:Emon_%4017711@localhost:5432/alphapulse"))
        self.engine = create_engine(self.database_url)
        
        # Model versioning
        self.active_model_versions = {}
        
        # LightGBM specific configuration
        self.lgb_params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': 0
        }
        
        # Performance metrics
        self.performance_metrics = {
            'predictions_made': 0,
            'accuracy_scores': deque(maxlen=1000),
            'model_retraining_count': 0,
            'last_retraining': None,
            'feature_importance': {},
            'model_versions': {},
            'training_history': deque(maxlen=100)
        }
        
        # Initialize models directory
        os.makedirs(self.models_dir, exist_ok=True)
        
    async def initialize(self):
        """Initialize the predictive analytics service"""
        try:
            self.logger.info("Initializing Predictive Analytics Service...")
            
            # Load existing models
            await self._load_models()
            
            # Initialize feature scalers
            self.scalers = {
                'liquidation': StandardScaler(),
                'orderbook': StandardScaler(),
                'microstructure': StandardScaler()
            }
            
            # Start background tasks
            self.retraining_task = asyncio.create_task(self._periodic_retraining())
            self.monitoring_task = asyncio.create_task(self._performance_monitoring())
            
            self.logger.info("Predictive Analytics Service initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Predictive Analytics Service: {e}")
            raise

    async def predict_liquidations(self, symbol: str, market_data: Dict[str, Any]) -> LiquidationPrediction:
        """Predict liquidation events for a symbol using LightGBM"""
        try:
            timestamp = datetime.now()
            
            # Engineer features
            features = await self._engineer_liquidation_features(symbol, market_data)
            
            # Load or train model if not available
            model_name = f"liquidation_predictor_{symbol}"
            if model_name not in self.liquidation_models:
                await self._load_or_train_liquidation_model(symbol)
            
            # Get predictions for different horizons
            predictions = {}
            feature_importance = {}
            
            for horizon in self.prediction_horizons:
                model_key = f"{model_name}_{horizon}"
                if model_key in self.liquidation_models:
                    model = self.liquidation_models[model_key]
                    
                    # Make prediction with LightGBM
                    features_array = np.array(features).reshape(1, -1)
                    prediction_proba = model.predict(features_array)[0]
                    predictions[horizon] = max(0, min(1, prediction_proba))
                    
                    # Get feature importance
                    if hasattr(model, 'feature_importance'):
                        feature_importance[horizon] = model.feature_importance().tolist()
                else:
                    # Fallback prediction if model not available
                    predictions[horizon] = self._fallback_liquidation_prediction(features)
            
            # Aggregate predictions with weighted average (shorter horizons get higher weight)
            if predictions:
                weights = [1.0 / h for h in predictions.keys()]
                total_weight = sum(weights)
                avg_probability = sum(pred * weight for pred, weight in zip(predictions.values(), weights)) / total_weight
                confidence_score = 1 - np.std(list(predictions.values())) if len(predictions) > 1 else 0.7
            else:
                avg_probability = 0.5
                confidence_score = 0.3
            
            # Calculate risk level based on probability and market conditions
            risk_level = self._calculate_risk_level(avg_probability, confidence_score)
            
            # Estimate expected liquidation volume
            expected_volume = self._estimate_liquidation_volume(symbol, avg_probability, market_data)
            
            # Extract key factors influencing prediction
            factors = await self._extract_prediction_factors(features, market_data)
            
            # Store prediction in database for model performance tracking
            await self._store_prediction_result(symbol, model_name, avg_probability, confidence_score, features, timestamp)
            
            # Update performance metrics
            self.performance_metrics['predictions_made'] += 1
            
            return LiquidationPrediction(
                symbol=symbol,
                timestamp=datetime.now(),
                prediction_horizon=max(self.prediction_horizons),
                liquidation_probability=avg_probability,
                expected_liquidation_volume=expected_volume,
                confidence_score=confidence_score,
                risk_level=risk_level,
                factors=self._extract_liquidation_factors(features),
                metadata={'horizon_predictions': predictions}
            )
            
        except Exception as e:
            self.logger.error(f"Error predicting liquidations for {symbol}: {e}")
            return self._get_default_liquidation_prediction(symbol)

    async def forecast_order_book(self, symbol: str, order_book_data: Dict[str, Any]) -> OrderBookForecast:
        """Forecast order book changes"""
        try:
            # Engineer features
            features = await self._engineer_orderbook_features(symbol, order_book_data)
            
            # Get model for symbol
            model_key = symbol
            if model_key in self.orderbook_models:
                model = self.orderbook_models[model_key]
                scaler = self.scalers['orderbook']
                
                # Scale features
                features_scaled = scaler.transform([features])
                
                # Make prediction
                prediction = model.predict(features_scaled)[0]
                
                # Parse prediction (assuming model outputs [spread, bid_depth, ask_depth, imbalance, volatility])
                predicted_spread = prediction[0]
                predicted_depth = {
                    'bid': prediction[1],
                    'ask': prediction[2]
                }
                predicted_imbalance = prediction[3]
                volatility_forecast = prediction[4]
                
                # Calculate confidence based on feature stability
                confidence_score = self._calculate_orderbook_confidence(features)
                
                # Store for performance tracking
                self.orderbook_data[symbol].append({
                    'timestamp': datetime.now(),
                    'prediction': prediction,
                    'actual': None,
                    'features': features
                })
                
                return OrderBookForecast(
                    symbol=symbol,
                    timestamp=datetime.now(),
                    forecast_horizon=15,  # 15 minutes
                    predicted_spread=predicted_spread,
                    predicted_depth=predicted_depth,
                    predicted_imbalance=predicted_imbalance,
                    confidence_score=confidence_score,
                    volatility_forecast=volatility_forecast,
                    metadata={'raw_prediction': prediction.tolist()}
                )
            
            return self._get_default_orderbook_forecast(symbol)
            
        except Exception as e:
            self.logger.error(f"Error forecasting order book for {symbol}: {e}")
            return self._get_default_orderbook_forecast(symbol)

    async def analyze_microstructure(self, symbol: str, market_data: Dict[str, Any]) -> MarketMicrostructureAnalysis:
        """Analyze market microstructure"""
        try:
            # Engineer features
            features = await self._engineer_microstructure_features(symbol, market_data)
            
            # Calculate microstructure metrics
            order_flow_toxicity = self._calculate_order_flow_toxicity(features)
            price_impact = self._calculate_price_impact(features)
            market_resilience = self._calculate_market_resilience(features)
            information_asymmetry = self._calculate_information_asymmetry(features)
            market_efficiency = self._calculate_market_efficiency(features)
            
            # Calculate overall microstructure score
            microstructure_score = np.mean([
                order_flow_toxicity,
                1 - price_impact,  # Lower impact is better
                market_resilience,
                1 - information_asymmetry,  # Lower asymmetry is better
                market_efficiency
            ])
            
            # Generate recommendations
            recommendations = self._generate_microstructure_recommendations(
                order_flow_toxicity, price_impact, market_resilience, 
                information_asymmetry, market_efficiency
            )
            
            return MarketMicrostructureAnalysis(
                symbol=symbol,
                timestamp=datetime.now(),
                order_flow_toxicity=order_flow_toxicity,
                price_impact=price_impact,
                market_resilience=market_resilience,
                information_asymmetry=information_asymmetry,
                market_efficiency=market_efficiency,
                microstructure_score=microstructure_score,
                recommendations=recommendations,
                metadata={'features': features}
            )
            
        except Exception as e:
            self.logger.error(f"Error analyzing microstructure for {symbol}: {e}")
            return self._get_default_microstructure_analysis(symbol)

    async def _engineer_liquidation_features(self, symbol: str, market_data: Dict[str, Any]) -> List[float]:
        """Engineer features for liquidation prediction"""
        try:
            features = []
            
            # Price-based features
            current_price = market_data.get('price', 0)
            price_change = market_data.get('price_change_24h', 0)
            price_volatility = market_data.get('price_volatility', 0)
            
            features.extend([current_price, price_change, price_volatility])
            
            # Volume-based features
            volume_24h = market_data.get('volume_24h', 0)
            volume_change = market_data.get('volume_change_24h', 0)
            
            features.extend([volume_24h, volume_change])
            
            # Leverage and margin features
            total_leverage = market_data.get('total_leverage', 0)
            margin_utilization = market_data.get('margin_utilization', 0)
            liquidation_risk_score = market_data.get('liquidation_risk_score', 0)
            
            features.extend([total_leverage, margin_utilization, liquidation_risk_score])
            
            # Order book features
            spread = market_data.get('spread', 0)
            depth_imbalance = market_data.get('depth_imbalance', 0)
            order_flow_toxicity = market_data.get('order_flow_toxicity', 0)
            
            features.extend([spread, depth_imbalance, order_flow_toxicity])
            
            # Market sentiment features
            sentiment_score = market_data.get('sentiment_score', 0)
            fear_greed_index = market_data.get('fear_greed_index', 50)
            
            features.extend([sentiment_score, fear_greed_index])
            
            # Technical indicators
            rsi = market_data.get('rsi', 50)
            macd = market_data.get('macd', 0)
            bollinger_position = market_data.get('bollinger_position', 0.5)
            
            features.extend([rsi, macd, bollinger_position])
            
            # Ensure we have enough features
            while len(features) < 20:
                features.append(0.0)
            
            return features[:20]  # Return exactly 20 features
            
        except Exception as e:
            self.logger.error(f"Error engineering liquidation features: {e}")
            return [0.0] * 20

    async def _engineer_orderbook_features(self, symbol: str, order_book_data: Dict[str, Any]) -> List[float]:
        """Engineer features for order book forecasting"""
        try:
            features = []
            
            # Current order book state
            bids = order_book_data.get('bids', [])
            asks = order_book_data.get('asks', [])
            
            # Spread features
            if bids and asks:
                spread = asks[0][0] - bids[0][0]
                spread_percentage = (spread / bids[0][0]) * 100
            else:
                spread = 0
                spread_percentage = 0
            
            features.extend([spread, spread_percentage])
            
            # Depth features
            bid_depth = sum(bid[1] for bid in bids[:10]) if bids else 0
            ask_depth = sum(ask[1] for ask in asks[:10]) if asks else 0
            total_depth = bid_depth + ask_depth
            
            features.extend([bid_depth, ask_depth, total_depth])
            
            # Imbalance features
            if total_depth > 0:
                imbalance = (bid_depth - ask_depth) / total_depth
            else:
                imbalance = 0
            
            features.extend([imbalance])
            
            # Order flow features
            order_flow_toxicity = order_book_data.get('order_flow_toxicity', 0)
            depth_pressure = order_book_data.get('depth_pressure', 0)
            
            features.extend([order_flow_toxicity, depth_pressure])
            
            # Volatility features
            price_volatility = order_book_data.get('price_volatility', 0)
            volume_volatility = order_book_data.get('volume_volatility', 0)
            
            features.extend([price_volatility, volume_volatility])
            
            # Market microstructure features
            market_resilience = order_book_data.get('market_resilience', 0)
            information_asymmetry = order_book_data.get('information_asymmetry', 0)
            
            features.extend([market_resilience, information_asymmetry])
            
            # Ensure we have enough features
            while len(features) < 15:
                features.append(0.0)
            
            return features[:15]  # Return exactly 15 features
            
        except Exception as e:
            self.logger.error(f"Error engineering orderbook features: {e}")
            return [0.0] * 15

    async def _engineer_microstructure_features(self, symbol: str, market_data: Dict[str, Any]) -> List[float]:
        """Engineer features for market microstructure analysis"""
        try:
            features = []
            
            # Order flow features
            order_flow_toxicity = market_data.get('order_flow_toxicity', 0)
            order_imbalance = market_data.get('order_imbalance', 0)
            trade_size_distribution = market_data.get('trade_size_distribution', 0)
            
            features.extend([order_flow_toxicity, order_imbalance, trade_size_distribution])
            
            # Price impact features
            price_impact = market_data.get('price_impact', 0)
            market_depth = market_data.get('market_depth', 0)
            resilience = market_data.get('resilience', 0)
            
            features.extend([price_impact, market_depth, resilience])
            
            # Information asymmetry features
            bid_ask_spread = market_data.get('bid_ask_spread', 0)
            order_book_imbalance = market_data.get('order_book_imbalance', 0)
            trade_flow_imbalance = market_data.get('trade_flow_imbalance', 0)
            
            features.extend([bid_ask_spread, order_book_imbalance, trade_flow_imbalance])
            
            # Market efficiency features
            price_efficiency = market_data.get('price_efficiency', 0)
            volume_efficiency = market_data.get('volume_efficiency', 0)
            liquidity_efficiency = market_data.get('liquidity_efficiency', 0)
            
            features.extend([price_efficiency, volume_efficiency, liquidity_efficiency])
            
            # Volatility features
            realized_volatility = market_data.get('realized_volatility', 0)
            implied_volatility = market_data.get('implied_volatility', 0)
            
            features.extend([realized_volatility, implied_volatility])
            
            # Ensure we have enough features
            while len(features) < 20:
                features.append(0.0)
            
            return features[:20]  # Return exactly 20 features
            
        except Exception as e:
            self.logger.error(f"Error engineering microstructure features: {e}")
            return [0.0] * 20

    def _calculate_risk_level(self, probability: float, confidence: float) -> str:
        """Calculate risk level based on probability and confidence"""
        if probability > 0.8 and confidence > 0.7:
            return 'critical'
        elif probability > 0.6 and confidence > 0.6:
            return 'high'
        elif probability > 0.4 and confidence > 0.5:
            return 'medium'
        else:
            return 'low'

    def _estimate_liquidation_volume(self, symbol: str, probability: float, market_data: Dict[str, Any]) -> float:
        """Estimate expected liquidation volume"""
        try:
            # Base volume from market data
            base_volume = market_data.get('volume_24h', 1000000)
            
            # Adjust based on probability and market conditions
            leverage_factor = market_data.get('total_leverage', 1.0)
            margin_factor = market_data.get('margin_utilization', 0.5)
            
            estimated_volume = base_volume * probability * leverage_factor * margin_factor
            
            return max(0, estimated_volume)
            
        except Exception as e:
            self.logger.error(f"Error estimating liquidation volume: {e}")
            return 0.0

    def _extract_liquidation_factors(self, features: List[float]) -> Dict[str, float]:
        """Extract key factors contributing to liquidation risk"""
        try:
            return {
                'price_volatility': features[2] if len(features) > 2 else 0,
                'leverage_ratio': features[5] if len(features) > 5 else 0,
                'margin_utilization': features[6] if len(features) > 6 else 0,
                'order_flow_toxicity': features[11] if len(features) > 11 else 0,
                'market_sentiment': features[13] if len(features) > 13 else 0,
                'technical_indicators': features[16] if len(features) > 16 else 0
            }
        except Exception as e:
            self.logger.error(f"Error extracting liquidation factors: {e}")
            return {}

    def _calculate_orderbook_confidence(self, features: List[float]) -> float:
        """Calculate confidence score for order book forecast"""
        try:
            # Use feature stability as confidence indicator
            feature_stability = 1 - np.std(features) if features else 0.5
            return max(0.1, min(1.0, feature_stability))
        except Exception as e:
            self.logger.error(f"Error calculating orderbook confidence: {e}")
            return 0.5

    def _calculate_order_flow_toxicity(self, features: List[float]) -> float:
        """Calculate order flow toxicity"""
        try:
            return features[0] if len(features) > 0 else 0.5
        except Exception as e:
            self.logger.error(f"Error calculating order flow toxicity: {e}")
            return 0.5

    def _calculate_price_impact(self, features: List[float]) -> float:
        """Calculate price impact"""
        try:
            return features[3] if len(features) > 3 else 0.5
        except Exception as e:
            self.logger.error(f"Error calculating price impact: {e}")
            return 0.5

    def _calculate_market_resilience(self, features: List[float]) -> float:
        """Calculate market resilience"""
        try:
            return features[5] if len(features) > 5 else 0.5
        except Exception as e:
            self.logger.error(f"Error calculating market resilience: {e}")
            return 0.5

    def _calculate_information_asymmetry(self, features: List[float]) -> float:
        """Calculate information asymmetry"""
        try:
            return features[8] if len(features) > 8 else 0.5
        except Exception as e:
            self.logger.error(f"Error calculating information asymmetry: {e}")
            return 0.5

    def _calculate_market_efficiency(self, features: List[float]) -> float:
        """Calculate market efficiency"""
        try:
            return features[11] if len(features) > 11 else 0.5
        except Exception as e:
            self.logger.error(f"Error calculating market efficiency: {e}")
            return 0.5

    def _generate_microstructure_recommendations(self, toxicity: float, impact: float, 
                                               resilience: float, asymmetry: float, 
                                               efficiency: float) -> List[str]:
        """Generate trading recommendations based on microstructure analysis"""
        recommendations = []
        
        if toxicity > 0.7:
            recommendations.append("High order flow toxicity detected - consider reducing position sizes")
        
        if impact > 0.6:
            recommendations.append("High price impact expected - use limit orders and avoid market orders")
        
        if resilience < 0.4:
            recommendations.append("Low market resilience - prepare for increased volatility")
        
        if asymmetry > 0.6:
            recommendations.append("High information asymmetry - be cautious of large orders")
        
        if efficiency < 0.5:
            recommendations.append("Low market efficiency - consider alternative execution strategies")
        
        if not recommendations:
            recommendations.append("Market conditions appear normal - standard trading strategies recommended")
        
        return recommendations

    async def _load_models(self):
        """Load pre-trained models"""
        try:
            # Load liquidation prediction models
            for horizon in self.prediction_horizons:
                model_path = os.path.join(self.models_dir, f"liquidation_model_{horizon}.pkl")
                if os.path.exists(model_path):
                    self.liquidation_models[f"model_{horizon}"] = joblib.load(model_path)
                    self.logger.info(f"Loaded liquidation model for {horizon}min horizon")
            
            # Load order book forecasting models
            orderbook_model_path = os.path.join(self.models_dir, "orderbook_model.pkl")
            if os.path.exists(orderbook_model_path):
                self.orderbook_models["default"] = joblib.load(orderbook_model_path)
                self.logger.info("Loaded order book forecasting model")
            
            # Load microstructure analysis models
            microstructure_model_path = os.path.join(self.models_dir, "microstructure_model.pkl")
            if os.path.exists(microstructure_model_path):
                self.microstructure_models["default"] = joblib.load(microstructure_model_path)
                self.logger.info("Loaded microstructure analysis model")
                
        except Exception as e:
            self.logger.error(f"Error loading models: {e}")

    async def _periodic_retraining(self):
        """Periodically retrain models with new data"""
        while True:
            try:
                await asyncio.sleep(3600)  # Retrain every hour
                
                # Check if we have enough new data
                if self._has_sufficient_data():
                    await self._retrain_models()
                    self.performance_metrics['model_retraining_count'] += 1
                    self.performance_metrics['last_retraining'] = datetime.now()
                    
            except Exception as e:
                self.logger.error(f"Error in periodic retraining: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes before retrying

    async def _performance_monitoring(self):
        """Monitor model performance and update metrics"""
        while True:
            try:
                await asyncio.sleep(300)  # Check every 5 minutes
                
                # Calculate accuracy scores
                accuracy = self._calculate_model_accuracy()
                if accuracy is not None:
                    self.performance_metrics['accuracy_scores'].append(accuracy)
                
                # Log performance metrics
                self.logger.info(f"Predictive Analytics Performance: {self.get_performance_metrics()}")
                
            except Exception as e:
                self.logger.error(f"Error in performance monitoring: {e}")

    def _has_sufficient_data(self) -> bool:
        """Check if we have sufficient data for retraining"""
        total_samples = sum(len(data) for data in self.liquidation_data.values())
        return total_samples > 1000

    async def _retrain_models(self):
        """Retrain models with new data"""
        try:
            self.logger.info("Starting model retraining...")
            
            # Prepare training data
            X_liquidation, y_liquidation = self._prepare_liquidation_training_data()
            X_orderbook, y_orderbook = self._prepare_orderbook_training_data()
            
            # Retrain liquidation models
            if len(X_liquidation) > 100:
                for horizon in self.prediction_horizons:
                    model = RandomForestRegressor(n_estimators=100, random_state=42)
                    model.fit(X_liquidation, y_liquidation)
                    
                    # Save model
                    model_path = os.path.join(self.models_dir, f"liquidation_model_{horizon}.pkl")
                    joblib.dump(model, model_path)
                    
                    self.liquidation_models[f"model_{horizon}"] = model
            
            # Retrain order book models
            if len(X_orderbook) > 100:
                model = GradientBoostingRegressor(n_estimators=100, random_state=42)
                model.fit(X_orderbook, y_orderbook)
                
                # Save model
                model_path = os.path.join(self.models_dir, "orderbook_model.pkl")
                joblib.dump(model, model_path)
                
                self.orderbook_models["default"] = model
            
            self.logger.info("Model retraining completed successfully")
            
        except Exception as e:
            self.logger.error(f"Error retraining models: {e}")

    def _prepare_liquidation_training_data(self) -> Tuple[List[List[float]], List[float]]:
        """Prepare training data for liquidation prediction"""
        X, y = [], []
        
        for symbol_data in self.liquidation_data.values():
            for i, data_point in enumerate(symbol_data[:-1]):
                if data_point['actual'] is not None:  # Only use data with actual outcomes
                    X.append(data_point['features'])
                    y.append(data_point['actual'])
        
        return X, y

    def _prepare_orderbook_training_data(self) -> Tuple[List[List[float]], List[List[float]]]:
        """Prepare training data for order book forecasting"""
        X, y = [], []
        
        for symbol_data in self.orderbook_data.values():
            for i, data_point in enumerate(symbol_data[:-1]):
                if data_point['actual'] is not None:
                    X.append(data_point['features'])
                    y.append(data_point['actual'])
        
        return X, y

    def _calculate_model_accuracy(self) -> Optional[float]:
        """Calculate model accuracy based on recent predictions"""
        try:
            recent_scores = list(self.performance_metrics['accuracy_scores'])[-100:]
            if recent_scores:
                return np.mean(recent_scores)
            return None
        except Exception as e:
            self.logger.error(f"Error calculating model accuracy: {e}")
            return None

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        return {
            'predictions_made': self.performance_metrics['predictions_made'],
            'model_retraining_count': self.performance_metrics['model_retraining_count'],
            'last_retraining': self.performance_metrics['last_retraining'],
            'avg_accuracy': np.mean(list(self.performance_metrics['accuracy_scores'])) if self.performance_metrics['accuracy_scores'] else 0,
            'models_loaded': len(self.liquidation_models) + len(self.orderbook_models) + len(self.microstructure_models)
        }

    async def _load_or_train_liquidation_model(self, symbol: str):
        """Load existing model or train new one for liquidation prediction"""
        try:
            model_name = f"liquidation_predictor_{symbol}"
            
            # Check if model exists in database
            with self.engine.connect() as conn:
                result = conn.execute(text("""
                    SELECT model_name, version, model_file_path, status 
                    FROM ml_model_versions 
                    WHERE model_name = :model_name AND status = 'active' 
                    ORDER BY version DESC LIMIT 1
                """), {"model_name": model_name})
                
                model_info = result.fetchone()
                
                if model_info and os.path.exists(model_info.model_file_path):
                    # Load existing model
                    for horizon in self.prediction_horizons:
                        model_path = f"{model_info.model_file_path}_{horizon}.joblib"
                        if os.path.exists(model_path):
                            model = joblib.load(model_path)
                            self.liquidation_models[f"{model_name}_{horizon}"] = model
                            self.active_model_versions[f"{model_name}_{horizon}"] = model_info.version
                    
                    self.logger.info(f"Loaded liquidation model for {symbol} version {model_info.version}")
                else:
                    # Train new model
                    await self._train_liquidation_model(symbol, model_name)
                    
        except Exception as e:
            self.logger.error(f"Error loading/training liquidation model for {symbol}: {e}")
            # Create fallback simple model
            await self._create_fallback_model(symbol, model_name)

    async def _train_liquidation_model(self, symbol: str, model_name: str):
        """Train new LightGBM model for liquidation prediction"""
        try:
            self.logger.info(f"Training new liquidation model for {symbol}...")
            
            # Fetch training data from database
            training_data = await self._fetch_training_data(symbol)
            
            if len(training_data) < 100:  # Minimum samples required
                self.logger.warning(f"Insufficient training data for {symbol}, using fallback model")
                await self._create_fallback_model(symbol, model_name)
                return
            
            # Prepare features and labels
            X, y = await self._prepare_training_data(training_data)
            
            if X.shape[0] == 0:
                await self._create_fallback_model(symbol, model_name)
                return
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train models for different horizons
            version = await self._get_next_model_version(model_name)
            
            for horizon in self.prediction_horizons:
                # Create LightGBM dataset
                train_data = lgb.Dataset(X_train, label=y_train)
                valid_data = lgb.Dataset(X_test, label=y_test)
                
                # Train model
                model = lgb.train(
                    self.lgb_params,
                    train_data,
                    valid_sets=[valid_data],
                    num_boost_round=100,
                    callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)]
                )
                
                # Save model
                model_path = f"{self.models_dir}/{model_name}_{horizon}_v{version}.joblib"
                joblib.dump(model, model_path)
                
                # Store in memory
                self.liquidation_models[f"{model_name}_{horizon}"] = model
                self.active_model_versions[f"{model_name}_{horizon}"] = version
                
                # Calculate accuracy
                y_pred = model.predict(X_test)
                y_pred_binary = (y_pred > 0.5).astype(int)
                accuracy = accuracy_score(y_test, y_pred_binary)
                
                # Store model metadata in database
                await self._store_model_version(model_name, version, 'lightgbm', model_path, accuracy, horizon)
            
            self.logger.info(f"Successfully trained liquidation model for {symbol} version {version}")
            
        except Exception as e:
            self.logger.error(f"Error training liquidation model for {symbol}: {e}")
            await self._create_fallback_model(symbol, model_name)

    async def _fetch_training_data(self, symbol: str) -> pd.DataFrame:
        """Fetch training data from database"""
        try:
            with self.engine.connect() as conn:
                # Fetch liquidation events and market data for training
                query = text("""
                    SELECT 
                        le.timestamp,
                        le.side,
                        le.price,
                        le.size,
                        le.impact_score,
                        obs.bid_price,
                        obs.ask_price,
                        obs.spread,
                        obs.liquidity_imbalance,
                        obs.depth_pressure,
                        obs.order_flow_toxicity,
                        ca.order_book_imbalance,
                        ca.liquidity_score,
                        ca.volatility_score
                    FROM liquidation_events le
                    LEFT JOIN order_book_snapshots obs ON 
                        obs.symbol = le.symbol 
                        AND obs.timestamp BETWEEN le.timestamp - INTERVAL '5 minutes' 
                        AND le.timestamp + INTERVAL '5 minutes'
                    LEFT JOIN comprehensive_analysis ca ON 
                        ca.symbol = le.symbol 
                        AND ca.timestamp BETWEEN le.timestamp - INTERVAL '5 minutes' 
                        AND le.timestamp + INTERVAL '5 minutes'
                    WHERE le.symbol = :symbol
                    AND le.timestamp >= NOW() - INTERVAL '30 days'
                    ORDER BY le.timestamp DESC
                    LIMIT 1000
                """)
                
                result = conn.execute(query, {"symbol": symbol})
                data = pd.DataFrame(result.fetchall(), columns=result.keys())
                
                return data
                
        except Exception as e:
            self.logger.error(f"Error fetching training data for {symbol}: {e}")
            return pd.DataFrame()

    async def _store_prediction_result(self, symbol: str, model_name: str, probability: float, 
                                     confidence: float, features: List[float], timestamp: datetime):
        """Store prediction result in database for performance tracking"""
        try:
            features_hash = hashlib.md5(str(features).encode()).hexdigest()
            
            with self.engine.connect() as conn:
                conn.execute(text("""
                    INSERT INTO ml_feature_store (
                        symbol, feature_name, feature_value, feature_metadata, 
                        timestamp, feature_group, is_training_data
                    ) VALUES (
                        :symbol, 'liquidation_prediction', :probability,
                        :metadata, :timestamp, 'prediction', false
                    )
                """), {
                    "symbol": symbol,
                    "probability": float(probability),
                    "metadata": json.dumps({
                        "model_name": model_name,
                        "confidence": float(confidence),
                        "features_hash": features_hash,
                        "feature_count": len(features)
                    }),
                    "timestamp": timestamp
                })
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Error storing prediction result: {e}")

    def _fallback_liquidation_prediction(self, features: List[float]) -> float:
        """Fallback prediction when model is not available"""
        try:
            # Simple heuristic based on features
            if len(features) >= 5:
                # Use order flow toxicity, liquidity imbalance, and volatility
                toxicity = features[0] if len(features) > 0 else 0.5
                imbalance = abs(features[1]) if len(features) > 1 else 0.5
                volatility = features[2] if len(features) > 2 else 0.5
                
                # Simple weighted combination
                prediction = (toxicity * 0.4 + imbalance * 0.3 + volatility * 0.3)
                return max(0.1, min(0.9, prediction))
            else:
                return 0.5
        except:
            return 0.5

    async def _extract_prediction_factors(self, features: List[float], market_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract key factors influencing the prediction"""
        try:
            factors = {}
            
            if len(features) >= 10:
                factors['order_flow_toxicity'] = features[0]
                factors['liquidity_imbalance'] = features[1] 
                factors['price_volatility'] = features[2]
                factors['spread_pressure'] = features[3]
                factors['volume_surge'] = features[4]
                factors['leverage_ratio'] = features[5] if len(features) > 5 else 0.0
                factors['margin_pressure'] = features[6] if len(features) > 6 else 0.0
                factors['correlation_risk'] = features[7] if len(features) > 7 else 0.0
            
            # Add market condition factors
            if 'current_price' in market_data and 'historical_volatility' in market_data:
                factors['market_stress'] = market_data.get('historical_volatility', 0.0)
                factors['price_momentum'] = market_data.get('price_change_24h', 0.0)
            
            return factors
            
        except Exception as e:
            self.logger.error(f"Error extracting prediction factors: {e}")
            return {}
    
    async def _create_fallback_model(self, symbol: str, model_name: str):
        """Create a simple fallback model when training fails"""
        try:
            self.logger.info(f"Creating fallback model for {symbol}")
            
            # Create a simple rule-based fallback
            class FallbackModel:
                def predict(self, X):
                    # Simple heuristic: return moderate risk for all predictions
                    return [0.5] * len(X)
            
            fallback_model = FallbackModel()
            
            # Store fallback model for all horizons
            for horizon in self.prediction_horizons:
                model_key = f"{model_name}_{horizon}"
                self.liquidation_models[model_key] = fallback_model
                self.active_model_versions[model_key] = 0  # Version 0 for fallback
            
            self.logger.info(f"✅ Fallback model created for {symbol}")
            
        except Exception as e:
            self.logger.error(f"Error creating fallback model for {symbol}: {e}")
    
    async def _get_next_model_version(self, model_name: str) -> int:
        """Get the next version number for a model"""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("""
                    SELECT MAX(version) FROM ml_model_versions WHERE model_name = :model_name
                """), {"model_name": model_name})
                
                max_version = result.scalar()
                return (max_version or 0) + 1
                
        except Exception as e:
            self.logger.error(f"Error getting next model version: {e}")
            return 1
    
    async def _store_model_version(self, model_name: str, version: int, model_type: str, 
                                 model_path: str, accuracy: float, horizon: int):
        """Store model version metadata in database"""
        try:
            with self.engine.connect() as conn:
                conn.execute(text("""
                    INSERT INTO ml_model_versions (
                        model_name, version, model_type, status, accuracy_score,
                        model_file_path, deployment_environment, created_at
                    ) VALUES (
                        :model_name, :version, :model_type, 'active', :accuracy_score,
                        :model_file_path, 'production', :created_at
                    )
                """), {
                    "model_name": f"{model_name}_{horizon}",
                    "version": version,
                    "model_type": model_type,
                    "accuracy_score": accuracy,
                    "model_file_path": model_path,
                    "created_at": datetime.now()
                })
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Error storing model version: {e}")
    
    async def _prepare_training_data(self, training_data):
        """Prepare training data for model training"""
        try:
            # This is a simplified data preparation
            # In production, implement proper feature engineering
            
            if len(training_data) == 0:
                return np.array([]), np.array([])
            
            # Create simple features from the data
            features = []
            labels = []
            
            for _, row in training_data.iterrows():
                feature_vector = [
                    float(row.get('spread', 0.0)),
                    float(row.get('liquidity_imbalance', 0.0)),
                    float(row.get('depth_pressure', 0.0)),
                    float(row.get('order_flow_toxicity', 0.0)),
                    float(row.get('impact_score', 0.0))
                ]
                
                # Simple binary label: high impact = 1, low impact = 0
                label = 1 if row.get('impact_score', 0.0) > 0.5 else 0
                
                features.append(feature_vector)
                labels.append(label)
            
            return np.array(features), np.array(labels)
            
        except Exception as e:
            self.logger.error(f"Error preparing training data: {e}")
            return np.array([]), np.array([])

    def _get_default_liquidation_prediction(self, symbol: str) -> LiquidationPrediction:
        """Get default liquidation prediction when model fails"""
        return LiquidationPrediction(
            symbol=symbol,
            timestamp=datetime.now(),
            prediction_horizon=30,
            liquidation_probability=0.5,
            expected_liquidation_volume=0.0,
            confidence_score=0.5,
            risk_level='medium',
            factors={},
            metadata={'error': 'Model prediction failed'}
        )

    def _get_default_orderbook_forecast(self, symbol: str) -> OrderBookForecast:
        """Get default order book forecast when model fails"""
        return OrderBookForecast(
            symbol=symbol,
            timestamp=datetime.now(),
            forecast_horizon=15,
            predicted_spread=0.0,
            predicted_depth={'bid': 0.0, 'ask': 0.0},
            predicted_imbalance=0.0,
            confidence_score=0.5,
            volatility_forecast=0.0,
            metadata={'error': 'Model prediction failed'}
        )

    def _get_default_microstructure_analysis(self, symbol: str) -> MarketMicrostructureAnalysis:
        """Get default microstructure analysis when model fails"""
        return MarketMicrostructureAnalysis(
            symbol=symbol,
            timestamp=datetime.now(),
            order_flow_toxicity=0.5,
            price_impact=0.5,
            market_resilience=0.5,
            information_asymmetry=0.5,
            market_efficiency=0.5,
            microstructure_score=0.5,
            recommendations=['Analysis unavailable'],
            metadata={'error': 'Model analysis failed'}
        )

    async def close(self):
        """Close the predictive analytics service"""
        try:
            # Cancel background tasks
            if hasattr(self, 'retraining_task'):
                self.retraining_task.cancel()
            if hasattr(self, 'monitoring_task'):
                self.monitoring_task.cancel()
            
            self.logger.info("Predictive Analytics Service closed successfully")
            
        except Exception as e:
            self.logger.error(f"Error closing Predictive Analytics Service: {e}")
