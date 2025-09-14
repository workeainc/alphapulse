"""
Advanced Model Fusion & Calibration (Phase 7)
Implements ensemble methods, probability calibration, and performance tracking
"""

import asyncio
import asyncpg
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from enum import Enum
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FusionMethod(Enum):
    """Model fusion methods"""
    WEIGHTED_AVERAGE = "weighted_average"
    VOTING = "voting"
    STACKING = "stacking"
    BAGGING = "bagging"

class CalibrationMethod(Enum):
    """Probability calibration methods"""
    ISOTONIC = "isotonic"
    PLATT = "platt"
    TEMPERATURE = "temperature"

@dataclass
class ModelPrediction:
    """Individual model prediction"""
    model_name: str
    probability: float  # 0-1 probability
    confidence: float  # 0-1 confidence
    features_used: List[str]
    timestamp: datetime = None

@dataclass
class EnsemblePrediction:
    """Ensemble prediction result"""
    symbol: str
    timeframe: str
    timestamp: datetime
    model_predictions: Dict[str, float]
    ensemble_prediction: float
    consensus_score: float
    agreement_count: int
    confidence_score: float
    calibrated_confidence: Optional[float]
    signal_direction: str  # 'LONG', 'SHORT', 'FLAT'
    signal_strength: float
    fusion_method: str
    
    @property
    def final_probability(self) -> float:
        """Alias for ensemble_prediction for backward compatibility"""
        return self.ensemble_prediction

class AdvancedModelFusion:
    """Advanced model fusion and calibration system"""
    
    def __init__(self, db_connection: asyncpg.Connection = None):
        self.db_connection = db_connection
        self.fusion_config = None
        self.calibration_models = {}
        self.performance_trackers = {}
        
    async def initialize(self):
        """Initialize the fusion system"""
        if self.db_connection:
            await self._load_fusion_config()
            await self._load_calibration_models()
            await self._initialize_performance_trackers()
            logger.info("✅ Advanced Model Fusion initialized")
    
    async def _load_fusion_config(self):
        """Load fusion configuration from database"""
        try:
            config = await self.db_connection.fetchrow("""
                SELECT fusion_method, model_weights, consensus_threshold, 
                       min_agreement_count, calibration_method, confidence_threshold
                FROM sde_model_fusion_config 
                WHERE is_active = TRUE 
                ORDER BY created_at DESC 
                LIMIT 1
            """)
            
            if config:
                # Handle JSONB model_weights properly
                model_weights = config['model_weights']
                if isinstance(model_weights, str):
                    import json
                    model_weights = json.loads(model_weights)
                
                self.fusion_config = {
                    'fusion_method': config['fusion_method'],
                    'model_weights': model_weights,
                    'consensus_threshold': float(config['consensus_threshold']),
                    'min_agreement_count': config['min_agreement_count'],
                    'calibration_method': config['calibration_method'],
                    'confidence_threshold': float(config['confidence_threshold'])
                }
                logger.info(f"✅ Loaded fusion config: {self.fusion_config['fusion_method']}")
            else:
                # Default configuration
                self.fusion_config = {
                    'fusion_method': 'weighted_average',
                    'model_weights': {'catboost': 0.4, 'logistic': 0.2, 'decision_tree': 0.2, 'rule_based': 0.2},
                    'consensus_threshold': 0.7,
                    'min_agreement_count': 3,
                    'calibration_method': 'isotonic',
                    'confidence_threshold': 0.85
                }
                logger.info("✅ Using default fusion config")
                
        except Exception as e:
            logger.error(f"❌ Failed to load fusion config: {e}")
            raise
    
    async def _load_calibration_models(self):
        """Load calibration models from database"""
        try:
            calibrations = await self.db_connection.fetch("""
                SELECT model_name, calibration_type, calibration_params, calibration_data
                FROM sde_model_calibration 
                WHERE is_active = TRUE
            """)
            
            for cal in calibrations:
                model_key = f"{cal['model_name']}_{cal['calibration_type']}"
                self.calibration_models[model_key] = {
                    'params': cal['calibration_params'],
                    'data': cal['calibration_data']
                }
            
            logger.info(f"✅ Loaded {len(self.calibration_models)} calibration models")
            
        except Exception as e:
            logger.error(f"❌ Failed to load calibration models: {e}")
    
    async def _initialize_performance_trackers(self):
        """Initialize performance tracking for each model"""
        try:
            models = ['catboost', 'logistic', 'decision_tree', 'rule_based']
            for model in models:
                self.performance_trackers[model] = {
                    'total_predictions': 0,
                    'correct_predictions': 0,
                    'accuracy': 0.0,
                    'last_updated': datetime.now()
                }
            
            logger.info(f"✅ Initialized performance trackers for {len(models)} models")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize performance trackers: {e}")
    
    async def fuse_predictions(self, predictions: List[ModelPrediction], 
                             symbol: str = None, timeframe: str = None,
                             fusion_method: str = None, weights: Dict[str, float] = None) -> EnsemblePrediction:
        """Fuse multiple model predictions into ensemble prediction"""
        try:
            if not predictions:
                raise ValueError("No predictions provided")
            
            # Extract model predictions
            model_preds = {pred.model_name: pred.probability for pred in predictions}
            model_confidences = {pred.model_name: pred.confidence for pred in predictions}
            
            # Use provided fusion method or default
            method_to_use = fusion_method or self.fusion_config.get('fusion_method', 'weighted_average')
            
            # Apply fusion method
            if method_to_use == 'weighted_average':
                ensemble_pred, consensus_score, agreement_count = self._weighted_average_fusion(
                    model_preds, model_confidences, weights
                )
            elif method_to_use == 'voting':
                ensemble_pred, consensus_score, agreement_count = self._voting_fusion(
                    model_preds, model_confidences
                )
            else:
                ensemble_pred, consensus_score, agreement_count = self._weighted_average_fusion(
                    model_preds, model_confidences, weights
                )
            
            # Calculate confidence score
            confidence_score = self._calculate_confidence_score(model_confidences, consensus_score)
            
            # Apply calibration if available
            calibrated_confidence = await self._apply_calibration(
                ensemble_pred, symbol, timeframe
            )
            
            # Determine signal direction and strength
            signal_direction, signal_strength = self._determine_signal(
                ensemble_pred, calibrated_confidence or confidence_score
            )
            
            # Create ensemble prediction
            ensemble = EnsemblePrediction(
                symbol=symbol,
                timeframe=timeframe,
                timestamp=datetime.now(),
                model_predictions=model_preds,
                ensemble_prediction=ensemble_pred,
                consensus_score=consensus_score,
                agreement_count=agreement_count,
                confidence_score=confidence_score,
                calibrated_confidence=calibrated_confidence,
                signal_direction=signal_direction,
                signal_strength=signal_strength,
                fusion_method=method_to_use
            )
            
            # Store ensemble prediction
            if self.db_connection:
                await self._store_ensemble_prediction(ensemble)
            
            return ensemble
            
        except Exception as e:
            logger.error(f"❌ Fusion failed: {e}")
            raise
    
    def _weighted_average_fusion(self, model_preds: Dict[str, float], 
                                model_confidences: Dict[str, float], 
                                weights: Dict[str, float] = None) -> Tuple[float, float, int]:
        """Weighted average fusion method"""
        weights = weights or self.fusion_config.get('model_weights', {})
        
        # Calculate weighted prediction
        weighted_sum = 0.0
        total_weight = 0.0
        
        for model_name, pred in model_preds.items():
            if model_name in weights:
                weight = weights[model_name] * model_confidences.get(model_name, 1.0)
                weighted_sum += pred * weight
                total_weight += weight
        
        ensemble_prediction = weighted_sum / total_weight if total_weight > 0 else 0.5
        
        # Calculate consensus score (how many models agree on direction)
        threshold = 0.5
        long_votes = sum(1 for pred in model_preds.values() if pred > threshold)
        short_votes = len(model_preds) - long_votes
        consensus_score = max(long_votes, short_votes) / len(model_preds)
        agreement_count = max(long_votes, short_votes)
        
        return ensemble_prediction, consensus_score, agreement_count
    
    def _voting_fusion(self, model_preds: Dict[str, float], 
                      model_confidences: Dict[str, float]) -> Tuple[float, float, int]:
        """Voting fusion method"""
        threshold = 0.5
        long_votes = sum(1 for pred in model_preds.values() if pred > threshold)
        short_votes = len(model_preds) - long_votes
        
        # Determine majority
        if long_votes > short_votes:
            ensemble_prediction = 0.75  # Strong long
        elif short_votes > long_votes:
            ensemble_prediction = 0.25  # Strong short
        else:
            ensemble_prediction = 0.5   # Neutral
        
        consensus_score = max(long_votes, short_votes) / len(model_preds)
        agreement_count = max(long_votes, short_votes)
        
        return ensemble_prediction, consensus_score, agreement_count
    
    def _calculate_confidence_score(self, model_confidences: Dict[str, float], 
                                  consensus_score: float) -> float:
        """Calculate overall confidence score"""
        # Average model confidence
        avg_confidence = np.mean(list(model_confidences.values())) if model_confidences else 0.5
        
        # Combine with consensus score
        confidence_score = 0.7 * avg_confidence + 0.3 * consensus_score
        
        return min(confidence_score, 1.0)
    
    async def _apply_calibration(self, prediction: float, symbol: str, 
                                timeframe: str) -> Optional[float]:
        """Apply calibration to prediction"""
        try:
            # For now, use simple temperature scaling
            # In production, load actual calibration models
            temperature = 1.2  # Calibration parameter
            
            # Apply temperature scaling
            logit = np.log(prediction / (1 - prediction))
            calibrated_logit = logit / temperature
            calibrated_pred = 1 / (1 + np.exp(-calibrated_logit))
            
            return calibrated_pred
            
        except Exception as e:
            logger.warning(f"⚠️ Calibration failed: {e}")
            return None
    
    def _determine_signal(self, prediction: float, confidence: float) -> Tuple[str, float]:
        """Determine signal direction and strength"""
        threshold = self.fusion_config['confidence_threshold']
        
        if confidence < threshold:
            return 'FLAT', 0.0
        
        if prediction > 0.6:
            return 'LONG', prediction
        elif prediction < 0.4:
            return 'SHORT', 1.0 - prediction
        else:
            return 'FLAT', 0.0
    
    async def _store_ensemble_prediction(self, ensemble: EnsemblePrediction):
        """Store ensemble prediction in database"""
        try:
            # Ensure required fields are not None
            symbol = ensemble.symbol or 'UNKNOWN'
            timeframe = ensemble.timeframe or '1h'
            timestamp = ensemble.timestamp or datetime.now()
            
            await self.db_connection.execute("""
                INSERT INTO sde_ensemble_predictions 
                (symbol, timeframe, timestamp, model_predictions, ensemble_prediction,
                 consensus_score, agreement_count, confidence_score, calibrated_confidence,
                 signal_direction, signal_strength, fusion_method)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
            """, symbol, timeframe, timestamp,
                 json.dumps(ensemble.model_predictions), ensemble.ensemble_prediction,
                 ensemble.consensus_score, ensemble.agreement_count, ensemble.confidence_score,
                 ensemble.calibrated_confidence, ensemble.signal_direction,
                 ensemble.signal_strength, ensemble.fusion_method)
            
        except Exception as e:
            logger.error(f"❌ Failed to store ensemble prediction: {e}")
    
    async def update_model_performance(self, model_name: str, symbol: str, 
                                     timeframe: str, prediction: float, 
                                     actual_outcome: float, timestamp: datetime):
        """Update model performance tracking"""
        try:
            # Update in-memory tracker
            if model_name in self.performance_trackers:
                tracker = self.performance_trackers[model_name]
                tracker['total_predictions'] += 1
                
                # Determine if prediction was correct
                predicted_direction = 'LONG' if prediction > 0.5 else 'SHORT'
                actual_direction = 'LONG' if actual_outcome > 0 else 'SHORT'
                
                if predicted_direction == actual_direction:
                    tracker['correct_predictions'] += 1
                
                tracker['accuracy'] = tracker['correct_predictions'] / tracker['total_predictions']
                tracker['last_updated'] = timestamp
            
            # Store in database
            if self.db_connection:
                await self._store_performance_update(
                    model_name, symbol, timeframe, prediction, actual_outcome, timestamp
                )
            
        except Exception as e:
            logger.error(f"❌ Failed to update performance: {e}")
    
    async def _store_performance_update(self, model_name: str, symbol: str, 
                                      timeframe: str, prediction: float, 
                                      actual_outcome: float, timestamp: datetime):
        """Store performance update in database"""
        try:
            # Calculate period (daily)
            period_start = timestamp.replace(hour=0, minute=0, second=0, microsecond=0)
            period_end = period_start + timedelta(days=1)
            
            # Check if record exists
            existing = await self.db_connection.fetchrow("""
                SELECT id, total_signals, winning_signals, losing_signals, win_rate
                FROM sde_model_performance
                WHERE model_name = $1 AND symbol = $2 AND timeframe = $3 
                      AND period_start = $4 AND period_end = $5
            """, model_name, symbol, timeframe, period_start, period_end)
            
            if existing:
                # Update existing record
                new_total = existing['total_signals'] + 1
                new_winning = existing['winning_signals']
                new_losing = existing['losing_signals']
                
                predicted_direction = 'LONG' if prediction > 0.5 else 'SHORT'
                actual_direction = 'LONG' if actual_outcome > 0 else 'SHORT'
                
                if predicted_direction == actual_direction:
                    new_winning += 1
                else:
                    new_losing += 1
                
                new_win_rate = new_winning / new_total if new_total > 0 else 0.0
                
                await self.db_connection.execute("""
                    UPDATE sde_model_performance
                    SET total_signals = $1, winning_signals = $2, losing_signals = $3, 
                        win_rate = $4
                    WHERE id = $5
                """, new_total, new_winning, new_losing, new_win_rate, existing['id'])
                
            else:
                # Create new record
                is_win = (prediction > 0.5) == (actual_outcome > 0)
                win_rate = 1.0 if is_win else 0.0
                
                await self.db_connection.execute("""
                    INSERT INTO sde_model_performance
                    (model_name, symbol, timeframe, period_start, period_end,
                     total_signals, winning_signals, losing_signals, win_rate, 
                     avg_profit, avg_loss, profit_factor, sharpe_ratio, max_drawdown, total_return, created_at)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16)
                """, model_name, symbol, timeframe, period_start, period_end,
                     1, 1 if is_win else 0, 0 if is_win else 1, win_rate,
                     0.0, 0.0, 0.0, 0.0, 0.0, 0.0, timestamp)
            
        except Exception as e:
            logger.error(f"❌ Failed to store performance update: {e}")
    
    async def detect_model_drift(self, model_name: str, symbol: str, 
                               timeframe: str, recent_predictions: List[float],
                               recent_actuals: List[float]) -> Dict[str, Any]:
        """Detect model drift"""
        try:
            if len(recent_predictions) < 10:
                return {'drift_detected': False, 'reason': 'Insufficient data'}
            
            # Calculate drift metrics
            recent_accuracy = np.mean([
                1.0 if (pred > 0.5) == (actual > 0) else 0.0
                for pred, actual in zip(recent_predictions, recent_actuals)
            ])
            
            # Get historical accuracy
            historical = await self.db_connection.fetchrow("""
                SELECT AVG(win_rate) as avg_win_rate
                FROM sde_model_performance
                WHERE model_name = $1 AND symbol = $2 AND timeframe = $3
                      AND period_start < $4
            """, model_name, symbol, timeframe, datetime.now() - timedelta(days=7))
            
            if not historical or historical['avg_win_rate'] is None:
                return {'drift_detected': False, 'reason': 'No historical data'}
            
            historical_accuracy = float(historical['avg_win_rate'])
            drift_score = abs(recent_accuracy - historical_accuracy)
            drift_threshold = 0.1  # 10% accuracy drop
            
            drift_detected = drift_score > drift_threshold
            
            # Store drift detection
            if self.db_connection:
                await self.db_connection.execute("""
                    INSERT INTO sde_model_drift
                    (model_name, symbol, timeframe, drift_type, drift_score, 
                     drift_threshold, is_drift_detected, drift_metrics, detection_date)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                """, model_name, symbol, timeframe, 'performance', drift_score,
                     drift_threshold, drift_detected, 
                     json.dumps({'recent_accuracy': recent_accuracy, 
                               'historical_accuracy': historical_accuracy}),
                     datetime.now())
            
            return {
                'drift_detected': drift_detected,
                'drift_score': drift_score,
                'drift_threshold': drift_threshold,
                'recent_accuracy': recent_accuracy,
                'historical_accuracy': historical_accuracy,
                'recommended_action': 'retrain' if drift_detected else 'monitor'
            }
            
        except Exception as e:
            logger.error(f"❌ Drift detection failed: {e}")
            return {'drift_detected': False, 'error': str(e)}
    
    async def get_performance_summary(self, model_name: str = None, 
                                    symbol: str = None, timeframe: str = None) -> Dict[str, Any]:
        """Get performance summary"""
        try:
            query = """
                SELECT model_name, symbol, timeframe, 
                       AVG(win_rate) as avg_win_rate,
                       COUNT(*) as periods,
                       SUM(total_signals) as total_signals,
                       SUM(winning_signals) as total_winning
                FROM sde_model_performance
                WHERE 1=1
            """
            params = []
            param_count = 0
            
            if model_name:
                param_count += 1
                query += f" AND model_name = ${param_count}"
                params.append(model_name)
            
            if symbol:
                param_count += 1
                query += f" AND symbol = ${param_count}"
                params.append(symbol)
            
            if timeframe:
                param_count += 1
                query += f" AND timeframe = ${param_count}"
                params.append(timeframe)
            
            query += " GROUP BY model_name, symbol, timeframe ORDER BY avg_win_rate DESC"
            
            results = await self.db_connection.fetch(query, *params)
            
            summary = []
            for row in results:
                summary.append({
                    'model_name': row['model_name'],
                    'symbol': row['symbol'],
                    'timeframe': row['timeframe'],
                    'avg_win_rate': float(row['avg_win_rate']) if row['avg_win_rate'] else 0.0,
                    'periods': row['periods'],
                    'total_signals': row['total_signals'],
                    'total_winning': row['total_winning']
                })
            
            return {'performance_summary': summary}
            
        except Exception as e:
            logger.error(f"❌ Failed to get performance summary: {e}")
            return {'error': str(e)}
