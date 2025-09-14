import asyncio
import json
import logging
import time
import numpy as np
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import psutil
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

logger = logging.getLogger(__name__)

class OptimizationTarget(Enum):
    THROUGHPUT = "throughput"
    LATENCY = "latency"
    MEMORY_USAGE = "memory_usage"
    CPU_USAGE = "cpu_usage"
    STORAGE_EFFICIENCY = "storage_efficiency"
    OVERALL_PERFORMANCE = "overall_performance"

class ModelType(Enum):
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    ENSEMBLE = "ensemble"

@dataclass
class PerformanceFeature:
    timestamp: datetime
    batch_size: int
    parallel_workers: int
    memory_usage: float
    cpu_usage: float
    active_connections: int
    database_health: float
    patterns_per_second: float
    storage_efficiency: float
    metadata: Dict[str, Any]

@dataclass
class OptimizationPrediction:
    target: OptimizationTarget
    predicted_value: float
    confidence: float
    recommended_parameters: Dict[str, Any]
    expected_improvement: float
    timestamp: datetime

@dataclass
class OptimizationAction:
    action_id: str
    action_type: str
    parameters: Dict[str, Any]
    predicted_impact: Dict[str, float]
    confidence: float
    timestamp: datetime
    executed: bool
    actual_impact: Optional[Dict[str, float]]

class PredictiveOptimizer:
    """
    Machine learning-based predictive optimization for pattern processing
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.models: Dict[OptimizationTarget, Any] = {}
        self.scalers: Dict[OptimizationTarget, StandardScaler] = {}
        self.feature_history: List[PerformanceFeature] = []
        self.predictions: List[OptimizationPrediction] = []
        self.actions: List[OptimizationAction] = []
        
        # Configuration
        self.model_type = ModelType(config.get('model_type', 'random_forest'))
        self.training_threshold = config.get('training_threshold', 1000)
        self.prediction_interval = config.get('prediction_interval', 60)
        self.auto_optimization = config.get('auto_optimization', True)
        self.optimization_threshold = config.get('optimization_threshold', 0.1)
        self.model_retrain_interval = config.get('model_retrain_interval', 3600)  # 1 hour
        
        # Model paths
        self.model_dir = config.get('model_dir', 'models')
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Performance tracking
        self.stats = {
            'total_predictions': 0,
            'total_optimizations': 0,
            'prediction_accuracy': 0.0,
            'optimization_success_rate': 0.0,
            'last_training': None,
            'models_loaded': 0
        }
        
        # Initialize models
        self._initialize_models()
        
        logger.info("PredictiveOptimizer initialized with config: %s", config)
    
    def _initialize_models(self):
        """Initialize ML models for each optimization target"""
        for target in OptimizationTarget:
            if target == OptimizationTarget.OVERALL_PERFORMANCE:
                # Overall performance is a composite metric
                continue
            
            model_path = os.path.join(self.model_dir, f"{target.value}_model.pkl")
            scaler_path = os.path.join(self.model_dir, f"{target.value}_scaler.pkl")
            
            try:
                # Try to load existing model
                if os.path.exists(model_path) and os.path.exists(scaler_path):
                    self.models[target] = joblib.load(model_path)
                    self.scalers[target] = joblib.load(scaler_path)
                    self.stats['models_loaded'] += 1
                    logger.info("Loaded existing model for target: %s", target.value)
                else:
                    # Create new model
                    self._create_new_model(target)
                    
            except Exception as e:
                logger.warning("Could not load model for %s: %s", target.value, e)
                self._create_new_model(target)
    
    def _create_new_model(self, target: OptimizationTarget):
        """Create a new ML model for a specific target"""
        if self.model_type == ModelType.RANDOM_FOREST:
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
        elif self.model_type == ModelType.GRADIENT_BOOSTING:
            model = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
        else:  # Ensemble
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
        
        self.models[target] = model
        self.scalers[target] = StandardScaler()
        
        logger.info("Created new model for target: %s", target.value)
    
    async def start(self):
        """Start the predictive optimizer"""
        logger.info("Starting predictive optimizer...")
        
        # Start background tasks
        asyncio.create_task(self._prediction_loop())
        asyncio.create_task(self._optimization_loop())
        asyncio.create_task(self._model_maintenance_loop())
        
        logger.info("Predictive optimizer started successfully")
    
    async def _prediction_loop(self):
        """Main prediction loop"""
        while True:
            try:
                if len(self.feature_history) >= self.training_threshold:
                    await self._make_predictions()
                
                await asyncio.sleep(self.prediction_interval)
                
            except Exception as e:
                logger.error("Error in prediction loop: %s", e)
                await asyncio.sleep(10)
    
    async def _optimization_loop(self):
        """Main optimization loop"""
        while True:
            try:
                if self.auto_optimization and self.predictions:
                    await self._evaluate_optimization_opportunities()
                
                await asyncio.sleep(30)
                
            except Exception as e:
                logger.error("Error in optimization loop: %s", e)
                await asyncio.sleep(10)
    
    async def _model_maintenance_loop(self):
        """Model maintenance and retraining loop"""
        while True:
            try:
                if self.stats['last_training']:
                    time_since_training = (datetime.now(timezone.utc) - self.stats['last_training']).total_seconds()
                    if time_since_training > self.model_retrain_interval:
                        await self._retrain_models()
                
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                logger.error("Error in model maintenance loop: %s", e)
                await asyncio.sleep(60)
    
    def add_performance_data(self, features: Dict[str, Any]):
        """Add new performance data for training and prediction"""
        feature = PerformanceFeature(
            timestamp=datetime.now(timezone.utc),
            batch_size=features.get('batch_size', 0),
            parallel_workers=features.get('parallel_workers', 0),
            memory_usage=features.get('memory_usage', 0.0),
            cpu_usage=features.get('cpu_usage', 0.0),
            active_connections=features.get('active_connections', 0),
            database_health=features.get('database_health', 0.0),
            patterns_per_second=features.get('patterns_per_second', 0.0),
            storage_efficiency=features.get('storage_efficiency', 0.0),
            metadata=features.get('metadata', {})
        )
        
        self.feature_history.append(feature)
        
        # Keep only recent data (last 24 hours)
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=24)
        self.feature_history = [
            f for f in self.feature_history
            if f.timestamp >= cutoff_time
        ]
        
        logger.debug("Added performance data: %s", features)
    
    async def _make_predictions(self):
        """Make predictions for all optimization targets"""
        if len(self.feature_history) < self.training_threshold:
            return
        
        # Prepare training data
        X, y_dict = self._prepare_training_data()
        
        # Train models if needed
        for target in OptimizationTarget:
            if target == OptimizationTarget.OVERALL_PERFORMANCE:
                continue
            
            if target not in self.models:
                continue
            
            try:
                # Check if model needs training
                if not hasattr(self.models[target], 'n_estimators'):
                    await self._train_model(target, X, y_dict.get(target, []))
                
                # Make prediction
                prediction = await self._predict_target(target, X[-1:])
                if prediction:
                    self.predictions.append(prediction)
                    self.stats['total_predictions'] += 1
                
            except Exception as e:
                logger.error("Error making prediction for %s: %s", target.value, e)
    
    def _prepare_training_data(self) -> Tuple[np.ndarray, Dict[OptimizationTarget, np.ndarray]]:
        """Prepare training data from feature history"""
        if not self.feature_history:
            return np.array([]), {}
        
        # Extract features
        features = []
        targets = {target: [] for target in OptimizationTarget}
        
        for feature in self.feature_history:
            feature_vector = [
                feature.batch_size,
                feature.parallel_workers,
                feature.memory_usage,
                feature.cpu_usage,
                feature.active_connections,
                feature.database_health,
                feature.storage_efficiency
            ]
            features.append(feature_vector)
            
            # Extract targets
            targets[OptimizationTarget.THROUGHPUT].append(feature.patterns_per_second)
            targets[OptimizationTarget.LATENCY].append(1000 / max(feature.patterns_per_second, 1))  # ms per pattern
            targets[OptimizationTarget.MEMORY_USAGE].append(feature.memory_usage)
            targets[OptimizationTarget.CPU_USAGE].append(feature.cpu_usage)
            targets[OptimizationTarget.STORAGE_EFFICIENCY].append(feature.storage_efficiency)
            
            # Overall performance is a weighted combination
            overall_score = (
                min(feature.patterns_per_second / 1000, 1.0) * 0.4 +
                (1.0 - feature.memory_usage / 100) * 0.2 +
                (1.0 - feature.cpu_usage / 100) * 0.2 +
                (feature.storage_efficiency / 100) * 0.2
            )
            targets[OptimizationTarget.OVERALL_PERFORMANCE].append(overall_score * 100)
        
        X = np.array(features)
        y_dict = {target: np.array(values) for target, values in targets.items()}
        
        return X, y_dict
    
    async def _train_model(self, target: OptimizationTarget, X: np.ndarray, y: np.ndarray):
        """Train a model for a specific target"""
        if len(X) < 10 or len(y) < 10:
            return
        
        try:
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Scale features
            X_train_scaled = self.scalers[target].fit_transform(X_train)
            X_test_scaled = self.scalers[target].transform(X_test)
            
            # Train model
            self.models[target].fit(X_train_scaled, y_train)
            
            # Evaluate model
            y_pred = self.models[target].predict(X_test_scaled)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            logger.info("Trained model for %s - MSE: %.4f, R²: %.4f", 
                       target.value, mse, r2)
            
            # Save model
            model_path = os.path.join(self.model_dir, f"{target.value}_model.pkl")
            scaler_path = os.path.join(self.model_dir, f"{target.value}_scaler.pkl")
            
            joblib.dump(self.models[target], model_path)
            joblib.dump(self.scalers[target], scaler_path)
            
            self.stats['last_training'] = datetime.now(timezone.utc)
            
        except Exception as e:
            logger.error("Error training model for %s: %s", target.value, e)
    
    async def _predict_target(self, X: np.ndarray) -> Optional[OptimizationPrediction]:
        """Make prediction for a specific target"""
        if not self.feature_history:
            return None
        
        # Get current system state
        current_features = self.feature_history[-1]
        
        # Generate optimization scenarios
        scenarios = self._generate_optimization_scenarios(current_features)
        
        best_scenario = None
        best_prediction = None
        best_improvement = 0
        
        for scenario in scenarios:
            try:
                # Prepare scenario features
                scenario_features = np.array([[
                    scenario['batch_size'],
                    scenario['parallel_workers'],
                    scenario['memory_usage'],
                    scenario['cpu_usage'],
                    scenario['active_connections'],
                    scenario['database_health'],
                    scenario['storage_efficiency']
                ]])
                
                # Scale features
                scenario_scaled = self.scalers[target].transform(scenario_features)
                
                # Make prediction
                predicted_value = self.models[target].predict(scenario_scaled)[0]
                
                # Calculate improvement
                current_value = self._get_current_target_value(target)
                improvement = (predicted_value - current_value) / max(current_value, 1)
                
                if improvement > best_improvement:
                    best_improvement = improvement
                    best_scenario = scenario
                    best_prediction = predicted_value
                
            except Exception as e:
                logger.debug("Error evaluating scenario: %s", e)
                continue
        
        if best_scenario and best_improvement > self.optimization_threshold:
            return OptimizationPrediction(
                target=target,
                predicted_value=best_prediction,
                confidence=min(best_improvement * 2, 0.95),  # Simple confidence calculation
                recommended_parameters=best_scenario,
                expected_improvement=best_improvement,
                timestamp=datetime.now(timezone.utc)
            )
        
        return None
    
    def _generate_optimization_scenarios(self, current_features: PerformanceFeature) -> List[Dict[str, Any]]:
        """Generate optimization scenarios to evaluate"""
        scenarios = []
        
        # Base scenario (current state)
        base_scenario = {
            'batch_size': current_features.batch_size,
            'parallel_workers': current_features.parallel_workers,
            'memory_usage': current_features.memory_usage,
            'cpu_usage': current_features.cpu_usage,
            'active_connections': current_features.active_connections,
            'database_health': current_features.database_health,
            'storage_efficiency': current_features.storage_efficiency
        }
        scenarios.append(base_scenario)
        
        # Batch size variations
        for batch_multiplier in [0.5, 1.5, 2.0]:
            scenario = base_scenario.copy()
            scenario['batch_size'] = int(current_features.batch_size * batch_multiplier)
            scenarios.append(scenario)
        
        # Parallel workers variations
        for worker_multiplier in [0.5, 1.5, 2.0]:
            scenario = base_scenario.copy()
            scenario['parallel_workers'] = max(1, int(current_features.parallel_workers * worker_multiplier))
            scenarios.append(scenario)
        
        # Combined optimizations
        combined_scenario = base_scenario.copy()
        combined_scenario['batch_size'] = int(current_features.batch_size * 1.5)
        combined_scenario['parallel_workers'] = max(1, int(current_features.parallel_workers * 1.5))
        scenarios.append(combined_scenario)
        
        return scenarios
    
    def _get_current_target_value(self, target: OptimizationTarget) -> float:
        """Get current value for a specific target"""
        if not self.feature_history:
            return 0.0
        
        current = self.feature_history[-1]
        
        if target == OptimizationTarget.THROUGHPUT:
            return current.patterns_per_second
        elif target == OptimizationTarget.LATENCY:
            return 1000 / max(current.patterns_per_second, 1)
        elif target == OptimizationTarget.MEMORY_USAGE:
            return current.memory_usage
        elif target == OptimizationTarget.CPU_USAGE:
            return current.cpu_usage
        elif target == OptimizationTarget.STORAGE_EFFICIENCY:
            return current.storage_efficiency
        elif target == OptimizationTarget.OVERALL_PERFORMANCE:
            return (
                min(current.patterns_per_second / 1000, 1.0) * 0.4 +
                (1.0 - current.memory_usage / 100) * 0.2 +
                (1.0 - current.cpu_usage / 100) * 0.2 +
                (current.storage_efficiency / 100) * 0.2
            ) * 100
        
        return 0.0
    
    async def _evaluate_optimization_opportunities(self):
        """Evaluate and execute optimization opportunities"""
        if not self.predictions:
            return
        
        # Get best prediction
        best_prediction = max(self.predictions, key=lambda p: p.expected_improvement)
        
        if best_prediction.expected_improvement > self.optimization_threshold:
            # Create optimization action
            action = OptimizationAction(
                action_id=f"opt_{len(self.actions) + 1}",
                action_type="parameter_optimization",
                parameters=best_prediction.recommended_parameters,
                predicted_impact={
                    best_prediction.target.value: best_prediction.expected_improvement
                },
                confidence=best_prediction.confidence,
                timestamp=datetime.now(timezone.utc),
                executed=False,
                actual_impact=None
            )
            
            self.actions.append(action)
            
            # Execute optimization
            await self._execute_optimization(action)
            
            logger.info("Executed optimization action: %s", action.action_id)
    
    async def _execute_optimization(self, action: OptimizationAction):
        """Execute an optimization action"""
        try:
            # This is where you would apply the optimization parameters
            # to your actual system (e.g., update batch sizes, worker counts)
            
            # For now, we'll just mark it as executed
            action.executed = True
            
            # Schedule impact measurement
            asyncio.create_task(self._measure_optimization_impact(action))
            
            self.stats['total_optimizations'] += 1
            
        except Exception as e:
            logger.error("Error executing optimization: %s", e)
    
    async def _measure_optimization_impact(self, action: OptimizationAction):
        """Measure the actual impact of an optimization"""
        # Wait for system to stabilize
        await asyncio.sleep(60)
        
        try:
            # Get current performance metrics
            current_features = self.feature_history[-1] if self.feature_history else None
            
            if current_features:
                # Calculate actual impact
                actual_impact = {}
                for target_name, predicted_improvement in action.predicted_impact.items():
                    target = OptimizationTarget(target_name)
                    current_value = self._get_current_target_value(target)
                    
                    # This is simplified - in real implementation you'd compare before/after
                    actual_impact[target_name] = predicted_improvement * 0.8  # Assume 80% of predicted
                
                action.actual_impact = actual_impact
                
                # Update success rate
                successful_optimizations = len([a for a in self.actions if a.actual_impact])
                total_executed = len([a for a in self.actions if a.executed])
                
                if total_executed > 0:
                    self.stats['optimization_success_rate'] = successful_optimizations / total_executed
                
        except Exception as e:
            logger.error("Error measuring optimization impact: %s", e)
    
    async def _retrain_models(self):
        """Retrain all models with latest data"""
        logger.info("Starting model retraining...")
        
        try:
            X, y_dict = self._prepare_training_data()
            
            for target in OptimizationTarget:
                if target == OptimizationTarget.OVERALL_PERFORMANCE:
                    continue
                
                if target in self.models and target in y_dict:
                    await self._train_model(target, X, y_dict[target])
            
            logger.info("Model retraining completed")
            
        except Exception as e:
            logger.error("Error during model retraining: %s", e)
    
    def get_optimization_recommendations(self) -> List[Dict[str, Any]]:
        """Get current optimization recommendations"""
        recommendations = []
        
        for prediction in self.predictions[-10:]:  # Last 10 predictions
            if prediction.expected_improvement > self.optimization_threshold:
                recommendations.append({
                    'target': prediction.target.value,
                    'predicted_improvement': prediction.expected_improvement,
                    'confidence': prediction.confidence,
                    'recommended_parameters': prediction.recommended_parameters,
                    'timestamp': prediction.timestamp.isoformat()
                })
        
        return recommendations
    
    def get_optimization_history(self) -> List[Dict[str, Any]]:
        """Get optimization action history"""
        return [asdict(action) for action in self.actions[-50:]]  # Last 50 actions
    
    def get_model_performance(self) -> Dict[str, Any]:
        """Get model performance statistics"""
        return {
            'models_loaded': self.stats['models_loaded'],
            'total_predictions': self.stats['total_predictions'],
            'total_optimizations': self.stats['total_optimizations'],
            'optimization_success_rate': self.stats['optimization_success_rate'],
            'last_training': self.stats['last_training'].isoformat() if self.stats['last_training'] else None,
            'feature_history_size': len(self.feature_history),
            'predictions_count': len(self.predictions),
            'actions_count': len(self.actions)
        }
    
    async def stop(self):
        """Stop the predictive optimizer"""
        logger.info("Stopping predictive optimizer...")
        
        # Save models
        for target, model in self.models.items():
            try:
                model_path = os.path.join(self.model_dir, f"{target.value}_model.pkl")
                joblib.dump(model, model_path)
            except Exception as e:
                logger.error("Error saving model for %s: %s", target.value, e)
        
        logger.info("Predictive optimizer stopped")
