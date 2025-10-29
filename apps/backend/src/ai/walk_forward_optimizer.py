"""
Walk-Forward Optimizer for AlphaPulse
Phase 4: Walk-forward optimization with Optuna for hyperparameter tuning
"""

import asyncio
import logging
import time
import numpy as np
import json
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import deque
from datetime import datetime, timedelta

# Machine learning imports
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split, TimeSeriesSplit
    from sklearn.metrics import accuracy_score, f1_score
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print("Scikit-learn not available - ML features disabled")

# Optuna for hyperparameter optimization
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("Optuna not available - hyperparameter optimization disabled")

logger = logging.getLogger(__name__)

@dataclass
class OptimizationResult:
    """Optimization result"""
    window_id: int
    best_params: Dict[str, Any]
    best_score: float
    final_accuracy: float
    final_f1_score: float
    n_trials: int
    window_size: int
    validation_size: int
    optimization_time: float
    timestamp: datetime

class WalkForwardOptimizer:
    """Walk-forward optimization with Optuna for hyperparameter tuning"""
    
    def __init__(self, 
                 window_size: int = 1000,
                 step_size: int = 100,
                 n_trials: int = 50,
                 optimization_metric: str = 'f1_score'):
        
        self.window_size = window_size
        self.step_size = step_size
        self.n_trials = n_trials
        self.optimization_metric = optimization_metric
        
        self.optimization_history = []
        self.best_params_history = []
        
        if not OPTUNA_AVAILABLE:
            logger.warning("Optuna not available - walk-forward optimization disabled")
        
        logger.info("Walk-Forward Optimizer initialized")
    
    async def optimize_window(self, 
                            training_data: List[Dict[str, Any]],
                            validation_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Optimize hyperparameters for a specific window"""
        if not OPTUNA_AVAILABLE or not ML_AVAILABLE:
            return {'status': 'dependencies_not_available'}
        
        try:
            def objective(trial):
                # Suggest hyperparameters
                n_estimators = trial.suggest_int('n_estimators', 50, 300)
                max_depth = trial.suggest_int('max_depth', 3, 20)
                min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
                min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
                
                # Create model with suggested parameters
                model = RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf,
                    random_state=42
                )
                
                # Train and evaluate
                X_train = [item['features'] for item in training_data]
                y_train = [item['label'] for item in training_data]
                X_val = [item['features'] for item in validation_data]
                y_val = [item['label'] for item in validation_data]
                
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
                
                # Return metric to optimize
                if self.optimization_metric == 'f1_score':
                    return f1_score(y_val, y_pred, average='weighted')
                elif self.optimization_metric == 'accuracy':
                    return accuracy_score(y_val, y_pred)
                else:
                    return f1_score(y_val, y_pred, average='weighted')
            
            # Create study and optimize
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=self.n_trials)
            
            # Get best parameters
            best_params = study.best_params
            best_score = study.best_value
            
            # Train final model with best parameters
            final_model = RandomForestClassifier(
                **best_params,
                random_state=42
            )
            
            X_train = [item['features'] for item in training_data]
            y_train = [item['label'] for item in training_data]
            final_model.fit(X_train, y_train)
            
            # Evaluate on validation set
            X_val = [item['features'] for item in validation_data]
            y_val = [item['label'] for item in validation_data]
            y_pred = final_model.predict(X_val)
            
            final_accuracy = accuracy_score(y_val, y_pred)
            final_f1 = f1_score(y_val, y_pred, average='weighted')
            
            result = {
                'status': 'success',
                'best_params': best_params,
                'best_score': best_score,
                'final_accuracy': final_accuracy,
                'final_f1_score': final_f1,
                'n_trials': self.n_trials,
                'window_size': len(training_data),
                'validation_size': len(validation_data)
            }
            
            self.optimization_history.append(result)
            self.best_params_history.append(best_params)
            
            logger.info(f"✅ Window optimization completed - F1: {final_f1:.3f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in window optimization: {e}")
            return {'status': 'error', 'error': str(e)}
    
    async def walk_forward_optimization(self, 
                                      data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform walk-forward optimization on the entire dataset"""
        if len(data) < self.window_size + self.step_size:
            return {'status': 'insufficient_data'}
        
        try:
            results = []
            current_pos = 0
            window_id = 0
            
            while current_pos + self.window_size < len(data):
                # Define training and validation windows
                train_start = current_pos
                train_end = current_pos + self.window_size
                val_start = train_end
                val_end = min(val_start + self.step_size, len(data))
                
                training_data = data[train_start:train_end]
                validation_data = data[val_start:val_end]
                
                # Optimize for this window
                window_result = await self.optimize_window(training_data, validation_data)
                window_result['window_id'] = window_id
                results.append(window_result)
                
                # Move window forward
                current_pos += self.step_size
                window_id += 1
            
            # Aggregate results
            successful_results = [r for r in results if r['status'] == 'success']
            
            if successful_results:
                avg_f1 = np.mean([r['final_f1_score'] for r in successful_results])
                avg_accuracy = np.mean([r['final_accuracy'] for r in successful_results])
                
                return {
                    'status': 'success',
                    'total_windows': len(results),
                    'successful_windows': len(successful_results),
                    'average_f1_score': avg_f1,
                    'average_accuracy': avg_accuracy,
                    'window_results': results
                }
            else:
                return {'status': 'no_successful_windows'}
                
        except Exception as e:
            logger.error(f"Error in walk-forward optimization: {e}")
            return {'status': 'error', 'error': str(e)}
    
    async def rolling_retraining(self, 
                               data: List[Dict[str, Any]],
                               retrain_interval: int = 1000) -> Dict[str, Any]:
        """Perform rolling retraining with periodic model updates"""
        if len(data) < retrain_interval:
            return {'status': 'insufficient_data'}
        
        try:
            results = []
            current_pos = 0
            
            while current_pos + retrain_interval < len(data):
                # Get training data up to current position
                training_data = data[:current_pos + retrain_interval]
                
                # Get validation data (next batch)
                val_start = current_pos + retrain_interval
                val_end = min(val_start + self.step_size, len(data))
                validation_data = data[val_start:val_end]
                
                # Optimize and retrain
                retrain_result = await self.optimize_window(training_data, validation_data)
                retrain_result['retrain_position'] = current_pos
                results.append(retrain_result)
                
                # Move forward
                current_pos += retrain_interval
            
            # Aggregate results
            successful_results = [r for r in results if r['status'] == 'success']
            
            if successful_results:
                avg_f1 = np.mean([r['final_f1_score'] for r in successful_results])
                avg_accuracy = np.mean([r['final_accuracy'] for r in successful_results])
                
                return {
                    'status': 'success',
                    'total_retrains': len(results),
                    'successful_retrains': len(successful_results),
                    'average_f1_score': avg_f1,
                    'average_accuracy': avg_accuracy,
                    'retrain_results': results
                }
            else:
                return {'status': 'no_successful_retrains'}
                
        except Exception as e:
            logger.error(f"Error in rolling retraining: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get summary of optimization history"""
        if not self.optimization_history:
            return {'status': 'no_optimization_history'}
        
        try:
            f1_scores = [r['final_f1_score'] for r in self.optimization_history if r['status'] == 'success']
            accuracies = [r['final_accuracy'] for r in self.optimization_history if r['status'] == 'success']
            
            return {
                'total_optimizations': len(self.optimization_history),
                'successful_optimizations': len(f1_scores),
                'average_f1_score': np.mean(f1_scores) if f1_scores else 0.0,
                'average_accuracy': np.mean(accuracies) if accuracies else 0.0,
                'best_f1_score': max(f1_scores) if f1_scores else 0.0,
                'best_accuracy': max(accuracies) if accuracies else 0.0,
                'f1_score_std': np.std(f1_scores) if f1_scores else 0.0,
                'accuracy_std': np.std(accuracies) if accuracies else 0.0
            }
            
        except Exception as e:
            logger.error(f"Error getting optimization summary: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def get_best_params(self) -> Dict[str, Any]:
        """Get the best parameters from the most recent optimization"""
        if not self.best_params_history:
            return {}
        
        return self.best_params_history[-1]
    
    def get_optimization_trends(self) -> Dict[str, Any]:
        """Analyze optimization trends over time"""
        if len(self.optimization_history) < 2:
            return {'status': 'insufficient_history'}
        
        try:
            successful_results = [r for r in self.optimization_history if r['status'] == 'success']
            
            if len(successful_results) < 2:
                return {'status': 'insufficient_successful_results'}
            
            f1_scores = [r['final_f1_score'] for r in successful_results]
            accuracies = [r['final_accuracy'] for r in successful_results]
            
            # Calculate trends
            f1_trend = np.polyfit(range(len(f1_scores)), f1_scores, 1)[0]
            accuracy_trend = np.polyfit(range(len(accuracies)), accuracies, 1)[0]
            
            return {
                'f1_score_trend': f1_trend,
                'accuracy_trend': accuracy_trend,
                'f1_improving': f1_trend > 0,
                'accuracy_improving': accuracy_trend > 0,
                'total_optimizations': len(successful_results)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing optimization trends: {e}")
            return {'status': 'error', 'error': str(e)}

# Global instance
walk_forward_optimizer = WalkForwardOptimizer(
    window_size=1000,
    step_size=100,
    n_trials=50,
    optimization_metric='f1_score'
)
