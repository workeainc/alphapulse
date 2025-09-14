#!/usr/bin/env python3
"""
Hyperparameter Optimization Script
Optimizes ML model hyperparameters using GridSearchCV
"""

import asyncio
import logging
import sys
import os
import json
from datetime import datetime

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

import asyncpg
import numpy as np
from sklearn.model_selection import GridSearchCV
from services.ml_models import NewsMLModels
from services.training_data_collector import TrainingDataCollector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class HyperparameterOptimizer:
    """Hyperparameter optimization for ML models"""
    
    def __init__(self, config, db_pool):
        self.config = config
        self.db_pool = db_pool
        self.ml_config = config.get('machine_learning', {})
        
        # Define hyperparameter grids for each model type
        self.hyperparameter_grids = {
            'lightgbm': {
                'n_estimators': [100, 200, 300],
                'max_depth': [6, 8, 10],
                'learning_rate': [0.01, 0.05, 0.1],
                'num_leaves': [31, 63, 127],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            },
            'xgboost': {
                'n_estimators': [100, 200, 300],
                'max_depth': [6, 8, 10],
                'learning_rate': [0.01, 0.05, 0.1],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0],
                'reg_alpha': [0, 0.1, 0.5],
                'reg_lambda': [0, 0.1, 0.5]
            },
            'random_forest': {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None]
            }
        }
    
    async def optimize_hyperparameters(self):
        """Optimize hyperparameters for all models"""
        try:
            logger.info("🔧 Starting hyperparameter optimization...")
            
            # Initialize training data collector
            training_collector = TrainingDataCollector(self.db_pool, self.config)
            
            # Collect training data
            data_result = await training_collector.collect_training_data()
            
            if data_result['total_samples'] == 0:
                logger.error("❌ No training data available for optimization")
                return {}
            
            logger.info(f"📊 Using {data_result['total_samples']} samples for optimization")
            
            # Optimize each model
            optimization_results = {}
            
            for model_name, model_config in self.ml_config.get('prediction_models', {}).items():
                if not model_config.get('enabled', False):
                    continue
                
                logger.info(f"🔧 Optimizing {model_name} model...")
                
                result = await self._optimize_model_hyperparameters(
                    model_name, model_config, training_collector
                )
                
                optimization_results[model_name] = result
            
            # Save optimization results
            await self._save_optimization_results(optimization_results)
            
            logger.info("✅ Hyperparameter optimization completed")
            return optimization_results
            
        except Exception as e:
            logger.error(f"❌ Error in hyperparameter optimization: {e}")
            return {}
    
    async def _optimize_model_hyperparameters(self, model_name, model_config, training_collector):
        """Optimize hyperparameters for a specific model"""
        try:
            model_type = model_config.get('model_type', 'lightgbm')
            
            # Prepare training data
            X_train, y_train = self._prepare_training_data(model_name, training_collector.training_data)
            X_val, y_val = self._prepare_training_data(model_name, training_collector.validation_data)
            
            if len(X_train) < 100:
                logger.warning(f"⚠️ Insufficient data for {model_name}: {len(X_train)} samples")
                return {'error': 'Insufficient training data'}
            
            # Get hyperparameter grid
            param_grid = self.hyperparameter_grids.get(model_type, {})
            
            if not param_grid:
                logger.warning(f"⚠️ No hyperparameter grid defined for {model_type}")
                return {'error': 'No hyperparameter grid defined'}
            
            # Create base model
            base_model = self._create_base_model(model_type)
            
            # Perform grid search
            logger.info(f"🔍 Performing grid search for {model_name}...")
            
            grid_search = GridSearchCV(
                estimator=base_model,
                param_grid=param_grid,
                cv=3,
                scoring='neg_mean_squared_error',
                n_jobs=-1,
                verbose=1
            )
            
            # Fit grid search
            grid_search.fit(X_train, y_train)
            
            # Get best parameters and score
            best_params = grid_search.best_params_
            best_score = -grid_search.best_score_  # Convert back to positive MSE
            
            # Evaluate on validation set
            best_model = grid_search.best_estimator_
            y_pred_val = best_model.predict(X_val)
            
            # Calculate validation metrics
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            val_mse = mean_squared_error(y_val, y_pred_val)
            val_mae = mean_absolute_error(y_val, y_pred_val)
            val_r2 = r2_score(y_val, y_pred_val)
            
            result = {
                'model_name': model_name,
                'model_type': model_type,
                'best_parameters': best_params,
                'best_cv_score': best_score,
                'validation_metrics': {
                    'mse': val_mse,
                    'mae': val_mae,
                    'r2': val_r2
                },
                'cv_results': {
                    'mean_test_score': grid_search.cv_results_['mean_test_score'].tolist(),
                    'std_test_score': grid_search.cv_results_['std_test_score'].tolist(),
                    'params': grid_search.cv_results_['params']
                },
                'optimization_timestamp': datetime.utcnow().isoformat()
            }
            
            logger.info(f"✅ {model_name} optimization completed:")
            logger.info(f"   - Best CV Score (MSE): {best_score:.4f}")
            logger.info(f"   - Validation MSE: {val_mse:.4f}")
            logger.info(f"   - Validation R²: {val_r2:.3f}")
            logger.info(f"   - Best Parameters: {best_params}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error optimizing {model_name}: {e}")
            return {'error': str(e)}
    
    def _prepare_training_data(self, model_name, data_points):
        """Prepare training data for optimization"""
        try:
            features_list = []
            targets = []
            
            for data_point in data_points:
                # Extract features
                features = list(data_point.features.values())
                features_list.append(features)
                
                # Extract target based on model type
                if model_name == 'impact_prediction':
                    target = data_point.labels['price_impact_24h']
                elif model_name == 'sentiment_enhancement':
                    target = data_point.labels['enhanced_sentiment']
                else:  # timing_optimization
                    target = data_point.labels['optimal_timing_score']
                
                targets.append(target)
            
            return np.array(features_list), np.array(targets)
            
        except Exception as e:
            logger.error(f"❌ Error preparing training data: {e}")
            return np.array([]), np.array([])
    
    def _create_base_model(self, model_type):
        """Create base model for optimization"""
        try:
            if model_type == 'lightgbm':
                import lightgbm as lgb
                return lgb.LGBMRegressor(random_state=42, verbose=-1)
            elif model_type == 'xgboost':
                import xgboost as xgb
                return xgb.XGBRegressor(random_state=42)
            elif model_type == 'random_forest':
                from sklearn.ensemble import RandomForestRegressor
                return RandomForestRegressor(random_state=42)
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
                
        except Exception as e:
            logger.error(f"❌ Error creating base model: {e}")
            raise
    
    async def _save_optimization_results(self, optimization_results):
        """Save optimization results to file"""
        try:
            # Create models directory if it doesn't exist
            models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
            os.makedirs(models_dir, exist_ok=True)
            
            # Save results
            results_path = os.path.join(models_dir, f"hyperparameter_optimization_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json")
            
            # Convert to serializable format
            serializable_results = {}
            for model_name, result in optimization_results.items():
                if 'error' not in result:
                    serializable_results[model_name] = {
                        'model_name': result['model_name'],
                        'model_type': result['model_type'],
                        'best_parameters': result['best_parameters'],
                        'best_cv_score': result['best_cv_score'],
                        'validation_metrics': result['validation_metrics'],
                        'optimization_timestamp': result['optimization_timestamp']
                    }
                else:
                    serializable_results[model_name] = {'error': result['error']}
            
            with open(results_path, 'w') as f:
                json.dump(serializable_results, f, indent=2)
            
            logger.info(f"✅ Optimization results saved to {results_path}")
            
            # Update config with best parameters
            await self._update_config_with_best_parameters(optimization_results)
            
        except Exception as e:
            logger.error(f"❌ Error saving optimization results: {e}")
    
    async def _update_config_with_best_parameters(self, optimization_results):
        """Update config file with best hyperparameters"""
        try:
            config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'enhanced_news_config.json')
            
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Update hyperparameters for each model
            for model_name, result in optimization_results.items():
                if 'error' not in result and 'best_parameters' in result:
                    model_config_path = ['enhanced_news_events', 'machine_learning', 'prediction_models', model_name, 'hyperparameters']
                    
                    # Navigate to the correct location in config
                    current = config
                    for path_part in model_config_path[:-1]:
                        if path_part not in current:
                            current[path_part] = {}
                        current = current[path_part]
                    
                    # Update hyperparameters
                    current[model_config_path[-1]] = result['best_parameters']
            
            # Save updated config
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            logger.info("✅ Configuration updated with best hyperparameters")
            
        except Exception as e:
            logger.error(f"❌ Error updating config: {e}")

async def main():
    """Main optimization function"""
    try:
        logger.info("🚀 Starting Hyperparameter Optimization Process")
        
        # Load configuration
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'enhanced_news_config.json')
        
        if not os.path.exists(config_path):
            logger.error(f"❌ Configuration file not found: {config_path}")
            return
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Database connection
        db_config = config['enhanced_news_events']['database']
        
        logger.info("🔌 Connecting to database...")
        db_pool = await asyncpg.create_pool(
            host=db_config['host'],
            port=db_config['port'],
            database=db_config['database'],
            user=db_config['username'],
            password=db_config['password'],
            min_size=5,
            max_size=20
        )
        
        logger.info("✅ Database connected successfully")
        
        # Initialize optimizer
        optimizer = HyperparameterOptimizer(config['enhanced_news_events'], db_pool)
        
        # Run optimization
        results = await optimizer.optimize_hyperparameters()
        
        # Display results
        logger.info("📈 Optimization Results Summary")
        logger.info("=" * 60)
        
        for model_name, result in results.items():
            if 'error' in result:
                logger.error(f"❌ {model_name}: {result['error']}")
            else:
                logger.info(f"✅ {model_name} Model:")
                logger.info(f"   - Model Type: {result['model_type']}")
                logger.info(f"   - Best CV Score (MSE): {result['best_cv_score']:.4f}")
                logger.info(f"   - Validation MSE: {result['validation_metrics']['mse']:.4f}")
                logger.info(f"   - Validation R²: {result['validation_metrics']['r2']:.3f}")
                logger.info(f"   - Best Parameters: {result['best_parameters']}")
                logger.info("")
        
        logger.info("=" * 60)
        logger.info("🎉 Hyperparameter Optimization Completed Successfully!")
        
        # Summary
        successful_optimizations = sum(1 for result in results.values() if 'error' not in result)
        total_models = len(results)
        
        logger.info(f"📊 Final Summary:")
        logger.info(f"   - Models Optimized: {successful_optimizations}/{total_models}")
        logger.info(f"   - Config Updated: {successful_optimizations > 0}")
        
    except Exception as e:
        logger.error(f"❌ Error in optimization process: {e}")
        raise
    finally:
        if 'db_pool' in locals():
            await db_pool.close()
            logger.info("🔌 Database connection closed")

if __name__ == "__main__":
    asyncio.run(main())
