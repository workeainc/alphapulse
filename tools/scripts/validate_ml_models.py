#!/usr/bin/env python3
"""
ML Model Validation Script
Comprehensive validation and backtesting of trained ML models
"""

import asyncio
import logging
import sys
import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

import asyncpg
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from services.ml_models import NewsMLModels
from services.training_data_collector import TrainingDataCollector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelValidator:
    """Comprehensive model validation and backtesting"""
    
    def __init__(self, config, db_pool):
        self.config = config
        self.db_pool = db_pool
        self.ml_config = config.get('machine_learning', {})
        
        # Validation settings
        self.validation_split = 0.2
        self.backtest_days = 7
        self.performance_thresholds = {
            'min_accuracy': 0.6,
            'min_f1_score': 0.5,
            'max_mse': 0.1,
            'min_r2': 0.3
        }
    
    async def validate_models(self):
        """Comprehensive model validation"""
        try:
            logger.info("🔍 Starting comprehensive model validation...")
            
            # Initialize components
            training_collector = TrainingDataCollector(self.db_pool, self.config)
            ml_models = NewsMLModels(self.config, self.db_pool)
            
            # Load trained models
            await ml_models.load_trained_models()
            
            # Collect validation data
            data_result = await training_collector.collect_training_data()
            
            if data_result['total_samples'] == 0:
                logger.error("❌ No validation data available")
                return {}
            
            logger.info(f"📊 Using {len(data_result['validation'])} samples for validation")
            
            # Perform validation for each model
            validation_results = {}
            
            for model_name, model_config in self.ml_config.get('prediction_models', {}).items():
                if not model_config.get('enabled', False):
                    continue
                
                logger.info(f"🔍 Validating {model_name} model...")
                
                result = await self._validate_model(
                    model_name, model_config, ml_models, data_result['validation']
                )
                
                validation_results[model_name] = result
            
            # Perform backtesting
            logger.info("📈 Performing backtesting...")
            backtest_results = await self._perform_backtesting(ml_models)
            
            # Generate comprehensive report
            report = await self._generate_validation_report(validation_results, backtest_results)
            
            # Save validation results
            await self._save_validation_results(report)
            
            logger.info("✅ Model validation completed")
            return report
            
        except Exception as e:
            logger.error(f"❌ Error in model validation: {e}")
            return {}
    
    async def _validate_model(self, model_name, model_config, ml_models, validation_data):
        """Validate a specific model"""
        try:
            model_type = model_config.get('model_type', 'lightgbm')
            
            # Prepare validation data
            X_val, y_val = self._prepare_validation_data(model_name, validation_data)
            
            if len(X_val) == 0:
                return {'error': 'No validation data available'}
            
            # Get predictions
            predictions = []
            confidences = []
            
            for i, features in enumerate(X_val):
                feature_dict = dict(zip(self._get_feature_names(), features))
                
                if model_name == 'impact_prediction':
                    pred = await ml_models.predict_news_impact(feature_dict)
                elif model_name == 'sentiment_enhancement':
                    pred = await ml_models.enhance_sentiment_prediction(feature_dict)
                else:  # timing_optimization
                    pred = await ml_models.optimize_timing_prediction(feature_dict)
                
                predictions.append(pred.prediction)
                confidences.append(pred.confidence)
            
            predictions = np.array(predictions)
            confidences = np.array(confidences)
            
            # Calculate metrics
            metrics = self._calculate_validation_metrics(y_val, predictions, confidences)
            
            # Check performance thresholds
            performance_status = self._check_performance_thresholds(metrics)
            
            result = {
                'model_name': model_name,
                'model_type': model_type,
                'validation_samples': len(X_val),
                'metrics': metrics,
                'performance_status': performance_status,
                'validation_timestamp': datetime.utcnow().isoformat()
            }
            
            logger.info(f"✅ {model_name} validation completed:")
            logger.info(f"   - Validation Samples: {len(X_val)}")
            logger.info(f"   - MSE: {metrics['mse']:.4f}")
            logger.info(f"   - R²: {metrics['r2']:.3f}")
            logger.info(f"   - Performance Status: {performance_status}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error validating {model_name}: {e}")
            return {'error': str(e)}
    
    def _prepare_validation_data(self, model_name, validation_data):
        """Prepare validation data"""
        try:
            features_list = []
            targets = []
            
            for data_point in validation_data:
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
            logger.error(f"❌ Error preparing validation data: {e}")
            return np.array([]), np.array([])
    
    def _get_feature_names(self):
        """Get feature names"""
        return [
            'title_length', 'content_length', 'entity_count',
            'sentiment_score', 'normalized_sentiment', 'sentiment_confidence',
            'market_regime_score', 'btc_dominance', 'market_volatility',
            'correlation_30m', 'correlation_2h', 'correlation_24h',
            'hour_of_day', 'day_of_week', 'is_market_hours',
            'social_volume', 'cross_source_validation', 'feed_credibility'
        ]
    
    def _calculate_validation_metrics(self, y_true, y_pred, confidences):
        """Calculate comprehensive validation metrics"""
        try:
            metrics = {}
            
            # Regression metrics
            metrics['mse'] = mean_squared_error(y_true, y_pred)
            metrics['mae'] = mean_absolute_error(y_true, y_pred)
            metrics['r2'] = r2_score(y_true, y_pred)
            metrics['rmse'] = np.sqrt(metrics['mse'])
            
            # Additional metrics
            metrics['mean_confidence'] = np.mean(confidences)
            metrics['confidence_std'] = np.std(confidences)
            
            # Correlation between confidence and accuracy
            errors = np.abs(y_true - y_pred)
            confidence_accuracy_corr = np.corrcoef(confidences, -errors)[0, 1]
            metrics['confidence_accuracy_correlation'] = confidence_accuracy_corr if not np.isnan(confidence_accuracy_corr) else 0.0
            
            # Prediction distribution
            metrics['prediction_mean'] = np.mean(y_pred)
            metrics['prediction_std'] = np.std(y_pred)
            metrics['prediction_min'] = np.min(y_pred)
            metrics['prediction_max'] = np.max(y_pred)
            
            return metrics
            
        except Exception as e:
            logger.error(f"❌ Error calculating metrics: {e}")
            return {}
    
    def _check_performance_thresholds(self, metrics):
        """Check if model meets performance thresholds"""
        try:
            status = {
                'overall_status': 'PASS',
                'failed_checks': []
            }
            
            # Check MSE
            if metrics.get('mse', float('inf')) > self.performance_thresholds['max_mse']:
                status['overall_status'] = 'FAIL'
                status['failed_checks'].append('mse_too_high')
            
            # Check R²
            if metrics.get('r2', -float('inf')) < self.performance_thresholds['min_r2']:
                status['overall_status'] = 'FAIL'
                status['failed_checks'].append('r2_too_low')
            
            # Check confidence
            if metrics.get('mean_confidence', 0) < 0.3:
                status['overall_status'] = 'WARNING'
                status['failed_checks'].append('low_confidence')
            
            return status
            
        except Exception as e:
            logger.error(f"❌ Error checking performance thresholds: {e}")
            return {'overall_status': 'ERROR', 'failed_checks': ['error']}
    
    async def _perform_backtesting(self, ml_models):
        """Perform backtesting on recent data"""
        try:
            logger.info("📈 Performing backtesting on recent data...")
            
            # Get recent news data for backtesting
            backtest_data = await self._get_backtest_data()
            
            if not backtest_data:
                logger.warning("⚠️ No recent data available for backtesting")
                return {}
            
            backtest_results = {
                'impact_prediction': await self._backtest_model('impact_prediction', ml_models, backtest_data),
                'sentiment_enhancement': await self._backtest_model('sentiment_enhancement', ml_models, backtest_data),
                'timing_optimization': await self._backtest_model('timing_optimization', ml_models, backtest_data)
            }
            
            return backtest_results
            
        except Exception as e:
            logger.error(f"❌ Error in backtesting: {e}")
            return {}
    
    async def _get_backtest_data(self):
        """Get recent data for backtesting"""
        try:
            async with self.db_pool.acquire() as conn:
                # Get recent news from the last N days
                lookback_date = datetime.utcnow() - timedelta(days=self.backtest_days)
                
                query = """
                    SELECT
                        id, title, content, published_at, sentiment_score,
                        entities, market_regime_score, btc_dominance,
                        market_volatility, social_volume, cross_source_validation,
                        feed_credibility, correlation_30m, correlation_2h, correlation_24h,
                        normalized_sentiment, sentiment_confidence,
                        hour_of_day, day_of_week, is_market_hours
                    FROM raw_news_content
                    WHERE published_at >= $1
                    AND sentiment_score IS NOT NULL
                    ORDER BY published_at DESC
                    LIMIT 1000;
                """
                
                rows = await conn.fetch(query, lookback_date)
                
                backtest_data = []
                for row in rows:
                    backtest_data.append({
                        'id': row['id'],
                        'title': row['title'],
                        'content': row['content'],
                        'published_at': row['published_at'],
                        'sentiment_score': float(row['sentiment_score'] or 0.0),
                        'entities': row['entities'] or [],
                        'market_regime_score': float(row['market_regime_score'] or 0.0),
                        'btc_dominance': float(row['btc_dominance'] or 50.0),
                        'market_volatility': float(row['market_volatility'] or 0.02),
                        'social_volume': float(row['social_volume'] or 0.0),
                        'cross_source_validation': float(row['cross_source_validation'] or 0.0),
                        'feed_credibility': float(row['feed_credibility'] or 0.5),
                        'correlation_30m': float(row['correlation_30m'] or 0.0),
                        'correlation_2h': float(row['correlation_2h'] or 0.0),
                        'correlation_24h': float(row['correlation_24h'] or 0.0),
                        'normalized_sentiment': float(row['normalized_sentiment'] or 0.0),
                        'sentiment_confidence': float(row['sentiment_confidence'] or 0.5),
                        'hour_of_day': int(row['hour_of_day'] or 12),
                        'day_of_week': int(row['day_of_week'] or 3),
                        'is_market_hours': int(row['is_market_hours'] or 1)
                    })
                
                logger.info(f"📊 Collected {len(backtest_data)} samples for backtesting")
                return backtest_data
                
        except Exception as e:
            logger.error(f"❌ Error getting backtest data: {e}")
            return []
    
    async def _backtest_model(self, model_name, ml_models, backtest_data):
        """Backtest a specific model"""
        try:
            predictions = []
            confidences = []
            timestamps = []
            
            for data_point in backtest_data:
                # Prepare features
                features = {
                    'title_length': len(data_point['title']),
                    'content_length': len(data_point['content']) if data_point['content'] else 0,
                    'entity_count': len(data_point['entities']),
                    'sentiment_score': data_point['sentiment_score'],
                    'normalized_sentiment': data_point['normalized_sentiment'],
                    'sentiment_confidence': data_point['sentiment_confidence'],
                    'market_regime_score': data_point['market_regime_score'],
                    'btc_dominance': data_point['btc_dominance'],
                    'market_volatility': data_point['market_volatility'],
                    'correlation_30m': data_point['correlation_30m'],
                    'correlation_2h': data_point['correlation_2h'],
                    'correlation_24h': data_point['correlation_24h'],
                    'hour_of_day': data_point['hour_of_day'],
                    'day_of_week': data_point['day_of_week'],
                    'is_market_hours': data_point['is_market_hours'],
                    'social_volume': data_point['social_volume'],
                    'cross_source_validation': data_point['cross_source_validation'],
                    'feed_credibility': data_point['feed_credibility']
                }
                
                # Get prediction
                if model_name == 'impact_prediction':
                    pred = await ml_models.predict_news_impact(features)
                elif model_name == 'sentiment_enhancement':
                    pred = await ml_models.enhance_sentiment_prediction(features)
                else:  # timing_optimization
                    pred = await ml_models.optimize_timing_prediction(features)
                
                predictions.append(pred.prediction)
                confidences.append(pred.confidence)
                timestamps.append(data_point['published_at'])
            
            # Calculate backtest metrics
            backtest_metrics = {
                'total_predictions': len(predictions),
                'mean_prediction': np.mean(predictions),
                'prediction_std': np.std(predictions),
                'mean_confidence': np.mean(confidences),
                'confidence_std': np.std(confidences),
                'prediction_range': [np.min(predictions), np.max(predictions)],
                'high_confidence_predictions': sum(1 for c in confidences if c > 0.7),
                'low_confidence_predictions': sum(1 for c in confidences if c < 0.3)
            }
            
            return {
                'model_name': model_name,
                'backtest_metrics': backtest_metrics,
                'predictions': predictions,
                'confidences': confidences,
                'timestamps': [ts.isoformat() for ts in timestamps]
            }
            
        except Exception as e:
            logger.error(f"❌ Error backtesting {model_name}: {e}")
            return {'error': str(e)}
    
    async def _generate_validation_report(self, validation_results, backtest_results):
        """Generate comprehensive validation report"""
        try:
            report = {
                'validation_summary': {
                    'total_models': len(validation_results),
                    'passed_models': sum(1 for r in validation_results.values() if 'error' not in r and r.get('performance_status', {}).get('overall_status') == 'PASS'),
                    'failed_models': sum(1 for r in validation_results.values() if 'error' not in r and r.get('performance_status', {}).get('overall_status') == 'FAIL'),
                    'warning_models': sum(1 for r in validation_results.values() if 'error' not in r and r.get('performance_status', {}).get('overall_status') == 'WARNING'),
                    'error_models': sum(1 for r in validation_results.values() if 'error' in r)
                },
                'validation_results': validation_results,
                'backtest_results': backtest_results,
                'report_timestamp': datetime.utcnow().isoformat()
            }
            
            # Add recommendations
            report['recommendations'] = self._generate_recommendations(validation_results, backtest_results)
            
            return report
            
        except Exception as e:
            logger.error(f"❌ Error generating validation report: {e}")
            return {}
    
    def _generate_recommendations(self, validation_results, backtest_results):
        """Generate recommendations based on validation results"""
        try:
            recommendations = []
            
            for model_name, result in validation_results.items():
                if 'error' in result:
                    recommendations.append(f"❌ {model_name}: Fix model loading/training issues")
                    continue
                
                status = result.get('performance_status', {})
                metrics = result.get('metrics', {})
                
                if status.get('overall_status') == 'FAIL':
                    recommendations.append(f"🔧 {model_name}: Retrain with more data or adjust hyperparameters")
                
                if status.get('overall_status') == 'WARNING':
                    recommendations.append(f"⚠️ {model_name}: Monitor performance and consider retraining")
                
                if metrics.get('mean_confidence', 0) < 0.5:
                    recommendations.append(f"📊 {model_name}: Low confidence predictions - consider feature engineering")
                
                if metrics.get('r2', 0) < 0.5:
                    recommendations.append(f"📈 {model_name}: Low R² score - consider model selection or feature selection")
            
            # Add general recommendations
            if not recommendations:
                recommendations.append("✅ All models performing well - continue monitoring")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"❌ Error generating recommendations: {e}")
            return ["Error generating recommendations"]
    
    async def _save_validation_results(self, report):
        """Save validation results to file"""
        try:
            # Create models directory if it doesn't exist
            models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
            os.makedirs(models_dir, exist_ok=True)
            
            # Save report
            report_path = os.path.join(models_dir, f"model_validation_report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json")
            
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            logger.info(f"✅ Validation report saved to {report_path}")
            
        except Exception as e:
            logger.error(f"❌ Error saving validation results: {e}")

async def main():
    """Main validation function"""
    try:
        logger.info("🚀 Starting ML Model Validation Process")
        
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
        
        # Initialize validator
        validator = ModelValidator(config['enhanced_news_events'], db_pool)
        
        # Run validation
        report = await validator.validate_models()
        
        # Display results
        logger.info("📊 Validation Results Summary")
        logger.info("=" * 60)
        
        if report:
            summary = report.get('validation_summary', {})
            logger.info(f"📈 Validation Summary:")
            logger.info(f"   - Total Models: {summary.get('total_models', 0)}")
            logger.info(f"   - Passed: {summary.get('passed_models', 0)}")
            logger.info(f"   - Failed: {summary.get('failed_models', 0)}")
            logger.info(f"   - Warnings: {summary.get('warning_models', 0)}")
            logger.info(f"   - Errors: {summary.get('error_models', 0)}")
            
            logger.info("")
            logger.info("🔍 Detailed Results:")
            
            for model_name, result in report.get('validation_results', {}).items():
                if 'error' in result:
                    logger.error(f"❌ {model_name}: {result['error']}")
                else:
                    metrics = result.get('metrics', {})
                    status = result.get('performance_status', {})
                    logger.info(f"✅ {model_name}:")
                    logger.info(f"   - Status: {status.get('overall_status', 'UNKNOWN')}")
                    logger.info(f"   - MSE: {metrics.get('mse', 0):.4f}")
                    logger.info(f"   - R²: {metrics.get('r2', 0):.3f}")
                    logger.info(f"   - Confidence: {metrics.get('mean_confidence', 0):.3f}")
            
            logger.info("")
            logger.info("💡 Recommendations:")
            for rec in report.get('recommendations', []):
                logger.info(f"   - {rec}")
        
        logger.info("=" * 60)
        logger.info("🎉 Model Validation Completed Successfully!")
        
    except Exception as e:
        logger.error(f"❌ Error in validation process: {e}")
        raise
    finally:
        if 'db_pool' in locals():
            await db_pool.close()
            logger.info("🔌 Database connection closed")

if __name__ == "__main__":
    asyncio.run(main())
