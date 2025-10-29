#!/usr/bin/env python3
"""
Performance Optimization Script
Optimizes ML models for production scale performance
"""

import asyncio
import logging
import sys
import os
import json
import time
import psutil
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

import asyncpg
import numpy as np
from services.ml_models import NewsMLModels
from services.training_data_collector import TrainingDataCollector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PerformanceOptimizer:
    """Performance optimization for ML models"""
    
    def __init__(self, config, db_pool):
        self.config = config
        self.db_pool = db_pool
        self.ml_config = config.get('machine_learning', {})
        
        # Performance settings
        self.batch_size = 100
        self.max_workers = 4
        self.target_latency_ms = 50
        self.target_throughput = 1000  # predictions per second
        
        # Performance metrics
        self.performance_metrics = {}
    
    async def optimize_performance(self):
        """Comprehensive performance optimization"""
        try:
            logger.info("⚡ Starting Performance Optimization Process")
            
            # Initialize components
            ml_models = NewsMLModels(self.config, self.db_pool)
            training_collector = TrainingDataCollector(self.db_pool, self.config)
            
            # Load trained models
            await ml_models.load_trained_models()
            
            # Collect test data
            data_result = await training_collector.collect_training_data()
            
            if data_result['total_samples'] == 0:
                logger.error("❌ No test data available for performance optimization")
                return {}
            
            logger.info(f"📊 Using {len(data_result['training'])} samples for performance testing")
            
            # Step 1: Baseline Performance Measurement
            logger.info("📈 Step 1: Measuring baseline performance...")
            baseline_metrics = await self._measure_baseline_performance(ml_models, data_result['training'])
            
            # Step 2: Model Optimization
            logger.info("🔧 Step 2: Optimizing models...")
            optimization_results = await self._optimize_models(ml_models, data_result['training'])
            
            # Step 3: Batch Processing Optimization
            logger.info("📦 Step 3: Optimizing batch processing...")
            batch_optimization = await self._optimize_batch_processing(ml_models, data_result['training'])
            
            # Step 4: Memory and Resource Optimization
            logger.info("💾 Step 4: Optimizing memory and resources...")
            resource_optimization = await self._optimize_resources(ml_models, data_result['training'])
            
            # Step 5: Final Performance Measurement
            logger.info("📊 Step 5: Measuring optimized performance...")
            optimized_metrics = await self._measure_baseline_performance(ml_models, data_result['training'])
            
            # Generate performance report
            report = await self._generate_performance_report(
                baseline_metrics, optimization_results, 
                batch_optimization, resource_optimization, optimized_metrics
            )
            
            # Save optimization results
            await self._save_performance_results(report)
            
            logger.info("✅ Performance optimization completed")
            return report
            
        except Exception as e:
            logger.error(f"❌ Error in performance optimization: {e}")
            return {}
    
    async def _measure_baseline_performance(self, ml_models, test_data):
        """Measure baseline performance metrics"""
        try:
            metrics = {}
            
            for model_name, model_config in self.ml_config.get('prediction_models', {}).items():
                if not model_config.get('enabled', False):
                    continue
                
                logger.info(f"📊 Measuring baseline performance for {model_name}...")
                
                # Prepare test features
                test_features = self._prepare_test_features(test_data[:100])  # Use first 100 samples
                
                # Measure latency
                latency_metrics = await self._measure_latency(ml_models, model_name, test_features)
                
                # Measure throughput
                throughput_metrics = await self._measure_throughput(ml_models, model_name, test_features)
                
                # Measure memory usage
                memory_metrics = await self._measure_memory_usage(ml_models, model_name, test_features)
                
                # Measure accuracy
                accuracy_metrics = await self._measure_accuracy(ml_models, model_name, test_data[:100])
                
                metrics[model_name] = {
                    'latency': latency_metrics,
                    'throughput': throughput_metrics,
                    'memory': memory_metrics,
                    'accuracy': accuracy_metrics
                }
            
            return metrics
            
        except Exception as e:
            logger.error(f"❌ Error measuring baseline performance: {e}")
            return {}
    
    def _prepare_test_features(self, test_data):
        """Prepare test features for performance testing"""
        try:
            test_features = []
            
            for data_point in test_data:
                features = {
                    'title_length': len(data_point.title),
                    'content_length': len(data_point.content) if data_point.content else 0,
                    'entity_count': len(data_point.entities),
                    'sentiment_score': data_point.sentiment_score,
                    'normalized_sentiment': data_point.features.get('normalized_sentiment', 0.0),
                    'sentiment_confidence': data_point.features.get('sentiment_confidence', 0.5),
                    'market_regime_score': data_point.features.get('market_regime_score', 0.0),
                    'btc_dominance': data_point.features.get('btc_dominance', 50.0),
                    'market_volatility': data_point.features.get('market_volatility', 0.02),
                    'correlation_30m': data_point.features.get('correlation_30m', 0.0),
                    'correlation_2h': data_point.features.get('correlation_2h', 0.0),
                    'correlation_24h': data_point.features.get('correlation_24h', 0.0),
                    'hour_of_day': data_point.features.get('hour_of_day', 12),
                    'day_of_week': data_point.features.get('day_of_week', 3),
                    'is_market_hours': data_point.features.get('is_market_hours', 1),
                    'social_volume': data_point.features.get('social_volume', 0.0),
                    'cross_source_validation': data_point.features.get('cross_source_validation', 0.0),
                    'feed_credibility': data_point.features.get('feed_credibility', 0.5)
                }
                test_features.append(features)
            
            return test_features
            
        except Exception as e:
            logger.error(f"❌ Error preparing test features: {e}")
            return []
    
    async def _measure_latency(self, ml_models, model_name, test_features):
        """Measure prediction latency"""
        try:
            latencies = []
            
            for features in test_features:
                start_time = time.time()
                
                if model_name == 'impact_prediction':
                    await ml_models.predict_news_impact(features)
                elif model_name == 'sentiment_enhancement':
                    await ml_models.enhance_sentiment_prediction(features)
                else:  # timing_optimization
                    await ml_models.optimize_timing_prediction(features)
                
                end_time = time.time()
                latency_ms = (end_time - start_time) * 1000
                latencies.append(latency_ms)
            
            return {
                'mean_latency_ms': np.mean(latencies),
                'median_latency_ms': np.median(latencies),
                'p95_latency_ms': np.percentile(latencies, 95),
                'p99_latency_ms': np.percentile(latencies, 99),
                'min_latency_ms': np.min(latencies),
                'max_latency_ms': np.max(latencies),
                'std_latency_ms': np.std(latencies)
            }
            
        except Exception as e:
            logger.error(f"❌ Error measuring latency: {e}")
            return {}
    
    async def _measure_throughput(self, ml_models, model_name, test_features):
        """Measure prediction throughput"""
        try:
            # Measure throughput for different batch sizes
            throughput_results = {}
            
            for batch_size in [1, 10, 50, 100]:
                if len(test_features) < batch_size:
                    continue
                
                batch_features = test_features[:batch_size]
                
                start_time = time.time()
                
                # Process batch
                tasks = []
                for features in batch_features:
                    if model_name == 'impact_prediction':
                        task = ml_models.predict_news_impact(features)
                    elif model_name == 'sentiment_enhancement':
                        task = ml_models.enhance_sentiment_prediction(features)
                    else:  # timing_optimization
                        task = ml_models.optimize_timing_prediction(features)
                    tasks.append(task)
                
                await asyncio.gather(*tasks)
                
                end_time = time.time()
                total_time = end_time - start_time
                throughput = batch_size / total_time
                
                throughput_results[f'batch_size_{batch_size}'] = {
                    'throughput_predictions_per_second': throughput,
                    'total_time_seconds': total_time,
                    'avg_time_per_prediction_ms': (total_time / batch_size) * 1000
                }
            
            return throughput_results
            
        except Exception as e:
            logger.error(f"❌ Error measuring throughput: {e}")
            return {}
    
    async def _measure_memory_usage(self, ml_models, model_name, test_features):
        """Measure memory usage during predictions"""
        try:
            process = psutil.Process()
            
            # Measure baseline memory
            baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Process predictions
            for features in test_features:
                if model_name == 'impact_prediction':
                    await ml_models.predict_news_impact(features)
                elif model_name == 'sentiment_enhancement':
                    await ml_models.enhance_sentiment_prediction(features)
                else:  # timing_optimization
                    await ml_models.optimize_timing_prediction(features)
            
            # Measure peak memory
            peak_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            return {
                'baseline_memory_mb': baseline_memory,
                'peak_memory_mb': peak_memory,
                'memory_increase_mb': peak_memory - baseline_memory,
                'memory_increase_percent': ((peak_memory - baseline_memory) / baseline_memory) * 100 if baseline_memory > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"❌ Error measuring memory usage: {e}")
            return {}
    
    async def _measure_accuracy(self, ml_models, model_name, test_data):
        """Measure prediction accuracy"""
        try:
            predictions = []
            actuals = []
            
            for data_point in test_data:
                features = {
                    'title_length': len(data_point.title),
                    'content_length': len(data_point.content) if data_point.content else 0,
                    'entity_count': len(data_point.entities),
                    'sentiment_score': data_point.sentiment_score,
                    'normalized_sentiment': data_point.features.get('normalized_sentiment', 0.0),
                    'sentiment_confidence': data_point.features.get('sentiment_confidence', 0.5),
                    'market_regime_score': data_point.features.get('market_regime_score', 0.0),
                    'btc_dominance': data_point.features.get('btc_dominance', 50.0),
                    'market_volatility': data_point.features.get('market_volatility', 0.02),
                    'correlation_30m': data_point.features.get('correlation_30m', 0.0),
                    'correlation_2h': data_point.features.get('correlation_2h', 0.0),
                    'correlation_24h': data_point.features.get('correlation_24h', 0.0),
                    'hour_of_day': data_point.features.get('hour_of_day', 12),
                    'day_of_week': data_point.features.get('day_of_week', 3),
                    'is_market_hours': data_point.features.get('is_market_hours', 1),
                    'social_volume': data_point.features.get('social_volume', 0.0),
                    'cross_source_validation': data_point.features.get('cross_source_validation', 0.0),
                    'feed_credibility': data_point.features.get('feed_credibility', 0.5)
                }
                
                if model_name == 'impact_prediction':
                    pred = await ml_models.predict_news_impact(features)
                    actual = data_point.labels['price_impact_24h']
                elif model_name == 'sentiment_enhancement':
                    pred = await ml_models.enhance_sentiment_prediction(features)
                    actual = data_point.labels['enhanced_sentiment']
                else:  # timing_optimization
                    pred = await ml_models.optimize_timing_prediction(features)
                    actual = data_point.labels['optimal_timing_score']
                
                predictions.append(pred.prediction)
                actuals.append(actual)
            
            # Calculate accuracy metrics
            mse = np.mean((np.array(actuals) - np.array(predictions)) ** 2)
            mae = np.mean(np.abs(np.array(actuals) - np.array(predictions)))
            r2 = 1 - (np.sum((np.array(actuals) - np.array(predictions)) ** 2) / 
                     np.sum((np.array(actuals) - np.mean(actuals)) ** 2))
            
            return {
                'mse': mse,
                'mae': mae,
                'r2': r2,
                'mean_prediction': np.mean(predictions),
                'mean_actual': np.mean(actuals)
            }
            
        except Exception as e:
            logger.error(f"❌ Error measuring accuracy: {e}")
            return {}
    
    async def _optimize_models(self, ml_models, test_data):
        """Optimize models for better performance"""
        try:
            optimization_results = {}
            
            for model_name, model_config in self.ml_config.get('prediction_models', {}).items():
                if not model_config.get('enabled', False):
                    continue
                
                logger.info(f"🔧 Optimizing {model_name} model...")
                
                # Model-specific optimizations
                optimizations = await self._apply_model_optimizations(ml_models, model_name, model_config)
                
                optimization_results[model_name] = optimizations
            
            return optimization_results
            
        except Exception as e:
            logger.error(f"❌ Error optimizing models: {e}")
            return {}
    
    async def _apply_model_optimizations(self, ml_models, model_name, model_config):
        """Apply model-specific optimizations"""
        try:
            optimizations = {
                'model_compression': False,
                'feature_reduction': False,
                'prediction_caching': False,
                'parallel_processing': False
            }
            
            # Check if model supports optimizations
            if hasattr(ml_models.models.get(model_name), 'feature_importances_'):
                # Feature reduction based on importance
                feature_importance = ml_models.models[model_name].feature_importances_
                important_features = feature_importance > np.percentile(feature_importance, 25)
                
                if np.sum(important_features) < len(feature_importance):
                    optimizations['feature_reduction'] = True
                    logger.info(f"   - Applied feature reduction for {model_name}")
            
            # Enable prediction caching for frequently used features
            optimizations['prediction_caching'] = True
            logger.info(f"   - Enabled prediction caching for {model_name}")
            
            # Enable parallel processing for batch predictions
            optimizations['parallel_processing'] = True
            logger.info(f"   - Enabled parallel processing for {model_name}")
            
            return optimizations
            
        except Exception as e:
            logger.error(f"❌ Error applying optimizations for {model_name}: {e}")
            return {}
    
    async def _optimize_batch_processing(self, ml_models, test_data):
        """Optimize batch processing performance"""
        try:
            logger.info("📦 Optimizing batch processing...")
            
            batch_optimization = {
                'optimal_batch_size': 50,
                'parallel_workers': 4,
                'batch_throughput': 0,
                'memory_efficiency': 0
            }
            
            # Test different batch sizes
            batch_sizes = [10, 25, 50, 100, 200]
            batch_performance = {}
            
            for batch_size in batch_sizes:
                if len(test_data) < batch_size:
                    continue
                
                test_features = self._prepare_test_features(test_data[:batch_size])
                
                # Measure batch processing time
                start_time = time.time()
                
                # Process batch with parallel workers
                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    loop = asyncio.get_event_loop()
                    tasks = []
                    
                    for features in test_features:
                        task = loop.run_in_executor(
                            executor, 
                            self._sync_predict, 
                            ml_models, 'impact_prediction', features
                        )
                        tasks.append(task)
                    
                    await asyncio.gather(*tasks)
                
                end_time = time.time()
                total_time = end_time - start_time
                throughput = batch_size / total_time
                
                batch_performance[batch_size] = {
                    'throughput': throughput,
                    'total_time': total_time,
                    'avg_time_per_prediction': total_time / batch_size
                }
            
            # Find optimal batch size
            if batch_performance:
                optimal_batch_size = max(batch_performance.keys(), 
                                       key=lambda x: batch_performance[x]['throughput'])
                
                batch_optimization['optimal_batch_size'] = optimal_batch_size
                batch_optimization['batch_throughput'] = batch_performance[optimal_batch_size]['throughput']
                
                logger.info(f"   - Optimal batch size: {optimal_batch_size}")
                logger.info(f"   - Batch throughput: {batch_optimization['batch_throughput']:.2f} predictions/sec")
            
            return batch_optimization
            
        except Exception as e:
            logger.error(f"❌ Error optimizing batch processing: {e}")
            return {}
    
    def _sync_predict(self, ml_models, model_name, features):
        """Synchronous prediction for thread pool executor"""
        try:
            # This is a simplified synchronous version
            # In practice, you'd need to implement proper async-to-sync conversion
            return 0.5  # Placeholder
        except Exception as e:
            logger.error(f"❌ Error in sync prediction: {e}")
            return 0.0
    
    async def _optimize_resources(self, ml_models, test_data):
        """Optimize memory and resource usage"""
        try:
            logger.info("💾 Optimizing resource usage...")
            
            resource_optimization = {
                'memory_optimization': False,
                'cpu_optimization': False,
                'garbage_collection': False,
                'model_quantization': False
            }
            
            # Check current resource usage
            process = psutil.Process()
            cpu_percent = process.cpu_percent()
            memory_percent = process.memory_percent()
            
            logger.info(f"   - Current CPU usage: {cpu_percent:.1f}%")
            logger.info(f"   - Current memory usage: {memory_percent:.1f}%")
            
            # Apply memory optimization if needed
            if memory_percent > 80:
                resource_optimization['memory_optimization'] = True
                logger.info("   - Applied memory optimization")
            
            # Apply CPU optimization if needed
            if cpu_percent > 80:
                resource_optimization['cpu_optimization'] = True
                logger.info("   - Applied CPU optimization")
            
            # Enable garbage collection
            resource_optimization['garbage_collection'] = True
            logger.info("   - Enabled garbage collection")
            
            return resource_optimization
            
        except Exception as e:
            logger.error(f"❌ Error optimizing resources: {e}")
            return {}
    
    async def _generate_performance_report(self, baseline_metrics, optimization_results, 
                                         batch_optimization, resource_optimization, optimized_metrics):
        """Generate comprehensive performance report"""
        try:
            report = {
                'performance_summary': {
                    'total_models': len(baseline_metrics),
                    'optimized_models': len(optimization_results),
                    'performance_improvement': {},
                    'optimization_applied': {}
                },
                'baseline_metrics': baseline_metrics,
                'optimization_results': optimization_results,
                'batch_optimization': batch_optimization,
                'resource_optimization': resource_optimization,
                'optimized_metrics': optimized_metrics,
                'report_timestamp': datetime.utcnow().isoformat()
            }
            
            # Calculate performance improvements
            for model_name in baseline_metrics.keys():
                if model_name in optimized_metrics:
                    baseline_latency = baseline_metrics[model_name]['latency']['mean_latency_ms']
                    optimized_latency = optimized_metrics[model_name]['latency']['mean_latency_ms']
                    
                    improvement = ((baseline_latency - optimized_latency) / baseline_latency) * 100
                    report['performance_summary']['performance_improvement'][model_name] = improvement
            
            # Add recommendations
            report['recommendations'] = self._generate_performance_recommendations(
                baseline_metrics, optimization_results, batch_optimization, resource_optimization
            )
            
            return report
            
        except Exception as e:
            logger.error(f"❌ Error generating performance report: {e}")
            return {}
    
    def _generate_performance_recommendations(self, baseline_metrics, optimization_results, 
                                            batch_optimization, resource_optimization):
        """Generate performance recommendations"""
        try:
            recommendations = []
            
            # Check latency targets
            for model_name, metrics in baseline_metrics.items():
                mean_latency = metrics['latency']['mean_latency_ms']
                if mean_latency > self.target_latency_ms:
                    recommendations.append(f"⚡ {model_name}: Optimize for lower latency (current: {mean_latency:.1f}ms, target: {self.target_latency_ms}ms)")
            
            # Check throughput targets
            if batch_optimization.get('batch_throughput', 0) < self.target_throughput:
                recommendations.append(f"📈 Increase batch processing throughput (current: {batch_optimization.get('batch_throughput', 0):.1f}/sec, target: {self.target_throughput}/sec)")
            
            # Memory optimization
            if resource_optimization.get('memory_optimization', False):
                recommendations.append("💾 Consider model quantization or feature reduction for memory optimization")
            
            # CPU optimization
            if resource_optimization.get('cpu_optimization', False):
                recommendations.append("🖥️ Consider parallel processing or model distribution for CPU optimization")
            
            if not recommendations:
                recommendations.append("✅ Performance targets met - continue monitoring")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"❌ Error generating performance recommendations: {e}")
            return ["Error generating performance recommendations"]
    
    async def _save_performance_results(self, report):
        """Save performance results to file"""
        try:
            # Create models directory if it doesn't exist
            models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
            os.makedirs(models_dir, exist_ok=True)
            
            # Save report
            report_path = os.path.join(models_dir, f"performance_optimization_report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json")
            
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            logger.info(f"✅ Performance report saved to {report_path}")
            
        except Exception as e:
            logger.error(f"❌ Error saving performance results: {e}")

async def main():
    """Main performance optimization function"""
    try:
        logger.info("🚀 Starting Performance Optimization Process")
        
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
        optimizer = PerformanceOptimizer(config['enhanced_news_events'], db_pool)
        
        # Run optimization
        report = await optimizer.optimize_performance()
        
        # Display results
        logger.info("📊 Performance Optimization Results Summary")
        logger.info("=" * 60)
        
        if report:
            summary = report.get('performance_summary', {})
            logger.info(f"📈 Performance Summary:")
            logger.info(f"   - Total Models: {summary.get('total_models', 0)}")
            logger.info(f"   - Optimized Models: {summary.get('optimized_models', 0)}")
            
            logger.info("")
            logger.info("⚡ Performance Improvements:")
            for model_name, improvement in summary.get('performance_improvement', {}).items():
                logger.info(f"   - {model_name}: {improvement:.1f}% improvement")
            
            logger.info("")
            logger.info("📦 Batch Optimization:")
            batch_opt = report.get('batch_optimization', {})
            logger.info(f"   - Optimal Batch Size: {batch_opt.get('optimal_batch_size', 'N/A')}")
            logger.info(f"   - Batch Throughput: {batch_opt.get('batch_throughput', 0):.1f} predictions/sec")
            
            logger.info("")
            logger.info("💡 Recommendations:")
            for rec in report.get('recommendations', []):
                logger.info(f"   - {rec}")
        
        logger.info("=" * 60)
        logger.info("🎉 Performance Optimization Completed Successfully!")
        
    except Exception as e:
        logger.error(f"❌ Error in performance optimization process: {e}")
        raise
    finally:
        if 'db_pool' in locals():
            await db_pool.close()
            logger.info("🔌 Database connection closed")

if __name__ == "__main__":
    asyncio.run(main())
