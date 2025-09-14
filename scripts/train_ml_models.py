#!/usr/bin/env python3
"""
ML Model Training Script
Trains all ML models using collected training data
"""

import asyncio
import logging
import sys
import os
from datetime import datetime

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

import asyncpg
from services.ml_models import NewsMLModels
from services.training_data_collector import TrainingDataCollector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def main():
    """Main training function"""
    try:
        logger.info("🚀 Starting ML Model Training Process")
        
        # Load configuration
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'enhanced_news_config.json')
        
        if not os.path.exists(config_path):
            logger.error(f"❌ Configuration file not found: {config_path}")
            return
        
        import json
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
        
        # Initialize training data collector
        logger.info("📊 Initializing training data collector...")
        training_collector = TrainingDataCollector(db_pool, config['enhanced_news_events'])
        
        # Initialize ML models
        logger.info("🤖 Initializing ML models...")
        ml_models = NewsMLModels(config['enhanced_news_events'], db_pool)
        await ml_models.initialize_models()
        
        # Step 1: Collect training data
        logger.info("🔄 Step 1: Collecting training data...")
        data_result = await training_collector.collect_training_data()
        
        if data_result['total_samples'] == 0:
            logger.error("❌ No training data available")
            return
        
        logger.info(f"✅ Collected {data_result['total_samples']} training samples")
        
        # Step 2: Get training data summary
        logger.info("📋 Step 2: Generating training data summary...")
        summary = await training_collector.get_training_data_summary()
        
        if 'error' not in summary:
            logger.info(f"📊 Training Data Summary:")
            logger.info(f"   - Total samples: {summary['total_samples']}")
            logger.info(f"   - Training samples: {summary['training_samples']}")
            logger.info(f"   - Validation samples: {summary['validation_samples']}")
            logger.info(f"   - Features: {summary['feature_count']}")
            logger.info(f"   - Labels: {summary['label_count']}")
        else:
            logger.warning(f"⚠️ Could not generate summary: {summary['error']}")
        
        # Step 3: Train models
        logger.info("🤖 Step 3: Training ML models...")
        training_results = await ml_models.train_models_with_training_data(training_collector)
        
        if not training_results:
            logger.error("❌ Model training failed")
            return
        
        # Step 4: Display training results
        logger.info("📈 Step 4: Training Results Summary")
        logger.info("=" * 60)
        
        for model_name, result in training_results.items():
            if 'error' in result:
                logger.error(f"❌ {model_name}: {result['error']}")
            else:
                logger.info(f"✅ {model_name} Model:")
                logger.info(f"   - Model Type: {result['model_type']}")
                logger.info(f"   - Training Samples: {result['training_samples']}")
                logger.info(f"   - Validation Samples: {result['validation_samples']}")
                
                # Training performance
                train_perf = result['training_performance']
                logger.info(f"   - Training Performance:")
                logger.info(f"     * F1 Score: {train_perf['f1_score']:.3f}")
                logger.info(f"     * Accuracy: {train_perf['accuracy']:.3f}")
                logger.info(f"     * MSE: {train_perf['mse']:.4f}")
                
                # Validation performance
                val_perf = result['validation_performance']
                logger.info(f"   - Validation Performance:")
                logger.info(f"     * F1 Score: {val_perf['f1_score']:.3f}")
                logger.info(f"     * Accuracy: {val_perf['accuracy']:.3f}")
                logger.info(f"     * MSE: {val_perf['mse']:.4f}")
                
                # Feature importance (top 5)
                feature_importance = result['feature_importance']
                if feature_importance:
                    top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
                    logger.info(f"   - Top 5 Features:")
                    for feature, importance in top_features:
                        logger.info(f"     * {feature}: {importance:.3f}")
                
                logger.info("")
        
        # Step 5: Save training data
        logger.info("💾 Step 5: Saving training data...")
        training_data_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'training_data.json')
        await training_collector.save_training_data(training_data_path)
        
        # Step 6: Load trained models and test
        logger.info("🧪 Step 6: Testing trained models...")
        await ml_models.load_trained_models()
        
        # Test predictions
        test_features = {
            'title_length': 50,
            'content_length': 500,
            'entity_count': 3,
            'sentiment_score': 0.2,
            'normalized_sentiment': 0.2,
            'sentiment_confidence': 0.7,
            'market_regime_score': 0.1,
            'btc_dominance': 45.0,
            'market_volatility': 0.025,
            'correlation_30m': 0.1,
            'correlation_2h': 0.05,
            'correlation_24h': 0.02,
            'hour_of_day': 14,
            'day_of_week': 2,
            'is_market_hours': 1,
            'social_volume': 1000,
            'cross_source_validation': 0.3,
            'feed_credibility': 0.8
        }
        
        logger.info("🧪 Testing model predictions...")
        
        # Test impact prediction
        impact_prediction = await ml_models.predict_news_impact(test_features)
        logger.info(f"   - Impact Prediction: {impact_prediction.prediction:.3f} (confidence: {impact_prediction.confidence:.3f})")
        
        # Test sentiment enhancement
        sentiment_prediction = await ml_models.enhance_sentiment_prediction(test_features)
        logger.info(f"   - Sentiment Enhancement: {sentiment_prediction.prediction:.3f} (confidence: {sentiment_prediction.confidence:.3f})")
        
        # Test timing optimization
        timing_prediction = await ml_models.optimize_timing_prediction(test_features)
        logger.info(f"   - Timing Optimization: {timing_prediction.prediction:.3f} (confidence: {timing_prediction.confidence:.3f})")
        
        # Step 7: Get model status
        logger.info("📊 Step 7: Model Status")
        model_status = await ml_models.get_model_status()
        
        for model_name, status in model_status.items():
            if status['status'] == 'trained':
                logger.info(f"✅ {model_name}: {status['model_type']} - Trained")
            else:
                logger.warning(f"⚠️ {model_name}: {status['status']}")
        
        logger.info("=" * 60)
        logger.info("🎉 ML Model Training Process Completed Successfully!")
        
        # Summary
        successful_models = sum(1 for result in training_results.values() if 'error' not in result)
        total_models = len(training_results)
        
        logger.info(f"📊 Final Summary:")
        logger.info(f"   - Models Trained: {successful_models}/{total_models}")
        logger.info(f"   - Training Data: {data_result['total_samples']} samples")
        logger.info(f"   - Models Saved: {successful_models}")
        logger.info(f"   - Training Data Saved: {training_data_path}")
        
    except Exception as e:
        logger.error(f"❌ Error in training process: {e}")
        raise
    finally:
        if 'db_pool' in locals():
            await db_pool.close()
            logger.info("🔌 Database connection closed")

if __name__ == "__main__":
    asyncio.run(main())
