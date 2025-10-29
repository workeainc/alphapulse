#!/usr/bin/env python3
"""
Comprehensive Test for ML Auto-Retraining System
Tests database setup, training, evaluation, and inference
"""

import os
import sys
import json
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.ai.ml_auto_retraining.train_model import MLModelTrainer
from src.ai.ml_auto_retraining.evaluate_and_promote import ModelEvaluator
from src.ai.ml_auto_retraining.ml_inference_engine import MLInferenceEngine, EnhancedPatternDetector

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Database configuration
DB_CONFIG = {
    'host': 'localhost',
    'database': 'alphapulse',
    'user': 'alpha_emon',
    'password': 'Emon_@17711',
    'port': 5432
}

def create_sample_data():
    """Create sample OHLCV and pattern performance data for testing"""
    logger.info("📊 Creating sample data for testing...")
    
    # Create sample OHLCV data
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='1h')
    n_samples = len(dates)
    
    # Generate realistic price data with trends and volatility
    np.random.seed(42)
    base_price = 50000
    price_changes = np.random.normal(0, 0.01, n_samples)  # 1% daily volatility
    prices = [base_price]
    
    for change in price_changes[1:]:
        new_price = prices[-1] * (1 + change)
        prices.append(new_price)
    
    # Create OHLCV data
    ohlcv_data = []
    for i, (date, price) in enumerate(zip(dates, prices)):
        # Add some realistic OHLC variation
        volatility = abs(price_changes[i]) if i < len(price_changes) else 0.005
        high = price * (1 + volatility * np.random.uniform(0.5, 1.5))
        low = price * (1 - volatility * np.random.uniform(0.5, 1.5))
        open_price = prices[i-1] if i > 0 else price
        close_price = price
        volume = np.random.uniform(1000, 5000)
        
        ohlcv_data.append({
            'time': date,
            'open': open_price,
            'high': high,
            'low': low,
            'close': close_price,
            'volume': volume
        })
    
    ohlcv_df = pd.DataFrame(ohlcv_data)
    
    # Create sample pattern performance data
    pattern_performance = []
    for i in range(0, len(ohlcv_df), 24):  # One pattern per day
        if i + 24 < len(ohlcv_df):
            pattern_performance.append({
                'timestamp': ohlcv_df.iloc[i]['time'],
                'tracking_id': f"pattern_{i}",
                'pattern_id': f"pattern_{i}",
                'symbol': 'BTCUSDT',
                'pattern_name': np.random.choice(['hammer', 'engulfing', 'doji', 'morning_star']),
                'timeframe': '1h',
                'pattern_confidence': np.random.uniform(0.6, 0.9),
                'predicted_outcome': np.random.choice(['success', 'failure']),
                'actual_outcome': np.random.choice(['success', 'failure']),
                'market_regime': np.random.choice(['trending', 'sideways', 'volatile', 'consolidation']),
                'volume_ratio': np.random.uniform(0.8, 1.5),
                'volatility_level': np.random.uniform(0.005, 0.02),
                'spread_impact': np.random.uniform(0.001, 0.005),
                'noise_filter_score': np.random.uniform(0.7, 1.0),
                'performance_score': np.random.uniform(0.5, 0.9),
                'outcome_timestamp': ohlcv_df.iloc[i + 24]['time'],
                'outcome_price': ohlcv_df.iloc[i + 24]['close'],
                'profit_loss': np.random.uniform(-100, 200)
            })
    
    perf_df = pd.DataFrame(pattern_performance)
    
    logger.info(f"✅ Created {len(ohlcv_df)} OHLCV records and {len(perf_df)} pattern performance records")
    return ohlcv_df, perf_df

def insert_sample_data_to_db(ohlcv_df, perf_df):
    """Insert sample data into database"""
    logger.info("💾 Inserting sample data into database...")
    
    try:
        import psycopg2
        
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        
        # Insert OHLCV data
        for _, row in ohlcv_df.iterrows():
            cursor.execute("""
                INSERT INTO ohlcv (time, open, high, low, close, volume, symbol)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (time, symbol) DO NOTHING
            """, (
                row['time'], row['open'], row['high'], row['low'], 
                row['close'], row['volume'], 'BTCUSDT'
            ))
        
        # Insert pattern performance data
        for _, row in perf_df.iterrows():
            cursor.execute("""
                INSERT INTO pattern_performance_tracking (
                    timestamp, tracking_id, pattern_id, symbol, pattern_name, timeframe,
                    pattern_confidence, predicted_outcome, actual_outcome, market_regime,
                    volume_ratio, volatility_level, spread_impact, noise_filter_score,
                    performance_score, outcome_timestamp, outcome_price, profit_loss
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                ) ON CONFLICT (timestamp, tracking_id) DO NOTHING
            """, (
                row['timestamp'], row['tracking_id'], row['pattern_id'], row['symbol'],
                row['pattern_name'], row['timeframe'], row['pattern_confidence'],
                row['predicted_outcome'], row['actual_outcome'], row['market_regime'],
                row['volume_ratio'], row['volatility_level'], row['spread_impact'],
                row['noise_filter_score'], row['performance_score'],
                row['outcome_timestamp'], row['outcome_price'], row['profit_loss']
            ))
        
        conn.commit()
        logger.info("✅ Sample data inserted successfully")
        
    except Exception as e:
        logger.error(f"❌ Failed to insert sample data: {e}")
        if conn:
            conn.rollback()
        raise
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

async def test_ml_training():
    """Test ML model training"""
    logger.info("🚀 Testing ML model training...")
    
    try:
        trainer = MLModelTrainer(DB_CONFIG)
        
        # Initialize components
        await trainer.initialize_components()
        
        # Test training for trending regime
        logger.info("📊 Testing training for trending regime...")
        
        # Create a simple training test - use data from 2024
        end_date = datetime(2024, 12, 31)
        start_date = datetime(2024, 11, 1)
        
        # Load data
        ohlcv_df = trainer.load_ohlcv_data('BTCUSDT', start_date, end_date)
        performance_df = trainer.load_pattern_performance_data('BTCUSDT', start_date, end_date)
        
        if ohlcv_df.empty:
            logger.warning("⚠️ No OHLCV data found, skipping training test")
            return None
        
        # Add symbol column
        ohlcv_df['symbol'] = 'BTCUSDT'
        
        # Create technical features
        feature_df = trainer.create_technical_features(ohlcv_df)
        
        # Apply noise filtering (simplified)
        filtered_df = feature_df.copy()  # Skip actual filtering for test
        
        # Classify market regime (simplified)
        filtered_df['market_regime'] = 'trending'  # Force trending for test
        
        # Create labels
        labeled_df = trainer.create_labels(filtered_df, performance_df, horizon=10)
        
        if len(labeled_df) < 100:
            logger.warning("⚠️ Insufficient data for training, skipping")
            return None
        
        # Prepare features
        X, y = trainer.prepare_features(labeled_df, 'trending')
        
        if len(X) < 50:
            logger.warning("⚠️ Insufficient features for training, skipping")
            return None
        
        # Split data
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
        
        # Train model
        params = {
            'n_estimators': 100,  # Reduced for testing
            'max_depth': 3,
            'learning_rate': 0.1,
            'subsample': 0.9,
            'colsample_bytree': 0.9,
            'reg_lambda': 1.0
        }
        
        model = trainer.train_model(X_train, y_train, params)
        
        # Evaluate model
        metrics = trainer.evaluate_model(model, X_val, y_val)
        
        # Save model
        model_path = trainer.save_model(model, 'alphaplus_pattern_classifier', 'trending', 'BTCUSDT', metrics, params)
        
        logger.info(f"✅ Training test completed: {model_path}")
        logger.info(f"📊 Model metrics: {metrics}")
        
        await trainer.cleanup()
        return model_path
        
    except Exception as e:
        logger.error(f"❌ Training test failed: {e}")
        return None

def test_model_evaluation(model_path):
    """Test model evaluation and promotion"""
    logger.info("🔍 Testing model evaluation and promotion...")
    
    if not model_path or not os.path.exists(model_path):
        logger.warning("⚠️ No model path provided, skipping evaluation test")
        return None
    
    try:
        evaluator = ModelEvaluator(DB_CONFIG)
        
        # Test evaluation
        result = evaluator.load_candidate_model(model_path)
        if not result:
            logger.warning("⚠️ Failed to load candidate model")
            return None
        
        # Load validation data - use data from 2024
        end_date = datetime(2024, 12, 31)
        start_date = datetime(2024, 12, 24)
        
        # Load OHLCV data
        ohlcv_query = """
            SELECT 
                time as timestamp,
                open, high, low, close, volume
            FROM ohlcv
            WHERE symbol = %s AND time >= %s AND time < %s
            ORDER BY time ASC
        """
        
        import psycopg2
        conn = psycopg2.connect(**DB_CONFIG)
        ohlcv_df = pd.read_sql(ohlcv_query, conn, params=('BTCUSDT', start_date, end_date))
        
        # Load pattern performance data
        perf_query = """
            SELECT 
                timestamp, pattern_id, actual_outcome, profit_loss, performance_score
            FROM pattern_performance_tracking
            WHERE symbol = %s AND timestamp >= %s AND timestamp < %s
            ORDER BY timestamp ASC
        """
        
        perf_df = pd.read_sql(perf_query, conn, params=('BTCUSDT', start_date, end_date))
        conn.close()
        
        if ohlcv_df.empty:
            logger.warning("⚠️ No validation data found, skipping evaluation test")
            return None
        
        # Create validation features
        feature_df = evaluator.create_validation_features(ohlcv_df)
        labeled_df = evaluator.create_validation_labels(feature_df, perf_df)
        
        # Prepare validation features
        X_val = evaluator.prepare_validation_features(labeled_df)
        y_val = labeled_df['target'].values
        
        if len(X_val) < 10:
            logger.warning("⚠️ Insufficient validation data, skipping evaluation test")
            return None
        
        # Evaluate model
        candidate_metrics = evaluator.evaluate_model_performance(result, X_val, y_val)
        
        # Calculate drift (simplified)
        drift_scores = {}
        drift_max = 0.0
        
        # Decide promotion
        promote, notes = evaluator.decide_promotion(candidate_metrics, None, drift_max)
        
        logger.info(f"✅ Evaluation test completed")
        logger.info(f"📊 Candidate metrics: {candidate_metrics}")
        logger.info(f"🎯 Promotion decision: {promote}")
        logger.info(f"📝 Notes: {notes}")
        
        return {
            'candidate_metrics': candidate_metrics,
            'promote': promote,
            'notes': notes
        }
        
    except Exception as e:
        logger.error(f"❌ Evaluation test failed: {e}")
        return None

def test_ml_inference():
    """Test ML inference engine"""
    logger.info("🔮 Testing ML inference engine...")
    
    try:
        engine = MLInferenceEngine(DB_CONFIG)
        
        # Create sample market data with trending characteristics
        dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='1h')
        
        # Create trending market data (upward trend)
        base_prices = np.linspace(45000, 55000, len(dates))  # Trending upward
        noise = np.random.normal(0, 200, len(dates))  # Small noise
        prices = base_prices + noise
        
        sample_data = pd.DataFrame({
            'open': prices,
            'high': prices * 1.005,  # Small high above close
            'low': prices * 0.995,   # Small low below close
            'close': prices,
            'volume': np.random.uniform(2000, 8000, len(dates))  # Higher volume for trending
        }, index=dates)
        
        # Sample pattern data
        pattern_data = {
            'pattern_id': 'test_pattern_001',
            'symbol': 'BTCUSDT',
            'pattern_name': 'hammer',
            'confidence': 0.8
        }
        
        # Test prediction (will fail if no models available, which is expected)
        prediction = engine.predict_pattern_success(pattern_data, sample_data)
        
        if prediction:
            logger.info(f"✅ ML prediction successful: {prediction.prediction_class}")
            logger.info(f"📊 Confidence: {prediction.prediction_confidence:.3f}")
            logger.info(f"🎯 Regime: {prediction.regime}")
        else:
            logger.info("ℹ️ No ML prediction available (expected for first run)")
        
        # Test performance summary
        summary = engine.get_model_performance_summary('alphaplus_pattern_classifier', 'trending', 'BTCUSDT', days=30)
        logger.info(f"📈 Performance summary: {summary}")
        
        engine.cleanup()
        return prediction
        
    except Exception as e:
        logger.error(f"❌ Inference test failed: {e}")
        return None

async def test_enhanced_pattern_detection():
    """Test enhanced pattern detection with ML"""
    logger.info("🔍 Testing enhanced pattern detection...")
    
    try:
        detector = EnhancedPatternDetector(DB_CONFIG)
        
        # Create sample pattern signals
        class MockSignal:
            def __init__(self, pattern_name, confidence):
                self.pattern_name = pattern_name
                self.confidence = confidence
        
        mock_signals = [
            MockSignal('hammer', 0.8),
            MockSignal('engulfing', 0.7),
            MockSignal('doji', 0.6)
        ]
        
        # Create sample market data with trending characteristics
        dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='1h')
        
        # Create trending market data (upward trend)
        base_prices = np.linspace(45000, 55000, len(dates))  # Trending upward
        noise = np.random.normal(0, 200, len(dates))  # Small noise
        prices = base_prices + noise
        
        sample_data = pd.DataFrame({
            'open': prices,
            'high': prices * 1.005,  # Small high above close
            'low': prices * 0.995,   # Small low below close
            'close': prices,
            'volume': np.random.uniform(2000, 8000, len(dates))  # Higher volume for trending
        }, index=dates)
        
        # Test enhanced detection
        enhanced_signals = await detector.detect_patterns_with_ml(sample_data, 'BTCUSDT', mock_signals)
        
        logger.info(f"✅ Enhanced pattern detection test completed")
        logger.info(f"📊 Enhanced {len(enhanced_signals)} signals")
        
        # Test performance summary
        summary = detector.get_ml_performance_summary('BTCUSDT', days=30)
        logger.info(f"📈 ML performance summary: {summary}")
        
        detector.cleanup()
        return enhanced_signals
        
    except Exception as e:
        logger.error(f"❌ Enhanced pattern detection test failed: {e}")
        return None

def verify_database_tables():
    """Verify that all ML tables exist and are accessible"""
    logger.info("🔍 Verifying database tables...")
    
    try:
        import psycopg2
        
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        
        tables_to_check = [
            'ml_models',
            'ml_eval_history',
            'ml_training_jobs',
            'ml_performance_tracking'
        ]
        
        for table in tables_to_check:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            count = cursor.fetchone()[0]
            logger.info(f"✅ {table}: {count} records")
        
        # Check if we have any production models
        cursor.execute("""
            SELECT model_name, regime, symbol, version, status
            FROM ml_models
            WHERE status = 'production'
            ORDER BY created_at DESC
        """)
        
        production_models = cursor.fetchall()
        logger.info(f"📊 Production models: {len(production_models)}")
        
        for model in production_models:
            logger.info(f"   - {model[0]} {model[1]} {model[2]} v{model[3]} ({model[4]})")
        
        cursor.close()
        conn.close()
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Database verification failed: {e}")
        return False

async def run_comprehensive_test():
    """Run comprehensive test of the ML auto-retraining system"""
    logger.info("🚀 Starting comprehensive ML auto-retraining system test")
    logger.info("=" * 80)
    
    test_results = {
        'database_verification': False,
        'sample_data_creation': False,
        'ml_training': False,
        'model_evaluation': False,
        'ml_inference': False,
        'enhanced_pattern_detection': False
    }
    
    try:
        # Step 1: Verify database tables
        logger.info("📋 Step 1: Verifying database tables...")
        test_results['database_verification'] = verify_database_tables()
        
        # Step 2: Create sample data
        logger.info("📋 Step 2: Creating sample data...")
        try:
            ohlcv_df, perf_df = create_sample_data()
            insert_sample_data_to_db(ohlcv_df, perf_df)
            test_results['sample_data_creation'] = True
        except Exception as e:
            logger.error(f"❌ Sample data creation failed: {e}")
        
        # Step 3: Test ML training
        logger.info("📋 Step 3: Testing ML training...")
        model_path = await test_ml_training()
        test_results['ml_training'] = model_path is not None
        
        # Step 4: Test model evaluation
        logger.info("📋 Step 4: Testing model evaluation...")
        eval_result = test_model_evaluation(model_path)
        test_results['model_evaluation'] = eval_result is not None
        
        # Step 5: Test ML inference
        logger.info("📋 Step 5: Testing ML inference...")
        inference_result = test_ml_inference()
        test_results['ml_inference'] = inference_result is not None
        
        # Step 6: Test enhanced pattern detection
        logger.info("📋 Step 6: Testing enhanced pattern detection...")
        enhanced_result = await test_enhanced_pattern_detection()
        test_results['enhanced_pattern_detection'] = enhanced_result is not None
        
    except Exception as e:
        logger.error(f"❌ Comprehensive test failed: {e}")
    
    # Print test summary
    logger.info("=" * 80)
    logger.info("📊 ML Auto-Retraining System Test Summary")
    logger.info("=" * 80)
    
    for test_name, result in test_results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{test_name.replace('_', ' ').title()}: {status}")
    
    passed_tests = sum(test_results.values())
    total_tests = len(test_results)
    
    logger.info(f"📈 Overall Result: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        logger.info("🎉 All tests passed! ML auto-retraining system is ready for production.")
    else:
        logger.info("⚠️ Some tests failed. Please review the logs and fix issues.")
    
    return test_results

if __name__ == "__main__":
    asyncio.run(run_comprehensive_test())
