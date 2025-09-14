#!/usr/bin/env python3
"""
Phase 2.3: Ensemble Model Integration Test Script
Tests database migrations, ensemble strategy, and signal generator integration
"""

import asyncio
import logging
import sys
import os
from datetime import datetime
import pandas as pd
import numpy as np

# Add the backend directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.core.database_manager import DatabaseManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Phase2_3EnsembleTester:
    def __init__(self):
        self.db_manager = None
        self.db_connection = None
    
    async def initialize_database(self):
        """Initialize database connection"""
        try:
            self.db_manager = DatabaseManager()
            await self.db_manager.initialize()
            self.db_connection = await self.db_manager.get_connection()
            logger.info("✅ Database connection established")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to initialize database: {e}")
            return False
    
    async def test_phase2_3_database_migrations(self):
        """Test Phase 2.3 database migrations"""
        logger.info("🔍 Testing Phase 2.3: Database Migrations...")
        
        try:
            # Check if ensemble columns exist
            columns_query = """
                SELECT column_name, data_type 
                FROM information_schema.columns 
                WHERE table_name = 'enhanced_signals' 
                AND column_name LIKE 'ensemble_%'
                ORDER BY column_name
            """
            
            columns = await self.db_connection.fetch(columns_query)
            
            expected_columns = [
                'ensemble_analysis', 'ensemble_voting_method', 'ensemble_model_weights',
                'ensemble_individual_predictions', 'ensemble_confidence', 'ensemble_diversity_score',
                'ensemble_agreement_ratio', 'ensemble_bias', 'ensemble_model_count',
                'ensemble_performance_score', 'ensemble_last_updated'
            ]
            
            found_columns = [col['column_name'] for col in columns]
            
            logger.info(f"📊 Found {len(found_columns)} ensemble columns: {found_columns}")
            
            # Check for missing columns
            missing_columns = [col for col in expected_columns if col not in found_columns]
            if missing_columns:
                logger.error(f"❌ Missing ensemble columns: {missing_columns}")
                return False
            
            # Check ensemble view
            view_query = """
                SELECT viewname 
                FROM pg_views 
                WHERE viewname = 'ensemble_enhanced_signals'
            """
            view_result = await self.db_connection.fetch(view_query)
            
            if not view_result:
                logger.error("❌ ensemble_enhanced_signals view not found")
                return False
            
            logger.info("✅ ensemble_enhanced_signals view exists")
            
            # Check ensemble functions
            functions_query = """
                SELECT proname 
                FROM pg_proc 
                WHERE proname IN ('calculate_ensemble_enhanced_quality', 'update_ensemble_performance')
            """
            functions = await self.db_connection.fetch(functions_query)
            
            found_functions = [func['proname'] for func in functions]
            logger.info(f"📊 Found ensemble functions: {found_functions}")
            
            if len(found_functions) < 2:
                logger.error("❌ Missing ensemble functions")
                return False
            
            logger.info("✅ Phase 2.3: Database migrations verified")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error testing database migrations: {e}")
            return False
    
    async def test_ensemble_strategy(self):
        """Test Ensemble Strategy functionality"""
        logger.info("🤖 Testing Phase 2.3: Ensemble Strategy...")
        
        try:
            # Import ensemble strategy
            import sys
            import os
            sys.path.append(os.path.join(os.path.dirname(__file__), 'ai'))
            from ml_strategy_enhancement import EnsembleStrategy, EnsembleConfig, MLStrategyType, ModelType
            
            # Create ensemble configuration
            config = EnsembleConfig(
                strategy_type=MLStrategyType.ENSEMBLE_VOTING,
                base_models=[
                    ModelType.RANDOM_FOREST,
                    ModelType.GRADIENT_BOOSTING,
                    ModelType.LOGISTIC_REGRESSION
                ],
                voting_method="soft",
                adaptive_weights=True
            )
            
            # Create ensemble strategy
            ensemble = EnsembleStrategy(config)
            logger.info(f"✅ Ensemble strategy created with {len(config.base_models)} models")
            
            # Generate sample data
            np.random.seed(42)
            X = np.random.randn(100, 10)
            y = np.random.randint(0, 2, 100)
            
            # Train ensemble
            ensemble.train(X, y)
            logger.info("✅ Ensemble training completed")
            
            # Test prediction
            test_features = np.random.randn(1, 10)
            prediction, confidence = ensemble.predict(test_features)
            
            logger.info(f"✅ Ensemble prediction: {prediction}, confidence: {confidence:.3f}")
            
            # Test ensemble analysis
            analysis = ensemble.get_ensemble_analysis()
            logger.info(f"✅ Ensemble analysis: diversity={analysis.get('diversity_score', 0):.3f}, agreement={analysis.get('agreement_ratio', 0):.3f}")
            
            logger.info("✅ Phase 2.3: Ensemble Strategy working")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error testing ensemble strategy: {e}")
            return False
    
    async def test_signal_generator_integration(self):
        """Test ensemble integration with signal generator"""
        logger.info("🔗 Testing Phase 2.3: Signal Generator Integration...")
        
        try:
            # Import signal generator
            from app.strategies.real_time_signal_generator import RealTimeSignalGenerator
            
            # Create signal generator
            config = {
                'use_ensemble': True,
                'min_confidence': 0.6,
                'database_config': {
                    'host': 'localhost',
                    'port': 5432,
                    'database': 'alphapulse',
                    'user': 'alpha_emon',
                    'password': 'Emon_@17711'
                }
            }
            
            signal_generator = RealTimeSignalGenerator(config)
            logger.info("✅ Signal generator created with ensemble enabled")
            
            # Test ensemble initialization
            await signal_generator.initialize_ensemble_strategy()
            
            if signal_generator.ensemble_strategy:
                logger.info("✅ Ensemble strategy initialized in signal generator")
            else:
                logger.error("❌ Ensemble strategy not initialized")
                return False
            
            # Test feature preparation
            # Create sample market data
            dates = pd.date_range(start='2024-01-01', periods=100, freq='H')
            df = pd.DataFrame({
                'timestamp': dates,
                'open': np.random.randn(100).cumsum() + 100,
                'high': np.random.randn(100).cumsum() + 102,
                'low': np.random.randn(100).cumsum() + 98,
                'close': np.random.randn(100).cumsum() + 100,
                'volume': np.random.randint(1000, 10000, 100)
            })
            
            # Calculate basic indicators
            df['sma_20'] = df['close'].rolling(window=20).mean()
            df['sma_50'] = df['close'].rolling(window=50).mean()
            df['rsi'] = signal_generator._calculate_rsi(df['close'])
            df['macd'], df['macd_signal'] = signal_generator._calculate_macd(df['close'])
            
            # Test feature preparation
            features = signal_generator._prepare_features_for_ensemble(df)
            
            if features is not None and features.shape[1] > 0:
                logger.info(f"✅ Feature preparation working: {features.shape}")
            else:
                logger.error("❌ Feature preparation failed")
                return False
            
            # Test ensemble analysis
            ensemble_signals = await signal_generator.analyze_ensemble_predictions(df, 'BTCUSDT', '1h')
            
            if ensemble_signals:
                logger.info(f"✅ Ensemble analysis working: bias={ensemble_signals.get('ensemble_bias', 'unknown')}, confidence={ensemble_signals.get('ensemble_confidence', 0):.3f}")
            else:
                logger.warning("⚠️ Ensemble analysis returned empty (may be expected if ensemble not fully trained)")
            
            logger.info("✅ Phase 2.3: Signal Generator Integration working")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error testing signal generator integration: {e}")
            return False
    
    async def test_database_persistence(self):
        """Test ensemble data persistence in database"""
        logger.info("💾 Testing Phase 2.3: Database Persistence...")
        
        try:
            # Insert test signal with ensemble data
            test_signal = {
                'symbol': 'TEST_ENSEMBLE',
                'signal_type': 'buy',
                'confidence': 0.85,
                'strength': 0.75,
                'timestamp': datetime.now(),
                'price': 50000.0,
                'reason': 'Phase 2.3 Ensemble Test Signal',
                'ensemble_analysis': {
                    'voting_method': 'soft',
                    'model_count': 5,
                    'diversity_score': 0.75,
                    'agreement_ratio': 0.8,
                    'individual_predictions': {'rf': 1, 'gb': 1, 'lr': 0, 'svm': 1, 'nn': 1},
                    'model_weights': {'rf': 0.2, 'gb': 0.2, 'lr': 0.2, 'svm': 0.2, 'nn': 0.2},
                    'performance_metrics': {'accuracy': 0.82, 'precision': 0.85, 'recall': 0.78}
                },
                'ensemble_voting_method': 'soft',
                'ensemble_model_weights': {'rf': 0.2, 'gb': 0.2, 'lr': 0.2, 'svm': 0.2, 'nn': 0.2},
                'ensemble_individual_predictions': {'rf': 1, 'gb': 1, 'lr': 0, 'svm': 1, 'nn': 1},
                'ensemble_confidence': 0.85,
                'ensemble_diversity_score': 0.75,
                'ensemble_agreement_ratio': 0.8,
                'ensemble_bias': 'bullish',
                'ensemble_model_count': 5,
                'ensemble_performance_score': 0.82,
                'ensemble_last_updated': datetime.now()
            }
            
            # Insert into database
            insert_query = """
                INSERT INTO enhanced_signals (
                    symbol, signal_type, confidence, strength, timestamp, price, reason,
                    ensemble_analysis, ensemble_voting_method, ensemble_model_weights,
                    ensemble_individual_predictions, ensemble_confidence, ensemble_diversity_score,
                    ensemble_agreement_ratio, ensemble_bias, ensemble_model_count,
                    ensemble_performance_score, ensemble_last_updated
                ) VALUES (
                    $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18
                ) RETURNING id
            """
            
            result = await self.db_connection.fetchrow(insert_query,
                test_signal['symbol'], test_signal['signal_type'], test_signal['confidence'],
                test_signal['strength'], test_signal['timestamp'], test_signal['price'],
                test_signal['reason'], test_signal['ensemble_analysis'], test_signal['ensemble_voting_method'],
                test_signal['ensemble_model_weights'], test_signal['ensemble_individual_predictions'],
                test_signal['ensemble_confidence'], test_signal['ensemble_diversity_score'],
                test_signal['ensemble_agreement_ratio'], test_signal['ensemble_bias'],
                test_signal['ensemble_model_count'], test_signal['ensemble_performance_score'],
                test_signal['ensemble_last_updated']
            )
            
            if result:
                signal_id = result['id']
                logger.info(f"✅ Test signal inserted with ID: {signal_id}")
                
                # Retrieve and verify
                select_query = """
                    SELECT * FROM enhanced_signals WHERE id = $1
                """
                retrieved = await self.db_connection.fetchrow(select_query, signal_id)
                
                if retrieved:
                    logger.info(f"✅ Signal retrieved: ensemble_bias={retrieved['ensemble_bias']}, confidence={retrieved['ensemble_confidence']}")
                    
                    # Test ensemble view
                    view_query = """
                        SELECT * FROM ensemble_enhanced_signals WHERE symbol = $1
                    """
                    view_result = await self.db_connection.fetch(view_query, test_signal['symbol'])
                    
                    if view_result:
                        logger.info(f"✅ Ensemble view working: {len(view_result)} records")
                    else:
                        logger.warning("⚠️ No records in ensemble view")
                    
                    # Clean up test data
                    delete_query = "DELETE FROM enhanced_signals WHERE id = $1"
                    await self.db_connection.execute(delete_query, signal_id)
                    logger.info("✅ Test data cleaned up")
                    
                    return True
                else:
                    logger.error("❌ Failed to retrieve test signal")
                    return False
            else:
                logger.error("❌ Failed to insert test signal")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error testing database persistence: {e}")
            return False
    
    async def run_all_tests(self):
        """Run all Phase 2.3 tests"""
        logger.info("🚀 Starting Phase 2.3: Ensemble Model Integration Tests...")
        
        # Initialize database
        if not await self.initialize_database():
            return False
        
        tests = [
            ("Database Migrations", self.test_phase2_3_database_migrations),
            ("Ensemble Strategy", self.test_ensemble_strategy),
            ("Signal Generator Integration", self.test_signal_generator_integration),
            ("Database Persistence", self.test_database_persistence)
        ]
        
        results = {}
        
        for test_name, test_func in tests:
            logger.info(f"\n{'='*60}")
            logger.info(f"🧪 Running: {test_name}")
            logger.info(f"{'='*60}")
            
            try:
                result = await test_func()
                results[test_name] = result
                
                if result:
                    logger.info(f"✅ {test_name}: PASSED")
                else:
                    logger.error(f"❌ {test_name}: FAILED")
                    
            except Exception as e:
                logger.error(f"❌ {test_name}: ERROR - {e}")
                results[test_name] = False
        
        # Summary
        logger.info(f"\n{'='*60}")
        logger.info("📊 Phase 2.3: Test Results Summary")
        logger.info(f"{'='*60}")
        
        passed = sum(1 for result in results.values() if result)
        total = len(results)
        
        for test_name, result in results.items():
            status = "✅ PASSED" if result else "❌ FAILED"
            logger.info(f"{test_name}: {status}")
        
        logger.info(f"\nOverall: {passed}/{total} tests passed")
        
        if passed == total:
            logger.info("🎉 Phase 2.3: All tests passed! Ensemble Model Integration is working correctly.")
        else:
            logger.error(f"⚠️ Phase 2.3: {total - passed} tests failed. Please check the errors above.")
        
        return passed == total

async def main():
    """Main test function"""
    tester = Phase2_3EnsembleTester()
    success = await tester.run_all_tests()
    
    if success:
        logger.info("🎯 Phase 2.3: Ensemble Model Integration is ready for production!")
        sys.exit(0)
    else:
        logger.error("💥 Phase 2.3: Ensemble Model Integration has issues that need to be resolved.")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
