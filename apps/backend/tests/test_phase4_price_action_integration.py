"""
Test Phase 4: Advanced Price Action Integration
Tests the integration of sophisticated price action models with signal generator
"""

import asyncio
import logging
import asyncpg
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, List

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Database configuration
db_config = {
    'host': 'localhost',
    'port': 5432,
    'user': 'alpha_emon',
    'password': 'Emon_@17711',
    'database': 'alphapulse'
}

class Phase4PriceActionIntegrationTest:
    """Test suite for Phase 4: Advanced Price Action Integration"""
    
    def __init__(self):
        self.db_pool = None
        self.test_results = {
            'database_tables': False,
            'price_action_engine': False,
            'signal_integration': False,
            'performance_tracking': False,
            'configuration_loading': False
        }
    
    async def setup(self):
        """Setup test environment"""
        try:
            self.db_pool = await asyncpg.create_pool(**db_config)
            logger.info("✅ Database connection established")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to connect to database: {e}")
            return False
    
    async def test_database_tables(self) -> bool:
        """Test Phase 4 database tables creation"""
        try:
            logger.info("🔍 Testing Phase 4 database tables...")
            
            required_tables = [
                'price_action_ml_models',
                'support_resistance_levels',
                'market_structure_analysis',
                'demand_supply_zones',
                'price_action_ml_predictions',
                'price_action_signal_integration',
                'price_action_performance',
                'price_action_config',
                'price_action_alerts'
            ]
            
            async with self.db_pool.acquire() as conn:
                for table in required_tables:
                    # Check if table exists
                    exists = await conn.fetchval("""
                        SELECT EXISTS (
                            SELECT FROM information_schema.tables 
                            WHERE table_schema = 'public' 
                            AND table_name = $1
                        )
                    """, table)
                    
                    if not exists:
                        logger.error(f"❌ Table {table} does not exist")
                        return False
                    
                    logger.info(f"✅ Table {table} exists")
                
                # Check default configurations
                config_count = await conn.fetchval("""
                    SELECT COUNT(*) FROM price_action_config 
                    WHERE is_active = true
                """)
                
                if config_count < 4:
                    logger.error(f"❌ Expected 4+ default configurations, found {config_count}")
                    return False
                
                logger.info(f"✅ Found {config_count} default configurations")
                
                # Check indexes
                index_count = await conn.fetchval("""
                    SELECT COUNT(*) FROM pg_indexes 
                    WHERE tablename IN (
                        'support_resistance_levels',
                        'market_structure_analysis',
                        'demand_supply_zones',
                        'price_action_ml_predictions',
                        'price_action_signal_integration'
                    )
                """)
                
                if index_count < 20:
                    logger.error(f"❌ Expected 20+ indexes, found {index_count}")
                    return False
                
                logger.info(f"✅ Found {index_count} performance indexes")
            
            logger.info("✅ All Phase 4 database tables and configurations verified")
            return True
            
        except Exception as e:
            logger.error(f"❌ Database tables test failed: {e}")
            return False
    
    async def test_price_action_engine(self) -> bool:
        """Test Advanced Price Action Integration Engine"""
        try:
            logger.info("🔍 Testing Advanced Price Action Integration Engine...")
            
            # Import the engine
            try:
                from src.strategies.advanced_price_action_integration import AdvancedPriceActionIntegration
                logger.info("✅ Advanced Price Action Integration Engine imported successfully")
            except ImportError as e:
                logger.error(f"❌ Failed to import Advanced Price Action Integration Engine: {e}")
                return False
            
            # Initialize the engine
            engine = AdvancedPriceActionIntegration(db_pool=self.db_pool)
            await engine.initialize()
            logger.info("✅ Advanced Price Action Integration Engine initialized")
            
            # Test configuration loading
            if not engine.config:
                logger.error("❌ Engine configuration is empty")
                return False
            
            required_config_keys = [
                'support_resistance_weight',
                'market_structure_weight',
                'demand_supply_weight',
                'pattern_ml_weight',
                'min_combined_score',
                'min_confidence_threshold'
            ]
            
            for key in required_config_keys:
                if key not in engine.config:
                    logger.error(f"❌ Missing configuration key: {key}")
                    return False
            
            logger.info("✅ Engine configuration loaded successfully")
            
            # Test performance stats
            stats = await engine.get_performance_stats()
            if not isinstance(stats, dict):
                logger.error("❌ Performance stats not returned as dict")
                return False
            
            logger.info("✅ Performance stats working correctly")
            
            # Cleanup
            await engine.cleanup()
            logger.info("✅ Engine cleanup completed")
            
            logger.info("✅ Advanced Price Action Integration Engine test passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Price action engine test failed: {e}")
            return False
    
    async def test_signal_integration(self) -> bool:
        """Test signal integration with price action analysis"""
        try:
            logger.info("🔍 Testing signal integration with price action analysis...")
            
            # Import required components
            try:
                from src.strategies.advanced_price_action_integration import AdvancedPriceActionIntegration, EnhancedSignal
                logger.info("✅ Signal integration components imported")
            except ImportError as e:
                logger.error(f"❌ Failed to import signal integration components: {e}")
                return False
            
            # Initialize engine
            engine = AdvancedPriceActionIntegration(db_pool=self.db_pool)
            await engine.initialize()
            
            # Create mock market data
            mock_market_data = {
                'ohlcv': [
                    [datetime.now().timestamp(), 50000.0, 51000.0, 49000.0, 50500.0, 1000000.0],
                    [(datetime.now() - timedelta(minutes=1)).timestamp(), 49500.0, 50500.0, 48500.0, 50000.0, 950000.0],
                    [(datetime.now() - timedelta(minutes=2)).timestamp(), 49000.0, 50000.0, 48000.0, 49500.0, 900000.0],
                ],
                'indicators': {
                    'rsi': 65.0,
                    'macd': 0.5,
                    'bollinger_upper': 52000.0,
                    'bollinger_lower': 48000.0,
                    'volume_sma': 1000000.0
                }
            }
            
            # Test price action analysis
            analysis = await engine.analyze_price_action('BTCUSDT', '1h', mock_market_data)
            
            if not analysis:
                logger.error("❌ Price action analysis returned None")
                return False
            
            # Verify analysis structure
            required_attrs = [
                'support_resistance_score',
                'market_structure_score',
                'demand_supply_score',
                'pattern_ml_score',
                'combined_price_action_score',
                'price_action_confidence'
            ]
            
            for attr in required_attrs:
                if not hasattr(analysis, attr):
                    logger.error(f"❌ Analysis missing attribute: {attr}")
                    return False
            
            logger.info(f"✅ Price action analysis completed (score: {analysis.combined_price_action_score:.3f})")
            
            # Test signal enhancement
            enhanced_signal = await engine.enhance_signal(
                signal_id=str(uuid.uuid4()),
                symbol="BTCUSDT",
                timeframe="1h",
                original_confidence=0.75,
                original_risk_reward=2.5,
                original_entry_price=50000.0,
                original_stop_loss=48000.0,
                original_take_profit=52000.0,
                market_data=mock_market_data
            )
            
            if not enhanced_signal:
                logger.error("❌ Signal enhancement returned None")
                return False
            
            # Verify enhanced signal structure
            if not isinstance(enhanced_signal, EnhancedSignal):
                logger.error("❌ Enhanced signal is not of correct type")
                return False
            
            logger.info(f"✅ Signal enhancement completed (confidence: {enhanced_signal.enhanced_confidence:.3f})")
            
            # Cleanup
            await engine.cleanup()
            
            logger.info("✅ Signal integration test passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Signal integration test failed: {e}")
            return False
    
    async def test_performance_tracking(self) -> bool:
        """Test performance tracking and database storage"""
        try:
            logger.info("🔍 Testing performance tracking...")
            
            async with self.db_pool.acquire() as conn:
                # Check if performance data is being stored
                prediction_count = await conn.fetchval("""
                    SELECT COUNT(*) FROM price_action_ml_predictions
                """)
                
                integration_count = await conn.fetchval("""
                    SELECT COUNT(*) FROM price_action_signal_integration
                """)
                
                logger.info(f"✅ Found {prediction_count} price action predictions")
                logger.info(f"✅ Found {integration_count} signal integrations")
                
                # Test inserting performance data
                test_performance = {
                    'symbol': 'TESTUSDT',
                    'timeframe': '1h',
                    'model_type': 'support_resistance',
                    'timestamp': datetime.now(),
                    'accuracy_score': 0.85,
                    'precision_score': 0.82,
                    'recall_score': 0.88,
                    'f1_score': 0.85,
                    'profit_factor': 1.5,
                    'win_rate': 0.65,
                    'avg_win': 100.0,
                    'avg_loss': 50.0,
                    'max_drawdown': 0.15,
                    'sharpe_ratio': 1.2,
                    'signal_count': 100,
                    'successful_signals': 65,
                    'failed_signals': 35,
                    'avg_confidence': 0.78,
                    'avg_risk_reward': 2.1,
                    'inference_latency_ms': 25.0,
                    'feature_importance': {'feature1': 0.3, 'feature2': 0.7},
                    'model_confidence': 0.85,
                    'market_regime': 'trending',
                    'volatility_level': 'medium',
                    'trend_strength': 0.75
                }
                
                await conn.execute("""
                    INSERT INTO price_action_performance (
                        symbol, timeframe, model_type, timestamp,
                        accuracy_score, precision_score, recall_score, f1_score,
                        profit_factor, win_rate, avg_win, avg_loss, max_drawdown, sharpe_ratio,
                        signal_count, successful_signals, failed_signals, avg_confidence, avg_risk_reward,
                        inference_latency_ms, feature_importance, model_confidence,
                        market_regime, volatility_level, trend_strength
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25)
                """, 
                test_performance['symbol'], test_performance['timeframe'], test_performance['model_type'], test_performance['timestamp'],
                test_performance['accuracy_score'], test_performance['precision_score'], test_performance['recall_score'], test_performance['f1_score'],
                test_performance['profit_factor'], test_performance['win_rate'], test_performance['avg_win'], test_performance['avg_loss'],
                test_performance['max_drawdown'], test_performance['sharpe_ratio'], test_performance['signal_count'], test_performance['successful_signals'],
                test_performance['failed_signals'], test_performance['avg_confidence'], test_performance['avg_risk_reward'], test_performance['inference_latency_ms'],
                json.dumps(test_performance['feature_importance']), test_performance['model_confidence'], test_performance['market_regime'],
                test_performance['volatility_level'], test_performance['trend_strength']
                )
                
                logger.info("✅ Performance data inserted successfully")
                
                # Verify insertion
                new_count = await conn.fetchval("""
                    SELECT COUNT(*) FROM price_action_performance WHERE symbol = 'TESTUSDT'
                """)
                
                if new_count == 0:
                    logger.error("❌ Performance data not found after insertion")
                    return False
                
                logger.info("✅ Performance data verified")
            
            logger.info("✅ Performance tracking test passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Performance tracking test failed: {e}")
            return False
    
    async def test_configuration_loading(self) -> bool:
        """Test configuration loading from database"""
        try:
            logger.info("🔍 Testing configuration loading...")
            
            async with self.db_pool.acquire() as conn:
                # Test loading integration configuration
                config = await conn.fetchrow("""
                    SELECT config_data FROM price_action_config 
                    WHERE config_name = 'price_action_integration_default' AND is_active = true
                """)
                
                if not config:
                    logger.error("❌ Integration configuration not found")
                    return False
                
                config_data = config['config_data']
                
                # Parse JSON if it's a string
                if isinstance(config_data, str):
                    import json
                    config_data = json.loads(config_data)
                
                # Verify configuration structure
                required_keys = [
                    'support_resistance_weight',
                    'market_structure_weight',
                    'demand_supply_weight',
                    'pattern_ml_weight',
                    'min_combined_score',
                    'min_confidence_threshold',
                    'enhancement_factor',
                    'risk_reward_improvement'
                ]
                
                for key in required_keys:
                    if key not in config_data:
                        logger.error(f"❌ Configuration missing key: {key}")
                        return False
                
                logger.info("✅ Configuration structure verified")
                
                # Test configuration values
                if not (0 <= config_data['support_resistance_weight'] <= 1):
                    logger.error("❌ Invalid support_resistance_weight")
                    return False
                
                if not (0 <= config_data['min_combined_score'] <= 1):
                    logger.error("❌ Invalid min_combined_score")
                    return False
                
                logger.info("✅ Configuration values verified")
            
            logger.info("✅ Configuration loading test passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Configuration loading test failed: {e}")
            return False
    
    async def run_all_tests(self):
        """Run all Phase 4 tests"""
        logger.info("🚀 Starting Phase 4: Advanced Price Action Integration Tests")
        
        if not await self.setup():
            logger.error("❌ Test setup failed")
            return False
        
        try:
            # Run all tests
            self.test_results['database_tables'] = await self.test_database_tables()
            self.test_results['price_action_engine'] = await self.test_price_action_engine()
            self.test_results['signal_integration'] = await self.test_signal_integration()
            self.test_results['performance_tracking'] = await self.test_performance_tracking()
            self.test_results['configuration_loading'] = await self.test_configuration_loading()
            
            # Calculate overall result
            passed_tests = sum(self.test_results.values())
            total_tests = len(self.test_results)
            
            logger.info("\n" + "="*60)
            logger.info("📊 PHASE 4 TEST RESULTS")
            logger.info("="*60)
            
            for test_name, result in self.test_results.items():
                status = "✅ PASSED" if result else "❌ FAILED"
                logger.info(f"{test_name.replace('_', ' ').title()}: {status}")
            
            logger.info("="*60)
            logger.info(f"Overall Result: {passed_tests}/{total_tests} tests passed")
            
            if passed_tests == total_tests:
                logger.info("🎉 PHASE 4: Advanced Price Action Integration - ALL TESTS PASSED!")
                return True
            else:
                logger.error("❌ PHASE 4: Some tests failed")
                return False
                
        except Exception as e:
            logger.error(f"❌ Test execution failed: {e}")
            return False
        finally:
            if self.db_pool:
                await self.db_pool.close()

async def main():
    """Main test execution"""
    test_suite = Phase4PriceActionIntegrationTest()
    success = await test_suite.run_all_tests()
    
    if success:
        logger.info("✅ Phase 4 implementation is ready for production!")
    else:
        logger.error("❌ Phase 4 implementation needs fixes")

if __name__ == "__main__":
    asyncio.run(main())
