"""
Test Enhanced Model Heads Phase 5
Comprehensive testing of enhanced model head creation, ONNX integration, and database operations
"""

import asyncio
import logging
import asyncpg
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import sys
import os

# Add the backend directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ai.sde_framework import SDEFramework, ModelHead, SignalDirection, ModelHeadResult
from ai.sde_integration_manager import SDEIntegrationManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EnhancedModelHeadsTester:
    """Test enhanced model heads functionality"""
    
    def __init__(self):
        self.db_pool = None
        self.sde_framework = None
        self.sde_integration_manager = None
        
    async def setup(self):
        """Setup database connection and components"""
        try:
            # Database connection
            self.db_pool = await asyncpg.create_pool(
                host='localhost',
                port=5432,
                user='alpha_emon',
                password='Emon_@17711',
                database='alphapulse'
            )
            
            # Initialize SDE components
            self.sde_framework = SDEFramework(self.db_pool)
            self.sde_integration_manager = SDEIntegrationManager(self.db_pool)
            
            logger.info("✅ Setup completed successfully")
            
        except Exception as e:
            logger.error(f"❌ Setup failed: {e}")
            raise
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.db_pool:
            await self.db_pool.close()
    
    def create_mock_market_data(self) -> pd.DataFrame:
        """Create mock market data for testing"""
        dates = pd.date_range(start='2024-01-01', periods=100, freq='1H')
        
        # Create realistic price data
        np.random.seed(42)
        base_price = 50000
        returns = np.random.normal(0, 0.02, 100)
        prices = [base_price]
        
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        df = pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
            'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
            'close': prices,
            'volume': np.random.uniform(1000, 10000, 100)
        })
        
        # Calculate technical indicators
        df['rsi'] = np.random.uniform(20, 80, 100)
        df['macd'] = np.random.uniform(-100, 100, 100)
        df['volume_ratio'] = np.random.uniform(0.5, 2.0, 100)
        df['atr'] = np.random.uniform(100, 500, 100)
        
        return df
    
    def create_mock_analysis_results(self) -> dict:
        """Create mock analysis results"""
        return {
            # Technical analysis
            'technical_confidence': 0.75,
            'technical_direction': 'long',
            'rsi': 65.0,
            'macd_signal': 0.8,
            'volume_ratio': 1.2,
            'bollinger_position': 0.6,
            'ema_trend': 0.7,
            'atr_percentile': 60.0,
            
            # Sentiment analysis
            'sentiment_score': 0.65,
            'news_impact': 0.1,
            'social_sentiment': 0.6,
            'fear_greed_index': 55.0,
            
            # Volume analysis
            'volume_confidence': 0.7,
            'volume_direction': 'long',
            'volume_delta': 0.15,
            'orderbook_imbalance': 0.2,
            'liquidity_score': 0.8,
            'spread_atr_ratio': 0.08,
            
            # Market regime
            'market_regime_confidence': 0.6,
            'market_regime_direction': 'long',
            
            # Price action
            'zone_score': 7.5,
            'structure_score': 8.0,
            'pattern_score': 7.0,
            'confluence_score': 7.8
        }
    
    def create_mock_market_data_dict(self) -> dict:
        """Create mock market data dictionary"""
        return {
            'current_price': 50000.0,
            'stop_loss': 49000.0,
            'atr_value': 300.0,
            'market_data_df': self.create_mock_market_data()
        }
    
    async def test_enhanced_model_head_creation(self):
        """Test enhanced model head creation"""
        logger.info("🧪 Testing enhanced model head creation...")
        
        try:
            # Create mock data
            analysis_results = self.create_mock_analysis_results()
            market_data = self.create_mock_market_data_dict()
            
            # Test enhanced model head creation
            model_heads = await self.sde_framework.create_enhanced_model_head_results(
                analysis_results, market_data, 'BTCUSDT', '1h'
            )
            
            # Validate results
            assert len(model_heads) == 4, f"Expected 4 model heads, got {len(model_heads)}"
            
            # Check each head
            for i, head in enumerate(model_heads):
                assert isinstance(head, ModelHeadResult), f"Head {i} is not ModelHeadResult"
                assert head.head_type in [ModelHead.HEAD_A, ModelHead.HEAD_B, ModelHead.HEAD_C, ModelHead.HEAD_D]
                assert 0.0 <= head.probability <= 1.0, f"Invalid probability: {head.probability}"
                assert 0.0 <= head.confidence <= 1.0, f"Invalid confidence: {head.confidence}"
                assert isinstance(head.features_used, list), f"Features used should be list: {head.features_used}"
                assert isinstance(head.reasoning, str), f"Reasoning should be string: {head.reasoning}"
            
            logger.info("✅ Enhanced model head creation test passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Enhanced model head creation test failed: {e}")
            return False
    
    async def test_model_consensus_with_enhanced_heads(self):
        """Test model consensus with enhanced heads"""
        logger.info("🧪 Testing model consensus with enhanced heads...")
        
        try:
            # Create mock data
            analysis_results = self.create_mock_analysis_results()
            market_data = self.create_mock_market_data_dict()
            
            # Create enhanced model heads
            model_heads = await self.sde_framework.create_enhanced_model_head_results(
                analysis_results, market_data, 'BTCUSDT', '1h'
            )
            
            # Test consensus
            consensus_result = await self.sde_framework.check_model_consensus(model_heads)
            
            # Validate consensus result
            assert isinstance(consensus_result.consensus_achieved, bool)
            assert isinstance(consensus_result.consensus_score, float)
            assert 0.0 <= consensus_result.consensus_score <= 1.0
            assert isinstance(consensus_result.agreeing_heads, list)
            assert isinstance(consensus_result.disagreeing_heads, list)
            
            logger.info(f"✅ Consensus test passed - Achieved: {consensus_result.consensus_achieved}, Score: {consensus_result.consensus_score:.3f}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Model consensus test failed: {e}")
            return False
    
    async def test_sde_integration_with_enhanced_heads(self):
        """Test SDE integration with enhanced model heads"""
        logger.info("🧪 Testing SDE integration with enhanced heads...")
        
        try:
            # Create mock data
            analysis_results = self.create_mock_analysis_results()
            market_data = self.create_mock_market_data_dict()
            
            # Test SDE integration
            integration_result = await self.sde_integration_manager.integrate_sde_with_signal(
                signal_id="test_signal_001",
                symbol="BTCUSDT",
                timeframe="1h",
                analysis_results=analysis_results,
                market_data=market_data,
                account_id="test_account"
            )
            
            # Validate integration result
            assert integration_result.signal_id == "test_signal_001"
            assert integration_result.symbol == "BTCUSDT"
            assert integration_result.timeframe == "1h"
            assert isinstance(integration_result.consensus_result.consensus_achieved, bool)
            assert isinstance(integration_result.final_confidence, float)
            assert 0.0 <= integration_result.final_confidence <= 1.0
            
            logger.info(f"✅ SDE integration test passed - Final confidence: {integration_result.final_confidence:.3f}")
            return True
            
        except Exception as e:
            logger.error(f"❌ SDE integration test failed: {e}")
            return False
    
    async def test_database_tables_creation(self):
        """Test that all database tables were created successfully"""
        logger.info("🧪 Testing database tables creation...")
        
        try:
            async with self.db_pool.acquire() as conn:
                # Check sde_enhanced_model_heads table
                result = await conn.fetchrow("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = 'sde_enhanced_model_heads'
                    )
                """)
                assert result['exists'], "sde_enhanced_model_heads table not found"
                
                # Check sde_onnx_model_registry table
                result = await conn.fetchrow("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = 'sde_onnx_model_registry'
                    )
                """)
                assert result['exists'], "sde_onnx_model_registry table not found"
                
                # Check sde_enhanced_features table
                result = await conn.fetchrow("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = 'sde_enhanced_features'
                    )
                """)
                assert result['exists'], "sde_enhanced_features table not found"
                
                # Check sde_model_head_performance table
                result = await conn.fetchrow("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = 'sde_model_head_performance'
                    )
                """)
                assert result['exists'], "sde_model_head_performance table not found"
                
                # Check default ONNX model configurations
                models = await conn.fetch("""
                    SELECT model_name, model_type, is_active 
                    FROM sde_onnx_model_registry 
                    WHERE is_active = true
                """)
                assert len(models) >= 3, f"Expected at least 3 ONNX models, got {len(models)}"
                
                model_names = [m['model_name'] for m in models]
                expected_models = ['catboost_technical', 'logistic_sentiment', 'tree_orderflow']
                for expected in expected_models:
                    assert expected in model_names, f"Expected model {expected} not found"
            
            logger.info("✅ Database tables creation test passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Database tables creation test failed: {e}")
            return False
    
    async def test_enhanced_model_head_storage(self):
        """Test storing enhanced model head results in database"""
        logger.info("🧪 Testing enhanced model head storage...")
        
        try:
            # Create mock data
            analysis_results = self.create_mock_analysis_results()
            market_data = self.create_mock_market_data_dict()
            
            # Create enhanced model heads
            model_heads = await self.sde_framework.create_enhanced_model_head_results(
                analysis_results, market_data, 'BTCUSDT', '1h'
            )
            
            # Store in database
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO sde_enhanced_model_heads (
                        signal_id, symbol, timeframe,
                        head_a_type, head_a_direction, head_a_probability, head_a_confidence,
                        head_a_features_used, head_a_reasoning, head_a_model_version,
                        head_b_type, head_b_direction, head_b_probability, head_b_confidence,
                        head_b_features_used, head_b_reasoning, head_b_model_version,
                        head_c_type, head_c_direction, head_c_probability, head_c_confidence,
                        head_c_features_used, head_c_reasoning, head_c_model_version,
                        head_d_type, head_d_direction, head_d_probability, head_d_confidence,
                        head_d_features_used, head_d_reasoning, head_d_model_version,
                        consensus_achieved, consensus_direction, consensus_score,
                        agreeing_heads_count, disagreeing_heads_count,
                        processing_time_ms, model_creation_success, onnx_inference_used, feature_engineering_used
                    ) VALUES (
                        $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16,
                        $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28, $29, $30,
                        $31, $32, $33, $34, $35, $36, $37, $38, $39, $40
                    )
                """, 
                'test_signal_002', 'BTCUSDT', '1h',
                model_heads[0].head_type.value, model_heads[0].direction.value, 
                model_heads[0].probability, model_heads[0].confidence,
                json.dumps(model_heads[0].features_used), model_heads[0].reasoning, 'v1.0',
                model_heads[1].head_type.value, model_heads[1].direction.value,
                model_heads[1].probability, model_heads[1].confidence,
                json.dumps(model_heads[1].features_used), model_heads[1].reasoning, 'v1.0',
                model_heads[2].head_type.value, model_heads[2].direction.value,
                model_heads[2].probability, model_heads[2].confidence,
                json.dumps(model_heads[2].features_used), model_heads[2].reasoning, 'v1.0',
                model_heads[3].head_type.value, model_heads[3].direction.value,
                model_heads[3].probability, model_heads[3].confidence,
                json.dumps(model_heads[3].features_used), model_heads[3].reasoning, 'v1.0',
                True, 'LONG', 0.75, 3, 1, 150, True, False, False
                )
                
                # Verify storage
                result = await conn.fetchrow("""
                    SELECT signal_id, symbol, timeframe, model_creation_success
                    FROM sde_enhanced_model_heads 
                    WHERE signal_id = $1
                """, 'test_signal_002')
                
                assert result is not None, "Enhanced model head result not stored"
                assert result['signal_id'] == 'test_signal_002'
                assert result['symbol'] == 'BTCUSDT'
                assert result['timeframe'] == '1h'
                assert result['model_creation_success'] is True
            
            logger.info("✅ Enhanced model head storage test passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Enhanced model head storage test failed: {e}")
            return False
    
    async def run_all_tests(self):
        """Run all tests"""
        logger.info("🚀 Starting Enhanced Model Heads Phase 5 tests...")
        
        tests = [
            ("Database Tables Creation", self.test_database_tables_creation),
            ("Enhanced Model Head Creation", self.test_enhanced_model_head_creation),
            ("Model Consensus with Enhanced Heads", self.test_model_consensus_with_enhanced_heads),
            ("SDE Integration with Enhanced Heads", self.test_sde_integration_with_enhanced_heads),
            ("Enhanced Model Head Storage", self.test_enhanced_model_head_storage)
        ]
        
        results = []
        for test_name, test_func in tests:
            logger.info(f"\n{'='*60}")
            logger.info(f"Running: {test_name}")
            logger.info(f"{'='*60}")
            
            try:
                result = await test_func()
                results.append((test_name, result))
                
                if result:
                    logger.info(f"✅ {test_name}: PASSED")
                else:
                    logger.error(f"❌ {test_name}: FAILED")
                    
            except Exception as e:
                logger.error(f"❌ {test_name}: ERROR - {e}")
                results.append((test_name, False))
        
        # Summary
        logger.info(f"\n{'='*60}")
        logger.info("TEST SUMMARY")
        logger.info(f"{'='*60}")
        
        passed = sum(1 for _, result in results if result)
        total = len(results)
        
        for test_name, result in results:
            status = "✅ PASSED" if result else "❌ FAILED"
            logger.info(f"{test_name}: {status}")
        
        logger.info(f"\nOverall: {passed}/{total} tests passed")
        
        if passed == total:
            logger.info("🎉 All tests passed! Enhanced Model Heads Phase 5 is working correctly.")
        else:
            logger.error(f"⚠️ {total - passed} tests failed. Please check the implementation.")
        
        return passed == total

async def main():
    """Main test function"""
    tester = EnhancedModelHeadsTester()
    
    try:
        await tester.setup()
        success = await tester.run_all_tests()
        return success
    finally:
        await tester.cleanup()

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
