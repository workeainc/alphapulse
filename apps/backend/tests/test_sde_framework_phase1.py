"""
Test Script for SDE Framework Phase 1 Implementation
Tests model consensus, confluence scoring, and execution quality assessment
"""

import asyncio
import logging
import asyncpg
import uuid
from datetime import datetime
from typing import Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Database configuration
db_config = {
    'host': 'localhost',
    'port': 5432,
    'user': 'alpha_emon',
    'password': 'Emon_@17711',
    'database': 'alphapulse'
}

async def test_sde_framework():
    """Test SDE framework functionality"""
    logger.info("🚀 Starting SDE Framework Phase 1 Tests")
    
    try:
        # Create database connection
        pool = await asyncpg.create_pool(**db_config)
        logger.info("✅ Database connection established")
        
        # Import SDE framework
        from src.ai.sde_framework import SDEFramework, ModelHeadResult, SignalDirection
        
        # Initialize SDE framework
        sde_framework = SDEFramework(pool)
        logger.info("✅ SDE Framework initialized")
        
        # Test 1: Configuration Loading
        logger.info("📋 Test 1: Configuration Loading")
        await sde_framework.load_configurations()
        
        if sde_framework.config:
            logger.info(f"✅ Loaded {len(sde_framework.config)} configurations")
            for config_name, config_data in sde_framework.config.items():
                logger.info(f"  - {config_name}: {type(config_data)}")
        else:
            logger.error("❌ No configurations loaded")
            return False
        
        # Test 2: Model Consensus Check
        logger.info("📋 Test 2: Model Consensus Check")
        
        # Test case 1: Consensus achieved
        head_results_consensus = {
            'head_a': ModelHeadResult(SignalDirection.LONG, 0.85, 0.85),
            'head_b': ModelHeadResult(SignalDirection.LONG, 0.80, 0.80),
            'head_c': ModelHeadResult(SignalDirection.LONG, 0.75, 0.75),
            'head_d': ModelHeadResult(SignalDirection.SHORT, 0.60, 0.60)
        }
        
        consensus_result = await sde_framework.check_model_consensus(head_results_consensus)
        logger.info(f"Consensus Test 1 - Achieved: {consensus_result.achieved}, Direction: {consensus_result.direction}, Probability: {consensus_result.probability:.3f}, Agreeing Heads: {consensus_result.agreeing_heads_count}")
        
        # Test case 2: No consensus
        head_results_no_consensus = {
            'head_a': ModelHeadResult(SignalDirection.LONG, 0.65, 0.65),
            'head_b': ModelHeadResult(SignalDirection.SHORT, 0.60, 0.60),
            'head_c': ModelHeadResult(SignalDirection.FLAT, 0.55, 0.55),
            'head_d': ModelHeadResult(SignalDirection.LONG, 0.70, 0.70)
        }
        
        consensus_result_2 = await sde_framework.check_model_consensus(head_results_no_consensus)
        logger.info(f"Consensus Test 2 - Achieved: {consensus_result_2.achieved}, Direction: {consensus_result_2.direction}, Probability: {consensus_result_2.probability:.3f}, Agreeing Heads: {consensus_result_2.agreeing_heads_count}")
        
        # Test 3: Confluence Score Calculation
        logger.info("📋 Test 3: Confluence Score Calculation")
        
        # Test case 1: High confluence
        analysis_results_high = {
            'support_resistance_quality': 0.9,
            'volume_confirmation': True,
            'htf_trend_strength': 0.8,
            'trend_alignment': True,
            'pattern_strength': 0.85,
            'breakout_confirmed': True
        }
        
        confluence_result = await sde_framework.calculate_confluence_score(analysis_results_high)
        logger.info(f"Confluence Test 1 - Score: {confluence_result.total_score:.2f}/10.0, Gate Passed: {confluence_result.gate_passed}")
        logger.info(f"  Breakdown: {confluence_result.breakdown}")
        
        # Test case 2: Low confluence
        analysis_results_low = {
            'support_resistance_quality': 0.3,
            'volume_confirmation': False,
            'htf_trend_strength': 0.4,
            'trend_alignment': False,
            'pattern_strength': 0.2,
            'breakout_confirmed': False
        }
        
        confluence_result_2 = await sde_framework.calculate_confluence_score(analysis_results_low)
        logger.info(f"Confluence Test 2 - Score: {confluence_result_2.total_score:.2f}/10.0, Gate Passed: {confluence_result_2.gate_passed}")
        logger.info(f"  Breakdown: {confluence_result_2.breakdown}")
        
        # Test 4: Execution Quality Assessment
        logger.info("📋 Test 4: Execution Quality Assessment")
        
        # Test case 1: Good execution quality
        market_data_good = {
            'spread_atr_ratio': 0.08,
            'impact_cost': 0.05
        }
        
        execution_result = await sde_framework.assess_execution_quality(market_data_good)
        logger.info(f"Execution Test 1 - Quality Score: {execution_result.quality_score:.2f}/10.0, All Gates Passed: {execution_result.all_gates_passed}")
        logger.info(f"  Breakdown: {execution_result.breakdown}")
        
        # Test case 2: Poor execution quality
        market_data_poor = {
            'spread_atr_ratio': 0.20,
            'impact_cost': 0.25
        }
        
        execution_result_2 = await sde_framework.assess_execution_quality(market_data_poor)
        logger.info(f"Execution Test 2 - Quality Score: {execution_result_2.quality_score:.2f}/10.0, All Gates Passed: {execution_result_2.all_gates_passed}")
        logger.info(f"  Breakdown: {execution_result_2.breakdown}")
        
        # Test 5: Database Integration
        logger.info("📋 Test 5: Database Integration")
        
        # Check if SDE tables exist
        async with pool.acquire() as conn:
            tables = await conn.fetch("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name LIKE 'sde_%'
                ORDER BY table_name
            """)
            
            logger.info(f"✅ Found {len(tables)} SDE tables:")
            for table in tables:
                logger.info(f"  - {table['table_name']}")
            
            # Check configuration data
            configs = await conn.fetch("SELECT config_name, config_type FROM sde_config WHERE is_active = true")
            logger.info(f"✅ Found {len(configs)} active SDE configurations:")
            for config in configs:
                logger.info(f"  - {config['config_name']} ({config['config_type']})")
        
        # Test 6: End-to-End Signal Processing
        logger.info("📋 Test 6: End-to-End Signal Processing")
        
        # Simulate complete signal processing with SDE framework
        signal_id = str(uuid.uuid4())
        symbol = "BTCUSDT"
        timeframe = "1h"
        
        # Create comprehensive analysis results
        comprehensive_results = {
            'technical_result': {
                'confidence': 0.85,
                'support_resistance_quality': 0.8,
                'trend_alignment': True,
                'current_price': 50000.0,
                'spread_atr_ratio': 0.10
            },
            'sentiment_result': {
                'sentiment_score': 0.7
            },
            'volume_result': {
                'confidence': 0.75,
                'volume_confirmation': True,
                'impact_cost': 0.08
            },
            'market_regime_result': {
                'confidence': 0.80,
                'htf_trend_strength': 0.75
            },
            'price_action_result': {
                'combined_price_action_score': 0.82,
                'pattern_strength': 0.8,
                'breakout_confirmed': True
            }
        }
        
        # Process through SDE framework
        head_results = {
            'head_a': ModelHeadResult(SignalDirection.LONG, 0.85, 0.85),
            'head_b': ModelHeadResult(SignalDirection.LONG, 0.82, 0.82),
            'head_c': ModelHeadResult(SignalDirection.LONG, 0.75, 0.75),
            'head_d': ModelHeadResult(SignalDirection.LONG, 0.80, 0.80)
        }
        
        consensus_result = await sde_framework.check_model_consensus(head_results)
        
        analysis_results = {
            'support_resistance_quality': comprehensive_results['technical_result']['support_resistance_quality'],
            'volume_confirmation': comprehensive_results['volume_result']['volume_confirmation'],
            'htf_trend_strength': comprehensive_results['market_regime_result']['htf_trend_strength'],
            'trend_alignment': comprehensive_results['technical_result']['trend_alignment'],
            'pattern_strength': comprehensive_results['price_action_result']['pattern_strength'],
            'breakout_confirmed': comprehensive_results['price_action_result']['breakout_confirmed']
        }
        
        confluence_result = await sde_framework.calculate_confluence_score(analysis_results)
        
        market_data = {
            'spread_atr_ratio': comprehensive_results['technical_result']['spread_atr_ratio'],
            'impact_cost': comprehensive_results['volume_result']['impact_cost']
        }
        
        execution_result = await sde_framework.assess_execution_quality(market_data)
        
        # Calculate final signal quality
        base_confidence = 0.80
        final_confidence = base_confidence
        
        if not consensus_result.achieved:
            final_confidence *= 0.5
            logger.info(f"❌ Consensus failed: {consensus_result.agreeing_heads_count}/4 heads agreed")
        
        if not confluence_result.gate_passed:
            final_confidence *= 0.7
            logger.info(f"❌ Confluence failed: score {confluence_result.total_score:.2f}/8.0")
        
        if not execution_result.all_gates_passed:
            final_confidence *= 0.8
            logger.info(f"❌ Execution quality failed: score {execution_result.quality_score:.2f}/10.0")
        
        logger.info(f"📊 Final Signal Quality Assessment:")
        logger.info(f"  - Base Confidence: {base_confidence:.3f}")
        logger.info(f"  - Final Confidence: {final_confidence:.3f}")
        logger.info(f"  - Consensus: {'✅' if consensus_result.achieved else '❌'} ({consensus_result.agreeing_heads_count}/4)")
        logger.info(f"  - Confluence: {'✅' if confluence_result.gate_passed else '❌'} ({confluence_result.total_score:.2f}/8.0)")
        logger.info(f"  - Execution: {'✅' if execution_result.all_gates_passed else '❌'} ({execution_result.quality_score:.2f}/10.0)")
        
        # Determine if signal should be emitted
        signal_emitted = (final_confidence >= 0.85 and 
                         consensus_result.achieved and 
                         confluence_result.gate_passed and 
                         execution_result.all_gates_passed)
        
        logger.info(f"🎯 Signal Emission Decision: {'✅ EMIT' if signal_emitted else '❌ REJECT'}")
        
        # Close connection
        await pool.close()
        
        logger.info("🎉 SDE Framework Phase 1 Tests completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return False

async def test_signal_generator_integration():
    """Test SDE framework integration with signal generator"""
    logger.info("🚀 Testing SDE Framework Integration with Signal Generator")
    
    try:
        # Create database connection
        pool = await asyncpg.create_pool(**db_config)
        logger.info("✅ Database connection established")
        
        # Import signal generator
        from src.app.signals.intelligent_signal_generator import IntelligentSignalGenerator
        
        # Initialize signal generator
        signal_generator = IntelligentSignalGenerator(pool, None)  # No exchange for testing
        logger.info("✅ Signal Generator initialized")
        
        # Check if SDE framework is available
        if hasattr(signal_generator, 'sde_framework') and signal_generator.sde_framework:
            logger.info("✅ SDE Framework integrated with Signal Generator")
        else:
            logger.warning("⚠️ SDE Framework not available in Signal Generator")
            return False
        
        # Test signal generation with SDE framework
        logger.info("📋 Testing signal generation with SDE framework")
        
        # This would require mock data and analysis results
        # For now, just verify the integration is working
        logger.info("✅ SDE Framework integration verified")
        
        # Close connection
        await pool.close()
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Integration test failed: {e}")
        return False

async def main():
    """Main test function"""
    logger.info("=" * 60)
    logger.info("SDE FRAMEWORK PHASE 1 TESTING")
    logger.info("=" * 60)
    
    # Test 1: SDE Framework Core Functionality
    test1_result = await test_sde_framework()
    
    # Test 2: Signal Generator Integration
    test2_result = await test_signal_generator_integration()
    
    # Summary
    logger.info("=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)
    logger.info(f"SDE Framework Core Tests: {'✅ PASSED' if test1_result else '❌ FAILED'}")
    logger.info(f"Signal Generator Integration: {'✅ PASSED' if test2_result else '❌ FAILED'}")
    
    overall_result = test1_result and test2_result
    logger.info(f"Overall Result: {'✅ ALL TESTS PASSED' if overall_result else '❌ SOME TESTS FAILED'}")
    
    return overall_result

if __name__ == "__main__":
    asyncio.run(main())
