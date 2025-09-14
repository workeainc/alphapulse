#!/usr/bin/env python3
"""
Phase 11: System Integration & End-to-End Testing
Comprehensive test of the complete SDE framework and signal generation pipeline
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any
import asyncpg
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FullSystemIntegrationTest:
    """Comprehensive end-to-end integration test for the complete SDE framework"""
    
    def __init__(self):
        self.db_pool = None
        self.test_results = {}
        self.start_time = None
        
    async def setup_database(self):
        """Initialize database connection"""
        try:
            self.db_pool = await asyncpg.create_pool(
                host='localhost',
                port=5432,
                user='postgres',
                password='Emon_@17711',
                database='alphapulse',
                min_size=5,
                max_size=20
            )
            logger.info("✅ Database connection established")
            return True
        except Exception as e:
            logger.error(f"❌ Database connection failed: {e}")
            return False
    
    async def test_phase1_basic_sde_framework(self):
        """Test Phase 1: Basic SDE Framework"""
        logger.info("🧪 Testing Phase 1: Basic SDE Framework")
        
        try:
            from ai.sde_framework import SDEFramework
            
            # Initialize SDE Framework
            sde = SDEFramework(self.db_pool)
            await sde.load_configurations()
            
            # Test model consensus
            model_heads = [
                {'model_name': 'catboost_technical', 'probability': 0.75, 'confidence': 0.8},
                {'model_name': 'logistic_sentiment', 'probability': 0.65, 'confidence': 0.7},
                {'model_name': 'decision_tree_orderflow', 'probability': 0.85, 'confidence': 0.9}
            ]
            
            consensus_result = await sde.check_model_consensus(model_heads)
            
            # Test confluence scoring
            confluence_result = await sde.calculate_confluence_score(
                technical_score=0.8,
                sentiment_score=0.7,
                orderflow_score=0.9,
                market_regime='trending'
            )
            
            # Test execution quality
            execution_result = await sde.assess_execution_quality(
                signal_strength=0.85,
                market_volatility=0.3,
                liquidity_score=0.8,
                news_impact=0.1
            )
            
            logger.info(f"✅ Phase 1 - Consensus: {consensus_result.consensus_score:.3f}")
            logger.info(f"✅ Phase 1 - Confluence: {confluence_result.confluence_score:.3f}")
            logger.info(f"✅ Phase 1 - Execution: {execution_result.execution_quality_score:.3f}")
            
            self.test_results['phase1'] = {
                'status': 'PASSED',
                'consensus_score': consensus_result.consensus_score,
                'confluence_score': confluence_result.confluence_score,
                'execution_quality': execution_result.execution_quality_score
            }
            
        except Exception as e:
            logger.error(f"❌ Phase 1 failed: {e}")
            self.test_results['phase1'] = {'status': 'FAILED', 'error': str(e)}
    
    async def test_phase2_enhanced_execution_quality(self):
        """Test Phase 2: Enhanced Execution Quality"""
        logger.info("🧪 Testing Phase 2: Enhanced Execution Quality")
        
        try:
            from ai.sde_framework import SDEFramework
            
            sde = SDEFramework(self.db_pool)
            await sde.load_configurations()
            
            # Test news blackout
            news_result = await sde.check_news_blackout(
                symbol='EURUSD',
                current_time=datetime.now()
            )
            
            # Test signal limits
            limits_result = await sde.check_signal_limits(
                symbol='EURUSD',
                signal_type='BUY',
                current_time=datetime.now()
            )
            
            # Test TP structure
            tp_result = await sde.calculate_tp_structure(
                entry_price=1.1000,
                signal_strength=0.85,
                market_volatility=0.3,
                risk_reward_ratio=2.0
            )
            
            logger.info(f"✅ Phase 2 - News Blackout: {news_result.is_blackout}")
            logger.info(f"✅ Phase 2 - Signal Limits: {limits_result.is_allowed}")
            logger.info(f"✅ Phase 2 - TP Levels: {len(tp_result.tp_levels)}")
            
            self.test_results['phase2'] = {
                'status': 'PASSED',
                'news_blackout': news_result.is_blackout,
                'signal_limits': limits_result.is_allowed,
                'tp_levels': len(tp_result.tp_levels)
            }
            
        except Exception as e:
            logger.error(f"❌ Phase 2 failed: {e}")
            self.test_results['phase2'] = {'status': 'FAILED', 'error': str(e)}
    
    async def test_phase3_divergence_analysis(self):
        """Test Phase 3: Divergence Analysis"""
        logger.info("🧪 Testing Phase 3: Divergence Analysis")
        
        try:
            from ai.divergence_analyzer import AdvancedDivergenceAnalyzer
            
            analyzer = AdvancedDivergenceAnalyzer(self.db_pool)
            
            # Create mock price data
            dates = pd.date_range(start='2024-01-01', periods=100, freq='1h')
            prices = np.random.randn(100).cumsum() + 100
            volumes = np.random.randint(1000, 10000, 100)
            
            price_data = pd.DataFrame({
                'timestamp': dates,
                'close': prices,
                'volume': volumes
            })
            
            # Test RSI divergence
            rsi_analysis = await analyzer.detect_rsi_divergence(
                price_data=price_data,
                lookback_periods=20,
                rsi_period=14
            )
            
            # Test MACD divergence
            macd_analysis = await analyzer.detect_macd_divergence(
                price_data=price_data,
                lookback_periods=20
            )
            
            # Test Volume divergence
            volume_analysis = await analyzer.detect_volume_divergence(
                price_data=price_data,
                lookback_periods=20
            )
            
            logger.info(f"✅ Phase 3 - RSI Signals: {len(rsi_analysis.signals)}")
            logger.info(f"✅ Phase 3 - MACD Signals: {len(macd_analysis.signals)}")
            logger.info(f"✅ Phase 3 - Volume Signals: {len(volume_analysis.signals)}")
            
            self.test_results['phase3'] = {
                'status': 'PASSED',
                'rsi_signals': len(rsi_analysis.signals),
                'macd_signals': len(macd_analysis.signals),
                'volume_signals': len(volume_analysis.signals)
            }
            
        except Exception as e:
            logger.error(f"❌ Phase 3 failed: {e}")
            self.test_results['phase3'] = {'status': 'FAILED', 'error': str(e)}
    
    async def test_phase4_sde_integration(self):
        """Test Phase 4: SDE Integration Manager"""
        logger.info("🧪 Testing Phase 4: SDE Integration Manager")
        
        try:
            from ai.sde_integration_manager import SDEIntegrationManager
            
            integration_manager = SDEIntegrationManager(self.db_pool)
            
            # Create mock signal data
            signal_data = {
                'symbol': 'EURUSD',
                'timestamp': datetime.now(),
                'signal_type': 'BUY',
                'confidence': 0.85,
                'price': 1.1000,
                'volume': 1000,
                'technical_indicators': {
                    'rsi': 65.5,
                    'macd': 0.002,
                    'bollinger_upper': 1.1050,
                    'bollinger_lower': 1.0950
                }
            }
            
            # Test SDE integration
            integrated_signal = await integration_manager.integrate_sde_with_signal(signal_data)
            
            logger.info(f"✅ Phase 4 - Integration Success: {integrated_signal is not None}")
            logger.info(f"✅ Phase 4 - Final Confidence: {integrated_signal.final_confidence:.3f}")
            
            self.test_results['phase4'] = {
                'status': 'PASSED',
                'integration_success': integrated_signal is not None,
                'final_confidence': integrated_signal.final_confidence if integrated_signal else 0
            }
            
        except Exception as e:
            logger.error(f"❌ Phase 4 failed: {e}")
            self.test_results['phase4'] = {'status': 'FAILED', 'error': str(e)}
    
    async def test_phase5_enhanced_model_heads(self):
        """Test Phase 5: Enhanced Model Heads"""
        logger.info("🧪 Testing Phase 5: Enhanced Model Heads")
        
        try:
            from ai.sde_framework import SDEFramework
            
            sde = SDEFramework(self.db_pool)
            
            # Test enhanced model head creation
            enhanced_heads = await sde.create_enhanced_model_head_results(
                symbol='EURUSD',
                timestamp=datetime.now(),
                market_data={
                    'price': 1.1000,
                    'volume': 1000,
                    'rsi': 65.5,
                    'macd': 0.002
                }
            )
            
            logger.info(f"✅ Phase 5 - Enhanced Heads Created: {len(enhanced_heads)}")
            
            # Test consensus with enhanced heads
            consensus_result = await sde.check_model_consensus(enhanced_heads)
            
            logger.info(f"✅ Phase 5 - Enhanced Consensus: {consensus_result.consensus_score:.3f}")
            
            self.test_results['phase5'] = {
                'status': 'PASSED',
                'enhanced_heads': len(enhanced_heads),
                'enhanced_consensus': consensus_result.consensus_score
            }
            
        except Exception as e:
            logger.error(f"❌ Phase 5 failed: {e}")
            self.test_results['phase5'] = {'status': 'FAILED', 'error': str(e)}
    
    async def test_phase6_advanced_feature_engineering(self):
        """Test Phase 6: Advanced Feature Engineering"""
        logger.info("🧪 Testing Phase 6: Advanced Feature Engineering")
        
        try:
            from ai.advanced_feature_engineering import AdvancedFeatureEngineering
            
            feature_engineer = AdvancedFeatureEngineering({'db_pool': self.db_pool})
            
            # Test multitimeframe features
            mtf_features = await feature_engineer.create_multitimeframe_features(
                symbol='EURUSD',
                base_timeframe='1h',
                timeframes=['5m', '15m', '4h', '1d']
            )
            
            # Test market regime features
            regime_features = await feature_engineer.create_market_regime_features(
                symbol='EURUSD',
                lookback_days=30
            )
            
            # Test news sentiment features
            sentiment_features = await feature_engineer.create_news_sentiment_features(
                symbol='EURUSD',
                hours_back=24
            )
            
            # Test volume profile features
            volume_features = await feature_engineer.create_volume_profile_features(
                symbol='EURUSD',
                lookback_days=7
            )
            
            logger.info(f"✅ Phase 6 - MTF Features: {len(mtf_features) if mtf_features else 0}")
            logger.info(f"✅ Phase 6 - Regime Features: {len(regime_features) if regime_features else 0}")
            logger.info(f"✅ Phase 6 - Sentiment Features: {len(sentiment_features) if sentiment_features else 0}")
            logger.info(f"✅ Phase 6 - Volume Features: {len(volume_features) if volume_features else 0}")
            
            self.test_results['phase6'] = {
                'status': 'PASSED',
                'mtf_features': len(mtf_features) if mtf_features else 0,
                'regime_features': len(regime_features) if regime_features else 0,
                'sentiment_features': len(sentiment_features) if sentiment_features else 0,
                'volume_features': len(volume_features) if volume_features else 0
            }
            
        except Exception as e:
            logger.error(f"❌ Phase 6 failed: {e}")
            self.test_results['phase6'] = {'status': 'FAILED', 'error': str(e)}
    
    async def test_phase7_advanced_model_fusion(self):
        """Test Phase 7: Advanced Model Fusion"""
        logger.info("🧪 Testing Phase 7: Advanced Model Fusion")
        
        try:
            from ai.advanced_model_fusion import AdvancedModelFusion
            
            fusion = AdvancedModelFusion(self.db_pool)
            
            # Create mock predictions
            predictions = [
                {'model_name': 'catboost', 'probability': 0.75, 'confidence': 0.8},
                {'model_name': 'logistic', 'probability': 0.65, 'confidence': 0.7},
                {'model_name': 'decision_tree', 'probability': 0.85, 'confidence': 0.9}
            ]
            
            # Test weighted average fusion
            weighted_result = await fusion.fuse_predictions(
                predictions=predictions,
                method='weighted_average',
                weights={'catboost': 0.4, 'logistic': 0.3, 'decision_tree': 0.3}
            )
            
            # Test voting fusion
            voting_result = await fusion.fuse_predictions(
                predictions=predictions,
                method='voting'
            )
            
            logger.info(f"✅ Phase 7 - Weighted Fusion: {weighted_result.final_probability:.3f}")
            logger.info(f"✅ Phase 7 - Voting Fusion: {voting_result.final_probability:.3f}")
            
            self.test_results['phase7'] = {
                'status': 'PASSED',
                'weighted_fusion': weighted_result.final_probability,
                'voting_fusion': voting_result.final_probability
            }
            
        except Exception as e:
            logger.error(f"❌ Phase 7 failed: {e}")
            self.test_results['phase7'] = {'status': 'FAILED', 'error': str(e)}
    
    async def test_phase8_advanced_calibration(self):
        """Test Phase 8: Advanced Calibration"""
        logger.info("🧪 Testing Phase 8: Advanced Calibration")
        
        try:
            from ai.advanced_calibration_system import AdvancedCalibrationSystem
            
            calibration = AdvancedCalibrationSystem(self.db_pool)
            
            # Test different calibration methods
            raw_probability = 0.75
            
            # Isotonic calibration
            isotonic_result = await calibration.calibrate_probability(
                raw_probability=raw_probability,
                method='isotonic',
                symbol='EURUSD'
            )
            
            # Platt calibration
            platt_result = await calibration.calibrate_probability(
                raw_probability=raw_probability,
                method='platt',
                symbol='EURUSD'
            )
            
            # Temperature scaling
            temp_result = await calibration.calibrate_probability(
                raw_probability=raw_probability,
                method='temperature',
                symbol='EURUSD'
            )
            
            logger.info(f"✅ Phase 8 - Isotonic: {isotonic_result.calibrated_probability:.3f}")
            logger.info(f"✅ Phase 8 - Platt: {platt_result.calibrated_probability:.3f}")
            logger.info(f"✅ Phase 8 - Temperature: {temp_result.calibrated_probability:.3f}")
            
            self.test_results['phase8'] = {
                'status': 'PASSED',
                'isotonic': isotonic_result.calibrated_probability,
                'platt': platt_result.calibrated_probability,
                'temperature': temp_result.calibrated_probability
            }
            
        except Exception as e:
            logger.error(f"❌ Phase 8 failed: {e}")
            self.test_results['phase8'] = {'status': 'FAILED', 'error': str(e)}
    
    async def test_phase9_signal_quality_validation(self):
        """Test Phase 9: Signal Quality Validation"""
        logger.info("🧪 Testing Phase 9: Signal Quality Validation")
        
        try:
            from ai.advanced_signal_quality_validator import AdvancedSignalQualityValidator
            
            validator = AdvancedSignalQualityValidator(self.db_pool)
            
            # Test signal quality validation
            signal_data = {
                'symbol': 'EURUSD',
                'signal_type': 'BUY',
                'confidence': 0.85,
                'price': 1.1000,
                'volume': 1000,
                'volatility': 0.3,
                'trend_strength': 0.7,
                'market_regime': 'trending'
            }
            
            quality_result = await validator.validate_signal_quality(signal_data)
            
            logger.info(f"✅ Phase 9 - Quality Score: {quality_result.quality_score:.3f}")
            logger.info(f"✅ Phase 9 - Is Valid: {quality_result.is_valid}")
            logger.info(f"✅ Phase 9 - Market Regime: {quality_result.market_regime}")
            
            self.test_results['phase9'] = {
                'status': 'PASSED',
                'quality_score': quality_result.quality_score,
                'is_valid': quality_result.is_valid,
                'market_regime': quality_result.market_regime
            }
            
        except Exception as e:
            logger.error(f"❌ Phase 9 failed: {e}")
            self.test_results['phase9'] = {'status': 'FAILED', 'error': str(e)}
    
    async def test_phase10_production_monitoring(self):
        """Test Phase 10: Production Monitoring"""
        logger.info("🧪 Testing Phase 10: Production Monitoring")
        
        try:
            from ai.production_monitoring_system import ProductionMonitoringSystem
            
            monitoring = ProductionMonitoringSystem(self.db_pool)
            
            # Start monitoring
            await monitoring.start_monitoring()
            
            # Wait for initial metrics collection
            await asyncio.sleep(2)
            
            # Get system status
            status = await monitoring.get_system_status()
            
            # Stop monitoring
            monitoring.stop_monitoring()
            
            logger.info(f"✅ Phase 10 - System Status: {status.overall_health}")
            logger.info(f"✅ Phase 10 - Active Alerts: {len(status.active_alerts)}")
            logger.info(f"✅ Phase 10 - Services Monitored: {len(status.service_health)}")
            
            self.test_results['phase10'] = {
                'status': 'PASSED',
                'system_health': status.overall_health,
                'active_alerts': len(status.active_alerts),
                'services_monitored': len(status.service_health)
            }
            
        except Exception as e:
            logger.error(f"❌ Phase 10 failed: {e}")
            self.test_results['phase10'] = {'status': 'FAILED', 'error': str(e)}
    
    async def test_end_to_end_signal_generation(self):
        """Test complete end-to-end signal generation workflow"""
        logger.info("🧪 Testing End-to-End Signal Generation")
        
        try:
            from app.signals.intelligent_signal_generator import IntelligentSignalGenerator
            
            # Initialize signal generator
            signal_generator = IntelligentSignalGenerator(self.db_pool)
            
            # Create mock market data
            market_data = {
                'symbol': 'EURUSD',
                'timestamp': datetime.now(),
                'open': 1.1000,
                'high': 1.1020,
                'low': 1.0980,
                'close': 1.1010,
                'volume': 1000,
                'technical_indicators': {
                    'rsi': 65.5,
                    'macd': 0.002,
                    'bollinger_upper': 1.1050,
                    'bollinger_lower': 1.0950,
                    'sma_20': 1.0990,
                    'sma_50': 1.0985
                }
            }
            
            # Generate signal
            signal = await signal_generator.generate_intelligent_signal(market_data)
            
            logger.info(f"✅ E2E - Signal Generated: {signal is not None}")
            if signal:
                logger.info(f"✅ E2E - Signal Type: {signal.signal_type}")
                logger.info(f"✅ E2E - Confidence: {signal.confidence:.3f}")
                logger.info(f"✅ E2E - Calibrated Confidence: {signal.calibrated_confidence:.3f}")
                logger.info(f"✅ E2E - Quality Metrics: {signal.quality_metrics is not None}")
            
            self.test_results['end_to_end'] = {
                'status': 'PASSED',
                'signal_generated': signal is not None,
                'signal_type': signal.signal_type if signal else None,
                'confidence': signal.confidence if signal else 0,
                'calibrated_confidence': signal.calibrated_confidence if signal else 0,
                'quality_metrics': signal.quality_metrics is not None if signal else False
            }
            
        except Exception as e:
            logger.error(f"❌ End-to-End test failed: {e}")
            self.test_results['end_to_end'] = {'status': 'FAILED', 'error': str(e)}
    
    async def run_all_tests(self):
        """Run all integration tests"""
        logger.info("🚀 Starting Full System Integration Test")
        self.start_time = time.time()
        
        # Setup database
        if not await self.setup_database():
            logger.error("❌ Cannot proceed without database connection")
            return
        
        # Run all phase tests
        test_methods = [
            self.test_phase1_basic_sde_framework,
            self.test_phase2_enhanced_execution_quality,
            self.test_phase3_divergence_analysis,
            self.test_phase4_sde_integration,
            self.test_phase5_enhanced_model_heads,
            self.test_phase6_advanced_feature_engineering,
            self.test_phase7_advanced_model_fusion,
            self.test_phase8_advanced_calibration,
            self.test_phase9_signal_quality_validation,
            self.test_phase10_production_monitoring,
            self.test_end_to_end_signal_generation
        ]
        
        for test_method in test_methods:
            try:
                await test_method()
                await asyncio.sleep(1)  # Brief pause between tests
            except Exception as e:
                logger.error(f"❌ Test {test_method.__name__} failed: {e}")
        
        # Generate test report
        await self.generate_test_report()
        
        # Cleanup
        if self.db_pool:
            await self.db_pool.close()
    
    async def generate_test_report(self):
        """Generate comprehensive test report"""
        logger.info("\n" + "="*80)
        logger.info("📊 FULL SYSTEM INTEGRATION TEST REPORT")
        logger.info("="*80)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result.get('status') == 'PASSED')
        failed_tests = total_tests - passed_tests
        
        logger.info(f"📈 Test Summary:")
        logger.info(f"   Total Tests: {total_tests}")
        logger.info(f"   Passed: {passed_tests}")
        logger.info(f"   Failed: {failed_tests}")
        logger.info(f"   Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        
        if self.start_time:
            duration = time.time() - self.start_time
            logger.info(f"   Duration: {duration:.2f} seconds")
        
        logger.info(f"\n📋 Detailed Results:")
        for test_name, result in self.test_results.items():
            status = result.get('status', 'UNKNOWN')
            status_icon = "✅" if status == 'PASSED' else "❌"
            logger.info(f"   {status_icon} {test_name}: {status}")
            
            if status == 'FAILED' and 'error' in result:
                logger.info(f"      Error: {result['error']}")
            elif status == 'PASSED':
                # Log key metrics for passed tests
                for key, value in result.items():
                    if key != 'status' and isinstance(value, (int, float)):
                        logger.info(f"      {key}: {value}")
        
        # System readiness assessment
        logger.info(f"\n🎯 System Readiness Assessment:")
        if passed_tests == total_tests:
            logger.info("   🟢 FULLY OPERATIONAL - All systems integrated and functional")
            logger.info("   🚀 Ready for production deployment")
        elif passed_tests >= total_tests * 0.8:
            logger.info("   🟡 MOSTLY OPERATIONAL - Minor issues detected")
            logger.info("   ⚠️  Review failed tests before production deployment")
        else:
            logger.info("   🔴 CRITICAL ISSUES - Multiple system failures")
            logger.info("   🛑 Requires immediate attention before deployment")
        
        logger.info("="*80)

async def main():
    """Main test execution"""
    test_suite = FullSystemIntegrationTest()
    await test_suite.run_all_tests()

if __name__ == "__main__":
    asyncio.run(main())

