#!/usr/bin/env python3
"""
Test Script for ML + Risk Integration System
Validates the integration of ML predictions with risk management for actionable signals
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
import os
from typing import Dict, Any

# Import the ML + Risk Integration service
from app.services.ml_risk_integration_service import MLRiskIntegrationService, ActionableTradeSignal, SignalStrength, RiskLevel, MarketRegime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MLRiskIntegrationTester:
    """Test class for ML + Risk Integration system"""
    
    def __init__(self):
        self.service = None
        self.test_results = {
            'total_tests': 0,
            'passed_tests': 0,
            'failed_tests': 0,
            'test_details': []
        }
    
    async def setup(self):
        """Setup the test environment"""
        logger.info("🚀 Setting up ML + Risk Integration test environment...")
        
        try:
            # Initialize the service
            config = {
                'database_url': os.getenv("DATABASE_URL", "postgresql://alpha_emon:Emon_%4017711@localhost:5432/alphapulse")
            }
            
            self.service = MLRiskIntegrationService(config)
            await self.service.initialize()
            
            logger.info("✅ Test environment setup completed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Test environment setup failed: {e}")
            return False
    
    async def teardown(self):
        """Cleanup test environment"""
        logger.info("🧹 Cleaning up test environment...")
        
        try:
            if self.service:
                await self.service.stop()
            logger.info("✅ Test environment cleanup completed")
        except Exception as e:
            logger.error(f"❌ Test environment cleanup failed: {e}")
    
    def generate_test_market_data(self, symbol: str = "BTC/USDT") -> Dict[str, Any]:
        """Generate realistic test market data"""
        try:
            # Generate price data (last 100 points)
            base_price = 50000.0
            price_changes = np.random.normal(0, 0.02, 100)  # 2% daily volatility
            prices = [base_price]
            
            for change in price_changes:
                new_price = prices[-1] * (1 + change)
                prices.append(new_price)
            
            # Generate volume data
            base_volume = 1000000.0
            volume_changes = np.random.normal(0, 0.3, 100)  # 30% volume volatility
            volumes = [base_volume]
            
            for change in volume_changes:
                new_volume = volumes[-1] * (1 + change)
                volumes.append(max(new_volume, 100000))  # Minimum volume
            
            # Generate OHLCV data
            ohlcv_data = []
            for i in range(len(prices) - 1):
                open_price = prices[i]
                close_price = prices[i + 1]
                high_price = max(open_price, close_price) * (1 + abs(np.random.normal(0, 0.01)))
                low_price = min(open_price, close_price) * (1 - abs(np.random.normal(0, 0.01)))
                volume = volumes[i]
                
                ohlcv_data.append({
                    'timestamp': datetime.now() - timedelta(minutes=len(prices) - i),
                    'open': open_price,
                    'high': high_price,
                    'low': low_price,
                    'close': close_price,
                    'volume': volume
                })
            
            market_data = {
                'symbol': symbol,
                'close_prices': prices,
                'volumes': volumes,
                'ohlcv_data': ohlcv_data,
                'current_price': prices[-1],
                'current_volume': volumes[-1],
                'price_change_24h': (prices[-1] - prices[0]) / prices[0],
                'volume_change_24h': (volumes[-1] - volumes[0]) / volumes[0]
            }
            
            return market_data
            
        except Exception as e:
            logger.error(f"Error generating test market data: {e}")
            return {}
    
    async def test_service_initialization(self):
        """Test service initialization"""
        test_name = "Service Initialization"
        logger.info(f"🧪 Testing {test_name}...")
        
        try:
            self.test_results['total_tests'] += 1
            
            # Check if service is properly initialized
            assert self.service is not None, "Service should be initialized"
            assert self.service.ensemble_service is not None, "Ensemble service should be initialized"
            assert self.service.risk_manager is not None, "Risk manager should be initialized"
            # Monitoring service is optional
            
            logger.info(f"✅ {test_name} PASSED")
            self.test_results['passed_tests'] += 1
            self.test_results['test_details'].append({
                'test': test_name,
                'status': 'PASSED',
                'details': 'Service initialized successfully with all components'
            })
            
        except Exception as e:
            logger.error(f"❌ {test_name} FAILED: {e}")
            self.test_results['failed_tests'] += 1
            self.test_results['test_details'].append({
                'test': test_name,
                'status': 'FAILED',
                'details': str(e)
            })
    
    async def test_actionable_signal_generation(self):
        """Test actionable signal generation"""
        test_name = "Actionable Signal Generation"
        logger.info(f"🧪 Testing {test_name}...")
        
        try:
            self.test_results['total_tests'] += 1
            
            # Generate test market data
            market_data = self.generate_test_market_data("BTC/USDT")
            
            # Generate actionable signal
            signal = await self.service.generate_actionable_signal("BTC/USDT", market_data)
            
            # Validate signal structure
            assert isinstance(signal, ActionableTradeSignal), "Should return ActionableTradeSignal"
            assert signal.symbol == "BTC/USDT", "Symbol should match"
            assert isinstance(signal.timestamp, datetime), "Timestamp should be datetime"
            assert signal.signal_type in ['long', 'short', 'hold'], "Signal type should be valid"
            assert isinstance(signal.signal_strength, SignalStrength), "Signal strength should be enum"
            assert 0.0 <= signal.confidence_score <= 1.0, "Confidence score should be 0-1"
            assert isinstance(signal.risk_level, RiskLevel), "Risk level should be enum"
            assert isinstance(signal.market_regime, MarketRegime), "Market regime should be enum"
            assert signal.recommended_leverage >= 1, "Leverage should be >= 1"
            assert signal.position_size_usdt >= 0.0, "Position size should be >= 0"
            assert signal.stop_loss_price > 0.0, "Stop loss should be > 0"
            assert signal.take_profit_price > 0.0, "Take profit should be > 0"
            assert signal.risk_reward_ratio > 0.0, "Risk-reward ratio should be > 0"
            assert 0.0 <= signal.risk_score <= 100.0, "Risk score should be 0-100"
            assert 0.0 <= signal.liquidation_risk <= 100.0, "Liquidation risk should be 0-100"
            assert 0.0 <= signal.volatility_score <= 1.0, "Volatility score should be 0-1"
            assert 0.0 <= signal.liquidity_score <= 1.0, "Liquidity score should be 0-1"
            
            logger.info(f"✅ {test_name} PASSED")
            logger.info(f"   Generated signal: {signal.signal_type} ({signal.signal_strength.value})")
            logger.info(f"   Confidence: {signal.confidence_score:.3f}")
            logger.info(f"   Risk Level: {signal.risk_level.value}")
            logger.info(f"   Market Regime: {signal.market_regime.value}")
            logger.info(f"   Leverage: {signal.recommended_leverage}x")
            logger.info(f"   Position Size: ${signal.position_size_usdt:.2f}")
            logger.info(f"   Risk-Reward: {signal.risk_reward_ratio:.2f}")
            
            self.test_results['passed_tests'] += 1
            self.test_results['test_details'].append({
                'test': test_name,
                'status': 'PASSED',
                'details': f'Generated valid signal: {signal.signal_type} with {signal.confidence_score:.3f} confidence'
            })
            
        except Exception as e:
            logger.error(f"❌ {test_name} FAILED: {e}")
            self.test_results['failed_tests'] += 1
            self.test_results['test_details'].append({
                'test': test_name,
                'status': 'FAILED',
                'details': str(e)
            })
    
    async def test_multiple_symbols(self):
        """Test signal generation for multiple symbols"""
        test_name = "Multiple Symbols Test"
        logger.info(f"🧪 Testing {test_name}...")
        
        try:
            self.test_results['total_tests'] += 1
            
            symbols = ["BTC/USDT", "ETH/USDT", "ADA/USDT"]
            signals = []
            
            for symbol in symbols:
                market_data = self.generate_test_market_data(symbol)
                signal = await self.service.generate_actionable_signal(symbol, market_data)
                signals.append(signal)
                
                # Validate each signal
                assert signal.symbol == symbol, f"Symbol should match for {symbol}"
                assert signal.signal_type in ['long', 'short', 'hold'], f"Valid signal type for {symbol}"
            
            logger.info(f"✅ {test_name} PASSED")
            logger.info(f"   Generated signals for {len(symbols)} symbols")
            for signal in signals:
                logger.info(f"   {signal.symbol}: {signal.signal_type} ({signal.confidence_score:.3f})")
            
            self.test_results['passed_tests'] += 1
            self.test_results['test_details'].append({
                'test': test_name,
                'status': 'PASSED',
                'details': f'Generated signals for {len(symbols)} symbols successfully'
            })
            
        except Exception as e:
            logger.error(f"❌ {test_name} FAILED: {e}")
            self.test_results['failed_tests'] += 1
            self.test_results['test_details'].append({
                'test': test_name,
                'status': 'FAILED',
                'details': str(e)
            })
    
    async def test_risk_analysis(self):
        """Test risk analysis functionality"""
        test_name = "Risk Analysis"
        logger.info(f"🧪 Testing {test_name}...")
        
        try:
            self.test_results['total_tests'] += 1
            
            market_data = self.generate_test_market_data("BTC/USDT")
            
            # Test risk analysis
            risk_analysis = await self.service._analyze_risk("BTC/USDT", market_data, None)
            
            # Validate risk analysis structure
            assert isinstance(risk_analysis, dict), "Risk analysis should be a dictionary"
            assert 'portfolio_metrics' in risk_analysis, "Should contain portfolio metrics"
            assert 'liquidation_risk' in risk_analysis, "Should contain liquidation risk"
            assert 'dynamic_leverage' in risk_analysis, "Should contain dynamic leverage"
            assert 'volatility_score' in risk_analysis, "Should contain volatility score"
            assert 'liquidity_score' in risk_analysis, "Should contain liquidity score"
            assert 'overall_risk_score' in risk_analysis, "Should contain overall risk score"
            
            # Validate risk scores
            assert 0.0 <= risk_analysis['liquidation_risk'] <= 100.0, "Liquidation risk should be 0-100"
            assert risk_analysis['dynamic_leverage'] >= 1, "Dynamic leverage should be >= 1"
            assert 0.0 <= risk_analysis['volatility_score'] <= 1.0, "Volatility score should be 0-1"
            assert 0.0 <= risk_analysis['liquidity_score'] <= 1.0, "Liquidity score should be 0-1"
            assert 0.0 <= risk_analysis['overall_risk_score'] <= 100.0, "Overall risk score should be 0-100"
            
            logger.info(f"✅ {test_name} PASSED")
            logger.info(f"   Liquidation Risk: {risk_analysis['liquidation_risk']:.2f}")
            logger.info(f"   Dynamic Leverage: {risk_analysis['dynamic_leverage']}x")
            logger.info(f"   Volatility Score: {risk_analysis['volatility_score']:.3f}")
            logger.info(f"   Liquidity Score: {risk_analysis['liquidity_score']:.3f}")
            logger.info(f"   Overall Risk Score: {risk_analysis['overall_risk_score']:.2f}")
            
            self.test_results['passed_tests'] += 1
            self.test_results['test_details'].append({
                'test': test_name,
                'status': 'PASSED',
                'details': f'Risk analysis completed with overall risk score: {risk_analysis["overall_risk_score"]:.2f}'
            })
            
        except Exception as e:
            logger.error(f"❌ {test_name} FAILED: {e}")
            self.test_results['failed_tests'] += 1
            self.test_results['test_details'].append({
                'test': test_name,
                'status': 'FAILED',
                'details': str(e)
            })
    
    async def test_market_regime_detection(self):
        """Test market regime detection"""
        test_name = "Market Regime Detection"
        logger.info(f"🧪 Testing {test_name}...")
        
        try:
            self.test_results['total_tests'] += 1
            
            # Test different market scenarios
            scenarios = [
                ("trending_up", [50000 + i * 100 for i in range(100)]),  # Upward trend
                ("trending_down", [50000 - i * 100 for i in range(100)]),  # Downward trend
                ("ranging", [50000 + np.sin(i/10) * 1000 for i in range(100)]),  # Ranging
                ("volatile", [50000 + np.random.normal(0, 2000) for i in range(100)])  # Volatile
            ]
            
            for scenario_name, prices in scenarios:
                market_data = {
                    'close_prices': prices,
                    'volumes': [1000000 + np.random.normal(0, 100000) for _ in range(len(prices))]
                }
                
                # Mock ensemble prediction
                class MockEnsemblePrediction:
                    def __init__(self):
                        self.unified_signal = 'hold'
                        self.risk_level = 'medium'
                
                regime = await self.service._determine_market_regime("BTC/USDT", market_data, MockEnsemblePrediction())
                
                assert isinstance(regime, MarketRegime), f"Regime should be MarketRegime enum for {scenario_name}"
                logger.info(f"   {scenario_name}: {regime.value}")
            
            logger.info(f"✅ {test_name} PASSED")
            
            self.test_results['passed_tests'] += 1
            self.test_results['test_details'].append({
                'test': test_name,
                'status': 'PASSED',
                'details': 'Market regime detection working for all scenarios'
            })
            
        except Exception as e:
            logger.error(f"❌ {test_name} FAILED: {e}")
            self.test_results['failed_tests'] += 1
            self.test_results['test_details'].append({
                'test': test_name,
                'status': 'FAILED',
                'details': str(e)
            })
    
    async def test_position_sizing(self):
        """Test position sizing calculations"""
        test_name = "Position Sizing"
        logger.info(f"🧪 Testing {test_name}...")
        
        try:
            self.test_results['total_tests'] += 1
            
            # Mock ensemble prediction
            class MockEnsemblePrediction:
                def __init__(self, signal='buy', confidence=0.8):
                    self.unified_signal = signal
                    self.confidence_score = confidence
            
            # Mock risk analysis
            risk_analysis = {
                'overall_risk_score': 30.0,
                'dynamic_leverage': 2,
                'volatility_score': 0.6,
                'liquidity_score': 0.8
            }
            
            # Test different scenarios
            scenarios = [
                (MarketRegime.TRENDING_UP, MockEnsemblePrediction('buy', 0.9)),
                (MarketRegime.RANGING, MockEnsemblePrediction('hold', 0.6)),
                (MarketRegime.VOLATILE, MockEnsemblePrediction('sell', 0.7))
            ]
            
            for regime, prediction in scenarios:
                sizing = await self.service._calculate_position_sizing("BTC/USDT", prediction, risk_analysis, regime)
                
                assert isinstance(sizing, dict), "Position sizing should return dictionary"
                assert 'leverage' in sizing, "Should contain leverage"
                assert 'position_size' in sizing, "Should contain position size"
                assert 'stop_loss' in sizing, "Should contain stop loss"
                assert 'take_profit' in sizing, "Should contain take profit"
                assert 'risk_reward_ratio' in sizing, "Should contain risk-reward ratio"
                
                assert sizing['leverage'] >= 1, "Leverage should be >= 1"
                assert sizing['position_size'] >= 0.0, "Position size should be >= 0"
                assert sizing['stop_loss'] > 0.0, "Stop loss should be > 0"
                assert sizing['take_profit'] > 0.0, "Take profit should be > 0"
                assert sizing['risk_reward_ratio'] > 0.0, "Risk-reward ratio should be > 0"
                
                logger.info(f"   {regime.value}: {sizing['leverage']}x leverage, ${sizing['position_size']:.2f} size")
            
            logger.info(f"✅ {test_name} PASSED")
            
            self.test_results['passed_tests'] += 1
            self.test_results['test_details'].append({
                'test': test_name,
                'status': 'PASSED',
                'details': 'Position sizing calculations working correctly'
            })
            
        except Exception as e:
            logger.error(f"❌ {test_name} FAILED: {e}")
            self.test_results['failed_tests'] += 1
            self.test_results['test_details'].append({
                'test': test_name,
                'status': 'FAILED',
                'details': str(e)
            })
    
    async def test_performance_metrics(self):
        """Test performance metrics tracking"""
        test_name = "Performance Metrics"
        logger.info(f"🧪 Testing {test_name}...")
        
        try:
            self.test_results['total_tests'] += 1
            
            # Get initial metrics
            initial_metrics = self.service.performance_metrics.copy()
            
            # Generate a signal to update metrics
            market_data = self.generate_test_market_data("BTC/USDT")
            signal = await self.service.generate_actionable_signal("BTC/USDT", market_data)
            
            # Get updated metrics
            updated_metrics = self.service.performance_metrics
            
            # Validate metrics update
            assert updated_metrics['signals_generated'] > initial_metrics['signals_generated'], "Signals generated should increase"
            assert updated_metrics['average_confidence'] >= 0.0, "Average confidence should be >= 0"
            assert updated_metrics['average_risk_score'] >= 0.0, "Average risk score should be >= 0"
            
            # Get performance summary
            summary = await self.service.get_performance_summary()
            assert isinstance(summary, dict), "Performance summary should be dictionary"
            assert 'performance_metrics' in summary, "Should contain performance metrics"
            assert 'integration_params' in summary, "Should contain integration parameters"
            assert 'signal_thresholds' in summary, "Should contain signal thresholds"
            
            logger.info(f"✅ {test_name} PASSED")
            logger.info(f"   Signals Generated: {updated_metrics['signals_generated']}")
            logger.info(f"   Average Confidence: {updated_metrics['average_confidence']:.3f}")
            logger.info(f"   Average Risk Score: {updated_metrics['average_risk_score']:.2f}")
            
            self.test_results['passed_tests'] += 1
            self.test_results['test_details'].append({
                'test': test_name,
                'status': 'PASSED',
                'details': f'Performance metrics tracking working: {updated_metrics["signals_generated"]} signals generated'
            })
            
        except Exception as e:
            logger.error(f"❌ {test_name} FAILED: {e}")
            self.test_results['failed_tests'] += 1
            self.test_results['test_details'].append({
                'test': test_name,
                'status': 'FAILED',
                'details': str(e)
            })
    
    async def run_all_tests(self):
        """Run all tests"""
        logger.info("🚀 Starting ML + Risk Integration System Tests...")
        
        # Setup
        if not await self.setup():
            logger.error("❌ Test setup failed, aborting tests")
            return
        
        try:
            # Run all tests
            await self.test_service_initialization()
            await self.test_actionable_signal_generation()
            await self.test_multiple_symbols()
            await self.test_risk_analysis()
            await self.test_market_regime_detection()
            await self.test_position_sizing()
            await self.test_performance_metrics()
            
        finally:
            # Teardown
            await self.teardown()
        
        # Print results
        self.print_results()
    
    def print_results(self):
        """Print test results"""
        logger.info("\n" + "="*60)
        logger.info("📊 ML + Risk Integration System Test Results")
        logger.info("="*60)
        
        total = self.test_results['total_tests']
        passed = self.test_results['passed_tests']
        failed = self.test_results['failed_tests']
        
        logger.info(f"Total Tests: {total}")
        logger.info(f"Passed: {passed} ✅")
        logger.info(f"Failed: {failed} ❌")
        logger.info(f"Success Rate: {(passed/total*100):.1f}%" if total > 0 else "N/A")
        
        logger.info("\n📋 Test Details:")
        for detail in self.test_results['test_details']:
            status_icon = "✅" if detail['status'] == 'PASSED' else "❌"
            logger.info(f"  {status_icon} {detail['test']}: {detail['details']}")
        
        logger.info("\n" + "="*60)
        
        # Save results to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"ml_risk_integration_test_results_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(self.test_results, f, indent=2, default=str)
        
        logger.info(f"📄 Detailed results saved to: {results_file}")

async def main():
    """Main test runner"""
    tester = MLRiskIntegrationTester()
    await tester.run_all_tests()

if __name__ == "__main__":
    asyncio.run(main())
