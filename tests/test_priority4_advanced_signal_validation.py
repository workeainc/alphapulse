#!/usr/bin/env python3
"""
Test script for Priority 4: Advanced Signal Validation

Tests core functionality for advanced signal validation and quality scoring.
"""

import asyncio
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any
import warnings
warnings.filterwarnings('ignore')

# Import Priority 4 system
from ..ai.priority4_advanced_signal_validation import (
    Priority4AdvancedSignalValidation,
    SignalQualityLevel,
    ValidationResult
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Priority4AdvancedSignalValidationTester:
    """Test suite for Priority 4 Advanced Signal Validation"""
    
    def __init__(self):
        self.priority4_system = Priority4AdvancedSignalValidation(
            enable_adaptive_thresholds=True
        )
        
    def generate_test_data(self, n_samples: int = 500) -> pd.DataFrame:
        """Generate synthetic test data with realistic market patterns"""
        logger.info(f"Generating {n_samples} synthetic test samples...")
        
        # Generate realistic price data
        np.random.seed(42)
        
        # Base price trend
        base_price = 50000 + np.cumsum(np.random.normal(0, 100, n_samples))
        
        # Add market cycles
        cycle = 50 * np.sin(np.arange(n_samples) * 2 * np.pi / 200)
        base_price += cycle
        
        # Generate OHLCV data
        data = pd.DataFrame({
            'open': base_price + np.random.normal(0, 50, n_samples),
            'high': base_price + np.random.normal(50, 100, n_samples),
            'low': base_price + np.random.normal(-50, 100, n_samples),
            'close': base_price + np.random.normal(0, 50, n_samples),
            'volume': np.random.lognormal(10, 1, n_samples)
        })
        
        # Ensure OHLC consistency
        data['high'] = np.maximum(data['high'], np.maximum(data['open'], data['close']))
        data['low'] = np.minimum(data['low'], np.minimum(data['open'], data['close']))
        
        # Clean up NaN values
        data = data.dropna()
        
        logger.info(f"✅ Generated test data with {len(data)} samples")
        return data
    
    def generate_test_signals(self, n_signals: int = 5) -> List[Dict[str, Any]]:
        """Generate test trading signals with various quality levels"""
        signals = []
        
        signal_types = ['buy', 'sell', 'mean_reversion', 'breakout', 'momentum']
        strategies = ['trend_following', 'mean_reversion', 'breakout_detection']
        timeframes = ['1m', '5m', '15m', '1h', '4h']
        
        for i in range(n_signals):
            # Create more varied signal characteristics
            if i == 0:
                # High quality signal
                signal_strength = np.random.uniform(0.8, 0.95)
                historical_accuracy = np.random.uniform(0.8, 0.95)
                pattern_confidence = np.random.uniform(0.8, 0.95)
            elif i == 1:
                # Medium quality signal
                signal_strength = np.random.uniform(0.5, 0.7)
                historical_accuracy = np.random.uniform(0.5, 0.7)
                pattern_confidence = np.random.uniform(0.5, 0.7)
            else:
                # Low quality signal
                signal_strength = np.random.uniform(0.3, 0.5)
                historical_accuracy = np.random.uniform(0.3, 0.5)
                pattern_confidence = np.random.uniform(0.3, 0.5)
            
            signal = {
                'symbol': 'BTCUSDT',
                'signal_type': np.random.choice(signal_types),
                'strategy_name': np.random.choice(strategies),
                'signal_strength': signal_strength,
                'historical_accuracy': historical_accuracy,
                'pattern_confidence': pattern_confidence,
                'entry_price': 50000 + np.random.uniform(-1000, 1000),
                'stop_loss': 0.0,
                'take_profit': 0.0,
                'contributing_timeframes': np.random.choice(timeframes, size=np.random.randint(1, 4), replace=False).tolist(),
                'timestamp': datetime.now() - timedelta(hours=np.random.randint(0, 24))
            }
            
            # Calculate SL/TP based on signal type
            if signal['signal_type'] in ['buy', 'momentum']:
                signal['stop_loss'] = signal['entry_price'] * 0.98
                signal['take_profit'] = signal['entry_price'] * 1.04
            elif signal['signal_type'] in ['sell']:
                signal['stop_loss'] = signal['entry_price'] * 1.02
                signal['take_profit'] = signal['entry_price'] * 0.96
            else:
                signal['stop_loss'] = signal['entry_price'] * 0.99
                signal['take_profit'] = signal['entry_price'] * 1.02
            
            signals.append(signal)
        
        return signals
    
    async def test_advanced_signal_validation(self) -> bool:
        """Test advanced signal validation system"""
        logger.info("🧪 Testing Advanced Signal Validation...")
        
        try:
            # Generate test data and signals
            test_data = self.generate_test_data(300)
            test_signals = self.generate_test_signals(5)
            
            success_count = 0
            total_signals = len(test_signals)
            
            for i, signal in enumerate(test_signals):
                try:
                    # Validate signal using our actual method
                    validation_result, quality_metrics = await self.priority4_system.validate_signal(
                        signal, test_data
                    )
                    
                    # Validate quality metrics
                    if (quality_metrics and 
                        0.0 <= quality_metrics.overall_quality <= 1.0 and
                        0.0 <= quality_metrics.confidence_score <= 1.0 and
                        0.0 <= quality_metrics.volatility_score <= 1.0 and
                        0.0 <= quality_metrics.trend_strength <= 1.0 and
                        0.0 <= quality_metrics.volume_confirmation <= 1.0 and
                        0.0 <= quality_metrics.market_regime_score <= 1.0):
                        
                        success_count += 1
                        logger.info(f"  ✅ Signal {i+1}: Quality Score: {quality_metrics.overall_quality:.3f}, "
                                  f"Result: {validation_result.value}")
                    else:
                        logger.warning(f"  ⚠️ Signal {i+1}: Invalid quality metrics")
                        
                except Exception as e:
                    logger.error(f"  ❌ Signal {i+1} validation failed: {e}")
            
            logger.info(f"✅ Advanced Signal Validation: {success_count}/{total_signals} signals validated successfully")
            return success_count > 0
            
        except Exception as e:
            logger.error(f"❌ Advanced Signal Validation test failed: {e}")
            return False
    
    async def test_signal_quality_scoring(self) -> bool:
        """Test signal quality scoring system"""
        logger.info("🧪 Testing Signal Quality Scoring...")
        
        try:
            # Generate test data and signals
            test_data = self.generate_test_data(200)
            test_signals = self.generate_test_signals(3)
            
            quality_scores = []
            validation_results = []
            
            for signal in test_signals:
                # Validate signal to get quality metrics
                validation_result, quality_metrics = await self.priority4_system.validate_signal(
                    signal, test_data
                )
                
                if quality_metrics:
                    quality_scores.append(quality_metrics.overall_quality)
                    validation_results.append(validation_result.value)
                    
                    # Check quality level assignment based on overall quality
                    if quality_metrics.overall_quality >= 0.6:
                        expected_result = ValidationResult.APPROVED.value
                    elif quality_metrics.overall_quality >= 0.42:  # 0.6 * 0.7 for review threshold
                        expected_result = ValidationResult.NEEDS_REVIEW.value
                    else:
                        expected_result = ValidationResult.REJECTED.value
                    
                    if validation_result.value != expected_result:
                        logger.warning(f"  ⚠️ Validation result mismatch: Expected {expected_result}, got {validation_result.value}")
            
            # Validate quality score distribution
            if quality_scores:
                avg_quality = np.mean(quality_scores)
                quality_std = np.std(quality_scores)
                
                logger.info(f"  Quality Score Stats: Mean: {avg_quality:.3f}, Std: {quality_std:.3f}")
                logger.info(f"  Validation Results: {set(validation_results)}")
                
                # Check if we have a reasonable distribution of quality scores
                if 0.4 <= avg_quality <= 0.8 and quality_std > 0.03:
                    logger.info("✅ Signal Quality Scoring: Good score distribution")
                    return True
                else:
                    logger.warning("⚠️ Signal Quality Scoring: Poor score distribution")
                    return False
            else:
                logger.warning("⚠️ Signal Quality Scoring: No quality scores generated")
                return False
                
        except Exception as e:
            logger.error(f"❌ Signal Quality Scoring test failed: {e}")
            return False
    
    async def test_validation_result_determination(self) -> bool:
        """Test validation result determination logic"""
        logger.info("🧪 Testing Validation Result Determination...")
        
        try:
            # Generate test data and signals
            test_data = self.generate_test_data(150)
            test_signals = self.generate_test_signals(4)
            
            validation_results = []
            
            for signal in test_signals:
                # Validate signal
                validation_result, quality_metrics = await self.priority4_system.validate_signal(
                    signal, test_data
                )
                
                if quality_metrics:
                    validation_results.append(validation_result.value)
                    
                    # Check validation logic consistency
                    if quality_metrics.overall_quality < 0.4:
                        if validation_result != ValidationResult.REJECTED:
                            logger.warning(f"  ⚠️ Low quality signal not rejected: Score {quality_metrics.overall_quality:.3f}")
                    
                    if quality_metrics.overall_quality >= 0.8:
                        if validation_result not in [ValidationResult.APPROVED, ValidationResult.CONDITIONAL_APPROVAL]:
                            logger.warning(f"  ⚠️ High quality signal not approved: Score {quality_metrics.overall_quality:.3f}")
            
            # Check validation result distribution
            if validation_results:
                result_counts = {}
                for result in validation_results:
                    result_counts[result] = result_counts.get(result, 0) + 1
                
                logger.info(f"  Validation Results: {result_counts}")
                
                # Should have at least 2 different result types
                if len(result_counts) >= 2:
                    logger.info("✅ Validation Result Determination: Good result distribution")
                    return True
                else:
                    logger.warning("⚠️ Validation Result Determination: Limited result distribution")
                    return False
            else:
                logger.warning("⚠️ Validation Result Determination: No validation results")
                return False
                
        except Exception as e:
            logger.error(f"❌ Validation Result Determination test failed: {e}")
            return False
    
    async def test_performance_tracking(self) -> bool:
        """Test performance tracking system"""
        logger.info("🧪 Testing Performance Tracking...")
        
        try:
            # Get initial performance
            initial_performance = self.priority4_system.get_validation_performance()
            
            # Generate and validate some test signals
            test_data = self.generate_test_data(100)
            test_signals = self.generate_test_signals(2)
            
            for signal in test_signals:
                await self.priority4_system.validate_signal(signal, test_data)
            
            # Get updated performance
            updated_performance = self.priority4_system.get_validation_performance()
            
            # Check if performance was updated
            if (updated_performance['total_signals'] > initial_performance['total_signals']):
                
                logger.info("✅ Performance Tracking: Performance metrics updated successfully")
                logger.info(f"  Total Signals: {updated_performance['total_signals']}")
                logger.info(f"  Approved Signals: {updated_performance['approved_signals']}")
                logger.info(f"  Rejected Signals: {updated_performance['rejected_signals']}")
                logger.info(f"  Accuracy Rate: {updated_performance['accuracy_rate']:.3f}")
                
                return True
            else:
                logger.warning("⚠️ Performance Tracking: No performance updates")
                return False
                
        except Exception as e:
            logger.error(f"❌ Performance Tracking test failed: {e}")
            return False
    
    async def run_all_tests(self) -> Dict[str, bool]:
        """Run all Priority 4 tests"""
        logger.info("🚀 Starting Priority 4 Advanced Signal Validation Tests")
        logger.info("=" * 70)
        
        test_results = {}
        
        # Test 1: Advanced Signal Validation
        test_results['advanced_signal_validation'] = await self.test_advanced_signal_validation()
        
        # Test 2: Signal Quality Scoring
        test_results['signal_quality_scoring'] = await self.test_signal_quality_scoring()
        
        # Test 3: Validation Result Determination
        test_results['validation_result_determination'] = await self.test_validation_result_determination()
        
        # Test 4: Performance Tracking
        test_results['performance_tracking'] = await self.test_performance_tracking()
        
        # Summary
        logger.info("=" * 70)
        logger.info("📊 Priority 4 Test Results Summary:")
        
        passed_tests = sum(test_results.values())
        total_tests = len(test_results)
        
        for test_name, result in test_results.items():
            status = "✅ PASSED" if result else "❌ FAILED"
            logger.info(f"  {test_name}: {status}")
        
        logger.info(f"\n🎯 Overall: {passed_tests}/{total_tests} tests passed")
        
        if passed_tests == total_tests:
            logger.info("🎉 All Priority 4 tests passed successfully!")
        else:
            logger.warning(f"⚠️ {total_tests - passed_tests} tests failed")
        
        return test_results

async def main():
    """Main test execution"""
    tester = Priority4AdvancedSignalValidationTester()
    results = await tester.run_all_tests()
    
    # Return results for potential CI/CD integration
    return results

if __name__ == "__main__":
    asyncio.run(main())
