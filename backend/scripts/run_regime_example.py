#!/usr/bin/env python3
"""
Market Regime Detection Example Usage
Demonstrates integration with AlphaPulse and real-time regime detection
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
import time
from typing import Dict, List

# Import market regime detection modules
from market_regime_detector import MarketRegimeDetector, MarketRegime, RegimeState
from backtest_regime import RegimeBacktester

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RegimeDetectionExample:
    """Example usage of Market Regime Detection module"""
    
    def __init__(self):
        """Initialize the example"""
        self.detector = None
        self.backtester = None
        self.results = []
    
    def generate_sample_data(self, periods: int = 1000) -> pd.DataFrame:
        """Generate realistic sample market data"""
        logger.info(f"Generating {periods} periods of sample data...")
        
        # Create time series
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', periods=periods)
        
        # Generate realistic price movement with different regimes
        np.random.seed(42)
        base_price = 50000.0
        prices = []
        
        for i in range(periods):
            # Simulate different market regimes
            if i < 200:  # Strong trend bull
                price_change = np.random.normal(50, 20)  # Upward trend
            elif i < 400:  # Strong trend bear
                price_change = np.random.normal(-50, 20)  # Downward trend
            elif i < 600:  # Ranging
                price_change = np.random.normal(0, 10)  # Sideways
            elif i < 800:  # Volatile breakout
                price_change = np.random.normal(0, 100)  # High volatility
            else:  # Choppy
                price_change = np.random.normal(0, 30)  # Random movement
            
            base_price += price_change
            prices.append(base_price)
        
        # Create OHLCV data
        data = []
        for i, (date, price) in enumerate(zip(dates, prices)):
            # Generate realistic OHLCV
            volatility = abs(price_change) if i > 0 else 100
            high = price + np.random.uniform(0, volatility * 0.5)
            low = price - np.random.uniform(0, volatility * 0.5)
            open_price = prices[i-1] if i > 0 else price
            close_price = price
            
            # Volume based on volatility
            volume = 1000000 + np.random.randint(-200000, 200000) + int(volatility * 1000)
            
            data.append({
                'timestamp': date,
                'open': open_price,
                'high': high,
                'low': low,
                'close': close_price,
                'volume': volume
            })
        
        return pd.DataFrame(data)
    
    def save_sample_data(self, data: pd.DataFrame, filename: str = "sample_market_data.csv"):
        """Save sample data to CSV"""
        data.to_csv(filename, index=False)
        logger.info(f"Sample data saved to {filename}")
        return filename
    
    def demonstrate_regime_detection(self):
        """Demonstrate basic regime detection functionality"""
        logger.info("=== Demonstrating Basic Regime Detection ===")
        
        # Initialize detector
        detector = MarketRegimeDetector(
            symbol='BTC/USDT',
            timeframe='15m',
            lookback_period=10,
            min_regime_duration=5,
            hysteresis_threshold=0.2,
            enable_ml=False  # Use rule-based for demo
        )
        
        # Sample indicators and candlestick
        indicators = {
            'adx': 35.0,
            'bb_upper': 52000.0,
            'bb_lower': 48000.0,
            'bb_middle': 50000.0,
            'atr': 1500.0,
            'rsi': 65.0,
            'volume_sma': 1000000.0
        }
        
        candlestick = {
            'open': 50000.0,
            'high': 51000.0,
            'low': 49000.0,
            'close': 50500.0,
            'volume': 1500000.0
        }
        
        # Add some price history
        for i in range(20):
            detector.price_history.append(50000.0 + i * 100)
        
        # Update regime
        regime_state = detector.update_regime(indicators, candlestick)
        
        logger.info(f"Current Regime: {regime_state.regime.value}")
        logger.info(f"Confidence: {regime_state.confidence:.2f}")
        logger.info(f"Duration: {regime_state.duration_candles} candles")
        logger.info(f"Stability Score: {regime_state.stability_score:.2f}")
        
        # Test signal filtering
        test_confidences = [0.6, 0.7, 0.8, 0.9]
        for confidence in test_confidences:
            should_filter = detector.should_filter_signal(confidence)
            logger.info(f"Signal confidence {confidence}: {'FILTERED' if should_filter else 'ALLOWED'}")
        
        return detector
    
    def demonstrate_regime_transitions(self):
        """Demonstrate regime transitions"""
        logger.info("=== Demonstrating Regime Transitions ===")
        
        detector = MarketRegimeDetector(
            symbol='BTC/USDT',
            timeframe='15m',
            enable_ml=False
        )
        
        # Add price history
        for i in range(20):
            detector.price_history.append(50000.0 + i * 100)
        
        # Test different market conditions
        test_scenarios = [
            {
                'name': 'Strong Trend Bull',
                'indicators': {
                    'adx': 40.0, 'bb_upper': 52000.0, 'bb_lower': 48000.0,
                    'bb_middle': 50000.0, 'atr': 1500.0, 'rsi': 70.0, 'volume_sma': 1000000.0
                },
                'candlestick': {
                    'open': 50000.0, 'high': 51000.0, 'low': 49000.0,
                    'close': 50500.0, 'volume': 1500000.0
                }
            },
            {
                'name': 'Strong Trend Bear',
                'indicators': {
                    'adx': 40.0, 'bb_upper': 52000.0, 'bb_lower': 48000.0,
                    'bb_middle': 50000.0, 'atr': 1500.0, 'rsi': 30.0, 'volume_sma': 1000000.0
                },
                'candlestick': {
                    'open': 50000.0, 'high': 51000.0, 'low': 49000.0,
                    'close': 49500.0, 'volume': 1500000.0
                }
            },
            {
                'name': 'Ranging',
                'indicators': {
                    'adx': 20.0, 'bb_upper': 50200.0, 'bb_lower': 49800.0,
                    'bb_middle': 50000.0, 'atr': 1000.0, 'rsi': 50.0, 'volume_sma': 1000000.0
                },
                'candlestick': {
                    'open': 50000.0, 'high': 50100.0, 'low': 49900.0,
                    'close': 50050.0, 'volume': 1000000.0
                }
            },
            {
                'name': 'Volatile Breakout',
                'indicators': {
                    'adx': 30.0, 'bb_upper': 53000.0, 'bb_lower': 47000.0,
                    'bb_middle': 50000.0, 'atr': 2000.0, 'rsi': 55.0, 'volume_sma': 1000000.0
                },
                'candlestick': {
                    'open': 50000.0, 'high': 52000.0, 'low': 48000.0,
                    'close': 51000.0, 'volume': 2000000.0
                }
            }
        ]
        
        for scenario in test_scenarios:
            logger.info(f"\nTesting: {scenario['name']}")
            
            # Update regime
            regime_state = detector.update_regime(scenario['indicators'], scenario['candlestick'])
            
            logger.info(f"  Detected Regime: {regime_state.regime.value}")
            logger.info(f"  Confidence: {regime_state.confidence:.2f}")
            logger.info(f"  Expected: {scenario['name'].replace(' ', '_').upper()}")
            
            # Check if detection matches expectation
            expected_regime = scenario['name'].replace(' ', '_').upper()
            if expected_regime in regime_state.regime.value.upper():
                logger.info("  ✓ Detection matches expectation")
            else:
                logger.info("  ✗ Detection differs from expectation")
    
    def demonstrate_backtesting(self, data_path: str):
        """Demonstrate backtesting functionality"""
        logger.info("=== Demonstrating Backtesting ===")
        
        # Initialize backtester
        backtester = RegimeBacktester(
            data_path=data_path,
            symbol='BTC/USDT',
            timeframe='15m',
            train_ratio=0.7,
            random_state=42
        )
        
        # Load data
        if not backtester.load_data():
            logger.error("Failed to load data for backtesting")
            return None
        
        logger.info(f"Loaded {len(backtester.data)} data points")
        logger.info(f"Training set: {len(backtester.train_data)} points")
        logger.info(f"Test set: {len(backtester.test_data)} points")
        
        # Test with default thresholds
        default_thresholds = {
            'adx_trend': 25.0,
            'adx_strong_trend': 35.0,
            'ma_slope_bull': 0.0001,
            'ma_slope_bear': -0.0001,
            'bb_width_volatile': 0.05,
            'bb_width_breakout': 0.07,
            'rsi_overbought': 60.0,
            'rsi_oversold': 40.0,
            'volume_ratio_high': 1.5,
            'breakout_strength_high': 70.0
        }
        
        logger.info("Running backtest with default thresholds...")
        result = backtester.backtest_regime_detector(default_thresholds, enable_ml=False)
        
        if result:
            logger.info(f"Backtest Results:")
            logger.info(f"  Accuracy: {result.accuracy:.3f}")
            logger.info(f"  Stability Score: {result.stability_score:.3f}")
            logger.info(f"  Average Regime Duration: {result.avg_regime_duration:.1f} candles")
            logger.info(f"  Regime Changes: {result.regime_changes}")
            logger.info(f"  Signal Filter Rate: {result.signal_filter_rate:.3f}")
            logger.info(f"  Win Rate: {result.win_rate:.3f}")
            logger.info(f"  Average Latency: {result.latency_ms:.2f} ms")
            
            logger.info(f"Regime Distribution:")
            for regime, count in result.regime_distribution.items():
                percentage = (count / len(backtester.test_data)) * 100
                logger.info(f"  {regime}: {count} ({percentage:.1f}%)")
        
        return backtester
    
    def demonstrate_optimization(self, backtester):
        """Demonstrate threshold optimization"""
        logger.info("=== Demonstrating Threshold Optimization ===")
        
        # Run optimization with limited trials for demo
        logger.info("Running threshold optimization (10 trials)...")
        optimization_result = backtester.optimize_thresholds(n_trials=10)
        
        if optimization_result:
            logger.info(f"Optimization Results:")
            logger.info(f"  Best Accuracy: {optimization_result.best_accuracy:.3f}")
            logger.info(f"  Best Stability: {optimization_result.best_stability:.3f}")
            logger.info(f"  Best Win Rate: {optimization_result.model_performance['win_rate']:.3f}")
            
            logger.info(f"Optimal Thresholds:")
            for param, value in optimization_result.best_thresholds.items():
                logger.info(f"  {param}: {value}")
            
            # Compare with default thresholds
            logger.info("Performance Improvement:")
            default_result = backtester.backtest_regime_detector({
                'adx_trend': 25.0, 'adx_strong_trend': 35.0,
                'ma_slope_bull': 0.0001, 'ma_slope_bear': -0.0001,
                'bb_width_volatile': 0.05, 'bb_width_breakout': 0.07,
                'rsi_overbought': 60.0, 'rsi_oversold': 40.0,
                'volume_ratio_high': 1.5, 'breakout_strength_high': 70.0
            }, enable_ml=False)
            
            if default_result:
                accuracy_improvement = optimization_result.best_accuracy - default_result.accuracy
                stability_improvement = optimization_result.best_stability - default_result.stability_score
                
                logger.info(f"  Accuracy Improvement: {accuracy_improvement:+.3f}")
                logger.info(f"  Stability Improvement: {stability_improvement:+.3f}")
    
    def demonstrate_ml_integration(self, backtester):
        """Demonstrate ML model integration"""
        logger.info("=== Demonstrating ML Model Integration ===")
        
        # Train ML model
        logger.info("Training ML model...")
        success = backtester.train_ml_model()
        
        if success:
            logger.info("✓ ML model trained successfully")
            
            # Test ML-based detection
            logger.info("Testing ML-based regime detection...")
            
            # Create detector with ML model
            detector = MarketRegimeDetector(
                symbol='BTC/USDT',
                timeframe='15m',
                enable_ml=True,
                model_path=f"models/regime_detector_BTC_USDT_15m"
            )
            
            # Test with sample data
            sample_indicators = {
                'adx': 35.0,
                'bb_upper': 52000.0,
                'bb_lower': 48000.0,
                'bb_middle': 50000.0,
                'atr': 1500.0,
                'rsi': 65.0,
                'volume_sma': 1000000.0
            }
            
            sample_candlestick = {
                'open': 50000.0,
                'high': 51000.0,
                'low': 49000.0,
                'close': 50500.0,
                'volume': 1500000.0
            }
            
            # Add price history
            for i in range(20):
                detector.price_history.append(50000.0 + i * 100)
            
            # Update regime with ML
            regime_state = detector.update_regime(sample_indicators, sample_candlestick)
            
            logger.info(f"ML-based Regime Detection:")
            logger.info(f"  Regime: {regime_state.regime.value}")
            logger.info(f"  Confidence: {regime_state.confidence:.3f}")
            logger.info(f"  ML Model Used: {detector.ml_model is not None}")
        else:
            logger.error("✗ ML model training failed")
    
    def demonstrate_performance_benchmark(self):
        """Demonstrate performance benchmarking"""
        logger.info("=== Demonstrating Performance Benchmark ===")
        
        detector = MarketRegimeDetector(
            symbol='BTC/USDT',
            timeframe='15m',
            enable_ml=False  # Disable ML for performance test
        )
        
        # Add price history
        for i in range(20):
            detector.price_history.append(50000.0 + i * 100)
        
        sample_indicators = {
            'adx': 35.0,
            'bb_upper': 52000.0,
            'bb_lower': 48000.0,
            'bb_middle': 50000.0,
            'atr': 1500.0,
            'rsi': 65.0,
            'volume_sma': 1000000.0
        }
        
        sample_candlestick = {
            'open': 50000.0,
            'high': 51000.0,
            'low': 49000.0,
            'close': 50500.0,
            'volume': 1500000.0
        }
        
        # Performance test
        iterations = 1000
        start_time = time.perf_counter()
        
        for _ in range(iterations):
            regime_state = detector.update_regime(sample_indicators, sample_candlestick)
        
        end_time = time.perf_counter()
        total_time = end_time - start_time
        avg_latency = (total_time / iterations) * 1000  # Convert to ms
        throughput = iterations / total_time
        
        logger.info(f"Performance Results:")
        logger.info(f"  Total Time: {total_time:.3f} seconds")
        logger.info(f"  Average Latency: {avg_latency:.2f} ms")
        logger.info(f"  Throughput: {throughput:.0f} updates/second")
        logger.info(f"  Total Updates: {detector.update_count}")
        
        # Performance requirements check
        logger.info(f"Performance Requirements:")
        logger.info(f"  Latency < 50ms: {'✓' if avg_latency < 50 else '✗'} ({avg_latency:.2f}ms)")
        logger.info(f"  Throughput > 100/sec: {'✓' if throughput > 100 else '✗'} ({throughput:.0f}/sec)")
        
        # Get performance metrics
        metrics = detector.get_performance_metrics()
        logger.info(f"  Average Latency (tracked): {metrics['avg_latency_ms']:.2f} ms")
        logger.info(f"  Current Regime: {metrics['current_regime']}")
        logger.info(f"  Regime Confidence: {metrics['regime_confidence']:.3f}")
        logger.info(f"  Stability Score: {metrics['stability_score']:.3f}")
    
    def demonstrate_integration_with_alphapulse(self):
        """Demonstrate integration with AlphaPulse"""
        logger.info("=== Demonstrating AlphaPulse Integration ===")
        
        # Simulate AlphaPulse integration
        class MockAlphaPulse:
            def __init__(self):
                self.regime_detector = MarketRegimeDetector(
                    symbol='BTC/USDT',
                    timeframe='15m',
                    enable_ml=True
                )
                self.signals = []
                self.filtered_signals = 0
            
            def process_candlestick(self, indicators, candlestick):
                """Process candlestick with regime detection"""
                # Update regime
                regime_state = self.regime_detector.update_regime(indicators, candlestick)
                
                # Simulate signal generation
                signal_confidence = np.random.uniform(0.5, 0.95)
                
                # Check if signal should be filtered
                if self.regime_detector.should_filter_signal(signal_confidence):
                    self.filtered_signals += 1
                    logger.info(f"Signal filtered (confidence: {signal_confidence:.2f}, regime: {regime_state.regime.value})")
                    return None
                else:
                    signal = {
                        'confidence': signal_confidence,
                        'regime': regime_state.regime.value,
                        'regime_confidence': regime_state.confidence,
                        'timestamp': datetime.now()
                    }
                    self.signals.append(signal)
                    logger.info(f"Signal generated (confidence: {signal_confidence:.2f}, regime: {regime_state.regime.value})")
                    return signal
        
        # Create mock AlphaPulse
        alphapulse = MockAlphaPulse()
        
        # Add price history
        for i in range(20):
            alphapulse.regime_detector.price_history.append(50000.0 + i * 100)
        
        # Test different market conditions
        test_conditions = [
            {'name': 'Strong Trend', 'adx': 40.0, 'rsi': 70.0},
            {'name': 'Ranging', 'adx': 20.0, 'rsi': 50.0},
            {'name': 'Choppy', 'adx': 15.0, 'rsi': 45.0},
            {'name': 'Volatile', 'adx': 30.0, 'rsi': 55.0}
        ]
        
        for condition in test_conditions:
            logger.info(f"\nTesting {condition['name']} market conditions:")
            
            indicators = {
                'adx': condition['adx'],
                'bb_upper': 52000.0,
                'bb_lower': 48000.0,
                'bb_middle': 50000.0,
                'atr': 1500.0,
                'rsi': condition['rsi'],
                'volume_sma': 1000000.0
            }
            
            candlestick = {
                'open': 50000.0,
                'high': 51000.0,
                'low': 49000.0,
                'close': 50500.0,
                'volume': 1500000.0
            }
            
            # Process multiple candlesticks
            for _ in range(10):
                signal = alphapulse.process_candlestick(indicators, candlestick)
        
        logger.info(f"\nIntegration Results:")
        logger.info(f"  Total Signals Generated: {len(alphapulse.signals)}")
        logger.info(f"  Total Signals Filtered: {alphapulse.filtered_signals}")
        logger.info(f"  Filter Rate: {alphapulse.filtered_signals / (len(alphapulse.signals) + alphapulse.filtered_signals) * 100:.1f}%")
        
        if alphapulse.signals:
            avg_confidence = np.mean([s['confidence'] for s in alphapulse.signals])
            logger.info(f"  Average Signal Confidence: {avg_confidence:.3f}")
    
    def run_comprehensive_demo(self):
        """Run comprehensive demonstration"""
        logger.info("🚀 Starting Market Regime Detection Comprehensive Demo")
        logger.info("=" * 60)
        
        try:
            # Generate sample data
            sample_data = self.generate_sample_data(periods=1000)
            data_path = self.save_sample_data(sample_data)
            
            # 1. Basic regime detection
            self.demonstrate_regime_detection()
            
            # 2. Regime transitions
            self.demonstrate_regime_transitions()
            
            # 3. Backtesting
            backtester = self.demonstrate_backtesting(data_path)
            
            if backtester:
                # 4. Optimization
                self.demonstrate_optimization(backtester)
                
                # 5. ML integration
                self.demonstrate_ml_integration(backtester)
            
            # 6. Performance benchmark
            self.demonstrate_performance_benchmark()
            
            # 7. AlphaPulse integration
            self.demonstrate_integration_with_alphapulse()
            
            logger.info("=" * 60)
            logger.info("✅ Market Regime Detection Demo Completed Successfully!")
            
        except Exception as e:
            logger.error(f"❌ Demo failed: {e}")
            raise

def main():
    """Main function"""
    example = RegimeDetectionExample()
    example.run_comprehensive_demo()

if __name__ == "__main__":
    main()
