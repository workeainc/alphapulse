#!/usr/bin/env python3
"""
Simple test script for AlphaPulse Data Pipeline
Tests the complete data pipeline workflow
"""

import asyncio
import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.insert(0, os.getcwd())

from backend.data.pipeline import DataPipeline, PipelineConfig
from backend.data.storage import DataStorage
from backend.data.validation import CandlestickValidator
from backend.strategies.indicators import TechnicalIndicators
from backend.strategies.pattern_detector import CandlestickPatternDetector

async def test_pipeline_components():
    """Test individual pipeline components"""
    print("Testing pipeline components...")
    
    # Test 1: Storage initialization
    try:
        storage = DataStorage("test_data", "sqlite")
        print("✅ Storage initialization: PASSED")
    except Exception as e:
        print(f"❌ Storage initialization: FAILED - {e}")
        return False
    
    # Test 2: Validation initialization
    try:
        validator = CandlestickValidator()
        print("✅ Validation initialization: PASSED")
    except Exception as e:
        print(f"❌ Validation initialization: FAILED - {e}")
        return False
    
    # Test 3: Technical indicators initialization
    try:
        indicators = TechnicalIndicators()
        print("✅ Technical indicators initialization: PASSED")
    except Exception as e:
        print(f"❌ Technical indicators initialization: FAILED - {e}")
        return False
    
    # Test 4: Pattern detector initialization
    try:
        pattern_detector = CandlestickPatternDetector()
        print("✅ Pattern detector initialization: PASSED")
    except Exception as e:
        print(f"❌ Pattern detector initialization: FAILED - {e}")
        return False
    
    return True

async def test_pipeline_integration():
    """Test pipeline integration with minimal configuration"""
    print("\nTesting pipeline integration...")
    
    try:
        # Create minimal test configuration
        config = PipelineConfig(
            symbols=['BTCUSDT'],
            intervals=['1h'],
            exchanges=['binance'],
            batch_size=100,
            retry_attempts=2,
            validation_enabled=True,
            storage_type='sqlite',
            storage_path='test_data',
            max_workers=1,
            update_frequency_minutes=5,
            analysis_enabled=True,
            pattern_detection_enabled=True,
            technical_indicators_enabled=True
        )
        
        # Initialize pipeline
        pipeline = DataPipeline(config)
        print("✅ Pipeline initialization: PASSED")
        
        # Test pipeline status initialization
        summary = pipeline.get_pipeline_summary()
        if summary['total_combinations'] == 1:
            print("✅ Pipeline status initialization: PASSED")
        else:
            print(f"❌ Pipeline status initialization: FAILED - Expected 1, got {summary['total_combinations']}")
            return False
        
        # Test pipeline reset
        pipeline.reset_pipeline_status()
        summary_after_reset = pipeline.get_pipeline_summary()
        if summary_after_reset['pending'] == 1:
            print("✅ Pipeline reset functionality: PASSED")
        else:
            print(f"❌ Pipeline reset functionality: FAILED - Expected 1 pending, got {summary_after_reset['pending']}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Pipeline integration test: FAILED - {e}")
        return False

async def test_data_validation():
    """Test data validation with sample data"""
    print("\nTesting data validation...")
    
    try:
        import pandas as pd
        import numpy as np
        
        # Create sample candlestick data
        dates = pd.date_range('2024-01-01', periods=100, freq='1H')
        sample_data = pd.DataFrame({
            'open': np.random.uniform(40000, 50000, 100),
            'high': np.random.uniform(40000, 50000, 100),
            'low': np.random.uniform(40000, 50000, 100),
            'close': np.random.uniform(40000, 50000, 100),
            'volume': np.random.uniform(100, 1000, 100)
        }, index=dates)
        
        # Ensure high >= low, high >= open, high >= close, low <= open, low <= close
        for i in range(len(sample_data)):
            high = max(sample_data.iloc[i]['open'], sample_data.iloc[i]['close'])
            low = min(sample_data.iloc[i]['open'], sample_data.iloc[i]['close'])
            sample_data.iloc[i, sample_data.columns.get_loc('high')] = high + np.random.uniform(0, 1000)
            sample_data.iloc[i, sample_data.columns.get_loc('low')] = low - np.random.uniform(0, 1000)
        
        # Test validation
        validator = CandlestickValidator()
        cleaned_data, validation_report = validator.validate_candlestick_data(sample_data)
        
        if len(cleaned_data) > 0 and validation_report['quality_score'] > 0.8:
            print("✅ Data validation: PASSED")
            print(f"   Quality score: {validation_report['quality_score']:.2f}")
            print(f"   Records cleaned: {validation_report['cleaned_count']}")
        else:
            print(f"❌ Data validation: FAILED - Quality score: {validation_report['quality_score']}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Data validation test: FAILED - {e}")
        return False

async def test_technical_indicators():
    """Test technical indicators calculation"""
    print("\nTesting technical indicators...")
    
    try:
        import pandas as pd
        import numpy as np
        
        # Create sample data
        dates = pd.date_range('2024-01-01', periods=100, freq='1H')
        sample_data = pd.DataFrame({
            'open': np.random.uniform(40000, 50000, 100),
            'high': np.random.uniform(40000, 50000, 100),
            'low': np.random.uniform(40000, 50000, 100),
            'close': np.random.uniform(40000, 50000, 100),
            'volume': np.random.uniform(100, 1000, 100)
        }, index=dates)
        
        # Ensure high >= low, high >= open, high >= close, low <= open, low <= close
        for i in range(len(sample_data)):
            high = max(sample_data.iloc[i]['open'], sample_data.iloc[i]['close'])
            low = min(sample_data.iloc[i]['open'], sample_data.iloc[i]['close'])
            sample_data.iloc[i, sample_data.columns.get_loc('high')] = high + np.random.uniform(0, 1000)
            sample_data.iloc[i, sample_data.columns.get_loc('low')] = low - np.random.uniform(0, 1000)
        
        # Test indicators
        indicators_calc = TechnicalIndicators()
        indicators = indicators_calc.calculate_all_indicators(sample_data)
        
        if len(indicators) > 0:
            print("✅ Technical indicators: PASSED")
            print(f"   Indicators calculated: {len(indicators)}")
            print(f"   Available indicators: {list(indicators.keys())}")
        else:
            print("❌ Technical indicators: FAILED - No indicators calculated")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Technical indicators test: FAILED - {e}")
        return False

async def test_pattern_detection():
    """Test candlestick pattern detection"""
    print("\nTesting pattern detection...")
    
    try:
        import pandas as pd
        import numpy as np
        
        # Create sample data with some patterns
        dates = pd.date_range('2024-01-01', periods=100, freq='1H')
        sample_data = pd.DataFrame({
            'open': np.random.uniform(40000, 50000, 100),
            'high': np.random.uniform(40000, 50000, 100),
            'low': np.random.uniform(40000, 50000, 100),
            'close': np.random.uniform(40000, 50000, 100),
            'volume': np.random.uniform(100, 1000, 100)
        }, index=dates)
        
        # Ensure high >= low, high >= open, high >= close, low <= open, low <= close
        for i in range(len(sample_data)):
            high = max(sample_data.iloc[i]['open'], sample_data.iloc[i]['close'])
            low = min(sample_data.iloc[i]['open'], sample_data.iloc[i]['close'])
            sample_data.iloc[i, sample_data.columns.get_loc('high')] = high + np.random.uniform(0, 1000)
            sample_data.iloc[i, sample_data.columns.get_loc('low')] = low - np.random.uniform(0, 1000)
        
        # Test pattern detection
        pattern_detector = CandlestickPatternDetector()
        patterns = pattern_detector.detect_patterns_from_dataframe(sample_data)
        
        print("✅ Pattern detection: PASSED")
        print(f"   Patterns detected: {len(patterns)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Pattern detection test: FAILED - {e}")
        return False

async def cleanup_test_data():
    """Clean up test data"""
    try:
        import shutil
        test_data_path = Path("test_data")
        if test_data_path.exists():
            shutil.rmtree(test_data_path)
            print("✅ Test data cleaned up")
    except Exception as e:
        print(f"⚠️  Cleanup warning: {e}")

async def main():
    """Main test function"""
    print("🚀 Starting AlphaPulse Data Pipeline Tests\n")
    
    tests = [
        test_pipeline_components,
        test_pipeline_integration,
        test_data_validation,
        test_technical_indicators,
        test_pattern_detection
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if await test():
                passed += 1
            else:
                print(f"Test {test.__name__} failed")
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The pipeline is working correctly.")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
    
    # Cleanup
    await cleanup_test_data()
    
    return passed == total

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
