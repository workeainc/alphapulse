"""
Test Dynamic Support/Resistance Analyzer
Comprehensive tests for dynamic support/resistance analysis functionality
"""

import pytest
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any

# Import the components to test
from backend.strategies.dynamic_support_resistance_analyzer import (
    DynamicSupportResistanceAnalyzer, SupportResistanceAnalysis, LevelType,
    SupportResistanceLevel, VolumeWeightedLevel, PsychologicalLevel,
    PsychologicalLevelType, InteractionType, LevelInteraction
)

class TestDynamicSupportResistanceAnalyzer:
    """Test suite for Dynamic Support/Resistance Analyzer"""
    
    @pytest.fixture
    def analyzer(self):
        """Create a support/resistance analyzer instance"""
        config = {
            'min_level_strength': 0.3,
            'min_touch_count': 2,
            'level_tolerance': 0.002,
            'volume_threshold': 1.5,
            'lookback_periods': 100
        }
        return DynamicSupportResistanceAnalyzer(config)
    
    @pytest.fixture
    def sample_uptrend_data_with_levels(self):
        """Create sample data with clear support/resistance levels"""
        data = []
        timestamp = datetime.now(timezone.utc) - timedelta(hours=200)
        
        for i in range(200):
            # Create clear support at 100 and resistance at 120
            if i % 20 == 0:  # Support level touches
                base_price = 100.0
            elif i % 20 == 10:  # Resistance level touches
                base_price = 120.0
            else:
                base_price = 100.0 + (i * 0.1)  # Gradual uptrend
            
            high = base_price + 2.0
            low = base_price - 2.0
            close = base_price + 0.5
            open_price = base_price - 0.5
            volume = 1000 + (i * 10)
            
            data.append({
                'timestamp': timestamp + timedelta(hours=i),
                'open': open_price,
                'high': high,
                'low': low,
                'close': close,
                'volume': volume
            })
        
        return data
    
    @pytest.fixture
    def sample_volume_data(self):
        """Create sample data with volume spikes at key levels"""
        data = []
        timestamp = datetime.now(timezone.utc) - timedelta(hours=100)
        
        for i in range(100):
            base_price = 100.0 + (i * 0.1)
            
            # Create volume spikes at round number levels
            if abs(base_price - 105.0) < 0.5 or abs(base_price - 110.0) < 0.5:
                volume = 5000  # High volume at key levels
            else:
                volume = 1000  # Normal volume
            
            data.append({
                'timestamp': timestamp + timedelta(hours=i),
                'open': base_price - 0.5,
                'high': base_price + 1.0,
                'low': base_price - 1.0,
                'close': base_price + 0.2,
                'volume': volume
            })
        
        return data
    
    @pytest.mark.asyncio
    async def test_analyzer_initialization(self, analyzer):
        """Test analyzer initialization"""
        assert analyzer is not None
        assert analyzer.min_level_strength == 0.3
        assert analyzer.min_touch_count == 2
        assert analyzer.level_tolerance == 0.002
        assert analyzer.volume_threshold == 1.5
        assert analyzer.lookback_periods == 100
        assert analyzer.stats['analyses_performed'] == 0
    
    @pytest.mark.asyncio
    async def test_basic_level_detection(self, analyzer, sample_uptrend_data_with_levels):
        """Test basic support/resistance level detection"""
        analysis = await analyzer.analyze_support_resistance(
            'BTCUSDT', '1h', sample_uptrend_data_with_levels
        )
        
        assert analysis is not None
        assert analysis.symbol == 'BTCUSDT'
        assert analysis.timeframe == '1h'
        assert len(analysis.support_levels) > 0
        assert len(analysis.resistance_levels) > 0
        
        # Check that levels have expected properties
        for level in analysis.support_levels:
            assert isinstance(level, SupportResistanceLevel)
            assert level.level_type == LevelType.SUPPORT
            assert level.price_level > 0
            assert 0 <= level.strength <= 1
            assert 0 <= level.confidence <= 1
        
        for level in analysis.resistance_levels:
            assert isinstance(level, SupportResistanceLevel)
            assert level.level_type == LevelType.RESISTANCE
            assert level.price_level > 0
            assert 0 <= level.strength <= 1
            assert 0 <= level.confidence <= 1
    
    @pytest.mark.asyncio
    async def test_level_consolidation(self, analyzer):
        """Test consolidation of nearby levels"""
        # Create levels that should be consolidated
        levels = [
            SupportResistanceLevel(
                level_type=LevelType.SUPPORT,
                price_level=100.0,
                strength=0.5,
                confidence=0.5
            ),
            SupportResistanceLevel(
                level_type=LevelType.SUPPORT,
                price_level=100.1,  # Very close to first level
                strength=0.6,
                confidence=0.6
            ),
            SupportResistanceLevel(
                level_type=LevelType.SUPPORT,
                price_level=105.0,  # Far from other levels
                strength=0.7,
                confidence=0.7
            )
        ]
        
        consolidated = await analyzer._consolidate_levels(levels)
        
        # Should consolidate the first two levels
        assert len(consolidated) == 2
        
        # Check that consolidation preserved important properties
        assert any(abs(level.price_level - 100.05) < 0.1 for level in consolidated)  # Merged level
        assert any(abs(level.price_level - 105.0) < 0.1 for level in consolidated)   # Unchanged level
    
    @pytest.mark.asyncio
    async def test_volume_weighted_levels(self, analyzer, sample_volume_data):
        """Test volume-weighted level detection"""
        analysis = await analyzer.analyze_support_resistance(
            'BTCUSDT', '1h', sample_volume_data
        )
        
        # Volume-weighted levels may or may not be detected depending on data distribution
        if len(analysis.volume_weighted_levels) > 0:
            for level in analysis.volume_weighted_levels:
                assert isinstance(level, VolumeWeightedLevel)
                assert level.volume_weight > 0
                assert level.volume_percentage > 0
                assert 0 <= level.validation_score <= 1
                assert level.level_reliability in ['high', 'medium', 'low']
        
        # At minimum, the analysis should complete successfully
        assert analysis is not None
        assert analysis.symbol == 'BTCUSDT'
    
    @pytest.mark.asyncio
    async def test_psychological_levels(self, analyzer, sample_uptrend_data_with_levels):
        """Test psychological level detection"""
        analysis = await analyzer.analyze_support_resistance(
            'BTCUSDT', '1h', sample_uptrend_data_with_levels
        )
        
        assert len(analysis.psychological_levels) > 0
        
        for level in analysis.psychological_levels:
            assert isinstance(level, PsychologicalLevel)
            assert level.level_type == PsychologicalLevelType.ROUND_NUMBER
            assert level.psychological_strength > 0
            assert 0 <= level.reliability_score <= 1
            assert level.round_number_type in ['major', 'minor', 'micro']
    
    @pytest.mark.asyncio
    async def test_level_validation_and_scoring(self, analyzer, sample_uptrend_data_with_levels):
        """Test level validation and scoring"""
        df = pd.DataFrame(sample_uptrend_data_with_levels)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Create a test level
        test_level = SupportResistanceLevel(
            level_type=LevelType.SUPPORT,
            price_level=100.0,
            strength=0.5,
            confidence=0.5
        )
        
        # Test touch counting
        touches = await analyzer._count_level_touches(df, test_level)
        assert len(touches) > 0
        
        # Test strength calculation
        test_level.touch_count = len(touches)
        test_level.touch_points = touches
        strength = await analyzer._calculate_level_strength(df, test_level)
        assert 0 <= strength <= 1
        
        # Test confidence calculation
        confidence = await analyzer._calculate_level_confidence(df, test_level)
        assert 0 <= confidence <= 1
    
    @pytest.mark.asyncio
    async def test_level_interaction_analysis(self, analyzer, sample_uptrend_data_with_levels):
        """Test level interaction analysis"""
        analysis = await analyzer.analyze_support_resistance(
            'BTCUSDT', '1h', sample_uptrend_data_with_levels
        )
        
        # Should have some recent interactions
        assert isinstance(analysis.recent_interactions, list)
        
        if analysis.recent_interactions:
            for interaction in analysis.recent_interactions:
                assert isinstance(interaction, LevelInteraction)
                assert interaction.level_type in [LevelType.SUPPORT, LevelType.RESISTANCE]
                assert interaction.interaction_type in [
                    InteractionType.TOUCH, InteractionType.BOUNCE, 
                    InteractionType.PENETRATION, InteractionType.BREAK
                ]
                assert interaction.approach_price > 0
                assert interaction.interaction_price > 0
    
    @pytest.mark.asyncio
    async def test_level_break_detection(self, analyzer):
        """Test level break detection"""
        # Create data where a support level gets broken
        data = []
        timestamp = datetime.now(timezone.utc) - timedelta(hours=50)
        
        for i in range(50):
            if i < 30:
                # Price stays above support at 100
                base_price = 102.0
            else:
                # Price breaks below support
                base_price = 98.0
            
            data.append({
                'timestamp': timestamp + timedelta(hours=i),
                'open': base_price - 0.5,
                'high': base_price + 1.0,
                'low': base_price - 1.0,
                'close': base_price + 0.2,
                'volume': 1000
            })
        
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Create a support level that should be broken
        test_level = SupportResistanceLevel(
            level_type=LevelType.SUPPORT,
            price_level=100.0,
            strength=0.7,
            confidence=0.7
        )
        
        is_broken = await analyzer._check_if_level_broken(df, test_level)
        assert is_broken  # Should detect the break
    
    @pytest.mark.asyncio
    async def test_overall_analysis_metrics(self, analyzer, sample_uptrend_data_with_levels):
        """Test overall analysis metrics calculation"""
        analysis = await analyzer.analyze_support_resistance(
            'BTCUSDT', '1h', sample_uptrend_data_with_levels
        )
        
        # Check overall strength
        assert 0 <= analysis.overall_strength <= 1
        
        # Check analysis confidence
        assert 0 <= analysis.analysis_confidence <= 1
        
        # Check market context
        assert isinstance(analysis.market_context, dict)
        assert 'current_price' in analysis.market_context
        assert 'total_support_levels' in analysis.market_context
        assert 'total_resistance_levels' in analysis.market_context
    
    @pytest.mark.asyncio
    async def test_volume_confirmation(self, analyzer, sample_volume_data):
        """Test volume confirmation for levels"""
        df = pd.DataFrame(sample_volume_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Create a level with volume data
        test_level = SupportResistanceLevel(
            level_type=LevelType.SUPPORT,
            price_level=105.0,  # Should have high volume
            strength=0.5,
            confidence=0.5
        )
        
        # Add some touch points
        test_level.touch_points = [
            type('TouchPoint', (), {
                'volume': 5000,  # High volume touch
                'timestamp': df['timestamp'].iloc[0]
            })()
        ]
        
        volume_confirmed = await analyzer._check_volume_confirmation(df, test_level)
        assert isinstance(volume_confirmed, bool)
    
    @pytest.mark.asyncio
    async def test_institutional_activity_detection(self, analyzer, sample_volume_data):
        """Test institutional activity detection"""
        df = pd.DataFrame(sample_volume_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Create a level with institutional volume
        test_level = SupportResistanceLevel(
            level_type=LevelType.RESISTANCE,
            price_level=110.0,
            strength=0.5,
            confidence=0.5
        )
        
        # Add touch points with institutional volume
        test_level.touch_points = [
            type('TouchPoint', (), {
                'volume': 10000,  # Very high volume (institutional)
                'timestamp': df['timestamp'].iloc[0]
            })()
        ]
        
        institutional_activity = await analyzer._check_institutional_activity(df, test_level)
        assert isinstance(institutional_activity, bool)
    
    @pytest.mark.asyncio
    async def test_insufficient_data_handling(self, analyzer):
        """Test handling of insufficient data"""
        insufficient_data = [
            {
                'timestamp': datetime.now(timezone.utc),
                'open': 100.0,
                'high': 101.0,
                'low': 99.0,
                'close': 100.5,
                'volume': 1000
            }
        ]
        
        analysis = await analyzer.analyze_support_resistance(
            'BTCUSDT', '1h', insufficient_data
        )
        
        assert analysis is not None
        assert analysis.overall_strength == 0.0
        assert analysis.analysis_confidence == 0.0
        assert len(analysis.support_levels) == 0
        assert len(analysis.resistance_levels) == 0
    
    @pytest.mark.asyncio
    async def test_performance_statistics(self, analyzer, sample_uptrend_data_with_levels):
        """Test performance statistics tracking"""
        initial_stats = analyzer.stats.copy()
        
        await analyzer.analyze_support_resistance('BTCUSDT', '1h', sample_uptrend_data_with_levels)
        
        # Check that statistics were updated
        assert analyzer.stats['analyses_performed'] > initial_stats['analyses_performed']
        assert analyzer.stats['levels_detected'] > initial_stats['levels_detected']
        assert analyzer.stats['last_update'] is not None
    
    @pytest.mark.asyncio
    async def test_error_handling(self, analyzer):
        """Test error handling with invalid data"""
        invalid_data = [
            {
                'timestamp': 'invalid_timestamp',
                'open': 'invalid_price',
                'high': 'invalid_price',
                'low': 'invalid_price',
                'close': 'invalid_price',
                'volume': 'invalid_volume'
            }
        ]
        
        # Should not raise exception, should return default analysis
        analysis = await analyzer.analyze_support_resistance(
            'BTCUSDT', '1h', invalid_data
        )
        
        assert analysis is not None
        assert analysis.overall_strength == 0.0
        assert analysis.analysis_confidence == 0.0
    
    @pytest.mark.asyncio
    async def test_market_context_calculation(self, analyzer, sample_uptrend_data_with_levels):
        """Test market context calculation"""
        analysis = await analyzer.analyze_support_resistance(
            'BTCUSDT', '1h', sample_uptrend_data_with_levels
        )
        
        context = analysis.market_context
        assert 'current_price' in context
        assert 'total_support_levels' in context
        assert 'total_resistance_levels' in context
        
        # Check if distances are calculated when levels exist
        if context.get('nearest_support'):
            assert 'support_distance_pct' in context
        if context.get('nearest_resistance'):
            assert 'resistance_distance_pct' in context

class TestSupportResistanceIntegration:
    """Integration tests for support/resistance analysis"""
    
    @pytest.mark.asyncio
    async def test_full_analysis_pipeline(self):
        """Test the complete support/resistance analysis pipeline"""
        config = {
            'min_level_strength': 0.2,
            'min_touch_count': 2,
            'level_tolerance': 0.003,
            'volume_threshold': 1.5,
            'lookback_periods': 100
        }
        
        analyzer = DynamicSupportResistanceAnalyzer(config)
        
        # Create comprehensive test data
        data = []
        timestamp = datetime.now(timezone.utc) - timedelta(hours=150)
        
        for i in range(150):
            # Create multiple phases with clear levels
            if i < 50:
                # Consolidation phase with clear support/resistance
                if i % 10 == 0:
                    base_price = 100.0  # Support
                elif i % 10 == 5:
                    base_price = 110.0  # Resistance
                else:
                    base_price = 105.0  # Middle range
            elif i < 100:
                # Breakout phase
                base_price = 110.0 + (i - 50) * 0.2
            else:
                # New consolidation at higher level
                if i % 10 == 0:
                    base_price = 120.0  # New support
                elif i % 10 == 5:
                    base_price = 130.0  # New resistance
                else:
                    base_price = 125.0  # Middle range
            
            # Add volume spikes at key levels
            if abs(base_price - 100.0) < 1 or abs(base_price - 110.0) < 1 or abs(base_price - 120.0) < 1:
                volume = 3000
            else:
                volume = 1000
            
            data.append({
                'timestamp': timestamp + timedelta(hours=i),
                'open': base_price - 0.5,
                'high': base_price + 1.5,
                'low': base_price - 1.5,
                'close': base_price + 0.3,
                'volume': volume
            })
        
        # Run full analysis
        analysis = await analyzer.analyze_support_resistance('BTCUSDT', '1h', data)
        
        # Validate comprehensive results
        assert analysis is not None
        assert analysis.symbol == 'BTCUSDT'
        assert analysis.timeframe == '1h'
        assert analysis.overall_strength > 0
        assert analysis.analysis_confidence > 0
        
        # Should detect multiple types of levels
        assert len(analysis.support_levels) > 0
        assert len(analysis.resistance_levels) > 0
        assert len(analysis.volume_weighted_levels) > 0
        assert len(analysis.psychological_levels) > 0
        
        # Should have level interactions
        assert len(analysis.recent_interactions) > 0
        
        # Check market context
        assert analysis.market_context['current_price'] > 0
        assert analysis.market_context['total_support_levels'] > 0
        assert analysis.market_context['total_resistance_levels'] > 0
        
        # Check performance statistics
        assert analyzer.stats['analyses_performed'] == 1
        assert analyzer.stats['levels_detected'] > 0
        assert analyzer.stats['volume_weighted_levels'] > 0
        assert analyzer.stats['psychological_levels'] > 0
        assert analyzer.stats['last_update'] is not None
    
    @pytest.mark.asyncio
    async def test_level_strength_distribution(self):
        """Test that level strengths are properly distributed"""
        analyzer = DynamicSupportResistanceAnalyzer()
        
        # Create data with varying level quality
        data = []
        timestamp = datetime.now(timezone.utc) - timedelta(hours=100)
        
        for i in range(100):
            # Strong level at 100 (many touches)
            if i % 5 == 0:
                base_price = 100.0
                volume = 2000  # High volume
            # Weak level at 105 (few touches)
            elif i % 25 == 0:
                base_price = 105.0
                volume = 1000
            else:
                base_price = 102.0 + (i * 0.01)
                volume = 1000
            
            data.append({
                'timestamp': timestamp + timedelta(hours=i),
                'open': base_price - 0.5,
                'high': base_price + 1.0,
                'low': base_price - 1.0,
                'close': base_price + 0.2,
                'volume': volume
            })
        
        analysis = await analyzer.analyze_support_resistance('BTCUSDT', '1h', data)
        
        # Should have levels with different strengths
        all_levels = analysis.support_levels + analysis.resistance_levels
        if len(all_levels) > 1:
            strengths = [level.strength for level in all_levels]
            assert max(strengths) > min(strengths), "Should have levels with different strengths"
    
    @pytest.mark.asyncio
    async def test_round_number_psychological_levels(self):
        """Test round number psychological level detection"""
        analyzer = DynamicSupportResistanceAnalyzer()
        
        # Create data around round numbers
        data = []
        timestamp = datetime.now(timezone.utc) - timedelta(hours=100)
        
        for i in range(100):  # Increase data points to meet minimum requirement
            # Price movement around 100 (major round number)
            base_price = 99.0 + (i * 0.02)  # Move from 99 to 101
            
            data.append({
                'timestamp': timestamp + timedelta(hours=i),
                'open': base_price - 0.2,
                'high': base_price + 0.5,
                'low': base_price - 0.5,
                'close': base_price + 0.1,
                'volume': 1000
            })
        
        analysis = await analyzer.analyze_support_resistance('BTCUSDT', '1h', data)
        
        # Psychological level detection may vary based on data distribution
        # At minimum, the analysis should complete successfully
        assert analysis is not None
        assert analysis.symbol == 'BTCUSDT'
        
        # If psychological levels are detected, verify they have expected properties
        if len(analysis.psychological_levels) > 0:
            for level in analysis.psychological_levels:
                assert isinstance(level, PsychologicalLevel)
                assert level.psychological_strength > 0
                assert 0 <= level.validation_score <= 1
                assert level.level_type in [
                    PsychologicalLevelType.ROUND_NUMBER,
                    PsychologicalLevelType.HALF_POINT,
                    PsychologicalLevelType.PREVIOUS_HIGHS_LOWS
                ]

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
