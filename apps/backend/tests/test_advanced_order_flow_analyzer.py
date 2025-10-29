"""
Tests for Advanced Order Flow Analyzer
Tests comprehensive order flow analysis including toxicity, market maker vs taker, large orders, and patterns
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock, patch

# Import the analyzer
import sys
sys.path.append('backend')
from src.strategies.advanced_order_flow_analyzer import (
    AdvancedOrderFlowAnalyzer,
    OrderFlowToxicityAnalysis,
    MarketMakerTakerAnalysis,
    LargeOrder,
    OrderFlowPattern,
    OrderFlowAlert,
    AdvancedOrderFlowAnalysis,
    ToxicityTrend,
    MarketMakerActivity,
    TakerAggression,
    OrderSizeCategory,
    OrderFlowPatternType,
    AlertLevel
)

@pytest.fixture
def analyzer():
    """Create a test analyzer instance"""
    config = {
        'toxicity_threshold': 0.3,
        'large_order_threshold': 0.1,
        'whale_order_threshold': 0.5,
        'pattern_confidence_threshold': 0.7,
        'min_data_points': 50,
        'volume_threshold': 0.05
    }
    return AdvancedOrderFlowAnalyzer(config)

@pytest.fixture
def sample_order_book_data():
    """Create sample order book data"""
    data = []
    base_time = datetime.now(timezone.utc) - timedelta(hours=100)
    
    for i in range(100):
        # Create realistic order book data
        base_price = 50000 + (i * 10)
        
        # Create bids and asks
        bids = []
        asks = []
        
        for j in range(10):
            bid_price = base_price - (j * 5)
            ask_price = base_price + (j * 5)
            
            # Vary order sizes to create toxicity
            if i % 20 == 0:  # Every 20th snapshot has large orders
                bid_size = 100 + (j * 50) + (i % 10 * 100)  # Large orders
                ask_size = 100 + (j * 50) + (i % 10 * 100)
            else:
                bid_size = 10 + (j * 5) + (i % 5 * 10)  # Normal orders
                ask_size = 10 + (j * 5) + (i % 5 * 10)
            
            bids.append([bid_price, bid_size])
            asks.append([ask_price, ask_size])
        
        data.append({
            'timestamp': base_time + timedelta(minutes=i),
            'bids': bids,
            'asks': asks,
            'symbol': 'BTCUSDT',
            'exchange': 'binance'
        })
    
    return pd.DataFrame(data)

@pytest.fixture
def sample_trade_data():
    """Create sample trade data"""
    data = []
    base_time = datetime.now(timezone.utc) - timedelta(hours=100)
    
    for i in range(100):
        base_price = 50000 + (i * 10)
        
        # Create trades with varying sizes
        if i % 10 == 0:  # Large trades every 10th trade
            quantity = 100 + (i % 5 * 50)
            side = 'buy' if i % 2 == 0 else 'sell'
        else:
            quantity = 10 + (i % 5 * 5)
            side = 'buy' if i % 2 == 0 else 'sell'
        
        data.append({
            'timestamp': base_time + timedelta(minutes=i),
            'price': base_price + (i % 10 - 5),
            'quantity': quantity,
            'quote_quantity': quantity * (base_price + (i % 10 - 5)),
            'side': side,
            'order_type': 'market' if i % 3 == 0 else 'limit',
            'symbol': 'BTCUSDT'
        })
    
    return pd.DataFrame(data)

@pytest.fixture
def sample_volume_data():
    """Create sample volume data"""
    data = []
    base_time = datetime.now(timezone.utc) - timedelta(hours=100)
    
    for i in range(100):
        # Create volume data with some spikes
        if i % 20 == 0:  # Volume spikes every 20th period
            volume = 10000 + (i % 10 * 5000)
        else:
            volume = 1000 + (i % 10 * 500)
        
        data.append({
            'timestamp': base_time + timedelta(minutes=i),
            'volume': volume,
            'symbol': 'BTCUSDT'
        })
    
    return pd.DataFrame(data)

class TestAdvancedOrderFlowAnalyzer:
    """Test the Advanced Order Flow Analyzer"""
    
    def test_analyzer_initialization(self, analyzer):
        """Test analyzer initialization"""
        assert analyzer is not None
        assert analyzer.toxicity_threshold == 0.3
        assert analyzer.large_order_threshold == 0.1
        assert analyzer.whale_order_threshold == 0.5
        assert analyzer.pattern_confidence_threshold == 0.7
        assert analyzer.min_data_points == 50
        assert analyzer.volume_threshold == 0.05
        
        # Check stats initialization
        assert analyzer.stats['total_analyses'] == 0
        assert analyzer.stats['toxicity_analyses'] == 0
        assert analyzer.stats['maker_taker_analyses'] == 0
        assert analyzer.stats['large_orders_detected'] == 0
        assert analyzer.stats['patterns_detected'] == 0
        assert analyzer.stats['alerts_generated'] == 0
        assert analyzer.stats['errors'] == 0
    
    @pytest.mark.asyncio
    async def test_order_flow_toxicity_analysis(self, analyzer, sample_order_book_data, sample_trade_data):
        """Test order flow toxicity analysis"""
        toxicity_analysis = await analyzer._analyze_order_flow_toxicity(
            'BTCUSDT', '1h', sample_order_book_data, sample_trade_data
        )
        
        assert toxicity_analysis is not None
        assert isinstance(toxicity_analysis, OrderFlowToxicityAnalysis)
        assert toxicity_analysis.symbol == 'BTCUSDT'
        assert toxicity_analysis.timeframe == '1h'
        assert -1.0 <= toxicity_analysis.toxicity_score <= 1.0
        assert -1.0 <= toxicity_analysis.bid_toxicity <= 1.0
        assert -1.0 <= toxicity_analysis.ask_toxicity <= 1.0
        assert 0.0 <= toxicity_analysis.large_order_ratio <= 1.0
        assert 0.0 <= toxicity_analysis.toxicity_confidence <= 1.0
        assert 0.0 <= toxicity_analysis.market_impact_score <= 1.0
        assert isinstance(toxicity_analysis.toxicity_trend, ToxicityTrend)
        assert isinstance(toxicity_analysis.order_size_distribution, dict)
        
        # Check order size distribution
        assert 'mean_size' in toxicity_analysis.order_size_distribution
        assert 'std_size' in toxicity_analysis.order_size_distribution
        assert 'size_percentiles' in toxicity_analysis.order_size_distribution
    
    @pytest.mark.asyncio
    async def test_market_maker_taker_analysis(self, analyzer, sample_trade_data, sample_volume_data):
        """Test market maker vs taker analysis"""
        maker_taker_analysis = await analyzer._analyze_market_maker_taker(
            'BTCUSDT', '1h', sample_trade_data, sample_volume_data
        )
        
        assert maker_taker_analysis is not None
        assert isinstance(maker_taker_analysis, MarketMakerTakerAnalysis)
        assert maker_taker_analysis.symbol == 'BTCUSDT'
        assert maker_taker_analysis.timeframe == '1h'
        assert 0.0 <= maker_taker_analysis.maker_volume_ratio <= 1.0
        assert 0.0 <= maker_taker_analysis.taker_volume_ratio <= 1.0
        assert abs(maker_taker_analysis.maker_volume_ratio + maker_taker_analysis.taker_volume_ratio - 1.0) < 0.01
        assert -1.0 <= maker_taker_analysis.maker_taker_imbalance <= 1.0
        assert isinstance(maker_taker_analysis.market_maker_activity, MarketMakerActivity)
        assert isinstance(maker_taker_analysis.taker_aggression, TakerAggression)
        assert 0.0 <= maker_taker_analysis.liquidity_provision_score <= 1.0
    
    @pytest.mark.asyncio
    async def test_large_order_detection(self, analyzer, sample_trade_data, sample_volume_data):
        """Test large order detection"""
        large_orders = await analyzer._detect_large_orders(
            'BTCUSDT', sample_trade_data, sample_volume_data
        )
        
        assert isinstance(large_orders, list)
        
        if large_orders:  # May or may not detect large orders depending on data
            for order in large_orders:
                assert isinstance(order, LargeOrder)
                assert order.symbol == 'BTCUSDT'
                assert order.side in ['buy', 'sell', 'unknown']
                assert order.price > 0
                assert order.quantity > 0
                assert order.quote_quantity > 0
                assert isinstance(order.size_category, OrderSizeCategory)
                assert 0.0 <= order.size_percentile <= 1.0
                assert 0.0 <= order.market_impact <= 1.0
                assert isinstance(order.institutional_indicator, bool)
                assert isinstance(order.analysis_metadata, dict)
    
    @pytest.mark.asyncio
    async def test_order_flow_pattern_detection(self, analyzer, sample_order_book_data, sample_trade_data):
        """Test order flow pattern detection"""
        patterns = await analyzer._detect_order_flow_patterns(
            'BTCUSDT', '1h', sample_order_book_data, sample_trade_data
        )
        
        assert isinstance(patterns, list)
        
        if patterns:  # May or may not detect patterns depending on data
            for pattern in patterns:
                assert isinstance(pattern, OrderFlowPattern)
                assert pattern.symbol == 'BTCUSDT'
                assert pattern.timeframe == '1h'
                assert isinstance(pattern.pattern_type, OrderFlowPatternType)
                assert 0.0 <= pattern.pattern_confidence <= 1.0
                assert 0.0 <= pattern.pattern_strength <= 1.0
                assert isinstance(pattern.volume_profile, dict)
                assert isinstance(pattern.price_action, dict)
                assert isinstance(pattern.order_flow_signature, dict)
                assert isinstance(pattern.analysis_metadata, dict)
    
    @pytest.mark.asyncio
    async def test_absorption_pattern_detection(self, analyzer):
        """Test absorption pattern detection"""
        # Create data that should trigger absorption pattern
        data = []
        base_time = datetime.now(timezone.utc) - timedelta(hours=10)
        
        for i in range(20):
            # Stable price with high volume
            price = 50000 + (i % 3 - 1)  # Very stable price
            volume = 1000 + (i * 100)  # High volume
            
            data.append({
                'timestamp': base_time + timedelta(minutes=i),
                'price': price,
                'quantity': volume,
                'side': 'buy' if i % 2 == 0 else 'sell'
            })
        
        df = pd.DataFrame(data)
        
        pattern = await analyzer._detect_absorption_pattern('BTCUSDT', '1h', df)
        
        # May or may not detect pattern depending on exact conditions
        if pattern:
            assert isinstance(pattern, OrderFlowPattern)
            assert pattern.pattern_type == OrderFlowPatternType.ABSORPTION
            assert pattern.pattern_confidence >= analyzer.pattern_confidence_threshold
    
    @pytest.mark.asyncio
    async def test_distribution_pattern_detection(self, analyzer):
        """Test distribution pattern detection"""
        # Create data that should trigger distribution pattern
        data = []
        base_time = datetime.now(timezone.utc) - timedelta(hours=10)
        
        for i in range(20):
            # Declining price with high volume
            price = 50000 - (i * 50)  # Declining price
            volume = 1000 + (i * 100)  # High volume
            
            data.append({
                'timestamp': base_time + timedelta(minutes=i),
                'price': price,
                'quantity': volume,
                'side': 'sell'  # Mostly selling
            })
        
        df = pd.DataFrame(data)
        
        pattern = await analyzer._detect_distribution_pattern('BTCUSDT', '1h', df)
        
        # May or may not detect pattern depending on exact conditions
        if pattern:
            assert isinstance(pattern, OrderFlowPattern)
            assert pattern.pattern_type == OrderFlowPatternType.DISTRIBUTION
            assert pattern.pattern_confidence >= analyzer.pattern_confidence_threshold
    
    @pytest.mark.asyncio
    async def test_accumulation_pattern_detection(self, analyzer):
        """Test accumulation pattern detection"""
        # Create data that should trigger accumulation pattern
        data = []
        base_time = datetime.now(timezone.utc) - timedelta(hours=10)
        
        for i in range(20):
            # Rising price with high volume
            price = 50000 + (i * 50)  # Rising price
            volume = 1000 + (i * 100)  # High volume
            
            data.append({
                'timestamp': base_time + timedelta(minutes=i),
                'price': price,
                'quantity': volume,
                'side': 'buy'  # Mostly buying
            })
        
        df = pd.DataFrame(data)
        
        pattern = await analyzer._detect_accumulation_pattern('BTCUSDT', '1h', df)
        
        # May or may not detect pattern depending on exact conditions
        if pattern:
            assert isinstance(pattern, OrderFlowPattern)
            assert pattern.pattern_type == OrderFlowPatternType.ACCUMULATION
            assert pattern.pattern_confidence >= analyzer.pattern_confidence_threshold
    
    @pytest.mark.asyncio
    async def test_alert_generation(self, analyzer):
        """Test alert generation"""
        # Create mock analysis results
        toxicity_analysis = OrderFlowToxicityAnalysis(
            timestamp=datetime.now(timezone.utc),
            symbol='BTCUSDT',
            timeframe='1h',
            toxicity_score=0.8,  # High toxicity
            bid_toxicity=0.7,
            ask_toxicity=0.9,
            large_order_ratio=0.3,
            order_size_distribution={},
            toxicity_trend=ToxicityTrend.INCREASING,
            toxicity_confidence=0.9,
            market_impact_score=0.8,
            analysis_metadata={}
        )
        
        large_orders = [
            LargeOrder(
                timestamp=datetime.now(timezone.utc),
                symbol='BTCUSDT',
                order_id='test123',
                side='buy',
                price=50000,
                quantity=1000,
                quote_quantity=50000000,
                order_type='market',
                size_category=OrderSizeCategory.WHALE,
                size_percentile=0.95,
                market_impact=0.8,
                execution_time=None,
                fill_ratio=None,
                slippage=None,
                order_flow_pattern=None,
                institutional_indicator=True,
                analysis_metadata={}
            )
        ]
        
        alerts = await analyzer._generate_alerts(
            'BTCUSDT', toxicity_analysis, None, large_orders, []
        )
        
        assert isinstance(alerts, list)
        assert len(alerts) > 0
        
        for alert in alerts:
            assert isinstance(alert, OrderFlowAlert)
            assert alert.symbol == 'BTCUSDT'
            assert alert.alert_triggered == True
            assert isinstance(alert.alert_level, AlertLevel)
            assert alert.metric_value > 0
            assert alert.threshold_value > 0
            assert isinstance(alert.alert_metadata, dict)
    
    @pytest.mark.asyncio
    async def test_analysis_confidence_calculation(self, analyzer):
        """Test analysis confidence calculation"""
        # Create mock analysis results
        toxicity_analysis = OrderFlowToxicityAnalysis(
            timestamp=datetime.now(timezone.utc),
            symbol='BTCUSDT',
            timeframe='1h',
            toxicity_score=0.5,
            bid_toxicity=0.4,
            ask_toxicity=0.6,
            large_order_ratio=0.2,
            order_size_distribution={},
            toxicity_trend=ToxicityTrend.STABLE,
            toxicity_confidence=0.8,
            market_impact_score=0.5,
            analysis_metadata={}
        )
        
        large_orders = [Mock()] * 5  # 5 large orders
        
        confidence = await analyzer._calculate_analysis_confidence(
            toxicity_analysis, None, large_orders, []
        )
        
        assert 0.0 <= confidence <= 1.0
        assert confidence > 0.5  # Should be reasonably high with good data
    
    @pytest.mark.asyncio
    async def test_market_context_calculation(self, analyzer, sample_order_book_data, sample_trade_data, sample_volume_data):
        """Test market context calculation"""
        context = await analyzer._calculate_market_context(
            'BTCUSDT', sample_order_book_data, sample_trade_data, sample_volume_data
        )
        
        assert isinstance(context, dict)
        assert context['symbol'] == 'BTCUSDT'
        assert context['data_points'] == 100
        assert 'time_range' in context
        assert 'volume_metrics' in context
        assert 'price_metrics' in context
        assert 'order_book_metrics' in context
        
        # Check volume metrics
        assert context['volume_metrics']['total_volume'] > 0
        assert context['volume_metrics']['avg_volume'] > 0
        assert context['volume_metrics']['volume_volatility'] >= 0
        
        # Check price metrics
        assert context['price_metrics']['price_range'] > 0
        assert context['price_metrics']['price_volatility'] >= 0
    
    @pytest.mark.asyncio
    async def test_insufficient_data_handling(self, analyzer):
        """Test handling of insufficient data"""
        # Create minimal data that should trigger insufficient data handling
        minimal_data = pd.DataFrame([{
            'timestamp': datetime.now(timezone.utc),
            'bids': [[50000, 10]],
            'asks': [[50001, 10]]
        }])
        
        analysis = await analyzer.analyze_order_flow(
            'BTCUSDT', '1h', minimal_data, minimal_data, minimal_data
        )
        
        assert isinstance(analysis, AdvancedOrderFlowAnalysis)
        assert analysis.symbol == 'BTCUSDT'
        assert analysis.timeframe == '1h'
        assert analysis.toxicity_analysis is None
        assert analysis.maker_taker_analysis is None
        assert len(analysis.large_orders) == 0
        assert len(analysis.order_flow_patterns) == 0
        assert len(analysis.alerts) == 0
        assert analysis.analysis_confidence == 0.0
        assert analysis.market_context['insufficient_data'] == True
    
    @pytest.mark.asyncio
    async def test_error_handling(self, analyzer):
        """Test error handling"""
        # Test with invalid data that should cause errors
        invalid_data = pd.DataFrame([{
            'timestamp': 'invalid_timestamp',
            'invalid_column': 'invalid_value'
        }])
        
        analysis = await analyzer.analyze_order_flow(
            'BTCUSDT', '1h', invalid_data, invalid_data, invalid_data
        )
        
        assert isinstance(analysis, AdvancedOrderFlowAnalysis)
        assert analysis.symbol == 'BTCUSDT'
        assert analysis.timeframe == '1h'
        # The analyzer handles invalid data gracefully by treating it as insufficient data
        assert analysis.market_context['insufficient_data'] == True
    
    @pytest.mark.asyncio
    async def test_full_analysis_pipeline(self, analyzer, sample_order_book_data, sample_trade_data, sample_volume_data):
        """Test the full analysis pipeline"""
        analysis = await analyzer.analyze_order_flow(
            'BTCUSDT', '1h', sample_order_book_data, sample_trade_data, sample_volume_data
        )
        
        assert isinstance(analysis, AdvancedOrderFlowAnalysis)
        assert analysis.symbol == 'BTCUSDT'
        assert analysis.timeframe == '1h'
        assert -1.0 <= analysis.overall_toxicity_score <= 1.0
        assert 0.0 <= analysis.market_maker_dominance <= 1.0
        assert analysis.large_order_activity >= 0.0
        assert analysis.pattern_activity >= 0.0
        assert 0.0 <= analysis.analysis_confidence <= 1.0
        assert isinstance(analysis.market_context, dict)
        assert isinstance(analysis.analysis_metadata, dict)
        
        # Check that stats were updated
        assert analyzer.stats['total_analyses'] > 0
        
        # Check that we have some analysis results
        assert (analysis.toxicity_analysis is not None or 
                analysis.maker_taker_analysis is not None or 
                len(analysis.large_orders) > 0 or 
                len(analysis.order_flow_patterns) > 0)
    
    @pytest.mark.asyncio
    async def test_analyzer_close(self, analyzer):
        """Test analyzer cleanup"""
        await analyzer.close()
        # Should not raise any exceptions

class TestOrderFlowIntegration:
    """Integration tests for order flow analysis"""
    
    @pytest.mark.asyncio
    async def test_realistic_market_scenario(self):
        """Test with realistic market scenario data"""
        analyzer = AdvancedOrderFlowAnalyzer()
        
        # Create realistic market data
        order_book_data = []
        trade_data = []
        volume_data = []
        
        base_time = datetime.now(timezone.utc) - timedelta(hours=24)
        base_price = 50000
        
        for i in range(1440):  # 24 hours of minute data
            # Simulate market session patterns
            if 8 <= (base_time + timedelta(minutes=i)).hour <= 16:  # Active hours
                volatility = 0.02
                volume_multiplier = 2.0
            else:  # Quiet hours
                volatility = 0.005
                volume_multiplier = 0.5
            
            # Price movement
            price_change = np.random.normal(0, volatility)
            current_price = base_price * (1 + price_change)
            
            # Order book
            bids = []
            asks = []
            for j in range(10):
                bid_price = current_price - (j * 5)
                ask_price = current_price + (j * 5)
                bid_size = (10 + j * 5) * volume_multiplier
                ask_size = (10 + j * 5) * volume_multiplier
                
                # Add some large orders occasionally
                if i % 100 == 0:
                    bid_size *= 5
                    ask_size *= 5
                
                bids.append([bid_price, bid_size])
                asks.append([ask_price, ask_size])
            
            order_book_data.append({
                'timestamp': base_time + timedelta(minutes=i),
                'bids': bids,
                'asks': asks
            })
            
            # Trades
            trade_quantity = (10 + np.random.exponential(20)) * volume_multiplier
            trade_data.append({
                'timestamp': base_time + timedelta(minutes=i),
                'price': current_price,
                'quantity': trade_quantity,
                'quote_quantity': trade_quantity * current_price,
                'side': 'buy' if np.random.random() > 0.5 else 'sell'
            })
            
            # Volume
            volume_data.append({
                'timestamp': base_time + timedelta(minutes=i),
                'volume': trade_quantity * volume_multiplier
            })
            
            base_price = current_price
        
        df_order_book = pd.DataFrame(order_book_data)
        df_trades = pd.DataFrame(trade_data)
        df_volume = pd.DataFrame(volume_data)
        
        # Run analysis
        analysis = await analyzer.analyze_order_flow(
            'BTCUSDT', '1h', df_order_book, df_trades, df_volume
        )
        
        # Verify results
        assert isinstance(analysis, AdvancedOrderFlowAnalysis)
        assert analysis.symbol == 'BTCUSDT'
        assert analysis.timeframe == '1h'
        assert analysis.analysis_confidence > 0.0
        
        # Should detect some patterns in 24 hours of data
        assert (analysis.toxicity_analysis is not None or 
                analysis.maker_taker_analysis is not None or 
                len(analysis.large_orders) > 0 or 
                len(analysis.order_flow_patterns) > 0)
        
        # Check market context
        assert analysis.market_context['data_points'] == 1440
        assert analysis.market_context['volume_metrics']['total_volume'] > 0
        assert analysis.market_context['price_metrics']['price_range'] > 0

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
