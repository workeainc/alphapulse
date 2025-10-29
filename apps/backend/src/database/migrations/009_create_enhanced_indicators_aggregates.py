"""
Migration: Create Enhanced Indicators Continuous Aggregates
TimescaleDB continuous aggregates for ultra-fast indicator queries
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB, TIMESTAMP

# revision identifiers, used by Alembic.
revision = '009_create_enhanced_indicators_aggregates'
down_revision = '008_create_enhanced_data_tables'
branch_labels = None
depends_on = None

def upgrade():
    """Create continuous aggregates for enhanced indicators"""
    
    # 1. Create 5-minute continuous aggregate for enhanced market data
    op.execute("""
        CREATE MATERIALIZED VIEW enhanced_market_data_5m
        WITH (timescaledb.continuous) AS
        SELECT 
            time_bucket('5 minutes', timestamp) AS bucket,
            symbol,
            timeframe,
            first(open, timestamp) AS open,
            max(high) AS high,
            min(low) AS low,
            last(close, timestamp) AS close,
            sum(volume) AS volume,
            avg(price_change) AS avg_price_change,
            avg(volume_change) AS avg_volume_change,
            avg(volatility) AS avg_volatility,
            avg(rsi) AS avg_rsi,
            avg(macd) AS avg_macd,
            avg(macd_signal) AS avg_macd_signal,
            avg(bollinger_upper) AS avg_bollinger_upper,
            avg(bollinger_middle) AS avg_bollinger_middle,
            avg(bollinger_lower) AS avg_bollinger_lower,
            avg(atr) AS avg_atr,
            avg(support_level) AS avg_support_level,
            avg(resistance_level) AS avg_resistance_level,
            avg(market_sentiment) AS avg_market_sentiment,
            avg(data_quality_score) AS avg_data_quality_score,
            count(*) AS data_points
        FROM enhanced_market_data
        GROUP BY bucket, symbol, timeframe
        WITH NO DATA;
    """)
    
    # 2. Create 15-minute continuous aggregate
    op.execute("""
        CREATE MATERIALIZED VIEW enhanced_market_data_15m
        WITH (timescaledb.continuous) AS
        SELECT 
            time_bucket('15 minutes', timestamp) AS bucket,
            symbol,
            timeframe,
            first(open, timestamp) AS open,
            max(high) AS high,
            min(low) AS low,
            last(close, timestamp) AS close,
            sum(volume) AS volume,
            avg(price_change) AS avg_price_change,
            avg(volume_change) AS avg_volume_change,
            avg(volatility) AS avg_volatility,
            avg(rsi) AS avg_rsi,
            avg(macd) AS avg_macd,
            avg(macd_signal) AS avg_macd_signal,
            avg(bollinger_upper) AS avg_bollinger_upper,
            avg(bollinger_middle) AS avg_bollinger_middle,
            avg(bollinger_lower) AS avg_bollinger_lower,
            avg(atr) AS avg_atr,
            avg(support_level) AS avg_support_level,
            avg(resistance_level) AS avg_resistance_level,
            avg(market_sentiment) AS avg_market_sentiment,
            avg(data_quality_score) AS avg_data_quality_score,
            count(*) AS data_points
        FROM enhanced_market_data
        GROUP BY bucket, symbol, timeframe
        WITH NO DATA;
    """)
    
    # 3. Create 1-hour continuous aggregate
    op.execute("""
        CREATE MATERIALIZED VIEW enhanced_market_data_1h
        WITH (timescaledb.continuous) AS
        SELECT 
            time_bucket('1 hour', timestamp) AS bucket,
            symbol,
            timeframe,
            first(open, timestamp) AS open,
            max(high) AS high,
            min(low) AS low,
            last(close, timestamp) AS close,
            sum(volume) AS volume,
            avg(price_change) AS avg_price_change,
            avg(volume_change) AS avg_volume_change,
            avg(volatility) AS avg_volatility,
            avg(rsi) AS avg_rsi,
            avg(macd) AS avg_macd,
            avg(macd_signal) AS avg_macd_signal,
            avg(bollinger_upper) AS avg_bollinger_upper,
            avg(bollinger_middle) AS avg_bollinger_middle,
            avg(bollinger_lower) AS avg_bollinger_lower,
            avg(atr) AS avg_atr,
            avg(support_level) AS avg_support_level,
            avg(resistance_level) AS avg_resistance_level,
            avg(market_sentiment) AS avg_market_sentiment,
            avg(data_quality_score) AS avg_data_quality_score,
            count(*) AS data_points
        FROM enhanced_market_data
        GROUP BY bucket, symbol, timeframe
        WITH NO DATA;
    """)
    
    # 4. Create 4-hour continuous aggregate
    op.execute("""
        CREATE MATERIALIZED VIEW enhanced_market_data_4h
        WITH (timescaledb.continuous) AS
        SELECT 
            time_bucket('4 hours', timestamp) AS bucket,
            symbol,
            timeframe,
            first(open, timestamp) AS open,
            max(high) AS high,
            min(low) AS low,
            last(close, timestamp) AS close,
            sum(volume) AS volume,
            avg(price_change) AS avg_price_change,
            avg(volume_change) AS avg_volume_change,
            avg(volatility) AS avg_volatility,
            avg(rsi) AS avg_rsi,
            avg(macd) AS avg_macd,
            avg(macd_signal) AS avg_macd_signal,
            avg(bollinger_upper) AS avg_bollinger_upper,
            avg(bollinger_middle) AS avg_bollinger_middle,
            avg(bollinger_lower) AS avg_bollinger_lower,
            avg(atr) AS avg_atr,
            avg(support_level) AS avg_support_level,
            avg(resistance_level) AS avg_resistance_level,
            avg(market_sentiment) AS avg_market_sentiment,
            avg(data_quality_score) AS avg_data_quality_score,
            count(*) AS data_points
        FROM enhanced_market_data
        GROUP BY bucket, symbol, timeframe
        WITH NO DATA;
    """)
    
    # 5. Create 1-day continuous aggregate
    op.execute("""
        CREATE MATERIALIZED VIEW enhanced_market_data_1d
        WITH (timescaledb.continuous) AS
        SELECT 
            time_bucket('1 day', timestamp) AS bucket,
            symbol,
            timeframe,
            first(open, timestamp) AS open,
            max(high) AS high,
            min(low) AS low,
            last(close, timestamp) AS close,
            sum(volume) AS volume,
            avg(price_change) AS avg_price_change,
            avg(volume_change) AS avg_volume_change,
            avg(volatility) AS avg_volatility,
            avg(rsi) AS avg_rsi,
            avg(macd) AS avg_macd,
            avg(macd_signal) AS avg_macd_signal,
            avg(bollinger_upper) AS avg_bollinger_upper,
            avg(bollinger_middle) AS avg_bollinger_middle,
            avg(bollinger_lower) AS avg_bollinger_lower,
            avg(atr) AS avg_atr,
            avg(support_level) AS avg_support_level,
            avg(resistance_level) AS avg_resistance_level,
            avg(market_sentiment) AS avg_market_sentiment,
            avg(data_quality_score) AS avg_data_quality_score,
            count(*) AS data_points
        FROM enhanced_market_data
        GROUP BY bucket, symbol, timeframe
        WITH NO DATA;
    """)
    
    # 6. Create RSI trend analysis continuous aggregate
    op.execute("""
        CREATE MATERIALIZED VIEW rsi_trend_analysis
        WITH (timescaledb.continuous) AS
        SELECT 
            time_bucket('1 hour', timestamp) AS bucket,
            symbol,
            timeframe,
            avg(rsi) AS avg_rsi,
            min(rsi) AS min_rsi,
            max(rsi) AS max_rsi,
            stddev(rsi) AS rsi_volatility,
            count(*) AS data_points,
            CASE 
                WHEN avg(rsi) > 70 THEN 'overbought'
                WHEN avg(rsi) < 30 THEN 'oversold'
                ELSE 'neutral'
            END AS rsi_regime
        FROM enhanced_market_data
        WHERE rsi IS NOT NULL
        GROUP BY bucket, symbol, timeframe
        WITH NO DATA;
    """)
    
    # 7. Create MACD signal analysis continuous aggregate
    op.execute("""
        CREATE MATERIALIZED VIEW macd_signal_analysis
        WITH (timescaledb.continuous) AS
        SELECT 
            time_bucket('1 hour', timestamp) AS bucket,
            symbol,
            timeframe,
            avg(macd) AS avg_macd,
            avg(macd_signal) AS avg_macd_signal,
            avg(macd - macd_signal) AS avg_macd_histogram,
            count(*) AS data_points,
            CASE 
                WHEN avg(macd) > avg(macd_signal) THEN 'bullish'
                WHEN avg(macd) < avg(macd_signal) THEN 'bearish'
                ELSE 'neutral'
            END AS macd_signal
        FROM enhanced_market_data
        WHERE macd IS NOT NULL AND macd_signal IS NOT NULL
        GROUP BY bucket, symbol, timeframe
        WITH NO DATA;
    """)
    
    # 8. Create Bollinger Bands analysis continuous aggregate
    op.execute("""
        CREATE MATERIALIZED VIEW bollinger_bands_analysis
        WITH (timescaledb.continuous) AS
        SELECT 
            time_bucket('1 hour', timestamp) AS bucket,
            symbol,
            timeframe,
            avg(bollinger_upper) AS avg_bb_upper,
            avg(bollinger_middle) AS avg_bb_middle,
            avg(bollinger_lower) AS avg_bb_lower,
            avg(bollinger_upper - bollinger_lower) AS avg_bb_width,
            avg((close - bollinger_lower) / (bollinger_upper - bollinger_lower)) AS avg_bb_position,
            count(*) AS data_points,
            CASE 
                WHEN avg(close) > avg(bollinger_upper) THEN 'above_upper'
                WHEN avg(close) < avg(bollinger_lower) THEN 'below_lower'
                ELSE 'within_bands'
            END AS bb_position
        FROM enhanced_market_data
        WHERE bollinger_upper IS NOT NULL AND bollinger_lower IS NOT NULL
        GROUP BY bucket, symbol, timeframe
        WITH NO DATA;
    """)
    
    # 9. Create market sentiment analysis continuous aggregate
    op.execute("""
        CREATE MATERIALIZED VIEW market_sentiment_analysis
        WITH (timescaledb.continuous) AS
        SELECT 
            time_bucket('1 hour', timestamp) AS bucket,
            symbol,
            timeframe,
            avg(market_sentiment) AS avg_sentiment,
            min(market_sentiment) AS min_sentiment,
            max(market_sentiment) AS max_sentiment,
            stddev(market_sentiment) AS sentiment_volatility,
            count(*) AS data_points,
            CASE 
                WHEN avg(market_sentiment) > 0.7 THEN 'bullish'
                WHEN avg(market_sentiment) < 0.3 THEN 'bearish'
                ELSE 'neutral'
            END AS sentiment_regime
        FROM enhanced_market_data
        WHERE market_sentiment IS NOT NULL
        GROUP BY bucket, symbol, timeframe
        WITH NO DATA;
    """)
    
    # 10. Create volatility analysis continuous aggregate
    op.execute("""
        CREATE MATERIALIZED VIEW volatility_analysis
        WITH (timescaledb.continuous) AS
        SELECT 
            time_bucket('1 hour', timestamp) AS bucket,
            symbol,
            timeframe,
            avg(volatility) AS avg_volatility,
            min(volatility) AS min_volatility,
            max(volatility) AS max_volatility,
            avg(atr) AS avg_atr,
            count(*) AS data_points,
            CASE 
                WHEN avg(volatility) > 0.05 THEN 'high'
                WHEN avg(volatility) < 0.01 THEN 'low'
                ELSE 'normal'
            END AS volatility_regime
        FROM enhanced_market_data
        WHERE volatility IS NOT NULL
        GROUP BY bucket, symbol, timeframe
        WITH NO DATA;
    """)
    
    # Add refresh policies for continuous aggregates
    # Refresh 5-minute aggregate every 5 minutes
    op.execute("""
        SELECT add_continuous_aggregate_policy('enhanced_market_data_5m',
            start_offset => INTERVAL '1 hour',
            end_offset => INTERVAL '5 minutes',
            schedule_interval => INTERVAL '5 minutes');
    """)
    
    # Refresh 15-minute aggregate every 15 minutes
    op.execute("""
        SELECT add_continuous_aggregate_policy('enhanced_market_data_15m',
            start_offset => INTERVAL '3 hours',
            end_offset => INTERVAL '15 minutes',
            schedule_interval => INTERVAL '15 minutes');
    """)
    
    # Refresh 1-hour aggregate every hour
    op.execute("""
        SELECT add_continuous_aggregate_policy('enhanced_market_data_1h',
            start_offset => INTERVAL '12 hours',
            end_offset => INTERVAL '1 hour',
            schedule_interval => INTERVAL '1 hour');
    """)
    
    # Refresh 4-hour aggregate every 4 hours
    op.execute("""
        SELECT add_continuous_aggregate_policy('enhanced_market_data_4h',
            start_offset => INTERVAL '2 days',
            end_offset => INTERVAL '4 hours',
            schedule_interval => INTERVAL '4 hours');
    """)
    
    # Refresh 1-day aggregate every day
    op.execute("""
        SELECT add_continuous_aggregate_policy('enhanced_market_data_1d',
            start_offset => INTERVAL '7 days',
            end_offset => INTERVAL '1 day',
            schedule_interval => INTERVAL '1 day');
    """)
    
    # Refresh analysis aggregates every hour
    op.execute("""
        SELECT add_continuous_aggregate_policy('rsi_trend_analysis',
            start_offset => INTERVAL '12 hours',
            end_offset => INTERVAL '1 hour',
            schedule_interval => INTERVAL '1 hour');
    """)
    
    op.execute("""
        SELECT add_continuous_aggregate_policy('macd_signal_analysis',
            start_offset => INTERVAL '12 hours',
            end_offset => INTERVAL '1 hour',
            schedule_interval => INTERVAL '1 hour');
    """)
    
    op.execute("""
        SELECT add_continuous_aggregate_policy('bollinger_bands_analysis',
            start_offset => INTERVAL '12 hours',
            end_offset => INTERVAL '1 hour',
            schedule_interval => INTERVAL '1 hour');
    """)
    
    op.execute("""
        SELECT add_continuous_aggregate_policy('market_sentiment_analysis',
            start_offset => INTERVAL '12 hours',
            end_offset => INTERVAL '1 hour',
            schedule_interval => INTERVAL '1 hour');
    """)
    
    op.execute("""
        SELECT add_continuous_aggregate_policy('volatility_analysis',
            start_offset => INTERVAL '12 hours',
            end_offset => INTERVAL '1 hour',
            schedule_interval => INTERVAL '1 hour');
    """)
    
    # Create indexes for better query performance
    op.execute("""
        CREATE INDEX idx_enhanced_market_data_5m_symbol_bucket 
        ON enhanced_market_data_5m (symbol, bucket DESC);
    """)
    
    op.execute("""
        CREATE INDEX idx_enhanced_market_data_15m_symbol_bucket 
        ON enhanced_market_data_15m (symbol, bucket DESC);
    """)
    
    op.execute("""
        CREATE INDEX idx_enhanced_market_data_1h_symbol_bucket 
        ON enhanced_market_data_1h (symbol, bucket DESC);
    """)
    
    op.execute("""
        CREATE INDEX idx_enhanced_market_data_4h_symbol_bucket 
        ON enhanced_market_data_4h (symbol, bucket DESC);
    """)
    
    op.execute("""
        CREATE INDEX idx_enhanced_market_data_1d_symbol_bucket 
        ON enhanced_market_data_1d (symbol, bucket DESC);
    """)
    
    # Create indexes for analysis views
    op.execute("""
        CREATE INDEX idx_rsi_trend_analysis_symbol_bucket 
        ON rsi_trend_analysis (symbol, bucket DESC);
    """)
    
    op.execute("""
        CREATE INDEX idx_macd_signal_analysis_symbol_bucket 
        ON macd_signal_analysis (symbol, bucket DESC);
    """)
    
    op.execute("""
        CREATE INDEX idx_bollinger_bands_analysis_symbol_bucket 
        ON bollinger_bands_analysis (symbol, bucket DESC);
    """)
    
    op.execute("""
        CREATE INDEX idx_market_sentiment_analysis_symbol_bucket 
        ON market_sentiment_analysis (symbol, bucket DESC);
    """)
    
    op.execute("""
        CREATE INDEX idx_volatility_analysis_symbol_bucket 
        ON volatility_analysis (symbol, bucket DESC);
    """)

def downgrade():
    """Drop continuous aggregates"""
    
    # Drop continuous aggregates
    op.execute("DROP MATERIALIZED VIEW IF EXISTS enhanced_market_data_5m CASCADE;")
    op.execute("DROP MATERIALIZED VIEW IF EXISTS enhanced_market_data_15m CASCADE;")
    op.execute("DROP MATERIALIZED VIEW IF EXISTS enhanced_market_data_1h CASCADE;")
    op.execute("DROP MATERIALIZED VIEW IF EXISTS enhanced_market_data_4h CASCADE;")
    op.execute("DROP MATERIALIZED VIEW IF EXISTS enhanced_market_data_1d CASCADE;")
    op.execute("DROP MATERIALIZED VIEW IF EXISTS rsi_trend_analysis CASCADE;")
    op.execute("DROP MATERIALIZED VIEW IF EXISTS macd_signal_analysis CASCADE;")
    op.execute("DROP MATERIALIZED VIEW IF EXISTS bollinger_bands_analysis CASCADE;")
    op.execute("DROP MATERIALIZED VIEW IF EXISTS market_sentiment_analysis CASCADE;")
    op.execute("DROP MATERIALIZED VIEW IF EXISTS volatility_analysis CASCADE;")
