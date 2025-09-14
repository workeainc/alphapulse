"""
Migration: Live Market Data Integration
Revision: 059_live_market_data_integration
Adds tables and columns for live market data integration and advanced features
"""

import asyncio
import asyncpg
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

async def create_live_market_data_tables():
    """Create tables for live market data integration"""
    db_config = {
        'host': 'postgres',
        'port': 5432,
        'user': 'alpha_emon',
        'password': 'Emon_@17711',
        'database': 'alphapulse'
    }
    
    try:
        conn = await asyncpg.connect(**db_config)
        logger.info("✅ Connected to database")
        
        # Create live market data table
        live_market_data_table = """
        CREATE TABLE IF NOT EXISTS live_market_data (
            timestamp TIMESTAMPTZ NOT NULL,
            symbol VARCHAR(20) NOT NULL,
            price DECIMAL(20,8) NOT NULL,
            volume DECIMAL(20,8),
            bid DECIMAL(20,8),
            ask DECIMAL(20,8),
            spread DECIMAL(20,8),
            high_24h DECIMAL(20,8),
            low_24h DECIMAL(20,8),
            change_24h DECIMAL(10,4),
            change_percent_24h DECIMAL(10,4),
            market_cap DECIMAL(30,8),
            circulating_supply DECIMAL(30,8),
            total_supply DECIMAL(30,8),
            max_supply DECIMAL(30,8),
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
        """
        await conn.execute(live_market_data_table)
        logger.info("✅ Created live_market_data table")
        
        # Create order book data table
        order_book_table = """
        CREATE TABLE IF NOT EXISTS order_book_data (
            timestamp TIMESTAMPTZ NOT NULL,
            symbol VARCHAR(20) NOT NULL,
            side VARCHAR(4) NOT NULL,  -- 'bid' or 'ask'
            price DECIMAL(20,8) NOT NULL,
            volume DECIMAL(20,8) NOT NULL,
            order_count INTEGER,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
        """
        await conn.execute(order_book_table)
        logger.info("✅ Created order_book_data table")
        
        # Create trade execution table
        trade_execution_table = """
        CREATE TABLE IF NOT EXISTS trade_executions (
            execution_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            signal_id VARCHAR(100),
            symbol VARCHAR(20) NOT NULL,
            side VARCHAR(4) NOT NULL,  -- 'buy' or 'sell'
            order_type VARCHAR(20) NOT NULL,  -- 'market', 'limit', 'stop'
            quantity DECIMAL(20,8) NOT NULL,
            price DECIMAL(20,8) NOT NULL,
            executed_price DECIMAL(20,8),
            status VARCHAR(20) NOT NULL,  -- 'pending', 'filled', 'cancelled', 'rejected'
            exchange_order_id VARCHAR(100),
            exchange_trade_id VARCHAR(100),
            commission DECIMAL(20,8),
            commission_asset VARCHAR(10),
            executed_at TIMESTAMPTZ,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            updated_at TIMESTAMPTZ DEFAULT NOW()
        );
        """
        await conn.execute(trade_execution_table)
        logger.info("✅ Created trade_executions table")
        
        # Create performance metrics table
        performance_metrics_table = """
        CREATE TABLE IF NOT EXISTS performance_metrics (
            timestamp TIMESTAMPTZ NOT NULL,
            metric_type VARCHAR(50) NOT NULL,
            metric_name VARCHAR(100) NOT NULL,
            metric_value DECIMAL(20,8) NOT NULL,
            metric_unit VARCHAR(20),
            symbol VARCHAR(20),
            timeframe VARCHAR(10),
            metadata JSONB,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
        """
        await conn.execute(performance_metrics_table)
        logger.info("✅ Created performance_metrics table")
        
        # Create system alerts table
        system_alerts_table = """
        CREATE TABLE IF NOT EXISTS system_alerts (
            alert_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            alert_type VARCHAR(50) NOT NULL,
            severity VARCHAR(20) NOT NULL,  -- 'info', 'warning', 'error', 'critical'
            title VARCHAR(200) NOT NULL,
            message TEXT NOT NULL,
            component VARCHAR(100),
            symbol VARCHAR(20),
            metadata JSONB,
            acknowledged BOOLEAN DEFAULT FALSE,
            acknowledged_by VARCHAR(100),
            acknowledged_at TIMESTAMPTZ,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
        """
        await conn.execute(system_alerts_table)
        logger.info("✅ Created system_alerts table")
        
        # Create ML model performance table
        ml_model_performance_table = """
        CREATE TABLE IF NOT EXISTS ml_model_performance (
            timestamp TIMESTAMPTZ NOT NULL,
            model_name VARCHAR(100) NOT NULL,
            model_version VARCHAR(50) NOT NULL,
            accuracy DECIMAL(8,4),
            precision DECIMAL(8,4),
            recall DECIMAL(8,4),
            f1_score DECIMAL(8,4),
            auc_score DECIMAL(8,4),
            confusion_matrix JSONB,
            feature_importance JSONB,
            training_time_seconds DECIMAL(10,2),
            inference_time_ms DECIMAL(10,2),
            model_size_mb DECIMAL(10,2),
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
        """
        await conn.execute(ml_model_performance_table)
        logger.info("✅ Created ml_model_performance table")
        
        # Create TimescaleDB hypertables
        try:
            hypertables = [
                "SELECT create_hypertable('live_market_data', 'timestamp', if_not_exists => TRUE, chunk_time_interval => INTERVAL '1 minute');",
                "SELECT create_hypertable('order_book_data', 'timestamp', if_not_exists => TRUE, chunk_time_interval => INTERVAL '1 minute');",
                "SELECT create_hypertable('performance_metrics', 'timestamp', if_not_exists => TRUE, chunk_time_interval => INTERVAL '5 minutes');",
                "SELECT create_hypertable('ml_model_performance', 'timestamp', if_not_exists => TRUE, chunk_time_interval => INTERVAL '1 hour');"
            ]
            for hypertable in hypertables:
                await conn.execute(hypertable)
            logger.info("✅ Created TimescaleDB hypertables")
        except Exception as e:
            logger.warning(f"⚠️ Hypertable creation warnings: {e}")
        
        # Create indexes
        indexes = [
            # Live market data indexes
            "CREATE INDEX IF NOT EXISTS idx_live_market_data_symbol_timestamp ON live_market_data (symbol, timestamp DESC);",
            "CREATE INDEX IF NOT EXISTS idx_live_market_data_price ON live_market_data (price);",
            
            # Order book indexes
            "CREATE INDEX IF NOT EXISTS idx_order_book_symbol_side_timestamp ON order_book_data (symbol, side, timestamp DESC);",
            "CREATE INDEX IF NOT EXISTS idx_order_book_price ON order_book_data (price);",
            
            # Trade execution indexes
            "CREATE INDEX IF NOT EXISTS idx_trade_executions_signal_id ON trade_executions (signal_id);",
            "CREATE INDEX IF NOT EXISTS idx_trade_executions_symbol_status ON trade_executions (symbol, status);",
            "CREATE INDEX IF NOT EXISTS idx_trade_executions_executed_at ON trade_executions (executed_at DESC);",
            
            # Performance metrics indexes
            "CREATE INDEX IF NOT EXISTS idx_performance_metrics_type_name ON performance_metrics (metric_type, metric_name);",
            "CREATE INDEX IF NOT EXISTS idx_performance_metrics_symbol ON performance_metrics (symbol);",
            
            # System alerts indexes
            "CREATE INDEX IF NOT EXISTS idx_system_alerts_severity ON system_alerts (severity);",
            "CREATE INDEX IF NOT EXISTS idx_system_alerts_acknowledged ON system_alerts (acknowledged);",
            "CREATE INDEX IF NOT EXISTS idx_system_alerts_created_at ON system_alerts (created_at DESC);",
            
            # ML model performance indexes
            "CREATE INDEX IF NOT EXISTS idx_ml_model_performance_name_version ON ml_model_performance (model_name, model_version);",
            "CREATE INDEX IF NOT EXISTS idx_ml_model_performance_accuracy ON ml_model_performance (accuracy DESC);"
        ]
        
        for index in indexes:
            try:
                await conn.execute(index)
                logger.info(f"✅ Created index: {index[:50]}...")
            except Exception as e:
                logger.warning(f"⚠️ Could not create index: {e}")
        
        # Insert sample data
        now = datetime.now()
        
        # Sample live market data
        live_market_data = [
            (now, 'BTC/USDT', 45000.0, 1000000.0, 44999.0, 45001.0, 2.0, 45500.0, 44500.0, 500.0, 1.12, 850000000000, 19000000, 21000000, 21000000),
            (now, 'ETH/USDT', 2800.0, 500000.0, 2799.5, 2800.5, 1.0, 2850.0, 2750.0, 50.0, 1.82, 350000000000, 120000000, 120000000, 120000000),
            (now, 'SOL/USDT', 95.0, 200000.0, 94.9, 95.1, 0.2, 96.0, 94.0, 1.0, 1.06, 40000000000, 400000000, 400000000, 400000000)
        ]
        
        await conn.executemany("""
            INSERT INTO live_market_data (
                timestamp, symbol, price, volume, bid, ask, spread, high_24h, low_24h,
                change_24h, change_percent_24h, market_cap, circulating_supply, total_supply, max_supply
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15)
        """, live_market_data)
        logger.info("✅ Inserted sample live market data")
        
        # Sample performance metrics
        performance_data = [
            (now, 'trading', 'win_rate', 0.75, 'percentage', 'BTC/USDT', '1h'),
            (now, 'trading', 'profit_factor', 2.5, 'ratio', 'BTC/USDT', '1h'),
            (now, 'system', 'api_latency', 150.0, 'milliseconds', None, None),
            (now, 'system', 'memory_usage', 512.0, 'megabytes', None, None),
            (now, 'ml', 'model_accuracy', 0.85, 'percentage', None, None)
        ]
        
        await conn.executemany("""
            INSERT INTO performance_metrics (
                timestamp, metric_type, metric_name, metric_value, metric_unit, symbol, timeframe
            ) VALUES ($1, $2, $3, $4, $5, $6, $7)
        """, performance_data)
        logger.info("✅ Inserted sample performance metrics")
        
        # Sample system alerts (only if table has timestamp column)
        try:
            alert_data = [
                (now, 'performance', 'info', 'System Performance', 'All systems operating normally', 'monitoring', None),
                (now, 'trading', 'warning', 'Low Win Rate', 'Win rate dropped below 60% for BTC/USDT', 'trading_engine', 'BTC/USDT'),
                (now, 'system', 'info', 'Data Collection', 'Market data collection completed successfully', 'data_collection', None)
            ]
            
            await conn.executemany("""
                INSERT INTO system_alerts (
                    timestamp, alert_type, severity, title, message, component, symbol
                ) VALUES ($1, $2, $3, $4, $5, $6, $7)
            """, alert_data)
            logger.info("✅ Inserted sample system alerts")
        except Exception as e:
            logger.warning(f"⚠️ Could not insert system alerts: {e}")
        
        await conn.close()
        logger.info("✅ Live market data integration migration completed successfully")
        
    except Exception as e:
        logger.error(f"❌ Error creating live market data tables: {e}")
        raise

async def main():
    """Main function to run the migration"""
    logger.info("🚀 Starting live market data integration migration...")
    await create_live_market_data_tables()
    logger.info("✅ Live market data integration migration completed!")

if __name__ == "__main__":
    asyncio.run(main())
