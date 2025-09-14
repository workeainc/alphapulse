#!/usr/bin/env python3
"""
Simple script to run the accuracy benchmarks table migration
"""

import asyncio
import logging
from sqlalchemy import text
from ..database.connection import TimescaleDBConnection

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def create_accuracy_benchmarks_table():
    """Create the model_accuracy_benchmarks table"""
    
    logger.info("🚀 Creating model_accuracy_benchmarks table...")
    
    try:
        # Initialize database connection
        db_connection = TimescaleDBConnection()
        db_connection.initialize()
        
        async with db_connection.get_async_session() as session:
            try:
                # Check if table already exists
                check_query = """
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'model_accuracy_benchmarks'
                );
                """
                result = await session.execute(text(check_query))
                table_exists = result.scalar()
                
                if table_exists:
                    logger.info("✅ model_accuracy_benchmarks table already exists")
                    return
                
                # Create the table
                create_table_sql = """
                CREATE TABLE model_accuracy_benchmarks (
                    id SERIAL PRIMARY KEY,
                    model_id VARCHAR(100) NOT NULL,
                    symbol VARCHAR(20) NOT NULL,
                    strategy_name VARCHAR(100) NOT NULL,
                    
                    -- ML Classification Metrics
                    precision FLOAT NOT NULL,
                    recall FLOAT NOT NULL,
                    f1_score FLOAT NOT NULL,
                    accuracy FLOAT NOT NULL,
                    roc_auc FLOAT NOT NULL,
                    
                    -- Trading Performance Metrics
                    win_rate FLOAT NOT NULL,
                    profit_factor FLOAT NOT NULL,
                    avg_win FLOAT NOT NULL,
                    avg_loss FLOAT NOT NULL,
                    total_return FLOAT NOT NULL,
                    sharpe_ratio FLOAT NOT NULL,
                    max_drawdown FLOAT NOT NULL,
                    
                    -- Additional Metrics
                    total_trades INTEGER NOT NULL,
                    winning_trades INTEGER NOT NULL,
                    losing_trades INTEGER NOT NULL,
                    avg_holding_period FLOAT NOT NULL,
                    risk_reward_ratio FLOAT NOT NULL,
                    
                    -- Test Configuration
                    test_period_days INTEGER NOT NULL,
                    frozen_test_set BOOLEAN NOT NULL DEFAULT TRUE,
                    
                    -- Metadata
                    evaluation_date TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    benchmark_version VARCHAR(20) NOT NULL DEFAULT 'v1.0'
                );
                """
                
                await session.execute(text(create_table_sql))
                logger.info("✅ Table created successfully")
                
                # Create indexes
                indexes = [
                    "CREATE INDEX ix_model_accuracy_benchmarks_model_id ON model_accuracy_benchmarks(model_id);",
                    "CREATE INDEX ix_model_accuracy_benchmarks_symbol ON model_accuracy_benchmarks(symbol);",
                    "CREATE INDEX ix_model_accuracy_benchmarks_strategy_name ON model_accuracy_benchmarks(strategy_name);",
                    "CREATE INDEX ix_model_accuracy_benchmarks_evaluation_date ON model_accuracy_benchmarks(evaluation_date);",
                    "CREATE INDEX ix_model_accuracy_benchmarks_model_symbol ON model_accuracy_benchmarks(model_id, symbol);",
                    "CREATE INDEX ix_model_accuracy_benchmarks_model_strategy ON model_accuracy_benchmarks(model_id, strategy_name);",
                    "CREATE INDEX ix_model_accuracy_benchmarks_symbol_date ON model_accuracy_benchmarks(symbol, evaluation_date);"
                ]
                
                for index_sql in indexes:
                    try:
                        await session.execute(text(index_sql))
                        logger.info(f"✅ Index created: {index_sql.split()[2]}")
                    except Exception as e:
                        logger.warning(f"Could not create index: {e}")
                
                # Commit the transaction
                await session.commit()
                logger.info("✅ model_accuracy_benchmarks table created successfully!")
                
            except Exception as e:
                await session.rollback()
                logger.error(f"❌ Error in transaction: {e}")
                raise
            
    except Exception as e:
        logger.error(f"❌ Error creating model_accuracy_benchmarks table: {e}")
        raise

async def main():
    """Main function"""
    await create_accuracy_benchmarks_table()

if __name__ == "__main__":
    asyncio.run(main())
