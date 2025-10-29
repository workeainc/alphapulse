#!/usr/bin/env python3
"""
Database Migration: Rename trades table to signal_recommendations
Migration Date: October 26, 2025
Purpose: Refactor from execution platform to signal recommendation engine

This migration renames the 'trades' table to 'signal_recommendations'
and updates column names to reflect the new purpose.
"""

import asyncpg
import asyncio
import logging
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TradestoRecommendationsMigration:
    """Migration to rename trades to signal_recommendations"""
    
    def __init__(self, database_url: str):
        self.database_url = database_url
        self.conn: Optional[asyncpg.Connection] = None
    
    async def connect(self):
        """Connect to database"""
        self.conn = await asyncpg.connect(self.database_url)
        logger.info("Connected to database")
    
    async def disconnect(self):
        """Disconnect from database"""
        if self.conn:
            await self.conn.close()
            logger.info("Disconnected from database")
    
    async def check_table_exists(self, table_name: str) -> bool:
        """Check if a table exists"""
        result = await self.conn.fetchval(
            """
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_schema = 'public'
                AND table_name = $1
            )
            """,
            table_name
        )
        return result
    
    async def migrate_up(self):
        """Run the migration (rename trades to signal_recommendations)"""
        logger.info("Starting migration: trades -> signal_recommendations")
        
        try:
            # Check if trades table exists
            trades_exists = await self.check_table_exists('trades')
            if not trades_exists:
                logger.warning("trades table does not exist, skipping migration")
                return
            
            # Check if signal_recommendations already exists
            recommendations_exists = await self.check_table_exists('signal_recommendations')
            if recommendations_exists:
                logger.warning("signal_recommendations table already exists, skipping migration")
                return
            
            # Start transaction
            async with self.conn.transaction():
                # Rename table
                await self.conn.execute(
                    'ALTER TABLE trades RENAME TO signal_recommendations'
                )
                logger.info("✓ Renamed table: trades -> signal_recommendations")
                
                # Rename columns
                column_renames = [
                    ('entry_price', 'suggested_entry_price'),
                    ('exit_price', 'suggested_exit_price'),
                    ('quantity', 'suggested_quantity'),
                    ('leverage', 'suggested_leverage'),
                    ('pnl', 'hypothetical_pnl'),
                    ('pnl_percentage', 'hypothetical_pnl_percentage'),
                    ('stop_loss', 'suggested_stop_loss'),
                    ('take_profit', 'suggested_take_profit'),
                    ('trailing_stop', 'suggested_trailing_stop'),
                    ('entry_time', 'recommendation_time'),
                    ('exit_time', 'expiry_time'),
                ]
                
                for old_col, new_col in column_renames:
                    try:
                        await self.conn.execute(
                            f'ALTER TABLE signal_recommendations RENAME COLUMN {old_col} TO {new_col}'
                        )
                        logger.info(f"✓ Renamed column: {old_col} -> {new_col}")
                    except asyncpg.exceptions.UndefinedColumnError:
                        logger.warning(f"Column {old_col} does not exist, skipping")
                
                # Update status values
                await self.conn.execute(
                    """
                    UPDATE signal_recommendations
                    SET status = CASE
                        WHEN status = 'open' THEN 'pending'
                        WHEN status = 'closed' THEN 'user_executed'
                        ELSE status
                    END
                    """
                )
                logger.info("✓ Updated status values")
                
                logger.info("Migration completed successfully!")
        
        except Exception as e:
            logger.error(f"Migration failed: {e}")
            raise
    
    async def migrate_down(self):
        """Rollback the migration (rename back to trades)"""
        logger.info("Rolling back migration: signal_recommendations -> trades")
        
        try:
            # Check if signal_recommendations table exists
            recommendations_exists = await self.check_table_exists('signal_recommendations')
            if not recommendations_exists:
                logger.warning("signal_recommendations table does not exist, skipping rollback")
                return
            
            # Start transaction
            async with self.conn.transaction():
                # Rename columns back
                column_renames = [
                    ('suggested_entry_price', 'entry_price'),
                    ('suggested_exit_price', 'exit_price'),
                    ('suggested_quantity', 'quantity'),
                    ('suggested_leverage', 'leverage'),
                    ('hypothetical_pnl', 'pnl'),
                    ('hypothetical_pnl_percentage', 'pnl_percentage'),
                    ('suggested_stop_loss', 'stop_loss'),
                    ('suggested_take_profit', 'take_profit'),
                    ('suggested_trailing_stop', 'trailing_stop'),
                    ('recommendation_time', 'entry_time'),
                    ('expiry_time', 'exit_time'),
                ]
                
                for old_col, new_col in column_renames:
                    try:
                        await self.conn.execute(
                            f'ALTER TABLE signal_recommendations RENAME COLUMN {old_col} TO {new_col}'
                        )
                        logger.info(f"✓ Renamed column: {old_col} -> {new_col}")
                    except asyncpg.exceptions.UndefinedColumnError:
                        logger.warning(f"Column {old_col} does not exist, skipping")
                
                # Update status values back
                await self.conn.execute(
                    """
                    UPDATE signal_recommendations
                    SET status = CASE
                        WHEN status = 'pending' THEN 'open'
                        WHEN status = 'user_executed' THEN 'closed'
                        ELSE status
                    END
                    """
                )
                logger.info("✓ Updated status values")
                
                # Rename table back
                await self.conn.execute(
                    'ALTER TABLE signal_recommendations RENAME TO trades'
                )
                logger.info("✓ Renamed table: signal_recommendations -> trades")
                
                logger.info("Rollback completed successfully!")
        
        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            raise


async def main():
    """Main migration function"""
    import os
    from dotenv import load_dotenv
    
    # Load environment variables
    load_dotenv()
    
    # Get database URL (note: @ in password is URL-encoded as %40)
    # Using port 55433 to avoid conflict with other PostgreSQL instances
    database_url = os.getenv(
        'DATABASE_URL',
        'postgresql://alpha_emon:Emon_%4017711@localhost:55433/alphapulse'
    )
    
    # Create migration instance
    migration = TradestoRecommendationsMigration(database_url)
    
    try:
        await migration.connect()
        
        # Run migration
        import sys
        if len(sys.argv) > 1 and sys.argv[1] == 'down':
            await migration.migrate_down()
        else:
            await migration.migrate_up()
    
    finally:
        await migration.disconnect()


if __name__ == '__main__':
    asyncio.run(main())

