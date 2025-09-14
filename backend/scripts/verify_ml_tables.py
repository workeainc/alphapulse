#!/usr/bin/env python3
"""
Verify ML pipeline tables were created successfully
"""

import psycopg2
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def verify_ml_tables():
    """Verify ML pipeline tables exist and are properly configured"""
    
    try:
        # Connect to database
        conn = psycopg2.connect(
            host='localhost',
            port=5432,
            database='alphapulse',
            user='alpha_emon',
            password='Emon_@17711'
        )
        
        cur = conn.cursor()
        
        # Check if tables exist
        logger.info("🔍 Verifying ML pipeline tables...")
        
        tables = ['ml_predictions', 'ml_signals', 'ml_model_performance']
        
        for table in tables:
            cur.execute("SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = %s);", (table,))
            exists = cur.fetchone()[0]
            if exists:
                logger.info(f"✅ Table '{table}' exists")
            else:
                logger.error(f"❌ Table '{table}' does not exist")
                return False
        
        # Check if tables are hypertables
        logger.info("🔍 Checking hypertable status...")
        
        for table in tables:
            try:
                cur.execute("SELECT EXISTS (SELECT FROM timescaledb_information.hypertables WHERE hypertable_name = %s);", (table,))
                is_hypertable = cur.fetchone()[0]
                if is_hypertable:
                    logger.info(f"✅ Table '{table}' is a TimescaleDB hypertable")
                else:
                    logger.info(f"ℹ️  Table '{table}' is a regular table")
            except Exception as e:
                logger.info(f"ℹ️  Could not check hypertable status: {e}")
        
        # Check table structure
        logger.info("📋 Table structure summary:")
        for table in tables:
            cur.execute("SELECT COUNT(*) FROM information_schema.columns WHERE table_name = %s;", (table,))
            column_count = cur.fetchone()[0]
            logger.info(f"   {table}: {column_count} columns")
        
        # Check indexes
        logger.info("📈 Checking indexes...")
        for table in tables:
            cur.execute("""
                SELECT COUNT(*) 
                FROM pg_indexes 
                WHERE tablename = %s;
            """, (table,))
            index_count = cur.fetchone()[0]
            logger.info(f"   {table}: {index_count} indexes")
        
        cur.close()
        conn.close()
        
        logger.info("🎉 All ML pipeline tables verified successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error verifying tables: {e}")
        return False

if __name__ == "__main__":
    success = verify_ml_tables()
    exit(0 if success else 1)
