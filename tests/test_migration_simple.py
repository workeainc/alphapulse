#!/usr/bin/env python3
"""
Simple Migration Test
Test the enhanced market intelligence database migration
"""

import asyncio
import logging
import sys
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_migration():
    """Test the database migration"""
    try:
        logger.info("🔄 Testing enhanced market intelligence migration...")
        
        # Import the migration function
        sys.path.append(os.path.join(os.path.dirname(__file__), 'database', 'migrations'))
        
        from database.migrations.004_enhanced_market_intelligence_tables import create_enhanced_market_intelligence_tables
        
        # Run migration
        success = await create_enhanced_market_intelligence_tables()
        
        if success:
            logger.info("✅ Migration completed successfully!")
            return True
        else:
            logger.error("❌ Migration failed!")
            return False
            
    except Exception as e:
        logger.error(f"❌ Migration test error: {e}")
        return False

async def main():
    """Main function"""
    try:
        success = await test_migration()
        
        if success:
            logger.info("🎉 Migration test PASSED!")
            return 0
        else:
            logger.error("💥 Migration test FAILED!")
            return 1
            
    except Exception as e:
        logger.error(f"💥 Test crashed: {e}")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
