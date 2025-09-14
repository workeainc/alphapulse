#!/usr/bin/env python3
"""
Performance Monitoring Starter Script
Starts the comprehensive performance monitoring system for AlphaPlus
"""

import asyncio
import logging
import sys
import os
from pathlib import Path

# Add the backend directory to the Python path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from app.monitoring.performance_monitor import PerformanceMonitor
from database.connection import TimescaleDBConnection

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/performance_monitoring.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

async def main():
    """Main function to start performance monitoring"""
    try:
        logger.info("🚀 Starting AlphaPlus Performance Monitoring System...")
        
        # Initialize database connection
        db_connection = TimescaleDBConnection()
        await db_connection.initialize()
        logger.info("✅ Database connection established")
        
        # Initialize performance monitor
        performance_monitor = PerformanceMonitor(db_connection.pool)
        logger.info("✅ Performance monitor initialized")
        
        # Start monitoring
        await performance_monitor.start_monitoring()
        logger.info("✅ Performance monitoring started successfully")
        
        # Keep the script running
        logger.info("📊 Performance monitoring is now active. Press Ctrl+C to stop.")
        
        try:
            # Run indefinitely
            while True:
                await asyncio.sleep(60)  # Check every minute
                
                # Log periodic status
                summary = performance_monitor.get_performance_summary()
                if summary:
                    logger.info("📈 Performance monitoring active - collecting metrics...")
                
        except KeyboardInterrupt:
            logger.info("🛑 Received stop signal...")
        
        finally:
            # Stop monitoring
            await performance_monitor.stop_monitoring()
            await db_connection.close()
            logger.info("✅ Performance monitoring stopped and cleaned up")
    
    except Exception as e:
        logger.error(f"❌ Error in performance monitoring: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
