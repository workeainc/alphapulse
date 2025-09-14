#!/usr/bin/env python3
"""
Advanced Analytics Dashboard Launcher
Starts the Advanced Analytics Dashboard service
"""

import uvicorn
import logging
import sys
import os

# Add the backend directory to the path
sys.path.append(os.path.dirname(__file__))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('analytics_dashboard.log')
    ]
)

logger = logging.getLogger(__name__)

def main():
    """Main function to start the analytics dashboard"""
    try:
        logger.info("🚀 Starting AlphaPulse Advanced Analytics Dashboard...")
        
        # Import the service
        from app.services.analytics_dashboard_service import app
        
        # Start the server
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8085,
            log_level="info",
            reload=False,
            access_log=True
        )
        
    except KeyboardInterrupt:
        logger.info("🛑 Analytics Dashboard stopped by user")
    except Exception as e:
        logger.error(f"❌ Error starting Analytics Dashboard: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
