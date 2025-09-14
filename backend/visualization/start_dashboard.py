#!/usr/bin/env python3
"""
Dashboard Startup Script for AlphaPulse
Week 8: Real-Time Dashboards & Reporting

Quick start script for launching the dashboard with various configurations.
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent.parent
sys.path.append(str(backend_path))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_dependencies():
    """Check if required dependencies are installed"""
    missing_deps = []
    
    try:
        import plotly
        logger.info("✅ Plotly available")
    except ImportError:
        missing_deps.append("plotly")
        logger.warning("❌ Plotly not found")
    
    try:
        import dash
        logger.info("✅ Dash available")
    except ImportError:
        missing_deps.append("dash")
        logger.warning("❌ Dash not found")
    
    try:
        import flask
        logger.info("✅ Flask available")
    except ImportError:
        missing_deps.append("flask")
        logger.warning("❌ Flask not found")
    
    if missing_deps:
        logger.error(f"❌ Missing dependencies: {', '.join(missing_deps)}")
        logger.info("Install with: pip install -r backend/visualization/requirements.txt")
        return False
    
    return True

def start_dashboard(mode='flask', host='localhost', port=8050, debug=False):
    """Start the dashboard in specified mode"""
    try:
        if mode == 'flask':
            from backend.visualization.dashboard_server import DashboardServer
            
            config = {
                'host': host,
                'port': port,
                'debug': debug,
                'db_config': {
                    'host': os.getenv('DB_HOST', 'localhost'),
                    'port': int(os.getenv('DB_PORT', 5432)),
                    'database': os.getenv('DB_NAME', 'alphapulse'),
                    'user': os.getenv('DB_USER', 'postgres'),
                    'password': os.getenv('DB_PASSWORD', 'password')
                }
            }
            
            server = DashboardServer(config)
            logger.info(f"🚀 Starting Flask dashboard server on {host}:{port}")
            # Initialize and run server
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(server.initialize())
            server.run()
            
        elif mode == 'dash':
            from backend.visualization.dashboard_service import DashboardService
            
            # Mock database connection for testing
            class MockDB:
                async def fetch(self, *args):
                    return []
                async def connect(self):
                    pass
                async def close(self):
                    pass
            
            dashboard = DashboardService(MockDB(), host=host, port=port)
            logger.info(f"🚀 Starting Dash dashboard on {host}:{port}")
            dashboard.run()
            
        else:
            logger.error(f"❌ Unknown mode: {mode}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error starting dashboard: {e}")
        return False
    
    return True

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Start AlphaPulse Dashboard')
    parser.add_argument('--mode', choices=['flask', 'dash'], default='flask',
                       help='Dashboard mode (flask or dash)')
    parser.add_argument('--host', default='localhost',
                       help='Host to bind to')
    parser.add_argument('--port', type=int, default=8050,
                       help='Port to bind to')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode')
    parser.add_argument('--check-deps', action='store_true',
                       help='Check dependencies only')
    
    args = parser.parse_args()
    
    logger.info("🚀 AlphaPulse Dashboard Startup")
    logger.info("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        if args.check_deps:
            sys.exit(1)
        logger.warning("⚠️ Some dependencies missing, but continuing...")
    
    if args.check_deps:
        logger.info("✅ Dependency check complete")
        return
    
    # Start dashboard
    success = start_dashboard(
        mode=args.mode,
        host=args.host,
        port=args.port,
        debug=args.debug
    )
    
    if success:
        logger.info("✅ Dashboard started successfully")
        logger.info(f"🌐 Access at: http://{args.host}:{args.port}")
    else:
        logger.error("❌ Failed to start dashboard")
        sys.exit(1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("⚠️ Dashboard stopped by user")
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        sys.exit(1)
