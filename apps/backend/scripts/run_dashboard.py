#!/usr/bin/env python3
"""
Launcher script for the Production Monitoring Dashboard
"""

import asyncio
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(__file__))

async def main():
    """Main function to start the dashboard"""
    try:
        print("🚀 Starting AlphaPulse Performance Dashboard...")
        
        # Import and start dashboard
        from src.app.services.monitoring_dashboard import MonitoringDashboard
        
        dashboard = MonitoringDashboard()
        
        # Start the dashboard
        await dashboard.start(host="0.0.0.0", port=8000)
        
    except KeyboardInterrupt:
        print("\n🛑 Dashboard stopped by user")
    except Exception as e:
        print(f"❌ Failed to start dashboard: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
