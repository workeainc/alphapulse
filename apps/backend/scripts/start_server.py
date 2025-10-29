#!/usr/bin/env python3
"""
Simple server startup script
"""
import uvicorn
from src.app.main_intelligent import app

if __name__ == "__main__":
    print("Starting AlphaPlus Trading System...")
    print(f"App title: {app.title}")
    print(f"App version: {app.version}")
    
    # List chart routes
    routes = [route for route in app.routes if hasattr(route, 'path')]
    chart_routes = [route for route in routes if 'charts' in route.path]
    print(f"Chart routes found: {len(chart_routes)}")
    for route in chart_routes:
        print(f"  {route.path}")
    
    print("Starting server on http://0.0.0.0:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
