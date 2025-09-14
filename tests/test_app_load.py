#!/usr/bin/env python3
"""
Test app loading
"""
from app.main_intelligent import app

print("App loaded successfully")
print(f"App title: {app.title}")
print(f"App version: {app.version}")

routes = [r for r in app.routes if hasattr(r, "path")]
print(f"Total routes: {len(routes)}")

chart_routes = [r for r in routes if "charts" in r.path]
print(f"Chart routes: {len(chart_routes)}")

for route in chart_routes:
    print(f"  {route.path}")

print("App is ready to start!")
