#!/usr/bin/env python3
"""
Test to directly call chart endpoint functions
"""
import asyncio
from src.app.main_intelligent import app
from fastapi.testclient import TestClient

async def test_direct_function():
    """Test the chart endpoint functions directly"""
    client = TestClient(app)
    
    # Test health endpoint
    response = client.get("/health")
    print(f"Health: {response.status_code}")
    print(f"Health data: {response.json()}")
    
    # Test chart endpoints
    endpoints = [
        "/api/charts/candlestick/BTCUSDT",
        "/api/charts/technical-indicators/BTCUSDT",
        "/api/charts/real-time/BTCUSDT"
    ]
    
    for endpoint in endpoints:
        try:
            response = client.get(endpoint)
            print(f"{endpoint}: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                print(f"Data: {data}")
            else:
                print(f"Error: {response.text}")
        except Exception as e:
            print(f"{endpoint} error: {e}")

if __name__ == "__main__":
    asyncio.run(test_direct_function())
