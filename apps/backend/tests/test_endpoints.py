#!/usr/bin/env python3
"""
Test script to verify endpoints are working
"""
import asyncio
import aiohttp
import json

async def test_endpoints():
    """Test the chart endpoints"""
    base_url = "http://localhost:8000"
    
    async with aiohttp.ClientSession() as session:
        # Test health endpoint
        try:
            async with session.get(f"{base_url}/health") as response:
                print(f"Health endpoint: {response.status}")
                if response.status == 200:
                    data = await response.json()
                    print(f"Health data: {data}")
        except Exception as e:
            print(f"Health endpoint error: {e}")
        
        # Test chart endpoints
        endpoints = [
            "/api/charts/candlestick/BTCUSDT",
            "/api/charts/technical-indicators/BTCUSDT",
            "/api/charts/real-time/BTCUSDT"
        ]
        
        for endpoint in endpoints:
            try:
                async with session.get(f"{base_url}{endpoint}") as response:
                    print(f"{endpoint}: {response.status}")
                    if response.status == 200:
                        data = await response.json()
                        print(f"Data: {json.dumps(data, indent=2)[:200]}...")
                    else:
                        text = await response.text()
                        print(f"Error: {text}")
            except Exception as e:
                print(f"{endpoint} error: {e}")

if __name__ == "__main__":
    asyncio.run(test_endpoints())
