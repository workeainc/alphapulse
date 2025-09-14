#!/usr/bin/env python3
"""
Test script to verify AlphaPlus API endpoints
"""

import requests
import json
import time
from datetime import datetime

BASE_URL = "http://localhost:8000"

def test_endpoint(endpoint, description):
    """Test an API endpoint"""
    try:
        print(f"\n🔍 Testing {description}...")
        response = requests.get(f"{BASE_URL}{endpoint}", timeout=10)
        
        if response.status_code == 200:
            print(f"✅ {description} - Status: {response.status_code}")
            try:
                data = response.json()
                print(f"   Response: {json.dumps(data, indent=2)}")
            except:
                print(f"   Response: {response.text[:200]}...")
        else:
            print(f"❌ {description} - Status: {response.status_code}")
            print(f"   Error: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print(f"❌ {description} - Connection failed (server not running?)")
    except Exception as e:
        print(f"❌ {description} - Error: {e}")

def main():
    """Main test function"""
    print("🚀 AlphaPlus API Endpoint Tests")
    print("=" * 50)
    print(f"Testing server at: {BASE_URL}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Test endpoints
    test_endpoint("/", "Root endpoint")
    test_endpoint("/health", "Health check")
    test_endpoint("/config", "Configuration")
    test_endpoint("/services/status", "Services status")
    
    print("\n" + "=" * 50)
    print("🎉 API endpoint tests completed!")

if __name__ == "__main__":
    main()
