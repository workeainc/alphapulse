#!/usr/bin/env python3
"""
Quick script to check backend status and verify WebSocket connection
"""

import requests
import json
import sys

def check_backend():
    base_url = "http://localhost:8000"
    
    print("=" * 60)
    print("BACKEND STATUS CHECK")
    print("=" * 60)
    
    # Check health
    try:
        health = requests.get(f"{base_url}/health", timeout=5)
        if health.status_code == 200:
            data = health.json()
            print("✅ Backend is running")
            print(f"   Database: {data.get('database', {}).get('status', 'unknown')}")
            print(f"   WebSocket: {data.get('websocket', {}).get('status', 'unknown')}")
            stats = data.get('statistics', {})
            print(f"   Scans: {stats.get('scans_performed', 0)}")
            print(f"   Signals: {stats.get('signals_generated', 0)}")
        else:
            print(f"❌ Backend returned status {health.status_code}")
            return
    except Exception as e:
        print(f"❌ Cannot connect to backend: {e}")
        print("   Make sure backend is running: python main.py")
        return
    
    # Check workflow status
    try:
        workflow = requests.get(f"{base_url}/api/system/workflow", timeout=5)
        if workflow.status_code == 200:
            data = workflow.json()
            ws = data.get('workflow_status', {})
            
            print("\n" + "=" * 60)
            print("WORKFLOW STATUS")
            print("=" * 60)
            
            # Data Collection
            dc = ws.get('data_collection', {})
            print(f"\n1️⃣  Data Collection: {dc.get('status', 'unknown').upper()}")
            print(f"   Candles Received: {dc.get('candles_received', 0)}")
            print(f"   Candles Stored: {dc.get('candles_stored', 0)}")
            
            if dc.get('time_since_last_candle'):
                print("\n   Last Candle Times:")
                for symbol, info in dc.get('time_since_last_candle', {}).items():
                    status = info.get('status', 'unknown')
                    seconds = info.get('seconds', -1)
                    if seconds >= 0:
                        print(f"      {symbol}: {seconds}s ago ({status})")
                    else:
                        print(f"      {symbol}: No data")
            
            # Indicator Calculation
            ic = ws.get('indicator_calculation', {})
            print(f"\n2️⃣  Indicator Calculation: {ic.get('status', 'unknown').upper()}")
            print(f"   Calculations: {ic.get('calculations_performed', 0)}")
            
            # Consensus
            cs = ws.get('consensus_system', {})
            print(f"\n3️⃣  9-Head Consensus: {cs.get('status', 'unknown').upper()}")
            print(f"   Calculations: {cs.get('calculations_performed', 0)}")
            
            # Signal Generation
            sg = ws.get('signal_generation', {})
            print(f"\n4️⃣  Signal Generation: {sg.get('status', 'unknown').upper()}")
            print(f"   Scans: {sg.get('scans_performed', 0)}")
            print(f"   Signals: {sg.get('signals_generated', 0)}")
            print(f"   Rejection Rate: {sg.get('rejection_rate', '0%')}")
            
            # Diagnosis
            print("\n" + "=" * 60)
            print("DIAGNOSIS")
            print("=" * 60)
            
            if dc.get('candles_received', 0) == 0:
                print("⚠️  NO CANDLES RECEIVED")
                print("   → Backend may need restart to apply new code")
                print("   → Binance WebSocket may not be connected")
                print("   → Check backend logs for 'Connected to Binance' message")
            elif dc.get('candles_stored', 0) < dc.get('candles_received', 0):
                print("⚠️  CANDLES NOT BEING STORED")
                print(f"   → Received: {dc.get('candles_received')}, Stored: {dc.get('candles_stored')}")
                print("   → Check database connection and logs")
            else:
                print("✅ Data collection working!")
                
        else:
            print(f"❌ Workflow endpoint returned status {workflow.status_code}")
            print(f"   Response: {workflow.text[:200]}")
    except Exception as e:
        print(f"❌ Error checking workflow: {e}")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    check_backend()

