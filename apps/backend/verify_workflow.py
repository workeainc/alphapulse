import requests
import json

response = requests.get('http://localhost:8000/api/system/workflow', timeout=5)
data = response.json()

print("\n" + "="*60)
print("WORKFLOW STATUS VERIFICATION")
print("="*60)

dc = data['workflow_status']['data_collection']
print(f"\n1️⃣  Data Collection: {dc['status'].upper()}")
print(f"   Candles Received: {dc['candles_received']}")
print(f"   Candles Stored: {dc['candles_stored']}")

print(f"\n   Last Candle Times:")
for symbol, info in dc['time_since_last_candle'].items():
    status_color = "✅" if info['status'] == 'realtime' else "⚠️" if info['status'] == 'delayed' else "❌"
    print(f"   {status_color} {symbol}: {info['seconds']}s ago ({info['status']})")

ic = data['workflow_status']['indicator_calculation']
print(f"\n2️⃣  Indicator Calculation: {ic['status'].upper()}")
print(f"   Calculations: {ic['calculations_performed']}")

cs = data['workflow_status']['consensus_system']
print(f"\n3️⃣  9-Head Consensus: {cs['status'].upper()}")
print(f"   Calculations: {cs['calculations_performed']}")

sg = data['workflow_status']['signal_generation']
print(f"\n4️⃣  Signal Generation: {sg['status'].upper()}")
print(f"   Scans: {sg['scans_performed']}")
print(f"   Signals: {sg['signals_generated']}")
print(f"   Rejection Rate: {sg['rejection_rate']}")

print("\n" + "="*60)
print("✅ DASHBOARD SHOULD SHOW THIS DATA!")
print("="*60)

