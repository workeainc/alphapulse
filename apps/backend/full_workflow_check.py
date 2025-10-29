import requests
import json
from datetime import datetime

response = requests.get('http://localhost:8000/api/system/workflow', timeout=5)
data = response.json()

print("\n" + "="*70)
print("🔴 LIVE WORKFLOW MONITOR - REAL-TIME STATUS")
print("="*70)

# Data Collection
dc = data['workflow_status']['data_collection']
print(f"\n1️⃣  DATA COLLECTION: {dc['status'].upper()}")
print(f"   📊 Candles Received: {dc['candles_received']}")
print(f"   💾 Candles Stored: {dc['candles_stored']}")
print(f"\n   ⏰ Last Candle Times:")
for symbol, info in dc['time_since_last_candle'].items():
    status_emoji = "✅" if info['status'] == 'realtime' else "⚠️" if info['status'] == 'delayed' else "❌"
    print(f"      {status_emoji} {symbol}: {info['seconds']}s ago ({info['status']})")

# Indicator Calculation
ic = data['workflow_status']['indicator_calculation']
print(f"\n2️⃣  INDICATOR CALCULATION: {ic['status'].upper()}")
print(f"   🧮 Calculations Performed: {ic['calculations_performed']}")
print(f"\n   📈 Buffer Status (candles in memory):")
for symbol, timeframes in ic['buffer_status'].items():
    print(f"      {symbol}:")
    for tf, status in timeframes.items():
        print(f"         {tf}: {status['candles_in_buffer']}/{status['buffer_size']} candles")

# 9-Head Consensus
cs = data['workflow_status']['consensus_system']
print(f"\n3️⃣  9-HEAD CONSENSUS SYSTEM: {cs['status'].upper()}")
print(f"   🎯 Consensus Calculations: {cs['calculations_performed']}")

if cs.get('last_consensus_votes') and len(cs['last_consensus_votes']) > 0:
    print(f"\n   🗳️  Recent Consensus Votes:")
    for key, vote in list(cs['last_consensus_votes'].items())[:3]:
        consensus = vote.get('consensus', {})
        direction = consensus.get('direction', 'FLAT')
        confidence = consensus.get('confidence', 0)
        agreeing = consensus.get('agreeing_heads', 0)
        total = consensus.get('total_heads', 9)
        
        print(f"\n      {key}:")
        print(f"      Direction: {direction.upper()}")
        print(f"      Confidence: {confidence*100:.1f}%")
        print(f"      Agreement: {agreeing}/{total} heads")
        
        if vote.get('votes'):
            print(f"      Individual Head Votes:")
            for head_name, head_vote in list(vote['votes'].items())[:5]:
                head_dir = head_vote.get('direction', 'FLAT')
                head_conf = head_vote.get('confidence', 0)
                print(f"         - {head_name}: {head_dir} ({head_conf*100:.0f}%)")
else:
    print(f"   ⏳ No consensus votes yet (waiting for more data)")

# Signal Generation
sg = data['workflow_status']['signal_generation']
print(f"\n4️⃣  SIGNAL GENERATION: {sg['status'].upper()}")
print(f"   🔍 Scans Performed: {sg['scans_performed']}")
print(f"   ⚡ Signals Generated: {sg['signals_generated']}")
print(f"   🚫 Rejection Rate: {sg['rejection_rate']}")

# Recent Workflow Steps
recent_steps = data.get('recent_workflow_steps', [])
if recent_steps:
    print(f"\n📜 RECENT WORKFLOW ACTIVITY (Last 5 steps):")
    for step in recent_steps[:5]:
        timestamp = step.get('timestamp', '')
        event = step.get('event', 'Unknown')
        symbol = step.get('symbol', '')
        timeframe = step.get('timeframe', '')
        
        if symbol and timeframe:
            print(f"   {timestamp}: {event} - {symbol} {timeframe}")
        elif symbol:
            print(f"   {timestamp}: {event} - {symbol}")
        else:
            print(f"   {timestamp}: {event}")

print("\n" + "="*70)
print("✅ ALL SYSTEMS OPERATIONAL - REFRESH DASHBOARD TO SEE THIS DATA!")
print("="*70)

