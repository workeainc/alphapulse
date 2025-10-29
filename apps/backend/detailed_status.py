import requests
from datetime import datetime

response = requests.get('http://localhost:8000/api/system/workflow', timeout=5)
data = response.json()

print("\n" + "="*80)
print("🚀 ALPHAPULSE LIVE WORKFLOW STATUS")
print("="*80)
print(f"Timestamp: {datetime.now().strftime('%H:%M:%S')}")

dc = data['workflow_status']['data_collection']
ic = data['workflow_status']['indicator_calculation']
cs = data['workflow_status']['consensus_system']
sg = data['workflow_status']['signal_generation']

# Summary
print(f"\n📊 SUMMARY:")
print(f"   Candles: {dc['candles_received']} received, {dc['candles_stored']} stored")
print(f"   Indicators: {ic['calculations_performed']} calculations")
print(f"   Consensus: {cs['calculations_performed']} votes")
print(f"   Scans: {sg['scans_performed']}, Signals: {sg['signals_generated']}")

# Data Collection Detail
print(f"\n1️⃣  DATA COLLECTION: {dc['status'].upper()}")
for symbol, info in dc['time_since_last_candle'].items():
    status_emoji = "✅" if info['status'] == 'realtime' else "❌"
    print(f"   {status_emoji} {symbol}: {info['status']}")

# Buffer Status Detail
print(f"\n2️⃣  INDICATOR BUFFERS:")
for symbol, timeframes in ic['buffer_status'].items():
    candles_1m = timeframes.get('1m', {}).get('candles_in_buffer', 0)
    print(f"   {symbol}: {candles_1m} 1m candles in buffer")

# Consensus Status
print(f"\n3️⃣  9-HEAD CONSENSUS:")
if cs['calculations_performed'] > 0:
    print(f"   ✅ {cs['calculations_performed']} consensus calculations completed")
    
    if cs.get('last_consensus_votes'):
        print(f"\n   Recent Votes:")
        for key, vote in list(cs['last_consensus_votes'].items())[:2]:
            consensus = vote.get('consensus', {})
            print(f"\n   {key}:")
            print(f"      Direction: {consensus.get('direction', 'FLAT')}")
            print(f"      Agreement: {consensus.get('agreeing_heads', 0)}/{consensus.get('total_heads', 9)} heads")
            print(f"      Confidence: {consensus.get('confidence', 0)*100:.1f}%")
            
            # Show individual head votes
            if vote.get('votes'):
                print(f"      Head Breakdown:")
                for head, data in list(vote['votes'].items())[:9]:
                    direction = data.get('direction', 'FLAT')
                    conf = data.get('confidence', 0)
                    print(f"         {head}: {direction} ({conf*100:.0f}%)")
else:
    print(f"   ⏳ Waiting for more candles (need ~20-50 for full indicators)")
    print(f"   Current: 2 candles per symbol")

# Signal Generation
print(f"\n4️⃣  SIGNAL GENERATION:")
print(f"   Scans: {sg['scans_performed']}")
print(f"   Signals: {sg['signals_generated']}")
print(f"   Rejection: {sg['rejection_rate']} (normal: 98-99%)")

print("\n" + "="*80)
print("Dashboard: http://localhost:3000")
print("="*80)

