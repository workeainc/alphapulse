# ✅ Live Data Collection & Monitoring - FIXED

## 🔴 **CRITICAL: Backend Needs Restart**

The backend is currently running **OLD code** without the new features. You need to restart it.

---

## **🔧 What Was Fixed**

### **1. Database Storage for New Candles** ✅
- **Location**: `apps/backend/main.py` lines 493-520
- **What it does**: Every new 1m candle from Binance WebSocket is now stored in database
- **Result**: Data persists and survives restarts

### **2. Workflow Monitoring API** ✅
- **New Endpoint**: `GET /api/system/workflow`
- **What it shows**:
  - Candles received/stored count
  - Last candle time per symbol
  - Indicator calculation status
  - 9-head consensus votes
  - Signal generation stats
  - Recent workflow steps

### **3. Frontend Dashboard Component** ✅
- **New Component**: `apps/web/src/components/workflow/WorkflowMonitor.tsx`
- **Auto-fetches** workflow status every 5 seconds
- **Shows error** if backend needs restart
- **Displays live updates** via WebSocket

---

## **🚀 RESTART INSTRUCTIONS**

### **Step 1: Stop Current Backend**

Find the terminal running the backend and press:
```
Ctrl + C
```

### **Step 2: Start Backend with New Code**

```powershell
cd apps\backend
python main.py
```

### **Step 3: Wait for Connection**

Look for these messages in backend logs:
```
✓ Database connection pool created
✓ TechnicalIndicatorAggregator initialized
✓ ModelHeadsManager initialized (9 heads: Technical, Volume with CVD...)
✓ Live market connector started (Binance WebSocket)
Connected to Binance! Streaming 10 data feeds
```

### **Step 4: Verify Data Flow**

Within 1-2 minutes, you should see in backend logs:
```
💾 Stored new candle: BTCUSDT 1m @ 2025-10-29...
📊 Added BTCUSDT 1m candle to indicator calculator
🔄 Processed BTCUSDT 1m candle through MTF manager
```

---

## **📊 What You'll See in Dashboard**

### **After Backend Restart:**

1. **Workflow Monitor Panel** (left side of dashboard):
   - ✅ **Data Collection**: Shows candles received/stored
   - ✅ **Indicator Calculation**: Shows calculation count
   - ✅ **9-Head Consensus**: Shows consensus votes
   - ✅ **Signal Generation**: Shows scans/signals/rejection rate
   - ✅ **Last Candle Times**: Shows how recent (should be < 60s)

2. **Real-Time Updates**:
   - Dashboard refreshes every 5 seconds
   - WebSocket sends updates when events occur
   - Recent activity log shows last 10 steps

3. **When Candle Closes**:
   - Indicator calculation count increases
   - Consensus calculation count increases
   - Head votes appear (if consensus achieved)
   - Signal may be generated (rare: 1-2% pass rate)

---

## **🔍 Verification Steps**

### **1. Check API Directly:**
```powershell
# Should work after restart
curl http://localhost:8000/api/system/workflow
```

**Expected Response:**
```json
{
  "workflow_status": {
    "data_collection": {
      "candles_received": 125,
      "candles_stored": 123,
      "status": "active",
      "time_since_last_candle": {
        "BTCUSDT": {"seconds": 45, "status": "realtime"},
        "ETHUSDT": {"seconds": 47, "status": "realtime"}
      }
    },
    ...
  }
}
```

### **2. Check Database:**
```sql
-- See new candles being stored
SELECT COUNT(*), source, MAX(timestamp) as latest
FROM ohlcv_data 
WHERE symbol='BTCUSDT' AND timeframe='1m'
GROUP BY source
ORDER BY source;

-- Should show:
-- historical_1year: ~525,600 candles
-- websocket: Increasing count (grows every minute)
```

### **3. Check Dashboard:**
- Open: http://localhost:3000
- Look at "🔴 Live Workflow Monitor"
- Should show **"LIVE"** badge in green
- Candles received should increase every minute
- Last candle times should show < 60 seconds ago

---

## **📈 Expected Data Flow**

### **Every Minute:**
```
Binance WebSocket → New 1m candle arrives
  ↓
✅ Stored in database (ohlcv_data table, source='websocket')
  ↓
✅ Added to indicator calculator buffer
  ↓
✅ Processed through MTF manager
  ↓
✅ Dashboard updates (candles_received +1)
```

### **When Candle Closes (Every 1m, 5m, 15m, 1h):**
```
Candle closes → on_candle_complete()
  ↓
✅ Calculate 69 indicators
  ↓
✅ Run 9-head consensus
  ↓
✅ Quality gates check
  ↓
✅ Signal generated (if all pass)
  ↓
✅ Dashboard updates (shows consensus votes)
```

---

## **⚠️ Troubleshooting**

### **If Dashboard Shows Error:**

**Error**: "Backend needs restart - workflow endpoint not available"

**Solution**: Backend is running old code
1. Stop backend (Ctrl+C)
2. Restart: `cd apps\backend && python main.py`
3. Wait for "Connected to Binance!" message

---

### **If Candles Received = 0:**

**Possible Causes:**
1. Binance WebSocket not connected
   - **Check**: Backend logs should show "Connected to Binance!"
   - **Fix**: Check internet connection

2. Backend not running new code
   - **Check**: `python check_backend_status.py`
   - **Fix**: Restart backend

3. WebSocket connection error
   - **Check**: Backend logs for WebSocket errors
   - **Fix**: Restart backend, check firewall

---

### **If Candles Received But Not Stored:**

**Possible Causes:**
1. Database connection error
   - **Check**: `GET /health` should show database: "connected"
   - **Fix**: Check database credentials in `DB_CONFIG`

2. Unique constraint error
   - **Check**: Backend logs for database errors
   - **Fix**: Run `python scripts/fix_unique_constraint.py`

---

## **✅ Success Indicators**

You'll know it's working when:

1. ✅ Backend logs show "💾 Stored new candle" every minute
2. ✅ Dashboard shows candles_received > 0
3. ✅ Last candle times show < 60 seconds ago
4. ✅ Workflow monitor shows "LIVE" status
5. ✅ Consensus calculations increase when candles close
6. ✅ Database shows growing count in `websocket` source

---

## **🎯 Next Steps After Restart**

1. **Monitor for 5 minutes** to verify:
   - Candles received increases every minute
   - Candles stored matches received
   - Last candle times stay recent

2. **Watch for candle closes**:
   - Indicator calculations increase
   - Consensus votes appear
   - Signal may generate (rare)

3. **Check database**:
   - Query shows both `historical_1year` and `websocket` sources
   - `websocket` count grows every minute

---

**Once backend is restarted, the dashboard will show live data aggregation and collection!** 🚀

