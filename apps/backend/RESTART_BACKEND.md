# 🔄 RESTART BACKEND - IMPORTANT!

## **Backend needs restart to apply new code**

The current backend is running **OLD code** without:
- ❌ Database storage for new candles
- ❌ Workflow monitoring endpoint (`/api/system/workflow`)
- ❌ Real-time workflow tracking

---

## **Steps to Restart:**

### **1. Stop Current Backend**
- Press `Ctrl+C` in the terminal where backend is running
- OR find and kill the Python process

### **2. Start Backend with New Code**
```powershell
cd apps\backend
python main.py
```

### **3. Look for These Messages:**

```
✓ Database connection pool created
✓ TechnicalIndicatorAggregator initialized (50+ indicators)
✓ ModelHeadsManager initialized (9 heads...)
✓ Live market connector started (Binance WebSocket)
Connected to Binance! Streaming 10 data feeds
```

### **4. Watch for Live Data:**

After 1-2 minutes, you should see:
```
💾 Stored new candle: BTCUSDT 1m @ 2025-10-29...
🕯️ Candle closed: BTCUSDT 1m @ 65000
📈 Calculating indicators for BTCUSDT 1m...
🎯 Processing through 9-head consensus for BTCUSDT 1m...
```

---

## **Verify It's Working:**

### **Check API:**
```powershell
# This should work after restart
curl http://localhost:8000/api/system/workflow
```

### **Check Dashboard:**
- Open: http://localhost:3000
- Look at "🔴 Live Workflow Monitor"
- Should show:
  - ✅ Candles Received: > 0
  - ✅ Candles Stored: > 0
  - ✅ Last Candle Times: Recent (< 60s ago)

---

## **If Still Not Working:**

1. **Check Binance Connection:**
   - Backend logs should show "Connected to Binance!"
   - If not, check internet connection

2. **Check Database:**
   - Verify connection: `GET /health` should show database: "connected"
   - Check credentials in `DB_CONFIG`

3. **Check Code is Latest:**
   ```powershell
   # Verify main.py has database storage code
   cd apps\backend
   Select-String -Path main.py -Pattern "Store in database.*NEW.*CRITICAL FIX"
   ```

