# 🚀 AlphaPulse Frontend - Quick Setup Guide

## Prerequisites

- **Node.js** 18+ or **Bun**
- **Backend** running on `http://localhost:8000`

---

## 📦 Installation Steps

### 1. Navigate to the Web Directory

```bash
cd apps/web
```

### 2. Install Dependencies

Using npm:
```bash
npm install
```

Using bun (faster):
```bash
bun install
```

### 3. Configure Environment

The `.env.local` file is already created with default values:

```env
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_WS_URL=ws://localhost:8000
```

If your backend runs on a different port, update these values.

### 4. Start Development Server

Using npm:
```bash
npm run dev
```

Using bun:
```bash
bun dev
```

The app will be available at: **http://localhost:3000**

---

## ✅ Verification Checklist

### 1. **Backend is Running**

Make sure your backend is running on port 8000:

```bash
# From the backend directory
cd ../backend
python main.py
```

You should see:
```
🚀 Starting Unified AlphaPlus Application...
✅ Database connection initialized
✅ Service manager initialized
```

### 2. **Frontend Starts Successfully**

You should see:
```
  ▲ Next.js 14.2.3
  - Local:        http://localhost:3000
  - Environments: .env.local
```

### 3. **WebSocket Connection**

Open http://localhost:3000 and check the browser console:

✅ **Success**: `✅ WebSocket connected`

❌ **Failure**: `❌ WebSocket error` or `🔌 WebSocket disconnected`

**Solution**: Ensure backend is running and WebSocket endpoint is accessible.

### 4. **API Connectivity**

Check the Status Bar at the bottom:
- **Green Wifi Icon**: Connected ✅
- **Red Wifi Icon**: Disconnected ❌

If disconnected, verify:
```bash
# Test API endpoint
curl http://localhost:8000/health
```

Should return:
```json
{
  "status": "healthy",
  "timestamp": "2025-10-27T..."
}
```

---

## 🎨 What You'll See

### Dashboard (Main Page)

```
┌─────────────────────────────────────────────────────────────┐
│  HEADER: AlphaPulse | Notifications | Settings              │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────────────────────┐  ┌───────────────────────┐│
│  │                              │  │  🟢 LIVE SIGNALS     ││
│  │    MAIN CHART                │  │  ┌─────────────────┐  ││
│  │    (Placeholder for now)     │  │  │ BTCUSDT LONG    │  ││
│  │                              │  │  │ 95% Confidence  │  ││
│  │                              │  │  └─────────────────┘  ││
│  └──────────────────────────────┘  └───────────────────────┘│
│  ┌──────────────────────────────┐  ┌───────────────────────┐│
│  │  9-HEAD CONSENSUS            │  │  QUICK STATS          ││
│  │  ● Technical      ✅ LONG    │  │  Active Signals: 12   ││
│  │  ● Sentiment      ✅ LONG    │  │  Win Rate: 78%        ││
│  │  ● Volume         ✅ LONG    │  │  Market: Bullish      ││
│  │  ...                         │  └───────────────────────┘│
│  └──────────────────────────────┘  ┌───────────────────────┐│
│                                     │  MTF ANALYSIS         ││
│                                     │  Perfect Alignment ✅ ││
│                                     └───────────────────────┘│
├─────────────────────────────────────────────────────────────┤
│  🟢 Connected | Latency: 45ms | AlphaPulse v1.0.0          │
└─────────────────────────────────────────────────────────────┘
```

### Analytics Page

Visit: **http://localhost:3000/analytics**

- Performance charts
- Win rate over time
- Recent signal performance table
- Key statistics

---

## 🐛 Troubleshooting

### Issue: "Cannot connect to backend"

**Symptoms:**
- Red WiFi icon in Status Bar
- No signals appearing
- Console error: `Failed to fetch`

**Solutions:**

1. **Check Backend Status**
```bash
curl http://localhost:8000/health
```

2. **Check CORS Settings** in backend (`main_unified.py`):
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Add this
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

3. **Restart Both Services**
```bash
# Terminal 1: Backend
cd apps/backend
python main.py

# Terminal 2: Frontend
cd apps/web
npm run dev
```

### Issue: "WebSocket not connecting"

**Symptoms:**
- Console: `🔌 WebSocket disconnected`
- No real-time updates

**Solutions:**

1. **Check WebSocket URL** in `.env.local`:
```env
NEXT_PUBLIC_WS_URL=ws://localhost:8000
```

2. **Test WebSocket** in browser console:
```javascript
const ws = new WebSocket('ws://localhost:8000/ws');
ws.onopen = () => console.log('Connected!');
ws.onerror = (e) => console.error('Error:', e);
```

3. **Check Backend WebSocket Endpoint**:
```bash
curl --include \
     --no-buffer \
     --header "Connection: Upgrade" \
     --header "Upgrade: websocket" \
     --header "Sec-WebSocket-Version: 13" \
     --header "Sec-WebSocket-Key: SGVsbG8sIHdvcmxkIQ==" \
     http://localhost:8000/ws
```

### Issue: "No signals appearing"

**Symptoms:**
- Dashboard shows "No signals yet"
- API returns empty array

**Solutions:**

1. **Check if backend has signals**:
```bash
curl http://localhost:8000/api/signals/latest
```

2. **Generate test signals** (backend):
```python
# In backend, generate sample signals
# Check if signal_generator is running
```

3. **Lower confidence filter**:
```typescript
// In SignalFeed.tsx, signals are filtered at 60%+
// Temporarily lower to see all signals
if (confidence < 0.0) {  // Show ALL signals
  continue;
}
```

---

## 🔧 Development Tips

### 1. **Hot Reload**
Code changes automatically refresh the browser. No need to restart.

### 2. **TypeScript Errors**
```bash
# Check types
npm run type-check
```

### 3. **Component Development**
All components are in `src/components/`. Modify and see changes instantly.

### 4. **Adding New Pages**
```bash
# Create new page
apps/web/src/app/my-page/page.tsx
```

Automatically available at: http://localhost:3000/my-page

### 5. **Debugging WebSocket**
```typescript
// In browser console
wsManager.getStatus()  // Check connection status
```

### 6. **Zustand State Debugging**
```typescript
// In browser console
useSignalStore.getState()  // Check signal state
useMarketStore.getState()  // Check market state
```

---

## 📚 Next Steps

1. **Customize Colors**: Edit `tailwind.config.ts`
2. **Add More Pages**: Create in `src/app/`
3. **Integrate TradingView**: Add chart library
4. **Add Notifications**: Implement browser notifications
5. **Mobile Optimization**: Test on mobile devices

---

## 🎯 Quick Commands Reference

```bash
# Development
npm run dev          # Start dev server
npm run build        # Build for production
npm start            # Start production server

# Quality
npm run lint         # Lint code
npm run type-check   # Type checking

# Cleanup
rm -rf .next         # Clear build cache
rm -rf node_modules  # Remove dependencies
npm install          # Reinstall
```

---

## ✨ Features Working Out of the Box

- ✅ Real-time signal feed
- ✅ 9-head SDE consensus visualization  
- ✅ Multi-timeframe analysis panel
- ✅ WebSocket connection with auto-reconnect
- ✅ Performance analytics page
- ✅ Responsive dark theme UI
- ✅ Automatic API polling
- ✅ State management (Zustand)
- ✅ Type-safe development (TypeScript)

---

## 🆘 Still Having Issues?

1. **Check Browser Console** (F12) for errors
2. **Check Backend Logs** for API errors
3. **Verify Ports**: Backend (8000), Frontend (3000)
4. **Clear Cache**: `rm -rf .next && npm run dev`
5. **Restart Everything**: Kill all processes and restart

---

**Happy Trading! 🚀📈**

