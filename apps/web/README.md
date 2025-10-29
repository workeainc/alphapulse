# AlphaPulse Frontend

> **Professional-Grade Crypto Trading Signals Dashboard**

Built with Next.js 14, TypeScript, TailwindCSS, and real-time WebSocket integration.

---

## 🚀 Features

### Core Features
- ✅ **Real-Time Signal Feed** - Live trading signals with WebSocket updates
- ✅ **9-Head SDE Consensus** - Visual representation of AI consensus voting
- ✅ **Multi-Timeframe Analysis** - MTF signal alignment and boost calculation
- ✅ **Performance Analytics** - Track win rates, profits, and signal accuracy
- ✅ **Dark Mode First** - Professional Bloomberg-style UI
- ✅ **Responsive Design** - Full functionality on all devices

### Technical Features
- ⚡ **Ultra-Low Latency** - <100ms WebSocket updates
- 🎨 **Framer Motion** - Smooth animations and transitions
- 📊 **Recharts Integration** - Beautiful performance charts
- 🔄 **React Query** - Automatic caching and refetching
- 💾 **Zustand State** - Lightweight global state management
- 🎯 **TypeScript** - Full type safety

---

## 📁 Project Structure

```
apps/web/
├── src/
│   ├── app/                      # Next.js 14 App Router
│   │   ├── page.tsx              # Main dashboard
│   │   ├── analytics/            # Analytics page
│   │   ├── layout.tsx            # Root layout
│   │   ├── providers.tsx         # React Query provider
│   │   └── globals.css           # Global styles
│   ├── components/               # React components
│   │   ├── ui/                   # Base UI components
│   │   │   ├── Button.tsx
│   │   │   ├── Card.tsx
│   │   │   └── Badge.tsx
│   │   ├── signals/              # Signal components
│   │   │   ├── SignalCard.tsx
│   │   │   └── SignalFeed.tsx
│   │   ├── sde/                  # SDE components
│   │   │   └── SDEConsensusDashboard.tsx
│   │   ├── mtf/                  # MTF components
│   │   │   └── MTFAnalysisPanel.tsx
│   │   ├── analytics/            # Analytics components
│   │   │   └── PerformanceChart.tsx
│   │   └── layout/               # Layout components
│   │       ├── Header.tsx
│   │       └── StatusBar.tsx
│   ├── lib/                      # Utilities & Logic
│   │   ├── api/                  # API client
│   │   │   └── client.ts
│   │   ├── websocket/            # WebSocket manager
│   │   │   └── manager.ts
│   │   ├── hooks/                # Custom hooks
│   │   │   ├── useWebSocket.ts
│   │   │   ├── useSignals.ts
│   │   │   └── useMarket.ts
│   │   └── utils/                # Helper functions
│   │       ├── cn.ts
│   │       ├── format.ts
│   │       └── confidence.ts
│   ├── store/                    # Zustand stores
│   │   ├── signalStore.ts
│   │   ├── marketStore.ts
│   │   └── notificationStore.ts
│   ├── types/                    # TypeScript types
│   │   ├── signal.ts
│   │   ├── sde.ts
│   │   ├── mtf.ts
│   │   └── websocket.ts
│   └── config/                   # Configuration
│       ├── api.ts
│       └── constants.ts
├── public/                       # Static assets
├── package.json
├── tsconfig.json
├── tailwind.config.ts
└── next.config.js
```

---

## 🛠️ Installation & Setup

### Prerequisites
- Node.js 18+ or Bun
- Backend running on `http://localhost:8000`

### Install Dependencies

```bash
cd apps/web
npm install
```

### Environment Variables

Create `.env.local`:

```env
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_WS_URL=ws://localhost:8000
```

### Run Development Server

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000)

---

## 📡 API Integration

### REST API Endpoints

The frontend connects to these backend endpoints:

```typescript
// Signals
GET  /api/signals/latest              // Latest signals (60%+ confidence)
GET  /api/signals/high-quality        // High-quality signals
GET  /api/signals/performance         // Performance metrics
GET  /api/signals/history             // Historical signals

// Market
GET  /api/market/status               // Market condition
GET  /market-data                     // OHLCV data

// Patterns
GET  /api/patterns/latest             // Latest patterns
GET  /api/patterns/history            // Historical patterns

// AI Performance
GET  /api/ai/performance              // AI model metrics
GET  /api/performance/analytics       // Analytics data
```

### WebSocket Endpoints

Real-time updates via WebSocket:

```typescript
WS   /ws                              // General WebSocket
WS   /ws/signals                      // Real-time signals
WS   /ws/market-data                  // Market data stream
```

### WebSocket Message Format

```typescript
{
  "type": "signal",
  "data": {
    "symbol": "BTCUSDT",
    "direction": "long",
    "confidence": 0.95,
    "pattern_type": "wyckoff_spring",
    "entry_price": 42000,
    "stop_loss": 40500,
    "take_profit": 45000
  },
  "timestamp": "2025-10-27T12:00:00Z"
}
```

---

## 🎨 Component Usage

### Signal Card

```tsx
import { SignalCard } from '@/components/signals/SignalCard';

<SignalCard 
  signal={signal}
  onClick={() => handleSignalClick(signal)}
  selected={isSelected}
  variant="detailed"
/>
```

### SDE Consensus Dashboard

```tsx
import { SDEConsensusDashboard } from '@/components/sde/SDEConsensusDashboard';

<SDEConsensusDashboard 
  consensus={consensusData}
  animated={true}
/>
```

### MTF Analysis Panel

```tsx
import { MTFAnalysisPanel } from '@/components/mtf/MTFAnalysisPanel';

<MTFAnalysisPanel 
  mtfSignal={mtfData}
  showBoostCalculation={true}
/>
```

---

## 🎯 Key Features Explained

### 1. Real-Time Signal Updates

Signals are updated via two methods:
- **Polling**: REST API every 10 seconds
- **WebSocket**: Instant push notifications

```typescript
// Automatic polling with React Query
const { data: signals } = useSignals({ autoRefresh: true });

// WebSocket updates
const { lastMessage } = useWebSocket();
```

### 2. 9-Head SDE Consensus Visualization

Shows individual head votes with:
- Color-coded confidence bars
- Direction indicators (LONG/SHORT/FLAT)
- Final consensus calculation
- Visual agreement status

### 3. Multi-Timeframe Analysis

Displays:
- Base timeframe signal
- Higher timeframe votes
- MTF boost calculation
- Alignment status (perfect/strong/weak/divergent)

### 4. State Management

**Zustand Stores:**
- `signalStore` - Manages signals and filters
- `marketStore` - Manages market data and watchlist
- `notificationStore` - Manages notifications

```typescript
// Using stores
const { signals, addSignal } = useSignalStore();
const { selectedSymbol, setSelectedSymbol } = useMarketStore();
```

---

## 🎨 Design System

### Colors

```typescript
// Base
background-primary: '#0B0E11'
background-secondary: '#141820'
background-tertiary: '#1E2329'

// Signals
signal-long: '#10B981' (green)
signal-short: '#EF4444' (red)

// Confidence
very-high: '#34D399' (85%+)
high: '#10B981' (75-85%)
medium: '#FBBF24' (65-75%)
low: '#F87171' (<65%)

// SDE Heads
technical: '#8B5CF6'
sentiment: '#EC4899'
volume: '#14B8A6'
ict: '#06B6D4'
wyckoff: '#10B981'
...
```

### Typography

- **Display**: Inter (headings)
- **Body**: Inter (text)
- **Mono**: JetBrains Mono (numbers, prices)

---

## 📊 Performance

- **First Load**: < 2s
- **Time to Interactive**: < 3s
- **WebSocket Latency**: < 100ms
- **Chart Rendering**: < 500ms

---

## 🚀 Deployment

### Build for Production

```bash
npm run build
```

### Start Production Server

```bash
npm start
```

### Deploy to Vercel

```bash
vercel deploy
```

---

## 📝 Development

### Type Checking

```bash
npm run type-check
```

### Linting

```bash
npm run lint
```

---

## 🎯 Roadmap

- [ ] TradingView chart integration
- [ ] Advanced filtering and search
- [ ] Signal backtesting UI
- [ ] Mobile app (React Native)
- [ ] Export signals to CSV/PDF
- [ ] Custom alert preferences
- [ ] Dark/Light theme toggle
- [ ] Multi-language support

---

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

---

## 📄 License

Proprietary - AlphaPulse Trading System

---

## 🆘 Support

For issues or questions:
- Check the [API documentation](../backend/README.md)
- Review the [backend endpoints](../backend/src/app/main_unified.py)
- Check WebSocket connection in browser console

---

**Built with ❤️ for professional crypto traders**
