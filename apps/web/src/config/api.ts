export const API_CONFIG = {
  baseURL: process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000',
  wsURL: process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8000',
  timeout: 10000,
  retryAttempts: 3,
  retryDelay: 1000,
};

export const API_ENDPOINTS = {
  // Core
  health: '/health',
  config: '/config',
  services: '/services/status',
  
  // Signals
  signals: {
    latest: '/api/signals/latest',
    highQuality: '/api/signals/high-quality',
    smcEnhanced: '/api/signals/smc-enhanced',
    aiEnhanced: '/api/signals/ai-enhanced',
    rlEnhanced: '/api/signals/rl-enhanced',
    nlpEnhanced: '/api/signals/nlp-enhanced',
    performance: '/api/signals/performance',
    history: '/api/signals/history',
  },
  
  // Patterns
  patterns: {
    latest: '/api/patterns/latest',
    history: '/api/patterns/history',
  },
  
  // Market
  market: {
    status: '/api/market/status',
    data: '/market-data',
  },
  
  // AI & Performance
  ai: {
    performance: '/api/ai/performance',
  },
  
  performance: {
    analytics: '/api/performance/analytics',
  },
  
  // WebSocket
  websocket: {
    main: '/ws',
    marketData: '/ws/market-data',
    signals: '/ws/signals',
    streaming: '/ws/streaming',
  },
} as const;

