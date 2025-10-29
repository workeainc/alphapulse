/**
 * API Configuration
 * Shared API endpoint and connection configuration
 */

export const API_CONFIG = {
  backend: {
    url: process.env.NEXT_PUBLIC_API_URL || process.env.BACKEND_URL || 'http://localhost:8000',
    wsUrl: process.env.NEXT_PUBLIC_WS_URL || process.env.BACKEND_WS_URL || 'ws://localhost:8000',
    timeout: 30000,
  },
  endpoints: {
    signals: '/api/v1/signals',
    recommendations: '/api/v1/recommendations',
    patterns: '/api/patterns/latest',
    marketStatus: '/api/market/status',
    health: '/health',
    websocket: '/ws',
  },
} as const;

export const DEFAULT_HEADERS = {
  'Content-Type': 'application/json',
  'Accept': 'application/json',
};

