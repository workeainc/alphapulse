/**
 * Shared Constants
 * Application-wide constants used across backend and frontend
 */

export const TIMEFRAMES = ['1m', '5m', '15m', '30m', '1h', '4h', '1d', '1w'] as const;
export type Timeframe = typeof TIMEFRAMES[number];

export const SYMBOLS = [
  'BTC/USDT',
  'ETH/USDT',
  'ADA/USDT',
  'SOL/USDT',
  'BNB/USDT',
  'XRP/USDT',
  'DOT/USDT',
  'LINK/USDT'
] as const;
export type Symbol = typeof SYMBOLS[number];

export const SIGNAL_CONFIDENCE_THRESHOLDS = {
  low: 0.60,
  medium: 0.75,
  high: 0.85,
  veryHigh: 0.90,
} as const;

export const RISK_TOLERANCE_LEVELS = {
  low: { maxPositionSize: 0.02, maxLeverage: 1 },
  medium: { maxPositionSize: 0.05, maxLeverage: 2 },
  high: { maxPositionSize: 0.10, maxLeverage: 3 },
} as const;

