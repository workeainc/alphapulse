// Design tokens
export const COLORS = {
  background: {
    primary: '#0B0E11',
    secondary: '#141820',
    tertiary: '#1E2329',
  },
  signal: {
    long: '#10B981',
    short: '#EF4444',
    neutral: '#6B7280',
  },
  confidence: {
    veryHigh: '#34D399',
    high: '#10B981',
    medium: '#FBBF24',
    low: '#F87171',
  },
  heads: {
    technical: '#8B5CF6',
    sentiment: '#EC4899',
    volume: '#14B8A6',
    rules: '#F59E0B',
    ict: '#06B6D4',
    wyckoff: '#10B981',
    harmonic: '#EAB308',
    structure: '#3B82F6',
    crypto: '#A855F7',
  },
} as const;

// Confidence thresholds
export const CONFIDENCE_THRESHOLDS = {
  veryHigh: 0.85,
  high: 0.75,
  medium: 0.65,
  low: 0,
} as const;

// Timeframe weights for MTF
export const TIMEFRAME_WEIGHTS = {
  '1d': 0.4,
  '4h': 0.3,
  '1h': 0.2,
  '15m': 0.1,
  '5m': 0.05,
  '1m': 0.02,
} as const;

// Head names
export const HEAD_NAMES = [
  'technical',
  'sentiment',
  'volume',
  'rules',
  'ict',
  'wyckoff',
  'harmonic',
  'structure',
  'crypto',
] as const;

// Refresh intervals (ms)
export const REFRESH_INTERVALS = {
  signals: 10000,      // 10 seconds
  marketData: 5000,    // 5 seconds
  performance: 30000,  // 30 seconds
  health: 60000,       // 1 minute
} as const;

// Notification settings
export const NOTIFICATION_CONFIG = {
  minConfidenceForSound: 0.85,
  minConfidenceForBrowser: 0.75,
  soundEnabled: true,
  browserEnabled: true,
} as const;

