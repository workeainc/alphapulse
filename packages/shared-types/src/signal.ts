/**
 * Signal Types
 * Shared types for trading signals across AlphaPulse
 */

export type SignalDirection = 'long' | 'short';
export type SignalSource = 'pattern' | 'ml_ensemble' | 'hybrid' | 'manual';

export interface Signal {
  signal_id: string;
  symbol: string;
  timeframe: string;
  direction: SignalDirection;
  confidence: number;
  entry_price: number;
  tp1?: number;
  tp2?: number;
  tp3?: number;
  tp4?: number;
  stop_loss?: number;
  risk_reward_ratio?: number;
  pattern_type?: string;
  volume_confirmation?: boolean;
  trend_alignment?: boolean;
  market_regime?: string;
  indicators?: Record<string, number>;
  validation_metrics?: Record<string, number>;
  timestamp: string;
  outcome?: 'pending' | 'success' | 'failure' | 'partial';
  source?: SignalSource;
  source_model?: string;
}

export interface SignalFilters {
  symbol?: string;
  timeframe?: string;
  direction?: SignalDirection;
  min_confidence?: number;
  source?: SignalSource;
  start_date?: string;
  end_date?: string;
}

export interface SignalResponse {
  signals: Signal[];
  total: number;
  page: number;
  page_size: number;
}

