/**
 * Signal Recommendation Types
 * Types for trading recommendations provided to users
 */

export type RecommendationStatus = 'pending' | 'user_executed' | 'expired' | 'cancelled';
export type RecommendationSide = 'long' | 'short';

export interface SignalRecommendation {
  id: number;
  signal_id: string;
  symbol: string;
  side: RecommendationSide;
  suggested_entry_price: number;
  suggested_exit_price?: number;
  suggested_quantity: number;
  suggested_leverage: number;
  hypothetical_pnl?: number;
  hypothetical_pnl_percentage?: number;
  strategy_name: string;
  strategy_signals?: Record<string, any>;
  suggested_stop_loss?: number;
  suggested_take_profit?: number;
  suggested_trailing_stop?: number;
  market_regime?: string;
  sentiment_score?: number;
  volatility?: number;
  timeframe_signals?: Record<string, any>;
  news_impact?: string;
  status: RecommendationStatus;
  recommendation_time: string;
  expiry_time?: string;
  created_at: string;
  updated_at: string;
  notes?: string;
  tags?: string[];
}

export interface RecommendationFilters {
  symbol?: string;
  side?: RecommendationSide;
  status?: RecommendationStatus;
  min_confidence?: number;
  start_date?: string;
  end_date?: string;
}

export interface RecommendationResponse {
  recommendations: SignalRecommendation[];
  total: number;
  page: number;
  page_size: number;
}

