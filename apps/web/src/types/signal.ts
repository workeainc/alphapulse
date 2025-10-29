// Signal Types
export interface Signal {
  symbol: string;
  direction: 'long' | 'short';
  confidence: number; // 0-1
  pattern_type: string;
  timestamp: string;
  entry_price: number;
  stop_loss: number;
  take_profit: number;
  tp1?: number;
  tp2?: number;
  tp3?: number;
  tp4?: number;
  risk_reward_ratio?: number;
  timeframe?: string;
  market_regime?: string;
  indicators?: Record<string, any>;
  validation_metrics?: Record<string, any>;
  metadata?: Record<string, any>;
}

export interface Pattern {
  symbol: string;
  pattern_type: string;
  confidence: number;
  strength: 'strong' | 'medium' | 'weak';
  timestamp: string;
  timeframe: string;
  price_level: number;
}

export interface MarketStatus {
  market_condition: 'bullish' | 'bearish' | 'neutral' | 'volatile';
  volatility: number;
  trend_direction: 'upward' | 'downward' | 'sideways';
  timestamp: string;
  websocket_activity?: {
    messages_received: number;
    messages_processed: number;
    avg_latency_ms: number;
    connected: boolean;
  };
}

export interface PerformanceMetrics {
  accuracy: number;
  total_signals: number;
  profitable_signals: number;
  average_return: number;
  win_rate?: number;
  profit_factor?: number;
  sharpe_ratio?: number;
  max_drawdown?: number;
  timestamp: string;
}

export interface AIPerformance {
  accuracy: number;
  total_signals: number;
  profitable_signals: number;
  average_return: number;
  timestamp: string;
  deep_learning_status?: {
    available: boolean;
    models_trained: number;
    last_training?: string;
    prediction_accuracy: number;
  };
  reinforcement_learning_status?: {
    available: boolean;
    training_episodes: number;
    avg_reward: number;
    best_reward: number;
    trading_agent_available: boolean;
    signal_agent_available: boolean;
  };
  natural_language_processing_status?: {
    available: boolean;
    analyses_performed: number;
    cache_hit_rate: number;
    overall_sentiment_accuracy: number;
    news_analyzer_available: boolean;
    social_media_analyzer_available: boolean;
    sentiment_analyzer_available: boolean;
  };
}

