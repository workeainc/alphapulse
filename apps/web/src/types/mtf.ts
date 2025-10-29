// MTF (Multi-Timeframe) Types

export type Timeframe = '1m' | '5m' | '15m' | '1h' | '4h' | '1d';

export interface TimeframeVote {
  signal_type: 'LONG' | 'SHORT' | 'NEUTRAL';
  confidence: number;
  contribution: number;
  weight: number;
}

export interface MTFSignal {
  symbol: string;
  base_timeframe: Timeframe;
  base_confidence: number;
  mtf_boost: number;
  final_confidence: number;
  contributing_timeframes: Timeframe[];
  timeframe_votes: Record<Timeframe, TimeframeVote>;
  timestamp: string;
  alignment_status: 'perfect' | 'strong' | 'weak' | 'divergent';
}

export interface MTFAnalysis {
  symbol: string;
  timeframes: Timeframe[];
  alignment_score: number;
  best_entry_timeframe: Timeframe;
  higher_timeframe_bias: 'bullish' | 'bearish' | 'neutral';
  divergence_warning: boolean;
  timestamp: string;
}

