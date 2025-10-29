// SDE (Single-Decision Engine) Types

export type HeadDirection = 'LONG' | 'SHORT' | 'FLAT';

export interface HeadVote {
  direction: HeadDirection;
  confidence: number; // 0-1
  probability: number; // 0-1
  weight: number;
  reasoning?: string;
}

export interface SDEConsensus {
  heads: {
    technical: HeadVote;
    sentiment: HeadVote;
    volume: HeadVote;
    rules: HeadVote;
    ict: HeadVote;
    wyckoff: HeadVote;
    harmonic: HeadVote;
    structure: HeadVote;
    crypto: HeadVote;
  };
  consensus_achieved: boolean;
  agreeing_heads: number;
  total_heads: number;
  consensus_score: number;
  final_direction: HeadDirection;
  final_confidence: number;
  final_probability: number;
  timestamp: string;
}

export type HeadName = 
  | 'technical'
  | 'sentiment'
  | 'volume'
  | 'rules'
  | 'ict'
  | 'wyckoff'
  | 'harmonic'
  | 'structure'
  | 'crypto';

export interface HeadPerformance {
  head_name: HeadName;
  accuracy: number;
  total_votes: number;
  correct_votes: number;
  avg_confidence: number;
  vote_distribution: {
    long: number;
    short: number;
    flat: number;
  };
}

