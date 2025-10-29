// WebSocket Types

export type WebSocketMessageType = 
  | 'signal'
  | 'market_update'
  | 'tp_hit'
  | 'sl_hit'
  | 'system_alert'
  | 'performance_update'
  | 'initial_signals'
  | 'new_signal'
  | 'signal_removed'
  | 'workflow_status'
  | 'workflow_update'
  | 'ping';

export interface WebSocketMessage<T = any> {
  type: WebSocketMessageType;
  data: T;
  timestamp: string;
}

export interface SignalUpdate {
  symbol: string;
  direction: 'long' | 'short';
  confidence: number;
  pattern_type: string;
  entry_price: number;
  stop_loss: number;
  take_profit: number;
}

export interface MarketUpdate {
  symbol: string;
  price: number;
  volume_24h: number;
  change_24h: number;
  high_24h: number;
  low_24h: number;
}

export interface TPHitUpdate {
  symbol: string;
  tp_level: number;
  price: number;
  profit: number;
}

export interface SLHitUpdate {
  symbol: string;
  price: number;
  loss: number;
}

export interface SystemAlert {
  message: string;
  severity: 'info' | 'warning' | 'error' | 'success';
}

export interface WebSocketStatus {
  connected: boolean;
  latency: number;
  reconnectAttempts: number;
  lastMessageTime?: string;
}

