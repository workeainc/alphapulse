// Export all types
export * from './signal';
export * from './sde';
export * from './mtf';
export * from './websocket';

// Common types
export interface HealthStatus {
  status: 'healthy' | 'unhealthy';
  timestamp: string;
  database: {
    status: 'healthy' | 'unhealthy';
  };
  websocket: any;
  services: any;
}

export interface Phase3Status {
  service: string;
  database: string;
  websocket: string;
  patterns_detected: number;
  signals_generated: number;
  timestamp: string;
  status: string;
  connection_status: string;
}

export interface ConfigData {
  websocket: {
    symbols: string[];
    timeframes: string[];
    performance_mode: string;
    enable_shared_memory: boolean;
  };
  database: {
    host: string;
    port: number;
    database: string;
  };
  redis: {
    enabled: boolean;
    url?: string;
  };
}

