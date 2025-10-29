'use client';

import * as React from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/Card';
import { Badge } from '@/components/ui/Badge';

interface WorkflowStatus {
  data_collection: {
    candles_received: number;
    candles_stored: number;
    last_candle_times: Record<string, string>;
    time_since_last_candle: Record<string, { seconds: number; status: string }>;
    status: string;
  };
  indicator_calculation: {
    calculations_performed: number;
    buffer_status: Record<string, Record<string, { candles_in_buffer: number; buffer_size: number }>>;
    status: string;
  };
  consensus_system: {
    calculations_performed: number;
    last_consensus_votes: Record<string, any>;
    status: string;
  };
  signal_generation: {
    scans_performed: number;
    signals_generated: number;
    rejection_rate: string;
    status: string;
  };
}

interface WorkflowStep {
  timestamp: string;
  event: string;
  symbol?: string;
  timeframe?: string;
  direction?: string;
  confidence?: number;
}

interface WorkflowMonitorProps {
  workflowStatus: WorkflowStatus | null;
  recentSteps: WorkflowStep[];
  lastUpdate?: any;
}

interface WorkflowMonitorWithError extends WorkflowMonitorProps {
  error?: Error | null;
  isLoading?: boolean;
}

export function WorkflowMonitor({ workflowStatus, recentSteps, lastUpdate, error, isLoading }: WorkflowMonitorWithError) {
  // Show error state (backend needs restart)
  if (error) {
    return (
      <Card className="h-full">
        <CardHeader>
          <CardTitle className="flex items-center justify-between">
            <span>🔴 Live Workflow Monitor</span>
            <Badge className="bg-red-500/20 text-red-400">ERROR</Badge>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="p-4 bg-red-500/10 border border-red-500/30 rounded">
            <div className="font-semibold text-red-400 mb-2">⚠️ Backend Restart Required</div>
            <div className="text-sm text-gray-300 mb-3">
              The backend is running old code. Please restart it:
            </div>
            <div className="text-xs font-mono bg-gray-900 p-2 rounded mb-2">
              <div>1. Stop backend (Ctrl+C)</div>
              <div>2. Run: <span className="text-green-400">cd apps\backend && python main.py</span></div>
              <div>3. Wait for "Connected to Binance!" message</div>
            </div>
            <div className="text-xs text-gray-400">
              Error: {error.message}
            </div>
          </div>
        </CardContent>
      </Card>
    );
  }

  // Show loading state
  if (isLoading || !workflowStatus) {
    return (
      <Card className="h-full">
        <CardHeader>
          <CardTitle>🔴 Live Workflow Monitor</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-center text-gray-400 py-8">
            <div className="animate-pulse">Loading workflow status...</div>
            <div className="text-xs text-gray-500 mt-2">Connecting to backend...</div>
          </div>
        </CardContent>
      </Card>
    );
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active':
      case 'realtime':
        return 'bg-green-500/20 text-green-400';
      case 'waiting':
      case 'delayed':
        return 'bg-yellow-500/20 text-yellow-400';
      case 'stale':
      case 'no_data':
        return 'bg-red-500/20 text-red-400';
      default:
        return 'bg-gray-500/20 text-gray-400';
    }
  };

  const formatTimeAgo = (seconds: number) => {
    if (seconds < 0) return 'N/A';
    if (seconds < 60) return `${seconds}s ago`;
    if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
    return `${Math.floor(seconds / 3600)}h ago`;
  };

  return (
    <Card className="h-full">
      <CardHeader>
        <CardTitle className="flex items-center justify-between">
          <span>🔴 Live Workflow Monitor</span>
          <Badge className={workflowStatus?.data_collection?.status === 'active' ? 'bg-green-500/20 text-green-400' : 'bg-gray-500/20 text-gray-400'}>
            {workflowStatus?.data_collection?.status === 'active' ? 'LIVE' : 'WAITING'}
          </Badge>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4 max-h-[600px] overflow-y-auto">
        {/* Data Collection Status */}
        <div className="space-y-2">
          <h3 className="text-sm font-semibold text-gray-300">1️⃣ Data Collection</h3>
          <div className="grid grid-cols-2 gap-2 text-xs">
            <div>
              <span className="text-gray-400">Candles Received:</span>
              <span className="ml-2 font-mono font-bold text-blue-400">
                {workflowStatus?.data_collection?.candles_received || 0}
              </span>
            </div>
            <div>
              <span className="text-gray-400">Candles Stored:</span>
              <span className="ml-2 font-mono font-bold text-green-400">
                {workflowStatus?.data_collection?.candles_stored || 0}
              </span>
            </div>
          </div>
          <div className="space-y-1">
            <div className="text-xs text-gray-400">Last Candle Times:</div>
            {workflowStatus?.data_collection?.time_since_last_candle && Object.keys(workflowStatus.data_collection.time_since_last_candle).length > 0 ? (
              Object.entries(workflowStatus.data_collection.time_since_last_candle).map(([symbol, info]: [string, any]) => (
                <div key={symbol} className="flex items-center justify-between text-xs">
                  <span className="text-gray-300">{symbol}:</span>
                  <Badge className={getStatusColor(info?.status || 'unknown')}>
                    {info?.seconds !== undefined && info.seconds >= 0 ? formatTimeAgo(info.seconds) : 'No data'}
                  </Badge>
                </div>
              ))
            ) : (
              <div className="text-xs text-gray-500 italic">Waiting for candles...</div>
            )}
          </div>
        </div>

        {/* Indicator Calculation Status */}
        <div className="space-y-2">
          <h3 className="text-sm font-semibold text-gray-300">2️⃣ Indicator Calculation</h3>
          <div className="text-xs">
            <span className="text-gray-400">Calculations:</span>
            <span className="ml-2 font-mono font-bold text-purple-400">
              {workflowStatus?.indicator_calculation?.calculations_performed || 0}
            </span>
          </div>
          <Badge className={getStatusColor(workflowStatus?.indicator_calculation?.status || 'waiting')}>
            {workflowStatus?.indicator_calculation?.status || 'waiting'}
          </Badge>
        </div>

        {/* Consensus System Status */}
        <div className="space-y-2">
          <h3 className="text-sm font-semibold text-gray-300">3️⃣ 9-Head Consensus</h3>
          <div className="text-xs">
            <span className="text-gray-400">Consensus Calculations:</span>
            <span className="ml-2 font-mono font-bold text-yellow-400">
              {workflowStatus?.consensus_system?.calculations_performed || 0}
            </span>
          </div>
          {workflowStatus?.consensus_system?.last_consensus_votes && Object.keys(workflowStatus.consensus_system.last_consensus_votes).length > 0 ? (
            <div className="space-y-1 text-xs">
              {Object.entries(workflowStatus.consensus_system.last_consensus_votes).slice(0, 3).map(([key, vote]: [string, any]) => (
                <div key={key} className="bg-gray-800/50 p-2 rounded">
                  <div className="font-semibold text-gray-300">{vote?.consensus?.direction || 'FLAT'}</div>
                  <div className="text-gray-400">
                    {vote?.consensus?.agreeing_heads || 0}/{vote?.consensus?.total_heads || 9} heads ({Math.round((vote?.consensus?.confidence || 0) * 100)}%)
                  </div>
                  {vote?.votes && (
                    <div className="text-[10px] text-gray-500 mt-1">
                      {Object.entries(vote.votes).slice(0, 3).map(([head, data]: [string, any]) => (
                        <div key={head}>{head}: {data?.direction || 'FLAT'}</div>
                      ))}
                    </div>
                  )}
                </div>
              ))}
            </div>
          ) : (
            <div className="text-xs text-gray-500 italic">No consensus votes yet (waiting for candle close)</div>
          )}
          <Badge className={getStatusColor(workflowStatus?.consensus_system?.status || 'waiting')}>
            {workflowStatus?.consensus_system?.status || 'waiting'}
          </Badge>
        </div>

        {/* Signal Generation Status */}
        <div className="space-y-2">
          <h3 className="text-sm font-semibold text-gray-300">4️⃣ Signal Generation</h3>
          <div className="grid grid-cols-2 gap-2 text-xs">
            <div>
              <span className="text-gray-400">Scans:</span>
              <span className="ml-2 font-mono font-bold text-cyan-400">
                {workflowStatus?.signal_generation?.scans_performed || 0}
              </span>
            </div>
            <div>
              <span className="text-gray-400">Signals:</span>
              <span className="ml-2 font-mono font-bold text-green-400">
                {workflowStatus?.signal_generation?.signals_generated || 0}
              </span>
            </div>
          </div>
          <div className="text-xs">
            <span className="text-gray-400">Rejection Rate:</span>
            <span className="ml-2 font-mono font-bold text-orange-400">
              {workflowStatus?.signal_generation?.rejection_rate || '0%'}
            </span>
          </div>
          <Badge className={getStatusColor(workflowStatus?.signal_generation?.status || 'waiting')}>
            {workflowStatus?.signal_generation?.status || 'waiting'}
          </Badge>
        </div>

        {/* Recent Workflow Steps */}
        {recentSteps && recentSteps.length > 0 && (
          <div className="space-y-2 border-t border-gray-700 pt-4">
            <h3 className="text-sm font-semibold text-gray-300">📋 Recent Activity</h3>
            <div className="space-y-1 max-h-[150px] overflow-y-auto">
              {recentSteps.slice(-10).reverse().map((step, idx) => (
                <div key={idx} className="text-xs bg-gray-800/30 p-2 rounded">
                  <div className="flex items-center justify-between">
                    <span className="text-gray-300 font-semibold">{step.event}</span>
                    <span className="text-gray-500 text-[10px]">
                      {new Date(step.timestamp).toLocaleTimeString()}
                    </span>
                  </div>
                  {step.symbol && (
                    <div className="text-gray-400 mt-1">
                      {step.symbol} {step.timeframe} {step.direction && `→ ${step.direction.toUpperCase()}`}
                      {step.confidence && ` (${Math.round(step.confidence * 100)}%)`}
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}

          {/* Live Update Indicator */}
        {lastUpdate && (
          <div className="text-[10px] text-gray-500 text-center pt-2 border-t border-gray-700">
            Last update: {new Date(lastUpdate.timestamp || Date.now()).toLocaleTimeString()}
          </div>
        )}

        {/* No Data Warning */}
        {workflowStatus?.data_collection?.candles_received === 0 && (
          <div className="mt-4 p-3 bg-yellow-500/10 border border-yellow-500/30 rounded text-xs">
            <div className="font-semibold text-yellow-400 mb-1">⚠️ Waiting for Data</div>
            <div className="text-yellow-300/80">
              Backend may need restart to apply new code. Check:
              <ul className="list-disc list-inside mt-1 ml-2">
                <li>Backend is running latest code</li>
                <li>Binance WebSocket is connected</li>
                <li>Check backend logs for "Connected to Binance!"</li>
              </ul>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

