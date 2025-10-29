'use client';

import * as React from 'react';
import { Wifi, WifiOff, Database } from 'lucide-react';
import { cn } from '@/lib/utils/cn';
import { formatLatency } from '@/lib/utils/format';
import { useWebSocket } from '@/lib/hooks';

export function StatusBar() {
  const { status } = useWebSocket();

  return (
    <div className="border-t border-gray-800 bg-background-secondary px-6 py-2">
      <div className="flex items-center justify-between text-xs">
        {/* Left: Connection Status */}
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2">
            {status.connected ? (
              <>
                <Wifi className="h-4 w-4 text-green-400" />
                <span className="text-green-400">Connected</span>
              </>
            ) : (
              <>
                <WifiOff className="h-4 w-4 text-red-400" />
                <span className="text-red-400">Disconnected</span>
              </>
            )}
          </div>

          {status.connected && (
            <div className="flex items-center gap-2">
              <Database className="h-4 w-4 text-blue-400" />
              <span className="text-gray-400">
                Latency: <span className="font-mono text-blue-400">{formatLatency(status.latency)}</span>
              </span>
            </div>
          )}
        </div>

        {/* Right: System Info */}
        <div className="flex items-center gap-4 text-gray-400">
          <span>AlphaPulse v1.0.0</span>
          <span className={cn(
            'h-2 w-2 rounded-full',
            status.connected ? 'bg-green-400 animate-pulse' : 'bg-red-400'
          )} />
        </div>
      </div>
    </div>
  );
}

