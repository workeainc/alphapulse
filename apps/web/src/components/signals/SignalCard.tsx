'use client';

import * as React from 'react';
import { TrendingUp, TrendingDown, Clock } from 'lucide-react';
import { Card, CardContent } from '@/components/ui/Card';
import { Badge } from '@/components/ui/Badge';
import { cn } from '@/lib/utils/cn';
import { formatPrice, formatConfidence, formatDate } from '@/lib/utils/format';
import { getConfidenceColor, getSignalColor } from '@/lib/utils/confidence';
import type { Signal } from '@/types';

interface SignalCardProps {
  signal: Signal;
  onClick?: () => void;
  selected?: boolean;
  variant?: 'compact' | 'detailed';
}

export function SignalCard({
  signal,
  onClick,
  selected = false,
  variant = 'compact',
}: SignalCardProps) {
  const isLong = signal.direction === 'long';
  const signalColor = getSignalColor(signal.direction);
  const confidenceColor = getConfidenceColor(signal.confidence);

  return (
    <Card
      className={cn(
        'cursor-pointer transition-all hover:shadow-lg',
        selected && 'ring-2 ring-blue-500'
      )}
      onClick={onClick}
    >
      <CardContent className="p-4">
        <div className="flex items-start justify-between gap-4">
          {/* Left: Symbol & Direction */}
          <div className="flex items-center gap-3">
            <div
              className="flex h-10 w-10 items-center justify-center rounded-lg"
              style={{ backgroundColor: `${signalColor}20` }}
            >
              {isLong ? (
                <TrendingUp className="h-5 w-5" style={{ color: signalColor }} />
              ) : (
                <TrendingDown className="h-5 w-5" style={{ color: signalColor }} />
              )}
            </div>
            <div>
              <h4 className="font-mono text-lg font-bold text-white">
                {signal.symbol}
              </h4>
              <Badge
                variant={isLong ? 'success' : 'danger'}
                className="mt-1"
              >
                {signal.direction.toUpperCase()}
              </Badge>
            </div>
          </div>

          {/* Right: Confidence */}
          <div className="text-right">
            <div
              className="text-2xl font-bold font-mono"
              style={{ color: confidenceColor }}
            >
              {formatConfidence(signal.confidence)}
            </div>
            <div className="text-xs text-gray-400 mt-1">Confidence</div>
          </div>
        </div>

        {/* Pattern Type & Quality */}
        <div className="mt-3 flex items-center gap-2 flex-wrap">
          <Badge variant="info" className="text-xs">
            {signal.pattern_type.replace(/_/g, ' ')}
          </Badge>
          {signal.metadata?.quality_score && (
            <Badge variant="success" className="text-xs">
              Quality: {(signal.metadata.quality_score * 100).toFixed(0)}%
            </Badge>
          )}
          {signal.metadata?.sde_consensus && (
            <Badge variant="warning" className="text-xs">
              SDE: {signal.metadata.sde_consensus.agreeing_heads}/9
            </Badge>
          )}
        </div>

        {/* Prices */}
        {variant === 'detailed' && (
          <div className="mt-4 grid grid-cols-3 gap-4 text-sm">
            <div>
              <div className="text-gray-400">Entry</div>
              <div className="font-mono font-semibold text-white">
                ${formatPrice(signal.entry_price)}
              </div>
            </div>
            <div>
              <div className="text-gray-400">Stop Loss</div>
              <div className="font-mono font-semibold text-red-400">
                ${formatPrice(signal.stop_loss)}
              </div>
            </div>
            <div>
              <div className="text-gray-400">Take Profit</div>
              <div className="font-mono font-semibold text-green-400">
                ${formatPrice(signal.take_profit)}
              </div>
            </div>
          </div>
        )}

        {/* Entry Proximity Indicator */}
        {signal.metadata?.entry_proximity_status && (
          <div className="mt-3">
            {signal.metadata.entry_proximity_status === 'imminent' && (
              <div className="flex items-center gap-2 rounded-lg bg-green-500/20 border border-green-500/40 p-2">
                <div className="h-2 w-2 rounded-full bg-green-500 animate-pulse" />
                <span className="text-xs font-semibold text-green-400">
                  ENTRY IMMINENT - Ready to Execute
                </span>
              </div>
            )}
            {signal.metadata.entry_proximity_status === 'soon' && (
              <div className="flex items-center gap-2 rounded-lg bg-yellow-500/20 border border-yellow-500/40 p-2">
                <div className="h-2 w-2 rounded-full bg-yellow-500" />
                <span className="text-xs font-semibold text-yellow-400">
                  Entry Soon - Prepare
                </span>
              </div>
            )}
          </div>
        )}

        {/* Timestamp */}
        <div className="mt-3 flex items-center gap-2 text-xs text-gray-400">
          <Clock className="h-3 w-3" />
          {formatDate(signal.timestamp)}
        </div>
      </CardContent>
    </Card>
  );
}

