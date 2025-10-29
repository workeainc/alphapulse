'use client';

import * as React from 'react';
import { TrendingUp, TrendingDown, AlertTriangle } from 'lucide-react';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/Card';
import { Badge } from '@/components/ui/Badge';
import { cn } from '@/lib/utils/cn';
import { formatConfidence } from '@/lib/utils/format';
import type { MTFSignal, Timeframe } from '@/types';

interface MTFAnalysisPanelProps {
  mtfSignal?: MTFSignal | null;
  showBoostCalculation?: boolean;
}

const TIMEFRAME_ORDER: Timeframe[] = ['1d', '4h', '1h', '15m', '5m', '1m'];

export function MTFAnalysisPanel({
  mtfSignal,
  showBoostCalculation = true,
}: MTFAnalysisPanelProps) {
  // Handle both full MTFSignal and simplified mtf_analysis from backend
  const mtfData = mtfSignal as any;
  
  if (!mtfData || (!mtfData.timeframe_votes && !mtfData.base_confidence)) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Multi-Timeframe Analysis</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex h-64 items-center justify-center text-gray-400">
            <div className="text-center">
              <p className="text-lg">Waiting for MTF data...</p>
              <p className="text-sm mt-2">Click a signal to see multi-timeframe analysis</p>
            </div>
          </div>
        </CardContent>
      </Card>
    );
  }

  const isPerfectAlignment = mtfData.alignment_status === 'perfect';
  const isDivergent = mtfData.alignment_status === 'divergent';
  const timeframeVotes = mtfData.timeframe_votes || {};
  const contributingTimeframes = Object.keys(timeframeVotes) as Timeframe[];

  return (
    <Card className={cn(
      'transition-all',
      isPerfectAlignment && 'ring-2 ring-green-500',
      isDivergent && 'ring-2 ring-red-500'
    )}>
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle>Multi-Timeframe Analysis</CardTitle>
          <Badge
            variant={isPerfectAlignment ? 'success' : isDivergent ? 'danger' : 'warning'}
          >
            {mtfSignal.alignment_status}
          </Badge>
        </div>
      </CardHeader>

      <CardContent className="space-y-4">
        {/* Timeframe Breakdown */}
        <div className="space-y-2">
          {TIMEFRAME_ORDER.filter(tf => contributingTimeframes.includes(tf)).map((timeframe) => {
            const vote = timeframeVotes[timeframe];
            if (!vote) return null;

            const isBase = timeframe === mtfData.base_timeframe;
            const isLong = vote.signal_type === 'LONG';
            const signalColor = isLong ? '#10B981' : vote.signal_type === 'SHORT' ? '#EF4444' : '#6B7280';

            return (
              <div
                key={timeframe}
                className={cn(
                  'flex items-center gap-3 rounded-lg p-3',
                  isBase && 'bg-blue-500/10 border border-blue-500/30'
                )}
              >
                {/* Timeframe Label */}
                <div className="flex items-center gap-2 w-20">
                  <span className={cn(
                    'text-sm font-semibold font-mono',
                    isBase ? 'text-blue-400' : 'text-gray-300'
                  )}>
                    {timeframe}
                  </span>
                  {isBase && (
                    <Badge variant="info" className="text-xs px-1.5 py-0">
                      BASE
                    </Badge>
                  )}
                </div>

                {/* Confidence Bar */}
                <div className="flex-1">
                  <div className="h-3 w-full rounded-full bg-gray-800 overflow-hidden">
                    <div
                      className="h-full transition-all duration-500 rounded-full"
                      style={{
                        width: `${vote.confidence * 100}%`,
                        backgroundColor: signalColor,
                      }}
                    />
                  </div>
                </div>

                {/* Signal Info */}
                <div className="flex items-center gap-3 w-48">
                  <div className="flex items-center gap-1">
                    {isLong ? (
                      <TrendingUp className="h-4 w-4 text-green-400" />
                    ) : vote.signal_type === 'SHORT' ? (
                      <TrendingDown className="h-4 w-4 text-red-400" />
                    ) : null}
                    <span className={cn(
                      'text-sm font-semibold',
                      isLong ? 'text-green-400' : vote.signal_type === 'SHORT' ? 'text-red-400' : 'text-gray-400'
                    )}>
                      {vote.signal_type}
                    </span>
                  </div>
                  <span className="text-sm text-gray-400 font-mono">
                    {formatConfidence(vote.confidence)}
                  </span>
                  {showBoostCalculation && !isBase && (
                    <span className="text-xs text-blue-400 font-mono">
                      +{formatConfidence(vote.contribution)}
                    </span>
                  )}
                </div>
              </div>
            );
          })}
        </div>

        {/* MTF Boost Summary */}
        {showBoostCalculation && mtfData.base_confidence !== undefined && (
          <div className="mt-4 rounded-lg bg-gray-800/50 p-4">
            <div className="grid grid-cols-3 gap-4 text-center">
              <div>
                <div className="text-xl font-bold font-mono text-white">
                  {formatConfidence(mtfData.base_confidence)}
                </div>
                <div className="text-xs text-gray-400 mt-1">Base Confidence</div>
              </div>
              <div>
                <div className="text-xl font-bold font-mono text-blue-400">
                  +{formatConfidence(mtfData.mtf_boost || 0)}
                </div>
                <div className="text-xs text-gray-400 mt-1">MTF Boost</div>
              </div>
              <div>
                <div className="text-xl font-bold font-mono text-green-400">
                  {formatConfidence(mtfData.final_confidence)}
                </div>
                <div className="text-xs text-gray-400 mt-1">Final Confidence</div>
              </div>
            </div>
          </div>
        )}

        {/* Divergence Warning */}
        {isDivergent && (
          <div className="flex items-center gap-2 rounded-lg bg-red-500/10 border border-red-500/30 p-3">
            <AlertTriangle className="h-5 w-5 text-red-400" />
            <div className="flex-1">
              <div className="text-sm font-semibold text-red-400">
                Timeframe Divergence Detected
              </div>
              <div className="text-xs text-gray-400 mt-1">
                Lower timeframes contradict higher timeframes. Exercise caution.
              </div>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

