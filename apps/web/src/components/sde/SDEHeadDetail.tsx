'use client';

import * as React from 'react';
import { ChevronDown, ChevronUp, Activity } from 'lucide-react';
import { cn } from '@/lib/utils/cn';

interface SDEHeadDetailProps {
  headName: string;
  headLabel: string;
  vote: any;
  color: string;
  isExpanded: boolean;
  onToggle: () => void;
}

export function SDEHeadDetail({
  headName,
  headLabel,
  vote,
  color,
  isExpanded,
  onToggle,
}: SDEHeadDetailProps) {
  return (
    <div className="rounded-lg border border-gray-700 overflow-hidden">
      {/* Head Summary (Clickable) */}
      <button
        onClick={onToggle}
        className="w-full flex items-center gap-3 p-3 hover:bg-gray-800/30 transition-all"
      >
        {/* Head Name & Icon */}
        <div className="flex items-center gap-2 w-48">
          <div
            className="h-2 w-2 rounded-full"
            style={{ backgroundColor: color }}
          />
          <span className="text-sm font-medium text-gray-300">
            {headLabel}
          </span>
        </div>

        {/* Confidence Bar */}
        <div className="flex-1">
          <div className="h-2 w-full rounded-full bg-gray-800 overflow-hidden">
            <div
              className="h-full transition-all duration-500 rounded-full"
              style={{
                width: `${vote.confidence * 100}%`,
                backgroundColor: color,
              }}
            />
          </div>
        </div>

        {/* Direction & Confidence */}
        <div className="flex items-center gap-2 w-32">
          <span className={cn(
            'text-sm font-semibold',
            vote.direction === 'LONG' && 'text-green-400',
            vote.direction === 'SHORT' && 'text-red-400',
            vote.direction === 'FLAT' && 'text-gray-400'
          )}>
            {vote.direction}
          </span>
          <span className="text-sm text-gray-400 font-mono">
            {(vote.confidence * 100).toFixed(1)}%
          </span>
        </div>

        {/* Expand Icon */}
        <div className="w-6">
          {isExpanded ? (
            <ChevronUp className="h-4 w-4 text-gray-400" />
          ) : (
            <ChevronDown className="h-4 w-4 text-gray-400" />
          )}
        </div>
      </button>

      {/* Detailed Breakdown (Expandable) */}
      {isExpanded && (
        <div className="border-t border-gray-700 bg-gray-900/50 p-4 space-y-4">
          {/* Timestamp & Update Status */}
          {(vote.timestamp || vote.last_updated) && (
            <div className="flex items-center justify-between pb-2 border-b border-gray-700/50">
              <div className="flex items-center gap-2">
                <div className="h-2 w-2 rounded-full bg-green-500 animate-pulse" />
                <span className="text-xs text-green-400 font-semibold">
                  {vote.last_updated || 'Live'}
                </span>
              </div>
              {vote.timestamp && (
                <span className="text-xs text-gray-500">
                  {new Date(vote.timestamp).toLocaleTimeString()}
                </span>
              )}
            </div>
          )}

          {/* Logic & Reasoning */}
          <div>
            <div className="flex items-center gap-2 mb-2">
              <Activity className="h-4 w-4" style={{ color }} />
              <span className="text-xs font-semibold text-gray-400 uppercase">
                Analysis Logic
              </span>
            </div>
            <p className="text-sm text-gray-300 mb-1">{vote.logic}</p>
            <p className="text-sm text-gray-400 italic">{vote.reasoning}</p>
          </div>

          {/* Score Breakdown (if available) */}
          {vote.score_breakdown && (
            <div>
              <div className="text-xs font-semibold text-gray-400 uppercase mb-2">
                Score Breakdown
              </div>
              <div className="space-y-2">
                {Object.entries(vote.score_breakdown).map(([key, value]: [string, any]) => (
                  <div key={key} className="space-y-1">
                    <div className="flex justify-between text-xs">
                      <span className="text-gray-400">{key.replace(/_/g, ' ')}</span>
                      <span className="font-mono text-white">
                        {typeof value === 'number' ? (value * 100).toFixed(0) + '%' : value}
                      </span>
                    </div>
                    {typeof value === 'number' && (
                      <div className="h-1.5 w-full rounded-full bg-gray-800 overflow-hidden">
                        <div
                          className="h-full rounded-full transition-all"
                          style={{
                            width: `${value * 100}%`,
                            backgroundColor: color,
                          }}
                        />
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Indicators Used */}
          {vote.indicators && (
            <div>
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs font-semibold text-gray-400 uppercase">
                  Real-Time Indicators
                </span>
                <span className="text-xs text-green-400 font-mono">LIVE</span>
              </div>
              <div className="grid grid-cols-2 gap-2">
                {Object.entries(vote.indicators).map(([key, value]: [string, any]) => {
                  const isNumeric = typeof value === 'number';
                  const displayValue = isNumeric
                    ? value % 1 === 0
                      ? value.toString()
                      : value.toFixed(value < 1 ? 4 : 2)
                    : value;
                  
                  return (
                    <div
                      key={key}
                      className="flex flex-col bg-gray-800/50 rounded px-3 py-2 hover:bg-gray-800/70 transition-colors"
                    >
                      <span className="text-xs text-gray-500 mb-1">
                        {key.replace(/_/g, ' ')}
                      </span>
                      <span className={cn(
                        "text-sm font-mono font-semibold",
                        isNumeric ? "text-cyan-400" : "text-white"
                      )}>
                        {value === null || value === undefined ? 'N/A' : displayValue}
                      </span>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {/* Confirming Factors */}
          {vote.factors && vote.factors.length > 0 && (
            <div>
              <div className="text-xs font-semibold text-gray-400 uppercase mb-2">
                Confirming Factors ({vote.factors.length})
              </div>
              <ul className="space-y-1">
                {vote.factors.map((factor: string, idx: number) => (
                  <li
                    key={idx}
                    className="flex items-start gap-2 text-sm text-gray-300 py-1"
                  >
                    <span
                      className="mt-1.5 h-1.5 w-1.5 rounded-full flex-shrink-0 animate-pulse"
                      style={{ backgroundColor: color }}
                    />
                    <span>{factor}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

