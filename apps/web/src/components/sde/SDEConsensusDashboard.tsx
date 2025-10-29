'use client';

import * as React from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/Card';
import { Badge } from '@/components/ui/Badge';
import { formatConfidence } from '@/lib/utils/format';
import { getHeadColor } from '@/lib/utils/confidence';
import { SDEHeadDetail } from './SDEHeadDetail';
import type { SDEConsensus } from '@/types';

interface SDEConsensusDashboardProps {
  consensus?: SDEConsensus | null;
  animated?: boolean;
}

const HEAD_LABELS = {
  technical: 'Technical Analysis',
  sentiment: 'Sentiment Analysis',
  volume: 'Volume Analysis',
  rules: 'Rule-Based',
  ict: 'ICT Concepts',
  wyckoff: 'Wyckoff Method',
  harmonic: 'Harmonic Patterns',
  structure: 'Market Structure',
  crypto: 'Crypto Metrics',
};

export function SDEConsensusDashboard({
  consensus,
  animated = true,
}: SDEConsensusDashboardProps) {
  const [expandedHead, setExpandedHead] = React.useState<string | null>(null);

  if (!consensus || !consensus.heads) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>9-Head Consensus System</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex h-64 items-center justify-center text-gray-400">
            <div className="text-center">
              <p className="text-lg">Waiting for signal selection...</p>
              <p className="text-sm mt-2">Click a signal to see 9-head consensus</p>
            </div>
          </div>
        </CardContent>
      </Card>
    );
  }

  const toggleHead = (headName: string) => {
    setExpandedHead(expandedHead === headName ? null : headName);
  };

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle>9-Head Consensus System</CardTitle>
          <Badge
            variant={consensus.consensus_achieved ? 'success' : 'warning'}
          >
            {consensus.agreeing_heads}/{consensus.total_heads} Agree
          </Badge>
        </div>
        <p className="text-sm text-gray-400 mt-1">
          Click any head to see detailed analysis breakdown
        </p>
      </CardHeader>

      <CardContent className="space-y-4">
        {/* Individual Heads (Expandable) */}
        <div className="space-y-2">
          {Object.entries(consensus.heads).map(([headName, vote]) => {
            const color = getHeadColor(headName);
            const label = HEAD_LABELS[headName as keyof typeof HEAD_LABELS];

            return (
              <SDEHeadDetail
                key={headName}
                headName={headName}
                headLabel={label}
                vote={vote}
                color={color}
                isExpanded={expandedHead === headName}
                onToggle={() => toggleHead(headName)}
              />
            );
          })}
        </div>

        {/* Final Decision */}
        <div className="mt-6 rounded-lg bg-gray-800/50 p-4">
          <div className="grid grid-cols-3 gap-4 text-center">
            <div>
              <div className="text-2xl font-bold font-mono text-white">
                {consensus.final_direction || consensus.direction}
              </div>
              <div className="text-xs text-gray-400 mt-1">Final Direction</div>
            </div>
            <div>
              <div className="text-2xl font-bold font-mono text-green-400">
                {formatConfidence(consensus.final_confidence || consensus.confidence)}
              </div>
              <div className="text-xs text-gray-400 mt-1">Confidence</div>
            </div>
            <div>
              <div className="text-2xl font-bold font-mono text-blue-400">
                {formatConfidence(consensus.consensus_score || consensus.confidence)}
              </div>
              <div className="text-xs text-gray-400 mt-1">Consensus Score</div>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

