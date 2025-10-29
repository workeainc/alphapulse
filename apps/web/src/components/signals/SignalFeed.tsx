'use client';

import * as React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { SignalCard } from './SignalCard';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/Card';
import { useSignalStore } from '@/store/signalStore';
import type { Signal } from '@/types';

interface SignalFeedProps {
  signals: Signal[];
  maxItems?: number;
}

export function SignalFeed({ signals, maxItems = 10 }: SignalFeedProps) {
  const { selectedSignal, selectSignal } = useSignalStore();
  const displaySignals = signals.slice(0, maxItems);
  
  const handleSignalClick = (signal: Signal) => {
    console.log('Signal clicked:', signal);
    console.log('Has SDE data:', !!signal.metadata?.sde_consensus);
    console.log('Has MTF data:', !!signal.metadata?.mtf_analysis);
    selectSignal(signal);
  };

  return (
    <Card className="h-full flex flex-col">
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle>Live Signals</CardTitle>
          <div className="flex items-center gap-2">
            <div className="h-2 w-2 rounded-full bg-green-500 animate-pulse" />
            <span className="text-sm text-gray-400">{signals.length} Active</span>
          </div>
        </div>
      </CardHeader>

      <CardContent className="flex-1 overflow-y-auto">
        {displaySignals.length === 0 ? (
          <div className="flex h-full items-center justify-center text-gray-400">
            <div className="text-center">
              <p className="text-lg font-semibold">No signals yet</p>
              <p className="text-sm mt-2">Waiting for new trading signals...</p>
            </div>
          </div>
        ) : (
          <div className="space-y-3">
            <AnimatePresence mode="popLayout">
              {displaySignals.map((signal, index) => (
                <motion.div
                  key={`${signal.symbol}-${signal.timestamp}`}
                  initial={{ opacity: 0, y: -20 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, x: -100 }}
                  transition={{ duration: 0.3, delay: index * 0.05 }}
                >
                  <SignalCard
                    signal={signal}
                    selected={selectedSignal?.symbol === signal.symbol}
                    onClick={() => handleSignalClick(signal)}
                    variant="detailed"
                  />
                </motion.div>
              ))}
            </AnimatePresence>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

