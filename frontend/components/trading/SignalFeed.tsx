import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { TrendingUp, TrendingDown, Clock, AlertTriangle, CheckCircle, XCircle } from 'lucide-react';
import { Signal } from '../../lib/api';

interface SignalFeedProps {
  signals: Signal[];
  onSignalClick: (signal: Signal) => void;
  selectedSignal?: Signal;
}

export const SignalFeed: React.FC<SignalFeedProps> = ({
  signals,
  onSignalClick,
  selectedSignal,
}) => {
  const [filter, setFilter] = useState<'all' | 'long' | 'short' | 'neutral'>('all');

  const filteredSignals = signals.filter(signal => {
    if (filter === 'all') return true;
    return signal.direction === filter;
  });

  const getSignalIcon = (direction: string) => {
    if (direction === 'long') {
      return <TrendingUp className="w-4 h-4 text-green-500" />;
    } else if (direction === 'short') {
      return <TrendingDown className="w-4 h-4 text-red-500" />;
    } else {
      return <AlertTriangle className="w-4 h-4 text-yellow-500" />;
    }
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence > 0.8) return 'text-green-600 bg-green-50 border-green-200';
    if (confidence > 0.6) return 'text-yellow-600 bg-yellow-50 border-yellow-200';
    return 'text-red-600 bg-red-50 border-red-200';
  };

  const getSignalStatus = (signal: Signal) => {
    // This would be calculated based on current price vs entry/stop loss
    return 'active'; // active, hit_sl, hit_tp, expired
  };

  return (
    <div className="bg-white rounded-lg shadow-sm border border-gray-200">
      {/* Header */}
      <div className="p-4 border-b border-gray-200">
        <div className="flex items-center justify-between">
          <h2 className="text-lg font-semibold text-gray-900">Live Signal Feed</h2>
          <div className="flex space-x-2">
            <button
              onClick={() => setFilter('all')}
              className={`px-3 py-1 rounded-md text-sm font-medium transition-colors ${
                filter === 'all'
                  ? 'bg-blue-100 text-blue-700'
                  : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
              }`}
            >
              All
            </button>
            <button
              onClick={() => setFilter('long')}
              className={`px-3 py-1 rounded-md text-sm font-medium transition-colors ${
                filter === 'long'
                  ? 'bg-green-100 text-green-700'
                  : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
              }`}
            >
              Long
            </button>
            <button
              onClick={() => setFilter('short')}
              className={`px-3 py-1 rounded-md text-sm font-medium transition-colors ${
                filter === 'short'
                  ? 'bg-red-100 text-red-700'
                  : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
              }`}
            >
              Short
            </button>
            <button
              onClick={() => setFilter('neutral')}
              className={`px-3 py-1 rounded-md text-sm font-medium transition-colors ${
                filter === 'neutral'
                  ? 'bg-gray-100 text-gray-700'
                  : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
              }`}
            >
              Neutral
            </button>
          </div>
        </div>
      </div>

      {/* Signal List */}
      <div className="max-h-96 overflow-y-auto">
        <AnimatePresence>
          {filteredSignals.length === 0 ? (
            <div className="p-8 text-center text-gray-500">
              <Clock className="w-8 h-8 mx-auto mb-2 text-gray-400" />
              <p>No signals available</p>
              <p className="text-sm">New signals will appear here in real-time</p>
            </div>
          ) : (
            filteredSignals.map((signal, index) => (
              <motion.div
                key={`${signal.symbol}-${signal.timestamp}`}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                transition={{ duration: 0.3, delay: index * 0.1 }}
                onClick={() => onSignalClick(signal)}
                className={`p-4 border-b border-gray-100 cursor-pointer transition-all hover:bg-gray-50 ${
                  selectedSignal?.symbol === signal.symbol &&
                  selectedSignal?.timestamp === signal.timestamp
                    ? 'bg-blue-50 border-blue-200'
                    : ''
                }`}
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-3">
                    {getSignalIcon(signal.direction)}
                    <div>
                      <div className="flex items-center space-x-2">
                        <span className="font-semibold text-gray-900">
                          {signal.symbol}
                        </span>
                        <span
                          className={`px-2 py-1 rounded-full text-xs font-medium ${
                            signal.direction === 'long'
                              ? 'bg-green-100 text-green-800'
                              : 'bg-red-100 text-red-800'
                          }`}
                        >
                          {signal.direction.toUpperCase()}
                        </span>
                      </div>
                      <div className="flex items-center space-x-4 mt-1 text-sm text-gray-600">
                        <span>Pattern: {signal.pattern_type}</span>
                        {signal.entry_price && (
                          <span>Entry: ${signal.entry_price.toFixed(2)}</span>
                        )}
                      </div>
                    </div>
                  </div>
                  
                  <div className="flex items-center space-x-3">
                    <div className="text-right">
                      <div
                        className={`px-2 py-1 rounded text-xs font-medium border ${getConfidenceColor(
                          signal.confidence
                        )}`}
                      >
                        {(signal.confidence * 100).toFixed(1)}%
                      </div>
                      <div className="text-xs text-gray-500 mt-1">
                        {new Date(signal.timestamp).toLocaleTimeString()}
                      </div>
                    </div>
                    
                    {/* Status Indicator */}
                    <div className="flex items-center space-x-1">
                      {getSignalStatus(signal) === 'active' && (
                        <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
                      )}
                      {getSignalStatus(signal) === 'hit_sl' && (
                        <XCircle className="w-4 h-4 text-red-500" />
                      )}
                      {getSignalStatus(signal) === 'hit_tp' && (
                        <CheckCircle className="w-4 h-4 text-green-500" />
                      )}
                    </div>
                  </div>
                </div>
              </motion.div>
            ))
          )}
        </AnimatePresence>
      </div>
    </div>
  );
};
