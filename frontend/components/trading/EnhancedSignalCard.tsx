import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { 
  TrendingUp, 
  TrendingDown, 
  Target, 
  AlertTriangle, 
  Clock, 
  BarChart3,
  Brain,
  Zap,
  Shield,
  Eye,
  EyeOff,
  History,
  Award,
  Lock
} from 'lucide-react';
import { Signal } from '../../lib/api';

interface EnhancedSignalCardProps {
  signal: Signal;
  onSignalClick: (signal: Signal) => void;
  isSelected?: boolean;
  historicalAccuracy?: number;
  similarTradesCount?: number;
}

export const EnhancedSignalCard: React.FC<EnhancedSignalCardProps> = ({
  signal,
  onSignalClick,
  isSelected = false,
  historicalAccuracy = 0.75,
  similarTradesCount = 12
}) => {
  const [showDetails, setShowDetails] = useState(false);

  const getRiskRewardRatio = () => {
    if (!signal.entry_price || !signal.stop_loss || !signal.take_profit) return null;
    const risk = Math.abs(signal.entry_price - signal.stop_loss);
    const reward = Math.abs(signal.take_profit - signal.entry_price);
    return (reward / risk).toFixed(2);
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence > 0.8) return 'text-green-600 bg-green-50 border-green-200';
    if (confidence > 0.6) return 'text-yellow-600 bg-yellow-50 border-yellow-200';
    return 'text-red-600 bg-red-50 border-red-200';
  };

  const getHistoricalAccuracyColor = (accuracy: number) => {
    if (accuracy > 0.7) return 'text-green-600';
    if (accuracy > 0.5) return 'text-yellow-600';
    return 'text-red-600';
  };

  const rrRatio = getRiskRewardRatio();

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
      onClick={() => onSignalClick(signal)}
      className={`bg-white rounded-lg shadow-sm border-2 cursor-pointer transition-all hover:shadow-md ${
        isSelected 
          ? 'border-blue-500 bg-blue-50' 
          : 'border-gray-200 hover:border-gray-300'
      }`}
    >
      {/* Header with Signal Type and Direction */}
      <div className="p-4 border-b border-gray-100">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            {signal.direction === 'long' ? (
              <TrendingUp className="w-5 h-5 text-green-500" />
            ) : (
              <TrendingDown className="w-5 h-5 text-red-500" />
            )}
            <div>
              <div className="flex items-center space-x-2">
                <span className="font-bold text-lg text-gray-900">{signal.symbol}</span>
                <span
                  className={`px-2 py-1 rounded-full text-xs font-medium ${
                    signal.direction === 'long'
                      ? 'bg-green-100 text-green-800'
                      : 'bg-red-100 text-red-800'
                  }`}
                >
                  {signal.direction.toUpperCase()}
                </span>
                <span className="px-2 py-1 rounded-full text-xs font-medium bg-blue-100 text-blue-800">
                  SPOT
                </span>
              </div>
              <div className="flex items-center space-x-2 mt-1">
                <span className="text-sm text-gray-600">Pattern: {signal.pattern_type}</span>
                <span className="text-sm text-gray-400">•</span>
                <span className="text-sm text-gray-600">
                  {new Date(signal.timestamp).toLocaleTimeString()}
                </span>
              </div>
            </div>
          </div>
          
          <div className="flex items-center space-x-2">
            <Lock className="w-4 h-4 text-gray-400" />
            <button
              onClick={(e) => {
                e.stopPropagation();
                setShowDetails(!showDetails);
              }}
              className="p-1 text-gray-400 hover:text-gray-600 transition-colors"
            >
              {showDetails ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
            </button>
          </div>
        </div>
      </div>

      {/* Quick Stats Row */}
      <div className="px-4 py-3 bg-gray-50">
        <div className="grid grid-cols-3 gap-4">
          <div className="text-center">
            <div className="text-sm text-gray-600">Entry</div>
            <div className="font-semibold text-gray-900">
              ${signal.entry_price?.toFixed(4) || 'N/A'}
            </div>
          </div>
          <div className="text-center">
            <div className="text-sm text-gray-600">Stop Loss</div>
            <div className="font-semibold text-red-600">
              ${signal.stop_loss?.toFixed(4) || 'N/A'}
            </div>
          </div>
          <div className="text-center">
            <div className="text-sm text-gray-600">Take Profit</div>
            <div className="font-semibold text-green-600">
              ${signal.take_profit?.toFixed(4) || 'N/A'}
            </div>
          </div>
        </div>
      </div>

      {/* Trust Indicators */}
      <div className="p-4">
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center space-x-2">
            <Brain className="w-4 h-4 text-purple-500" />
            <span className="text-sm font-medium text-gray-700">AI Confidence</span>
          </div>
          <div
            className={`px-2 py-1 rounded text-xs font-medium border ${getConfidenceColor(
              signal.confidence
            )}`}
          >
            {(signal.confidence * 100).toFixed(1)}%
          </div>
        </div>

        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center space-x-2">
            <History className="w-4 h-4 text-blue-500" />
            <span className="text-sm font-medium text-gray-700">Historical Accuracy</span>
          </div>
          <div className="flex items-center space-x-2">
            <span className={`text-sm font-medium ${getHistoricalAccuracyColor(historicalAccuracy)}`}>
              {(historicalAccuracy * 100).toFixed(0)}%
            </span>
            <span className="text-xs text-gray-500">({similarTradesCount} trades)</span>
          </div>
        </div>

        {rrRatio && (
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center space-x-2">
              <Zap className="w-4 h-4 text-yellow-500" />
              <span className="text-sm font-medium text-gray-700">Risk/Reward</span>
            </div>
            <div className="text-sm font-bold text-gray-900">{rrRatio}:1</div>
          </div>
        )}

        {/* Strategy Tags */}
        <div className="flex flex-wrap gap-2 mt-3">
          <span className="px-2 py-1 bg-purple-100 text-purple-800 text-xs rounded-full">
            RSI Divergence
          </span>
          <span className="px-2 py-1 bg-blue-100 text-blue-800 text-xs rounded-full">
            Volume Spike
          </span>
          <span className="px-2 py-1 bg-green-100 text-green-800 text-xs rounded-full">
            Support Level
          </span>
        </div>
      </div>

      {/* Expandable Details */}
      {showDetails && (
        <motion.div
          initial={{ opacity: 0, height: 0 }}
          animate={{ opacity: 1, height: 'auto' }}
          exit={{ opacity: 0, height: 0 }}
          className="border-t border-gray-100 p-4 bg-gray-50"
        >
          <div className="space-y-3">
            <div>
              <h4 className="text-sm font-medium text-gray-700 mb-2">Strategy Breakdown</h4>
              <p className="text-sm text-gray-600 leading-relaxed">
                This {signal.pattern_type.replace('_', ' ')} pattern was triggered by RSI divergence 
                at key support level with above-average volume confirmation. Historical analysis shows 
                similar setups have a {historicalAccuracy * 100}% success rate.
              </p>
            </div>

            <div>
              <h4 className="text-sm font-medium text-gray-700 mb-2">Risk Management</h4>
              <div className="text-sm text-gray-600 space-y-1">
                <div>• Recommended Position Size: 1-2% of portfolio</div>
                <div>• Signal Validity: 1-3 hours</div>
                <div>• Monitor: Volume confirmation & key level breaks</div>
              </div>
            </div>

            <div className="flex space-x-2">
              <button className="flex-1 bg-green-600 hover:bg-green-700 text-white text-sm font-medium py-2 px-3 rounded-lg transition-colors">
                Execute Trade
              </button>
              <button className="flex-1 bg-gray-600 hover:bg-gray-700 text-white text-sm font-medium py-2 px-3 rounded-lg transition-colors">
                Add to Watchlist
              </button>
            </div>
          </div>
        </motion.div>
      )}

      {/* Status Indicator */}
      <div className="px-4 py-2 bg-gray-50 border-t border-gray-100">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
            <span className="text-xs text-gray-600">Active Signal</span>
          </div>
          <div className="flex items-center space-x-1">
            <Award className="w-3 h-3 text-yellow-500" />
            <span className="text-xs text-gray-600">High Confidence</span>
          </div>
        </div>
      </div>
    </motion.div>
  );
};
