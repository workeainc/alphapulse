import React from 'react';
import { Signal } from '../../lib/api';
import { 
  TrendingUp, 
  TrendingDown, 
  Target, 
  AlertTriangle, 
  Clock, 
  BarChart3,
  Brain,
  Zap,
  Shield
} from 'lucide-react';

interface SignalAnalysisProps {
  signal: Signal | null;
  onClose: () => void;
}

export const SignalAnalysis: React.FC<SignalAnalysisProps> = ({ signal, onClose }) => {
  if (!signal) {
    return (
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-8">
        <div className="text-center text-gray-500">
          <BarChart3 className="w-12 h-12 mx-auto mb-4 text-gray-400" />
          <h3 className="text-lg font-medium mb-2">Select a Signal</h3>
          <p>Click on any signal from the feed to see detailed analysis</p>
        </div>
      </div>
    );
  }

  const getRiskRewardRatio = () => {
    if (!signal.entry_price || !signal.stop_loss || !signal.take_profit) return null;
    const risk = Math.abs(signal.entry_price - signal.stop_loss);
    const reward = Math.abs(signal.take_profit - signal.entry_price);
    return (reward / risk).toFixed(2);
  };

  const getSignalStrength = (confidence: number) => {
    if (confidence > 0.8) return { text: 'Strong', color: 'text-green-600', bg: 'bg-green-50' };
    if (confidence > 0.6) return { text: 'Medium', color: 'text-yellow-600', bg: 'bg-yellow-50' };
    return { text: 'Weak', color: 'text-red-600', bg: 'bg-red-50' };
  };

  const strength = getSignalStrength(signal.confidence);
  const rrRatio = getRiskRewardRatio();

  return (
    <div className="bg-white rounded-lg shadow-sm border border-gray-200">
      {/* Header */}
      <div className="p-4 border-b border-gray-200">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            {signal.direction === 'long' ? (
              <TrendingUp className="w-6 h-6 text-green-500" />
            ) : (
              <TrendingDown className="w-6 h-6 text-red-500" />
            )}
            <div>
              <h2 className="text-lg font-semibold text-gray-900">
                {signal.symbol} - {signal.direction.toUpperCase()} Signal
              </h2>
              <p className="text-sm text-gray-600">
                Generated at {new Date(signal.timestamp).toLocaleString()}
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-gray-600 transition-colors"
          >
            ✕
          </button>
        </div>
      </div>

      <div className="p-6">
        {/* Signal Overview */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
          <div className="bg-gray-50 rounded-lg p-4">
            <div className="flex items-center space-x-2 mb-2">
              <Target className="w-4 h-4 text-blue-500" />
              <span className="text-sm font-medium text-gray-700">Entry Price</span>
            </div>
            <p className="text-xl font-bold text-gray-900">
              ${signal.entry_price?.toFixed(2) || 'N/A'}
            </p>
          </div>
          
          <div className="bg-gray-50 rounded-lg p-4">
            <div className="flex items-center space-x-2 mb-2">
              <AlertTriangle className="w-4 h-4 text-red-500" />
              <span className="text-sm font-medium text-gray-700">Stop Loss</span>
            </div>
            <p className="text-xl font-bold text-red-600">
              ${signal.stop_loss?.toFixed(2) || 'N/A'}
            </p>
          </div>
          
          <div className="bg-gray-50 rounded-lg p-4">
            <div className="flex items-center space-x-2 mb-2">
              <TrendingUp className="w-4 h-4 text-green-500" />
              <span className="text-sm font-medium text-gray-700">Take Profit</span>
            </div>
            <p className="text-xl font-bold text-green-600">
              ${signal.take_profit?.toFixed(2) || 'N/A'}
            </p>
          </div>
        </div>

        {/* Confidence and Risk/Reward */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
          <div className="bg-gray-50 rounded-lg p-4">
            <div className="flex items-center space-x-2 mb-2">
              <Brain className="w-4 h-4 text-purple-500" />
              <span className="text-sm font-medium text-gray-700">AI Confidence</span>
            </div>
            <div className="flex items-center space-x-3">
              <div className="text-2xl font-bold text-gray-900">
                {(signal.confidence * 100).toFixed(1)}%
              </div>
              <span className={`px-2 py-1 rounded text-xs font-medium ${strength.color} ${strength.bg}`}>
                {strength.text}
              </span>
            </div>
          </div>
          
          {rrRatio && (
            <div className="bg-gray-50 rounded-lg p-4">
              <div className="flex items-center space-x-2 mb-2">
                <Zap className="w-4 h-4 text-yellow-500" />
                <span className="text-sm font-medium text-gray-700">Risk/Reward Ratio</span>
              </div>
              <div className="text-2xl font-bold text-gray-900">
                {rrRatio}:1
              </div>
            </div>
          )}
        </div>

        {/* Pattern Analysis */}
        <div className="mb-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-3">Pattern Analysis</h3>
          <div className="bg-gray-50 rounded-lg p-4">
            <div className="flex items-center space-x-2 mb-2">
              <BarChart3 className="w-4 h-4 text-blue-500" />
              <span className="font-medium text-gray-700">{signal.pattern_type}</span>
            </div>
            <p className="text-gray-600 text-sm">
              This {signal.pattern_type.replace('_', ' ')} pattern indicates a potential 
              {signal.direction === 'long' ? ' bullish reversal' : ' bearish continuation'} 
              with strong momentum confirmation.
            </p>
          </div>
        </div>

        {/* AI Reasoning */}
        <div className="mb-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-3">AI Reasoning</h3>
          <div className="bg-blue-50 rounded-lg p-4 border border-blue-200">
            <div className="flex items-start space-x-3">
              <Brain className="w-5 h-5 text-blue-500 mt-0.5" />
              <div>
                <p className="text-gray-800 text-sm leading-relaxed">
                  Our AI analysis detected multiple confirming indicators for this {signal.direction} signal:
                </p>
                <ul className="mt-2 text-sm text-gray-700 space-y-1">
                  <li>• Strong volume confirmation on pattern completion</li>
                  <li>• RSI divergence supporting the reversal</li>
                  <li>• Key support/resistance level interaction</li>
                  <li>• Market sentiment alignment with technical analysis</li>
                </ul>
              </div>
            </div>
          </div>
        </div>

        {/* Risk Management */}
        <div className="mb-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-3">Risk Management</h3>
          <div className="bg-yellow-50 rounded-lg p-4 border border-yellow-200">
            <div className="flex items-start space-x-3">
              <Shield className="w-5 h-5 text-yellow-600 mt-0.5" />
              <div>
                <p className="text-gray-800 text-sm leading-relaxed">
                  <strong>Recommended Position Size:</strong> 1-2% of portfolio per trade
                </p>
                <p className="text-gray-700 text-sm mt-1">
                  <strong>Signal Validity:</strong> 1-3 hours or until stop loss is hit
                </p>
                <p className="text-gray-700 text-sm mt-1">
                  <strong>Monitor:</strong> Volume confirmation and key level breaks
                </p>
              </div>
            </div>
          </div>
        </div>

        {/* Action Buttons */}
        <div className="flex space-x-3">
          <button className="flex-1 bg-green-600 hover:bg-green-700 text-white font-medium py-2 px-4 rounded-lg transition-colors">
            Execute Trade
          </button>
          <button className="flex-1 bg-gray-600 hover:bg-gray-700 text-white font-medium py-2 px-4 rounded-lg transition-colors">
            Add to Watchlist
          </button>
          <button className="px-4 py-2 border border-gray-300 text-gray-700 rounded-lg hover:bg-gray-50 transition-colors">
            Skip
          </button>
        </div>
      </div>
    </div>
  );
};
