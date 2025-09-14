import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { 
  Calendar, 
  Filter, 
  TrendingUp, 
  TrendingDown, 
  BarChart3, 
  Clock,
  DollarSign,
  Target,
  AlertTriangle,
  CheckCircle,
  XCircle
} from 'lucide-react';
import { useHistoricalSignals, useHistoricalPatterns, usePerformanceAnalytics } from '../../lib/hooks';

interface HistoricalDataProps {
  className?: string;
}

export const HistoricalData: React.FC<HistoricalDataProps> = ({ className = '' }) => {
  const [activeTab, setActiveTab] = useState<'signals' | 'patterns' | 'analytics'>('signals');
  const [filters, setFilters] = useState({
    symbol: '',
    timeframe: '',
    from_date: '',
    to_date: '',
    days: 30
  });

  // Fetch historical data
  const { data: signalsData, isLoading: signalsLoading } = useHistoricalSignals(
    filters.symbol || filters.timeframe || filters.from_date || filters.to_date ? filters : undefined
  );
  
  const { data: patternsData, isLoading: patternsLoading } = useHistoricalPatterns(
    filters.symbol || filters.timeframe || filters.from_date || filters.to_date ? filters : undefined
  );
  
  const { data: analyticsData, isLoading: analyticsLoading } = usePerformanceAnalytics({
    symbol: filters.symbol || undefined,
    timeframe: filters.timeframe || undefined,
    days: filters.days
  });

  const signals = signalsData?.signals || [];
  const patterns = patternsData?.patterns || [];

  const handleFilterChange = (key: string, value: string) => {
    setFilters(prev => ({ ...prev, [key]: value }));
  };

  const getSignalIcon = (direction: string) => {
    if (direction === 'long') {
      return <TrendingUp className="w-4 h-4 text-green-500" />;
    } else if (direction === 'short') {
      return <TrendingDown className="w-4 h-4 text-red-500" />;
    } else {
      return <AlertTriangle className="w-4 h-4 text-yellow-500" />;
    }
  };

  const getSignalStatus = (signal: any) => {
    if (signal.status === 'completed') {
      return signal.pnl && signal.pnl > 0 ? 'profit' : 'loss';
    }
    return signal.status || 'active';
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'profit':
        return <CheckCircle className="w-4 h-4 text-green-500" />;
      case 'loss':
        return <XCircle className="w-4 h-4 text-red-500" />;
      case 'active':
        return <Clock className="w-4 h-4 text-blue-500" />;
      default:
        return <Clock className="w-4 h-4 text-gray-500" />;
    }
  };

  return (
    <div className={`bg-white rounded-lg shadow-sm border border-gray-200 ${className}`}>
      {/* Header */}
      <div className="p-4 border-b border-gray-200">
        <div className="flex items-center justify-between">
          <h2 className="text-lg font-semibold text-gray-900">Historical Data Analysis</h2>
          <div className="flex items-center space-x-2">
            <Filter className="w-4 h-4 text-gray-500" />
            <span className="text-sm text-gray-600">Filters</span>
          </div>
        </div>
      </div>

      {/* Filters */}
      <div className="p-4 border-b border-gray-200 bg-gray-50">
        <div className="grid grid-cols-1 md:grid-cols-5 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Symbol</label>
            <select
              value={filters.symbol}
              onChange={(e) => handleFilterChange('symbol', e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              <option value="">All Symbols</option>
              <option value="BTC/USDT">BTC/USDT</option>
              <option value="ETH/USDT">ETH/USDT</option>
              <option value="ADA/USDT">ADA/USDT</option>
              <option value="SOL/USDT">SOL/USDT</option>
              <option value="BNB/USDT">BNB/USDT</option>
              <option value="XRP/USDT">XRP/USDT</option>
            </select>
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Timeframe</label>
            <select
              value={filters.timeframe}
              onChange={(e) => handleFilterChange('timeframe', e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              <option value="">All Timeframes</option>
              <option value="1m">1m</option>
              <option value="5m">5m</option>
              <option value="15m">15m</option>
              <option value="1h">1h</option>
              <option value="4h">4h</option>
            </select>
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">From Date</label>
            <input
              type="date"
              value={filters.from_date}
              onChange={(e) => handleFilterChange('from_date', e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">To Date</label>
            <input
              type="date"
              value={filters.to_date}
              onChange={(e) => handleFilterChange('to_date', e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Days</label>
            <select
              value={filters.days}
              onChange={(e) => handleFilterChange('days', e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              <option value={7}>7 Days</option>
              <option value={30}>30 Days</option>
              <option value={90}>90 Days</option>
              <option value={180}>180 Days</option>
            </select>
          </div>
        </div>
      </div>

      {/* Tabs */}
      <div className="border-b border-gray-200">
        <nav className="flex space-x-8 px-4">
          <button
            onClick={() => setActiveTab('signals')}
            className={`py-4 px-1 border-b-2 font-medium text-sm ${
              activeTab === 'signals'
                ? 'border-blue-500 text-blue-600'
                : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
            }`}
          >
            Historical Signals ({signals.length})
          </button>
          <button
            onClick={() => setActiveTab('patterns')}
            className={`py-4 px-1 border-b-2 font-medium text-sm ${
              activeTab === 'patterns'
                ? 'border-blue-500 text-blue-600'
                : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
            }`}
          >
            Historical Patterns ({patterns.length})
          </button>
          <button
            onClick={() => setActiveTab('analytics')}
            className={`py-4 px-1 border-b-2 font-medium text-sm ${
              activeTab === 'analytics'
                ? 'border-blue-500 text-blue-600'
                : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
            }`}
          >
            Performance Analytics
          </button>
        </nav>
      </div>

      {/* Content */}
      <div className="p-4">
        {activeTab === 'signals' && (
          <div>
            {signalsLoading ? (
              <div className="text-center py-8">
                <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mx-auto"></div>
                <p className="mt-2 text-gray-600">Loading historical signals...</p>
              </div>
            ) : signals.length === 0 ? (
              <div className="text-center py-8 text-gray-500">
                <Clock className="w-12 h-12 mx-auto mb-4 text-gray-400" />
                <p>No historical signals found</p>
                <p className="text-sm">Try adjusting your filters</p>
              </div>
            ) : (
              <div className="space-y-3">
                {signals.map((signal: any, index: number) => (
                  <motion.div
                    key={signal.signal_id}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.3, delay: index * 0.1 }}
                    className="p-4 border border-gray-200 rounded-lg hover:bg-gray-50 transition-colors"
                  >
                    <div className="flex items-center justify-between">
                      <div className="flex items-center space-x-3">
                        {getSignalIcon(signal.direction)}
                        <div>
                          <div className="flex items-center space-x-2">
                            <span className="font-semibold text-gray-900">{signal.symbol}</span>
                            <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                              signal.direction === 'long' ? 'bg-green-100 text-green-800' :
                              signal.direction === 'short' ? 'bg-red-100 text-red-800' :
                              'bg-yellow-100 text-yellow-800'
                            }`}>
                              {signal.direction.toUpperCase()}
                            </span>
                          </div>
                          <div className="text-sm text-gray-600 mt-1">
                            Pattern: {signal.pattern_type} • Confidence: {(signal.confidence * 100).toFixed(1)}%
                          </div>
                        </div>
                      </div>
                      
                      <div className="flex items-center space-x-4">
                        <div className="text-right">
                          {signal.entry_price && (
                            <div className="text-sm text-gray-600">
                              Entry: ${signal.entry_price.toFixed(2)}
                            </div>
                          )}
                          {signal.pnl !== null && (
                            <div className={`text-sm font-medium ${
                              signal.pnl > 0 ? 'text-green-600' : 'text-red-600'
                            }`}>
                              P&L: ${signal.pnl.toFixed(2)}
                            </div>
                          )}
                        </div>
                        
                        <div className="flex items-center space-x-2">
                          {getStatusIcon(getSignalStatus(signal))}
                          <span className="text-xs text-gray-500">
                            {new Date(signal.signal_generated_at).toLocaleDateString()}
                          </span>
                        </div>
                      </div>
                    </div>
                  </motion.div>
                ))}
              </div>
            )}
          </div>
        )}

        {activeTab === 'patterns' && (
          <div>
            {patternsLoading ? (
              <div className="text-center py-8">
                <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mx-auto"></div>
                <p className="mt-2 text-gray-600">Loading historical patterns...</p>
              </div>
            ) : patterns.length === 0 ? (
              <div className="text-center py-8 text-gray-500">
                <BarChart3 className="w-12 h-12 mx-auto mb-4 text-gray-400" />
                <p>No historical patterns found</p>
                <p className="text-sm">Try adjusting your filters</p>
              </div>
            ) : (
              <div className="space-y-3">
                {patterns.map((pattern: any, index: number) => (
                  <motion.div
                    key={pattern.pattern_id}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.3, delay: index * 0.1 }}
                    className="p-4 border border-gray-200 rounded-lg hover:bg-gray-50 transition-colors"
                  >
                    <div className="flex items-center justify-between">
                      <div className="flex items-center space-x-3">
                        <BarChart3 className="w-4 h-4 text-blue-500" />
                        <div>
                          <div className="flex items-center space-x-2">
                            <span className="font-semibold text-gray-900">{pattern.symbol}</span>
                            <span className="px-2 py-1 rounded-full text-xs font-medium bg-blue-100 text-blue-800">
                              {pattern.pattern_type.replace('_', ' ').toUpperCase()}
                            </span>
                          </div>
                          <div className="text-sm text-gray-600 mt-1">
                            Category: {pattern.pattern_category} • Strength: {pattern.strength}
                          </div>
                        </div>
                      </div>
                      
                      <div className="flex items-center space-x-4">
                        <div className="text-right">
                          <div className="text-sm text-gray-600">
                            Confidence: {(pattern.confidence * 100).toFixed(1)}%
                          </div>
                          <div className="text-sm text-gray-600">
                            R:R: {pattern.risk_reward_ratio.toFixed(2)}
                          </div>
                        </div>
                        
                        <div className="text-xs text-gray-500">
                          {new Date(pattern.pattern_start_time).toLocaleDateString()}
                        </div>
                      </div>
                    </div>
                  </motion.div>
                ))}
              </div>
            )}
          </div>
        )}

        {activeTab === 'analytics' && (
          <div>
            {analyticsLoading ? (
              <div className="text-center py-8">
                <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mx-auto"></div>
                <p className="mt-2 text-gray-600">Loading performance analytics...</p>
              </div>
            ) : !analyticsData ? (
              <div className="text-center py-8 text-gray-500">
                <Target className="w-12 h-12 mx-auto mb-4 text-gray-400" />
                <p>No analytics data available</p>
                <p className="text-sm">Try adjusting your filters</p>
              </div>
            ) : (
              <div className="space-y-6">
                {/* Performance Summary */}
                <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                  <div className="bg-blue-50 rounded-lg p-4">
                    <div className="flex items-center space-x-2 mb-2">
                      <BarChart3 className="w-5 h-5 text-blue-500" />
                      <span className="text-sm font-medium text-gray-700">Total Signals</span>
                    </div>
                    <div className="text-2xl font-bold text-gray-900">{analyticsData.total_signals}</div>
                  </div>
                  
                  <div className="bg-green-50 rounded-lg p-4">
                    <div className="flex items-center space-x-2 mb-2">
                      <Target className="w-5 h-5 text-green-500" />
                      <span className="text-sm font-medium text-gray-700">Win Rate</span>
                    </div>
                    <div className="text-2xl font-bold text-gray-900">{analyticsData.win_rate}%</div>
                  </div>
                  
                  <div className="bg-purple-50 rounded-lg p-4">
                    <div className="flex items-center space-x-2 mb-2">
                      <DollarSign className="w-5 h-5 text-purple-500" />
                      <span className="text-sm font-medium text-gray-700">Avg Return</span>
                    </div>
                    <div className="text-2xl font-bold text-gray-900">${analyticsData.avg_return}</div>
                  </div>
                  
                  <div className="bg-yellow-50 rounded-lg p-4">
                    <div className="flex items-center space-x-2 mb-2">
                      <TrendingUp className="w-5 h-5 text-yellow-500" />
                      <span className="text-sm font-medium text-gray-700">Total P&L</span>
                    </div>
                    <div className={`text-2xl font-bold ${analyticsData.total_pnl >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                      ${analyticsData.total_pnl}
                    </div>
                  </div>
                </div>

                {/* Confidence Analysis */}
                {analyticsData.confidence_analysis && Object.keys(analyticsData.confidence_analysis).length > 0 && (
                  <div className="bg-gray-50 rounded-lg p-4">
                    <h3 className="text-lg font-semibold text-gray-900 mb-3">Confidence Analysis</h3>
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                      <div className="text-center">
                        <div className="text-2xl font-bold text-green-600">{analyticsData.confidence_analysis.high || 0}</div>
                                                 <div className="text-sm text-gray-600">High Confidence (&gt;80%)</div>
                      </div>
                      <div className="text-center">
                        <div className="text-2xl font-bold text-yellow-600">{analyticsData.confidence_analysis.medium || 0}</div>
                        <div className="text-sm text-gray-600">Medium Confidence (60-80%)</div>
                      </div>
                      <div className="text-center">
                        <div className="text-2xl font-bold text-red-600">{analyticsData.confidence_analysis.low || 0}</div>
                                                 <div className="text-sm text-gray-600">Low Confidence (&lt;60%)</div>
                      </div>
                    </div>
                  </div>
                )}

                {/* Direction Analysis */}
                {analyticsData.direction_analysis && Object.keys(analyticsData.direction_analysis).length > 0 && (
                  <div className="bg-gray-50 rounded-lg p-4">
                    <h3 className="text-lg font-semibold text-gray-900 mb-3">Direction Analysis</h3>
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                      {Object.entries(analyticsData.direction_analysis).map(([direction, count]) => (
                        <div key={direction} className="text-center">
                          <div className="text-2xl font-bold text-gray-900">{count as number}</div>
                          <div className="text-sm text-gray-600 capitalize">{direction} Signals</div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};
