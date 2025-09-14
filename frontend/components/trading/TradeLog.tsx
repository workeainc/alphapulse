import React, { useState } from 'react';
import { TrendingUp, TrendingDown, CheckCircle, XCircle, Clock, DollarSign } from 'lucide-react';

interface Trade {
  id: string;
  symbol: string;
  direction: 'long' | 'short';
  entryPrice: number;
  exitPrice?: number;
  stopLoss: number;
  takeProfit: number;
  status: 'open' | 'closed' | 'hit_sl' | 'hit_tp';
  entryTime: string;
  exitTime?: string;
  pnl?: number;
  pnlPercentage?: number;
  riskAmount: number;
  notes?: string;
}

interface TradeLogProps {
  trades: Trade[];
  onAddTrade: (trade: Partial<Trade>) => void;
}

export const TradeLog: React.FC<TradeLogProps> = ({ trades, onAddTrade }) => {
  const [activeTab, setActiveTab] = useState<'all' | 'open' | 'closed'>('all');
  const [showAddForm, setShowAddForm] = useState(false);

  const filteredTrades = trades.filter(trade => {
    if (activeTab === 'all') return true;
    if (activeTab === 'open') return trade.status === 'open';
    if (activeTab === 'closed') return trade.status !== 'open';
    return true;
  });

  const getTotalPnL = () => {
    return trades
      .filter(trade => trade.pnl !== undefined)
      .reduce((sum, trade) => sum + (trade.pnl || 0), 0);
  };

  const getWinRate = () => {
    const closedTrades = trades.filter(trade => trade.status !== 'open');
    const winningTrades = closedTrades.filter(trade => (trade.pnl || 0) > 0);
    return closedTrades.length > 0 ? (winningTrades.length / closedTrades.length) * 100 : 0;
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'open':
        return <Clock className="w-4 h-4 text-blue-500" />;
      case 'hit_tp':
        return <CheckCircle className="w-4 h-4 text-green-500" />;
      case 'hit_sl':
        return <XCircle className="w-4 h-4 text-red-500" />;
      case 'closed':
        return <CheckCircle className="w-4 h-4 text-gray-500" />;
      default:
        return <Clock className="w-4 h-4 text-gray-400" />;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'open':
        return 'text-blue-600 bg-blue-50 border-blue-200';
      case 'hit_tp':
        return 'text-green-600 bg-green-50 border-green-200';
      case 'hit_sl':
        return 'text-red-600 bg-red-50 border-red-200';
      case 'closed':
        return 'text-gray-600 bg-gray-50 border-gray-200';
      default:
        return 'text-gray-600 bg-gray-50 border-gray-200';
    }
  };

  return (
    <div className="bg-white rounded-lg shadow-sm border border-gray-200">
      {/* Header */}
      <div className="p-4 border-b border-gray-200">
        <div className="flex items-center justify-between">
          <h2 className="text-lg font-semibold text-gray-900">Trade Log</h2>
          <button
            onClick={() => setShowAddForm(!showAddForm)}
            className="px-3 py-1 bg-blue-600 text-white text-sm rounded-md hover:bg-blue-700 transition-colors"
          >
            {showAddForm ? 'Cancel' : 'Add Trade'}
          </button>
        </div>
      </div>

      {/* Performance Summary */}
      <div className="p-4 border-b border-gray-200 bg-gray-50">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div className="text-center">
            <div className="text-2xl font-bold text-gray-900">{trades.length}</div>
            <div className="text-sm text-gray-600">Total Trades</div>
          </div>
          <div className="text-center">
            <div className={`text-2xl font-bold ${getTotalPnL() >= 0 ? 'text-green-600' : 'text-red-600'}`}>
              ${getTotalPnL().toFixed(2)}
            </div>
            <div className="text-sm text-gray-600">Total P&L</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-gray-900">{getWinRate().toFixed(1)}%</div>
            <div className="text-sm text-gray-600">Win Rate</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-gray-900">
              {trades.filter(t => t.status === 'open').length}
            </div>
            <div className="text-sm text-gray-600">Open Trades</div>
          </div>
        </div>
      </div>

      {/* Tabs */}
      <div className="flex border-b border-gray-200">
        <button
          onClick={() => setActiveTab('all')}
          className={`px-4 py-2 text-sm font-medium transition-colors ${
            activeTab === 'all'
              ? 'text-blue-600 border-b-2 border-blue-600'
              : 'text-gray-500 hover:text-gray-700'
          }`}
        >
          All Trades
        </button>
        <button
          onClick={() => setActiveTab('open')}
          className={`px-4 py-2 text-sm font-medium transition-colors ${
            activeTab === 'open'
              ? 'text-blue-600 border-b-2 border-blue-600'
              : 'text-gray-500 hover:text-gray-700'
          }`}
        >
          Open
        </button>
        <button
          onClick={() => setActiveTab('closed')}
          className={`px-4 py-2 text-sm font-medium transition-colors ${
            activeTab === 'closed'
              ? 'text-blue-600 border-b-2 border-blue-600'
              : 'text-gray-500 hover:text-gray-700'
          }`}
        >
          Closed
        </button>
      </div>

      {/* Trade List */}
      <div className="max-h-96 overflow-y-auto">
        {filteredTrades.length === 0 ? (
          <div className="p-8 text-center text-gray-500">
            <DollarSign className="w-8 h-8 mx-auto mb-2 text-gray-400" />
            <p>No trades found</p>
            <p className="text-sm">Your executed trades will appear here</p>
          </div>
        ) : (
          <div className="divide-y divide-gray-200">
            {filteredTrades.map((trade) => (
              <div key={trade.id} className="p-4 hover:bg-gray-50 transition-colors">
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-3">
                    {trade.direction === 'long' ? (
                      <TrendingUp className="w-4 h-4 text-green-500" />
                    ) : (
                      <TrendingDown className="w-4 h-4 text-red-500" />
                    )}
                    <div>
                      <div className="flex items-center space-x-2">
                        <span className="font-semibold text-gray-900">{trade.symbol}</span>
                        <span
                          className={`px-2 py-1 rounded-full text-xs font-medium ${
                            trade.direction === 'long'
                              ? 'bg-green-100 text-green-800'
                              : 'bg-red-100 text-red-800'
                          }`}
                        >
                          {trade.direction.toUpperCase()}
                        </span>
                        <span
                          className={`px-2 py-1 rounded text-xs font-medium border ${getStatusColor(
                            trade.status
                          )}`}
                        >
                          {trade.status.replace('_', ' ').toUpperCase()}
                        </span>
                      </div>
                      <div className="flex items-center space-x-4 mt-1 text-sm text-gray-600">
                        <span>Entry: ${trade.entryPrice.toFixed(2)}</span>
                        {trade.exitPrice && (
                          <span>Exit: ${trade.exitPrice.toFixed(2)}</span>
                        )}
                        <span>Risk: ${trade.riskAmount.toFixed(2)}</span>
                      </div>
                    </div>
                  </div>
                  
                  <div className="text-right">
                    {trade.pnl !== undefined && (
                      <div
                        className={`text-lg font-bold ${
                          trade.pnl >= 0 ? 'text-green-600' : 'text-red-600'
                        }`}
                      >
                        ${trade.pnl.toFixed(2)}
                      </div>
                    )}
                    <div className="text-xs text-gray-500">
                      {new Date(trade.entryTime).toLocaleDateString()}
                    </div>
                  </div>
                </div>
                
                {trade.notes && (
                  <div className="mt-2 text-sm text-gray-600 bg-gray-50 p-2 rounded">
                    {trade.notes}
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};
