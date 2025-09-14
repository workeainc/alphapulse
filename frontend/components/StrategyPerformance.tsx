import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { Target } from 'lucide-react';

interface StrategyPerformanceProps {
  data?: Record<string, {
    total_trades: number;
    winning_trades: number;
    losing_trades: number;
    total_pnl: number;
    win_rate: number;
    avg_win: number;
    avg_loss: number;
  }>;
}

export default function StrategyPerformance({ data }: StrategyPerformanceProps) {
  // Mock data for demonstration
  const mockData = {
    'Mean Reversion': {
      total_trades: 45,
      winning_trades: 28,
      losing_trades: 17,
      total_pnl: 1250,
      win_rate: 62.2,
      avg_win: 85,
      avg_loss: -45,
    },
    'Momentum': {
      total_trades: 32,
      winning_trades: 19,
      losing_trades: 13,
      total_pnl: 890,
      win_rate: 59.4,
      avg_win: 72,
      avg_loss: -38,
    },
    'Arbitrage': {
      total_trades: 18,
      winning_trades: 15,
      losing_trades: 3,
      total_pnl: 420,
      win_rate: 83.3,
      avg_win: 35,
      avg_loss: -25,
    },
  };

  const strategyData = data || mockData;

  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 0,
    }).format(value);
  };

  const chartData = Object.entries(strategyData).map(([name, stats]) => ({
    name,
    winRate: stats.win_rate,
    totalPnl: stats.total_pnl,
  }));

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Target className="h-5 w-5" />
          Strategy Performance
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          {/* Performance Chart */}
          <div className="h-48">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" />
                <YAxis />
                <Tooltip />
                <Bar dataKey="winRate" fill="#8884d8" name="Win Rate %" />
              </BarChart>
            </ResponsiveContainer>
          </div>

          {/* Strategy Details */}
          <div className="grid grid-cols-1 gap-3">
            {Object.entries(strategyData).map(([name, stats]) => (
              <div key={name} className="p-3 border rounded-lg">
                <div className="font-medium mb-2">{name}</div>
                <div className="grid grid-cols-2 gap-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Total Trades:</span>
                    <span>{stats.total_trades}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Win Rate:</span>
                    <span className="font-medium">{stats.win_rate}%</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Total P&L:</span>
                    <span className={`font-medium ${
                      stats.total_pnl >= 0 ? 'text-green-600' : 'text-red-600'
                    }`}>
                      {formatCurrency(stats.total_pnl)}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Avg Win:</span>
                    <span className="text-green-600">{formatCurrency(stats.avg_win)}</span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
