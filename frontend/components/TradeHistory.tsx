import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { History, TrendingUp, TrendingDown } from 'lucide-react';

interface TradeHistoryProps {
  data?: Array<{
    id: string;
    symbol: string;
    side: 'buy' | 'sell';
    amount: number;
    price: number;
    timestamp: string;
    pnl?: number;
  }>;
}

export default function TradeHistory({ data }: TradeHistoryProps) {
  // Mock data for demonstration
  const mockData = [
    { id: '1', symbol: 'BTC/USDT', side: 'buy', amount: 0.1, price: 45000, timestamp: '2024-01-15 14:30', pnl: 150 },
    { id: '2', symbol: 'ETH/USDT', side: 'sell', amount: 1.5, price: 2800, timestamp: '2024-01-15 13:45', pnl: -50 },
    { id: '3', symbol: 'BTC/USDT', side: 'sell', amount: 0.05, price: 45200, timestamp: '2024-01-15 12:20', pnl: 100 },
    { id: '4', symbol: 'SOL/USDT', side: 'buy', amount: 10, price: 95, timestamp: '2024-01-15 11:15', pnl: 0 },
  ];

  const tradeData = data || mockData;

  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2,
    }).format(value);
  };

  const formatTimestamp = (timestamp: string) => {
    return new Date(timestamp).toLocaleString();
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <History className="h-5 w-5" />
          Recent Trades
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-3">
          {tradeData.map((trade) => (
            <div key={trade.id} className="flex items-center justify-between p-3 border rounded-lg">
              <div className="flex items-center gap-3">
                <div className={`p-2 rounded-full ${
                  trade.side === 'buy' ? 'bg-green-100 text-green-600' : 'bg-red-100 text-red-600'
                }`}>
                  {trade.side === 'buy' ? <TrendingUp className="h-4 w-4" /> : <TrendingDown className="h-4 w-4" />}
                </div>
                <div>
                  <div className="font-medium">{trade.symbol}</div>
                  <div className="text-sm text-muted-foreground">
                    {trade.amount} @ {formatCurrency(trade.price)}
                  </div>
                </div>
              </div>
              <div className="text-right">
                <div className="text-sm text-muted-foreground">{formatTimestamp(trade.timestamp)}</div>
                {trade.pnl !== undefined && (
                  <div className={`font-medium ${
                    trade.pnl >= 0 ? 'text-green-600' : 'text-red-600'
                  }`}>
                    {trade.pnl >= 0 ? '+' : ''}{formatCurrency(trade.pnl)}
                  </div>
                )}
              </div>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}
