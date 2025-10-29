'use client';

import * as React from 'react';
import { Header } from '@/components/layout/Header';
import { StatusBar } from '@/components/layout/StatusBar';
import { PerformanceChart } from '@/components/analytics/PerformanceChart';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/Card';
import { Badge } from '@/components/ui/Badge';
import { TrendingUp, TrendingDown, Target, Award } from 'lucide-react';
import { useSignalPerformance } from '@/lib/hooks/useSignals';

export default function AnalyticsPage() {
  const { data: performance } = useSignalPerformance();

  return (
    <div className="flex h-screen flex-col">
      <Header />

      <main className="flex-1 overflow-auto">
        <div className="container mx-auto px-6 py-6">
          <div className="mb-6">
            <h1 className="text-3xl font-bold text-white">Analytics & Performance</h1>
            <p className="text-gray-400 mt-2">
              Track signal performance, win rates, and system metrics
            </p>
          </div>

          {/* Stats Grid */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-6">
            <Card>
              <CardContent className="p-6">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm text-gray-400">Win Rate</p>
                    <p className="text-3xl font-bold text-green-400 mt-2">78.5%</p>
                    <Badge variant="success" className="mt-2">
                      <TrendingUp className="h-3 w-3 mr-1" />
                      +2.3%
                    </Badge>
                  </div>
                  <Award className="h-12 w-12 text-green-400 opacity-20" />
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardContent className="p-6">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm text-gray-400">Total Signals</p>
                    <p className="text-3xl font-bold text-blue-400 mt-2">1,247</p>
                    <Badge variant="info" className="mt-2">
                      This Month
                    </Badge>
                  </div>
                  <Target className="h-12 w-12 text-blue-400 opacity-20" />
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardContent className="p-6">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm text-gray-400">Avg Return</p>
                    <p className="text-3xl font-bold text-yellow-400 mt-2">4.2%</p>
                    <Badge variant="warning" className="mt-2">
                      Per Signal
                    </Badge>
                  </div>
                  <TrendingUp className="h-12 w-12 text-yellow-400 opacity-20" />
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardContent className="p-6">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm text-gray-400">Profit Factor</p>
                    <p className="text-3xl font-bold text-purple-400 mt-2">2.3</p>
                    <Badge variant="info" className="mt-2">
                      Excellent
                    </Badge>
                  </div>
                  <Award className="h-12 w-12 text-purple-400 opacity-20" />
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Charts Grid */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
            <PerformanceChart metric="win_rate" type="area" />
            <PerformanceChart metric="profit" type="line" />
          </div>

          {/* Recent Signals Table */}
          <Card>
            <CardHeader>
              <CardTitle>Recent Signal Performance</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead>
                    <tr className="border-b border-gray-700">
                      <th className="px-4 py-3 text-left text-sm font-semibold text-gray-400">Symbol</th>
                      <th className="px-4 py-3 text-left text-sm font-semibold text-gray-400">Direction</th>
                      <th className="px-4 py-3 text-left text-sm font-semibold text-gray-400">Confidence</th>
                      <th className="px-4 py-3 text-left text-sm font-semibold text-gray-400">Entry</th>
                      <th className="px-4 py-3 text-left text-sm font-semibold text-gray-400">Exit</th>
                      <th className="px-4 py-3 text-left text-sm font-semibold text-gray-400">P&L</th>
                      <th className="px-4 py-3 text-left text-sm font-semibold text-gray-400">Status</th>
                    </tr>
                  </thead>
                  <tbody>
                    {[
                      { symbol: 'BTCUSDT', direction: 'LONG', confidence: 95, entry: 42000, exit: 43500, pl: 3.57, status: 'win' },
                      { symbol: 'ETHUSDT', direction: 'LONG', confidence: 88, entry: 2800, exit: 2950, pl: 5.36, status: 'win' },
                      { symbol: 'BNBUSDT', direction: 'SHORT', confidence: 82, entry: 580, exit: 565, pl: 2.59, status: 'win' },
                      { symbol: 'SOLUSDT', direction: 'LONG', confidence: 76, entry: 145, exit: 140, pl: -3.45, status: 'loss' },
                      { symbol: 'ADAUSDT', direction: 'LONG', confidence: 91, entry: 0.65, exit: 0.70, pl: 7.69, status: 'win' },
                    ].map((signal, i) => (
                      <tr key={i} className="border-b border-gray-800 hover:bg-gray-800/50">
                        <td className="px-4 py-3 font-mono font-semibold text-white">{signal.symbol}</td>
                        <td className="px-4 py-3">
                          <Badge variant={signal.direction === 'LONG' ? 'success' : 'danger'}>
                            {signal.direction}
                          </Badge>
                        </td>
                        <td className="px-4 py-3 font-mono text-blue-400">{signal.confidence}%</td>
                        <td className="px-4 py-3 font-mono text-gray-300">${signal.entry}</td>
                        <td className="px-4 py-3 font-mono text-gray-300">${signal.exit}</td>
                        <td className={`px-4 py-3 font-mono font-semibold ${signal.pl > 0 ? 'text-green-400' : 'text-red-400'}`}>
                          {signal.pl > 0 ? '+' : ''}{signal.pl}%
                        </td>
                        <td className="px-4 py-3">
                          <Badge variant={signal.status === 'win' ? 'success' : 'danger'}>
                            {signal.status.toUpperCase()}
                          </Badge>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </CardContent>
          </Card>
        </div>
      </main>

      <StatusBar />
    </div>
  );
}

