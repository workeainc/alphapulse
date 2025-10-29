'use client';

import * as React from 'react';
import { LineChart, Line, AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/Card';

interface PerformanceChartProps {
  data?: any[];
  type?: 'line' | 'area';
  metric?: 'win_rate' | 'profit' | 'accuracy';
}

export function PerformanceChart({
  data = [],
  type = 'area',
  metric = 'win_rate',
}: PerformanceChartProps) {
  // Sample data if none provided
  const sampleData = [
    { date: 'Day 1', win_rate: 72, profit: 1200, accuracy: 68 },
    { date: 'Day 2', win_rate: 75, profit: 1450, accuracy: 71 },
    { date: 'Day 3', win_rate: 78, profit: 1680, accuracy: 75 },
    { date: 'Day 4', win_rate: 76, profit: 1520, accuracy: 73 },
    { date: 'Day 5', win_rate: 80, profit: 1890, accuracy: 78 },
    { date: 'Day 6', win_rate: 82, profit: 2100, accuracy: 80 },
    { date: 'Day 7', win_rate: 78, profit: 1950, accuracy: 76 },
  ];

  const displayData = data.length > 0 ? data : sampleData;

  return (
    <Card>
      <CardHeader>
        <CardTitle>Performance Over Time</CardTitle>
      </CardHeader>
      <CardContent>
        <ResponsiveContainer width="100%" height={300}>
          {type === 'area' ? (
            <AreaChart data={displayData}>
              <defs>
                <linearGradient id="colorMetric" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#3B82F6" stopOpacity={0.8}/>
                  <stop offset="95%" stopColor="#3B82F6" stopOpacity={0}/>
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis dataKey="date" stroke="#9CA3AF" />
              <YAxis stroke="#9CA3AF" />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#1E2329',
                  border: '1px solid #374151',
                  borderRadius: '8px',
                }}
              />
              <Area
                type="monotone"
                dataKey={metric}
                stroke="#3B82F6"
                fillOpacity={1}
                fill="url(#colorMetric)"
              />
            </AreaChart>
          ) : (
            <LineChart data={displayData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis dataKey="date" stroke="#9CA3AF" />
              <YAxis stroke="#9CA3AF" />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#1E2329',
                  border: '1px solid #374151',
                  borderRadius: '8px',
                }}
              />
              <Legend />
              <Line type="monotone" dataKey={metric} stroke="#3B82F6" strokeWidth={2} />
            </LineChart>
          )}
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
}

