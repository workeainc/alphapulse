import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { AlertTriangle, TrendingDown, TrendingUp, Activity, Clock, DollarSign } from 'lucide-react';

interface LiquidationEvent {
  id: string;
  symbol: string;
  side: 'long' | 'short';
  price: number;
  size: number;
  value: number;
  timestamp: number;
  impact_score: number;
  cluster_id?: string;
  exchange: string;
  liquidation_type: 'isolated' | 'cross' | 'partial';
  distance_from_price: number; // Distance from current market price
}

interface LiquidationEventsProps {
  events: LiquidationEvent[];
  currentPrice?: number;
  maxEvents?: number;
  showImpactAnalysis?: boolean;
}

export default function LiquidationEvents({ 
  events, 
  currentPrice, 
  maxEvents = 10, 
  showImpactAnalysis = true 
}: LiquidationEventsProps) {
  const [filterSide, setFilterSide] = useState<'all' | 'long' | 'short'>('all');
  const [sortBy, setSortBy] = useState<'timestamp' | 'value' | 'impact'>('timestamp');
  const [showHighImpact, setShowHighImpact] = useState(true);

  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 0,
    }).format(value);
  };

  const formatTime = (timestamp: number) => {
    const date = new Date(timestamp);
    return date.toLocaleTimeString();
  };

  const getImpactColor = (impact: number) => {
    if (impact > 0.8) return 'text-red-600';
    if (impact > 0.6) return 'text-orange-600';
    if (impact > 0.4) return 'text-yellow-600';
    return 'text-green-600';
  };

  const getImpactIcon = (impact: number) => {
    if (impact > 0.8) return <AlertTriangle className="h-4 w-4 text-red-600" />;
    if (impact > 0.6) return <TrendingDown className="h-4 w-4 text-orange-600" />;
    if (impact > 0.4) return <Activity className="h-4 w-4 text-yellow-600" />;
    return <TrendingUp className="h-4 w-4 text-green-600" />;
  };

  const getSideColor = (side: string) => {
    return side === 'long' ? 'text-red-600' : 'text-green-600';
  };

  const getSideIcon = (side: string) => {
    return side === 'long' ? <TrendingDown className="h-4 w-4" /> : <TrendingUp className="h-4 w-4" />;
  };

  const getTypeColor = (type: string) => {
    switch (type) {
      case 'isolated': return 'bg-blue-100 text-blue-800';
      case 'cross': return 'bg-red-100 text-red-800';
      case 'partial': return 'bg-yellow-100 text-yellow-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  // Filter and sort events
  const filteredEvents = events
    .filter(event => {
      if (filterSide !== 'all' && event.side !== filterSide) return false;
      if (showHighImpact && event.impact_score < 0.5) return false;
      return true;
    })
    .sort((a, b) => {
      switch (sortBy) {
        case 'value':
          return b.value - a.value;
        case 'impact':
          return b.impact_score - a.impact_score;
        case 'timestamp':
        default:
          return b.timestamp - a.timestamp;
      }
    })
    .slice(0, maxEvents);

  // Calculate summary statistics
  const totalValue = filteredEvents.reduce((sum, event) => sum + event.value, 0);
  const averageImpact = filteredEvents.reduce((sum, event) => sum + event.impact_score, 0) / filteredEvents.length || 0;
  const longLiquidations = filteredEvents.filter(event => event.side === 'long').length;
  const shortLiquidations = filteredEvents.filter(event => event.side === 'short').length;

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <AlertTriangle className="h-5 w-5 text-red-600" />
          Liquidation Events
        </CardTitle>
        <div className="flex items-center gap-4 text-sm text-muted-foreground">
          <span>Total Value: {formatCurrency(totalValue)}</span>
          <span>Avg Impact: {(averageImpact * 100).toFixed(1)}%</span>
          <span>Long: {longLiquidations} | Short: {shortLiquidations}</span>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Controls */}
        <div className="flex items-center gap-4 text-sm">
          <div className="flex items-center gap-2">
            <span>Filter:</span>
            <select 
              value={filterSide} 
              onChange={(e) => setFilterSide(e.target.value as any)}
              className="border rounded px-2 py-1"
            >
              <option value="all">All</option>
              <option value="long">Long</option>
              <option value="short">Short</option>
            </select>
          </div>
          
          <div className="flex items-center gap-2">
            <span>Sort by:</span>
            <select 
              value={sortBy} 
              onChange={(e) => setSortBy(e.target.value as any)}
              className="border rounded px-2 py-1"
            >
              <option value="timestamp">Time</option>
              <option value="value">Value</option>
              <option value="impact">Impact</option>
            </select>
          </div>
          
          <label className="flex items-center gap-2">
            <input
              type="checkbox"
              checked={showHighImpact}
              onChange={(e) => setShowHighImpact(e.target.checked)}
              className="rounded"
            />
            High Impact Only
          </label>
        </div>

        {/* Impact Analysis */}
        {showImpactAnalysis && (
          <div className="border-t pt-4">
            <h4 className="text-sm font-medium mb-3 flex items-center gap-2">
              <Activity className="h-4 w-4" />
              Impact Analysis
            </h4>
            <div className="grid grid-cols-3 gap-4">
              <div className="text-center p-3 border rounded-lg">
                <div className="text-lg font-bold text-red-600">{formatCurrency(totalValue)}</div>
                <div className="text-sm text-muted-foreground">Total Liquidated</div>
              </div>
              
              <div className="text-center p-3 border rounded-lg">
                <div className={`text-lg font-bold ${getImpactColor(averageImpact)}`}>
                  {(averageImpact * 100).toFixed(1)}%
                </div>
                <div className="text-sm text-muted-foreground">Average Impact</div>
              </div>
              
              <div className="text-center p-3 border rounded-lg">
                <div className="text-lg font-bold text-blue-600">{filteredEvents.length}</div>
                <div className="text-sm text-muted-foreground">Active Events</div>
              </div>
            </div>
          </div>
        )}

        {/* Liquidation Events List */}
        <div className="space-y-2">
          {filteredEvents.length === 0 ? (
            <div className="text-center text-muted-foreground py-8">
              No liquidation events found
            </div>
          ) : (
            filteredEvents.map((event) => (
              <div key={event.id} className="border rounded-lg p-3 hover:bg-gray-50">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    {getImpactIcon(event.impact_score)}
                    <div>
                      <div className="flex items-center gap-2">
                        <span className="font-medium">{event.symbol}</span>
                        <span className={`text-xs px-2 py-1 rounded ${getTypeColor(event.liquidation_type)}`}>
                          {event.liquidation_type.toUpperCase()}
                        </span>
                      </div>
                      <div className="flex items-center gap-2 text-sm text-muted-foreground">
                        <span>{event.exchange}</span>
                        <span>•</span>
                        <span>{formatTime(event.timestamp)}</span>
                      </div>
                    </div>
                  </div>
                  
                  <div className="text-right">
                    <div className="flex items-center gap-2">
                      <span className={`font-semibold ${getSideColor(event.side)}`}>
                        {event.side.toUpperCase()}
                      </span>
                      {getSideIcon(event.side)}
                    </div>
                    <div className="text-sm font-semibold">{formatCurrency(event.value)}</div>
                    <div className="text-xs text-muted-foreground">
                      {formatCurrency(event.price)} @ {event.size.toFixed(4)}
                    </div>
                  </div>
                </div>
                
                <div className="mt-2 flex items-center justify-between text-xs text-muted-foreground">
                  <div className="flex items-center gap-4">
                    <span>Impact: <span className={getImpactColor(event.impact_score)}>{(event.impact_score * 100).toFixed(1)}%</span></span>
                    {currentPrice && (
                      <span>Distance: {Math.abs(event.price - currentPrice).toFixed(2)}</span>
                    )}
                  </div>
                  {event.cluster_id && (
                    <span>Cluster: {event.cluster_id}</span>
                  )}
                </div>
              </div>
            ))
          )}
        </div>

        {/* Real-time Status */}
        <div className="border-t pt-4">
          <div className="flex items-center justify-between text-sm">
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
              <span>Real-time monitoring active</span>
            </div>
            <span className="text-muted-foreground">
              Last update: {new Date().toLocaleTimeString()}
            </span>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
