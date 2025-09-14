import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Layers, TrendingUp, TrendingDown, AlertTriangle, Activity } from 'lucide-react';

interface OrderBookData {
  symbol: string;
  timestamp: number;
  bids: Array<[number, number]>; // [price, size]
  asks: Array<[number, number]>; // [price, size]
  spread: number;
  spread_percentage: number;
  total_bid_volume: number;
  total_ask_volume: number;
  liquidity_imbalance: number;
  depth_pressure: number;
  order_flow_toxicity: number;
  liquidity_walls: Array<{
    price: number;
    size: number;
    side: 'bid' | 'ask';
    strength: number;
  }>;
  order_clusters: Array<{
    price_range: [number, number];
    total_volume: number;
    order_count: number;
    side: 'bid' | 'ask';
  }>;
}

interface OrderBookVisualizationProps {
  data: OrderBookData | null;
  maxDepth?: number;
  pricePrecision?: number;
  sizePrecision?: number;
}

export default function OrderBookVisualization({ 
  data, 
  maxDepth = 20, 
  pricePrecision = 2, 
  sizePrecision = 4 
}: OrderBookVisualizationProps) {
  const [selectedSide, setSelectedSide] = useState<'bid' | 'ask'>('bid');
  const [showLiquidityWalls, setShowLiquidityWalls] = useState(true);
  const [showOrderClusters, setShowOrderClusters] = useState(true);

  if (!data) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Layers className="h-5 w-5" />
            Order Book Analysis
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-center text-muted-foreground">Loading order book data...</div>
        </CardContent>
      </Card>
    );
  }

  const formatPrice = (price: number) => price.toFixed(pricePrecision);
  const formatSize = (size: number) => size.toFixed(sizePrecision);
  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2,
    }).format(value);
  };

  const getLiquidityWallColor = (strength: number) => {
    if (strength > 0.8) return 'bg-red-500';
    if (strength > 0.6) return 'bg-orange-500';
    if (strength > 0.4) return 'bg-yellow-500';
    return 'bg-green-500';
  };

  const getDepthPressureColor = (pressure: number) => {
    if (pressure > 0.8) return 'text-red-600';
    if (pressure > 0.6) return 'text-orange-600';
    if (pressure > 0.4) return 'text-yellow-600';
    return 'text-green-600';
  };

  const getToxicityColor = (toxicity: number) => {
    if (toxicity > 0.7) return 'text-red-600';
    if (toxicity > 0.5) return 'text-orange-600';
    if (toxicity > 0.3) return 'text-yellow-600';
    return 'text-green-600';
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Layers className="h-5 w-5" />
          Order Book Analysis - {data.symbol}
        </CardTitle>
        <div className="flex items-center gap-4 text-sm text-muted-foreground">
          <span>Spread: {formatPrice(data.spread)} ({data.spread_percentage.toFixed(4)}%)</span>
          <span>Imbalance: {(data.liquidity_imbalance * 100).toFixed(1)}%</span>
        </div>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* Market Depth Overview */}
        <div className="grid grid-cols-3 gap-4">
          <div className="text-center p-3 border rounded-lg">
            <div className="text-lg font-bold text-green-600">{formatCurrency(data.total_bid_volume)}</div>
            <div className="text-sm text-muted-foreground">Total Bid Volume</div>
          </div>
          
          <div className="text-center p-3 border rounded-lg">
            <div className="text-lg font-bold text-red-600">{formatCurrency(data.total_ask_volume)}</div>
            <div className="text-sm text-muted-foreground">Total Ask Volume</div>
          </div>
          
          <div className="text-center p-3 border rounded-lg">
            <div className={`text-lg font-bold ${getDepthPressureColor(data.depth_pressure)}`}>
              {(data.depth_pressure * 100).toFixed(1)}%
            </div>
            <div className="text-sm text-muted-foreground">Depth Pressure</div>
          </div>
        </div>

        {/* Order Flow Toxicity */}
        <div className="border-t pt-4">
          <h4 className="text-sm font-medium mb-3 flex items-center gap-2">
            <Activity className="h-4 w-4" />
            Order Flow Analysis
          </h4>
          <div className="flex items-center justify-between p-3 border rounded-lg">
            <span className="text-sm text-muted-foreground">Order Flow Toxicity</span>
            <div className="flex items-center gap-2">
              <div className="w-32 bg-gray-200 rounded-full h-2">
                <div 
                  className={`h-2 rounded-full ${getToxicityColor(data.order_flow_toxicity).replace('text-', 'bg-')}`}
                  style={{width: `${data.order_flow_toxicity * 100}%`}}
                ></div>
              </div>
              <span className={`font-semibold ${getToxicityColor(data.order_flow_toxicity)}`}>
                {(data.order_flow_toxicity * 100).toFixed(1)}%
              </span>
            </div>
          </div>
        </div>

        {/* Liquidity Walls */}
        {showLiquidityWalls && data.liquidity_walls.length > 0 && (
          <div className="border-t pt-4">
            <h4 className="text-sm font-medium mb-3 flex items-center gap-2">
              <AlertTriangle className="h-4 w-4" />
              Liquidity Walls
            </h4>
            <div className="space-y-2">
              {data.liquidity_walls.slice(0, 5).map((wall, index) => (
                <div key={index} className="flex items-center justify-between p-2 border rounded-lg">
                  <div className="flex items-center gap-2">
                    <div className={`w-3 h-3 rounded-full ${getLiquidityWallColor(wall.strength)}`}></div>
                    <span className="text-sm font-medium">{formatPrice(wall.price)}</span>
                    <span className={`text-xs px-2 py-1 rounded ${
                      wall.side === 'bid' ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'
                    }`}>
                      {wall.side.toUpperCase()}
                    </span>
                  </div>
                  <div className="text-right">
                    <div className="text-sm font-semibold">{formatSize(wall.size)}</div>
                    <div className="text-xs text-muted-foreground">Strength: {(wall.strength * 100).toFixed(0)}%</div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Order Clusters */}
        {showOrderClusters && data.order_clusters.length > 0 && (
          <div className="border-t pt-4">
            <h4 className="text-sm font-medium mb-3 flex items-center gap-2">
              <TrendingUp className="h-4 w-4" />
              Order Clusters
            </h4>
            <div className="space-y-2">
              {data.order_clusters.slice(0, 5).map((cluster, index) => (
                <div key={index} className="flex items-center justify-between p-2 border rounded-lg">
                  <div className="flex items-center gap-2">
                    <span className="text-sm font-medium">
                      {formatPrice(cluster.price_range[0])} - {formatPrice(cluster.price_range[1])}
                    </span>
                    <span className={`text-xs px-2 py-1 rounded ${
                      cluster.side === 'bid' ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'
                    }`}>
                      {cluster.side.toUpperCase()}
                    </span>
                  </div>
                  <div className="text-right">
                    <div className="text-sm font-semibold">{formatSize(cluster.total_volume)}</div>
                    <div className="text-xs text-muted-foreground">{cluster.order_count} orders</div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Order Book Depth Visualization */}
        <div className="border-t pt-4">
          <h4 className="text-sm font-medium mb-3">Order Book Depth</h4>
          <div className="grid grid-cols-2 gap-4">
            {/* Bids */}
            <div>
              <h5 className="text-sm font-medium text-green-600 mb-2">Bids</h5>
              <div className="space-y-1">
                {data.bids.slice(0, maxDepth).map((bid, index) => (
                  <div key={index} className="flex items-center justify-between text-sm">
                    <span className="text-green-600">{formatPrice(bid[0])}</span>
                    <span>{formatSize(bid[1])}</span>
                    <div className="w-16 bg-gray-200 rounded h-1">
                      <div 
                        className="bg-green-500 h-1 rounded"
                        style={{width: `${(bid[1] / Math.max(...data.bids.map(b => b[1]))) * 100}%`}}
                      ></div>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Asks */}
            <div>
              <h5 className="text-sm font-medium text-red-600 mb-2">Asks</h5>
              <div className="space-y-1">
                {data.asks.slice(0, maxDepth).map((ask, index) => (
                  <div key={index} className="flex items-center justify-between text-sm">
                    <div className="w-16 bg-gray-200 rounded h-1">
                      <div 
                        className="bg-red-500 h-1 rounded"
                        style={{width: `${(ask[1] / Math.max(...data.asks.map(a => a[1]))) * 100}%`}}
                      ></div>
                    </div>
                    <span>{formatSize(ask[1])}</span>
                    <span className="text-red-600">{formatPrice(ask[0])}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* Controls */}
        <div className="border-t pt-4">
          <div className="flex items-center gap-4">
            <label className="flex items-center gap-2 text-sm">
              <input
                type="checkbox"
                checked={showLiquidityWalls}
                onChange={(e) => setShowLiquidityWalls(e.target.checked)}
                className="rounded"
              />
              Show Liquidity Walls
            </label>
            <label className="flex items-center gap-2 text-sm">
              <input
                type="checkbox"
                checked={showOrderClusters}
                onChange={(e) => setShowOrderClusters(e.target.checked)}
                className="rounded"
              />
              Show Order Clusters
            </label>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
