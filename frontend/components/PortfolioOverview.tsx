import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { TrendingUp, TrendingDown, DollarSign, BarChart3, AlertTriangle, Gauge, Activity, Layers } from 'lucide-react';

interface PortfolioOverviewProps {
  data: {
    total_balance: number;
    available_balance: number;
    total_pnl: number;
    total_pnl_percentage: number;
    daily_pnl: number;
    daily_pnl_percentage: number;
    open_positions: number;
    consecutive_losses: number;
    daily_loss_limit: number;
    // Enhanced leverage and risk metrics
    total_leverage: number;
    average_leverage: number;
    max_leverage: number;
    margin_utilization: number;
    liquidation_risk_score: number;
    portfolio_var: number;
    correlation_risk: number;
    // Advanced analytics
    liquidity_score: number;
    market_depth_analysis: {
      bid_liquidity: number;
      ask_liquidity: number;
      liquidity_imbalance: number;
    };
    order_book_analysis: {
      spread: number;
      depth_pressure: number;
      order_flow_toxicity: number;
    };
  } | null;
}

export default function PortfolioOverview({ data }: PortfolioOverviewProps) {
  if (!data) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Portfolio Overview</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-center text-muted-foreground">Loading...</div>
        </CardContent>
      </Card>
    );
  }

  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2,
    }).format(value);
  };

  const formatPercentage = (value: number) => {
    return `${value >= 0 ? '+' : ''}${value.toFixed(2)}%`;
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <BarChart3 className="h-5 w-5" />
          Portfolio Overview
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* Basic Portfolio Metrics */}
        <div className="grid grid-cols-2 gap-4">
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-sm text-muted-foreground">Total Balance</span>
              <span className="font-semibold">{formatCurrency(data.total_balance)}</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-sm text-muted-foreground">Available Balance</span>
              <span className="font-semibold">{formatCurrency(data.available_balance)}</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-sm text-muted-foreground">Open Positions</span>
              <span className="font-semibold">{data.open_positions}</span>
            </div>
          </div>
          
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-sm text-muted-foreground">Total P&L</span>
              <span className={`font-semibold flex items-center gap-1 ${
                data.total_pnl >= 0 ? 'text-green-600' : 'text-red-600'
              }`}>
                {data.total_pnl >= 0 ? <TrendingUp className="h-4 w-4" /> : <TrendingDown className="h-4 w-4" />}
                {formatCurrency(data.total_pnl)} ({formatPercentage(data.total_pnl_percentage)})
              </span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-sm text-muted-foreground">Daily P&L</span>
              <span className={`font-semibold flex items-center gap-1 ${
                data.daily_pnl >= 0 ? 'text-green-600' : 'text-red-600'
              }`}>
                {data.daily_pnl >= 0 ? <TrendingUp className="h-4 w-4" /> : <TrendingDown className="h-4 w-4" />}
                {formatCurrency(data.daily_pnl)} ({formatPercentage(data.daily_pnl_percentage)})
              </span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-sm text-muted-foreground">Daily Loss Limit</span>
              <span className="font-semibold text-orange-600">{formatCurrency(data.daily_loss_limit)}</span>
            </div>
          </div>
        </div>

        {/* Enhanced Leverage and Risk Metrics */}
        <div className="border-t pt-4">
          <h4 className="text-sm font-medium mb-3 flex items-center gap-2">
            <Gauge className="h-4 w-4" />
            Leverage & Risk Analytics
          </h4>
          <div className="grid grid-cols-3 gap-4">
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-xs text-muted-foreground">Total Leverage</span>
                <span className="font-semibold text-blue-600">{data.total_leverage}x</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-xs text-muted-foreground">Avg Leverage</span>
                <span className="font-semibold">{data.average_leverage}x</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-xs text-muted-foreground">Max Leverage</span>
                <span className="font-semibold text-red-600">{data.max_leverage}x</span>
              </div>
            </div>
            
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-xs text-muted-foreground">Margin Util.</span>
                <span className={`font-semibold ${data.margin_utilization > 80 ? 'text-red-600' : 'text-green-600'}`}>
                  {data.margin_utilization.toFixed(1)}%
                </span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-xs text-muted-foreground">Liquidation Risk</span>
                <span className={`font-semibold flex items-center gap-1 ${
                  data.liquidation_risk_score > 70 ? 'text-red-600' : 
                  data.liquidation_risk_score > 40 ? 'text-orange-600' : 'text-green-600'
                }`}>
                  {data.liquidation_risk_score > 70 && <AlertTriangle className="h-3 w-3" />}
                  {data.liquidation_risk_score.toFixed(0)}
                </span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-xs text-muted-foreground">Portfolio VaR</span>
                <span className="font-semibold text-purple-600">{formatCurrency(data.portfolio_var)}</span>
              </div>
            </div>

            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-xs text-muted-foreground">Correlation Risk</span>
                <span className={`font-semibold ${data.correlation_risk > 0.7 ? 'text-red-600' : 'text-green-600'}`}>
                  {(data.correlation_risk * 100).toFixed(1)}%
                </span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-xs text-muted-foreground">Liquidity Score</span>
                <span className={`font-semibold ${data.liquidity_score > 80 ? 'text-green-600' : 
                  data.liquidity_score > 50 ? 'text-orange-600' : 'text-red-600'}`}>
                  {data.liquidity_score.toFixed(0)}
                </span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-xs text-muted-foreground">Spread</span>
                <span className="font-semibold text-gray-600">{data.order_book_analysis.spread.toFixed(4)}%</span>
              </div>
            </div>
          </div>
        </div>

        {/* Market Depth Analysis */}
        <div className="border-t pt-4">
          <h4 className="text-sm font-medium mb-3 flex items-center gap-2">
            <Layers className="h-4 w-4" />
            Market Depth Analysis
          </h4>
          <div className="grid grid-cols-3 gap-4">
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-xs text-muted-foreground">Bid Liquidity</span>
                <span className="font-semibold text-green-600">{formatCurrency(data.market_depth_analysis.bid_liquidity)}</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-xs text-muted-foreground">Ask Liquidity</span>
                <span className="font-semibold text-red-600">{formatCurrency(data.market_depth_analysis.ask_liquidity)}</span>
              </div>
            </div>
            
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-xs text-muted-foreground">Liquidity Imbalance</span>
                <span className={`font-semibold ${
                  Math.abs(data.market_depth_analysis.liquidity_imbalance) > 0.3 ? 'text-orange-600' : 'text-green-600'
                }`}>
                  {(data.market_depth_analysis.liquidity_imbalance * 100).toFixed(1)}%
                </span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-xs text-muted-foreground">Depth Pressure</span>
                <span className={`font-semibold ${
                  data.order_book_analysis.depth_pressure > 0.7 ? 'text-red-600' : 'text-green-600'
                }`}>
                  {(data.order_book_analysis.depth_pressure * 100).toFixed(1)}%
                </span>
              </div>
            </div>

            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-xs text-muted-foreground">Order Flow Toxicity</span>
                <span className={`font-semibold ${
                  data.order_book_analysis.order_flow_toxicity > 0.6 ? 'text-red-600' : 'text-green-600'
                }`}>
                  {(data.order_book_analysis.order_flow_toxicity * 100).toFixed(1)}%
                </span>
              </div>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
