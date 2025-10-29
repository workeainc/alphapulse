'use client';

import * as React from 'react';
import { Header } from '@/components/layout/Header';
import { StatusBar } from '@/components/layout/StatusBar';
import { SignalFeed } from '@/components/signals/SignalFeed';
import { SDEConsensusDashboard } from '@/components/sde/SDEConsensusDashboard';
import { MTFAnalysisPanel } from '@/components/mtf/MTFAnalysisPanel';
import { WorkflowMonitor } from '@/components/workflow/WorkflowMonitor';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/Card';
import { useSignals } from '@/lib/hooks/useSignals';
import { useWorkflow, useSystemStats } from '@/lib/hooks/useWorkflow';
import { useMarketStatus } from '@/lib/hooks/useMarket';
import { useWebSocket } from '@/lib/hooks/useWebSocket';
import { useSignalStore } from '@/store/signalStore';
import { useNotificationStore } from '@/store/notificationStore';
import type { WebSocketMessage, SignalUpdate } from '@/types';

export default function DashboardPage() {
  const { data: rawSignals = [], isLoading, refetch } = useSignals({ autoRefresh: false });
  
  // Deduplicate signals - only keep best signal per symbol
  const signals = React.useMemo(() => {
    const bySymbol = new Map<string, typeof rawSignals[0]>();
    rawSignals.forEach(signal => {
      const existing = bySymbol.get(signal.symbol);
      if (!existing || signal.confidence > existing.confidence) {
        bySymbol.set(signal.symbol, signal);
      }
    });
    return Array.from(bySymbol.values())
      .sort((a, b) => b.confidence - a.confidence)
      .slice(0, 5); // Show max 5 quality signals
  }, [rawSignals]);
  const { data: marketStatus } = useMarketStatus();
  const { lastMessage } = useWebSocket();
  const { addSignal } = useSignalStore();
  const { addNotification } = useNotificationStore();

  // Store SDE and MTF data
  const [sdeData, setSdeData] = React.useState<any>(null);
  const [mtfData, setMtfData] = React.useState<any>(null);
  const { selectedSignal } = useSignalStore();

  // Fetch workflow status via API (with polling)
  const { data: workflowData, isLoading: workflowLoading, error: workflowError } = useWorkflow();
  const { data: systemStats } = useSystemStats();

  // Store workflow status (from API + WebSocket updates)
  const [workflowStatus, setWorkflowStatus] = React.useState<any>(null);
  const [recentSteps, setRecentSteps] = React.useState<any[]>([]);
  const [lastWorkflowUpdate, setLastWorkflowUpdate] = React.useState<any>(null);

  // Update workflow status from API data
  React.useEffect(() => {
    if (workflowData) {
      setWorkflowStatus(workflowData.workflow_status);
      setRecentSteps(workflowData.recent_workflow_steps || []);
      setLastWorkflowUpdate({
        ...workflowData,
        timestamp: workflowData.timestamp
      });
    }
  }, [workflowData]);

  // Handle WebSocket messages
  React.useEffect(() => {
    if (!lastMessage) return;

    const message = lastMessage as WebSocketMessage;

    // Initial signals load
    if (message.type === 'initial_signals') {
      refetch();  // Refresh signal list
      return;
    }

    // New signal generated
    if (message.type === 'new_signal') {
      refetch();  // Refresh to get new signal
      
      const signalData = message.data as any;

      // Send notification ONLY if entry is imminent
      if (signalData.entry_proximity_status === 'imminent' && 
          signalData.confidence >= 0.85 &&
          signalData.agreeing_heads >= 5) {
        addNotification({
          type: 'signal',
          title: 'Entry Imminent - Action Required!',
          message: `${signalData.symbol} ${signalData.direction.toUpperCase()} @ ${(signalData.confidence * 100).toFixed(0)}% - Entry within 0.5%`,
          data: signalData,
        });
      }
    }

    // Signal removed (invalidated/filled/expired)
    if (message.type === 'signal_removed') {
      refetch();  // Refresh to remove signal
    }

    // Workflow status update
    if (message.type === 'workflow_status') {
      const data = message.data as any;
      setWorkflowStatus(data.workflow_status);
      setRecentSteps(data.recent_workflow_steps || []);
      setLastWorkflowUpdate(data);
    }

    // Workflow update (real-time events)
    if (message.type === 'workflow_update') {
      const update = message.data as any;
      // Update workflow status if needed
      if (update.type === 'candle_received') {
        // Trigger refresh of workflow status
        // The backend will send periodic updates
      }
    }
  }, [lastMessage, addNotification, refetch]);

  // Update SDE and MTF data when signal is selected OR when signals update
  React.useEffect(() => {
    let targetSignal = null;
    
    if (selectedSignal && signals.length > 0) {
      targetSignal = signals.find(s => s.symbol === selectedSignal.symbol);
    } else if (signals.length > 0) {
      targetSignal = signals[0]; // Auto-select first signal
    }
    
    if (targetSignal) {
      console.log('Selected signal:', targetSignal);
      console.log('Metadata:', targetSignal.metadata);
      
      if (targetSignal.metadata?.sde_consensus) {
        setSdeData(targetSignal.metadata.sde_consensus);
      }
      if (targetSignal.metadata?.mtf_analysis) {
        setMtfData(targetSignal.metadata.mtf_analysis);
      }
    }
  }, [selectedSignal, signals]);

  return (
    <div className="flex h-screen flex-col">
      <Header />

      <main className="flex-1 overflow-y-auto">
        <div className="container mx-auto px-6 py-6">
          <div className="grid grid-cols-12 gap-6">
            {/* Left Column: Chart + SDE + Workflow */}
            <div className="col-span-8 flex flex-col gap-6">
              {/* Main Chart */}
              <Card className="h-[400px]">
                <CardHeader>
                  <CardTitle>BTCUSDT / 1H</CardTitle>
                </CardHeader>
                <CardContent className="h-[calc(100%-4rem)]">
                  <div className="flex h-full items-center justify-center text-gray-400">
                    <div className="text-center">
                      <p className="text-lg font-semibold">TradingView Chart</p>
                      <p className="text-sm mt-2">Chart integration coming soon</p>
                    </div>
                  </div>
                </CardContent>
              </Card>

              {/* SDE Consensus Dashboard */}
              <div className="mb-8">
                <SDEConsensusDashboard consensus={sdeData} animated />
              </div>

              {/* Workflow Monitor */}
              <div className="mb-8">
                <WorkflowMonitor 
                  workflowStatus={workflowStatus}
                  recentSteps={recentSteps}
                  lastUpdate={lastWorkflowUpdate}
                  error={workflowError}
                  isLoading={workflowLoading}
                />
              </div>
            </div>

            {/* Right Column: Signals + MTF + Stats */}
            <div className="col-span-4 flex flex-col gap-6 pb-8">
              {/* Signal Feed */}
              <div className="h-[500px]">
                <SignalFeed signals={signals} maxItems={5} />
              </div>

              {/* MTF Analysis */}
              <MTFAnalysisPanel mtfSignal={mtfData} showBoostCalculation />

              {/* Quick Stats */}
              <Card>
                <CardHeader>
                  <CardTitle>Quick Stats</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <div className="text-2xl font-bold font-mono text-green-400">
                        {signals.length}
                      </div>
                      <div className="text-xs text-gray-400 mt-1">Quality Signals</div>
                    </div>
                    <div>
                      <div className="text-2xl font-bold font-mono text-blue-400">
                        {signals.length > 0 
                          ? Math.round(signals.reduce((sum, s) => sum + s.confidence, 0) / signals.length * 100)
                          : 0}%
                      </div>
                      <div className="text-xs text-gray-400 mt-1">Avg Confidence</div>
                    </div>
                    <div>
                      <div className="text-2xl font-bold font-mono text-yellow-400">
                        {marketStatus?.market_condition || 'N/A'}
                      </div>
                      <div className="text-xs text-gray-400 mt-1">Market</div>
                    </div>
                    <div>
                      <div className="text-2xl font-bold font-mono text-purple-400">
                        {signals.filter(s => s.metadata?.quality_score >= 0.75).length}
                      </div>
                      <div className="text-xs text-gray-400 mt-1">High Quality</div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>
          </div>
        </div>
      </main>

      <StatusBar />
    </div>
  );
}

