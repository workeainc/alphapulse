import React from 'react';
import { StatusCard } from '../ui/StatusCard';
import { useHealthStatus, useAIPerformance } from '../../lib/hooks';
import { Database, Brain, TrendingUp, Activity } from 'lucide-react';

export const SystemStatus: React.FC = () => {
  const { data: health, isLoading: healthLoading, error: healthError } = useHealthStatus();
  const { data: aiPerformance, isLoading: aiLoading, error: aiError } = useAIPerformance();

  if (healthLoading || aiLoading) {
    return (
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {[...Array(4)].map((_, i) => (
          <div key={i} className="bg-white rounded-lg border p-4 animate-pulse">
            <div className="h-4 bg-gray-200 rounded w-3/4 mb-2"></div>
            <div className="h-8 bg-gray-200 rounded w-1/2"></div>
          </div>
        ))}
      </div>
    );
  }

  if (healthError || aiError) {
    return (
      <div className="bg-red-50 border border-red-200 rounded-lg p-4">
        <p className="text-red-800">Failed to load system status</p>
      </div>
    );
  }

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
      <StatusCard
        title="Service Status"
        value={health?.service || 'Unknown'}
        status={health?.service === 'healthy' ? 'success' : 'error'}
        icon={<Activity className="w-5 h-5" />}
      />
      
      <StatusCard
        title="Database Status"
        value={health?.database || 'Unknown'}
        status={health?.database === 'connected' ? 'success' : 'error'}
        icon={<Database className="w-5 h-5" />}
      />
      
      <StatusCard
        title="Patterns Detected"
        value={health?.patterns_detected || 0}
        status="info"
        subtitle="Total patterns"
        icon={<TrendingUp className="w-5 h-5" />}
      />
      
      <StatusCard
        title="AI Accuracy"
        value={aiPerformance ? `${(aiPerformance.accuracy * 100).toFixed(1)}%` : 'N/A'}
        status={
          aiPerformance?.accuracy && aiPerformance.accuracy > 0.7 ? 'success' :
          aiPerformance?.accuracy && aiPerformance.accuracy > 0.5 ? 'warning' : 'error'
        }
        subtitle={`${aiPerformance?.profitable_signals || 0}/${aiPerformance?.total_signals || 0} profitable`}
        icon={<Brain className="w-5 h-5" />}
      />
    </div>
  );
};
