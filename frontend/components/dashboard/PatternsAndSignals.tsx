import React from 'react';
import { DataTable } from '../ui/DataTable';
import { useLatestPatterns, useLatestSignals } from '../../lib/hooks';

export const PatternsAndSignals: React.FC = () => {
  const { data: patternsData, isLoading: patternsLoading, error: patternsError } = useLatestPatterns();
  const { data: signalsData, isLoading: signalsLoading, error: signalsError } = useLatestSignals();

  const patterns = patternsData || [];
  const signals = signalsData || [];

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      <DataTable
        title="Latest Patterns"
        data={patterns}
        type="patterns"
        className={patternsLoading ? 'animate-pulse' : ''}
      />
      
      <DataTable
        title="Latest Signals"
        data={signals}
        type="signals"
        className={signalsLoading ? 'animate-pulse' : ''}
      />
      
      {(patternsError || signalsError) && (
        <div className="col-span-full bg-red-50 border border-red-200 rounded-lg p-4">
          <p className="text-red-800">
            {patternsError && 'Failed to load patterns. '}
            {signalsError && 'Failed to load signals.'}
          </p>
        </div>
      )}
    </div>
  );
};
