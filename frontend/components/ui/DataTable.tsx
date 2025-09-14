import React from 'react';
import { format } from 'date-fns';

interface DataTableProps {
  title: string;
  data: Array<{
    id?: string | number;
    symbol: string;
    pattern_type?: string;
    direction?: string;
    confidence: number;
    strength?: string;
    timestamp: string;
    [key: string]: any;
  }>;
  type: 'patterns' | 'signals';
  className?: string;
}

export const DataTable: React.FC<DataTableProps> = ({
  title,
  data,
  type,
  className = '',
}) => {
  // Ensure data is always an array
  const safeData = Array.isArray(data) ? data : [];
  const getConfidenceColor = (confidence: number) => {
    if (confidence > 0.7) return 'bg-green-100 text-green-800';
    if (confidence > 0.5) return 'bg-yellow-100 text-yellow-800';
    return 'bg-red-100 text-red-800';
  };

  const getDirectionColor = (direction: string) => {
    return direction === 'long' ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800';
  };

  return (
    <div className={`bg-white rounded-lg shadow-sm p-6 ${className}`}>
      <h2 className="text-xl font-semibold text-gray-900 mb-4">{title}</h2>
      
      {safeData.length === 0 ? (
        <div className="text-center py-8">
          <p className="text-gray-500">No {type} available yet...</p>
        </div>
      ) : (
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Symbol
                </th>
                {type === 'patterns' && (
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Pattern Type
                  </th>
                )}
                {type === 'signals' && (
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Direction
                  </th>
                )}
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Confidence
                </th>
                {type === 'patterns' && (
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Strength
                  </th>
                )}
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Time
                </th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {safeData.slice(0, 10).map((item, index) => (
                <tr key={item.id || index} className="hover:bg-gray-50">
                  <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                    {item.symbol}
                  </td>
                  {type === 'patterns' && (
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      {item.pattern_type}
                    </td>
                  )}
                  {type === 'signals' && (
                    <td className="px-6 py-4 whitespace-nowrap">
                      <span className={`px-2 py-1 rounded-full text-xs font-medium ${getDirectionColor(item.direction || '')}`}>
                        {item.direction?.toUpperCase()}
                      </span>
                    </td>
                  )}
                  <td className="px-6 py-4 whitespace-nowrap">
                    <span className={`px-2 py-1 rounded-full text-xs font-medium ${getConfidenceColor(item.confidence)}`}>
                      {(item.confidence * 100).toFixed(1)}%
                    </span>
                  </td>
                  {type === 'patterns' && (
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      {item.strength}
                    </td>
                  )}
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                    {format(new Date(item.timestamp), 'HH:mm:ss')}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
};
