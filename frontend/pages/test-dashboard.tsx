import React, { useState, useEffect } from 'react';
import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000';

interface Pattern {
  symbol: string;
  pattern_type: string;
  confidence: number;
  strength: string;
  timestamp: string;
}

interface Signal {
  symbol: string;
  direction: string;
  confidence: number;
  pattern_type: string;
  timestamp: string;
}

export default function TestDashboard() {
  const [patterns, setPatterns] = useState<Pattern[]>([]);
  const [signals, setSignals] = useState<Signal[]>([]);
  const [health, setHealth] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchData = async () => {
    try {
      setLoading(true);
      setError(null);

      // Fetch health status
      const healthResponse = await axios.get(`${API_BASE_URL}/health`);
      setHealth(healthResponse.data);

      // Fetch patterns
      const patternsResponse = await axios.get(`${API_BASE_URL}/api/patterns/latest`);
      setPatterns(patternsResponse.data.patterns || []);

      // Fetch signals
      const signalsResponse = await axios.get(`${API_BASE_URL}/api/signals/latest`);
      setSignals(signalsResponse.data.signals || []);

    } catch (err) {
      setError('Failed to fetch data from backend');
      console.error('Error fetching data:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 10000); // Refresh every 10 seconds
    return () => clearInterval(interval);
  }, []);

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto"></div>
          <p className="mt-4 text-gray-600">Loading AlphaPulse Dashboard...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50 p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="bg-white rounded-lg shadow-sm p-6 mb-6">
          <h1 className="text-3xl font-bold text-gray-900 mb-2">AlphaPulse AI Trading System</h1>
          <p className="text-gray-600">Phase 3 - Real-time AI Trading Dashboard</p>
          
          {health && (
            <div className="mt-4 p-4 bg-green-50 rounded-lg">
              <h3 className="font-semibold text-green-800">System Status</h3>
              <p className="text-green-700">Service: {health.service}</p>
              <p className="text-green-700">Database: {health.database}</p>
              <p className="text-green-700">Patterns Detected: {health.patterns_detected}</p>
              <p className="text-green-700">Signals Generated: {health.signals_generated}</p>
            </div>
          )}
        </div>

        {error && (
          <div className="bg-red-50 border border-red-200 rounded-lg p-4 mb-6">
            <p className="text-red-800">{error}</p>
          </div>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Patterns Section */}
          <div className="bg-white rounded-lg shadow-sm p-6">
            <h2 className="text-xl font-semibold text-gray-900 mb-4">Latest Patterns</h2>
            {patterns.length > 0 ? (
              <div className="space-y-3">
                {patterns.slice(0, 5).map((pattern, index) => (
                  <div key={index} className="border border-gray-200 rounded-lg p-4">
                    <div className="flex justify-between items-start">
                      <div>
                        <h3 className="font-medium text-gray-900">{pattern.symbol}</h3>
                        <p className="text-sm text-gray-600">{pattern.pattern_type}</p>
                      </div>
                      <div className="text-right">
                        <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                          pattern.confidence > 0.7 ? 'bg-green-100 text-green-800' :
                          pattern.confidence > 0.5 ? 'bg-yellow-100 text-yellow-800' :
                          'bg-red-100 text-red-800'
                        }`}>
                          {(pattern.confidence * 100).toFixed(1)}%
                        </span>
                        <p className="text-xs text-gray-500 mt-1">{pattern.strength}</p>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <p className="text-gray-500">No patterns detected yet...</p>
            )}
          </div>

          {/* Signals Section */}
          <div className="bg-white rounded-lg shadow-sm p-6">
            <h2 className="text-xl font-semibold text-gray-900 mb-4">Latest Signals</h2>
            {signals.length > 0 ? (
              <div className="space-y-3">
                {signals.slice(0, 5).map((signal, index) => (
                  <div key={index} className="border border-gray-200 rounded-lg p-4">
                    <div className="flex justify-between items-start">
                      <div>
                        <h3 className="font-medium text-gray-900">{signal.symbol}</h3>
                        <p className="text-sm text-gray-600">{signal.pattern_type}</p>
                      </div>
                      <div className="text-right">
                        <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                          signal.direction === 'long' ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'
                        }`}>
                          {signal.direction.toUpperCase()}
                        </span>
                        <p className="text-xs text-gray-500 mt-1">
                          {(signal.confidence * 100).toFixed(1)}% confidence
                        </p>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <p className="text-gray-500">No signals generated yet...</p>
            )}
          </div>
        </div>

        {/* Refresh Button */}
        <div className="mt-6 text-center">
          <button
            onClick={fetchData}
            className="bg-blue-600 hover:bg-blue-700 text-white font-medium py-2 px-4 rounded-lg transition-colors"
          >
            Refresh Data
          </button>
        </div>
      </div>
    </div>
  );
}
