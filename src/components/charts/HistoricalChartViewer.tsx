'use client';

import { useState, useEffect } from 'react';
import { CandlestickChart } from './CandlestickChart';

interface Candle {
  timestamp: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

interface AssetOption {
  category: string;
  asset: string;
  candleCount: number;
}

interface HistoricalChartViewerProps {
  category: string;
  defaultAssets?: string[];
}

const TIMEFRAMES = [
  { label: '30m', value: '30m' },
  { label: '1H', value: '1h' },
  { label: '4H', value: '4h' },
  { label: '1D', value: '1d' },
  { label: '1W', value: '1w' },
];

// Helper to format date for input
function formatDateForInput(date: Date): string {
  return date.toISOString().split('T')[0];
}

export function HistoricalChartViewer({ category }: HistoricalChartViewerProps) {
  const [assets, setAssets] = useState<AssetOption[]>([]);
  const [selectedAsset, setSelectedAsset] = useState<string>('');
  const [selectedTimeframe, setSelectedTimeframe] = useState<string>('1h');
  const [candles, setCandles] = useState<Candle[]>([]);
  const [summary, setSummary] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Custom date/time range state
  const [showCustomRange, setShowCustomRange] = useState(false);
  const [startDate, setStartDate] = useState(formatDateForInput(new Date(Date.now() - 24 * 60 * 60 * 1000)));
  const [startTime, setStartTime] = useState('00:00');
  const [endDate, setEndDate] = useState(formatDateForInput(new Date()));
  const [endTime, setEndTime] = useState('23:59');

  // Fetch available assets on mount
  useEffect(() => {
    async function fetchAssets() {
      try {
        const res = await fetch('/api/historical', { method: 'POST' });
        const data = await res.json();

        if (data.assets) {
          // Filter assets by category
          const filtered = data.assets.filter((a: AssetOption) => a.category === category);
          setAssets(filtered);

          // Auto-select first asset if available
          if (filtered.length > 0 && !selectedAsset) {
            setSelectedAsset(filtered[0].asset);
          }
        }
      } catch (err) {
        console.error('Failed to load assets:', err);
      }
    }

    fetchAssets();
  }, [category]);

  // Fetch candles when asset or timeframe changes
  useEffect(() => {
    if (!selectedAsset) return;
    if (selectedTimeframe === 'custom') return; // Custom fetches on button click

    async function fetchCandles() {
      setLoading(true);
      setError(null);

      try {
        const res = await fetch(
          `/api/historical?category=${encodeURIComponent(category)}&asset=${encodeURIComponent(selectedAsset)}&timeframe=${selectedTimeframe}`
        );
        const data = await res.json();

        if (data.error) {
          setError(data.message || data.error);
          setCandles([]);
        } else {
          setCandles(data.candles || []);
          setSummary(data.summary);
        }
      } catch (err) {
        setError('Failed to fetch historical data');
        setCandles([]);
      } finally {
        setLoading(false);
      }
    }

    fetchCandles();
  }, [selectedAsset, selectedTimeframe, category]);

  // Fetch custom range data
  const fetchCustomRange = async () => {
    if (!selectedAsset) return;

    setLoading(true);
    setError(null);

    try {
      const params = new URLSearchParams({
        category,
        asset: selectedAsset,
        startDate,
        startTime,
        endDate,
        endTime,
      });

      const res = await fetch(`/api/historical?${params.toString()}`);
      const data = await res.json();

      if (data.error) {
        setError(data.message || data.error);
        setCandles([]);
      } else {
        setCandles(data.candles || []);
        setSummary(data.summary);
      }
    } catch (err) {
      setError('Failed to fetch historical data');
      setCandles([]);
    } finally {
      setLoading(false);
    }
  };

  // Handle timeframe selection
  const handleTimeframeSelect = (value: string) => {
    if (value === 'custom') {
      setShowCustomRange(true);
      setSelectedTimeframe('custom');
    } else {
      setShowCustomRange(false);
      setSelectedTimeframe(value);
    }
  };

  const isPositive = summary?.percentChange >= 0;

  return (
    <div className="bg-zinc-900 rounded-xl border border-zinc-800 p-4 mb-8">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-4">
        <h2 className="text-lg font-semibold text-white">Historical Chart</h2>

        <div className="flex flex-wrap items-center gap-3">
          {/* Asset selector */}
          <select
            value={selectedAsset}
            onChange={(e) => setSelectedAsset(e.target.value)}
            className="bg-zinc-800 text-white border border-zinc-700 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-violet-500"
          >
            {assets.length === 0 ? (
              <option value="">No data available</option>
            ) : (
              assets.map((asset) => (
                <option key={asset.asset} value={asset.asset}>
                  {asset.asset} ({(asset.candleCount / 1000).toFixed(0)}k candles)
                </option>
              ))
            )}
          </select>

          {/* Timeframe buttons */}
          <div className="flex gap-1">
            {TIMEFRAMES.map((tf) => (
              <button
                key={tf.value}
                onClick={() => handleTimeframeSelect(tf.value)}
                className={`px-3 py-1.5 text-xs font-medium rounded-lg transition-colors ${
                  selectedTimeframe === tf.value && !showCustomRange
                    ? 'bg-violet-600 text-white'
                    : 'bg-zinc-800 text-zinc-400 hover:bg-zinc-700 hover:text-zinc-300'
                }`}
              >
                {tf.label}
              </button>
            ))}
            <button
              onClick={() => handleTimeframeSelect('custom')}
              className={`px-3 py-1.5 text-xs font-medium rounded-lg transition-colors ${
                showCustomRange
                  ? 'bg-violet-600 text-white'
                  : 'bg-zinc-800 text-zinc-400 hover:bg-zinc-700 hover:text-zinc-300'
              }`}
            >
              Custom
            </button>
          </div>
        </div>
      </div>

      {/* Custom date/time range picker */}
      {showCustomRange && (
        <div className="bg-zinc-800/50 rounded-lg p-4 mb-4">
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-3 items-end">
            <div>
              <label className="block text-xs text-zinc-400 mb-1">Start Date</label>
              <input
                type="date"
                value={startDate}
                onChange={(e) => setStartDate(e.target.value)}
                className="w-full bg-zinc-700 text-white border border-zinc-600 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-violet-500"
              />
            </div>
            <div>
              <label className="block text-xs text-zinc-400 mb-1">Start Time</label>
              <input
                type="time"
                value={startTime}
                onChange={(e) => setStartTime(e.target.value)}
                className="w-full bg-zinc-700 text-white border border-zinc-600 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-violet-500"
              />
            </div>
            <div>
              <label className="block text-xs text-zinc-400 mb-1">End Date</label>
              <input
                type="date"
                value={endDate}
                onChange={(e) => setEndDate(e.target.value)}
                className="w-full bg-zinc-700 text-white border border-zinc-600 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-violet-500"
              />
            </div>
            <div>
              <label className="block text-xs text-zinc-400 mb-1">End Time</label>
              <input
                type="time"
                value={endTime}
                onChange={(e) => setEndTime(e.target.value)}
                className="w-full bg-zinc-700 text-white border border-zinc-600 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-violet-500"
              />
            </div>
            <button
              onClick={fetchCustomRange}
              disabled={loading}
              className="bg-violet-600 hover:bg-violet-700 disabled:bg-violet-800 text-white font-medium rounded-lg px-4 py-2 text-sm transition-colors"
            >
              {loading ? 'Loading...' : 'Load Data'}
            </button>
          </div>
          <p className="text-xs text-zinc-500 mt-2">
            Example: Select January 6, 2024 from 12:20 to 12:47 to view that specific time range
          </p>
        </div>
      )}

      {/* Summary stats */}
      {summary && !loading && !error && (
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-4">
          <div className="bg-zinc-800/50 rounded-lg p-3">
            <div className="text-xs text-zinc-500 mb-1">Current</div>
            <div className="text-lg font-bold text-white font-mono">
              ${summary.currentPrice.toFixed(2)}
            </div>
          </div>
          <div className="bg-zinc-800/50 rounded-lg p-3">
            <div className="text-xs text-zinc-500 mb-1">Change</div>
            <div className={`text-lg font-bold font-mono ${isPositive ? 'text-emerald-400' : 'text-red-400'}`}>
              {isPositive ? '+' : ''}{summary.percentChange.toFixed(2)}%
            </div>
          </div>
          <div className="bg-zinc-800/50 rounded-lg p-3">
            <div className="text-xs text-zinc-500 mb-1">High</div>
            <div className="text-lg font-bold text-white font-mono">
              ${summary.highPrice.toFixed(2)}
            </div>
          </div>
          <div className="bg-zinc-800/50 rounded-lg p-3">
            <div className="text-xs text-zinc-500 mb-1">Low</div>
            <div className="text-lg font-bold text-white font-mono">
              ${summary.lowPrice.toFixed(2)}
            </div>
          </div>
        </div>
      )}

      {/* Chart area */}
      <div>
        {loading ? (
          <div className="flex items-center justify-center h-[400px] bg-zinc-800/30 rounded-lg">
            <div className="text-center">
              <div className="animate-spin text-3xl mb-2">⚙️</div>
              <p className="text-zinc-400">Loading chart data...</p>
            </div>
          </div>
        ) : error ? (
          <div className="flex items-center justify-center h-[400px] bg-zinc-800/30 rounded-lg">
            <div className="text-center">
              <div className="text-3xl mb-2">📊</div>
              <p className="text-zinc-400">{error}</p>
            </div>
          </div>
        ) : candles.length > 0 ? (
          <CandlestickChart
            realCandles={candles}
            predictedCandles={[]}
            height={400}
            autoFit={true}
          />
        ) : (
          <div className="flex items-center justify-center h-[400px] bg-zinc-800/30 rounded-lg">
            <div className="text-center">
              <div className="text-3xl mb-2">📈</div>
              <p className="text-zinc-400">Select an asset to view historical data</p>
            </div>
          </div>
        )}
      </div>

      {/* Info footer */}
      {candles.length > 0 && (
        <div className="mt-3 text-xs text-zinc-500 text-center">
          Showing {candles.length} candles ({selectedTimeframe} of 1-minute data) for {selectedAsset}
        </div>
      )}
    </div>
  );
}
