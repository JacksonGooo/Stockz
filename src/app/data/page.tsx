'use client';

import { useState, useEffect } from 'react';
import { Layout } from '@/components/layout';

interface AssetInfo {
  name: string;
  category: string;
  totalCandles: number;
  totalDays: number;
  dateRange: { start: string; end: string } | null;
  lastUpdated: string | null;
  sizeBytes: number;
  status: 'live' | 'sleeping' | 'ready' | 'collecting' | 'error' | 'empty' | 'complete';
  hasProcessedData: boolean;
  completenessPercentage: number;
}

interface CategoryInfo {
  name: string;
  assets: AssetInfo[];
  totalCandles: number;
  totalSize: number;
}

interface DataStatus {
  categories: CategoryInfo[];
  totalCandles: number;
  totalSize: number;
  lastScan: string;
}

interface BackfillItem {
  running: boolean;
  current: string | null;
  processed: number;
  total: number;
  newCandles?: number;
  lastUpdate: string | null;
  completed?: boolean;
  source?: string;
}

interface BackfillProgress {
  stocks?: BackfillItem;
  crypto?: BackfillItem;
  forex?: BackfillItem;
  commodities?: BackfillItem;
}

function formatBytes(bytes: number): string {
  if (bytes === 0) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function formatNumber(num: number): string {
  return num.toLocaleString();
}

function getStatusColor(status: string): string {
  switch (status) {
    case 'live': return 'bg-green-500 animate-pulse';
    case 'sleeping': return 'bg-amber-500';
    case 'ready': return 'bg-emerald-500';
    case 'collecting': return 'bg-blue-500 animate-pulse';
    case 'complete': return 'bg-emerald-600';
    case 'error': return 'bg-red-500';
    default: return 'bg-zinc-400';
  }
}

function getStatusLabel(status: string): string {
  switch (status) {
    case 'live': return 'Live';
    case 'sleeping': return 'Sleeping';
    case 'ready': return 'Ready';
    case 'collecting': return 'Collecting';
    case 'complete': return 'Complete';
    case 'error': return 'Error';
    default: return 'Empty';
  }
}

function getCategoryIcon(category: string): string {
  switch (category) {
    case 'Crypto': return '₿';
    case 'Stock Market': return '📈';
    case 'Commodities': return '🥇';
    case 'Currencies': return '💱';
    default: return '📊';
  }
}

function getCompletenessColor(pct: number): string {
  if (pct >= 80) return 'bg-green-500';
  if (pct >= 50) return 'bg-amber-500';
  return 'bg-red-500';
}

export default function DataPage() {
  const [data, setData] = useState<DataStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [expandedCategory, setExpandedCategory] = useState<string | null>(null);
  const [backfillProgress, setBackfillProgress] = useState<BackfillProgress | null>(null);
  const [checkResults, setCheckResults] = useState<{
    summary: { assetsChecked: number; totalIssues: number; totalGaps: number; avgCompleteness: number };
    results: Array<{
      asset: string;
      category: string;
      completeness: number;
      gaps: Array<{ start: string; end: string; minutes: number }>;
      issues: string[];
      duplicates: number;
    }>;
  } | null>(null);
  const [checking, setChecking] = useState(false);

  const runDataCheck = async () => {
    setChecking(true);
    try {
      const res = await fetch('/api/data/check');
      if (res.ok) {
        const json = await res.json();
        setCheckResults(json);
      }
    } catch (err) {
      console.error('Data check failed:', err);
    } finally {
      setChecking(false);
    }
  };

  const fetchData = async () => {
    try {
      const res = await fetch('/api/data');
      if (!res.ok) throw new Error('Failed to fetch data');
      const json = await res.json();
      setData(json);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setLoading(false);
    }
  };

  const fetchProgress = async () => {
    try {
      const res = await fetch('/api/backfill-progress');
      if (res.ok) {
        const json = await res.json();
        setBackfillProgress(json);
      }
    } catch {
      // Ignore progress fetch errors
    }
  };

  useEffect(() => {
    fetchData();
    fetchProgress();
    // Refresh data every 30 seconds
    const dataInterval = setInterval(fetchData, 30000);
    // Refresh progress every 3 seconds
    const progressInterval = setInterval(fetchProgress, 3000);
    return () => {
      clearInterval(dataInterval);
      clearInterval(progressInterval);
    };
  }, []);

  if (loading) {
    return (
      <Layout>
        <div className="animate-pulse space-y-4">
          <div className="h-8 bg-zinc-200 dark:bg-zinc-800 rounded w-48" />
          <div className="h-32 bg-zinc-200 dark:bg-zinc-800 rounded" />
          <div className="h-32 bg-zinc-200 dark:bg-zinc-800 rounded" />
        </div>
      </Layout>
    );
  }

  if (error) {
    return (
      <Layout>
        <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-xl p-4">
          <h2 className="text-red-800 dark:text-red-200 font-semibold">Error Loading Data</h2>
          <p className="text-red-600 dark:text-red-400 text-sm mt-1">{error}</p>
          <button
            onClick={fetchData}
            className="mt-3 px-4 py-2 bg-red-600 text-white rounded-lg text-sm hover:bg-red-700"
          >
            Retry
          </button>
        </div>
      </Layout>
    );
  }

  return (
    <Layout>
      <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-zinc-900 dark:text-zinc-100">Data</h1>
          <p className="text-zinc-600 dark:text-zinc-400 text-sm mt-1">
            Historical and live market data collection
          </p>
        </div>
        <div className="flex gap-2">
          <button
            onClick={runDataCheck}
            disabled={checking}
            className="px-4 py-2 bg-purple-600 text-white rounded-lg text-sm hover:bg-purple-700 flex items-center gap-2 disabled:opacity-50"
          >
            {checking ? (
              <svg className="w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
              </svg>
            ) : (
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
            )}
            {checking ? 'Checking...' : 'Check Data'}
          </button>
          <button
            onClick={fetchData}
            className="px-4 py-2 bg-blue-600 text-white rounded-lg text-sm hover:bg-blue-700 flex items-center gap-2"
          >
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
            </svg>
            Refresh
          </button>
        </div>
      </div>

      {/* Data Check Results Panel */}
      {checkResults && (
        <div className="bg-gradient-to-r from-purple-50 to-indigo-50 dark:from-purple-900/20 dark:to-indigo-900/20 border border-purple-200 dark:border-purple-800 rounded-xl p-4">
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center gap-2">
              <svg className="w-5 h-5 text-purple-600 dark:text-purple-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              <h2 className="font-semibold text-purple-900 dark:text-purple-100">Data Check Results</h2>
            </div>
            <button
              onClick={() => setCheckResults(null)}
              className="text-purple-600 hover:text-purple-800 dark:text-purple-400"
            >
              <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>

          {/* Summary */}
          <div className="grid grid-cols-4 gap-4 mb-4">
            <div className="bg-white/50 dark:bg-zinc-800/50 rounded-lg p-3 text-center">
              <div className="text-2xl font-bold text-purple-700 dark:text-purple-300">{checkResults.summary.assetsChecked}</div>
              <div className="text-xs text-purple-600 dark:text-purple-400">Assets Checked</div>
            </div>
            <div className="bg-white/50 dark:bg-zinc-800/50 rounded-lg p-3 text-center">
              <div className={`text-2xl font-bold ${checkResults.summary.avgCompleteness >= 90 ? 'text-green-600' : checkResults.summary.avgCompleteness >= 70 ? 'text-amber-600' : 'text-red-600'}`}>
                {checkResults.summary.avgCompleteness}%
              </div>
              <div className="text-xs text-purple-600 dark:text-purple-400">Avg Completeness</div>
            </div>
            <div className="bg-white/50 dark:bg-zinc-800/50 rounded-lg p-3 text-center">
              <div className={`text-2xl font-bold ${checkResults.summary.totalGaps === 0 ? 'text-green-600' : 'text-amber-600'}`}>
                {checkResults.summary.totalGaps}
              </div>
              <div className="text-xs text-purple-600 dark:text-purple-400">Data Gaps</div>
            </div>
            <div className="bg-white/50 dark:bg-zinc-800/50 rounded-lg p-3 text-center">
              <div className={`text-2xl font-bold ${checkResults.summary.totalIssues === 0 ? 'text-green-600' : 'text-red-600'}`}>
                {checkResults.summary.totalIssues}
              </div>
              <div className="text-xs text-purple-600 dark:text-purple-400">Issues Found</div>
            </div>
          </div>

          {/* Issues List */}
          {checkResults.results.some(r => r.issues.length > 0) && (
            <div className="bg-white/50 dark:bg-zinc-800/50 rounded-lg p-3">
              <h3 className="text-sm font-medium text-purple-800 dark:text-purple-200 mb-2">Issues:</h3>
              <ul className="space-y-1">
                {checkResults.results
                  .filter(r => r.issues.length > 0)
                  .map((r, i) => (
                    <li key={i} className="text-xs text-red-600 dark:text-red-400 flex items-start gap-2">
                      <span className="font-mono bg-red-100 dark:bg-red-900/30 px-1 rounded">{r.asset}</span>
                      <span>{r.issues.join(', ')}</span>
                    </li>
                  ))
                }
              </ul>
            </div>
          )}
        </div>
      )}

      {/* Backfill Progress Panel */}
      {backfillProgress && (backfillProgress.stocks?.running || backfillProgress.crypto?.running || backfillProgress.forex?.running || backfillProgress.commodities?.running) && (
        <div className="bg-gradient-to-r from-blue-50 to-purple-50 dark:from-blue-900/20 dark:to-purple-900/20 border border-blue-200 dark:border-blue-800 rounded-xl p-4">
          <div className="flex items-center gap-2 mb-3">
            <div className="w-2 h-2 bg-blue-500 rounded-full animate-pulse" />
            <h2 className="font-semibold text-blue-900 dark:text-blue-100">Historical Data Backfill In Progress</h2>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            {/* Stocks Progress */}
            {backfillProgress.stocks && (
              <div className="bg-white/50 dark:bg-zinc-800/50 rounded-lg p-3">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm font-medium text-zinc-700 dark:text-zinc-300">📈 Stocks ({backfillProgress.stocks.source || 'Alpaca'})</span>
                  {backfillProgress.stocks.running ? (
                    <span className="text-xs bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400 px-2 py-0.5 rounded-full">Running</span>
                  ) : backfillProgress.stocks.completed ? (
                    <span className="text-xs bg-emerald-100 text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-400 px-2 py-0.5 rounded-full">Complete</span>
                  ) : (
                    <span className="text-xs bg-zinc-100 text-zinc-600 dark:bg-zinc-800 dark:text-zinc-400 px-2 py-0.5 rounded-full">Idle</span>
                  )}
                </div>
                {backfillProgress.stocks.running && (
                  <>
                    <div className="text-xs text-zinc-600 dark:text-zinc-400 mb-1">
                      Current: <span className="font-mono text-zinc-900 dark:text-zinc-100">{backfillProgress.stocks.current}</span>
                      {backfillProgress.stocks.newCandles ? (
                        <span className="ml-2 text-green-600 dark:text-green-400">+{backfillProgress.stocks.newCandles.toLocaleString()}</span>
                      ) : null}
                    </div>
                    <div className="w-full h-2 bg-zinc-200 dark:bg-zinc-700 rounded-full overflow-hidden mb-1">
                      <div
                        className="h-full bg-blue-500 transition-all duration-500"
                        style={{ width: `${(backfillProgress.stocks.processed / backfillProgress.stocks.total) * 100}%` }}
                      />
                    </div>
                    <div className="text-xs text-zinc-500 dark:text-zinc-500">
                      {backfillProgress.stocks.processed} / {backfillProgress.stocks.total} ({((backfillProgress.stocks.processed / backfillProgress.stocks.total) * 100).toFixed(1)}%)
                    </div>
                  </>
                )}
              </div>
            )}
            {/* Crypto Progress */}
            {backfillProgress.crypto && (
              <div className="bg-white/50 dark:bg-zinc-800/50 rounded-lg p-3">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm font-medium text-zinc-700 dark:text-zinc-300">₿ Crypto ({backfillProgress.crypto.source || 'Binance'})</span>
                  {backfillProgress.crypto.running ? (
                    <span className="text-xs bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400 px-2 py-0.5 rounded-full">Running</span>
                  ) : backfillProgress.crypto.completed ? (
                    <span className="text-xs bg-emerald-100 text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-400 px-2 py-0.5 rounded-full">Complete</span>
                  ) : (
                    <span className="text-xs bg-zinc-100 text-zinc-600 dark:bg-zinc-800 dark:text-zinc-400 px-2 py-0.5 rounded-full">Idle</span>
                  )}
                </div>
                {backfillProgress.crypto.running && (
                  <>
                    <div className="text-xs text-zinc-600 dark:text-zinc-400 mb-1">
                      Current: <span className="font-mono text-zinc-900 dark:text-zinc-100">{backfillProgress.crypto.current}</span>
                      {backfillProgress.crypto.newCandles ? (
                        <span className="ml-2 text-green-600 dark:text-green-400">+{backfillProgress.crypto.newCandles.toLocaleString()}</span>
                      ) : null}
                    </div>
                    <div className="w-full h-2 bg-zinc-200 dark:bg-zinc-700 rounded-full overflow-hidden mb-1">
                      <div
                        className="h-full bg-purple-500 transition-all duration-500"
                        style={{ width: `${(backfillProgress.crypto.processed / backfillProgress.crypto.total) * 100}%` }}
                      />
                    </div>
                    <div className="text-xs text-zinc-500 dark:text-zinc-500">
                      {backfillProgress.crypto.processed} / {backfillProgress.crypto.total} ({((backfillProgress.crypto.processed / backfillProgress.crypto.total) * 100).toFixed(1)}%)
                    </div>
                  </>
                )}
              </div>
            )}
            {/* Forex Progress */}
            {backfillProgress.forex && (
              <div className="bg-white/50 dark:bg-zinc-800/50 rounded-lg p-3">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm font-medium text-zinc-700 dark:text-zinc-300">💱 Forex ({backfillProgress.forex.source || 'Polygon'})</span>
                  {backfillProgress.forex.running ? (
                    <span className="text-xs bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400 px-2 py-0.5 rounded-full">Running</span>
                  ) : backfillProgress.forex.completed ? (
                    <span className="text-xs bg-emerald-100 text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-400 px-2 py-0.5 rounded-full">Complete</span>
                  ) : (
                    <span className="text-xs bg-zinc-100 text-zinc-600 dark:bg-zinc-800 dark:text-zinc-400 px-2 py-0.5 rounded-full">Idle</span>
                  )}
                </div>
                {backfillProgress.forex.running && (
                  <>
                    <div className="text-xs text-zinc-600 dark:text-zinc-400 mb-1">
                      Current: <span className="font-mono text-zinc-900 dark:text-zinc-100">{backfillProgress.forex.current}</span>
                      {backfillProgress.forex.newCandles ? (
                        <span className="ml-2 text-green-600 dark:text-green-400">+{backfillProgress.forex.newCandles.toLocaleString()}</span>
                      ) : null}
                    </div>
                    <div className="w-full h-2 bg-zinc-200 dark:bg-zinc-700 rounded-full overflow-hidden mb-1">
                      <div
                        className="h-full bg-amber-500 transition-all duration-500"
                        style={{ width: `${(backfillProgress.forex.processed / backfillProgress.forex.total) * 100}%` }}
                      />
                    </div>
                    <div className="text-xs text-zinc-500 dark:text-zinc-500">
                      {backfillProgress.forex.processed} / {backfillProgress.forex.total} ({((backfillProgress.forex.processed / backfillProgress.forex.total) * 100).toFixed(1)}%)
                    </div>
                  </>
                )}
              </div>
            )}
            {/* Commodities Progress */}
            {backfillProgress.commodities && (
              <div className="bg-white/50 dark:bg-zinc-800/50 rounded-lg p-3">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm font-medium text-zinc-700 dark:text-zinc-300">🥇 Commodities ({backfillProgress.commodities.source || 'Alpaca'})</span>
                  {backfillProgress.commodities.running ? (
                    <span className="text-xs bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400 px-2 py-0.5 rounded-full">Running</span>
                  ) : backfillProgress.commodities.completed ? (
                    <span className="text-xs bg-emerald-100 text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-400 px-2 py-0.5 rounded-full">Complete</span>
                  ) : (
                    <span className="text-xs bg-zinc-100 text-zinc-600 dark:bg-zinc-800 dark:text-zinc-400 px-2 py-0.5 rounded-full">Idle</span>
                  )}
                </div>
                {backfillProgress.commodities.running && (
                  <>
                    <div className="text-xs text-zinc-600 dark:text-zinc-400 mb-1">
                      Current: <span className="font-mono text-zinc-900 dark:text-zinc-100">{backfillProgress.commodities.current}</span>
                      {backfillProgress.commodities.newCandles ? (
                        <span className="ml-2 text-green-600 dark:text-green-400">+{backfillProgress.commodities.newCandles.toLocaleString()}</span>
                      ) : null}
                    </div>
                    <div className="w-full h-2 bg-zinc-200 dark:bg-zinc-700 rounded-full overflow-hidden mb-1">
                      <div
                        className="h-full bg-yellow-500 transition-all duration-500"
                        style={{ width: `${(backfillProgress.commodities.processed / backfillProgress.commodities.total) * 100}%` }}
                      />
                    </div>
                    <div className="text-xs text-zinc-500 dark:text-zinc-500">
                      {backfillProgress.commodities.processed} / {backfillProgress.commodities.total} ({((backfillProgress.commodities.processed / backfillProgress.commodities.total) * 100).toFixed(1)}%)
                    </div>
                  </>
                )}
              </div>
            )}
          </div>
        </div>
      )}

      {/* Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="bg-white dark:bg-zinc-900 rounded-xl border border-zinc-200 dark:border-zinc-800 p-4">
          <div className="text-sm text-zinc-600 dark:text-zinc-400">Total Candles</div>
          <div className="text-2xl font-bold text-zinc-900 dark:text-zinc-100 mt-1">
            {formatNumber(data?.totalCandles || 0)}
          </div>
        </div>
        <div className="bg-white dark:bg-zinc-900 rounded-xl border border-zinc-200 dark:border-zinc-800 p-4">
          <div className="text-sm text-zinc-600 dark:text-zinc-400">Data Size</div>
          <div className="text-2xl font-bold text-zinc-900 dark:text-zinc-100 mt-1">
            {formatBytes(data?.totalSize || 0)}
          </div>
        </div>
        <div className="bg-white dark:bg-zinc-900 rounded-xl border border-zinc-200 dark:border-zinc-800 p-4">
          <div className="text-sm text-zinc-600 dark:text-zinc-400">Categories</div>
          <div className="text-2xl font-bold text-zinc-900 dark:text-zinc-100 mt-1">
            {data?.categories.length || 0}
          </div>
        </div>
        <div className="bg-white dark:bg-zinc-900 rounded-xl border border-zinc-200 dark:border-zinc-800 p-4">
          <div className="text-sm text-zinc-600 dark:text-zinc-400">Last Scan</div>
          <div className="text-lg font-medium text-zinc-900 dark:text-zinc-100 mt-1">
            {data?.lastScan ? new Date(data.lastScan).toLocaleTimeString() : '-'}
          </div>
        </div>
      </div>

      {/* Categories */}
      <div className="space-y-4">
        {data?.categories.map((category) => (
          <div
            key={category.name}
            className="bg-white dark:bg-zinc-900 rounded-xl border border-zinc-200 dark:border-zinc-800 overflow-hidden"
          >
            {/* Category Header */}
            <button
              onClick={() => setExpandedCategory(
                expandedCategory === category.name ? null : category.name
              )}
              className="w-full px-4 py-4 flex items-center justify-between hover:bg-zinc-50 dark:hover:bg-zinc-800/50 transition-colors"
            >
              <div className="flex items-center gap-3">
                <span className="text-2xl">{getCategoryIcon(category.name)}</span>
                <div className="text-left">
                  <h2 className="font-semibold text-zinc-900 dark:text-zinc-100">
                    {category.name}
                  </h2>
                  <p className="text-sm text-zinc-600 dark:text-zinc-400">
                    {category.assets.length} assets • {formatNumber(category.totalCandles)} candles • {formatBytes(category.totalSize)}
                  </p>
                </div>
              </div>
              <div className="flex items-center gap-3">
                {/* Status indicators */}
                <div className="flex gap-1">
                  {category.assets.slice(0, 5).map((asset) => (
                    <div
                      key={asset.name}
                      className={`w-2 h-2 rounded-full ${getStatusColor(asset.status)}`}
                      title={`${asset.name}: ${asset.status}`}
                    />
                  ))}
                </div>
                <svg
                  className={`w-5 h-5 text-zinc-400 transition-transform ${
                    expandedCategory === category.name ? 'rotate-180' : ''
                  }`}
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                >
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                </svg>
              </div>
            </button>

            {/* Assets List */}
            {expandedCategory === category.name && (
              <div className="border-t border-zinc-200 dark:border-zinc-800">
                <div className="overflow-x-auto">
                  <table className="w-full">
                    <thead className="bg-zinc-50 dark:bg-zinc-800/50">
                      <tr>
                        <th className="px-4 py-3 text-left text-xs font-medium text-zinc-500 dark:text-zinc-400 uppercase tracking-wider">Asset</th>
                        <th className="px-4 py-3 text-left text-xs font-medium text-zinc-500 dark:text-zinc-400 uppercase tracking-wider">Status</th>
                        <th className="px-4 py-3 text-right text-xs font-medium text-zinc-500 dark:text-zinc-400 uppercase tracking-wider">Candles</th>
                        <th className="px-4 py-3 text-right text-xs font-medium text-zinc-500 dark:text-zinc-400 uppercase tracking-wider">Days</th>
                        <th className="px-4 py-3 text-left text-xs font-medium text-zinc-500 dark:text-zinc-400 uppercase tracking-wider">Date Range</th>
                        <th className="px-4 py-3 text-right text-xs font-medium text-zinc-500 dark:text-zinc-400 uppercase tracking-wider">Size</th>
                        <th className="px-4 py-3 text-center text-xs font-medium text-zinc-500 dark:text-zinc-400 uppercase tracking-wider">Completeness</th>
                        <th className="px-4 py-3 text-center text-xs font-medium text-zinc-500 dark:text-zinc-400 uppercase tracking-wider">Processed</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-zinc-200 dark:divide-zinc-800">
                      {category.assets.map((asset) => (
                        <tr key={asset.name} className="hover:bg-zinc-50 dark:hover:bg-zinc-800/30">
                          <td className="px-4 py-3">
                            <span className="font-medium text-zinc-900 dark:text-zinc-100">{asset.name}</span>
                          </td>
                          <td className="px-4 py-3">
                            <span className={`inline-flex items-center gap-1.5 px-2 py-1 rounded-full text-xs font-medium ${
                              asset.status === 'live'
                                ? 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400'
                                : asset.status === 'sleeping'
                                ? 'bg-amber-100 text-amber-700 dark:bg-amber-900/30 dark:text-amber-400'
                                : asset.status === 'ready'
                                ? 'bg-emerald-100 text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-400'
                                : asset.status === 'collecting'
                                ? 'bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-400'
                                : asset.status === 'complete'
                                ? 'bg-emerald-100 text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-400'
                                : asset.status === 'error'
                                ? 'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400'
                                : 'bg-zinc-100 text-zinc-700 dark:bg-zinc-800 dark:text-zinc-400'
                            }`}>
                              <span className={`w-1.5 h-1.5 rounded-full ${getStatusColor(asset.status)}`} />
                              {getStatusLabel(asset.status)}
                            </span>
                          </td>
                          <td className="px-4 py-3 text-right font-mono text-sm text-zinc-900 dark:text-zinc-100">
                            {formatNumber(asset.totalCandles)}
                          </td>
                          <td className="px-4 py-3 text-right font-mono text-sm text-zinc-600 dark:text-zinc-400">
                            {asset.totalDays}
                          </td>
                          <td className="px-4 py-3 text-sm text-zinc-600 dark:text-zinc-400">
                            {asset.dateRange
                              ? `${asset.dateRange.start} to ${asset.dateRange.end}`
                              : '-'
                            }
                          </td>
                          <td className="px-4 py-3 text-right text-sm text-zinc-600 dark:text-zinc-400">
                            {formatBytes(asset.sizeBytes)}
                          </td>
                          <td className="px-4 py-3">
                            <div className="flex items-center gap-2">
                              <div className="w-16 h-2 bg-zinc-200 dark:bg-zinc-700 rounded-full overflow-hidden">
                                <div
                                  className={`h-full ${getCompletenessColor(asset.completenessPercentage)}`}
                                  style={{ width: `${asset.completenessPercentage}%` }}
                                />
                              </div>
                              <span className="text-xs font-mono text-zinc-600 dark:text-zinc-400">
                                {asset.completenessPercentage}%
                              </span>
                            </div>
                          </td>
                          <td className="px-4 py-3 text-center">
                            {asset.hasProcessedData ? (
                              <span className="text-emerald-500">✓</span>
                            ) : (
                              <span className="text-zinc-400">-</span>
                            )}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          </div>
        ))}
      </div>

      </div>
    </Layout>
  );
}
