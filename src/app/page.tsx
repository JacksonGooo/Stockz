'use client';

import { useState, useEffect } from 'react';
import Link from 'next/link';
import { Layout } from '@/components/layout';

interface AssetInfo {
  name: string;
  category: string;
  totalCandles: number;
  totalDays: number;
  status: 'live' | 'sleeping' | 'ready' | 'collecting' | 'error' | 'empty';
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

function formatNumber(num: number): string {
  if (num >= 1000000) return (num / 1000000).toFixed(1) + 'M';
  if (num >= 1000) return (num / 1000).toFixed(1) + 'K';
  return num.toLocaleString();
}

function formatBytes(bytes: number): string {
  if (bytes === 0) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
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

function getCategoryHref(category: string): string {
  switch (category) {
    case 'Crypto': return '/crypto';
    case 'Stock Market': return '/stocks';
    case 'Commodities': return '/commodities';
    case 'Currencies': return '/currencies';
    default: return '/data';
  }
}

export default function Dashboard() {
  const [data, setData] = useState<DataStatus | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function fetchData() {
      try {
        const res = await fetch('/api/data');
        if (res.ok) {
          const json = await res.json();
          setData(json);
        }
      } catch (error) {
        console.error('Failed to fetch data:', error);
      } finally {
        setLoading(false);
      }
    }

    fetchData();
    const interval = setInterval(fetchData, 30000);
    return () => clearInterval(interval);
  }, []);

  const totalAssets = data?.categories.reduce((sum, cat) => sum + cat.assets.length, 0) || 0;
  const liveAssets = data?.categories.reduce((sum, cat) =>
    sum + cat.assets.filter(a => a.status === 'live').length, 0) || 0;

  return (
    <Layout>
      {/* Hero Section */}
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-zinc-900 dark:text-zinc-100 mb-3">
          Welcome to S.U.P.I.D.
        </h1>
        <p className="text-sm text-zinc-500 dark:text-zinc-500 mb-2 font-medium">
          Stock Trader Usually Predicts Insane Dollarydoos
        </p>
        <p className="text-lg text-zinc-600 dark:text-zinc-400 max-w-3xl">
          A comprehensive market data platform collecting real-time and historical data
          for cryptocurrencies, stocks, commodities, and currencies. Powered by TradingView
          and Coinbase APIs with AI-driven predictions coming soon.
        </p>
      </div>

      {/* Data Collection Status */}
      <div className="bg-white dark:bg-zinc-900 rounded-xl border border-zinc-200 dark:border-zinc-800 p-6 mb-8">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-xl font-semibold text-zinc-900 dark:text-zinc-100">
            Data Collection Status
          </h2>
          <div className="flex items-center gap-2">
            <span className="relative flex h-2.5 w-2.5">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-green-400 opacity-75" />
              <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-green-500" />
            </span>
            <span className="text-sm text-green-600 dark:text-green-400 font-medium">Live</span>
          </div>
        </div>

        {loading ? (
          <div className="animate-pulse space-y-4">
            <div className="h-20 bg-zinc-200 dark:bg-zinc-800 rounded" />
          </div>
        ) : (
          <>
            {/* Stats Grid */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
              <div className="bg-zinc-50 dark:bg-zinc-800/50 rounded-lg p-4">
                <div className="text-2xl font-bold text-zinc-900 dark:text-zinc-100">
                  {formatNumber(data?.totalCandles || 0)}
                </div>
                <div className="text-sm text-zinc-600 dark:text-zinc-400">Total Candles</div>
              </div>
              <div className="bg-zinc-50 dark:bg-zinc-800/50 rounded-lg p-4">
                <div className="text-2xl font-bold text-zinc-900 dark:text-zinc-100">
                  {totalAssets}
                </div>
                <div className="text-sm text-zinc-600 dark:text-zinc-400">Assets Tracked</div>
              </div>
              <div className="bg-zinc-50 dark:bg-zinc-800/50 rounded-lg p-4">
                <div className="text-2xl font-bold text-green-600 dark:text-green-400">
                  {liveAssets}
                </div>
                <div className="text-sm text-zinc-600 dark:text-zinc-400">Live Now</div>
              </div>
              <div className="bg-zinc-50 dark:bg-zinc-800/50 rounded-lg p-4">
                <div className="text-2xl font-bold text-zinc-900 dark:text-zinc-100">
                  {formatBytes(data?.totalSize || 0)}
                </div>
                <div className="text-sm text-zinc-600 dark:text-zinc-400">Data Size</div>
              </div>
            </div>

            {/* Category Breakdown */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
              {data?.categories.map((category) => {
                const liveCount = category.assets.filter(a => a.status === 'live').length;
                const sleepingCount = category.assets.filter(a => a.status === 'sleeping').length;
                const readyCount = category.assets.filter(a => a.status === 'ready').length;

                return (
                  <Link
                    key={category.name}
                    href={getCategoryHref(category.name)}
                    className="bg-zinc-50 dark:bg-zinc-800/50 rounded-lg p-4 hover:bg-zinc-100 dark:hover:bg-zinc-800 transition-colors"
                  >
                    <div className="flex items-center gap-2 mb-2">
                      <span className="text-xl">{getCategoryIcon(category.name)}</span>
                      <span className="font-semibold text-zinc-900 dark:text-zinc-100">
                        {category.name}
                      </span>
                    </div>
                    <div className="text-sm text-zinc-600 dark:text-zinc-400 mb-2">
                      {category.assets.length} assets • {formatNumber(category.totalCandles)} candles
                    </div>
                    <div className="flex gap-2">
                      {liveCount > 0 && (
                        <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400">
                          <span className="w-1.5 h-1.5 rounded-full bg-green-500 animate-pulse" />
                          {liveCount} live
                        </span>
                      )}
                      {sleepingCount > 0 && (
                        <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs bg-amber-100 text-amber-700 dark:bg-amber-900/30 dark:text-amber-400">
                          <span className="w-1.5 h-1.5 rounded-full bg-amber-500" />
                          {sleepingCount} sleeping
                        </span>
                      )}
                      {readyCount > 0 && liveCount === 0 && sleepingCount === 0 && (
                        <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs bg-emerald-100 text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-400">
                          <span className="w-1.5 h-1.5 rounded-full bg-emerald-500" />
                          {readyCount} ready
                        </span>
                      )}
                    </div>
                  </Link>
                );
              })}
            </div>
          </>
        )}
      </div>

      {/* Features Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-8">
        <FeatureCard
          icon={
            <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
            </svg>
          }
          title="Real-Time Data"
          description="Live minute-by-minute candle data from TradingView for 49 assets across 4 categories. Updates every 60 seconds."
          color="blue"
        />
        <FeatureCard
          icon={
            <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 7v10c0 2.21 3.582 4 8 4s8-1.79 8-4V7M4 7c0 2.21 3.582 4 8 4s8-1.79 8-4M4 7c0-2.21 3.582-4 8-4s8 1.79 8 4" />
            </svg>
          }
          title="Historical Data"
          description="Up to 2 years of historical 1-minute candle data for deep analysis and backtesting. Fetched via Coinbase API."
          color="purple"
        />
        <FeatureCard
          icon={
            <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
            </svg>
          }
          title="AI Predictions"
          description="TensorFlow.js powered machine learning models trained on historical data to predict price movements. Coming soon."
          color="amber"
        />
      </div>

      {/* About Section */}
      <div className="bg-white dark:bg-zinc-900 rounded-xl border border-zinc-200 dark:border-zinc-800 p-6 mb-8">
        <h2 className="text-xl font-semibold text-zinc-900 dark:text-zinc-100 mb-4">
          About S.U.P.I.D.
        </h2>
        <div className="prose prose-zinc dark:prose-invert max-w-none">
          <p className="text-zinc-600 dark:text-zinc-400 mb-4">
            S.U.P.I.D. is a personal market analysis platform designed to collect, store, and analyze
            financial data across multiple asset classes. The platform features:
          </p>
          <ul className="text-zinc-600 dark:text-zinc-400 space-y-2 mb-4">
            <li className="flex items-start gap-2">
              <span className="text-blue-500 mt-1">•</span>
              <span><strong>15 Cryptocurrencies:</strong> BTC, ETH, SOL, XRP, DOGE, ADA, AVAX, DOT, POL, LINK, LTC, UNI, ATOM, XLM, ALGO</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-blue-500 mt-1">•</span>
              <span><strong>20 Stocks:</strong> Major indices (SPY, QQQ, DIA, IWM) and top companies (AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA, JPM, V, MA, BAC, UNH, JNJ, WMT, PG, XOM)</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-blue-500 mt-1">•</span>
              <span><strong>6 Commodities:</strong> GOLD, SILVER, OIL, NATGAS, COPPER, PLATINUM</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-blue-500 mt-1">•</span>
              <span><strong>8 Currency Pairs:</strong> EURUSD, GBPUSD, USDJPY, AUDUSD, USDCAD, USDCHF, NZDUSD, EURGBP</span>
            </li>
          </ul>
          <p className="text-zinc-600 dark:text-zinc-400">
            Data is stored in an organized folder structure with weekly folders containing daily JSON files
            of OHLCV (Open, High, Low, Close, Volume) candle data. This enables efficient querying and
            analysis while keeping the dataset manageable.
          </p>
        </div>
      </div>

      {/* Quick Links */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <Link
          href="/data"
          className="bg-white dark:bg-zinc-900 rounded-xl border border-zinc-200 dark:border-zinc-800 p-4 hover:border-blue-500 dark:hover:border-blue-500 transition-colors group"
        >
          <div className="text-blue-600 dark:text-blue-400 mb-2">
            <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 7v10c0 2.21 3.582 4 8 4s8-1.79 8-4V7M4 7c0 2.21 3.582 4 8 4s8-1.79 8-4M4 7c0-2.21 3.582-4 8-4s8 1.79 8 4" />
            </svg>
          </div>
          <div className="font-medium text-zinc-900 dark:text-zinc-100 group-hover:text-blue-600 dark:group-hover:text-blue-400">
            View Data
          </div>
          <div className="text-sm text-zinc-500 dark:text-zinc-500">
            Browse collected data
          </div>
        </Link>
        <Link
          href="/crypto"
          className="bg-white dark:bg-zinc-900 rounded-xl border border-zinc-200 dark:border-zinc-800 p-4 hover:border-blue-500 dark:hover:border-blue-500 transition-colors group"
        >
          <div className="text-amber-600 dark:text-amber-400 mb-2">
            <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8c-1.657 0-3 .895-3 2s1.343 2 3 2 3 .895 3 2-1.343 2-3 2m0-8c1.11 0 2.08.402 2.599 1M12 8V7m0 1v8m0 0v1m0-1c-1.11 0-2.08-.402-2.599-1M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
          </div>
          <div className="font-medium text-zinc-900 dark:text-zinc-100 group-hover:text-blue-600 dark:group-hover:text-blue-400">
            Crypto
          </div>
          <div className="text-sm text-zinc-500 dark:text-zinc-500">
            15 cryptocurrencies
          </div>
        </Link>
        <Link
          href="/stocks"
          className="bg-white dark:bg-zinc-900 rounded-xl border border-zinc-200 dark:border-zinc-800 p-4 hover:border-blue-500 dark:hover:border-blue-500 transition-colors group"
        >
          <div className="text-emerald-600 dark:text-emerald-400 mb-2">
            <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 8v8m-4-5v5m-4-2v2m-2 4h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
            </svg>
          </div>
          <div className="font-medium text-zinc-900 dark:text-zinc-100 group-hover:text-blue-600 dark:group-hover:text-blue-400">
            Stocks
          </div>
          <div className="text-sm text-zinc-500 dark:text-zinc-500">
            20 major stocks
          </div>
        </Link>
        <Link
          href="/ai"
          className="bg-white dark:bg-zinc-900 rounded-xl border border-zinc-200 dark:border-zinc-800 p-4 hover:border-blue-500 dark:hover:border-blue-500 transition-colors group"
        >
          <div className="text-purple-600 dark:text-purple-400 mb-2">
            <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
            </svg>
          </div>
          <div className="font-medium text-zinc-900 dark:text-zinc-100 group-hover:text-blue-600 dark:group-hover:text-blue-400">
            AI Predictions
          </div>
          <div className="text-sm text-zinc-500 dark:text-zinc-500">
            ML-powered insights
          </div>
        </Link>
      </div>
    </Layout>
  );
}

// Feature Card Component
interface FeatureCardProps {
  icon: React.ReactNode;
  title: string;
  description: string;
  color: 'blue' | 'purple' | 'amber' | 'emerald';
}

function FeatureCard({ icon, title, description, color }: FeatureCardProps) {
  const colorStyles = {
    blue: 'bg-blue-50 dark:bg-blue-900/20 text-blue-600 dark:text-blue-400',
    purple: 'bg-purple-50 dark:bg-purple-900/20 text-purple-600 dark:text-purple-400',
    amber: 'bg-amber-50 dark:bg-amber-900/20 text-amber-600 dark:text-amber-400',
    emerald: 'bg-emerald-50 dark:bg-emerald-900/20 text-emerald-600 dark:text-emerald-400',
  };

  return (
    <div className="bg-white dark:bg-zinc-900 rounded-xl border border-zinc-200 dark:border-zinc-800 p-6">
      <div className={`inline-flex p-3 rounded-lg ${colorStyles[color]} mb-4`}>
        {icon}
      </div>
      <h3 className="text-lg font-semibold text-zinc-900 dark:text-zinc-100 mb-2">
        {title}
      </h3>
      <p className="text-sm text-zinc-600 dark:text-zinc-400">
        {description}
      </p>
    </div>
  );
}
