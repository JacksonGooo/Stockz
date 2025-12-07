'use client';

import { useState } from 'react';
import Link from 'next/link';
import { Layout } from '@/components/layout';
import { Card, Badge, Input } from '@/components/ui';
import { useCyclingFetch } from '@/hooks';
import { formatTimeCST } from '@/lib/time';

interface CryptoAsset {
  symbol: string;
  name: string;
  price: number;
  change: number;
  changePercent: number;
  volume: number;
  high: number;
  low: number;
  recommendation?: string;
}

const DEFAULT_CRYPTO_SYMBOLS = [
  'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT', 'SOLUSDT',
  'ADAUSDT', 'DOGEUSDT', 'AVAXUSDT', 'DOTUSDT', 'LINKUSDT',
];

// Fetch function for crypto
async function fetchCryptoData(symbols: string[]): Promise<CryptoAsset[]> {
  try {
    const response = await fetch(`/api/crypto?symbols=${symbols.join(',')}`);
    const data = await response.json();
    return data.quotes || [];
  } catch (error) {
    console.error('Failed to fetch crypto data:', error);
    return [];
  }
}

function formatPrice(price: number): string {
  if (price >= 1000) return price.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  if (price >= 1) return price.toFixed(2);
  if (price >= 0.01) return price.toFixed(4);
  return price.toFixed(6);
}

function formatVolume(volume: number): string {
  if (volume >= 1e12) return `${(volume / 1e12).toFixed(2)}T`;
  if (volume >= 1e9) return `${(volume / 1e9).toFixed(2)}B`;
  if (volume >= 1e6) return `${(volume / 1e6).toFixed(2)}M`;
  if (volume >= 1e3) return `${(volume / 1e3).toFixed(2)}K`;
  return volume.toFixed(2);
}

function CryptoCard({ crypto, isUpdating }: { crypto: CryptoAsset; isUpdating?: boolean }) {
  const isPositive = crypto.change >= 0;

  const getBaseSymbol = (symbol: string) => {
    return symbol.replace('USDT', '').replace('USD', '');
  };

  const getCryptoIcon = (symbol: string) => {
    const baseSymbol = getBaseSymbol(symbol);
    const icons: Record<string, string> = {
      BTC: 'B', ETH: 'E', BNB: 'B', XRP: 'X', SOL: 'S',
      ADA: 'A', DOGE: 'D', AVAX: 'A', DOT: 'D', LINK: 'L',
    };
    return icons[baseSymbol] || baseSymbol.charAt(0);
  };

  const getCryptoColor = (symbol: string) => {
    const baseSymbol = getBaseSymbol(symbol);
    const colors: Record<string, string> = {
      BTC: 'from-orange-500/20 to-amber-500/20 text-orange-600 dark:text-orange-400',
      ETH: 'from-indigo-500/20 to-purple-500/20 text-indigo-600 dark:text-indigo-400',
      BNB: 'from-yellow-500/20 to-amber-500/20 text-yellow-600 dark:text-yellow-400',
      XRP: 'from-slate-500/20 to-gray-500/20 text-slate-600 dark:text-slate-400',
      SOL: 'from-purple-500/20 to-pink-500/20 text-purple-600 dark:text-purple-400',
      ADA: 'from-blue-500/20 to-cyan-500/20 text-blue-600 dark:text-blue-400',
      DOGE: 'from-amber-500/20 to-yellow-500/20 text-amber-600 dark:text-amber-400',
      AVAX: 'from-red-500/20 to-rose-500/20 text-red-600 dark:text-red-400',
      DOT: 'from-pink-500/20 to-rose-500/20 text-pink-600 dark:text-pink-400',
      LINK: 'from-blue-500/20 to-indigo-500/20 text-blue-600 dark:text-blue-400',
    };
    return colors[baseSymbol] || 'from-zinc-500/20 to-gray-500/20 text-zinc-600 dark:text-zinc-400';
  };

  return (
    <Link href={`/crypto/${crypto.symbol}`}>
      <Card
        hover
        className={`
          group cursor-pointer transition-all duration-200 hover:shadow-lg
          ${isUpdating ? 'ring-2 ring-blue-400 dark:ring-blue-500' : ''}
        `}
      >
        <div className="flex items-start justify-between mb-4">
          <div className="flex items-center gap-3">
            <div
              className={`
                w-12 h-12 rounded-xl flex items-center justify-center
                font-bold text-lg bg-gradient-to-br ${getCryptoColor(crypto.symbol)}
                ${isUpdating ? 'animate-pulse' : ''}
              `}
            >
              {getCryptoIcon(crypto.symbol)}
            </div>
            <div>
              <h3 className="font-semibold text-zinc-900 dark:text-zinc-100 group-hover:text-blue-600 dark:group-hover:text-blue-400 transition-colors">
                {getBaseSymbol(crypto.symbol)}
              </h3>
              <p className="text-sm text-zinc-500 dark:text-zinc-400 truncate max-w-[150px]">
                {crypto.name}
              </p>
            </div>
          </div>
          {crypto.recommendation && (
            <Badge
              variant={
                crypto.recommendation.includes('BUY')
                  ? 'success'
                  : crypto.recommendation.includes('SELL')
                    ? 'danger'
                    : 'default'
              }
              size="sm"
            >
              {crypto.recommendation.replace('STRONG_', '')}
            </Badge>
          )}
        </div>

        <div className="flex items-end justify-between">
          <div>
            <p className="text-2xl font-bold font-mono text-zinc-900 dark:text-zinc-100">
              ${formatPrice(crypto.price)}
            </p>
            <div className="flex items-center gap-2 mt-1">
              <span
                className={`text-sm font-medium ${
                  isPositive
                    ? 'text-emerald-600 dark:text-emerald-400'
                    : 'text-red-600 dark:text-red-400'
                }`}
              >
                {isPositive ? '+' : ''}
                {crypto.changePercent.toFixed(2)}%
              </span>
              <span className="text-xs text-zinc-400">24h</span>
            </div>
          </div>
          <div className="text-right">
            <p className="text-xs text-zinc-500 dark:text-zinc-400">Volume</p>
            <p className="text-sm font-medium text-zinc-700 dark:text-zinc-300 font-mono">
              ${formatVolume(crypto.volume)}
            </p>
          </div>
        </div>
      </Card>
    </Link>
  );
}

export function CryptoContent() {
  const [searchQuery, setSearchQuery] = useState('');

  const {
    data: cryptos,
    isLive,
    setIsLive,
    currentSymbols,
    lastUpdated,
    isLoading,
  } = useCyclingFetch<CryptoAsset>({
    symbols: DEFAULT_CRYPTO_SYMBOLS,
    fetchFn: fetchCryptoData,
    cycleSize: 3,
    intervalMs: 3000,
    getSymbol: (crypto) => crypto.symbol,
  });

  const filteredCryptos = searchQuery
    ? cryptos.filter(
        (c) =>
          c.symbol.toLowerCase().includes(searchQuery.toLowerCase()) ||
          c.name.toLowerCase().includes(searchQuery.toLowerCase())
      )
    : cryptos;

  const marketStats = {
    totalVolume: cryptos.reduce((sum, c) => sum + c.volume, 0),
    avgChange: cryptos.length
      ? cryptos.reduce((sum, c) => sum + c.changePercent, 0) / cryptos.length
      : 0,
    gainers: cryptos.filter((c) => c.change > 0).length,
    losers: cryptos.filter((c) => c.change < 0).length,
  };

  const getBaseSymbol = (symbol: string) => symbol.replace('USDT', '').replace('USD', '');

  return (
    <Layout>
      <div className="flex flex-col md:flex-row md:items-start md:justify-between gap-4 mb-8">
        <div>
          <h1 className="text-3xl font-bold text-zinc-900 dark:text-zinc-100 mb-2">
            Crypto Markets
          </h1>
          <p className="text-zinc-500 dark:text-zinc-400">
            Real-time cryptocurrency prices and analysis
          </p>
        </div>
        <div className="flex flex-col items-end gap-2">
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2">
              {isLive && (
                <span className="relative flex h-2 w-2">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75" />
                  <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500" />
                </span>
              )}
              <span className="text-sm text-zinc-600 dark:text-zinc-400">
                {isLive ? 'Live' : 'Paused'}
              </span>
            </div>
            <button
              onClick={() => setIsLive(!isLive)}
              className={`
                px-3 py-1.5 rounded-lg text-sm font-medium transition-all
                ${isLive
                  ? 'bg-emerald-100 text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-400'
                  : 'bg-zinc-100 text-zinc-600 dark:bg-zinc-800 dark:text-zinc-400'
                }
              `}
            >
              {isLive ? 'Pause' : 'Resume'}
            </button>
          </div>
          {isLive && currentSymbols.length > 0 && (
            <div className="text-xs text-zinc-400 flex items-center gap-1">
              <span className="animate-pulse">Updating:</span>
              <span className="font-mono text-blue-500 dark:text-blue-400">
                {currentSymbols.map(s => getBaseSymbol(s)).join(', ')}
              </span>
            </div>
          )}
          {lastUpdated && (
            <span className="text-xs text-zinc-400">
              Last update: {formatTimeCST(lastUpdated)}
            </span>
          )}
        </div>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
        <Card>
          <p className="text-sm text-zinc-500 dark:text-zinc-400 mb-1">24h Volume</p>
          <p className="text-xl font-bold text-zinc-900 dark:text-zinc-100">
            ${formatVolume(marketStats.totalVolume)}
          </p>
        </Card>
        <Card>
          <p className="text-sm text-zinc-500 dark:text-zinc-400 mb-1">Avg Change</p>
          <p
            className={`text-xl font-bold ${
              marketStats.avgChange >= 0
                ? 'text-emerald-600 dark:text-emerald-400'
                : 'text-red-600 dark:text-red-400'
            }`}
          >
            {marketStats.avgChange >= 0 ? '+' : ''}
            {marketStats.avgChange.toFixed(2)}%
          </p>
        </Card>
        <Card>
          <p className="text-sm text-zinc-500 dark:text-zinc-400 mb-1">Gainers</p>
          <p className="text-xl font-bold text-emerald-600 dark:text-emerald-400">
            {marketStats.gainers}
          </p>
        </Card>
        <Card>
          <p className="text-sm text-zinc-500 dark:text-zinc-400 mb-1">Losers</p>
          <p className="text-xl font-bold text-red-600 dark:text-red-400">
            {marketStats.losers}
          </p>
        </Card>
      </div>

      <Card className="mb-8">
        <div className="flex flex-col md:flex-row gap-4">
          <div className="flex-1">
            <Input
              placeholder="Search cryptocurrencies..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              leftIcon={
                <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                </svg>
              }
            />
          </div>
        </div>
      </Card>

      <div className="flex items-center justify-between mb-4">
        <p className="text-sm text-zinc-500 dark:text-zinc-400">
          Showing {filteredCryptos.length} of {cryptos.length} cryptocurrencies
        </p>
      </div>

      {isLoading ? (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {Array.from({ length: 6 }).map((_, i) => (
            <Card key={i} className="animate-pulse">
              <div className="h-6 w-24 bg-zinc-200 dark:bg-zinc-800 rounded mb-2" />
              <div className="h-4 w-32 bg-zinc-200 dark:bg-zinc-800 rounded mb-4" />
              <div className="h-8 w-28 bg-zinc-200 dark:bg-zinc-800 rounded" />
            </Card>
          ))}
        </div>
      ) : filteredCryptos.length === 0 ? (
        <div className="text-center py-12">
          <svg className="w-12 h-12 mx-auto text-zinc-400 dark:text-zinc-600 mb-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9.172 16.172a4 4 0 015.656 0M9 10h.01M15 10h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
          <p className="text-zinc-500 dark:text-zinc-400">No cryptocurrencies found</p>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {filteredCryptos.map((crypto) => (
            <CryptoCard
              key={crypto.symbol}
              crypto={crypto}
              isUpdating={currentSymbols.includes(crypto.symbol)}
            />
          ))}
        </div>
      )}
    </Layout>
  );
}
