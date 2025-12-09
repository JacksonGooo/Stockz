'use client';

import { useState, useEffect } from 'react';
import { Layout } from '@/components/layout';
import { Card, Badge } from '@/components/ui';
import { StockList } from '@/components/stocks';
import { HistoricalChartViewer } from '@/components/charts';

// Crypto symbols (TradingView format: EXCHANGE:SYMBOL)
const CRYPTO_SYMBOLS = [
  'BINANCE:BTCUSDT',   // Bitcoin
  'BINANCE:ETHUSDT',   // Ethereum
  'BINANCE:BNBUSDT',   // Binance Coin
  'BINANCE:SOLUSDT',   // Solana
  'BINANCE:XRPUSDT',   // Ripple
  'BINANCE:ADAUSDT',   // Cardano
  'BINANCE:DOGEUSDT',  // Dogecoin
  'BINANCE:MATICUSDT', // Polygon
  'BINANCE:DOTUSDT',   // Polkadot
  'BINANCE:AVAXUSDT',  // Avalanche
  'BINANCE:LINKUSDT',  // Chainlink
  'BINANCE:UNIUSDT',   // Uniswap
  'BINANCE:LTCUSDT',   // Litecoin
  'BINANCE:ATOMUSDT',  // Cosmos
  'BINANCE:ETCUSDT',   // Ethereum Classic
  'BINANCE:XLMUSDT',   // Stellar
];

interface CryptoAsset {
  symbol: string;
  name: string;
  currentPrice: number;
  previousClose: number;
  change: number;
  changePercent: number;
  volume: number;
  marketCap: number;
  high52Week: number;
  low52Week: number;
  sector: string;
}

export default function CryptoPage() {
  const [cryptos, setCryptos] = useState<CryptoAsset[]>([]);
  const [topGainers, setTopGainers] = useState<CryptoAsset[]>([]);
  const [topLosers, setTopLosers] = useState<CryptoAsset[]>([]);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    async function fetchCryptoData() {
      try {
        setIsLoading(true);

        // Fetch crypto quotes from TradingView
        const response = await fetch(`/api/stocks/quotes?symbols=${CRYPTO_SYMBOLS.join(',')}`);
        const data = await response.json();

        if (data.quotes) {
          const cryptoData: CryptoAsset[] = data.quotes.map((q: any) => {
            // Extract crypto symbol (e.g., BINANCE:BTCUSDT -> BTC)
            const symbol = q.symbol.replace('USDT', '').replace('USD', '');

            return {
              symbol,
              name: getDisplayName(symbol),
              sector: 'Cryptocurrency',
              currentPrice: q.price || 0,
              previousClose: q.previousClose || q.price,
              change: q.change || 0,
              changePercent: q.changePercent || 0,
              volume: q.volume || 0,
              marketCap: 0, // TradingView doesn't provide market cap for crypto
              high52Week: q.price * 1.5,
              low52Week: q.price * 0.5,
            };
          });

          setCryptos(cryptoData);

          // Calculate top gainers (highest positive change)
          const gainers = cryptoData.filter(c => c.changePercent > 0).sort((a, b) => b.changePercent - a.changePercent);
          setTopGainers(gainers.slice(0, 3));

          // Calculate top losers (most negative change first)
          const losers = cryptoData.filter(c => c.changePercent < 0).sort((a, b) => a.changePercent - b.changePercent);
          setTopLosers(losers.slice(0, 3));
        }
      } catch (error) {
        console.error('Failed to fetch crypto data:', error);
      } finally {
        setIsLoading(false);
      }
    }

    fetchCryptoData();

    // Refresh every 30 seconds
    const interval = setInterval(fetchCryptoData, 30000);
    return () => clearInterval(interval);
  }, []);

  const totalMarketCap = cryptos.reduce((sum, crypto) => sum + crypto.marketCap, 0);
  const avgChangePercent = cryptos.length > 0
    ? cryptos.reduce((sum, c) => sum + c.changePercent, 0) / cryptos.length
    : 0;

  return (
    <Layout>
      {/* Page header */}
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-zinc-900 dark:text-zinc-100 mb-2">
          Cryptocurrency Market
        </h1>
        <p className="text-zinc-500 dark:text-zinc-400">
          Real-time cryptocurrency prices and market data
        </p>
      </div>

      {/* Historical Chart Viewer */}
      <HistoricalChartViewer category="Crypto" />

      {/* Stats overview */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 mb-8">
        <StatCard
          title="Total Assets"
          value={cryptos.length.toString()}
          subtitle="Tracked cryptocurrencies"
          icon={
            <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8c-1.657 0-3 .895-3 2s1.343 2 3 2 3 .895 3 2-1.343 2-3 2m0-8c1.11 0 2.08.402 2.599 1M12 8V7m0 1v8m0 0v1m0-1c-1.11 0-2.08-.402-2.599-1M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
          }
          color="blue"
        />
        <StatCard
          title="Market Trend"
          value={avgChangePercent >= 0 ? 'BULLISH' : 'BEARISH'}
          subtitle={`Avg: ${avgChangePercent.toFixed(2)}%`}
          icon={
            <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
            </svg>
          }
          color={avgChangePercent >= 0 ? 'green' : 'red'}
        />
        <StatCard
          title="24h Volume"
          value={formatLargeNumber(cryptos.reduce((sum, c) => sum + c.volume * c.currentPrice, 0))}
          subtitle="Trading volume"
          icon={
            <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
            </svg>
          }
          color="purple"
        />
      </div>

      {/* Top movers */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
        <div>
          <div className="flex items-center gap-2 mb-4">
            <h2 className="text-xl font-semibold text-zinc-900 dark:text-zinc-100">
              Top Gainers
            </h2>
            <Badge variant="success" size="sm">
              24h
            </Badge>
          </div>
          <StockList stocks={topGainers} isLoading={isLoading} compact />
        </div>
        <div>
          <div className="flex items-center gap-2 mb-4">
            <h2 className="text-xl font-semibold text-zinc-900 dark:text-zinc-100">
              Top Losers
            </h2>
            <Badge variant="danger" size="sm">
              24h
            </Badge>
          </div>
          <StockList stocks={topLosers} isLoading={isLoading} compact />
        </div>
      </div>

      {/* All cryptos */}
      <StockList
        stocks={cryptos}
        isLoading={isLoading}
        showDetails
        title="All Cryptocurrencies"
      />
    </Layout>
  );
}

// Helper function to get display name
function getDisplayName(symbol: string): string {
  const names: Record<string, string> = {
    'BTC': 'Bitcoin',
    'ETH': 'Ethereum',
    'BNB': 'Binance Coin',
    'SOL': 'Solana',
    'XRP': 'Ripple',
    'ADA': 'Cardano',
    'DOGE': 'Dogecoin',
    'MATIC': 'Polygon',
    'DOT': 'Polkadot',
    'AVAX': 'Avalanche',
    'LINK': 'Chainlink',
    'UNI': 'Uniswap',
    'LTC': 'Litecoin',
    'ATOM': 'Cosmos',
    'ETC': 'Ethereum Classic',
    'XLM': 'Stellar',
  };
  return names[symbol] || symbol;
}

// Format large numbers
function formatLargeNumber(num: number): string {
  if (num >= 1e9) return `$${(num / 1e9).toFixed(2)}B`;
  if (num >= 1e6) return `$${(num / 1e6).toFixed(2)}M`;
  if (num >= 1e3) return `$${(num / 1e3).toFixed(2)}K`;
  return `$${num.toFixed(2)}`;
}

// Stat card component
interface StatCardProps {
  title: string;
  value: string;
  subtitle: string;
  icon: React.ReactNode;
  color: 'blue' | 'green' | 'red' | 'yellow' | 'purple' | 'emerald';
}

function StatCard({ title, value, subtitle, icon, color }: StatCardProps) {
  const colorStyles = {
    blue: 'from-blue-500/10 to-blue-500/5 text-blue-600 dark:text-blue-400',
    green: 'from-emerald-500/10 to-emerald-500/5 text-emerald-600 dark:text-emerald-400',
    red: 'from-red-500/10 to-red-500/5 text-red-600 dark:text-red-400',
    yellow: 'from-amber-500/10 to-amber-500/5 text-amber-600 dark:text-amber-400',
    purple: 'from-violet-500/10 to-violet-500/5 text-violet-600 dark:text-violet-400',
    emerald: 'from-emerald-500/10 to-emerald-500/5 text-emerald-600 dark:text-emerald-400',
  };

  return (
    <Card variant="default" padding="md">
      <div className="flex items-start justify-between">
        <div>
          <p className="text-sm text-zinc-500 dark:text-zinc-400 mb-1">{title}</p>
          <p className="text-2xl font-bold text-zinc-900 dark:text-zinc-100">
            {value}
          </p>
          <p className="text-xs text-zinc-400 dark:text-zinc-500 mt-1">
            {subtitle}
          </p>
        </div>
        <div
          className={`p-3 rounded-xl bg-gradient-to-br ${colorStyles[color]}`}
        >
          {icon}
        </div>
      </div>
    </Card>
  );
}
