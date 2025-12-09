'use client';

import { useState, useEffect } from 'react';
import { Layout } from '@/components/layout';
import { Card, Badge, Input } from '@/components/ui';
import { StockList } from '@/components/stocks';
import { stockService, Stock } from '@/ai';

const SECTORS = [
  'All',
  'Technology',
  'Consumer Cyclical',
  'Financial Services',
  'Healthcare',
  'Energy',
];

export default function MarketsPage() {
  const [stocks, setStocks] = useState<Stock[]>([]);
  const [filteredStocks, setFilteredStocks] = useState<Stock[]>([]);
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedSector, setSelectedSector] = useState('All');
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    async function fetchStocks() {
      try {
        const allStocks = await stockService.getAllStocks();
        setStocks(allStocks);
        setFilteredStocks(allStocks);
      } catch (error) {
        console.error('Failed to fetch stocks:', error);
      } finally {
        setIsLoading(false);
      }
    }
    fetchStocks();
  }, []);

  useEffect(() => {
    let result = stocks;

    // Filter by sector
    if (selectedSector !== 'All') {
      result = result.filter((s) => s.sector === selectedSector);
    }

    // Filter by search query
    if (searchQuery) {
      const query = searchQuery.toLowerCase();
      result = result.filter(
        (s) =>
          s.symbol.toLowerCase().includes(query) ||
          s.name.toLowerCase().includes(query)
      );
    }

    setFilteredStocks(result);
  }, [stocks, selectedSector, searchQuery]);

  // Calculate market stats
  const marketStats = {
    totalMarketCap: stocks.reduce((sum, s) => sum + s.marketCap, 0),
    avgChange: stocks.length
      ? stocks.reduce((sum, s) => sum + s.changePercent, 0) / stocks.length
      : 0,
    gainers: stocks.filter((s) => s.change > 0).length,
    losers: stocks.filter((s) => s.change < 0).length,
  };

  return (
    <Layout>
      {/* Page header */}
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-zinc-900 dark:text-zinc-100 mb-2">
          Markets
        </h1>
        <p className="text-zinc-500 dark:text-zinc-400">
          Browse and analyze stocks across different sectors
        </p>
      </div>

      {/* Market overview */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
        <Card>
          <p className="text-sm text-zinc-500 dark:text-zinc-400 mb-1">Total Market Cap</p>
          <p className="text-xl font-bold text-zinc-900 dark:text-zinc-100">
            ${(marketStats.totalMarketCap / 1e12).toFixed(2)}T
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

      {/* Filters */}
      <Card className="mb-8">
        <div className="flex flex-col md:flex-row gap-4">
          <div className="flex-1">
            <Input
              placeholder="Search stocks..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              leftIcon={
                <svg
                  className="w-5 h-5"
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"
                  />
                </svg>
              }
            />
          </div>
          <div className="flex gap-2 flex-wrap">
            {SECTORS.map((sector) => (
              <button
                key={sector}
                onClick={() => setSelectedSector(sector)}
                className={`
                  px-4 py-2 rounded-xl text-sm font-medium transition-all duration-200
                  ${
                    selectedSector === sector
                      ? 'bg-blue-600 text-white'
                      : 'bg-zinc-100 dark:bg-zinc-800 text-zinc-600 dark:text-zinc-400 hover:bg-zinc-200 dark:hover:bg-zinc-700'
                  }
                `}
              >
                {sector}
              </button>
            ))}
          </div>
        </div>
      </Card>

      {/* Results count */}
      <div className="flex items-center justify-between mb-4">
        <p className="text-sm text-zinc-500 dark:text-zinc-400">
          Showing {filteredStocks.length} of {stocks.length} stocks
        </p>
        {selectedSector !== 'All' && (
          <Badge variant="info" size="sm">
            {selectedSector}
          </Badge>
        )}
      </div>

      {/* Stock list */}
      <StockList stocks={filteredStocks} isLoading={isLoading} showDetails />
    </Layout>
  );
}
