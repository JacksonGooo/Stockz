'use client';

import { useState, useEffect, useMemo, useCallback } from 'react';
import { Layout } from '@/components/layout';
import { Card, Button, Badge, Input } from '@/components/ui';
import { StockCard } from '@/components/stocks';
import { stockService, Stock, WatchlistItem } from '@/ai';

export default function WatchlistPage() {
  const [watchlist, setWatchlist] = useState<WatchlistItem[]>([]);
  const [allStocks, setAllStocks] = useState<Stock[]>([]);
  const [searchQuery, setSearchQuery] = useState('');
  const [showAddModal, setShowAddModal] = useState(false);

  useEffect(() => {
    setWatchlist(stockService.getWatchlist());
    stockService.getAllStocks().then(setAllStocks);
  }, []);

  const handleRemove = useCallback((symbol: string) => {
    stockService.removeFromWatchlist(symbol);
    setWatchlist(stockService.getWatchlist());
  }, []);

  const handleAdd = useCallback(async (symbol: string) => {
    await stockService.addToWatchlist(symbol);
    setWatchlist(stockService.getWatchlist());
    setShowAddModal(false);
    setSearchQuery('');
  }, []);

  // Memoize filtered stocks to prevent recalculation on every render
  const filteredStocks = useMemo(() =>
    allStocks.filter(
      (s) =>
        !watchlist.some((w) => w.stock.symbol === s.symbol) &&
        (s.symbol.toLowerCase().includes(searchQuery.toLowerCase()) ||
          s.name.toLowerCase().includes(searchQuery.toLowerCase()))
    ), [allStocks, watchlist, searchQuery]);

  // Memoize totals calculation
  const { totalValue, totalChange } = useMemo(() => ({
    totalValue: watchlist.reduce((sum, w) => sum + w.stock.currentPrice, 0),
    totalChange: watchlist.reduce((sum, w) => sum + w.stock.change, 0),
  }), [watchlist]);

  return (
    <Layout>
      {/* Page header */}
      <div className="flex items-start justify-between mb-8">
        <div>
          <h1 className="text-3xl font-bold text-zinc-900 dark:text-zinc-100 mb-2">
            Watchlist
          </h1>
          <p className="text-zinc-500 dark:text-zinc-400">
            Track your favorite stocks and get alerts
          </p>
        </div>
        <Button
          variant="primary"
          onClick={() => setShowAddModal(true)}
          leftIcon={
            <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6v6m0 0v6m0-6h6m-6 0H6" />
            </svg>
          }
        >
          Add Stock
        </Button>
      </div>

      {/* Watchlist stats */}
      {watchlist.length > 0 && (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
          <Card>
            <p className="text-sm text-zinc-500 dark:text-zinc-400 mb-1">Stocks Watching</p>
            <p className="text-2xl font-bold text-zinc-900 dark:text-zinc-100">
              {watchlist.length}
            </p>
          </Card>
          <Card>
            <p className="text-sm text-zinc-500 dark:text-zinc-400 mb-1">Combined Price</p>
            <p className="text-2xl font-bold text-zinc-900 dark:text-zinc-100 font-mono">
              ${totalValue.toFixed(2)}
            </p>
          </Card>
          <Card>
            <p className="text-sm text-zinc-500 dark:text-zinc-400 mb-1">Total Change</p>
            <p
              className={`text-2xl font-bold font-mono ${
                totalChange >= 0
                  ? 'text-emerald-600 dark:text-emerald-400'
                  : 'text-red-600 dark:text-red-400'
              }`}
            >
              {totalChange >= 0 ? '+' : ''}${totalChange.toFixed(2)}
            </p>
          </Card>
        </div>
      )}

      {/* Watchlist items */}
      {watchlist.length > 0 ? (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {watchlist.map((item) => (
            <div key={item.stock.symbol} className="relative group">
              <StockCard stock={item.stock} showDetails />
              <button
                onClick={() => handleRemove(item.stock.symbol)}
                className="absolute top-4 right-4 p-2 rounded-lg bg-red-100 dark:bg-red-900/30 text-red-600 dark:text-red-400 opacity-0 group-hover:opacity-100 transition-opacity"
              >
                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
              <p className="text-xs text-zinc-400 mt-2">
                Added {new Date(item.addedAt).toLocaleDateString()}
              </p>
            </div>
          ))}
        </div>
      ) : (
        <Card className="py-16 text-center">
          <svg
            className="w-16 h-16 mx-auto text-zinc-400 mb-4"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={1.5}
              d="M11.049 2.927c.3-.921 1.603-.921 1.902 0l1.519 4.674a1 1 0 00.95.69h4.915c.969 0 1.371 1.24.588 1.81l-3.976 2.888a1 1 0 00-.363 1.118l1.518 4.674c.3.922-.755 1.688-1.538 1.118l-3.976-2.888a1 1 0 00-1.176 0l-3.976 2.888c-.783.57-1.838-.197-1.538-1.118l1.518-4.674a1 1 0 00-.363-1.118l-3.976-2.888c-.784-.57-.38-1.81.588-1.81h4.914a1 1 0 00.951-.69l1.519-4.674z"
            />
          </svg>
          <h3 className="text-lg font-semibold text-zinc-900 dark:text-zinc-100 mb-2">
            Your Watchlist is Empty
          </h3>
          <p className="text-zinc-500 dark:text-zinc-400 mb-4">
            Add stocks to your watchlist to track them easily
          </p>
          <Button variant="primary" onClick={() => setShowAddModal(true)}>
            Add Your First Stock
          </Button>
        </Card>
      )}

      {/* Add stock modal */}
      {showAddModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
          <div
            className="absolute inset-0 bg-black/50 backdrop-blur-sm"
            onClick={() => setShowAddModal(false)}
          />
          <Card className="relative w-full max-w-md max-h-[80vh] overflow-hidden">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-xl font-semibold text-zinc-900 dark:text-zinc-100">
                Add to Watchlist
              </h2>
              <button
                onClick={() => setShowAddModal(false)}
                className="p-2 rounded-lg hover:bg-zinc-100 dark:hover:bg-zinc-800"
              >
                <svg className="w-5 h-5 text-zinc-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>

            <Input
              placeholder="Search stocks..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="mb-4"
            />

            <div className="max-h-64 overflow-y-auto space-y-2">
              {filteredStocks.map((stock) => (
                <button
                  key={stock.symbol}
                  onClick={() => handleAdd(stock.symbol)}
                  className="w-full flex items-center justify-between p-3 rounded-xl hover:bg-zinc-100 dark:hover:bg-zinc-800 transition-colors text-left"
                >
                  <div>
                    <p className="font-medium text-zinc-900 dark:text-zinc-100">
                      {stock.symbol}
                    </p>
                    <p className="text-sm text-zinc-500 dark:text-zinc-400">
                      {stock.name}
                    </p>
                  </div>
                  <Badge variant={stock.change >= 0 ? 'success' : 'danger'} size="sm">
                    {stock.change >= 0 ? '+' : ''}{stock.changePercent.toFixed(2)}%
                  </Badge>
                </button>
              ))}
              {filteredStocks.length === 0 && (
                <p className="text-center text-zinc-500 py-4">No stocks found</p>
              )}
            </div>
          </Card>
        </div>
      )}
    </Layout>
  );
}
