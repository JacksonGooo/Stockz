'use client';

import { Stock } from '@/ai/types';
import { StockCard } from './StockCard';
import { StockCardSkeleton } from '../ui/Skeleton';

interface StockListProps {
  stocks: Stock[];
  isLoading?: boolean;
  showDetails?: boolean;
  compact?: boolean;
  title?: string;
}

export function StockList({
  stocks,
  isLoading = false,
  showDetails = false,
  compact = false,
  title,
}: StockListProps) {
  if (isLoading) {
    return (
      <div>
        {title && (
          <h2 className="text-xl font-semibold text-zinc-900 dark:text-zinc-100 mb-4">
            {title}
          </h2>
        )}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {Array.from({ length: 6 }).map((_, i) => (
            <StockCardSkeleton key={i} />
          ))}
        </div>
      </div>
    );
  }

  if (stocks.length === 0) {
    return (
      <div className="text-center py-12">
        <svg
          className="w-12 h-12 mx-auto text-zinc-400 dark:text-zinc-600 mb-4"
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={1.5}
            d="M9.172 16.172a4 4 0 015.656 0M9 10h.01M15 10h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
          />
        </svg>
        <p className="text-zinc-500 dark:text-zinc-400">No stocks found</p>
      </div>
    );
  }

  return (
    <div>
      {title && (
        <h2 className="text-xl font-semibold text-zinc-900 dark:text-zinc-100 mb-4">
          {title}
        </h2>
      )}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {stocks.map((stock) => (
          <StockCard key={stock.symbol} stock={stock} showDetails={showDetails} compact={compact} />
        ))}
      </div>
    </div>
  );
}
