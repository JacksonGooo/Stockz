'use client';

import { useState, useEffect, useCallback, useRef } from 'react';

interface UseCyclingFetchOptions<T> {
  symbols: string[];
  fetchFn: (symbols: string[]) => Promise<T[]>;
  cycleSize?: number;
  intervalMs?: number;
  getSymbol: (item: T) => string;
  enabled?: boolean;
}

interface UseCyclingFetchResult<T> {
  data: T[];
  isLive: boolean;
  setIsLive: (live: boolean) => void;
  currentSymbols: string[];
  lastUpdated: Date | null;
  isLoading: boolean;
}

export function useCyclingFetch<T>({
  symbols,
  fetchFn,
  cycleSize = 3,
  intervalMs = 3000,
  getSymbol,
  enabled = true,
}: UseCyclingFetchOptions<T>): UseCyclingFetchResult<T> {
  const [data, setData] = useState<T[]>([]);
  const [isLive, setIsLive] = useState(true);
  const [cycleIndex, setCycleIndex] = useState(0);
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [currentSymbols, setCurrentSymbols] = useState<string[]>([]);
  const dataRef = useRef<T[]>([]);

  // Keep dataRef in sync
  useEffect(() => {
    dataRef.current = data;
  }, [data]);

  // Calculate which symbols to fetch in current cycle
  const getSymbolsForCycle = useCallback((index: number) => {
    if (symbols.length === 0) return [];
    const totalCycles = Math.ceil(symbols.length / cycleSize);
    const normalizedIndex = index % totalCycles;
    const start = normalizedIndex * cycleSize;
    return symbols.slice(start, start + cycleSize);
  }, [symbols, cycleSize]);

  // Initial fetch - get all data first
  useEffect(() => {
    if (!enabled || symbols.length === 0) return;

    async function initialFetch() {
      setIsLoading(true);
      try {
        const result = await fetchFn(symbols);
        setData(result);
        dataRef.current = result;
        setLastUpdated(new Date());
      } catch (error) {
        console.error('Initial fetch error:', error);
      } finally {
        setIsLoading(false);
      }
    }

    initialFetch();
  }, [symbols, fetchFn, enabled]);

  // Cycling fetch - optimized to reduce re-renders
  useEffect(() => {
    if (!enabled || !isLive || symbols.length === 0 || isLoading) return;

    const interval = setInterval(async () => {
      const symbolsToFetch = getSymbolsForCycle(cycleIndex);

      try {
        const newData = await fetchFn(symbolsToFetch);

        // Batch all state updates together to reduce re-renders
        // Use functional updates to ensure we're working with latest state
        setData(prev => {
          const updated = [...prev];
          newData.forEach(item => {
            const symbol = getSymbol(item);
            const existingIndex = updated.findIndex(d => getSymbol(d) === symbol);
            if (existingIndex >= 0) {
              updated[existingIndex] = item;
            } else {
              updated.push(item);
            }
          });
          return updated;
        });

        // Only update these if they actually changed
        setCurrentSymbols(prev => {
          if (prev.join(',') === symbolsToFetch.join(',')) return prev;
          return symbolsToFetch;
        });
        setLastUpdated(new Date());
        setCycleIndex(prev => prev + 1);
      } catch (error) {
        console.error('Cycle fetch error:', error);
      }
    }, intervalMs);

    return () => clearInterval(interval);
  }, [enabled, isLive, symbols, cycleIndex, intervalMs, fetchFn, getSymbol, getSymbolsForCycle, isLoading]);

  return {
    data,
    isLive,
    setIsLive,
    currentSymbols,
    lastUpdated,
    isLoading,
  };
}

// Hook for focused single-item fetching (detail pages)
interface UseFocusedFetchOptions<T> {
  symbol: string;
  fetchFn: (symbol: string) => Promise<T | null>;
  intervalMs?: number;
  enabled?: boolean;
  onPriceChange?: (direction: 'up' | 'down') => void;
  getPrice?: (item: T) => number;
}

interface UseFocusedFetchResult<T> {
  data: T | null;
  isLive: boolean;
  setIsLive: (live: boolean) => void;
  lastUpdated: Date | null;
  isLoading: boolean;
}

export function useFocusedFetch<T>({
  symbol,
  fetchFn,
  intervalMs = 2000,
  enabled = true,
  onPriceChange,
  getPrice,
}: UseFocusedFetchOptions<T>): UseFocusedFetchResult<T> {
  const [data, setData] = useState<T | null>(null);
  const [isLive, setIsLive] = useState(true);
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const prevPriceRef = useRef<number | null>(null);

  // Initial fetch with retry
  useEffect(() => {
    if (!enabled || !symbol) return;

    let isMounted = true;
    let retryCount = 0;
    const maxRetries = 3;

    async function initialFetch() {
      setIsLoading(true);
      try {
        const result = await fetchFn(symbol);
        if (!isMounted) return;
        setData(result);
        if (result && getPrice) {
          prevPriceRef.current = getPrice(result);
        }
        setLastUpdated(new Date());
        retryCount = 0; // Reset on success
      } catch (error) {
        if (!isMounted) return;
        // Silently retry on network errors (common during dev hot reload)
        if (retryCount < maxRetries) {
          retryCount++;
          setTimeout(initialFetch, 1000 * retryCount);
          return;
        }
        // Only log after all retries failed
        console.warn('Fetch failed after retries:', symbol);
      } finally {
        if (isMounted) {
          setIsLoading(false);
        }
      }
    }

    initialFetch();

    return () => {
      isMounted = false;
    };
  }, [symbol, fetchFn, enabled, getPrice]);

  // Focused polling
  useEffect(() => {
    if (!enabled || !isLive || !symbol || isLoading) return;

    const interval = setInterval(async () => {
      try {
        const result = await fetchFn(symbol);

        if (result && getPrice && onPriceChange) {
          const newPrice = getPrice(result);
          if (prevPriceRef.current !== null && newPrice !== prevPriceRef.current) {
            onPriceChange(newPrice > prevPriceRef.current ? 'up' : 'down');
          }
          prevPriceRef.current = newPrice;
        }

        setData(result);
        setLastUpdated(new Date());
      } catch (error) {
        console.error('Focused fetch error:', error);
      }
    }, intervalMs);

    return () => clearInterval(interval);
  }, [enabled, isLive, symbol, intervalMs, fetchFn, getPrice, onPriceChange, isLoading]);

  return {
    data,
    isLive,
    setIsLive,
    lastUpdated,
    isLoading,
  };
}
