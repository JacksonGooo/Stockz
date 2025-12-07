/**
 * Indicator Caching Layer
 * Caches computed indicators per symbol to reduce redundant calculations
 *
 * Features:
 * - Per-symbol caching with TTL
 * - Price-change triggered invalidation
 * - Lazy indicator calculation
 * - Batch calculation support
 */

import {
  OHLCV,
  TechnicalIndicators,
  MarketContext,
  MarketRegime,
  calculateAllIndicators,
  indicatorsToFeatures,
  indicatorsToFastFeatures,
  detectMarketRegime,
} from './indicators';

export interface CachedIndicators {
  indicators: TechnicalIndicators;
  regime: MarketRegime;
  features: number[];
  fastFeatures: number[];
  computedAt: number;
  priceAtComputation: number;
  dataLength: number;
}

interface CacheEntry {
  data: CachedIndicators;
  timestamp: number;
}

// Configuration
const CACHE_TTL_MS = 60 * 1000; // 1 minute default TTL
const PRICE_CHANGE_THRESHOLD = 0.005; // 0.5% price change triggers recompute
const MAX_CACHE_SIZE = 100; // Maximum number of symbols to cache

/**
 * Indicator Cache Manager
 */
class IndicatorCacheManager {
  private cache = new Map<string, CacheEntry>();
  private hitCount = 0;
  private missCount = 0;

  /**
   * Get cached indicators for a symbol, recomputing if needed
   */
  getCachedIndicators(
    symbol: string,
    data: OHLCV[],
    marketContext?: MarketContext,
    options: { forceRecompute?: boolean; ttlMs?: number } = {}
  ): CachedIndicators {
    const { forceRecompute = false, ttlMs = CACHE_TTL_MS } = options;

    if (data.length === 0) {
      return this.getEmptyCachedIndicators();
    }

    const currentPrice = data[data.length - 1].close;
    const cacheKey = symbol.toUpperCase();
    const cached = this.cache.get(cacheKey);

    // Check if cache is valid
    if (cached && !forceRecompute) {
      const age = Date.now() - cached.timestamp;
      const priceChange = Math.abs((currentPrice - cached.data.priceAtComputation) / cached.data.priceAtComputation);
      const dataLengthMatch = cached.data.dataLength === data.length;

      // Cache hit conditions:
      // 1. Within TTL
      // 2. Price hasn't changed significantly
      // 3. Data length matches (no new candles)
      if (age < ttlMs && priceChange < PRICE_CHANGE_THRESHOLD && dataLengthMatch) {
        this.hitCount++;
        return cached.data;
      }
    }

    // Cache miss - compute indicators
    this.missCount++;
    const indicators = calculateAllIndicators(data, marketContext);
    const regime = detectMarketRegime(
      data.map(d => d.close),
      data.map(d => d.high),
      data.map(d => d.low)
    );

    const cachedData: CachedIndicators = {
      indicators,
      regime,
      features: indicatorsToFeatures(indicators),
      fastFeatures: indicatorsToFastFeatures(indicators),
      computedAt: Date.now(),
      priceAtComputation: currentPrice,
      dataLength: data.length,
    };

    // Store in cache
    this.cache.set(cacheKey, {
      data: cachedData,
      timestamp: Date.now(),
    });

    // Evict old entries if cache is full
    this.evictIfNeeded();

    return cachedData;
  }

  /**
   * Get cached indicators without data (returns null if not cached or expired)
   */
  getCachedOnly(symbol: string, ttlMs: number = CACHE_TTL_MS): CachedIndicators | null {
    const cacheKey = symbol.toUpperCase();
    const cached = this.cache.get(cacheKey);

    if (!cached) return null;

    const age = Date.now() - cached.timestamp;
    if (age >= ttlMs) return null;

    return cached.data;
  }

  /**
   * Batch calculate indicators for multiple symbols
   */
  async getBatchIndicators(
    symbolDataMap: Map<string, OHLCV[]>,
    marketContext?: MarketContext
  ): Promise<Map<string, CachedIndicators>> {
    const results = new Map<string, CachedIndicators>();

    // Process in parallel
    const entries = Array.from(symbolDataMap.entries());
    const promises = entries.map(async ([symbol, data]) => {
      const indicators = this.getCachedIndicators(symbol, data, marketContext);
      return { symbol, indicators };
    });

    const computed = await Promise.all(promises);

    for (const { symbol, indicators } of computed) {
      results.set(symbol, indicators);
    }

    return results;
  }

  /**
   * Invalidate cache for a specific symbol
   */
  invalidate(symbol: string): void {
    this.cache.delete(symbol.toUpperCase());
  }

  /**
   * Clear entire cache
   */
  clear(): void {
    this.cache.clear();
    this.hitCount = 0;
    this.missCount = 0;
  }

  /**
   * Get cache statistics
   */
  getStats(): {
    size: number;
    hitCount: number;
    missCount: number;
    hitRate: number;
  } {
    const total = this.hitCount + this.missCount;
    return {
      size: this.cache.size,
      hitCount: this.hitCount,
      missCount: this.missCount,
      hitRate: total > 0 ? this.hitCount / total : 0,
    };
  }

  /**
   * Evict old entries if cache exceeds max size
   */
  private evictIfNeeded(): void {
    if (this.cache.size <= MAX_CACHE_SIZE) return;

    // Find and remove oldest entries
    const entries = Array.from(this.cache.entries());
    entries.sort((a, b) => a[1].timestamp - b[1].timestamp);

    const toRemove = entries.slice(0, entries.length - MAX_CACHE_SIZE);
    for (const [key] of toRemove) {
      this.cache.delete(key);
    }
  }

  /**
   * Get empty cached indicators for edge cases
   */
  private getEmptyCachedIndicators(): CachedIndicators {
    const emptyIndicators: TechnicalIndicators = {
      sma5: 0, sma10: 0, sma20: 0, sma50: 0, sma200: 0,
      ema9: 0, ema12: 0, ema21: 0, ema26: 0,
      rsi: 50, rsi_sma: 50,
      macd: 0, macdSignal: 0, macdHistogram: 0,
      stochK: 50, stochD: 50,
      williamR: -50,
      roc: 0, roc5: 0,
      momentum: 0,
      cci: 0, mfi: 50,
      adx: 25, plusDI: 25, minusDI: 25,
      bollingerUpper: 0, bollingerMiddle: 0, bollingerLower: 0,
      bollingerWidth: 0, bollingerPercentB: 0.5,
      atr: 0, atrPercent: 0,
      volatility: 0,
      keltnerUpper: 0, keltnerLower: 0,
      donchianHigh: 0, donchianLow: 0,
      volumeSma: 0, volumeRatio: 1,
      obv: 0, obvSlope: 0,
      vwap: 0, cmf: 0, adLine: 0,
      priceChange: 0, priceChangePercent: 0,
      highLowRange: 0, bodySize: 0,
      upperShadow: 0, lowerShadow: 0,
      gapUp: 0, gapDown: 0,
      distanceFromHigh20: 0, distanceFromLow20: 0,
      distanceFromHigh52: 0, distanceFromLow52: 0,
      pivotPoint: 0, support1: 0, resistance1: 0,
      trendStrength: 0, pricePositionInRange: 0.5,
      consecutiveUpDays: 0, consecutiveDownDays: 0,
      shortTermTrend: 0, mediumTermTrend: 0, longTermTrend: 0,
      normalized: {
        price: 0.5, volume: 0.5,
        rsi: 0.5, macd: 0, stochastic: 0.5, momentum: 0, cci: 0, mfi: 0.5,
        bollinger: 0.5, atr: 0, volatility: 0,
        shortTrend: 0, mediumTrend: 0, longTrend: 0, trendAlignment: 0,
        volumeFlow: 0, moneyFlow: 0,
        pricePosition: 0.5, distanceFromResistance: 0, distanceFromSupport: 0,
        candlePattern: 0, gapStrength: 0,
        marketSentiment: 0, sectorStrength: 0, breadth: 0,
      },
    };

    const emptyRegime: MarketRegime = {
      regime: 'ranging',
      regimeStrength: 0.5,
      regimeDuration: 0,
      volatilityCluster: 1,
      hurstExponent: 0.5,
      features: {
        regimeEncoded: 0,
        regimeStrengthNorm: 0.5,
        volatilityClusterNorm: 0.5,
        hurstNorm: 0.5,
      },
    };

    return {
      indicators: emptyIndicators,
      regime: emptyRegime,
      features: new Array(30).fill(0),
      fastFeatures: new Array(15).fill(0),
      computedAt: Date.now(),
      priceAtComputation: 0,
      dataLength: 0,
    };
  }
}

// Singleton instance
let cacheInstance: IndicatorCacheManager | null = null;

/**
 * Get the indicator cache singleton
 */
export function getIndicatorCache(): IndicatorCacheManager {
  if (!cacheInstance) {
    cacheInstance = new IndicatorCacheManager();
  }
  return cacheInstance;
}

/**
 * Reset the indicator cache (for testing)
 */
export function resetIndicatorCache(): void {
  if (cacheInstance) {
    cacheInstance.clear();
  }
  cacheInstance = null;
}

// Export the class for testing
export { IndicatorCacheManager };
