import { NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';

// Server-side caching for performance
let cachedData: { categories: CategoryInfo[]; totalCandles: number; totalSize: number; lastScan: string } | null = null;
let cacheTimestamp = 0;
const CACHE_TTL = 30000; // 30 seconds

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

// Check if market is currently open
function isMarketOpen(category: string): boolean {
  const now = new Date();
  const day = now.getDay(); // 0 = Sunday, 6 = Saturday
  const hour = now.getUTCHours();

  if (category === 'Crypto') {
    return true; // Crypto is 24/7
  }

  // Stock markets: Mon-Fri, roughly 9:30 AM - 4 PM ET (14:30 - 21:00 UTC)
  if (category === 'Stock Market') {
    if (day === 0 || day === 6) return false; // Weekend
    return hour >= 14 && hour < 21;
  }

  // Commodities: Similar to stocks but some trade longer
  if (category === 'Commodities') {
    if (day === 0 || day === 6) return false;
    return hour >= 14 && hour < 21;
  }

  // Forex: Sun 5pm ET - Fri 5pm ET (mostly 24/5)
  if (category === 'Currencies') {
    if (day === 0) return hour >= 22; // Sunday opens late
    if (day === 6) return false; // Saturday closed
    if (day === 5) return hour < 22; // Friday closes early
    return true; // Mon-Thu 24h
  }

  return false;
}

// Calculate expected candles based on category and number of days
function getExpectedCandles(category: string, totalDays: number): number {
  switch (category) {
    case 'Crypto':
      return totalDays * 1440; // 24h * 60min
    case 'Stock Market':
    case 'Commodities':
      // Roughly 5/7 of days are trading days, 390 min per day (6.5 hours)
      return Math.floor(totalDays * (5 / 7) * 390);
    case 'Currencies':
      // 5/7 days, ~23 hours per day
      return Math.floor(totalDays * (5 / 7) * 1380);
    default:
      return totalDays * 1440;
  }
}

interface CategoryInfo {
  name: string;
  assets: AssetInfo[];
  totalCandles: number;
  totalSize: number;
}

// Fast candle estimation based on file size (avoids reading/parsing every file)
// Average candle JSON is ~80-100 bytes, use 85 as estimate
function estimateCandlesFromSize(sizeBytes: number): number {
  return Math.floor(sizeBytes / 85);
}

function getAssetInfo(dataDir: string, category: string, asset: string): AssetInfo {
  const assetPath = path.join(dataDir, category, asset);

  if (!fs.existsSync(assetPath)) {
    return {
      name: asset,
      category,
      totalCandles: 0,
      totalDays: 0,
      dateRange: null,
      lastUpdated: null,
      sizeBytes: 0,
      status: 'empty',
      hasProcessedData: false,
    };
  }

  let totalCandles = 0;
  let totalSize = 0;
  const dates: string[] = [];
  let lastModified: Date | null = null;

  // Scan week folders
  const weekFolders = fs.readdirSync(assetPath)
    .filter(f => f.startsWith('week_'))
    .sort();

  for (const weekFolder of weekFolders) {
    const weekPath = path.join(assetPath, weekFolder);
    const stat = fs.statSync(weekPath);

    if (stat.isDirectory()) {
      const dayFiles = fs.readdirSync(weekPath)
        .filter(f => f.endsWith('.json'))
        .sort();

      for (const dayFile of dayFiles) {
        const dayPath = path.join(weekPath, dayFile);
        const dayStat = fs.statSync(dayPath);

        totalSize += dayStat.size;
        dates.push(dayFile.replace('.json', ''));

        if (!lastModified || dayStat.mtime > lastModified) {
          lastModified = dayStat.mtime;
        }

        // Estimate candles from file size (much faster than parsing)
        totalCandles += estimateCandlesFromSize(dayStat.size);
      }
    }
  }

  // Check for processed data
  const processedPath = path.join(assetPath, 'processed');
  const masterFile = path.join(assetPath, `${asset.toLowerCase()}_master.pkl.gz`);
  const hasProcessedData = fs.existsSync(processedPath) || fs.existsSync(masterFile);

  if (hasProcessedData && fs.existsSync(masterFile)) {
    totalSize += fs.statSync(masterFile).size;
  }

  // Calculate completeness percentage first (needed for status)
  const expectedCandles = getExpectedCandles(category, dates.length);
  const completenessPercentage = expectedCandles > 0
    ? Math.min(100, Math.round((totalCandles / expectedCandles) * 100))
    : 0;

  // Determine status based on market hours, data freshness, and completeness
  let status: AssetInfo['status'] = 'empty';

  if (totalCandles > 0) {
    const marketOpen = isMarketOpen(category);
    const now = new Date();
    const isRecent = lastModified && (now.getTime() - lastModified.getTime()) < 5 * 60 * 1000; // Within 5 minutes
    // For crypto: consider "live" if data was updated today (24/7 market)
    const todayStr = now.toISOString().split('T')[0];
    const hasTodayData = dates.length > 0 && dates[dates.length - 1] >= todayStr;

    // Check if data is complete (>= 95% for days with data)
    if (completenessPercentage >= 95) {
      status = 'complete';
    } else if (marketOpen && isRecent) {
      status = 'live';
    } else if (category === 'Crypto' && hasTodayData) {
      // Crypto is 24/7, show as live if we have today's data
      status = 'live';
    } else if (marketOpen) {
      status = 'ready';
    } else {
      status = 'sleeping';
    }
  }

  return {
    name: asset,
    category,
    totalCandles,
    totalDays: dates.length,
    dateRange: dates.length > 0 ? { start: dates[0], end: dates[dates.length - 1] } : null,
    lastUpdated: lastModified?.toISOString() || null,
    sizeBytes: totalSize,
    status,
    hasProcessedData,
    completenessPercentage,
  };
}

export async function GET() {
  try {
    // Return cached data if still fresh (within 30 seconds)
    const now = Date.now();
    if (cachedData && (now - cacheTimestamp) < CACHE_TTL) {
      return NextResponse.json(cachedData, {
        headers: {
          'Cache-Control': 'public, max-age=30',
          'X-Cache': 'HIT',
        },
      });
    }

    const dataDir = path.join(process.cwd(), 'Data');

    if (!fs.existsSync(dataDir)) {
      return NextResponse.json({ categories: [], totalCandles: 0, totalSize: 0 });
    }

    const categories: CategoryInfo[] = [];
    let grandTotalCandles = 0;
    let grandTotalSize = 0;

    // Define ONLY the core assets to display (fast loading)
    // This prevents scanning 200+ assets on disk
    const expectedStructure: Record<string, string[]> = {
      'Crypto': ['BTC', 'ETH', 'SOL', 'XRP', 'DOGE', 'ADA', 'LINK', 'LTC'],
      'Stock Market': ['SPY', 'QQQ', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA'],
      'Commodities': ['GLD', 'SLV', 'USO', 'UNG'],
      'Currencies': ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCAD'],
    };

    for (const [categoryName, expectedAssets] of Object.entries(expectedStructure)) {
      const assets: AssetInfo[] = [];
      let categoryCandles = 0;
      let categorySize = 0;

      // ONLY load defined assets (not all existing ones on disk)
      for (const asset of expectedAssets) {
        const info = getAssetInfo(dataDir, categoryName, asset);
        assets.push(info);
        categoryCandles += info.totalCandles;
        categorySize += info.sizeBytes;
      }

      // Sort by candles (most data first)
      assets.sort((a, b) => b.totalCandles - a.totalCandles);

      categories.push({
        name: categoryName,
        assets,
        totalCandles: categoryCandles,
        totalSize: categorySize,
      });

      grandTotalCandles += categoryCandles;
      grandTotalSize += categorySize;
    }

    // Cache the result
    const responseData = {
      categories,
      totalCandles: grandTotalCandles,
      totalSize: grandTotalSize,
      lastScan: new Date().toISOString(),
    };

    cachedData = responseData;
    cacheTimestamp = Date.now();

    return NextResponse.json(responseData, {
      headers: {
        'Cache-Control': 'public, max-age=30',
        'X-Cache': 'MISS',
      },
    });
  } catch (error) {
    console.error('Error scanning data:', error);
    return NextResponse.json({ error: 'Failed to scan data' }, { status: 500 });
  }
}
