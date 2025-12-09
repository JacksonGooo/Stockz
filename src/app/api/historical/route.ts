import { NextRequest, NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';

interface Candle {
  timestamp: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

interface BufferData {
  recentCandles: Candle[];
  lastPrice: number;
  lastTimestamp: number;
  dayCount: number;
}

// Load recent candles from realtime buffer
function loadRealtimeBuffer(category: string, asset: string): Candle[] {
  const bufferFile = path.join(process.cwd(), 'Data', '.buffer', 'realtime.json');

  if (!fs.existsSync(bufferFile)) {
    return [];
  }

  try {
    const bufferContent = fs.readFileSync(bufferFile, 'utf8');
    const allBuffers: Record<string, BufferData> = JSON.parse(bufferContent);
    const key = `${category}/${asset}`;

    if (allBuffers[key] && allBuffers[key].recentCandles) {
      return allBuffers[key].recentCandles;
    }
  } catch {
    // Ignore errors
  }

  return [];
}

// Merge and deduplicate candles by timestamp
function mergeCandles(historical: Candle[], realtime: Candle[]): Candle[] {
  const map = new Map<number, Candle>();

  // Add historical first
  for (const c of historical) {
    map.set(c.timestamp, c);
  }

  // Realtime overwrites (more recent)
  for (const c of realtime) {
    map.set(c.timestamp, c);
  }

  // Sort by timestamp
  return Array.from(map.values()).sort((a, b) => a.timestamp - b.timestamp);
}

// Load historical candles for an asset by count (most recent)
function loadHistoricalCandles(category: string, asset: string, count: number = 60): Candle[] {
  const dataDir = path.join(process.cwd(), 'Data');
  const assetDir = path.join(dataDir, category, asset);

  if (!fs.existsSync(assetDir)) {
    return [];
  }

  const allCandles: Candle[] = [];
  const weekDirs = fs.readdirSync(assetDir)
    .filter(d => d.startsWith('week_'))
    .sort()
    .reverse();

  for (const weekDir of weekDirs) {
    const weekPath = path.join(assetDir, weekDir);
    const files = fs.readdirSync(weekPath)
      .filter(f => f.endsWith('.json'))
      .sort()
      .reverse();

    for (const file of files) {
      try {
        const data = JSON.parse(fs.readFileSync(path.join(weekPath, file), 'utf8'));
        // Data is stored oldest first, so reverse it
        allCandles.push(...data.reverse());

        if (allCandles.length >= count) {
          break;
        }
      } catch {
        // Skip invalid files
      }
    }

    if (allCandles.length >= count) {
      break;
    }
  }

  // Return most recent candles, oldest first for charting
  return allCandles.slice(0, count).reverse();
}

// Load historical candles for an asset by date/time range
function loadCandlesByDateRange(
  category: string,
  asset: string,
  startDate: string,
  startTime: string,
  endDate: string,
  endTime: string
): Candle[] {
  const dataDir = path.join(process.cwd(), 'Data');
  const assetDir = path.join(dataDir, category, asset);

  if (!fs.existsSync(assetDir)) {
    return [];
  }

  // Parse start and end timestamps
  const startTimestamp = new Date(`${startDate}T${startTime}:00`).getTime();
  const endTimestamp = new Date(`${endDate}T${endTime}:59`).getTime();

  const allCandles: Candle[] = [];
  const weekDirs = fs.readdirSync(assetDir)
    .filter(d => d.startsWith('week_'))
    .sort();

  // Get the date range we need to check
  const startDateObj = new Date(startDate);
  const endDateObj = new Date(endDate);

  for (const weekDir of weekDirs) {
    const weekPath = path.join(assetDir, weekDir);
    const files = fs.readdirSync(weekPath)
      .filter(f => f.endsWith('.json'))
      .sort();

    for (const file of files) {
      // Check if this file's date is within our range
      const fileDate = file.replace('.json', '');
      const fileDateObj = new Date(fileDate);

      // Skip files outside our date range
      if (fileDateObj < startDateObj || fileDateObj > endDateObj) {
        continue;
      }

      try {
        const data: Candle[] = JSON.parse(fs.readFileSync(path.join(weekPath, file), 'utf8'));

        // Filter candles by timestamp
        for (const candle of data) {
          if (candle.timestamp >= startTimestamp && candle.timestamp <= endTimestamp) {
            allCandles.push(candle);
          }
        }
      } catch {
        // Skip invalid files
      }
    }
  }

  // Sort by timestamp (oldest first for charting)
  allCandles.sort((a, b) => a.timestamp - b.timestamp);

  return allCandles;
}

// GET handler - fetch historical candles
export async function GET(request: NextRequest) {
  const searchParams = request.nextUrl.searchParams;
  const category = searchParams.get('category');
  const asset = searchParams.get('asset');
  const timeframe = searchParams.get('timeframe') || '1h';

  // Custom date/time range params
  const startDate = searchParams.get('startDate');
  const startTime = searchParams.get('startTime');
  const endDate = searchParams.get('endDate');
  const endTime = searchParams.get('endTime');

  if (!category || !asset) {
    return NextResponse.json({
      error: 'Missing parameters',
      message: 'category and asset are required',
    }, { status: 400 });
  }

  let candles: Candle[];

  // Load realtime buffer data
  const realtimeCandles = loadRealtimeBuffer(category, asset);

  // Check if custom date range is requested
  if (startDate && startTime && endDate && endTime) {
    const historical = loadCandlesByDateRange(category, asset, startDate, startTime, endDate, endTime);
    candles = mergeCandles(historical, realtimeCandles);
  } else {
    // Map timeframe to candle count (1-minute candles)
    const timeframeMap: Record<string, number> = {
      '30m': 30,
      '1h': 60,
      '4h': 240,
      '1d': 1440,
      '1w': 10080,
      '1m': 43200, // 30 days
    };

    const count = timeframeMap[timeframe] || 60;
    const historical = loadHistoricalCandles(category, asset, count);
    // Merge historical with realtime and take the most recent 'count' candles
    const merged = mergeCandles(historical, realtimeCandles);
    candles = merged.slice(-count);
  }

  if (candles.length === 0) {
    return NextResponse.json({
      error: 'No data',
      message: `No historical data found for ${category}/${asset}`,
    }, { status: 404 });
  }

  // Calculate summary stats
  const closes = candles.map(c => c.close);
  const currentPrice = closes[closes.length - 1];
  const openPrice = closes[0];
  const highPrice = Math.max(...candles.map(c => c.high));
  const lowPrice = Math.min(...candles.map(c => c.low));
  const priceChange = currentPrice - openPrice;
  const percentChange = ((currentPrice - openPrice) / openPrice) * 100;

  return NextResponse.json({
    asset,
    category,
    timeframe,
    candles,
    summary: {
      currentPrice,
      openPrice,
      highPrice,
      lowPrice,
      priceChange,
      percentChange,
      candleCount: candles.length,
    },
  });
}

// POST handler - list available assets with data
export async function POST() {
  const dataDir = path.join(process.cwd(), 'Data');
  const assets: { category: string; asset: string; candleCount: number }[] = [];

  const categories = ['Crypto', 'Stock Market', 'Commodities', 'Currencies'];

  for (const category of categories) {
    const categoryPath = path.join(dataDir, category);
    if (!fs.existsSync(categoryPath)) continue;

    const assetDirs = fs.readdirSync(categoryPath).filter(f =>
      fs.statSync(path.join(categoryPath, f)).isDirectory()
    );

    for (const asset of assetDirs) {
      // Count candles
      let candleCount = 0;
      const assetPath = path.join(categoryPath, asset);
      const weekDirs = fs.readdirSync(assetPath).filter(d => d.startsWith('week_'));

      for (const weekDir of weekDirs) {
        const weekPath = path.join(assetPath, weekDir);
        const files = fs.readdirSync(weekPath).filter(f => f.endsWith('.json'));

        for (const file of files) {
          try {
            const data = JSON.parse(fs.readFileSync(path.join(weekPath, file), 'utf8'));
            candleCount += data.length;
          } catch {
            // Skip
          }
        }
      }

      if (candleCount > 0) {
        assets.push({ category, asset, candleCount });
      }
    }
  }

  // Sort by candle count
  assets.sort((a, b) => b.candleCount - a.candleCount);

  return NextResponse.json({
    assets,
    totalAssets: assets.length,
  });
}
