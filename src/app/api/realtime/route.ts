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

// GET handler - fetch recent candles from memory buffer
export async function GET(request: NextRequest) {
  const searchParams = request.nextUrl.searchParams;
  const category = searchParams.get('category');
  const asset = searchParams.get('asset');
  const minutes = parseInt(searchParams.get('minutes') || '30', 10);

  if (!category || !asset) {
    return NextResponse.json({
      error: 'Missing parameters',
      message: 'category and asset are required',
    }, { status: 400 });
  }

  const bufferFile = path.join(process.cwd(), 'Data', '.buffer', 'realtime.json');

  // Check if buffer file exists
  if (!fs.existsSync(bufferFile)) {
    // Fall back to loading from disk
    return NextResponse.json({
      error: 'No realtime data',
      message: 'Realtime buffer not available. Is the collector running?',
      candles: [],
    }, { status: 404 });
  }

  try {
    const bufferContent = fs.readFileSync(bufferFile, 'utf8');
    const allBuffers: Record<string, BufferData> = JSON.parse(bufferContent);

    const key = `${category}/${asset}`;
    const buffer = allBuffers[key];

    if (!buffer || !buffer.recentCandles || buffer.recentCandles.length === 0) {
      return NextResponse.json({
        error: 'No data',
        message: `No realtime data for ${category}/${asset}`,
        candles: [],
      }, { status: 404 });
    }

    // Get the last N minutes of candles
    const candles = buffer.recentCandles.slice(-minutes);

    // Calculate summary
    const currentPrice = candles[candles.length - 1]?.close || 0;
    const openPrice = candles[0]?.open || currentPrice;
    const highPrice = Math.max(...candles.map(c => c.high));
    const lowPrice = Math.min(...candles.map(c => c.low));
    const priceChange = currentPrice - openPrice;
    const percentChange = openPrice > 0 ? ((currentPrice - openPrice) / openPrice) * 100 : 0;

    return NextResponse.json({
      asset,
      category,
      candles,
      lastPrice: buffer.lastPrice,
      lastUpdate: buffer.lastTimestamp,
      todaysCandleCount: buffer.dayCount,
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
  } catch (err) {
    console.error('Error reading realtime buffer:', err);
    return NextResponse.json({
      error: 'Read error',
      message: 'Failed to read realtime buffer',
    }, { status: 500 });
  }
}

// POST handler - list all available assets in buffer
export async function POST() {
  const bufferFile = path.join(process.cwd(), 'Data', '.buffer', 'realtime.json');

  if (!fs.existsSync(bufferFile)) {
    return NextResponse.json({
      error: 'No realtime data',
      message: 'Realtime buffer not available. Is the collector running?',
      assets: [],
    });
  }

  try {
    const bufferContent = fs.readFileSync(bufferFile, 'utf8');
    const allBuffers: Record<string, BufferData> = JSON.parse(bufferContent);

    const assets = Object.entries(allBuffers)
      .filter(([_, buffer]) => buffer.recentCandles && buffer.recentCandles.length > 0)
      .map(([key, buffer]) => {
        const [category, asset] = key.split('/');
        return {
          category,
          asset,
          lastPrice: buffer.lastPrice,
          lastUpdate: buffer.lastTimestamp,
          candleCount: buffer.recentCandles.length,
          todayCount: buffer.dayCount,
        };
      })
      .sort((a, b) => b.lastUpdate - a.lastUpdate);

    return NextResponse.json({
      assets,
      totalAssets: assets.length,
      bufferAge: Date.now() - (assets[0]?.lastUpdate || 0),
    });
  } catch (err) {
    console.error('Error reading realtime buffer:', err);
    return NextResponse.json({
      error: 'Read error',
      message: 'Failed to read realtime buffer',
      assets: [],
    });
  }
}
