import { NextRequest, NextResponse } from 'next/server';
import { getCandles } from '@/lib/tradingview';

// Timeframe mapping
const TIMEFRAME_MAP: Record<string, string> = {
  '1': '1',      // 1 minute
  '5': '5',      // 5 minutes
  '15': '15',    // 15 minutes
  '30': '30',    // 30 minutes
  '60': '60',    // 1 hour
  '240': '240',  // 4 hours
  'D': 'D',      // Daily
  'W': 'W',      // Weekly
  'M': 'M',      // Monthly
};

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ symbol: string }> }
) {
  const { symbol } = await params;
  const upperSymbol = symbol.toUpperCase();

  const searchParams = request.nextUrl.searchParams;
  const days = parseInt(searchParams.get('days') || '100', 10);
  const timeframe = TIMEFRAME_MAP[searchParams.get('timeframe') || 'D'] || 'D';

  try {
    const candles = await getCandles(upperSymbol, timeframe, days);

    if (!candles || candles.length === 0) {
      return NextResponse.json(
        { error: 'No historical data available for this symbol' },
        { status: 404 }
      );
    }

    return NextResponse.json({
      symbol: upperSymbol,
      timeframe,
      candles: candles.map((c) => ({
        timestamp: c.timestamp,
        open: c.open,
        high: c.high,
        low: c.low,
        close: c.close,
        volume: c.volume,
      })),
    });
  } catch (error) {
    console.error('Error fetching candles:', error);
    return NextResponse.json(
      { error: 'Failed to fetch historical data' },
      { status: 500 }
    );
  }
}
