import { NextRequest, NextResponse } from 'next/server';
import { getMultipleQuotes } from '@/lib/tradingview';

export async function GET(request: NextRequest) {
  const searchParams = request.nextUrl.searchParams;
  const symbolsParam = searchParams.get('symbols');

  if (!symbolsParam) {
    return NextResponse.json({ quotes: [] });
  }

  const symbols = symbolsParam.split(',').map((s) => s.trim().toUpperCase());

  if (symbols.length === 0) {
    return NextResponse.json({ quotes: [] });
  }

  try {
    // FAST: Single HTTP request for all stocks via TradingView Scanner API
    const quotes = await getMultipleQuotes(symbols);

    return NextResponse.json({
      quotes: quotes.map((q) => ({
        symbol: q.symbol,
        exchange: q.exchange,
        name: q.name,
        price: q.price,
        change: q.change,
        changePercent: q.changePercent,
        high: q.high,
        low: q.low,
        open: q.open,
        previousClose: q.previousClose,
        volume: q.volume,
        currency: q.currency,
        recommendation: q.recommendation,
        taScore: q.taScore,
      })),
    });
  } catch (error) {
    console.error('Error fetching quotes:', error);
    return NextResponse.json(
      { error: 'Failed to fetch stock quotes' },
      { status: 500 }
    );
  }
}
