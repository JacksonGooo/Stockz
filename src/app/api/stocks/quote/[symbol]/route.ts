import { NextRequest, NextResponse } from 'next/server';
import { getQuote } from '@/lib/tradingview';

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ symbol: string }> }
) {
  const { symbol } = await params;
  const upperSymbol = symbol.toUpperCase();

  try {
    // FAST: Single HTTP request - quote already includes TA data
    const quote = await getQuote(upperSymbol);

    if (!quote) {
      return NextResponse.json(
        { error: 'Symbol not found or no data available' },
        { status: 404 }
      );
    }

    return NextResponse.json({
      symbol: quote.symbol,
      exchange: quote.exchange,
      name: quote.name,
      price: quote.price,
      change: quote.change,
      changePercent: quote.changePercent,
      high: quote.high,
      low: quote.low,
      open: quote.open,
      previousClose: quote.previousClose,
      volume: quote.volume,
      currency: quote.currency,
      // Technical analysis data from Scanner API
      recommendation: quote.recommendation || null,
      taScore: quote.taScore || null,
    });
  } catch (error) {
    console.error('Error fetching stock quote:', error);
    return NextResponse.json(
      { error: 'Failed to fetch stock data' },
      { status: 500 }
    );
  }
}
