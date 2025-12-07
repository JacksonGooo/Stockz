import { NextRequest, NextResponse } from 'next/server';
import { getCryptoQuotes, getDefaultCryptoSymbols } from '@/lib/tradingview';

export async function GET(request: NextRequest) {
  const searchParams = request.nextUrl.searchParams;
  const symbolsParam = searchParams.get('symbols');

  // Use provided symbols or default crypto symbols
  const symbols = symbolsParam
    ? symbolsParam.split(',').map((s) => s.trim().toUpperCase())
    : getDefaultCryptoSymbols();

  try {
    const quotes = await getCryptoQuotes(symbols);

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
    console.error('Error fetching crypto quotes:', error);
    return NextResponse.json(
      { error: 'Failed to fetch crypto quotes' },
      { status: 500 }
    );
  }
}
