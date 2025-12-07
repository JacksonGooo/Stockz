import { NextRequest, NextResponse } from 'next/server';
import { searchStocks } from '@/lib/tradingview';

export async function GET(request: NextRequest) {
  const searchParams = request.nextUrl.searchParams;
  const query = searchParams.get('q');

  if (!query || query.length < 1) {
    return NextResponse.json({ results: [] });
  }

  try {
    const results = await searchStocks(query, 'stock');

    // Filter to show mainly US stocks and limit results
    const filtered = results
      .filter((r) =>
        ['NASDAQ', 'NYSE', 'AMEX', 'OTC'].includes(r.exchange.toUpperCase()) ||
        r.exchange.toUpperCase().includes('US')
      )
      .slice(0, 15)
      .map((r) => ({
        symbol: r.symbol,
        name: r.name,
        exchange: r.exchange,
        type: r.type,
        id: r.id,
      }));

    return NextResponse.json({ results: filtered });
  } catch (error) {
    console.error('Error searching stocks:', error);
    return NextResponse.json(
      { error: 'Failed to search stocks' },
      { status: 500 }
    );
  }
}
