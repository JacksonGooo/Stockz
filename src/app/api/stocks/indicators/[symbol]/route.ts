import { NextRequest, NextResponse } from 'next/server';
import { getTechnicalAnalysis } from '@/lib/tradingview';

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ symbol: string }> }
) {
  const { symbol } = await params;
  const upperSymbol = symbol.toUpperCase();

  try {
    const ta = await getTechnicalAnalysis(upperSymbol);

    if (!ta) {
      return NextResponse.json(
        { error: 'Technical analysis not available for this symbol' },
        { status: 404 }
      );
    }

    // Parse the TA periods into a more usable format
    const indicators: Record<string, unknown> = {
      symbol: upperSymbol,
      timestamp: new Date().toISOString(),
      recommendation: ta.recommendation,
      score: ta.score,
    };

    // Add period-specific data
    const periodLabels: Record<string, string> = {
      '1': '1m',
      '5': '5m',
      '15': '15m',
      '60': '1h',
      '240': '4h',
      '1D': 'daily',
      '1W': 'weekly',
      '1M': 'monthly',
    };

    for (const [period, data] of Object.entries(ta.periods)) {
      const label = periodLabels[period] || period;
      indicators[label] = {
        oscillators: data.Other,
        overall: data.All,
        movingAverages: data.MA,
        recommendation: getRecommendation(data.All),
      };
    }

    return NextResponse.json(indicators);
  } catch (error) {
    console.error('Error fetching indicators:', error);
    return NextResponse.json(
      { error: 'Failed to fetch technical indicators' },
      { status: 500 }
    );
  }
}

function getRecommendation(score: number): string {
  if (score >= 1) return 'STRONG_BUY';
  if (score >= 0.5) return 'BUY';
  if (score <= -1) return 'STRONG_SELL';
  if (score <= -0.5) return 'SELL';
  return 'NEUTRAL';
}
