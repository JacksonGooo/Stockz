/**
 * API Route: POST /api/predictions/[symbol]/generate
 * Force regenerate predictions for a symbol
 */

import { NextRequest, NextResponse } from 'next/server';
import { getMinuteCandles, getCryptoMinuteCandles, TVCandle } from '@/lib/tradingview';
import {
  generateMinutePredictions,
  getPredictionSummary,
} from '@/ai/ml/minutePredictionService';
import { savePrediction, deletePrediction } from '@/ai/ml/predictionStorage';
import { OHLCV } from '@/ai/ml/indicators';

/**
 * Convert TVCandle to OHLCV format
 */
function tvCandleToOHLCV(candle: TVCandle): OHLCV {
  return {
    open: candle.open,
    high: candle.high,
    low: candle.low,
    close: candle.close,
    volume: candle.volume,
    timestamp: new Date(candle.timestamp),
  };
}

export async function POST(
  request: NextRequest,
  { params }: { params: Promise<{ symbol: string }> }
) {
  const { symbol } = await params;

  let body: { type?: 'stock' | 'crypto' } = {};
  try {
    body = await request.json();
  } catch {
    // Empty body is okay
  }

  const assetType = body.type || 'stock';
  const upperSymbol = symbol.toUpperCase();

  try {
    // Delete existing prediction
    await deletePrediction(upperSymbol, assetType);

    // Fetch 30 1-minute historical candles
    let historicalCandles: TVCandle[];

    if (assetType === 'crypto') {
      historicalCandles = await getCryptoMinuteCandles(upperSymbol, 30);
    } else {
      historicalCandles = await getMinuteCandles(upperSymbol, 30);
    }

    if (historicalCandles.length === 0) {
      return NextResponse.json(
        { error: 'Failed to fetch historical data' },
        { status: 500 }
      );
    }

    // Convert to OHLCV format
    const ohlcvData = historicalCandles.map(tvCandleToOHLCV);

    // Generate new predictions
    const prediction = await generateMinutePredictions(
      upperSymbol,
      ohlcvData,
      assetType,
      60 // 60 minutes of predictions
    );

    // Save to storage
    await savePrediction(prediction);

    // Return summary
    const summary = getPredictionSummary(prediction);

    return NextResponse.json({
      success: true,
      symbol: upperSymbol,
      generatedAt: prediction.generatedAt,
      expiresAt: prediction.expiresAt,
      summary,
      message: 'Predictions generated successfully',
    });
  } catch (error) {
    console.error('Error generating predictions:', error);
    return NextResponse.json(
      { error: 'Failed to generate predictions' },
      { status: 500 }
    );
  }
}
