/**
 * API Route: GET /api/predictions/[symbol]/candles
 * Returns combined historical + predicted candles for the 90-minute chart
 */

import { NextRequest, NextResponse } from 'next/server';
import { getMinuteCandles, getCryptoMinuteCandles, TVCandle } from '@/lib/tradingview';
import {
  generateMinutePredictions,
  MinuteCandle,
  PredictionResult,
} from '@/ai/ml/minutePredictionService';
import {
  getValidPrediction,
  savePrediction,
  getPredictionAge,
  getTimeUntilExpiry,
} from '@/ai/ml/predictionStorage';
import { OHLCV } from '@/ai/ml/indicators';

interface CandleResponse {
  symbol: string;
  assetType: 'stock' | 'crypto';
  generatedAt: number;
  expiresAt: number;
  candles: Array<{
    timestamp: number;
    open: number;
    high: number;
    low: number;
    close: number;
    type: 'historical' | 'predicted';
    direction: 'up' | 'down';
  }>;
  metadata: {
    historicalCount: number;
    predictedCount: number;
    confidence: number;
    trend: string;
    ageMinutes: number;
    expiresInMinutes: number;
  };
}

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

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ symbol: string }> }
) {
  const { symbol } = await params;
  const searchParams = request.nextUrl.searchParams;
  const assetType = (searchParams.get('type') as 'stock' | 'crypto') || 'stock';
  const forceRegenerate = searchParams.get('force') === 'true';

  const upperSymbol = symbol.toUpperCase();

  try {
    // Check for valid cached prediction first (unless force regenerate)
    let prediction: PredictionResult | null = null;

    if (!forceRegenerate) {
      prediction = await getValidPrediction(upperSymbol, assetType);
    }

    // If no valid prediction, generate new one
    if (!prediction) {
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

      // Generate predictions
      prediction = await generateMinutePredictions(
        upperSymbol,
        ohlcvData,
        assetType,
        60 // 60 minutes of predictions
      );

      // Save to storage
      await savePrediction(prediction);
    }

    // Combine historical and predicted candles for response
    const allCandles: CandleResponse['candles'] = [
      ...prediction.historicalCandles.map((c: MinuteCandle) => ({
        timestamp: c.timestamp,
        open: c.open,
        high: c.high,
        low: c.low,
        close: c.close,
        type: 'historical' as const,
        direction: c.direction,
      })),
      ...prediction.predictedCandles.map((c: MinuteCandle) => ({
        timestamp: c.timestamp,
        open: c.open,
        high: c.high,
        low: c.low,
        close: c.close,
        type: 'predicted' as const,
        direction: c.direction,
      })),
    ];

    const response: CandleResponse = {
      symbol: prediction.symbol,
      assetType: prediction.assetType,
      generatedAt: prediction.generatedAt,
      expiresAt: prediction.expiresAt,
      candles: allCandles,
      metadata: {
        historicalCount: prediction.historicalCandles.length,
        predictedCount: prediction.predictedCandles.length,
        confidence: prediction.metadata.confidence,
        trend: prediction.metadata.trend,
        ageMinutes: getPredictionAge(prediction),
        expiresInMinutes: getTimeUntilExpiry(prediction),
      },
    };

    return NextResponse.json(response);
  } catch (error) {
    console.error('Error fetching prediction candles:', error);
    return NextResponse.json(
      { error: 'Failed to fetch prediction data' },
      { status: 500 }
    );
  }
}
