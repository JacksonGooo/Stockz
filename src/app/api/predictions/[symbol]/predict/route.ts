/**
 * API Route: GET /api/predictions/[symbol]/predict
 * Server-side ML prediction endpoint
 * Runs the full prediction pipeline on the server to keep TensorFlow.js out of client bundle
 */

import { NextRequest, NextResponse } from 'next/server';
import { getPredictionPipeline } from '@/ai/ml/predictionPipeline';
import { PredictionTimeframe } from '@/ai/types';

// Cache predictions for 2 minutes to reduce computation
const predictionCache = new Map<string, { data: any; timestamp: number }>();
const CACHE_TTL_MS = 2 * 60 * 1000; // 2 minutes

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ symbol: string }> }
) {
  const { symbol } = await params;
  const upperSymbol = symbol.toUpperCase();

  // Parse query parameters
  const searchParams = request.nextUrl.searchParams;
  const timeframe = (searchParams.get('timeframe') || '1w') as PredictionTimeframe;
  const skipCache = searchParams.get('fresh') === 'true';

  // Validate timeframe
  const validTimeframes: PredictionTimeframe[] = ['30m', '1d', '1w', '1m', '3m', '6m', '1y'];
  if (!validTimeframes.includes(timeframe)) {
    return NextResponse.json(
      { error: `Invalid timeframe. Valid options: ${validTimeframes.join(', ')}` },
      { status: 400 }
    );
  }

  // Check cache first
  const cacheKey = `${upperSymbol}-${timeframe}`;
  if (!skipCache) {
    const cached = predictionCache.get(cacheKey);
    if (cached && Date.now() - cached.timestamp < CACHE_TTL_MS) {
      return NextResponse.json({
        ...cached.data,
        cached: true,
        cacheAge: Math.round((Date.now() - cached.timestamp) / 1000),
      });
    }
  }

  try {
    // Get prediction pipeline and run prediction
    const pipeline = getPredictionPipeline();
    const prediction = await pipeline.predict(upperSymbol, timeframe);

    if (!prediction) {
      return NextResponse.json(
        { error: 'Failed to generate prediction. Insufficient data for symbol.' },
        { status: 404 }
      );
    }

    // Get additional context
    const sentiment = await pipeline.getMarketSentiment(upperSymbol);
    const status = await pipeline.getServiceStatus();

    const response = {
      prediction,
      sentiment,
      modelInfo: {
        version: status.modelVersion,
        accuracy: status.metrics.accuracy,
        lastTrained: status.metrics.lastTrainedAt,
      },
      cached: false,
    };

    // Cache the result
    predictionCache.set(cacheKey, {
      data: response,
      timestamp: Date.now(),
    });

    return NextResponse.json(response);
  } catch (error) {
    console.error('Prediction error:', error);
    return NextResponse.json(
      { error: 'Failed to generate prediction' },
      { status: 500 }
    );
  }
}

/**
 * POST /api/predictions/[symbol]/predict
 * Batch predictions for multiple timeframes
 */
export async function POST(
  request: NextRequest,
  { params }: { params: Promise<{ symbol: string }> }
) {
  const { symbol } = await params;
  const upperSymbol = symbol.toUpperCase();

  let body: { timeframes?: PredictionTimeframe[] } = {};
  try {
    body = await request.json();
  } catch {
    // Empty body - use defaults
  }

  const timeframes = body.timeframes || ['1d', '1w', '1m'];

  try {
    const pipeline = getPredictionPipeline();
    const predictions: Record<string, any> = {};

    // Generate predictions for each timeframe in parallel
    const results = await Promise.all(
      timeframes.map(async (tf) => {
        const prediction = await pipeline.predict(upperSymbol, tf);
        return { timeframe: tf, prediction };
      })
    );

    for (const result of results) {
      if (result.prediction) {
        predictions[result.timeframe] = result.prediction;
      }
    }

    if (Object.keys(predictions).length === 0) {
      return NextResponse.json(
        { error: 'Failed to generate any predictions' },
        { status: 404 }
      );
    }

    return NextResponse.json({
      symbol: upperSymbol,
      predictions,
      generatedAt: new Date().toISOString(),
    });
  } catch (error) {
    console.error('Batch prediction error:', error);
    return NextResponse.json(
      { error: 'Failed to generate predictions' },
      { status: 500 }
    );
  }
}
