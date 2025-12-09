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

interface PredictionResponse {
  asset: string;
  category: string;
  currentPrice: number;
  prediction: {
    price: number;
    direction: 'up' | 'down' | 'neutral';
    percentChange: number;
    confidence: number;
  };
  technicalIndicators: {
    sma20: number;
    sma50: number;
    rsi: number;
    trend: 'bullish' | 'bearish' | 'neutral';
  };
  recentCandles: number;
  timestamp: string;
}

// Load recent candles for an asset
function loadRecentCandles(category: string, asset: string, count: number = 100): Candle[] {
  const dataDir = path.join(process.cwd(), 'Data');
  const assetDir = path.join(dataDir, category, asset);

  if (!fs.existsSync(assetDir)) {
    return [];
  }

  const allCandles: Candle[] = [];
  const weekDirs = fs.readdirSync(assetDir)
    .filter(d => d.startsWith('week_'))
    .sort()
    .reverse(); // Most recent first

  for (const weekDir of weekDirs) {
    const weekPath = path.join(assetDir, weekDir);
    const files = fs.readdirSync(weekPath)
      .filter(f => f.endsWith('.json'))
      .sort()
      .reverse(); // Most recent first

    for (const file of files) {
      try {
        const data = JSON.parse(fs.readFileSync(path.join(weekPath, file), 'utf8'));
        allCandles.push(...data.reverse());

        if (allCandles.length >= count) {
          break;
        }
      } catch {
        // Skip invalid files
      }
    }

    if (allCandles.length >= count) {
      break;
    }
  }

  // Return in chronological order (oldest first)
  return allCandles.slice(0, count).reverse();
}

// Calculate Simple Moving Average
function calculateSMA(prices: number[], period: number): number {
  if (prices.length < period) return prices[prices.length - 1] || 0;
  const slice = prices.slice(-period);
  return slice.reduce((a, b) => a + b, 0) / period;
}

// Calculate RSI (Relative Strength Index)
function calculateRSI(prices: number[], period: number = 14): number {
  if (prices.length < period + 1) return 50;

  let gains = 0;
  let losses = 0;

  for (let i = prices.length - period; i < prices.length; i++) {
    const change = prices[i] - prices[i - 1];
    if (change > 0) {
      gains += change;
    } else {
      losses -= change;
    }
  }

  const avgGain = gains / period;
  const avgLoss = losses / period;

  if (avgLoss === 0) return 100;
  const rs = avgGain / avgLoss;
  return 100 - (100 / (1 + rs));
}

// Simple price prediction using technical analysis
function predictPrice(candles: Candle[]): { price: number; confidence: number; direction: 'up' | 'down' | 'neutral' } {
  if (candles.length < 20) {
    const lastPrice = candles[candles.length - 1]?.close || 0;
    return { price: lastPrice, confidence: 0, direction: 'neutral' };
  }

  const closes = candles.map(c => c.close);
  const currentPrice = closes[closes.length - 1];

  // Calculate indicators
  const sma5 = calculateSMA(closes, 5);
  const sma10 = calculateSMA(closes, 10);
  const sma20 = calculateSMA(closes, 20);
  const rsi = calculateRSI(closes);

  // Calculate momentum
  const momentum = (closes[closes.length - 1] - closes[closes.length - 5]) / closes[closes.length - 5];

  // Weighted prediction based on multiple factors
  let bullishSignals = 0;
  let bearishSignals = 0;

  // SMA crossovers
  if (sma5 > sma10) bullishSignals += 1;
  else bearishSignals += 1;

  if (sma10 > sma20) bullishSignals += 1;
  else bearishSignals += 1;

  // Price above/below SMAs
  if (currentPrice > sma20) bullishSignals += 1;
  else bearishSignals += 1;

  // RSI signals
  if (rsi < 30) bullishSignals += 2; // Oversold = buy signal
  else if (rsi > 70) bearishSignals += 2; // Overbought = sell signal

  // Momentum
  if (momentum > 0.01) bullishSignals += 1;
  else if (momentum < -0.01) bearishSignals += 1;

  // Calculate prediction
  const totalSignals = bullishSignals + bearishSignals;
  const bullishRatio = bullishSignals / totalSignals;

  // Predict price change
  const maxChange = 0.005; // Max 0.5% prediction
  const changeMultiplier = (bullishRatio - 0.5) * 2; // -1 to 1
  const predictedChange = changeMultiplier * maxChange;
  const predictedPrice = currentPrice * (1 + predictedChange);

  // Confidence based on signal strength
  const confidence = Math.min(Math.abs(bullishRatio - 0.5) * 200, 80);

  // Direction
  let direction: 'up' | 'down' | 'neutral' = 'neutral';
  if (bullishRatio > 0.6) direction = 'up';
  else if (bullishRatio < 0.4) direction = 'down';

  return { price: predictedPrice, confidence, direction };
}

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ category: string; asset: string }> }
) {
  const { category, asset } = await params;

  try {
    // Decode URL-encoded category names
    const decodedCategory = decodeURIComponent(category);

    // Load recent candles
    const candles = loadRecentCandles(decodedCategory, asset, 100);

    if (candles.length < 10) {
      return NextResponse.json({
        error: 'Not enough data',
        message: `Only ${candles.length} candles available for ${asset}. Need at least 10.`,
      }, { status: 400 });
    }

    const closes = candles.map(c => c.close);
    const currentPrice = closes[closes.length - 1];

    // Calculate technical indicators
    const sma20 = calculateSMA(closes, 20);
    const sma50 = closes.length >= 50 ? calculateSMA(closes, 50) : sma20;
    const rsi = calculateRSI(closes);

    // Determine trend
    let trend: 'bullish' | 'bearish' | 'neutral' = 'neutral';
    if (currentPrice > sma20 && sma20 > sma50) trend = 'bullish';
    else if (currentPrice < sma20 && sma20 < sma50) trend = 'bearish';

    // Get prediction
    const prediction = predictPrice(candles);
    const percentChange = ((prediction.price - currentPrice) / currentPrice) * 100;

    const response: PredictionResponse = {
      asset,
      category: decodedCategory,
      currentPrice,
      prediction: {
        price: prediction.price,
        direction: prediction.direction,
        percentChange,
        confidence: prediction.confidence,
      },
      technicalIndicators: {
        sma20,
        sma50,
        rsi,
        trend,
      },
      recentCandles: candles.length,
      timestamp: new Date().toISOString(),
    };

    return NextResponse.json(response);
  } catch (error) {
    console.error('Prediction error:', error);
    return NextResponse.json({
      error: 'Prediction failed',
      message: error instanceof Error ? error.message : 'Unknown error',
    }, { status: 500 });
  }
}
