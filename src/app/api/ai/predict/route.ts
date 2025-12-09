import { NextRequest, NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';
import { execSync } from 'child_process';

interface Candle {
  timestamp: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

interface DataStats {
  minPrice: number;
  maxPrice: number;
  minVolume: number;
  maxVolume: number;
}

interface BufferData {
  recentCandles: Candle[];
  lastPrice: number;
  lastTimestamp: number;
  dayCount: number;
}

// Load recent candles from realtime buffer (fresh live data)
function loadRealtimeBuffer(category: string, asset: string): Candle[] {
  const bufferFile = path.join(process.cwd(), 'Data', '.buffer', 'realtime.json');

  if (!fs.existsSync(bufferFile)) {
    return [];
  }

  try {
    const bufferContent = fs.readFileSync(bufferFile, 'utf8');
    const allBuffers: Record<string, BufferData> = JSON.parse(bufferContent);
    const key = `${category}/${asset}`;

    if (allBuffers[key] && allBuffers[key].recentCandles) {
      return allBuffers[key].recentCandles;
    }
  } catch {
    // Ignore errors
  }

  return [];
}

// Merge and deduplicate candles by timestamp
function mergeCandles(historical: Candle[], realtime: Candle[]): Candle[] {
  const map = new Map<number, Candle>();

  // Add historical first
  for (const c of historical) {
    map.set(c.timestamp, c);
  }

  // Realtime overwrites (more recent data)
  for (const c of realtime) {
    map.set(c.timestamp, c);
  }

  // Sort by timestamp
  return Array.from(map.values()).sort((a, b) => a.timestamp - b.timestamp);
}

// Load recent candles from disk (historical data)
function loadHistoricalCandles(category: string, asset: string, count: number): Candle[] {
  const dataDir = path.join(process.cwd(), 'Data');
  const assetDir = path.join(dataDir, category, asset);

  if (!fs.existsSync(assetDir)) {
    return [];
  }

  const allCandles: Candle[] = [];
  const weekDirs = fs.readdirSync(assetDir)
    .filter(d => d.startsWith('week_'))
    .sort()
    .reverse();

  for (const weekDir of weekDirs) {
    const weekPath = path.join(assetDir, weekDir);
    const files = fs.readdirSync(weekPath)
      .filter(f => f.endsWith('.json'))
      .sort()
      .reverse();

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

  return allCandles.slice(0, count).reverse();
}

// Load recent candles for an asset (merges disk + realtime buffer for freshest data)
function loadRecentCandles(category: string, asset: string, count: number = 30): Candle[] {
  // Load from disk (historical)
  const historical = loadHistoricalCandles(category, asset, count);

  // Load from realtime buffer (fresh live data)
  const realtime = loadRealtimeBuffer(category, asset);

  // Merge and take most recent 'count' candles
  const merged = mergeCandles(historical, realtime);

  return merged.slice(-count);
}

// Calculate normalization stats
function calculateStats(candles: Candle[]): DataStats {
  let minPrice = Infinity;
  let maxPrice = -Infinity;
  let minVolume = Infinity;
  let maxVolume = -Infinity;

  for (const candle of candles) {
    minPrice = Math.min(minPrice, candle.low);
    maxPrice = Math.max(maxPrice, candle.high);
    minVolume = Math.min(minVolume, candle.volume);
    maxVolume = Math.max(maxVolume, candle.volume);
  }

  return { minPrice, maxPrice, minVolume, maxVolume };
}

// Calculate SMA
function calculateSMA(prices: number[], period: number): number {
  if (prices.length < period) return prices[prices.length - 1] || 0;
  const slice = prices.slice(-period);
  return slice.reduce((a, b) => a + b, 0) / period;
}

// Calculate RSI
function calculateRSI(prices: number[], period: number = 14): number {
  if (prices.length < period + 1) return 50;

  let gains = 0;
  let losses = 0;

  for (let i = prices.length - period; i < prices.length; i++) {
    const change = prices[i] - prices[i - 1];
    if (change > 0) gains += change;
    else losses -= change;
  }

  const avgGain = gains / period;
  const avgLoss = losses / period;

  if (avgLoss === 0) return 100;
  const rs = avgGain / avgLoss;
  return 100 - (100 / (1 + rs));
}

// Calculate volatility (standard deviation)
function calculateVolatility(prices: number[]): number {
  if (prices.length < 2) return 0;
  const mean = prices.reduce((a, b) => a + b, 0) / prices.length;
  const squaredDiffs = prices.map(p => Math.pow(p - mean, 2));
  const variance = squaredDiffs.reduce((a, b) => a + b, 0) / prices.length;
  return Math.sqrt(variance);
}

// Check if a trained LSTM model exists for an asset
function hasTrainedModel(asset: string): boolean {
  const modelsDir = path.join(process.cwd(), 'models');
  const modelPath = path.join(modelsDir, `${asset.toLowerCase()}_lstm.keras`);
  const statsPath = path.join(modelsDir, `${asset.toLowerCase()}_lstm_stats.json`);
  return fs.existsSync(modelPath) && fs.existsSync(statsPath);
}

// Check if the new autoregressive BTC model exists
function hasAutoRegressiveModel(asset: string): boolean {
  if (asset.toUpperCase() !== 'BTC') return false;
  const modelPath = path.join(process.cwd(), 'btc_predictor', 'models', 'autoregressive_btc', 'model.keras');
  return fs.existsSync(modelPath);
}

// Run prediction using the autoregressive BTC model
function predictWithAutoRegressive(
  candles: Candle[],
  outputMinutes: number
): { predictions: Candle[]; summary: any } | null {
  try {
    const predictScript = path.join(process.cwd(), 'btc_predictor', 'predict_api.py');

    if (!fs.existsSync(predictScript)) {
      console.error('AutoRegressive predict script not found:', predictScript);
      return null;
    }

    const inputData = JSON.stringify({
      candles: candles,
      outputMinutes: outputMinutes
    });

    // Call Python script with input data
    const result = execSync(
      `py -3.11 "${predictScript}"`,
      {
        input: inputData,
        encoding: 'utf8',
        maxBuffer: 10 * 1024 * 1024, // 10MB buffer
        timeout: 120000, // 120 second timeout for model loading
        cwd: path.join(process.cwd(), 'btc_predictor')
      }
    );

    const parsed = JSON.parse(result);

    if (parsed.error) {
      console.error('AutoRegressive prediction error:', parsed.message);
      return null;
    }

    return {
      predictions: parsed.predictions,
      summary: parsed.summary
    };
  } catch (error) {
    console.error('AutoRegressive prediction failed:', error);
    return null;
  }
}

// Run prediction using the trained LSTM model
function predictWithLSTM(
  category: string,
  asset: string,
  candles: Candle[],
  outputMinutes: number
): { predictions: Candle[]; summary: any } | null {
  try {
    const predictScript = path.join(process.cwd(), 'btc_predictor', 'predict.py');

    if (!fs.existsSync(predictScript)) {
      console.error('Predict script not found:', predictScript);
      return null;
    }

    const inputData = JSON.stringify({
      candles: candles,
      outputMinutes: outputMinutes
    });

    // Call Python script with input data
    const result = execSync(
      `py -3.11 "${predictScript}" "${category}" "${asset}"`,
      {
        input: inputData,
        encoding: 'utf8',
        maxBuffer: 10 * 1024 * 1024, // 10MB buffer
        timeout: 60000, // 60 second timeout
      }
    );

    const parsed = JSON.parse(result);

    if (parsed.error) {
      console.error('LSTM prediction error:', parsed.message);
      return null;
    }

    return {
      predictions: parsed.predictions,
      summary: parsed.summary
    };
  } catch (error) {
    console.error('LSTM prediction failed:', error);
    return null;
  }
}

// Predict next 60 candles using technical analysis (fallback)
function predictCandles(
  recentCandles: Candle[],
  count: number = 60
): Candle[] {
  if (recentCandles.length < 10) {
    return [];
  }

  const predictions: Candle[] = [];
  const closes = recentCandles.map(c => c.close);
  const volumes = recentCandles.map(c => c.volume);

  // Calculate indicators
  const sma5 = calculateSMA(closes, 5);
  const sma10 = calculateSMA(closes, 10);
  const sma20 = calculateSMA(closes, 20);
  const rsi = calculateRSI(closes);
  const volatility = calculateVolatility(closes.slice(-20));
  const avgVolume = volumes.reduce((a, b) => a + b, 0) / volumes.length;

  // Determine trend
  let trendStrength = 0;
  if (sma5 > sma10) trendStrength += 0.3;
  else trendStrength -= 0.3;
  if (sma10 > sma20) trendStrength += 0.3;
  else trendStrength -= 0.3;
  if (closes[closes.length - 1] > sma20) trendStrength += 0.2;
  else trendStrength -= 0.2;

  // RSI influence
  if (rsi < 30) trendStrength += 0.3; // Oversold, likely to go up
  else if (rsi > 70) trendStrength -= 0.3; // Overbought, likely to go down

  // Normalize trend strength to a small percentage change per candle
  const baseChange = trendStrength * 0.001; // Max ~0.1% per candle

  // Generate predictions
  let lastCandle = recentCandles[recentCandles.length - 1];
  let currentPrice = lastCandle.close;

  for (let i = 0; i < count; i++) {
    // Add some randomness based on volatility
    const randomFactor = (Math.random() - 0.5) * 2;
    const volatilityFactor = (volatility / currentPrice) * randomFactor * 0.5;

    // Price change for this candle
    const priceChange = currentPrice * (baseChange + volatilityFactor);
    const newClose = currentPrice + priceChange;

    // Generate OHLC based on the direction
    const isUp = newClose >= currentPrice;
    const range = Math.abs(priceChange) + (volatility * Math.random() * 0.3);

    const open = currentPrice;
    const close = newClose;
    const high = Math.max(open, close) + range * Math.random() * 0.5;
    const low = Math.min(open, close) - range * Math.random() * 0.5;

    // Volume with some variation
    const volumeChange = (Math.random() - 0.5) * 0.4;
    const volume = avgVolume * (1 + volumeChange);

    predictions.push({
      timestamp: lastCandle.timestamp + (i + 1) * 60000, // +1 minute each
      open,
      high,
      low,
      close,
      volume,
    });

    currentPrice = newClose;
  }

  return predictions;
}

// GET handler - get recent candles and run prediction
export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { category, asset, inputMinutes = 30, outputMinutes = 60 } = body;

    if (!category || !asset) {
      return NextResponse.json({
        error: 'Missing parameters',
        message: 'category and asset are required',
      }, { status: 400 });
    }

    // Check which model is available - autoregressive BTC model takes priority
    const hasAutoReg = hasAutoRegressiveModel(asset);
    const hasOldModel = hasTrainedModel(asset);

    // Load recent candles - need more for models (60+ for feature calculation)
    const minCandles = (hasAutoReg || hasOldModel) ? 100 : 10;
    const recentCandles = loadRecentCandles(category, asset, Math.max(inputMinutes, minCandles));

    if (recentCandles.length < minCandles) {
      return NextResponse.json({
        error: 'Not enough data',
        message: `Only ${recentCandles.length} candles available. Need at least ${minCandles}.`,
      }, { status: 400 });
    }

    let predictedCandles: Candle[] = [];
    let mlSummary: any = null;
    let usedModel = 'none';

    // Try autoregressive model first (for BTC)
    if (hasAutoReg && outputMinutes > 0) {
      const autoRegResult = predictWithAutoRegressive(recentCandles, outputMinutes);
      if (autoRegResult) {
        predictedCandles = autoRegResult.predictions;
        mlSummary = autoRegResult.summary;
        usedModel = 'autoregressive_lstm';
      }
    }

    // Fall back to old LSTM model if autoregressive failed
    if (predictedCandles.length === 0 && hasOldModel && outputMinutes > 0) {
      const lstmResult = predictWithLSTM(category, asset, recentCandles, outputMinutes);
      if (lstmResult) {
        predictedCandles = lstmResult.predictions;
        mlSummary = lstmResult.summary;
        usedModel = 'lstm';
      }
    }

    // Fall back to technical analysis if LSTM failed or not available
    if (predictedCandles.length === 0 && outputMinutes > 0) {
      predictedCandles = predictCandles(recentCandles, outputMinutes);
    }

    // Calculate stats for context
    const closes = recentCandles.map(c => c.close);
    const currentPrice = closes[closes.length - 1];
    const predictedFinalPrice = predictedCandles.length > 0
      ? predictedCandles[predictedCandles.length - 1].close
      : currentPrice;

    const percentChange = mlSummary?.percentChange ??
      ((predictedFinalPrice - currentPrice) / currentPrice) * 100;

    return NextResponse.json({
      asset,
      category,
      realCandles: recentCandles,
      predictedCandles,
      summary: {
        currentPrice,
        predictedPrice: predictedFinalPrice,
        percentChange,
        direction: percentChange > 0.1 ? 'up' : percentChange < -0.1 ? 'down' : 'neutral',
        sma20: calculateSMA(closes, 20),
        rsi: calculateRSI(closes),
      },
      meta: {
        inputMinutes,
        outputMinutes,
        realCandleCount: recentCandles.length,
        predictedCandleCount: predictedCandles.length,
        timestamp: new Date().toISOString(),
        modelUsed: usedModel,
        usedLSTM: usedModel !== 'none',
      },
    });
  } catch (error) {
    console.error('AI Prediction error:', error);
    return NextResponse.json({
      error: 'Prediction failed',
      message: error instanceof Error ? error.message : 'Unknown error',
    }, { status: 500 });
  }
}

// GET handler - list available models
export async function GET() {
  const dataDir = path.join(process.cwd(), 'Data');
  const modelsDir = path.join(process.cwd(), 'models');

  const assets: { category: string; asset: string; hasModel: boolean; candleCount: number }[] = [];

  const categories = ['Crypto', 'Stock Market', 'Commodities', 'Currencies'];

  for (const category of categories) {
    const categoryPath = path.join(dataDir, category);
    if (!fs.existsSync(categoryPath)) continue;

    const assetDirs = fs.readdirSync(categoryPath).filter(f =>
      fs.statSync(path.join(categoryPath, f)).isDirectory()
    );

    for (const asset of assetDirs) {
      // Check if trained model exists (either old LSTM or new autoregressive)
      const modelPath = path.join(modelsDir, `${asset.toLowerCase()}_lstm`);
      const autoRegPath = path.join(process.cwd(), 'btc_predictor', 'models', 'autoregressive_btc', 'model.keras');
      const hasOldModel = fs.existsSync(modelPath);
      const hasAutoReg = asset.toUpperCase() === 'BTC' && fs.existsSync(autoRegPath);
      const hasModel = hasOldModel || hasAutoReg;

      // Count candles
      let candleCount = 0;
      const assetPath = path.join(categoryPath, asset);
      const weekDirs = fs.readdirSync(assetPath).filter(d => d.startsWith('week_'));

      for (const weekDir of weekDirs) {
        const weekPath = path.join(assetPath, weekDir);
        const files = fs.readdirSync(weekPath).filter(f => f.endsWith('.json'));

        for (const file of files) {
          try {
            const data = JSON.parse(fs.readFileSync(path.join(weekPath, file), 'utf8'));
            candleCount += data.length;
          } catch {
            // Skip
          }
        }
      }

      assets.push({ category, asset, hasModel, candleCount });
    }
  }

  // Sort by candle count (most data first)
  assets.sort((a, b) => b.candleCount - a.candleCount);

  return NextResponse.json({
    assets,
    totalAssets: assets.length,
    trainedModels: assets.filter(a => a.hasModel).length,
  });
}
