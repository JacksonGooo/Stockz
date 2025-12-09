/**
 * Data Loader for TensorFlow.js
 * Loads OHLCV candle data from the Data folder
 * Prepares data for neural network training and prediction
 */

import * as fs from 'fs';
import * as path from 'path';

// OHLCV candle format (same for all data sources)
export interface Candle {
  timestamp: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

// Normalized candle for neural network input
export interface NormalizedCandle {
  open: number;   // 0-1 normalized
  high: number;   // 0-1 normalized
  low: number;    // 0-1 normalized
  close: number;  // 0-1 normalized
  volume: number; // 0-1 normalized
}

// Data statistics for normalization
export interface DataStats {
  minPrice: number;
  maxPrice: number;
  minVolume: number;
  maxVolume: number;
}

const DATA_DIR = path.join(process.cwd(), 'Data');

/**
 * Load all candles for an asset
 */
export function loadAssetData(category: string, asset: string): Candle[] {
  const assetDir = path.join(DATA_DIR, category, asset);

  if (!fs.existsSync(assetDir)) {
    return [];
  }

  const allCandles: Candle[] = [];
  const weekDirs = fs.readdirSync(assetDir)
    .filter(d => d.startsWith('week_'))
    .sort();

  for (const weekDir of weekDirs) {
    const weekPath = path.join(assetDir, weekDir);
    const files = fs.readdirSync(weekPath)
      .filter(f => f.endsWith('.json'))
      .sort();

    for (const file of files) {
      try {
        const data = JSON.parse(fs.readFileSync(path.join(weekPath, file), 'utf8'));
        allCandles.push(...data);
      } catch (e) {
        // Skip invalid files
      }
    }
  }

  // Sort by timestamp and remove duplicates
  const uniqueCandles = Array.from(
    new Map(allCandles.map(c => [c.timestamp, c])).values()
  ).sort((a, b) => a.timestamp - b.timestamp);

  return uniqueCandles;
}

/**
 * Calculate statistics for normalization
 */
export function calculateStats(candles: Candle[]): DataStats {
  if (candles.length === 0) {
    return { minPrice: 0, maxPrice: 1, minVolume: 0, maxVolume: 1 };
  }

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

/**
 * Normalize candles to 0-1 range for neural network
 */
export function normalizeCandles(candles: Candle[], stats: DataStats): NormalizedCandle[] {
  const priceRange = stats.maxPrice - stats.minPrice || 1;
  const volumeRange = stats.maxVolume - stats.minVolume || 1;

  return candles.map(c => ({
    open: (c.open - stats.minPrice) / priceRange,
    high: (c.high - stats.minPrice) / priceRange,
    low: (c.low - stats.minPrice) / priceRange,
    close: (c.close - stats.minPrice) / priceRange,
    volume: (c.volume - stats.minVolume) / volumeRange,
  }));
}

/**
 * Denormalize price back to actual value
 */
export function denormalizePrice(normalized: number, stats: DataStats): number {
  const priceRange = stats.maxPrice - stats.minPrice || 1;
  return normalized * priceRange + stats.minPrice;
}

/**
 * Create training sequences for LSTM
 * Input: past N candles
 * Output: next candle's close price
 */
export function createSequences(
  normalizedCandles: NormalizedCandle[],
  sequenceLength: number = 60  // 60 minutes = 1 hour lookback
): { inputs: number[][][], outputs: number[] } {
  const inputs: number[][][] = [];
  const outputs: number[] = [];

  for (let i = sequenceLength; i < normalizedCandles.length; i++) {
    // Input: past sequenceLength candles (each with 5 features)
    const sequence = normalizedCandles.slice(i - sequenceLength, i).map(c => [
      c.open,
      c.high,
      c.low,
      c.close,
      c.volume,
    ]);
    inputs.push(sequence);

    // Output: next candle's close price
    outputs.push(normalizedCandles[i].close);
  }

  return { inputs, outputs };
}

/**
 * Split data into training and validation sets
 */
export function splitData<T>(
  data: T[],
  trainRatio: number = 0.8
): { train: T[], validation: T[] } {
  const splitIndex = Math.floor(data.length * trainRatio);
  return {
    train: data.slice(0, splitIndex),
    validation: data.slice(splitIndex),
  };
}

/**
 * Get list of all available assets
 */
export function getAvailableAssets(): { category: string; asset: string }[] {
  const assets: { category: string; asset: string }[] = [];

  if (!fs.existsSync(DATA_DIR)) {
    return assets;
  }

  const categories = fs.readdirSync(DATA_DIR).filter(d =>
    fs.statSync(path.join(DATA_DIR, d)).isDirectory()
  );

  for (const category of categories) {
    const categoryPath = path.join(DATA_DIR, category);
    const assetDirs = fs.readdirSync(categoryPath).filter(d =>
      fs.statSync(path.join(categoryPath, d)).isDirectory()
    );

    for (const asset of assetDirs) {
      assets.push({ category, asset });
    }
  }

  return assets;
}

/**
 * Load and prepare data for training
 */
export function prepareTrainingData(
  category: string,
  asset: string,
  sequenceLength: number = 60
) {
  // Load raw candles
  const candles = loadAssetData(category, asset);

  if (candles.length < sequenceLength + 100) {
    throw new Error(`Not enough data for ${asset}. Need at least ${sequenceLength + 100} candles.`);
  }

  // Calculate normalization stats
  const stats = calculateStats(candles);

  // Normalize
  const normalized = normalizeCandles(candles, stats);

  // Create sequences
  const { inputs, outputs } = createSequences(normalized, sequenceLength);

  // Split into train/validation
  const { train: trainInputs, validation: valInputs } = splitData(inputs);
  const { train: trainOutputs, validation: valOutputs } = splitData(outputs);

  return {
    trainInputs,
    trainOutputs,
    valInputs,
    valOutputs,
    stats,
    totalCandles: candles.length,
    sequenceLength,
  };
}
