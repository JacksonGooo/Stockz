/**
 * Model Training Service
 * Trains the LSTM model on real historical data to improve prediction accuracy
 */

import { StockPredictionModel, ModelMetrics } from './model';
import { calculateAllIndicators, indicatorsToFeatures, OHLCV } from './indicators';

// Training symbols - diverse set for generalization
const TRAINING_STOCKS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'JPM', 'V', 'WMT'];
const TRAINING_CRYPTO = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'ADAUSDT'];

// In-memory model state (persisted via API on server side)
let cachedModelState: ModelState | null = null;

// Candle interface for training data
interface TrainingCandle {
  timestamp: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface TrainingProgress {
  phase: 'fetching' | 'preparing' | 'training' | 'validating' | 'complete' | 'error';
  progress: number; // 0-100
  currentEpoch?: number;
  totalEpochs?: number;
  loss?: number;
  valLoss?: number;
  message: string;
  samplesCollected?: number;
  estimatedAccuracy?: number;
}

export interface ModelState {
  trainedAt: number;
  accuracy: number;
  loss: number;
  valLoss: number;
  trainingSamples: number;
  epochs: number;
  symbols: string[];
}

// Singleton for training state
let trainingInProgress = false;
let currentProgress: TrainingProgress = {
  phase: 'complete',
  progress: 100,
  message: 'Ready',
};

/**
 * Convert TrainingCandle to OHLCV format
 */
function candleToOHLCV(candle: TrainingCandle): OHLCV {
  return {
    open: candle.open,
    high: candle.high,
    low: candle.low,
    close: candle.close,
    volume: candle.volume,
    timestamp: new Date(candle.timestamp),
  };
}

// Multi-timeframe output days (1, 5, 10 days ahead)
const OUTPUT_TIMEFRAMES = [1, 5, 10];

/**
 * Create training sequences from OHLCV data
 * Each sequence is 60 timesteps, target is multi-timeframe % changes (1, 5, 10 days)
 */
function createTrainingData(
  candles: OHLCV[],
  sequenceLength: number = 60
): { sequences: number[][][]; targets: number[][] } {
  const sequences: number[][][] = [];
  const targets: number[][] = [];

  const maxLookahead = Math.max(...OUTPUT_TIMEFRAMES);

  if (candles.length < sequenceLength + maxLookahead) {
    return { sequences, targets };
  }

  // Create overlapping sequences
  for (let i = sequenceLength; i < candles.length - maxLookahead; i++) {
    const sequenceCandles = candles.slice(i - sequenceLength, i);

    // Calculate features for each timestep in sequence
    const sequence: number[][] = [];
    let valid = true;

    for (let j = 0; j < sequenceCandles.length; j++) {
      const windowStart = Math.max(0, j - 15);
      const window = sequenceCandles.slice(windowStart, j + 1);

      if (window.length >= 5) {
        const indicators = calculateAllIndicators(window);
        const features = indicatorsToFeatures(indicators);

        // Check for NaN/Infinity
        if (features.some(f => !isFinite(f))) {
          valid = false;
          break;
        }
        sequence.push(features);
      }
    }

    if (!valid || sequence.length < sequenceLength) continue;

    // Multi-timeframe targets: calculate price changes for 1, 5, 10 days ahead
    const currentClose = sequenceCandles[sequenceCandles.length - 1].close;
    const targetValues: number[] = [];
    let allValid = true;

    for (const days of OUTPUT_TIMEFRAMES) {
      const futureIndex = i + days - 1;
      if (futureIndex < candles.length) {
        const futureClose = candles[futureIndex].close;
        const priceChange = (futureClose - currentClose) / currentClose;
        // Normalize to -1 to 1 range using tanh for smoother scaling
        const normalizedTarget = Math.tanh(priceChange * 10);
        if (!isFinite(normalizedTarget)) {
          allValid = false;
          break;
        }
        targetValues.push(normalizedTarget);
      } else {
        allValid = false;
        break;
      }
    }

    if (allValid && targetValues.length === OUTPUT_TIMEFRAMES.length) {
      sequences.push(sequence.slice(-sequenceLength));
      targets.push(targetValues);
    }
  }

  return { sequences, targets };
}

/**
 * Fetch candles via API (works in both client and server)
 */
async function fetchCandlesViaAPI(symbol: string, isCrypto: boolean): Promise<TrainingCandle[]> {
  try {
    if (isCrypto) {
      // For crypto, use the predictions API which fetches minute candles
      const response = await fetch(`/api/predictions/${symbol}/candles?type=crypto`);
      if (response.ok) {
        const data = await response.json();
        return data.candles || [];
      }
    } else {
      // For stocks, use the candles API
      const response = await fetch(`/api/stocks/candles/${symbol}?timeframe=D&count=365`);
      if (response.ok) {
        const data = await response.json();
        return data.candles || data || [];
      }
    }
  } catch (error) {
    console.error(`Failed to fetch candles for ${symbol}:`, error);
  }
  return [];
}

/**
 * Fetch and prepare training data from multiple symbols
 */
async function fetchTrainingData(
  symbols: string[],
  isCrypto: boolean,
  onProgress?: (msg: string, collected: number) => void
): Promise<{ sequences: number[][][]; targets: number[][] }> {
  const allSequences: number[][][] = [];
  const allTargets: number[][] = [];

  for (let i = 0; i < symbols.length; i++) {
    const symbol = symbols[i];
    onProgress?.(`Fetching ${symbol} (${i + 1}/${symbols.length})...`, allSequences.length);

    try {
      // Fetch historical candles via API
      const candles = await fetchCandlesViaAPI(symbol, isCrypto);

      if (candles.length < 50) {
        console.log(`Skipping ${symbol} - insufficient data (${candles.length} candles)`);
        continue;
      }

      // Convert to OHLCV
      const ohlcvData = candles.map(candleToOHLCV);

      // Create training sequences
      const { sequences, targets } = createTrainingData(ohlcvData, 30);

      allSequences.push(...sequences);
      allTargets.push(...targets);

      console.log(`${symbol}: Created ${sequences.length} training samples`);
    } catch (error) {
      console.error(`Error fetching ${symbol}:`, error);
    }

    // Small delay between requests
    await new Promise(resolve => setTimeout(resolve, 500));
  }

  return { sequences: allSequences, targets: allTargets };
}

/**
 * Save model state (in-memory, can be persisted via API)
 */
export function saveModelState(state: ModelState): void {
  cachedModelState = state;
  console.log('Model state saved to memory');

  // Try to persist to localStorage if in browser
  if (typeof window !== 'undefined') {
    try {
      localStorage.setItem('stockz-model-state', JSON.stringify(state));
    } catch {
      // localStorage might not be available
    }
  }
}

/**
 * Load model state from memory or localStorage
 */
export async function loadModelState(): Promise<ModelState | null> {
  // Return cached state if available
  if (cachedModelState) {
    return cachedModelState;
  }

  // Try to load from localStorage if in browser
  if (typeof window !== 'undefined') {
    try {
      const stored = localStorage.getItem('stockz-model-state');
      if (stored) {
        cachedModelState = JSON.parse(stored);
        return cachedModelState;
      }
    } catch {
      // localStorage might not be available
    }
  }

  return null;
}

/**
 * Check if model needs training
 */
export async function needsTraining(): Promise<boolean> {
  const state = await loadModelState();
  if (!state) return true;

  // Retrain if older than 24 hours or accuracy below 60%
  const ageHours = (Date.now() - state.trainedAt) / (1000 * 60 * 60);
  return ageHours > 24 || state.accuracy < 60;
}

/**
 * Get current training progress
 */
export function getTrainingProgress(): TrainingProgress {
  return { ...currentProgress };
}

/**
 * Check if training is in progress
 */
export function isTrainingInProgress(): boolean {
  return trainingInProgress;
}

/**
 * Main training function
 */
export async function trainModel(
  model: StockPredictionModel,
  options: {
    epochs?: number;
    includeStocks?: boolean;
    includeCrypto?: boolean;
    onProgress?: (progress: TrainingProgress) => void;
  } = {}
): Promise<ModelMetrics> {
  const {
    epochs = 30,
    includeStocks = true,
    includeCrypto = true,
    onProgress,
  } = options;

  if (trainingInProgress) {
    throw new Error('Training already in progress');
  }

  trainingInProgress = true;

  const updateProgress = (update: Partial<TrainingProgress>) => {
    currentProgress = { ...currentProgress, ...update };
    onProgress?.(currentProgress);
  };

  try {
    // Phase 1: Fetch data
    updateProgress({
      phase: 'fetching',
      progress: 0,
      message: 'Fetching historical data...',
    });

    let allSequences: number[][][] = [];
    let allTargets: number[][] = [];
    const trainedSymbols: string[] = [];

    if (includeStocks) {
      updateProgress({ message: 'Fetching stock data...' });
      const stockData = await fetchTrainingData(
        TRAINING_STOCKS,
        false,
        (msg, collected) => updateProgress({ message: msg, samplesCollected: collected })
      );
      allSequences.push(...stockData.sequences);
      allTargets.push(...stockData.targets);
      trainedSymbols.push(...TRAINING_STOCKS);
    }

    if (includeCrypto) {
      updateProgress({ message: 'Fetching crypto data...' });
      const cryptoData = await fetchTrainingData(
        TRAINING_CRYPTO,
        true,
        (msg, collected) => updateProgress({
          message: msg,
          samplesCollected: allSequences.length + collected
        })
      );
      allSequences.push(...cryptoData.sequences);
      allTargets.push(...cryptoData.targets);
      trainedSymbols.push(...TRAINING_CRYPTO);
    }

    if (allSequences.length < 100) {
      throw new Error(`Insufficient training data: only ${allSequences.length} samples collected`);
    }

    updateProgress({
      phase: 'preparing',
      progress: 20,
      message: `Preparing ${allSequences.length} training samples...`,
      samplesCollected: allSequences.length,
    });

    // Shuffle data
    const indices = Array.from({ length: allSequences.length }, (_, i) => i);
    for (let i = indices.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [indices[i], indices[j]] = [indices[j], indices[i]];
    }

    const shuffledSequences = indices.map(i => allSequences[i]);
    const shuffledTargets = indices.map(i => allTargets[i]);

    // Phase 2: Build model if needed
    if (!model.isReady()) {
      model.buildModel();
    }

    // Phase 3: Training
    updateProgress({
      phase: 'training',
      progress: 25,
      message: 'Training neural network...',
      currentEpoch: 0,
      totalEpochs: epochs,
    });

    const metrics = await model.train(
      shuffledSequences,
      shuffledTargets,
      {
        epochs,
        batchSize: 32,
        validationSplit: 0.2,
      },
      (epoch, logs) => {
        const progressPercent = 25 + (epoch / epochs) * 65;
        updateProgress({
          progress: progressPercent,
          currentEpoch: epoch + 1,
          totalEpochs: epochs,
          loss: logs?.loss,
          valLoss: logs?.val_loss,
          message: `Epoch ${epoch + 1}/${epochs} - Loss: ${logs?.loss?.toFixed(4) || 'N/A'}`,
        });
      }
    );

    // Phase 4: Save model state
    updateProgress({
      phase: 'validating',
      progress: 92,
      message: 'Saving model...',
    });

    const modelState: ModelState = {
      trainedAt: Date.now(),
      accuracy: metrics.accuracy,
      loss: metrics.loss,
      valLoss: metrics.valLoss,
      trainingSamples: allSequences.length,
      epochs,
      symbols: trainedSymbols,
    };

    saveModelState(modelState);

    // Try to save model to IndexedDB (browser) or file (Node)
    try {
      await model.saveModel('indexeddb://stock-prediction-model');
    } catch {
      console.log('IndexedDB save skipped (server-side)');
    }

    updateProgress({
      phase: 'complete',
      progress: 100,
      message: `Training complete! Accuracy: ${metrics.accuracy.toFixed(1)}%`,
      estimatedAccuracy: metrics.accuracy,
    });

    console.log('Training completed:', metrics);
    return metrics;

  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : 'Unknown error';
    updateProgress({
      phase: 'error',
      progress: 0,
      message: `Training failed: ${errorMessage}`,
    });
    throw error;
  } finally {
    trainingInProgress = false;
  }
}

/**
 * Quick training with synthetic data for immediate use
 * Used when we need predictions fast but model is untrained
 */
export async function quickSyntheticTraining(model: StockPredictionModel): Promise<ModelMetrics> {
  console.log('Running quick synthetic training...');

  // Generate synthetic training data that mimics realistic market patterns
  const sequences: number[][][] = [];
  const targets: number[][] = []; // Multi-timeframe targets

  // Generate patterns: trending up, trending down, sideways, volatile
  const patterns = ['trend_up', 'trend_down', 'sideways', 'volatile'];

  for (let p = 0; p < 100; p++) {
    const pattern = patterns[p % patterns.length];
    const sequence: number[][] = [];

    let price = 100;
    let rsi = 50;
    let macd = 0;

    for (let i = 0; i < 60; i++) {
      // Simulate price movement based on pattern (60 timesteps to match model)
      let change = 0;
      if (pattern === 'trend_up') change = 0.002 + Math.random() * 0.003;
      else if (pattern === 'trend_down') change = -0.002 - Math.random() * 0.003;
      else if (pattern === 'sideways') change = (Math.random() - 0.5) * 0.002;
      else change = (Math.random() - 0.5) * 0.01; // volatile

      price *= (1 + change);

      // Update indicators based on pattern
      if (pattern === 'trend_up') {
        rsi = Math.min(80, rsi + Math.random() * 2);
        macd += 0.1;
      } else if (pattern === 'trend_down') {
        rsi = Math.max(20, rsi - Math.random() * 2);
        macd -= 0.1;
      } else {
        rsi = 50 + (Math.random() - 0.5) * 20;
        macd = (Math.random() - 0.5) * 0.5;
      }

      // Create feature vector (30 features to match enhanced model)
      const features = [
        // PRICE & POSITION (5 features)
        change,                           // price change
        Math.random(),                    // price position in range
        Math.random() - 0.5,              // distance from SMA20
        Math.random() - 0.5,              // distance from SMA50
        Math.random() - 0.5,              // Bollinger position
        // MOMENTUM (7 features)
        rsi / 100,                        // RSI normalized
        macd,                             // MACD
        (Math.random() - 0.5) * 2,        // MACD histogram
        Math.random(),                    // Stoch K
        Math.random(),                    // Stoch D
        (Math.random() - 0.5) * 2,        // CCI
        Math.random(),                    // MFI
        // TREND (5 features)
        Math.random() * 0.5,              // ADX
        Math.random() - 0.5,              // +DI - -DI
        pattern === 'trend_up' ? 0.5 : pattern === 'trend_down' ? -0.5 : 0, // SMA trend
        (Math.random() - 0.5) * 0.5,      // EMA divergence
        Math.random() - 0.5,              // Trend strength
        // VOLATILITY (4 features)
        Math.random() * 0.02,             // ATR ratio
        Math.random() * 0.5,              // BB width
        Math.random() * 0.3,              // Keltner width
        Math.random() * 0.2,              // Donchian position
        // VOLUME (4 features)
        Math.random(),                    // Volume ratio
        Math.random() - 0.5,              // OBV slope
        Math.random() - 0.5,              // CMF
        Math.random() - 0.5,              // AD line slope
        // PATTERN & STRUCTURE (3 features)
        Math.random() - 0.5,              // Pivot distance
        Math.random() * 5,                // Consecutive days
        Math.random() - 0.5,              // Candle pattern
        // MARKET CONTEXT (2 features)
        Math.random() - 0.5,              // Market trend
        Math.random() * 0.5,              // Market volatility
      ];

      sequence.push(features);
    }

    sequences.push(sequence);

    // Multi-timeframe targets: day1, day5, day10
    // Each follows the pattern with increasing magnitude
    let baseTarget = 0;
    if (pattern === 'trend_up') baseTarget = 0.3 + Math.random() * 0.2;
    else if (pattern === 'trend_down') baseTarget = -0.3 - Math.random() * 0.2;
    else if (pattern === 'sideways') baseTarget = (Math.random() - 0.5) * 0.1;
    else baseTarget = (Math.random() - 0.5) * 0.4;

    // Multi-timeframe targets scale with time (day1, day5, day10)
    targets.push([
      baseTarget * 0.5,                           // day1: smaller movement
      baseTarget * 1.0,                           // day5: base movement
      baseTarget * 1.5 + (Math.random() - 0.5) * 0.2, // day10: larger + noise
    ]);
  }

  // Build and train
  if (!model.isReady()) {
    model.buildModel();
  }

  const metrics = await model.train(sequences, targets, {
    epochs: 20,
    batchSize: 16,
    validationSplit: 0.2,
  });

  // Save a basic state
  saveModelState({
    trainedAt: Date.now(),
    accuracy: metrics.accuracy,
    loss: metrics.loss,
    valLoss: metrics.valLoss,
    trainingSamples: sequences.length,
    epochs: 20,
    symbols: ['synthetic'],
  });

  console.log('Quick synthetic training complete:', metrics);
  return metrics;
}
