/**
 * LSTM Prediction Model for TensorFlow.js
 * Predicts future price movements based on historical OHLCV data
 */

import * as tf from '@tensorflow/tfjs';
import {
  Candle,
  DataStats,
  loadAssetData,
  calculateStats,
  normalizeCandles,
  createSequences,
  denormalizePrice,
  splitData,
} from './dataLoader';

// Model configuration
export interface ModelConfig {
  sequenceLength: number;   // Number of candles to look back (default: 60)
  features: number;         // Number of features per candle (OHLCV = 5)
  lstmUnits: number[];      // LSTM layer units (default: [128, 64])
  dropoutRate: number;      // Dropout rate (default: 0.2)
  learningRate: number;     // Learning rate (default: 0.001)
}

// Prediction result
export interface PredictionResult {
  predictedPrice: number;
  confidence: number;
  direction: 'up' | 'down' | 'neutral';
  percentChange: number;
}

// Training progress callback
export type TrainingCallback = (epoch: number, logs: tf.Logs | undefined) => void;

// Default model configuration
const DEFAULT_CONFIG: ModelConfig = {
  sequenceLength: 60,     // 60 minutes = 1 hour lookback
  features: 5,            // OHLCV
  lstmUnits: [128, 64],   // Two LSTM layers
  dropoutRate: 0.2,
  learningRate: 0.001,
};

/**
 * Build LSTM model architecture
 */
export function buildModel(config: ModelConfig = DEFAULT_CONFIG): tf.Sequential {
  const model = tf.sequential();

  // First LSTM layer with return sequences for stacking
  model.add(tf.layers.lstm({
    units: config.lstmUnits[0],
    returnSequences: config.lstmUnits.length > 1,
    inputShape: [config.sequenceLength, config.features],
  }));
  model.add(tf.layers.dropout({ rate: config.dropoutRate }));

  // Additional LSTM layers
  for (let i = 1; i < config.lstmUnits.length; i++) {
    const isLast = i === config.lstmUnits.length - 1;
    model.add(tf.layers.lstm({
      units: config.lstmUnits[i],
      returnSequences: !isLast,
    }));
    model.add(tf.layers.dropout({ rate: config.dropoutRate }));
  }

  // Dense layers for output
  model.add(tf.layers.dense({ units: 32, activation: 'relu' }));
  model.add(tf.layers.dense({ units: 1 }));  // Output: predicted close price (normalized)

  // Compile model
  model.compile({
    optimizer: tf.train.adam(config.learningRate),
    loss: 'meanSquaredError',
    metrics: ['mae'],  // Mean Absolute Error
  });

  return model;
}

/**
 * Train the model on asset data
 */
export async function trainModel(
  model: tf.Sequential,
  category: string,
  asset: string,
  config: ModelConfig = DEFAULT_CONFIG,
  epochs: number = 50,
  batchSize: number = 32,
  onProgress?: TrainingCallback
): Promise<{ history: tf.History; stats: DataStats }> {
  // Load and prepare data
  const candles = loadAssetData(category, asset);

  if (candles.length < config.sequenceLength + 100) {
    throw new Error(`Not enough data for ${asset}. Need at least ${config.sequenceLength + 100} candles, got ${candles.length}`);
  }

  console.log(`Training on ${candles.length} candles for ${asset}`);

  // Calculate normalization stats
  const stats = calculateStats(candles);

  // Normalize data
  const normalized = normalizeCandles(candles, stats);

  // Create sequences
  const { inputs, outputs } = createSequences(normalized, config.sequenceLength);

  // Split data
  const { train: trainInputs, validation: valInputs } = splitData(inputs);
  const { train: trainOutputs, validation: valOutputs } = splitData(outputs);

  console.log(`Training samples: ${trainInputs.length}, Validation samples: ${valInputs.length}`);

  // Convert to tensors
  const xTrain = tf.tensor3d(trainInputs);
  const yTrain = tf.tensor2d(trainOutputs.map(y => [y]));
  const xVal = tf.tensor3d(valInputs);
  const yVal = tf.tensor2d(valOutputs.map(y => [y]));

  // Train
  const history = await model.fit(xTrain, yTrain, {
    epochs,
    batchSize,
    validationData: [xVal, yVal],
    shuffle: true,
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        if (onProgress) {
          onProgress(epoch, logs);
        }
        console.log(`Epoch ${epoch + 1}/${epochs} - Loss: ${logs?.loss?.toFixed(6)} - Val Loss: ${logs?.val_loss?.toFixed(6)}`);
      },
    },
  });

  // Clean up tensors
  xTrain.dispose();
  yTrain.dispose();
  xVal.dispose();
  yVal.dispose();

  return { history, stats };
}

/**
 * Make a prediction for the next candle
 */
export function predict(
  model: tf.Sequential,
  recentCandles: Candle[],
  stats: DataStats,
  sequenceLength: number = 60
): PredictionResult {
  if (recentCandles.length < sequenceLength) {
    throw new Error(`Need at least ${sequenceLength} candles for prediction, got ${recentCandles.length}`);
  }

  // Get the last sequenceLength candles
  const lastCandles = recentCandles.slice(-sequenceLength);

  // Normalize
  const normalized = normalizeCandles(lastCandles, stats);

  // Create input tensor
  const input = normalized.map(c => [c.open, c.high, c.low, c.close, c.volume]);
  const inputTensor = tf.tensor3d([input]);

  // Predict
  const prediction = model.predict(inputTensor) as tf.Tensor;
  const normalizedPrice = prediction.dataSync()[0];

  // Clean up
  inputTensor.dispose();
  prediction.dispose();

  // Denormalize to get actual price
  const predictedPrice = denormalizePrice(normalizedPrice, stats);
  const currentPrice = lastCandles[lastCandles.length - 1].close;
  const percentChange = ((predictedPrice - currentPrice) / currentPrice) * 100;

  // Calculate direction and confidence
  let direction: 'up' | 'down' | 'neutral';
  if (percentChange > 0.1) {
    direction = 'up';
  } else if (percentChange < -0.1) {
    direction = 'down';
  } else {
    direction = 'neutral';
  }

  // Confidence based on how far from neutral (simplified)
  const confidence = Math.min(Math.abs(percentChange) * 10, 100);

  return {
    predictedPrice,
    confidence,
    direction,
    percentChange,
  };
}

/**
 * Save model to file system (Node.js) or IndexedDB (browser)
 */
export async function saveModel(model: tf.Sequential, modelName: string): Promise<void> {
  const savePath = `file://./models/${modelName}`;
  await model.save(savePath);
  console.log(`Model saved to ${savePath}`);
}

/**
 * Load model from file system (Node.js) or IndexedDB (browser)
 */
export async function loadModel(modelName: string): Promise<tf.Sequential> {
  const loadPath = `file://./models/${modelName}`;
  const model = await tf.loadLayersModel(loadPath) as tf.Sequential;
  console.log(`Model loaded from ${loadPath}`);
  return model;
}

/**
 * Create and train a model for an asset (convenience function)
 */
export async function createTrainedModel(
  category: string,
  asset: string,
  epochs: number = 50,
  onProgress?: TrainingCallback
): Promise<{ model: tf.Sequential; stats: DataStats }> {
  const config = DEFAULT_CONFIG;
  const model = buildModel(config);

  console.log('Model architecture:');
  model.summary();

  const { stats } = await trainModel(model, category, asset, config, epochs, 32, onProgress);

  return { model, stats };
}

/**
 * Batch predict for multiple time horizons
 */
export function predictMultiStep(
  model: tf.Sequential,
  recentCandles: Candle[],
  stats: DataStats,
  steps: number = 5,  // Predict next 5 candles
  sequenceLength: number = 60
): PredictionResult[] {
  const predictions: PredictionResult[] = [];
  let currentCandles = [...recentCandles];

  for (let i = 0; i < steps; i++) {
    const prediction = predict(model, currentCandles, stats, sequenceLength);
    predictions.push(prediction);

    // Add predicted candle to sequence for next prediction
    const lastCandle = currentCandles[currentCandles.length - 1];
    const predictedCandle: Candle = {
      timestamp: lastCandle.timestamp + 60000, // +1 minute
      open: lastCandle.close,
      high: Math.max(lastCandle.close, prediction.predictedPrice),
      low: Math.min(lastCandle.close, prediction.predictedPrice),
      close: prediction.predictedPrice,
      volume: lastCandle.volume, // Assume same volume
    };
    currentCandles = [...currentCandles.slice(1), predictedCandle];
  }

  return predictions;
}
