/**
 * LSTM Model Training Script
 * Trains TensorFlow.js models for price prediction
 *
 * Usage:
 *   node train_model.js <category> <asset> [epochs]
 *
 * Examples:
 *   node train_model.js Crypto BTC 100
 *   node train_model.js "Stock Market" AAPL 50
 */

const tf = require('@tensorflow/tfjs');
const fs = require('fs');
const path = require('path');

const DATA_DIR = path.join(__dirname, '..', 'Data');
const MODELS_DIR = path.join(__dirname, '..', 'models');

// Model configuration
const CONFIG = {
  sequenceLength: 60,   // 60 minutes lookback
  features: 5,          // OHLCV
  lstmUnits: [128, 64], // Two LSTM layers
  dropoutRate: 0.2,
  learningRate: 0.001,
  trainRatio: 0.8,
  maxSamples: 100000,   // Limit samples for memory management
  batchSize: 64,
};

// Ensure models directory exists
if (!fs.existsSync(MODELS_DIR)) {
  fs.mkdirSync(MODELS_DIR, { recursive: true });
}

/**
 * Load candle data for an asset
 */
function loadAssetData(category, asset) {
  const assetDir = path.join(DATA_DIR, category, asset);

  if (!fs.existsSync(assetDir)) {
    throw new Error(`No data found for ${category}/${asset}`);
  }

  const allCandles = [];
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
 * Calculate normalization statistics
 */
function calculateStats(candles) {
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
 * Normalize candles to 0-1 range
 */
function normalizeCandles(candles, stats) {
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
 * Create training sequences
 */
function createSequences(normalizedCandles, sequenceLength) {
  const inputs = [];
  const outputs = [];

  for (let i = sequenceLength; i < normalizedCandles.length; i++) {
    const sequence = normalizedCandles.slice(i - sequenceLength, i).map(c => [
      c.open, c.high, c.low, c.close, c.volume
    ]);
    inputs.push(sequence);
    outputs.push(normalizedCandles[i].close);
  }

  return { inputs, outputs };
}

/**
 * Build LSTM model
 */
function buildModel() {
  const model = tf.sequential();

  // First LSTM layer
  model.add(tf.layers.lstm({
    units: CONFIG.lstmUnits[0],
    returnSequences: true,
    inputShape: [CONFIG.sequenceLength, CONFIG.features],
  }));
  model.add(tf.layers.dropout({ rate: CONFIG.dropoutRate }));

  // Second LSTM layer
  model.add(tf.layers.lstm({
    units: CONFIG.lstmUnits[1],
    returnSequences: false,
  }));
  model.add(tf.layers.dropout({ rate: CONFIG.dropoutRate }));

  // Dense layers
  model.add(tf.layers.dense({ units: 32, activation: 'relu' }));
  model.add(tf.layers.dense({ units: 1 }));

  model.compile({
    optimizer: tf.train.adam(CONFIG.learningRate),
    loss: 'meanSquaredError',
    metrics: ['mae'],
  });

  return model;
}

/**
 * Main training function
 */
async function trainModel(category, asset, epochs = 50) {
  console.log('============================================================');
  console.log('S.U.P.I.D. LSTM Model Training');
  console.log('============================================================');
  console.log(`Asset: ${category}/${asset}`);
  console.log(`Epochs: ${epochs}`);
  console.log(`Sequence Length: ${CONFIG.sequenceLength}`);
  console.log('============================================================\n');

  // Load data
  console.log('Loading data...');
  const candles = loadAssetData(category, asset);
  console.log(`Loaded ${candles.length.toLocaleString()} candles`);

  if (candles.length < CONFIG.sequenceLength + 100) {
    throw new Error(`Not enough data. Need at least ${CONFIG.sequenceLength + 100} candles.`);
  }

  // Calculate stats
  const stats = calculateStats(candles);
  console.log(`Price range: $${stats.minPrice.toFixed(2)} - $${stats.maxPrice.toFixed(2)}`);

  // Normalize
  console.log('Normalizing data...');
  const normalized = normalizeCandles(candles, stats);

  // Create sequences
  console.log('Creating training sequences...');
  let { inputs, outputs } = createSequences(normalized, CONFIG.sequenceLength);
  console.log(`Created ${inputs.length.toLocaleString()} sequences`);

  // Sample if too many sequences (memory management)
  if (inputs.length > CONFIG.maxSamples) {
    console.log(`Sampling ${CONFIG.maxSamples.toLocaleString()} sequences (memory optimization)...`);
    const step = Math.floor(inputs.length / CONFIG.maxSamples);
    const sampledInputs = [];
    const sampledOutputs = [];
    for (let i = 0; i < inputs.length; i += step) {
      if (sampledInputs.length >= CONFIG.maxSamples) break;
      sampledInputs.push(inputs[i]);
      sampledOutputs.push(outputs[i]);
    }
    inputs = sampledInputs;
    outputs = sampledOutputs;
    console.log(`Sampled to ${inputs.length.toLocaleString()} sequences`);
  }

  // Split data
  const splitIndex = Math.floor(inputs.length * CONFIG.trainRatio);
  const trainInputs = inputs.slice(0, splitIndex);
  const trainOutputs = outputs.slice(0, splitIndex);
  const valInputs = inputs.slice(splitIndex);
  const valOutputs = outputs.slice(splitIndex);

  console.log(`Training samples: ${trainInputs.length.toLocaleString()}`);
  console.log(`Validation samples: ${valInputs.length.toLocaleString()}`);

  // Create tensors
  console.log('\nCreating tensors...');
  const xTrain = tf.tensor3d(trainInputs);
  const yTrain = tf.tensor2d(trainOutputs.map(y => [y]));
  const xVal = tf.tensor3d(valInputs);
  const yVal = tf.tensor2d(valOutputs.map(y => [y]));

  // Build model
  console.log('Building model...');
  const model = buildModel();
  model.summary();

  // Train
  console.log('\n============================================================');
  console.log('Training started...');
  console.log('============================================================\n');

  const startTime = Date.now();

  await model.fit(xTrain, yTrain, {
    epochs,
    batchSize: CONFIG.batchSize,
    validationData: [xVal, yVal],
    shuffle: true,
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
        console.log(
          `Epoch ${(epoch + 1).toString().padStart(3)}/${epochs} | ` +
          `Loss: ${logs.loss.toFixed(6)} | ` +
          `Val Loss: ${logs.val_loss.toFixed(6)} | ` +
          `MAE: ${logs.mae.toFixed(6)} | ` +
          `Time: ${elapsed}s`
        );
      },
    },
  });

  const totalTime = ((Date.now() - startTime) / 1000).toFixed(1);
  console.log(`\nTraining completed in ${totalTime} seconds`);

  // Save model
  const modelName = `${asset.toLowerCase()}_lstm`;
  const modelPath = path.join(MODELS_DIR, modelName);
  await model.save(`file://${modelPath}`);
  console.log(`Model saved to: ${modelPath}`);

  // Save stats for later use in predictions
  const statsPath = path.join(MODELS_DIR, `${modelName}_stats.json`);
  fs.writeFileSync(statsPath, JSON.stringify({
    ...stats,
    asset,
    category,
    sequenceLength: CONFIG.sequenceLength,
    trainedAt: new Date().toISOString(),
    totalCandles: candles.length,
  }, null, 2));
  console.log(`Stats saved to: ${statsPath}`);

  // Clean up
  xTrain.dispose();
  yTrain.dispose();
  xVal.dispose();
  yVal.dispose();

  console.log('\n============================================================');
  console.log('Training Complete!');
  console.log('============================================================');

  return { model, stats };
}

// Run from command line
const args = process.argv.slice(2);
if (args.length < 2) {
  console.log('Usage: node train_model.js <category> <asset> [epochs]');
  console.log('');
  console.log('Examples:');
  console.log('  node train_model.js Crypto BTC 100');
  console.log('  node train_model.js "Stock Market" AAPL 50');
  console.log('  node train_model.js Commodities GOLD');
  process.exit(1);
}

const category = args[0];
const asset = args[1];
const epochs = parseInt(args[2]) || 50;

trainModel(category, asset, epochs)
  .then(() => process.exit(0))
  .catch(err => {
    console.error('Training failed:', err.message);
    process.exit(1);
  });
