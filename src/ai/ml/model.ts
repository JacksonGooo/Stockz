/**
 * Stock Prediction Neural Network Model - Enhanced Version
 * Features:
 * - Attention mechanism for focusing on important time steps
 * - Larger network capacity (100K+ params)
 * - Extended lookback (60 days)
 * - Multi-timeframe outputs (1-day, 5-day, 10-day predictions)
 * - Bidirectional LSTM for better pattern recognition
 */

import * as tf from '@tensorflow/tfjs';
import { getFeatureCount } from './indicators';

export interface ModelConfig {
  sequenceLength: number; // Number of time steps to look back
  featureCount: number; // Number of input features
  lstmUnits: number; // LSTM hidden units
  attentionHeads: number; // Number of attention heads
  denseUnits: number; // Dense layer units
  dropout: number; // Dropout rate
  learningRate: number;
  useAttention: boolean; // Enable attention mechanism
  useBidirectional: boolean; // Use bidirectional LSTM
  outputTimeframes: number[]; // Prediction timeframes in days
}

export interface TrainingConfig {
  epochs: number;
  batchSize: number;
  validationSplit: number;
  shuffle: boolean;
  earlyStopping: boolean;
  patience: number;
}

export interface ModelMetrics {
  loss: number;
  valLoss: number;
  mse: number;
  mae: number;
  accuracy: number;
  trainingSamples: number;
  lastTrainedAt: Date;
  epoch: number;
}

export interface MultiTimeframePrediction {
  day1: { prediction: number; confidence: number };
  day5: { prediction: number; confidence: number };
  day10: { prediction: number; confidence: number };
  overall: { direction: 'up' | 'down' | 'neutral'; strength: number };
}

const DEFAULT_MODEL_CONFIG: ModelConfig = {
  sequenceLength: 60, // Look back 60 days (increased from 30)
  featureCount: getFeatureCount(), // 30 features
  lstmUnits: 128, // Increased from 64
  attentionHeads: 4, // Multi-head attention
  denseUnits: 64, // Increased from 32
  dropout: 0.3, // Slightly higher dropout for regularization
  learningRate: 0.0005, // Lower learning rate for stability
  useAttention: true,
  useBidirectional: true,
  outputTimeframes: [1, 5, 10], // Predict 1, 5, and 10 days ahead
};

const DEFAULT_TRAINING_CONFIG: TrainingConfig = {
  epochs: 100, // Increased from 50
  batchSize: 64, // Increased from 32
  validationSplit: 0.2,
  shuffle: true,
  earlyStopping: true,
  patience: 10,
};

/**
 * Custom Attention Layer
 * Implements scaled dot-product attention for time series
 */
class AttentionLayer extends tf.layers.Layer {
  private units: number;
  private Wq!: tf.LayerVariable;
  private Wk!: tf.LayerVariable;
  private Wv!: tf.LayerVariable;
  private Wo!: tf.LayerVariable;

  constructor(config: { units: number; name?: string }) {
    super(config);
    this.units = config.units;
  }

  build(inputShape: tf.Shape | tf.Shape[]): void {
    const shape = Array.isArray(inputShape[0]) ? inputShape[0] : inputShape;
    const inputDim = shape[shape.length - 1] as number;

    this.Wq = this.addWeight('Wq', [inputDim, this.units], 'float32', tf.initializers.glorotUniform({}));
    this.Wk = this.addWeight('Wk', [inputDim, this.units], 'float32', tf.initializers.glorotUniform({}));
    this.Wv = this.addWeight('Wv', [inputDim, this.units], 'float32', tf.initializers.glorotUniform({}));
    this.Wo = this.addWeight('Wo', [this.units, inputDim], 'float32', tf.initializers.glorotUniform({}));

    super.build(inputShape);
  }

  call(inputs: tf.Tensor | tf.Tensor[], kwargs?: Record<string, unknown>): tf.Tensor {
    return tf.tidy(() => {
      const input = Array.isArray(inputs) ? inputs[0] : inputs;
      const inputShape = input.shape;
      const batchSize = inputShape[0] || 1;
      const seqLen = inputShape[1] || 1;
      const features = inputShape[2] || 1;

      // Reshape for 2D matrix multiplication: [batch * seq, features]
      const flatInput = input.reshape([batchSize * seqLen, features]);

      // Query, Key, Value projections
      const Q = tf.matMul(flatInput, this.Wq.read()).reshape([batchSize, seqLen, this.units]);
      const K = tf.matMul(flatInput, this.Wk.read()).reshape([batchSize, seqLen, this.units]);
      const V = tf.matMul(flatInput, this.Wv.read()).reshape([batchSize, seqLen, this.units]);

      // Scaled dot-product attention
      const scale = Math.sqrt(this.units);
      const scores = tf.div(tf.matMul(Q, K, false, true), scale);
      const weights = tf.softmax(scores, -1);

      // Apply attention to values
      const attended = tf.matMul(weights, V);

      // Output projection - reshape for 2D matmul then back to 3D
      const flatAttended = attended.reshape([batchSize * seqLen, this.units]);
      const output = tf.matMul(flatAttended, this.Wo.read()).reshape([batchSize, seqLen, this.units]);

      // Residual connection (note: requires same shape, so we project input)
      const flatInputProjected = tf.matMul(flatInput, this.Wq.read()).reshape([batchSize, seqLen, this.units]);
      return tf.add(flatInputProjected, output) as tf.Tensor;
    });
  }

  computeOutputShape(inputShape: tf.Shape | tf.Shape[]): tf.Shape {
    // Handle both single shape and array of shapes
    if (Array.isArray(inputShape[0])) {
      return inputShape[0] as tf.Shape;
    }
    return inputShape as tf.Shape;
  }

  getConfig(): tf.serialization.ConfigDict {
    return { ...super.getConfig(), units: this.units };
  }

  static get className(): string {
    return 'AttentionLayer';
  }
}

// Register custom layer
tf.serialization.registerClass(AttentionLayer);

/**
 * Enhanced Stock Prediction Model class
 */
export class StockPredictionModel {
  private model: tf.LayersModel | null = null;
  private config: ModelConfig;
  private metrics: ModelMetrics;
  private isTraining: boolean = false;
  private modelId: string;

  constructor(config: Partial<ModelConfig> = {}) {
    this.config = { ...DEFAULT_MODEL_CONFIG, ...config };
    this.modelId = `stock-model-v2-${Date.now()}`;
    this.metrics = {
      loss: 1,
      valLoss: 1,
      mse: 1,
      mae: 1,
      accuracy: 0,
      trainingSamples: 0,
      lastTrainedAt: new Date(),
      epoch: 0,
    };
  }

  /**
   * Build the enhanced model architecture
   */
  buildModel(): void {
    if (this.model) {
      this.model.dispose();
    }

    const {
      sequenceLength,
      featureCount,
      lstmUnits,
      denseUnits,
      dropout,
      learningRate,
      useAttention,
      useBidirectional,
      outputTimeframes,
    } = this.config;

    // Input layer
    const input = tf.input({ shape: [sequenceLength, featureCount], name: 'input' });

    let x: tf.SymbolicTensor = input;

    // Layer Normalization for input stability
    x = tf.layers.layerNormalization({ name: 'input_norm' }).apply(x) as tf.SymbolicTensor;

    // First LSTM block (Bidirectional if enabled)
    if (useBidirectional) {
      x = tf.layers.bidirectional({
        layer: tf.layers.lstm({
          units: lstmUnits,
          returnSequences: true,
          kernelInitializer: 'glorotUniform',
          recurrentInitializer: 'orthogonal',
          kernelRegularizer: tf.regularizers.l2({ l2: 0.001 }),
          recurrentRegularizer: tf.regularizers.l2({ l2: 0.001 }),
        }) as tf.RNN,
        name: 'bidirectional_lstm_1',
      }).apply(x) as tf.SymbolicTensor;
    } else {
      x = tf.layers.lstm({
        units: lstmUnits,
        returnSequences: true,
        kernelInitializer: 'glorotUniform',
        recurrentInitializer: 'orthogonal',
        kernelRegularizer: tf.regularizers.l2({ l2: 0.001 }),
        name: 'lstm_1',
      }).apply(x) as tf.SymbolicTensor;
    }

    x = tf.layers.dropout({ rate: dropout, name: 'dropout_1' }).apply(x) as tf.SymbolicTensor;
    x = tf.layers.layerNormalization({ name: 'norm_1' }).apply(x) as tf.SymbolicTensor;

    // Attention mechanism
    if (useAttention) {
      const attentionUnits = useBidirectional ? lstmUnits * 2 : lstmUnits;
      x = new AttentionLayer({ units: attentionUnits, name: 'attention' }).apply(x) as tf.SymbolicTensor;
      x = tf.layers.dropout({ rate: dropout * 0.5, name: 'attention_dropout' }).apply(x) as tf.SymbolicTensor;
    }

    // Second LSTM block
    if (useBidirectional) {
      x = tf.layers.bidirectional({
        layer: tf.layers.lstm({
          units: lstmUnits / 2,
          returnSequences: true,
          kernelInitializer: 'glorotUniform',
          recurrentInitializer: 'orthogonal',
          kernelRegularizer: tf.regularizers.l2({ l2: 0.001 }),
        }) as tf.RNN,
        name: 'bidirectional_lstm_2',
      }).apply(x) as tf.SymbolicTensor;
    } else {
      x = tf.layers.lstm({
        units: lstmUnits / 2,
        returnSequences: true,
        kernelInitializer: 'glorotUniform',
        recurrentInitializer: 'orthogonal',
        name: 'lstm_2',
      }).apply(x) as tf.SymbolicTensor;
    }

    x = tf.layers.dropout({ rate: dropout, name: 'dropout_2' }).apply(x) as tf.SymbolicTensor;
    x = tf.layers.layerNormalization({ name: 'norm_2' }).apply(x) as tf.SymbolicTensor;

    // Third LSTM block - final sequence processing
    x = tf.layers.lstm({
      units: lstmUnits / 4,
      returnSequences: false,
      kernelInitializer: 'glorotUniform',
      recurrentInitializer: 'orthogonal',
      name: 'lstm_3',
    }).apply(x) as tf.SymbolicTensor;

    x = tf.layers.dropout({ rate: dropout, name: 'dropout_3' }).apply(x) as tf.SymbolicTensor;

    // Dense layers with skip connections
    const dense1 = tf.layers.dense({
      units: denseUnits * 2,
      activation: 'relu',
      kernelRegularizer: tf.regularizers.l2({ l2: 0.001 }),
      name: 'dense_1',
    }).apply(x) as tf.SymbolicTensor;

    const dense1_drop = tf.layers.dropout({ rate: dropout * 0.5, name: 'dense_1_dropout' }).apply(dense1) as tf.SymbolicTensor;

    const dense2 = tf.layers.dense({
      units: denseUnits,
      activation: 'relu',
      kernelRegularizer: tf.regularizers.l2({ l2: 0.001 }),
      name: 'dense_2',
    }).apply(dense1_drop) as tf.SymbolicTensor;

    const dense2_drop = tf.layers.dropout({ rate: dropout * 0.5, name: 'dense_2_dropout' }).apply(dense2) as tf.SymbolicTensor;

    const dense3 = tf.layers.dense({
      units: denseUnits / 2,
      activation: 'relu',
      name: 'dense_3',
    }).apply(dense2_drop) as tf.SymbolicTensor;

    // Multi-timeframe output heads
    const outputs: tf.SymbolicTensor[] = [];

    for (let i = 0; i < outputTimeframes.length; i++) {
      const timeframe = outputTimeframes[i];
      const output = tf.layers.dense({
        units: 1,
        activation: 'tanh',
        name: `output_day${timeframe}`,
      }).apply(dense3) as tf.SymbolicTensor;
      outputs.push(output);
    }

    // Concatenate all outputs
    const finalOutput = outputs.length > 1
      ? tf.layers.concatenate({ name: 'output_concat' }).apply(outputs) as tf.SymbolicTensor
      : outputs[0];

    // Build model
    this.model = tf.model({ inputs: input, outputs: finalOutput });

    // Compile with custom loss for multi-output
    const optimizer = tf.train.adam(learningRate);
    this.model.compile({
      optimizer,
      loss: 'meanSquaredError',
      metrics: ['mse', 'mae'],
    });

    console.log('Model built successfully');
    this.model.summary();
  }

  /**
   * Prepare training data from feature sequences
   */
  prepareTrainingData(
    sequences: number[][][],
    targets: number[][] // Now multi-dimensional for multiple timeframes
  ): { xs: tf.Tensor3D; ys: tf.Tensor2D } {
    const xs = tf.tensor3d(sequences);
    const ys = tf.tensor2d(targets);
    return { xs, ys };
  }

  /**
   * Train the model with enhanced configuration
   */
  async train(
    sequences: number[][][],
    targets: number[][],
    config: Partial<TrainingConfig> = {},
    onProgress?: (epoch: number, logs: tf.Logs) => void
  ): Promise<ModelMetrics> {
    if (!this.model) {
      this.buildModel();
    }

    if (this.isTraining) {
      console.warn('Model is already training');
      return this.metrics;
    }

    this.isTraining = true;
    const trainingConfig = { ...DEFAULT_TRAINING_CONFIG, ...config };

    const { xs, ys } = this.prepareTrainingData(sequences, targets);

    try {
      const callbacks: tf.CustomCallbackArgs = {
        onEpochEnd: (epoch, logs) => {
          if (logs) {
            this.metrics = {
              loss: logs.loss || 1,
              valLoss: logs.val_loss || 1,
              mse: logs.mse || 1,
              mae: logs.mae || 1,
              accuracy: Math.max(0, 1 - (logs.val_loss || 1)) * 100,
              trainingSamples: sequences.length,
              lastTrainedAt: new Date(),
              epoch: epoch + 1,
            };
            onProgress?.(epoch, logs);
          }
        },
      };

      // Run training
      const history = await this.model!.fit(xs, ys, {
        epochs: trainingConfig.epochs,
        batchSize: trainingConfig.batchSize,
        validationSplit: trainingConfig.validationSplit,
        shuffle: trainingConfig.shuffle,
        callbacks: callbacks,
      });

      // Calculate final metrics
      const finalLoss = history.history.loss[history.history.loss.length - 1] as number;
      const finalValLoss = history.history.val_loss
        ? (history.history.val_loss[history.history.val_loss.length - 1] as number)
        : finalLoss;

      this.metrics = {
        ...this.metrics,
        loss: finalLoss,
        valLoss: finalValLoss,
        accuracy: Math.max(0, Math.min(100, (1 - Math.sqrt(finalValLoss)) * 100)),
        trainingSamples: sequences.length,
        lastTrainedAt: new Date(),
        epoch: history.history.loss.length,
      };

      console.log('Training completed', this.metrics);
    } finally {
      xs.dispose();
      ys.dispose();
      this.isTraining = false;
    }

    return this.metrics;
  }

  /**
   * Make predictions for multiple timeframes
   */
  predict(sequence: number[][]): MultiTimeframePrediction {
    if (!this.model) {
      console.warn('Model not trained, using random prediction');
      return this.getDefaultPrediction();
    }

    const paddedSequence = this.padSequence(sequence);
    const inputTensor = tf.tensor3d([paddedSequence]);
    const outputTensor = this.model.predict(inputTensor) as tf.Tensor;
    const predictions = outputTensor.dataSync();

    const baseConfidence = Math.max(0.4, Math.min(0.95, this.metrics.accuracy / 100));

    const result: MultiTimeframePrediction = {
      day1: this.createPrediction(predictions[0], baseConfidence),
      day5: this.createPrediction(predictions[1] ?? predictions[0], baseConfidence * 0.9),
      day10: this.createPrediction(predictions[2] ?? predictions[0], baseConfidence * 0.8),
      overall: { direction: 'neutral', strength: 0 },
    };

    // Calculate overall direction and strength
    const avgPrediction = (result.day1.prediction + result.day5.prediction + result.day10.prediction) / 3;
    result.overall = {
      direction: avgPrediction > 1 ? 'up' : avgPrediction < -1 ? 'down' : 'neutral',
      strength: Math.min(1, Math.abs(avgPrediction) / 5),
    };

    inputTensor.dispose();
    outputTensor.dispose();

    return result;
  }

  /**
   * Single timeframe prediction (backwards compatible)
   */
  predictSingle(sequence: number[][]): { prediction: number; confidence: number } {
    const multiPrediction = this.predict(sequence);
    return multiPrediction.day5; // Default to 5-day prediction
  }

  /**
   * Batch prediction for multiple sequences
   */
  predictBatch(sequences: number[][][]): MultiTimeframePrediction[] {
    if (!this.model) {
      return sequences.map(() => this.getDefaultPrediction());
    }

    const paddedSequences = sequences.map((s) => this.padSequence(s));
    const inputTensor = tf.tensor3d(paddedSequences);
    const outputTensor = this.model.predict(inputTensor) as tf.Tensor;
    const allPredictions = outputTensor.dataSync();

    const numOutputs = this.config.outputTimeframes.length;
    const baseConfidence = Math.max(0.4, Math.min(0.95, this.metrics.accuracy / 100));

    const results: MultiTimeframePrediction[] = [];

    for (let i = 0; i < sequences.length; i++) {
      const offset = i * numOutputs;
      const result: MultiTimeframePrediction = {
        day1: this.createPrediction(allPredictions[offset], baseConfidence),
        day5: this.createPrediction(allPredictions[offset + 1] ?? allPredictions[offset], baseConfidence * 0.9),
        day10: this.createPrediction(allPredictions[offset + 2] ?? allPredictions[offset], baseConfidence * 0.8),
        overall: { direction: 'neutral', strength: 0 },
      };

      const avgPrediction = (result.day1.prediction + result.day5.prediction + result.day10.prediction) / 3;
      result.overall = {
        direction: avgPrediction > 1 ? 'up' : avgPrediction < -1 ? 'down' : 'neutral',
        strength: Math.min(1, Math.abs(avgPrediction) / 5),
      };

      results.push(result);
    }

    inputTensor.dispose();
    outputTensor.dispose();

    return results;
  }

  private createPrediction(rawPrediction: number, baseConfidence: number): { prediction: number; confidence: number } {
    const prediction = rawPrediction * 10; // Scale to percentage
    const predictionStrength = Math.abs(rawPrediction);
    const confidence = baseConfidence * (0.7 + 0.3 * Math.min(predictionStrength * 2, 1));

    return {
      prediction,
      confidence: Math.min(0.95, Math.max(0.3, confidence)),
    };
  }

  private getDefaultPrediction(): MultiTimeframePrediction {
    return {
      day1: { prediction: 0, confidence: 0.5 },
      day5: { prediction: 0, confidence: 0.5 },
      day10: { prediction: 0, confidence: 0.5 },
      overall: { direction: 'neutral', strength: 0 },
    };
  }

  /**
   * Pad or truncate sequence to correct length
   */
  private padSequence(sequence: number[][]): number[][] {
    const { sequenceLength, featureCount } = this.config;

    if (sequence.length >= sequenceLength) {
      return sequence.slice(-sequenceLength);
    }

    const padding = Array(sequenceLength - sequence.length)
      .fill(null)
      .map(() => Array(featureCount).fill(0));

    return [...padding, ...sequence];
  }

  /**
   * Save model to storage
   */
  async saveModel(path: string = 'indexeddb://stock-prediction-model-v2'): Promise<void> {
    if (!this.model) {
      throw new Error('No model to save');
    }

    await this.model.save(path);
    console.log(`Model saved to ${path}`);
  }

  /**
   * Load model from storage
   */
  async loadModel(path: string = 'indexeddb://stock-prediction-model-v2'): Promise<boolean> {
    try {
      this.model = await tf.loadLayersModel(path);

      const optimizer = tf.train.adam(this.config.learningRate);
      this.model.compile({
        optimizer,
        loss: 'meanSquaredError',
        metrics: ['mse', 'mae'],
      });

      console.log(`Model loaded from ${path}`);
      return true;
    } catch {
      console.log('No saved model found, will build a new one');
      return false;
    }
  }

  getMetrics(): ModelMetrics {
    return { ...this.metrics };
  }

  isReady(): boolean {
    return this.model !== null;
  }

  isCurrentlyTraining(): boolean {
    return this.isTraining;
  }

  getConfig(): ModelConfig {
    return { ...this.config };
  }

  dispose(): void {
    if (this.model) {
      this.model.dispose();
      this.model = null;
    }
  }
}

/**
 * Create a new model instance with default config
 */
export function createModel(config?: Partial<ModelConfig>): StockPredictionModel {
  return new StockPredictionModel(config);
}
