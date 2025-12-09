/**
 * Stock Prediction Neural Network Model
 * Uses LSTM architecture for time series prediction
 */

import * as tf from '@tensorflow/tfjs';

export interface ModelConfig {
  sequenceLength: number; // Number of time steps to look back
  featureCount: number; // Number of input features
  lstmUnits: number; // LSTM hidden units
  denseUnits: number; // Dense layer units
  dropout: number; // Dropout rate
  learningRate: number;
}

export interface TrainingConfig {
  epochs: number;
  batchSize: number;
  validationSplit: number;
  shuffle: boolean;
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

const DEFAULT_MODEL_CONFIG: ModelConfig = {
  sequenceLength: 30, // Look back 30 days
  featureCount: 15, // Number of technical indicators
  lstmUnits: 64,
  denseUnits: 32,
  dropout: 0.2,
  learningRate: 0.001,
};

const DEFAULT_TRAINING_CONFIG: TrainingConfig = {
  epochs: 50,
  batchSize: 32,
  validationSplit: 0.2,
  shuffle: true,
};

/**
 * Stock Prediction Model class
 */
export class StockPredictionModel {
  private model: tf.LayersModel | null = null;
  private config: ModelConfig;
  private metrics: ModelMetrics;
  private isTraining: boolean = false;
  private modelId: string;

  constructor(config: Partial<ModelConfig> = {}) {
    this.config = { ...DEFAULT_MODEL_CONFIG, ...config };
    this.modelId = `stock-model-${Date.now()}`;
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
   * Build the LSTM model architecture
   */
  buildModel(): void {
    // Clean up any existing model
    if (this.model) {
      this.model.dispose();
    }

    const { sequenceLength, featureCount, lstmUnits, denseUnits, dropout, learningRate } =
      this.config;

    // Build sequential model
    const model = tf.sequential();

    // First LSTM layer with return sequences for stacking
    model.add(
      tf.layers.lstm({
        units: lstmUnits,
        returnSequences: true,
        inputShape: [sequenceLength, featureCount],
        kernelInitializer: 'glorotUniform',
        recurrentInitializer: 'glorotUniform',
        kernelRegularizer: tf.regularizers.l2({ l2: 0.01 }),
      })
    );
    model.add(tf.layers.dropout({ rate: dropout }));

    // Second LSTM layer
    model.add(
      tf.layers.lstm({
        units: lstmUnits / 2,
        returnSequences: false,
        kernelInitializer: 'glorotUniform',
        recurrentInitializer: 'glorotUniform',
        kernelRegularizer: tf.regularizers.l2({ l2: 0.01 }),
      })
    );
    model.add(tf.layers.dropout({ rate: dropout }));

    // Dense layers for prediction
    model.add(
      tf.layers.dense({
        units: denseUnits,
        activation: 'relu',
        kernelRegularizer: tf.regularizers.l2({ l2: 0.01 }),
      })
    );
    model.add(tf.layers.dropout({ rate: dropout / 2 }));

    model.add(
      tf.layers.dense({
        units: 16,
        activation: 'relu',
      })
    );

    // Output layer - predicts price change percentage
    model.add(
      tf.layers.dense({
        units: 1,
        activation: 'tanh', // Output between -1 and 1 for normalized price change
      })
    );

    // Compile model
    const optimizer = tf.train.adam(learningRate);
    model.compile({
      optimizer,
      loss: 'meanSquaredError',
      metrics: ['mse', 'mae'],
    });

    this.model = model;
    console.log('Model built successfully');
    this.model.summary();
  }

  /**
   * Prepare training data from feature sequences
   */
  prepareTrainingData(
    sequences: number[][][], // [samples, timesteps, features]
    targets: number[] // Target price change percentages
  ): { xs: tf.Tensor3D; ys: tf.Tensor2D } {
    const xs = tf.tensor3d(sequences);
    const ys = tf.tensor2d(targets.map((t) => [t]));
    return { xs, ys };
  }

  /**
   * Train the model
   */
  async train(
    sequences: number[][][],
    targets: number[],
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
      const history = await this.model!.fit(xs, ys, {
        epochs: trainingConfig.epochs,
        batchSize: trainingConfig.batchSize,
        validationSplit: trainingConfig.validationSplit,
        shuffle: trainingConfig.shuffle,
        callbacks: {
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
        },
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
        epoch: trainingConfig.epochs,
      };

      console.log('Training completed', this.metrics);
    } finally {
      // Clean up tensors
      xs.dispose();
      ys.dispose();
      this.isTraining = false;
    }

    return this.metrics;
  }

  /**
   * Make a prediction
   */
  predict(sequence: number[][]): { prediction: number; confidence: number } {
    if (!this.model) {
      console.warn('Model not trained, using random prediction');
      return { prediction: 0, confidence: 0.5 };
    }

    // Ensure sequence has correct shape
    const paddedSequence = this.padSequence(sequence);

    const inputTensor = tf.tensor3d([paddedSequence]);
    const outputTensor = this.model.predict(inputTensor) as tf.Tensor;
    const prediction = outputTensor.dataSync()[0];

    // Calculate confidence based on model metrics and prediction magnitude
    const baseConfidence = Math.max(0.4, Math.min(0.95, this.metrics.accuracy / 100));
    const predictionStrength = Math.abs(prediction);
    const confidence = baseConfidence * (0.7 + 0.3 * Math.min(predictionStrength * 2, 1));

    // Clean up
    inputTensor.dispose();
    outputTensor.dispose();

    return {
      prediction: prediction * 10, // Scale back to percentage
      confidence: Math.min(0.95, Math.max(0.3, confidence)),
    };
  }

  /**
   * Batch prediction for multiple sequences
   */
  predictBatch(sequences: number[][][]): Array<{ prediction: number; confidence: number }> {
    if (!this.model) {
      return sequences.map(() => ({ prediction: 0, confidence: 0.5 }));
    }

    const paddedSequences = sequences.map((s) => this.padSequence(s));
    const inputTensor = tf.tensor3d(paddedSequences);
    const outputTensor = this.model.predict(inputTensor) as tf.Tensor;
    const predictions = outputTensor.dataSync();

    const results = Array.from(predictions).map((pred) => {
      const baseConfidence = Math.max(0.4, Math.min(0.95, this.metrics.accuracy / 100));
      const predictionStrength = Math.abs(pred);
      const confidence = baseConfidence * (0.7 + 0.3 * Math.min(predictionStrength * 2, 1));

      return {
        prediction: pred * 10,
        confidence: Math.min(0.95, Math.max(0.3, confidence)),
      };
    });

    inputTensor.dispose();
    outputTensor.dispose();

    return results;
  }

  /**
   * Pad or truncate sequence to correct length
   */
  private padSequence(sequence: number[][]): number[][] {
    const { sequenceLength, featureCount } = this.config;

    if (sequence.length >= sequenceLength) {
      return sequence.slice(-sequenceLength);
    }

    // Pad with zeros at the beginning
    const padding = Array(sequenceLength - sequence.length)
      .fill(null)
      .map(() => Array(featureCount).fill(0));

    return [...padding, ...sequence];
  }

  /**
   * Save model to IndexedDB (browser) or file system (Node)
   */
  async saveModel(path: string = 'indexeddb://stock-prediction-model'): Promise<void> {
    if (!this.model) {
      throw new Error('No model to save');
    }

    await this.model.save(path);
    console.log(`Model saved to ${path}`);
  }

  /**
   * Load model from storage
   */
  async loadModel(path: string = 'indexeddb://stock-prediction-model'): Promise<boolean> {
    try {
      this.model = await tf.loadLayersModel(path);
      console.log(`Model loaded from ${path}`);
      return true;
    } catch {
      // Expected on first run - no saved model exists yet
      console.log('No saved model found, will build a new one');
      return false;
    }
  }

  /**
   * Get current model metrics
   */
  getMetrics(): ModelMetrics {
    return { ...this.metrics };
  }

  /**
   * Check if model is ready for predictions
   */
  isReady(): boolean {
    return this.model !== null;
  }

  /**
   * Check if model is currently training
   */
  isCurrentlyTraining(): boolean {
    return this.isTraining;
  }

  /**
   * Get model configuration
   */
  getConfig(): ModelConfig {
    return { ...this.config };
  }

  /**
   * Dispose of the model and free memory
   */
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
