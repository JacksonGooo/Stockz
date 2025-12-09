/**
 * Continuous Training Service
 * Manages background training of the stock prediction model
 */

import { StockPredictionModel, ModelMetrics, createModel } from './model';
import { fetchHistoricalData, getSupportedSymbols, HistoricalDataPoint } from './dataProvider';
import {
  calculateAllIndicators,
  indicatorsToFeatures,
  OHLCV,
} from './indicators';

export interface TrainingStatus {
  isRunning: boolean;
  currentEpoch: number;
  totalEpochs: number;
  currentSymbol: string;
  symbolsProcessed: number;
  totalSymbols: number;
  metrics: ModelMetrics;
  startTime: Date | null;
  estimatedTimeRemaining: number | null;
}

export interface TrainingServiceConfig {
  trainingIntervalMs: number; // How often to retrain
  epochsPerCycle: number; // Epochs per training cycle
  batchSize: number;
  sequenceLength: number;
  minSamplesPerSymbol: number;
}

const DEFAULT_CONFIG: TrainingServiceConfig = {
  trainingIntervalMs: 60000, // Train every minute
  epochsPerCycle: 10,
  batchSize: 32,
  sequenceLength: 30,
  minSamplesPerSymbol: 100,
};

type TrainingCallback = (status: TrainingStatus) => void;

/**
 * Training Service class - manages continuous model training
 */
class TrainingService {
  private model: StockPredictionModel;
  private config: TrainingServiceConfig;
  private status: TrainingStatus;
  private trainingInterval: ReturnType<typeof setInterval> | null = null;
  private isInitialized: boolean = false;
  private callbacks: Set<TrainingCallback> = new Set();
  private trainingData: Map<string, { sequences: number[][][]; targets: number[] }> = new Map();

  constructor(config: Partial<TrainingServiceConfig> = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config };
    this.model = createModel({ sequenceLength: this.config.sequenceLength });
    this.status = {
      isRunning: false,
      currentEpoch: 0,
      totalEpochs: 0,
      currentSymbol: '',
      symbolsProcessed: 0,
      totalSymbols: 0,
      metrics: this.model.getMetrics(),
      startTime: null,
      estimatedTimeRemaining: null,
    };
  }

  /**
   * Initialize the training service (FAST - no data fetching)
   */
  async initialize(): Promise<void> {
    if (this.isInitialized) return;

    console.log('Initializing Training Service (lightweight mode)...');

    // Try to load existing model
    const loaded = await this.model.loadModel();
    if (!loaded) {
      // Build a fresh model if none exists
      this.model.buildModel();
    }

    // SKIP data preparation - will be done on-demand when predictions are requested
    // This makes initialization instant instead of taking 10+ seconds

    this.isInitialized = true;
    console.log('Training Service initialized (ready)');
  }

  /**
   * Prepare training data for all supported symbols
   */
  private async prepareAllTrainingData(): Promise<void> {
    const symbols = getSupportedSymbols();
    console.log(`Preparing training data for ${symbols.length} symbols...`);

    for (const symbol of symbols) {
      try {
        const data = await this.prepareTrainingDataForSymbol(symbol);
        if (data.sequences.length >= this.config.minSamplesPerSymbol) {
          this.trainingData.set(symbol, data);
        }
      } catch (error) {
        console.warn(`Failed to prepare data for ${symbol}:`, error);
      }
    }

    console.log(`Training data prepared for ${this.trainingData.size} symbols`);
  }

  /**
   * Prepare training data for a single symbol
   */
  private async prepareTrainingDataForSymbol(
    symbol: string
  ): Promise<{ sequences: number[][][]; targets: number[] }> {
    // Fetch historical data
    const historicalData = await fetchHistoricalData(symbol, 500);

    if (historicalData.length < this.config.sequenceLength + 10) {
      return { sequences: [], targets: [] };
    }

    const sequences: number[][][] = [];
    const targets: number[] = [];

    // Create sliding window sequences
    for (let i = this.config.sequenceLength; i < historicalData.length - 5; i++) {
      const windowData = historicalData.slice(i - this.config.sequenceLength, i);
      const futureData = historicalData.slice(i, i + 5);

      // Calculate indicators for each timestep in the window
      const sequence: number[][] = [];
      for (let j = 0; j < windowData.length; j++) {
        const dataUpToPoint = historicalData.slice(0, i - this.config.sequenceLength + j + 1);
        const indicators = calculateAllIndicators(dataUpToPoint);
        sequence.push(indicatorsToFeatures(indicators));
      }

      // Calculate target: average price change over next 5 days
      const currentPrice = windowData[windowData.length - 1].close;
      const futurePrice = futureData[futureData.length - 1].close;
      const priceChange = ((futurePrice - currentPrice) / currentPrice) * 100;

      // Normalize target to -1 to 1 range (assuming max 10% change)
      const normalizedTarget = Math.tanh(priceChange / 10);

      sequences.push(sequence);
      targets.push(normalizedTarget);
    }

    return { sequences, targets };
  }

  /**
   * Start continuous training
   */
  async start(): Promise<void> {
    if (!this.isInitialized) {
      await this.initialize();
    }

    if (this.status.isRunning) {
      console.log('Training is already running');
      return;
    }

    this.status.isRunning = true;
    this.status.startTime = new Date();
    this.notifyCallbacks();

    // Run initial training
    await this.runTrainingCycle();

    // Set up continuous training interval
    this.trainingInterval = setInterval(async () => {
      if (!this.model.isCurrentlyTraining()) {
        await this.runTrainingCycle();
      }
    }, this.config.trainingIntervalMs);

    console.log('Continuous training started');
  }

  /**
   * Run a single training cycle
   */
  private async runTrainingCycle(): Promise<void> {
    const symbols = Array.from(this.trainingData.keys());
    if (symbols.length === 0) {
      console.warn('No training data available');
      return;
    }

    this.status.totalSymbols = symbols.length;
    this.status.symbolsProcessed = 0;
    this.status.totalEpochs = this.config.epochsPerCycle;

    // Combine all training data
    const allSequences: number[][][] = [];
    const allTargets: number[] = [];

    for (const symbol of symbols) {
      const data = this.trainingData.get(symbol);
      if (data) {
        this.status.currentSymbol = symbol;
        allSequences.push(...data.sequences);
        allTargets.push(...data.targets);
        this.status.symbolsProcessed++;
        this.notifyCallbacks();
      }
    }

    if (allSequences.length === 0) return;

    // Shuffle data
    const indices = Array.from({ length: allSequences.length }, (_, i) => i);
    for (let i = indices.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [indices[i], indices[j]] = [indices[j], indices[i]];
    }

    const shuffledSequences = indices.map((i) => allSequences[i]);
    const shuffledTargets = indices.map((i) => allTargets[i]);

    // Train the model
    const startTime = Date.now();
    await this.model.train(
      shuffledSequences,
      shuffledTargets,
      {
        epochs: this.config.epochsPerCycle,
        batchSize: this.config.batchSize,
        validationSplit: 0.2,
      },
      (epoch, logs) => {
        this.status.currentEpoch = epoch + 1;
        this.status.metrics = this.model.getMetrics();

        // Estimate remaining time
        const elapsedMs = Date.now() - startTime;
        const msPerEpoch = elapsedMs / (epoch + 1);
        const remainingEpochs = this.config.epochsPerCycle - (epoch + 1);
        this.status.estimatedTimeRemaining = remainingEpochs * msPerEpoch;

        this.notifyCallbacks();
      }
    );

    // Save model after training
    try {
      await this.model.saveModel();
    } catch (error) {
      console.warn('Failed to save model:', error);
    }

    this.status.metrics = this.model.getMetrics();
    this.notifyCallbacks();
  }

  /**
   * Stop continuous training
   */
  stop(): void {
    if (this.trainingInterval) {
      clearInterval(this.trainingInterval);
      this.trainingInterval = null;
    }
    this.status.isRunning = false;
    this.notifyCallbacks();
    console.log('Training stopped');
  }

  /**
   * Get the trained model for predictions
   */
  getModel(): StockPredictionModel {
    return this.model;
  }

  /**
   * Get current training status
   */
  getStatus(): TrainingStatus {
    return { ...this.status };
  }

  /**
   * Subscribe to training status updates
   */
  subscribe(callback: TrainingCallback): () => void {
    this.callbacks.add(callback);
    return () => this.callbacks.delete(callback);
  }

  /**
   * Notify all subscribers of status change
   */
  private notifyCallbacks(): void {
    const status = this.getStatus();
    this.callbacks.forEach((cb) => cb(status));
  }

  /**
   * Refresh training data for a specific symbol
   */
  async refreshSymbolData(symbol: string): Promise<void> {
    try {
      const data = await this.prepareTrainingDataForSymbol(symbol);
      if (data.sequences.length >= this.config.minSamplesPerSymbol) {
        this.trainingData.set(symbol, data);
      }
    } catch (error) {
      console.warn(`Failed to refresh data for ${symbol}:`, error);
    }
  }

  /**
   * Get training data statistics
   */
  getDataStats(): { symbol: string; samples: number }[] {
    return Array.from(this.trainingData.entries()).map(([symbol, data]) => ({
      symbol,
      samples: data.sequences.length,
    }));
  }

  /**
   * Clean up resources
   */
  dispose(): void {
    this.stop();
    this.model.dispose();
    this.trainingData.clear();
    this.callbacks.clear();
  }
}

// Singleton instance
let trainingServiceInstance: TrainingService | null = null;

/**
 * Get or create the training service instance
 */
export function getTrainingService(
  config?: Partial<TrainingServiceConfig>
): TrainingService {
  if (!trainingServiceInstance) {
    trainingServiceInstance = new TrainingService(config);
  }
  return trainingServiceInstance;
}

/**
 * Reset the training service (for testing)
 */
export function resetTrainingService(): void {
  if (trainingServiceInstance) {
    trainingServiceInstance.dispose();
    trainingServiceInstance = null;
  }
}
