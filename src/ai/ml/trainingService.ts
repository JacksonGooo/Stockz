/**
 * Enhanced Training Service
 * Manages background training of the enhanced stock prediction model
 * Supports multi-timeframe predictions and market context
 */

import { StockPredictionModel, ModelMetrics, createModel, ModelConfig } from './model';
import { fetchHistoricalData, getSupportedSymbols, HistoricalDataPoint } from './dataProvider';
import {
  calculateAllIndicators,
  indicatorsToFeatures,
  getFeatureCount,
  OHLCV,
  MarketContext,
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
  phase: 'idle' | 'preparing' | 'training' | 'validating';
}

export interface TrainingServiceConfig {
  trainingIntervalMs: number;
  epochsPerCycle: number;
  batchSize: number;
  sequenceLength: number;
  minSamplesPerSymbol: number;
  outputTimeframes: number[];
  useMarketContext: boolean;
}

const DEFAULT_CONFIG: TrainingServiceConfig = {
  trainingIntervalMs: 300000, // Train every 5 minutes (reduced frequency for speed)
  epochsPerCycle: 10, // Reduced epochs per cycle (was 20)
  batchSize: 128, // Increased batch size for faster training
  sequenceLength: 30, // Reduced to 30 days lookback (was 60) for speed
  minSamplesPerSymbol: 100, // Reduced samples needed (was 200)
  outputTimeframes: [1, 5, 10], // Predict 1, 5, and 10 days ahead
  useMarketContext: true,
};

type TrainingCallback = (status: TrainingStatus) => void;

/**
 * Enhanced Training Service class
 */
class TrainingService {
  private model: StockPredictionModel;
  private config: TrainingServiceConfig;
  private status: TrainingStatus;
  private trainingInterval: ReturnType<typeof setInterval> | null = null;
  private isInitialized: boolean = false;
  private callbacks: Set<TrainingCallback> = new Set();
  private trainingData: Map<string, { sequences: number[][][]; targets: number[][] }> = new Map();
  private marketContext: MarketContext | null = null;

  constructor(config: Partial<TrainingServiceConfig> = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config };

    const modelConfig: Partial<ModelConfig> = {
      sequenceLength: this.config.sequenceLength,
      featureCount: getFeatureCount(),
      outputTimeframes: this.config.outputTimeframes,
    };

    this.model = createModel(modelConfig);
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
      phase: 'idle',
    };
  }

  /**
   * Initialize the training service
   */
  async initialize(): Promise<void> {
    if (this.isInitialized) return;

    console.log('Initializing Enhanced Training Service...');
    this.status.phase = 'preparing';
    this.notifyCallbacks();

    // Try to load existing model
    const loaded = await this.model.loadModel();
    if (!loaded) {
      this.model.buildModel();
    }

    // Pre-fetch and prepare training data
    await this.prepareAllTrainingData();

    this.isInitialized = true;
    this.status.phase = 'idle';
    this.notifyCallbacks();
    console.log('Enhanced Training Service initialized');
  }

  /**
   * Update market context for training
   */
  updateMarketContext(context: MarketContext): void {
    this.marketContext = context;
  }

  /**
   * Calculate market context from aggregate market data
   */
  private calculateMarketContext(allData: Map<string, HistoricalDataPoint[]>): MarketContext {
    const symbols = Array.from(allData.keys());
    if (symbols.length === 0) {
      return {
        marketTrend: 0,
        marketVolatility: 0.5,
        sectorPerformance: 0,
        breadthRatio: 1,
        riskOnOff: 0,
      };
    }

    let advancers = 0;
    let decliners = 0;
    let totalChange = 0;
    let totalVolatility = 0;

    for (const [, data] of allData) {
      if (data.length < 2) continue;

      const lastClose = data[data.length - 1].close;
      const prevClose = data[data.length - 2].close;
      const change = (lastClose - prevClose) / prevClose;

      totalChange += change;
      if (change > 0) advancers++;
      else if (change < 0) decliners++;

      // Calculate recent volatility
      const returns = [];
      for (let i = Math.max(1, data.length - 20); i < data.length; i++) {
        returns.push(Math.log(data[i].close / data[i - 1].close));
      }
      if (returns.length > 0) {
        const mean = returns.reduce((a, b) => a + b, 0) / returns.length;
        const variance = returns.reduce((sum, r) => sum + Math.pow(r - mean, 2), 0) / returns.length;
        totalVolatility += Math.sqrt(variance * 252);
      }
    }

    const avgChange = symbols.length > 0 ? totalChange / symbols.length : 0;
    const avgVolatility = symbols.length > 0 ? totalVolatility / symbols.length : 0.2;
    const breadthRatio = decliners > 0 ? advancers / decliners : advancers > 0 ? 2 : 1;

    return {
      marketTrend: Math.tanh(avgChange * 100), // Normalize to -1 to 1
      marketVolatility: Math.min(avgVolatility, 1),
      sectorPerformance: Math.tanh(avgChange * 50),
      breadthRatio,
      riskOnOff: breadthRatio > 1.5 ? 1 : breadthRatio < 0.67 ? -1 : 0,
    };
  }

  /**
   * Prepare training data for all supported symbols
   * Optimized: fetch only 100 days (reduced from 500) and fetch in parallel
   */
  private async prepareAllTrainingData(): Promise<void> {
    const symbols = getSupportedSymbols().slice(0, 3); // Limit to 3 symbols for speed
    console.log(`Preparing training data for ${symbols.length} symbols...`);

    const allHistoricalData = new Map<string, HistoricalDataPoint[]>();

    // Fetch all data in parallel (much faster than sequential)
    const fetchPromises = symbols.map(async (symbol) => {
      try {
        const data = await fetchHistoricalData(symbol, 100); // Reduced from 500 to 100 days
        return { symbol, data };
      } catch (error) {
        console.warn(`Failed to fetch data for ${symbol}:`, error);
        return { symbol, data: [] };
      }
    });

    const results = await Promise.all(fetchPromises);

    for (const { symbol, data } of results) {
      if (data.length >= this.config.sequenceLength + 20) {
        allHistoricalData.set(symbol, data);
      }
    }

    // Calculate market context
    if (this.config.useMarketContext) {
      this.marketContext = this.calculateMarketContext(allHistoricalData);
    }

    // Second pass: prepare training data with context
    for (const [symbol, historicalData] of allHistoricalData) {
      try {
        const data = this.prepareTrainingDataForSymbol(historicalData);
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
   * Prepare training data for a single symbol (multi-timeframe)
   */
  private prepareTrainingDataForSymbol(
    historicalData: HistoricalDataPoint[]
  ): { sequences: number[][][]; targets: number[][] } {
    const sequences: number[][][] = [];
    const targets: number[][] = [];

    const maxFutureDay = Math.max(...this.config.outputTimeframes);

    // Create sliding window sequences
    for (let i = this.config.sequenceLength; i < historicalData.length - maxFutureDay; i++) {
      const windowData = historicalData.slice(i - this.config.sequenceLength, i);

      // Calculate indicators for each timestep in the window
      const sequence: number[][] = [];
      for (let j = 0; j < windowData.length; j++) {
        const dataUpToPoint = historicalData.slice(0, i - this.config.sequenceLength + j + 1);
        const indicators = calculateAllIndicators(dataUpToPoint, this.marketContext || undefined);
        sequence.push(indicatorsToFeatures(indicators));
      }

      // Calculate targets for multiple timeframes
      const currentPrice = windowData[windowData.length - 1].close;
      const targetValues: number[] = [];

      for (const days of this.config.outputTimeframes) {
        const futureIndex = i + days - 1;
        if (futureIndex < historicalData.length) {
          const futurePrice = historicalData[futureIndex].close;
          const priceChange = ((futurePrice - currentPrice) / currentPrice) * 100;
          // Normalize to -1 to 1 (assuming max 10% change)
          targetValues.push(Math.tanh(priceChange / 10));
        } else {
          targetValues.push(0);
        }
      }

      sequences.push(sequence);
      targets.push(targetValues);
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

    this.status.phase = 'training';
    this.status.totalSymbols = symbols.length;
    this.status.symbolsProcessed = 0;
    this.status.totalEpochs = this.config.epochsPerCycle;
    this.notifyCallbacks();

    // Combine all training data
    const allSequences: number[][][] = [];
    const allTargets: number[][] = [];

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
        earlyStopping: true,
        patience: 5,
      },
      (epoch, logs) => {
        this.status.currentEpoch = epoch + 1;
        this.status.metrics = this.model.getMetrics();

        const elapsedMs = Date.now() - startTime;
        const msPerEpoch = elapsedMs / (epoch + 1);
        const remainingEpochs = this.config.epochsPerCycle - (epoch + 1);
        this.status.estimatedTimeRemaining = remainingEpochs * msPerEpoch;

        this.notifyCallbacks();
      }
    );

    this.status.phase = 'validating';
    this.notifyCallbacks();

    // Save model after training
    try {
      await this.model.saveModel();
    } catch (error) {
      console.warn('Failed to save model:', error);
    }

    this.status.phase = 'idle';
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
    this.status.phase = 'idle';
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
      const historicalData = await fetchHistoricalData(symbol, 500);
      if (historicalData.length >= this.config.sequenceLength + 20) {
        const data = this.prepareTrainingDataForSymbol(historicalData);
        if (data.sequences.length >= this.config.minSamplesPerSymbol) {
          this.trainingData.set(symbol, data);
        }
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
   * Get current market context
   */
  getMarketContext(): MarketContext | null {
    return this.marketContext;
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
 * Reset the training service
 */
export function resetTrainingService(): void {
  if (trainingServiceInstance) {
    trainingServiceInstance.dispose();
    trainingServiceInstance = null;
  }
}
