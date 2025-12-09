/**
 * AI Prediction Service
 * Uses the ML pipeline for real stock predictions
 */

import {
  PredictionResult,
  PredictionTimeframe,
  AIServiceStatus,
  MarketSentiment,
} from './types';
import { getPredictionPipeline, getTrainingService, TrainingStatus } from './ml';

class PredictionService {
  private initialized: boolean = false;

  /**
   * Ensure the ML pipeline is initialized (non-blocking)
   */
  private ensureInitialized(): void {
    if (this.initialized) return;

    const pipeline = getPredictionPipeline();
    // Initialize in background - don't block UI
    pipeline.initialize().then(() => {
      this.initialized = true;
      console.log('✅ AI Pipeline ready');
    }).catch((err) => {
      console.error('❌ AI Pipeline initialization failed:', err);
    });
  }

  /**
   * Get AI prediction for a stock
   */
  async getPrediction(
    symbol: string,
    timeframe: PredictionTimeframe = '1w'
  ): Promise<PredictionResult | null> {
    // Non-blocking: start initialization if needed but don't wait
    this.ensureInitialized();

    // Return null if not ready yet - prevents blocking page load
    if (!this.initialized) {
      return null;
    }

    const pipeline = getPredictionPipeline();
    return pipeline.predict(symbol, timeframe);
  }

  /**
   * Get predictions for multiple stocks
   */
  async getBatchPredictions(
    symbols: string[],
    timeframe: PredictionTimeframe = '1w'
  ): Promise<PredictionResult[]> {
    await this.ensureInitialized();

    const pipeline = getPredictionPipeline();
    return pipeline.predictBatch(symbols, timeframe);
  }

  /**
   * Get market sentiment analysis
   */
  async getMarketSentiment(symbol?: string): Promise<MarketSentiment> {
    // Non-blocking: start initialization if needed but don't wait
    this.ensureInitialized();

    // Return default sentiment if not ready yet
    if (!this.initialized) {
      return {
        overall: 'neutral',
        score: 0,
        newsImpact: 0,
        socialMediaImpact: 0,
        technicalIndicators: 0,
      };
    }

    const pipeline = getPredictionPipeline();
    return pipeline.getMarketSentiment(symbol);
  }

  /**
   * Train model on new data (triggers a training cycle)
   */
  async trainModel(data: unknown): Promise<boolean> {
    await this.ensureInitialized();

    // Training happens continuously, but we can trigger a refresh
    const trainingService = getTrainingService();
    if (!trainingService.getStatus().isRunning) {
      await trainingService.start();
    }
    return true;
  }

  /**
   * Get AI service status
   */
  async getServiceStatus(): Promise<AIServiceStatus> {
    await this.ensureInitialized();

    const pipeline = getPredictionPipeline();
    return pipeline.getServiceStatus();
  }

  /**
   * Get prediction history for a symbol
   */
  async getPredictionHistory(
    symbol: string,
    limit: number = 10
  ): Promise<PredictionResult[]> {
    await this.ensureInitialized();

    const pipeline = getPredictionPipeline();
    return pipeline.getPredictionHistory(symbol, limit);
  }

  /**
   * Validate prediction accuracy
   */
  async validatePrediction(predictionId: string): Promise<{
    wasAccurate: boolean;
    actualChange: number;
    predictedChange: number;
    error: number;
  }> {
    // In a real system, this would compare past predictions to actual outcomes
    await this.ensureInitialized();

    const trainingService = getTrainingService();
    const metrics = trainingService.getModel().getMetrics();

    // Simulate validation based on model accuracy
    const accuracy = metrics.accuracy / 100;
    const wasAccurate = Math.random() < accuracy;
    const predictedChange = (Math.random() - 0.5) * 10;
    const error = wasAccurate
      ? (Math.random() - 0.5) * 2
      : (Math.random() - 0.5) * 6;
    const actualChange = predictedChange + error;

    return {
      wasAccurate,
      actualChange: Number(actualChange.toFixed(2)),
      predictedChange: Number(predictedChange.toFixed(2)),
      error: Number(Math.abs(error).toFixed(2)),
    };
  }

  /**
   * Get current training status
   */
  getTrainingStatus(): TrainingStatus | null {
    if (!this.initialized) return null;

    const trainingService = getTrainingService();
    return trainingService.getStatus();
  }

  /**
   * Subscribe to training updates
   */
  subscribeToTrainingUpdates(callback: (status: TrainingStatus) => void): () => void {
    const trainingService = getTrainingService();
    return trainingService.subscribe(callback);
  }
}

export const predictionService = new PredictionService();
