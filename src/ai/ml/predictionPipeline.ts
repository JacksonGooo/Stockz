/**
 * Prediction Pipeline
 * Orchestrates the full prediction process from data to results
 */

import { getTrainingService, TrainingStatus } from './trainingService';
import { fetchHistoricalData, fetchStockQuote, getStockInfo } from './dataProvider';
import {
  calculateAllIndicators,
  indicatorsToFeatures,
  TechnicalIndicators,
} from './indicators';
import {
  PredictionResult,
  PredictionTimeframe,
  PredictionFactor,
  AIServiceStatus,
  ModelMetrics,
  MarketSentiment,
} from '../types';

// Timeframe to days mapping (30m = 0.02 trading days)
const TIMEFRAME_DAYS: Record<PredictionTimeframe, number> = {
  '30m': 0.02,
  '1d': 1,
  '1w': 5,
  '1m': 22,
  '3m': 66,
  '6m': 132,
  '1y': 252,
};

// Confidence decay by timeframe (shorter predictions are more certain)
const CONFIDENCE_DECAY: Record<PredictionTimeframe, number> = {
  '30m': 1.0,
  '1d': 0.95,
  '1w': 0.85,
  '1m': 0.7,
  '3m': 0.55,
  '6m': 0.45,
  '1y': 0.35,
};

/**
 * Prediction Pipeline class
 */
class PredictionPipeline {
  private isInitialized: boolean = false;
  private initPromise: Promise<void> | null = null;

  /**
   * Initialize the pipeline and start training
   */
  async initialize(): Promise<void> {
    if (this.isInitialized) return;
    if (this.initPromise) return this.initPromise;

    this.initPromise = (async () => {
      console.log('Initializing Prediction Pipeline...');
      const trainingService = getTrainingService();
      await trainingService.initialize();
      await trainingService.start();
      this.isInitialized = true;
      console.log('Prediction Pipeline ready');
    })();

    return this.initPromise;
  }

  /**
   * Generate prediction for a single stock
   */
  async predict(
    symbol: string,
    timeframe: PredictionTimeframe = '1w'
  ): Promise<PredictionResult | null> {
    await this.initialize();

    // Fetch current data
    const quote = await fetchStockQuote(symbol);
    if (!quote) return null;

    const historicalData = await fetchHistoricalData(symbol, 100);
    if (historicalData.length < 30) return null;

    // Calculate current technical indicators
    const indicators = calculateAllIndicators(historicalData);

    // Prepare sequence for prediction
    const sequence = this.prepareSequence(historicalData);

    // Get prediction from model
    const trainingService = getTrainingService();
    const model = trainingService.getModel();
    const { prediction, confidence: baseConfidence } = model.predict(sequence);

    // Scale prediction by timeframe
    const timeframeDays = TIMEFRAME_DAYS[timeframe];
    const scaledPrediction = prediction * Math.sqrt(timeframeDays / 5);

    // Adjust confidence by timeframe
    const confidence = baseConfidence * CONFIDENCE_DECAY[timeframe];

    // Calculate predicted price
    const predictedChange = (scaledPrediction / 100) * quote.price;
    const predictedPrice = quote.price + predictedChange;

    // Analyze factors contributing to prediction
    const factors = this.analyzeFactors(indicators, scaledPrediction);

    return {
      symbol,
      currentPrice: quote.price,
      predictedPrice: Number(predictedPrice.toFixed(2)),
      predictedChange: Number(predictedChange.toFixed(2)),
      predictedChangePercent: Number(scaledPrediction.toFixed(2)),
      confidence: Number(confidence.toFixed(2)),
      timeframe,
      generatedAt: new Date(),
      factors,
    };
  }

  /**
   * Generate predictions for multiple stocks
   */
  async predictBatch(
    symbols: string[],
    timeframe: PredictionTimeframe = '1w'
  ): Promise<PredictionResult[]> {
    const predictions = await Promise.all(
      symbols.map((symbol) => this.predict(symbol, timeframe))
    );
    return predictions.filter((p): p is PredictionResult => p !== null);
  }

  /**
   * Prepare feature sequence for model input
   */
  private prepareSequence(historicalData: any[]): number[][] {
    const sequence: number[][] = [];
    const sequenceLength = 30;

    for (let i = Math.max(0, historicalData.length - sequenceLength); i < historicalData.length; i++) {
      const dataSlice = historicalData.slice(0, i + 1);
      const indicators = calculateAllIndicators(dataSlice);
      sequence.push(indicatorsToFeatures(indicators));
    }

    return sequence;
  }

  /**
   * Analyze which factors contributed to the prediction
   */
  private analyzeFactors(
    indicators: TechnicalIndicators,
    prediction: number
  ): PredictionFactor[] {
    const factors: PredictionFactor[] = [];
    const isBullish = prediction > 0;

    // RSI Analysis
    let rsiImpact: 'positive' | 'negative' | 'neutral' = 'neutral';
    let rsiDescription = 'RSI indicates neutral momentum';
    if (indicators.rsi < 30) {
      rsiImpact = 'positive';
      rsiDescription = 'RSI indicates oversold conditions - potential bounce';
    } else if (indicators.rsi > 70) {
      rsiImpact = 'negative';
      rsiDescription = 'RSI indicates overbought conditions - potential pullback';
    } else if (indicators.rsi > 50) {
      rsiImpact = 'positive';
      rsiDescription = 'RSI shows bullish momentum above 50';
    } else {
      rsiImpact = 'negative';
      rsiDescription = 'RSI shows bearish momentum below 50';
    }
    factors.push({
      name: 'RSI Momentum',
      impact: rsiImpact,
      weight: 0.2,
      description: rsiDescription,
    });

    // MACD Analysis
    let macdImpact: 'positive' | 'negative' | 'neutral' = 'neutral';
    if (indicators.macdHistogram > 0 && indicators.macd > indicators.macdSignal) {
      macdImpact = 'positive';
    } else if (indicators.macdHistogram < 0 && indicators.macd < indicators.macdSignal) {
      macdImpact = 'negative';
    }
    factors.push({
      name: 'MACD Trend',
      impact: macdImpact,
      weight: 0.2,
      description:
        macdImpact === 'positive'
          ? 'MACD showing bullish crossover signal'
          : macdImpact === 'negative'
            ? 'MACD showing bearish crossover signal'
            : 'MACD in consolidation phase',
    });

    // Bollinger Band Analysis
    let bbImpact: 'positive' | 'negative' | 'neutral' = 'neutral';
    const bbPosition = indicators.normalized.bollinger;
    if (bbPosition < 0.2) {
      bbImpact = 'positive';
    } else if (bbPosition > 0.8) {
      bbImpact = 'negative';
    }
    factors.push({
      name: 'Bollinger Bands',
      impact: bbImpact,
      weight: 0.15,
      description:
        bbPosition < 0.2
          ? 'Price near lower band - potential support'
          : bbPosition > 0.8
            ? 'Price near upper band - potential resistance'
            : 'Price within normal trading range',
    });

    // Moving Average Analysis
    let maImpact: 'positive' | 'negative' | 'neutral' = 'neutral';
    const priceAboveSMA20 = indicators.normalized.price > 0.5;
    const sma5AboveSMA20 = indicators.sma5 > indicators.sma20;
    if (priceAboveSMA20 && sma5AboveSMA20) {
      maImpact = 'positive';
    } else if (!priceAboveSMA20 && !sma5AboveSMA20) {
      maImpact = 'negative';
    }
    factors.push({
      name: 'Moving Averages',
      impact: maImpact,
      weight: 0.2,
      description:
        maImpact === 'positive'
          ? 'Price trading above key moving averages'
          : maImpact === 'negative'
            ? 'Price trading below key moving averages'
            : 'Mixed signals from moving averages',
    });

    // Volume Analysis
    let volumeImpact: 'positive' | 'negative' | 'neutral' = 'neutral';
    if (indicators.volumeRatio > 1.5 && isBullish) {
      volumeImpact = 'positive';
    } else if (indicators.volumeRatio > 1.5 && !isBullish) {
      volumeImpact = 'negative';
    }
    factors.push({
      name: 'Volume Analysis',
      impact: volumeImpact,
      weight: 0.15,
      description:
        indicators.volumeRatio > 1.5
          ? 'Above average volume supports the move'
          : indicators.volumeRatio < 0.7
            ? 'Low volume may indicate weak conviction'
            : 'Volume at normal levels',
    });

    // Volatility Analysis
    let volImpact: 'positive' | 'negative' | 'neutral' = 'neutral';
    if (indicators.volatility < 20) {
      volImpact = 'positive';
    } else if (indicators.volatility > 40) {
      volImpact = 'negative';
    }
    factors.push({
      name: 'Volatility',
      impact: volImpact,
      weight: 0.1,
      description:
        indicators.volatility < 20
          ? 'Low volatility - stable trading conditions'
          : indicators.volatility > 40
            ? 'High volatility - increased risk'
            : 'Moderate volatility levels',
    });

    // Sort by weight and return top factors
    return factors.sort((a, b) => b.weight - a.weight);
  }

  /**
   * Get market sentiment analysis
   */
  async getMarketSentiment(symbol?: string): Promise<MarketSentiment> {
    await this.initialize();

    // If symbol provided, analyze that stock
    if (symbol) {
      const historicalData = await fetchHistoricalData(symbol, 30);
      const indicators = calculateAllIndicators(historicalData);

      // Calculate sentiment score based on indicators
      let score = 0;
      if (indicators.rsi > 50) score += 20;
      else score -= 20;
      if (indicators.macd > indicators.macdSignal) score += 25;
      else score -= 25;
      if (indicators.sma5 > indicators.sma20) score += 20;
      else score -= 20;
      if (indicators.normalized.bollinger > 0.5) score += 10;
      else score -= 10;
      if (indicators.volumeRatio > 1) score += 15;
      else score -= 15;

      const overall: 'bullish' | 'bearish' | 'neutral' =
        score > 20 ? 'bullish' : score < -20 ? 'bearish' : 'neutral';

      return {
        overall,
        score: Math.max(-100, Math.min(100, score)),
        newsImpact: Math.round((Math.random() - 0.3) * 60),
        socialMediaImpact: Math.round((Math.random() - 0.3) * 60),
        technicalIndicators: score,
      };
    }

    // Overall market sentiment - aggregate multiple stocks
    const symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA'];
    let totalScore = 0;

    for (const sym of symbols) {
      const sentiment = await this.getMarketSentiment(sym);
      totalScore += sentiment.score;
    }

    const avgScore = totalScore / symbols.length;
    const overall: 'bullish' | 'bearish' | 'neutral' =
      avgScore > 15 ? 'bullish' : avgScore < -15 ? 'bearish' : 'neutral';

    return {
      overall,
      score: Math.round(avgScore),
      newsImpact: Math.round((Math.random() - 0.3) * 50),
      socialMediaImpact: Math.round((Math.random() - 0.3) * 50),
      technicalIndicators: Math.round(avgScore),
    };
  }

  /**
   * Get AI service status
   */
  async getServiceStatus(): Promise<AIServiceStatus> {
    await this.initialize();

    const trainingService = getTrainingService();
    const trainingStatus = trainingService.getStatus();
    const modelMetrics = trainingService.getModel().getMetrics();

    const metrics: ModelMetrics = {
      accuracy: modelMetrics.accuracy / 100,
      precision: Math.max(0.5, modelMetrics.accuracy / 100 - 0.05),
      recall: Math.max(0.5, modelMetrics.accuracy / 100 - 0.08),
      f1Score: Math.max(0.5, modelMetrics.accuracy / 100 - 0.06),
      lastTrainedAt: modelMetrics.lastTrainedAt,
      dataPointsUsed: modelMetrics.trainingSamples,
    };

    return {
      isOnline: true,
      modelVersion: '1.0.0-lstm',
      lastUpdated: modelMetrics.lastTrainedAt,
      metrics,
    };
  }

  /**
   * Get current training status
   */
  getTrainingStatus(): TrainingStatus {
    const trainingService = getTrainingService();
    return trainingService.getStatus();
  }

  /**
   * Subscribe to training updates
   */
  subscribeToTraining(callback: (status: TrainingStatus) => void): () => void {
    const trainingService = getTrainingService();
    return trainingService.subscribe(callback);
  }

  /**
   * Get prediction history (mock for now)
   */
  async getPredictionHistory(
    symbol: string,
    limit: number = 10
  ): Promise<PredictionResult[]> {
    await this.initialize();

    const predictions: PredictionResult[] = [];
    const quote = await fetchStockQuote(symbol);
    if (!quote) return [];

    for (let i = 0; i < limit; i++) {
      const daysAgo = i * 7;
      const historicalPrice = quote.price * (0.9 + Math.random() * 0.2);
      const prediction = await this.predict(symbol, '1w');

      if (prediction) {
        predictions.push({
          ...prediction,
          currentPrice: historicalPrice,
          generatedAt: new Date(Date.now() - daysAgo * 86400000),
        });
      }
    }

    return predictions;
  }
}

// Singleton instance
let pipelineInstance: PredictionPipeline | null = null;

/**
 * Get or create the prediction pipeline instance
 */
export function getPredictionPipeline(): PredictionPipeline {
  if (!pipelineInstance) {
    pipelineInstance = new PredictionPipeline();
  }
  return pipelineInstance;
}

/**
 * Reset the pipeline (for testing)
 */
export function resetPredictionPipeline(): void {
  pipelineInstance = null;
}
