/**
 * Model Ensemble for Improved Predictions
 *
 * Combines multiple specialized models:
 * - General model (trained on all data)
 * - Trend model (trained on trending periods)
 * - Mean-reversion model (trained on ranging periods)
 *
 * Uses weighted averaging based on current market regime
 */

import { LightPredictionModel, LightPrediction, createLightModel } from './lightModel';
import { MarketRegime, MarketRegimeType } from './indicators';

export interface EnsembleMember {
  model: LightPredictionModel;
  name: string;
  specialty: 'general' | 'trend' | 'mean_reversion' | 'volatility';
  weight: number;
  accuracy: number;
  trainingSamples: number;
}

export interface EnsemblePrediction {
  day1: { mean: number; confidence: number; agreement: number };
  day5: { mean: number; confidence: number; agreement: number };
  day10: { mean: number; confidence: number; agreement: number };
  overall: {
    direction: 'up' | 'down' | 'neutral';
    strength: number;
    confidence: number;
  };
  memberPredictions: {
    name: string;
    day1: number;
    day5: number;
    day10: number;
    weight: number;
  }[];
}

export interface EnsembleConfig {
  useAdaptiveWeights: boolean;
  minAgreementForHighConfidence: number;
  regimeWeightBoost: number;
}

const DEFAULT_CONFIG: EnsembleConfig = {
  useAdaptiveWeights: true,
  minAgreementForHighConfidence: 0.7,
  regimeWeightBoost: 1.5, // Boost weight of regime-appropriate model
};

/**
 * Model Ensemble class
 */
export class ModelEnsemble {
  private members: EnsembleMember[] = [];
  private config: EnsembleConfig;
  private isInitialized: boolean = false;

  constructor(config: Partial<EnsembleConfig> = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config };
  }

  /**
   * Initialize ensemble with default models
   */
  async initialize(): Promise<void> {
    if (this.isInitialized) return;

    // Create three specialized models
    const generalModel = createLightModel();
    const trendModel = createLightModel();
    const meanReversionModel = createLightModel();

    // Build all models
    generalModel.buildModel();
    trendModel.buildModel();
    meanReversionModel.buildModel();

    this.members = [
      {
        model: generalModel,
        name: 'General',
        specialty: 'general',
        weight: 0.4, // Base weight
        accuracy: 0.6,
        trainingSamples: 0,
      },
      {
        model: trendModel,
        name: 'Trend',
        specialty: 'trend',
        weight: 0.3,
        accuracy: 0.6,
        trainingSamples: 0,
      },
      {
        model: meanReversionModel,
        name: 'MeanReversion',
        specialty: 'mean_reversion',
        weight: 0.3,
        accuracy: 0.6,
        trainingSamples: 0,
      },
    ];

    this.isInitialized = true;
    console.log('Model ensemble initialized with 3 members');
  }

  /**
   * Add a trained model to the ensemble
   */
  addMember(member: EnsembleMember): void {
    // Check if member with same name exists
    const existingIndex = this.members.findIndex(m => m.name === member.name);
    if (existingIndex >= 0) {
      this.members[existingIndex] = member;
    } else {
      this.members.push(member);
    }
    this.normalizeWeights();
  }

  /**
   * Train a specific model in the ensemble
   */
  async trainMember(
    name: string,
    sequences: number[][][],
    targets: number[][],
    epochs: number = 10
  ): Promise<void> {
    const member = this.members.find(m => m.name === name);
    if (!member) {
      console.warn(`Member ${name} not found in ensemble`);
      return;
    }

    const metrics = await member.model.train(sequences, targets, { epochs });
    member.accuracy = metrics.accuracy / 100;
    member.trainingSamples = metrics.trainingSamples;

    // Update weight based on accuracy
    if (this.config.useAdaptiveWeights) {
      member.weight = Math.max(0.1, member.accuracy);
      this.normalizeWeights();
    }
  }

  /**
   * Make ensemble prediction with regime-aware weighting
   */
  predict(sequence: number[][], regime?: MarketRegime): EnsemblePrediction {
    if (this.members.length === 0) {
      return this.getDefaultPrediction();
    }

    // Get predictions from all members
    const memberPredictions = this.members.map(member => {
      const pred = member.model.predictFast(sequence);
      return {
        member,
        prediction: pred,
      };
    });

    // Calculate regime-adjusted weights
    const adjustedWeights = this.calculateRegimeWeights(regime);

    // Weighted average of predictions
    let day1Sum = 0;
    let day5Sum = 0;
    let day10Sum = 0;
    let weightSum = 0;

    const memberResults: EnsemblePrediction['memberPredictions'] = [];

    for (let i = 0; i < memberPredictions.length; i++) {
      const { member, prediction } = memberPredictions[i];
      const weight = adjustedWeights[i];

      day1Sum += prediction.day1 * weight;
      day5Sum += prediction.day5 * weight;
      day10Sum += prediction.day10 * weight;
      weightSum += weight;

      memberResults.push({
        name: member.name,
        day1: prediction.day1,
        day5: prediction.day5,
        day10: prediction.day10,
        weight,
      });
    }

    const day1Mean = weightSum > 0 ? day1Sum / weightSum : 0;
    const day5Mean = weightSum > 0 ? day5Sum / weightSum : 0;
    const day10Mean = weightSum > 0 ? day10Sum / weightSum : 0;

    // Calculate agreement (how much models agree)
    const day1Agreement = this.calculateAgreement(memberPredictions.map(p => p.prediction.day1));
    const day5Agreement = this.calculateAgreement(memberPredictions.map(p => p.prediction.day5));
    const day10Agreement = this.calculateAgreement(memberPredictions.map(p => p.prediction.day10));

    // Confidence is based on agreement and model accuracies
    const avgAccuracy = this.members.reduce((sum, m) => sum + m.accuracy, 0) / this.members.length;
    const day1Confidence = this.calculateConfidence(day1Agreement, avgAccuracy);
    const day5Confidence = this.calculateConfidence(day5Agreement, avgAccuracy) * 0.9;
    const day10Confidence = this.calculateConfidence(day10Agreement, avgAccuracy) * 0.8;

    // Overall direction
    const avgPrediction = (day1Mean + day5Mean + day10Mean) / 3;

    return {
      day1: { mean: day1Mean, confidence: day1Confidence, agreement: day1Agreement },
      day5: { mean: day5Mean, confidence: day5Confidence, agreement: day5Agreement },
      day10: { mean: day10Mean, confidence: day10Confidence, agreement: day10Agreement },
      overall: {
        direction: avgPrediction > 1 ? 'up' : avgPrediction < -1 ? 'down' : 'neutral',
        strength: Math.min(1, Math.abs(avgPrediction) / 5),
        confidence: (day1Confidence + day5Confidence + day10Confidence) / 3,
      },
      memberPredictions: memberResults,
    };
  }

  /**
   * Calculate regime-adjusted weights
   */
  private calculateRegimeWeights(regime?: MarketRegime): number[] {
    const weights = this.members.map(m => m.weight);

    if (!regime || !this.config.useAdaptiveWeights) {
      return weights;
    }

    // Boost weights based on regime
    return this.members.map((member, i) => {
      let boost = 1;

      if (regime.regime === 'trending_up' || regime.regime === 'trending_down') {
        if (member.specialty === 'trend') {
          boost = this.config.regimeWeightBoost;
        }
      } else if (regime.regime === 'ranging') {
        if (member.specialty === 'mean_reversion') {
          boost = this.config.regimeWeightBoost;
        }
      } else if (regime.regime === 'high_volatility') {
        if (member.specialty === 'volatility') {
          boost = this.config.regimeWeightBoost;
        }
        // Also boost general model in high volatility
        if (member.specialty === 'general') {
          boost = 1.2;
        }
      }

      return weights[i] * boost * regime.regimeStrength + weights[i] * (1 - regime.regimeStrength);
    });
  }

  /**
   * Calculate agreement between predictions (0 = no agreement, 1 = full agreement)
   */
  private calculateAgreement(predictions: number[]): number {
    if (predictions.length <= 1) return 1;

    // Check if all predictions agree on direction
    const positiveCount = predictions.filter(p => p > 0).length;
    const negativeCount = predictions.filter(p => p < 0).length;
    const maxAgreement = Math.max(positiveCount, negativeCount) / predictions.length;

    // Also check magnitude agreement (coefficient of variation)
    const mean = predictions.reduce((a, b) => a + b, 0) / predictions.length;
    const variance = predictions.reduce((sum, p) => sum + Math.pow(p - mean, 2), 0) / predictions.length;
    const std = Math.sqrt(variance);
    const cv = mean !== 0 ? Math.abs(std / mean) : 1;

    // Low CV = high magnitude agreement
    const magnitudeAgreement = Math.max(0, 1 - cv);

    // Combine direction and magnitude agreement
    return (maxAgreement + magnitudeAgreement) / 2;
  }

  /**
   * Calculate confidence from agreement and accuracy
   */
  private calculateConfidence(agreement: number, accuracy: number): number {
    // Base confidence from accuracy
    let confidence = accuracy;

    // Boost or penalize based on agreement
    if (agreement >= this.config.minAgreementForHighConfidence) {
      confidence *= 1 + (agreement - this.config.minAgreementForHighConfidence) * 0.3;
    } else {
      confidence *= 0.7 + agreement * 0.3;
    }

    return Math.max(0.3, Math.min(0.95, confidence));
  }

  /**
   * Normalize weights to sum to 1
   */
  private normalizeWeights(): void {
    const total = this.members.reduce((sum, m) => sum + m.weight, 0);
    if (total > 0) {
      for (const member of this.members) {
        member.weight /= total;
      }
    }
  }

  /**
   * Get default prediction when no models available
   */
  private getDefaultPrediction(): EnsemblePrediction {
    return {
      day1: { mean: 0, confidence: 0.5, agreement: 0 },
      day5: { mean: 0, confidence: 0.5, agreement: 0 },
      day10: { mean: 0, confidence: 0.5, agreement: 0 },
      overall: { direction: 'neutral', strength: 0, confidence: 0.5 },
      memberPredictions: [],
    };
  }

  /**
   * Get ensemble statistics
   */
  getStats(): {
    memberCount: number;
    members: { name: string; specialty: string; weight: number; accuracy: number }[];
    averageAccuracy: number;
  } {
    return {
      memberCount: this.members.length,
      members: this.members.map(m => ({
        name: m.name,
        specialty: m.specialty,
        weight: m.weight,
        accuracy: m.accuracy,
      })),
      averageAccuracy: this.members.length > 0
        ? this.members.reduce((sum, m) => sum + m.accuracy, 0) / this.members.length
        : 0,
    };
  }

  /**
   * Save all models in ensemble
   */
  async saveModels(basePath: string = 'indexeddb://ensemble'): Promise<void> {
    for (const member of this.members) {
      await member.model.saveModel(`${basePath}-${member.name.toLowerCase()}`);
    }
  }

  /**
   * Load all models in ensemble
   */
  async loadModels(basePath: string = 'indexeddb://ensemble'): Promise<boolean> {
    let allLoaded = true;
    for (const member of this.members) {
      const loaded = await member.model.loadModel(`${basePath}-${member.name.toLowerCase()}`);
      if (!loaded) {
        member.model.buildModel();
        allLoaded = false;
      }
    }
    return allLoaded;
  }

  /**
   * Dispose all models
   */
  dispose(): void {
    for (const member of this.members) {
      member.model.dispose();
    }
    this.members = [];
    this.isInitialized = false;
  }
}

// Singleton instance
let ensembleInstance: ModelEnsemble | null = null;

/**
 * Get or create the model ensemble singleton
 */
export function getModelEnsemble(): ModelEnsemble {
  if (!ensembleInstance) {
    ensembleInstance = new ModelEnsemble();
  }
  return ensembleInstance;
}

/**
 * Reset the ensemble (for testing)
 */
export function resetModelEnsemble(): void {
  if (ensembleInstance) {
    ensembleInstance.dispose();
  }
  ensembleInstance = null;
}
