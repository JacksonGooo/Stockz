/**
 * Confidence Calibration Module
 *
 * Calibrates raw model confidence scores to reflect actual accuracy
 * using Platt scaling and Expected Calibration Error (ECE)
 *
 * A well-calibrated model should have:
 * - Predictions with 70% confidence should be correct ~70% of the time
 * - Predictions with 90% confidence should be correct ~90% of the time
 */

export interface PredictionOutcome {
  symbol: string;
  predictedAt: Date;
  targetDate: Date;
  predictedChangePercent: number;
  actualChangePercent: number | null;
  rawConfidence: number;
  calibratedConfidence: number | null;
  wasCorrect: boolean | null; // Direction correct
  error: number | null; // Absolute error
}

export interface CalibrationBin {
  binStart: number;
  binEnd: number;
  meanConfidence: number;
  actualAccuracy: number;
  sampleCount: number;
  gap: number; // |accuracy - confidence|
}

export interface CalibrationStats {
  expectedCalibrationError: number; // ECE
  maxCalibrationError: number; // MCE
  bins: CalibrationBin[];
  totalSamples: number;
  overallAccuracy: number;
  isCalibrated: boolean;
}

/**
 * Platt scaling parameters
 * Transforms raw score to probability: P = 1 / (1 + exp(A * score + B))
 */
interface PlattParams {
  A: number;
  B: number;
}

/**
 * Confidence Calibrator class
 */
export class ConfidenceCalibrator {
  private plattParams: PlattParams = { A: -1, B: 0 };
  private isCalibrated: boolean = false;
  private outcomes: PredictionOutcome[] = [];
  private calibrationStats: CalibrationStats | null = null;

  // Number of bins for calibration analysis
  private numBins: number = 10;

  /**
   * Add a prediction outcome for calibration
   */
  addOutcome(outcome: PredictionOutcome): void {
    this.outcomes.push(outcome);

    // Recalibrate periodically
    if (this.outcomes.length % 50 === 0) {
      this.calibrate();
    }
  }

  /**
   * Add multiple outcomes at once
   */
  addOutcomes(outcomes: PredictionOutcome[]): void {
    this.outcomes.push(...outcomes);
    if (outcomes.length > 10) {
      this.calibrate();
    }
  }

  /**
   * Calibrate confidence scores using Platt scaling
   * Requires at least 20 samples with known outcomes
   */
  calibrate(): boolean {
    const validOutcomes = this.outcomes.filter(o => o.wasCorrect !== null);

    if (validOutcomes.length < 20) {
      console.log(`Insufficient data for calibration (${validOutcomes.length}/20 samples)`);
      return false;
    }

    // Fit Platt scaling parameters using gradient descent
    this.fitPlattScaling(validOutcomes);

    // Calculate calibration statistics
    this.calibrationStats = this.calculateCalibrationStats(validOutcomes);
    this.isCalibrated = this.calibrationStats.expectedCalibrationError < 0.1;

    console.log(`Calibration complete. ECE: ${this.calibrationStats.expectedCalibrationError.toFixed(3)}`);
    return this.isCalibrated;
  }

  /**
   * Calibrate a raw confidence score
   */
  calibrateConfidence(rawConfidence: number): number {
    if (!this.isCalibrated) {
      // Return raw confidence with slight compression toward 0.5
      return 0.5 + (rawConfidence - 0.5) * 0.8;
    }

    // Apply Platt scaling
    const { A, B } = this.plattParams;
    const calibrated = 1 / (1 + Math.exp(A * rawConfidence + B));

    // Clamp to reasonable range
    return Math.max(0.3, Math.min(0.95, calibrated));
  }

  /**
   * Get multi-factor calibrated confidence
   * Considers: model uncertainty, regime stability, prediction magnitude, historical accuracy
   */
  getMultiFactorConfidence(factors: {
    rawConfidence: number;
    modelUncertainty?: number; // Std dev from MC Dropout (lower = more confident)
    regimeStability?: number; // 0-1 (higher = more stable)
    predictionMagnitude?: number; // Absolute % change predicted
    historicalAccuracySimilar?: number; // Historical accuracy for similar conditions
    timeframeDecay?: number; // Decay factor for longer timeframes
  }): number {
    const {
      rawConfidence,
      modelUncertainty = 0,
      regimeStability = 0.5,
      predictionMagnitude = 0,
      historicalAccuracySimilar = 0.6,
      timeframeDecay = 1,
    } = factors;

    // Weight factors
    const weights = {
      rawConfidence: 0.25,
      modelUncertainty: 0.20,
      regimeStability: 0.15,
      predictionMagnitude: 0.10,
      historicalAccuracy: 0.20,
      timeframeDecay: 0.10,
    };

    // Convert uncertainty to confidence (lower uncertainty = higher confidence)
    const uncertaintyConfidence = Math.max(0, 1 - modelUncertainty / 3);

    // Convert magnitude to confidence (smaller predictions = more confident)
    const magnitudeConfidence = Math.max(0.3, 1 - Math.abs(predictionMagnitude) / 20);

    // Weighted combination
    const combinedConfidence =
      rawConfidence * weights.rawConfidence +
      uncertaintyConfidence * weights.modelUncertainty +
      regimeStability * weights.regimeStability +
      magnitudeConfidence * weights.predictionMagnitude +
      historicalAccuracySimilar * weights.historicalAccuracy +
      timeframeDecay * weights.timeframeDecay;

    // Apply calibration
    return this.calibrateConfidence(combinedConfidence);
  }

  /**
   * Fit Platt scaling parameters using simple gradient descent
   */
  private fitPlattScaling(outcomes: PredictionOutcome[]): void {
    // Initialize parameters
    let A = -2;
    let B = 0;
    const learningRate = 0.1;
    const iterations = 100;

    for (let iter = 0; iter < iterations; iter++) {
      let gradA = 0;
      let gradB = 0;

      for (const outcome of outcomes) {
        const score = outcome.rawConfidence;
        const target = outcome.wasCorrect ? 1 : 0;

        // Sigmoid
        const p = 1 / (1 + Math.exp(A * score + B));

        // Gradient of log loss
        const diff = p - target;
        gradA += diff * score;
        gradB += diff;
      }

      // Update with average gradient
      A -= learningRate * gradA / outcomes.length;
      B -= learningRate * gradB / outcomes.length;
    }

    this.plattParams = { A, B };
  }

  /**
   * Calculate calibration statistics
   */
  private calculateCalibrationStats(outcomes: PredictionOutcome[]): CalibrationStats {
    const bins: CalibrationBin[] = [];
    let ece = 0;
    let mce = 0;
    let totalCorrect = 0;

    // Create bins
    for (let i = 0; i < this.numBins; i++) {
      const binStart = i / this.numBins;
      const binEnd = (i + 1) / this.numBins;

      // Filter outcomes in this bin
      const binOutcomes = outcomes.filter(o =>
        o.rawConfidence >= binStart && o.rawConfidence < binEnd
      );

      if (binOutcomes.length === 0) {
        bins.push({
          binStart,
          binEnd,
          meanConfidence: (binStart + binEnd) / 2,
          actualAccuracy: 0,
          sampleCount: 0,
          gap: 0,
        });
        continue;
      }

      const meanConfidence = binOutcomes.reduce((sum, o) => sum + o.rawConfidence, 0) / binOutcomes.length;
      const correctCount = binOutcomes.filter(o => o.wasCorrect).length;
      const actualAccuracy = correctCount / binOutcomes.length;
      const gap = Math.abs(actualAccuracy - meanConfidence);

      bins.push({
        binStart,
        binEnd,
        meanConfidence,
        actualAccuracy,
        sampleCount: binOutcomes.length,
        gap,
      });

      // ECE: weighted average of gaps
      ece += gap * binOutcomes.length;

      // MCE: max gap
      mce = Math.max(mce, gap);

      totalCorrect += correctCount;
    }

    ece /= outcomes.length;

    return {
      expectedCalibrationError: ece,
      maxCalibrationError: mce,
      bins,
      totalSamples: outcomes.length,
      overallAccuracy: totalCorrect / outcomes.length,
      isCalibrated: ece < 0.1,
    };
  }

  /**
   * Get current calibration statistics
   */
  getStats(): CalibrationStats | null {
    return this.calibrationStats;
  }

  /**
   * Get historical accuracy for outcomes similar to the given prediction
   */
  getHistoricalAccuracyForSimilar(
    predictedChange: number,
    confidence: number,
    tolerance: number = 0.2
  ): number {
    const validOutcomes = this.outcomes.filter(o => o.wasCorrect !== null);

    // Filter for similar predictions
    const similar = validOutcomes.filter(o => {
      const changeSimilar = Math.abs(o.predictedChangePercent - predictedChange) < tolerance * Math.abs(predictedChange || 1);
      const confidenceSimilar = Math.abs(o.rawConfidence - confidence) < 0.1;
      return changeSimilar && confidenceSimilar;
    });

    if (similar.length < 5) {
      // Not enough similar samples, return default
      return 0.6;
    }

    const correct = similar.filter(o => o.wasCorrect).length;
    return correct / similar.length;
  }

  /**
   * Check if a prediction was correct (direction match)
   */
  static isPredictionCorrect(
    predictedChange: number,
    actualChange: number,
    threshold: number = 0.5
  ): boolean {
    // Both positive or both negative (with threshold for noise)
    if (Math.abs(predictedChange) < threshold && Math.abs(actualChange) < threshold) {
      return true; // Both neutral
    }
    return (predictedChange > 0 && actualChange > 0) || (predictedChange < 0 && actualChange < 0);
  }

  /**
   * Export outcomes for persistence
   */
  exportOutcomes(): PredictionOutcome[] {
    return [...this.outcomes];
  }

  /**
   * Import outcomes from persistence
   */
  importOutcomes(outcomes: PredictionOutcome[]): void {
    this.outcomes = outcomes;
    if (outcomes.length >= 20) {
      this.calibrate();
    }
  }

  /**
   * Clear all outcomes and reset calibration
   */
  reset(): void {
    this.outcomes = [];
    this.plattParams = { A: -1, B: 0 };
    this.isCalibrated = false;
    this.calibrationStats = null;
  }
}

// Singleton instance
let calibratorInstance: ConfidenceCalibrator | null = null;

/**
 * Get the confidence calibrator singleton
 */
export function getConfidenceCalibrator(): ConfidenceCalibrator {
  if (!calibratorInstance) {
    calibratorInstance = new ConfidenceCalibrator();
  }
  return calibratorInstance;
}

/**
 * Reset the calibrator (for testing)
 */
export function resetConfidenceCalibrator(): void {
  if (calibratorInstance) {
    calibratorInstance.reset();
  }
  calibratorInstance = null;
}
