/**
 * Prediction Storage Service
 * Saves and loads predictions to/from JSON files on the server
 */

import { promises as fs } from 'fs';
import path from 'path';
import { PredictionResult } from './minutePredictionService';

// Storage directory for predictions
const PREDICTIONS_DIR = path.join(process.cwd(), 'data', 'predictions');

// Prediction validity duration (60 minutes)
const PREDICTION_VALIDITY_MS = 60 * 60 * 1000;

/**
 * Ensure the predictions directory exists
 */
async function ensureDirectory(): Promise<void> {
  try {
    await fs.mkdir(PREDICTIONS_DIR, { recursive: true });
  } catch {
    // Directory might already exist
  }
}

/**
 * Get the file path for a symbol's predictions
 */
function getFilePath(symbol: string, assetType: 'stock' | 'crypto'): string {
  const sanitizedSymbol = symbol.replace(/[^a-zA-Z0-9]/g, '_').toUpperCase();
  return path.join(PREDICTIONS_DIR, `${assetType}_${sanitizedSymbol}.json`);
}

/**
 * Save prediction to JSON file
 */
export async function savePrediction(prediction: PredictionResult): Promise<void> {
  await ensureDirectory();

  const filePath = getFilePath(prediction.symbol, prediction.assetType);

  try {
    const jsonData = JSON.stringify(prediction, null, 2);
    await fs.writeFile(filePath, jsonData, 'utf-8');
    console.log(`Prediction saved for ${prediction.symbol} at ${filePath}`);
  } catch (error) {
    console.error(`Failed to save prediction for ${prediction.symbol}:`, error);
    throw error;
  }
}

/**
 * Load prediction from JSON file
 */
export async function loadPrediction(
  symbol: string,
  assetType: 'stock' | 'crypto'
): Promise<PredictionResult | null> {
  const filePath = getFilePath(symbol, assetType);

  try {
    const jsonData = await fs.readFile(filePath, 'utf-8');
    const prediction = JSON.parse(jsonData) as PredictionResult;
    return prediction;
  } catch {
    // File doesn't exist or is corrupted
    return null;
  }
}

/**
 * Check if a prediction is still valid (not expired)
 */
export function isPredictionValid(prediction: PredictionResult | null): boolean {
  if (!prediction) return false;

  const now = Date.now();
  return prediction.expiresAt > now;
}

/**
 * Check if predictions exist and are valid for a symbol
 */
export async function hasValidPrediction(
  symbol: string,
  assetType: 'stock' | 'crypto'
): Promise<boolean> {
  const prediction = await loadPrediction(symbol, assetType);
  return isPredictionValid(prediction);
}

/**
 * Get prediction if valid, or null if expired/missing
 */
export async function getValidPrediction(
  symbol: string,
  assetType: 'stock' | 'crypto'
): Promise<PredictionResult | null> {
  const prediction = await loadPrediction(symbol, assetType);

  if (isPredictionValid(prediction)) {
    return prediction;
  }

  return null;
}

/**
 * Delete prediction file for a symbol
 */
export async function deletePrediction(
  symbol: string,
  assetType: 'stock' | 'crypto'
): Promise<void> {
  const filePath = getFilePath(symbol, assetType);

  try {
    await fs.unlink(filePath);
    console.log(`Prediction deleted for ${symbol}`);
  } catch {
    // File might not exist
  }
}

/**
 * Clean up expired predictions
 */
export async function cleanupExpiredPredictions(): Promise<number> {
  await ensureDirectory();

  let cleaned = 0;

  try {
    const files = await fs.readdir(PREDICTIONS_DIR);

    for (const file of files) {
      if (!file.endsWith('.json')) continue;

      const filePath = path.join(PREDICTIONS_DIR, file);

      try {
        const jsonData = await fs.readFile(filePath, 'utf-8');
        const prediction = JSON.parse(jsonData) as PredictionResult;

        if (!isPredictionValid(prediction)) {
          await fs.unlink(filePath);
          cleaned++;
          console.log(`Cleaned up expired prediction: ${file}`);
        }
      } catch {
        // Skip corrupted files
      }
    }
  } catch {
    // Directory might not exist
  }

  return cleaned;
}

/**
 * List all stored predictions with their status
 */
export async function listPredictions(): Promise<
  Array<{
    symbol: string;
    assetType: 'stock' | 'crypto';
    generatedAt: number;
    expiresAt: number;
    isValid: boolean;
  }>
> {
  await ensureDirectory();

  const results: Array<{
    symbol: string;
    assetType: 'stock' | 'crypto';
    generatedAt: number;
    expiresAt: number;
    isValid: boolean;
  }> = [];

  try {
    const files = await fs.readdir(PREDICTIONS_DIR);

    for (const file of files) {
      if (!file.endsWith('.json')) continue;

      const filePath = path.join(PREDICTIONS_DIR, file);

      try {
        const jsonData = await fs.readFile(filePath, 'utf-8');
        const prediction = JSON.parse(jsonData) as PredictionResult;

        results.push({
          symbol: prediction.symbol,
          assetType: prediction.assetType,
          generatedAt: prediction.generatedAt,
          expiresAt: prediction.expiresAt,
          isValid: isPredictionValid(prediction),
        });
      } catch {
        // Skip corrupted files
      }
    }
  } catch {
    // Directory might not exist
  }

  return results;
}

/**
 * Get age of prediction in minutes
 */
export function getPredictionAge(prediction: PredictionResult): number {
  const ageMs = Date.now() - prediction.generatedAt;
  return Math.floor(ageMs / 60000);
}

/**
 * Get time until prediction expires in minutes
 */
export function getTimeUntilExpiry(prediction: PredictionResult): number {
  const remainingMs = prediction.expiresAt - Date.now();
  return Math.max(0, Math.floor(remainingMs / 60000));
}

// ============ PREDICTION TRACKING FOR CALIBRATION ============

/**
 * Tracked prediction for historical analysis and calibration
 */
export interface TrackedPrediction {
  id: string;
  symbol: string;
  assetType: 'stock' | 'crypto';
  predictedAt: number;
  targetDate: number; // When we expect the prediction to resolve
  timeframe: string;
  predictedChangePercent: number;
  actualChangePercent: number | null;
  rawConfidence: number;
  calibratedConfidence: number | null;
  wasCorrect: boolean | null;
  absoluteError: number | null;
  priceAtPrediction: number;
  priceAtTarget: number | null;
  validated: boolean;
}

// Storage file for tracked predictions
const TRACKED_FILE = path.join(process.cwd(), 'data', 'tracked_predictions.json');

/**
 * Load tracked predictions from file
 */
async function loadTrackedPredictions(): Promise<TrackedPrediction[]> {
  try {
    const jsonData = await fs.readFile(TRACKED_FILE, 'utf-8');
    return JSON.parse(jsonData) as TrackedPrediction[];
  } catch {
    return [];
  }
}

/**
 * Save tracked predictions to file
 */
async function saveTrackedPredictions(predictions: TrackedPrediction[]): Promise<void> {
  await ensureDirectory();
  // Keep only last 1000 predictions to avoid unbounded growth
  const trimmed = predictions.slice(-1000);
  await fs.writeFile(TRACKED_FILE, JSON.stringify(trimmed, null, 2), 'utf-8');
}

/**
 * Track a new prediction for later validation
 */
export async function trackPrediction(prediction: {
  symbol: string;
  assetType: 'stock' | 'crypto';
  predictedChangePercent: number;
  confidence: number;
  timeframe: string;
  priceAtPrediction: number;
  targetDate: number;
}): Promise<string> {
  const tracked = await loadTrackedPredictions();

  const id = `${prediction.symbol}-${prediction.timeframe}-${Date.now()}`;

  const newPrediction: TrackedPrediction = {
    id,
    symbol: prediction.symbol,
    assetType: prediction.assetType,
    predictedAt: Date.now(),
    targetDate: prediction.targetDate,
    timeframe: prediction.timeframe,
    predictedChangePercent: prediction.predictedChangePercent,
    actualChangePercent: null,
    rawConfidence: prediction.confidence,
    calibratedConfidence: null,
    wasCorrect: null,
    absoluteError: null,
    priceAtPrediction: prediction.priceAtPrediction,
    priceAtTarget: null,
    validated: false,
  };

  tracked.push(newPrediction);
  await saveTrackedPredictions(tracked);

  return id;
}

/**
 * Validate a tracked prediction with actual outcome
 */
export async function validateTrackedPrediction(
  id: string,
  actualPrice: number
): Promise<TrackedPrediction | null> {
  const tracked = await loadTrackedPredictions();

  const index = tracked.findIndex(p => p.id === id);
  if (index === -1) return null;

  const prediction = tracked[index];

  // Calculate actual change
  const actualChangePercent =
    ((actualPrice - prediction.priceAtPrediction) / prediction.priceAtPrediction) * 100;

  // Determine if prediction was correct (direction match)
  const threshold = 0.5; // 0.5% threshold for "neutral"
  let wasCorrect: boolean;

  if (Math.abs(prediction.predictedChangePercent) < threshold && Math.abs(actualChangePercent) < threshold) {
    wasCorrect = true; // Both neutral
  } else {
    wasCorrect =
      (prediction.predictedChangePercent > 0 && actualChangePercent > 0) ||
      (prediction.predictedChangePercent < 0 && actualChangePercent < 0);
  }

  // Update prediction
  prediction.actualChangePercent = actualChangePercent;
  prediction.priceAtTarget = actualPrice;
  prediction.wasCorrect = wasCorrect;
  prediction.absoluteError = Math.abs(prediction.predictedChangePercent - actualChangePercent);
  prediction.validated = true;

  tracked[index] = prediction;
  await saveTrackedPredictions(tracked);

  return prediction;
}

/**
 * Validate all pending predictions that have reached their target date
 * Returns list of predictions that need price data to validate
 */
export async function getPendingValidations(): Promise<TrackedPrediction[]> {
  const tracked = await loadTrackedPredictions();
  const now = Date.now();

  return tracked.filter(p => !p.validated && p.targetDate <= now);
}

/**
 * Get validated predictions for calibration
 */
export async function getValidatedPredictions(): Promise<TrackedPrediction[]> {
  const tracked = await loadTrackedPredictions();
  return tracked.filter(p => p.validated && p.wasCorrect !== null);
}

/**
 * Get accuracy statistics by confidence bucket
 */
export async function getAccuracyByConfidence(): Promise<Map<string, { accuracy: number; count: number }>> {
  const validated = await getValidatedPredictions();
  const buckets = new Map<string, { correct: number; total: number }>();

  // Create buckets: 0.3-0.4, 0.4-0.5, 0.5-0.6, 0.6-0.7, 0.7-0.8, 0.8-0.9, 0.9-1.0
  for (const pred of validated) {
    const bucketStart = Math.floor(pred.rawConfidence * 10) / 10;
    const bucketKey = `${bucketStart.toFixed(1)}-${(bucketStart + 0.1).toFixed(1)}`;

    if (!buckets.has(bucketKey)) {
      buckets.set(bucketKey, { correct: 0, total: 0 });
    }

    const bucket = buckets.get(bucketKey)!;
    bucket.total++;
    if (pred.wasCorrect) bucket.correct++;
  }

  // Convert to accuracy
  const result = new Map<string, { accuracy: number; count: number }>();
  for (const [key, value] of buckets) {
    result.set(key, {
      accuracy: value.total > 0 ? value.correct / value.total : 0,
      count: value.total,
    });
  }

  return result;
}

/**
 * Get overall prediction statistics
 */
export async function getPredictionStats(): Promise<{
  total: number;
  validated: number;
  pending: number;
  correctCount: number;
  accuracy: number;
  averageError: number;
  byTimeframe: Map<string, { accuracy: number; count: number }>;
}> {
  const tracked = await loadTrackedPredictions();

  const validated = tracked.filter(p => p.validated);
  const pending = tracked.filter(p => !p.validated);
  const correct = validated.filter(p => p.wasCorrect);

  const totalError = validated.reduce((sum, p) => sum + (p.absoluteError || 0), 0);

  // Group by timeframe
  const byTimeframe = new Map<string, { correct: number; total: number }>();
  for (const pred of validated) {
    if (!byTimeframe.has(pred.timeframe)) {
      byTimeframe.set(pred.timeframe, { correct: 0, total: 0 });
    }
    const tf = byTimeframe.get(pred.timeframe)!;
    tf.total++;
    if (pred.wasCorrect) tf.correct++;
  }

  const timeframeAccuracy = new Map<string, { accuracy: number; count: number }>();
  for (const [key, value] of byTimeframe) {
    timeframeAccuracy.set(key, {
      accuracy: value.total > 0 ? value.correct / value.total : 0,
      count: value.total,
    });
  }

  return {
    total: tracked.length,
    validated: validated.length,
    pending: pending.length,
    correctCount: correct.length,
    accuracy: validated.length > 0 ? correct.length / validated.length : 0,
    averageError: validated.length > 0 ? totalError / validated.length : 0,
    byTimeframe: timeframeAccuracy,
  };
}

/**
 * Clean up old tracked predictions (keep last 30 days)
 */
export async function cleanupOldTrackedPredictions(maxAgeDays: number = 30): Promise<number> {
  const tracked = await loadTrackedPredictions();
  const cutoff = Date.now() - maxAgeDays * 24 * 60 * 60 * 1000;

  const filtered = tracked.filter(p => p.predictedAt > cutoff);
  const removed = tracked.length - filtered.length;

  if (removed > 0) {
    await saveTrackedPredictions(filtered);
  }

  return removed;
}
