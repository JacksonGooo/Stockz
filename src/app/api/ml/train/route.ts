/**
 * API Route: POST /api/ml/train
 * Trigger model training with real historical data
 *
 * GET - Check training status
 * POST - Start training
 */

import { NextRequest, NextResponse } from 'next/server';
import { StockPredictionModel } from '@/ai/ml/model';
import {
  trainModel,
  quickSyntheticTraining,
  getTrainingProgress,
  isTrainingInProgress,
  loadModelState,
} from '@/ai/ml/modelTrainingService';

// Singleton model for training
let trainingModel: StockPredictionModel | null = null;

function getModel(): StockPredictionModel {
  if (!trainingModel) {
    trainingModel = new StockPredictionModel();
  }
  return trainingModel;
}

/**
 * GET - Check training status and model state
 */
export async function GET() {
  try {
    const progress = getTrainingProgress();
    const modelState = await loadModelState();
    const isTraining = isTrainingInProgress();

    return NextResponse.json({
      isTraining,
      progress,
      modelState: modelState ? {
        trainedAt: modelState.trainedAt,
        accuracy: modelState.accuracy,
        loss: modelState.loss,
        trainingSamples: modelState.trainingSamples,
        symbols: modelState.symbols,
        ageHours: (Date.now() - modelState.trainedAt) / (1000 * 60 * 60),
      } : null,
    });
  } catch (error) {
    console.error('Error checking training status:', error);
    return NextResponse.json(
      { error: 'Failed to check training status' },
      { status: 500 }
    );
  }
}

/**
 * POST - Start training
 */
export async function POST(request: NextRequest) {
  try {
    let body: {
      type?: 'full' | 'quick';
      epochs?: number;
      includeStocks?: boolean;
      includeCrypto?: boolean;
    } = {};

    try {
      body = await request.json();
    } catch {
      // Empty body is okay, use defaults
    }

    const {
      type = 'full',
      epochs = 30,
      includeStocks = true,
      includeCrypto = true,
    } = body;

    // Check if already training
    if (isTrainingInProgress()) {
      return NextResponse.json(
        {
          error: 'Training already in progress',
          progress: getTrainingProgress(),
        },
        { status: 409 }
      );
    }

    const model = getModel();

    // Start training in background
    if (type === 'quick') {
      // Quick synthetic training - faster but less accurate
      quickSyntheticTraining(model).catch(console.error);

      return NextResponse.json({
        success: true,
        message: 'Quick synthetic training started',
        type: 'quick',
      });
    } else {
      // Full training with real data
      trainModel(model, {
        epochs,
        includeStocks,
        includeCrypto,
      }).catch(console.error);

      return NextResponse.json({
        success: true,
        message: 'Full training started with real historical data',
        type: 'full',
        config: { epochs, includeStocks, includeCrypto },
      });
    }
  } catch (error) {
    console.error('Error starting training:', error);
    return NextResponse.json(
      { error: 'Failed to start training' },
      { status: 500 }
    );
  }
}
