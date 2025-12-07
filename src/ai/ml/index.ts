/**
 * ML Module Exports
 */

// Indicators
export {
  calculateAllIndicators,
  indicatorsToFeatures,
  getFeatureLabels,
  calculateSMA,
  calculateEMA,
  calculateRSI,
  calculateMACD,
  calculateBollingerBands,
  calculateStochastic,
  calculateATR,
  calculateOBV,
  calculateVWAP,
  type OHLCV,
  type TechnicalIndicators,
} from './indicators';

// Data Provider
export {
  fetchHistoricalData,
  fetchStockQuote,
  fetchMultipleQuotes,
  getSupportedSymbols,
  getDefaultSymbols,
  getStockInfo,
  getStockInfoSync,
  searchStocks,
  addSymbol,
  removeSymbol,
  createPriceStream,
  isMarketOpen,
  clearCache,
  type StockQuote,
  type HistoricalDataPoint,
} from './dataProvider';

// Model
export {
  StockPredictionModel,
  createModel,
  type ModelConfig,
  type TrainingConfig,
  type ModelMetrics,
} from './model';

// Training Service
export {
  getTrainingService,
  resetTrainingService,
  type TrainingStatus,
  type TrainingServiceConfig,
} from './trainingService';

// Prediction Pipeline
export {
  getPredictionPipeline,
  resetPredictionPipeline,
} from './predictionPipeline';

// Model Training Service (new - for minute predictions)
export {
  trainModel,
  quickSyntheticTraining,
  getTrainingProgress,
  isTrainingInProgress,
  loadModelState,
  needsTraining,
  type TrainingProgress,
  type ModelState,
} from './modelTrainingService';

// Minute Prediction Service
export {
  generateMinutePredictions,
  toMinuteCandle,
  getPredictionSummary,
  type MinuteCandle,
  type PredictionResult,
} from './minutePredictionService';
