// AI Stock Prediction Subproject
// Export all types and services for use in the UI
// NOTE: ML modules are NOT exported here to avoid bundling TensorFlow.js in client

export * from './types';
export { stockService } from './stockService';
export { predictionService, type TrainingStatus } from './predictionService';
