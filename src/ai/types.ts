// Stock data types for the AI prediction system

export interface Stock {
  symbol: string;
  name: string;
  sector: string;
  currentPrice: number;
  previousClose: number;
  change: number;
  changePercent: number;
  volume: number;
  marketCap: number;
  high52Week: number;
  low52Week: number;
}

export interface StockHistoricalData {
  symbol: string;
  timestamp: Date;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface PredictionResult {
  symbol: string;
  currentPrice: number;
  predictedPrice: number;
  predictedChange: number;
  predictedChangePercent: number;
  confidence: number; // 0-1 scale
  timeframe: PredictionTimeframe;
  generatedAt: Date;
  factors: PredictionFactor[];
}

export interface PredictionFactor {
  name: string;
  impact: 'positive' | 'negative' | 'neutral';
  weight: number; // 0-1 scale
  description: string;
}

export type PredictionTimeframe = '30m' | '1d' | '1w' | '1m' | '3m' | '6m' | '1y';

export interface ModelMetrics {
  accuracy: number;
  precision: number;
  recall: number;
  f1Score: number;
  lastTrainedAt: Date;
  dataPointsUsed: number;
}

export interface AIServiceStatus {
  isOnline: boolean;
  modelVersion: string;
  lastUpdated: Date;
  metrics: ModelMetrics;
}

export interface WatchlistItem {
  stock: Stock;
  addedAt: Date;
  alertPrice?: number;
  notes?: string;
}

export interface MarketSentiment {
  overall: 'bullish' | 'bearish' | 'neutral';
  score: number; // -100 to 100
  newsImpact: number;
  socialMediaImpact: number;
  technicalIndicators: number;
}
