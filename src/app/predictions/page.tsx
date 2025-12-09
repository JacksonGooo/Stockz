'use client';

import { useState, useEffect } from 'react';
import { Layout } from '@/components/layout';
import { Card, Button, Badge, Select, Input } from '@/components/ui';
import { PredictionCard } from '@/components/stocks';
import { stockService, predictionService, Stock, PredictionResult, PredictionTimeframe } from '@/ai';

export default function PredictionsPage() {
  const [stocks, setStocks] = useState<Stock[]>([]);
  const [selectedStock, setSelectedStock] = useState<string>('');
  const [selectedTimeframe, setSelectedTimeframe] = useState<PredictionTimeframe>('30m');
  const [predictions, setPredictions] = useState<PredictionResult[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isBatchLoading, setIsBatchLoading] = useState(false);

  useEffect(() => {
    async function fetchStocks() {
      const allStocks = await stockService.getAllStocks();
      setStocks(allStocks);
      if (allStocks.length > 0) {
        setSelectedStock(allStocks[0].symbol);
      }
    }
    fetchStocks();
  }, []);

  const handleSinglePrediction = async () => {
    if (!selectedStock) return;
    setIsLoading(true);
    try {
      const pred = await predictionService.getPrediction(selectedStock, selectedTimeframe);
      if (pred) {
        setPredictions((prev) => [pred, ...prev.filter((p) => p.symbol !== pred.symbol)]);
      }
    } catch (error) {
      console.error('Failed to get prediction:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleBatchPredictions = async () => {
    setIsBatchLoading(true);
    try {
      const symbols = stocks.map((s) => s.symbol);
      const preds = await predictionService.getBatchPredictions(symbols, selectedTimeframe);
      setPredictions(preds);
    } catch (error) {
      console.error('Failed to get batch predictions:', error);
    } finally {
      setIsBatchLoading(false);
    }
  };

  const bullishPredictions = predictions.filter((p) => p.predictedChange > 0);
  const bearishPredictions = predictions.filter((p) => p.predictedChange < 0);

  return (
    <Layout>
      {/* Page header */}
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-zinc-900 dark:text-zinc-100 mb-2">
          AI Predictions
        </h1>
        <p className="text-zinc-500 dark:text-zinc-400">
          Generate AI-powered stock price predictions
        </p>
      </div>

      {/* Prediction controls */}
      <Card className="mb-8">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <Select
            label="Stock"
            value={selectedStock}
            onChange={(e) => setSelectedStock(e.target.value)}
            options={stocks.map((s) => ({
              value: s.symbol,
              label: `${s.symbol} - ${s.name}`,
            }))}
          />
          <Select
            label="Timeframe"
            value={selectedTimeframe}
            onChange={(e) => setSelectedTimeframe(e.target.value as PredictionTimeframe)}
            options={[
              { value: '30m', label: '30 Minutes' },
              { value: '1d', label: '1 Day' },
              { value: '1w', label: '1 Week' },
              { value: '1m', label: '1 Month' },
              { value: '3m', label: '3 Months' },
              { value: '6m', label: '6 Months' },
              { value: '1y', label: '1 Year' },
            ]}
          />
          <div className="flex items-end">
            <Button
              variant="primary"
              className="w-full"
              onClick={handleSinglePrediction}
              isLoading={isLoading}
            >
              Get Prediction
            </Button>
          </div>
          <div className="flex items-end">
            <Button
              variant="secondary"
              className="w-full"
              onClick={handleBatchPredictions}
              isLoading={isBatchLoading}
            >
              Predict All
            </Button>
          </div>
        </div>
      </Card>

      {/* Summary stats */}
      {predictions.length > 0 && (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
          <Card>
            <div className="text-center">
              <p className="text-sm text-zinc-500 dark:text-zinc-400 mb-1">Total Predictions</p>
              <p className="text-3xl font-bold text-zinc-900 dark:text-zinc-100">
                {predictions.length}
              </p>
            </div>
          </Card>
          <Card>
            <div className="text-center">
              <p className="text-sm text-zinc-500 dark:text-zinc-400 mb-1">Bullish</p>
              <p className="text-3xl font-bold text-emerald-600 dark:text-emerald-400">
                {bullishPredictions.length}
              </p>
            </div>
          </Card>
          <Card>
            <div className="text-center">
              <p className="text-sm text-zinc-500 dark:text-zinc-400 mb-1">Bearish</p>
              <p className="text-3xl font-bold text-red-600 dark:text-red-400">
                {bearishPredictions.length}
              </p>
            </div>
          </Card>
        </div>
      )}

      {/* Predictions grid */}
      {predictions.length > 0 ? (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {predictions.map((prediction) => (
            <PredictionCard key={prediction.symbol} prediction={prediction} />
          ))}
        </div>
      ) : (
        <Card className="py-16 text-center">
          <svg
            className="w-16 h-16 mx-auto text-zinc-400 mb-4"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={1.5}
              d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z"
            />
          </svg>
          <h3 className="text-lg font-semibold text-zinc-900 dark:text-zinc-100 mb-2">
            No Predictions Yet
          </h3>
          <p className="text-zinc-500 dark:text-zinc-400">
            Select a stock and timeframe to generate an AI prediction
          </p>
        </Card>
      )}
    </Layout>
  );
}
