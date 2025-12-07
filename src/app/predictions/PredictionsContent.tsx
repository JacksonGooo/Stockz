'use client';

import { useState, useEffect, useMemo } from 'react';
import { Layout } from '@/components/layout';
import { Card, Button, Badge, Select } from '@/components/ui';
import { PredictionCard } from '@/components/stocks';
import { stockService, predictionService, Stock, PredictionResult, PredictionTimeframe } from '@/ai';

const CRYPTO_SYMBOLS = [
  { symbol: 'BTCUSDT', name: 'Bitcoin' },
  { symbol: 'ETHUSDT', name: 'Ethereum' },
  { symbol: 'BNBUSDT', name: 'BNB' },
  { symbol: 'XRPUSDT', name: 'XRP' },
  { symbol: 'SOLUSDT', name: 'Solana' },
  { symbol: 'ADAUSDT', name: 'Cardano' },
  { symbol: 'DOGEUSDT', name: 'Dogecoin' },
  { symbol: 'AVAXUSDT', name: 'Avalanche' },
  { symbol: 'DOTUSDT', name: 'Polkadot' },
  { symbol: 'LINKUSDT', name: 'Chainlink' },
];

type AssetType = 'stocks' | 'crypto' | 'all';

export function PredictionsContent() {
  const [stocks, setStocks] = useState<Stock[]>([]);
  const [selectedAsset, setSelectedAsset] = useState<string>('');
  const [selectedTimeframe, setSelectedTimeframe] = useState<PredictionTimeframe>('30m');
  const [assetType, setAssetType] = useState<AssetType>('all');
  const [predictions, setPredictions] = useState<PredictionResult[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isBatchLoading, setIsBatchLoading] = useState(false);

  useEffect(() => {
    async function fetchStocks() {
      const allStocks = await stockService.getAllStocks();
      setStocks(allStocks);
      if (allStocks.length > 0) {
        setSelectedAsset(allStocks[0].symbol);
      }
    }
    fetchStocks();
  }, []);

  const assetOptions = useMemo(() => {
    const stockOptions = stocks.map((s) => ({
      value: s.symbol,
      label: `${s.symbol} - ${s.name}`,
      type: 'stock' as const,
    }));

    const cryptoOptions = CRYPTO_SYMBOLS.map((c) => ({
      value: c.symbol,
      label: `${c.symbol} - ${c.name}`,
      type: 'crypto' as const,
    }));

    if (assetType === 'stocks') return stockOptions;
    if (assetType === 'crypto') return cryptoOptions;
    return [...stockOptions, ...cryptoOptions];
  }, [stocks, assetType]);

  useEffect(() => {
    if (assetOptions.length > 0 && !assetOptions.find(a => a.value === selectedAsset)) {
      setSelectedAsset(assetOptions[0].value);
    }
  }, [assetOptions, selectedAsset]);

  const handleSinglePrediction = async () => {
    if (!selectedAsset) return;
    setIsLoading(true);
    try {
      const pred = await predictionService.getPrediction(selectedAsset, selectedTimeframe);
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
      const symbols = assetOptions.map((a) => a.value);
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
  const stockPredictions = predictions.filter(p => !p.symbol.includes('USDT'));
  const cryptoPredictions = predictions.filter(p => p.symbol.includes('USDT'));

  return (
    <Layout>
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-zinc-900 dark:text-zinc-100 mb-2">
          AI Predictions
        </h1>
        <p className="text-zinc-500 dark:text-zinc-400">
          Generate AI-powered price predictions for stocks and cryptocurrencies
        </p>
      </div>

      <Card className="mb-8">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-2">
            <svg className="w-5 h-5 text-violet-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            <span className="text-sm font-medium text-zinc-700 dark:text-zinc-300">
              Predictions for the next <span className="text-violet-600 dark:text-violet-400 font-bold">60 minutes</span>
            </span>
          </div>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <Select
            label="Asset Type"
            value={assetType}
            onChange={(e) => setAssetType(e.target.value as AssetType)}
            options={[
              { value: 'all', label: 'All Assets' },
              { value: 'stocks', label: 'Stocks Only' },
              { value: 'crypto', label: 'Crypto Only' },
            ]}
          />
          <Select
            label="Asset"
            value={selectedAsset}
            onChange={(e) => setSelectedAsset(e.target.value)}
            options={assetOptions}
          />
          <div className="flex items-end">
            <Button
              variant="primary"
              className="w-full"
              onClick={handleSinglePrediction}
              isLoading={isLoading}
            >
              Predict 60 Min
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

      {predictions.length > 0 && (
        <div className="grid grid-cols-2 md:grid-cols-5 gap-4 mb-8">
          <Card>
            <div className="text-center">
              <p className="text-sm text-zinc-500 dark:text-zinc-400 mb-1">Total</p>
              <p className="text-2xl font-bold text-zinc-900 dark:text-zinc-100">
                {predictions.length}
              </p>
            </div>
          </Card>
          <Card>
            <div className="text-center">
              <p className="text-sm text-zinc-500 dark:text-zinc-400 mb-1">Stocks</p>
              <p className="text-2xl font-bold text-blue-600 dark:text-blue-400">
                {stockPredictions.length}
              </p>
            </div>
          </Card>
          <Card>
            <div className="text-center">
              <p className="text-sm text-zinc-500 dark:text-zinc-400 mb-1">Crypto</p>
              <p className="text-2xl font-bold text-orange-600 dark:text-orange-400">
                {cryptoPredictions.length}
              </p>
            </div>
          </Card>
          <Card>
            <div className="text-center">
              <p className="text-sm text-zinc-500 dark:text-zinc-400 mb-1">Bullish</p>
              <p className="text-2xl font-bold text-emerald-600 dark:text-emerald-400">
                {bullishPredictions.length}
              </p>
            </div>
          </Card>
          <Card>
            <div className="text-center">
              <p className="text-sm text-zinc-500 dark:text-zinc-400 mb-1">Bearish</p>
              <p className="text-2xl font-bold text-red-600 dark:text-red-400">
                {bearishPredictions.length}
              </p>
            </div>
          </Card>
        </div>
      )}

      {predictions.length > 0 ? (
        <div className="space-y-8">
          {stockPredictions.length > 0 && (
            <div>
              <h2 className="text-xl font-semibold text-zinc-900 dark:text-zinc-100 mb-4 flex items-center gap-2">
                <svg className="w-5 h-5 text-blue-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
                </svg>
                Stock Predictions
                <Badge variant="default" size="sm">{stockPredictions.length}</Badge>
              </h2>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                {stockPredictions.map((prediction) => (
                  <PredictionCard key={prediction.symbol} prediction={prediction} />
                ))}
              </div>
            </div>
          )}

          {cryptoPredictions.length > 0 && (
            <div>
              <h2 className="text-xl font-semibold text-zinc-900 dark:text-zinc-100 mb-4 flex items-center gap-2">
                <svg className="w-5 h-5 text-orange-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8c-1.657 0-3 .895-3 2s1.343 2 3 2 3 .895 3 2-1.343 2-3 2m0-8c1.11 0 2.08.402 2.599 1M12 8V7m0 1v8m0 0v1m0-1c-1.11 0-2.08-.402-2.599-1M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                Crypto Predictions
                <Badge variant="warning" size="sm">{cryptoPredictions.length}</Badge>
              </h2>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                {cryptoPredictions.map((prediction) => (
                  <PredictionCard key={prediction.symbol} prediction={prediction} />
                ))}
              </div>
            </div>
          )}
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
          <p className="text-zinc-500 dark:text-zinc-400 mb-4">
            Select a stock or cryptocurrency and timeframe to generate an AI prediction
          </p>
          <div className="flex justify-center gap-4">
            <Badge variant="default">Stocks</Badge>
            <Badge variant="warning">Crypto</Badge>
          </div>
        </Card>
      )}
    </Layout>
  );
}
