'use client';

import { useState, useEffect } from 'react';
import { Layout } from '@/components/layout';
import { CandlestickChart } from '@/components/charts/CandlestickChart';

interface Asset {
  category: string;
  asset: string;
  hasModel: boolean;
  candleCount: number;
}

interface Candle {
  timestamp: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

interface PredictionResult {
  asset: string;
  category: string;
  realCandles: Candle[];
  predictedCandles: Candle[];
  summary: {
    currentPrice: number;
    predictedPrice: number;
    percentChange: number;
    direction: 'up' | 'down' | 'neutral';
    sma20: number;
    rsi: number;
  };
  meta: {
    inputMinutes: number;
    outputMinutes: number;
    realCandleCount: number;
    predictedCandleCount: number;
    timestamp: string;
  };
}

export default function AIPage() {
  const [assets, setAssets] = useState<Asset[]>([]);
  const [selectedAsset, setSelectedAsset] = useState<Asset | null>(null);
  const [selectedCategory, setSelectedCategory] = useState<string>('All');
  const [loading, setLoading] = useState(false);
  const [predicting, setPredicting] = useState(false);
  const [prediction, setPrediction] = useState<PredictionResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Load asset data
  const loadAsset = async (asset: Asset) => {
    setSelectedAsset(asset);
    setLoading(true);
    setError(null);
    setPrediction(null);

    try {
      // Just load the real candles first (without prediction)
      const res = await fetch('/api/ai/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          category: asset.category,
          asset: asset.asset,
          inputMinutes: 30,
          outputMinutes: 0, // Don't predict yet
        }),
      });

      const data = await res.json();

      if (data.error) {
        setError(data.message || data.error);
      } else {
        setPrediction({
          ...data,
          predictedCandles: [], // Clear predictions until user runs network
        });
      }
    } catch (err) {
      setError('Failed to load asset data');
    } finally {
      setLoading(false);
    }
  };

  // Run the neural network prediction
  const runPrediction = async () => {
    if (!selectedAsset) return;

    setPredicting(true);
    setError(null);

    try {
      const res = await fetch('/api/ai/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          category: selectedAsset.category,
          asset: selectedAsset.asset,
          inputMinutes: 30,
          outputMinutes: 60,
        }),
      });

      const data = await res.json();

      if (data.error) {
        setError(data.message || data.error);
      } else {
        setPrediction(data);
      }
    } catch (err) {
      setError('Prediction failed');
    } finally {
      setPredicting(false);
    }
  };

  // BTC is the only trained model currently - auto-load on mount
  useEffect(() => {
    const btcAsset: Asset = {
      category: 'Crypto',
      asset: 'BTC',
      hasModel: true,
      candleCount: 1000000
    };
    setAssets([btcAsset]);
    loadAsset(btcAsset);
  }, []);

  const categoryOrder = ['Crypto'];  // Only crypto available currently

  // Filter assets by selected category
  const filteredAssets = selectedCategory === 'All'
    ? assets
    : assets.filter(a => a.category === selectedCategory);

  return (
    <Layout>
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-zinc-900 dark:text-white mb-2">BTC AI Prediction</h1>
        <p className="text-zinc-500 dark:text-zinc-400">
          LSTM Neural Network trained on 1M+ candles. Analyzes 30 minutes of data to predict the next 60 minutes.
          <span className="text-yellow-500 ml-2">(Experimental - 50% direction accuracy on backtests)</span>
        </p>
      </div>

        <div className="w-full">
          {/* Chart and Controls */}
          <div className="w-full">
            {!selectedAsset ? (
              <div className="bg-zinc-900 rounded-xl border border-zinc-800 p-8 text-center">
                <div className="text-6xl mb-4">🧠</div>
                <h2 className="text-xl font-semibold text-white mb-2">No Asset Selected</h2>
                <p className="text-zinc-400">
                  Select an asset from the left panel to load its neural network model.
                </p>
              </div>
            ) : loading ? (
              <div className="bg-zinc-900 rounded-xl border border-zinc-800 p-8 text-center">
                <div className="animate-spin text-4xl mb-4">⚙️</div>
                <h2 className="text-xl font-semibold text-white mb-2">Loading {selectedAsset.asset}...</h2>
                <p className="text-zinc-400">Fetching recent market data</p>
              </div>
            ) : error ? (
              <div className="bg-zinc-900 rounded-xl border border-zinc-800 p-8 text-center">
                <div className="text-6xl mb-4">⚠️</div>
                <h2 className="text-xl font-semibold text-red-400 mb-2">Error</h2>
                <p className="text-zinc-400">{error}</p>
              </div>
            ) : prediction ? (
              <div className="space-y-6">
                {/* Header with Run Button */}
                <div className="bg-zinc-900 rounded-xl border border-zinc-800 p-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <h2 className="text-2xl font-bold text-white">
                        {prediction.asset}
                        <span className="text-zinc-500 text-lg ml-2">
                          {prediction.category}
                        </span>
                      </h2>
                      <p className="text-zinc-400 text-sm mt-1">
                        {prediction.meta.realCandleCount} minutes of data loaded
                        {prediction.predictedCandles.length > 0 &&
                          ` | ${prediction.meta.predictedCandleCount} minutes predicted`}
                      </p>
                    </div>
                    <button
                      onClick={runPrediction}
                      disabled={predicting}
                      className={`px-6 py-3 rounded-lg font-semibold transition-all ${
                        predicting
                          ? 'bg-zinc-700 text-zinc-400 cursor-not-allowed'
                          : 'bg-gradient-to-r from-violet-600 to-purple-600 hover:from-violet-500 hover:to-purple-500 text-white shadow-lg shadow-violet-500/25'
                      }`}
                    >
                      {predicting ? (
                        <span className="flex items-center gap-2">
                          <span className="animate-spin">⚙️</span>
                          Running...
                        </span>
                      ) : (
                        '🧠 Run Network'
                      )}
                    </button>
                  </div>
                </div>

                {/* Chart */}
                <div className="bg-zinc-900 rounded-xl border border-zinc-800 p-4 overflow-x-auto">
                  <CandlestickChart
                    realCandles={prediction.realCandles}
                    predictedCandles={prediction.predictedCandles}
                    width={800}
                    height={400}
                  />
                </div>

                {/* Summary Stats */}
                {prediction.predictedCandles.length > 0 && (
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    <div className="bg-zinc-900 rounded-xl border border-zinc-800 p-4">
                      <div className="text-zinc-500 text-sm">Current Price</div>
                      <div className="text-2xl font-bold text-white">
                        ${prediction.summary.currentPrice.toFixed(2)}
                      </div>
                    </div>
                    <div className="bg-zinc-900 rounded-xl border border-zinc-800 p-4">
                      <div className="text-zinc-500 text-sm">Predicted Price (60m)</div>
                      <div className={`text-2xl font-bold ${
                        prediction.summary.direction === 'up'
                          ? 'text-green-400'
                          : prediction.summary.direction === 'down'
                          ? 'text-red-400'
                          : 'text-zinc-400'
                      }`}>
                        ${prediction.summary.predictedPrice.toFixed(2)}
                      </div>
                    </div>
                    <div className="bg-zinc-900 rounded-xl border border-zinc-800 p-4">
                      <div className="text-zinc-500 text-sm">Expected Change</div>
                      <div className={`text-2xl font-bold ${
                        prediction.summary.percentChange > 0
                          ? 'text-green-400'
                          : prediction.summary.percentChange < 0
                          ? 'text-red-400'
                          : 'text-zinc-400'
                      }`}>
                        {prediction.summary.percentChange > 0 ? '+' : ''}
                        {prediction.summary.percentChange.toFixed(2)}%
                      </div>
                    </div>
                    <div className="bg-zinc-900 rounded-xl border border-zinc-800 p-4">
                      <div className="text-zinc-500 text-sm">RSI</div>
                      <div className={`text-2xl font-bold ${
                        prediction.summary.rsi > 70
                          ? 'text-red-400'
                          : prediction.summary.rsi < 30
                          ? 'text-green-400'
                          : 'text-zinc-400'
                      }`}>
                        {prediction.summary.rsi.toFixed(1)}
                      </div>
                    </div>
                  </div>
                )}

                {/* Instructions */}
                {prediction.predictedCandles.length === 0 && (
                  <div className="bg-zinc-800/50 rounded-xl border border-zinc-700 p-6 text-center">
                    <p className="text-zinc-400">
                      Click <span className="text-violet-400 font-semibold">Run Network</span> to
                      analyze the past 30 minutes and predict the next 60 minutes.
                    </p>
                  </div>
                )}
              </div>
            ) : null}
          </div>
        </div>
    </Layout>
  );
}
