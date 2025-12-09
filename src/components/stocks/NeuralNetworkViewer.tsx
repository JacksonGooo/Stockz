'use client';

import { useState, useEffect } from 'react';
import { Card, Badge } from '@/components/ui';

interface NetworkNode {
  id: string;
  label: string;
  value?: number;
}

interface NetworkLayer {
  name: string;
  nodes: NetworkNode[];
}

export interface NeuralNetworkData {
  inputs: Array<{
    name: string;
    value: number;
    min: number;
    max: number;
  }>;
  outputs: {
    buy: number; // percentage 0-100
    dontBuy: number; // percentage 0-100
    notSure: number; // percentage 0-100
  };
  confidence: number;
}

interface Props {
  data?: NeuralNetworkData;
  isLoading?: boolean;
}

export function NeuralNetworkViewer({ data, isLoading = false }: Props) {
  const [selectedInput, setSelectedInput] = useState<string | null>(null);

  const defaultData: NeuralNetworkData = {
    inputs: [
      { name: 'RSI (14)', value: 65, min: 0, max: 100 },
      { name: 'MACD', value: 0.45, min: -2, max: 2 },
      { name: 'SMA 20', value: 155.30, min: 140, max: 170 },
      { name: 'SMA 50', value: 152.10, min: 140, max: 170 },
      { name: 'Bollinger Upper', value: 160.20, min: 150, max: 170 },
      { name: 'Volume (M)', value: 2.5, min: 0.5, max: 5 },
      { name: 'ATR', value: 2.15, min: 0.5, max: 4 },
      { name: 'Stochastic %K', value: 72, min: 0, max: 100 },
    ],
    outputs: {
      buy: 68,
      dontBuy: 12,
      notSure: 20,
    },
    confidence: 0.78,
  };

  const displayData = data || defaultData;

  const getOutputColor = (type: 'buy' | 'dontBuy' | 'notSure') => {
    switch (type) {
      case 'buy':
        return 'from-green-500 to-green-600';
      case 'dontBuy':
        return 'from-red-500 to-red-600';
      case 'notSure':
        return 'from-yellow-500 to-yellow-600';
    }
  };

  const getOutputLabel = (type: 'buy' | 'dontBuy' | 'notSure') => {
    switch (type) {
      case 'buy':
        return 'BUY (+1)';
      case 'dontBuy':
        return "DON'T BUY (-1)";
      case 'notSure':
        return 'NOT SURE (0)';
    }
  };

  return (
    <div className="space-y-6">
      {/* Model Status */}
      <Card className="p-6">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-lg font-semibold text-zinc-900 dark:text-zinc-100">
              Neural Network Status
            </h3>
            <p className="text-sm text-zinc-500 dark:text-zinc-400 mt-1">
              LSTM-based stock prediction model
            </p>
          </div>
          <div className="flex items-center gap-3">
            <div className="w-3 h-3 bg-green-500 rounded-full animate-pulse"></div>
            <span className="text-sm font-medium text-green-600 dark:text-green-400">Active</span>
          </div>
        </div>
      </Card>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Input Layer */}
        <Card className="p-6 lg:col-span-1">
          <h3 className="text-lg font-semibold text-zinc-900 dark:text-zinc-100 mb-4">
            Input Features
          </h3>
          <div className="space-y-3">
            {isLoading ? (
              <div className="space-y-3">
                {[...Array(4)].map((_, i) => (
                  <div key={i} className="h-12 bg-zinc-200 dark:bg-zinc-700 rounded animate-pulse" />
                ))}
              </div>
            ) : (
              displayData.inputs.map((input, idx) => {
                const normalized = ((input.value - input.min) / (input.max - input.min)) * 100;
                return (
                  <div
                    key={idx}
                    className={`p-3 rounded-lg border cursor-pointer transition-all ${
                      selectedInput === input.name
                        ? 'border-blue-500 bg-blue-50 dark:bg-blue-950'
                        : 'border-zinc-200 dark:border-zinc-700 hover:border-zinc-300 dark:hover:border-zinc-600'
                    }`}
                    onClick={() => setSelectedInput(selectedInput === input.name ? null : input.name)}
                  >
                    <div className="flex justify-between items-center mb-2">
                      <span className="text-sm font-medium text-zinc-900 dark:text-zinc-100">
                        {input.name}
                      </span>
                      <span className="text-sm font-semibold text-blue-600 dark:text-blue-400">
                        {input.value}
                      </span>
                    </div>
                    <div className="w-full bg-zinc-200 dark:bg-zinc-700 rounded-full h-2">
                      <div
                        className="h-full bg-gradient-to-r from-blue-500 to-blue-600 rounded-full transition-all"
                        style={{ width: `${Math.min(normalized, 100)}%` }}
                      ></div>
                    </div>
                    <div className="text-xs text-zinc-500 dark:text-zinc-400 mt-1">
                      Range: {input.min} - {input.max}
                    </div>
                  </div>
                );
              })
            )}
          </div>
        </Card>

        {/* Network Visualization */}
        <div className="lg:col-span-1 flex flex-col items-center justify-center">
          <div className="w-full space-y-6">
            {/* Input to Hidden */}
            <div className="flex justify-between items-center px-4">
              <div className="flex flex-col items-center gap-2">
                <div className="text-sm font-medium text-zinc-600 dark:text-zinc-400">Input</div>
                <div className="flex gap-1">
                  {[...Array(8)].map((_, i) => (
                    <div
                      key={i}
                      className="w-6 h-6 rounded-full bg-blue-500"
                      style={{ opacity: 0.4 + (i / 8) * 0.6 }}
                    ></div>
                  ))}
                </div>
              </div>

              <div className="flex-1 flex justify-center">
                <div className="text-xs text-zinc-500 dark:text-zinc-400">LSTM Layer</div>
              </div>

              <div className="flex flex-col items-center gap-2">
                <div className="text-sm font-medium text-zinc-600 dark:text-zinc-400">Hidden</div>
                <div className="flex gap-1">
                  {[...Array(5)].map((_, i) => (
                    <div
                      key={i}
                      className="w-6 h-6 rounded-full bg-purple-500"
                      style={{ opacity: 0.4 + (i / 5) * 0.6 }}
                    ></div>
                  ))}
                </div>
              </div>
            </div>

            {/* Hidden to Output */}
            <div className="flex justify-between items-center px-4">
              <div className="flex flex-col items-center gap-2">
                <div className="text-sm font-medium text-zinc-600 dark:text-zinc-400">Hidden</div>
                <div className="flex gap-1">
                  {[...Array(5)].map((_, i) => (
                    <div
                      key={i}
                      className="w-6 h-6 rounded-full bg-purple-500"
                      style={{ opacity: 0.4 + (i / 5) * 0.6 }}
                    ></div>
                  ))}
                </div>
              </div>

              <div className="flex-1 flex justify-center">
                <div className="text-xs text-zinc-500 dark:text-zinc-400">Dense Layer</div>
              </div>

              <div className="flex flex-col items-center gap-2">
                <div className="text-sm font-medium text-zinc-600 dark:text-zinc-400">Output</div>
                <div className="flex gap-1">
                  {[...Array(3)].map((_, i) => (
                    <div
                      key={i}
                      className="w-6 h-6 rounded-full bg-orange-500"
                      style={{ opacity: 0.4 + (i / 3) * 0.6 }}
                    ></div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Output Layer */}
        <Card className="p-6 lg:col-span-1">
          <h3 className="text-lg font-semibold text-zinc-900 dark:text-zinc-100 mb-4">
            Prediction Output
          </h3>

          {isLoading ? (
            <div className="space-y-4">
              {[...Array(3)].map((_, i) => (
                <div key={i} className="h-20 bg-zinc-200 dark:bg-zinc-700 rounded animate-pulse" />
              ))}
            </div>
          ) : (
            <div className="space-y-4">
              {/* Buy */}
              <div className="p-4 rounded-lg border border-green-200 dark:border-green-900 bg-green-50 dark:bg-green-950">
                <div className="flex items-center justify-between mb-2">
                  <span className="font-semibold text-green-900 dark:text-green-100">
                    {getOutputLabel('buy')}
                  </span>
                  <Badge className="bg-green-600 hover:bg-green-700">
                    {displayData.outputs.buy}%
                  </Badge>
                </div>
                <div className="w-full bg-green-200 dark:bg-green-900 rounded-full h-3">
                  <div
                    className={`h-full bg-gradient-to-r ${getOutputColor('buy')} rounded-full transition-all`}
                    style={{ width: `${displayData.outputs.buy}%` }}
                  ></div>
                </div>
              </div>

              {/* Don't Buy */}
              <div className="p-4 rounded-lg border border-red-200 dark:border-red-900 bg-red-50 dark:bg-red-950">
                <div className="flex items-center justify-between mb-2">
                  <span className="font-semibold text-red-900 dark:text-red-100">
                    {getOutputLabel('dontBuy')}
                  </span>
                  <Badge className="bg-red-600 hover:bg-red-700">
                    {displayData.outputs.dontBuy}%
                  </Badge>
                </div>
                <div className="w-full bg-red-200 dark:bg-red-900 rounded-full h-3">
                  <div
                    className={`h-full bg-gradient-to-r ${getOutputColor('dontBuy')} rounded-full transition-all`}
                    style={{ width: `${displayData.outputs.dontBuy}%` }}
                  ></div>
                </div>
              </div>

              {/* Not Sure */}
              <div className="p-4 rounded-lg border border-yellow-200 dark:border-yellow-900 bg-yellow-50 dark:bg-yellow-950">
                <div className="flex items-center justify-between mb-2">
                  <span className="font-semibold text-yellow-900 dark:text-yellow-100">
                    {getOutputLabel('notSure')}
                  </span>
                  <Badge className="bg-yellow-600 hover:bg-yellow-700">
                    {displayData.outputs.notSure}%
                  </Badge>
                </div>
                <div className="w-full bg-yellow-200 dark:bg-yellow-900 rounded-full h-3">
                  <div
                    className={`h-full bg-gradient-to-r ${getOutputColor('notSure')} rounded-full transition-all`}
                    style={{ width: `${displayData.outputs.notSure}%` }}
                  ></div>
                </div>
              </div>

              {/* Confidence */}
              <div className="mt-4 p-3 rounded-lg bg-zinc-100 dark:bg-zinc-800 border border-zinc-200 dark:border-zinc-700">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm font-medium text-zinc-700 dark:text-zinc-300">
                    Model Confidence
                  </span>
                  <span className="text-lg font-bold text-zinc-900 dark:text-zinc-100">
                    {(displayData.confidence * 100).toFixed(0)}%
                  </span>
                </div>
                <div className="w-full bg-zinc-300 dark:bg-zinc-700 rounded-full h-2">
                  <div
                    className="h-full bg-gradient-to-r from-indigo-500 to-indigo-600 rounded-full"
                    style={{ width: `${displayData.confidence * 100}%` }}
                  ></div>
                </div>
              </div>
            </div>
          )}
        </Card>
      </div>

      {/* Information Card */}
      <Card className="p-6">
        <h3 className="text-lg font-semibold text-zinc-900 dark:text-zinc-100 mb-3">
          How It Works
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
          <div>
            <h4 className="font-medium text-zinc-900 dark:text-zinc-100 mb-2">Input Features</h4>
            <p className="text-zinc-600 dark:text-zinc-400">
              Technical indicators (RSI, MACD, Moving Averages, etc.) feed into the neural network.
            </p>
          </div>
          <div>
            <h4 className="font-medium text-zinc-900 dark:text-zinc-100 mb-2">Processing</h4>
            <p className="text-zinc-600 dark:text-zinc-400">
              LSTM and Dense layers process patterns in historical price data and indicators.
            </p>
          </div>
          <div>
            <h4 className="font-medium text-zinc-900 dark:text-zinc-100 mb-2">Output Decision</h4>
            <p className="text-zinc-600 dark:text-zinc-400">
              Outputs probabilities for Buy (1), Don't Buy (-1), or Not Sure (0) signals.
            </p>
          </div>
        </div>
      </Card>
    </div>
  );
}
