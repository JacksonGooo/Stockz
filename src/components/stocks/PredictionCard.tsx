'use client';

import { memo, useMemo } from 'react';
import { Card, CardHeader, CardTitle } from '../ui/Card';
import { Badge } from '../ui/Badge';
import { PredictionResult } from '@/ai/types';
import { formatDateTimeCST } from '@/lib/time';

interface PredictionCardProps {
  prediction: PredictionResult;
}

// Memoized PredictionCard to prevent unnecessary re-renders
export const PredictionCard = memo(function PredictionCard({ prediction }: PredictionCardProps) {
  // Memoize computed values
  const { isPositive, confidenceColor, formattedDate } = useMemo(() => ({
    isPositive: prediction.predictedChange >= 0,
    confidenceColor: prediction.confidence >= 0.7
      ? 'success' as const
      : prediction.confidence >= 0.5
        ? 'warning' as const
        : 'danger' as const,
    formattedDate: formatDateTimeCST(prediction.generatedAt),
  }), [prediction.predictedChange, prediction.confidence, prediction.generatedAt]);

  // Memoize factors to avoid re-creating slice
  const displayFactors = useMemo(
    () => prediction.factors.slice(0, 3),
    [prediction.factors]
  );

  return (
    <Card variant="glass" className="relative overflow-hidden">
      {/* Background gradient accent */}
      <div
        className={`
          absolute top-0 right-0 w-32 h-32 rounded-full blur-3xl opacity-20
          ${isPositive ? 'bg-emerald-500' : 'bg-red-500'}
        `}
      />

      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <svg
            className="w-5 h-5 text-violet-500"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z"
            />
          </svg>
          AI Prediction
        </CardTitle>
        <Badge variant={confidenceColor} size="sm">
          {Math.round(prediction.confidence * 100)}% confidence
        </Badge>
      </CardHeader>

      <div className="space-y-4">
        {/* Price prediction */}
        <div className="flex items-center justify-between p-4 rounded-xl bg-zinc-100/50 dark:bg-zinc-800/50">
          <div>
            <p className="text-sm text-zinc-500 dark:text-zinc-400">
              Current Price
            </p>
            <p className="text-xl font-bold font-mono text-zinc-900 dark:text-zinc-100">
              ${prediction.currentPrice.toFixed(2)}
            </p>
          </div>
          <div className="flex items-center gap-2">
            <svg
              className={`w-6 h-6 ${
                isPositive
                  ? 'text-emerald-500 rotate-0'
                  : 'text-red-500 rotate-180'
              }`}
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M5 10l7-7m0 0l7 7m-7-7v18"
              />
            </svg>
          </div>
          <div className="text-right">
            <p className="text-sm text-zinc-500 dark:text-zinc-400">
              Predicted (60 min)
            </p>
            <p
              className={`text-xl font-bold font-mono ${
                isPositive
                  ? 'text-emerald-600 dark:text-emerald-400'
                  : 'text-red-600 dark:text-red-400'
              }`}
            >
              ${prediction.predictedPrice.toFixed(2)}
            </p>
          </div>
        </div>

        {/* Change stats */}
        <div className="grid grid-cols-2 gap-3">
          <div className="p-3 rounded-xl bg-zinc-100/50 dark:bg-zinc-800/50">
            <p className="text-xs text-zinc-500 dark:text-zinc-400 mb-1">
              Expected Change
            </p>
            <p
              className={`text-lg font-bold font-mono ${
                isPositive
                  ? 'text-emerald-600 dark:text-emerald-400'
                  : 'text-red-600 dark:text-red-400'
              }`}
            >
              {isPositive ? '+' : ''}${prediction.predictedChange.toFixed(2)}
            </p>
          </div>
          <div className="p-3 rounded-xl bg-zinc-100/50 dark:bg-zinc-800/50">
            <p className="text-xs text-zinc-500 dark:text-zinc-400 mb-1">
              % Change
            </p>
            <p
              className={`text-lg font-bold font-mono ${
                isPositive
                  ? 'text-emerald-600 dark:text-emerald-400'
                  : 'text-red-600 dark:text-red-400'
              }`}
            >
              {isPositive ? '+' : ''}
              {prediction.predictedChangePercent.toFixed(2)}%
            </p>
          </div>
        </div>

        {/* Factors */}
        <div>
          <p className="text-sm font-medium text-zinc-700 dark:text-zinc-300 mb-2">
            Key Factors
          </p>
          <div className="space-y-2">
            {displayFactors.map((factor, index) => (
              <div
                key={index}
                className="flex items-center gap-2 p-2 rounded-lg bg-zinc-50 dark:bg-zinc-800/30"
              >
                <div
                  className={`
                    w-2 h-2 rounded-full
                    ${factor.impact === 'positive' ? 'bg-emerald-500' : ''}
                    ${factor.impact === 'negative' ? 'bg-red-500' : ''}
                    ${factor.impact === 'neutral' ? 'bg-zinc-400' : ''}
                  `}
                />
                <span className="text-sm text-zinc-600 dark:text-zinc-400 flex-1">
                  {factor.name}
                </span>
                <span className="text-xs text-zinc-500 dark:text-zinc-500">
                  {Math.round(factor.weight * 100)}%
                </span>
              </div>
            ))}
          </div>
        </div>

        {/* Generated timestamp */}
        <p className="text-xs text-zinc-400 dark:text-zinc-500 text-center">
          Generated {formattedDate}
        </p>
      </div>
    </Card>
  );
});
