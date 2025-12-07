'use client';

import { useState, useEffect } from 'react';
import { Layout } from '@/components/layout';
import { Card, Badge } from '@/components/ui';
import { predictionService, AIServiceStatus } from '@/ai';

// Generate deterministic chart data using seeded random
function seededRandokom(seed: number): number {
  const x = Math.sin(seed) * 10000;
  return x - Math.floor(x);
}

function generateChartData(): number[] {
  return Array.from({ length: 30 }, (_, i) => {
    const random = seededRandokom(i + 1);
    return 50 + random * 40;
  });
}

const CHART_DATA = generateChartData();

interface MetricBarProps {
  label: string;
  value: number;
  color: 'blue' | 'emerald' | 'violet' | 'amber';
}

function MetricBar({ label, value, color }: MetricBarProps) {
  const colorStyles = {
    blue: 'from-blue-500 to-blue-400',
    emerald: 'from-emerald-500 to-emerald-400',
    violet: 'from-violet-500 to-violet-400',
    amber: 'from-amber-500 to-amber-400',
  };

  const percentage = value * 100;

  return (
    <div>
      <div className="flex justify-between items-center mb-2">
        <span className="text-sm text-zinc-600 dark:text-zinc-400">{label}</span>
        <span className="text-sm font-semibold text-zinc-900 dark:text-zinc-100">
          {percentage.toFixed(1)}%
        </span>
      </div>
      <div className="h-3 bg-zinc-100 dark:bg-zinc-800 rounded-full overflow-hidden">
        <div
          className={`h-full bg-gradient-to-r ${colorStyles[color]} rounded-full transition-all duration-500`}
          style={{ width: `${percentage}%` }}
        />
      </div>
    </div>
  );
}

interface StatCardProps {
  title: string;
  value: string;
  change: string;
  positive: boolean;
}

function StatCard({ title, value, change, positive }: StatCardProps) {
  return (
    <Card>
      <p className="text-sm text-zinc-500 dark:text-zinc-400 mb-1">{title}</p>
      <p className="text-2xl font-bold text-zinc-900 dark:text-zinc-100">{value}</p>
      <p
        className={`text-sm font-medium mt-1 ${
          positive ? 'text-emerald-600 dark:text-emerald-400' : 'text-red-600 dark:text-red-400'
        }`}
      >
        {change} vs last month
      </p>
    </Card>
  );
}

export function AnalyticsContent() {
  const [serviceStatus, setServiceStatus] = useState<AIServiceStatus | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    async function fetchStatus() {
      try {
        const status = await predictionService.getServiceStatus();
        setServiceStatus(status);
      } catch (error) {
        console.error('Failed to fetch service status:', error);
      } finally {
        setIsLoading(false);
      }
    }
    fetchStatus();
  }, []);

  return (
    <Layout>
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-zinc-900 dark:text-zinc-100 mb-2">
          Analytics
        </h1>
        <p className="text-zinc-500 dark:text-zinc-400">
          AI model performance and system analytics
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
        <Card>
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-lg font-semibold text-zinc-900 dark:text-zinc-100">
              Model Status
            </h2>
            <Badge variant={serviceStatus?.isOnline ? 'success' : 'danger'} pulse>
              {serviceStatus?.isOnline ? 'Online' : 'Offline'}
            </Badge>
          </div>

          <div className="space-y-4">
            <div className="flex justify-between items-center p-3 rounded-xl bg-zinc-50 dark:bg-zinc-800/50">
              <span className="text-zinc-600 dark:text-zinc-400">Version</span>
              <span className="font-mono text-zinc-900 dark:text-zinc-100">
                {serviceStatus?.modelVersion || '—'}
              </span>
            </div>
            <div className="flex justify-between items-center p-3 rounded-xl bg-zinc-50 dark:bg-zinc-800/50">
              <span className="text-zinc-600 dark:text-zinc-400">Last Updated</span>
              <span className="text-zinc-900 dark:text-zinc-100">
                {serviceStatus?.lastUpdated
                  ? new Date(serviceStatus.lastUpdated).toLocaleDateString('en-US')
                  : '—'}
              </span>
            </div>
            <div className="flex justify-between items-center p-3 rounded-xl bg-zinc-50 dark:bg-zinc-800/50">
              <span className="text-zinc-600 dark:text-zinc-400">Data Points</span>
              <span className="font-mono text-zinc-900 dark:text-zinc-100">
                {serviceStatus?.metrics.dataPointsUsed.toLocaleString('en-US') || '—'}
              </span>
            </div>
            <div className="flex justify-between items-center p-3 rounded-xl bg-zinc-50 dark:bg-zinc-800/50">
              <span className="text-zinc-600 dark:text-zinc-400">Last Trained</span>
              <span className="text-zinc-900 dark:text-zinc-100">
                {serviceStatus?.metrics.lastTrainedAt
                  ? new Date(serviceStatus.metrics.lastTrainedAt).toLocaleDateString('en-US')
                  : '—'}
              </span>
            </div>
          </div>
        </Card>

        <Card>
          <h2 className="text-lg font-semibold text-zinc-900 dark:text-zinc-100 mb-6">
            Performance Metrics
          </h2>

          <div className="space-y-6">
            <MetricBar label="Accuracy" value={serviceStatus?.metrics.accuracy || 0} color="blue" />
            <MetricBar label="Precision" value={serviceStatus?.metrics.precision || 0} color="emerald" />
            <MetricBar label="Recall" value={serviceStatus?.metrics.recall || 0} color="violet" />
            <MetricBar label="F1 Score" value={serviceStatus?.metrics.f1Score || 0} color="amber" />
          </div>
        </Card>
      </div>

      <Card>
        <h2 className="text-lg font-semibold text-zinc-900 dark:text-zinc-100 mb-6">
          Prediction Accuracy Over Time
        </h2>

        <div className="h-64 flex items-end gap-2">
          {CHART_DATA.map((height, i) => {
            const isGood = height > 65;
            return (
              <div
                key={i}
                className={`flex-1 rounded-t transition-all duration-300 hover:opacity-80 cursor-pointer ${
                  isGood ? 'bg-emerald-400 dark:bg-emerald-500' : 'bg-amber-400 dark:bg-amber-500'
                }`}
                style={{ height: `${height}%` }}
                title={`Day ${i + 1}: ${height.toFixed(0)}% accuracy`}
              />
            );
          })}
        </div>
        <div className="flex justify-between mt-2 text-xs text-zinc-500 dark:text-zinc-400">
          <span>30 days ago</span>
          <span>Today</span>
        </div>
      </Card>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-8">
        <StatCard title="Total Predictions" value="12,847" change="+8.2%" positive />
        <StatCard title="Correct Predictions" value="9,456" change="+12.1%" positive />
        <StatCard title="Avg Confidence" value="74%" change="+2.3%" positive />
        <StatCard title="Avg Error" value="3.2%" change="-0.5%" positive />
      </div>
    </Layout>
  );
}
