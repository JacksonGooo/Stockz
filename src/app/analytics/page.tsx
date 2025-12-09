'use client';

import { Layout } from '@/components/layout';
import { NeuralNetworkViewer } from '@/components/stocks';

export default function AnalyticsPage() {
  return (
    <Layout>
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-zinc-900 dark:text-zinc-100 mb-2">
          Neural Network
        </h1>
        <p className="text-zinc-500 dark:text-zinc-400">
          Visualize the AI model's inputs, network structure, and prediction outputs
        </p>
      </div>

      <NeuralNetworkViewer />
    </Layout>
  );
}
