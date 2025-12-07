'use client';

import dynamic from 'next/dynamic';
import { Layout } from '@/components/layout';
import { Card } from '@/components/ui';

const PredictionsContent = dynamic(
  () => import('./PredictionsContent').then((mod) => mod.PredictionsContent),
  {
    loading: () => (
      <Layout>
        <div className="animate-pulse">
          <div className="h-8 w-48 bg-zinc-200 dark:bg-zinc-800 rounded mb-2" />
          <div className="h-4 w-80 bg-zinc-200 dark:bg-zinc-800 rounded mb-8" />
          <Card className="mb-8">
            <div className="h-6 w-48 bg-zinc-200 dark:bg-zinc-800 rounded mb-4" />
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
              {[1, 2, 3, 4].map((i) => (
                <div key={i} className="h-16 bg-zinc-100 dark:bg-zinc-800/50 rounded-xl" />
              ))}
            </div>
          </Card>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {[1, 2, 3].map((i) => (
              <Card key={i}>
                <div className="h-6 w-24 bg-zinc-200 dark:bg-zinc-800 rounded mb-4" />
                <div className="h-20 bg-zinc-100 dark:bg-zinc-800/50 rounded" />
              </Card>
            ))}
          </div>
        </div>
      </Layout>
    ),
  }
);

export default function PredictionsPage() {
  return <PredictionsContent />;
}
