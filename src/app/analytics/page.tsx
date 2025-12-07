'use client';

import dynamic from 'next/dynamic';
import { Layout } from '@/components/layout';
import { Card } from '@/components/ui';

const AnalyticsContent = dynamic(
  () => import('./AnalyticsContent').then((mod) => mod.AnalyticsContent),
  {
    loading: () => (
      <Layout>
        <div className="animate-pulse">
          <div className="h-8 w-36 bg-zinc-200 dark:bg-zinc-800 rounded mb-2" />
          <div className="h-4 w-64 bg-zinc-200 dark:bg-zinc-800 rounded mb-8" />
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
            <Card>
              <div className="h-6 w-32 bg-zinc-200 dark:bg-zinc-800 rounded mb-6" />
              <div className="space-y-4">
                {[1, 2, 3, 4].map((i) => (
                  <div key={i} className="h-12 bg-zinc-100 dark:bg-zinc-800/50 rounded-xl" />
                ))}
              </div>
            </Card>
            <Card>
              <div className="h-6 w-40 bg-zinc-200 dark:bg-zinc-800 rounded mb-6" />
              <div className="space-y-6">
                {[1, 2, 3, 4].map((i) => (
                  <div key={i} className="h-8 bg-zinc-100 dark:bg-zinc-800/50 rounded" />
                ))}
              </div>
            </Card>
          </div>
        </div>
      </Layout>
    ),
  }
);

export default function AnalyticsPage() {
  return <AnalyticsContent />;
}
