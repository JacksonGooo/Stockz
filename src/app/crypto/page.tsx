'use client';

import dynamic from 'next/dynamic';
import { Layout } from '@/components/layout';
import { Card } from '@/components/ui';

const CryptoContent = dynamic(
  () => import('./CryptoContent').then((mod) => mod.CryptoContent),
  {
    loading: () => (
      <Layout>
        <div className="animate-pulse">
          <div className="h-8 w-48 bg-zinc-200 dark:bg-zinc-800 rounded mb-2" />
          <div className="h-4 w-72 bg-zinc-200 dark:bg-zinc-800 rounded mb-8" />
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
            {[1, 2, 3, 4].map((i) => (
              <Card key={i}>
                <div className="h-4 w-20 bg-zinc-200 dark:bg-zinc-800 rounded mb-2" />
                <div className="h-6 w-24 bg-zinc-200 dark:bg-zinc-800 rounded" />
              </Card>
            ))}
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {[1, 2, 3, 4, 5, 6].map((i) => (
              <Card key={i}>
                <div className="h-6 w-24 bg-zinc-200 dark:bg-zinc-800 rounded mb-2" />
                <div className="h-4 w-32 bg-zinc-200 dark:bg-zinc-800 rounded mb-4" />
                <div className="h-8 w-28 bg-zinc-200 dark:bg-zinc-800 rounded" />
              </Card>
            ))}
          </div>
        </div>
      </Layout>
    ),
  }
);

export default function CryptoPage() {
  return <CryptoContent />;
}
