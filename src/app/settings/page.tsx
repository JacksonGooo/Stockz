'use client';

import dynamic from 'next/dynamic';
import { Layout } from '@/components/layout';

const SettingsContent = dynamic(
  () => import('./SettingsContent').then((mod) => mod.SettingsContent),
  {
    loading: () => (
      <Layout>
        <div className="animate-pulse">
          <div className="h-8 w-32 bg-zinc-200 dark:bg-zinc-800 rounded mb-2" />
          <div className="h-4 w-64 bg-zinc-200 dark:bg-zinc-800 rounded mb-8" />
          <div className="max-w-2xl space-y-6">
            {[1, 2, 3, 4].map((i) => (
              <div key={i} className="h-48 bg-zinc-200 dark:bg-zinc-800 rounded-2xl" />
            ))}
          </div>
        </div>
      </Layout>
    ),
    ssr: false,
  }
);

export default function SettingsPage() {
  return <SettingsContent />;
}
