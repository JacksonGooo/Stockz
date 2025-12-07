'use client';

import { HTMLAttributes } from 'react';

interface SkeletonProps extends HTMLAttributes<HTMLDivElement> {
  variant?: 'text' | 'circular' | 'rectangular';
  width?: string | number;
  height?: string | number;
}

export function Skeleton({
  variant = 'text',
  width,
  height,
  className = '',
  ...props
}: SkeletonProps) {
  const baseStyles = 'animate-pulse bg-zinc-200 dark:bg-zinc-800';

  const variantStyles = {
    text: 'rounded-md',
    circular: 'rounded-full',
    rectangular: 'rounded-xl',
  };

  const defaultSizes = {
    text: { width: '100%', height: '1rem' },
    circular: { width: '40px', height: '40px' },
    rectangular: { width: '100%', height: '120px' },
  };

  return (
    <div
      className={`${baseStyles} ${variantStyles[variant]} ${className}`}
      style={{
        width: width || defaultSizes[variant].width,
        height: height || defaultSizes[variant].height,
      }}
      {...props}
    />
  );
}

// Pre-built skeleton patterns
export function StockCardSkeleton() {
  return (
    <div className="p-6 rounded-2xl border border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-900">
      <div className="flex items-start justify-between mb-4">
        <div className="flex items-center gap-3">
          <Skeleton variant="circular" width={48} height={48} />
          <div>
            <Skeleton variant="text" width={80} height={20} className="mb-2" />
            <Skeleton variant="text" width={120} height={14} />
          </div>
        </div>
        <Skeleton variant="text" width={60} height={24} />
      </div>
      <div className="flex items-end justify-between">
        <Skeleton variant="text" width={100} height={32} />
        <Skeleton variant="text" width={80} height={20} />
      </div>
    </div>
  );
}

export function ChartSkeleton() {
  return (
    <div className="w-full">
      <div className="flex items-center justify-between mb-4">
        <Skeleton variant="text" width={150} height={24} />
        <div className="flex gap-2">
          <Skeleton variant="text" width={40} height={32} />
          <Skeleton variant="text" width={40} height={32} />
          <Skeleton variant="text" width={40} height={32} />
        </div>
      </div>
      <Skeleton variant="rectangular" height={300} />
    </div>
  );
}

export function TableRowSkeleton({ columns = 5 }: { columns?: number }) {
  return (
    <tr className="border-b border-zinc-100 dark:border-zinc-800">
      {Array.from({ length: columns }).map((_, i) => (
        <td key={i} className="py-4 px-4">
          <Skeleton variant="text" width={i === 0 ? 100 : 60} height={16} />
        </td>
      ))}
    </tr>
  );
}
