'use client';

import { HTMLAttributes, forwardRef } from 'react';

type BadgeVariant = 'default' | 'success' | 'warning' | 'danger' | 'info';
type BadgeSize = 'sm' | 'md' | 'lg';

interface BadgeProps extends HTMLAttributes<HTMLSpanElement> {
  variant?: BadgeVariant;
  size?: BadgeSize;
  pulse?: boolean;
}

const variantStyles: Record<BadgeVariant, string> = {
  default: 'bg-zinc-100 text-zinc-700 dark:bg-zinc-800 dark:text-zinc-300',
  success: 'bg-emerald-100 text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-400',
  warning: 'bg-amber-100 text-amber-700 dark:bg-amber-900/30 dark:text-amber-400',
  danger: 'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400',
  info: 'bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-400',
};

const sizeStyles: Record<BadgeSize, string> = {
  sm: 'px-2 py-0.5 text-xs',
  md: 'px-2.5 py-1 text-xs',
  lg: 'px-3 py-1.5 text-sm',
};

export const Badge = forwardRef<HTMLSpanElement, BadgeProps>(
  (
    {
      variant = 'default',
      size = 'md',
      pulse = false,
      className = '',
      children,
      ...props
    },
    ref
  ) => {
    return (
      <span
        ref={ref}
        className={`
          inline-flex items-center justify-center font-medium rounded-full
          transition-colors duration-200
          ${variantStyles[variant]}
          ${sizeStyles[size]}
          ${className}
        `}
        {...props}
      >
        {pulse && (
          <span className="relative flex h-2 w-2 mr-1.5">
            <span
              className={`
                animate-ping absolute inline-flex h-full w-full rounded-full opacity-75
                ${variant === 'success' ? 'bg-emerald-400' : ''}
                ${variant === 'danger' ? 'bg-red-400' : ''}
                ${variant === 'warning' ? 'bg-amber-400' : ''}
                ${variant === 'info' ? 'bg-blue-400' : ''}
                ${variant === 'default' ? 'bg-zinc-400' : ''}
              `}
            />
            <span
              className={`
                relative inline-flex rounded-full h-2 w-2
                ${variant === 'success' ? 'bg-emerald-500' : ''}
                ${variant === 'danger' ? 'bg-red-500' : ''}
                ${variant === 'warning' ? 'bg-amber-500' : ''}
                ${variant === 'info' ? 'bg-blue-500' : ''}
                ${variant === 'default' ? 'bg-zinc-500' : ''}
              `}
            />
          </span>
        )}
        {children}
      </span>
    );
  }
);

Badge.displayName = 'Badge';

// Price change badge component
interface PriceChangeBadgeProps {
  change: number;
  changePercent: number;
  showPercent?: boolean;
  size?: BadgeSize;
}

export function PriceChangeBadge({
  change,
  changePercent,
  showPercent = true,
  size = 'md',
}: PriceChangeBadgeProps) {
  const isPositive = change >= 0;

  return (
    <Badge
      variant={isPositive ? 'success' : 'danger'}
      size={size}
      className="font-mono"
    >
      {isPositive ? '+' : ''}
      {showPercent ? `${changePercent.toFixed(2)}%` : `$${change.toFixed(2)}`}
    </Badge>
  );
}
