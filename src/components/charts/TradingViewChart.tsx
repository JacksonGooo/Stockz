'use client';

import { useEffect, useRef, memo } from 'react';

interface TradingViewChartProps {
  symbol: string;
  exchange?: string;
  interval?: string;
  theme?: 'light' | 'dark';
  height?: number;
  autosize?: boolean;
  range?: '1D' | '5D' | '1W' | '1M' | '3M' | '6M' | 'YTD' | '1Y' | '5Y' | 'ALL';
}

function TradingViewChartComponent({
  symbol,
  exchange = 'NASDAQ',
  interval = 'D',
  theme = 'dark',
  height = 400,
  autosize = true,
  range = '1W',
}: TradingViewChartProps) {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!containerRef.current) return;

    // Clear any existing widget
    containerRef.current.innerHTML = '';

    // Create the TradingView widget script
    const script = document.createElement('script');
    script.src = 'https://s3.tradingview.com/external-embedding/embed-widget-advanced-chart.js';
    script.type = 'text/javascript';
    script.async = true;

    // Determine the full symbol with exchange
    let fullSymbol = symbol;
    if (!symbol.includes(':')) {
      // Auto-detect exchange for common assets
      if (['BTC', 'ETH', 'SOL', 'XRP', 'ADA', 'DOGE', 'MATIC', 'DOT', 'AVAX', 'LINK', 'UNI', 'LTC', 'ATOM', 'ETC', 'XLM', 'BNB'].includes(symbol.toUpperCase())) {
        fullSymbol = `BINANCE:${symbol.toUpperCase()}USDT`;
      } else if (['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD', 'USDCHF', 'NZDUSD', 'EURGBP'].includes(symbol.toUpperCase())) {
        fullSymbol = `FX:${symbol.toUpperCase()}`;
      } else if (['GOLD', 'SILVER', 'PLATINUM'].includes(symbol.toUpperCase())) {
        fullSymbol = `TVC:${symbol.toUpperCase()}`;
      } else if (['OIL', 'USOIL'].includes(symbol.toUpperCase())) {
        fullSymbol = 'TVC:USOIL';
      } else if (symbol.toUpperCase() === 'NATGAS') {
        fullSymbol = 'TVC:NATGAS';
      } else if (symbol.toUpperCase() === 'COPPER') {
        fullSymbol = 'COMEX:HG1!';
      } else {
        fullSymbol = `${exchange}:${symbol.toUpperCase()}`;
      }
    }

    script.innerHTML = JSON.stringify({
      autosize: autosize,
      height: autosize ? '100%' : height,
      symbol: fullSymbol,
      interval: interval,
      timezone: 'Etc/UTC',
      theme: theme,
      style: '1',
      locale: 'en',
      range: range,
      allow_symbol_change: false,
      calendar: false,
      support_host: 'https://www.tradingview.com',
      hide_top_toolbar: false,
      hide_legend: false,
      save_image: false,
      hide_volume: false,
      backgroundColor: theme === 'dark' ? 'rgba(24, 24, 27, 1)' : 'rgba(255, 255, 255, 1)',
      gridColor: theme === 'dark' ? 'rgba(63, 63, 70, 0.3)' : 'rgba(228, 228, 231, 0.5)',
    });

    containerRef.current.appendChild(script);

    return () => {
      if (containerRef.current) {
        containerRef.current.innerHTML = '';
      }
    };
  }, [symbol, exchange, interval, theme, height, autosize, range]);

  return (
    <div className="tradingview-widget-container" style={{ height: autosize ? '100%' : height }}>
      <div ref={containerRef} style={{ height: '100%', width: '100%' }} />
    </div>
  );
}

export const TradingViewChart = memo(TradingViewChartComponent);
