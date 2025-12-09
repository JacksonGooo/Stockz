/**
 * Technical Indicators for Stock Analysis
 * Used as features for the ML model
 */

export interface OHLCV {
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  timestamp?: Date;
}

export interface TechnicalIndicators {
  // Trend Indicators
  sma5: number;
  sma10: number;
  sma20: number;
  sma50: number;
  ema12: number;
  ema26: number;

  // Momentum Indicators
  rsi: number;
  macd: number;
  macdSignal: number;
  macdHistogram: number;
  stochK: number;
  stochD: number;
  williamR: number;
  roc: number;
  momentum: number;

  // Volatility Indicators
  bollingerUpper: number;
  bollingerMiddle: number;
  bollingerLower: number;
  bollingerWidth: number;
  atr: number;
  volatility: number;

  // Volume Indicators
  volumeSma: number;
  volumeRatio: number;
  obv: number;
  vwap: number;

  // Price Patterns
  priceChange: number;
  priceChangePercent: number;
  highLowRange: number;
  bodySize: number;
  upperShadow: number;
  lowerShadow: number;

  // Normalized values (0-1 range for ML)
  normalized: {
    price: number;
    volume: number;
    rsi: number;
    macd: number;
    bollinger: number;
    momentum: number;
  };
}

/**
 * Calculate Simple Moving Average
 */
export function calculateSMA(data: number[], period: number): number {
  if (data.length < period) return data[data.length - 1] || 0;
  const slice = data.slice(-period);
  return slice.reduce((a, b) => a + b, 0) / period;
}

/**
 * Calculate Exponential Moving Average
 */
export function calculateEMA(data: number[], period: number): number {
  if (data.length === 0) return 0;
  if (data.length < period) return calculateSMA(data, data.length);

  const multiplier = 2 / (period + 1);
  let ema = calculateSMA(data.slice(0, period), period);

  for (let i = period; i < data.length; i++) {
    ema = (data[i] - ema) * multiplier + ema;
  }

  return ema;
}

/**
 * Calculate Relative Strength Index (RSI)
 */
export function calculateRSI(prices: number[], period: number = 14): number {
  if (prices.length < period + 1) return 50;

  const changes: number[] = [];
  for (let i = 1; i < prices.length; i++) {
    changes.push(prices[i] - prices[i - 1]);
  }

  const recentChanges = changes.slice(-period);
  let gains = 0;
  let losses = 0;

  for (const change of recentChanges) {
    if (change > 0) gains += change;
    else losses += Math.abs(change);
  }

  const avgGain = gains / period;
  const avgLoss = losses / period;

  if (avgLoss === 0) return 100;
  const rs = avgGain / avgLoss;
  return 100 - 100 / (1 + rs);
}

/**
 * Calculate MACD (Moving Average Convergence Divergence)
 */
export function calculateMACD(prices: number[]): {
  macd: number;
  signal: number;
  histogram: number;
} {
  const ema12 = calculateEMA(prices, 12);
  const ema26 = calculateEMA(prices, 26);
  const macd = ema12 - ema26;

  // Calculate signal line (9-period EMA of MACD)
  const macdHistory: number[] = [];
  for (let i = 26; i <= prices.length; i++) {
    const slice = prices.slice(0, i);
    const e12 = calculateEMA(slice, 12);
    const e26 = calculateEMA(slice, 26);
    macdHistory.push(e12 - e26);
  }

  const signal = calculateEMA(macdHistory, 9);
  const histogram = macd - signal;

  return { macd, signal, histogram };
}

/**
 * Calculate Bollinger Bands
 */
export function calculateBollingerBands(
  prices: number[],
  period: number = 20,
  stdDev: number = 2
): { upper: number; middle: number; lower: number; width: number } {
  const middle = calculateSMA(prices, period);
  const slice = prices.slice(-period);

  // Calculate standard deviation
  const variance =
    slice.reduce((sum, val) => sum + Math.pow(val - middle, 2), 0) / period;
  const std = Math.sqrt(variance);

  const upper = middle + stdDev * std;
  const lower = middle - stdDev * std;
  const width = (upper - lower) / middle;

  return { upper, middle, lower, width };
}

/**
 * Calculate Stochastic Oscillator
 */
export function calculateStochastic(
  highs: number[],
  lows: number[],
  closes: number[],
  kPeriod: number = 14,
  dPeriod: number = 3
): { k: number; d: number } {
  if (highs.length < kPeriod) return { k: 50, d: 50 };

  const recentHighs = highs.slice(-kPeriod);
  const recentLows = lows.slice(-kPeriod);
  const currentClose = closes[closes.length - 1];

  const highestHigh = Math.max(...recentHighs);
  const lowestLow = Math.min(...recentLows);

  const k =
    highestHigh === lowestLow
      ? 50
      : ((currentClose - lowestLow) / (highestHigh - lowestLow)) * 100;

  // Calculate %D (SMA of %K)
  const kValues: number[] = [];
  for (let i = kPeriod; i <= closes.length; i++) {
    const h = highs.slice(i - kPeriod, i);
    const l = lows.slice(i - kPeriod, i);
    const c = closes[i - 1];
    const hh = Math.max(...h);
    const ll = Math.min(...l);
    kValues.push(hh === ll ? 50 : ((c - ll) / (hh - ll)) * 100);
  }

  const d = calculateSMA(kValues, dPeriod);

  return { k, d };
}

/**
 * Calculate Williams %R
 */
export function calculateWilliamsR(
  highs: number[],
  lows: number[],
  closes: number[],
  period: number = 14
): number {
  if (highs.length < period) return -50;

  const recentHighs = highs.slice(-period);
  const recentLows = lows.slice(-period);
  const currentClose = closes[closes.length - 1];

  const highestHigh = Math.max(...recentHighs);
  const lowestLow = Math.min(...recentLows);

  if (highestHigh === lowestLow) return -50;
  return ((highestHigh - currentClose) / (highestHigh - lowestLow)) * -100;
}

/**
 * Calculate Average True Range (ATR)
 */
export function calculateATR(
  highs: number[],
  lows: number[],
  closes: number[],
  period: number = 14
): number {
  if (highs.length < 2) return 0;

  const trueRanges: number[] = [];
  for (let i = 1; i < highs.length; i++) {
    const tr = Math.max(
      highs[i] - lows[i],
      Math.abs(highs[i] - closes[i - 1]),
      Math.abs(lows[i] - closes[i - 1])
    );
    trueRanges.push(tr);
  }

  return calculateSMA(trueRanges, period);
}

/**
 * Calculate Rate of Change (ROC)
 */
export function calculateROC(prices: number[], period: number = 10): number {
  if (prices.length <= period) return 0;
  const currentPrice = prices[prices.length - 1];
  const pastPrice = prices[prices.length - 1 - period];
  return ((currentPrice - pastPrice) / pastPrice) * 100;
}

/**
 * Calculate On-Balance Volume (OBV)
 */
export function calculateOBV(closes: number[], volumes: number[]): number {
  if (closes.length < 2) return volumes[0] || 0;

  let obv = 0;
  for (let i = 1; i < closes.length; i++) {
    if (closes[i] > closes[i - 1]) {
      obv += volumes[i];
    } else if (closes[i] < closes[i - 1]) {
      obv -= volumes[i];
    }
  }

  return obv;
}

/**
 * Calculate Volume Weighted Average Price (VWAP)
 */
export function calculateVWAP(
  highs: number[],
  lows: number[],
  closes: number[],
  volumes: number[]
): number {
  let cumulativeTPV = 0;
  let cumulativeVolume = 0;

  for (let i = 0; i < closes.length; i++) {
    const typicalPrice = (highs[i] + lows[i] + closes[i]) / 3;
    cumulativeTPV += typicalPrice * volumes[i];
    cumulativeVolume += volumes[i];
  }

  return cumulativeVolume === 0 ? closes[closes.length - 1] : cumulativeTPV / cumulativeVolume;
}

/**
 * Calculate Historical Volatility
 */
export function calculateVolatility(prices: number[], period: number = 20): number {
  if (prices.length < 2) return 0;

  const returns: number[] = [];
  for (let i = 1; i < prices.length; i++) {
    returns.push(Math.log(prices[i] / prices[i - 1]));
  }

  const recentReturns = returns.slice(-period);
  const mean = recentReturns.reduce((a, b) => a + b, 0) / recentReturns.length;
  const variance =
    recentReturns.reduce((sum, r) => sum + Math.pow(r - mean, 2), 0) /
    recentReturns.length;

  // Annualized volatility (assuming 252 trading days)
  return Math.sqrt(variance * 252) * 100;
}

/**
 * Calculate all technical indicators for a given OHLCV dataset
 */
export function calculateAllIndicators(data: OHLCV[]): TechnicalIndicators {
  const closes = data.map((d) => d.close);
  const highs = data.map((d) => d.high);
  const lows = data.map((d) => d.low);
  const volumes = data.map((d) => d.volume);
  const opens = data.map((d) => d.open);

  const currentPrice = closes[closes.length - 1];
  const currentVolume = volumes[volumes.length - 1];
  const previousClose = closes[closes.length - 2] || currentPrice;

  // Trend Indicators
  const sma5 = calculateSMA(closes, 5);
  const sma10 = calculateSMA(closes, 10);
  const sma20 = calculateSMA(closes, 20);
  const sma50 = calculateSMA(closes, 50);
  const ema12 = calculateEMA(closes, 12);
  const ema26 = calculateEMA(closes, 26);

  // Momentum Indicators
  const rsi = calculateRSI(closes, 14);
  const macdData = calculateMACD(closes);
  const stochastic = calculateStochastic(highs, lows, closes, 14, 3);
  const williamR = calculateWilliamsR(highs, lows, closes, 14);
  const roc = calculateROC(closes, 10);
  const momentum = currentPrice - (closes[closes.length - 11] || currentPrice);

  // Volatility Indicators
  const bollinger = calculateBollingerBands(closes, 20, 2);
  const atr = calculateATR(highs, lows, closes, 14);
  const volatility = calculateVolatility(closes, 20);

  // Volume Indicators
  const volumeSma = calculateSMA(volumes, 20);
  const volumeRatio = volumeSma === 0 ? 1 : currentVolume / volumeSma;
  const obv = calculateOBV(closes, volumes);
  const vwap = calculateVWAP(highs, lows, closes, volumes);

  // Price Patterns
  const priceChange = currentPrice - previousClose;
  const priceChangePercent = (priceChange / previousClose) * 100;
  const currentHigh = highs[highs.length - 1];
  const currentLow = lows[lows.length - 1];
  const currentOpen = opens[opens.length - 1];
  const highLowRange = currentHigh - currentLow;
  const bodySize = Math.abs(currentPrice - currentOpen);
  const upperShadow = currentHigh - Math.max(currentPrice, currentOpen);
  const lowerShadow = Math.min(currentPrice, currentOpen) - currentLow;

  // Calculate normalized values for ML input
  const priceRange = Math.max(...closes) - Math.min(...closes);
  const minPrice = Math.min(...closes);
  const volumeRange = Math.max(...volumes) - Math.min(...volumes);
  const minVolume = Math.min(...volumes);

  const normalized = {
    price: priceRange === 0 ? 0.5 : (currentPrice - minPrice) / priceRange,
    volume: volumeRange === 0 ? 0.5 : (currentVolume - minVolume) / volumeRange,
    rsi: rsi / 100,
    macd: Math.tanh(macdData.macd / currentPrice), // Normalize to -1 to 1
    bollinger:
      bollinger.upper === bollinger.lower
        ? 0.5
        : (currentPrice - bollinger.lower) / (bollinger.upper - bollinger.lower),
    momentum: Math.tanh(momentum / currentPrice),
  };

  return {
    sma5,
    sma10,
    sma20,
    sma50,
    ema12,
    ema26,
    rsi,
    macd: macdData.macd,
    macdSignal: macdData.signal,
    macdHistogram: macdData.histogram,
    stochK: stochastic.k,
    stochD: stochastic.d,
    williamR,
    roc,
    momentum,
    bollingerUpper: bollinger.upper,
    bollingerMiddle: bollinger.middle,
    bollingerLower: bollinger.lower,
    bollingerWidth: bollinger.width,
    atr,
    volatility,
    volumeSma,
    volumeRatio,
    obv,
    vwap,
    priceChange,
    priceChangePercent,
    highLowRange,
    bodySize,
    upperShadow,
    lowerShadow,
    normalized,
  };
}

/**
 * Convert indicators to feature array for ML model
 */
export function indicatorsToFeatures(indicators: TechnicalIndicators): number[] {
  return [
    indicators.normalized.price,
    indicators.normalized.volume,
    indicators.normalized.rsi,
    indicators.normalized.macd,
    indicators.normalized.bollinger,
    indicators.normalized.momentum,
    indicators.rsi / 100,
    indicators.stochK / 100,
    indicators.stochD / 100,
    (indicators.williamR + 100) / 100,
    Math.tanh(indicators.roc / 100),
    indicators.bollingerWidth,
    Math.min(indicators.volumeRatio, 3) / 3,
    Math.tanh(indicators.priceChangePercent / 10),
    indicators.volatility / 100,
  ];
}

/**
 * Generate feature labels for debugging/analysis
 */
export function getFeatureLabels(): string[] {
  return [
    'normalized_price',
    'normalized_volume',
    'normalized_rsi',
    'normalized_macd',
    'normalized_bollinger',
    'normalized_momentum',
    'rsi',
    'stoch_k',
    'stoch_d',
    'williams_r',
    'roc',
    'bollinger_width',
    'volume_ratio',
    'price_change_pct',
    'volatility',
  ];
}
