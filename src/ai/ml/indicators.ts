/**
 * Technical Indicators for Stock Analysis - Enhanced Version
 * Expanded feature set for improved ML model accuracy
 */

export interface OHLCV {
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  timestamp?: Date;
}

export interface MarketContext {
  marketTrend: number; // -1 to 1 (bearish to bullish)
  marketVolatility: number; // VIX-like measure
  sectorPerformance: number; // Sector relative strength
  breadthRatio: number; // Advance/decline ratio
  riskOnOff: number; // Risk sentiment -1 to 1
}

export interface TechnicalIndicators {
  // Trend Indicators
  sma5: number;
  sma10: number;
  sma20: number;
  sma50: number;
  sma200: number;
  ema9: number;
  ema12: number;
  ema21: number;
  ema26: number;

  // Momentum Indicators
  rsi: number;
  rsi_sma: number; // Smoothed RSI
  macd: number;
  macdSignal: number;
  macdHistogram: number;
  stochK: number;
  stochD: number;
  williamR: number;
  roc: number;
  roc5: number;
  momentum: number;
  cci: number; // Commodity Channel Index
  mfi: number; // Money Flow Index
  adx: number; // Average Directional Index
  plusDI: number;
  minusDI: number;

  // Volatility Indicators
  bollingerUpper: number;
  bollingerMiddle: number;
  bollingerLower: number;
  bollingerWidth: number;
  bollingerPercentB: number;
  atr: number;
  atrPercent: number;
  volatility: number;
  keltnerUpper: number;
  keltnerLower: number;
  donchianHigh: number;
  donchianLow: number;

  // Volume Indicators
  volumeSma: number;
  volumeRatio: number;
  obv: number;
  obvSlope: number;
  vwap: number;
  cmf: number; // Chaikin Money Flow
  adLine: number; // Accumulation/Distribution

  // Price Patterns & Structure
  priceChange: number;
  priceChangePercent: number;
  highLowRange: number;
  bodySize: number;
  upperShadow: number;
  lowerShadow: number;
  gapUp: number;
  gapDown: number;

  // Support/Resistance
  distanceFromHigh20: number;
  distanceFromLow20: number;
  distanceFromHigh52: number;
  distanceFromLow52: number;
  pivotPoint: number;
  support1: number;
  resistance1: number;

  // Trend Strength
  trendStrength: number;
  pricePositionInRange: number;
  consecutiveUpDays: number;
  consecutiveDownDays: number;

  // Cross-timeframe
  shortTermTrend: number;
  mediumTermTrend: number;
  longTermTrend: number;

  // Market Context (when available)
  marketContext?: MarketContext;

  // Normalized values (0-1 or -1 to 1 range for ML)
  normalized: NormalizedFeatures;
}

export interface NormalizedFeatures {
  // Price & Volume (0-1)
  price: number;
  volume: number;

  // Momentum (-1 to 1 or 0-1)
  rsi: number;
  macd: number;
  stochastic: number;
  momentum: number;
  cci: number;
  mfi: number;

  // Volatility (0-1)
  bollinger: number;
  atr: number;
  volatility: number;

  // Trend (-1 to 1)
  shortTrend: number;
  mediumTrend: number;
  longTrend: number;
  trendAlignment: number;

  // Volume Flow (-1 to 1)
  volumeFlow: number;
  moneyFlow: number;

  // Support/Resistance (0-1)
  pricePosition: number;
  distanceFromResistance: number;
  distanceFromSupport: number;

  // Pattern Recognition
  candlePattern: number;
  gapStrength: number;

  // Market Context (-1 to 1)
  marketSentiment: number;
  sectorStrength: number;
  breadth: number;
}

// ============ CALCULATION FUNCTIONS ============

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

  const macdHistory: number[] = [];
  for (let i = 26; i <= prices.length; i++) {
    const slice = prices.slice(0, i);
    const e12 = calculateEMA(slice, 12);
    const e26 = calculateEMA(slice, 26);
    macdHistory.push(e12 - e26);
  }

  const signal = macdHistory.length > 0 ? calculateEMA(macdHistory, 9) : 0;
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
): { upper: number; middle: number; lower: number; width: number; percentB: number } {
  const middle = calculateSMA(prices, period);
  const slice = prices.slice(-period);
  const currentPrice = prices[prices.length - 1];

  const variance =
    slice.reduce((sum, val) => sum + Math.pow(val - middle, 2), 0) / period;
  const std = Math.sqrt(variance);

  const upper = middle + stdDev * std;
  const lower = middle - stdDev * std;
  const width = middle > 0 ? (upper - lower) / middle : 0;
  const percentB = upper !== lower ? (currentPrice - lower) / (upper - lower) : 0.5;

  return { upper, middle, lower, width, percentB };
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
  return pastPrice !== 0 ? ((currentPrice - pastPrice) / pastPrice) * 100 : 0;
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
 * Calculate OBV Slope (trend of OBV)
 */
export function calculateOBVSlope(closes: number[], volumes: number[], period: number = 10): number {
  if (closes.length < period + 1) return 0;

  const obvValues: number[] = [];
  let obv = 0;

  for (let i = 1; i < closes.length; i++) {
    if (closes[i] > closes[i - 1]) obv += volumes[i];
    else if (closes[i] < closes[i - 1]) obv -= volumes[i];
    obvValues.push(obv);
  }

  if (obvValues.length < period) return 0;

  const recentOBV = obvValues.slice(-period);
  const firstOBV = recentOBV[0];
  const lastOBV = recentOBV[recentOBV.length - 1];

  return firstOBV !== 0 ? (lastOBV - firstOBV) / Math.abs(firstOBV) : 0;
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
    if (prices[i - 1] > 0) {
      returns.push(Math.log(prices[i] / prices[i - 1]));
    }
  }

  const recentReturns = returns.slice(-period);
  if (recentReturns.length === 0) return 0;

  const mean = recentReturns.reduce((a, b) => a + b, 0) / recentReturns.length;
  const variance =
    recentReturns.reduce((sum, r) => sum + Math.pow(r - mean, 2), 0) /
    recentReturns.length;

  return Math.sqrt(variance * 252) * 100;
}

/**
 * Calculate Commodity Channel Index (CCI)
 */
export function calculateCCI(
  highs: number[],
  lows: number[],
  closes: number[],
  period: number = 20
): number {
  if (closes.length < period) return 0;

  const typicalPrices: number[] = [];
  for (let i = 0; i < closes.length; i++) {
    typicalPrices.push((highs[i] + lows[i] + closes[i]) / 3);
  }

  const sma = calculateSMA(typicalPrices, period);
  const recentTP = typicalPrices.slice(-period);
  const meanDeviation = recentTP.reduce((sum, tp) => sum + Math.abs(tp - sma), 0) / period;

  if (meanDeviation === 0) return 0;
  return (typicalPrices[typicalPrices.length - 1] - sma) / (0.015 * meanDeviation);
}

/**
 * Calculate Money Flow Index (MFI)
 */
export function calculateMFI(
  highs: number[],
  lows: number[],
  closes: number[],
  volumes: number[],
  period: number = 14
): number {
  if (closes.length < period + 1) return 50;

  let positiveFlow = 0;
  let negativeFlow = 0;

  for (let i = closes.length - period; i < closes.length; i++) {
    const typicalPrice = (highs[i] + lows[i] + closes[i]) / 3;
    const prevTypicalPrice = (highs[i - 1] + lows[i - 1] + closes[i - 1]) / 3;
    const moneyFlow = typicalPrice * volumes[i];

    if (typicalPrice > prevTypicalPrice) {
      positiveFlow += moneyFlow;
    } else if (typicalPrice < prevTypicalPrice) {
      negativeFlow += moneyFlow;
    }
  }

  if (negativeFlow === 0) return 100;
  const mfRatio = positiveFlow / negativeFlow;
  return 100 - 100 / (1 + mfRatio);
}

/**
 * Calculate ADX (Average Directional Index) with +DI and -DI
 */
export function calculateADX(
  highs: number[],
  lows: number[],
  closes: number[],
  period: number = 14
): { adx: number; plusDI: number; minusDI: number } {
  if (highs.length < period + 1) return { adx: 25, plusDI: 25, minusDI: 25 };

  const plusDM: number[] = [];
  const minusDM: number[] = [];
  const tr: number[] = [];

  for (let i = 1; i < highs.length; i++) {
    const highDiff = highs[i] - highs[i - 1];
    const lowDiff = lows[i - 1] - lows[i];

    plusDM.push(highDiff > lowDiff && highDiff > 0 ? highDiff : 0);
    minusDM.push(lowDiff > highDiff && lowDiff > 0 ? lowDiff : 0);

    const trueRange = Math.max(
      highs[i] - lows[i],
      Math.abs(highs[i] - closes[i - 1]),
      Math.abs(lows[i] - closes[i - 1])
    );
    tr.push(trueRange);
  }

  const smoothedPlusDM = calculateEMA(plusDM, period);
  const smoothedMinusDM = calculateEMA(minusDM, period);
  const smoothedTR = calculateEMA(tr, period);

  const plusDI = smoothedTR !== 0 ? (smoothedPlusDM / smoothedTR) * 100 : 0;
  const minusDI = smoothedTR !== 0 ? (smoothedMinusDM / smoothedTR) * 100 : 0;

  const dx = plusDI + minusDI !== 0 ? (Math.abs(plusDI - minusDI) / (plusDI + minusDI)) * 100 : 0;

  // Calculate ADX as smoothed DX
  const dxValues: number[] = [];
  for (let i = period; i < highs.length; i++) {
    // Simplified DX calculation for each point
    dxValues.push(dx);
  }
  const adx = dxValues.length > 0 ? calculateEMA(dxValues, period) : 25;

  return { adx, plusDI, minusDI };
}

/**
 * Calculate Chaikin Money Flow (CMF)
 */
export function calculateCMF(
  highs: number[],
  lows: number[],
  closes: number[],
  volumes: number[],
  period: number = 20
): number {
  if (closes.length < period) return 0;

  let mfvSum = 0;
  let volumeSum = 0;

  for (let i = closes.length - period; i < closes.length; i++) {
    const highLowDiff = highs[i] - lows[i];
    const mfMultiplier = highLowDiff !== 0
      ? ((closes[i] - lows[i]) - (highs[i] - closes[i])) / highLowDiff
      : 0;
    mfvSum += mfMultiplier * volumes[i];
    volumeSum += volumes[i];
  }

  return volumeSum !== 0 ? mfvSum / volumeSum : 0;
}

/**
 * Calculate Accumulation/Distribution Line
 */
export function calculateADLine(
  highs: number[],
  lows: number[],
  closes: number[],
  volumes: number[]
): number {
  let adLine = 0;

  for (let i = 0; i < closes.length; i++) {
    const highLowDiff = highs[i] - lows[i];
    const mfMultiplier = highLowDiff !== 0
      ? ((closes[i] - lows[i]) - (highs[i] - closes[i])) / highLowDiff
      : 0;
    adLine += mfMultiplier * volumes[i];
  }

  return adLine;
}

/**
 * Calculate Keltner Channels
 */
export function calculateKeltnerChannels(
  highs: number[],
  lows: number[],
  closes: number[],
  period: number = 20,
  multiplier: number = 2
): { upper: number; middle: number; lower: number } {
  const middle = calculateEMA(closes, period);
  const atr = calculateATR(highs, lows, closes, period);

  return {
    upper: middle + multiplier * atr,
    middle,
    lower: middle - multiplier * atr,
  };
}

/**
 * Calculate Donchian Channels
 */
export function calculateDonchianChannels(
  highs: number[],
  lows: number[],
  period: number = 20
): { high: number; low: number; mid: number } {
  const recentHighs = highs.slice(-period);
  const recentLows = lows.slice(-period);

  const high = Math.max(...recentHighs);
  const low = Math.min(...recentLows);

  return { high, low, mid: (high + low) / 2 };
}

/**
 * Calculate Pivot Points
 */
export function calculatePivotPoints(
  high: number,
  low: number,
  close: number
): { pivot: number; s1: number; r1: number; s2: number; r2: number } {
  const pivot = (high + low + close) / 3;
  const s1 = 2 * pivot - high;
  const r1 = 2 * pivot - low;
  const s2 = pivot - (high - low);
  const r2 = pivot + (high - low);

  return { pivot, s1, r1, s2, r2 };
}

/**
 * Count consecutive up/down days
 */
export function countConsecutiveDays(closes: number[]): { up: number; down: number } {
  let up = 0;
  let down = 0;

  for (let i = closes.length - 1; i > 0; i--) {
    if (closes[i] > closes[i - 1]) {
      if (down > 0) break;
      up++;
    } else if (closes[i] < closes[i - 1]) {
      if (up > 0) break;
      down++;
    } else {
      break;
    }
  }

  return { up, down };
}

/**
 * Calculate trend strength using multiple EMAs
 */
export function calculateTrendStrength(closes: number[]): {
  shortTerm: number;
  mediumTerm: number;
  longTerm: number;
  overall: number;
} {
  const ema9 = calculateEMA(closes, 9);
  const ema21 = calculateEMA(closes, 21);
  const ema50 = calculateEMA(closes, 50);
  const ema200 = closes.length >= 200 ? calculateEMA(closes, 200) : calculateEMA(closes, Math.min(closes.length, 50));

  const currentPrice = closes[closes.length - 1];

  // Trend signals (-1 to 1)
  const shortTerm = currentPrice > ema9 ? (ema9 > ema21 ? 1 : 0.5) : (ema9 < ema21 ? -1 : -0.5);
  const mediumTerm = ema21 > ema50 ? 1 : -1;
  const longTerm = ema50 > ema200 ? 1 : -1;

  // Overall trend alignment
  const overall = (shortTerm + mediumTerm + longTerm) / 3;

  return { shortTerm, mediumTerm, longTerm, overall };
}

/**
 * Calculate all technical indicators for a given OHLCV dataset
 */
export function calculateAllIndicators(
  data: OHLCV[],
  marketContext?: MarketContext
): TechnicalIndicators {
  const closes = data.map((d) => d.close);
  const highs = data.map((d) => d.high);
  const lows = data.map((d) => d.low);
  const volumes = data.map((d) => d.volume);
  const opens = data.map((d) => d.open);

  const currentPrice = closes[closes.length - 1];
  const currentVolume = volumes[volumes.length - 1];
  const previousClose = closes[closes.length - 2] || currentPrice;
  const currentHigh = highs[highs.length - 1];
  const currentLow = lows[lows.length - 1];
  const currentOpen = opens[opens.length - 1];

  // Trend Indicators
  const sma5 = calculateSMA(closes, 5);
  const sma10 = calculateSMA(closes, 10);
  const sma20 = calculateSMA(closes, 20);
  const sma50 = calculateSMA(closes, 50);
  const sma200 = closes.length >= 200 ? calculateSMA(closes, 200) : calculateSMA(closes, Math.min(closes.length, 50));
  const ema9 = calculateEMA(closes, 9);
  const ema12 = calculateEMA(closes, 12);
  const ema21 = calculateEMA(closes, 21);
  const ema26 = calculateEMA(closes, 26);

  // Momentum Indicators
  const rsi = calculateRSI(closes, 14);
  const rsiValues: number[] = [];
  for (let i = 20; i <= closes.length; i++) {
    rsiValues.push(calculateRSI(closes.slice(0, i), 14));
  }
  const rsi_sma = rsiValues.length > 0 ? calculateSMA(rsiValues, 5) : rsi;

  const macdData = calculateMACD(closes);
  const stochastic = calculateStochastic(highs, lows, closes, 14, 3);
  const williamR = calculateWilliamsR(highs, lows, closes, 14);
  const roc = calculateROC(closes, 10);
  const roc5 = calculateROC(closes, 5);
  const momentum = currentPrice - (closes[closes.length - 11] || currentPrice);
  const cci = calculateCCI(highs, lows, closes, 20);
  const mfi = calculateMFI(highs, lows, closes, volumes, 14);
  const adxData = calculateADX(highs, lows, closes, 14);

  // Volatility Indicators
  const bollinger = calculateBollingerBands(closes, 20, 2);
  const atr = calculateATR(highs, lows, closes, 14);
  const atrPercent = currentPrice > 0 ? (atr / currentPrice) * 100 : 0;
  const volatility = calculateVolatility(closes, 20);
  const keltner = calculateKeltnerChannels(highs, lows, closes, 20, 2);
  const donchian = calculateDonchianChannels(highs, lows, 20);

  // Volume Indicators
  const volumeSma = calculateSMA(volumes, 20);
  const volumeRatio = volumeSma === 0 ? 1 : currentVolume / volumeSma;
  const obv = calculateOBV(closes, volumes);
  const obvSlope = calculateOBVSlope(closes, volumes, 10);
  const vwap = calculateVWAP(highs, lows, closes, volumes);
  const cmf = calculateCMF(highs, lows, closes, volumes, 20);
  const adLine = calculateADLine(highs, lows, closes, volumes);

  // Price Patterns
  const priceChange = currentPrice - previousClose;
  const priceChangePercent = previousClose !== 0 ? (priceChange / previousClose) * 100 : 0;
  const highLowRange = currentHigh - currentLow;
  const bodySize = Math.abs(currentPrice - currentOpen);
  const upperShadow = currentHigh - Math.max(currentPrice, currentOpen);
  const lowerShadow = Math.min(currentPrice, currentOpen) - currentLow;

  // Gap detection
  const prevHigh = highs.length > 1 ? highs[highs.length - 2] : currentHigh;
  const prevLow = lows.length > 1 ? lows[lows.length - 2] : currentLow;
  const gapUp = currentLow > prevHigh ? (currentLow - prevHigh) / prevHigh * 100 : 0;
  const gapDown = currentHigh < prevLow ? (prevLow - currentHigh) / prevLow * 100 : 0;

  // Support/Resistance
  const high20 = Math.max(...highs.slice(-20));
  const low20 = Math.min(...lows.slice(-20));
  const high52 = Math.max(...highs.slice(-252));
  const low52 = Math.min(...lows.slice(-252));
  const distanceFromHigh20 = high20 > 0 ? (high20 - currentPrice) / high20 * 100 : 0;
  const distanceFromLow20 = low20 > 0 ? (currentPrice - low20) / low20 * 100 : 0;
  const distanceFromHigh52 = high52 > 0 ? (high52 - currentPrice) / high52 * 100 : 0;
  const distanceFromLow52 = low52 > 0 ? (currentPrice - low52) / low52 * 100 : 0;

  const pivots = calculatePivotPoints(
    highs[highs.length - 2] || currentHigh,
    lows[lows.length - 2] || currentLow,
    closes[closes.length - 2] || currentPrice
  );

  // Trend Strength
  const trendData = calculateTrendStrength(closes);
  const consecutiveDays = countConsecutiveDays(closes);
  const priceRange = high20 - low20;
  const pricePositionInRange = priceRange > 0 ? (currentPrice - low20) / priceRange : 0.5;

  // Calculate normalized values for ML input
  const priceRangeAll = Math.max(...closes) - Math.min(...closes);
  const minPrice = Math.min(...closes);
  const volumeRange = Math.max(...volumes) - Math.min(...volumes);
  const minVolume = Math.min(...volumes);

  const normalized: NormalizedFeatures = {
    // Price & Volume (0-1)
    price: priceRangeAll === 0 ? 0.5 : (currentPrice - minPrice) / priceRangeAll,
    volume: volumeRange === 0 ? 0.5 : (currentVolume - minVolume) / volumeRange,

    // Momentum
    rsi: rsi / 100,
    macd: Math.tanh(macdData.macd / (currentPrice * 0.01)), // Normalize relative to 1% of price
    stochastic: stochastic.k / 100,
    momentum: Math.tanh(momentum / (currentPrice * 0.05)), // Normalize relative to 5% of price
    cci: Math.tanh(cci / 200), // CCI typically ranges -200 to 200
    mfi: mfi / 100,

    // Volatility (0-1)
    bollinger: bollinger.percentB,
    atr: Math.min(atrPercent / 5, 1), // Cap at 5% ATR
    volatility: Math.min(volatility / 50, 1), // Cap at 50% annualized vol

    // Trend (-1 to 1)
    shortTrend: trendData.shortTerm,
    mediumTrend: trendData.mediumTerm,
    longTrend: trendData.longTerm,
    trendAlignment: trendData.overall,

    // Volume Flow (-1 to 1)
    volumeFlow: Math.tanh(obvSlope),
    moneyFlow: cmf, // Already -1 to 1

    // Support/Resistance (0-1)
    pricePosition: pricePositionInRange,
    distanceFromResistance: Math.min(distanceFromHigh20 / 20, 1),
    distanceFromSupport: Math.min(distanceFromLow20 / 20, 1),

    // Pattern Recognition
    candlePattern: highLowRange > 0 ? (bodySize / highLowRange) * (currentPrice > currentOpen ? 1 : -1) : 0,
    gapStrength: Math.tanh((gapUp - gapDown) / 2),

    // Market Context (-1 to 1)
    marketSentiment: marketContext?.marketTrend || 0,
    sectorStrength: marketContext?.sectorPerformance || 0,
    breadth: marketContext?.breadthRatio ? Math.tanh(marketContext.breadthRatio - 1) : 0,
  };

  return {
    sma5,
    sma10,
    sma20,
    sma50,
    sma200,
    ema9,
    ema12,
    ema21,
    ema26,
    rsi,
    rsi_sma,
    macd: macdData.macd,
    macdSignal: macdData.signal,
    macdHistogram: macdData.histogram,
    stochK: stochastic.k,
    stochD: stochastic.d,
    williamR,
    roc,
    roc5,
    momentum,
    cci,
    mfi,
    adx: adxData.adx,
    plusDI: adxData.plusDI,
    minusDI: adxData.minusDI,
    bollingerUpper: bollinger.upper,
    bollingerMiddle: bollinger.middle,
    bollingerLower: bollinger.lower,
    bollingerWidth: bollinger.width,
    bollingerPercentB: bollinger.percentB,
    atr,
    atrPercent,
    volatility,
    keltnerUpper: keltner.upper,
    keltnerLower: keltner.lower,
    donchianHigh: donchian.high,
    donchianLow: donchian.low,
    volumeSma,
    volumeRatio,
    obv,
    obvSlope,
    vwap,
    cmf,
    adLine,
    priceChange,
    priceChangePercent,
    highLowRange,
    bodySize,
    upperShadow,
    lowerShadow,
    gapUp,
    gapDown,
    distanceFromHigh20,
    distanceFromLow20,
    distanceFromHigh52,
    distanceFromLow52,
    pivotPoint: pivots.pivot,
    support1: pivots.s1,
    resistance1: pivots.r1,
    trendStrength: trendData.overall,
    pricePositionInRange,
    consecutiveUpDays: consecutiveDays.up,
    consecutiveDownDays: consecutiveDays.down,
    shortTermTrend: trendData.shortTerm,
    mediumTermTrend: trendData.mediumTerm,
    longTermTrend: trendData.longTerm,
    marketContext,
    normalized,
  };
}

/**
 * Convert indicators to feature array for ML model - ENHANCED VERSION
 * Now outputs 30 features instead of 15
 */
export function indicatorsToFeatures(indicators: TechnicalIndicators): number[] {
  const n = indicators.normalized;

  return [
    // === PRICE & POSITION (5 features) ===
    n.price,                    // 0: Normalized price position in range
    n.pricePosition,            // 1: Price position in 20-day range
    n.distanceFromResistance,   // 2: Distance from 20-day high
    n.distanceFromSupport,      // 3: Distance from 20-day low
    n.bollinger,                // 4: Bollinger %B (position in bands)

    // === MOMENTUM (7 features) ===
    n.rsi,                      // 5: RSI normalized
    n.stochastic,               // 6: Stochastic %K
    n.macd,                     // 7: MACD normalized
    n.momentum,                 // 8: Price momentum
    n.cci,                      // 9: CCI normalized
    n.mfi,                      // 10: Money Flow Index
    Math.tanh(indicators.roc / 20), // 11: Rate of change

    // === TREND (5 features) ===
    n.shortTrend,               // 12: Short-term trend direction
    n.mediumTrend,              // 13: Medium-term trend direction
    n.longTrend,                // 14: Long-term trend direction
    n.trendAlignment,           // 15: Overall trend alignment
    Math.tanh((indicators.adx - 25) / 25), // 16: ADX trend strength

    // === VOLATILITY (4 features) ===
    n.atr,                      // 17: ATR as % of price
    n.volatility,               // 18: Historical volatility
    indicators.bollingerWidth,  // 19: Bollinger bandwidth
    Math.tanh(indicators.atrPercent - 2), // 20: ATR deviation from norm

    // === VOLUME (4 features) ===
    n.volume,                   // 21: Normalized volume
    Math.min(indicators.volumeRatio, 3) / 3, // 22: Volume ratio (capped)
    n.volumeFlow,               // 23: OBV trend
    n.moneyFlow,                // 24: Chaikin Money Flow

    // === PATTERN & STRUCTURE (3 features) ===
    n.candlePattern,            // 25: Candle body analysis
    n.gapStrength,              // 26: Gap up/down strength
    Math.tanh((indicators.consecutiveUpDays - indicators.consecutiveDownDays) / 5), // 27: Consecutive days

    // === MARKET CONTEXT (2 features) ===
    n.marketSentiment,          // 28: Overall market trend
    n.sectorStrength,           // 29: Sector relative strength
  ];
}

/**
 * Generate feature labels for debugging/analysis
 */
export function getFeatureLabels(): string[] {
  return [
    // Price & Position
    'price_normalized',
    'price_position_20d',
    'distance_from_resistance',
    'distance_from_support',
    'bollinger_percent_b',

    // Momentum
    'rsi',
    'stochastic_k',
    'macd',
    'momentum',
    'cci',
    'mfi',
    'roc',

    // Trend
    'trend_short',
    'trend_medium',
    'trend_long',
    'trend_alignment',
    'adx_strength',

    // Volatility
    'atr_percent',
    'volatility',
    'bollinger_width',
    'atr_deviation',

    // Volume
    'volume_normalized',
    'volume_ratio',
    'obv_slope',
    'chaikin_mf',

    // Pattern
    'candle_pattern',
    'gap_strength',
    'consecutive_days',

    // Market Context
    'market_sentiment',
    'sector_strength',
  ];
}

/**
 * Get the number of features (for model configuration)
 */
export function getFeatureCount(): number {
  return 30;
}

/**
 * Get the number of fast features (for light model)
 */
export function getFastFeatureCount(): number {
  return 15;
}

// ============ FAST FEATURE EXTRACTION ============

/**
 * Convert indicators to FAST feature array for light ML model
 * Selects 15 most predictive features for speed
 *
 * Feature categories:
 * - Momentum (4): RSI, MACD, Stochastic, ROC
 * - Trend (4): Short/medium/long trend, alignment
 * - Volatility (2): ATR, Bollinger width
 * - Volume (2): Volume ratio, OBV slope
 * - Price (2): Price position, Bollinger %B
 * - Market (1): Market sentiment
 */
export function indicatorsToFastFeatures(indicators: TechnicalIndicators): number[] {
  const n = indicators.normalized;

  return [
    // === MOMENTUM (4 features) ===
    n.rsi,                              // 0: RSI normalized (most predictive momentum)
    n.macd,                             // 1: MACD normalized
    n.stochastic,                       // 2: Stochastic %K
    Math.tanh(indicators.roc / 20),     // 3: Rate of change

    // === TREND (4 features) ===
    n.shortTrend,                       // 4: Short-term trend
    n.mediumTrend,                      // 5: Medium-term trend
    n.longTrend,                        // 6: Long-term trend
    n.trendAlignment,                   // 7: Trend alignment

    // === VOLATILITY (2 features) ===
    n.atr,                              // 8: ATR as % of price
    indicators.bollingerWidth,          // 9: Bollinger bandwidth

    // === VOLUME (2 features) ===
    Math.min(indicators.volumeRatio, 3) / 3, // 10: Volume ratio (capped at 3)
    n.volumeFlow,                       // 11: OBV slope

    // === PRICE POSITION (2 features) ===
    n.pricePosition,                    // 12: Price position in range
    n.bollinger,                        // 13: Bollinger %B

    // === MARKET CONTEXT (1 feature) ===
    n.marketSentiment,                  // 14: Market sentiment
  ];
}

/**
 * Get fast feature labels for debugging
 */
export function getFastFeatureLabels(): string[] {
  return [
    'rsi', 'macd', 'stochastic', 'roc',
    'trend_short', 'trend_medium', 'trend_long', 'trend_align',
    'atr', 'bb_width',
    'volume_ratio', 'obv_slope',
    'price_position', 'bb_percent',
    'market_sentiment',
  ];
}

// ============ MARKET REGIME DETECTION ============

export type MarketRegimeType = 'trending_up' | 'trending_down' | 'ranging' | 'high_volatility';

export interface MarketRegime {
  regime: MarketRegimeType;
  regimeStrength: number;        // 0-1 how confident we are in the regime
  regimeDuration: number;        // Estimated days in current regime
  volatilityCluster: number;     // Current vs historical volatility
  hurstExponent: number;         // Trend persistence (>0.5 = trending, <0.5 = mean reverting)
  features: {
    regimeEncoded: number;       // -1 to 1 (bearish trend to bullish trend)
    regimeStrengthNorm: number;  // 0-1
    volatilityClusterNorm: number; // 0-1
    hurstNorm: number;           // 0-1 (0.5 = random walk)
  };
}

/**
 * Detect the current market regime
 * Uses ADX for trend strength, Bollinger width for volatility,
 * and price position relative to MAs for direction
 */
export function detectMarketRegime(
  closes: number[],
  highs: number[],
  lows: number[],
  period: number = 20
): MarketRegime {
  if (closes.length < period) {
    return getDefaultRegime();
  }

  // Calculate components
  const adxData = calculateADX(highs, lows, closes, 14);
  const volatility = calculateVolatility(closes, period);
  const avgVolatility = calculateAverageVolatility(closes, 60); // 60-day average
  const trendData = calculateTrendStrength(closes);
  const hurst = calculateSimpleHurst(closes, Math.min(100, closes.length));

  // Volatility clustering
  const volatilityCluster = avgVolatility > 0 ? volatility / avgVolatility : 1;

  // Determine regime based on ADX and trend direction
  let regime: MarketRegimeType;
  let regimeStrength: number;

  if (volatilityCluster > 1.5) {
    // High volatility regime
    regime = 'high_volatility';
    regimeStrength = Math.min(1, (volatilityCluster - 1) / 1.5);
  } else if (adxData.adx > 25) {
    // Trending regime
    if (trendData.overall > 0.3) {
      regime = 'trending_up';
      regimeStrength = Math.min(1, adxData.adx / 50) * Math.min(1, trendData.overall + 0.5);
    } else if (trendData.overall < -0.3) {
      regime = 'trending_down';
      regimeStrength = Math.min(1, adxData.adx / 50) * Math.min(1, Math.abs(trendData.overall) + 0.5);
    } else {
      regime = 'ranging';
      regimeStrength = 0.5;
    }
  } else {
    // Ranging/consolidation regime
    regime = 'ranging';
    regimeStrength = Math.min(1, (30 - adxData.adx) / 20);
  }

  // Estimate regime duration (simplified)
  const regimeDuration = estimateRegimeDuration(closes, regime, period);

  // Create normalized features for ML
  const regimeEncoded =
    regime === 'trending_up' ? 1 :
    regime === 'trending_down' ? -1 :
    regime === 'high_volatility' ? 0 : 0;

  return {
    regime,
    regimeStrength,
    regimeDuration,
    volatilityCluster,
    hurstExponent: hurst,
    features: {
      regimeEncoded: regimeEncoded * regimeStrength,
      regimeStrengthNorm: regimeStrength,
      volatilityClusterNorm: Math.min(1, volatilityCluster / 2),
      hurstNorm: hurst,
    },
  };
}

/**
 * Calculate average volatility over a longer period
 */
function calculateAverageVolatility(closes: number[], period: number): number {
  if (closes.length < period) {
    return calculateVolatility(closes, closes.length);
  }

  const volatilities: number[] = [];
  for (let i = 20; i <= period; i += 10) {
    volatilities.push(calculateVolatility(closes.slice(0, -period + i), 20));
  }

  return volatilities.length > 0
    ? volatilities.reduce((a, b) => a + b, 0) / volatilities.length
    : calculateVolatility(closes, 20);
}

/**
 * Simplified Hurst exponent calculation
 * H > 0.5: trending (persistent)
 * H = 0.5: random walk
 * H < 0.5: mean reverting
 */
function calculateSimpleHurst(closes: number[], period: number): number {
  if (closes.length < period || period < 20) {
    return 0.5; // Default to random walk
  }

  const returns: number[] = [];
  for (let i = 1; i < closes.length; i++) {
    if (closes[i - 1] > 0) {
      returns.push(Math.log(closes[i] / closes[i - 1]));
    }
  }

  if (returns.length < 20) return 0.5;

  // Rescaled range analysis (simplified)
  const recentReturns = returns.slice(-period);
  const mean = recentReturns.reduce((a, b) => a + b, 0) / recentReturns.length;

  // Cumulative deviation
  let cumDev = 0;
  let maxDev = -Infinity;
  let minDev = Infinity;

  for (const r of recentReturns) {
    cumDev += r - mean;
    maxDev = Math.max(maxDev, cumDev);
    minDev = Math.min(minDev, cumDev);
  }

  const range = maxDev - minDev;
  const std = Math.sqrt(recentReturns.reduce((sum, r) => sum + Math.pow(r - mean, 2), 0) / recentReturns.length);

  if (std === 0) return 0.5;

  // Simplified Hurst calculation
  const rs = range / std;
  const n = recentReturns.length;

  // H = log(R/S) / log(n) - simplified approximation
  const hurst = Math.log(rs) / Math.log(n);

  // Clamp to reasonable range
  return Math.max(0.2, Math.min(0.8, hurst));
}

/**
 * Estimate how long the current regime has been in effect
 */
function estimateRegimeDuration(
  closes: number[],
  currentRegime: MarketRegimeType,
  period: number
): number {
  if (closes.length < period * 2) return period;

  // Look back and count days with similar characteristics
  let duration = 0;
  const recentTrend = calculateTrendStrength(closes);

  for (let i = closes.length - 1; i >= period && i > closes.length - 60; i--) {
    const slice = closes.slice(0, i);
    const trendData = calculateTrendStrength(slice);

    // Check if trend characteristics match current regime
    const matchesCurrent =
      (currentRegime === 'trending_up' && trendData.overall > 0.2) ||
      (currentRegime === 'trending_down' && trendData.overall < -0.2) ||
      (currentRegime === 'ranging' && Math.abs(trendData.overall) < 0.3);

    if (matchesCurrent) {
      duration++;
    } else {
      break;
    }
  }

  return Math.max(1, duration);
}

/**
 * Get default regime when data is insufficient
 */
function getDefaultRegime(): MarketRegime {
  return {
    regime: 'ranging',
    regimeStrength: 0.5,
    regimeDuration: 0,
    volatilityCluster: 1,
    hurstExponent: 0.5,
    features: {
      regimeEncoded: 0,
      regimeStrengthNorm: 0.5,
      volatilityClusterNorm: 0.5,
      hurstNorm: 0.5,
    },
  };
}

/**
 * Enhanced feature extraction including regime features
 * Returns 19 features (15 fast + 4 regime)
 */
export function indicatorsToEnhancedFeatures(
  indicators: TechnicalIndicators,
  regime: MarketRegime
): number[] {
  const fastFeatures = indicatorsToFastFeatures(indicators);

  return [
    ...fastFeatures,
    regime.features.regimeEncoded,       // 15: Regime direction and strength
    regime.features.regimeStrengthNorm,  // 16: Regime confidence
    regime.features.volatilityClusterNorm, // 17: Volatility clustering
    regime.features.hurstNorm,           // 18: Trend persistence
  ];
}
