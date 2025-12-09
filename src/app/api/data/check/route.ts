import { NextResponse } from 'next/server';
import { existsSync, readdirSync, statSync, readFileSync } from 'fs';
import { join } from 'path';

interface GapInfo {
  start: string;
  end: string;
  minutes: number;
}

interface DataCheckResult {
  asset: string;
  category: string;
  totalCandles: number;
  dateRange: { start: string; end: string } | null;
  gaps: GapInfo[];
  totalGapMinutes: number;
  completeness: number;
  duplicates: number;
  issues: string[];
}

function parseJsonlFile(filePath: string): Array<{ timestamp: number }> {
  const content = readFileSync(filePath, 'utf-8');
  const lines = content.trim().split('\n').filter(l => l.trim());
  return lines.map(line => {
    try {
      return JSON.parse(line);
    } catch {
      return null;
    }
  }).filter(Boolean) as Array<{ timestamp: number }>;
}

function checkAssetData(assetDir: string, assetName: string, category: string): DataCheckResult {
  const issues: string[] = [];
  const gaps: GapInfo[] = [];
  let totalCandles = 0;
  let duplicates = 0;
  let allTimestamps: number[] = [];

  // Read all week files
  const files = existsSync(assetDir)
    ? readdirSync(assetDir).filter(f => f.endsWith('.jsonl') && f.startsWith('week_'))
    : [];

  if (files.length === 0) {
    return {
      asset: assetName,
      category,
      totalCandles: 0,
      dateRange: null,
      gaps: [],
      totalGapMinutes: 0,
      completeness: 0,
      duplicates: 0,
      issues: ['No data files found']
    };
  }

  // Read all candles
  for (const file of files) {
    const filePath = join(assetDir, file);
    try {
      const candles = parseJsonlFile(filePath);
      totalCandles += candles.length;
      allTimestamps.push(...candles.map(c => c.timestamp));
    } catch (err) {
      issues.push(`Error reading ${file}: ${err instanceof Error ? err.message : 'Unknown error'}`);
    }
  }

  if (allTimestamps.length === 0) {
    return {
      asset: assetName,
      category,
      totalCandles: 0,
      dateRange: null,
      gaps: [],
      totalGapMinutes: 0,
      completeness: 0,
      duplicates: 0,
      issues: ['No valid candles found']
    };
  }

  // Sort timestamps
  allTimestamps.sort((a, b) => a - b);

  // Check for duplicates
  const uniqueTimestamps = [...new Set(allTimestamps)];
  duplicates = allTimestamps.length - uniqueTimestamps.length;
  if (duplicates > 0) {
    issues.push(`Found ${duplicates} duplicate candles`);
  }

  // Check for gaps (more than 2 minutes between consecutive candles)
  const MINUTE_MS = 60 * 1000;
  const GAP_THRESHOLD_MS = 2 * MINUTE_MS; // 2 minutes

  let totalGapMinutes = 0;
  for (let i = 1; i < uniqueTimestamps.length; i++) {
    const diff = uniqueTimestamps[i] - uniqueTimestamps[i - 1];
    if (diff > GAP_THRESHOLD_MS) {
      const gapMinutes = Math.floor(diff / MINUTE_MS);
      totalGapMinutes += gapMinutes - 1; // -1 because 1 minute gap is normal

      // Only record significant gaps (> 5 minutes)
      if (gapMinutes > 5) {
        gaps.push({
          start: new Date(uniqueTimestamps[i - 1]).toISOString(),
          end: new Date(uniqueTimestamps[i]).toISOString(),
          minutes: gapMinutes
        });
      }
    }
  }

  // Calculate completeness
  const startTime = uniqueTimestamps[0];
  const endTime = uniqueTimestamps[uniqueTimestamps.length - 1];
  const expectedCandles = Math.floor((endTime - startTime) / MINUTE_MS) + 1;
  const completeness = Math.min(100, Math.round((uniqueTimestamps.length / expectedCandles) * 100));

  if (completeness < 90) {
    issues.push(`Data completeness is only ${completeness}%`);
  }

  if (gaps.length > 10) {
    issues.push(`Found ${gaps.length} significant gaps in data`);
  }

  return {
    asset: assetName,
    category,
    totalCandles: uniqueTimestamps.length,
    dateRange: {
      start: new Date(startTime).toISOString(),
      end: new Date(endTime).toISOString()
    },
    gaps: gaps.slice(0, 20), // Limit to 20 largest gaps
    totalGapMinutes,
    completeness,
    duplicates,
    issues
  };
}

export async function GET() {
  try {
    const dataDir = join(process.cwd(), 'btc_predictor', 'data');
    const results: DataCheckResult[] = [];

    // Check crypto data
    const cryptoDir = join(dataDir, 'crypto');
    if (existsSync(cryptoDir)) {
      const cryptoAssets = readdirSync(cryptoDir).filter(f =>
        statSync(join(cryptoDir, f)).isDirectory()
      );

      for (const asset of cryptoAssets) {
        const result = checkAssetData(
          join(cryptoDir, asset),
          asset,
          'Crypto'
        );
        results.push(result);
      }
    }

    // Check stocks data
    const stocksDir = join(dataDir, 'stocks');
    if (existsSync(stocksDir)) {
      const stockAssets = readdirSync(stocksDir).filter(f =>
        statSync(join(stocksDir, f)).isDirectory()
      );

      for (const asset of stockAssets.slice(0, 10)) { // Limit to 10 for speed
        const result = checkAssetData(
          join(stocksDir, asset),
          asset,
          'Stocks'
        );
        results.push(result);
      }
    }

    // Summary
    const totalIssues = results.reduce((sum, r) => sum + r.issues.length, 0);
    const totalGaps = results.reduce((sum, r) => sum + r.gaps.length, 0);
    const avgCompleteness = results.length > 0
      ? Math.round(results.reduce((sum, r) => sum + r.completeness, 0) / results.length)
      : 0;

    return NextResponse.json({
      timestamp: new Date().toISOString(),
      summary: {
        assetsChecked: results.length,
        totalIssues,
        totalGaps,
        avgCompleteness
      },
      results
    });
  } catch (error) {
    console.error('Data check error:', error);
    return NextResponse.json(
      { error: 'Failed to check data', details: error instanceof Error ? error.message : 'Unknown error' },
      { status: 500 }
    );
  }
}
