import { NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';

export async function GET() {
  try {
    const progressFile = path.join(process.cwd(), 'btc_predictor', 'backfill_progress.json');

    if (!fs.existsSync(progressFile)) {
      return NextResponse.json({
        stocks: { running: false, completed: false },
        crypto: { running: false, completed: false }
      });
    }

    const content = fs.readFileSync(progressFile, 'utf-8');
    const progress = JSON.parse(content);

    return NextResponse.json(progress);
  } catch (error) {
    console.error('Error reading progress:', error);
    return NextResponse.json({
      stocks: { running: false, completed: false },
      crypto: { running: false, completed: false }
    });
  }
}
