import { fetchDecoderLatent } from '@/lib/utils/saelens';
import { NextRequest, NextResponse } from 'next/server';

export async function GET(request: NextRequest) {
  const { searchParams } = request.nextUrl;
  const repo = searchParams.get('repo');
  const path = searchParams.get('path');
  const indexStr = searchParams.get('index');

  if (!repo || !path || !indexStr) {
    return NextResponse.json({ error: 'Missing required query parameters: repo, path, index' }, { status: 400 });
  }

  const index = parseInt(indexStr, 10);
  if (Number.isNaN(index) || index < 0) {
    return NextResponse.json({ error: 'index must be a non-negative integer' }, { status: 400 });
  }

  try {
    const result = await fetchDecoderLatent(repo, path, index);
    return NextResponse.json(result);
  } catch (e: unknown) {
    const message = e instanceof Error ? e.message : String(e);
    return NextResponse.json({ error: message }, { status: 400 });
  }
}
