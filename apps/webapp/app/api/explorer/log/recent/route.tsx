import { getRecentProblemCreations } from '@/lib/db/problem';
import { NextResponse } from 'next/server';

export const POST = async () => {
  const logs = await getRecentProblemCreations(8);
  return NextResponse.json(logs);
};
