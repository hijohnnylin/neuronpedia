import { env } from '@/lib/env';
import { NextResponse } from 'next/server';

export async function POST() {
  if (!env.IS_LOCALHOST) {
    return NextResponse.json({ error: 'This route is only available on localhost or to admin users' }, { status: 400 });
  }

  return NextResponse.json({ message: 'Not enabled' });

  // try {
  //   // Get all table names except 'User'
  //   const tables = await prisma.$queryRaw<{ tablename: string }[]>`
  //     SELECT tablename
  //     FROM pg_tables
  //     WHERE schemaname = 'public'
  //     AND tablename != 'User'
  //     AND tablename != '_prisma_migrations'
  //   `;

  //   // Truncate each table
  //   for (const { tablename } of tables) {
  //     console.log(`Truncating table: ${tablename}`);
  //     await prisma.$executeRawUnsafe(`TRUNCATE TABLE "${tablename}" CASCADE`);
  //   }

  //   return NextResponse.json({ message: 'Database reset successful' });
  // } catch (error) {
  //   console.error('Error resetting database:', error);
  //   return new NextResponse('Internal Server Error', { status: 500 });
  // }
}
