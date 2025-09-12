/* eslint-disable no-restricted-syntax */
/* eslint-disable no-await-in-loop */

import { prisma } from '@/lib/db';
import { CONFIG_BASE_PATH, downloadFileJsonlParsedLines } from '@/lib/utils/s3';
import { Pool } from 'pg';

const WORK_MEM = '2GB';

export async function importConfigFromS3() {
  const explanationModelTypeLines = await downloadFileJsonlParsedLines(
    `${CONFIG_BASE_PATH}explanation_model_type.jsonl`,
  );
  for (const line of explanationModelTypeLines) {
    await prisma.explanationModelType.upsert({
      where: { name: line.name },
      update: line,
      create: line,
    });
  }

  const explanationTypeLines = await downloadFileJsonlParsedLines(`${CONFIG_BASE_PATH}explanation_type.jsonl`);
  for (const line of explanationTypeLines) {
    await prisma.explanationType.upsert({
      where: { name: line.name },
      update: line,
      create: line,
    });
  }

  const explanationScoreModelLines = await downloadFileJsonlParsedLines(
    `${CONFIG_BASE_PATH}explanation_score_model.jsonl`,
  );
  for (const line of explanationScoreModelLines) {
    await prisma.explanationScoreModel.upsert({
      where: { name: line.name },
      update: line,
      create: line,
    });
  }

  const explanationScoreTypeLines = await downloadFileJsonlParsedLines(
    `${CONFIG_BASE_PATH}explanation_score_type.jsonl`,
  );
  for (const line of explanationScoreTypeLines) {
    await prisma.explanationScoreType.upsert({
      where: { name: line.name },
      update: line,
      create: line,
    });
  }

  const evalTypeLines = await downloadFileJsonlParsedLines(`${CONFIG_BASE_PATH}eval_type.jsonl`);
  for (const line of evalTypeLines) {
    await prisma.evalType.upsert({
      where: { name: line.name },
      update: line,
      create: line,
    });
  }
}

export async function importJsonlString(tableName: string, jsonlData: string) {
  let pool;
  let client;
  // replace all \u0000 with ' ' because it's not supported by postgres
  // eslint-disable-next-line no-param-reassign
  jsonlData = jsonlData.replaceAll('\\u0000', ' ');
  try {
    try {
      pool = new Pool({
        connectionString: process.env.POSTGRES_URL_NON_POOLING || '',
        ssl: { rejectUnauthorized: false },
      });
      client = await pool.connect();
      // Set work_mem for this connection
      await client.query(`SET work_mem = '${WORK_MEM}'`);
    } catch (error) {
      pool = new Pool({
        connectionString: process.env.POSTGRES_URL_NON_POOLING || '',
        ssl: false,
      });
      client = await pool.connect();
      // Set work_mem for this connection
      await client.query(`SET work_mem = '${WORK_MEM}'`);
    }

    // Parse first line to get available columns
    const firstLine = jsonlData.trim().split('\n')[0];
    const availableColumns = Object.keys(JSON.parse(firstLine));
    // Get column information only for columns that exist in the JSON
    const columnQuery = `
      SELECT 
        column_name,
        CASE 
          WHEN data_type = 'ARRAY' THEN 
            udt_name::regtype::text || '[]'
          WHEN data_type = 'USER-DEFINED' THEN (
            SELECT t.typname::text
            FROM pg_type t
            WHERE t.typname = c.udt_name
          )
          ELSE 
            data_type
        END as data_type
      FROM information_schema.columns c
      WHERE table_name = $1
        AND column_name = ANY($2)
      ORDER BY ordinal_position;
    `;
    const { rows: columns } = await client.query(columnQuery, [tableName, availableColumns]);

    // Build the column definition list
    const columnDefs = columns.map((col) => `"${col.column_name}" ${col.data_type}`).join(', ');
    const columnList = columns.map((col) => `"${col.column_name}"`).join(', ');

    const query = `
      INSERT INTO "${tableName}" (${columnList})
      SELECT ${columnList} FROM jsonb_to_recordset($1::jsonb) as t(${columnDefs})
      ON CONFLICT DO NOTHING
    `;

    const lines = jsonlData.trim().split('\n');
    const chunkSize = 65000; // there's a limit of ~200MB per insert

    for (let i = 0; i < lines.length; i += chunkSize) {
      const chunk = lines.slice(i, i + chunkSize);
      const jsonArray = `[${chunk.join(',')}]`;
      await client.query(query, [jsonArray]);
    }
  } catch (err) {
    console.error('Error importing data:', err);
    throw err;
  } finally {
    if (client) {
      client.release();
    }
  }
}
