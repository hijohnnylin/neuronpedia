/* eslint-disable no-restricted-syntax */
/* eslint-disable no-await-in-loop */

import { prisma } from '@/lib/db';
import { CONFIG_BASE_PATH, downloadFileJsonlParsedLines } from '@/lib/utils/s3';
import { Pool } from 'pg';

// Reduced from 2GB to 256MB to fit within Kubernetes PostgreSQL memory limits (512Mi)
// If PostgreSQL has more memory, this can be increased for better performance
const WORK_MEM = '256MB';

// Singleton pool for import operations to avoid connection churn
let importPool: Pool | null = null;
let useSSL: boolean | null = null; // null = not determined yet, true/false = determined

function getImportPool(): Pool {
  if (!importPool) {
    // For localhost/Docker Compose with internal Postgres, SSL is typically not needed
    // For cloud/managed Postgres, SSL is usually required
    // Default to no SSL for internal Kubernetes postgres
    const shouldTrySSL = useSSL ?? false;

    importPool = new Pool({
      connectionString: process.env.POSTGRES_URL_NON_POOLING || '',
      ssl: shouldTrySSL ? { rejectUnauthorized: false } : false,
      max: 5, // Limit concurrent connections
      idleTimeoutMillis: 30000, // Close idle connections after 30s
      connectionTimeoutMillis: 10000, // Timeout after 10s if can't connect
      // statement_timeout is set per-query instead
    });

    console.log(`Created import pool with SSL=${shouldTrySSL}`);
  }
  return importPool;
}

// Helper to reset pool if SSL settings need to change
function resetImportPool() {
  if (importPool) {
    importPool.end();
    importPool = null;
  }
}

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
  let client;
  // replace all \u0000 with ' ' because it's not supported by postgres
  // eslint-disable-next-line no-param-reassign
  jsonlData = jsonlData.replaceAll('\\u0000', ' ');
  try {
    const pool = getImportPool();
    console.log(`Connecting to database for table ${tableName}...`);

    try {
      client = await pool.connect();
      console.log(`Connected successfully. Setting work_mem=${WORK_MEM} and statement_timeout=600s`);
    } catch (sslError: any) {
      // Handle SSL connection errors specifically
      if (sslError?.message?.includes('SSL') || sslError?.message?.includes('ssl')) {
        console.log(`SSL connection failed: ${sslError.message}. Retrying without SSL...`);
        resetImportPool();
        useSSL = false;
        const newPool = getImportPool();
        client = await newPool.connect();
        console.log(`Connected successfully without SSL. Setting work_mem=${WORK_MEM} and statement_timeout=600s`);
      } else {
        throw sslError;
      }
    }

    // Set work_mem and statement_timeout for this connection
    // statement_timeout prevents queries from running indefinitely
    await client.query(`SET work_mem = '${WORK_MEM}'`);
    await client.query(`SET statement_timeout = '600000'`); // 10 minutes per statement
    console.log(`Database configuration applied for ${tableName}`);

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
    const totalChunks = Math.ceil(lines.length / chunkSize);
    console.log(`Importing ${lines.length} lines into ${tableName} in ${totalChunks} chunks`);

    for (let i = 0; i < lines.length; i += chunkSize) {
      const chunk = lines.slice(i, i + chunkSize);
      const jsonArray = `[${chunk.join(',')}]`;
      const chunkNum = Math.floor(i / chunkSize) + 1;
      console.log(`Executing insert for ${tableName} chunk ${chunkNum}/${totalChunks} (${chunk.length} rows)`);
      await client.query(query, [jsonArray]);
      console.log(`Completed ${tableName} chunk ${chunkNum}/${totalChunks}`);
    }
    console.log(`Successfully imported all data into ${tableName}`);
  } catch (err) {
    console.error(`Error importing data into ${tableName}:`, err);
    if (err instanceof Error) {
      console.error('Error details:', { message: err.message, stack: err.stack, name: err.name });
    }
    throw err;
  } finally {
    if (client) {
      console.log(`Releasing database connection for ${tableName}`);
      client.release();
    }
    // Don't end the pool here - it's a singleton that will be reused
  }
}
