import { createHash } from 'crypto';

/** SHA-256 hex over UTF-8 bytes; must match Postgres backfill using digest(convert_to("text", 'UTF8'), 'sha256'). */
export function nlaExplainTextHash(text: string): string {
  return createHash('sha256').update(text, 'utf8').digest('hex');
}
