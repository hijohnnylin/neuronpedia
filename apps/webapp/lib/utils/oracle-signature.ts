// Signatures on oracle reads. A read is not repeatable (bf16 noise can flip a near
// tie), so a share stores the text the viewer saw instead of reading again. This
// server signs each read it serves; the share route stores only reads it signed.

import type { JlensExportOracle, JlensExportOracleRead } from '@/components/jlens/jlens-export';
import { createHash, createHmac, timingSafeEqual } from 'crypto';

export type SignedOracleReadFields = {
  modelId: string;
  adapter: string;
  maxBullets: number;
  position: number;
  // `oraclePrefixDigest` of the token ids 0..position.
  prefixDigest: string;
  layer: number;
  bullets: string[];
  text: string;
  finish: string;
};

const SIGNATURE_VERSION = 'v1';

function signingKey(): Buffer | null {
  const secret = process.env.NEXTAUTH_SECRET || '';
  if (!secret) {
    return null;
  }
  return createHash('sha256').update(`neuronpedia oracle read ${SIGNATURE_VERSION}\0${secret}`).digest();
}

export function oraclePrefixDigest(prefix: number[]): string {
  return createHash('sha256').update(prefix.join(',')).digest('hex');
}

function payload(f: SignedOracleReadFields): string {
  return JSON.stringify([
    SIGNATURE_VERSION,
    f.modelId,
    f.adapter,
    f.maxBullets,
    f.position,
    f.prefixDigest,
    f.layer,
    f.bullets,
    f.text,
    f.finish,
  ]);
}

export function oracleSigningEnabled(): boolean {
  return signingKey() !== null;
}

// Null when the server has no secret to sign with.
export function signOracleRead(fields: SignedOracleReadFields): string | null {
  const key = signingKey();
  if (!key) {
    return null;
  }
  return `${SIGNATURE_VERSION}.${createHmac('sha256', key).update(payload(fields)).digest('base64url')}`;
}

export function verifyOracleRead(fields: SignedOracleReadFields, sig: string): boolean {
  const expected = signOracleRead(fields);
  if (!expected || typeof sig !== 'string') {
    return false;
  }
  const a = Buffer.from(expected);
  const b = Buffer.from(sig);
  return a.length === b.length && timingSafeEqual(a, b);
}

// The reads a share stores, ordered by position and layer. A read outside
// `tokenIds` or with no valid signature is dropped, so the share still saves.
// Undefined when no read is left.
export function verifySharedOracleReads(
  modelId: string,
  tokenIds: number[],
  maxBullets: number,
  reads: JlensExportOracleRead[],
): JlensExportOracle | undefined {
  if (!oracleSigningEnabled()) {
    return undefined;
  }
  const digests = new Map<number, string>();
  const seen = new Set<string>();
  const stored: JlensExportOracle['reads'] = [];
  for (const r of reads) {
    const read = {
      position: r.position,
      layer: r.layer,
      adapter: r.adapter,
      bullets: r.bullets,
      text: r.text,
      finish: r.finish,
      sig: r.sig,
    };
    const id = `${read.position}:${read.layer}`;
    if (read.position >= tokenIds.length || seen.has(id)) {
      continue;
    }
    let prefixDigest = digests.get(read.position);
    if (prefixDigest === undefined) {
      prefixDigest = oraclePrefixDigest(tokenIds.slice(0, read.position + 1));
      digests.set(read.position, prefixDigest);
    }
    if (!verifyOracleRead({ modelId, maxBullets, prefixDigest, ...read }, read.sig)) {
      continue;
    }
    seen.add(id);
    stored.push(read);
  }
  if (stored.length === 0) {
    return undefined;
  }
  stored.sort((a, b) => a.position - b.position || a.layer - b.layer);
  return { maxBullets, reads: stored };
}
