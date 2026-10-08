import type { JlensExportOracleRead } from '@/components/jlens/jlens-export';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import {
  oraclePrefixDigest,
  SignedOracleReadFields,
  signOracleRead,
  verifyOracleRead,
  verifySharedOracleReads,
} from './oracle-signature';

const TOKEN_IDS = [11, 22, 33, 44];
const MODEL = 'qwen3.6-27b';
const ADAPTER = 'neuronpedia/jacobian-lens:qwen3.6-27b/oracle';

function fields(overrides: Partial<SignedOracleReadFields> = {}): SignedOracleReadFields {
  return {
    modelId: MODEL,
    adapter: ADAPTER,
    maxBullets: 2,
    position: 2,
    prefixDigest: oraclePrefixDigest(TOKEN_IDS.slice(0, 3)),
    layer: 24,
    bullets: ['The capital of France', 'A European city'],
    text: '- The capital of France\n- A European city\n',
    finish: 'bullets',
    ...overrides,
  };
}

function read(position: number, layer: number, text = `read ${position}/${layer}`): JlensExportOracleRead {
  const f = fields({
    position,
    prefixDigest: oraclePrefixDigest(TOKEN_IDS.slice(0, position + 1)),
    layer,
    bullets: [text],
    text: `- ${text}\n`,
  });
  return {
    position,
    layer,
    adapter: f.adapter,
    bullets: f.bullets,
    text: f.text,
    finish: f.finish,
    sig: signOracleRead(f)!,
  };
}

describe('oracle read signatures', () => {
  const saved = process.env.NEXTAUTH_SECRET;
  beforeEach(() => {
    process.env.NEXTAUTH_SECRET = 'test-secret';
  });
  afterEach(() => {
    process.env.NEXTAUTH_SECRET = saved;
  });

  it('verifies a read it signed', () => {
    const sig = signOracleRead(fields());
    expect(sig).toMatch(/^v1\./);
    expect(verifyOracleRead(fields(), sig!)).toBe(true);
  });

  it('rejects a changed field', () => {
    const sig = signOracleRead(fields())!;
    expect(verifyOracleRead(fields({ text: '- Something else\n' }), sig)).toBe(false);
    expect(verifyOracleRead(fields({ bullets: ['The capital of France'] }), sig)).toBe(false);
    expect(verifyOracleRead(fields({ layer: 28 }), sig)).toBe(false);
    expect(verifyOracleRead(fields({ maxBullets: 3 }), sig)).toBe(false);
    expect(verifyOracleRead(fields({ modelId: 'gemma-3-12b' }), sig)).toBe(false);
    expect(verifyOracleRead(fields({ adapter: 'other' }), sig)).toBe(false);
    expect(verifyOracleRead(fields({ prefixDigest: oraclePrefixDigest([11, 22, 34]) }), sig)).toBe(false);
  });

  it('rejects a signature made with another secret', () => {
    const sig = signOracleRead(fields())!;
    process.env.NEXTAUTH_SECRET = 'rotated';
    expect(verifyOracleRead(fields(), sig)).toBe(false);
  });

  it('does not sign or verify without a secret', () => {
    const sig = signOracleRead(fields())!;
    process.env.NEXTAUTH_SECRET = '';
    expect(signOracleRead(fields())).toBeNull();
    expect(verifyOracleRead(fields(), sig)).toBe(false);
    expect(verifySharedOracleReads(MODEL, TOKEN_IDS, 2, [read(1, 24)])).toBeUndefined();
  });

  it('stores signed reads in order, once each, with only the read fields', () => {
    const a = read(3, 24);
    const b = read(1, 28);
    const c = read(1, 24);
    const withExtra = { ...a, injected: '<script>' } as JlensExportOracleRead;
    const result = verifySharedOracleReads(MODEL, TOKEN_IDS, 2, [withExtra, b, c, a]);
    expect(result).toEqual({ maxBullets: 2, reads: [c, b, a] });
  });

  it('drops a read with a bad signature and keeps the others', () => {
    const tampered = { ...read(1, 24), text: '- Not what the oracle said\n' };
    expect(verifySharedOracleReads(MODEL, TOKEN_IDS, 2, [read(0, 24), tampered])).toEqual({
      maxBullets: 2,
      reads: [read(0, 24)],
    });
  });

  it('keeps a valid copy of a read after a tampered copy', () => {
    const tampered = { ...read(1, 24), text: '- Not what the oracle said\n' };
    expect(verifySharedOracleReads(MODEL, TOKEN_IDS, 2, [tampered, read(1, 24)])).toEqual({
      maxBullets: 2,
      reads: [read(1, 24)],
    });
  });

  it('drops reads of other tokens, and gives nothing when no read is left', () => {
    expect(verifySharedOracleReads(MODEL, [11, 22, 99, 44], 2, [read(2, 24)])).toBeUndefined();
    expect(verifySharedOracleReads(MODEL, TOKEN_IDS.slice(0, 2), 2, [read(3, 24)])).toBeUndefined();
  });

  it('drops reads when the bullet cap differs from the reads', () => {
    expect(verifySharedOracleReads(MODEL, TOKEN_IDS, 3, [read(1, 24)])).toBeUndefined();
  });
});
