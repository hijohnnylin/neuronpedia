import { describe, expect, it } from 'vitest';
import { describeImportError } from './import-error';

describe('describeImportError', () => {
  it('returns the server message from an SSE error frame', () => {
    const frame = { type: 'error', isTrusted: true, data: JSON.stringify({ message: 'S3 download failed' }) };
    expect(describeImportError(frame)).toBe('S3 download failed');
  });

  it('falls back to the raw payload when it is not JSON', () => {
    expect(describeImportError({ data: 'boom' })).toBe('boom');
  });

  it('does not surface {"isTrusted":true} for a dropped connection', () => {
    const msg = describeImportError({ type: 'error', isTrusted: true });
    expect(msg).not.toContain('isTrusted');
    expect(msg).toContain('connection');
  });

  it('handles Error instances and strings', () => {
    expect(describeImportError(new Error('nope'))).toBe('nope');
    expect(describeImportError('plain')).toBe('plain');
  });

  it('handles null and undefined', () => {
    expect(describeImportError(null)).toContain('connection');
    expect(describeImportError(undefined)).toContain('connection');
  });
});
