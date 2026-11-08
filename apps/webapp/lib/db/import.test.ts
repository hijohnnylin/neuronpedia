import { describe, expect, it, vi } from 'vitest';
import { dropJsonNulls } from './import';

vi.mock('@/lib/db', () => ({ prisma: {} }));

describe('dropJsonNulls', () => {
  it('replaces a null escape with a space', () => {
    const line = JSON.stringify({ text: 'a\u0000b' });
    expect(JSON.parse(dropJsonNulls(line))).toEqual({ text: 'a b' });
  });

  it('keeps a literal backslash followed by "u0000"', () => {
    const line = JSON.stringify({ text: 'path\\u0000' });
    expect(dropJsonNulls(line)).toBe(line);
    expect(JSON.parse(dropJsonNulls(line))).toEqual({ text: 'path\\u0000' });
  });

  it('replaces a null escape after an escaped backslash', () => {
    const line = JSON.stringify({ text: '\\\u0000' });
    expect(JSON.parse(dropJsonNulls(line))).toEqual({ text: '\\ ' });
  });

  it('replaces every null escape on every line', () => {
    const lines = [JSON.stringify({ a: '\u0000\u0000' }), JSON.stringify({ b: 'x\u0000' })].join('\n');
    expect(
      dropJsonNulls(lines)
        .split('\n')
        .map((l) => JSON.parse(l)),
    ).toEqual([{ a: '  ' }, { b: 'x ' }]);
  });
});
