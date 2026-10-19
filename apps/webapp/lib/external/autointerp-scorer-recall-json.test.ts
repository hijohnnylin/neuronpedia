import { describe, expect, it } from 'vitest';
import { parseMatchResponse } from './autointerp-scorer-recall-json';

describe('parseMatchResponse', () => {
  it('reads plain JSON', () => {
    expect(parseMatchResponse('{"match": true, "reason": "cities"}')).toEqual({ match: true, reason: 'cities' });
  });

  it('reads JSON inside a ```json fence', () => {
    expect(parseMatchResponse('```json\n{\n  "match": false,\n  "reason": "no cats"\n}\n```')).toEqual({
      match: false,
      reason: 'no cats',
    });
  });

  it('accepts match as a string', () => {
    expect(parseMatchResponse('{"match": "true", "reason": "x"}').match).toBe(true);
    expect(parseMatchResponse('{"match": "false", "reason": "x"}').match).toBe(false);
  });

  it('throws when there is no match value or no JSON', () => {
    expect(() => parseMatchResponse('{"reason": "x"}')).toThrow();
    expect(() => parseMatchResponse('')).toThrow();
    expect(() => parseMatchResponse('yes it matches')).toThrow();
  });
});
