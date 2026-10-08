import { describe, expect, it } from 'vitest';
import { LensColumn, lensTabFromColumns, shareLensColumns } from './lens';

const { JACOBIAN_LENS: J, ORACLE_LENS: O, LOGIT_LENS: L } = LensColumn;

describe('shareLensColumns', () => {
  it('uses lensColumns, in display order and once each', () => {
    expect(shareLensColumns([L, O, J, L], 'JACOBIAN_LENS')).toEqual([J, O, L]);
    expect(shareLensColumns([O], 'DIFF')).toEqual([O]);
  });

  it('drops values that are not columns', () => {
    expect(shareLensColumns(['DIFF', L], null)).toEqual([L]);
  });

  it('falls back to the legacy tab when lensColumns has no columns', () => {
    expect(shareLensColumns([], 'DIFF')).toEqual([J, L]);
    expect(shareLensColumns(null, 'LOGIT_LENS')).toEqual([L]);
    expect(shareLensColumns(undefined, 'JACOBIAN_LENS')).toEqual([J]);
    expect(shareLensColumns(['nope'], undefined)).toEqual([J]);
  });
});

describe('lensTabFromColumns', () => {
  it('stores the token lenses of a set as the legacy tab', () => {
    expect(lensTabFromColumns([J, O, L])).toBe('DIFF');
    expect(lensTabFromColumns([J, L])).toBe('DIFF');
    expect(lensTabFromColumns([O, L])).toBe('LOGIT_LENS');
    expect(lensTabFromColumns([L])).toBe('LOGIT_LENS');
    expect(lensTabFromColumns([J, O])).toBe('JACOBIAN_LENS');
    expect(lensTabFromColumns([O])).toBe('JACOBIAN_LENS');
  });
});
