import { describe, expect, test } from 'vitest';
import { cn } from '../ui';

describe('cn', () => {
  test('merges multiple class strings', () => {
    const result = cn('class1', 'class2', 'class3');
    expect(result).toBe('class1 class2 class3');
  });

  test('handles empty input', () => {
    const result = cn();
    expect(result).toBe('');
  });

  test('resolves Tailwind class conflicts', () => {
    const result = cn('p-4', 'p-2');
    expect(result).toBe('p-2');
  });

  test('handles undefined and null values', () => {
    const result = cn('class1', undefined, 'class2', null);
    expect(result).toBe('class1 class2');
  });

  test('merges conditional classes', () => {
    const result = cn('base-class', true && 'conditional-class', false && 'hidden-class');
    expect(result).toBe('base-class conditional-class');
  });
});
