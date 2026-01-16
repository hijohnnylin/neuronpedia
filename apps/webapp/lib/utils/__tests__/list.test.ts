import { describe, expect, test } from 'vitest';
import {
  MAX_LIST_FEATURES_FOR_TEST_TEXT,
  MAX_LIST_TEST_TEXT_LENGTH_CHARS,
} from '../list';

describe('MAX_LIST_FEATURES_FOR_TEST_TEXT', () => {
  test('has correct value', () => {
    expect(MAX_LIST_FEATURES_FOR_TEST_TEXT).toBe(20);
  });

  test('is a number', () => {
    expect(typeof MAX_LIST_FEATURES_FOR_TEST_TEXT).toBe('number');
  });
});

describe('MAX_LIST_TEST_TEXT_LENGTH_CHARS', () => {
  test('has correct value', () => {
    expect(MAX_LIST_TEST_TEXT_LENGTH_CHARS).toBe(500);
  });

  test('is a number', () => {
    expect(typeof MAX_LIST_TEST_TEXT_LENGTH_CHARS).toBe('number');
  });
});
