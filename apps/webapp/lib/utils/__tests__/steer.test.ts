import { describe, expect, test } from 'vitest';
import {
  replaceSteerModelIdIfNeeded,
  convertOldSteerOutputToChatMessages,
} from '../steer';

describe('replaceSteerModelIdIfNeeded', () => {
  test('returns modelId unchanged when not in allow list', () => {
    const result = replaceSteerModelIdIfNeeded('test-model');
    expect(result).toBe('test-model');
  });

  test('returns modelId unchanged when not ending with -it', () => {
    const result = replaceSteerModelIdIfNeeded('model-without-it');
    expect(result).toBe('model-without-it');
  });

  test('handles empty string', () => {
    const result = replaceSteerModelIdIfNeeded('');
    expect(result).toBe('');
  });
});

describe('convertOldSteerOutputToChatMessages', () => {
  test('converts simple user message', () => {
    const input = '<start_of_turn>user\nHello<end_of_turn>';
    const result = convertOldSteerOutputToChatMessages(input);
    // split('<end_of_turn>') creates an extra empty string after the last delimiter
    expect(result.length).toBeGreaterThanOrEqual(1);
    expect(result[0].role).toBe('user');
    expect(result[0].content).toBe('Hello');
  });

  test('converts multiple messages', () => {
    const input = '<start_of_turn>user\nHello<end_of_turn><start_of_turn>model\nHi there<end_of_turn>';
    const result = convertOldSteerOutputToChatMessages(input);
    // split('<end_of_turn>') creates an extra empty string after the last delimiter
    expect(result.length).toBeGreaterThanOrEqual(2);
    expect(result[0].role).toBe('user');
    expect(result[0].content).toBe('Hello');
    expect(result[1].role).toBe('model');
    expect(result[1].content).toBe('Hi there');
  });

  test('removes bos and eos tags', () => {
    const input = '<bos><start_of_turn>user\nHello<eos><end_of_turn>';
    const result = convertOldSteerOutputToChatMessages(input);
    expect(result[0].content).toBe('Hello');
    expect(result[0].content).not.toContain('<bos>');
    expect(result[0].content).not.toContain('<eos>');
  });

  test('handles unknown role as system', () => {
    const input = '<start_of_turn>unknown\nSome content<end_of_turn>';
    const result = convertOldSteerOutputToChatMessages(input);
    expect(result[0].role).toBe('system');
    expect(result[0].content).toBe('Some content');
  });

  test('handles multiline content', () => {
    const input = '<start_of_turn>user\nLine 1\nLine 2<end_of_turn>';
    const result = convertOldSteerOutputToChatMessages(input);
    expect(result[0].content).toBe('Line 1\nLine 2');
  });

  test('handles empty content', () => {
    const input = '<start_of_turn>user\n<end_of_turn>';
    const result = convertOldSteerOutputToChatMessages(input);
    expect(result[0].content).toBe('');
  });

  test('handles empty string', () => {
    const result = convertOldSteerOutputToChatMessages('');
    // Empty string splits into [''] which creates one message with empty content
    expect(result.length).toBe(1);
    expect(result[0].content).toBe('');
    expect(result[0].role).toBe('system');
  });
});
