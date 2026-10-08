import { describe, expect, it } from 'vitest';
import { lensChatMessageSchema, lensChatToolsSchema } from './lens-chat-schema';

const schema = lensChatMessageSchema(100);
const call = { name: 'get_weather', arguments: { city: 'Paris' }, id: 'call_1' };

describe('lensChatMessageSchema', () => {
  it('accepts an assistant tool call with empty content', () => {
    expect(schema.isValidSync({ role: 'assistant', content: '', toolCalls: [call] })).toBe(true);
  });

  it('accepts a tool result with its call id', () => {
    expect(schema.isValidSync({ role: 'tool', content: 'Sunny', toolCallId: 'call_1' })).toBe(true);
  });

  it('refuses tool calls on a non-assistant message', () => {
    expect(schema.isValidSync({ role: 'user', content: 'Hi', toolCalls: [call] })).toBe(false);
  });

  it('refuses a call id on a non-tool message', () => {
    expect(schema.isValidSync({ role: 'assistant', content: 'Hi', toolCallId: 'call_1' })).toBe(false);
  });

  it('refuses unknown roles, missing content and content over the cap', () => {
    expect(schema.isValidSync({ role: 'function', content: 'Hi' })).toBe(false);
    expect(schema.isValidSync({ role: 'user' })).toBe(false);
    expect(schema.isValidSync({ role: 'user', content: 'x'.repeat(101) })).toBe(false);
  });

  it('refuses tool call arguments that are not an object', () => {
    expect(schema.isValidSync({ role: 'assistant', content: '', toolCalls: [{ name: 'f', arguments: '{}' }] })).toBe(
      false,
    );
  });
});

describe('lensChatToolsSchema', () => {
  it('accepts tool definitions or none', () => {
    expect(lensChatToolsSchema.isValidSync([{ type: 'function', function: { name: 'f' } }])).toBe(true);
    expect(lensChatToolsSchema.isValidSync(undefined)).toBe(true);
  });

  it('refuses tool definitions over the size cap', () => {
    expect(lensChatToolsSchema.isValidSync([{ description: 'x'.repeat(60000) }])).toBe(false);
  });
});
