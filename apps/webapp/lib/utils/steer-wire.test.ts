import { describe, expect, it } from 'vitest';
import { STEER_COMPLETION_VERSION } from './steer';
import { assistantAxisFromStored, assistantAxisToStored, storedOutputTextIncludesPrompt } from './steer-wire';

// A row as it was written before the inference wire became camelCase. Rows like this are still
// in the database and are never migrated, so reading one has to keep working indefinitely.
const LEGACY_ROW = {
  pc_titles: ['warmth', 'formality'],
  turns: [
    { pc_values: { warmth: 0.5 }, pc_values_post_cap: { warmth: 0.2 }, snippet: 'hello' },
    { pc_values: { warmth: -1 }, pc_values_post_cap: { warmth: -1 }, snippet: 'bye' },
  ],
  type: 'STEERED',
};

describe('assistantAxisFromStored', () => {
  it('reads a legacy snake_case row', () => {
    expect(assistantAxisFromStored(LEGACY_ROW)).toEqual({
      pcTitles: ['warmth', 'formality'],
      turns: [
        { pcValues: { warmth: 0.5 }, pcValuesPostCap: { warmth: 0.2 }, snippet: 'hello' },
        { pcValues: { warmth: -1 }, pcValuesPostCap: { warmth: -1 }, snippet: 'bye' },
      ],
      type: 'STEERED',
    });
  });

  it('reads a camelCase row, which is what newer writes produce upstream', () => {
    const axis = assistantAxisFromStored({
      pcTitles: ['warmth'],
      turns: [{ pcValues: { warmth: 1 }, pcValuesPostCap: { warmth: 1 }, snippet: 'hi' }],
      type: 'DEFAULT',
    });
    expect(axis.pcTitles).toEqual(['warmth']);
    expect(axis.turns?.[0].pcValues).toEqual({ warmth: 1 });
    expect(axis.type).toBe('DEFAULT');
  });

  it('prefers camelCase when a row somehow carries both', () => {
    const axis = assistantAxisFromStored({
      pc_titles: ['old'],
      pcTitles: ['new'],
      turns: [{ pc_values: { a: 0 }, pcValues: { a: 1 } }],
    });
    expect(axis.pcTitles).toEqual(['new']);
    expect(axis.turns?.[0].pcValues).toEqual({ a: 1 });
  });

  it('survives the shapes a nullable json column actually yields', () => {
    // `capMonitorOutput` is nullable and rows predate parts of the shape, so absent is normal.
    for (const empty of [null, undefined, {}]) {
      expect(assistantAxisFromStored(empty)).toEqual({ pcTitles: undefined, turns: [], type: undefined });
    }
    expect(assistantAxisFromStored({ pc_titles: ['a'] }).turns).toEqual([]);
  });
});

describe('assistantAxisToStored', () => {
  it('always writes snake_case, whatever existing readers were built against', () => {
    const stored = assistantAxisToStored({
      pcTitles: ['warmth'],
      turns: [{ pcValues: { warmth: 0.5 }, pcValuesPostCap: { warmth: 0.2 }, snippet: 'hello' }],
      type: 'STEERED',
    });
    expect(stored).toEqual({
      pc_titles: ['warmth'],
      turns: [{ pc_values: { warmth: 0.5 }, pc_values_post_cap: { warmth: 0.2 }, snippet: 'hello' }],
      type: 'STEERED',
    });
    expect(Object.keys(stored)).not.toContain('pcTitles');
  });

  it('round-trips a legacy row back to the same stored shape', () => {
    // The property that matters: reading and rewriting a row must not silently change its
    // format, or a row rewritten once becomes unreadable to anything expecting the old names.
    expect(assistantAxisToStored(assistantAxisFromStored(LEGACY_ROW))).toEqual(LEGACY_ROW);
  });

  it('writes an empty turn list rather than dropping the key', () => {
    expect(assistantAxisToStored({})).toEqual({ pc_titles: undefined, turns: [], type: undefined });
  });
});

describe('storedOutputTextIncludesPrompt', () => {
  it('flags a completion saved before inference stopped returning the prompt', () => {
    expect(
      storedOutputTextIncludesPrompt({ version: STEER_COMPLETION_VERSION - 1, outputTextChatTemplate: null }),
    ).toBe(true);
  });

  it('does not flag a completion saved at the current version', () => {
    expect(storedOutputTextIncludesPrompt({ version: STEER_COMPLETION_VERSION, outputTextChatTemplate: null })).toBe(
      false,
    );
  });

  it('never flags a chat row, whose outputText is not rendered on its own', () => {
    // Chat rows share this table and are still at older versions, but they render from
    // outputTextChatTemplate, so treating one as prompt-prefixed would strip a real message.
    expect(
      storedOutputTextIncludesPrompt({ version: STEER_COMPLETION_VERSION - 1, outputTextChatTemplate: '[]' }),
    ).toBe(false);
  });
});
