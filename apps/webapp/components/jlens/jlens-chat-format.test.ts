import { LensTokenMessage } from '@/lib/utils/lens';
import { describe, expect, it } from 'vitest';
import { extractAssistantText, tokensToText } from './jlens-chat-format';

// The generated half of an assistant turn, as the stream delivers it: one token
// per position, spanned by the server as message content.
function generatedTokens(tokens: string[], overrides: Partial<LensTokenMessage> = {}): LensTokenMessage[] {
  return tokens.map((token, i) => ({
    kind: 'token' as const,
    position: i,
    token,
    id: i,
    is_generated: true,
    section: 'content',
    results: [],
    ...overrides,
  }));
}

// One character spread over `count` tokens, as the server sends it: the whole glyph at every
// position (so each chip shows the emoji rather than a `), with the repeats flagged.
function splitCharTokens(char: string, count = 2): LensTokenMessage[] {
  return generatedTokens(Array(count).fill(char)).map((t, i) => ({ ...t, is_char_continuation: i > 0 }));
}

describe('extractAssistantText', () => {
  it('keeps the space the first generated token carries after a prefill', () => {
    // The whole point: `hi` + ` how are you?` must not become `hihow are you?`,
    // which retokenizes differently when the turn is re-sent as history.
    const text = extractAssistantText(generatedTokens([' how', ' are', ' you', '?']), 'hi');
    expect(text).toBe('hi how are you?');
  });

  it('still drops surrounding whitespace when there is no prefill', () => {
    expect(extractAssistantText(generatedTokens(['\n', 'hello', ' there', '\n']))).toBe('hello there');
  });

  it('drops trailing whitespace after a prefill, since the template re-adds its own', () => {
    expect(extractAssistantText(generatedTokens([' sure', '.', '\n'], {}), 'Yes,')).toBe('Yes, sure.');
  });

  it('returns the prefill unchanged when nothing was generated', () => {
    expect(extractAssistantText([], 'hi')).toBe('hi');
  });

  it('keeps only the final channel of a reasoning turn', () => {
    const tokens: LensTokenMessage[] = [
      ...generatedTokens(['thinking', ' hard'], { channel: 'analysis' }),
      ...generatedTokens([' the', ' answer'], { channel: 'final' }),
    ];
    expect(extractAssistantText(tokens)).toBe('the answer');
    expect(extractAssistantText(tokens, 'A:')).toBe('A: the answer');
  });

  it('strips residual markers on an unspanned (legacy) turn', () => {
    const tokens = generatedTokens([' how', ' are', ' you', '<|im_end|>'], { section: null });
    expect(extractAssistantText(tokens)).toBe('how are you');
    expect(extractAssistantText(tokens, 'hi')).toBe('hi how are you');
  });

  it('counts a split emoji once, however many tokens showed it', () => {
    // An emoji split across tokens has its whole glyph repeated at every contributing
    // position so each chip renders it. Joining those strings is what doubled the emoji
    // in the previous turn once the conversation was re-sent as history.
    // The run's combined string carries whatever its first fragment held, the leading space
    // included (` 😀`), exactly as the tokenizer decodes it.
    const tokens = [
      ...generatedTokens(['nice']),
      ...splitCharTokens(' 😀'),
      ...splitCharTokens('🎉', 3),
      ...generatedTokens(['!']),
    ];
    expect(extractAssistantText(tokens)).toBe('nice 😀🎉!');
  });
});

describe('tokensToText', () => {
  it('joins whole tokens as-is', () => {
    expect(tokensToText(generatedTokens(['Hi', ' there']))).toBe('Hi there');
  });

  it('keeps two adjacent split emoji distinct', () => {
    // The case the repeated glyph cannot express on its own, and why the server flags the
    // repeats instead of leaving the client to collapse equal neighbours: two split emoji
    // look exactly like one shown across its fragments.
    expect(tokensToText([...splitCharTokens('😀'), ...splitCharTokens('😀')])).toBe('😀😀');
  });

  it('treats a token missing the flag as its own character', () => {
    // Runs stored before the flag existed carry no `is_char_continuation`; they keep their
    // old (joined) reading rather than losing characters.
    expect(tokensToText([{ token: '😀' }, { token: '😀' }])).toBe('😀😀');
  });
});
