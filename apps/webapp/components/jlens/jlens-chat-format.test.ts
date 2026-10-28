import { LensTokenMessage } from '@/lib/utils/lens';
import { describe, expect, it } from 'vitest';
import {
  extractAssistantText,
  groupTokensBySpans,
  lastTurnPrefill,
  messageIndicesForGroups,
  toChatPayload,
  tokensToText,
} from './jlens-chat-format';

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

// A prompt token with the server's span fields, positioned after `start`.
function spanned(tokens: [string, Partial<LensTokenMessage>][], start = 0): LensTokenMessage[] {
  return tokens.map(([token, span], i) => ({
    kind: 'token' as const,
    position: start + i,
    token,
    id: start + i,
    is_generated: false,
    results: [],
    ...span,
  }));
}

describe('groupTokensBySpans', () => {
  const sys = { role: 'system', message_index: 0 };
  const usr = { role: 'user', message_index: 1 };
  const explicitSystemChat = spanned([
    ['<|begin_of_text|>', { ...sys, section: 'header' }],
    ['system', { ...sys, section: 'header' }],
    ['Be', { ...sys, section: 'content' }],
    [' terse', { ...sys, section: 'content' }],
    ['<|eot_id|>', { ...sys, section: 'footer' }],
    ['user', { ...usr, section: 'header' }],
    ['Hi', { ...usr, section: 'content' }],
    ['<|eot_id|>', { ...usr, section: 'footer' }],
    ['assistant', { role: 'assistant', section: 'header' }],
    ['Hello', { role: 'assistant', section: 'content', is_generated: true }],
  ]);

  it('gives an explicit system message its own bubble with header and footer', () => {
    const { messages: groups } = groupTokensBySpans(explicitSystemChat);
    expect(groups.map((g) => g.role)).toEqual(['system', 'user', 'assistant']);
    expect(groups[0].headerTokens).toHaveLength(2);
    expect(tokensToText(groups[0].contentTokens)).toBe('Be terse');
    expect(groups[0].footerTokens).toHaveLength(1);
    expect(groups[0].messageIndex).toBe(0);
  });

  it('maps a system bubble to its message like any other input turn', () => {
    const { messages: groups } = groupTokensBySpans(explicitSystemChat);
    const idxs = messageIndicesForGroups(groups, [{ role: 'system' }, { role: 'user' }, { role: 'assistant' }]);
    expect(idxs).toEqual([0, 1, 2]);
  });

  it('keeps a template-injected system turn (no message index) as a separate unmapped bubble', () => {
    // What the engine sends once it labels the preamble it injects itself.
    const injected = { role: 'system', message_index: null };
    const tokens = spanned([
      ['<|begin_of_text|>', { ...injected, section: 'header' }],
      ['Cutting Knowledge', { ...injected, section: 'content' }],
      ['<|eot_id|>', { ...injected, section: 'footer' }],
      ['user', { role: 'user', message_index: 0, section: 'header' }],
      ['Hi', { role: 'user', message_index: 0, section: 'content' }],
    ]);
    const { messages: groups } = groupTokensBySpans(tokens);
    expect(groups.map((g) => g.role)).toEqual(['system', 'user']);
    expect(groups[0].messageIndex).toBeUndefined();
    expect(messageIndicesForGroups(groups, [{ role: 'user' }])).toEqual([null, 0]);
  });

  it('still folds the preamble into the first user header when the engine labels it that way', () => {
    // What every deployed engine sends today; the bubble must not split.
    const usr0 = { role: 'user', message_index: 0 };
    const tokens = spanned([
      ['<|begin_of_text|>', { ...usr0, section: 'header' }],
      ['system', { ...usr0, section: 'header' }],
      ['Cutting Knowledge', { ...usr0, section: 'header' }],
      ['user', { ...usr0, section: 'header' }],
      ['Hi', { ...usr0, section: 'content' }],
    ]);
    const { messages: groups } = groupTokensBySpans(tokens);
    expect(groups).toHaveLength(1);
    expect(groups[0].role).toBe('user');
    expect(groups[0].headerTokens).toHaveLength(4);
  });

  it('puts a tool result on the assistant side, mapped to its own message', () => {
    const tokens = spanned([
      ['<tool_call>', { role: 'assistant', message_index: 0, section: 'footer' }],
      ['<tool_response>', { role: 'tool', message_index: 1, section: 'header' }],
      ['Sunny', { role: 'tool', message_index: 1, section: 'content' }],
      ['It', { role: 'assistant', message_index: 2, section: 'content' }],
    ]);
    const { messages: groups } = groupTokensBySpans(tokens);
    expect(groups.map((g) => g.role)).toEqual(['assistant', 'assistant', 'assistant']);
    expect(messageIndicesForGroups(groups, [{ role: 'assistant' }, { role: 'tool' }, { role: 'assistant' }])).toEqual([
      0, 1, 2,
    ]);
  });

  it('splits an input assistant message into its reasoning and its answer, both mapped to it', () => {
    const asst = { role: 'assistant', message_index: 1 };
    const tokens = spanned([
      ['Hi', { ...usr, message_index: 0, section: 'content' }],
      ['<|im_start|>', { ...asst, section: 'header' }],
      ['<think>', { ...asst, channel: 'analysis', section: 'header' }],
      ['Say hi', { ...asst, channel: 'analysis', section: 'content' }],
      ['</think>', { ...asst, channel: 'analysis', section: 'footer' }],
      ['Hello', { ...asst, channel: 'final', section: 'content' }],
    ]);
    const { messages: groups } = groupTokensBySpans(tokens);
    expect(groups.map((g) => g.channel)).toEqual([undefined, 'analysis', 'final']);
    expect(tokensToText(groups[1].contentTokens)).toBe('Say hi');
    expect(groups[1].headerTokens.map((t) => t.token)).toEqual(['<|im_start|>', '<think>']);
    expect(groups[1].footerTokens.map((t) => t.token)).toEqual(['</think>']);
    expect(tokensToText(groups[2].contentTokens)).toBe('Hello');
    expect(messageIndicesForGroups(groups, [{ role: 'user' }, { role: 'assistant' }])).toEqual([0, 1, 1]);
  });
});

describe('lastTurnPrefill', () => {
  const usr = { role: 'user', message_index: 0, section: 'content' };
  const asst = { role: 'assistant', message_index: 1 };
  const prefill = (tokens: [string, Partial<LensTokenMessage>][]) =>
    lastTurnPrefill(groupTokensBySpans(spanned(tokens)).messages);

  it('returns the input text before the first generated token', () => {
    expect(
      prefill([
        ['Hi', usr],
        ['<|im_start|>assistant', { ...asst, section: 'header' }],
        ['Sure', { ...asst, section: 'content' }],
        [',', { ...asst, section: 'content', is_generated: true }],
      ]),
    ).toBe('Sure');
  });

  it('keeps the reasoning and its tags when the prefill is two bubbles', () => {
    expect(
      prefill([
        ['Hi', usr],
        ['<|im_start|>assistant\n', { ...asst, section: 'header' }],
        ['<think>', { ...asst, channel: 'analysis', section: 'header' }],
        ['\nSay hi.\n', { ...asst, channel: 'analysis', section: 'content' }],
        ['</think>', { ...asst, channel: 'analysis', section: 'footer' }],
        ['\n\nHello', { ...asst, channel: 'final', section: 'content' }],
        [' there', { ...asst, channel: 'final', section: 'content', is_generated: true }],
      ]),
    ).toBe('<think>\nSay hi.\n</think>\n\nHello');
  });

  it('is empty with no generated token, or only whitespace before it', () => {
    expect(
      prefill([
        ['Hi', usr],
        ['Hello', { ...asst, section: 'content' }],
      ]),
    ).toBe('');
    expect(
      prefill([
        ['Hi', usr],
        ['<think>\n\n</think>', { ...asst, section: 'scaffold' }],
        ['\n\n', { ...asst, section: 'content' }],
        ['Hello', { ...asst, section: 'content', is_generated: true }],
      ]),
    ).toBe('');
  });
});

describe('toChatPayload', () => {
  it('keeps tool calls and call ids, and drops empty tool fields', () => {
    const call = { name: 'get_weather', arguments: { city: 'Paris' }, id: 'call_1' };
    expect(
      toChatPayload([
        { role: 'user', content: 'Hi', toolCalls: [] },
        { role: 'assistant', content: '', toolCalls: [call] },
        { role: 'tool', content: 'Sunny', toolCallId: 'call_1' },
      ]),
    ).toEqual([
      { role: 'user', content: 'Hi' },
      { role: 'assistant', content: '', toolCalls: [call] },
      { role: 'tool', content: 'Sunny', toolCallId: 'call_1' },
    ]);
  });
});
