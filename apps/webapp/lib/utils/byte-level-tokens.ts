// Byte-level (GPT-2 / Qwen) token decoding. Pure, so it is safe on the server
// and in the browser.

// Reproduces HF's `bytes_to_unicode`: a reversible map from each of the 256
// byte values to a printable unicode codepoint. We build the inverse here
// (printable char -> byte) so we can turn a raw vocab string back into bytes.
function buildByteDecoder(): Map<string, number> {
  const bs: number[] = [];
  const addRange = (from: string, to: string) => {
    for (let i = from.codePointAt(0)!; i <= to.codePointAt(0)!; i += 1) {
      bs.push(i);
    }
  };
  addRange('!', '~');
  addRange('\u00a1', '\u00ac');
  addRange('\u00ae', '\u00ff');

  const cs = [...bs];
  let n = 0;
  for (let b = 0; b < 256; b += 1) {
    if (!bs.includes(b)) {
      bs.push(b);
      cs.push(256 + n);
      n += 1;
    }
  }

  const decoder = new Map<string, number>();
  for (let i = 0; i < bs.length; i += 1) {
    decoder.set(String.fromCodePoint(cs[i]), bs[i]);
  }
  return decoder;
}

export const BYTE_DECODER = buildByteDecoder();
const UTF8_DECODER = typeof TextDecoder !== 'undefined' ? new TextDecoder('utf-8', { fatal: false }) : null;
const UTF8_ENCODER = typeof TextEncoder !== 'undefined' ? new TextEncoder() : null;

// Decode a raw byte-level token string back into its real text, or null if the
// string isn't a byte-level token (e.g. it is already real characters) or
// decodes to invalid UTF-8 (a partial multi-byte token).
export function decodeRawToken(raw: string): string | null {
  if (!UTF8_DECODER) {
    return null;
  }
  const bytes: number[] = [];
  for (const ch of raw) {
    const b = BYTE_DECODER.get(ch);
    if (b === undefined) {
      return null;
    }
    bytes.push(b);
  }
  const text = UTF8_DECODER.decode(Uint8Array.from(bytes));
  return text.includes('\ufffd') ? null : text;
}

// Stored activation tokens are often half-decoded: spaces are real but
// newlines are still `Ċ` and curly quotes are still `âĢĻ`. This maps the
// byte-level characters back to bytes, passes real characters through, and
// returns the input unchanged when the result is not valid UTF-8.
export function decodeMixedToken(token: string): string {
  if (!UTF8_DECODER || !UTF8_ENCODER) {
    return token;
  }
  const bytes: number[] = [];
  for (const ch of token) {
    const b = BYTE_DECODER.get(ch);
    if (b !== undefined) {
      bytes.push(b);
    } else {
      bytes.push(...UTF8_ENCODER.encode(ch));
    }
  }
  const text = UTF8_DECODER.decode(Uint8Array.from(bytes));
  return text.includes('\ufffd') ? token : text;
}
