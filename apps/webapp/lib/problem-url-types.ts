// ─── URL → Type Detection Rules ────────────────────────────────────────────
// Shared between client (draft-node) and server (problem-utils, submit-urls).
// Add new rules here. First match wins.

type TypeRule = { match: (url: URL) => boolean; type: string };

const TYPE_RULES: TypeRule[] = [
  { match: (u) => u.hostname === 'arxiv.org' || u.hostname === 'www.arxiv.org', type: 'paper' },
  { match: (u) => u.hostname === 'openreview.net' || u.hostname === 'www.openreview.net', type: 'paper' },
  { match: (u) => u.hostname === 'scholar.google.com', type: 'paper' },
  {
    match: (u) => u.hostname === 'semanticscholar.org' || u.hostname.endsWith('.semanticscholar.org'),
    type: 'paper',
  },
  { match: (u) => u.hostname === 'paperswithcode.com' || u.hostname === 'www.paperswithcode.com', type: 'paper' },
  { match: (u) => u.hostname === 'github.com' || u.hostname === 'www.github.com', type: 'tool' },
  { match: (u) => u.hostname === 'gitlab.com' || u.hostname === 'www.gitlab.com', type: 'tool' },
  { match: (u) => u.hostname === 'pypi.org' || u.hostname === 'www.pypi.org', type: 'tool' },
  { match: (u) => u.hostname === 'npmjs.com' || u.hostname === 'www.npmjs.com', type: 'tool' },
  {
    match: (u) =>
      (u.hostname === 'huggingface.co' ||
        u.hostname === 'www.huggingface.co' ||
        u.hostname === 'huggingface.com' ||
        u.hostname === 'www.huggingface.com') &&
      u.pathname.includes('/datasets'),
    type: 'dataset',
  },
  {
    match: (u) =>
      u.hostname === 'huggingface.co' ||
      u.hostname === 'www.huggingface.co' ||
      u.hostname === 'huggingface.com' ||
      u.hostname === 'www.huggingface.com',
    type: 'model',
  },
  { match: (u) => u.hostname === 'kaggle.com' || u.hostname === 'www.kaggle.com', type: 'dataset' },
];

/**
 * Normalize known URL variants to their canonical form.
 * e.g. arxiv.org/pdf/... and arxiv.org/html/... → arxiv.org/abs/...
 */
export function normalizeUrl(url: string): string {
  try {
    const parsed = new URL(url);
    if (parsed.hostname === 'arxiv.org' || parsed.hostname === 'www.arxiv.org') {
      const pdfOrHtml = parsed.pathname.match(/^\/(pdf|html)\/(.+)/);
      if (pdfOrHtml) {
        const id = pdfOrHtml[2].replace(/v\d+$/, '').replace(/\.pdf$/, '');
        parsed.pathname = `/abs/${id}`;
        return parsed.toString();
      }
    }
    return url;
  } catch {
    return url;
  }
}

export function detectTypeFromUrl(url: string): string {
  try {
    const parsed = new URL(url);
    for (const rule of TYPE_RULES) {
      if (rule.match(parsed)) return rule.type;
    }
  } catch {
    // invalid URL
  }
  return 'paper';
}
