import { NPSteerType } from '@/lib/api/inference-types';
import { ASSISTANT_AXIS_ID } from '@/lib/utils/steer';
import { PublicAxisReadout } from '@/lib/utils/steer-axis-public';

export const CAP_GITHUB_URL = 'https://github.com/safety-research/assistant-axis';
export const CAP_PAPER_URL = 'https://arxiv.org/abs/2601.10387';
export const CAP_BLOG_URL = 'https://www.anthropic.com/research/assistant-axis';
export const CAP_CONTACT_EMAIL = 'jacklindsey@anthropic.com,christina.lu@cs.ox.ac.uk';
export const CAP_VECTOR_URL = '/llama3.3-70b-it/40-neuronpedia-resid-post/101874252';
export const DEMO_BUTTONS = [
  { id: 'cmkjhhsu0000fgfu5pkv3zlmv', emoji: '😢', label: 'Isolation' },
  { id: 'cmkhii9hk0015ruw6zpzwan1z', emoji: '🌀', label: 'Sycophancy' },
  { id: 'cmkhj4zb5000vmj34bcicslcg', emoji: '💸', label: 'Tax Fraud' },
  { id: null, emoji: '✏️', label: 'Free Chat' },
] as const;

/**
 * This page's one axis, as read for one steer type, out of everything a response carried.
 *
 * The page asks for `lu_assistant-axis` and plots that, so it looks the reading up by the id it
 * asked for. Null when a response carried no reading of it -- a conversation saved before the axis
 * was measured, or a run that only produced the other column.
 */
export function assistantAxisFor(axes: PublicAxisReadout[] | undefined, type: NPSteerType): PublicAxisReadout | null {
  return axes?.find((axis) => axis.id === ASSISTANT_AXIS_ID && axis.type === type) ?? null;
}
