// Model IDs without their dots. Old LessWrong posts link to these, because its hover preview
// once failed on dots in a URL. Keep them so those links still resolve.
export const LEGACY_MODEL_ID_ALIASES = new Map<string, string>([
  ['llama31-8b-it', 'llama3.1-8b-it'],
  ['qwen25-7b-it', 'qwen2.5-7b-it'],
]);

const DEFAULT_REDIRECT_TO_STEER_STRENGTH = '10';

/**
 * Returns the path and query that a legacy model ID URL redirects to, or null when the first path
 * segment is not a legacy model ID.
 */
export function getLegacyModelIdRedirect(pathname: string, searchParams: URLSearchParams): string | null {
  const [, legacyModelId, ...rest] = pathname.split('/');
  const modelId = LEGACY_MODEL_ID_ALIASES.get(legacyModelId);
  if (!modelId) {
    return null;
  }

  // Old posts link a feature with redirectToSteer=true to get a steer preview.
  if (rest.length === 2 && searchParams.get('redirectToSteer') === 'true') {
    const steerParams = new URLSearchParams({
      embed: 'true',
      source: rest[0],
      index: rest[1],
      hideInitialSettingsOnMobile: 'true',
      strength: searchParams.get('strength') || DEFAULT_REDIRECT_TO_STEER_STRENGTH,
    });
    const saved = searchParams.get('saved');
    if (saved) {
      steerParams.set('saved', saved);
    }
    return `/${modelId}/steer?${steerParams.toString()}`;
  }

  const query = searchParams.toString();
  return `/${[modelId, ...rest].join('/')}${query ? `?${query}` : ''}`;
}
