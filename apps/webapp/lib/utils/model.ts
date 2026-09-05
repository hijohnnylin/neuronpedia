// A HuggingFace repo id, as stored in `Model.hfRepoId`.
//
// Exactly one slash is required, so `gpt2` is rejected even though the Hub still resolves it as a
// legacy canonical repo. That is the point rather than an oversight: a slash is what tells a
// Neuronpedia model id apart from a HuggingFace one, here and in the SQL that audits the column.
// A bare name would be ambiguous with our own ids, and the canonical spelling exists anyway
// (`openai-community/gpt2`).
//
// Each half must begin with a letter or digit, which is also what stops a component being `..`
// -- this value reaches URL building in `lib/utils/saelens.ts`, with our HF_TOKEN attached.
export const HF_REPO_ID_REGEX = /^[A-Za-z0-9][A-Za-z0-9._-]*\/[A-Za-z0-9][A-Za-z0-9._-]*$/;

// The Hub's own limit per component. Two of those plus the separator.
export const MAX_HF_REPO_ID_CHARS = 193;

export const HF_REPO_ID_ERROR_MESSAGE =
  "HuggingFace repo id must be `namespace/name`, using only letters, digits, '.', '_' and '-'";

/**
 * Report whether a string is a well-formed HuggingFace repo id.
 *
 * Case is preserved by every caller. Unlike `Model.id`, which is lowercased on the way in, a repo
 * id is case-sensitive on the Hub -- `google/gemma-4-E2B` is not `google/gemma-4-e2b`.
 */
export function isValidHfRepoId(value: string): boolean {
  return value.length <= MAX_HF_REPO_ID_CHARS && HF_REPO_ID_REGEX.test(value);
}
