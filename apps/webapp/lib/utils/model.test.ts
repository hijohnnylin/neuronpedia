import { describe, expect, it } from 'vitest';
import { MAX_HF_REPO_ID_CHARS, isValidHfRepoId } from './model';

// Every value the migration backfills, so the validator cannot reject something already in the
// column. Taken from np_model_to_hf.json, including the awkward ones.
const BACKFILLED_REPO_IDS = [
  'EleutherAI/pythia-70m-deduped',
  'openai-community/gpt2',
  'google/gemma-3-1b-pt',
  'Qwen/Qwen3.5-0.8B',
  'Qwen/Qwen2.5-1.5B-Instruct',
  'google/gemma-4-E2B',
  'meta-llama/Llama-3.3-70B-Instruct',
  'meta-models/Muse-Glimmer-30B',
  'allenai/Olmo-3-1025-7B',
  'deepseek-ai/DeepSeek-R1-Distill-Llama-8B',
];

describe('isValidHfRepoId', () => {
  it.each(BACKFILLED_REPO_IDS)('accepts %s', (repoId) => {
    expect(isValidHfRepoId(repoId)).toBe(true);
  });

  it('accepts a repo id whose name carries dots, underscores and dashes', () => {
    expect(isValidHfRepoId('org/Some_model.v2-final')).toBe(true);
  });

  it('rejects a bare name, which the Hub resolves but we do not store', () => {
    expect(isValidHfRepoId('gpt2')).toBe(false);
  });

  it('rejects a Neuronpedia model id, which is the whole point of requiring the slash', () => {
    expect(isValidHfRepoId('gpt2-small')).toBe(false);
    expect(isValidHfRepoId('gemma-2-2b-it')).toBe(false);
  });

  it('rejects more than one slash', () => {
    expect(isValidHfRepoId('org/repo/extra')).toBe(false);
  });

  it('rejects an empty half', () => {
    expect(isValidHfRepoId('/repo')).toBe(false);
    expect(isValidHfRepoId('org/')).toBe(false);
    expect(isValidHfRepoId('/')).toBe(false);
  });

  it('rejects a component that is a relative path, since this value reaches URL building', () => {
    expect(isValidHfRepoId('../secrets')).toBe(false);
    expect(isValidHfRepoId('org/..')).toBe(false);
    expect(isValidHfRepoId('../..')).toBe(false);
  });

  it('rejects query and fragment characters that would redirect a fetch', () => {
    expect(isValidHfRepoId('org/repo?x=1')).toBe(false);
    expect(isValidHfRepoId('org/repo#main')).toBe(false);
    expect(isValidHfRepoId('org/repo with space')).toBe(false);
  });

  it('rejects a component starting with punctuation', () => {
    expect(isValidHfRepoId('.org/repo')).toBe(false);
    expect(isValidHfRepoId('org/-repo')).toBe(false);
  });

  it('rejects a value longer than the Hub allows', () => {
    const half = 'a'.repeat(MAX_HF_REPO_ID_CHARS);
    expect(isValidHfRepoId(`${half}/${half}`)).toBe(false);
  });

  it('is case-preserving rather than case-insensitive, so both spellings validate', () => {
    // The column stores what the Hub calls it. `gemma-4-E2B` and `gemma-4-e2b` are different repos
    // as far as this validator is concerned, and the unique index treats them as different rows.
    expect(isValidHfRepoId('google/gemma-4-E2B')).toBe(true);
    expect(isValidHfRepoId('google/gemma-4-e2b')).toBe(true);
  });
});
