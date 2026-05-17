import { prisma } from '@/lib/db';
import { nlaFetch } from '@/lib/db/nla-source';
import { getOAIEmbedding } from '@/lib/external/embedding';
import { getNlaSaeFormat } from '@/lib/utils/nla';
import { fetchDecoderLatent, getSAEFormatConfig, SAEFormat } from '@/lib/utils/saelens';
import { getLayerNumFromSource } from '@/lib/utils/source';
import { Explanation } from '@prisma/client';
import { AuthenticatedUser } from '../with-user';

const EMBEDDING_MODEL = 'text-embedding-3-large';
const EMBEDDING_DIMENSIONS = 256;

/**
 * Loads everything the NLA scorers need to talk to the right NLA server
 * about the right activation:
 *   - the W_dec row (one decoder latent), pulled with format-aware
 *     filename + tensor key
 *   - the matching `NlaSource` row, used to resolve which GPU server
 *     cluster owns this (model, layer)
 *
 * SAE format inference mirrors the client: we consult the
 * (model, layer) -> SAEFormat allowlist in `lib/utils/nla.ts`. Layers
 * not in the allowlist default to SAELens for backward compatibility
 * with legacy SAE checkpoints.
 */
async function fetchLatentForExplanation(explanation: Explanation) {
  const source = await prisma.source.findUnique({
    where: {
      modelId_id: {
        modelId: explanation.modelId,
        id: explanation.layer,
      },
    },
  });
  if (!source?.hfRepoId || !source?.hfFolderId) {
    throw new Error('Source does not have HuggingFace repo/folder configured');
  }

  const index = parseInt(explanation.index, 10);
  if (Number.isNaN(index) || index < 0) {
    throw new Error(`Invalid feature index: ${explanation.index}`);
  }

  const layerNum = getLayerNumFromSource(explanation.layer);
  const saeFormat = getNlaSaeFormat(explanation.modelId, layerNum) ?? SAEFormat.SAELens;
  const { weightsFilename } = getSAEFormatConfig(saeFormat);
  const safetensorsPath = `${source.hfFolderId}/${weightsFilename}`;
  const latent = await fetchDecoderLatent(source.hfRepoId, safetensorsPath, index, saeFormat);

  // Resolve the NlaSource row that owns this (model, layer). The scorer
  // can't pass an explicit nlaSourceId (no UI surface for it today), so
  // we do the same (modelId, layerNum) lookup that the explain-saelens
  // route does. Stable ordering by id so repeat scorings of the same
  // (modelId, layerNum) hit the same row when multiple match (rare;
  // differs only by ar/av).
  const nlaSource = await prisma.nlaSource.findFirst({
    where: { modelId: explanation.modelId, layerNum },
    orderBy: { id: 'asc' },
  });
  if (!nlaSource) {
    throw new Error(
      `No NLA source configured for modelId="${explanation.modelId}", layerNum=${layerNum}. ` +
        `Configure an NlaSource for this (model, layer) pair before running NLA scorers.`,
    );
  }

  return { latent, source, index, saeFormat, layerNum, nlaSourceId: nlaSource.id };
}

function cosineSimilarity(a: number[], b: number[]): number {
  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < a.length; i += 1) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  const denom = Math.sqrt(normA) * Math.sqrt(normB);
  return denom === 0 ? 0 : dot / denom;
}

// ─── nla_reconstructor ──────────────────────────────────────────────────────

export const generateScoreNlaReconstructor = async (explanation: Explanation, user: AuthenticatedUser) => {
  const { latent, source, index, nlaSourceId } = await fetchLatentForExplanation(explanation);

  // `nlaFetch` (vs the legacy single-shot `getNlaServerUrl()`) gives us
  // shuffle + failover across the NlaSource row's `servers[]`, plus
  // automatic auth-header injection. The request-level fail-fast
  // semaphores on the NLA server (`NLA_RECONSTRUCTOR_MAX_CONCURRENT`)
  // mean we may see 429s under load — those bubble up as scorer errors
  // for the user to retry, which is the desired contract.
  const nlaResponse = await nlaFetch(explanation.modelId, nlaSourceId, '/score', {
    method: 'POST',
    body: JSON.stringify({
      description: explanation.description,
      activation: latent.values,
    }),
  });

  if (!nlaResponse.ok) {
    const errorText = await nlaResponse.text();
    throw new Error(`NLA server error: ${nlaResponse.status} - ${errorText}`);
  }

  const nlaResult = (await nlaResponse.json()) as { mse: number; cosine_similarity: number };

  const jsonDetails = {
    mse: nlaResult.mse,
    cosine_similarity: nlaResult.cosine_similarity,
    hfRepoId: source.hfRepoId,
    hfFolderId: source.hfFolderId,
    index,
  };

  return prisma.explanationScore.create({
    data: {
      value: nlaResult.cosine_similarity,
      explanationId: explanation.id,
      explanationScoreTypeName: 'nla_reconstructor',
      explanationScoreModelName: `nla_reconstructor`,
      initiatedByUserId: user.id,
      jsonDetails: JSON.stringify(jsonDetails),
    },
  });
};

// ─── nla_verbalizer / nla_verbalizer_last ───────────────────────────────────

async function generateScoreNlaVerbalizerInner(
  explanation: Explanation,
  user: AuthenticatedUser,
  useLastParagraph: boolean,
) {
  const scorerType = useLastParagraph ? 'nla_verbalizer_last' : 'nla_verbalizer';
  const { latent, source, index, nlaSourceId } = await fetchLatentForExplanation(explanation);

  // 1. Get NLA explanation via /describe (non-streaming). Same nlaFetch
  // semantics as nla_reconstructor above — shuffle + failover across the
  // (model, layer) NlaSource's `servers[]`.
  const describeResponse = await nlaFetch(explanation.modelId, nlaSourceId, '/describe', {
    method: 'POST',
    body: JSON.stringify({
      activations: [latent.values],
      temperature: 0.7,
      max_new_tokens: 200,
      stream: false,
    }),
  });

  if (!describeResponse.ok) {
    const errorText = await describeResponse.text();
    throw new Error(`NLA server error: ${describeResponse.status} - ${errorText}`);
  }

  const describeResult = (await describeResponse.json()) as {
    results: { description: string; mse?: number; cosine_similarity?: number }[];
  };
  const fullNlaExplanation = describeResult.results[0]?.description;
  if (!fullNlaExplanation) {
    throw new Error('NLA server returned no description');
  }

  // 2. Determine the NLA text to compare
  let nlaText = fullNlaExplanation;
  if (useLastParagraph) {
    const paragraphs = fullNlaExplanation.split('\n').filter((p) => p.trim().length > 0);
    nlaText = paragraphs[paragraphs.length - 1] || fullNlaExplanation;
  }

  // 3. Get embeddings for both texts in one call
  const embeddings = (await getOAIEmbedding(EMBEDDING_MODEL, EMBEDDING_DIMENSIONS, [
    nlaText,
    explanation.description,
  ])) as number[][];

  const nlaEmbedding = embeddings[0];
  const explanationEmbedding = embeddings[1];

  // 4. Compute cosine similarity
  const cossim = cosineSimilarity(nlaEmbedding, explanationEmbedding);

  const jsonDetails = {
    nla_explanation: fullNlaExplanation,
    nla_text_used: nlaText,
    original_explanation: explanation.description,
    nla_embedding: nlaEmbedding,
    explanation_embedding: explanationEmbedding,
    embedding_model: EMBEDDING_MODEL,
    embedding_dimensions: EMBEDDING_DIMENSIONS,
    cosine_similarity: cossim,
    hfRepoId: source.hfRepoId,
    hfFolderId: source.hfFolderId,
    index,
  };

  return prisma.explanationScore.create({
    data: {
      value: cossim,
      explanationId: explanation.id,
      explanationScoreTypeName: scorerType,
      explanationScoreModelName: scorerType,
      initiatedByUserId: user.id,
      jsonDetails: JSON.stringify(jsonDetails),
    },
  });
}

export const generateScoreNlaVerbalizer = async (explanation: Explanation, user: AuthenticatedUser) =>
  generateScoreNlaVerbalizerInner(explanation, user, false);

export const generateScoreNlaVerbalizerLast = async (explanation: Explanation, user: AuthenticatedUser) =>
  generateScoreNlaVerbalizerInner(explanation, user, true);
