import OpenAI from 'openai';
import pgvector from 'pgvector';

const VALID_EMBEDDING_MODELS = ['text-embedding-3-large'];

export async function getOAIEmbedding(embeddingModel: string, dimensions: number, text: string | string[]) {
  if (!VALID_EMBEDDING_MODELS.includes(embeddingModel)) {
    throw new Error('Invalid embedding model');
  }
  // if text is empty string or array, return an empty array
  if (Array.isArray(text) && text.length === 0) {
    return [];
  }
  const openai = new OpenAI();
  // TODO: this fails silently when the key is invalid
  const response = await openai.embeddings.create({
    input: text,
    model: embeddingModel,
    dimensions,
  });
  const queryEmbeddingResult = response.data.map((v: any) => v.embedding);
  if (queryEmbeddingResult.length === 0) {
    throw new Error('No embedding result found');
  }
  // if we have an array of strings, return an array of arrays
  if (Array.isArray(text)) {
    return queryEmbeddingResult.map((v: number[]) => v as number[]);
  }
  // otherwise just return the first one
  return queryEmbeddingResult[0] as number[];
}

export async function getExplanationEmbeddingSql(explanation: string) {
  return pgvector.toSql(await getOAIEmbedding('text-embedding-3-large', 256, explanation));
}

export async function getExplanationEmbedding(explanation: string) {
  return getOAIEmbedding('text-embedding-3-large', 256, explanation);
}
