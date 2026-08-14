/**
 * Named aliases for the graph wire types the webapp uses.
 *
 * `graph.d.ts` is generated from `apps/graph/openapi.json` and reaches everything through
 * `components['schemas'][...]`, which is unreadable at a call site.
 *
 * Field names here are snake_case, and that is deliberate rather than an oversight: unlike
 * inference and autointerp, these shapes are public in three directions. `/api/graph/tokenize`
 * and `/api/steer-logits` forward responses nearly verbatim with snake_case in their swagger,
 * and `/api/steer-logits` publishes `SteerFeature`'s own field names because it forwards
 * `features` untouched. See `apps/graph/neuronpedia_graph/schemas.py` and the note in AGENTS.md.
 *
 * Not covered here: the graph JSON uploaded to S3. Its keys come from `circuit_tracer` and are
 * pinned by the published `graph-schema.json`, so they live in `graph-types.ts` under
 * `app/[modelId]/graph/` rather than being generated from our models.
 */
import type { components } from '@/lib/api/graph';

type Schemas = components['schemas'];

export type GraphChatMessage = Schemas['GraphChatMessage'];
export type SalientLogit = Schemas['SalientLogit'];
export type ForwardPassResponse = Schemas['ForwardPassResponse'];
export type ParseChatPromptResponse = Schemas['ParseChatPromptResponse'];
export type SteerFeature = Schemas['SteerFeature'];
export type SteerResponse = Schemas['SteerResponse'];
export type LogitsByToken = Schemas['LogitsByToken'];
export type TopLogit = Schemas['TopLogit'];
export type GraphGenerationResponse = Schemas['GraphGenerationResponse'];
