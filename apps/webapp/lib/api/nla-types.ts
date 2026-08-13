/**
 * Named aliases for the NLA wire types the webapp uses.
 *
 * `nla.d.ts` is generated from `apps/nla/openapi.json` and reaches everything through
 * `components['schemas'][...]`, which is unreadable at a call site.
 *
 * Field names here are snake_case, and that is deliberate rather than an oversight: unlike
 * inference and autointerp, these names are persisted and published. `/api/nla/explain` writes
 * `ExplainResult` records verbatim into `NlaExplainCache.resultJson`, and those rows back
 * permanent `/nla/[shareId]` URLs, so a rename would strand every existing share. See the note
 * in AGENTS.md and `apps/nla/tests/test_frame_contract.py`.
 *
 * Not covered here: the SSE frames from `/completion`, `/describe` and `/explain`. They are not
 * response bodies, so no spec can describe them; they are pinned on the python side by
 * `apps/nla/tests/test_frame_contract.py` and typed by hand in `app/[modelId]/nla/nla-types.ts`.
 */
import type { components } from '@/lib/api/nla';

type Schemas = components['schemas'];

export type TokenInfo = Schemas['TokenInfo'];
export type TokenizeResponse = Schemas['TokenizeResponse'];
export type ExplainResult = Schemas['ExplainResult'];
export type ExplainResponse = Schemas['ExplainResponse'];
export type DescriptionResult = Schemas['DescriptionResult'];
export type DescribeResponse = Schemas['DescribeResponse'];
export type ScoreResponse = Schemas['ScoreResponse'];
export type ChatMessageInput = Schemas['ChatMessageInput'];
