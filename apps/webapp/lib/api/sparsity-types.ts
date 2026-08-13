/**
 * Named aliases for the sparsity wire types the webapp uses.
 *
 * `sparsity.d.ts` is generated from `apps/sparsity/openapi.json` and reaches everything through
 * `components['schemas'][...]`, which is unreadable at a call site.
 *
 * Field names here are snake_case, and that is deliberate rather than an oversight: unlike
 * inference and autointerp, this server's responses are forwarded nearly verbatim by the
 * documented public route `/api/sparsity/connected-neurons`, so its names are already a public
 * contract. See `apps/sparsity/schemas.py` and the note in AGENTS.md.
 *
 * `TraceNode` in particular used to be hand-copied into two files that agreed with the python
 * only by convention, so a rename compiled fine and broke at runtime.
 */
import type { components } from '@/lib/api/sparsity';

type Schemas = components['schemas'];

export type TraceNode = Schemas['TraceNode'];
export type NeuronConnectionsResponse = Schemas['NeuronConnectionsResponse'];
export type ChannelConnectionsResponse = Schemas['ChannelConnectionsResponse'];
export type ChannelNeuron = Schemas['ChannelNeuron'];
