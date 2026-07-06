// Shared constants + types for jlens (Jacobian Lens) share links.
//
// A "share" stores the heavy run data (messages, meta, and the full unfiltered
// per-position token stream) gzipped in S3, and the small UI-restore state in
// the `JlensShare` DB row. Loading a share fetches the DB row (server-side) for
// the S3 url + UI state, then the browser downloads + decompresses the S3 blob.

import { LensType } from '@/lib/utils/lens';

// We reuse the existing graph S3 bucket, under a `jlens/` subdirectory.
export const JLENS_S3_DIR = 'jlens';
// Anonymous (logged-out) uploads go under `jlens/anonymous/...`.
export const JLENS_ANONYMOUS_USER_ID = 'anonymous';

// Hard cap on the UNCOMPRESSED share JSON. Enforced before gzip + upload.
export const MAX_JLENS_SHARE_UPLOAD_SIZE_BYTES = 100 * 1024 * 1024;

// Per-IP/day cap on share creations (mirrors the graph put-request limit).
export const MAX_JLENS_SHARE_PUT_REQUESTS_PER_DAY = 200;

// Max length of the optional user-supplied share description.
export const MAX_JLENS_SHARE_DESCRIPTION_LENGTH = 512;

// A locked sidebar predicted-token selection. Keyed by the decoded token STRING
// (the read-out exposes predicted tokens as strings, not ids), plus its lens
// type. Position-independent, so robust across re-runs.
export interface JlensShareLockedToken {
  key: string;
  type: LensType;
}

// The steer configuration saved with a share: the exact decoded token string
// being steered, its lens type, the model layers the steer was injected at, and
// the signed strength. The heavy steered token stream + meta live in the S3
// blob; this small config is also mirrored to `JlensShare` DB columns and is
// used to re-run the steered read-out (server-side) and re-enter steer mode.
export interface JlensShareSteer {
  token: string;
  type: LensType;
  layers: number[];
  strength: number;
  // When true, the steer ablates (projects out) the readout direction instead
  // of additively steering — mutually exclusive with strength.
  ablate: boolean;
  // Intervention mode: 'steer' (add/suppress) or 'swap' (replace source readout
  // with `swapToken`). Optional for backward-compat with pre-swap shares.
  mode?: 'steer' | 'swap';
  // The free-typed target token for a swap (e.g. " Rugby"). Empty/absent for
  // plain steers.
  swapToken?: string;
  // Whether the intervention was applied to generated tokens too (default
  // false). Optional for backward-compat with older shares.
  steerGenerated?: boolean;
}

// Small UI state restored when opening a shared link. Stored in DB columns
// (not the S3 blob) so the page can rehydrate without parsing the blob.
export interface JlensShareUiState {
  lockedTokens: JlensShareLockedToken[];
  selectedPositions: number[];
  // 'JACOBIAN_LENS' | 'LOGIT_LENS' | 'DIFF'
  activeLensModeTab: string;
  topN: number;
  hideNonWordTokens: boolean;
  // Run settings restored on load (also drive any new steer the viewer runs).
  temperature: number;
  numCompletionTokens: number;
  // Number of leading prompt tokens (the rest were generated), so the loaded
  // run can mark the prompt→generated boundary. Null for older shares that
  // didn't persist it — in which case the boundary is unknown and not shown.
  numPromptTokens: number | null;
}

// Build the public-facing share URL for a given share id.
export function makeJlensSharePath(sharedId: string) {
  return `/jlens/${sharedId}`;
}
