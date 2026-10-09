'use client';

// The oracle lens: a LoRA that describes the activation at one position in
// words, at each oracle layer (`/api/lens/oracle`). A read starts only when
// the user asks for it in the token popup. Reads are greedy, so a read depends
// only on the token prefix, which keys the client cache.

import { LensMetaMessage, LensOracleReadMessage, LensTokenMessage, ORACLE_BULLETS } from '@/lib/utils/lens';
import {
  createContext,
  ReactNode,
  UIEvent,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
  useSyncExternalStore,
} from 'react';
import { JlensExportOracle } from './jlens-export';
import { OracleUnavailableError, runOracleStream } from './jlens-stream';
import { LayerRange, visibleLayerIndices } from './jlens-token-popup';

// The background sweep. Layers not served are left out (every other served
// layer if fewer than two are left). A chunk of 12 x 5 reads fits in one batch on an A100.
export const ORACLE_SWEEP_LAYERS = [20, 36, 44, 52, 60];
export const ORACLE_SWEEP_POSITIONS_PER_REQUEST = 12;
export const ORACLE_SWEEP_MAX_POSITIONS = 1024;

const MAX_CACHED_READS = 4 * ORACLE_SWEEP_MAX_POSITIONS;

export type OracleEntry = {
  status: 'loading' | 'done' | 'error';
  position: number;
  adapter: string;
  reads: Map<number, LensOracleReadMessage>;
  // Each layer's text so far, until its read ends.
  partials: Map<number, string>;
  // True when all served layers were asked for (or the reads came from a
  // share); false when only the sweep layers were.
  allLayers: boolean;
  error?: string;
};

// A position the sweep has not read yet, is reading, or has read.
export type OracleSweepStatus = 'queued' | 'loading' | 'done';

// Sweep status per position. Each token chip listens to its own position only,
// so a status change draws one chip again, not the transcript.
export class OracleSweepStore {
  private status = new Map<number, OracleSweepStatus>();

  private listeners = new Map<number, Set<() => void>>();

  get(position: number): OracleSweepStatus | null {
    return this.status.get(position) ?? null;
  }

  set(position: number, status: OracleSweepStatus | null) {
    if (this.get(position) === status) {
      return;
    }
    if (status) {
      this.status.set(position, status);
    } else {
      this.status.delete(position);
    }
    this.listeners.get(position)?.forEach((l) => l());
  }

  clear() {
    const positions = Array.from(this.status.keys());
    this.status.clear();
    positions.forEach((p) => this.listeners.get(p)?.forEach((l) => l()));
  }

  subscribe(position: number, listener: () => void): () => void {
    let set = this.listeners.get(position);
    if (!set) {
      set = new Set();
      this.listeners.set(position, set);
    }
    set.add(listener);
    return () => {
      set!.delete(listener);
      if (set!.size === 0) {
        this.listeners.delete(position);
      }
    };
  }
}

// A stable value, so a chip draws again only when its own status changes.
export const OracleSweepContext = createContext<OracleSweepStore | null>(null);

const NO_SWEEP = () => () => {};

export function useOracleSweepStatus(position: number): OracleSweepStatus | null {
  const store = useContext(OracleSweepContext);
  const subscribe = useCallback(
    (listener: () => void) => (store ? store.subscribe(position, listener) : NO_SWEEP()),
    [store, position],
  );
  return useSyncExternalStore(
    subscribe,
    () => store?.get(position) ?? null,
    () => null,
  );
}

// One key per prefix: two 32-bit hashes of the token ids 0..i, for each i.
// Tokens after a gap in the positions get no key.
function prefixHashes(tokens: LensTokenMessage[]): string[] {
  const out: string[] = [];
  let a = 0x811c9dc5;
  let b = 0x9e3779b9;
  for (const t of tokens) {
    if (t.position !== out.length) {
      break;
    }
    a = Math.imul(a ^ t.id, 0x01000193) >>> 0;
    b = Math.imul(b ^ (t.id + 0x632be5ab), 0x85ebca6b) >>> 0;
    b = (b ^ (b >>> 13)) >>> 0;
    out.push(`${a.toString(36)}.${b.toString(36)}`);
  }
  return out;
}

function oracleKey(modelId: string, position: number, hash: string): string {
  return `${modelId}|${position}|${hash}`;
}

function newEntry(position: number): OracleEntry {
  return { status: 'loading', position, adapter: '', reads: new Map(), partials: new Map(), allLayers: false };
}

// True when `entry` has a read at every layer of `layers`.
function hasLayers(entry: OracleEntry | undefined, layers: number[]): boolean {
  return !!entry && layers.length > 0 && layers.every((l) => entry.reads.has(l));
}

// `entry` with `read` in it. A read already kept stays: it may be a share's.
function withRead(entry: OracleEntry, read: LensOracleReadMessage): OracleEntry {
  const partials = new Map(entry.partials);
  partials.delete(read.layer);
  if (entry.reads.has(read.layer)) {
    return { ...entry, partials };
  }
  return { ...entry, reads: new Map(entry.reads).set(read.layer, read), partials };
}

// Resolves when the page is visible, or on abort.
function whenVisible(signal: AbortSignal): Promise<void> {
  if (typeof document === 'undefined' || !document.hidden) {
    return Promise.resolve();
  }
  return new Promise((resolve) => {
    const done = () => {
      document.removeEventListener('visibilitychange', check);
      signal.removeEventListener('abort', done);
      resolve();
    };
    const check = () => {
      if (!document.hidden) {
        done();
      }
    };
    document.addEventListener('visibilitychange', check);
    signal.addEventListener('abort', done);
  });
}

const PARTIAL_BULLET = /^\s*[-*]\s+(.*\S)/;

// The bullets of a read's text so far, the last one maybe cut.
function partialBullets(text: string | undefined): string[] {
  if (!text) {
    return [];
  }
  const bullets: string[] = [];
  for (const line of text.split('\n')) {
    const m = PARTIAL_BULLET.exec(line);
    if (m) {
      bullets.push(m[1]);
    }
  }
  return bullets.slice(0, ORACLE_BULLETS);
}

// A layer's bullets: its read's, else those of its text so far.
function layerBullets(entry: OracleEntry | null, layer: number): string[] | null {
  const read = entry?.reads.get(layer);
  if (read) {
    return read.bullets;
  }
  const partial = partialBullets(entry?.partials.get(layer));
  return partial.length > 0 ? partial : null;
}

export type OracleState = {
  modelId: string;
  // True when the servers can read, may read (a loaded run, before its first
  // read), or the run has stored reads.
  available: boolean;
  layers: number[];
  // A lens run is streaming, so a request does nothing.
  busy: boolean;
  entryFor: (position: number) => OracleEntry | null;
  request: (position: number) => void;
  // Set when a read found that the servers have no oracle after all.
  unavailableReason: string | null;
  // The signed reads of this run, for a share.
  sharedReads: () => JlensExportOracle | undefined;
  // Call when a share or file loads: puts its reads (if any) into the cache.
  // `tokens` and `meta` are the loaded run's.
  seed: (oracle: JlensExportOracle | undefined, tokens: LensTokenMessage[], meta: LensMetaMessage | null) => void;
  // True while the oracle column shows: the sweep runs only then.
  setSweepActive: (active: boolean) => void;
  sweepStore: OracleSweepStore;
};

export const OracleContext = createContext<OracleState | null>(null);

// Who trained each model's Oracle Lens adapter, shown under the load button.
const ORACLE_CREDITS: Record<string, { label: string; href: string }> = {
  'qwen3.6-27b': {
    label: 'Bhatia, Blank et al.',
    href: 'https://huggingface.co/agu18dec/olens_and_ar/tree/main/olens_s3d_rl600',
  },
};

type Sweep = {
  // Positions 0..count-1 are the sweep's.
  count: number;
  queue: number[];
  chunk: Set<number>;
};

export function useOracleReads({
  modelId,
  tokens,
  meta,
  enabled,
  busy,
}: {
  modelId: string;
  tokens: LensTokenMessage[];
  meta: LensMetaMessage | null;
  // False for runs the oracle cannot read, e.g. a steered run: a read uses the
  // unsteered activations.
  enabled: boolean;
  // A lens run is streaming. Reads wait, so they do not queue behind it.
  busy: boolean;
}): OracleState {
  const [unavailableReason, setUnavailableReason] = useState<string | null>(null);
  // Layers of a share's stored reads, shown when the servers read none.
  const [seededLayers, setSeededLayers] = useState<number[]>([]);
  // A loaded run's meta tells which layers its servers read then, not now.
  const [loadedMeta, setLoadedMeta] = useState<LensMetaMessage | null>(null);
  // The layers of the last reply to a read of all layers.
  const [repliedLayers, setRepliedLayers] = useState<number[] | null>(null);
  const isLoadedMeta = meta != null && meta === loadedMeta;
  // Null: not known until a read replies.
  const servedLayers = useMemo((): number[] | null => {
    if (!enabled || unavailableReason) {
      return [];
    }
    return isLoadedMeta ? repliedLayers : (meta?.oracle_layers ?? []);
  }, [enabled, unavailableReason, isLoadedMeta, repliedLayers, meta]);
  const canRead = servedLayers === null || servedLayers.length > 0;
  const layers = useMemo(() => {
    if (!enabled) {
      return [];
    }
    if (servedLayers && servedLayers.length > 0) {
      return servedLayers;
    }
    if (seededLayers.length > 0 || servedLayers) {
      return seededLayers;
    }
    return meta?.oracle_layers ?? [];
  }, [enabled, servedLayers, seededLayers, meta]);
  // Empty: the sweep reads all layers (the served layers are not known yet).
  const sweepLayers = useMemo(() => {
    const base = servedLayers ?? meta?.oracle_layers ?? [];
    const chosen = ORACLE_SWEEP_LAYERS.filter((l) => base.includes(l));
    return chosen.length >= 2 ? chosen : base.filter((_, i) => i % 2 === 0);
  }, [servedLayers, meta]);
  const sweepLayersKey = sweepLayers.join(',');
  const sweepLayersRef = useRef(sweepLayers);
  sweepLayersRef.current = sweepLayers;
  const cacheRef = useRef(new Map<string, OracleEntry>());
  const inflightRef = useRef<{ key: string; controller: AbortController } | null>(null);
  const [version, setVersion] = useState(0);
  const bump = useCallback(() => setVersion((v) => v + 1), []);
  const frameRef = useRef<number | null>(null);
  const bumpSoon = useCallback(() => {
    if (frameRef.current == null) {
      frameRef.current = requestAnimationFrame(() => {
        frameRef.current = null;
        bump();
      });
    }
  }, [bump]);
  const tokensRef = useRef(tokens);
  tokensRef.current = tokens;
  const hashes = useMemo(() => prefixHashes(tokens), [tokens]);
  const hashesRef = useRef(hashes);
  hashesRef.current = hashes;
  // The last prefix key names the whole sequence.
  const sequenceKey = hashes.length > 0 ? `${hashes.length}|${hashes[hashes.length - 1]}` : '';
  const [sweepActive, setSweepActive] = useState(false);
  const sweepStore = useMemo(() => new OracleSweepStore(), []);
  const sweepRef = useRef<Sweep | null>(null);
  // Changes when a sweep starts or ends, so hovers can read again.
  const [sweeping, setSweeping] = useState(false);

  // Writes `change(entry)` (null: removes the entry) under `key`.
  const update = useCallback(
    (key: string, position: number, change: (entry: OracleEntry) => OracleEntry | null, soon = false) => {
      const cache = cacheRef.current;
      const next = change(cache.get(key) ?? newEntry(position));
      cache.delete(key);
      if (next) {
        cache.set(key, next);
        while (cache.size > MAX_CACHED_READS) {
          const oldest = cache.keys().next().value;
          if (oldest === undefined) {
            break;
          }
          cache.delete(oldest);
        }
      }
      if (soon) {
        bumpSoon();
      } else {
        bump();
      }
    },
    [bump, bumpSoon],
  );

  // After a read that did not end: keep the layers that did.
  const settle = useCallback(
    (key: string, position: number) =>
      update(key, position, (e) => {
        if (e.status !== 'loading') {
          return e;
        }
        return e.reads.size > 0 ? { ...e, status: 'done', partials: new Map() } : null;
      }),
    [update],
  );

  // A new model has other activations: drop the cache and any read in flight.
  useEffect(() => {
    cacheRef.current.clear();
    inflightRef.current?.controller.abort();
    inflightRef.current = null;
    setUnavailableReason(null);
    setSeededLayers([]);
    setLoadedMeta(null);
    setRepliedLayers(null);
    bump();
  }, [modelId, bump]);
  useEffect(
    () => () => {
      inflightRef.current?.controller.abort();
      if (frameRef.current != null) {
        cancelAnimationFrame(frameRef.current);
      }
    },
    [],
  );

  const keyFor = useCallback(
    (position: number): string | null =>
      position < hashes.length ? oracleKey(modelId, position, hashes[position]) : null,
    [modelId, hashes],
  );

  // The sweep: reads positions 0..count-1 at the sweep layers, one chunk at a
  // time, in order (a hovered position goes first). It starts again on a new
  // run or layers, and skips positions read before.
  const sweepOn = sweepActive && enabled && canRead && !busy && sequenceKey !== '';
  useEffect(() => {
    if (!sweepOn) {
      return undefined;
    }
    const ids = tokensRef.current.slice(0, hashesRef.current.length).map((t) => t.id);
    const count = Math.min(ids.length, ORACLE_SWEEP_MAX_POSITIONS);
    const hashAt = hashesRef.current;
    const layersToRead = sweepLayersRef.current;
    const keyAt = (p: number) => oracleKey(modelId, p, hashAt[p]);
    const swept = (p: number) => {
      const e = cacheRef.current.get(keyAt(p));
      return layersToRead.length > 0 ? hasLayers(e, layersToRead) : e?.status === 'done';
    };
    const queue: number[] = [];
    for (let p = 0; p < count; p += 1) {
      if (swept(p)) {
        sweepStore.set(p, 'done');
      } else {
        queue.push(p);
      }
    }
    if (queue.length === 0) {
      return () => sweepStore.clear();
    }
    const sweep: Sweep = { count, queue, chunk: new Set() };
    sweepRef.current = sweep;
    queue.forEach((p) => sweepStore.set(p, 'queued'));
    setSweeping(true);
    bump();
    const controller = new AbortController();
    const { signal } = controller;

    const readChunk = async (chunk: number[]) => {
      // False when the server read only the first position.
      let readsAll = true;
      await runOracleStream({
        modelId,
        tokenIds: ids.slice(0, Math.max(...chunk) + 1),
        position: chunk[0],
        positions: chunk.slice(1),
        layers: layersToRead,
        partial: false,
        signal,
        onMeta: (m) => {
          readsAll = chunk.length === 1 || m.positions != null;
          if (layersToRead.length === 0) {
            setRepliedLayers(m.layers);
          }
          chunk.forEach((p) => update(keyAt(p), p, (e) => ({ ...e, adapter: m.adapter }), true));
        },
        onRead: (read) => {
          const p = read.position ?? chunk[0];
          if (sweep.chunk.has(p)) {
            update(keyAt(p), p, (e) => withRead(e, read), true);
          }
        },
      });
      return readsAll;
    };

    const run = async () => {
      let failures = 0;
      while (sweep.queue.length > 0 && !signal.aborted) {
        // One chunk at a time, in order.
        // eslint-disable-next-line no-await-in-loop
        await whenVisible(signal);
        if (signal.aborted) {
          break;
        }
        const chunk = sweep.queue.splice(0, ORACLE_SWEEP_POSITIONS_PER_REQUEST).filter((p) => !swept(p));
        if (chunk.length === 0) {
          continue;
        }
        chunk.forEach((p) => {
          sweep.chunk.add(p);
          sweepStore.set(p, 'loading');
          update(keyAt(p), p, (e) => ({ ...e, status: 'loading', error: undefined }), true);
        });
        let readsAll = true;
        let error: string | null = null;
        try {
          // eslint-disable-next-line no-await-in-loop
          readsAll = await readChunk(chunk);
          failures = 0;
        } catch (err) {
          if (signal.aborted) {
            break;
          }
          if (err instanceof OracleUnavailableError) {
            setUnavailableReason(err.message);
            break;
          }
          failures += 1;
          error = err instanceof Error ? err.message : String(err);
        }
        const retry = error != null && failures === 1;
        chunk.forEach((p) => {
          sweep.chunk.delete(p);
          if (error != null && !retry) {
            update(keyAt(p), p, (e) => ({ ...e, status: 'error', error: error! }));
          } else {
            settle(keyAt(p), p);
          }
          sweepStore.set(p, retry ? 'queued' : swept(p) ? 'done' : null);
        });
        if (retry) {
          sweep.queue.unshift(...chunk);
        } else if (error != null) {
          failures = 0;
        }
        // A server that reads one position per request: leave the rest to hovers.
        if (!readsAll) {
          break;
        }
      }
      if (!signal.aborted) {
        sweep.queue.forEach((p) => sweepStore.set(p, null));
        sweepRef.current = null;
        setSweeping(false);
      }
    };
    run();

    return () => {
      controller.abort();
      sweep.chunk.forEach((p) => settle(keyAt(p), p));
      sweepStore.clear();
      sweepRef.current = null;
      setSweeping(false);
    };
    // `sequenceKey` and `sweepLayersKey` stand for the tokens and layers read through refs.
  }, [sweepOn, sequenceKey, sweepLayersKey, modelId, sweepStore, update, settle, bump]);

  const entryFor = useCallback(
    (position: number) => {
      const key = keyFor(position);
      const entry = key ? cacheRef.current.get(key) : undefined;
      const status = sweepStore.get(position);
      if (!entry && (status === 'queued' || status === 'loading')) {
        return newEntry(position);
      }
      return entry ?? null;
    },
    // `version` makes a new function (and so a new context value) per update.
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [keyFor, sweepStore, version],
  );

  // Reads the layers `position` does not have yet. While the sweep runs, a
  // position it has not read goes to the front of its queue instead.
  const request = useCallback(
    (position: number) => {
      if (!canRead || busy) {
        return;
      }
      const sweep = sweepRef.current;
      if (sweep && position < sweep.count) {
        const at = sweep.queue.indexOf(position);
        if (at > 0) {
          sweep.queue.splice(at, 1);
          sweep.queue.unshift(position);
        }
        return;
      }
      const key = keyFor(position);
      if (!key) {
        return;
      }
      const cached = cacheRef.current.get(key);
      // Empty reads every layer the server serves.
      const wanted = servedLayers ? servedLayers.filter((l) => !cached?.reads.has(l)) : [];
      if (cached && cached.status === 'done' && (servedLayers ? wanted.length === 0 : cached.allLayers)) {
        return;
      }
      if (inflightRef.current) {
        if (inflightRef.current.key === key) {
          return;
        }
        // One read at a time: the newest request wins.
        inflightRef.current.controller.abort();
      }
      const controller = new AbortController();
      inflightRef.current = { key, controller };
      update(key, position, (e) => ({ ...e, status: 'loading', error: undefined }));
      runOracleStream({
        modelId,
        tokenIds: tokensRef.current.slice(0, position + 1).map((t) => t.id),
        position,
        layers: wanted,
        signal: controller.signal,
        onMeta: (m) => {
          if (wanted.length === 0) {
            setRepliedLayers(m.layers);
          }
          update(key, position, (e) => ({ ...e, adapter: m.adapter }));
        },
        // `soon`: one render per frame, since partials come once per token per layer.
        onPartial: (p) =>
          update(key, position, (e) => ({ ...e, partials: new Map(e.partials).set(p.layer, p.text) }), true),
        onRead: (read) => update(key, position, (e) => withRead(e, read)),
      })
        .then(() => update(key, position, (e) => ({ ...e, status: 'done', allLayers: true })))
        .catch((err: unknown) => {
          if (controller.signal.aborted) {
            // An aborted read is not an answer; the next request starts again
            // (the server's cache keeps the layers that ended).
            settle(key, position);
            return;
          }
          if (err instanceof OracleUnavailableError) {
            setUnavailableReason(err.message);
          }
          const error = err instanceof Error ? err.message : String(err);
          update(key, position, (e) => ({ ...e, status: 'error', error }));
        })
        .finally(() => {
          if (inflightRef.current?.controller === controller) {
            inflightRef.current = null;
          }
        });
    },
    // `busy` and `sweeping` are dependencies so the auto-read effects run again when they change.
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [canRead, servedLayers, busy, sweeping, keyFor, modelId, update, settle],
  );

  const sharedReads = useCallback((): JlensExportOracle | undefined => {
    const reads: JlensExportOracle['reads'] = [];
    for (const [key, entry] of cacheRef.current) {
      // Only reads of this run's tokens.
      if (key !== keyFor(entry.position)) {
        continue;
      }
      for (const read of entry.reads.values()) {
        if (read.sig) {
          reads.push({
            position: entry.position,
            layer: read.layer,
            adapter: entry.adapter,
            bullets: read.bullets,
            text: read.text,
            finish: read.finish,
            sig: read.sig,
          });
        }
      }
    }
    return reads.length > 0 ? { maxBullets: ORACLE_BULLETS, reads } : undefined;
  }, [keyFor]);

  const seed = useCallback(
    (oracle: JlensExportOracle | undefined, runTokens: LensTokenMessage[], runMeta: LensMetaMessage | null) => {
      setLoadedMeta(runMeta);
      setRepliedLayers(null);
      setUnavailableReason(null);
      setSeededLayers([]);
      if (!oracle) {
        return;
      }
      // A read stored at another bullet cap shows its first bullet. Its
      // signature is for that cap, so a share does not store it again.
      const sameCap = oracle.maxBullets === ORACLE_BULLETS;
      const runHashes = prefixHashes(runTokens);
      const byPosition = new Map<number, OracleEntry>();
      for (const r of oracle.reads) {
        if (r.position >= runHashes.length) {
          continue;
        }
        let entry = byPosition.get(r.position);
        if (!entry) {
          entry = { ...newEntry(r.position), status: 'done', adapter: r.adapter, allLayers: true };
          byPosition.set(r.position, entry);
        }
        entry.reads.set(r.layer, {
          kind: 'read',
          layer: r.layer,
          bullets: sameCap ? r.bullets : r.bullets.slice(0, ORACLE_BULLETS),
          text: r.text,
          finish: r.finish,
          cached: true,
          sig: sameCap ? r.sig : undefined,
        });
      }
      for (const [position, entry] of byPosition) {
        cacheRef.current.set(oracleKey(modelId, position, runHashes[position]), entry);
      }
      setSeededLayers(Array.from(new Set(oracle.reads.map((r) => r.layer))).sort((a, b) => a - b));
      bump();
    },
    [modelId, bump],
  );

  return useMemo(
    () => ({
      modelId,
      available: enabled && (canRead || seededLayers.length > 0),
      layers,
      busy,
      entryFor,
      request,
      unavailableReason,
      sharedReads,
      seed,
      setSweepActive,
      sweepStore,
    }),
    [
      modelId,
      enabled,
      canRead,
      seededLayers,
      layers,
      busy,
      entryFor,
      request,
      unavailableReason,
      sharedReads,
      seed,
      sweepStore,
    ],
  );
}

// The oracle's readout under the popup's shared layer strip, laid out like the
// lens readouts beside it. With no active layer, one row per visible layer at
// the lens rows' height (so the scroll can be mirrored); layers the oracle
// does not read stay blank. With an active layer, that layer's full bullets.
// Before a read, a button in the center starts it.
export function OracleLayerReadout({
  token,
  layers,
  activeLayer,
  range,
  scrollRef,
  onScroll,
}: {
  token: LensTokenMessage;
  layers: number[];
  activeLayer: number | null;
  range: LayerRange | null;
  scrollRef?: (el: HTMLDivElement | null) => void;
  onScroll?: (e: UIEvent<HTMLDivElement>) => void;
}) {
  const oracle = useContext(OracleContext);
  if (!oracle) {
    return null;
  }
  const entry = oracle.entryFor(token.position);
  const oracleLayers = new Set(oracle.layers);
  const cell = (layer: number) => {
    if (!oracleLayers.has(layer)) {
      return null;
    }
    const bullets = layerBullets(entry, layer);
    if (bullets) {
      return bullets.length > 0 ? bullets : null;
    }
    return entry?.status === 'loading' ? 'loading' : null;
  };
  const activeLayerNumber = activeLayer != null ? layers[activeLayer] : null;

  let body: ReactNode;
  if (!entry) {
    const credit = ORACLE_CREDITS[oracle.modelId];
    body = (
      <div className="flex flex-1 flex-col items-center justify-center gap-y-1.5 px-3">
        <button
          type="button"
          onClick={() => oracle.request(token.position)}
          disabled={oracle.busy}
          title={oracle.busy ? 'Wait for the run to end.' : undefined}
          className="rounded-md bg-slate-400 px-3 py-1.5 text-[11px] font-semibold text-white transition-colors hover:bg-slate-500 disabled:cursor-not-allowed disabled:opacity-40"
        >
          Load Oracle Lens
        </button>
        {credit && (
          <a
            href={credit.href}
            className="mt-0.5 text-[10px] text-slate-400 hover:underline"
            target="_blank"
            rel="noopener noreferrer"
          >
            {credit.label}
          </a>
        )}
      </div>
    );
  } else if (entry.status === 'error') {
    body = <div className="px-3 py-3 text-center text-[11px] leading-snug text-red-500">{entry.error}</div>;
  } else if (activeLayerNumber != null) {
    const c = cell(activeLayerNumber);
    body =
      c === 'loading' ? (
        <span className="h-3 w-full animate-pulse rounded bg-slate-100" />
      ) : c ? (
        <ul className="flex flex-col gap-y-1">
          {c.map((b, i) => (
            <li key={i} className="break-words text-[11px] leading-snug text-slate-700">
              <span className="mr-1 text-slate-300">•</span>
              {b}
            </li>
          ))}
        </ul>
      ) : !oracleLayers.has(activeLayerNumber) ? (
        <div className="text-[10px] leading-snug text-slate-400">
          No oracle read at this layer. It reads layers {oracle.layers.join(', ')}.
        </div>
      ) : null;
  } else {
    body = [...visibleLayerIndices(layers, range)].reverse().map((idx) => {
      const c = cell(layers[idx]);
      return (
        <div
          key={layers[idx]}
          className="flex h-5 max-h-5 min-h-5 w-full flex-row items-center gap-x-1.5 px-2 py-0.5 text-[10px]"
        >
          <span className="w-[21px] shrink-0 font-mono text-[9px] tabular-nums text-slate-400 sm:w-10">
            {layers[idx]}
          </span>
          {c === 'loading' ? (
            <span className="h-2.5 flex-1 animate-pulse rounded bg-slate-100" />
          ) : c ? (
            <span className="min-w-0 flex-1 truncate text-slate-700" title={c.join('\n')}>
              {c.join(' · ')}
            </span>
          ) : null}
        </div>
      );
    });
  }

  return (
    <div className="flex max-h-[378px] min-h-[378px] w-full flex-col border-t-4 border-slate-200">
      <div className="flex shrink-0 flex-row items-center gap-x-1.5 border-b border-slate-200 px-2 py-1.5 text-[9px] uppercase tracking-wide text-slate-500 sm:px-5">
        {activeLayerNumber != null ? (
          <span className="min-w-0 flex-1 whitespace-nowrap">
            Layer <span className="font-bold text-slate-700">{activeLayerNumber}</span> Oracle
          </span>
        ) : (
          <>
            <span className="hidden w-10 shrink-0 sm:block">Layer</span>
            <span className="min-w-0 flex-1 whitespace-nowrap text-center sm:text-left">Oracle</span>
          </>
        )}
      </div>
      <div
        ref={scrollRef}
        onScroll={onScroll}
        className={`flex min-h-0 flex-1 flex-col overflow-y-auto py-2 ${
          activeLayerNumber != null ? 'px-5' : 'gap-y-0.5 px-2 sm:px-5'
        }`}
      >
        {body}
      </div>
    </div>
  );
}
