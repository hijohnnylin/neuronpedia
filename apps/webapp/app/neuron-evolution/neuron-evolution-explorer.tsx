'use client';

/**
 * The Neuron Evolution explorer: pick one of the sampled Pythia MLP neurons on the left, and read how
 * the concepts it responds to appeared, extended and faded across training on the right.
 *
 * All data is static JSON fetched from DATA_BASE_URL (see shared.ts); nothing here touches the
 * database or an inference server. The URL carries the selection (?model=&neuron=&step=) so a view
 * can be linked.
 */

import { Database, ExternalLink, Github } from 'lucide-react';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import GuideDialog from './guide-dialog';
import LineagePanel from './lineage-panel';
import SemanticMap from './semantic-map';
import {
  BLOG_URL,
  ColorBy,
  DecodedSim,
  decodeSim,
  DEFAULT_NEURON_ID,
  fetchCatalog,
  fetchModelJson,
  fmtStep,
  groupDesc,
  groupKey,
  groupName,
  groupTitle,
  isModelAvailable,
  label,
  LayerData,
  matches,
  Meta,
  ModelCatalog,
  ModelEntry,
  modelStatus,
  NEURON_EVOLUTION_PATH,
  NeuronMeta,
  neuronName,
  neuronRef,
  parseNeuronRef,
  PIPELINE_URL,
  SimData,
  StepRecord,
  tr,
} from './shared';
import { ExplanationTable, StepPanel } from './step-panel';

const PANEL = 'rounded-lg border border-slate-200 bg-white p-3';
const CHIP = 'mr-1 inline-block rounded-full bg-slate-100 px-2 text-xs text-slate-600';
const SELECT = 'max-w-full rounded-md border border-slate-200 bg-white px-1.5 py-0.5 text-xs text-slate-700';
const LINK =
  'inline-flex items-center gap-1 whitespace-nowrap rounded-full border border-slate-200 bg-white px-2.5 py-0.5 text-xs text-slate-700 hover:bg-slate-100';

function Confidence({ c }: { c: number | null | undefined }) {
  if (c == null) return null;
  return (
    <span className="text-slate-400" title="The AI's own confidence in this call, not a statistical probability">
      {' '}
      · AI confidence {Math.round(c * 100)}%
    </span>
  );
}

export default function NeuronEvolutionExplorer({
  initialModelId,
  initialNeuron,
  initialStep,
}: {
  initialModelId?: string;
  initialNeuron?: string;
  initialStep?: number;
}) {
  const [catalog, setCatalog] = useState<ModelCatalog | null>(null);
  const [model, setModel] = useState<ModelEntry | null>(null);
  const [meta, setMeta] = useState<Meta | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [sel, setSel] = useState<string | null>(null);
  const [rows, setRows] = useState<StepRecord[] | null>(null);
  const [loadingLayer, setLoadingLayer] = useState<number | null>(null);
  const [step, setStep] = useState<number | null>(null);
  const [cmp, setCmp] = useState<number | null>(null);
  const [grouping, setGrouping] = useState('category');
  const [group, setGroup] = useState<string | null>(null);
  const [query, setQuery] = useState('');
  const [colorBy, setColorBy] = useState<ColorBy>('mech');
  const [famFocus, setFamFocus] = useState<string | null>(null);
  const [topN, setTopN] = useState(5);

  // The neuron and step named by the URL apply to the first model only; a model switch starts fresh.
  const initialRef = useRef<{ neuron: string | null; step?: number } | null>({
    neuron: parseNeuronRef(initialNeuron),
    step: initialStep,
  });
  const layersRef = useRef<Record<string, LayerData>>({});
  const simRef = useRef<Record<string, Promise<DecodedSim>>>({});
  const requestRef = useRef(0);

  useEffect(() => {
    fetchCatalog()
      .then((c) => {
        setCatalog(c);
        // Only models whose bundle has been published are offered; sizes still being extracted stay out.
        const available = c.models.filter(isModelAvailable);
        const m =
          available.find((x) => x.id === initialModelId) ||
          available.find((x) => x.id === c.default_model) ||
          available[0];
        if (!m) throw new Error('No model has published results yet');
        setModel(m);
      })
      .catch((e: unknown) => setError(e instanceof Error ? e.message : String(e)));
  }, [initialModelId]);

  const models = useMemo(() => catalog?.models.filter(isModelAvailable) ?? [], [catalog]);

  const selectNeuron = useCallback(async (m: ModelEntry, md: Meta, id: string, requestedStep?: number) => {
    const n = md.neurons.find((x) => x.id === id);
    if (!n) return;
    const request = (requestRef.current += 1);
    setSel(id);
    setFamFocus(null);
    setCmp(null);
    const key = `${m.id}/L${n.layer}`;
    let layer = layersRef.current[key];
    if (!layer) {
      setRows(null);
      setLoadingLayer(n.layer);
      try {
        layer = await fetchModelJson<LayerData>(m, `L${n.layer}`);
        layersRef.current[key] = layer;
      } catch (e: unknown) {
        if (request === requestRef.current) setError(e instanceof Error ? e.message : String(e));
        return;
      } finally {
        if (request === requestRef.current) setLoadingLayer(null);
      }
    }
    if (request !== requestRef.current) return;
    const r = layer[id] ?? [];
    setRows(r);
    setStep((cur) => {
      if (requestedStep != null && r.some((x) => x.step === requestedStep)) return requestedStep;
      if (cur != null && r.some((x) => x.step === cur)) return cur;
      return r.length ? r[r.length - 1].step : null;
    });
  }, []);

  useEffect(() => {
    if (!model) return undefined;
    let cancelled = false;
    setMeta(null);
    setSel(null);
    setRows(null);
    setStep(null);
    setGroup(null);
    setQuery('');
    setError(null);
    fetchModelJson<Meta>(model, 'meta')
      .then((md) => {
        if (cancelled) return;
        setMeta(md);
        const initial = initialRef.current;
        initialRef.current = null;
        const wanted = initial?.neuron;
        const first =
          (wanted && md.neurons.find((x) => x.id === wanted)) ||
          md.neurons.find((x) => x.id === DEFAULT_NEURON_ID) ||
          md.neurons[0];
        if (first) selectNeuron(model, md, first.id, initial?.step);
      })
      .catch((e: unknown) => {
        if (!cancelled) setError(e instanceof Error ? e.message : String(e));
      });
    return () => {
      cancelled = true;
    };
  }, [model, selectNeuron]);

  // Keep the URL in step with the view, so it can be copied as a link to this neuron at this stage.
  useEffect(() => {
    if (!model || !meta || typeof window === 'undefined') return;
    const n = sel ? meta.neurons.find((x) => x.id === sel) : undefined;
    const params = new URLSearchParams();
    params.set('model', model.id);
    if (n) params.set('neuron', neuronRef(n));
    if (n && step != null) params.set('step', String(step));
    window.history.replaceState(null, '', `${NEURON_EVOLUTION_PATH}?${params.toString()}`);
  }, [model, meta, sel, step]);

  const moveStep = useCallback(
    (delta: number) => {
      if (!rows || !rows.length) return;
      const i = Math.max(
        0,
        rows.findIndex((r) => r.step === step),
      );
      setStep(rows[Math.max(0, Math.min(rows.length - 1, i + delta))].step);
    },
    [rows, step],
  );

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement | null)?.tagName;
      if (!rows || tag === 'INPUT' || tag === 'SELECT' || tag === 'TEXTAREA') return;
      if (e.key === 'ArrowLeft') moveStep(-1);
      if (e.key === 'ArrowRight') moveStep(1);
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [rows, moveStep]);

  const isMatch = useCallback((n: NeuronMeta) => matches(n, grouping, group, query), [grouping, group, query]);
  const list = useMemo(() => (meta ? meta.neurons.filter(isMatch) : []), [meta, isMatch]);
  const selected = sel && meta ? meta.neurons.find((x) => x.id === sel) : undefined;

  const loadSim = useCallback(() => {
    if (!model) return Promise.reject(new Error('No model'));
    if (!simRef.current[model.id]) {
      simRef.current[model.id] = fetchModelJson<SimData>(model, 'sim').then(decodeSim);
    }
    return simRef.current[model.id];
  }, [model]);

  const switchModel = (id: string) => {
    const m = models.find((x) => x.id === id);
    if (!m || m.id === model?.id) return;
    initialRef.current = null;
    setModel(m);
  };

  const groups = meta ? (meta.groups[grouping] ?? Object.values(meta.groups)[0]) : null;
  const counts: Record<string, number> = {};
  if (meta) for (const n of meta.neurons) counts[groupKey(n, grouping)] = (counts[groupKey(n, grouping)] || 0) + 1;
  const groupKeys = groups
    ? groups.order.filter((k) => counts[k]).concat(Object.keys(counts).filter((k) => !groups.order.includes(k)))
    : [];
  const maxCount = Math.max(1, ...Object.values(counts));

  const info = () => {
    if (!model) return '';
    if (!meta) return `${model.label} · ${modelStatus(model)}`;
    let s = `${model.label} · ${meta.neurons.length} neurons · ${meta.steps.length} training stages`;
    if (model.status !== 'ready') s += ` · partial results ${meta.steps.length}/${meta.planned_steps?.length || '?'}`;
    if (model.corpus) s += ` · ${model.corpus}`;
    return s;
  };

  return (
    <div className="w-full max-w-screen-2xl px-3 pb-10 pt-3 text-[14px] text-slate-900 sm:px-5">
      <div className="mb-3 flex flex-wrap items-center gap-x-4 gap-y-2">
        <h1 className="text-base font-semibold">Neuron Evolution</h1>
        <label className="flex items-center gap-1.5 text-xs text-slate-600">
          Model
          <select
            className={SELECT}
            aria-label="Select model"
            value={model?.id ?? ''}
            disabled={!catalog}
            onChange={(e) => switchModel(e.target.value)}
          >
            {models.map((m) => (
              <option key={m.id} value={m.id}>
                {m.label} · {modelStatus(m)}
              </option>
            ))}
          </select>
        </label>
        <label className="flex items-center gap-1.5 text-xs text-slate-600">
          Group by
          <select
            className={SELECT}
            value={grouping}
            disabled={!meta}
            onChange={(e) => {
              setGrouping(e.target.value);
              setGroup(null);
            }}
          >
            {meta
              ? Object.entries(meta.groups)
                  .filter(([k]) => k !== 'cluster')
                  .map(([k, g]) => (
                    <option key={k} value={k}>
                      {groupTitle(g)}
                    </option>
                  ))
              : null}
          </select>
        </label>
        <label className="flex items-center gap-1.5 text-xs text-slate-600">
          Search
          <input
            className={`${SELECT} w-44`}
            placeholder="e.g. L6/N201, or a word"
            value={query}
            disabled={!meta}
            onChange={(e) => setQuery(e.target.value)}
          />
        </label>
        <span className="text-xs text-slate-500">{info()}</span>
        <span className="ml-auto flex flex-wrap items-center gap-1.5">
          <GuideDialog model={model} />
          <a className={LINK} href={PIPELINE_URL} target="_blank" rel="noreferrer">
            <Github className="h-3.5 w-3.5" /> Pipeline
          </a>
          {model?.dataset_url && (
            <a className={LINK} href={model.dataset_url} target="_blank" rel="noreferrer">
              <Database className="h-3.5 w-3.5" /> Dataset
            </a>
          )}
          <a className={LINK} href={BLOG_URL} target="_blank" rel="noreferrer">
            <ExternalLink className="h-3.5 w-3.5" /> Blog
          </a>
        </span>
      </div>

      {error && <div className="mb-3 rounded-lg border border-red-200 bg-red-50 p-3 text-sm text-red-700">{error}</div>}

      <div className="grid grid-cols-1 items-start gap-3.5 lg:grid-cols-[380px_minmax(0,1fr)]">
        <aside className="flex min-w-0 flex-col gap-3.5 lg:sticky lg:top-2 lg:max-h-[calc(100vh-1rem)] lg:overflow-auto">
          <div className={PANEL}>
            {meta && model ? (
              <SemanticMap
                key={model.id}
                neurons={meta.neurons}
                steps={meta.steps}
                selected={sel}
                isMatch={isMatch}
                onSelect={(id) => selectNeuron(model, meta, id)}
                loadSim={loadSim}
              />
            ) : (
              <div className="py-10 text-center text-xs text-slate-400">Loading…</div>
            )}
          </div>
          {meta && groups && (
            <div className={PANEL}>
              <h2 className="mb-1.5 text-sm font-semibold text-slate-800">{groupTitle(groups)}</h2>
              <div>
                <button
                  type="button"
                  className={`block w-full rounded-md border px-2 py-1.5 text-left ${group == null ? 'border-sky-700 bg-sky-50' : 'border-transparent hover:bg-slate-50'}`}
                  onClick={() => setGroup(null)}
                >
                  <div className="flex justify-between gap-2 text-[13px] font-semibold">
                    <span>All</span>
                    <span>{meta.neurons.length}</span>
                  </div>
                </button>
                {groupKeys.map((k) => {
                  const it = groups.items[k];
                  return (
                    <button
                      key={k}
                      type="button"
                      className={`block w-full rounded-md border px-2 py-1.5 text-left ${group === k ? 'border-sky-700 bg-sky-50' : 'border-transparent hover:bg-slate-50'}`}
                      onClick={() => setGroup(k)}
                    >
                      <div className="flex justify-between gap-2 text-[13px] font-semibold">
                        <span>{groupName(it, k)}</span>
                        <span>{counts[k]}</span>
                      </div>
                      <div
                        className="mt-0.5 h-1 rounded-sm bg-sky-600"
                        style={{ width: `${(100 * counts[k]) / maxCount}%` }}
                      />
                      {groupDesc(it) && <div className="text-xs text-slate-600">{groupDesc(it)}</div>}
                    </button>
                  );
                })}
              </div>
            </div>
          )}
          {meta && model && (
            <div className={PANEL}>
              <h2 className="mb-1.5 text-sm font-semibold text-slate-800">
                Neurons <span className="text-xs font-normal text-slate-500">({list.length})</span>
              </h2>
              <div className="max-h-[360px] overflow-auto lg:max-h-none">
                {list.length ? (
                  list.map((n) => (
                    <button
                      key={n.id}
                      type="button"
                      onClick={() => selectNeuron(model, meta, n.id)}
                      className={`grid w-full grid-cols-[76px_minmax(0,1fr)] gap-1.5 rounded-md px-1.5 py-1 text-left text-[13px] ${sel === n.id ? 'bg-sky-50' : 'hover:bg-slate-50'}`}
                    >
                      <span className="font-semibold tabular-nums">{neuronName(n)}</span>
                      <span>
                        {tr(n, 'label')}
                        <br />
                        <span className={CHIP}>{label('dom', n.dominant)}</span>
                        <span className={CHIP}>{label('diff', n.diff)}</span>
                      </span>
                    </button>
                  ))
                ) : (
                  <div className="text-xs text-slate-500">No neurons match.</div>
                )}
              </div>
            </div>
          )}
        </aside>

        <main className="flex min-w-0 flex-col gap-3.5">
          {!model || (!meta && !error) ? (
            <div className={`${PANEL} py-10 text-center text-slate-400`}>Loading…</div>
          ) : meta && !selected ? (
            <div className={PANEL}>
              <h2 className="text-base font-semibold">What this is</h2>
              <p className="max-w-[72ch] text-[13px] text-slate-600">
                Explanations for {meta.neurons.length} neurons of {model.label} across {meta.steps.length} published
                training stages. Pick a neuron on the left.
              </p>
            </div>
          ) : meta && selected ? (
            <>
              <div className={PANEL}>
                <h2 className="text-lg font-semibold">
                  {neuronName(selected)} <span className="font-normal text-slate-600">{tr(selected, 'label')}</span>
                </h2>
                <span className={CHIP}>
                  Content type <b>{groupName(meta.groups.category?.items[selected.category], selected.category)}</b>
                </span>
                <span className={CHIP}>
                  How it evolved <b>{label('rg', selected.rg)}</b>
                  <Confidence c={selected.rg_conf} />
                </span>
                <span
                  className={CHIP}
                  title="The mechanism holding over half of the activation on average late in training (from step 16000)"
                >
                  Late-training makeup <b>{label('dom', selected.dominant)}</b>
                </span>
                <span className={CHIP}>
                  Like denoising? <b>{label('diff', selected.diff)}</b>
                  <Confidence c={selected.diff_conf} />
                </span>
              </div>
              {selected.rg_conf == null && selected.diff_conf == null ? (
                <div className={`${PANEL} text-slate-500`}>
                  Completed checkpoint explanations and source contexts are available. Full trajectory judgments and
                  concept lineages are still being analyzed.
                </div>
              ) : (
                <div className="grid grid-cols-1 gap-3.5 md:grid-cols-2">
                  <div className={PANEL}>
                    <h3 className="mb-1.5 text-sm font-semibold">
                      AI summary: how it evolved · {label('rg', selected.rg)}
                      <Confidence c={selected.rg_conf} />
                    </h3>
                    <div>{tr(selected, 'rg_summary')}</div>
                    <div className="my-1.5 grid grid-cols-[max-content_minmax(0,1fr)] gap-x-2.5 text-[13px]">
                      <div className="text-slate-500">Core idea</div>
                      <div>{tr(selected, 'core')}</div>
                    </div>
                    <details>
                      <summary className="cursor-pointer text-xs text-slate-500">AI&apos;s reasoning</summary>
                      <div className="text-xs">{tr(selected, 'rg_evidence')}</div>
                    </details>
                  </div>
                  <div className={PANEL}>
                    <h3 className="mb-1.5 text-sm font-semibold">
                      AI summary: did it sharpen gradually, like denoising? · {label('diff', selected.diff)}
                      <Confidence c={selected.diff_conf} />
                    </h3>
                    <div>{tr(selected, 'diff_summary')}</div>
                    <div className="my-1.5 grid grid-cols-[max-content_minmax(0,1fr)] gap-x-2.5 gap-y-0.5 text-[13px]">
                      <div className="text-slate-500">Final idea first appears</div>
                      <div>{selected.commit != null ? fmtStep(selected.commit) : '—'}</div>
                      <div className="text-slate-500">Stops getting clearer at</div>
                      <div>{selected.saturation != null ? fmtStep(selected.saturation) : '—'}</div>
                      <div className="text-slate-500">Broad direction settled before details</div>
                      <div>{label('cbf', selected.cbf)}</div>
                      <div className="text-slate-500">How clarity changed</div>
                      <div>{label('trend', selected.trend)}</div>
                    </div>
                    <details>
                      <summary className="cursor-pointer text-xs text-slate-500">AI&apos;s reasoning</summary>
                      <div className="text-xs">{tr(selected, 'diff_evidence')}</div>
                    </details>
                  </div>
                </div>
              )}
              {selected.lineages?.length && selected.lineage_shares?.length ? (
                <LineagePanel
                  n={selected}
                  steps={meta.steps}
                  step={step}
                  onStep={setStep}
                  colorBy={colorBy}
                  onColorBy={setColorBy}
                  famFocus={famFocus}
                  onFamFocus={setFamFocus}
                />
              ) : null}
              {rows && rows.length && step != null ? (
                <>
                  <ExplanationTable rows={rows} steps={meta.steps} loss={meta.loss} step={step} onStep={setStep} />
                  <StepPanel
                    n={selected}
                    rows={rows}
                    step={step}
                    cmp={cmp}
                    onCmp={setCmp}
                    onMove={moveStep}
                    topN={topN}
                    onTopN={setTopN}
                    colorBy={colorBy}
                  />
                </>
              ) : (
                <div className={`${PANEL} py-10 text-center text-slate-400`}>
                  {loadingLayer != null ? `Loading layer ${loadingLayer}…` : 'No per-step records for this neuron.'}
                </div>
              )}
            </>
          ) : null}
        </main>
      </div>
    </div>
  );
}
