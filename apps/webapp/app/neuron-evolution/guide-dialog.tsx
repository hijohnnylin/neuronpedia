'use client';

import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from '@/components/shadcn/dialog';
import { CircleHelp } from 'lucide-react';
import { BLOG_URL, ModelEntry, PIPELINE_URL } from './shared';

const GUIDE: [string, string][] = [
  [
    'Pick a neuron',
    'Use the map, the groups or search (e.g. L6/N201) on the left. Closer dots have more alike descriptions; drag the slider or press play to move through training; double-click a dot to center the map on it.',
  ],
  [
    'Concept evolution chart',
    'Each line is one concept; training runs left to right. A fork from another line = derived from an older concept; a color change on the same line = replacement; entering with a hollow circle = an unrelated new concept; bottom row = seen only once.',
  ],
  ['Click a line', 'Shows only its family; click again to restore. The toggle colors by mechanism or by family.'],
  [
    'Bar chart',
    'Which concepts carry the activation of the 10 strongest passages at each stage. Hover for percentages; click a column to jump to that stage.',
  ],
  [
    'Table and text',
    "Click a row to see the AI's explanation and the original text at that stage; the bluer, the stronger. Use ← → to change stage, or compare two stages side by side.",
  ],
];

export default function GuideDialog({ model }: { model: ModelEntry | null }) {
  const pile10k = model?.corpus === 'NeelNanda/pile-10k';
  return (
    <Dialog>
      <DialogTrigger asChild>
        <button
          type="button"
          className="inline-flex items-center gap-1 rounded-full border border-slate-200 bg-white px-2.5 py-0.5 text-xs text-slate-700 hover:bg-slate-100"
        >
          <CircleHelp className="h-3.5 w-3.5" />
          Guide
        </button>
      </DialogTrigger>
      <DialogContent className="max-h-[calc(100vh-3rem)] overflow-y-auto bg-white sm:max-w-[560px]">
        <DialogHeader>
          <DialogTitle>How to read this page</DialogTitle>
          <DialogDescription>Hover lines, numbers and titles for details.</DialogDescription>
        </DialogHeader>
        <ol className="m-0 list-none p-0">
          {GUIDE.map(([title, body], i) => (
            <li key={title} className="my-2.5 grid grid-cols-[22px_minmax(0,1fr)] gap-2 text-[13px] text-slate-600">
              <span className="grid h-5 w-5 place-items-center rounded-full bg-slate-100 text-xs font-semibold text-slate-800">
                {i + 1}
              </span>
              <div>
                <b className="text-slate-800">{title}</b>
                <br />
                {body}
              </div>
            </li>
          ))}
        </ol>
        <div className="text-xs text-slate-600">
          <b className="text-slate-800">Method.</b> Neurons are sampled uniformly per layer. At each published
          checkpoint an LLM explains the strongest positive activations; the explanation is scored by detection (8
          held-out top-activation texts against 8 random texts, so 0.5 is chance). The per-step patterns are then
          chained into concept lineages, each born by branching from an older one, replacing a dead one, arriving
          unrelated, or appearing once.
          {pile10k
            ? ' This model uses the fixed pile-10k probe (6,356,992 tokens); the original Pythia-160M release used a different corpus, so check corpus and probe before comparing models.'
            : ''}
        </div>
        <div className="text-xs text-slate-600">
          Pipeline and data:{' '}
          <a href={PIPELINE_URL} target="_blank" rel="noreferrer" className="text-sky-700 underline">
            HeartyHaven/neuron-evolution
          </a>
          {' · '}
          <a href={BLOG_URL} target="_blank" rel="noreferrer" className="text-sky-700 underline">
            blog post
          </a>
          . Gao, Zhang and Chen, <i>Recursive Generalization and Superposition Are Both Right</i>, 2026.
        </div>
      </DialogContent>
    </Dialog>
  );
}
