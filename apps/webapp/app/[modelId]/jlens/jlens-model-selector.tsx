'use client';

import CustomTooltip from '@/components/custom-tooltip';
import { JPP_CREDITS } from '@/components/jlens/jlens-lens-mode';
import { useGlobalContext } from '@/components/provider/global-provider';
import { compareModelIdsBySize } from '@/lib/utils/general';
import { jppLensAvailable } from '@/lib/utils/lens';
import * as DropdownMenu from '@radix-ui/react-dropdown-menu';
import { QuestionMarkCircledIcon, StarFilledIcon } from '@radix-ui/react-icons';
import { ChevronDownIcon } from 'lucide-react';
import { type ReactNode, type SyntheticEvent, useEffect, useRef } from 'react';
import { JLENS_DEMOS_BAR_ID } from './jlens-tour-constants';

export interface JlensDemo {
  shareId: string;
  displayName: string;
}

// Pinned models that get their own dedicated tabs (plus curated demos). Every
// other inference-enabled model is reachable via the "Select Model" dropdown.
export const JLENS_MODEL_TOGGLE_OPTIONS = [
  {
    modelId: 'qwen3.6-27b',
    displayName: 'Qwen 27B',
    featured: true,
    demos: [
      { shareId: 'cmr1qav5a0008pt2xhsvp0scq', displayName: '💬 Verbal Report' },
      { shareId: 'cmr1hlmkj0001pt2x8udm0029', displayName: '🤔 Directed Modulation' },
      { shareId: 'cmr2l0fqc000npt2xbku41y8u', displayName: '🕷️ Multi-Hop Reasoning' },
      { shareId: 'cmr1s0uq90009pt2xbk2o3wqr', displayName: '🛜 General Broadcast' },
      { shareId: 'cmrfaefrz0000nw2xaty0eats', displayName: '🔀 Selective Mediation' },
      { shareId: 'cmr33z92d0000dw2xd7972w77', displayName: '🤼 Versus Logit-Lens' },
      { shareId: 'cmuyz763o0001182xcprl8ar8', displayName: '🫀 J++ Lens' },
      // { shareId: 'cmr1jc4nm0006pt2xg8a39vii', displayName: '🇦🇺 Australia' },
      // { shareId: 'cmr1j5ocu0005pt2x2178homz', displayName: '💰 Blackmail' },
      // { shareId: 'cmr1iqhr40004pt2x5ndl7go8', displayName: '🇷🇺 Spontaneous Russian' },
      // { shareId: 'cmr1hsw4e0002pt2xesotfigr', displayName: '🌊 Anti-Ocean Creatures' },
      // { shareId: 'cmr1hd7x90000pt2xdj7m8yuf', displayName: '💶 Multi-Hop Currency' },
      // { shareId: 'cmr1ifwtl0003pt2xcbld88kj', displayName: '🇪🇸 Multilingual Abstraction' },
    ],
  },
  {
    modelId: 'gemma-3-12b',
    displayName: 'Gemma 12B',
    featured: true,
    demos: [
      { shareId: 'cmr47wh200000dw2xbls3cm95', displayName: '💬 Verbal Report' },
      { shareId: 'cmr48a0230001dw2x72fe4khn', displayName: '🤔 Directed Modulation' },
      { shareId: 'cmr48qc9o0003dw2x6lgl1gc7', displayName: '🕷️ Multi-Hop Reasoning' },
      { shareId: 'cmr48ump60004dw2xcpyz4564', displayName: '🛜 General Broadcast' },
      { shareId: 'cmr4dwlm6000004le6mkd3we1', displayName: '🤼 Versus Logit-Lens' },
    ],
  },
  {
    modelId: 'deepseek-v4-flash',
    displayName: 'DeepSeek V4 Flash',
    featured: true,
    demos: [
      { shareId: 'cmt28t2od0000332x3qvbbwiw', displayName: '💬 Verbal Report' },
      { shareId: 'cmt293qnp0001332x512g3nkm', displayName: '🤔 Directed Modulation' },
      { shareId: 'cmt29hbqf0003332xa7oe5ezl', displayName: '🕷️ Multi-Hop Reasoning' },
      { shareId: 'cmt2aym590000332xc7o88l99', displayName: '🛜 General Broadcast' },
      { shareId: 'cmt2b6yd70000012x6voje2h7', displayName: '🔀 Selective Mediation' },
      { shareId: 'cmt2bcgw40001012xgbleezs7', displayName: '🤼 Versus Logit-Lens' },
      { shareId: 'cmvaai4k4000007jj7vhecp5s', displayName: '🫀 J++ Lens' },
    ],
  },
  {
    modelId: 'gemma-3-1b',
    displayName: 'Gemma 1B',
    demos: [{ shareId: 'cmre470y20000vf2xaw0xccb9', displayName: '🏛️ Multi-Hop Reasoning' }],
  },
  {
    modelId: 'gemma-3-4b',
    displayName: 'Gemma 4B',
    demos: [{ shareId: 'cmreg8wka0000ut2xf5m67mda', displayName: '🕷️ Multi-Hop Reasoning' }],
  },
  {
    modelId: 'gpt-oss-20b',
    displayName: 'GPT-OSS 20B',
    demos: [{ shareId: 'cmsqi0iyh0000an2x4ysk1ct8', displayName: '🕷️ Multi-Hop Reasoning' }],
  },
  {
    modelId: 'llama3.1-8b-it',
    displayName: 'Llama 3.1 8B IT',
    demos: [{ shareId: 'cmreirhmk0000vc2xfq1697b7', displayName: '🕷️ Multi-Hop Reasoning' }],
  },
  {
    modelId: 'llama3.3-70b-it',
    displayName: 'Llama 3.3 70B IT',
    demos: [{ shareId: 'cmsr81qce0000fd2x7mbn7wby', displayName: '🤕 Confused Capital' }],
  },
] as const satisfies readonly {
  modelId: string;
  displayName: string;
  featured?: boolean;
  demos: readonly JlensDemo[];
}[];

// Models shown with a "Featured" badge and surfaced at the top of the dropdown.
const FEATURED_MODEL_IDS = JLENS_MODEL_TOGGLE_OPTIONS.filter((o) => 'featured' in o && o.featured).map(
  (o) => o.modelId,
) as readonly string[];

// Models that have their own dedicated toggle tab (hardcoded below). When one of
// these is active, its name shows on the tab, so the dropdown trigger just reads
// "Models". Any other model (even if it has curated demos) shows its name in the
// dropdown trigger instead.
const TAB_MODEL_IDS: readonly string[] = ['qwen3.6-27b', 'gemma-3-12b', 'deepseek-v4-flash'];

// Inference models with no fitted Jacobian lens.
const NO_JLENS_MODEL_IDS: readonly string[] = ['gemma-4-31b-it'];

// Given the model + currently-viewed shareId, return where a "Next demo" button
// should go: the next demo in that model's curated list, or free chat when on
// the last demo (labelled "Free Chat" accordingly). Returns null when the
// current run isn't one of the predefined demos (so no next button is shown for
// arbitrary shared links).
export function getNextJlensDemo(
  modelId: string,
  currentShareId: string | null,
): { href: string; label: string } | null {
  if (!currentShareId) return null;
  const option = JLENS_MODEL_TOGGLE_OPTIONS.find((o) => o.modelId === modelId);
  if (!option) return null;
  const idx = option.demos.findIndex((d) => d.shareId === currentShareId);
  if (idx === -1) return null;
  if (idx === option.demos.length - 1) return { href: `/${modelId}/jlens`, label: 'Free Chat' };
  return { href: `/${modelId}/jlens?shareId=${option.demos[idx + 1].shareId}`, label: 'Next demo' };
}

// What each model's server runs, for the "?" hover. Models with no entry show no "?".
// The model and its lenses share one dtype; the KV cache does too unless `kvCache` says otherwise.
interface ModelLensInfo {
  hfModelId: string;
  hfModelRevision?: string;
  // The neuronpedia/jacobian-lens revision that has the lens files.
  lensRevision?: string;
  jlensPath: string;
  jppPath?: string;
  dtype?: string;
  quantization?: string;
  kvCache?: string;
  maxLength: number;
}

// The revision with the J++ lenses for Gemma 3 12B and Llama 3.1 8B IT.
const LENS_REVISION = 'b87d2a42d715f50e3a8359d9c29ccdd5514509d2';

const MODEL_LENS_INFO: Record<string, ModelLensInfo> = {
  'qwen3.6-27b': {
    hfModelId: 'Qwen/Qwen3.6-27B',
    hfModelRevision: '6a9e13bd6fc8f0983b9b99948120bc37f49c13e9',
    lensRevision: 'e166deb5ad540932100ef803c8c9bcc2a5453694',
    jlensPath: 'qwen3.6-27b/jlens/Salesforce-wikitext/Qwen3.6-27B_jacobian_lens_n1000.pt',
    jppPath: 'qwen3.6-27b/jpp/Salesforce-wikitext/Qwen3.6-27B_jpp_lens.pt',
    maxLength: 4096,
  },
  'gemma-3-12b': {
    hfModelId: 'google/gemma-3-12b-pt',
    jlensPath: 'gemma-3-12b/jlens/Salesforce-wikitext/gemma-3-12b-pt_jacobian_lens.pt',
    jppPath: 'gemma-3-12b/jpp/Salesforce-wikitext/gemma-3-12b-pt_jpp_lens.pt',
    maxLength: 1024,
  },
  'llama3.1-8b-it': {
    hfModelId: 'meta-llama/Llama-3.1-8B-Instruct',
    jlensPath: 'llama3.1-8b-it/jlens/Salesforce-wikitext/Llama-3.1-8B-Instruct_jacobian_lens.pt',
    jppPath: 'llama3.1-8b-it/jpp/Salesforce-wikitext/Llama-3.1-8B-Instruct_jpp_lens.pt',
    maxLength: 4096,
  },
  'deepseek-v4-flash': {
    hfModelId: 'deepseek-ai/DeepSeek-V4-Flash',
    jlensPath: 'deepseek-v4-flash/jlens/NeelNanda-pile-10k/deepseek-v4-flash_jacobian_lens_cblank.pt',
    jppPath: 'deepseek-v4-flash/jpp/Salesforce-wikitext/DeepSeek-V4-Flash_jpp_lens.pt',
    quantization: 'fp8',
    kvCache: 'fp8',
    maxLength: 2048,
  },
  'gemma-2-2b': {
    hfModelId: 'google/gemma-2-2b',
    jlensPath: 'gemma-2-2b/jlens/Salesforce-wikitext/gemma-2-2b_jacobian_lens.pt',
    maxLength: 1024,
  },
  'gemma-2-2b-it': {
    hfModelId: 'google/gemma-2-2b-it',
    jlensPath: 'gemma-2-2b-it/jlens/Salesforce-wikitext/gemma-2-2b-it_jacobian_lens.pt',
    maxLength: 1024,
  },
  'gemma-2-9b-it': {
    hfModelId: 'google/gemma-2-9b-it',
    jlensPath: 'gemma-2-9b-it/jlens/Salesforce-wikitext/gemma-2-9b-it_jacobian_lens.pt',
    maxLength: 1024,
  },
  'gemma-3-270m': {
    hfModelId: 'google/gemma-3-270m',
    jlensPath: 'gemma-3-270m/jlens/Salesforce-wikitext/gemma-3-270m_jacobian_lens.pt',
    maxLength: 1024,
  },
  'gemma-3-1b': {
    hfModelId: 'google/gemma-3-1b-pt',
    jlensPath: 'gemma-3-1b/jlens/Salesforce-wikitext/gemma-3-1b-pt_jacobian_lens.pt',
    maxLength: 1024,
  },
  'gemma-3-4b': {
    hfModelId: 'google/gemma-3-4b-pt',
    jlensPath: 'gemma-3-4b/jlens/Salesforce-wikitext/gemma-3-4b-pt_jacobian_lens.pt',
    maxLength: 1024,
  },
  'gpt-oss-20b': {
    hfModelId: 'openai/gpt-oss-20b',
    jlensPath: 'gpt-oss-20b/jlens/Salesforce-wikitext/gpt-oss-20b_jacobian_lens.pt',
    quantization: 'mxfp4',
    maxLength: 4096,
  },
  'gpt2-small': {
    hfModelId: 'openai-community/gpt2',
    jlensPath: 'gpt2-small/jlens/Salesforce-wikitext/gpt2_jacobian_lens.pt',
    dtype: 'fp32',
    maxLength: 1024,
  },
  'llama3.1-8b': {
    hfModelId: 'meta-llama/Llama-3.1-8B',
    jlensPath: 'llama3.1-8b/jlens/Salesforce-wikitext/Llama-3.1-8B_jacobian_lens.pt',
    maxLength: 1024,
  },
  'llama3.3-70b-it': {
    hfModelId: 'meta-llama/Llama-3.3-70B-Instruct',
    jlensPath: 'llama3.3-70b-it/jlens/Salesforce-wikitext/Llama-3.3-70B-Instruct_jacobian_lens.pt',
    maxLength: 1024,
  },
  'qwen3-8b': {
    hfModelId: 'Qwen/Qwen3-8B',
    jlensPath: 'qwen3-8b/jlens/Salesforce-wikitext/Qwen3-8B_jacobian_lens.pt',
    maxLength: 4096,
  },
  'qwen3-32b': {
    hfModelId: 'Qwen/Qwen3-32B',
    jlensPath: 'qwen3-32b/jlens/Salesforce-wikitext/Qwen3-32B_jacobian_lens.pt',
    maxLength: 4096,
  },
};

function LensFileLink({ revision, path }: { revision: string; path: string }) {
  return (
    <a
      href={`https://huggingface.co/neuronpedia/jacobian-lens/blob/${revision}/${path}`}
      target="_blank"
      rel="noopener noreferrer"
    >
      <code className="text-sky-600 hover:underline">{path.split('/').pop()}</code>
    </a>
  );
}

function ModelLensInfoContent({ modelId, info }: { modelId: string; info: ModelLensInfo }) {
  const jppCredit = JPP_CREDITS[modelId];
  const lensRevision = info.lensRevision ?? LENS_REVISION;
  const dtype = info.dtype ?? 'bf16';
  return (
    <div className="flex flex-col gap-y-1.5">
      <p>
        <strong>Model:</strong>{' '}
        <a
          href={`https://huggingface.co/${info.hfModelId}${info.hfModelRevision ? `/tree/${info.hfModelRevision}` : ''}`}
          target="_blank"
          rel="noopener noreferrer"
        >
          <code className="text-sky-600 hover:underline">{info.hfModelId}</code>
        </a>
        {info.hfModelRevision && <span className="text-slate-400"> @ {info.hfModelRevision.slice(0, 7)}</span>}
      </p>
      <p>
        <strong>Jacobian Lens:</strong> <LensFileLink revision={lensRevision} path={info.jlensPath} />
      </p>
      {info.jppPath && jppLensAvailable(modelId) && (
        <p>
          <strong>J++ Lens:</strong> <LensFileLink revision={lensRevision} path={info.jppPath} />
          {jppCredit && (
            <span className="text-slate-400">
              {' '}
              by{' '}
              <a href={jppCredit.href} target="_blank" rel="noopener noreferrer" className="hover:underline">
                {jppCredit.label}
              </a>
            </span>
          )}
        </p>
      )}
      <div className="flex flex-col gap-y-1">
        <strong>Deployment</strong>
        <ul className="ml-4 list-inside list-disc">
          <li>
            <strong>Model:</strong> <code>{dtype}</code>,{' '}
            {info.quantization ? (
              <>
                <code>{info.quantization}</code> quantization
              </>
            ) : (
              'no quantization'
            )}
          </li>
          <li>
            <strong>KV Cache:</strong> <code>{info.kvCache ?? dtype}</code>
          </li>
          <li>
            <strong>Lenses:</strong> <code>{dtype}</code>
          </li>
          <li>
            <strong>Max Length:</strong> {info.maxLength.toLocaleString('en-US')} tokens
          </li>
        </ul>
      </div>
    </div>
  );
}

// The "?" that opens a model's lens info. It sits inside a button or a menu item,
// so pointer and click events stop here and do not select the model.
function ModelLensInfoHover({
  modelId,
  side,
  className,
}: {
  modelId: string;
  side: 'bottom' | 'right';
  className?: string;
}) {
  const info = MODEL_LENS_INFO[modelId];
  if (!info) {
    return null;
  }
  const stop = (e: SyntheticEvent) => e.stopPropagation();
  return (
    <CustomTooltip
      side={side}
      delayDuration={0}
      wide
      trigger={
        <span
          role="button"
          tabIndex={-1}
          aria-label={`About ${info.hfModelId}`}
          onClick={stop}
          onPointerDown={stop}
          onPointerUp={stop}
          className={`items-center text-current opacity-70 transition-opacity hover:opacity-100 ${className ?? ''}`}
        >
          <QuestionMarkCircledIcon className="h-3 w-3" />
        </span>
      }
    >
      <ModelLensInfoContent modelId={modelId} info={info} />
    </CustomTooltip>
  );
}

// Absolute "?" at the right edge of a tab; the tab needs `sm:pr-7` to make room.
const TAB_INFO_CLASS = 'right-2 top-1/2 hidden -translate-y-1/2 sm:absolute sm:block';

const tabClass = (active: boolean, rounding: string) =>
  `relative flex h-7 cursor-pointer items-center justify-center border text-[10px] sm:text-[11px] font-medium transition-colors hover:z-20 hover:border-sky-700 hover:bg-sky-100 sm:px-3 ${rounding} ${
    active ? 'z-10 border-sky-700 bg-sky-700 text-white hover:bg-sky-700' : 'border-slate-200 bg-white text-slate-500'
  }`;

export default function JlensModelSelector({
  modelId,
  currentShareId,
  onModelChange,
  onNavigate,
  mobileLeftSlot,
}: {
  modelId: string;
  // The active shared run's id (null in free chat). Drives the active-demo
  // highlight. Passed in (rather than read from the URL) so it can be cleared
  // in place when the user edits a shared run's sidebar selection.
  currentShareId: string | null;
  onModelChange: (modelId: string) => void;
  // Navigate to a jlens URL (demo / free chat). Routed through the page
  // client's transition so the loading skeleton shows immediately on click.
  onNavigate: (href: string) => void;
  mobileLeftSlot?: ReactNode;
}) {
  const { globalModels, getInferenceEnabledModels } = useGlobalContext();

  const currentOption = JLENS_MODEL_TOGGLE_OPTIONS.find((o) => o.modelId === modelId);
  const hasDedicatedTab = TAB_MODEL_IDS.includes(modelId);
  const demos: readonly JlensDemo[] = currentOption?.demos ?? [];

  // Show every inference-enabled model with a lens in the dropdown, but surface
  // the pinned models at the top as "Featured" (same treatment as the releases
  // dropdown) rather than hiding them.
  const dropdownModelIds = [...getInferenceEnabledModels()]
    .filter((id) => !NO_JLENS_MODEL_IDS.includes(id))
    .sort((a, b) => {
      const aIsFeatured = FEATURED_MODEL_IDS.includes(a);
      const bIsFeatured = FEATURED_MODEL_IDS.includes(b);
      if (aIsFeatured && !bIsFeatured) return -1;
      if (!aIsFeatured && bIsFeatured) return 1;
      return compareModelIdsBySize(a, b);
    });

  // When switching to a demo, horizontally scroll the active demo button into
  // view within its own scroll container only (no ancestor scrolling, which is
  // what scrollIntoView would do).
  const demosScrollerRef = useRef<HTMLDivElement>(null);
  const activeDemoRef = useRef<HTMLButtonElement>(null);
  useEffect(() => {
    const scroller = demosScrollerRef.current;
    const btn = activeDemoRef.current;
    if (!scroller || !btn) return;
    const scrollerRect = scroller.getBoundingClientRect();
    const btnRect = btn.getBoundingClientRect();
    const delta = btnRect.left - scrollerRect.left - (scrollerRect.width - btnRect.width) / 2;
    scroller.scrollLeft += delta;
  }, [currentShareId, modelId]);

  return (
    <div className="absolute left-0 top-0 flex w-full flex-col items-center justify-center gap-x-5 gap-y-2 sm:-top-2.5 sm:left-1/2 sm:w-auto sm:-translate-x-1/2">
      <div
        className="flex w-full flex-col items-end justify-center gap-y-1.5 sm:w-auto sm:items-center sm:gap-y-1.5"
        id="jlens-model-selector"
      >
        <div className="z-10 inline-flex h-7">
          <button
            type="button"
            onClick={() => onModelChange('qwen3.6-27b')}
            className={tabClass(modelId === 'qwen3.6-27b', 'rounded-l-md px-2.5 sm:rounded-l-lg sm:pr-7')}
          >
            Qwen <span className="hidden sm:inline sm:px-1">3.6</span> 27B
            <ModelLensInfoHover modelId="qwen3.6-27b" side="bottom" className={TAB_INFO_CLASS} />
          </button>
          {/* <button
            type="button"
            onClick={() => onModelChange('llama3.1-8b')}
            className={`-ml-px ${tabClass(modelId === 'llama3.1-8b', '')}`}
          >
            Llama <span className="hidden sm:inline sm:px-1">3.1</span> 8B
          </button> */}
          <button
            type="button"
            onClick={() => onModelChange('gemma-3-12b')}
            className={`-ml-px ${tabClass(modelId === 'gemma-3-12b', 'px-2.5 sm:pr-7')}`}
          >
            Gemma <span className="hidden sm:inline sm:px-1">3</span> 12B
            <ModelLensInfoHover modelId="gemma-3-12b" side="bottom" className={TAB_INFO_CLASS} />
          </button>
          <button
            type="button"
            onClick={() => onModelChange('deepseek-v4-flash')}
            className={`-ml-px ${tabClass(modelId === 'deepseek-v4-flash', 'px-2.5 sm:pr-7')}`}
          >
            <span className="sm:hidden">DSv4</span>
            <span className="hidden sm:inline">DeepSeek V4 Flash</span>
            <ModelLensInfoHover modelId="deepseek-v4-flash" side="bottom" className={TAB_INFO_CLASS} />
          </button>
          <DropdownMenu.Root>
            <DropdownMenu.Trigger asChild>
              <button
                type="button"
                className={`relative -ml-px flex ${!hasDedicatedTab ? 'w-[110px] min-w-[110px] max-w-[110px] sm:w-[130px] sm:min-w-[130px] sm:max-w-[120px]' : 'w-[70px] min-w-[70px] max-w-[70px] sm:w-[120px] sm:min-w-[120px] sm:max-w-[120px]'} justify-center gap-x-1 px-2 text-[10px] ${tabClass(!hasDedicatedTab, 'rounded-r-md sm:rounded-r-lg')}`}
              >
                <span className="flex-1 truncate pr-2">
                  {!hasDedicatedTab ? (
                    (globalModels[modelId]?.displayName ?? modelId)
                  ) : (
                    <span className="">
                      <span className="hidden sm:inline">Other</span> Models
                    </span>
                  )}
                </span>
                <ChevronDownIcon className="absolute right-1 h-3 w-3 shrink-0 sm:right-2 sm:h-3.5 sm:w-3.5" />
              </button>
            </DropdownMenu.Trigger>
            <DropdownMenu.Portal>
              <DropdownMenu.Content
                align="center"
                sideOffset={4}
                className="z-30 max-h-[400px] w-[230px] overflow-y-auto rounded-md border border-slate-300 bg-white text-xs font-medium text-sky-700 shadow-[0px_10px_38px_-10px_rgba(22,_23,_24,_0.35),_0px_10px_20px_-15px_rgba(22,_23,_24,_0.2)]"
              >
                {dropdownModelIds.length === 0 ? (
                  <div className="px-3 py-2.5 text-[11px] font-normal text-slate-400">No models available</div>
                ) : (
                  dropdownModelIds.map((mId) => (
                    <DropdownMenu.Item
                      key={mId}
                      onSelect={() => onModelChange(mId)}
                      className={`flex cursor-pointer flex-row items-center justify-between gap-x-1.5 border-b border-b-slate-200 px-3 py-2.5 text-xs focus:outline-none ${
                        mId === modelId ? 'bg-sky-100 text-sky-700' : 'text-slate-500'
                      } hover:bg-slate-100`}
                    >
                      <span className="flex flex-row items-center gap-x-1.5 whitespace-nowrap font-sans">
                        {globalModels[mId]?.displayNameShort ?? mId}
                        <ModelLensInfoHover modelId={mId} side="right" className="flex" />
                      </span>
                      {FEATURED_MODEL_IDS.includes(mId) && (
                        <span className="flex flex-row items-center gap-x-0.5 text-[8.5px] font-bold uppercase text-emerald-600">
                          <StarFilledIcon className="h-2 w-2" /> Featured
                        </span>
                      )}
                    </DropdownMenu.Item>
                  ))
                )}
              </DropdownMenu.Content>
            </DropdownMenu.Portal>
          </DropdownMenu.Root>
        </div>

        <div
          className={`flex w-full flex-row items-center justify-between gap-x-1 rounded-lg sm:justify-center sm:gap-x-0 ${
            demos.length > 0 ? 'sm:-mt-5 sm:rounded-2xl sm:bg-slate-200 sm:px-2 sm:py-2 sm:pt-5' : ''
          }`}
        >
          {mobileLeftSlot && <div className="shrink-0 sm:hidden">{mobileLeftSlot}</div>}
          <div className="flex min-w-0 flex-row items-center justify-end gap-x-[0px] gap-y-[1px]">
            {demos.length > 0 && (
              <div
                ref={demosScrollerRef}
                id={JLENS_DEMOS_BAR_ID}
                className="flex min-w-0 max-w-[calc(100dvw-170px)] flex-row items-center justify-start gap-x-[0px] gap-y-[1px] divide-x divide-slate-200 overflow-x-scroll rounded-lg border border-slate-200 sm:w-auto sm:max-w-none sm:justify-center sm:overflow-hidden sm:rounded-xl sm:border-none"
              >
                {demos.map((demo) => {
                  const isActive = currentShareId === demo.shareId;
                  const label = demo.displayName;
                  return (
                    <button
                      key={demo.shareId}
                      ref={isActive ? activeDemoRef : undefined}
                      type="button"
                      onClick={() => {
                        if (!isActive) {
                          onNavigate(`/${modelId}/jlens?shareId=${demo.shareId}`);
                        }
                      }}
                      className={`sm:px-auto flex-1 px-1 py-1.5 text-[11px] font-medium transition-colors disabled:cursor-not-allowed disabled:opacity-50 sm:w-[105px] sm:min-w-[105px] sm:max-w-[105px] sm:flex-auto ${
                        isActive
                          ? 'border-sky-500 bg-sky-700 text-white'
                          : 'border-slate-200 bg-white text-slate-600 hover:bg-sky-100'
                      }`}
                    >
                      {(() => {
                        const firstSpace = label.indexOf(' ');
                        if (firstSpace === -1) {
                          return <span className="text-sm">{label}</span>;
                        }
                        const first = label.slice(0, firstSpace);
                        const second = label.slice(firstSpace + 1);
                        return (
                          <span className="flex flex-row items-center justify-center gap-x-1 px-1.5 sm:gap-x-2 sm:px-2">
                            <span className="text-sm sm:text-xl">{first}</span>
                            {(() => {
                              const secondFirstSpace = second.indexOf(' ');
                              if (secondFirstSpace === -1) {
                                return <span className="text-[9px] sm:text-[10px]">{second}</span>;
                              }
                              const secondFirst = second.slice(0, secondFirstSpace);
                              const secondRest = second.slice(secondFirstSpace + 1);
                              return (
                                <span className="flex flex-col text-[9px] leading-tight sm:text-[10px]">
                                  <span className="whitespace-nowrap">{secondFirst}</span>
                                  <span className="whitespace-nowrap">{secondRest}</span>
                                </span>
                              );
                            })()}
                          </span>
                        );
                      })()}
                    </button>
                  );
                })}
              </div>
            )}
            <button
              type="button"
              onClick={() => {
                // Only navigate when leaving a shared run; clicking Free Chat
                // while already in free chat would needlessly wipe the chat.
                if (currentShareId) {
                  onNavigate(`/${modelId}/jlens`);
                }
              }}
              className={`rounded-lg border px-1 py-1.5 text-[11px] font-medium transition-colors disabled:cursor-not-allowed disabled:opacity-50 sm:w-[90px] sm:min-w-[90px] sm:max-w-[90px] sm:rounded-xl sm:border-none sm:py-1.5 ${
                demos.length > 0 ? 'ml-1 sm:ml-2' : ''
              } ${
                !currentShareId
                  ? 'border-sky-700 bg-sky-700 text-white'
                  : 'border-slate-200 bg-white text-slate-600 hover:bg-sky-100'
              }`}
            >
              <span className="flex flex-row items-center justify-center gap-x-1 px-1.5 sm:gap-x-2.5 sm:px-2">
                <span className="text-xs sm:text-xl">✏️</span>
                <span className="text-[9px] leading-tight sm:text-[11px]">
                  Free
                  <br />
                  Chat
                </span>
              </span>
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
