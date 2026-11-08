import { ExplanationMetadataButton } from '@/components/explanation-source';
import { replaceHtmlAnomalies } from '@/lib/utils/activations';
import * as DropdownMenu from '@radix-ui/react-dropdown-menu';
import * as Tooltip from '@radix-ui/react-tooltip';
import { ChevronDown, ChevronRight, ExternalLink, HelpCircle } from 'lucide-react';
import { ExplanationPartialWithRelations, NeuronTokenListPartialWithRelations } from 'prisma/generated/zod';

export const MISSING_LOGIT_VALUE = -10000;
export const LOGITS_KEY = 'logits';

export type TokenColumnData = {
  title: string;
  help: string;
  tone: 'negative' | 'positive' | 'both';
  tokens: string[];
  values: number[];
  metadata?: unknown;
};

// The lists of one type and name, for example the "Output Read" lists of Neuron Lens, one per pole.
export type TokenListGroup = {
  key: string;
  typeName: string;
  name: string;
  lists: NeuronTokenListPartialWithRelations[];
};

const POLE_ORDER = ['NEGATIVE', 'BOTH', 'POSITIVE'] as const;
const POLE_TITLE = { NEGATIVE: 'Negative', BOTH: 'Both signs', POSITIVE: 'Positive' } as const;
const POLE_TONE = { NEGATIVE: 'negative', BOTH: 'both', POSITIVE: 'positive' } as const;

export function groupTokenLists(lists: NeuronTokenListPartialWithRelations[]): TokenListGroup[] {
  const groups = new Map<string, TokenListGroup>();
  lists.forEach((l) => {
    const typeName = l.typeName ?? '';
    const name = l.name ?? '';
    const key = JSON.stringify([typeName, name]);
    if (!groups.has(key)) {
      groups.set(key, { key, typeName, name, lists: [] });
    }
    groups.get(key)!.lists.push(l);
  });
  return Array.from(groups.values());
}

// The group of the explanation's type that best matches its aspect, then its pole. The first group wins a tie.
export function tokenListKeyForExplanation(
  groups: TokenListGroup[],
  explanation: Pick<ExplanationPartialWithRelations, 'typeName' | 'pole' | 'aspect'>,
): string | undefined {
  const score = (g: TokenListGroup) =>
    Math.max(...g.lists.map((l) => (l.aspect === explanation.aspect ? 2 : 0) + (l.pole === explanation.pole ? 1 : 0)));
  let best: TokenListGroup | undefined;
  groups
    .filter((g) => g.typeName === explanation.typeName)
    .forEach((g) => {
      if (!best || score(g) > score(best)) {
        best = g;
      }
    });
  return best?.key;
}

export function columnsForGroup(group: TokenListGroup): TokenColumnData[] {
  const sourceName = group.lists[0]?.type?.displayName || group.typeName;
  return POLE_ORDER.flatMap((pole) =>
    group.lists
      .filter((l) => l.pole === pole)
      .map((l) => ({
        title: POLE_TITLE[pole],
        help: `${sourceName}, ${group.name}: ${
          l.aspect === 'STEERING' ? 'what this pole writes' : 'what drives this pole'
        }. Highest first.`,
        tone: POLE_TONE[pole],
        tokens: l.tokens ?? [],
        values: l.values ?? [],
        metadata: l.metadata,
      })),
  );
}

function background(tone: TokenColumnData['tone'], ratio: number) {
  if (tone === 'negative') {
    return `rgba(255, calc(255 * (1 - ${ratio})), calc(255 * (1 - ${ratio})), 0.5)`;
  }
  if (tone === 'positive') {
    return `rgba(calc(255 - 239 * ${ratio}), calc(255 - 70 * ${ratio}), calc(255 - 126 * ${ratio}), 0.5)`;
  }
  return `rgba(calc(255 - 200 * ${ratio}), calc(255 - 120 * ${ratio}), 255, 0.5)`;
}

export function TokenColumn({
  column,
  maxValue,
  showHeader,
  isRowHidden,
}: {
  column: TokenColumnData;
  maxValue: number;
  showHeader: boolean;
  isRowHidden: (i: number) => boolean;
}) {
  const hasValues = column.values.length > 0;
  return (
    <div className="min-w-0 flex-1">
      {showHeader && (
        <div className="mb-0 flex flex-row items-center gap-x-1 whitespace-pre text-[9px] font-bold uppercase text-slate-500 sm:mb-1 sm:text-[11px]">
          <div className="mb-0.5 flex flex-row items-center justify-center gap-x-1 font-sans text-[10px] font-normal uppercase text-slate-400 sm:mb-0">
            {column.title}
          </div>
          <Tooltip.Provider delayDuration={300} skipDelayDuration={0}>
            <Tooltip.Root>
              <Tooltip.Trigger asChild>
                <button type="button">
                  <HelpCircle className="h-2.5 w-2.5 text-slate-400 sm:h-2.5 sm:w-2.5" />
                </button>
              </Tooltip.Trigger>
              <Tooltip.Portal>
                <Tooltip.Content
                  className="z-20 max-w-xs whitespace-normal rounded bg-slate-500 px-3 py-2 font-sans text-xs normal-case text-white"
                  sideOffset={5}
                >
                  {column.help}
                  <Tooltip.Arrow className="fill-slate-500" />
                </Tooltip.Content>
              </Tooltip.Portal>
            </Tooltip.Root>
          </Tooltip.Provider>
          <ExplanationMetadataButton metadata={column.metadata} />
        </div>
      )}
      <div className="flex flex-col gap-y-0.5 text-[9px] sm:text-[11px]">
        {column.tokens.map((s, i) => {
          const value = column.values[i];
          const ratio = hasValues ? (maxValue ? Math.abs(value) / maxValue : undefined) : 0.35;
          return (
            <div className={`flex-row justify-between gap-x-1 ${isRowHidden(i) ? 'hidden' : 'flex'}`} key={i}>
              <pre
                style={{ backgroundColor: ratio !== undefined ? background(column.tone, ratio) : undefined }}
                className="max-w-[80px] overflow-x-hidden rounded-sm font-bold sm:max-w-[100px] sm:rounded"
              >
                {replaceHtmlAnomalies(s)}
              </pre>
              {hasValues && <div>{value === MISSING_LOGIT_VALUE ? 'N/A' : value.toFixed(2)}</div>}
            </div>
          );
        })}
      </div>
    </div>
  );
}

const LOGITS_LABEL = 'Top Logits';
const LOGITS_HELP =
  "The tokens that this feature's output raises and lowers most, read directly through the unembedding. It uses only the weights.";

// What each imported list shows, by type and list name. Other lists get a general description.
const LIST_DESCRIPTIONS: Record<string, Record<string, string>> = {
  'neuron-lens': {
    Drivers:
      'The tokens that make this pole fire hardest. Neuron Lens puts each token into 8 real documents and averages the activation at that token.',
    'Input Read': "What the neuron's input looks for, as tokens.",
    'Output Read': 'What this pole writes into the model, as tokens.',
  },
  modelmapper: {
    'Generated Words':
      "Words that Model Mapper's labeller wrote from the label alone, as a test. Only lists where the neuron fires on these words more than on look-alike words are shown.",
  },
  'persona-neurons': {
    'Persona Vectors':
      'The persona vectors that have this neuron in their top 100, ordered by how much this neuron adds to each one.',
  },
};

// What each source's lists show, for a source menu. Other sources use their type description.
const SOURCE_DESCRIPTIONS: Record<string, string> = {
  'neuron-lens':
    'Neuron Lens shows each pole of a neuron as tokens, in three ways: the tokens that drive it, what its input looks for, and what it writes.',
  modelmapper: 'Model Mapper tests each label with words written from the label alone.',
  'persona-neurons': 'Persona Neurons finds the neurons that contribute most to each persona vector.',
};

// Neuron Lens chooses the lens for each layer, and puts its name in metadata, for example "read_R_z".
function lensSentence(lens: unknown) {
  if (typeof lens !== 'string') {
    return '';
  }
  const kind = lens.includes('_S')
    ? 'the S-lens, which shows which input tokens put the model in this state'
    : 'the R-lens, which shows what the model is about to say';
  const z = lens.endsWith('_z') ? ', with each score compared to the other neurons in this layer' : '';
  return ` At this layer it uses ${kind}${z}.`;
}

function sourceName(group: TokenListGroup) {
  return group.lists[0]?.type?.displayName || group.typeName;
}

function groupLabel(group: TokenListGroup) {
  return `${sourceName(group)} · ${group.name}`;
}

function groupHelp(group: TokenListGroup) {
  const known = LIST_DESCRIPTIONS[group.typeName]?.[group.name];
  if (known) {
    const metadata = group.lists[0]?.metadata as { lens?: unknown } | null | undefined;
    return known + lensSentence(metadata?.lens);
  }
  const aspects = new Set(group.lists.map((l) => l.aspect));
  if (!aspects.has('STEERING')) {
    return 'What drives each pole.';
  }
  if (!aspects.has('ACTIVATION')) {
    return 'What each pole writes.';
  }
  return 'What drives each pole, and what it writes.';
}

function sourceHelp(group: TokenListGroup) {
  return SOURCE_DESCRIPTIONS[group.typeName] || group.lists[0]?.type?.description || sourceName(group);
}

type HelpAuthor = { name: string; url?: string | null };

function groupAuthor(group: TokenListGroup): HelpAuthor | undefined {
  const type = group.lists[0]?.type;
  return type?.creatorName ? { name: type.creatorName, url: type.url } : undefined;
}

// Marks help content. A click on its link must not close the menu or select the item.
const HELP_CONTENT_ATTR = 'data-token-list-help';

function HelpTip({ text, author, side = 'top' }: { text: string; author?: HelpAuthor; side?: 'top' | 'right' }) {
  return (
    <Tooltip.Provider delayDuration={200} skipDelayDuration={0}>
      <Tooltip.Root>
        <Tooltip.Trigger asChild>
          <span className="inline-flex shrink-0 cursor-help">
            <HelpCircle className="h-2.5 w-2.5 text-slate-400" />
          </span>
        </Tooltip.Trigger>
        <Tooltip.Portal>
          <Tooltip.Content
            {...{ [HELP_CONTENT_ATTR]: '' }}
            onClick={(e) => e.stopPropagation()}
            onPointerUp={(e) => e.stopPropagation()}
            className="z-50 max-w-xs whitespace-normal rounded bg-slate-500 px-3 py-2 font-sans text-xs normal-case text-white"
            side={side}
            sideOffset={5}
          >
            {text}
            {author && (
              <div className="mt-1.5 text-slate-200">
                From{' '}
                {author.url ? (
                  <a
                    href={author.url}
                    target="_blank"
                    rel="noreferrer"
                    className="inline-flex items-center gap-x-0.5 font-medium text-white underline hover:text-sky-100"
                  >
                    {author.name}
                    <ExternalLink className="h-3 w-3" />
                  </a>
                ) : (
                  <span className="font-medium text-white">{author.name}</span>
                )}
              </div>
            )}
            <Tooltip.Arrow className="fill-slate-500" />
          </Tooltip.Content>
        </Tooltip.Portal>
      </Tooltip.Root>
    </Tooltip.Provider>
  );
}

const ITEM_CLASS =
  'flex cursor-pointer flex-row items-center gap-x-1.5 px-2 py-1 outline-none data-[highlighted]:bg-slate-100 data-[state=open]:bg-slate-100';
const CONTENT_CLASS =
  'z-30 overflow-hidden rounded-sm border border-slate-200 bg-white py-0.5 font-sans text-[11px] font-medium text-slate-600 shadow';

// A sub-menu's left edge, from its item's left edge, and its top, from its item's top.
// Radix keeps a sub-menu open only for a pointer that moves right. This one opens below, so its
// item does not close it on pointer leave. The next item that gets the pointer closes it.
const SUBMENU_INDENT_PX = 20;
const SUBMENU_DROP_PX = 18;

function keepOpenForHelp(e: { target: EventTarget | null; preventDefault: () => void }) {
  if (e.target instanceof Element && e.target.closest(`[${HELP_CONTENT_ATTR}]`)) {
    e.preventDefault();
  }
}

// A menu to show the logits or one group of token lists in the columns below it.
export function TokenListDropdown({
  groups,
  shown,
  onShow,
}: {
  groups: TokenListGroup[];
  shown: string;
  onShow: (key: string) => void;
}) {
  const shownGroup = groups.find((g) => g.key === shown);
  const sources = new Map<string, TokenListGroup[]>();
  groups.forEach((g) => sources.set(g.typeName, [...(sources.get(g.typeName) ?? []), g]));

  const item = (key: string, label: string, help: string, author?: HelpAuthor) => (
    <DropdownMenu.Item
      key={key}
      onSelect={() => onShow(key)}
      className={`${ITEM_CLASS} ${key === shown ? 'bg-slate-50 text-slate-800' : ''}`}
    >
      <span className="truncate">{label}</span>
      <HelpTip text={help} author={author} side="right" />
    </DropdownMenu.Item>
  );

  return (
    <DropdownMenu.Root>
      <DropdownMenu.Trigger asChild>
        <button
          type="button"
          className="mb-1.5 flex w-full flex-row items-center justify-between rounded-sm bg-slate-50 px-2 py-1 font-sans text-[10px] font-medium text-slate-600 outline-none hover:bg-slate-100 focus-visible:ring-1 focus-visible:ring-slate-300 sm:text-[11px]"
        >
          <span className="flex min-w-0 flex-row items-center gap-x-1.5">
            <span className="truncate">{shownGroup ? groupLabel(shownGroup) : LOGITS_LABEL}</span>
            <HelpTip
              text={shownGroup ? groupHelp(shownGroup) : LOGITS_HELP}
              author={shownGroup ? groupAuthor(shownGroup) : undefined}
            />
          </span>
          <ChevronDown className="h-3 w-3 shrink-0 text-slate-400" />
        </button>
      </DropdownMenu.Trigger>
      <DropdownMenu.Portal>
        <DropdownMenu.Content
          className={CONTENT_CLASS}
          style={{ width: 'var(--radix-dropdown-menu-trigger-width)' }}
          sideOffset={2}
          align="start"
          onInteractOutside={keepOpenForHelp}
        >
          {item(LOGITS_KEY, LOGITS_LABEL, LOGITS_HELP)}
          {Array.from(sources.values()).map((sourceGroups) => {
            if (sourceGroups.length === 1) {
              const g = sourceGroups[0];
              return item(g.key, groupLabel(g), groupHelp(g), groupAuthor(g));
            }
            const hasShown = sourceGroups.some((g) => g.key === shown);
            return (
              <DropdownMenu.Sub key={sourceGroups[0].typeName}>
                <DropdownMenu.SubTrigger
                  className={`${ITEM_CLASS} ${hasShown ? 'bg-slate-50 text-slate-800' : ''}`}
                  onPointerLeave={(e) => e.preventDefault()}
                >
                  <span className="truncate">{sourceName(sourceGroups[0])}</span>
                  <HelpTip text={sourceHelp(sourceGroups[0])} author={groupAuthor(sourceGroups[0])} />
                  <ChevronRight className="-ml-0.5 h-3 w-3 shrink-0 text-slate-400" />
                </DropdownMenu.SubTrigger>
                <DropdownMenu.Portal>
                  <DropdownMenu.SubContent
                    className={`${CONTENT_CLASS} w-max min-w-[9rem]`}
                    style={{ marginLeft: `calc(${SUBMENU_INDENT_PX}px - var(--radix-popper-anchor-width))` }}
                    alignOffset={SUBMENU_DROP_PX}
                    avoidCollisions={false}
                    onInteractOutside={keepOpenForHelp}
                  >
                    {sourceGroups.map((g) => item(g.key, g.name, groupHelp(g), groupAuthor(g)))}
                  </DropdownMenu.SubContent>
                </DropdownMenu.Portal>
              </DropdownMenu.Sub>
            );
          })}
        </DropdownMenu.Content>
      </DropdownMenu.Portal>
    </DropdownMenu.Root>
  );
}
