'use client';

import SearchTopkByToken from '@/app/search-topk-by-token/search-topk-by-token';
import ExplanationsSearcher from '@/components/explanations-searcher';
import FeatureSelector from '@/components/feature-selector/feature-selector';
import ModelSelector from '@/components/feature-selector/model-selector';
import InferenceSearcher from '@/components/inference-searcher/inference-searcher';
import JumpToSAE from '@/components/jump-to-sae';
import { useGlobalContext } from '@/components/provider/global-provider';
import RandomReleaseFeature from '@/components/random-release-feature';
import { DEFAULT_MODELID, DEFAULT_SOURCE, DEMO_MODE, IS_LOCALHOST } from '@/lib/env';
import { getSourceSetNameFromSource, NEURONS_SOURCESET } from '@/lib/utils/source';
import { SourceReleaseWithPartialRelations } from '@/prisma/generated/zod';
import * as DropdownMenu from '@radix-ui/react-dropdown-menu';
import * as Select from '@radix-ui/react-select';
import { ChevronDown, ChevronDownIcon, ChevronUpIcon, Plus } from 'lucide-react';
import { useRouter } from 'next-nprogress-bar';
import Link from 'next/link';
import { useState } from 'react';
import InferenceActivationAllProvider from '../provider/inference-activation-all-provider';
import ModelsDropdown from './models-dropdown';
import ReleasesDropdown from './releases-dropdown';

export default function NavBarButtons() {
  const router = useRouter();
  const { getSourceSetsForModelId, globalModels, releases, getDefaultModel, getFirstSourceForSourceSet } =
    useGlobalContext();
  const [jumpToOpen, setJumpToOpen] = useState(false);
  const [jumpToModelModelId, setJumpToModelModelId] = useState(DEFAULT_MODELID || getDefaultModel()?.id || '');

  const defaultSource =
    DEFAULT_SOURCE ||
    (getSourceSetsForModelId(jumpToModelModelId, true).length > 0
      ? getFirstSourceForSourceSet(jumpToModelModelId, getSourceSetsForModelId(jumpToModelModelId, true)[0].name)
      : '');

  return (
    <>
      {IS_LOCALHOST && !DEMO_MODE && (
        <Link
          className="flex cursor-pointer items-center whitespace-nowrap rounded-full px-2.5 py-1 text-[13px] transition-all hover:bg-sky-100 hover:text-sky-700 focus:outline-none data-[state=open]:bg-sky-700 data-[state=open]:text-white"
          href="/admin"
          prefetch={false}
        >
          Admin
        </Link>
      )}

      <DropdownMenu.Root>
        <DropdownMenu.Trigger className="flex cursor-pointer items-center whitespace-nowrap rounded-full px-2.5 py-1 text-[13px] transition-all hover:bg-sky-100 hover:text-sky-700 focus:outline-none data-[state=open]:bg-sky-700 data-[state=open]:text-white">
          Get Started
          <ChevronDown className="-mr-0.5 ml-0.5 h-3.5 w-3.5" />
        </DropdownMenu.Trigger>
        <DropdownMenu.Portal>
          <DropdownMenu.Content
            sideOffset={-3}
            className="z-50 min-w-[8rem] max-w-[24rem] overflow-hidden rounded-md bg-white p-1 shadow-[0px_10px_38px_-10px_rgba(22,_23,_24,_0.35),_0px_10px_20px_-15px_rgba(22,_23,_24,_0.2)]"
          >
            <DropdownMenu.Item className="flex cursor-pointer items-center border-b border-slate-100 px-5 py-3 text-sm font-medium text-sky-700 outline-none hover:bg-sky-100 hover:text-sky-700">
              <Link
                href="https://github.com/hijohnnylin/neuronpedia#readme"
                target="_blank"
                prefetch={false}
                rel="noreferrer"
                className="flex w-full cursor-pointer flex-col items-start justify-center text-[13px]"
              >
                Running Neuronpedia
                <span className="ml-0 pt-0.5 text-[11px] font-normal leading-snug text-slate-500">
                  The latest README for getting started with Neuronpedia.
                </span>
              </Link>
            </DropdownMenu.Item>
            <DropdownMenu.Item className="flex cursor-pointer items-center border-b border-slate-100 px-5 py-3 text-sm font-medium text-sky-700 outline-none hover:bg-sky-100 hover:text-sky-700">
              <Link
                href="https://docs.neuronpedia.org"
                target="_blank"
                prefetch={false}
                rel="noreferrer"
                className="flex w-full cursor-pointer flex-col items-start justify-center text-[13px]"
              >
                Usage Docs (To Be Updated)
                <span className="ml-0 pt-0.5 text-[11px] font-normal leading-snug text-slate-500">
                  What is Neuronpedia? How do I use it?
                </span>
              </Link>
            </DropdownMenu.Item>
            {releases.filter((r) => r.name === 'gemma-scope').length > 0 && (
              <DropdownMenu.Item className="flex cursor-pointer items-center border-b border-slate-100 px-5 py-3 text-sm font-medium text-sky-700 outline-none hover:bg-sky-100 hover:text-sky-700">
                <Link
                  href="/gemma-scope"
                  prefetch={false}
                  rel="noreferrer"
                  className="flex w-full cursor-pointer flex-col items-start justify-center text-[13px]"
                >
                  Gemma Scope
                  <span className="ml-0 pt-0.5 text-[11px] font-normal leading-snug text-slate-500">
                    A guided, beginner-friendly introduction to AI interpretability.
                  </span>
                </Link>
              </DropdownMenu.Item>
            )}

            <DropdownMenu.Arrow className="fill-white" />
          </DropdownMenu.Content>
        </DropdownMenu.Portal>
      </DropdownMenu.Root>

      <Link
        href="/api-doc"
        prefetch={false}
        rel="noreferrer"
        className="flex cursor-pointer items-center whitespace-nowrap rounded-full px-2.5 py-1 text-[13px] transition-all hover:bg-sky-100 hover:text-sky-700 focus:outline-none data-[state=open]:bg-sky-700 data-[state=open]:text-white"
      >
        API
      </Link>

      <ReleasesDropdown />

      <DropdownMenu.Root open={jumpToOpen}>
        <DropdownMenu.Trigger
          onClick={() => {
            setJumpToOpen(!jumpToOpen);
          }}
          className="flex cursor-pointer items-center whitespace-nowrap rounded-full px-2.5 py-1 text-[13px] transition-all hover:bg-sky-100 hover:text-sky-700 focus:outline-none data-[state=open]:bg-sky-700 data-[state=open]:text-white"
        >
          Jump To
          <ChevronDown className="-mr-0.5 ml-0.5 h-3.5 w-3.5" />
        </DropdownMenu.Trigger>
        <DropdownMenu.Portal>
          <DropdownMenu.Content
            sideOffset={-3}
            onPointerDownOutside={() => {
              setJumpToOpen(false);
            }}
            className="z-30 flex min-w-[8rem] flex-col divide-slate-100 overflow-hidden rounded-md bg-white p-1 py-1 shadow-[0px_10px_38px_-10px_rgba(22,_23,_24,_0.35),_0px_10px_20px_-15px_rgba(22,_23,_24,_0.2)]"
          >
            <div className="flex w-full flex-row justify-between border-b border-b-slate-100 px-7 py-4">
              <div
                onSelect={(e) => {
                  e.preventDefault();
                }}
                className="flex cursor-pointer flex-col items-start text-sm font-medium text-sky-700 outline-none"
              >
                <div className="text-[10px] uppercase text-slate-500">Jump to Model</div>
                <div className="flex flex-row gap-x-2">
                  <ModelSelector
                    modelId={jumpToModelModelId}
                    modelIdChangedCallback={(modelId) => {
                      setJumpToModelModelId(modelId);
                    }}
                  />
                  <button
                    type="button"
                    onClick={(e) => {
                      if (e.ctrlKey || e.metaKey) {
                        window.open(`/${jumpToModelModelId}`, '_blank');
                      } else {
                        router.push(`/${jumpToModelModelId}`);
                      }
                      setJumpToOpen(false);
                    }}
                    className="flex h-10 max-h-[40px] min-h-[40px] select-none items-center justify-center rounded bg-slate-200 px-3 text-[11px] font-medium uppercase text-slate-500 hover:bg-sky-700 hover:text-white"
                  >
                    Go
                  </button>
                </div>
              </div>
              <div className="flex flex-col pt-1">
                <div className="mb-1 font-sans text-[9px] font-medium uppercase text-slate-500">Random Feature</div>
                <RandomReleaseFeature
                  callback={() => {
                    setJumpToOpen(false);
                  }}
                  release={releases[Math.floor(Math.random() * releases.length)] as SourceReleaseWithPartialRelations}
                />
              </div>
            </div>
            <div className="px-7 py-4">
              <JumpToSAE
                modelId={jumpToModelModelId}
                layer={defaultSource}
                callback={() => {
                  setJumpToOpen(false);
                }}
                modelOnSeparateRow
              />
            </div>
            <div
              onSelect={(e) => {
                e.preventDefault();
              }}
              className="flex cursor-pointer flex-col items-start border-t border-b-slate-100 px-7 py-4 text-sm font-medium text-sky-700 outline-none"
            >
              <div className="text-[10px] font-medium uppercase text-slate-500">Jump to Feature</div>
              <FeatureSelector
                showModel
                openInNewTab={false}
                defaultModelId={jumpToModelModelId}
                defaultSourceSet={getSourceSetNameFromSource(defaultSource)}
                defaultIndex="0"
                filterToPublic
                modelOnSeparateRow
                callback={() => {
                  setJumpToOpen(false);
                }}
              />
            </div>
            <DropdownMenu.Arrow className="fill-white" />
          </DropdownMenu.Content>
        </DropdownMenu.Portal>
      </DropdownMenu.Root>

      <DropdownMenu.Root>
        <DropdownMenu.Trigger className="flex cursor-pointer items-center whitespace-nowrap rounded-full px-2.5 py-1 text-[13px] transition-all hover:bg-sky-100 hover:text-sky-700 focus:outline-none data-[state=open]:bg-sky-700 data-[state=open]:text-white">
          Search
          <ChevronDown className="ml-0.5 h-3.5 w-3.5" />
        </DropdownMenu.Trigger>
        <DropdownMenu.Portal>
          <DropdownMenu.Content
            sideOffset={-3}
            onChange={(e) => {
              e.preventDefault();
            }}
            className="z-30 min-w-[600px] max-w-[600px] overflow-hidden rounded-md bg-white shadow-[0px_10px_38px_-10px_rgba(22,_23,_24,_0.35),_0px_10px_20px_-15px_rgba(22,_23,_24,_0.2)]"
          >
            <div className="flex items-center border-b border-slate-200 px-8 pb-6 pt-6 text-sm font-medium text-sky-700 outline-none">
              <div className="flex w-full flex-col items-start justify-center text-[15px] text-slate-600">
                <div className="text-xl font-semibold leading-none tracking-tight text-slate-800">
                  Search Explanations
                </div>
                <span className="mb-2.5 ml-0 pt-1 text-[11px] leading-snug text-slate-500">
                  Find features by searching the text of feature explanations.
                </span>
                <ExplanationsSearcher initialSelectedLayers={[]} showTabs fromNav />
              </div>
            </div>
            <div className="flex items-center px-8 pb-2 pt-7 text-sm font-medium text-sky-700 outline-none">
              <div className="flex w-full flex-col items-start justify-center text-[15px] text-slate-600">
                <div className="text-xl font-semibold leading-none tracking-tight text-slate-800">
                  Search via Inference
                </div>
                <span className="mb-2.5 ml-0 pt-1 text-[11px] leading-snug text-slate-500">
                  Run custom text through a model to find top features that fire.
                </span>
                <InferenceActivationAllProvider>
                  <InferenceSearcher
                    initialModelId={jumpToModelModelId}
                    initialSourceSet={getSourceSetNameFromSource(defaultSource)}
                    initialSelectedLayers={[]}
                    showModels
                    initialSortIndexes={[]}
                    showExamples={false}
                  />
                </InferenceActivationAllProvider>
              </div>
            </div>
            <div className="flex items-center border-t border-slate-200 px-8 pb-4 pt-6 text-sm font-medium text-sky-700 outline-none">
              <div className="flex w-full flex-col items-start justify-center text-[15px] text-slate-600">
                <div className="text-xl font-semibold leading-none tracking-tight text-slate-800">
                  Search TopK by Token
                </div>
                <span className="mb-2.5 ml-0 pt-1 text-[11px] leading-snug text-slate-500">
                  Similar to search via inference, but returns top activating features by token for a single source.
                </span>
                <SearchTopkByToken
                  initialModelId={jumpToModelModelId}
                  initialSource={defaultSource}
                  initialText=""
                  initialIgnoreBos
                  hideSettings
                  showResultsInNewPage
                />
              </div>
            </div>

            <DropdownMenu.Arrow className="fill-white" />
          </DropdownMenu.Content>
        </DropdownMenu.Portal>
      </DropdownMenu.Root>

      <ModelsDropdown isInBreadcrumb={false} />

      <Select.Root
        defaultValue="saes"
        value="saes"
        onValueChange={(newVal) => {
          router.push(`/${newVal.replace('|', '/')}`);
        }}
      >
        <Select.Trigger className="hidden w-full flex-1 select-none flex-row items-center justify-center gap-x-1 whitespace-pre rounded px-0 text-[13px] hover:text-sky-700 hover:underline focus:outline-none sm:pl-1 sm:uppercase">
          <Select.Value />
          <Select.Icon>
            <ChevronDown className="ml-0.5 h-3.5 w-3.5" />
          </Select.Icon>
        </Select.Trigger>
        <Select.Portal>
          <Select.Content
            position="popper"
            align="center"
            sideOffset={0}
            className="z-30 max-h-[650px] cursor-pointer overflow-hidden rounded-md border border-slate-300 bg-white text-xs font-medium text-sky-700 shadow"
          >
            <Select.ScrollUpButton className="flex justify-center border-b bg-slate-100 text-slate-300">
              <ChevronUpIcon className="h-4 w-4" />
            </Select.ScrollUpButton>
            <Select.Viewport className="text-left">
              <Select.Item
                key="saes"
                value="saes"
                className="hidden flex-col overflow-hidden border-b py-2.5 font-sans text-xs text-slate-400 hover:bg-slate-100 focus:outline-none"
              >
                <Select.ItemText>Sparse Autoencoders</Select.ItemText>
              </Select.Item>
              {Object.keys(globalModels).map((modelId) =>
                getSourceSetsForModelId(modelId).map((s) => {
                  if (s.name === NEURONS_SOURCESET) {
                    return <div className="hidden" key={s.name} />;
                  }
                  return (
                    <Select.Item
                      key={s.name}
                      value={`${modelId}|${s.name}`}
                      className="gzap-y-0.5 flex flex-col overflow-hidden border-b px-3 py-2.5 font-sans text-xs text-sky-700 hover:bg-slate-100 focus:outline-none"
                    >
                      <Select.ItemText>
                        <span className="uppercase">
                          {modelId}・{s.name}
                        </span>
                      </Select.ItemText>
                      <div className="mt-0 flex flex-row justify-between gap-x-3 text-[10px] font-normal text-slate-500">
                        <div>{s.type}</div>
                        <div>{s.creatorName}</div>
                      </div>
                    </Select.Item>
                  );
                }),
              )}
              <Link
                href={IS_LOCALHOST ? '/sae/new' : 'https://forms.gle/Yg51TYFutJysiyDP7'}
                target={IS_LOCALHOST ? undefined : '_blank'}
                prefetch={false}
                rel="noreferrer"
                className="col-span-2 flex h-10 flex-row items-center justify-center gap-x-0.5 rounded bg-white px-3 py-1 text-xs text-slate-700 hover:bg-slate-100"
              >
                <Plus className="h-3.5 w-3.5" />
                Add New SAEs
              </Link>
            </Select.Viewport>
            <Select.ScrollDownButton className="flex justify-center border-b bg-slate-100 text-slate-300">
              <ChevronDownIcon className="h-4 w-4" />
            </Select.ScrollDownButton>
          </Select.Content>
        </Select.Portal>
      </Select.Root>

      <Link
        href="/gemma-2-2b/graph"
        className="relative flex cursor-pointer items-center whitespace-nowrap rounded-full px-2.5 py-1 text-[13px] transition-all hover:bg-sky-100 hover:text-sky-700 focus:outline-none data-[state=open]:bg-sky-700 data-[state=open]:text-white"
      >
        Circuit Tracer
        <span className="absolute -right-2 -top-1.5 flex h-3.5 w-8 items-center justify-center rounded-full bg-red-600 text-[8px] font-bold text-white">
          NEW
        </span>
      </Link>

      <Link
        href="/steer"
        prefetch={false}
        rel="noreferrer"
        className="flex cursor-pointer items-center whitespace-nowrap rounded-full px-2.5 py-1 text-[13px] transition-all hover:bg-sky-100 hover:text-sky-700 focus:outline-none data-[state=open]:bg-sky-700 data-[state=open]:text-white"
      >
        Steer
      </Link>

      {!DEMO_MODE && (
        <Link
          href="/sae-bench"
          prefetch={false}
          rel="noreferrer"
          className="flex cursor-pointer items-center whitespace-nowrap rounded-full px-2.5 py-1 text-[13px] transition-all hover:bg-sky-100 hover:text-sky-700 focus:outline-none data-[state=open]:bg-sky-700 data-[state=open]:text-white"
        >
          SAE Evals
        </Link>
      )}

      <Link
        href="https://neuronpedia-datasets.s3.us-east-1.amazonaws.com/index.html?prefix=v1/"
        target="_blank"
        rel="noreferrer"
        className="relative flex cursor-pointer items-center whitespace-nowrap rounded-full px-2.5 py-1 text-[13px] transition-all hover:bg-sky-100 hover:text-sky-700 focus:outline-none data-[state=open]:bg-sky-700 data-[state=open]:text-white"
      >
        Exports
      </Link>

      <Link
        href="https://join.slack.com/t/opensourcemechanistic/shared_invite/zt-3m2fulfeu-0LnVnF8yCrKJYQvWLuCQaQ"
        target="_blank"
        rel="noreferrer"
        className="flex cursor-pointer items-center whitespace-nowrap rounded-full px-2.5 py-1 text-[13px] transition-all hover:bg-sky-100 hover:text-sky-700 focus:outline-none data-[state=open]:bg-sky-700 data-[state=open]:text-white"
      >
        Slack
      </Link>
      <Link
        href="/blog"
        className="relative flex cursor-pointer items-center whitespace-nowrap rounded-full bg-sky-100 px-4 py-1 text-[13px] font-medium text-sky-700 transition-all hover:bg-sky-200 hover:text-sky-700 focus:outline-none data-[state=open]:bg-sky-700 data-[state=open]:text-white"
      >
        Blog
      </Link>

      <Link
        href="/privacy"
        className="flex cursor-pointer items-center whitespace-nowrap px-0 py-0.5 text-[13px] transition-all hover:text-sky-700 hover:underline sm:mb-0 sm:hidden sm:px-1 sm:py-0"
      >
        Privacy & Terms
      </Link>
      <Link
        href="/contact"
        className="flex cursor-pointer items-center whitespace-nowrap px-0 py-0.5 text-[13px] transition-all hover:text-sky-700 hover:underline sm:mb-0 sm:hidden sm:px-1 sm:py-0"
      >
        Contact
      </Link>
    </>
  );
}
