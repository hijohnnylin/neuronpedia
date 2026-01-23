'use client';

import JumpToPane from '@/components/panes/jump-to-pane';
import SearchExplanationsPane from '@/components/panes/search-explanations-pane';
import SearchInferenceReleasePane from '@/components/panes/search-inference-release-pane';
import { useGlobalContext } from '@/components/provider/global-provider';
import { getDefaultSourceSetAndSourceForRelease } from '@/lib/utils/source';
import * as Tabs from '@radix-ui/react-tabs';
import { ArrowUpRight, BellDot, CircuitBoard, Home, Notebook, RowsIcon, ScrollText, Shield, Smile } from 'lucide-react';
import { useSession } from 'next-auth/react';
import Link from 'next/link';
import { SourceReleaseWithPartialRelations } from 'prisma/generated/zod';
import React, { useEffect, useState } from 'react';
import TabCircuit from './tab-circuit';
import TabMain, { blogPostLink, codingTutorialLink, hfLink } from './tab-main';
import TabSafety from './tab-safety';

export default function GemmaScope2Home({ release }: { release: SourceReleaseWithPartialRelations }) {
  const session = useSession();
  const { setSignInModalOpen } = useGlobalContext();

  const defaultModelIdNonSteer = 'gemma-3-4b-it';

  const { defaultSourceSet, defaultSource } = getDefaultSourceSetAndSourceForRelease(release);

  function getDefaultTabFromHash(hash: string) {
    if (hash === '#main') {
      return 'main';
    }
    if (hash === '#safety') {
      return 'safety';
    }
    if (hash === '#circuit') {
      return 'circuit';
    }
    if (hash === '#browse') {
      return 'browse';
    }
    return 'main';
  }

  const [tabValue, setTabValue] = useState(
    typeof window !== 'undefined' ? getDefaultTabFromHash(window.location.hash) : 'main',
  );

  const mainRef = React.createRef<HTMLDivElement>();
  const safetyRef = React.createRef<HTMLDivElement>();
  const circuitRef = React.createRef<HTMLDivElement>();
  const browseRef = React.createRef<HTMLDivElement>();

  useEffect(() => {
    if (tabValue === 'main') {
      window.history.pushState({}, '', '#main');
      mainRef.current?.scroll({
        behavior: 'instant',
        top: 0,
      });
    } else if (tabValue === 'safety') {
      window.history.pushState({}, '', '#safety');

      safetyRef.current?.scroll({
        behavior: 'instant',
        top: 0,
      });
    } else if (tabValue === 'circuit') {
      window.history.pushState({}, '', '#circuit');
      circuitRef.current?.scroll({
        behavior: 'instant',
        top: 0,
      });
    } else if (tabValue === 'browse') {
      window.history.pushState({}, '', '#browse');
      browseRef.current?.scroll({
        behavior: 'instant',
        top: 0,
      });
    }
  }, [tabValue]);

  return (
    <div className="flex h-full w-full flex-col items-start pb-0 pt-0 text-slate-700">
      <div className="sticky top-12 z-20 flex w-full flex-row items-center justify-between border-b border-slate-200 bg-white px-4 pb-2 sm:px-6 sm:pb-3 sm:pt-1">
        <div className="flex w-full flex-col gap-x-3 gap-y-1 sm:flex-row sm:items-center sm:justify-between">
          <button
            type="button"
            onClick={() => {
              setTabValue('main');
            }}
            className="mt-0 flex flex-col text-center font-sans text-[14px] font-bold text-slate-800 sm:hidden sm:text-center sm:text-[18px] sm:leading-none"
          >
            {release.descriptionShort}
          </button>
          <div className="mt-0 hidden flex-col text-center font-sans text-[12px] font-bold text-slate-600 sm:flex sm:text-left sm:text-[18px] sm:leading-none">
            {release.description}
          </div>
          <div className="flex flex-row justify-start text-left font-sans text-[10px] font-medium leading-none text-slate-500 sm:justify-end sm:text-left sm:text-[14px]">
            {release.creatorName}
            <span className="hidden sm:ml-1 sm:flex">
              ·{' '}
              {release.createdAt.toLocaleDateString('en-US', {
                year: 'numeric',
                month: 'long',
              })}
              {release?.creatorEmail && (
                <>
                  {' '}
                  ·{' '}
                  <Link
                    className="ml-1 text-sky-700"
                    href={`mailto:${
                      release?.creatorEmail
                    }?subject=SAE%20Detail%20Request%3A%20${release.name.toUpperCase()}&body=I'd%20like%20to%20contact%20the%20researcher%20who%20created%20the%20${release.name.toUpperCase()}%20SAEs.%20Please%20put%20me%20in%20touch%20-%20thanks!`}
                  >
                    Admin Support Contact
                  </Link>
                </>
              )}
            </span>
          </div>
        </div>
      </div>
      <Tabs.Root
        className="flex h-full w-full flex-row border-b border-slate-200 bg-white sm:h-[calc(100vh-48px-43px-26px)] sm:max-h-[calc(100vh-48px-43px-26px)] sm:min-h-[calc(100vh-48px-43px-26px)]"
        defaultValue={tabValue}
        value={tabValue}
        onValueChange={(v) => {
          setTabValue(v);
        }}
        orientation="vertical"
      >
        <Tabs.List
          className="hidden w-[230px] min-w-[230px] max-w-[230px] flex-col border-0 border-slate-200 bg-slate-50 py-2 text-left text-sm font-semibold text-slate-400 sm:flex"
          aria-label="Sidebar for Navigation"
        >
          <Tabs.Trigger
            className="group relative flex flex-row items-center gap-x-2 px-1 py-1 text-left transition-all data-[state=active]:pl-5 data-[state=active]:text-sky-700 hover:data-[state=inactive]:bg-slate-100 hover:data-[state=inactive]:text-slate-500"
            value="main"
          >
            <div className="flex min-w-[160px] flex-row gap-x-2 rounded-full px-4 py-2.5 transition-all group-data-[state=active]:bg-slate-600 group-data-[state=active]:text-white">
              <Home className="h-5 w-5" /> Introduction{' '}
            </div>
          </Tabs.Trigger>
          <Tabs.Trigger
            className="group relative flex flex-row items-center gap-x-2 px-1 py-1 text-left transition-all data-[state=active]:pl-5 data-[state=active]:text-sky-700 hover:data-[state=inactive]:bg-slate-100 hover:data-[state=inactive]:text-slate-500"
            value="safety"
          >
            <div className="flex min-w-[160px] flex-row gap-x-2 rounded-full px-4 py-2.5 transition-all group-data-[state=active]:bg-gGreen group-data-[state=active]:text-white">
              <Shield className="h-5 w-5" /> Safety Features{' '}
            </div>
          </Tabs.Trigger>
          <Tabs.Trigger
            className="group flex flex-row items-center gap-x-2 px-1 py-1 text-left transition-all data-[state=active]:pl-5 data-[state=active]:text-sky-700 hover:data-[state=inactive]:bg-slate-100 hover:data-[state=inactive]:text-slate-500"
            value="circuit"
          >
            <div className="flex min-w-[160px] flex-row gap-x-2 whitespace-nowrap rounded-full px-4 py-2.5 transition-all group-data-[state=active]:bg-gYellow group-data-[state=active]:text-white">
              <CircuitBoard className="h-5 w-5" /> Circuit Tracing{' '}
            </div>
          </Tabs.Trigger>
          <Tabs.Trigger
            className="group flex flex-row items-center gap-x-2 px-1 py-1 text-left transition-all data-[state=active]:pl-5 data-[state=active]:text-indigo-600 hover:data-[state=inactive]:bg-slate-100 hover:data-[state=inactive]:text-slate-500"
            value="browse"
          >
            <button
              type="button"
              onClick={() => {
                setTabValue('browse');
              }}
              className="flex min-w-[160px] flex-row gap-x-2 whitespace-nowrap rounded-full px-4 py-2.5 transition-all group-data-[state=active]:bg-gBlue group-data-[state=active]:text-white"
            >
              <RowsIcon className="h-5 w-5" />
              Browse & Search
            </button>
          </Tabs.Trigger>

          {!session.data?.user && (
            <button
              type="button"
              onClick={() => {
                setSignInModalOpen(true);
              }}
              className="group flex flex-row items-center gap-x-2 px-1 py-1 text-left transition-all data-[state=active]:pl-5 data-[state=active]:text-sky-700 hover:data-[state=inactive]:bg-slate-100 hover:data-[state=inactive]:text-slate-500"
            >
              <div className="flex flex-row gap-x-2 rounded-full px-4 py-2.5 text-gYellow transition-all group-data-[state=active]:bg-gYellow group-data-[state=active]:text-white">
                <BellDot className="h-5 w-5" />
                Get Notified
              </div>
            </button>
          )}
          <a
            href={codingTutorialLink}
            target="_blank"
            rel="noreferrer"
            className="group flex flex-row items-center gap-x-2 px-1 py-1 text-left transition-all data-[state=active]:pl-5 data-[state=active]:text-sky-700 hover:data-[state=inactive]:bg-slate-100 hover:data-[state=inactive]:text-slate-500"
          >
            <div className="flex flex-row gap-x-2 rounded-full px-4 py-2.5 transition-all group-data-[state=active]:bg-gGreen group-data-[state=active]:text-white">
              <Notebook className="h-5 w-5" />
              Tutorial Notebook <ArrowUpRight className="h-4 w-4" />
            </div>
          </a>
          <a
            href={hfLink}
            target="_blank"
            rel="noreferrer"
            className="group flex flex-row items-center gap-x-2 px-1 py-1 text-left transition-all data-[state=active]:pl-5 data-[state=active]:text-sky-700 hover:data-[state=inactive]:bg-slate-100 hover:data-[state=inactive]:text-slate-500"
          >
            <div className="flex flex-row gap-x-2 rounded-full px-4 py-2.5 transition-all group-data-[state=active]:bg-gGreen group-data-[state=active]:text-white">
              <Smile className="h-5 w-5" />
              HuggingFace <ArrowUpRight className="h-4 w-4" />
            </div>
          </a>
          <a
            href={blogPostLink}
            target="_blank"
            rel="noreferrer"
            className="group flex flex-row items-center gap-x-2 px-1 py-1 text-left transition-all data-[state=active]:pl-5 data-[state=active]:text-sky-700 hover:data-[state=inactive]:bg-slate-100 hover:data-[state=inactive]:text-slate-500"
          >
            <div className="flex flex-row gap-x-2 rounded-full px-4 py-2.5 transition-all group-data-[state=active]:bg-gGreen group-data-[state=active]:text-white">
              <ScrollText className="h-5 w-5" />
              DeepMind Blog <ArrowUpRight className="h-4 w-4" />
            </div>
          </a>
        </Tabs.List>
        <Tabs.Content className="h-full w-full" value="main">
          <div
            ref={mainRef}
            className="flex h-full w-full flex-row items-start justify-center overflow-y-scroll border-slate-400 bg-white px-0 py-5 pb-1"
          >
            {/* eslint-disable-next-line */}
            <TabMain tabUpdater={setTabValue} />
          </div>
        </Tabs.Content>

        <Tabs.Content className="h-full w-full" value="safety">
          <div
            ref={safetyRef}
            className="flex h-full w-full flex-col items-center overflow-y-scroll border-slate-400 bg-white px-0 pb-1"
          >
            <TabSafety tabUpdater={setTabValue} />
          </div>
        </Tabs.Content>

        <Tabs.Content ref={circuitRef} className="h-full w-full flex-col bg-white" value="circuit">
          <div className="flex h-full w-full flex-col items-center overflow-y-scroll bg-white px-0 pb-1">
            {/* eslint-disable-next-line */}
            <TabCircuit tabUpdater={setTabValue} />
          </div>
        </Tabs.Content>

        <Tabs.Content ref={browseRef} className="h-full w-full flex-col bg-white" value="browse">
          <div className="flex h-full w-full flex-col items-center overflow-y-scroll border-l bg-slate-50 px-0 py-0">
            <div className="mb-8 mt-5 flex w-full max-w-screen-lg flex-col items-center">
              <JumpToPane
                release={release}
                defaultModelId={defaultSourceSet?.modelId || defaultModelIdNonSteer}
                defaultSourceSetName={defaultSourceSet?.name || ''}
                defaultSourceId={defaultSource?.id || ''}
                vertical
              />

              <div className="mt-6 w-full">
                <SearchExplanationsPane
                  initialModelId={defaultSourceSet?.modelId || defaultModelIdNonSteer}
                  initialSourceSetName={defaultSourceSet?.name || ''}
                  filterToRelease={release}
                />
              </div>

              <div className="mt-6 w-full">
                <SearchInferenceReleasePane release={release} />
              </div>
            </div>
          </div>
        </Tabs.Content>
      </Tabs.Root>
    </div>
  );
}
