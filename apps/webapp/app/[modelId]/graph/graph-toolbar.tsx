import { CLTGraph, FilterGraphType, getGraphBaseUrlToName } from '@/app/[modelId]/graph/utils';
import { useGlobalContext } from '@/components/provider/global-provider';
import { useGraphContext } from '@/components/provider/graph-provider';
import { Button } from '@/components/shadcn/button';
import * as Select from '@radix-ui/react-select';
import * as ToggleGroup from '@radix-ui/react-toggle-group';
import { ChevronDownIcon, ChevronUpIcon, ExternalLinkIcon, Share2, Trash, UploadCloud } from 'lucide-react';
import { useSession } from 'next-auth/react';
import { useRouter } from 'next-nprogress-bar';
import { useSearchParams } from 'next/navigation';
import { Fragment, useState } from 'react';
import GenerateGraphModal from './generate-graph-modal';
import GraphInfoModal from './graph-info-modal';
import UploadGraphModal from './upload-graph-modal';

export default function GraphToolbar() {
  const session = useSession();
  const searchParams = useSearchParams();
  const isEmbed = searchParams.get('embed') === 'true';
  const { setSignInModalOpen } = useGlobalContext();
  const router = useRouter();
  const [isDeleting, setIsDeleting] = useState(false);
  const {
    modelIdToMetadataMap,
    selectedModelId,
    selectedMetadataGraph,
    selectedGraph,
    setSelectedMetadataGraph,
    modelIdToModelDisplayName,
    filterGraphsSetting,
    setFilterGraphsSetting,
    shouldShowGraphToCurrentUser,
    setIsCopyModalOpen,
  } = useGraphContext();

  if (isEmbed) {
    return (
      <div className="flex w-full flex-col pt-1">
        <div className="flex w-full flex-row items-center justify-between gap-x-2 text-sm">
          <div className="flex flex-row items-center justify-center gap-x-2 text-xs text-slate-700">
            <Button
              variant="outline"
              onClick={() => {
                const url = new URL(window.location.href);
                url.searchParams.delete('embed');
                window.open(url.toString(), '_blank');
              }}
              className="h-7 gap-x-2 px-2.5 py-0 font-mono text-xs font-medium leading-snug text-sky-700 hover:border-sky-500 hover:bg-sky-100 hover:text-sky-800"
            >
              <span className="rounded-sm bg-slate-100 px-1.5 py-0.5 text-[10px] font-bold text-slate-600">
                {modelIdToModelDisplayName.get(selectedModelId)}
              </span>
              <span className="px-1">{selectedMetadataGraph?.slug}</span>
              <ExternalLinkIcon className="h-3.5 w-3.5" />
            </Button>
            <div
              title={selectedMetadataGraph?.prompt}
              className="flex-1 overflow-hidden text-ellipsis whitespace-nowrap"
            >
              {selectedMetadataGraph?.prompt}
            </div>
          </div>
          <Button
            variant="outline"
            size="sm"
            title="Share Graph, Subgraph, and Custom Labels"
            aria-label="Share Graph Subgraph, and Custom Labels"
            className="flex h-8 w-8 items-center justify-center whitespace-nowrap border-slate-300 px-0 text-sm text-slate-500 hover:bg-slate-50"
            onClick={() => {
              setIsCopyModalOpen(true);
            }}
            disabled={selectedMetadataGraph === null}
          >
            <Share2 className="h-4 w-4" />
          </Button>
        </div>
      </div>
    );
  }

  function makeCreatorNameFromGraph(graph: CLTGraph) {
    if (graph.metadata.info?.creator_name) {
      return graph.metadata.info?.creator_name;
    }
    if (selectedMetadataGraph?.user?.name) {
      return selectedMetadataGraph?.user.name;
    }
    if (selectedMetadataGraph?.url) {
      return getGraphBaseUrlToName(selectedMetadataGraph.url);
    }
    return 'Anonymous';
  }

  return (
    <div className="flex w-full flex-col pt-2.5">
      <div className="flex w-full flex-row items-end gap-x-2">
        <GenerateGraphModal />

        <div className="flex flex-col">
          <div className="w-full pb-0.5 text-center text-[9px] font-medium uppercase text-slate-400">Select Model</div>
          <Select.Root
            value={selectedModelId}
            onValueChange={(newVal) => {
              window.location.href = `/${newVal}/graph`;
            }}
          >
            <Select.Trigger
              onKeyDown={(e) => {
                if (e.key === 'g') {
                  e.preventDefault();
                  e.stopPropagation();
                }
              }}
              className="inline-flex h-12 w-52 max-w-52 items-center justify-between gap-1 rounded border border-slate-300 bg-white px-4 py-2 text-sm leading-none focus:outline-none focus:ring-0"
            >
              <Select.Value>
                {modelIdToModelDisplayName.get(selectedModelId) ? (
                  <div className="flex flex-col items-start justify-start gap-y-0.5 text-left">
                    <div className="text-xs font-medium text-slate-600">
                      {modelIdToModelDisplayName.get(selectedModelId)}
                    </div>
                    <div className="text-[9px] font-normal text-slate-400">{selectedModelId}</div>
                  </div>
                ) : (
                  <div className="text-slate-400">Select a model</div>
                )}
              </Select.Value>
              <Select.Icon>
                <ChevronDownIcon className="w-5 text-slate-500" />
              </Select.Icon>
            </Select.Trigger>
            <Select.Portal>
              <Select.Content
                position="popper"
                align="center"
                sideOffset={3}
                className="z-[99999] max-h-[400px] overflow-hidden rounded-md border bg-white shadow-lg"
              >
                <Select.ScrollUpButton className="flex h-7 cursor-pointer items-center justify-center bg-white text-slate-700 hover:bg-slate-100">
                  <ChevronUpIcon className="w-5 text-slate-500" />
                </Select.ScrollUpButton>
                <Select.Viewport className="w-full divide-y divide-slate-100 p-2 text-slate-700">
                  {Object.keys(modelIdToMetadataMap).map((modelId) => (
                    <Select.Item
                      key={modelId}
                      value={modelId}
                      className="relative flex h-12 w-full cursor-pointer select-none items-center overflow-x-hidden whitespace-pre rounded py-0 pl-4 pr-6 text-xs hover:bg-slate-100 data-[highlighted]:bg-slate-100 data-[highlighted]:outline-none"
                    >
                      <Select.ItemText className="w-full">
                        <div className="flex w-full flex-col items-start justify-start gap-y-0">
                          <div className="w-full truncate text-left">{modelIdToModelDisplayName.get(modelId)}</div>
                          <div className="w-full text-[9px] font-normal text-slate-400">{modelId}</div>
                        </div>
                      </Select.ItemText>
                    </Select.Item>
                  ))}
                </Select.Viewport>
                <Select.ScrollDownButton className="flex h-7 cursor-pointer items-center justify-center bg-white text-slate-700 hover:bg-slate-100">
                  <ChevronDownIcon className="w-5 text-slate-500" />
                </Select.ScrollDownButton>
              </Select.Content>
            </Select.Portal>
          </Select.Root>
        </div>
        <div className="hidden flex-col">
          <div className="w-full pb-0.5 text-center text-[9px] font-medium uppercase text-slate-400">Filter Graphs</div>
          <ToggleGroup.Root
            type="multiple"
            value={filterGraphsSetting}
            onValueChange={(value) => {
              if (value) setFilterGraphsSetting(value as FilterGraphType[]);
            }}
            className="flex h-12 rounded border border-sky-600 bg-slate-50"
          >
            <ToggleGroup.Item
              value={FilterGraphType.Featured}
              aria-label="Only shows your graphs and featured graphs."
              className="w-[86px] rounded-l border-r border-sky-600 px-2.5 text-xs leading-none text-slate-400 data-[state=on]:bg-sky-100 data-[state=on]:text-sky-700 data-[state=off]:hover:bg-slate-100"
            >
              Featured
            </ToggleGroup.Item>

            <ToggleGroup.Item
              value={FilterGraphType.Mine}
              aria-label="Shows only your uploaded graphs."
              className="w-[86px] rounded-r border-sky-600 px-2.5 text-xs text-slate-400 data-[state=on]:bg-sky-100 data-[state=on]:text-sky-700 data-[state=off]:hover:bg-slate-100"
            >
              Mine
            </ToggleGroup.Item>
            {/* <ToggleGroup.Item
              aria-label="Shows all graphs, including other user uploaded graphs."
              value={FilterGraphType.Community}
              className="w-[86px] rounded-r border-sky-600 px-2.5 text-xs text-slate-400 data-[state=on]:bg-sky-100 data-[state=on]:text-sky-700 data-[state=off]:hover:bg-slate-100"
            >
              Community
            </ToggleGroup.Item> */}
          </ToggleGroup.Root>
        </div>
        <div className="flex min-w-0 flex-1 flex-col">
          <div className="w-full pb-0.5 text-center text-[9px] font-medium uppercase text-slate-400">Select Graph</div>
          <Select.Root
            value={selectedMetadataGraph?.slug}
            onValueChange={(newVal) => {
              const newGraph = modelIdToMetadataMap[selectedModelId]?.find((graph) => graph.slug === newVal);
              if (newGraph) {
                setSelectedMetadataGraph(newGraph);
              }
            }}
          >
            <Select.Trigger
              onKeyDown={(e) => {
                if (e.key === 'g') {
                  e.preventDefault();
                  e.stopPropagation();
                }
              }}
              className="relative inline-flex h-12 max-w-full items-center justify-between gap-1 overflow-x-hidden rounded border border-slate-300 bg-white px-4 py-0 pr-10 text-sm leading-none focus:outline-none focus:ring-0"
            >
              {selectedMetadataGraph !== null ? (
                <Select.Value asChild>
                  <div className="flex w-full flex-col items-start justify-start gap-y-2 overflow-y-visible">
                    <div className="flex w-full flex-row items-center justify-between">
                      <div className="font-mono text-[12px] font-medium text-sky-700">{selectedMetadataGraph.slug}</div>
                    </div>
                    <div className="text-overflow-ellipsis whitespace-nowrap text-[10px] font-normal leading-none text-slate-500">
                      {/* <span className="rounded bg-slate-200 px-1 py-0.5 text-[8px] font-bold text-slate-600">
                        PROMPT
                      </span>{' '} */}
                      {selectedMetadataGraph.promptTokens.map((token, i) => (
                        <span
                          key={`${token}-${i}`}
                          className="mx-0.5 rounded bg-slate-100 px-[3px] py-0.5 font-mono text-slate-700"
                        >
                          {token.replaceAll('\n', ' ').replaceAll(' ', '\u00A0')}
                        </span>
                      ))}
                    </div>
                  </div>
                </Select.Value>
              ) : (
                <div className="w-full flex-1 flex-col items-center justify-center text-slate-400">Select a Prompt</div>
              )}
              <Select.Icon className="justfiy-center absolute right-0 flex h-[100%] w-8 items-center bg-white">
                <ChevronDownIcon className="ml-1 w-5 text-slate-500" />
              </Select.Icon>
            </Select.Trigger>
            <Select.Portal>
              <Select.Content
                position="popper"
                align="start"
                sideOffset={3}
                className="z-[99999] max-h-[640px] w-full max-w-[800px] overflow-hidden overscroll-y-none rounded-md border bg-white shadow-lg"
              >
                <Select.Viewport className="w-full p-0 text-slate-700">
                  {(() => {
                    const allVisibleGraphs =
                      modelIdToMetadataMap[selectedModelId]?.filter((graph) => shouldShowGraphToCurrentUser(graph)) ??
                      [];

                    const myGraphs = allVisibleGraphs.filter((graph) => session.data?.user?.id === graph.userId);
                    const featuredGraphs = allVisibleGraphs.filter(
                      (graph) => graph.isFeatured && session.data?.user?.id !== graph.userId,
                    );
                    const otherSelectedGraph =
                      selectedMetadataGraph &&
                      !myGraphs.some(
                        (graph) =>
                          graph.slug === selectedMetadataGraph.slug && graph.modelId === selectedMetadataGraph.modelId,
                      ) &&
                      !featuredGraphs.some(
                        (graph) =>
                          graph.slug === selectedMetadataGraph.slug && graph.modelId === selectedMetadataGraph.modelId,
                      )
                        ? selectedMetadataGraph
                        : null;
                    const communityGraphs = allVisibleGraphs.filter(
                      (graph) => !graph.isFeatured && session.data?.user?.id !== graph.userId,
                    );

                    let graphsDisplayedCount = 0;
                    if (filterGraphsSetting.includes(FilterGraphType.Mine)) graphsDisplayedCount += myGraphs.length;
                    if (filterGraphsSetting.includes(FilterGraphType.Featured))
                      graphsDisplayedCount += featuredGraphs.length;
                    if (filterGraphsSetting.includes(FilterGraphType.Community))
                      graphsDisplayedCount += communityGraphs.length;

                    const renderGraphItem = (graph: (typeof allVisibleGraphs)[0], isMyGraph: boolean) => (
                      <div className="relative flex w-full flex-row items-center hover:bg-sky-100">
                        <Select.Item
                          key={graph.slug}
                          value={graph.slug}
                          className="group relative flex w-full cursor-pointer select-none items-center overflow-x-hidden py-2.5 pl-4 pr-4 text-xs outline-none hover:bg-slate-100 data-[highlighted]:bg-sky-50"
                        >
                          <Select.ItemText className="w-full min-w-full" asChild>
                            <div className="flex w-full min-w-full flex-col items-start justify-start gap-y-0">
                              <div className="flex w-full flex-row items-center justify-between">
                                <div className="font-mono text-[12px] font-medium text-sky-700">{graph.slug}</div>
                                {!isMyGraph && (
                                  <div className="mr-0 flex flex-row items-center gap-x-2">
                                    <div className="text-[10px] font-normal text-slate-500">
                                      {graph.user?.name
                                        ? graph.user?.name
                                        : getGraphBaseUrlToName(graph.url) || 'Anonymous'}
                                    </div>
                                  </div>
                                )}
                              </div>
                              <div className="mt-2 w-full whitespace-pre-line pl-0 text-[10px] leading-tight text-slate-500">
                                <div className="flex flex-wrap">
                                  {graph.promptTokens.map((token, i) => (
                                    <Fragment key={`${token}-${i}`}>
                                      <span className="mx-[1px] mb-1 rounded bg-slate-100 px-[2px] py-0.5 font-mono text-slate-700 group-hover:bg-sky-200 group-hover:text-sky-700 group-data-[highlighted]:bg-sky-200 group-data-[highlighted]:text-sky-700">
                                        {token.replaceAll(' ', '\u00A0')}
                                      </span>
                                      {(token === '⏎' || token === '⏎⏎') && <div className="w-full" />}
                                    </Fragment>
                                  ))}
                                </div>
                              </div>
                            </div>
                          </Select.ItemText>
                        </Select.Item>
                        {isMyGraph && (
                          <Button
                            variant="ghost"
                            size="sm"
                            className="absolute right-1.5 top-1.5 h-6 w-6 p-0 text-red-500 hover:bg-red-100 hover:text-red-700"
                            onClick={async (e) => {
                              e.preventDefault();
                              e.stopPropagation();

                              // eslint-disable-next-line
                              if (confirm(`Are you sure you want to delete graph "${graph.slug}"?`)) {
                                try {
                                  setIsDeleting(true);
                                  const res = await fetch('/api/graph/delete', {
                                    method: 'POST',
                                    headers: { 'Content-Type': 'application/json' },
                                    body: JSON.stringify({ modelId: selectedModelId, slug: graph.slug }),
                                  });
                                  if (res.ok) {
                                    // Reload to the first available graph
                                    const remainingGraphs = allVisibleGraphs.filter((g) => g.slug !== graph.slug);
                                    if (remainingGraphs.length > 0) {
                                      const nextGraph = remainingGraphs.sort((a, b) => a.slug.localeCompare(b.slug))[0];
                                      router.replace(`/${selectedModelId}/graph?slug=${nextGraph.slug}`);
                                      setSelectedMetadataGraph(nextGraph);
                                    } else {
                                      // find first graph in any model
                                      const anyModelId = Object.keys(modelIdToMetadataMap).find(
                                        (mId) => modelIdToMetadataMap[mId]?.length > 0,
                                      );
                                      if (anyModelId) {
                                        const nextGraph = modelIdToMetadataMap[anyModelId]!.sort((a, b) =>
                                          a.slug.localeCompare(b.slug),
                                        )[0];
                                        router.replace(`/${anyModelId}/graph?slug=${nextGraph.slug}`);
                                        setSelectedMetadataGraph(nextGraph);
                                      } else {
                                        router.replace(`${anyModelId}/graph`);
                                      }
                                    }
                                    window.location.reload();
                                  } else {
                                    const errorData = await res.json();
                                    alert(`Failed to delete graph: ${errorData.message || 'Unknown error'}`);
                                  }
                                } catch (error) {
                                  console.error('Error deleting graph:', error);
                                  alert('An unexpected error occurred while deleting the graph.');
                                } finally {
                                  setIsDeleting(false);
                                }
                              }
                            }}
                          >
                            <Trash className="h-3.5 w-3.5" />
                          </Button>
                        )}
                      </div>
                    );

                    return (
                      <>
                        {otherSelectedGraph && (
                          <Select.Group className="divide-y divide-slate-200">
                            {renderGraphItem(otherSelectedGraph, session.data?.user?.id === otherSelectedGraph.userId)}
                          </Select.Group>
                        )}
                        {filterGraphsSetting.includes(FilterGraphType.Mine) && myGraphs.length > 0 && (
                          <Select.Group className="divide-y divide-slate-200">
                            <Select.Label className="sticky top-0 z-10 border-b border-t border-slate-100 bg-slate-50 py-2 pl-4 pr-6 pt-2.5 text-center text-xs font-bold text-slate-500">
                              My Graphs
                            </Select.Label>
                            {myGraphs.sort((a, b) => a.slug.localeCompare(b.slug)).map((g) => renderGraphItem(g, true))}
                          </Select.Group>
                        )}
                        {filterGraphsSetting.includes(FilterGraphType.Featured) && featuredGraphs.length > 0 && (
                          <Select.Group className="divide-y divide-slate-200">
                            <Select.Label className="sticky top-0 z-10 border-b border-t border-slate-100 bg-slate-50 py-2 pl-4 pr-6 pt-2.5 text-center text-xs font-bold text-slate-500">
                              Featured Graphs
                            </Select.Label>
                            {featuredGraphs
                              .sort((a, b) => a.slug.localeCompare(b.slug))
                              .map((g) => renderGraphItem(g, session.data?.user?.id === g.userId))}
                          </Select.Group>
                        )}
                        {/* {filterGraphsSetting.includes(FilterGraphType.Community) && communityGraphs.length > 0 && (
                          <Select.Group className="divide-y divide-slate-200">
                            <Select.Label className="sticky top-0 z-10 border-b border-t border-slate-100 bg-slate-50 py-2 pl-4 pr-6 pt-2.5 text-center text-xs font-bold text-slate-500">
                              Community-Submitted Graphs
                            </Select.Label>
                            {communityGraphs
                              .sort((a, b) => a.slug.localeCompare(b.slug))
                              .map((g) => renderGraphItem(g, session.data?.user?.id === g.userId))}
                          </Select.Group>
                        )} */}
                        {graphsDisplayedCount === 0 && (
                          <div className="relative w-full cursor-default select-none px-5 py-3 text-center text-sm text-slate-400">
                            No graphs matched your filters. Include more filters on the left.
                          </div>
                        )}
                      </>
                    );
                  })()}
                </Select.Viewport>
                <Select.ScrollDownButton className="flex h-7 cursor-pointer items-center justify-center bg-white text-slate-700 hover:bg-slate-100">
                  <ChevronDownIcon className="w-5 text-slate-500" />
                </Select.ScrollDownButton>
              </Select.Content>
            </Select.Portal>
          </Select.Root>
        </div>
        <div className="flex flex-col">
          <div className="w-full pb-0.5 text-center text-[9px] font-medium uppercase text-slate-400">Tools</div>
          <div className="flex flex-row gap-x-2">
            <Button
              variant="outline"
              size="sm"
              disabled={!selectedGraph}
              onClick={() => {
                if (selectedGraph?.metadata.info?.creator_url) {
                  if (
                    // eslint-disable-next-line
                    confirm(
                      `Confirm navigation to ${selectedGraph.metadata.info?.creator_name ? selectedGraph.metadata.info?.creator_name : 'the creator'}'s website.\n\n${selectedGraph.metadata.info.creator_url}.`,
                    )
                  ) {
                    window.open(selectedGraph.metadata.info.creator_url, '_blank');
                  }
                } else if (selectedGraph) {
                  alert(`This graph was generated by ${makeCreatorNameFromGraph(selectedGraph)}.`);
                }
              }}
              className="group flex h-12 w-32 max-w-32 flex-col items-center justify-center gap-x-2 gap-y-[5px] overflow-x-clip border-slate-200 bg-slate-200 px-0 text-xs leading-none text-slate-500 transition-none hover:border-sky-200 hover:bg-sky-200"
            >
              <div className="text-[8px] font-semibold leading-none text-slate-400 group-hover:text-sky-600">
                GENERATED BY
              </div>
              <div className="text-center text-[11px] leading-none text-sky-700 group-hover:text-sky-800">
                {selectedGraph ? makeCreatorNameFromGraph(selectedGraph) : 'Loading...'}
              </div>
            </Button>
            <GraphInfoModal cltGraph={selectedGraph} selectedMetadataGraph={selectedMetadataGraph} />
            {session.data?.user ? (
              <UploadGraphModal />
            ) : (
              <Button
                variant="outline"
                size="sm"
                className="flex h-12 items-center justify-center gap-x-2 border-slate-300 text-xs"
                onClick={() => {
                  setSignInModalOpen(true);
                }}
              >
                <UploadCloud className="h-4 w-4" />
                Upload
              </Button>
            )}
            <Button
              variant="outline"
              size="sm"
              title="Share Graph, Subgraph, and Custom Labels"
              aria-label="Share Graph Subgraph, and Custom Labels"
              className="flex h-12 items-center justify-center gap-x-2 whitespace-nowrap border-slate-300 text-xs text-slate-500 hover:bg-slate-50"
              onClick={() => {
                setIsCopyModalOpen(true);
              }}
              disabled={selectedMetadataGraph === null || !selectedGraph}
            >
              <Share2 className="h-4 w-4" />
              Share & Embed
            </Button>
          </div>
        </div>
      </div>
      {isDeleting && (
        <div className="fixed inset-0 z-[999999] flex items-center justify-center bg-slate-800/50">
          <div className="cursor-default select-none rounded-lg bg-white p-6 text-lg font-semibold text-slate-700 shadow-xl">
            Deleting Graph...
          </div>
        </div>
      )}
    </div>
  );
}
