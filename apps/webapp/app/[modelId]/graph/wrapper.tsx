'use client';

import { GraphModalProvider } from '@/components/provider/graph-modal-provider';
import { useGraphContext } from '@/components/provider/graph-provider';
import { LoadingSquare } from '@/components/svg/loading-square';
import { useSearchParams } from 'next/navigation';
import GraphFeatureDetail from './feature-detail';
import GenerateGraphModal from './generate-graph-modal';
import GraphToolbar from './graph-toolbar';
import LinkGraph from './link-graph';
import CopyModal from './modals/copy-modal';
import LoadSubgraphModal from './modals/load-subgraph-modal';
import SaveSubgraphModal from './modals/save-subgraph-modal';
import SteerModal from './modals/steer-modal';
import WelcomeModal from './modals/welcome-modal';
import GraphNodeConnections from './node-connections';
import Subgraph from './subgraph';

export default function GraphWrapper({ hasSlug }: { hasSlug: boolean }) {
  const { isLoadingGraphData, selectedMetadataGraph, loadingGraphLabel } = useGraphContext();

  const searchParams = useSearchParams();
  const isEmbed = searchParams.get('embed') === 'true';
  return (
    <GraphModalProvider>
      <div
        className={`${isEmbed ? 'h-[calc(100%_-_20px)] max-h-screen min-h-[calc(100%_-_20px)]' : 'h-[calc(100vh_-_75px)] max-h-[calc(100vh_-_75px)] min-h-[calc(100vh_-_75px)]'} flex w-full flex-col justify-center px-1 text-slate-700 sm:px-4`}
      >
        <div className="flex w-full flex-1 flex-col items-center justify-center overflow-hidden">
          {/* <div>{JSON.stringify(visState)}</div> */}
          <div className="flex w-full flex-col">
            <GraphToolbar />
          </div>

          <div className="w-full flex-1 overflow-hidden pt-1">
            {isLoadingGraphData ? (
              <div className="flex h-full w-full flex-col items-center justify-center gap-y-3">
                <LoadingSquare className="h-6 w-6" />
                <div className="text-sm text-slate-400">
                  {loadingGraphLabel.length > 0 ? loadingGraphLabel : 'Loading...'}
                </div>
              </div>
            ) : selectedMetadataGraph ? (
              <div className="flex h-full max-h-full w-full flex-col">
                <div className="flex h-[50%] max-h-[50%] min-h-[50%] w-full flex-row pb-2">
                  <LinkGraph />
                  <GraphNodeConnections />
                </div>
                <div className="relative flex h-[50%] w-full flex-row pb-1 pt-1">
                  <div className="w-full sm:w-[53%] sm:min-w-[53%] sm:max-w-[53%]">
                    <Subgraph />
                  </div>
                  <GraphFeatureDetail />
                </div>
              </div>
            ) : (
              <div className="flex h-full w-full items-center justify-center">
                <div className="text-center text-lg text-slate-400">
                  No graph selected. Choose one from the dropdown above.
                </div>
              </div>
            )}
          </div>
        </div>
        <LoadSubgraphModal />
        <SaveSubgraphModal />
        <WelcomeModal hasSlug={hasSlug} />
        <GenerateGraphModal />
        <CopyModal />
        <SteerModal />
      </div>
    </GraphModalProvider>
  );
}
