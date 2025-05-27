'use client';

import { useGraphContext } from '@/components/provider/graph-provider';
import { Button } from '@/components/shadcn/button';
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from '@/components/shadcn/dialog';
import { LoadingSquare } from '@/components/svg/loading-square';
import copy from 'copy-to-clipboard';
import { CodeIcon, CopyIcon, LinkIcon } from 'lucide-react';
import { useSearchParams } from 'next/navigation';
import GraphFeatureDetail from './feature-detail';
import GraphToolbar from './graph-toolbar';
import LinkGraph from './link-graph';
import GraphNodeConnections from './node-connections';
import Subgraph from './subgraph';

function CopyModal() {
  const { isCopyModalOpen, setIsCopyModalOpen } = useGraphContext();

  const handleCopyOption = (type: 'embed' | 'iframe' | 'normal') => {
    const currentUrl = window.location.href;

    switch (type) {
      case 'embed': {
        const url = new URL(currentUrl);
        url.searchParams.set('embed', 'true');
        copy(url.toString());
        alert('Embed link copied to clipboard!');
        break;
      }
      case 'iframe': {
        const url = new URL(currentUrl);
        url.searchParams.set('embed', 'true');
        const iframeCode = `<iframe src="${url.toString()}" width="100%" height="600" frameborder="0"></iframe>`;
        copy(iframeCode);
        alert('Iframe code copied to clipboard!');
        break;
      }
      case 'normal': {
        const url = new URL(currentUrl);
        url.searchParams.delete('embed');
        copy(url.toString());
        alert('URL copied to clipboard!');
        break;
      }
      default:
        break;
    }

    setIsCopyModalOpen(false);
  };

  return (
    <Dialog open={isCopyModalOpen} onOpenChange={setIsCopyModalOpen}>
      <DialogContent className="bg-white sm:max-w-md">
        <DialogHeader>
          <DialogTitle>Share Graph & Subgraph</DialogTitle>
          <DialogDescription>{`Choose how you'd like to copy this graph for sharing. This will share the graph, your current subgraph, and any custom labels you have created.`}</DialogDescription>
        </DialogHeader>
        <div className="py-22 flex flex-col gap-3">
          <Button
            variant="outline"
            className="h-auto justify-start gap-3 py-3"
            onClick={() => handleCopyOption('normal')}
          >
            <LinkIcon className="h-4 w-4" />
            <div className="text-left">
              <div className="font-medium">Copy Link</div>
              <div className="text-sm text-slate-500">Share as a normal URL</div>
            </div>
          </Button>

          <Button
            variant="outline"
            className="h-auto justify-start gap-3 py-3"
            onClick={() => handleCopyOption('embed')}
          >
            <CopyIcon className="h-4 w-4" />
            <div className="text-left">
              <div className="font-medium">Copy Embed Link</div>
              <div className="text-sm text-slate-500">URL optimized for iFrame embed</div>
            </div>
          </Button>

          <Button
            variant="outline"
            className="h-auto justify-start gap-3 py-3"
            onClick={() => handleCopyOption('iframe')}
          >
            <CodeIcon className="h-4 w-4" />
            <div className="text-left">
              <div className="font-medium">Copy iFrame Code</div>
              <div className="text-sm text-slate-500">HTML code snippet to embed this graph</div>
            </div>
          </Button>
        </div>
      </DialogContent>
    </Dialog>
  );
}

export default function GraphWrapper() {
  const { isLoadingGraphData, selectedMetadataGraph, loadingGraphLabel } = useGraphContext();

  const searchParams = useSearchParams();
  const isEmbed = searchParams.get('embed') === 'true';
  return (
    <div
      className={`${isEmbed ? 'h-[calc(100%_-_20px)] max-h-screen min-h-[calc(100%_-_20px)]' : 'h-[calc(100vh_-_75px)] max-h-[calc(100vh_-_75px)] min-h-[calc(100vh_-_75px)]'} flex w-full flex-col justify-center px-4 text-slate-700`}
    >
      <div className="flex w-full flex-col items-center justify-center sm:hidden">
        <div className="mb-2 w-full pt-8 text-center text-sm text-red-500">
          Sorry, this page is not optimized for mobile. Please visit this link on a desktop browser.
        </div>
      </div>
      <div className="hidden w-full flex-1 flex-col items-center justify-center overflow-hidden sm:flex">
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
              <div className="flex h-[50%] w-full flex-row pb-1 pt-1">
                <div className="w-[53%] min-w-[53%] max-w-[53%]">
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
      <CopyModal />
    </div>
  );
}
