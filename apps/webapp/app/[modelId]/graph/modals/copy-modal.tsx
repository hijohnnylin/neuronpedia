'use client';

import { useGraphModalContext } from '@/components/provider/graph-modal-provider';
import { Button } from '@/components/shadcn/button';
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from '@/components/shadcn/dialog';
import copy from 'copy-to-clipboard';
import { CodeIcon, CopyIcon, LinkIcon } from 'lucide-react';

export default function CopyModal() {
  const { isCopyModalOpen, setIsCopyModalOpen } = useGraphModalContext();

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
