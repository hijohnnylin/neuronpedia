import { getLayerNumFromSource } from '@/lib/utils/source';
import { NeuronWithPartialRelations } from '@/prisma/generated/zod';
import copy from 'copy-to-clipboard';
import { Copy, ExternalLink, Joystick, Trash } from 'lucide-react';
import Link from 'next/link';
import { useGlobalContext } from './provider/global-provider';
import { Button } from './shadcn/button';

export default function VectorItem({
  vector,
  isInNeuronDetails,
}: {
  vector: NeuronWithPartialRelations;
  isInNeuronDetails: boolean;
}) {
  const { getInferenceEnabledForModel } = useGlobalContext();

  async function deleteVector() {
    if (window.confirm('Are you sure you want to delete this vector?')) {
      await fetch(`/api/vector/delete`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          modelId: vector.modelId,
          source: vector.layer,
          index: vector.index,
        }),
      });
      // reload the page
      window.location.reload();
    }
  }

  return (
    <div className="flex w-full flex-col px-2 py-2 pb-3 text-xs">
      <div className="flex flex-1 flex-col items-center justify-center">
        {vector.vectorLabel && (
          <div className="flex flex-1 grid-cols-1 flex-col items-center justify-center">
            <div className="mb-0.5 mt-0 text-[9px] uppercase text-slate-400">Vector Label</div>
            <Link
              href={`/${vector.modelId}/${vector.layer}/${vector.index}`}
              className="text-sm leading-none text-sky-700 hover:underline"
              prefetch={false}
            >
              {vector.vectorLabel}
            </Link>
          </div>
        )}
      </div>
      <div
        className={`grid w-full ${isInNeuronDetails ? 'mt-1 grid-cols-4 gap-y-0.5' : 'mt-2 grid-cols-2 gap-y-1'
          } items-start justify-center gap-x-2`}
      >
        <div className="flex flex-1 grid-cols-1 flex-col items-center justify-center">
          <div className="mb-0.5 mt-1.5 text-[9px] uppercase text-slate-400">Model</div>
          <div className="mt-0 text-xs text-slate-500">{vector.modelId}</div>
        </div>

        <div className="flex flex-1 grid-cols-1 flex-col items-center justify-center">
          <div className="mb-0.5 mt-1.5 text-[9px] uppercase text-slate-400">Layer #</div>
          <div className="mt-0 font-mono text-sm text-slate-500">{getLayerNumFromSource(vector.layer)}</div>
        </div>

        <div className="flex flex-1 grid-cols-1 flex-col items-center justify-center">
          <div className="mb-0.5 mt-1.5 text-[9px] uppercase text-slate-400">Steering Hook</div>
          <div className="mt-0 text-xs text-slate-500">{vector.hookName}</div>
        </div>

        <div className="flex flex-1 flex-col items-center justify-center">
          <div className="mb-0.5 mt-1.5 text-[9px] uppercase text-slate-400">Steering Strength</div>
          <div className="mt-0 text-xs text-slate-500">{vector.vectorDefaultSteerStrength}</div>
        </div>

        {vector.creator && (
          <div className="flex flex-1 flex-col items-center justify-center">
            <div className="mb-0.5 mt-1.5 text-[9px] uppercase text-slate-400">Uploader</div>
            <div className="mt-0 text-xs text-slate-500">{vector.creator?.name}</div>
          </div>
        )}

        <div className="flex flex-1 flex-col items-center justify-center">
          <div className="mb-0.5 mt-1.5 text-[9px] uppercase text-slate-400">Created At</div>
          <div className="mt-0 text-xs text-slate-500">
            {`${new Date(vector.createdAt).toLocaleDateString()} ${new Date(vector.createdAt).toLocaleTimeString()}`}
          </div>
        </div>

        <div className="col-span-1 flex flex-1 flex-col items-center justify-center">
          <div className="mb-0.5 mt-1.5 text-[9px] uppercase text-slate-400">Raw Vector</div>
          <div className="mt-0 flex flex-row items-center justify-center gap-x-2">
            <input
              readOnly
              className="h-8 min-h-8 w-32 min-w-32 rounded border-slate-300 px-2.5 py-1 font-mono text-[10px] leading-tight text-slate-400 outline-none focus:border-slate-400 focus:outline-none focus:ring-0"
              value={JSON.stringify(vector.vector)}
            />
            <Button
              variant="outline"
              size="icon"
              onClick={() => {
                copy(JSON.stringify(vector.vector));
                alert('Copied to clipboard.');
              }}
              className="h-8 w-8 min-w-8"
            >
              <Copy className="h-4 w-4 text-slate-500" />
            </Button>
          </div>
        </div>
        <div className="col-span-1 flex flex-1 flex-col items-center justify-center">
          <div className="mb-0.5 mt-1.5 text-[9px] uppercase text-slate-400">Actions</div>
          <div className="mt-0 flex flex-row items-center justify-center gap-x-1.5">
            <Button
              variant="outline"
              size="icon"
              onClick={() => {
                const isAssistantAxis =
                  vector.modelId === 'llama3.3-70b-it' &&
                  vector.layer === '40-neuronpedia-resid-post' &&
                  vector.index === '101874252';

                if (isAssistantAxis) {
                  window.open('/assistant-axis', '_blank');
                } else {
                  window.open(`/${vector.modelId}/steer/?source=${vector.layer}&index=${vector.index}`, '_blank');
                }
              }}
              aria-label="Steer"
              className={`h-8 w-20 gap-x-1.5 text-xs ${getInferenceEnabledForModel(vector.modelId) ? '' : 'hidden'}`}
            >
              <Joystick className="h-4 w-4 text-slate-500" />
              Steer
            </Button>
            {!isInNeuronDetails && (
              <>
                <Button
                  variant="outline"
                  size="icon"
                  onClick={() => {
                    window.open(`/${vector.modelId}/${vector.layer}/${vector.index}`, '_blank');
                  }}
                  aria-label="Open in New Tab"
                  className="h-8 w-8 min-w-8"
                >
                  <ExternalLink className="h-4 w-4 text-slate-500" />
                </Button>
                <Button
                  variant="outline"
                  size="icon"
                  onClick={() => {
                    deleteVector();
                  }}
                  className="h-8 w-8 border border-red-200 text-red-400 hover:bg-red-50"
                >
                  <Trash className="h-4 w-4" />
                </Button>
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
