'use client';

import PanelLoader from '@/components/panel-loader';
import { useGlobalContext } from '@/components/provider/global-provider';
import * as Dialog from '@radix-ui/react-dialog';
import FeatureDashboard from '../app/[modelId]/[layer]/[index]/feature-dashboard';

export default function FeatureModal() {
  const { featureModalFeature, featureModalopen, setFeatureModalOpen } = useGlobalContext();

  return (
    <Dialog.Root open={featureModalopen}>
      <Dialog.Portal>
        <Dialog.Overlay className="fixed inset-0 z-50 bg-slate-600/20" />
        <Dialog.Content
          onPointerDownOutside={() => {
            setFeatureModalOpen(false);
          }}
          className="fixed left-[50%] top-[50%] z-50 mt-12 flex h-[100vh] max-h-[100vh] w-[100vw] max-w-[100%] translate-x-[-50%] translate-y-[-50%] flex-col overflow-y-scroll bg-slate-50 shadow-xl focus:outline-none sm:top-[50%] sm:mt-0 sm:h-[90vh] sm:max-h-[90vh] sm:w-[65vw] sm:max-w-[65%] sm:rounded-md"
        >
          <div className="sticky top-0 z-20 w-full flex-col items-center border-slate-300">
            <div className="mb-0 flex w-full flex-row items-start justify-between gap-x-4 rounded-t-md border-b bg-white px-2 pb-2 pt-2 sm:px-4">
              <button
                type="button"
                className="flex flex-row items-center justify-center gap-x-1 rounded-full bg-slate-300 px-3 py-1.5 text-[11px] text-slate-600 hover:bg-sky-700 hover:text-white focus:outline-none"
                aria-label="Close"
                onClick={() => {
                  setFeatureModalOpen(false);
                }}
              >
                Done
              </button>
              <a
                href={`/${featureModalFeature?.modelId}/${featureModalFeature?.layer}/${featureModalFeature?.index}`}
                target="_blank"
                rel="noreferrer"
                className="flex flex-row items-center justify-center gap-x-1 rounded-full bg-slate-300 px-3 py-1.5 text-[11px] text-slate-600 hover:bg-sky-700 hover:text-white focus:outline-none"
              >
                ↗ Popup
              </a>
            </div>
          </div>
          {featureModalFeature && featureModalFeature.activations ? (
            <div className="flex h-full w-full flex-row overflow-y-scroll">
              <FeatureDashboard initialNeuron={featureModalFeature} embed />
            </div>
          ) : (
            <div className="flex w-full flex-1 flex-col items-center justify-center">
              <PanelLoader showBackground={false} />
            </div>
          )}
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}
