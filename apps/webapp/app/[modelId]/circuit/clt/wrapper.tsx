'use client';

import { useCircuitCLT } from '@/components/provider/circuit-clt-provider';
import { Card, CardContent } from '@/components/shadcn/card';
import { LoadingSquare } from '@/components/svg/loading-square';
import CLTFeatureDetail from './clt-feature-detail';
import CLTLinkGraph from './clt-link-graph';
import CLTNodeConnections from './clt-node-connections';
import CLTSubgraph from './clt-subgraph';
import CLTModelPromptSelector from './model-prompt-selector';

export default function CLTWrapper() {
  const { isLoadingGraphData } = useCircuitCLT();
  return (
    <div className="mt-4 flex w-full flex-col justify-center px-4 text-slate-700">
      <div className="flex w-full flex-col items-center justify-center sm:hidden">
        <div className="mb-2 w-full pt-8 text-center text-sm text-red-500">
          Sorry, this page is not optimized for mobile. Please visit this link on a desktop browser.
        </div>
      </div>
      <div className="hidden w-full flex-col items-center justify-center sm:flex">
        <div className="flex w-full max-w-screen-xl flex-col">
          {/* <div className="mb-1 w-full text-center text-base font-medium text-slate-600">
            Circuit Tracing <sup className="text-xs">alpha</sup>
          </div> */}
          <div className="mb-2 w-full text-center text-[11px] text-red-500">
            This is a work in progress and not linked to from public Neuronpedia pages.
          </div>
          <CLTModelPromptSelector />
        </div>

        <Card className="mb-10 mt-5 w-full bg-white">
          <CardContent className="py-6 pt-3">
            {isLoadingGraphData ? (
              <div className="flex h-full min-h-[800px] w-full items-center justify-center">
                <LoadingSquare className="h-6 w-6" />
              </div>
            ) : (
              <>
                <div className="flex w-full flex-row gap-x-2 pl-2">
                  <CLTLinkGraph />
                  <CLTNodeConnections />
                </div>
                <div className="flex w-full flex-row gap-x-2 pl-2">
                  <div className="w-[53%] min-w-[53%] max-w-[53%]">
                    <CLTSubgraph />
                  </div>
                  <div className="w-[45%] min-w-[45%] overflow-hidden">
                    <CLTFeatureDetail />
                  </div>
                </div>
              </>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
