'use client';

import { Provider as TooltipProvider } from '@radix-ui/react-tooltip';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { AppProgressBar as ProgressBar } from 'next-nprogress-bar';
import { PrimeReactProvider } from 'primereact/api';
import {
  ExplanationModelType,
  ExplanationScoreModel,
  ExplanationScoreType,
  ExplanationType,
  ModelWithPartialRelations,
  SourceReleaseWithPartialRelations,
} from 'prisma/generated/zod';
import CommentProvider from './comment-provider';
import ExplanationScoreDetailProvider from './explanation-score-detail-provider';
import FeatureProvider from './feature-provider';
import GlobalProvider from './global-provider';
import { ImportProvider } from './import-provider';
import UmapProvider from './umap-provider';
// Note: We don't include CircuitCLTProvider here as it needs to be initialized with metadata
// and is used specifically in the CLT page, not globally

const queryClient = new QueryClient();

export function Providers({
  children,
  initialModels,
  initialExplanationTypes,
  initialExplanationModels,
  initialReleases,
  initialExplanationScoreTypes,
  initialExplanationScoreModelTypes,
}: {
  children: React.ReactNode;
  initialModels: {
    [key: string]: ModelWithPartialRelations;
  };
  initialExplanationTypes: ExplanationType[];
  initialExplanationModels: ExplanationModelType[];
  initialReleases: SourceReleaseWithPartialRelations[];
  initialExplanationScoreTypes: ExplanationScoreType[];
  initialExplanationScoreModelTypes: ExplanationScoreModel[];
}) {
  return (
    <>
      <QueryClientProvider client={queryClient}>
        <PrimeReactProvider>
          <GlobalProvider
            initialModels={initialModels}
            initialExplanationTypes={initialExplanationTypes}
            initialExplanationModels={initialExplanationModels}
            initialReleases={initialReleases}
            initialExplanationScoreTypes={initialExplanationScoreTypes}
            initialExplanationScoreModelTypes={initialExplanationScoreModelTypes}
          >
            <ImportProvider>
              <UmapProvider>
                <TooltipProvider>
                  <FeatureProvider>
                    <CommentProvider>
                      <ExplanationScoreDetailProvider>
                        <CommentProvider>{children}</CommentProvider>
                      </ExplanationScoreDetailProvider>
                    </CommentProvider>
                  </FeatureProvider>
                </TooltipProvider>
              </UmapProvider>
            </ImportProvider>
          </GlobalProvider>
        </PrimeReactProvider>
      </QueryClientProvider>
      <ProgressBar height="4px" color="#0369a1" options={{ showSpinner: false }} shallowRouting />
    </>
  );
}
