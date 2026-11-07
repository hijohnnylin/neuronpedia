'use client';

import { ExplanationScore } from '@prisma/client';
import { ExplanationWithPartialRelations } from 'prisma/generated/zod';
import { useMemo, useState } from 'react';
import createContextWrapper from './provider-util';

export const [ExplanationScoreDetailContext, useExplanationScoreDetailContext] = createContextWrapper<{
  explanationScoreDetailOpen: boolean;
  setExplanationScoreDetailOpen: (open: boolean) => void;
  explanationScoreDetailScore: ExplanationScore | undefined;
  setExplanationScoreDetailScore: (score: ExplanationScore) => void;
  explanationScoreDetailExplanation: ExplanationWithPartialRelations | undefined;
  setExplanationScoreDetailExplanation: (explanation: ExplanationWithPartialRelations) => void;
  loadExplanationScoreDetail: (
    explanation: ExplanationWithPartialRelations,
    explanationScore: ExplanationScore,
  ) => void;
  explanationScoreDetailLoading: boolean;
  setExplanationScoreDetailLoading: (loading: boolean) => void;
}>('ExplanationScoreDetailContext');

export default function ExplanationScoreDetailProvider({ children }: { children: React.ReactNode }) {
  const [explanationScoreDetailOpen, setExplanationScoreDetailOpen] = useState(false);
  const [explanationScoreDetailScore, setExplanationScoreDetailScore] = useState<ExplanationScore>();
  const [explanationScoreDetailExplanation, setExplanationScoreDetailExplanation] =
    useState<ExplanationWithPartialRelations>();
  const [explanationScoreDetailLoading, setExplanationScoreDetailLoading] = useState(false);

  function loadExplanationScoreDetail(
    explanation: ExplanationWithPartialRelations,
    explanationScore: ExplanationScore,
  ) {
    setExplanationScoreDetailExplanation(explanation);
    setExplanationScoreDetailOpen(true);
    if (explanationScore.jsonDetails) {
      setExplanationScoreDetailScore(explanationScore);
    } else {
      setExplanationScoreDetailLoading(true);
      fetch(`/api/explanation/${explanation.id}`, {
        method: 'GET',
        headers: { 'Content-Type': 'application/json' },
      })
        .then((response) => response.json())
        .then((result: ExplanationWithPartialRelations) => {
          const scores = result.scores as ExplanationScore[];
          setExplanationScoreDetailScore(scores[0]);
          setExplanationScoreDetailLoading(false);
        });
    }
  }

  return (
    <ExplanationScoreDetailContext.Provider
      value={useMemo(
        () => ({
          explanationScoreDetailOpen,
          setExplanationScoreDetailOpen,
          explanationScoreDetailScore,
          setExplanationScoreDetailScore,
          explanationScoreDetailExplanation,
          setExplanationScoreDetailExplanation,
          loadExplanationScoreDetail,
          explanationScoreDetailLoading,
          setExplanationScoreDetailLoading,
        }),
        [
          explanationScoreDetailOpen,
          explanationScoreDetailScore,
          explanationScoreDetailExplanation,
          explanationScoreDetailLoading,
          loadExplanationScoreDetail,
        ],
      )}
    >
      {children}
    </ExplanationScoreDetailContext.Provider>
  );
}

// // Old Activation Score V1 Dialog (not used currently)
// // TODO: bring this back
// activationOpen: boolean;
// setActivationOpen: (open: boolean) => void;
// loadExplanationWithActivations: (explanationId: string) => void;
// activationExplanation: ExplanationWithPartialRelations | undefined;
// setActivationExplanation: (
//   explanation: ExplanationWithPartialRelations,
// ) => void;
// activationLoading: boolean;
// setActivationLoading: (loading: boolean) => void;
