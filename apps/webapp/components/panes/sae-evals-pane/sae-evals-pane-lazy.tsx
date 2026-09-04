'use client';

import dynamic from 'next/dynamic';

const SAEEvalsPaneLazy = dynamic(() => import('./sae-evals-pane'), {
  ssr: false,
});

export default SAEEvalsPaneLazy;