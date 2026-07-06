'use client';

import dynamic from 'next/dynamic';

const SearchExplanationsPaneLazy = dynamic(() => import('./search-explanations-pane'), {
  ssr: false,
});

export default SearchExplanationsPaneLazy;