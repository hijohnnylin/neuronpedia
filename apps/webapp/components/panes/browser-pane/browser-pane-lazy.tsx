'use client';

import dynamic from 'next/dynamic';

const BrowserPaneLazy = dynamic(() => import('./browser-pane'), {
  ssr: false,
});

export default BrowserPaneLazy;