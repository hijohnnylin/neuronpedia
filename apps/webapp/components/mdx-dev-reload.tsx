'use client';

import { useEffect } from 'react';

// Mounted by the root layout under `next dev` only. A full reload rather than `router.refresh()`:
// refresh refetches the route but leaves the rendered post unchanged, and a plain page load is what
// already picks up an edit today.
export default function MdxDevReload() {
  useEffect(() => {
    const source = new EventSource('/api/dev/mdx-reload');
    source.onmessage = () => {
      window.location.reload();
    };
    return () => {
      source.close();
    };
  }, []);

  return null;
}
