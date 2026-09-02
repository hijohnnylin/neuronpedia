import { NODE_ENV } from '@/lib/env';
import fs from 'fs';
import { NextResponse } from 'next/server';
import path from 'path';

// Blog posts are read with `fs` at request time (app/blog/blog-util.ts), so they are not in the
// bundler's module graph: editing one produces no rebuild and no reload, and the page only changes
// if you refresh it by hand. This streams one message per edit, which `MdxDevReload` turns into a
// `router.refresh()`. Dev only — there is nothing to watch once the posts are built into a bundle.

const POSTS_DIR = path.join(process.cwd(), 'app', 'blog', 'posts');

// A single save arrives as several watch events, because editors write a temp file and rename it
// over the original. Waiting for them to stop costs one re-render instead of three.
const SETTLE_MS = 100;

export function GET() {
  if (NODE_ENV !== 'development') {
    return new NextResponse(null, { status: 404 });
  }

  const encoder = new TextEncoder();
  let watcher: fs.FSWatcher | null = null;
  let settle: NodeJS.Timeout | null = null;
  let open = true;

  function close() {
    open = false;
    if (settle) {
      clearTimeout(settle);
      settle = null;
    }
    watcher?.close();
    watcher = null;
  }

  const stream = new ReadableStream({
    start(controller) {
      watcher = fs.watch(POSTS_DIR, (_event, filename) => {
        if (!open || !filename?.endsWith('.mdx')) {
          return;
        }
        if (settle) {
          clearTimeout(settle);
        }
        settle = setTimeout(() => {
          try {
            controller.enqueue(encoder.encode(`data: ${filename}\n\n`));
          } catch {
            // The browser went away between the edit and the flush.
            close();
          }
        }, SETTLE_MS);
      });
    },
    cancel() {
      close();
    },
  });

  return new NextResponse(stream, {
    headers: {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache',
      Connection: 'keep-alive',
    },
  });
}
