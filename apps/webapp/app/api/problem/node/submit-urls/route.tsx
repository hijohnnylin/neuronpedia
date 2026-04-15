import { createProblemNode } from '@/lib/db/problem';
import { detectTypeFromUrl, fetchUrlMetadata } from '@/lib/problem-utils';
import { normalizeUrl } from '@/lib/problem-url-types';
import { RequestAuthedUser, withAuthedUser } from '@/lib/with-user';
import { NextResponse } from 'next/server';

export const POST = withAuthedUser(async (request: RequestAuthedUser) => {
  const body = await request.json();
  const { urls, parentId } = body;

  if (!urls || !Array.isArray(urls) || urls.length === 0) {
    return NextResponse.json({ error: 'urls is required and must be a non-empty array' }, { status: 400 });
  }

  if (urls.length > 20) {
    return NextResponse.json({ error: 'Maximum 20 URLs per request' }, { status: 400 });
  }

  for (const url of urls) {
    if (typeof url !== 'string' || !/^https?:\/\/.+/i.test(url)) {
      return NextResponse.json({ error: `Invalid URL: ${url}` }, { status: 400 });
    }
  }

  const results: { url: string; nodeId?: number; error?: string }[] = [];

  const normalizedUrls = urls.map(normalizeUrl);
  const mainUrl = normalizedUrls[0];
  const additionalUrls = normalizedUrls.slice(1);

  // Detect types from all URLs, deduplicated
  const detectedTypes = new Set<string>();
  for (const url of normalizedUrls) {
    detectedTypes.add(detectTypeFromUrl(url));
  }
  const nodeTypes = Array.from(detectedTypes);

  // Fetch metadata from main URL
  let title: string | null = null;
  let description: string | null = null;
  let author: string | null = null;

  try {
    const meta = await fetchUrlMetadata(mainUrl);
    title = meta.title;
    description = meta.description;
    author = meta.author;
  } catch (err) {
    const message = err instanceof Error ? err.message : 'Unknown error';
    console.error('[submit-urls] Failed to fetch metadata for:', mainUrl, message);
    // Continue with null metadata — still create the node
  }

  try {
    const node = await createProblemNode(
      {
        nodeTypes: nodeTypes as any,
        parentId: parentId != null ? Number(parentId) : null,
        title,
        description,
        author,
        mainUrl,
        additionalUrls,
      },
      request.user,
    );

    results.push({ url: mainUrl, nodeId: node.id });
  } catch (err) {
    const message = err instanceof Error ? err.message : 'Unknown error';
    results.push({ url: mainUrl, error: message });
  }

  return NextResponse.json({ results });
});
