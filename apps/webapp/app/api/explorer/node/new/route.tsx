import { createProblemNode } from '@/lib/db/problem';
import { RequestAuthedUser, withAuthedUser } from '@/lib/with-user';
import { NextResponse } from 'next/server';

export const POST = withAuthedUser(async (request: RequestAuthedUser) => {
  const body = await request.json();

  if (!body.type && (!body.nodeTypes || body.nodeTypes.length === 0)) {
    return NextResponse.json({ error: 'type or nodeTypes is required' }, { status: 400 });
  }

  try {
    const node = await createProblemNode(
      {
        nodeTypes: body.nodeTypes,
        parentId: body.parentId != null ? Number(body.parentId) : null,
        title: body.title,
        description: body.description,
        author: body.author,
        mainUrl: body.mainUrl,
        additionalUrls: body.additionalUrls,
        applicationTags: body.applicationTags,
      },
      request.user,
    );
    return NextResponse.json(node);
  } catch (error) {
    console.error('[node/new] Error creating node:', error);
    const message = error instanceof Error ? error.message : 'Unknown error';
    const isValidation = message.includes('must be') || message.includes('permission');
    return NextResponse.json({ error: message }, { status: isValidation ? 400 : 500 });
  }
});
