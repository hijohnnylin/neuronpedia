import { updateProblemNode } from '@/lib/db/problem';
import { RequestAuthedUser, withAuthedUser } from '@/lib/with-user';
import { NextResponse } from 'next/server';

export const POST = withAuthedUser(async (request: RequestAuthedUser) => {
  const body = await request.json();

  const id = Number(body.id);
  if (!id || Number.isNaN(id)) {
    return NextResponse.json({ error: 'id is required' }, { status: 400 });
  }

  try {
    const node = await updateProblemNode(
      id,
      {
        nodeTypes: body.nodeTypes,
        parentId: body.parentId != null ? Number(body.parentId) : body.parentId,
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
    return NextResponse.json({ error: error instanceof Error ? error.message : 'Unknown error' }, { status: 500 });
  }
});
