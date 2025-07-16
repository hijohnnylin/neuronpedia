import { DeleteSubgraphRequestSchema } from '@/app/[modelId]/graph/utils';
import { prisma } from '@/lib/db';
import { RequestAuthedUser, withAuthedUser } from '@/lib/with-user';
import { NextResponse } from 'next/server';

export const POST = withAuthedUser(async (request: RequestAuthedUser) => {
  try {
    const body = await request.json();
    const validatedData = DeleteSubgraphRequestSchema.parse(body);
    const { subgraphId } = validatedData;

    // Find the subgraph to check ownership
    const subgraph = await prisma.graphMetadataSubgraph.findUnique({
      where: { id: subgraphId },
    });

    if (!subgraph) {
      return NextResponse.json({ error: 'Subgraph not found' }, { status: 404 });
    }

    // Check that the logged in user equals the owner of the subgraph
    if (subgraph.userId !== request.user.id) {
      return NextResponse.json({ error: 'You are not authorized to delete this subgraph' }, { status: 403 });
    }

    // Delete the subgraph
    await prisma.graphMetadataSubgraph.delete({
      where: { id: subgraphId },
    });

    return NextResponse.json({ success: true, message: 'Subgraph deleted successfully' });
  } catch (error) {
    console.error('Error deleting subgraph:', error);
    return NextResponse.json({ error: 'Failed to delete subgraph' }, { status: 500 });
  }
});
