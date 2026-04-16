import { MAX_TITLE_LENGTH } from '@/app/explorer/explorer-shared';
import { prisma } from '@/lib/db';
import { ProblemEdgeType, ProblemNodeApprovalState, ProblemNodeType } from '@prisma/client';
import { AuthenticatedUser } from '../with-user';

// ─── Helpers ────────────────────────────────────────────────────────────────

const VALID_NODE_TYPES = new Set<string>(Object.values(ProblemNodeType));
const VALID_EDGE_TYPES = new Set<string>(Object.values(ProblemEdgeType));

const MAX_DESCRIPTION_LENGTH = 5000;
const MAX_URL_LENGTH = 2048;
const MAX_ADDITIONAL_URLS = 20;
const MAX_APPLICATION_TAGS = 20;
const MAX_TAG_LENGTH = 100;
const MAX_AUTHOR_LENGTH = 200;

function validateTitle(title: string | null | undefined) {
  if (title && title.length > MAX_TITLE_LENGTH) {
    throw new Error(`Title must be ${MAX_TITLE_LENGTH} characters or fewer`);
  }
}

function isValidHttpUrl(url: string): boolean {
  try {
    const parsed = new URL(url);
    return parsed.protocol === 'http:' || parsed.protocol === 'https:';
  } catch {
    return false;
  }
}

function validateNodeData(data: {
  description?: string | null;
  author?: string | null;
  mainUrl?: string | null;
  additionalUrls?: string[];
  applicationTags?: string[];
}) {
  if (data.description && data.description.length > MAX_DESCRIPTION_LENGTH) {
    throw new Error(`Description must be ${MAX_DESCRIPTION_LENGTH} characters or fewer`);
  }
  if (data.author && data.author.length > MAX_AUTHOR_LENGTH) {
    throw new Error(`Author must be ${MAX_AUTHOR_LENGTH} characters or fewer`);
  }
  if (data.mainUrl) {
    if (data.mainUrl.length > MAX_URL_LENGTH) {
      throw new Error(`Main URL must be ${MAX_URL_LENGTH} characters or fewer`);
    }
    if (!isValidHttpUrl(data.mainUrl)) {
      throw new Error('Main URL must be a valid http:// or https:// URL');
    }
  }
  if (data.additionalUrls) {
    if (data.additionalUrls.length > MAX_ADDITIONAL_URLS) {
      throw new Error(`Maximum ${MAX_ADDITIONAL_URLS} additional URLs allowed`);
    }

    data.additionalUrls.forEach((url) => {
      if (url.length > MAX_URL_LENGTH) {
        throw new Error(`Each URL must be ${MAX_URL_LENGTH} characters or fewer`);
      }
      if (!isValidHttpUrl(url)) {
        throw new Error('Each additional URL must be a valid http:// or https:// URL');
      }
    });
  }
  if (data.applicationTags) {
    if (data.applicationTags.length > MAX_APPLICATION_TAGS) {
      throw new Error(`Maximum ${MAX_APPLICATION_TAGS} application tags allowed`);
    }

    data.applicationTags.forEach((tag) => {
      if (tag.length > MAX_TAG_LENGTH) {
        throw new Error(`Each tag must be ${MAX_TAG_LENGTH} characters or fewer`);
      }
    });
  }
}

function validateNodeTypes(nodeTypes: unknown): asserts nodeTypes is ProblemNodeType[] {
  if (!Array.isArray(nodeTypes)) throw new Error('nodeTypes must be an array');
  for (const t of nodeTypes) {
    if (typeof t !== 'string' || !VALID_NODE_TYPES.has(t)) {
      throw new Error(`Invalid node type: ${String(t).slice(0, 50)}. Valid types: ${[...VALID_NODE_TYPES].join(', ')}`);
    }
  }
}

function validateEdgeType(type: unknown): asserts type is ProblemEdgeType {
  if (typeof type !== 'string' || !VALID_EDGE_TYPES.has(type)) {
    throw new Error(
      `Invalid edge type: ${String(type).slice(0, 50)}. Valid types: ${[...VALID_EDGE_TYPES].join(', ')}`,
    );
  }
}

const problemNodeInclude = {
  createdBy: { select: { id: true, name: true, image: true } },
  approver: { select: { id: true, name: true, image: true } },
  children: {
    select: { id: true, title: true, nodeTypes: true, approvalState: true, author: true },
    where: { approvalState: { not: ProblemNodeApprovalState.REJECTED } },
    orderBy: { title: 'asc' as const },
  },
  parent: {
    select: { id: true, title: true, nodeTypes: true },
  },
};

async function assertUserCanEditNode(nodeId: number, user: AuthenticatedUser) {
  const dbUser = await prisma.user.findUniqueOrThrow({
    where: { id: user.id },
    select: { admin: true, isProblemEditor: true },
  });
  if (dbUser.admin || dbUser.isProblemEditor) return;

  const node = await prisma.problemNode.findUniqueOrThrow({ where: { id: nodeId } });
  if (node.createdById !== user.id) {
    throw new Error('You do not have permission to edit this node');
  }
}

// ─── Nodes ──────────────────────────────────────────────────────────────────

export async function getProblemNodes(includeUnapproved = false, currentUserId?: string) {
  const where = includeUnapproved
    ? { approvalState: { not: ProblemNodeApprovalState.REJECTED } }
    : currentUserId
      ? {
          OR: [
            { approvalState: ProblemNodeApprovalState.APPROVED },
            { createdById: currentUserId, approvalState: { not: ProblemNodeApprovalState.REJECTED } },
          ],
        }
      : { approvalState: ProblemNodeApprovalState.APPROVED };

  return prisma.problemNode.findMany({
    where,
    include: problemNodeInclude,
    orderBy: { title: 'asc' },
  });
}

export async function getProblemNode(id: number) {
  return prisma.problemNode.findUniqueOrThrow({
    where: { id },
    include: {
      ...problemNodeInclude,
      comments: {
        include: {
          user: { select: { id: true, name: true, image: true } },
          replies: {
            include: {
              user: { select: { id: true, name: true, image: true } },
            },
            orderBy: { createdAt: 'asc' as const },
          },
        },
        where: { parentCommentId: null },
        orderBy: { createdAt: 'asc' as const },
      },
      edgesAsSource: {
        include: { targetNode: { select: { id: true, title: true, nodeTypes: true } } },
      },
      edgesAsTarget: {
        include: { sourceNode: { select: { id: true, title: true, nodeTypes: true } } },
      },
    },
  });
}

export async function createProblemNode(
  data: {
    nodeTypes?: ProblemNodeType[];
    parentId?: number | null;
    title?: string | null;
    description?: string | null;
    author?: string | null;
    mainUrl?: string | null;
    additionalUrls?: string[];
    applicationTags?: string[];
  },
  user: AuthenticatedUser,
) {
  validateTitle(data.title);
  validateNodeData(data);
  if (data.nodeTypes) validateNodeTypes(data.nodeTypes);

  const dbUser = await prisma.user.findUniqueOrThrow({
    where: { id: user.id },
    select: { admin: true, isProblemEditor: true },
  });
  const autoApprove = dbUser.admin || dbUser.isProblemEditor;

  const node = await prisma.problemNode.create({
    data: {
      nodeTypes: data.nodeTypes ?? [ProblemNodeType.topic],
      parentId: data.parentId ?? null,
      title: data.title ?? null,
      description: data.description ?? null,
      author: data.author ?? null,
      mainUrl: data.mainUrl ?? null,
      additionalUrls: data.additionalUrls ?? [],
      applicationTags: data.applicationTags ?? [],
      createdById: user.id,
      ...(autoApprove && {
        approvalState: ProblemNodeApprovalState.APPROVED,
        approverId: user.id,
      }),
    },
    include: problemNodeInclude,
  });

  await prisma.problemNodeLog.create({
    data: {
      userId: user.id,
      problemNodeId: node.id,
      action: 'CREATED_NODE',
      details: `Created ${(data.nodeTypes ?? ['topic']).join(', ')} node: ${data.title ?? '(untitled)'}`,
    },
  });

  return node;
}

export async function updateProblemNode(
  id: number,
  data: {
    nodeTypes?: ProblemNodeType[];
    parentId?: number | null;
    title?: string | null;
    description?: string | null;
    author?: string | null;
    mainUrl?: string | null;
    additionalUrls?: string[];
    applicationTags?: string[];
  },
  user: AuthenticatedUser,
) {
  await assertUserCanEditNode(id, user);
  validateTitle(data.title);
  validateNodeData(data);
  if (data.nodeTypes) validateNodeTypes(data.nodeTypes);

  const node = await prisma.problemNode.update({
    where: { id },
    data,
    include: problemNodeInclude,
  });

  await prisma.problemNodeLog.create({
    data: {
      userId: user.id,
      problemNodeId: node.id,
      action: 'UPDATED_NODE',
      details: `Updated node: ${node.title ?? '(untitled)'}`,
    },
  });

  return node;
}

export async function deleteProblemNode(id: number, user: AuthenticatedUser) {
  await assertUserCanEditNode(id, user);

  await prisma.problemNodeLog.create({
    data: {
      userId: user.id,
      problemNodeId: id,
      action: 'DELETED_NODE',
      details: `Deleted node ${id}`,
    },
  });

  return prisma.problemNode.delete({ where: { id } });
}

// ─── Approval ───────────────────────────────────────────────────────────────

export async function approveProblemNode(id: number, approved: boolean, user: AuthenticatedUser) {
  const dbUser = await prisma.user.findUniqueOrThrow({
    where: { id: user.id },
    select: { admin: true, isProblemEditor: true },
  });
  if (!dbUser.admin && !dbUser.isProblemEditor) {
    throw new Error('You do not have permission to approve/reject nodes');
  }

  const node = await prisma.problemNode.update({
    where: { id },
    data: {
      approvalState: approved ? ProblemNodeApprovalState.APPROVED : ProblemNodeApprovalState.REJECTED,
      approverId: user.id,
    },
    include: problemNodeInclude,
  });

  await prisma.problemNodeLog.create({
    data: {
      userId: user.id,
      problemNodeId: node.id,
      action: approved ? 'APPROVED_NODE' : 'REJECTED_NODE',
      details: `${approved ? 'Approved' : 'Rejected'} node: ${node.title ?? '(untitled)'}`,
    },
  });

  return node;
}

// ─── Edges ──────────────────────────────────────────────────────────────────

export async function createProblemEdge(
  data: {
    sourceNodeId: number;
    targetNodeId: number;
    type: string;
  },
  user: AuthenticatedUser,
) {
  validateEdgeType(data.type);

  const dbUser = await prisma.user.findUniqueOrThrow({
    where: { id: user.id },
    select: { admin: true, isProblemEditor: true },
  });
  const autoApprove = dbUser.admin || dbUser.isProblemEditor;

  return prisma.problemEdge.create({
    data: {
      sourceNodeId: data.sourceNodeId,
      targetNodeId: data.targetNodeId,
      type: data.type,
      createdById: user.id,
      ...(autoApprove && {
        approvalState: ProblemNodeApprovalState.APPROVED,
        approverId: user.id,
      }),
    },
  });
}

export async function deleteProblemEdge(id: string, user: AuthenticatedUser) {
  const edge = await prisma.problemEdge.findUniqueOrThrow({ where: { id } });
  const dbUser = await prisma.user.findUniqueOrThrow({
    where: { id: user.id },
    select: { admin: true, isProblemEditor: true },
  });
  if (!dbUser.admin && !dbUser.isProblemEditor && edge.createdById !== user.id) {
    throw new Error('You do not have permission to delete this edge');
  }
  return prisma.problemEdge.delete({ where: { id } });
}

// ─── Comments ───────────────────────────────────────────────────────────────

export async function createProblemComment(
  data: {
    problemNodeId: number;
    text: string;
    parentCommentId?: string | null;
  },
  user: AuthenticatedUser,
) {
  const comment = await prisma.problemNodeComment.create({
    data: {
      problemNodeId: data.problemNodeId,
      text: data.text,
      parentCommentId: data.parentCommentId ?? null,
      userId: user.id,
    },
    include: {
      user: { select: { id: true, name: true, image: true } },
    },
  });

  await prisma.problemNodeLog.create({
    data: {
      userId: user.id,
      problemNodeId: data.problemNodeId,
      action: 'COMMENT_ADDED',
      details: `Added comment on node ${data.problemNodeId}`,
    },
  });

  return comment;
}

export async function deleteProblemComment(id: string, user: AuthenticatedUser) {
  const comment = await prisma.problemNodeComment.findUniqueOrThrow({ where: { id } });
  const dbUser = await prisma.user.findUniqueOrThrow({
    where: { id: user.id },
    select: { admin: true, isProblemEditor: true },
  });
  if (!dbUser.admin && !dbUser.isProblemEditor && comment.userId !== user.id) {
    throw new Error('You do not have permission to delete this comment');
  }
  return prisma.problemNodeComment.delete({ where: { id } });
}

// ─── Logs ───────────────────────────────────────────────────────────────────

export async function getProblemLogs(limit = 50) {
  return prisma.problemNodeLog.findMany({
    include: {
      user: { select: { id: true, name: true, image: true } },
      problemNode: { select: { id: true, title: true, nodeTypes: true } },
    },
    orderBy: { timestamp: 'desc' },
    take: limit,
  });
}

export async function getRecentProblemCreations(limit = 10) {
  return prisma.problemNodeLog.findMany({
    where: {
      action: 'CREATED_NODE',
      problemNode: { approvalState: ProblemNodeApprovalState.APPROVED },
    },
    include: {
      user: { select: { id: true, name: true } },
      problemNode: { select: { id: true, title: true, nodeTypes: true } },
    },
    orderBy: { timestamp: 'desc' },
    take: limit,
  });
}
