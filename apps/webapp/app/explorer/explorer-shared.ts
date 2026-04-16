export const DRAFT_ID = -1;
export const MAX_TITLE_LENGTH = 128;

export const NODE_TYPES_LIST = ['topic', 'paper', 'tool', 'dataset', 'eval', 'replication', 'model'] as const;

export type ProblemNodeData = {
  id: number;
  nodeTypes: string[];
  parentId: number | null;
  title: string | null;
  description: string | null;
  author: string | null;
  mainUrl: string | null;
  additionalUrls: string[];
  applicationTags: string[];
  approvalState: string;
  children: { id: number; title: string | null; nodeTypes: string[]; approvalState: string; author: string | null }[];
  parent: { id: number; title: string | null; nodeTypes: string[] } | null;
  createdBy: { id: string; name: string };
  approver: { id: string; name: string } | null;
};

export type DetailNode = ProblemNodeData & {
  comments: Comment[];
  edgesAsSource: { id: string; type: string; targetNode: { id: number; title: string | null; type: string } }[];
  edgesAsTarget: { id: string; type: string; sourceNode: { id: number; title: string | null; type: string } }[];
  createdAt: string | Date;
  updatedAt: string | Date;
};

export type Comment = {
  id: string;
  text: string;
  createdAt: string | Date;
  user: { id: string; name: string; image: string | null };
  replies?: Comment[];
};

export const TYPE_COLORS: Record<string, string> = {
  topic: 'bg-sky-600 text-white',
  paper: 'bg-emerald-600 text-white',
  tool: 'bg-indigo-600 text-white',
  dataset: 'bg-amber-600 text-white',
  eval: 'bg-rose-600 text-white',
  replication: 'bg-fuchsia-600 text-white',
  model: 'bg-stone-600 text-white',
};

export const TYPE_HEX_COLORS: Record<string, string> = {
  topic: '#3b82f6',
  paper: '#10b981',
  tool: '#4f46e5',
  dataset: '#f59e0b',
  eval: '#f43f5e',
  replication: '#c026d3',
  model: '#78716c',
};
