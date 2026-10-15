import { prisma } from '@/lib/db';
import { getPersonaAxisDefinitions } from '@/lib/db/persona-axis';
import { makeAuthedUserFromSessionOrReturnNull } from '@/lib/db/user';
import { ChatMessage } from '@/lib/utils/steer';
import { axisReadoutsToPublic, PublicAxisReadout } from '@/lib/utils/steer-axis-public';
import { axisReadoutsFromStored } from '@/lib/utils/steer-wire';
import { SteerOutputType } from '@prisma/client';
import { Metadata } from 'next';
import NeuralTransparencyChat from './neural-transparency-chat';
import { NT_AXIS_IDS, NT_MODEL_ID, NT_TITLE } from './shared';

export async function generateMetadata(): Promise<Metadata> {
  const title = NT_TITLE;
  const description = `Chat with a language model while reading the traits its activations carry.`;

  return {
    title,
    description,
    openGraph: {
      title,
      description,
    },
  };
}

/** A saved chat on this page's model, or null for any id that is not one. */
async function loadSavedChat(
  savedId: string,
): Promise<{ messages: ChatMessage[]; readouts: PublicAxisReadout[] } | null> {
  const row = await prisma.steerOutput.findUnique({
    where: { id: savedId },
    select: { modelId: true, outputTextChatTemplate: true, capMonitorOutput: true },
  });
  if (!row || row.modelId !== NT_MODEL_ID || !row.outputTextChatTemplate) {
    return null;
  }
  try {
    const messages = JSON.parse(row.outputTextChatTemplate) as ChatMessage[];
    const stored = row.capMonitorOutput ? JSON.parse(row.capMonitorOutput) : null;
    const readouts = axisReadoutsToPublic(axisReadoutsFromStored(stored, SteerOutputType.DEFAULT)).filter(
      (readout) => readout.type !== SteerOutputType.STEERED,
    );
    return { messages, readouts };
  } catch {
    return null;
  }
}

export default async function Page({ searchParams }: { searchParams: Promise<{ savedId?: string | string[] }> }) {
  const { savedId: savedIdParam } = await searchParams;
  const savedId = typeof savedIdParam === 'string' ? savedIdParam : undefined;

  // Read here rather than from the client, because the wheel has to be drawn before anything has
  // been measured: its frame is the axes themselves, and a page that fetched them after mounting
  // would show an empty square first and then rearrange under the first reply.
  const [axes, saved] = await Promise.all([
    getPersonaAxisDefinitions(NT_MODEL_ID, NT_AXIS_IDS, await makeAuthedUserFromSessionOrReturnNull()),
    savedId ? loadSavedChat(savedId) : null,
  ]);

  return (
    <div className="flex h-full w-full flex-col items-center overflow-hidden bg-slate-50">
      {/* Keyed by the saved chat, so that picking another example starts from its state. */}
      <NeuralTransparencyChat
        key={saved ? savedId : 'new'}
        axes={axes}
        savedId={saved ? savedId : undefined}
        initialMessages={saved?.messages}
        initialReadouts={saved?.readouts}
      />
    </div>
  );
}
