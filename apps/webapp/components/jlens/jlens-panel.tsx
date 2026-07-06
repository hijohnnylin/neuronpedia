'use client';

import { useGlobalContext } from '@/components/provider/global-provider';
import { ChineseTranslationsProvider } from '@/lib/utils/chinese-translations';
import JlensChat from './jlens-chat';
import JlensCompletion from './jlens-completion';

// Fraction of layers to discard from the *start* (lowest layers) when
// rendering lens predictions in the UI. With a value of 3, the first 1/3 of
// layers are hidden (e.g. 8 of 24 layers). The remaining layers keep their
// true layer numbers — only display is trimmed, the data is untouched.
export const START_LAYER_FRACTION = 2;

// Top-level jlens panel. Picks the interface based on the model's `instruct`
// flag: instruct models get the chat interface, base models get the
// single-shot completion interface.
export default function JlensPanel({
  modelId,
  inferenceAvailable = true,
}: {
  modelId: string;
  inferenceAvailable?: boolean;
}) {
  const { globalModels } = useGlobalContext();
  const isInstruct = !!globalModels[modelId]?.instruct;

  return (
    <ChineseTranslationsProvider>
      <div className="flex min-h-0 w-full flex-1 flex-col sm:gap-y-3">
        {isInstruct ? (
          <JlensChat key="live" modelId={modelId} inferenceAvailable={inferenceAvailable} />
        ) : (
          <JlensCompletion key="live" modelId={modelId} inferenceAvailable={inferenceAvailable} />
        )}
      </div>
    </ChineseTranslationsProvider>
  );
}
