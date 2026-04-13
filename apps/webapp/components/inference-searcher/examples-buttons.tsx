'use client';

import { Button } from '@/components/shadcn/button';
import { INFERENCE_EXAMPLE_TEXTS, INFERENCE_EXAMPLE_TEXTS_CATEGORIZED } from '@/lib/utils/inference-example-texts';
import { useRouter } from '@bprogress/next';
import { Dices } from 'lucide-react';

export default function ExamplesButtons({ makeUrl }: { makeUrl: (query: string) => string }) {
  const router = useRouter();
  return (
    <>
      <div className="mb-1.5 flex flex-row items-center justify-start gap-x-2 text-center font-sans text-[10px] uppercase text-slate-500">
        Run Example Search
      </div>
      <div className="grid grid-cols-3 gap-x-1.5 gap-y-1.5 sm:grid-cols-6">
        <Button
          onClick={(e) => {
            e.preventDefault();
            const randomSentence = INFERENCE_EXAMPLE_TEXTS[Math.floor(Math.random() * INFERENCE_EXAMPLE_TEXTS.length)];
            router.push(makeUrl(randomSentence));
          }}
          variant="outline"
          size="sm"
          className="h-12 gap-x-2 shadow-sm"
        >
          <Dices className="h-4 w-4" />
          Random
        </Button>
        {INFERENCE_EXAMPLE_TEXTS_CATEGORIZED.map((category) => (
          <Button
            key={category.type}
            onClick={(e) => {
              e.preventDefault();
              const sentences = INFERENCE_EXAMPLE_TEXTS_CATEGORIZED.find((c) => c.type === category.type)?.examples;
              if (sentences) {
                const randomSentence = sentences[Math.floor(Math.random() * sentences.length)];
                router.push(makeUrl(randomSentence));
              }
            }}
            variant="outline"
            size="sm"
            className="h-12 gap-x-2 shadow-sm"
          >
            {category.type}
          </Button>
        ))}
      </div>
    </>
  );
}
