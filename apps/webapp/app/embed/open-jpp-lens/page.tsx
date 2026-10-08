import { Button } from '@/components/shadcn/button';

// Temporary: what the /qwen3 embed shows. The share opens in a new tab.
const JPP_LENS_SHARE_URL = 'https://www.neuronpedia.org/qwen3.6-27b/jlens?shareId=cmuyz763o0001182xcprl8ar8';

export default function Page() {
  return (
    <div className="flex min-h-[100dvh] w-full items-center justify-center">
      <Button asChild size="lg">
        <a href={JPP_LENS_SHARE_URL} target="_blank" rel="noopener noreferrer">
          Open J++ Lens
        </a>
      </Button>
    </div>
  );
}
