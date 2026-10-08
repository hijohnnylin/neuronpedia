import { Button } from '@/components/shadcn/button';
import Image from 'next/image';

// Temporary: what the /qwen3 embed shows. The share opens in a new tab.
const JPP_LENS_SHARE_URL = 'https://www.neuronpedia.org/qwen3.6-27b/jlens?shareId=cmuyz763o0001182xcprl8ar8';

export default function Page() {
  return (
    <div className="flex min-h-[100dvh] w-full flex-col items-center justify-center gap-y-5">
      <Button asChild size="lg">
        <a href={JPP_LENS_SHARE_URL} target="_blank" rel="noopener noreferrer">
          Open J++ Lens
        </a>
      </Button>
      <a
        href="https://www.neuronpedia.org"
        target="_blank"
        rel="noopener noreferrer"
        className="flex items-center text-base text-sky-800"
      >
        <Image
          src="/logo.png"
          alt="Neuronpedia logo - a computer chip with a rounded viewfinder border around it"
          width="20"
          height="20"
          className="mr-1.5"
        />
        Neuronpedia
      </a>
    </div>
  );
}
