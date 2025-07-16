'use client';

import { Button } from '@/components/shadcn/button';
import { Card, CardContent } from '@/components/shadcn/card';
import { ArrowRight } from 'lucide-react';
import { useRouter } from 'next-nprogress-bar';
import { useEffect } from 'react';

const DEFAULT_GRAPH_MODEL_ID = 'gemma-2-2b';

export default function CircuitCLTRedirectPage({
  searchParams,
}: {
  searchParams: {
    logitDiff?: string;
    model?: string;
    slug?: string;
    pinnedIds?: string;
    supernodes?: string;
    clerps?: string;
  };
}) {
  const router = useRouter();
  let { model } = searchParams;

  // if no modelId, then use the default modelId
  if (!model || model.length === 0) {
    model = DEFAULT_GRAPH_MODEL_ID;
  }
  const searchParamsEntries = Object.entries(searchParams).filter(([key]) => key !== 'model');
  const queryString =
    searchParamsEntries.length > 0 ? `?${new URLSearchParams(Object.fromEntries(searchParamsEntries)).toString()}` : '';
  const newUrl = `/${model}/graph${queryString}`;

  // Automatically redirect after a short delay
  useEffect(() => {
    const timer = setTimeout(() => {
      router.push(newUrl);
    }, 3000);

    return () => clearTimeout(timer);
  }, [router, newUrl]);

  return (
    <div className="container mx-auto py-12">
      <Card className="mx-auto w-full max-w-2xl">
        <CardContent className="rounded-md border border-slate-200 bg-white pb-6 pt-6">
          <div className="flex flex-col items-center gap-6 text-center text-slate-700">
            <h1 className="text-xl font-bold">Attribution graphs have moved to a new URL format.</h1>
            <p className="text-muted-foreground">Redirecting you now...</p>
            <Button onClick={() => router.push(newUrl)} className="flex items-center gap-2">
              Redirect <ArrowRight className="h-4 w-4" />
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
