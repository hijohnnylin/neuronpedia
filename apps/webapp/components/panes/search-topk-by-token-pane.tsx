'use client';

import SearchTopkByToken from '@/app/search-topk-by-token/search-topk-by-token';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/shadcn/card';

export default function SearchTopkByTokenPane({
  modelId,
  source,
  text,
  initialDensityThreshold,
  initialIgnoreBos,
  initialSortBy,
  showResultsInNewPage = false,
}: {
  modelId: string;
  source: string;
  text?: string | undefined;
  initialDensityThreshold?: number | undefined;
  initialIgnoreBos?: boolean | undefined;
  initialSortBy?: 'frequency' | 'strength' | 'density' | undefined;
  showResultsInNewPage?: boolean;
}) {
  return (
    <div className="flex w-full flex-col items-center">
      <Card className="w-full bg-white">
        <CardHeader className="w-full pb-3 pt-6">
          <div className="flex w-full flex-row items-center justify-between">
            <CardTitle>Search TopK by Token</CardTitle>
          </div>
        </CardHeader>
        <CardContent className="h-full w-full overflow-y-hidden pt-3">
          <SearchTopkByToken
            initialModelId={modelId}
            initialSource={source}
            initialText={text}
            initialDensityThreshold={initialDensityThreshold}
            initialIgnoreBos={initialIgnoreBos}
            initialSortBy={initialSortBy}
            showResultsInNewPage={showResultsInNewPage}
          />
        </CardContent>
      </Card>
    </div>
  );
}
