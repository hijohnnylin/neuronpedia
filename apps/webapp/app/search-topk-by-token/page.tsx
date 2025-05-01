import BreadcrumbsComponent from '@/components/breadcrumbs-component';
import { BreadcrumbLink } from '@/components/shadcn/breadcrumbs';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/shadcn/card';
import SearchTopkByToken from './search-topk-by-token';

export default async function Page({
  searchParams,
}: {
  searchParams: { [key: string]: string | string[] | undefined };
}) {
  const modelId = typeof searchParams.modelId === 'string' ? searchParams.modelId : undefined;
  const source = typeof searchParams.source === 'string' ? searchParams.source : undefined;
  const text = typeof searchParams.text === 'string' ? searchParams.text : undefined;
  const densityThresholdParam =
    typeof searchParams.densityThreshold === 'string' ? searchParams.densityThreshold : undefined;
  const initialDensityThreshold = densityThresholdParam ? parseFloat(densityThresholdParam) : undefined;
  const ignoreBosParam = typeof searchParams.ignoreBos === 'string' ? searchParams.ignoreBos : undefined;
  const initialIgnoreBos = ignoreBosParam === 'true' ? true : ignoreBosParam === 'false' ? false : undefined;

  const sortByParam = typeof searchParams.sortBy === 'string' ? searchParams.sortBy : undefined;
  const initialSortBy =
    sortByParam === 'frequency' || sortByParam === 'strength' || sortByParam === 'density' ? sortByParam : undefined;

  return (
    <div className="flex h-full w-full flex-col items-center justify-center sm:max-h-[calc(100vh-80px)]">
      <BreadcrumbsComponent
        crumbsArray={[
          <BreadcrumbLink href="/search-topk-by-token" key={1}>
            Search TopK by Token
          </BreadcrumbLink>,
          ...(text && modelId && source
            ? [
                <BreadcrumbLink href={`/search-topk-by-token?modelId=${modelId}&source=${source}&text=${text}`} key={2}>
                  {`"${text.trim().slice(0, 12)}..."`}
                </BreadcrumbLink>,
              ]
            : []),
        ]}
      />
      <Card className="mt-3 w-full max-w-screen-xl overflow-y-hidden bg-white sm:h-full">
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
          />
        </CardContent>
      </Card>
    </div>
  );
}
