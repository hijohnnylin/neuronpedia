import { getGraphUrl, useCircuitCLT } from '@/components/provider/circuit-clt-provider';
import { Button } from '@/components/shadcn/button';
import * as Select from '@radix-ui/react-select';
import { ChevronDownIcon, ChevronUpIcon, DownloadIcon } from 'lucide-react';

export default function CLTModelPromptSelector() {
  const {
    metadata,
    modelToBaseUrl,
    setSelectedModelId,
    selectedModelId,
    selectedMetadataGraph,
    setSelectedMetadataGraph,
    metadataScanToModelDisplayName,
  } = useCircuitCLT();

  return (
    <div className="flex w-full flex-col">
      <div className="pb-1 text-[9px] font-medium uppercase text-slate-400">Select a Model and Prompt</div>
      <div className="flex w-full flex-row gap-x-2">
        <Select.Root
          value={selectedModelId}
          onValueChange={(newVal) => {
            setSelectedModelId(newVal);
            setSelectedMetadataGraph(metadata[newVal][0]);
          }}
        >
          <Select.Trigger
            onKeyDown={(e) => {
              if (e.key === 'g') {
                e.preventDefault();
                e.stopPropagation();
              }
            }}
            className="inline-flex h-12 w-52 max-w-52 items-center justify-between gap-1 rounded border border-slate-300 bg-white px-4 py-2 text-sm leading-none"
          >
            <Select.Value>
              {metadataScanToModelDisplayName.get(selectedModelId) ? (
                <div className="flex flex-col items-start justify-start gap-y-0.5 text-left">
                  <div className="text-xs font-medium text-slate-600">
                    {metadataScanToModelDisplayName.get(selectedModelId)}
                  </div>
                  <div className="text-[9px] font-normal text-slate-400">{selectedModelId}</div>
                </div>
              ) : (
                <div className="text-slate-400">Select a model</div>
              )}
            </Select.Value>
            <Select.Icon>
              <ChevronDownIcon className="w-5 text-slate-500" />
            </Select.Icon>
          </Select.Trigger>
          <Select.Portal>
            <Select.Content
              position="popper"
              align="center"
              sideOffset={3}
              className="z-[99999] max-h-[400px] overflow-hidden rounded-md border bg-white shadow-lg"
            >
              <Select.ScrollUpButton className="flex h-7 cursor-pointer items-center justify-center bg-white text-slate-700 hover:bg-slate-100">
                <ChevronUpIcon className="w-5 text-slate-500" />
              </Select.ScrollUpButton>
              <Select.Viewport className="w-full divide-y divide-slate-100 p-2 text-slate-700">
                {Object.keys(metadata).map((modelId) => (
                  <Select.Item
                    key={modelId}
                    value={modelId}
                    className="relative flex h-12 w-full cursor-pointer select-none items-center overflow-x-hidden whitespace-pre rounded py-0 pl-4 pr-6 text-xs hover:bg-slate-100 data-[highlighted]:bg-slate-100 data-[highlighted]:outline-none"
                  >
                    <Select.ItemText className="w-full">
                      <div className="flex w-full flex-col items-start justify-start gap-y-0">
                        <div className="w-full truncate text-left">{metadataScanToModelDisplayName.get(modelId)}</div>
                        <div className="w-full text-[9px] font-normal text-slate-400">{modelId}</div>
                      </div>
                    </Select.ItemText>
                  </Select.Item>
                ))}
              </Select.Viewport>
              <Select.ScrollDownButton className="flex h-7 cursor-pointer items-center justify-center bg-white text-slate-700 hover:bg-slate-100">
                <ChevronDownIcon className="w-5 text-slate-500" />
              </Select.ScrollDownButton>
            </Select.Content>
          </Select.Portal>
        </Select.Root>

        <Select.Root
          value={selectedMetadataGraph?.slug}
          onValueChange={(newVal) => {
            const newGraph = metadata[selectedModelId]?.find((graph) => graph.slug === newVal);
            if (newGraph) {
              setSelectedMetadataGraph(newGraph);
            }
          }}
        >
          <Select.Trigger
            onKeyDown={(e) => {
              if (e.key === 'g') {
                e.preventDefault();
                e.stopPropagation();
              }
            }}
            className="relative inline-flex h-12 w-full flex-1 items-center justify-between gap-1 overflow-hidden whitespace-pre rounded border border-slate-300 bg-white px-4 py-2 text-sm leading-none"
          >
            <Select.Value>
              {selectedMetadataGraph ? (
                <div className="flex flex-col items-start justify-start gap-y-0.5">
                  <div className="text-xs font-medium text-slate-600">
                    {selectedMetadataGraph.prompt.replaceAll('\n', ' ').trim()}
                  </div>
                  <div className="text-[9px] font-normal text-slate-400">{selectedMetadataGraph.slug}</div>
                </div>
              ) : (
                <div className="text-slate-400">Select a prompt</div>
              )}
            </Select.Value>
            <Select.Icon className="absolute right-0 w-8 bg-white">
              <ChevronDownIcon className="ml-1 w-5 text-slate-500" />
            </Select.Icon>
          </Select.Trigger>
          <Select.Portal>
            <Select.Content
              position="popper"
              align="start"
              sideOffset={3}
              className="z-[99999] max-h-[640px] w-full max-w-[800px] overflow-hidden rounded-md border bg-white shadow-lg"
            >
              <Select.ScrollUpButton className="flex h-7 cursor-pointer items-center justify-center bg-white text-slate-700 hover:bg-slate-100">
                <ChevronUpIcon className="w-5 text-slate-500" />
              </Select.ScrollUpButton>
              <Select.Viewport className="divide-y divide-slate-100 p-2 text-slate-700">
                {metadata[selectedModelId]?.map((graph) => (
                  <Select.Item
                    key={graph.slug}
                    value={graph.slug}
                    className="relative flex w-full cursor-pointer select-none items-center overflow-x-hidden rounded py-2.5 pl-4 pr-6 text-xs hover:bg-slate-100 data-[highlighted]:bg-slate-100 data-[highlighted]:outline-none"
                  >
                    <Select.ItemText>
                      <div className="flex flex-col items-start justify-start gap-y-0">
                        <div className="whitespace-pre-line text-[11px] text-slate-600">{graph.prompt.trim()}</div>
                        <div className="text-[9px] font-normal text-slate-400">{graph.slug}</div>
                      </div>
                    </Select.ItemText>
                  </Select.Item>
                ))}
              </Select.Viewport>
              <Select.ScrollDownButton className="flex h-7 cursor-pointer items-center justify-center bg-white text-slate-700 hover:bg-slate-100">
                <ChevronDownIcon className="w-5 text-slate-500" />
              </Select.ScrollDownButton>
            </Select.Content>
          </Select.Portal>
        </Select.Root>
        {selectedMetadataGraph && (
          <Button
            variant="outline"
            size="sm"
            aria-label="Download Graph JSON"
            className="flex h-12 items-center justify-center whitespace-nowrap border-slate-300 text-sm text-slate-600 hover:bg-slate-50"
            onClick={() => {
              if (selectedMetadataGraph) {
                const url = getGraphUrl(selectedMetadataGraph.slug, modelToBaseUrl[selectedModelId]);
                window.open(url, '_blank');
              }
            }}
          >
            <DownloadIcon className="h-4 w-4" />
          </Button>
        )}
      </div>
    </div>
  );
}
