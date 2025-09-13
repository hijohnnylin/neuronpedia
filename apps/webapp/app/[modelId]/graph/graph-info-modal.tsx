'use client';

import { useGraphContext } from '@/components/provider/graph-provider';
import { Button } from '@/components/shadcn/button';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from '@/components/shadcn/dialog';
import { DownloadIcon, Info } from 'lucide-react';
import { CLTGraph, CLTGraphQParams } from './graph-types';

interface GraphInfoModalProps {
  cltGraph: CLTGraph | null;
  selectedMetadataGraph: any;
}

export default function GraphInfoModal({ cltGraph, selectedMetadataGraph }: GraphInfoModalProps) {
  const { getOverrideClerpForNode, selectedGraph, visState } = useGraphContext();

  const handleDownload = async () => {
    if (selectedMetadataGraph && selectedGraph) {
      try {
        // download a fresh copy of the graph
        const response = await fetch(selectedMetadataGraph.url);
        const data = (await response.json()) as CLTGraph;

        // add our custom labels to it
        data.nodes.forEach((node) => {
          // find the node in the original selectedGraph
          const originalNode = selectedGraph.nodes.find((n) => n.node_id === node.node_id);
          if (originalNode) {
            // eslint-disable-next-line no-param-reassign
            node.clerp = getOverrideClerpForNode(originalNode) || '';
          }
        });

        // add subgraph state to it
        const qParams: CLTGraphQParams = {
          linkType: 'both',
          pinnedIds: visState.pinnedIds,
          clickedId: '',
          supernodes: visState.subgraph?.supernodes || [],
          sg_pos: '',
        };
        data.qParams = qParams;

        // add current window url to metadata
        const location = window.location.href;
        // const sourceSetUrl = `${NEXT_PUBLIC_URL}/${selectedMetadataGraph.modelId}/${selectedMetadataGraph.sourceSetName}`;
        if (data.metadata.info) {
          data.metadata.info.neuronpedia_link = location;
          data.metadata.info.neuronpedia_source_set = selectedMetadataGraph.sourceSetName;
        } else {
          data.metadata.info = {
            neuronpedia_link: location,
            neuronpedia_source_set: selectedMetadataGraph.sourceSetName,
          };
        }

        // download it
        const blob = new Blob([JSON.stringify(data)], { type: 'application/json' });
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `${selectedMetadataGraph.slug}_${new Date().toISOString().replace(/[:.]/g, '-')}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        window.URL.revokeObjectURL(url);
      } catch (error) {
        console.error('Failed to download graph, no custom labels will be added', error);
        // Fallback to opening in new tab
        window.open(selectedMetadataGraph.url, '_blank');
      }
    }
  };

  if (!cltGraph) {
    return (
      <Button
        variant="outline"
        size="sm"
        disabled
        className="flex h-12 items-center justify-center gap-x-2 border-slate-300 text-xs text-slate-300"
      >
        <Info className="h-4 w-4" />
        <span className="hidden sm:block">Info</span>
      </Button>
    );
  }

  const { metadata } = cltGraph;

  return (
    <Dialog>
      <DialogTrigger asChild>
        <Button
          variant="outline"
          size="sm"
          title="Graph Information"
          aria-label="Graph Information"
          className="flex h-12 items-center justify-center gap-x-2 border-slate-300 text-xs text-slate-500 hover:bg-slate-50"
        >
          <Info className="h-4 w-4" />
          <span className="hidden sm:block">Info</span>
        </Button>
      </DialogTrigger>
      <DialogContent className="max-h-[80vh] max-w-2xl overflow-y-auto bg-white text-slate-700">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <Info className="h-5 w-5" />
            Graph Information
          </DialogTitle>
        </DialogHeader>

        <div className="space-y-3">
          {/* Basic Information */}
          <div className="space-y-3">
            <div className="grid grid-cols-2 text-sm">
              <div>
                <span className="text-[11px] font-medium uppercase text-slate-400">Graph Name (Slug/ID)</span>
                <p className="font-mono text-slate-800">{metadata.slug}</p>
              </div>
              <div>
                <span className="text-[11px] font-medium uppercase text-slate-400">Model</span>
                <p className="text-slate-800">{metadata.scan}</p>
              </div>
            </div>
          </div>

          {/* Prompt Information */}
          <div className="space-y-3">
            <div className="space-y-1.5">
              <div>
                <span className="text-[11px] font-medium uppercase text-slate-400">Prompt</span>
                <p className="rounded py-1.5 pb-0.5 text-xs text-slate-800">{metadata.prompt}</p>
              </div>
              <div>
                <div className="mt-1 flex flex-wrap gap-1">
                  {metadata.prompt_tokens.map((token, i) => (
                    <span
                      key={`${token}-${i}`}
                      className="mx-[1px] mb-1 rounded bg-slate-200 px-[2px] py-0.5 font-mono text-[11px] text-slate-800"
                    >
                      {token.replaceAll('\n', '⏎').replaceAll(' ', '\u00A0')}
                    </span>
                  ))}
                </div>
              </div>
            </div>
          </div>

          {/* Graph Statistics */}
          <div className="space-y-1.5 pt-0">
            <div className="grid grid-cols-2 text-sm">
              <div>
                <span className="text-[11px] font-medium uppercase text-slate-400"># Nodes</span>
                <p className="text-slate-800">{cltGraph.nodes?.length.toLocaleString() || 0}</p>
              </div>
              <div>
                <span className="text-[11px] font-medium uppercase text-slate-400"># Links</span>
                <p className="text-slate-800">{cltGraph.links?.length.toLocaleString() || 0}</p>
              </div>
            </div>
          </div>

          {/* Creator Information */}
          {metadata.info && (
            <div className="space-y-3">
              <div className="grid grid-cols-1 text-sm">
                <div className="grid grid-cols-2">
                  {metadata.info.creator_name && (
                    <div>
                      <span className="text-[11px] font-medium uppercase text-slate-400">Created By</span>
                      <p>
                        <a
                          href={metadata.info.creator_url}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="text-sky-700 hover:underline"
                        >
                          {metadata.info.creator_name}
                        </a>
                      </p>
                    </div>
                  )}
                  {metadata.info.create_time_ms && (
                    <div>
                      <span className="text-[11px] font-medium uppercase text-slate-400">Created On</span>
                      <p className="text-slate-800">{new Date(metadata.info.create_time_ms).toLocaleString()}</p>
                    </div>
                  )}
                </div>
                {metadata.info.description && (
                  <div>
                    <span className="text-[11px] font-medium uppercase text-slate-400">
                      Additional Description/Notes
                    </span>
                    <p className="text-slate-800">helloooooo {metadata.info.description}</p>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Generation Settings */}
          {(metadata.generation_settings || metadata.pruning_settings) && (
            <div className="flex w-full flex-row pt-3">
              {metadata.generation_settings && (
                <div className="flex-1 space-y-1.5">
                  <h3 className="mb-0 text-center text-xs font-medium uppercase leading-none text-slate-400">
                    Generation Settings
                  </h3>
                  <div className="grid grid-cols-2 gap-y-1 text-sm">
                    {metadata.generation_settings.max_n_logits && (
                      <div>
                        <span className="text-[11px] font-medium uppercase text-slate-400">Max N Logits</span>
                        <p className="text-slate-800">{metadata.generation_settings.max_n_logits}</p>
                      </div>
                    )}
                    {metadata.generation_settings.desired_logit_prob && (
                      <div>
                        <span className="text-[11px] font-medium uppercase text-slate-400">Desired Logit Prob</span>
                        <p className="text-slate-800">{metadata.generation_settings.desired_logit_prob}</p>
                      </div>
                    )}
                    {metadata.generation_settings.batch_size && (
                      <div>
                        <span className="text-[11px] font-medium uppercase text-slate-400">Batch Size</span>
                        <p className="text-slate-800">{metadata.generation_settings.batch_size}</p>
                      </div>
                    )}
                    {metadata.generation_settings.max_feature_nodes && (
                      <div>
                        <span className="text-[11px] font-medium uppercase text-slate-400">Max Feature Nodes</span>
                        <p className="text-slate-800">{metadata.generation_settings.max_feature_nodes}</p>
                      </div>
                    )}
                  </div>
                </div>
              )}

              {/* Pruning Settings */}
              {metadata.pruning_settings && (
                <div className="flex-1 space-y-1.5">
                  <h3 className="mb-0 text-center text-xs font-medium uppercase leading-none text-slate-400">
                    Pruning Settings
                  </h3>
                  <div className="grid grid-cols-2 gap-y-1 text-sm">
                    {metadata.pruning_settings.node_threshold && (
                      <div>
                        <span className="text-[11px] font-medium uppercase text-slate-400">Node Threshold</span>
                        <p className="text-slate-800">{metadata.pruning_settings.node_threshold}</p>
                      </div>
                    )}
                    {metadata.pruning_settings.edge_threshold && (
                      <div>
                        <span className="text-[11px] font-medium uppercase text-slate-400">Edge Threshold</span>
                        <p className="text-slate-800">{metadata.pruning_settings.edge_threshold}</p>
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Feature Details */}
          {metadata.feature_details && (
            <div className="space-y-3">
              <h3 className="text-lg font-semibold">Feature Details</h3>
              <div className="grid grid-cols-1 gap-4 text-sm">
                {metadata.feature_details.feature_json_base_url && (
                  <div>
                    <span className="font-medium text-slate-600">Feature JSON Base URL:</span>
                    <p className="break-all font-mono text-xs text-slate-800">
                      {metadata.feature_details.feature_json_base_url}
                    </p>
                  </div>
                )}
                {metadata.feature_details.neuronpedia_source_set && (
                  <div>
                    <span className="font-medium text-slate-600">Neuronpedia Source Set:</span>
                    <p className="text-slate-800">{metadata.feature_details.neuronpedia_source_set}</p>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Generator Information */}
          {metadata.info?.generator && (
            <div className="space-y-1 pt-1">
              <h3 className="text-left text-xs font-medium uppercase text-slate-400">Generator Used</h3>
              <div className="grid grid-cols-1 items-center justify-start gap-0 text-sm">
                {metadata.info.generator.name && (
                  <div>
                    <p className="text-left text-slate-800">{metadata.info.generator.name}</p>
                  </div>
                )}
                {metadata.info.generator.version && (
                  <div className="mt-0.5">
                    <p className="text-left font-mono text-xs text-slate-800">{metadata.info.generator.version}</p>
                  </div>
                )}
                {metadata.info.generator.url && (
                  <div>
                    <a
                      href={metadata.info.generator.url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="break-all text-left text-[11px] text-sky-600 hover:underline"
                    >
                      {metadata.info.generator.url}
                    </a>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Source URLs */}
          {metadata.info?.source_urls && metadata.info.source_urls.length > 0 && (
            <div className="space-y-1.5">
              <h3 className="text-sm font-semibold">Source URLs</h3>
              <div className="space-y-1">
                {metadata.info.source_urls.map((url, index) => (
                  <a
                    key={index}
                    href={url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="block break-all text-xs text-sky-600 hover:underline"
                  >
                    {url}
                  </a>
                ))}
                <a
                  href={selectedMetadataGraph.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="block break-all text-xs text-sky-600 hover:underline"
                >
                  Raw Graph JSON (no labels)
                </a>
              </div>
            </div>
          )}

          {/* Download Section */}
          <div className="border-t pt-4">
            <Button onClick={handleDownload} className="w-full gap-2" disabled={!selectedMetadataGraph}>
              <DownloadIcon className="h-4 w-4" />
              Download Graph JSON + Labels
            </Button>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}
