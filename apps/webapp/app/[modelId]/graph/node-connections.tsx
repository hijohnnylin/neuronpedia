import { useGraphContext } from '@/components/provider/graph-provider';
import { Circle } from 'lucide-react';
import { useEffect, useState } from 'react';
import GraphFeatureLink from './np-feature-link';
import {
  CLTGraphNode,
  CltVisState,
  featureTypeToText,
  graphModelHasNpDashboards,
  shouldShowNodeForDensityThreshold,
  shouldShowNodeForInfluenceThreshold,
} from './utils';

function FeatureList({
  title,
  nodes,
  linkType,
  hasNPDashboards,
  visState,
  updateVisStateField,
  isEditingLabel,
  getNodeSupernodeAndOverrideLabel,
}: {
  title: string;
  nodes: CLTGraphNode[];
  linkType: 'source' | 'target';
  hasNPDashboards: boolean;
  visState: any;
  updateVisStateField: (field: keyof CltVisState, value: any) => void;
  isEditingLabel: boolean;
  getNodeSupernodeAndOverrideLabel: (node: CLTGraphNode) => string;
}) {
  const linkProp = linkType === 'source' ? 'tmpClickedSourceLink' : 'tmpClickedTargetLink';
  const filteredNodes = nodes
    ?.toSorted((a, b) => (b[linkProp]?.pctInput ?? 0) - (a[linkProp]?.pctInput ?? 0))
    .filter((node) => {
      // no input = don't show
      if (node[linkProp]?.pctInput === null || node[linkProp]?.pctInput === undefined) {
        return false;
      }

      // otherwise there is some input, but check if we should check both influence and density filters
      return (
        shouldShowNodeForInfluenceThreshold(node, visState) &&
        shouldShowNodeForDensityThreshold(hasNPDashboards, node, visState)
      );
    });
  return (
    <div className="flex flex-1 flex-col gap-y-0.5 overflow-y-scroll overscroll-none px-1 pb-1 text-slate-800">
      <div className="sticky top-0 bg-white pb-0.5 text-[10px] font-medium uppercase text-slate-500">
        {title} ({filteredNodes.length})
      </div>
      {filteredNodes.map((node, idx) => (
        <button
          type="button"
          key={`${node.featureId}-${idx}`}
          className={`flex cursor-pointer flex-row items-center justify-between gap-x-1.5 rounded bg-slate-50 px-2 py-[3px] text-[10px] hover:bg-sky-100 ${
            node.featureId === visState.hoveredId ? 'z-20 outline-dotted outline-[3px] outline-[#f0f]' : ''
          } ${(node[linkProp]?.pctInput ?? 0) > 0.25 || (node[linkProp]?.pctInput ?? 0) < -0.25 ? 'text-white' : ''}`}
          style={{ backgroundColor: node[linkProp]?.tmpColor }}
          onMouseEnter={() => {
            if (!isEditingLabel) {
              updateVisStateField('hoveredId', node.featureId);
            }
          }}
          onMouseLeave={() => {
            updateVisStateField('hoveredId', null);
          }}
          onClick={(e) => {
            if (e.ctrlKey || e.metaKey) {
              // If control or command key is pressed, add to pinnedIds
              updateVisStateField('pinnedIds', [...(visState.pinnedIds || []), node.nodeId]);
            } else {
              // Otherwise just set as clicked
              updateVisStateField('clickedId', node.nodeId);
            }
          }}
        >
          <svg width={10} height={14} className="mr-0 inline-block">
            <g>
              <g
                className={`default-icon block fill-none ${(node[linkProp]?.pctInput ?? 0) > 0.25 || (node[linkProp]?.pctInput ?? 0) < -0.25 ? 'stroke-white' : 'stroke-slate-800'} ${node.nodeId && visState.pinnedIds?.includes(node.nodeId) ? 'stroke-[1.7]' : 'stroke-[0.7]'}`}
              >
                <text fontSize={15} textAnchor="middle" dominantBaseline="central" dx={5} dy={5}>
                  {featureTypeToText(node.feature_type)}
                </text>
              </g>
            </g>
          </svg>
          <div className="flex-1 text-left leading-snug">{getNodeSupernodeAndOverrideLabel(node)}</div>
          {node[linkProp]?.tmpClickedCtxOffset !== undefined &&
            (node[linkProp]?.tmpClickedCtxOffset > 0 ? (
              <div>→</div>
            ) : node[linkProp]?.tmpClickedCtxOffset < 0 ? (
              <div>←</div>
            ) : (
              ''
            ))}
          <div className="flex flex-col items-center justify-center">
            <div className="font-mono">
              {node[linkProp]?.pctInput !== null && node[linkProp]?.pctInput !== undefined
                ? node[linkProp]?.pctInput > 0
                  ? `+${node[linkProp]?.pctInput?.toFixed(3)}`
                  : node[linkProp]?.pctInput?.toFixed(3)
                : ''}
            </div>
            <div className="font-mono">{node.layer !== 'E' ? `L${node.layer}` : ''}</div>
          </div>
        </button>
      ))}
    </div>
  );
}

export default function GraphNodeConnections() {
  const {
    visState,
    selectedGraph,
    updateVisStateField,
    isEditingLabel,
    getNodeSupernodeAndOverrideLabel,
    setFullNPFeatureDetail,
  } = useGraphContext();

  const [clickedNode, setClickedNode] = useState<CLTGraphNode | null>(null);

  useEffect(() => {
    if (visState.clickedId) {
      const cNode = selectedGraph?.nodes.find((e) => e.nodeId === visState.clickedId);
      if (cNode) {
        setClickedNode(cNode);
      }
    } else {
      setClickedNode(null);
    }
  }, [visState.clickedId, selectedGraph]);

  useEffect(() => {
    if (clickedNode) {
      setFullNPFeatureDetail(setClickedNode, clickedNode);
    }
  }, [clickedNode]);

  return (
    <div className="node-connections relative mt-2 flex max-w-[420px] flex-1 flex-row overflow-y-hidden rounded-lg border border-slate-200 bg-white px-2 py-2 shadow-sm transition-all">
      {/* <div className="absolute top-0 mx-auto h-3 w-24 rounded-b bg-[#f0f] text-[6px] text-white">Clicked Node</div> */}

      <div className="flex w-full flex-col text-slate-700">
        {clickedNode ? (
          <div className="flex flex-row items-center gap-x-2 text-xs font-medium text-slate-600">
            {!clickedNode?.featureDetailNP && <div className="">F#{clickedNode?.feature}</div>}
            <Circle className="h-3.5 max-h-3.5 min-h-3.5 w-3.5 min-w-3.5 max-w-3.5 text-[#f0f]" />
            <div className="flex-1 leading-tight">{getNodeSupernodeAndOverrideLabel(clickedNode)}</div>
            <GraphFeatureLink selectedGraph={selectedGraph} node={clickedNode} />
          </div>
        ) : (
          <div className="flex h-[100%] flex-col items-center justify-center text-center text-sm font-medium text-slate-700">
            <div className="mb-2 text-lg font-bold">Node Connections</div>
            <div className="">Click a node on the left to see its connections.</div>
          </div>
        )}
        {clickedNode && (
          <div
            className={`mt-2 flex h-full w-full flex-1 flex-row gap-x-0 ${clickedNode?.featureDetailNP ? 'pb-9' : 'pb-5'}`}
          >
            <FeatureList
              title="Input Features"
              nodes={selectedGraph?.nodes.filter((node) => node.feature_type !== 'mlp reconstruction error') || []}
              linkType="source"
              hasNPDashboards={selectedGraph ? graphModelHasNpDashboards(selectedGraph) : false}
              visState={visState}
              updateVisStateField={updateVisStateField}
              isEditingLabel={isEditingLabel}
              getNodeSupernodeAndOverrideLabel={getNodeSupernodeAndOverrideLabel}
            />
            <FeatureList
              title="Output Features"
              nodes={selectedGraph?.nodes.filter((node) => node.feature_type !== 'mlp reconstruction error') || []}
              linkType="target"
              hasNPDashboards={selectedGraph ? graphModelHasNpDashboards(selectedGraph) : false}
              visState={visState}
              updateVisStateField={updateVisStateField}
              isEditingLabel={isEditingLabel}
              getNodeSupernodeAndOverrideLabel={getNodeSupernodeAndOverrideLabel}
            />
          </div>
        )}
      </div>
    </div>
  );
}
