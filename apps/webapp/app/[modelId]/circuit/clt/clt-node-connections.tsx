import CustomTooltip from '@/components/custom-tooltip';
import { useCircuitCLT } from '@/components/provider/circuit-clt-provider';
import { QuestionMarkCircledIcon } from '@radix-ui/react-icons';
import { Circle } from 'lucide-react';
import { useEffect, useState } from 'react';
import { CLTGraphNode, CltVisState, featureTypeToText } from './clt-utils';

function FeatureList({
  title,
  nodes,
  linkType,
  visState,
  updateVisStateField,
}: {
  title: string;
  nodes: CLTGraphNode[];
  linkType: 'source' | 'target';
  visState: any;
  updateVisStateField: (field: keyof CltVisState, value: any) => void;
}) {
  const linkProp = linkType === 'source' ? 'tmpClickedSourceLink' : 'tmpClickedTargetLink';

  return (
    <div className="flex max-h-[320px] flex-1 flex-col gap-y-0.5 overflow-y-scroll px-1 text-slate-800">
      <div className="sticky top-0 bg-white pb-1 text-sm font-medium text-slate-600">{title}</div>
      {nodes
        ?.toSorted((a, b) => (b[linkProp]?.pctInput ?? 0) - (a[linkProp]?.pctInput ?? 0))
        .filter((node) => node[linkProp]?.pctInput !== null && node[linkProp]?.pctInput !== undefined)
        .map((node) => (
          <button
            type="button"
            key={node.nodeId}
            className={`flex cursor-pointer flex-row items-center justify-between gap-x-1.5 rounded bg-slate-50 px-2 py-[3px] text-[10px] hover:bg-sky-100 ${
              node.featureId === visState.hoveredId ? 'z-20 outline-dotted outline-[3px] outline-[#f0f]' : ''
            } ${(node[linkProp]?.pctInput ?? 0) > 0.25 ? 'text-white' : ''}`}
            style={{ backgroundColor: node[linkProp]?.tmpColor }}
            onMouseEnter={() => {
              updateVisStateField('hoveredId', node.featureId);
            }}
            onMouseLeave={() => {
              updateVisStateField('hoveredId', null);
            }}
            onClick={() => {
              updateVisStateField('clickedId', node.nodeId);
            }}
          >
            <svg width={10} height={10} className="mr-0 inline-block">
              <g>
                <g
                  className={`default-icon block fill-none ${(node[linkProp]?.pctInput ?? 0) > 0.25 ? 'stroke-white' : 'stroke-slate-800'} ${node.nodeId && visState.pinnedIds?.includes(node.nodeId) ? 'stroke-[1.7]' : 'stroke-[0.7]'}`}
                >
                  <text fontSize={15} textAnchor="middle" dominantBaseline="central" dx={5} dy={4}>
                    {featureTypeToText(node.feature_type)}
                  </text>
                </g>
              </g>
            </svg>
            <div className="flex-1 text-left leading-snug">{node.ppClerp}</div>
            {node[linkProp]?.tmpClickedCtxOffset !== undefined &&
              (node[linkProp]?.tmpClickedCtxOffset > 0 ? (
                <div
                  className={`${node[linkProp]?.pctInput !== null && node[linkProp]?.pctInput !== undefined && node[linkProp]?.pctInput > 0.25 ? 'text-white' : 'text-slate-600'}`}
                >
                  →
                </div>
              ) : node[linkProp]?.tmpClickedCtxOffset < 0 ? (
                <div
                  className={`${node[linkProp]?.pctInput !== null && node[linkProp]?.pctInput !== undefined && node[linkProp]?.pctInput > 0.25 ? 'text-white' : 'text-slate-600'}`}
                >
                  ←
                </div>
              ) : (
                ''
              ))}
            <div className="font-mono">
              {node[linkProp]?.pctInput !== null && node[linkProp]?.pctInput !== undefined
                ? node[linkProp]?.pctInput > 0
                  ? `+${node[linkProp]?.pctInput?.toFixed(3)}`
                  : node[linkProp]?.pctInput?.toFixed(3)
                : ''}
            </div>
          </button>
        ))}
    </div>
  );
}

export default function CLTNodeConnections() {
  const { visState, selectedGraph, updateVisStateField } = useCircuitCLT();

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

  return (
    <div className="node-connections relative mt-2 min-h-[320px] flex-1">
      <div className="mb-3 mt-2 flex w-full flex-row items-center justify-start gap-x-2 px-1">
        <div className="text-sm font-bold text-slate-600">Node Connections</div>
        <CustomTooltip wide trigger={<QuestionMarkCircledIcon className="h-4 w-4 text-slate-500" />}>
          <div className="flex flex-col">
            TODO: https://transformer-circuits.pub/2025/attribution-graphs/methods.html
          </div>
        </CustomTooltip>
      </div>
      <div className="flex w-full flex-col text-slate-700">
        {clickedNode ? (
          <div className="flex flex-row items-center gap-x-2 px-1 text-sm font-medium text-slate-600">
            <div className="">F#{clickedNode?.feature}</div>
            <Circle className="h-3.5 w-3.5 text-[#f0f]" />
            <div>{clickedNode?.ppClerp}</div>
          </div>
        ) : (
          <div className="px-1 text-sm font-medium text-slate-500">Click a feature on the left for details</div>
        )}
        {clickedNode && (
          <div className="mt-2 flex w-full flex-row gap-x-0">
            <FeatureList
              title="Input Features"
              nodes={selectedGraph?.nodes || []}
              linkType="source"
              visState={visState}
              updateVisStateField={updateVisStateField}
            />
            <FeatureList
              title="Output Features"
              nodes={selectedGraph?.nodes || []}
              linkType="target"
              visState={visState}
              updateVisStateField={updateVisStateField}
            />
          </div>
        )}
      </div>
    </div>
  );
}
