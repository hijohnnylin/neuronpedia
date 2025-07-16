import { CLTGraph, CLTGraphNode } from '@/app/[modelId]/graph/utils';
import { Button } from '@/components/shadcn/button';
import { BOS_TOKENS } from '@/lib/utils/activations';
import { SteerLogitFeature, SteeredPositionIdentifier } from '@/lib/utils/graph';
import {
  STEER_MULTIPLIER_STEP,
  STEER_STRENGTH_ADDED_MULTIPLIER_CUSTOM_GRAPH,
  STEER_STRENGTH_ADDED_MULTIPLIER_GRAPH,
  STEER_STRENGTH_ADDED_MULTIPLIER_MAX,
  STEER_STRENGTH_ADDED_MULTIPLIER_MIN,
} from '@/lib/utils/steer';
import * as Slider from '@radix-ui/react-slider';
import { MousePointerClick, Trash2 } from 'lucide-react';
import { useEffect, useRef } from 'react';
import MiniDashboardHoverCard from './mini-dashboard-hover-card';

// Each NodeToSteer is a single feature for a single active "original" token position.
// Each node can have multiple positions that it's steered at.
export default function NodeToSteer({
  node,
  sourceId,
  nodeSteerIdentifier,
  label,
  selectedGraph,
  steeredPositions,
  setSteeredPositions,
  isSteered,
  findSteeredPositionByPosition,
  removeSteeredPosition,
  setSteeredPositionDeltaByPosition,
  getDeltaToAddForMultiplier,
  getTopActivationValue,
  isCustomSteerNode = false,
  isInSupernode = false,
  allTokensHaveSameDelta,
  findSteeredPositionSteerGeneratedTokens,
  setSteeredPositionDeltaSteerGeneratedTokens,
  setAllTokensDelta,
  isSteering,
}: {
  node: CLTGraphNode;
  sourceId: string;
  nodeSteerIdentifier: SteeredPositionIdentifier;
  label: string;
  selectedGraph: CLTGraph;
  steeredPositions: SteerLogitFeature[];
  setSteeredPositions: (features: SteerLogitFeature[]) => void;
  isSteered: (identifier: SteeredPositionIdentifier) => boolean;
  findSteeredPositionByPosition: (
    identifier: SteeredPositionIdentifier,
    position: number,
  ) => SteerLogitFeature | undefined;
  removeSteeredPosition: (identifier: SteeredPositionIdentifier) => void;
  setSteeredPositionDeltaByPosition: (identifier: SteeredPositionIdentifier, position: number, delta: number) => void;
  getDeltaToAddForMultiplier: (node: CLTGraphNode, multiplier: number) => number;
  getTopActivationValue: (node: CLTGraphNode) => number;
  isCustomSteerNode?: boolean;
  isInSupernode?: boolean;
  allTokensHaveSameDelta: (nodeSteerIdentifier: SteeredPositionIdentifier) => number | false | null;
  findSteeredPositionSteerGeneratedTokens: (
    nodeSteerIdentifier: SteeredPositionIdentifier,
  ) => SteerLogitFeature | undefined;
  setSteeredPositionDeltaSteerGeneratedTokens: (nodeSteerIdentifier: SteeredPositionIdentifier, delta: number) => void;
  setAllTokensDelta: (nodeSteerIdentifier: SteeredPositionIdentifier, delta: number) => void;
  isSteering: boolean;
}) {
  const scrollIntoViewRef = useRef<HTMLDivElement>(null);

  // Scroll to the position when the component mounts or when steered features change
  useEffect(() => {
    // for some reason the scroll doesn't set to the correct position if a timeout is not set
    setTimeout(() => {
      if (scrollIntoViewRef.current && isSteered(nodeSteerIdentifier)) {
        // Find the immediate parent container (the horizontal scroll container)
        const parentContainer = scrollIntoViewRef.current.parentElement;
        if (parentContainer) {
          const elementRect = scrollIntoViewRef.current.getBoundingClientRect();
          const parentRect = parentContainer.getBoundingClientRect();

          // Calculate the scroll position to center the element in the parent
          const scrollLeft =
            parentContainer.scrollLeft +
            (elementRect.left - parentRect.left) -
            (parentRect.width - elementRect.width) / 2;

          parentContainer.scrollTo({
            left: scrollLeft,
            behavior: 'smooth',
          });
        }
      }
    }, 1);
  }, [isSteered, nodeSteerIdentifier]);

  return (
    <div key={node.nodeId} className="relative flex w-full flex-col gap-y-1 rounded-md pl-6 pt-0.5">
      {isInSupernode && (
        <div className={`absolute left-3 top-4 z-0 h-[1px] w-[12px] bg-sky-700 ${isSteering ? 'opacity-50' : ''}`} />
      )}
      <div className="flex flex-row items-center gap-x-3 text-[10px]">
        <Button
          disabled={isSteering}
          onClick={() => {
            if (isSteered(nodeSteerIdentifier)) {
              removeSteeredPosition(nodeSteerIdentifier);
            } else {
              setSteeredPositionDeltaByPosition(
                nodeSteerIdentifier,
                node.ctx_idx,
                getDeltaToAddForMultiplier(
                  node,
                  isCustomSteerNode
                    ? STEER_STRENGTH_ADDED_MULTIPLIER_CUSTOM_GRAPH
                    : STEER_STRENGTH_ADDED_MULTIPLIER_GRAPH,
                ),
              );
            }
          }}
          className={`h-6 w-16 rounded-full border text-[9px] font-medium uppercase ${
            isSteered(nodeSteerIdentifier)
              ? 'border-red-600 bg-red-50 text-red-600 hover:bg-red-100'
              : 'border-sky-700 bg-white text-sky-800 hover:bg-sky-200'
          }`}
        >
          {isSteered(nodeSteerIdentifier) ? 'Unsteer' : 'Steer'}
        </Button>
        <div className="flex flex-1 basis-1/2 flex-col gap-y-[1px]">
          <div className="-mt-[1px] line-clamp-1 flex-1 text-xs" title={label}>
            {label}
          </div>
          <div className="flex w-full flex-row items-center justify-start text-[9px] font-medium leading-none text-slate-400 group-hover:text-slate-500">
            Activation <span className="px-1 font-mono text-sky-700">{node.activation?.toFixed(2)}</span> at{' '}
            <div className="mx-1 rounded-sm bg-slate-200 px-0.5 py-0.5 font-mono text-slate-600">
              {selectedGraph.metadata.prompt_tokens[nodeSteerIdentifier.tokenActivePosition]
                .toString()
                .replaceAll(' ', '\u00A0')
                .replaceAll('\n', '↵')}
            </div>{' '}
            position {nodeSteerIdentifier.tokenActivePosition}
          </div>
        </div>

        <div className="flex flex-col gap-x-1 pl-0">
          <MiniDashboardHoverCard sourceId={sourceId} nodeSteerIdentifier={nodeSteerIdentifier} node={node} />
        </div>
      </div>
      {isSteered(nodeSteerIdentifier) && (
        <div className="mt-0.5 flex w-full flex-row items-center gap-x-1">
          <div className="flex w-full flex-col">
            <div className="flex max-w-full flex-1 flex-row items-start gap-x-1.5">
              <div className="flex min-w-11 flex-col items-center justify-center gap-y-1 rounded bg-slate-200/50 py-2">
                <Slider.Root
                  orientation="vertical"
                  defaultValue={[0]}
                  min={STEER_STRENGTH_ADDED_MULTIPLIER_MIN}
                  max={STEER_STRENGTH_ADDED_MULTIPLIER_MAX}
                  step={STEER_MULTIPLIER_STEP}
                  value={(() => {
                    const delta = allTokensHaveSameDelta(nodeSteerIdentifier);
                    if (delta === null || delta === false) {
                      return [0];
                    }
                    return [Number((delta / getTopActivationValue(node)).toFixed(1)) || 0];
                  })()}
                  onValueChange={(value) => {
                    setAllTokensDelta(nodeSteerIdentifier, getDeltaToAddForMultiplier(node, value[0]));
                  }}
                  className={`group relative flex h-24 w-2 items-center justify-center overflow-visible ${
                    allTokensHaveSameDelta(nodeSteerIdentifier) === false
                      ? 'opacity-50 hover:opacity-70'
                      : 'cursor-pointer'
                  }`}
                  disabled={isSteering}
                >
                  <Slider.Track className="relative h-full w-[4px] grow cursor-pointer rounded-full border border-sky-600 bg-sky-600 disabled:border-slate-300 disabled:bg-slate-100 group-hover:bg-sky-700">
                    <Slider.Range className="absolute h-full rounded-full" />
                  </Slider.Track>
                  <Slider.Thumb
                    onClick={() => {
                      if (!isSteering && !allTokensHaveSameDelta(nodeSteerIdentifier)) {
                        setAllTokensDelta(nodeSteerIdentifier, 0);
                      }
                    }}
                    className="relative flex h-5 w-9 cursor-pointer select-none items-center justify-center overflow-visible rounded-full border border-sky-700 bg-white text-[10px] font-medium leading-none text-sky-700 shadow disabled:border-slate-300 disabled:bg-slate-100 group-hover:bg-sky-100"
                  >
                    {allTokensHaveSameDelta(nodeSteerIdentifier) === false ? (
                      <MousePointerClick className="h-3.5 w-3.5" />
                    ) : allTokensHaveSameDelta(nodeSteerIdentifier) === null ? (
                      <span className="text-[7px] font-bold">ABLATE</span>
                    ) : (
                      // @ts-ignore
                      `${allTokensHaveSameDelta(nodeSteerIdentifier) ? (allTokensHaveSameDelta(nodeSteerIdentifier) > 0 ? '+' : '') : ''}${(allTokensHaveSameDelta(nodeSteerIdentifier) / getTopActivationValue(node)).toFixed(1) || '0'}×`
                    )}
                  </Slider.Thumb>
                </Slider.Root>
                <div className="mt-0.5 flex h-5 flex-col items-center justify-center gap-y-[1px] rounded px-1 py-0 text-center text-[7px] font-medium leading-none text-slate-500">
                  <div>ALL</div>
                  <div>TOKENS</div>
                </div>
              </div>
              <div className="forceShowScrollBarHorizontal flex max-w-full flex-row items-end overflow-x-scroll px-2 py-2 pb-0.5">
                {selectedGraph?.metadata.prompt_tokens.map((token, i) => (
                  <div
                    key={`${node.nodeId}-${i}`}
                    ref={node.ctx_idx === i ? scrollIntoViewRef : undefined}
                    className={`mx-1.5 min-w-fit flex-col items-center justify-center gap-y-1 ${
                      BOS_TOKENS.includes(token) ? 'hidden' : 'flex'
                    }`}
                  >
                    <Slider.Root
                      orientation="vertical"
                      defaultValue={[0]}
                      min={STEER_STRENGTH_ADDED_MULTIPLIER_MIN}
                      max={STEER_STRENGTH_ADDED_MULTIPLIER_MAX}
                      step={STEER_MULTIPLIER_STEP}
                      value={[
                        (findSteeredPositionByPosition(nodeSteerIdentifier, i)?.delta || 0) /
                          getTopActivationValue(node) || 0,
                      ]}
                      onValueChange={(value) => {
                        setSteeredPositionDeltaByPosition(
                          nodeSteerIdentifier,
                          i,
                          getDeltaToAddForMultiplier(node, value[0]),
                        );
                      }}
                      className={`group relative flex h-24 w-2 items-center justify-center overflow-visible ${
                        !findSteeredPositionByPosition(nodeSteerIdentifier, i)
                          ? 'opacity-50 hover:opacity-70'
                          : 'cursor-pointer'
                      }`}
                      disabled={isSteering}
                    >
                      <Slider.Track className="relative h-full w-[4px] grow cursor-pointer rounded-full border border-sky-600 bg-sky-600 disabled:border-slate-300 disabled:bg-slate-100 group-hover:bg-sky-700">
                        <Slider.Range className="absolute h-full rounded-full" />
                      </Slider.Track>
                      <Slider.Thumb
                        onClick={() => {
                          if (!isSteering && !findSteeredPositionByPosition(nodeSteerIdentifier, i)) {
                            setSteeredPositionDeltaByPosition(nodeSteerIdentifier, i, 0);
                          }
                        }}
                        className="relative flex h-5 w-9 cursor-pointer select-none items-center justify-center overflow-visible rounded-full border border-sky-700 bg-white text-[10px] font-medium leading-none text-sky-700 shadow disabled:border-slate-300 disabled:bg-slate-100 group-hover:bg-sky-100"
                      >
                        {!findSteeredPositionByPosition(nodeSteerIdentifier, i) ? (
                          <MousePointerClick className="h-3.5 w-3.5" />
                        ) : findSteeredPositionByPosition(nodeSteerIdentifier, i)?.ablate ? (
                          <span className="text-[7px] font-bold">ABLATE</span>
                        ) : (
                          // @ts-ignore
                          `${findSteeredPositionByPosition(nodeSteerIdentifier, i)?.delta ? (findSteeredPositionByPosition(nodeSteerIdentifier, i)?.delta > 0 ? '+' : '') : ''}${((findSteeredPositionByPosition(nodeSteerIdentifier, i)?.delta || 0) / getTopActivationValue(node)).toFixed(1) || '0'}×`
                        )}
                      </Slider.Thumb>
                    </Slider.Root>
                    <div className="mt-0.5 h-5 rounded bg-slate-200 px-1 py-0.5 font-mono text-[8.5px] text-slate-700">
                      {token.toString().replaceAll(' ', '\u00A0').replaceAll('\n', '↵')}
                    </div>
                    <Button
                      variant="ghost"
                      size="sm"
                      className={`flex h-6 w-6 min-w-6 rounded p-0 ${
                        findSteeredPositionByPosition(nodeSteerIdentifier, i)
                          ? 'text-red-700 hover:bg-red-100'
                          : 'text-slate-400'
                      }`}
                      disabled={isSteering || !findSteeredPositionByPosition(nodeSteerIdentifier, i)}
                      onClick={() =>
                        setSteeredPositions(
                          steeredPositions.filter(
                            (f) =>
                              !(
                                f.layer === nodeSteerIdentifier.layer &&
                                f.index === nodeSteerIdentifier.index &&
                                f.steer_position === i &&
                                f.token_active_position === nodeSteerIdentifier.tokenActivePosition
                              ),
                          ),
                        )
                      }
                    >
                      <Trash2 className="h-3.5 w-3.5" />
                    </Button>
                  </div>
                ))}
              </div>
              <div className="flex w-11 min-w-11 max-w-11 flex-col items-center justify-center gap-y-1 rounded bg-slate-200/50 py-2 pb-1">
                <Slider.Root
                  orientation="vertical"
                  defaultValue={[0]}
                  min={STEER_STRENGTH_ADDED_MULTIPLIER_MIN}
                  max={STEER_STRENGTH_ADDED_MULTIPLIER_MAX}
                  step={STEER_MULTIPLIER_STEP}
                  value={[
                    (findSteeredPositionSteerGeneratedTokens(nodeSteerIdentifier)?.delta || 0) /
                      getTopActivationValue(node) || 0,
                  ]}
                  onValueChange={(value) => {
                    setSteeredPositionDeltaSteerGeneratedTokens(
                      nodeSteerIdentifier,
                      getDeltaToAddForMultiplier(node, value[0]),
                    );
                  }}
                  className={`group relative flex h-24 w-2 items-center justify-center overflow-visible ${
                    !findSteeredPositionSteerGeneratedTokens(nodeSteerIdentifier)
                      ? 'opacity-50 hover:opacity-70'
                      : 'cursor-pointer'
                  }`}
                  disabled={isSteering}
                >
                  <Slider.Track className="relative h-full w-[4px] grow cursor-pointer rounded-full border border-sky-600 bg-sky-600 disabled:border-slate-300 disabled:bg-slate-100 group-hover:bg-sky-700">
                    <Slider.Range className="absolute h-full rounded-full" />
                  </Slider.Track>
                  <Slider.Thumb
                    onClick={() => {
                      if (!isSteering && !findSteeredPositionSteerGeneratedTokens(nodeSteerIdentifier)) {
                        setSteeredPositionDeltaSteerGeneratedTokens(nodeSteerIdentifier, 0);
                      }
                    }}
                    className="relative flex h-5 w-9 cursor-pointer select-none items-center justify-center overflow-visible rounded-full border border-sky-700 bg-white text-[10px] font-medium leading-none text-sky-700 shadow disabled:border-slate-300 disabled:bg-slate-100 group-hover:bg-sky-100"
                  >
                    {!findSteeredPositionSteerGeneratedTokens(nodeSteerIdentifier) ? (
                      <MousePointerClick className="h-3.5 w-3.5" />
                    ) : findSteeredPositionSteerGeneratedTokens(nodeSteerIdentifier)?.ablate ? (
                      <span className="text-[7px] font-bold">ABLATE</span>
                    ) : (
                      // @ts-ignore
                      `${findSteeredPositionSteerGeneratedTokens(nodeSteerIdentifier)?.delta ? (findSteeredPositionSteerGeneratedTokens(nodeSteerIdentifier)?.delta > 0 ? '+' : '') : ''}${((findSteeredPositionSteerGeneratedTokens(nodeSteerIdentifier)?.delta || 0) / getTopActivationValue(node)).toFixed(1) || '0'}×`
                    )}
                  </Slider.Thumb>
                </Slider.Root>
                <div className="mt-0.5 flex h-5 flex-col items-center justify-center gap-y-[1px] rounded px-1 py-0 text-center text-[7px] font-medium leading-none text-slate-500">
                  <div>NEW</div>
                  <div>TOKENS</div>
                </div>
                <Button
                  variant="ghost"
                  size="sm"
                  className={`flex h-6 w-6 min-w-6 rounded p-0 text-red-700 hover:bg-red-100 ${
                    findSteeredPositionSteerGeneratedTokens(nodeSteerIdentifier)
                      ? 'text-red-700 hover:bg-red-100'
                      : 'text-slate-400'
                  }`}
                  disabled={isSteering || !findSteeredPositionSteerGeneratedTokens(nodeSteerIdentifier)}
                  onClick={() =>
                    setSteeredPositions(
                      steeredPositions.filter(
                        (f) =>
                          !(
                            f.layer === nodeSteerIdentifier.layer &&
                            f.index === nodeSteerIdentifier.index &&
                            f.steer_generated_tokens
                          ),
                      ),
                    )
                  }
                >
                  <Trash2 className="h-3.5 w-3.5" />
                </Button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
