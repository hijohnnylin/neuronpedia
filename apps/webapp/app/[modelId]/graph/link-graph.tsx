/* eslint-disable no-param-reassign */

import { useGraphContext } from '@/components/provider/graph-provider';
import { useGraphStateContext } from '@/components/provider/graph-state-provider';
import { useScreenSize } from '@/lib/hooks/use-screen-size';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import d3 from './d3-jetpack';
import GraphControls, { MIN_TOKENS_TO_ALLOW_HORIZONTAL_SCROLL } from './graph-controls';
import { CLTGraphLink, CLTGraphNode } from './graph-types';
import {
  clientCheckClaudeMode,
  CLTGraphExtended,
  featureTypeToText,
  featureTypeToTextSize,
  filterNodes,
  hideTooltip,
  isHideLayer,
  showTooltip,
} from './utils';

const PERCENT_INCREASE_PER_ADDITIONAL_TOKEN = 0.03;
const X_LABEL_ROTATION_DEGREES = 30;
const X_LABEL_OFFSET = '-12,8';

const MAX_LUMINANCE_LINK_GRAPH = 0.9;
const MINIMUM_LINK_GRAPH_STROKE_WIDTH = 0.5;
// Extended type for context count object
interface ContextCount {
  ctx_idx: number;
  maxCount: number;
  cumsum: number;
  width?: number;
  minS?: number;
}

// Extended type for the graph configuration object
interface GraphConfig {
  width: number;
  height: number;
  totalWidth: number;
  totalHeight: number;
  margin: { left: number; right: number; top: number; bottom: number };
  svg: d3.Selection<SVGGElement, unknown, null, undefined>;
  svgBot: d3.Selection<SVGGElement, unknown, null, undefined>;
  layers: Array<d3.Selection<SVGGElement, unknown, null, undefined> | CanvasRenderingContext2D | null | undefined>;
  x: d3.ScaleLinear<number, number>;
  y: d3.ScaleBand<number>;
  yAxis: d3.Axis<number>;
}

function clerpUUID(d: CLTGraphNode) {
  return `🤖${d.featureIndex}`;
}

function saveHClerpsToLocalStorage(hClerps: Map<string, any>) {
  const key = 'local-clerp';
  const hClerpArray = Array.from(hClerps.entries()).filter((d) => d[1]);
  localStorage.setItem(key, JSON.stringify(hClerpArray));
}

function getHClerpsFromLocalStorage() {
  const key = 'local-clerp';
  // We want to set on load here so that any page load will fix the key.
  if (localStorage.getItem(key) === null) localStorage.setItem(key, '[]');
  const hClerpArray = JSON.parse(localStorage.getItem(key) || '[]').filter((d: any) => d[0] !== clerpUUID(d));
  return new Map(hClerpArray);
}

function hClerpUpdateFn(params: [any, any] | null, data: CLTGraphExtended) {
  const localClerps = getHClerpsFromLocalStorage();
  if (params) {
    const [node, hClerp] = params;
    localClerps.set(clerpUUID(node), hClerp);
    // @ts-ignore
    saveHClerpsToLocalStorage(localClerps);
  }

  if (!data.features) return;

  data.features.forEach((node) => {
    node.localClerp = localClerps.get(clerpUUID(node));
    node.ppClerp = node.localClerp || node.remoteClerp || node.clerp;
  });

  data.nodes?.forEach((node) => {
    // @ts-ignore
    if (!data.features?.idToFeature) return;
    // @ts-ignore
    const feature = data.features.idToFeature[node.featureId];
    if (!feature) return;
    node.localClerp = feature.localClerp;
    node.ppClerp = feature.ppClerp;
  });
}

// Helper function to combine links for supernodes
function combineLinks(
  linksList: CLTGraphLink[],
  isSrc: boolean,
  node: any,
  supernodeId: string | null,
): CLTGraphLink[] {
  return d3
    .nestBy(linksList, (d) => (isSrc ? d.sourceNode?.nodeId || '' : d.targetNode?.nodeId || ''))
    .map((nestedLinks) => ({
      source: (isSrc ? nestedLinks[0].sourceNode?.nodeId : supernodeId) || '',
      target: (isSrc ? supernodeId : nestedLinks[0].targetNode?.nodeId) || '',
      sourceNode: isSrc ? nestedLinks[0].sourceNode : node,
      targetNode: isSrc ? node : nestedLinks[0].targetNode,
      weight: d3.sum(nestedLinks, (d) => d.weight || 0),
      absWeight: Math.abs(d3.sum(nestedLinks, (d) => d.weight || 0)),
    })) as CLTGraphLink[];
}

export default function LinkGraph() {
  const screenSize = useScreenSize();
  const svgRef = useRef<SVGSVGElement>(null);
  const middleRef = useRef<SVGSVGElement>(null);
  const bottomRef = useRef<SVGSVGElement>(null);
  const [allowScroll, setAllowScroll] = useState(false);
  const [isExpanded, setIsExpanded] = useState(false);
  const canvasRefs = useRef<Array<HTMLCanvasElement | null>>([null, null, null, null, null]);
  const { visState, selectedGraph, updateVisStateField, togglePin, isEditingLabel, makeTooltipText } =
    useGraphContext();
  const {
    hoveredIdRef,
    updateHoverState,
    clearHoverState,
    clickedIdRef,
    updateClickedState,
    clearClickedState,
    registerHoverCallback,
    registerClickedCallback,
  } = useGraphStateContext();
  const isEditingLabelRef = useRef(isEditingLabel);
  const cRef = useRef<GraphConfig | null>(null);

  function colorNodes() {
    selectedGraph?.nodes.forEach((d) => {
      d.nodeColor = '#ffffff';
    });
  }

  function colorLinks() {
    const linearPctScale = d3.scaleLinear().domain([-0.4, 0.4]);
    const linearTScale = d3
      .scaleLinear()
      .domain([0, 0.5, 0.5, 1])
      .range([0, 0.5 - 0.001, 0.5 + 0.001, 1]);

    const widthScale = d3.scaleSqrt().domain([0, 1]).range([0.00001, 3]);

    const pctInputColorFn = (d: number) => d3.interpolatePRGn(linearTScale(linearPctScale(d)));

    selectedGraph?.links.forEach((d) => {
      // d.color = d3.interpolatePRGn(_linearTScale(_linearAbsScale(d.weight)))
      if (d.pctInput !== undefined) {
        d.strokeWidth = widthScale(Math.abs(d.pctInput));
        d.pctInputColor = pctInputColorFn(d.pctInput);
        d.color = pctInputColorFn(d.pctInput);
      }
    });
  }

  // Avoid doing heavy coloring work during render; only run when selectedGraph changes
  const lastColoredGraphSlugRef = useRef<string | null>(null);
  useEffect(() => {
    if (!selectedGraph) return;
    const slug = selectedGraph?.metadata?.slug || 'unknown-graph';
    if (lastColoredGraphSlugRef.current === slug) return;
    colorNodes();
    colorLinks();
    lastColoredGraphSlugRef.current = slug;
  }, [selectedGraph]);

  function distance(x1: number, y1: number, x2: number, y2: number) {
    return Math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2);
  }

  function findClosestPoint(mouseX: number, mouseY: number, points: CLTGraphNode[]): [CLTGraphNode, number] | null {
    if (points.length === 0) return null;

    let closestPoint = points[0];
    if (!closestPoint.pos) return null;
    let closestDistance = distance(mouseX, mouseY, closestPoint.pos[0], closestPoint.pos[1]);

    // eslint-disable-next-line
    for (let i = 1; i < points.length; i++) {
      const point = points[i];
      // eslint-disable-next-line
      if (!point.pos) continue;
      const dist = distance(mouseX, mouseY, point.pos[0], point.pos[1]);
      if (dist < closestDistance) {
        closestPoint = point;
        closestDistance = dist;
      }
    }
    return [closestPoint, closestDistance];
  }

  // Effect to keep the isEditingLabelRef updated
  useEffect(() => {
    isEditingLabelRef.current = isEditingLabel;
  }, [isEditingLabel]);

  // Function to update hover links data
  const updateHoverLinksData = useCallback(
    (hoveredId: string | null) => {
      if (!selectedGraph) return;
      const data = selectedGraph as CLTGraphExtended;
      if (!data.nodes) return;

      // Clear existing tmpHoveredLink values if no hoveredId
      if (!hoveredId) {
        data.nodes.forEach((d) => {
          d.tmpHoveredLink = undefined;
          d.tmpHoveredSourceLink = undefined;
          d.tmpHoveredTargetLink = undefined;
        });
        return;
      }
      // Get the hovered node
      const node: CLTGraphNode | undefined = data.nodes.find((n) => n.featureId === hoveredId);

      // If we couldn't find a node, clear all tmpHoveredLink values
      if (!node) {
        data.nodes.forEach((d) => {
          d.tmpHoveredLink = undefined;
          d.tmpHoveredSourceLink = undefined;
          d.tmpHoveredTargetLink = undefined;
        });
        return;
      }

      // Process all connected links
      const connectedLinks = [...(node.sourceLinks || []), ...(node.targetLinks || [])].filter(Boolean);

      // Map links by node ID for easier lookup
      const nodeIdToSourceLink: Record<string, CLTGraphLink> = {};
      const nodeIdToTargetLink: Record<string, CLTGraphLink> = {};
      const featureIdToLink: Record<string, CLTGraphLink> = {};

      connectedLinks.forEach((link) => {
        if (link.sourceNode === node) {
          if (link.targetNode?.nodeId) {
            nodeIdToTargetLink[link.targetNode.nodeId] = link;
          }
          if (link.targetNode?.featureId) {
            featureIdToLink[link.targetNode.featureId] = link;
          }
          link.tmpHoveredCtxOffset = (link.targetNode?.ctx_idx || 0) - (node.ctx_idx || 0);
        }

        if (link.targetNode === node) {
          if (link.sourceNode?.nodeId) {
            nodeIdToSourceLink[link.sourceNode.nodeId] = link;
          }
          if (link.sourceNode?.featureId) {
            featureIdToLink[link.sourceNode.featureId] = link;
          }
          link.tmpHoveredCtxOffset = (link.sourceNode?.ctx_idx || 0) - (node.ctx_idx || 0);
        }

        // Set color for the link
        link.tmpColor = link.pctInputColor;
      });

      // Update all nodes with the appropriate links
      data.nodes.forEach((d) => {
        d.tmpHoveredLink = nodeIdToSourceLink[d.nodeId || ''] || nodeIdToTargetLink[d.nodeId || ''];
        d.tmpHoveredSourceLink = nodeIdToSourceLink[d.nodeId || ''];
        d.tmpHoveredTargetLink = nodeIdToTargetLink[d.nodeId || ''];
      });

      // Update features with links if they exist
      if (data.features) {
        data.features.forEach((d) => {
          d.tmpHoveredLink = featureIdToLink[d.featureId];
        });
      }
    },
    [selectedGraph],
  );

  function filterLinks(featureIds: string[], data: CLTGraphExtended) {
    if (!selectedGraph) return [];
    const filteredLinks: CLTGraphLink[] = [];
    const filteredNodes = filterNodes(data, data.nodes, selectedGraph, visState, clickedIdRef.current);

    featureIds.forEach((nodeId) => {
      filteredNodes
        .filter((n) => n.nodeId === nodeId)
        .forEach((node) => {
          if (visState.linkType === 'input' || visState.linkType === 'either') {
            if (node.sourceLinks) {
              Array.prototype.push.apply(filteredLinks, node.sourceLinks);
            }
          }
          if (visState.linkType === 'output' || visState.linkType === 'either') {
            if (node.targetLinks) {
              Array.prototype.push.apply(filteredLinks, node.targetLinks);
            }
          }
          if (visState.linkType === 'both') {
            if (node.sourceLinks) {
              filteredLinks.push(
                ...node.sourceLinks.filter(
                  (link) => link.sourceNode && visState.pinnedIds.includes(link.sourceNode.nodeId || ''),
                ),
              );
            }
            if (node.targetLinks) {
              filteredLinks.push(
                ...node.targetLinks.filter(
                  (link) => link.targetNode && visState.pinnedIds.includes(link.targetNode.nodeId || ''),
                ),
              );
            }
          }
        });
    });

    return filteredLinks;
  }

  function calculateLuminance(color: string) {
    const rgbMatch = color.match(/rgb\((\d+),\s*(\d+),\s*(\d+)\)/);
    if (!rgbMatch) return 1;
    const r = parseInt(rgbMatch[1], 10);
    const g = parseInt(rgbMatch[2], 10);
    const b = parseInt(rgbMatch[3], 10);
    return (0.299 * r + 0.587 * g + 0.114 * b) / 255;
  }

  // Utility function to draw links
  function drawLinks(
    linkArray: CLTGraphLink[],
    ctx: CanvasRenderingContext2D | null,
    strokeWidthOffset = 0,
    colorOverride?: string,
    maxLuminance?: number,
  ) {
    if (!ctx) return;
    if (cRef.current) {
      ctx.clearRect(
        -cRef.current.margin.left,
        -cRef.current.margin.top,
        cRef.current.totalWidth,
        cRef.current.totalHeight,
      );
    }

    d3.sort(linkArray, (d) => d.strokeWidth || 0).forEach((d) => {
      if (!d.sourceNode?.pos || !d.targetNode?.pos) return;

      ctx.beginPath();
      ctx.moveTo(d.sourceNode.pos[0], d.sourceNode.pos[1]);
      ctx.lineTo(d.targetNode.pos[0], d.targetNode.pos[1]);

      let colorToUse = d.color || 'rgb(0, 0, 0, 1)';
      if (maxLuminance !== undefined && !colorOverride) {
        if (calculateLuminance(colorToUse) > maxLuminance) {
          colorToUse = '#ddd';
        }
      }
      ctx.strokeStyle = colorOverride || colorToUse;
      ctx.lineWidth = Math.max(MINIMUM_LINK_GRAPH_STROKE_WIDTH, (d.strokeWidth || 1) + strokeWidthOffset);
      ctx.stroke();
    });
  }

  // Function to update hover visuals
  const updateHoverVisuals = useCallback(
    (hoveredId: string | null) => {
      if (!selectedGraph || !svgRef.current) return;

      const data = selectedGraph as CLTGraphExtended;
      if (!data.nodes) return;

      // Get canvas contexts for drawing hover links
      const hoveredCtx = canvasRefs.current[2]?.getContext('2d');
      const allLinksCtx = canvasRefs.current[0]?.getContext('2d');

      if (!hoveredCtx || !allLinksCtx) return;

      // Get current graph config from the SVG
      const svgContainer = d3.select(svgRef.current);
      const hoverSel = svgContainer.select('g.svg-top').selectAll('circle.hover-circle');

      // Update hover circle visibility
      hoverSel.style('display', (d: any) => {
        if (d.featureId === hoveredId) {
          return '';
        }
        return 'none';
      });

      // // Clear previous hover links
      // const svgBBox = svgRef.current.getBoundingClientRect();
      // const { width, height } = svgBBox;
      // const margin = {
      //   left: isHideLayer(data.metadata.scan) ? 0 : 30,
      //   right: 20,
      //   top: 30,
      //   bottom: 35,
      // };

      // hoveredCtx.clearRect(-margin.left, -margin.top, width, height);
      // allLinksCtx.clearRect(-margin.left, -margin.top, width, height);

      // // pruning/filtering
      // const filteredNodes = filterNodes(data, data.nodes, selectedGraph, visState, clickedIdRef.current);
      // const filteredNodeIds = new Set(filteredNodes.map((n) => n.nodeId));

      // // Get hovered links and filter them by ensuring both source and target nodes pass the filtering criteria
      // const hoveredLinks = filteredNodes
      //   .filter((d) => d.tmpHoveredLink)
      //   .map((d) => d.tmpHoveredLink)
      //   .filter((link): link is CLTGraphLink => {
      //     if (!link) return false;
      //     // Only include links where both source and target nodes pass the filtering criteria
      //     const sourceNodeId = link.sourceNode?.nodeId;
      //     const targetNodeId = link.targetNode?.nodeId;
      //     // @ts-ignore
      //     return sourceNodeId && targetNodeId && filteredNodeIds.has(sourceNodeId) && filteredNodeIds.has(targetNodeId);
      //   });

      // // Draw background and main hover links
      // drawLinks(hoveredLinks, allLinksCtx, 0.05, '#aaa');
      // drawLinks(hoveredLinks, hoveredCtx, 0.05);
    },
    [selectedGraph, clickedIdRef.current, visState],
  );

  // Function to update pinned visuals
  const updatePinnedVisuals = useCallback(
    (clickedId: string | null) => {
      if (!selectedGraph || !svgRef.current) return;

      const data = selectedGraph as CLTGraphExtended;
      if (!data.nodes) return;

      // Get canvas context for pinned links
      const pinnedCtx = canvasRefs.current[4]?.getContext('2d');
      if (!pinnedCtx) return;

      // Clear previous pinned links
      const svgBBox = svgRef.current.getBoundingClientRect();
      const { width, height } = svgBBox;
      const margin = {
        left: isHideLayer(data.metadata.scan) ? 0 : 30,
        right: 20,
        top: 30,
        bottom: 35,
      };

      pinnedCtx.clearRect(-margin.left, -margin.top, width, height);

      // Only draw pinned links if there's no clicked node
      if (!clickedId && visState.pinnedIds.length > 0) {
        // Filter links based on visState for pinned nodes
        const pinnedLinks = filterLinks(visState.pinnedIds, data);
        drawLinks(pinnedLinks, pinnedCtx, 0, undefined, MAX_LUMINANCE_LINK_GRAPH);
      }
    },
    [selectedGraph, visState.pinnedIds, visState.linkType, visState],
  );

  // Function to update clicked links data
  const updateClickedLinksData = useCallback(
    (clickedId: string | null) => {
      if (!selectedGraph) return;
      const data = selectedGraph as CLTGraphExtended;
      if (!data.nodes) return;

      // Clear existing tmpClickedLink values if no clickedId
      if (!clickedId) {
        data.nodes.forEach((d) => {
          d.tmpClickedLink = undefined;
          d.tmpClickedSourceLink = undefined;
          d.tmpClickedTargetLink = undefined;
        });
        return;
      }

      // Get the clicked node
      let node: CLTGraphNode | undefined = data.nodes.find((n) => n.nodeId === clickedId);

      // Handle supernodes
      if (!node && clickedId?.startsWith('supernode-')) {
        // For a clicked supernode, process the memberNodes if subgraph and supernodes exist
        if (visState.subgraph?.supernodes) {
          const supernodeId = +clickedId.split('-')[1];
          const memberNodeIds = visState.subgraph.supernodes[supernodeId]?.slice(1);

          if (memberNodeIds) {
            // Create a virtual node with member nodes
            const idToNode: Record<string, CLTGraphNode> = {};
            data.nodes.forEach((n) => {
              if (n.nodeId) {
                idToNode[n.nodeId] = n;
              }
            });

            const memberNodes: CLTGraphNode[] = memberNodeIds.map((id: string) => idToNode[id]).filter(Boolean);

            if (memberNodes.length > 0) {
              // @ts-ignore
              node = {
                nodeId: clickedId,
                memberNodes,
                memberSet: new Set(memberNodes.map((d: CLTGraphNode) => d.nodeId || '')),
                sourceLinks: [],
                targetLinks: [],
              };

              if (!node) return;
              const allSourceLinks = memberNodes.flatMap((d: CLTGraphNode) => d.sourceLinks || []).filter(Boolean);
              const allTargetLinks = memberNodes.flatMap((d: CLTGraphNode) => d.targetLinks || []).filter(Boolean);

              node.sourceLinks = combineLinks(allSourceLinks, true, node, clickedId);
              node.targetLinks = combineLinks(allTargetLinks, false, node, clickedId);
            }
          }
        }
      }

      // If we couldn't find a node, clear all tmpClickedLink values
      if (!node) {
        data.nodes.forEach((d) => {
          d.tmpClickedLink = undefined;
          d.tmpClickedSourceLink = undefined;
          d.tmpClickedTargetLink = undefined;
        });
        return;
      }

      // Process all connected links
      const connectedLinks = [...(node.sourceLinks || []), ...(node.targetLinks || [])].filter(Boolean);

      // Map links by node ID for easier lookup
      const nodeIdToSourceLink: Record<string, CLTGraphLink> = {};
      const nodeIdToTargetLink: Record<string, CLTGraphLink> = {};
      const featureIdToLink: Record<string, CLTGraphLink> = {};

      connectedLinks.forEach((link) => {
        if (link.sourceNode === node) {
          if (link.targetNode?.nodeId) {
            nodeIdToTargetLink[link.targetNode.nodeId] = link;
          }
          if (link.targetNode?.featureId) {
            featureIdToLink[link.targetNode.featureId] = link;
          }
          link.tmpClickedCtxOffset = (link.targetNode?.ctx_idx || 0) - (node.ctx_idx || 0);
        }

        if (link.targetNode === node) {
          if (link.sourceNode?.nodeId) {
            nodeIdToSourceLink[link.sourceNode.nodeId] = link;
          }
          if (link.sourceNode?.featureId) {
            featureIdToLink[link.sourceNode.featureId] = link;
          }
          link.tmpClickedCtxOffset = (link.sourceNode?.ctx_idx || 0) - (node.ctx_idx || 0);
        }

        // Set color for the link
        link.tmpColor = link.pctInputColor;
      });

      // Update all nodes with the appropriate links
      data.nodes.forEach((d) => {
        d.tmpClickedLink = nodeIdToSourceLink[d.nodeId || ''] || nodeIdToTargetLink[d.nodeId || ''];
        d.tmpClickedSourceLink = nodeIdToSourceLink[d.nodeId || ''];
        d.tmpClickedTargetLink = nodeIdToTargetLink[d.nodeId || ''];
      });

      // Update features with links if they exist
      if (data.features) {
        data.features.forEach((d) => {
          d.tmpClickedLink = featureIdToLink[d.featureId];
        });
      }
    },
    [selectedGraph, visState.subgraph],
  );

  // Function to update clicked visuals
  const updateClickedVisuals = useCallback(
    (clickedId: string | null) => {
      if (!selectedGraph || !svgRef.current) return;

      const data = selectedGraph as CLTGraphExtended;
      if (!data.nodes) return;

      // Get canvas contexts for drawing clicked links
      const clickedCtx = canvasRefs.current[3]?.getContext('2d');
      const bgCtx = canvasRefs.current[1]?.getContext('2d');

      if (!clickedCtx || !bgCtx) return;

      // Get current graph config from the SVG
      const svgContainer = d3.select(svgRef.current);
      const nodeSel = svgContainer.select('g.svg-top').selectAll('text.node');

      // Update clicked node styling
      nodeSel
        .classed('clicked', (d: any) => Boolean(d.nodeId === clickedId))
        .attr('fill', (d: any) =>
          d.tmpClickedLink ? d.tmpClickedLink.pctInputColor || d.nodeColor || '#000' : d.nodeColor || '#000',
        )
        .attr('stroke', (d: any) => (d.nodeId === clickedId ? '#f0f' : '#000'))
        .attr('stroke-width', (d: any) => (d.nodeId === clickedId ? 1.5 : 0.5));

      // Clear previous clicked links
      const svgBBox = svgRef.current.getBoundingClientRect();
      const { width, height } = svgBBox;
      const margin = {
        left: isHideLayer(data.metadata.scan) ? 0 : 30,
        right: 20,
        top: 30,
        bottom: 35,
      };

      clickedCtx.clearRect(-margin.left, -margin.top, width, height);
      bgCtx.clearRect(-margin.left, -margin.top, width, height);

      if (clickedId) {
        // pruning/filtering
        const filteredNodes = filterNodes(data, data.nodes, selectedGraph, visState, clickedId);
        const filteredNodeIds = new Set(filteredNodes.map((n) => n.nodeId));

        // Get clicked links and filter them by ensuring both source and target nodes pass filtering
        const clickedLinks = filteredNodes
          .filter((d) => d.tmpClickedLink)
          .map((d) => d.tmpClickedLink)
          .filter((link): link is CLTGraphLink => {
            if (!link) return false;
            // Only include links where both source and target nodes pass the filtering criteria
            const sourceNodeId = link.sourceNode?.nodeId;
            const targetNodeId = link.targetNode?.nodeId;

            // @ts-ignore
            return (
              sourceNodeId && targetNodeId && filteredNodeIds.has(sourceNodeId) && filteredNodeIds.has(targetNodeId)
            );
          });

        // Draw background and main clicked links
        drawLinks(clickedLinks, bgCtx, 0.05, '#555');
        drawLinks(clickedLinks, clickedCtx, 0.05);
      }
    },
    [selectedGraph, visState],
  );

  // Combined clicked update function
  const onClickedChange = useCallback(
    (clickedId: string | null) => {
      updateClickedLinksData(clickedId);
      updateClickedVisuals(clickedId);
      updatePinnedVisuals(clickedId);
    },
    [updateClickedLinksData, updateClickedVisuals, updatePinnedVisuals],
  );

  // Combined hover update function
  const onHoverChange = useCallback(
    (hoveredId: string | null) => {
      updateHoverLinksData(hoveredId);
      updateHoverVisuals(hoveredId);
    },
    [updateHoverLinksData, updateHoverVisuals],
  );

  // Register for hover change notifications from other components
  useEffect(() => {
    const unregister = registerHoverCallback((hoveredId) => {
      // Update hover visuals when hover state changes externally
      onHoverChange(hoveredId);
    });

    return unregister; // Cleanup on unmount
  }, [registerHoverCallback, onHoverChange]);

  // Register for clicked change notifications from other components
  useEffect(() => {
    const unregister = registerClickedCallback((clickedId) => {
      // Update clicked links data and visuals when clicked state changes externally
      onClickedChange(clickedId);
    });

    return unregister; // Cleanup on unmount
  }, [registerClickedCallback, onClickedChange]);

  // Update hClerp values - equivalent to hClerpUpdateFn
  const updateHClerps = useCallback(() => {
    if (!selectedGraph) return;
    const data = selectedGraph as CLTGraphExtended;
    hClerpUpdateFn(null, data);
  }, [selectedGraph]);

  // Run hClerp update on initial render
  useEffect(() => {
    updateHClerps();
  }, [updateHClerps]);

  // Initialize the D3 graph visualization
  useEffect(() => {
    if (!svgRef.current || !selectedGraph) return;

    // Clear any existing content
    d3.select(svgRef.current).selectAll('*').remove();

    const data = selectedGraph as CLTGraphExtended;
    let nodes = filterNodes(data, data.nodes, selectedGraph, visState, clickedIdRef.current);
    if (clientCheckClaudeMode()) {
      nodes = nodes.filter((d) => d.feature_type !== 'mlp reconstruction error');
    }

    // Set up the base SVG container
    const svgContainer = d3.select(svgRef.current);
    const svgBBox = svgRef.current.getBoundingClientRect();
    const { width, height } = svgBBox;

    const middleContainer = d3.select(middleRef.current);
    const bottomContainer = d3.select(bottomRef.current);

    // clear all containers/svgs
    svgContainer.selectAll('*').remove();
    middleContainer.selectAll('*').remove();
    bottomContainer.selectAll('*').remove();

    // Define margins based on visState
    const margin = {
      left: isHideLayer(data.metadata.scan) ? 0 : 30,
      right: 20,
      top: 30,
      bottom: 35,
    };

    const svgBot = bottomContainer.append('g').attr('class', 'svg-bot');

    // Create canvas elements for different link layers
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    canvasRefs.current.forEach((ref, i) => {
      const canvasId = `canvas-layer-${i}`;

      // Check if canvas already exists
      let canvas = middleContainer.select(`#${canvasId}`).node();

      if (!canvas) {
        // Create and position canvas element using foreignObject
        const foreignObject = middleContainer
          .append('foreignObject')
          .attr('width', width)
          .attr('height', height)
          .attr('x', 0)
          .attr('y', 0);

        canvas = foreignObject
          .append('xhtml:canvas')
          .attr('id', canvasId)
          .attr('width', width)
          .attr('height', height)
          .style('position', 'absolute')
          .style('top', '0px')
          .style('left', '0px')
          .node();

        canvasRefs.current[i] = canvas as HTMLCanvasElement;
      }
    });

    const svg = svgContainer.append('g').attr('class', 'svg-top');

    // Setup canvas contexts
    const allCtx = {
      allLinks: canvasRefs.current[0]?.getContext('2d'),
      bgLinks: canvasRefs.current[1]?.getContext('2d'),
      hoveredLinks: canvasRefs.current[2]?.getContext('2d'),
      clickedLinks: canvasRefs.current[3]?.getContext('2d'),
      pinnedLinks: canvasRefs.current[4]?.getContext('2d'),
    };

    // Transform all contexts to account for margins
    Object.values(allCtx).forEach((ctx) => {
      if (ctx) {
        ctx.translate(margin.left, margin.top);
      }
    });

    // Also transform SVG layers
    svgBot.attr('transform', `translate(${margin.left},${margin.top})`);
    svg.attr('transform', `translate(${margin.left},${margin.top})`);

    // Calculate graph dimensions
    cRef.current = {
      width: width - margin.left - margin.right,
      height: height - margin.top - margin.bottom,
      totalWidth: width,
      totalHeight: height,
      margin,
      svg,
      svgBot,
      layers: [svgBot, ...Object.values(allCtx), svg],
      // Initialize with dummy scales that will be replaced
      x: d3.scaleLinear().domain([0, 1]).range([0, 1]),
      y: d3.scaleBand<number>().domain([0]).range([0, 1]),
      yAxis: d3.axisLeft(d3.scaleBand<number>().domain([0]).range([0, 1])),
    };
    const c = cRef.current;

    // Count max number of nodes at each context to create a polylinear x scale
    const nonUndefinedNodes = nodes.filter((d) => d.ctx_idx !== undefined);
    const earliestCtxWithNodes = d3.min(nonUndefinedNodes, (d) => d.ctx_idx as number) || 0;

    let cumsum = 0;
    const ctxCounts: ContextCount[] = [];

    // Generate context counts
    const maxCtxIdx = d3.max(nonUndefinedNodes, (d) => d.ctx_idx as number) || 0;
    // eslint-disable-next-line
    for (let ctxIdx = 0; ctxIdx <= maxCtxIdx; ctxIdx++) {
      let maxCount = 1;
      if (ctxIdx >= earliestCtxWithNodes) {
        const group = nodes.filter((d) => d.ctx_idx === ctxIdx);
        if (group.length > 0) {
          const groupedByStream = d3.nestBy(group, (d) => d.streamIdx?.toString() || '0');
          const lengths = groupedByStream.map((g) => g.length);
          maxCount = Math.max(1, Math.max(...lengths));
        }
        cumsum += maxCount;
      }
      ctxCounts.push({ ctx_idx: ctxIdx, maxCount, cumsum });
    }

    // Create scales
    const xDomain = [-1].concat(ctxCounts.map((d) => d.ctx_idx));
    const xRange = [0].concat(ctxCounts.map((d) => (d.cumsum * c.width) / cumsum));
    c.x = d3
      .scaleLinear()
      .domain(xDomain.map((d) => d + 1))
      .range(xRange);

    // Get byStream from data or create a default with 19 items
    const byStreamLength = data.byStream?.length || 19;
    const numLayers = data.metadata.neuronpedia_internal_model?.layers || 0;
    const yNumTicks = isHideLayer(data.metadata.scan) ? byStreamLength : numLayers + 2;

    // Create an array of numbers for the y-axis
    c.y = d3.scaleBand(d3.range(yNumTicks), [c.height, 0]);

    // Create y-axis
    c.yAxis = d3
      .axisLeft(c.y)
      .tickValues(d3.range(yNumTicks))
      .tickFormat((i) =>
        // TODO: remove gpt2-small special case
        i === yNumTicks - 1 ? 'Lgt' : i === 0 ? 'Emb' : `L${data.metadata.scan === 'gpt2-small' ? i : i - 1}`,
      );

    // Background elements
    c.svgBot.append('rect').attr('width', c.width).attr('height', c.height).attr('fill', 'rgba(226, 232, 240, 0.6)');

    // Add background rectangles for even and odd rows
    c.svgBot
      .append('g')
      .selectAll('rect')
      .data([0, yNumTicks - 1])
      .enter()
      .append('rect')
      .attr('width', c.width)
      .attr('height', c.y.bandwidth())
      .attr('y', (i) => {
        const yPos = c.y(i);
        return yPos !== undefined ? yPos : 0;
      })
      .attr('fill', 'rgba(226, 232, 240, 0.8)');

    // Add horizontal grid lines
    c.svgBot
      .append('g')
      .selectAll('path')
      .data(d3.range(-1, yNumTicks - 1))
      .enter()
      .append('path')
      .attr('transform', (d) => {
        const yPos = c.y(d + 1);
        return `translate(0,${yPos !== undefined ? yPos : 0})`;
      })
      .attr('d', `M0,0H${c.width}`)
      .attr('stroke', 'white')
      .attr('stroke-width', 0.5);

    // Draw axis
    c.svgBot.append('g').attr('class', 'y axis').call(c.yAxis);

    // Set y-axis tick text color
    c.svgBot.selectAll('.y text').attr('fill', '#64748b').attr('font-size', '9px');

    if (isHideLayer(data.metadata.scan)) {
      c.svgBot.select('.y').remove();
    }
    c.svgBot.selectAll('.y line').remove();
    c.svgBot.selectAll('.y .domain').remove();

    // Spread nodes across each context
    ctxCounts.forEach((d, i) => {
      if (i < ctxCounts.length) {
        const colWidth = c.x(d.ctx_idx + 1) - c.x(ctxCounts[i].ctx_idx);
        // eslint-disable-next-line
        d.width = colWidth;
      }
    });

    // If default to 8px padding right, if pad right to center singletons
    const padR = Math.min(8, d3.min(ctxCounts.slice(1), (d) => (d.width || 0) / 2) || 8);

    // Find the tightest spacing between nodes and use for all ctx (but don't go below 20)
    ctxCounts.forEach((d) => {
      const availableWidth = (d.width || 0) - padR;
      // eslint-disable-next-line
      d.minS = availableWidth / d.maxCount;
    });

    const overallS = Math.max(20, d3.min(ctxCounts, (d) => d.minS || 20) || 20);

    // Apply to nodes - mutating the nodes array to add position data
    const nestByResult = d3.nestBy(nodes, (d) => [d.ctx_idx, d.streamIdx || 0].join('-'));
    nestByResult.forEach((ctxLayer) => {
      const ctxIdx = ctxLayer[0].ctx_idx;
      if (ctxIdx === undefined) return;

      const ctxWidth = c.x(ctxIdx + 1) - c.x(ctxIdx) - padR;
      const s = Math.min(overallS, ctxWidth / ctxLayer.length);

      // Sorting by logitPct stacks all the links
      const sortedLayer = d3.sort(ctxLayer, (d) => -(d.logitPct || 0));
      sortedLayer.forEach((d, i) => {
        if (d.feature_type === 'embedding') {
          d.xOffset = c.x(d.ctx_idx + 1) - c.x(d.ctx_idx) - (padR + 3.5);
        } else {
          d.xOffset = ctxWidth - (padR / 2 + i * s);
        }
        // eslint-disable-next-line
        d.yOffset = 0;
      });
    });

    // Calculate positions for all nodes
    nodes.forEach((d) => {
      if (d.ctx_idx === undefined || d.streamIdx === undefined) return;

      const effectiveStreamIdx =
        d.feature_type === 'embedding' || isHideLayer(data.metadata.scan) || data.metadata.scan === 'gpt2-small'
          ? d.streamIdx
          : d.streamIdx + 1;

      const xPos = c.x(d.ctx_idx) + (d.xOffset || 0);

      const yBand = c.y(effectiveStreamIdx);
      if (yBand === undefined) return;

      // eslint-disable-next-line
      d.pos = [xPos, yBand + c.y.bandwidth() / 2 + (d.yOffset || 0)];
    });

    // Add gradient for pinned/clicked nodes
    svg.append('defs').html(`
      <linearGradient id='pinned-clicked-gradient' x1='0' x2='2' gradientUnits='userSpaceOnUse' spreadMethod='repeat'>
        <stop offset='0'    stop-color='#f0f' />
        <stop offset='70%'  stop-color='#f0f' />
        <stop offset='71%'  stop-color='#000' />
        <stop offset='100%' stop-color='#000' />
      </linearGradient>
    `);

    // Set up nodes
    const isMobile = window.innerWidth < 640;
    const nodeSel = c.svg
      .selectAll('text.node')
      .data(nodes)
      .enter()
      .append('text')
      .attr('class', 'node')
      .attr('transform', (d) => {
        const pos = d.pos || [0, 0];
        return `translate(${pos[0]},${pos[1]})`;
      })
      .text((d) => featureTypeToText(d.feature_type))
      .attr('opacity', (d) => (d.feature_type === 'mlp reconstruction error' ? 0.35 : 1))
      .attr('font-family', 'Arial')
      .attr('font-size', (d) => featureTypeToTextSize(isMobile, d.feature_type)) // weird safari mobile bug where it renders the diamond too large
      .attr('fill', (d) => d.nodeColor || '#000')
      .attr('stroke', '#000')
      .attr('stroke-width', 2)
      .attr('text-anchor', 'middle')
      .attr('dominant-baseline', 'central');

    // Add hover circles for visual feedback
    c.svg
      .selectAll('circle.hover-circle')
      .data(nodes)
      .enter()
      .append('circle')
      .attr('class', 'hover-circle')
      .attr('transform', (d) => {
        const pos = d.pos || [0, 0];
        return `translate(${pos[0]},${pos[1]})`;
      })
      .attr('r', 6)
      .attr('cy', 0.5)
      .attr('stroke', '#f0f')
      .attr('stroke-width', 2)
      .attr('stroke-dasharray', '2 2')
      .attr('fill', 'none')
      .style('display', 'none')
      .style('pointer-events', 'none');

    // // Draw links for pinned nodes
    if (allCtx.pinnedLinks) {
      drawLinks(
        clickedIdRef.current ? [] : filterLinks(visState.pinnedIds, data),
        allCtx.pinnedLinks,
        0,
        undefined,
        MAX_LUMINANCE_LINK_GRAPH,
      );
    }

    // Highlight pinned nodes
    nodeSel.classed('pinned', (d) => Boolean(d.nodeId && visState.pinnedIds.includes(d.nodeId)));

    // Highlight clicked node
    nodeSel.classed('clicked', (d) => Boolean(d.nodeId === clickedIdRef.current));

    // Style nodes based on their tmp clicked link
    nodeSel
      .attr('fill', (d) =>
        d.tmpClickedLink ? d.tmpClickedLink.pctInputColor || d.nodeColor || '#000' : d.nodeColor || '#000',
      )
      .attr('stroke', (d) => (d.nodeId === clickedIdRef.current ? '#f0f' : '#000'))
      .attr('stroke-width', (d) => (d.nodeId === clickedIdRef.current ? 1.5 : 0.5));

    // Add mouse event handlers for graph interaction
    const maxHoverDistance = 30;

    // Variable to track the current hovered node id to avoid unnecessary state updates
    let currentHoveredFeatureId: string | null = hoveredIdRef.current;

    svgContainer
      .on('mousemove', (event) => {
        if (event.shiftKey) return;

        const [mouseX, mouseY] = d3.pointer(event);
        const result = findClosestPoint(mouseX - margin.left, mouseY - margin.top, nodes);

        if (!result) {
          // No nodes found - clear hover state if we had one
          hideTooltip();
          if (currentHoveredFeatureId) {
            clearHoverState(onHoverChange);
            currentHoveredFeatureId = null;
          }
          return;
        }

        const [closestNode, closestDistance] = result;

        if (closestDistance > maxHoverDistance) {
          // Un-hover behavior - hide tooltips, clear highlight
          hideTooltip();
          // Only update state if needed
          if (currentHoveredFeatureId) {
            clearHoverState(onHoverChange);
            currentHoveredFeatureId = null;
          }
        } else if (currentHoveredFeatureId !== closestNode.featureId && !isEditingLabelRef.current) {
          // Only update when the hovered node actually changes
          currentHoveredFeatureId = closestNode.featureId || null;

          // Hover behavior
          // console.log('Setting hover state:', currentHoveredFeatureId);
          updateHoverState(closestNode, onHoverChange);
          showTooltip(event, closestNode, makeTooltipText(closestNode));
          // NOTE: Hover circle display is now handled by the callback
        }
      })
      .on('mouseleave', (event) => {
        if (event.shiftKey) return;

        // Clear hover state
        hideTooltip();
        // Clear hover state to remove hover links
        if (currentHoveredFeatureId) {
          clearHoverState(onHoverChange);
          currentHoveredFeatureId = null;
        }
      })
      .on('click', (event) => {
        const [mouseX, mouseY] = d3.pointer(event);
        const result = findClosestPoint(mouseX - margin.left, mouseY - margin.top, nodes);

        if (!result) return;

        const [closestNode, closestDistance] = result;

        if (closestDistance > maxHoverDistance) {
          // Clear clicked state if clicking away
          clearClickedState(onClickedChange);
        } else {
          // Handle clicking on a node
          // eslint-disable-next-line
          if (event.metaKey || event.ctrlKey) {
            // Toggle pinned state with meta/ctrl key
            const newPinnedIds = togglePin(closestNode.nodeId || '');

            // Update pinned visualization
            nodeSel.classed('pinned', (d) => Boolean(d.nodeId && newPinnedIds.includes(d.nodeId)));

            // Redraw pinned links
            if (allCtx.pinnedLinks) {
              drawLinks(filterLinks(newPinnedIds, data), allCtx.pinnedLinks, 0, undefined, MAX_LUMINANCE_LINK_GRAPH);
            }
          } else {
            // Set as clicked node
            const newClickedId = clickedIdRef.current === closestNode.nodeId ? null : closestNode.nodeId;

            if (newClickedId) {
              updateClickedState(closestNode, onClickedChange);
            } else {
              clearClickedState(onClickedChange);
            }
          }
        }
      });

    // Add x axis text/lines for prompt tokens
    const promptTicks = data.metadata.prompt_tokens.slice(earliestCtxWithNodes).map((token, i) => {
      const ctxIdx = i + earliestCtxWithNodes;
      const mNodes = nodes.filter((d) => d.ctx_idx === ctxIdx);
      const hasEmbed = mNodes.some((d) => d.feature_type === 'embedding');
      return { token, ctx_idx: ctxIdx, mNodes, hasEmbed };
    });

    const xTickSel = c.svgBot
      .selectAll('g.prompt-token')
      .data(promptTicks)
      .enter()
      .append('g')
      .attr('class', 'prompt-token')
      .attr('transform', (d) => `translate(${c.x(d.ctx_idx + 1)},${c.height})`);

    xTickSel.append('path').attr('d', `M0,0v${-c.height}`).attr('stroke', '#fff').attr('stroke-width', 1);

    xTickSel
      .filter((d) => d.hasEmbed)
      .append('path')
      .attr('stroke', '#B0AEA6')
      .attr('d', `M-${padR + 3.5},${-c.y.bandwidth() / 2 + 6}V${8}`);

    xTickSel
      .filter((d) => d.hasEmbed)
      .append('g')
      .attr('transform', `translate(${X_LABEL_OFFSET})`)
      .append('text')
      .text((d) => d.token)
      .attr('x', -3)
      .attr('y', 10)
      .attr('text-anchor', 'middle')
      .attr('transform', `rotate(-${X_LABEL_ROTATION_DEGREES})`)
      .attr('dominant-baseline', 'middle')
      .attr('font-size', 10)
      .attr('fill', '#334155');

    // Add logit ticks
    const logitTickSel = c.svgBot
      .append('g')
      .attr('class', 'axis')
      .selectAll('g')
      .data(nodes.filter((d) => d.feature_type === 'logit'))
      .enter()
      .append('g')
      .attr('transform', (d) => {
        const pos = d.pos || [0, 0];
        return `translate(${pos[0]},${pos[1]})`;
      });

    logitTickSel
      .append('path')
      .attr('stroke', '#B0AEA6')
      .attr('d', `M0,${-6}V${-c.y.bandwidth() / 2 - 6}`);

    logitTickSel
      .append('g')
      .attr('transform', `translate(-5,${-c.y.bandwidth() / 2 - 8})`)
      .append('text')
      .text((d) => d.logitToken || '')
      .attr('x', 6)
      .attr('y', -3)
      .attr('transform', 'rotate(-35)')
      .attr('text-anchor', 'middle')
      .attr('dominant-baseline', 'middle')
      .attr('font-size', 10)
      .attr('fill', '#334155');

    // Initial display of hovered nodes
    // NOTE: Hover circle display is now handled in a separate useEffect for performance

    // Restore clicked state after graph reinitialization
    if (clickedIdRef.current) {
      // Find the clicked node
      const clickedNode = nodes.find((n) => n.nodeId === clickedIdRef.current || n.featureId === clickedIdRef.current);
      if (clickedNode) {
        // Restore clicked links data and visuals
        onClickedChange(clickedIdRef.current);
      }
    }
  }, [
    screenSize,
    selectedGraph,
    visState.pruningThreshold,
    visState.densityThreshold,
    visState.linkType,
    visState.pinnedIds,
    visState.subgraph,
    allowScroll,
    isExpanded,
  ]);

  const shouldDoHorizontalScroll = useMemo(() => {
    if (!selectedGraph) return false;
    const numTokens = selectedGraph?.metadata.prompt_tokens.length || 0;
    return allowScroll && numTokens > MIN_TOKENS_TO_ALLOW_HORIZONTAL_SCROLL;
  }, [selectedGraph, allowScroll]);

  const svgWidth = useMemo(() => {
    if (!selectedGraph) return 100;
    const numTokens = selectedGraph?.metadata.prompt_tokens.length || 0;
    if (!shouldDoHorizontalScroll) return 100;
    return 100 + 100 * PERCENT_INCREASE_PER_ADDITIONAL_TOKEN * (numTokens - MIN_TOKENS_TO_ALLOW_HORIZONTAL_SCROLL);
  }, [selectedGraph, allowScroll]);

  return (
    <>
      {/* Backdrop for dismissing overlay on background click */}
      {isExpanded && (
        <div
          className="fixed inset-0 z-[9998] bg-black/50 backdrop-blur-sm"
          onClick={() => setIsExpanded(false)}
          onKeyDown={(e) => e.key === 'Escape' && setIsExpanded(false)}
          role="button"
          tabIndex={0}
          aria-label="Close expanded view"
        />
      )}
      <div
        className={`link-graph select-none overscroll-none ${
          isExpanded
            ? 'bg-white'
            : `relative -mr-4 flex-1 sm:mr-0 ${shouldDoHorizontalScroll ? 'forceShowScrollBarHorizontal mt-1 max-w-full overflow-y-hidden overflow-x-scroll' : 'mt-1 w-full'}`
        }`}
        style={{
          ...(isExpanded && {
            position: 'fixed',
            top: '50%',
            left: '50%',
            transform: 'translate(-50%, -50%)',
            width: 'calc(100vw - 80px)',
            height: 'calc(100vh - 120px)',
            maxWidth: '1800px',
            maxHeight: '1000px',
            zIndex: 9999,
            borderTop: '20px solid white',
            borderBottom: '8px solid white',
            borderLeft: '20px solid white',
            borderRight: '6px solid white',
            borderRadius: '12px',
            boxShadow: '0 25px 50px -12px rgba(0, 0, 0, 0.25)',
          }),
          ...(shouldDoHorizontalScroll &&
            !isExpanded && {
              WebkitMask: 'linear-gradient(to right, black calc(100% - 25px), transparent 100%)',
              mask: 'linear-gradient(to right, black calc(100% - 25px), transparent 100%)',
            }),
        }}
      >
      {/* <div className="mb-3 mt-2 flex w-full flex-row items-center justify-start gap-x-2">
        <div className="text-sm font-bold text-slate-600">Link Graph</div>
        <CustomTooltip wide trigger={<QuestionMarkCircledIcon className="h-4 w-4 text-slate-500" />}>
          <div className="flex flex-col">
            TODO: https://transformer-circuits.pub/2025/attribution-graphs/methods.html
          </div>
        </CustomTooltip>
      </div> */}

      <GraphControls
        selectedGraph={selectedGraph}
        visState={visState}
        updateVisStateField={updateVisStateField}
        allowScroll={allowScroll}
        setAllowScroll={setAllowScroll}
        shouldDoHorizontalScroll={shouldDoHorizontalScroll}
        isExpanded={isExpanded}
        setIsExpanded={setIsExpanded}
      />
      <div className="tooltip tooltip-hidden" />
      <svg
        className="absolute top-0 z-0 h-full"
        style={{
          ...(shouldDoHorizontalScroll
            ? {
                overflowX: 'scroll',
                width: `${svgWidth}%`,
              }
            : { width: '100%', height: 'calc(100% - 6px)' }),
        }}
        ref={bottomRef}
      />
      <svg
        className="absolute top-0 z-0 h-full"
        style={{
          ...(shouldDoHorizontalScroll
            ? {
                overflowX: 'scroll',
                width: `${svgWidth}%`,
              }
            : { width: '100%', height: 'calc(100% - 6px)' }),
        }}
        ref={middleRef}
      />
      <svg
        className="absolute top-0 z-0 h-full"
        style={{
          ...(shouldDoHorizontalScroll
            ? {
                overflowX: 'scroll',
                width: `${svgWidth}%`,
              }
            : { width: '100%', height: 'calc(100% - 6px)' }),
        }}
        ref={svgRef}
      />
    </div>
    </>
  );
}
