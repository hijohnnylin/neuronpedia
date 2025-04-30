/* eslint-disable no-param-reassign */

import { useCircuitCLT } from '@/components/provider/circuit-clt-provider';
import { useScreenSize } from '@/lib/hooks/use-screen-size';
import { useCallback, useEffect, useRef } from 'react';
import {
  CLTGraph,
  CLTGraphLink,
  CLTGraphNode,
  cltModelToLayers,
  featureTypeToText,
  hideTooltip,
  isHideLayer,
  showTooltip,
} from './clt-utils';
import d3 from './d3-jetpack';

const HEIGHT = 320;

// Extended type for custom CLTGraph properties
interface CLTGraphExtended extends CLTGraph {
  byStream?: Array<any>;
  features?: Array<any>;
}

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

export default function CLTLinkGraph() {
  const screenSize = useScreenSize();
  const svgRef = useRef<SVGSVGElement>(null);
  const middleRef = useRef<SVGSVGElement>(null);
  const bottomRef = useRef<SVGSVGElement>(null);
  const canvasRefs = useRef<Array<HTMLCanvasElement | null>>([null, null, null, null]);
  const { visState, selectedGraph, updateVisStateField } = useCircuitCLT();

  function colorNodes() {
    selectedGraph?.nodes.forEach((d) => {
      d.nodeColor = '#ffffff';
    });
  }
  colorNodes();

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
  colorLinks();

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

  // Update hoverState when hoveredId changes - equivalent to renderAll.hoveredId.fns.push()
  useEffect(() => {
    if (!selectedGraph || !visState.hoveredId) return;

    // Use hovered node if possible, otherwise use last occurrence of feature
    const targetCtxIdx = visState.hoveredCtxIdx ?? 999;
    const hoveredNodes = selectedGraph.nodes.filter((n) => n.featureId === visState.hoveredNodeId);
    if (hoveredNodes.length > 0) {
      // if hovered node is the same as the hoveredId, do nothing
      if (hoveredNodes[0].nodeId === visState.hoveredNodeId) return;

      // Sort by closest ctx_idx to the target
      const node = d3.sort(hoveredNodes, (d) => Math.abs((d.ctx_idx || 0) - (targetCtxIdx || 0)))[0];
      updateVisStateField('hoveredNodeId', node?.nodeId || null);
    }
  }, [visState.hoveredId, visState.hoveredCtxIdx, selectedGraph, updateVisStateField]);

  // Update clickedState when clickedId changes - equivalent to renderAll.clickedId.fns.push()
  useEffect(() => {
    if (!selectedGraph) return;

    const data = selectedGraph as CLTGraphExtended;
    if (!data.nodes) return;

    // Clear existing tmpClickedLink values if no clickedId
    if (!visState.clickedId) {
      data.nodes.forEach((d) => {
        d.tmpClickedLink = undefined;
        d.tmpClickedSourceLink = undefined;
        d.tmpClickedTargetLink = undefined;
      });
      return;
    }

    // Get the clicked node
    let node: CLTGraphNode | undefined = data.nodes.find((n) => n.nodeId === visState.clickedId);

    // Handle supernodes
    if (!node && visState.clickedId?.startsWith('supernode-')) {
      // For a clicked supernode, process the memberNodes if subgraph and supernodes exist
      if (visState.subgraph?.supernodes) {
        const supernodeId = +visState.clickedId.split('-')[1];
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
              nodeId: visState.clickedId,
              memberNodes,
              memberSet: new Set(memberNodes.map((d: CLTGraphNode) => d.nodeId || '')),
              sourceLinks: [],
              targetLinks: [],
            };

            if (!node) return;
            const allSourceLinks = memberNodes.flatMap((d: CLTGraphNode) => d.sourceLinks || []).filter(Boolean);

            const allTargetLinks = memberNodes.flatMap((d: CLTGraphNode) => d.targetLinks || []).filter(Boolean);

            node.sourceLinks = combineLinks(allSourceLinks, true, node, visState.clickedId);
            node.targetLinks = combineLinks(allTargetLinks, false, node, visState.clickedId);
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
  }, [visState.clickedId, visState.subgraph, selectedGraph, updateVisStateField]);

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
    const { nodes, links } = data;

    // Set up the base SVG container
    const svgContainer = d3.select(svgRef.current);
    const svgBBox = svgRef.current.getBoundingClientRect();
    const { width } = svgBBox;
    const height = HEIGHT;

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
      top: 0,
      bottom: 0,
    };

    const svgBot = bottomContainer.append('g').attr('class', 'svg-bot');

    // Create canvas elements for different link layers
    canvasRefs.current.forEach((_, i) => {
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
      pinnedLinks: canvasRefs.current[1]?.getContext('2d'),
      bgLinks: canvasRefs.current[2]?.getContext('2d'),
      clickedLinks: canvasRefs.current[3]?.getContext('2d'),
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
    const c: GraphConfig = {
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
    const numLayers = cltModelToLayers[data.metadata.scan as keyof typeof cltModelToLayers];
    const yNumTicks = isHideLayer(data.metadata.scan) ? byStreamLength : numLayers + 1;

    // Create an array of numbers for the y-axis
    c.y = d3.scaleBand(d3.range(yNumTicks), [c.height, 0]);

    // Create y-axis
    c.yAxis = d3
      .axisLeft(c.y)
      .tickValues(d3.range(yNumTicks))
      .tickFormat((i) =>
        // if (i % 2 !== 0) return '';
        i === numLayers ? 'Lgt' : i === 0 ? 'Emb' : `L${i}`,
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

    if (isHideLayer(data.metadata.scan)) c.svgBot.select('.y').remove();
    c.svgBot.selectAll('.y line').remove();

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
        // These mutations are kept from the original code but marked explicitly
        const xOffset = d.feature_type === 'logit' ? ctxWidth - (padR / 2 + i * s) : ctxWidth - (padR / 2 + i * s);
        // eslint-disable-next-line
        d.xOffset = xOffset;
        // eslint-disable-next-line
        d.yOffset = 0;
      });
    });

    // Calculate positions for all nodes
    nodes.forEach((d) => {
      if (d.ctx_idx === undefined || d.streamIdx === undefined) return;

      const xPos = c.x(d.ctx_idx) + (d.xOffset || 0);
      const yBand = c.y(d.streamIdx);
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
      .attr('font-size', 14)
      .attr('fill', (d) => d.nodeColor || '#000')
      .attr('stroke', '#000')
      .attr('stroke-width', 2)
      .attr('text-anchor', 'middle')
      .attr('dominant-baseline', 'central');

    // Add hover circles for visual feedback
    const hoverSel = c.svg
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

    // Utility function to draw links
    function drawLinks(
      linkArray: CLTGraphLink[],
      ctx: CanvasRenderingContext2D | null,
      strokeWidthOffset = 0,
      colorOverride?: string,
    ) {
      if (!ctx) return;

      ctx.clearRect(-c.margin.left, -c.margin.top, c.totalWidth, c.totalHeight);
      d3.sort(linkArray, (d) => d.strokeWidth || 0).forEach((d) => {
        if (!d.sourceNode?.pos || !d.targetNode?.pos) return;

        ctx.beginPath();
        ctx.moveTo(d.sourceNode.pos[0], d.sourceNode.pos[1]);
        ctx.lineTo(d.targetNode.pos[0], d.targetNode.pos[1]);
        ctx.strokeStyle = colorOverride || d.color || '#000';
        ctx.lineWidth = (d.strokeWidth || 1) + strokeWidthOffset;
        ctx.stroke();
      });
    }

    // Filter links based on visState
    function filterLinks(featureIds: string[]) {
      const filteredLinks: CLTGraphLink[] = [];

      featureIds.forEach((nodeId) => {
        nodes
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

    // Draw all links with low opacity
    if (allCtx.allLinks) {
      drawLinks(links, allCtx.allLinks, 0, 'rgba(0,0,0,.05)');
    }

    // Draw links for pinned nodes
    if (allCtx.pinnedLinks) {
      drawLinks(filterLinks(visState.pinnedIds), allCtx.pinnedLinks);
    }

    // Draw links for clicked node
    // ensure that clickedId exists in the nodes array
    if (visState.clickedId && nodes.some((d) => d.nodeId === visState.clickedId)) {
      drawLinks([], allCtx.pinnedLinks || null);
    } else {
      // Filter links connected to clicked node
      const clickedLinks = nodes
        .filter((d) => d.tmpClickedLink)
        .map((d) => d.tmpClickedLink)
        .filter(Boolean) as CLTGraphLink[];
      drawLinks(clickedLinks, allCtx.clickedLinks || null, 1);
    }

    // Highlight pinned nodes
    nodeSel.classed('pinned', (d) => Boolean(d.nodeId && visState.pinnedIds.includes(d.nodeId)));

    // Highlight clicked node
    nodeSel.classed('clicked', (d) => Boolean(d.nodeId === visState.clickedId));

    // Style nodes based on their tmp clicked link
    nodeSel
      .attr('fill', (d) =>
        d.tmpClickedLink ? d.tmpClickedLink.pctInputColor || d.nodeColor || '#000' : d.nodeColor || '#000',
      )
      .attr('stroke', (d) => (d.nodeId === visState.clickedId ? '#f0f' : '#000'))
      .attr('stroke-width', (d) => (d.nodeId === visState.clickedId ? 1.5 : 0.5));

    // Add mouse event handlers for graph interaction
    const maxHoverDistance = 30;

    // Variable to track the current hovered node id to avoid unnecessary state updates
    let currentHoveredFeatureId: string | null = visState.hoveredId;

    svgContainer
      .on('mousemove', (event) => {
        if (event.shiftKey) return;

        const [mouseX, mouseY] = d3.pointer(event);
        const result = findClosestPoint(mouseX - margin.left, mouseY - margin.top, nodes);

        if (!result) return;

        const [closestNode, closestDistance] = result;

        if (closestDistance > maxHoverDistance) {
          // Un-hover behavior - hide tooltips, clear highlight
          hoverSel.style('display', 'none');
          hideTooltip();

          // Only update state if it needs to change
          if (currentHoveredFeatureId) {
            // console.log('Clearing hover state');
            // updateVisStateField('clickedId', null);
            // updateVisStateField('clickedCtxIdx', null);
            updateVisStateField('hoveredId', null);
            updateVisStateField('hoveredCtxIdx', null);
            currentHoveredFeatureId = null;
          }
        } else if (currentHoveredFeatureId !== closestNode.featureId) {
          // Only update when the hovered node actually changes
          currentHoveredFeatureId = closestNode.featureId || null;

          // Hover behavior
          // console.log('Setting hover state:', currentHoveredFeatureId);
          updateVisStateField('hoveredId', closestNode.featureId || null);
          updateVisStateField('hoveredCtxIdx', closestNode.ctx_idx);
          // updateVisStateField('clickedId', closestNode.nodeId || null);
          // updateVisStateField('clickedCtxIdx', closestNode.ctx_idx);
          showTooltip(event, closestNode);
          // Visual feedback for hover - direct DOM update without requiring a state update
          hoverSel.style('display', (d) => (d.featureId === currentHoveredFeatureId ? '' : 'none'));
        }
      })
      .on('mouseleave', (event) => {
        if (event.shiftKey) return;

        // Clear hover state
        hoverSel.style('display', 'none');

        // Only update state if needed
        if (currentHoveredFeatureId) {
          updateVisStateField('hoveredId', null);
          updateVisStateField('hoveredCtxIdx', null);
          // updateVisStateField('clickedId', null);
          // updateVisStateField('clickedCtxIdx', null);
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
          updateVisStateField('clickedId', null);
          updateVisStateField('clickedCtxIdx', null);

          // Update visual state
          nodeSel.classed('clicked', false);
        } else {
          // Handle clicking on a node
          // eslint-disable-next-line
          if (event.metaKey || event.ctrlKey) {
            // Toggle pinned state with meta/ctrl key
            const pinnedIndex = visState.pinnedIds.indexOf(closestNode.nodeId || '');
            const newPinnedIds = [...visState.pinnedIds];

            if (pinnedIndex === -1 && closestNode.nodeId) {
              newPinnedIds.push(closestNode.nodeId);
            } else if (pinnedIndex !== -1) {
              newPinnedIds.splice(pinnedIndex, 1);
            }

            updateVisStateField('pinnedIds', newPinnedIds);

            // Update pinned visualization
            nodeSel.classed('pinned', (d) => Boolean(d.nodeId && newPinnedIds.includes(d.nodeId)));

            // Redraw pinned links
            if (allCtx.pinnedLinks) {
              drawLinks(filterLinks(newPinnedIds), allCtx.pinnedLinks);
            }
          } else {
            // Set as clicked node
            const newClickedId = visState.clickedId === closestNode.nodeId ? null : closestNode.nodeId;
            updateVisStateField('clickedId', newClickedId || null);
            updateVisStateField('clickedCtxIdx', newClickedId ? closestNode.ctx_idx : null);

            // Update clicked visualization
            nodeSel.classed('clicked', (d) => Boolean(d.nodeId === newClickedId));
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
      .attr('transform', 'translate(-12,8)')
      .append('text')
      .text((d) => d.token)
      .attr('x', -3)
      .attr('y', 10)
      .attr('text-anchor', 'middle')
      .attr('transform', 'rotate(-30)')
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
      .attr('x', 5)
      .attr('y', -5)
      .attr('transform', 'rotate(-30)')
      .attr('text-anchor', 'middle')
      .attr('dominant-baseline', 'middle')
      .attr('font-size', 10)
      .attr('fill', '#334155');

    // Initial display of hovered nodes
    hoverSel.style('display', (d) => (d.featureId === visState.hoveredId ? '' : 'none'));
    // Remove visState.hoveredId from dependency array to prevent redraws on hover
  }, [screenSize, selectedGraph, visState.hoveredId, visState]);

  return (
    <div className="link-graph relative mt-3 min-h-[400px] w-[66%] min-w-[66%] max-w-[66%] pt-5">
      {/* <div className="mb-3 mt-2 flex w-full flex-row items-center justify-start gap-x-2">
        <div className="text-sm font-bold text-slate-600">Link Graph</div>
        <CustomTooltip wide trigger={<QuestionMarkCircledIcon className="h-4 w-4 text-slate-500" />}>
          <div className="flex flex-col">
            TODO: https://transformer-circuits.pub/2025/attribution-graphs/methods.html
          </div>
        </CustomTooltip>
      </div> */}
      <div className="tooltip tooltip-hidden" />
      <svg className="absolute z-0 w-full" height={HEIGHT} ref={bottomRef} />
      <svg className="absolute z-0 w-full" height={HEIGHT} ref={middleRef} />
      <svg className="absolute z-0 w-full" height={HEIGHT} ref={svgRef} />
    </div>
  );
}
