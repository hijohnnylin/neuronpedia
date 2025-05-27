/* eslint-disable no-param-reassign */

import { useGraphContext } from '@/components/provider/graph-provider';
import { Button } from '@/components/shadcn/button';
import { Card, CardContent } from '@/components/shadcn/card';
import { useScreenSize } from '@/lib/hooks/use-screen-size';
import * as Checkbox from '@radix-ui/react-checkbox';
import { Check, Circle, HelpCircleIcon, Share2, Trash2, XIcon } from 'lucide-react';
import { useCallback, useEffect, useRef, useState } from 'react';
import d3 from './d3-jetpack';
import { CLTGraphLink, CLTGraphNode, hideTooltip, showTooltip } from './utils';

const NODE_WIDTH = 75;
const NODE_HEIGHT = 25;

// Custom force container function to keep nodes within bounds
function forceContainer(bbox: [[number, number], [number, number]]) {
  let nodes: any[];

  function force(alpha: number) {
    let i;
    const n = nodes.length;
    let node;
    let x = 0;
    let y = 0;

    for (i = 0; i < n; i += 1) {
      node = nodes[i];
      x = node.x;
      y = node.y;

      if (x < bbox[0][0]) node.vx += (bbox[0][0] - x) * alpha;
      if (y < bbox[0][1]) node.vy += (bbox[0][1] - y) * alpha;
      if (x > bbox[1][0]) node.vx += (bbox[1][0] - x) * alpha;
      if (y > bbox[1][1]) node.vy += (bbox[1][1] - y) * alpha;
    }
  }

  force.initialize = function (_: any[]) {
    nodes = _;
  };

  return force;
}

// Extended node type for D3 force simulation
interface ForceNode {
  x: number;
  y: number;
  fx: number | null;
  fy: number | null;
  nodeId: string;
  featureId: string;
  node: CLTGraphNode;
  sortedSlug: string[]; // TODO: this doesn't make any sense? check logic
  dagrePositioned?: boolean;
  vx?: number;
  vy?: number;
}

// Extended link type for subgraph
interface SubgraphLink extends CLTGraphLink {
  color: string;
  pctInputColor: string;
  ogLinks?: SubgraphLink[];
  ogLink?: CLTGraphLink;
  source: any | ForceNode; // D3 force will replace this with the node object
  target: any | ForceNode; // D3 force will replace this with the node object
  sourceOffsetX?: number;
  targetOffsetX?: number;
}

export default function Subgraph() {
  const svgRef = useRef<SVGSVGElement>(null);
  const divRef = useRef<HTMLDivElement>(null);
  const {
    visState,
    selectedGraph,
    updateVisStateField,
    isEditingLabel,
    getOverrideClerpForNode,
    makeTooltipText,
    resetSelectedGraphToDefaultVisState,
    setIsCopyModalOpen,
  } = useGraphContext();
  const simulationRef = useRef<d3.Simulation<ForceNode, undefined> | null>(null);
  const nodeSelRef = useRef<d3.Selection<HTMLDivElement, ForceNode, HTMLDivElement, unknown> | null>(null);
  const memberNodeSelRef = useRef<d3.Selection<HTMLDivElement, CLTGraphNode, HTMLDivElement, ForceNode> | null>(null);
  const svgPathsRef = useRef<d3.Selection<SVGPathElement, SubgraphLink, SVGGElement, unknown> | null>(null);
  const edgeLabelsRef = useRef<d3.Selection<SVGTextElement, SubgraphLink, SVGGElement, unknown> | null>(null);
  const sgLinksRef = useRef<SubgraphLink[]>([]);
  const [selForceNodes, setSelForceNodes] = useState<ForceNode[]>([]);
  const [nodeIdToNode, setNodeIdToNode] = useState<Record<string, CLTGraphNode>>({});
  const [isMetaKeyHeld, setIsMetaKeyHeld] = useState(false);

  // Ref to hold the latest active grouping state for event handlers
  const activeGroupingRef = useRef(visState.subgraph?.activeGrouping);
  // Ref to hold the latest isEditingLabel state for event handlers
  const isEditingLabelRef = useRef(isEditingLabel);

  const lastClickedIdRef = useRef(visState.clickedId);

  const screenSize = useScreenSize();

  // Helper to create pct input color function
  const pctInputColorFn = useCallback((d: number) => {
    const linearScale = d3.scaleLinear().domain([-0.4, 0.4]);
    const linearTScale = d3
      .scaleLinear()
      .domain([0, 0.5, 0.5, 1])
      .range([0, 0.5 - 0.001, 0.5 + 0.001, 1]);
    return d3.interpolatePRGn(linearTScale(linearScale(d)));
  }, []);

  // Background to text color helper
  const bgColorToTextColor = useCallback((color?: string) => {
    if (!color) return undefined;
    // Simple luminance-based text color
    const rgb = d3.rgb(color);
    const luminance = 0.299 * rgb.r + 0.587 * rgb.g + 0.114 * rgb.b;
    return luminance > 160 ? 'black' : 'white';
  }, []);

  // Effect to keep the activeGroupingRef updated
  useEffect(() => {
    activeGroupingRef.current = visState.subgraph?.activeGrouping;
  }, [visState.subgraph?.activeGrouping]);

  // Effect to keep the isEditingLabelRef updated
  useEffect(() => {
    isEditingLabelRef.current = isEditingLabel;
  }, [isEditingLabel]);

  // Effect to keep the lastClickedIdRef updated
  useEffect(() => {
    lastClickedIdRef.current = visState.clickedId;
  }, [visState.clickedId]);

  // Handler for keydown event to enable grouping mode
  useEffect(() => {
    if (!selectedGraph) return;

    function handleKeyDown(ev: KeyboardEvent) {
      if (ev.repeat) return;
      if (!visState.isEditMode || ev.key !== 'g') return;

      if (visState.subgraph) {
        updateVisStateField('subgraph', {
          ...visState.subgraph,
          activeGrouping: {
            ...visState.subgraph.activeGrouping,
            isActive: true,
          },
        });
      }

      const subgraphEl = document.querySelector('.subgraph');
      if (subgraphEl) {
        subgraphEl.classList.add('is-grouping');
      }
    }

    function handleKeyUp(ev: KeyboardEvent) {
      if (!visState.isEditMode || ev.key !== 'g') return;
      if (!visState.subgraph) return;

      if (visState.subgraph.activeGrouping.selectedNodeIds.size > 1) {
        const allSelectedIds: string[] = [];
        let prevSupernodeLabel = '';
        const supernodesToRemove: string[][] = [];

        visState.subgraph.activeGrouping.selectedNodeIds.forEach((id) => {
          const node = nodeIdToNode[id];
          if (!node?.memberNodeIds) {
            allSelectedIds.push(id);
            return;
          }

          prevSupernodeLabel = node.ppClerp || '';

          // find the supernode to remove
          // eslint-disable-next-line
          const supernodeToRemove = visState.subgraph?.supernodes.find(([_, ...nodeIds]) =>
            nodeIds.every((d) => node.memberNodeIds?.includes(d)),
          );

          if (supernodeToRemove) {
            supernodesToRemove.push(supernodeToRemove);
          }

          // Add its member nodes to selection
          node.memberNodeIds.forEach((memberId) => allSelectedIds.push(memberId));
        });

        // Find a label for the new supernode
        const label =
          prevSupernodeLabel || allSelectedIds.map((id) => nodeIdToNode[id]?.ppClerp).find((d) => d) || 'supernode';

        // remove the supernodes to remove
        let newSupernodes = visState.subgraph?.supernodes.filter(
          // eslint-disable-next-line
          ([_, ...nodeIds]) => !supernodesToRemove.some((d) => nodeIds.every((e) => d.includes(e))),
        );

        if (visState.subgraph) {
          newSupernodes = [...(newSupernodes || []), [label, ...new Set(allSelectedIds)]];
          updateVisStateField('subgraph', {
            ...visState.subgraph,
            supernodes: newSupernodes,
            activeGrouping: {
              isActive: false,
              selectedNodeIds: new Set(),
            },
          });
        }
      } else {
        // reset grouping state
        updateVisStateField('subgraph', {
          ...visState.subgraph,
          activeGrouping: {
            isActive: false,
            selectedNodeIds: new Set(),
          },
        });
      }

      const subgraphEl = document.querySelector('.subgraph');
      if (subgraphEl) {
        subgraphEl.classList.remove('is-grouping');
      }
    }

    document.addEventListener('keydown', handleKeyDown);
    document.addEventListener('keyup', handleKeyUp);

    // eslint-disable-next-line
    return () => {
      document.removeEventListener('keydown', handleKeyDown);
      document.removeEventListener('keyup', handleKeyUp);
    };
  }, [selectedGraph, visState, nodeIdToNode, updateVisStateField]);

  // Handler for tracking meta/ctrl key state for pin/unpin mode
  useEffect(() => {
    function handleKeyDown(ev: KeyboardEvent) {
      if (ev.key === 'Meta' || ev.key === 'Control') {
        setIsMetaKeyHeld(true);
      }
    }

    function handleKeyUp(ev: KeyboardEvent) {
      if (ev.key === 'Meta' || ev.key === 'Control') {
        setIsMetaKeyHeld(false);
      }
    }

    document.addEventListener('keydown', handleKeyDown);
    document.addEventListener('keyup', handleKeyUp);

    return () => {
      document.removeEventListener('keydown', handleKeyDown);
      document.removeEventListener('keyup', handleKeyUp);
    };
  }, []);

  // Initialize subgraph state if not already initialized
  useEffect(() => {
    if (visState.subgraph) return;

    updateVisStateField('subgraph', {
      sticky: true,
      dagrefy: true,
      supernodes: visState.supernodes || [],
      activeGrouping: {
        isActive: false,
        selectedNodeIds: new Set(),
      },
    });
  }, [visState, updateVisStateField]);

  // Initialize the D3 subgraph visualization - runs only when layout needs reset
  useEffect(() => {
    console.log('re-rendering simulation setup'); // Add log for debugging resets
    if (!svgRef.current || !divRef.current || !selectedGraph) return;

    // Stop any existing simulation before clearing refs
    simulationRef.current?.stop();

    // Clear any existing content and refs
    d3.select(svgRef.current).selectAll('*').remove();
    d3.select(divRef.current).selectAll('*').remove();
    nodeSelRef.current = null;
    memberNodeSelRef.current = null;
    svgPathsRef.current = null;
    edgeLabelsRef.current = null;
    sgLinksRef.current = [];
    simulationRef.current = null;

    const data = selectedGraph;
    const { nodes, links } = data;

    // Set up margins and container dimensions
    const margin = {
      top: 30,
      bottom: 50,
      left: 35,
      right: 35,
    };

    const svgBBox = svgRef.current.getBoundingClientRect();
    const width = svgBBox.width - margin.left - margin.right;
    const height = 400 - margin.top - margin.bottom;

    const svg = d3.select(svgRef.current).append('g').attr('transform', `translate(${margin.left},${margin.top})`);

    const div = d3
      .select(divRef.current)
      .style('position', 'absolute')
      .style('top', `${margin.top}px`) //
      .style('left', `${margin.left}px`)
      .style('width', `${width}px`)
      .style('height', `${height}px`);

    // Add svg definitions for markers
    svg.append('defs').html(`
      <marker id="mid-positive" viewBox="0 0 10 10" refX="5" refY="5" 
        markerWidth="4" markerHeight="4" orient="auto-start-reverse">
        <path d="M 0 0 L 10 5 L 0 10 z" fill="#008000" />
      </marker>
      <marker id="mid-negative" viewBox="0 0 10 10" refX="5" refY="5" 
        markerWidth="4" markerHeight="4" orient="auto-start-reverse">
        <path d="M 0 0 L 10 5 L 0 10 z" fill="#800000" />
      </marker>
    `);

    // Pick out the subgraph and do supernode surgery
    nodes.forEach((d) => {
      d.supernodeId = undefined;
    });
    const pinnedIds = visState.pinnedIds.slice(0, 200); // max of 200 nodes
    const pinnedNodes = nodes.filter((d) => pinnedIds.includes(d.nodeId || ''));

    // Create mapping from nodeId to node
    const newNodeIdToNode: Record<string, CLTGraphNode> = Object.fromEntries(
      pinnedNodes.map((d) => [d.nodeId || '', d]),
    );
    setNodeIdToNode(newNodeIdToNode);

    // Create supernodes and mark their children
    const supernodes = (visState.subgraph?.supernodes || [])
      .map(([label, ...nodeIds], i) => {
        const nodeId = newNodeIdToNode[label] ? `supernode-${i}` : label;
        const memberNodes = nodeIds.map((id) => newNodeIdToNode[id]).filter(Boolean);
        memberNodes.forEach((d) => {
          if (d) d.supernodeId = nodeId;
        });

        const rv: CLTGraphNode = {
          nodeId,
          featureId: `supernode-${i}`,
          ppClerp: label,
          layer: d3.mean(memberNodes, (d) => +d.layer)?.toString() || '',
          ctx_idx: d3.mean(memberNodes, (d) => d.ctx_idx || 0) || 0,
          streamIdx: d3.mean(memberNodes, (d) => d.streamIdx || 0),
          memberNodeIds: nodeIds,
          memberNodes,
          isSuperNode: true,
          node_id: nodeId,
          feature: i,
          feature_type: 'supernode',
          token_prob: 0,
          is_target_logit: false,
          run_idx: 0,
          reverse_ctx_idx: 0,
          jsNodeId: nodeId,
          clerp: label,
        };
        newNodeIdToNode[rv.nodeId || ''] = rv;

        return rv;
      })
      .filter((d) => d.memberNodes?.length);

    // Update clerps - ensure nodes have proper display names
    nodes.forEach((d) => {
      d.ppClerp = d.clerp;
    });

    supernodes.forEach(({ ppClerp, memberNodes }) => {
      if (memberNodes && memberNodes.length === 1 && ppClerp === memberNodes[0].ppClerp) return;

      memberNodes?.forEach((d) => {
        if (d) {
          d.ppClerp = `[${ppClerp}]${ppClerp !== d.ppClerp ? ` ${d.ppClerp}` : ''}`;
        }
      });
    });

    // Calculate inputAbsSumExternalSn for each node
    pinnedNodes.forEach((d) => {
      d.inputAbsSumExternalSn = d3.sum(d.sourceLinks || [], (e) => {
        if (!e.sourceNode?.supernodeId) return Math.abs(e.weight || 0);
        return e.sourceNode?.supernodeId === d.supernodeId ? 0 : Math.abs(e.weight || 0);
      });

      d.sgSnInputWeighting = d.inputAbsSumExternalSn && d.inputAbsSum ? d.inputAbsSumExternalSn / d.inputAbsSum : 0;
    });

    // Collect nodes for the subgraph
    const sgNodes = pinnedNodes.filter((d) => !d.supernodeId).concat(supernodes);

    // For supernodes, calculate combined values
    sgNodes.forEach((d) => {
      if (d.isSuperNode && d.memberNodes) {
        d.inputAbsSum = d3.sum(d.memberNodes, (e) => e.inputAbsSum || 0);
        d.inputAbsSumExternalSn = d3.sum(d.memberNodes, (e) => e.inputAbsSumExternalSn || 0);
      } else {
        d.memberNodes = [d];
      }

      const sum = d3.sum(d.memberNodes || [], (e) => e.sgSnInputWeighting || 0);
      d.memberNodes?.forEach((e) => {
        if (e && sum > 0) {
          e.sgSnInputWeighting = (e.sgSnInputWeighting || 0) / sum;
        }
      });
    });

    // Select and process subgraph links
    const newSgLinks: SubgraphLink[] = links
      .filter((d) => newNodeIdToNode[d.sourceNode?.nodeId || ''] && newNodeIdToNode[d.targetNode?.nodeId || ''])
      .map((d) => ({
        source: d.sourceNode?.nodeId || '',
        target: d.targetNode?.nodeId || '',
        weight: d.weight || 0,
        color: d.pctInputColor || '',
        pctInputColor: d.pctInputColor || '',
        ogLink: d,
        sourceNode: d.sourceNode,
        targetNode: d.targetNode,
      }));

    // Remap source/target to supernodes
    newSgLinks.forEach((link) => {
      const sourceNode = newNodeIdToNode[link.source];
      const targetNode = newNodeIdToNode[link.target];

      if (sourceNode?.supernodeId) link.source = sourceNode.supernodeId;
      if (targetNode?.supernodeId) link.target = targetNode.supernodeId;
    });

    // Combine parallel links and remove self-links
    const combinedLinks: SubgraphLink[] = d3
      .nestBy(newSgLinks, (d) => `${d.source}-${d.target}`)
      .map((cLinks) => {
        const weight = d3.sum(cLinks, (link) => {
          const targetNode = link.ogLink?.targetNode;
          if (!targetNode) return 0;

          const { inputAbsSumExternalSn = 0, sgSnInputWeighting = 0 } = targetNode;
          return inputAbsSumExternalSn ? (link.weight / inputAbsSumExternalSn) * sgSnInputWeighting : 0;
        });

        return {
          source: cLinks[0].source,
          target: cLinks[0].target,
          weight,
          color: pctInputColorFn(weight),
          pctInput: weight,
          pctInputColor: pctInputColorFn(weight),
          ogLinks: cLinks,
          sourceNode: cLinks[0].sourceNode,
          targetNode: cLinks[0].targetNode,
        };
      })
      .filter((d) => d.source !== d.target);

    const sgLinks = d3.sort(combinedLinks, (d) => Math.abs(d.weight));
    sgLinksRef.current = sgLinks; // Store links in ref

    // Create scales for initial positioning
    const xScale = d3
      .scaleLinear()
      .domain(d3.extent(sgNodes, (d) => d.ctx_idx) as [number, number])
      .range([0, (width * 3) / 4]);

    const yScale = d3
      .scaleLinear()
      .domain(d3.extent(sgNodes, (d) => d.streamIdx).toReversed() as [number, number])
      .range([0, height - NODE_HEIGHT]);

    // Create force nodes with position data
    // Try to preserve positions across renders
    const existingForceNodes = selForceNodes.length > 0 ? new Map(selForceNodes.map((n) => [n.node.nodeId, n])) : null;

    const newForceNodes: ForceNode[] = sgNodes.map((node) => {
      const existing = existingForceNodes?.get(node.nodeId || '');
      return {
        x: existing ? existing.x : xScale(node.ctx_idx || 0),
        y: existing ? existing.y : yScale(node.streamIdx || 0),
        fx: existing?.fx || null,
        fy: existing?.fy || null,
        nodeId: node.nodeId || '',
        featureId: node.featureId || '',
        node,
        // TODO: this doesn't make any sense? check logic
        sortedSlug: d3.sort((node.memberNodes || []).map((d) => d.featureIndex).join(' ')),
      };
    });

    const sortedForceNodes = d3.sort(newForceNodes, (d) => d.sortedSlug);
    setSelForceNodes(sortedForceNodes);

    // Create new simulation
    const newSimulation = d3
      .forceSimulation(sortedForceNodes)
      .force(
        'link',
        d3.forceLink(sgLinks).id((d: any) => d.node.nodeId),
      )
      .force('charge', d3.forceManyBody().strength(-100))
      .force('collide', d3.forceCollide(Math.sqrt(NODE_HEIGHT ** 2 + NODE_WIDTH ** 2) / 2))
      .force(
        'container',
        forceContainer([
          [-10, 0],
          [width - NODE_HEIGHT, height - NODE_HEIGHT],
        ]),
      )
      .force('x', d3.forceX((d: ForceNode) => xScale(d.node.ctx_idx || 0)).strength(0.1))
      .force('y', d3.forceY((d: ForceNode) => yScale(d.node.streamIdx || 0)).strength(2));

    // Store the new simulation in the ref
    simulationRef.current = newSimulation;

    // Function to run on each simulation tick - defined inside so it closes over refs if needed
    function renderForce() {
      if (!nodeSelRef.current) return;
      nodeSelRef.current.translate((d) => [d.x, d.y]);

      // eslint-disable-next-line @typescript-eslint/no-use-before-define
      renderEdges(); // Calls renderEdges which uses refs

      // Only update edge labels if they exist and nodes aren't dagre-positioned
      edgeLabelsRef.current
        ?.filter((d) => {
          const source = typeof d.source === 'object' ? d.source : null;
          const target = typeof d.target === 'object' ? d.target : null;
          if (!source || !target) return false;
          return !(source.dagrePositioned && target.dagrePositioned);
        })
        .translate((d) => {
          const source = typeof d.source === 'object' ? d.source : null;
          const target = typeof d.target === 'object' ? d.target : null;
          if (!source || !target) return [0, 0];

          return [(source.x + target.x) / 2 + NODE_WIDTH / 2, (source.y + target.y) / 2 + NODE_HEIGHT / 2];
        });
    }

    // Function to render edges with offsets - defined inside to access refs
    function renderEdges() {
      if (!svgPathsRef.current) return;
      const currentSgLinks = sgLinksRef.current; // Use links from ref

      // Adjust link source positions based on source node member count
      // eslint-disable-next-line @typescript-eslint/no-shadow
      d3.nestBy(currentSgLinks, (d) => d.source).forEach((links) => {
        const sourceNode = typeof links[0].source === 'object' ? links[0].source : null;
        if (!sourceNode || !sourceNode.node) return;

        const numSlots = sourceNode.node.memberNodes?.length || 1;
        const totalWidth = Math.min(4, numSlots) * 8;

        d3.sort(links, (d) => {
          const target = typeof d.target === 'object' ? d.target : null;
          if (!target) return 0;
          return Math.atan2(target.y - sourceNode.y, target.x - sourceNode.x);
        }).forEach((d, i) => {
          d.sourceOffsetX = ((i - links.length / 2) * totalWidth) / links.length;
        });
      });

      // Adjust link target positions based on target node member count
      // eslint-disable-next-line @typescript-eslint/no-shadow
      d3.nestBy(currentSgLinks, (d) => d.target).forEach((links) => {
        const targetNode = typeof links[0].target === 'object' ? links[0].target : null;
        if (!targetNode || !targetNode.node) return;

        const numSlots = targetNode.node.memberNodes?.length || 1;
        const totalWidth = (Math.min(4, numSlots) + 1) * 3;

        d3.sort(links, (d) => {
          const source = typeof d.source === 'object' ? d.source : null;
          if (!source) return 0;
          return -Math.atan2(source.y - targetNode.y, source.x - targetNode.x);
        }).forEach((d, i) => {
          d.targetOffsetX = ((i - links.length / 2) * totalWidth) / links.length;
        });
      });

      // Update SVG paths using the ref
      svgPathsRef.current.attr('d', (d: SubgraphLink) => {
        const source = typeof d.source === 'object' ? d.source : null;
        const target = typeof d.target === 'object' ? d.target : null;
        if (!source || !target) return '';

        const x0 = source.x + NODE_WIDTH / 2 + (d.sourceOffsetX || 0);
        const y0 = source.y;
        const x1 = target.x + NODE_WIDTH / 2 + (d.targetOffsetX || 0);
        const y1 = target.y + (target.node.textHeight || 0); // + 28;

        return `M${x0},${y0} L${x1},${y1}`;
      });
    }

    // Create SVG path elements for links
    const svgPaths = svg
      .appendMany('path.link-path', sgLinks)
      .attr('fill', 'none')
      .attr('marker-mid', (d) => (d.weight > 0 ? 'url(#mid-positive)' : 'url(#mid-negative)'))
      .attr('stroke-width', (d) => Math.abs(d.weight) * 15)
      .attr('stroke', (d) => d.color)
      .attr('opacity', 0.8)
      .attr('stroke-linecap', 'round');

    // @ts-ignore
    svgPathsRef.current = svgPaths;

    // Add weight labels for links
    const edgeLabels = svg.appendMany('text.weight-label', sgLinks);
    // @ts-ignore
    edgeLabelsRef.current = edgeLabels;

    // Setup drag behavior
    const drag = d3
      .drag()
      .on('drag', (ev: any) => {
        // Only when actually dragging, mark as no longer dagre positioned and restart sim
        ev.subject.dagrePositioned = false;
        if (!ev.active) simulationRef.current?.alphaTarget(0.3).restart();
        ev.subject.fx = ev.x;
        ev.subject.x = ev.x;
        ev.subject.fy = ev.y;
        ev.subject.y = ev.y;
        hideTooltip();
        renderForce();
      })
      .on('end', (ev: any) => {
        if (!ev.active) simulationRef.current?.alphaTarget(0);
        if (!visState.subgraph?.sticky && !ev.subject.dagrePositioned) {
          ev.subject.fx = null;
          ev.subject.fy = null;
        }
      });

    // Create node elements
    const nodeSel = div
      .appendMany('div.supernode-container', sortedForceNodes)
      .translate((d) => [d.x, d.y])
      .style('width', `${NODE_WIDTH}px`)
      .style('height', `${NODE_HEIGHT}px`)
      .on('mouseover', (ev: MouseEvent, d: ForceNode) => {
        if (ev.buttons === 1) return; // Ignore if dragging/clicking
        if (!ev.metaKey && !ev.ctrlKey && !activeGroupingRef.current?.isActive && !isEditingLabelRef.current) {
          updateVisStateField('hoveredId', d.node.featureId || null);
          updateVisStateField('hoveredCtxIdx', d.node.ctx_idx);
          showTooltip(ev, d.node, makeTooltipText(d.node));
        }
      })
      .on('mouseleave', (ev: MouseEvent) => {
        if (ev.buttons === 1) return; // Ignore if dragging/clicking
        updateVisStateField('hoveredId', null);
        updateVisStateField('hoveredCtxIdx', null);
        hideTooltip();
      })
      .on('click', (ev: MouseEvent, d: ForceNode) => {
        const currentActiveGrouping = activeGroupingRef.current;

        // Handle grouping mode
        if (currentActiveGrouping?.isActive) {
          const nodeId = d.node.supernodeId || d.node.nodeId;
          if (nodeId) {
            // Read selected IDs from the ref, but create a new Set for modification
            const currentSelectedNodeIds = currentActiveGrouping?.selectedNodeIds || new Set();
            const selectedNodeIds = new Set(currentSelectedNodeIds);
            if (selectedNodeIds.has(nodeId)) {
              selectedNodeIds.delete(nodeId);
            } else {
              selectedNodeIds.add(nodeId);
            }

            // Update the main state (which will trigger the ref update and styling effect)
            updateVisStateField('subgraph', {
              sticky: visState.subgraph?.sticky || false,
              dagrefy: visState.subgraph?.dagrefy || false,
              supernodes: visState.subgraph?.supernodes || [],
              activeGrouping: {
                isActive: true,
                selectedNodeIds,
              },
            });
          }

          ev.stopPropagation();
          ev.preventDefault();
          return;
        }

        if (ev.metaKey || ev.ctrlKey) {
          // Toggle pinned state
          const newPinnedIds = [...visState.pinnedIds];
          const pinnedIndex = newPinnedIds.indexOf(d.node.nodeId || '');

          if (pinnedIndex === -1 && d.node.nodeId) {
            newPinnedIds.push(d.node.nodeId);
          } else if (pinnedIndex !== -1) {
            newPinnedIds.splice(pinnedIndex, 1);
          }

          updateVisStateField('pinnedIds', newPinnedIds);
        } else {
          // Set as clicked node
          const newClickedId = lastClickedIdRef.current === d.node.nodeId ? null : d.node.nodeId;
          if (newClickedId !== lastClickedIdRef.current) {
            updateVisStateField('clickedId', newClickedId || null);
            updateVisStateField('clickedCtxIdx', newClickedId ? d.node.ctx_idx : null);
            lastClickedIdRef.current = newClickedId || null;
          }
        }
      })
      // @ts-ignore
      .call(drag);

    // @ts-ignore
    nodeSelRef.current = nodeSel;

    // Style nodes as supernodes
    nodeSel.classed('is-supernode', true).style('height', `${NODE_HEIGHT + 12}px`);

    // Add member circles for each node in the supernode
    const memberNodeSel = nodeSel
      .append('div.member-circles')
      .style('width', (d) => (d.node.memberNodes && d.node.memberNodes?.length <= 4 ? 'auto' : 'calc(32px + 12px)'))
      .style('gap', (d) => (d.node.memberNodes && d.node.memberNodes?.length <= 4 ? '1px' : '0px'))
      .selectAll('div.member-circle')
      .data((d: ForceNode) => d.node.memberNodes || [])
      .join('div.member-circle')
      .classed('not-clt-feature', (d) => d.feature_type !== 'cross layer transcoder')
      .st({
        marginLeft(d: CLTGraphNode, i: number) {
          // @ts-ignore
          // eslint-disable-next-line
          const n = this.parentNode.childNodes.length;
          return n <= 4 ? 0 : i === 0 ? 0 : -((n - 4) * 8) / (n - 1);
        },
      })
      .on('mouseover', (ev: MouseEvent, d: CLTGraphNode) => {
        if (ev.buttons === 1) return; // Ignore if dragging
        if (activeGroupingRef.current?.isActive || ev.metaKey || ev.ctrlKey || isEditingLabelRef.current) return;
        updateVisStateField('hoveredId', d.featureId || null);
        updateVisStateField('hoveredCtxIdx', d.ctx_idx);
        showTooltip(ev, d, makeTooltipText(d));
        ev.stopPropagation();
      })
      .on('mouseleave', (ev: MouseEvent) => {
        if (ev.buttons === 1) return; // Ignore if dragging out
        updateVisStateField('hoveredId', null);
        updateVisStateField('hoveredCtxIdx', null);
        hideTooltip();
      })
      .on('click', (ev: MouseEvent, d: CLTGraphNode) => {
        if (!activeGroupingRef.current?.isActive) {
          ev.stopPropagation();
        } else {
          // grouping, don't select
          return;
        }
        // Regular click behavior
        if (ev.metaKey || ev.ctrlKey) {
          // Toggle pinned state
          const newPinnedIds = [...visState.pinnedIds];
          const pinnedIndex = newPinnedIds.indexOf(d.nodeId || '');

          if (pinnedIndex === -1 && d.nodeId) {
            newPinnedIds.push(d.nodeId);
          } else if (pinnedIndex !== -1) {
            newPinnedIds.splice(pinnedIndex, 1);
          }

          updateVisStateField('pinnedIds', newPinnedIds);
        } else {
          // Set as clicked node
          const newClickedId = lastClickedIdRef.current === d.nodeId ? null : d.nodeId;
          if (newClickedId !== lastClickedIdRef.current) {
            updateVisStateField('clickedId', newClickedId || null);
            updateVisStateField('clickedCtxIdx', newClickedId ? d.ctx_idx : null);
            lastClickedIdRef.current = newClickedId || null;
          }
        }
      })
      .attr('title', (d: CLTGraphNode) => getOverrideClerpForNode(d) || '');

    // @ts-ignore
    memberNodeSelRef.current = memberNodeSel;

    // Add ungroup button for supernodes in edit mode
    if (visState.isEditMode) {
      nodeSel
        .select('.member-circles')
        .filter((d) => d.node.isSuperNode || false)
        .append('div.ungroup-btn')
        .text('×')
        .style('top', '2px')
        .style('left', '-15px')
        .style('position', 'absolute')
        .style('cursor', 'pointer')
        .style('fontWeight', 'bold')
        .on('click', (ev: MouseEvent, d: ForceNode) => {
          ev.stopPropagation();

          if (visState.subgraph && d.node.memberNodeIds) {
            const newSupernodes = visState.subgraph.supernodes.filter(
              // eslint-disable-next-line
              ([_, ...nodeIds]) => !nodeIds.every((id) => d.node.memberNodeIds?.includes(id)),
            );

            updateVisStateField('subgraph', {
              ...visState.subgraph,
              supernodes: newSupernodes,
            });
          }
        });
    }

    // Add text labels
    const nodeTextSel = nodeSel.append('div.node-text-container');

    nodeTextSel
      .append('span')
      .text((d) => getOverrideClerpForNode(d.node) || '')
      .on('click', (ev, d) => {
        if (!visState.isEditMode) return;
        if (!d.node.isSuperNode) return;
        ev.stopPropagation();

        const spanSel = d3.select(ev.target).st({ display: 'none' });
        const input = d3
          .select(spanSel.node().parentNode)
          .append('input')
          .at({ class: 'temp-edit', value: spanSel.text() })
          // eslint-disable-next-line
          .on('blur', save)
          .on('keydown', (saveEvent) => {
            if (saveEvent.key === 'Enter') {
              // eslint-disable-next-line
              save();
              input.node()?.blur();
            }
            saveEvent.stopPropagation();
          });

        input.node()?.focus();

        function save() {
          // eslint-disable-next-line
          const idx = visState.subgraph?.supernodes.findIndex(([_, ...nodeIds]) =>
            nodeIds.every((id) => d.node.memberNodeIds?.includes(id)),
          );

          if (visState.subgraph && typeof idx === 'number' && idx >= 0) {
            const newValue = input.node()?.value || 'supernode';

            // We need to create a new one because otherwise useEffect won't trigger
            const newSupernodes = visState.subgraph.supernodes.map((supernodeEntry, i) => {
              if (i === idx) {
                // eslint-disable-next-line
                const [_, ...restIds] = supernodeEntry;
                return [newValue, ...restIds];
              }
              return supernodeEntry;
            });

            updateVisStateField('subgraph', {
              ...visState.subgraph,
              supernodes: newSupernodes,
            });

            input.remove();
            spanSel.st({ display: 'inline' });
          } else {
            // Handle case where idx is not found or subgraph doesn't exist
            console.error('Could not find supernode index or subgraph is missing.');
            input.remove();
            spanSel.st({ display: 'inline' });
          }
        }
      });

    // Get actual text height for each node
    nodeTextSel.each(function (d: ForceNode) {
      // @ts-ignore
      // eslint-disable-next-line
      d.node.textHeight = this.getBoundingClientRect().height || -8;
    });

    // Add weight indicators
    nodeSel
      .append('div.clicked-weight.source')
      .style('position', 'absolute')
      .style('bottom', '-15px')
      .style('left', '5px')
      .style('fontSize', '10px');

    nodeSel
      .append('div.clicked-weight.target')
      .style('position', 'absolute')
      .style('bottom', '-15px')
      .style('right', '5px')
      .style('fontSize', '10px');

    // Apply sticky and dagrefy settings
    if (visState.subgraph?.sticky === false) {
      // Unsticky nodes
      sortedForceNodes.forEach((d) => {
        d.fx = null;
        d.fy = null;
      });
      simulationRef.current?.alphaTarget(0.3).restart();
    }

    if (visState.subgraph?.dagrefy) {
      // Check for saved positions
      if (visState.sg_pos) {
        const nums = visState.sg_pos.split(' ').map((d) => +d);
        sortedForceNodes.forEach((d, i) => {
          d.fx = (nums[i * 2 + 0] / 1000) * width;
          d.x = d.fx;
          d.fy = (nums[i * 2 + 1] / 1000) * height;
          d.y = d.fy;
        });

        nodeSel.translate((d) => [d.x, d.y]);

        // Mark positions as stored
        updateVisStateField('og_sg_pos', visState.sg_pos);
        updateVisStateField('sg_pos', '');
      }
    }

    // Initial render of edges and force
    renderEdges();
    renderForce(); // Start simulation ticks

    // Add tick handler to simulation stored in ref
    simulationRef.current?.on('tick', renderForce);

    // Cleanup function
    // eslint-disable-next-line
    return () => {
      simulationRef.current?.stop();
    };
  }, [
    screenSize,
    selectedGraph,
    visState.pinnedIds,
    visState.subgraph?.supernodes,
    visState.subgraph?.sticky,
    visState.subgraph?.dagrefy,
    visState.sg_pos,
    visState.isHideLayer,
    visState.clerps,
    updateVisStateField,
    pctInputColorFn,
    bgColorToTextColor,
  ]);

  // Separate useEffect for applying styles based on interaction state
  useEffect(() => {
    // Ensure selections exist and we have the necessary state
    if (!nodeSelRef.current || !memberNodeSelRef.current || !visState.subgraph || !nodeIdToNode) return;

    const nodeSel = nodeSelRef.current;
    const memberNodeSel = memberNodeSelRef.current;
    const currentSgLinks = sgLinksRef.current; // Use links from ref for styling logic

    // Apply classes based on interaction state
    nodeSel
      .classed('clicked', (d: ForceNode) => d.node.nodeId === visState.clickedId)
      .classed('hovered', (d: ForceNode) => d.node.featureId === visState.hoveredId)
      .classed(
        'grouping-selected', // This class should now be applied correctly
        (d: ForceNode) => visState.subgraph?.activeGrouping.selectedNodeIds.has(d.node.nodeId || '') || false,
      );

    // --- Logic for calculating and applying tmpClickedLink styles ---
    // Clear previous highlighting state from actual node data used by D3
    Object.values(nodeIdToNode).forEach((node) => {
      node.tmpClickedLink = undefined;
      node.tmpClickedSgSource = undefined;
      node.tmpClickedSgTarget = undefined;
    });
    // Also clear temporary data possibly bound to elements if necessary (might be redundant)
    memberNodeSel.each((d) => {
      if (d) d.tmpClickedLink = undefined;
    });

    // Calculate and apply new clicked link highlighting state
    if (visState.clickedId && nodeIdToNode[visState.clickedId]) {
      currentSgLinks.forEach((link) => {
        // Resolve source/target, potentially using nodeIdToNode map if needed
        const sourceForceNode = typeof link.source === 'object' ? link.source : null;
        const targetForceNode = typeof link.target === 'object' ? link.target : null;
        const sourceNodeId = sourceForceNode?.node.nodeId || (link.source as string);
        const targetNodeId = targetForceNode?.node.nodeId || (link.target as string);
        const sourceNode = nodeIdToNode[sourceNodeId];
        const targetNode = nodeIdToNode[targetNodeId];

        if (link.ogLinks && sourceNode && targetNode) {
          if (sourceNodeId === visState.clickedId) {
            // Highlight target member nodes
            link.ogLinks.forEach((ogLink) => {
              if (ogLink.targetNode && nodeIdToNode[ogLink.targetNode.nodeId || '']) {
                // Ensure node exists in current map before assigning
                nodeIdToNode[ogLink.targetNode.nodeId || ''].tmpClickedLink = ogLink;
              }
            });
            // Store sg link info for target supernode/node styling if needed
            if (nodeIdToNode[targetNodeId]) {
              nodeIdToNode[targetNodeId].tmpClickedSgTarget = link;
            }
          } else if (targetNodeId === visState.clickedId) {
            // Highlight source member nodes
            link.ogLinks.forEach((ogLink) => {
              if (ogLink.sourceNode && nodeIdToNode[ogLink.sourceNode.nodeId || '']) {
                // Ensure node exists in current map before assigning
                nodeIdToNode[ogLink.sourceNode.nodeId || ''].tmpClickedLink = ogLink;
              }
            });
            // Store sg link info for source supernode/node styling if needed
            if (nodeIdToNode[sourceNodeId]) {
              nodeIdToNode[sourceNodeId].tmpClickedSgSource = link;
            }
          }
        }
      });
    }

    // Apply styles based on the potentially updated tmpClickedLink property
    memberNodeSel
      .classed('clicked', (d: CLTGraphNode) => d.nodeId === visState.clickedId)
      .classed('hovered', (d: CLTGraphNode) => d.featureId === visState.hoveredId)
      .style('background', (d: CLTGraphNode) => d?.tmpClickedLink?.pctInputColor || '#fff')
      .style('color', (d: CLTGraphNode) => bgColorToTextColor(d?.tmpClickedLink?.pctInputColor) || 'black');

    // NOTE: This effect intentionally avoids restarting the simulation or changing layout.
  }, [
    visState.clickedId,
    visState.hoveredId,
    visState.subgraph?.activeGrouping.selectedNodeIds,
    bgColorToTextColor,
    nodeIdToNode,
  ]);

  const [showSubgraphHelp, setShowSubgraphHelp] = useState(true);

  useEffect(() => {
    if (visState.pinnedIds.length === 0) {
      setShowSubgraphHelp(true);
    } else {
      setShowSubgraphHelp(false);
    }
  }, [visState.pinnedIds]);

  return (
    <Card className="h-full w-full flex-1 bg-white">
      <CardContent className="h-full px-0 py-0">
        {/* <div className="mb-3 mt-2 flex w-full flex-row items-center justify-start gap-x-2">
        <div className="text-sm font-bold text-slate-600">Subgraph</div>
        <CustomTooltip wide trigger={<QuestionMarkCircledIcon className="h-4 w-4 text-slate-500" />}>
          <div className="flex flex-col">
            <p>The subgraph shows a force-directed visualization of pinned nodes.</p>
            <p>Keyboard shortcuts:</p>
            <ul>
              <li>Press g to enter grouping mode, then click nodes to group them</li>
              <li>Hold Ctrl/Cmd to pin/unpin nodes</li>
            </ul>
          </div>
        </CustomTooltip>
      </div> */}
        <div className="subgraph relative h-full w-full">
          <svg className="absolute h-full w-full" ref={svgRef} />
          <div className="absolute h-full w-full" ref={divRef} />

          {/* Supernode Grouping Mode Label */}
          {visState.subgraph?.activeGrouping.isActive &&
            (visState.pinnedIds.length > 1 ? (
              <div className="absolute left-1/2 top-2.5 z-10 -translate-x-1/2 transform cursor-default whitespace-pre rounded-md bg-sky-600 px-2 py-1 text-center text-[11px] font-medium text-white">
                {visState.subgraph.activeGrouping.selectedNodeIds.size === 0 ||
                (visState.subgraph.activeGrouping.selectedNodeIds.size > 2 && visState.subgraph?.supernodes.length > 0)
                  ? `Grouping Mode: Click nodes below to select them, release 'g' to group. Click ✕ to ungroup.`
                  : visState.subgraph?.supernodes.length === 0
                    ? visState.subgraph.activeGrouping.selectedNodeIds.size === 1
                      ? `Grouping Mode: Nice! One node selected. Keep holding 'g' and click another node.`
                      : visState.subgraph.activeGrouping.selectedNodeIds.size === 2
                        ? `Grouping Mode: Neat! Two nodes selected. To group them, release 'g'. You can also pin and select more nodes.`
                        : `Grouping Mode: ${visState.subgraph.activeGrouping.selectedNodeIds.size} nodes selected. To group them, release 'g'. You can also pin and select more nodes.`
                    : `Grouping Mode: Click nodes below to select them, release 'g' to group. Click ✕ to ungroup.`}
              </div>
            ) : (
              <div className="absolute left-1/2 top-2.5 z-10 -translate-x-1/2 transform cursor-default whitespace-pre rounded-md bg-amber-600 px-2 py-1 text-center text-[11px] font-medium text-white">
                Cannot activate grouping mode. You need at least two pinned nodes.
                <br />
                Pin nodes by holding command/control + clicking a node in the link graph above.
              </div>
            ))}

          {!visState.subgraph?.activeGrouping.isActive &&
            visState.subgraph?.supernodes.length === 1 &&
            visState.subgraph?.supernodes.every((sn) => sn[0] === 'supernode') && (
              <div className="absolute left-1/2 top-2.5 z-10 -translate-x-1/2 transform cursor-default whitespace-pre rounded-md bg-sky-600 px-2 py-1 text-center text-[11px] font-medium text-white">
                {`Supernode created! Click the default 'supernode' label to rename it.`}
              </div>
            )}

          {/* Pin/Unpin Mode Label */}
          {isMetaKeyHeld &&
            !visState.subgraph?.activeGrouping.isActive &&
            visState.subgraph?.supernodes.length === 0 && (
              <div className="absolute left-1/2 top-2.5 z-10 -translate-x-1/2 transform cursor-default whitespace-pre rounded-md bg-emerald-600 px-2 py-1 text-center text-[11px] font-medium text-white">
                {visState.pinnedIds.length === 0 && (
                  <>
                    Pinning Mode: Click a <Circle className="mr-0.5 inline h-2.5 w-2.5" />
                    node in the link graph above or in the subgraph below to pin or unpin it.
                  </>
                )}
                {visState.pinnedIds.length === 1 && (
                  <>
                    Pinning Mode: Nice! One <Circle className="mr-0.5 inline h-2.5 w-2.5" />
                    node pinned. Pin one more to start grouping.
                  </>
                )}
                {visState.pinnedIds.length === 2 && (
                  <>
                    Pinning Mode: Great! Two <Circle className="mr-0.5 inline h-2.5 w-2.5" />
                    nodes pinned. Release command/control and hold {`'g'`} to start grouping.
                  </>
                )}
                {visState.pinnedIds.length > 2 && (
                  <>
                    Pinning Mode: {visState.pinnedIds.length} <Circle className="mr-0.5 inline h-2.5 w-2.5" />
                    nodes pinned. Release command/control and hold {`'g'`} to start grouping.
                  </>
                )}
              </div>
            )}
          {isMetaKeyHeld &&
            !visState.subgraph?.activeGrouping.isActive &&
            visState.subgraph &&
            visState.subgraph?.supernodes.length > 0 && (
              <div className="absolute left-1/2 top-2.5 z-10 -translate-x-1/2 transform cursor-default whitespace-pre rounded-md bg-emerald-600 px-2 py-1 text-center text-[11px] font-medium text-white">
                Pinning Mode: Click a <Circle className="ml-0.5 inline h-2.5 w-2.5" /> node to pin or unpin it.
              </div>
            )}

          {(visState.pinnedIds.length === 0 || showSubgraphHelp) && (
            <div className="absolute flex h-full min-h-full w-full flex-col items-start justify-center gap-y-1.5 rounded-xl bg-white/70 px-5 text-slate-700 backdrop-blur-sm">
              <div className="mb-1.5 w-full text-center text-lg font-bold">Creating a Subgraph</div>
              <div>
                <strong>
                  · Pin or Unpin a <Circle className="mb-1 mr-0.5 inline h-3 w-3" />
                  Node
                </strong>
                {`: Hold 'Command/Ctrl'`}, then click a <Circle className="mb-1 mr-0.5 inline h-3 w-3" />
                node in the link graph above.
              </div>
              <div>
                <strong>· Group Nodes into a Supernode</strong>
                {`: Hold 'g', then click multiple nodes in this subgraph to select them. When you release 'g', they'll be grouped into a "supernode".`}
              </div>
              <div>
                <strong>· Ungroup Supernodes</strong>
                {`: Hold 'g', then click the x next to a supernode to ungroup it.`}
              </div>
              <div>
                <strong>· Label a Supernode</strong>: Click the label under a supernode to edit its name.
              </div>
              <div className="flex flex-row flex-wrap items-center">
                <strong>· Share</strong>: To share this graph, subgraph, and your custom labels, click{' '}
                <Share2 className="mx-1 ml-2 h-4 w-4" /> Share in the top right.
              </div>
            </div>
          )}
          <Button
            variant="outline"
            size="sm"
            className={`${visState.pinnedIds.length === 0 ? 'hidden' : 'absolute left-3 top-3 h-8 w-28 gap-x-1.5 rounded-full border-0 bg-slate-100 px-0 py-0 text-slate-600 hover:bg-slate-200'}`}
            onClick={() => setShowSubgraphHelp(!showSubgraphHelp)}
          >
            {showSubgraphHelp ? (
              <>
                <XIcon className="h-5 w-5" />
                <span>Hide Help</span>
              </>
            ) : (
              <>
                <HelpCircleIcon className="h-5 w-5" />
                <span>Show Help</span>
              </>
            )}
          </Button>
          <Button
            variant="outline"
            size="sm"
            title="Copy Graph + Subgraph + Custom Labels to Clipboard"
            aria-label="Copy Graph + Subgraph + Custom Labels to Clipboard"
            className={`${visState.pinnedIds.length === 0 ? 'hidden' : 'absolute'} right-3 top-3 h-8 w-8 flex-col items-center justify-center gap-y-1.5 whitespace-nowrap bg-slate-100 px-0 text-[8px] font-medium leading-none text-slate-500 hover:bg-red-100 hover:bg-slate-200 hover:text-slate-600`}
            onClick={() => {
              setIsCopyModalOpen(true);
            }}
            disabled={visState.pinnedIds.length === 0}
          >
            <Share2 className="h-4 w-4" />
          </Button>
          <Button
            variant="outline"
            size="sm"
            title="Reset Graph to Defaults"
            aria-label="Reset Graph to Defaults"
            className={`${visState.pinnedIds.length === 0 ? 'hidden' : 'absolute'} bottom-3 right-3 h-8 w-8 flex-col items-center justify-center gap-y-1.5 whitespace-nowrap border-none border-slate-300 bg-slate-100 px-0 text-[8px] font-medium leading-none text-red-500 hover:bg-red-100 hover:text-red-600`}
            onClick={() => {
              // eslint-disable-next-line
              if (confirm('Are you sure you want to reset the graph to its default state?')) {
                resetSelectedGraphToDefaultVisState();
              }
            }}
            disabled={visState.pinnedIds.length === 0}
          >
            <Trash2 className="h-4 w-4" />
          </Button>
        </div>
        <div className="hidden w-full flex-row items-center justify-center gap-x-3">
          <label
            className="flex flex-row items-center gap-x-1 text-xs font-bold leading-none text-slate-600"
            htmlFor="stickyCheck"
          >
            <Checkbox.Root
              className="flex h-4 w-4 appearance-none items-center justify-center rounded-[3px] border border-slate-300 bg-white outline-none"
              defaultChecked
              checked={visState.subgraph?.sticky ? visState.subgraph.sticky : false}
              onCheckedChange={(e) => {
                updateVisStateField('subgraph', {
                  sticky: e === true,
                  dagrefy: visState.subgraph?.dagrefy ? visState.subgraph.dagrefy : false,
                  supernodes: visState.subgraph?.supernodes || [],
                  activeGrouping: visState.subgraph?.activeGrouping || {
                    isActive: false,
                    selectedNodeIds: new Set(),
                  },
                });
              }}
              id="stickyCheck"
            >
              <Checkbox.Indicator className="text-slate-600">
                <Check className="h-4 w-4" />
              </Checkbox.Indicator>
            </Checkbox.Root>
            Sticky
          </label>
          <label
            className="flex flex-row items-center gap-x-1 text-xs font-bold leading-none text-slate-600"
            htmlFor="dagrefyCheck"
          >
            <Checkbox.Root
              className="flex h-4 w-4 appearance-none items-center justify-center rounded-[3px] border border-slate-300 bg-white outline-none"
              defaultChecked
              checked={visState.subgraph?.dagrefy ? visState.subgraph.dagrefy : false}
              onCheckedChange={(e) => {
                updateVisStateField('subgraph', {
                  dagrefy: e === true,
                  sticky: visState.subgraph?.sticky ? visState.subgraph.sticky : false,
                  supernodes: visState.subgraph?.supernodes || [],
                  activeGrouping: visState.subgraph?.activeGrouping || {
                    isActive: false,
                    selectedNodeIds: new Set(),
                  },
                });
              }}
              id="dagrefyCheck"
            >
              <Checkbox.Indicator className="text-slate-600">
                <Check className="h-4 w-4" />
              </Checkbox.Indicator>
            </Checkbox.Root>
            Dagrefy
          </label>
        </div>
      </CardContent>
    </Card>
  );
}
