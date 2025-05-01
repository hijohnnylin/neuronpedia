/* eslint-disable no-param-reassign */

import { useCircuitCLT } from '@/components/provider/circuit-clt-provider';
import { useScreenSize } from '@/lib/hooks/use-screen-size';
import { useCallback, useEffect, useRef, useState } from 'react';
import { CLTGraphLink, CLTGraphNode, hideTooltip, showTooltip } from './clt-utils';
import d3 from './d3-jetpack';

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

export default function CLTSubgraph() {
  const svgRef = useRef<SVGSVGElement>(null);
  const divRef = useRef<HTMLDivElement>(null);
  const { visState, selectedGraph, updateVisStateField } = useCircuitCLT();
  const [simulation, setSimulation] = useState<d3.Simulation<ForceNode, undefined> | null>(null);
  const [selForceNodes, setSelForceNodes] = useState<ForceNode[]>([]);
  const [nodeIdToNode, setNodeIdToNode] = useState<Record<string, CLTGraphNode>>({});

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

  // Initialize the D3 subgraph visualization
  useEffect(() => {
    console.log('rendering');
    if (!svgRef.current || !divRef.current || !selectedGraph) return;

    // Stop any existing simulation
    if (simulation) {
      simulation.stop();
    }

    // Clear any existing content
    d3.select(svgRef.current).selectAll('*').remove();
    d3.select(divRef.current).selectAll('*').remove();

    const data = selectedGraph;
    const { nodes, links } = data;

    // Set up margins and container dimensions
    const margin = {
      top: 26,
      bottom: 5,
      left: visState.isHideLayer ? 0 : 30,
      right: 5,
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

    setSimulation(newSimulation);

    // Create SVG path elements for links
    const svgPaths = svg
      .appendMany('path.link-path', sgLinks)
      .attr('fill', 'none')
      .attr('marker-mid', (d) => (d.weight > 0 ? 'url(#mid-positive)' : 'url(#mid-negative)'))
      .attr('stroke-width', (d) => Math.abs(d.weight) * 15)
      .attr('stroke', (d) => d.color)
      .attr('opacity', 0.8)
      .attr('stroke-linecap', 'round');

    // Add weight labels for links
    const edgeLabels = svg.appendMany('text.weight-label', sgLinks);

    // Function to render edges with offsets
    function renderEdges() {
      // Adjust link source positions based on source node member count
      // eslint-disable-next-line @typescript-eslint/no-shadow
      d3.nestBy(sgLinks, (d) => d.source).forEach((links) => {
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
      d3.nestBy(sgLinks, (d) => d.target).forEach((links) => {
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

      // Update SVG paths
      svgPaths.attr('d', (d: SubgraphLink) => {
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

    // Setup drag behavior
    const drag = d3
      .drag()
      .on('drag', (ev: any) => {
        // Only when actually dragging, mark as no longer dagre positioned and restart sim
        ev.subject.dagrePositioned = false;
        if (!ev.active) newSimulation.alphaTarget(0.3).restart();
        ev.subject.fx = ev.x;
        ev.subject.x = ev.x;
        ev.subject.fy = ev.y;
        ev.subject.y = ev.y;
        // eslint-disable-next-line
        renderForce();
      })
      .on('end', (ev: any) => {
        if (!ev.active) newSimulation.alphaTarget(0);
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
        // check if it's a click. if so, don't run the rest of the code (or it will rerender and not run the click)
        if (ev.button === 0) {
          return;
        }
        if (!ev.shiftKey) {
          updateVisStateField('hoveredId', d.node.featureId || null);
          updateVisStateField('hoveredCtxIdx', d.node.ctx_idx);
          showTooltip(ev, d.node);
        }
      })
      .on('mouseleave', (ev: MouseEvent) => {
        if (ev.buttons === 1) {
          return;
        }
        updateVisStateField('hoveredId', null);
        updateVisStateField('hoveredCtxIdx', null);
        hideTooltip();
      })
      .on('click', (ev: MouseEvent, d: ForceNode) => {
        // Handle grouping mode
        if (visState.subgraph?.activeGrouping.isActive) {
          const nodeId = d.node.supernodeId || d.node.nodeId;
          if (nodeId) {
            const selectedNodeIds = new Set(visState.subgraph.activeGrouping.selectedNodeIds);
            if (selectedNodeIds.has(nodeId)) {
              selectedNodeIds.delete(nodeId);
            } else {
              selectedNodeIds.add(nodeId);
            }

            updateVisStateField('subgraph', {
              ...visState.subgraph,
              activeGrouping: {
                ...visState.subgraph.activeGrouping,
                selectedNodeIds,
              },
            });
          }

          ev.stopPropagation();
          ev.preventDefault();
          return;
        }

        // Regular click behavior
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
          const newClickedId = visState.clickedId === d.node.nodeId ? null : d.node.nodeId;
          updateVisStateField('clickedId', newClickedId || null);
          updateVisStateField('clickedCtxIdx', newClickedId ? d.node.ctx_idx : null);
        }
      })
      // @ts-ignore
      .call(drag);

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
        if (ev.buttons === 1) {
          return;
        }
        if (visState.subgraph?.activeGrouping.isActive || ev.metaKey || ev.ctrlKey) {
          // grouping, don't activate behavior
          return;
        }
        // updateVisStateField('clickedId', d.nodeId || null);
        // updateVisStateField('clickedCtxIdx', d.ctx_idx);
        updateVisStateField('hoveredId', d.featureId || null);
        updateVisStateField('hoveredCtxIdx', d.ctx_idx);
        showTooltip(ev, d);
        ev.stopPropagation();
      })
      .on('mouseleave', (ev: MouseEvent) => {
        if (ev.buttons === 1) {
          return;
        }
        // updateVisStateField('clickedId', null);
        // updateVisStateField('clickedCtxIdx', null);
        updateVisStateField('hoveredId', null);
        updateVisStateField('hoveredCtxIdx', null);
        hideTooltip();
      })
      .on('click', (ev: MouseEvent, d: CLTGraphNode) => {
        if (!visState.subgraph?.activeGrouping.isActive) {
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
          const newClickedId = visState.clickedId === d.nodeId ? null : d.nodeId;
          updateVisStateField('clickedId', newClickedId || null);
          updateVisStateField('clickedCtxIdx', newClickedId ? d.ctx_idx : null);
        }
      })
      .attr('title', (d: CLTGraphNode) => d.ppClerp || '');

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

    // Function to run on each simulation tick
    function renderForce() {
      nodeSel.translate((d) => [d.x, d.y]);
      renderEdges();

      edgeLabels
        .filter((d) => {
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

    // Add text labels
    const nodeTextSel = nodeSel.append('div.node-text-container');

    nodeTextSel.append('span').text((d) => d.node.ppClerp || '');

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
      newSimulation.alphaTarget(0.3).restart();
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

    // Function to update node styles
    function styleNodes() {
      nodeSel
        .classed('clicked', (d: ForceNode) => d.node.nodeId === visState.clickedId)
        .classed('hovered', (d: ForceNode) => d.node.featureId === visState.hoveredId)
        .style('z-index', (d: ForceNode) => `${Math.round(d.x * 20 + d.y) + 1000}`)
        .classed(
          'grouping-selected',
          (d: ForceNode) => visState.subgraph?.activeGrouping.selectedNodeIds.has(d.node.nodeId || '') || false,
        );

      memberNodeSel
        .classed('clicked', (d: CLTGraphNode) => d.nodeId === visState.clickedId)
        .classed('hovered', (d: CLTGraphNode) => d.featureId === visState.hoveredId)
        .style('background', (d: CLTGraphNode) => d.tmpClickedLink?.pctInputColor || '#fff')
        .style('color', (d: CLTGraphNode) => bgColorToTextColor(d.tmpClickedLink?.pctInputColor) || 'black');

      // Style clicked links using supernode adjusted graph
      sgNodes.forEach((d) => {
        d.tmpClickedSgSource = d.tmpClickedLink?.sourceNode === d ? d.tmpClickedLink : undefined;
        d.tmpClickedSgTarget = d.tmpClickedLink?.targetNode === d ? d.tmpClickedLink : undefined;
      });

      if (visState.clickedId) {
        sgLinks.forEach((d) => {
          const source = typeof d.source === 'object' ? d.source : null;
          const target = typeof d.target === 'object' ? d.target : null;

          if (source?.node.nodeId === visState.clickedId && target && nodeIdToNode[target.node.nodeId]) {
            nodeIdToNode[target.node.nodeId].tmpClickedSgTarget = d;
          }

          if (target?.node.nodeId === visState.clickedId && source && nodeIdToNode[source.node.nodeId]) {
            nodeIdToNode[source.node.nodeId].tmpClickedSgSource = d;
          }
        });
      }
    }

    // Start rendering and apply styles
    styleNodes();
    renderEdges();

    // Add tick handler to simulation
    newSimulation.on('tick', renderForce);

    // Cleanup function
    // eslint-disable-next-line
    return () => {
      newSimulation.stop();
    };
  }, [screenSize, selectedGraph, visState, updateVisStateField, pctInputColorFn, bgColorToTextColor]);

  return (
    <div className="relative mt-3 w-full">
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
      <div className="subgraph relative min-h-[400px] w-full">
        <svg className="absolute h-[400px] w-full" height={400} ref={svgRef} />
        <div className="absolute h-[400px] w-full" ref={divRef} />
      </div>
    </div>
  );
}
