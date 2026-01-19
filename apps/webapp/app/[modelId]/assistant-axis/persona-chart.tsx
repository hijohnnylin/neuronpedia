import { useState } from "react";
import { useRef } from "react";
import { useCallback } from "react";
import { useEffect } from "react";
import { useMemo } from "react";
import { PersonaCheckResult } from "./types";

// Color palette for PC lines (Steered)
const PC_COLORS = [
    '#1f77b4',
    '#ff7f0e',
    '#2ca02c',
    '#d62728',
    '#9467bd',
    '#8c564b',
    '#e377c2',
    '#7f7f7f',
    '#bcbd22',
    '#17becf',
    '#aec7e8',
    '#ffbb78',
];

// Color for Default lines (slate-400)
const DEFAULT_COLOR = '#94a3b8';

interface ChartDataPoint {
    turnIndex: number;
    similarity: number;
    snippet: string;
}

interface ChartSeries {
    name: string;
    color: string;
    points: ChartDataPoint[];
}

export interface ChartData {
    series: ChartSeries[];
    nTurns: number;
}

// Transform persona-check data into chart data
// Turn indices are offset by +1 so turn 0 is reserved for the initial "starting" dots
export function buildChartData(result: PersonaCheckResult, type: 'steered' | 'default' = 'steered'): ChartData {
    // nTurns is +1 because turn 0 is the initial state
    const nTurns = result.turns.length + 1;

    const series = result.pc_titles.map((pcTitle, pcIdx) => {
        const color = type === 'steered' ? PC_COLORS[pcIdx % PC_COLORS.length] : DEFAULT_COLOR;
        // Offset turnIndex by +1 so first message is at turn 1
        const points = result.turns.map((turn, idx) => ({
            turnIndex: idx + 1,
            similarity: turn.pc_values[pcTitle],
            snippet: turn.snippet,
        }));

        const suffix = type === 'steered' ? 'Capped Llama' : 'Default Llama';
        return { name: suffix, color, points };
    });

    return { series, nTurns };
}

// Combine two chart data results
export function combineChartData(steeredData: ChartData | null, defaultData: ChartData | null): ChartData | null {
    if (!steeredData && !defaultData) return null;
    if (!steeredData) return defaultData;
    if (!defaultData) return steeredData;

    return {
        series: [...steeredData.series, ...defaultData.series],
        nTurns: Math.max(steeredData.nTurns, defaultData.nTurns),
    };
}

// Helper to generate smooth curved path between points (defined outside component for use in animation)
const generateSmoothPath = (points: { x: number; y: number }[]): string => {
    if (points.length < 2) return '';
    if (points.length === 2) {
        return `M ${points[0].x} ${points[0].y} L ${points[1].x} ${points[1].y}`;
    }

    let path = `M ${points[0].x} ${points[0].y}`;

    for (let i = 0; i < points.length - 1; i++) {
        const p0 = points[Math.max(0, i - 1)];
        const p1 = points[i];
        const p2 = points[i + 1];
        const p3 = points[Math.min(points.length - 1, i + 2)];

        // Catmull-Rom to Bezier conversion
        const tension = 0.5;
        const cp1x = p1.x + ((p2.x - p0.x) * tension) / 3;
        const cp1y = p1.y + ((p2.y - p0.y) * tension) / 3;
        const cp2x = p2.x - ((p3.x - p1.x) * tension) / 3;
        const cp2y = p2.y - ((p3.y - p1.y) * tension) / 3;

        path += ` C ${cp1x} ${cp1y}, ${cp2x} ${cp2y}, ${p2.x} ${p2.y}`;
    }

    return path;
};

// Simple SVG Chart component with animations
export default function PersonaChart({
    data,
    width = 340,
    height = 600,
    loading = false,
    isSteering = false,
    skipAnimationRef,
    onPointClick,
}: {
    data: ChartData | null;
    width?: number;
    height?: number;
    loading?: boolean;
    isSteering?: boolean;
    skipAnimationRef?: React.MutableRefObject<boolean>;
    onPointClick?: (turn: number) => void;
}) {
    const [tooltip, setTooltip] = useState<{
        visible: boolean;
        x: number;
        y: number;
        content: { pcName: string; turn: number; similarity: number; snippet: string };
    } | null>(null);

    const [showStartTooltip, setShowStartTooltip] = useState<{ visible: boolean; x: number; y: number } | null>(null);

    // Track previous data for animations
    const prevDataRef = useRef<ChartData | null>(null);
    // Refs for animated elements - will be updated directly without React state
    const animatedDotsRef = useRef<Map<string, SVGCircleElement | null>>(new Map());
    const animatedPathsRef = useRef<Map<string, SVGPathElement | null>>(new Map());
    const animatedPingsRef = useRef<Map<string, SVGCircleElement | null>>(new Map());
    const animatedHoverRingsRef = useRef<Map<string, SVGCircleElement | null>>(new Map());
    // Animation state stored in ref to avoid re-renders
    const animationStateRef = useRef<{
        isAnimating: boolean;
        info: Map<
            string,
            {
                startX: number;
                startY: number;
                endX: number;
                endY: number;
                // All path points for this series - last point will be interpolated during animation
                pathPoints: { x: number; y: number }[];
            }
        >;
        startTime: number | null;
        frameId: number | null;
    }>({
        isAnimating: false,
        info: new Map(),
        startTime: null,
        frameId: null,
    });
    // Force re-render trigger (only used at start and end of animation)
    const [, forceUpdate] = useState(0);

    // Animation duration in ms
    const ANIMATION_DURATION = 3000;

    const padding = { top: 50, right: 16, bottom: 30, left: 16 };
    const chartWidth = width - padding.left - padding.right;
    const chartHeight = height - padding.top - padding.bottom;

    const minSim = -2.5;
    const maxSim = 2.5;
    const simRange = maxSim - minSim; // 6

    // Calculate scales
    const nTurns = data?.nTurns || 1;
    // Use minimum of 8 turns for y-axis scaling to prevent rescaling until we hit 8 turns
    const yAxisTurns = Math.max(nTurns, 8);

    const xScale = useCallback(
        (sim: number) => {
            return ((sim - minSim) / simRange) * chartWidth;
        },
        [chartWidth],
    );

    const yScale = useCallback(
        (turn: number) => {
            return (turn / Math.max(yAxisTurns - 1, 1)) * chartHeight;
        },
        [yAxisTurns, chartHeight],
    );

    // Easing function for smooth animation
    const easeOutCubic = (t: number): number => {
        return 1 - Math.pow(1 - t, 3);
    };

    // Animation loop - directly manipulates DOM elements via refs
    const runAnimation = useCallback(() => {
        const state = animationStateRef.current;

        const animateFrame = (timestamp: number) => {
            if (state.startTime === null) {
                state.startTime = timestamp;
            }

            const elapsed = timestamp - state.startTime;
            const rawProgress = Math.min(elapsed / ANIMATION_DURATION, 1);
            const easedProgress = easeOutCubic(rawProgress);

            // Update each animated element directly via DOM
            state.info.forEach((anim, seriesName) => {
                const currentX = anim.startX + (anim.endX - anim.startX) * easedProgress;
                const currentY = anim.startY + (anim.endY - anim.startY) * easedProgress;

                // Update dot position
                const dot = animatedDotsRef.current.get(seriesName);
                if (dot) {
                    dot.setAttribute('cx', String(currentX));
                    dot.setAttribute('cy', String(currentY));
                }

                // Update ping position
                const ping = animatedPingsRef.current.get(seriesName);
                if (ping) {
                    ping.setAttribute('cx', String(currentX));
                    ping.setAttribute('cy', String(currentY));
                }

                // Update hover ring position
                const hoverRing = animatedHoverRingsRef.current.get(seriesName);
                if (hoverRing) {
                    hoverRing.setAttribute('cx', String(currentX));
                    hoverRing.setAttribute('cy', String(currentY));
                }

                // Update the full curved path with interpolated last point
                const pathEl = animatedPathsRef.current.get(seriesName);
                if (pathEl && anim.pathPoints.length > 0) {
                    // Create path points with interpolated last point position
                    const interpolatedPathPoints = anim.pathPoints.map((p, idx) => {
                        if (idx === anim.pathPoints.length - 1) {
                            return { x: currentX, y: currentY };
                        }
                        return p;
                    });
                    const pathD = generateSmoothPath(interpolatedPathPoints);
                    pathEl.setAttribute('d', pathD);
                }
            });

            if (rawProgress < 1) {
                state.frameId = requestAnimationFrame(animateFrame);
            } else {
                // Animation complete - trigger React re-render to show final state
                state.isAnimating = false;
                state.info = new Map();
                state.startTime = null;
                state.frameId = null;
                forceUpdate((n) => n + 1);
            }
        };

        state.frameId = requestAnimationFrame(animateFrame);
    }, [ANIMATION_DURATION]);

    // Update prevDataRef when data changes and we're not loading
    useEffect(() => {
        // Reset prevDataRef when data is cleared (e.g., on reset)
        if (data === null) {
            prevDataRef.current = null;
            // Cancel any running animation
            if (animationStateRef.current.frameId) {
                cancelAnimationFrame(animationStateRef.current.frameId);
            }
            animationStateRef.current.isAnimating = false;
            animationStateRef.current.info = new Map();
            return;
        }

        if (!loading && !isSteering) {
            // Skip animation if requested (e.g., when loading saved data)
            if (skipAnimationRef?.current) {
                skipAnimationRef.current = false;
                prevDataRef.current = data;
                return;
            }

            // Trigger animation when new data arrives
            // nTurns starts at 1 (just turn 0 / "Start"), so first message has nTurns = 2
            // Use || 1 to handle the case where prevDataRef.current is null (first message)
            const prevNTurns = prevDataRef.current?.nTurns || 1;
            if (data.nTurns > prevNTurns) {
                // Calculate animation info for each series
                const newAnimationInfo = new Map<
                    string,
                    {
                        startX: number;
                        startY: number;
                        endX: number;
                        endY: number;
                        pathPoints: { x: number; y: number }[];
                    }
                >();

                data.series.forEach((series) => {
                    const prevSeries = prevDataRef.current?.series.find((s) => s.name === series.name);
                    const prevPointsLength = prevSeries?.points.length ?? 0;

                    if (series.points.length > prevPointsLength) {
                        // Get the new point that was just added
                        const newPoint = series.points[series.points.length - 1];
                        if (newPoint) {
                            // Calculate start position - either from previous last point, or from initial position at turn 0
                            let startX: number;
                            let startY: number;

                            if (prevSeries && prevSeries.points.length > 0) {
                                const prevLastPoint = prevSeries.points[prevSeries.points.length - 1];
                                startX = xScale(prevLastPoint.similarity);
                                startY = yScale(prevLastPoint.turnIndex);
                            } else {
                                // First point - animate from the initial position at turn 0 (center starting point)
                                startX = xScale(0);
                                startY = yScale(0);
                            }

                            const endX = xScale(newPoint.similarity);
                            const endY = yScale(newPoint.turnIndex);

                            // Calculate all path points (including initial point at turn 0)
                            // The last point will be interpolated during animation
                            const allPoints = [{ turnIndex: 0, similarity: 0 }, ...series.points];
                            const pathPoints = allPoints.map((p) => ({
                                x: xScale(p.similarity),
                                y: yScale(p.turnIndex),
                            }));

                            // Store animation info with path points
                            newAnimationInfo.set(series.name, { startX, startY, endX, endY, pathPoints });
                        }
                    }
                });

                // Set up animation state
                animationStateRef.current.isAnimating = true;
                animationStateRef.current.info = newAnimationInfo;
                animationStateRef.current.startTime = null;

                // Trigger re-render to show animated elements, then start animation
                forceUpdate((n) => n + 1);

                // Start animation after React has rendered the animated elements
                requestAnimationFrame(() => {
                    runAnimation();
                });
            }
            prevDataRef.current = data;
        }

        return () => {
            if (animationStateRef.current.frameId) {
                cancelAnimationFrame(animationStateRef.current.frameId);
            }
        };
    }, [data, loading, isSteering, xScale, yScale, runAnimation]);

    const handleMouseEnter = (e: React.MouseEvent, pcName: string, turn: number, similarity: number, snippet: string) => {
        const rect = e.currentTarget.getBoundingClientRect();
        setTooltip({
            visible: true,
            x: rect.right + 8,
            y: rect.top + rect.height / 2,
            content: { pcName, turn, similarity, snippet },
        });
    };

    const handleMouseLeave = () => {
        setTooltip(null);
    };

    const handleStartDotMouseEnter = (e: React.MouseEvent) => {
        const rect = e.currentTarget.getBoundingClientRect();
        setShowStartTooltip({
            visible: true,
            x: rect.right + 8,
            y: rect.top + rect.height / 2,
        });
    };

    const handleStartDotMouseLeave = () => {
        setShowStartTooltip(null);
    };

    const handleStartDotClick = () => {
        onPointClick?.(0);
    };

    // Find the maximum turn index (bottom-most point)
    const maxTurnIndex = data ? Math.max(...data.series.flatMap((s) => s.points.map((p) => p.turnIndex))) : 0;

    // Check if we have data
    const hasData = data && data.series.length > 0;

    // Show pending state when isSteering is true (user sent message)
    const showPending = isSteering;

    // Get last known positions for each series (for pending animation)
    // When no data, use initial position at turn 0, similarity 0
    const lastPositions = useMemo(() => {
        const positions = new Map<string, { x: number; similarity: number; y: number }>();
        if (!hasData) {
            // Initial positions for the two initial dots
            positions.set('initial-steered', { x: xScale(0), similarity: 0, y: yScale(0) });
            positions.set('initial-default', { x: xScale(0), similarity: 0, y: yScale(0) });
        } else if (data) {
            data.series.forEach((series) => {
                const lastPoint = series.points[series.points.length - 1];
                if (lastPoint) {
                    positions.set(series.name, {
                        x: xScale(lastPoint.similarity),
                        similarity: lastPoint.similarity,
                        y: yScale(lastPoint.turnIndex),
                    });
                }
            });
        }
        return positions;
    }, [data, hasData, xScale, yScale]);

    // Calculate pending animation info DURING RENDER (before useEffect runs)
    // This is needed because useEffect runs AFTER render, so without this,
    // the first render with new data would show elements at final positions (jump)
    const pendingAnimationInfo = useMemo(() => {
        // If animation is already running via refs, don't recalculate
        if (animationStateRef.current.isAnimating) return null;
        if (!data || loading || isSteering) return null;
        // Skip if animation should be skipped (e.g., loading saved data)
        if (skipAnimationRef?.current) return null;

        const prevNTurns = prevDataRef.current?.nTurns || 1;
        if (data.nTurns <= prevNTurns) return null;

        const info = new Map<
            string,
            {
                startX: number;
                startY: number;
                endX: number;
                endY: number;
                pathPoints: { x: number; y: number }[];
            }
        >();

        data.series.forEach((series) => {
            const prevSeries = prevDataRef.current?.series.find((s) => s.name === series.name);
            const prevPointsLength = prevSeries?.points.length ?? 0;

            if (series.points.length > prevPointsLength) {
                const newPoint = series.points[series.points.length - 1];
                if (newPoint) {
                    let startX: number;
                    let startY: number;

                    if (prevSeries && prevSeries.points.length > 0) {
                        const prevLastPoint = prevSeries.points[prevSeries.points.length - 1];
                        startX = xScale(prevLastPoint.similarity);
                        startY = yScale(prevLastPoint.turnIndex);
                    } else {
                        // First point - animate from the initial position at turn 0
                        startX = xScale(0);
                        startY = yScale(0);
                    }

                    const endX = xScale(newPoint.similarity);
                    const endY = yScale(newPoint.turnIndex);

                    // Calculate all path points (including initial point at turn 0)
                    const allPoints = [{ turnIndex: 0, similarity: 0 }, ...series.points];
                    const pathPoints = allPoints.map((p) => ({
                        x: xScale(p.similarity),
                        y: yScale(p.turnIndex),
                    }));

                    info.set(series.name, { startX, startY, endX, endY, pathPoints });
                }
            }
        });

        return info.size > 0 ? info : null;
    }, [data, loading, isSteering, xScale, yScale]);

    return (
        <div className="relative" style={{ width, height }}>
            <svg width={width} height={height}>
                {/* Labels and center line */}
                <g transform={`translate(${padding.left}, ${padding.top})`}>
                    {/* Vertical center line at x=0 to show neutral */}
                    <line x1={xScale(0)} y1={0} x2={xScale(0)} y2={chartHeight} stroke="#e2e8f0" strokeWidth={1} />
                    {/* Y axis grid lines (turns) - 0-indexed, show up to yAxisTurns */}
                    {Array.from({ length: yAxisTurns }).map((_, i) => (
                        <g key={`turn-${i}`}>
                            {/* Horizontal line at each turn (skip turn 0) */}
                            {i > 0 && i < yAxisTurns - 1 && (
                                <line
                                    x1={0}
                                    y1={yScale(i)}
                                    x2={chartWidth}
                                    y2={yScale(i)}
                                    stroke="#e2e8f0"
                                    strokeWidth={1}
                                    strokeDasharray="4,4"
                                />
                            )}
                            {/* <text x={-10} y={yScale(i)} textAnchor="end" alignmentBaseline="middle" fontSize={10} fill="#64748b">
                                {i === 0 ? '' : i}
                            </text> */}
                        </g>
                    ))}
                    {/* Axis titles
          <text
            x={-35}
            y={chartHeight / 2}
            textAnchor="middle"
            fontSize={11}
            fill="#475569"
            transform={`rotate(-90, -35, ${chartHeight / 2})`}
          >
            Turn
          </text>
          <text x={0} y={-16} textAnchor="start" fontSize={11} fill="#475569">
            Role-Playing
          </text>
          <text x={chartWidth} y={-16} textAnchor="end" fontSize={11} fill="#475569">
            Assistant
          </text> */}
                </g>

                {/* Data series */}
                <g transform={`translate(${padding.left}, ${padding.top})`}>
                    {/* Initial dots at turn 0 when no data */}
                    {!hasData && (
                        <>
                            {/* Default initial dot (rendered first, so it's below) */}
                            <g className="group">
                                {/* Ping animation only when loading */}
                                {showPending && (
                                    <circle cx={xScale(0)} cy={yScale(0)} r={6} fill={DEFAULT_COLOR} opacity={0.75} style={{ pointerEvents: 'none' }}>
                                        <animate attributeName="r" from="6" to="18" dur="1s" repeatCount="indefinite" />
                                        <animate attributeName="opacity" from="0.75" to="0" dur="1s" repeatCount="indefinite" />
                                    </circle>
                                )}
                                {/* Hover ring */}
                                <circle
                                    cx={xScale(0)}
                                    cy={yScale(0)}
                                    r={9}
                                    fill="none"
                                    stroke={DEFAULT_COLOR}
                                    strokeWidth={2}
                                    className="opacity-0 group-hover:opacity-100 transition-opacity"
                                    style={{ pointerEvents: 'none' }}
                                />
                                <circle cx={xScale(0)} cy={yScale(0)} r={6} fill={DEFAULT_COLOR} stroke="white" strokeWidth={2} />
                            </g>
                            {/* Steered initial dot (rendered second, so it's on top) - clickable */}
                            <g
                                className="cursor-pointer group"
                                onMouseEnter={handleStartDotMouseEnter}
                                onMouseLeave={handleStartDotMouseLeave}
                                onClick={handleStartDotClick}
                            >
                                {/* Ping animation only when loading */}
                                {showPending && (
                                    <circle cx={xScale(0)} cy={yScale(0)} r={6} fill={PC_COLORS[0]} opacity={0.75} style={{ pointerEvents: 'none' }}>
                                        <animate attributeName="r" from="6" to="18" dur="1s" repeatCount="indefinite" />
                                        <animate attributeName="opacity" from="0.75" to="0" dur="1s" repeatCount="indefinite" />
                                    </circle>
                                )}
                                {/* Hover ring */}
                                <circle
                                    cx={xScale(0)}
                                    cy={yScale(0)}
                                    r={9}
                                    fill="none"
                                    stroke={PC_COLORS[0]}
                                    strokeWidth={2}
                                    className="opacity-0 group-hover:opacity-100 transition-opacity"
                                    style={{ pointerEvents: 'none' }}
                                />
                                <circle cx={xScale(0)} cy={yScale(0)} r={6} fill={PC_COLORS[0]} stroke="white" strokeWidth={2} />
                            </g>
                        </>
                    )}

                    {/* Data series when we have data - sort so default (gray) renders first, steered (blue) on top */}
                    {hasData &&
                        [...data!.series]
                            .sort((a, b) => {
                                // Default series should come first (rendered below), Steered series last (rendered on top)
                                const aIsDefault = a.name.includes('Default');
                                const bIsDefault = b.name.includes('Default');
                                if (aIsDefault && !bIsDefault) return -1;
                                if (!aIsDefault && bIsDefault) return 1;
                                return 0;
                            })
                            .map((series) => {
                                const animState = animationStateRef.current;
                                // Use ref-based animation info if available, otherwise fall back to pending animation info
                                // This handles the first render after new data arrives (before useEffect runs)
                                const anim = animState.info.get(series.name) || pendingAnimationInfo?.get(series.name);
                                const isAnimating =
                                    (animState.isAnimating && animState.info.get(series.name)) ||
                                    !!pendingAnimationInfo?.get(series.name);

                                // Include the initial point at turn 0, similarity 0
                                const allPoints = [{ turnIndex: 0, similarity: 0, snippet: 'Starting point' }, ...series.points];

                                // Calculate path points - during animation, the path is updated via ref
                                const pathPoints = allPoints.map((p) => ({
                                    x: xScale(p.similarity),
                                    y: yScale(p.turnIndex),
                                }));

                                // Initial path - during animation, starts with last point at previous position
                                const initialPathPoints =
                                    isAnimating && anim
                                        ? pathPoints.map((p, idx) =>
                                            idx === pathPoints.length - 1 ? { x: anim.startX, y: anim.startY } : p,
                                        )
                                        : pathPoints;
                                const pathD = generateSmoothPath(initialPathPoints);

                                return (
                                    <g key={series.name}>
                                        {/* Full curved path - updated by animation loop during animation */}
                                        {pathPoints.length > 1 && (
                                            <path
                                                ref={
                                                    isAnimating
                                                        ? (el) => {
                                                            animatedPathsRef.current.set(series.name, el);
                                                        }
                                                        : undefined
                                                }
                                                fill="none"
                                                stroke={series.color}
                                                strokeWidth={2}
                                                d={pathD}
                                            />
                                        )}

                                        {/* Initial dot at turn 0 - clickable */}
                                        <g
                                            className="cursor-pointer group"
                                            onMouseEnter={handleStartDotMouseEnter}
                                            onMouseLeave={handleStartDotMouseLeave}
                                            onClick={handleStartDotClick}
                                        >
                                            {/* Hover ring */}
                                            <circle
                                                cx={xScale(0)}
                                                cy={yScale(0)}
                                                r={9}
                                                fill="none"
                                                stroke={series.color}
                                                strokeWidth={2}
                                                className="opacity-0 group-hover:opacity-100 transition-opacity"
                                            />
                                            <circle
                                                cx={xScale(0)}
                                                cy={yScale(0)}
                                                r={6}
                                                fill={series.color}
                                                stroke="white"
                                                strokeWidth={2}
                                            />
                                        </g>

                                        {/* Data points (starting from turn 1) */}
                                        {series.points.map((point, pointIdx) => {
                                            const isBottomMost = point.turnIndex === maxTurnIndex;
                                            const isNewPoint = isAnimating && pointIdx === series.points.length - 1;
                                            const endX = xScale(point.similarity);
                                            const endY = yScale(point.turnIndex);

                                            // For animated point, start at the animation start position
                                            const initialX = isNewPoint && anim ? anim.startX : endX;
                                            const initialY = isNewPoint && anim ? anim.startY : endY;

                                            return (
                                                <g key={`${series.name}-${point.turnIndex}`}>
                                                    {/* Ping animation for bottom-most point - only when loading (showPending) */}
                                                    {isBottomMost && showPending && !isNewPoint && (
                                                        <circle cx={endX} cy={endY} r={6} fill={series.color} opacity={0.75}>
                                                            <animate attributeName="r" from="6" to="18" dur="1s" repeatCount="indefinite" />
                                                            <animate attributeName="opacity" from="0.75" to="0" dur="1s" repeatCount="indefinite" />
                                                        </circle>
                                                    )}

                                                    {/* Data point - for animated points, position updated by animation loop */}
                                                    {isNewPoint ? (
                                                        <g
                                                            className="cursor-pointer group"
                                                            onMouseEnter={(e) =>
                                                                handleMouseEnter(e, series.name, point.turnIndex, point.similarity, point.snippet)
                                                            }
                                                            onMouseLeave={handleMouseLeave}
                                                            onClick={() => onPointClick?.(point.turnIndex)}
                                                        >
                                                            {/* Pulsing animation on the moving dot during animation - updated by animation loop */}
                                                            <circle
                                                                ref={(el) => {
                                                                    animatedPingsRef.current.set(series.name, el);
                                                                }}
                                                                cx={initialX}
                                                                cy={initialY}
                                                                r={6}
                                                                fill={series.color}
                                                                opacity={0.75}
                                                                style={{ pointerEvents: 'none' }}
                                                            >
                                                                <animate attributeName="r" from="6" to="18" dur="1s" repeatCount="indefinite" />
                                                                <animate attributeName="opacity" from="0.75" to="0" dur="1s" repeatCount="indefinite" />
                                                            </circle>
                                                            {/* Hover ring */}
                                                            <circle
                                                                ref={(el) => {
                                                                    animatedHoverRingsRef.current.set(series.name, el);
                                                                }}
                                                                cx={initialX}
                                                                cy={initialY}
                                                                r={9}
                                                                fill="none"
                                                                stroke={series.color}
                                                                strokeWidth={2}
                                                                className="opacity-0 group-hover:opacity-100 transition-opacity"
                                                                style={{ pointerEvents: 'none' }}
                                                            />
                                                            <circle
                                                                ref={(el) => {
                                                                    animatedDotsRef.current.set(series.name, el);
                                                                }}
                                                                cx={initialX}
                                                                cy={initialY}
                                                                r={6}
                                                                fill={series.color}
                                                                stroke="white"
                                                                strokeWidth={2}
                                                            />
                                                        </g>
                                                    ) : (
                                                        <g
                                                            className="cursor-pointer group"
                                                            onMouseEnter={(e) =>
                                                                handleMouseEnter(e, series.name, point.turnIndex, point.similarity, point.snippet)
                                                            }
                                                            onMouseLeave={handleMouseLeave}
                                                            onClick={() => onPointClick?.(point.turnIndex)}
                                                        >
                                                            {/* Hover ring */}
                                                            <circle
                                                                cx={endX}
                                                                cy={endY}
                                                                r={9}
                                                                fill="none"
                                                                stroke={series.color}
                                                                strokeWidth={2}
                                                                className="opacity-0 group-hover:opacity-100 transition-opacity"
                                                            />
                                                            <circle
                                                                cx={endX}
                                                                cy={endY}
                                                                r={6}
                                                                fill={series.color}
                                                                stroke="white"
                                                                strokeWidth={2}
                                                            />
                                                        </g>
                                                    )}
                                                </g>
                                            );
                                        })}
                                    </g>
                                );
                            })}
                </g>

                {/* Legend - horizontal, centered above chart */}
                <g className="hidden sm:block" transform={`translate(${width / 2}, 12)`}>
                    <g transform="translate(-85, 0)">
                        <line x1="0" y1="0" x2="16" y2="0" stroke={DEFAULT_COLOR} strokeWidth={2} />
                        <circle cx="16" cy="0" r="3" fill={DEFAULT_COLOR} stroke="white" strokeWidth={1} />
                        <text x="24" y="0" fontSize={11} fill="#64748b" alignmentBaseline="middle">Default</text>
                    </g>
                    <g transform="translate(15, 0)">
                        <line x1="0" y1="0" x2="16" y2="0" stroke={PC_COLORS[0]} strokeWidth={2} />
                        <circle cx="16" cy="0" r="3" fill={PC_COLORS[0]} stroke="white" strokeWidth={1} />
                        <text x="24" y="0" fontSize={11} fill="#64748b" alignmentBaseline="middle">Capped</text>
                    </g>
                </g>
            </svg>

            {/* Tooltip */}
            {tooltip && tooltip.visible && (
                <div
                    className={
                        'pointer-events-none flex flex-col gap-y-1 fixed z-50 max-w-[200px] sm:max-w-xs rounded-md py-1.5 px-2 sm:px-5 sm:py-4 text-[10px] sm:text-xs border shadow sm:shadow-lg ' +
                        (tooltip.content.pcName.toLowerCase().includes('capped')
                            ? 'bg-sky-50 text-sky-700 border-sky-700/70'
                            : 'bg-slate-50 text-slate-600 border-slate-700/70')
                    }
                    style={{
                        left: tooltip.x,
                        top: tooltip.y,
                        transform: 'translateY(-50%)',
                    }}
                >
                    <div className={'font-bold flex justify-between'}>
                        <span>{tooltip.content.pcName}</span>
                        <span className="font-medium">Turn #{tooltip.content.turn}</span>
                    </div>
                    <div className="flex justify-between">
                        <span>{tooltip.content.similarity < 0 ? '⬅️ Role-Playing' : '➡️ Assistant'}</span>
                        <span>Projection: {tooltip.content.similarity.toFixed(2)}</span>
                    </div>
                    <div
                        className={
                            'mt-2 border-t pt-2 text-[11px] italic ' +
                            (tooltip.content.pcName.toLowerCase().includes('capped')
                                ? 'border-sky-600/50 text-sky-700'
                                : 'border-slate-300 text-slate-600')
                        }
                    >
                        {`"`}{tooltip.content.snippet.length > 220
                            ? tooltip.content.snippet.slice(0, 220) + '...'
                            : tooltip.content.snippet}{`"`}
                    </div>
                </div>
            )}

            {/* Start of Conversation Tooltip */}
            {showStartTooltip && showStartTooltip.visible && (
                <div
                    className="pointer-events-none fixed z-50 rounded bg-slate-800 px-3 py-2 text-xs text-white shadow-lg"
                    style={{
                        left: showStartTooltip.x,
                        top: showStartTooltip.y,
                        transform: 'translateY(-50%)',
                    }}
                >
                    Start of Conversation
                </div>
            )}

        </div>
    );
}
